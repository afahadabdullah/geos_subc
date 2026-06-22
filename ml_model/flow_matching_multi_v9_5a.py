import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers import UNet2DModel
from tqdm.auto import tqdm

# Number of intermediate feature channels between shared UNet and per-week heads
HEAD_FEATURES = 64
HEAD_HIDDEN = 64
HEAD_BOTTLENECK = 32
CONTEXT_HIDDEN = 128
GLOBAL_TOKEN_DIM = 128
# Default v8 width: only widen the deepest stage so we gain bottleneck capacity
# while keeping the higher-resolution attention blocks unchanged.
UNET_BLOCK_OUT_CHANNELS = (128, 256, 512, 768)


def _num_groups(channels, max_groups=8):
    for groups in (max_groups, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class FiLMModulation(nn.Module):
    """
    Lightweight FiLM block that applies per-channel scale and shift from a
    pooled conditioning summary.
    """

    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.to_scale = nn.Linear(context_dim, channels)
        self.to_shift = nn.Linear(context_dim, channels)

    def forward(self, x, context):
        scale = self.to_scale(context).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(context).unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1.0 + scale) + shift


class WeekHead(nn.Module):
    """
    Stronger per-week decoder head than a bare 1x1 projection.
    """

    def __init__(self, in_channels, out_channels, context_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, HEAD_HIDDEN, kernel_size=3, padding=1)
        self.film1 = FiLMModulation(HEAD_HIDDEN, context_dim)
        self.conv2 = nn.Conv2d(HEAD_HIDDEN, HEAD_BOTTLENECK, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(HEAD_BOTTLENECK), HEAD_BOTTLENECK)
        self.out = nn.Conv2d(HEAD_BOTTLENECK, out_channels, kernel_size=1)

    def forward(self, x, context):
        x = self.conv1(x)
        x = F.silu(self.film1(x, context))
        x = self.conv2(x)
        x = F.silu(self.norm2(x))
        return self.out(x)


class CircularLongitudeConv2d(nn.Module):
    """Conv2d with circular longitude padding and replicated latitude padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.pad_h = kernel_size[0] // 2
        self.pad_w = kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        if self.pad_w:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        if self.pad_h:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="replicate")
        return self.conv(x)


class GlobalSpatialContextEncoder(nn.Module):
    """Preserve coarse global maps as tokens while retaining a pooled FiLM summary."""

    def __init__(self, in_channels, context_dim, token_dim=GLOBAL_TOKEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            CircularLongitudeConv2d(in_channels, 32, kernel_size=5, stride=2),
            nn.GroupNorm(_num_groups(32), 32),
            nn.SiLU(),
            CircularLongitudeConv2d(32, 64, kernel_size=5, stride=2),
            nn.GroupNorm(_num_groups(64), 64),
            nn.SiLU(),
            CircularLongitudeConv2d(64, 96, kernel_size=3, stride=2),
            nn.GroupNorm(_num_groups(96), 96),
            nn.SiLU(),
        )
        self.to_tokens = nn.Conv2d(96, token_dim, kernel_size=1)
        self.position_projection = nn.Linear(4, token_dim)
        self.summary_projection = nn.Sequential(
            nn.Linear(token_dim, context_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _position_features(height, width, device, dtype):
        lat = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        lon = torch.arange(width, device=device, dtype=dtype) * (2.0 * torch.pi / width)
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
        return torch.stack(
            [lat_grid, torch.sin(lon_grid), torch.cos(lon_grid), torch.cos(lat_grid * torch.pi / 2.0)],
            dim=-1,
        ).reshape(height * width, 4)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        features = self.encoder(x)
        token_map = self.to_tokens(features)
        batch, channels, height, width = token_map.shape
        tokens = token_map.flatten(2).transpose(1, 2)
        position = self.position_projection(
            self._position_features(height, width, token_map.device, token_map.dtype)
        )
        tokens = tokens + position.unsqueeze(0)
        summary = self.summary_projection(tokens.mean(dim=1))
        return tokens, summary, (height, width)


class BottleneckCrossAttention(nn.Module):
    """Cross-attend local UNet bottleneck cells to coarse global spatial tokens."""

    def __init__(self, local_channels, token_dim=GLOBAL_TOKEN_DIM, num_heads=4, num_layers=1):
        super().__init__()
        self.query_projection = nn.Linear(local_channels, token_dim)
        self.output_projection = nn.Linear(token_dim, local_channels)
        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(token_dim),
                        "token_norm": nn.LayerNorm(token_dim),
                        "attention": nn.MultiheadAttention(
                            token_dim,
                            int(num_heads),
                            dropout=0.0,
                            batch_first=True,
                        ),
                        "ff_norm": nn.LayerNorm(token_dim),
                        "ff": nn.Sequential(
                            nn.Linear(token_dim, token_dim * 2),
                            nn.SiLU(),
                            nn.Linear(token_dim * 2, token_dim),
                        ),
                    }
                )
            )

    def forward(self, local_features, global_tokens):
        batch, channels, height, width = local_features.shape
        queries = self.query_projection(local_features.flatten(2).transpose(1, 2))
        for layer in self.layers:
            attended, _ = layer["attention"](
                layer["query_norm"](queries),
                layer["token_norm"](global_tokens),
                layer["token_norm"](global_tokens),
                need_weights=False,
            )
            queries = queries + attended
            queries = queries + layer["ff"](layer["ff_norm"](queries))
        update = self.output_projection(queries).transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return local_features + update


class FlowMatchingModel(nn.Module):
    """
    Rectified-flow UNet with lightweight FiLM conditioning, deep lead embeddings,
    and richer week heads.
    """

    def __init__(
        self,
        in_channels=37,
        out_channels=2,
        block_out_channels=None,
        sample_size=(181, 360),
        global_context_channels=0,
        static_geography=None,
        use_global_cross_attention=True,
        global_attention_heads=4,
        global_attention_layers=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.cond_channels = in_channels - out_channels
        if int(out_channels) != 2:
            raise ValueError("v9.5a separate PR/T2M heads require out_channels=2.")
        self.sample_size = tuple(int(v) for v in sample_size)
        self.global_context_channels = int(global_context_channels or 0)
        self.use_global_cross_attention = bool(use_global_cross_attention)
        if static_geography is None:
            static_geography = torch.zeros(
                5, self.sample_size[0], self.sample_size[1], dtype=torch.float32
            )
        static_geography = torch.as_tensor(static_geography, dtype=torch.float32)
        if static_geography.ndim != 3 or tuple(static_geography.shape[-2:]) != self.sample_size:
            raise ValueError(
                f"static_geography must be [C,{self.sample_size[0]},{self.sample_size[1]}], "
                f"got {tuple(static_geography.shape)}."
            )
        self.static_geography_channels = int(static_geography.shape[0])
        self.register_buffer("static_geography", static_geography.contiguous(), persistent=True)
        backbone_in_channels = in_channels + self.static_geography_channels
        if block_out_channels is None:
            block_out_channels = UNET_BLOCK_OUT_CHANNELS
        self.block_out_channels = tuple(int(c) for c in block_out_channels)
        if len(self.block_out_channels) != 4:
            raise ValueError(
                f"Expected 4 UNet block widths, got {self.block_out_channels}."
            )
        if self.cond_channels <= 0:
            raise ValueError(
                f"Expected conditioning channels in addition to {out_channels} flow channels, "
                f"got in_channels={in_channels}."
            )

        context_input_dim = self.cond_channels * 2
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, CONTEXT_HIDDEN),
            nn.SiLU(),
            nn.Linear(CONTEXT_HIDDEN, CONTEXT_HIDDEN),
            nn.SiLU(),
        )
        self.global_context_encoder = (
            GlobalSpatialContextEncoder(self.global_context_channels, CONTEXT_HIDDEN)
            if self.global_context_channels > 0 else None
        )
        self.lead_embedding = nn.Embedding(4, CONTEXT_HIDDEN)
        self.input_adapter = nn.Conv2d(backbone_in_channels, backbone_in_channels, kernel_size=1)
        self.input_film = FiLMModulation(backbone_in_channels, CONTEXT_HIDDEN)

        # U-Net backbone with class embedding conditioning for lead week
        self.unet = UNet2DModel(
            sample_size=self.sample_size,
            in_channels=backbone_in_channels,
            out_channels=HEAD_FEATURES,  # Intermediate feature space
            layers_per_block=2,
            block_out_channels=self.block_out_channels,
            down_block_types=(
                "DownBlock2D",       # 181x360
                "DownBlock2D",       # 90x180
                "AttnDownBlock2D",   # 45x90
                "AttnDownBlock2D",   # 22x45
            ),
            up_block_types=(
                "AttnUpBlock2D",     # 22x45
                "AttnUpBlock2D",     # 45x90
                "UpBlock2D",         # 90x180
                "UpBlock2D",         # 181x360
            ),
            class_embed_type="timestep",
            num_class_embeds=4,
        )

        self.feature_film = FiLMModulation(HEAD_FEATURES, CONTEXT_HIDDEN)
        self.global_context = nn.Linear(CONTEXT_HIDDEN, HEAD_FEATURES)
        self.bottleneck_cross_attention = BottleneckCrossAttention(
            self.block_out_channels[-1],
            token_dim=GLOBAL_TOKEN_DIM,
            num_heads=global_attention_heads,
            num_layers=global_attention_layers,
        )
        self._active_global_tokens = None
        self._last_global_token_shape = None
        self._mid_block_hook = self.unet.mid_block.register_forward_hook(
            self._apply_bottleneck_cross_attention
        )

        self.pr_heads = nn.ModuleList([
            WeekHead(HEAD_FEATURES, 1, CONTEXT_HIDDEN) for _ in range(4)
        ])
        self.t2m_heads = nn.ModuleList([
            WeekHead(HEAD_FEATURES, 1, CONTEXT_HIDDEN) for _ in range(4)
        ])
        self.pr_var_heads = nn.ModuleList([
            WeekHead(HEAD_FEATURES, 1, CONTEXT_HIDDEN) for _ in range(4)
        ])
        self.t2m_var_heads = nn.ModuleList([
            WeekHead(HEAD_FEATURES, 1, CONTEXT_HIDDEN) for _ in range(4)
        ])
        self.out_channels = out_channels

    def _build_context(self, x_cond, lead_idx=None, global_context=None):
        cond_mean = x_cond.mean(dim=(2, 3))
        cond_var = x_cond.var(dim=(2, 3), unbiased=False)
        cond_std = torch.sqrt(cond_var + 1e-6)
        context = self.context_mlp(torch.cat([cond_mean, cond_std], dim=1))

        global_tokens = None
        if self.global_context_encoder is not None:
            if global_context is None:
                global_add = torch.zeros_like(context)
            else:
                if global_context.shape[1] != self.global_context_channels:
                    raise ValueError(
                        f"Expected {self.global_context_channels} global context channels, "
                        f"got {global_context.shape[1]}."
                    )
                global_tokens, global_add, token_grid = self.global_context_encoder(
                    global_context.to(dtype=x_cond.dtype)
                )
                self._last_global_token_shape = (
                    int(global_tokens.shape[1]),
                    int(global_tokens.shape[2]),
                    int(token_grid[0]),
                    int(token_grid[1]),
                )
            context = context + global_add

        if lead_idx is None:
            lead_emb = self.lead_embedding.weight[0].unsqueeze(0).expand(x_cond.shape[0], -1)
        else:
            lead_emb = self.lead_embedding(lead_idx.long())
        return context + lead_emb, global_tokens

    def _apply_bottleneck_cross_attention(self, module, inputs, output):
        del module, inputs
        if (
            not self.use_global_cross_attention
            or self._active_global_tokens is None
            or output.shape[0] != self._active_global_tokens.shape[0]
        ):
            return output
        return self.bottleneck_cross_attention(output, self._active_global_tokens)

    @staticmethod
    def _combine_variable_heads(pr_head, t2m_head, features, context):
        return torch.cat(
            [pr_head(features, context), t2m_head(features, context)],
            dim=1,
        )

    def forward(self, x_t, x_cond, t, lead_idx=None, compute_variance=True, global_context=None):
        """
        x_t:      [B, out_channels, H, W] - State at time t
        x_cond:   [B, cond_channels, H, W] - Conditioning variables
        t:        [B] or scalar in [0, 1]. Representing continuous flow time.
        lead_idx: [B] tensor with values in {0, 1, 2, 3} indicating forecast week.
                  If None, defaults to head 0.
        compute_variance: If False, skip the variance heads entirely and return
                          ``(velocity, None)``.
        """
        context, global_tokens = self._build_context(
            x_cond, lead_idx=lead_idx, global_context=global_context
        )

        # Spatial concatenation (x_t is [B, 2, H, W], x_cond is [B, cond_channels, H, W]).
        static = self.static_geography.to(dtype=x_cond.dtype).unsqueeze(0).expand(
            x_cond.shape[0], -1, -1, -1
        )
        x = torch.cat([x_t, x_cond, static], dim=1)

        # Pad to multiple of 16 for 4 down-blocks
        orig_H, orig_W = x.shape[2], x.shape[3]
        target_H = ((orig_H + 15) // 16) * 16
        target_W = ((orig_W + 15) // 16) * 16
        pad_h = target_H - orig_H
        pad_w = target_W - orig_W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        x = self.input_adapter(x)
        x = F.silu(self.input_film(x, context))

        # Scale time for diffusers embedding
        t_scaled = t * 1000.0

        # Construct class label tensor for U-Net deep conditioning
        if lead_idx is None:
            unet_labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            unet_labels = lead_idx.long()

        # Shared UNet forward pass -> intermediate features, passing class_labels for deep conditioning
        self._active_global_tokens = global_tokens
        try:
            features = self.unet(x, t_scaled, class_labels=unet_labels).sample
        finally:
            self._active_global_tokens = None

        # Crop back to original resolution
        features = features[..., :orig_H, :orig_W]
        features = self.feature_film(features, context)
        features = F.silu(features + self.global_context(context).unsqueeze(-1).unsqueeze(-1))

        # Route through dedicated per-week output heads
        if lead_idx is None:
            output = self._combine_variable_heads(
                self.pr_heads[0], self.t2m_heads[0], features, context
            ).to(features.dtype)
            if not compute_variance:
                return output, None
            variance = self._combine_variable_heads(
                self.pr_var_heads[0], self.t2m_var_heads[0], features, context
            )
            return output, F.softplus(variance).to(features.dtype)

        B = features.shape[0]
        output = torch.zeros(B, self.out_channels, orig_H, orig_W, device=features.device, dtype=features.dtype)
        var_output = None
        if compute_variance:
            var_output = torch.zeros(B, self.out_channels, orig_H, orig_W, device=features.device, dtype=features.dtype)

        for week_idx in range(4):
            mask = (lead_idx == week_idx)
            if mask.any():
                week_context = context[mask]
                output[mask] = self._combine_variable_heads(
                    self.pr_heads[week_idx],
                    self.t2m_heads[week_idx],
                    features[mask],
                    week_context,
                ).to(features.dtype)
                if compute_variance:
                    var_output[mask] = F.softplus(
                        self._combine_variable_heads(
                            self.pr_var_heads[week_idx],
                            self.t2m_var_heads[week_idx],
                            features[mask],
                            week_context,
                        )
                    ).to(features.dtype)

        return output, var_output


class CustomFlowMatcher:
    """
    ODE solver and interpolation logic for Rectified Flow / Flow Matching.
    Calculates straight paths from Noise (t=0) to Data (t=1).
    """
    def __init__(self, device="cpu"):
        self.device = device

    def sample_time_batch(
        self,
        batch_size,
        schedule="uniform",
        beta_alpha=1.5,
        beta_beta=2.0,
        low_fraction=0.35,
        mid_fraction=0.50,
        eps=1e-4,
    ):
        """
        Samples flow times for training.
        """
        schedule = str(schedule or "uniform").lower()
        eps = float(eps)

        if schedule == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif schedule == "beta":
            alpha = max(float(beta_alpha), eps)
            beta = max(float(beta_beta), eps)
            dist = torch.distributions.Beta(
                torch.tensor(alpha, device=self.device),
                torch.tensor(beta, device=self.device),
            )
            t = dist.sample((batch_size,))
        elif schedule == "logit_normal":
            scale = max(float(beta_alpha), eps)
            t = torch.sigmoid(torch.randn((batch_size,), device=self.device) * scale)
        elif schedule == "stratified_low_mid":
            low_fraction = min(max(float(low_fraction), 0.0), 1.0)
            mid_fraction = min(max(float(mid_fraction), 0.0), 1.0 - low_fraction)
            low_count = int(round(batch_size * low_fraction))
            mid_count = int(round(batch_size * mid_fraction))
            high_count = max(0, batch_size - low_count - mid_count)

            parts = []
            if low_count > 0:
                parts.append(eps + torch.rand((low_count,), device=self.device) * (0.35 - eps))
            if mid_count > 0:
                parts.append(0.35 + torch.rand((mid_count,), device=self.device) * 0.40)
            if high_count > 0:
                parts.append(0.75 + torch.rand((high_count,), device=self.device) * (1.0 - eps - 0.75))
            t = torch.cat(parts, dim=0) if parts else torch.empty((0,), device=self.device)
            if t.numel() > 1:
                t = t[torch.randperm(t.numel(), device=self.device)]
        else:
            raise ValueError(f"Unknown flow time sampling schedule: {schedule}")

        return t.clamp(eps, 1.0 - eps)

    def eof_sample(self, eof_bases, mjo_phases, num_samples, H, W, lead_ids=None):
        """
        Sample physically structured noise from MJO-phase × lead-week EOF subspace.
        """
        noise = torch.zeros((num_samples, 2, H, W), device=self.device)

        for i in range(num_samples):
            b_idx = i % len(mjo_phases)
            phase = int(mjo_phases[b_idx])
            lead = int(lead_ids[b_idx]) if lead_ids is not None else None

            key = (phase, lead) if lead is not None else phase
            if key not in eof_bases:
                key = phase
            if key not in eof_bases:
                key = (0, lead) if lead is not None else 0

            if key in eof_bases and 'eofs' in eof_bases[key]:
                eofs = eof_bases[key]['eofs'].to(self.device)
                eigenvals = eof_bases[key]['eigenvalues'].to(self.device)
                K = eofs.shape[0]
                for c in range(2):
                    alpha = torch.randn(K, device=self.device) * torch.sqrt(eigenvals)
                    noise_field = torch.einsum('k,khw->hw', alpha, eofs)

                    std = noise_field.std()
                    if std > 1e-6:
                        noise_field = noise_field / std

                    noise[i, c] = noise_field
            else:
                for c in range(2):
                    noise[i, c] = torch.randn(H, W, device=self.device)

        return noise

    def interpolate(self, target, noise, t):
        """
        Constructs the intermediate state x_t linearly between noise and target.
        """
        t = t.view(-1, 1, 1, 1).to(target.device)
        x_t = t * target + (1.0 - t) * noise
        v_target = target - noise
        return x_t, v_target

    @torch.no_grad()
    def prepare_initial_state(self, model, noise, x_cond, lead_idx=None, apply_flow_variance=False,
                               variance_channels=None, variance_beta=1.0,
                               variance_coarse_kernel=None, global_context=None):
        """
        Build the solver initial state x_0.
        """
        if isinstance(variance_beta, (list, tuple)):
            beta_vals = [float(max(0.0, min(1.0, b))) for b in variance_beta]
            beta = torch.as_tensor(
                beta_vals, device=noise.device, dtype=noise.dtype
            ).view(1, -1, 1, 1)
            beta_debug = beta_vals
        else:
            beta_scalar = float(max(0.0, min(1.0, variance_beta)))
            beta = beta_scalar
            beta_debug = beta_scalar
        debug = {
            "apply_flow_variance": bool(apply_flow_variance),
            "variance_beta": beta_debug,
            "variance_coarse_kernel": variance_coarse_kernel,
        }

        if not apply_flow_variance:
            x_t = noise.clone()
            debug["x_t_init"] = x_t
            return x_t, debug

        t_zero = torch.zeros((noise.shape[0],), device=noise.device, dtype=torch.float32)
        _, var_pred = model(noise, x_cond, t_zero, lead_idx=lead_idx, global_context=global_context)

        std_pred = torch.sqrt(var_pred + 1e-6)
        std_pred = torch.clamp(std_pred, min=0.1, max=2.0)
        std_pred_raw = std_pred

        if variance_channels is not None:
            channel_mask = torch.as_tensor(
                variance_channels, device=std_pred.device, dtype=torch.bool
            ).view(1, -1, 1, 1)
            if channel_mask.shape[1] != std_pred.shape[1]:
                raise ValueError(
                    f"variance_channels length {channel_mask.shape[1]} does not match "
                    f"model channel count {std_pred.shape[1]}"
                )
            std_pred = torch.where(channel_mask, std_pred, torch.ones_like(std_pred))

        if variance_coarse_kernel is not None:
            kernel = int(variance_coarse_kernel)
            if kernel > 1:
                coarse = F.avg_pool2d(
                    std_pred.float(),
                    kernel_size=kernel,
                    stride=kernel,
                    ceil_mode=True,
                    count_include_pad=False,
                )
                std_pred = F.interpolate(
                    coarse,
                    size=std_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).to(std_pred.dtype)
                std_pred = torch.clamp(std_pred, min=0.1, max=2.0)

        std_eff = 1.0 + beta * (std_pred - 1.0)
        x_t = noise * std_eff

        debug["std_pred_raw"] = std_pred_raw
        debug["std_pred"] = std_pred
        debug["std_eff"] = std_eff
        debug["x_t_init"] = x_t
        return x_t, debug

    @torch.no_grad()
    def euler_solve(self, model, noise, x_cond, num_steps=10, lead_idx=None, apply_flow_variance=False,
                    variance_channels=None, variance_beta=1.0,
                    variance_coarse_kernel=None, return_debug=False, global_context=None):
        """
        Inference routine using explicit Euler integration.
        """
        x_t, debug = self.prepare_initial_state(
            model,
            noise,
            x_cond,
            lead_idx=lead_idx,
            apply_flow_variance=apply_flow_variance,
            variance_channels=variance_channels,
            variance_beta=variance_beta,
            variance_coarse_kernel=variance_coarse_kernel,
            global_context=global_context,
        )

        dt = 1.0 / num_steps

        pbar = tqdm(range(num_steps), desc="ODE Solve", leave=False, disable=num_steps < 20)
        for step in pbar:
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)

            v_pred, _ = model(
                x_t,
                x_cond,
                t,
                lead_idx=lead_idx,
                compute_variance=False,
                global_context=global_context,
            )

            x_t = x_t + v_pred * dt

        if return_debug:
            debug["x_t_final"] = x_t
            return x_t, debug
        return x_t

    def euler_solve_differentiable(
        self,
        model,
        noise,
        x_cond,
        num_steps=10,
        lead_idx=None,
        use_checkpoint=False,
        global_context=None,
    ):
        """
        Training-time Euler integration with gradients enabled.
        """
        x_t = noise
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            if use_checkpoint:
                if global_context is None:
                    def velocity_only(x_state, cond_state, t_state, lead_state):
                        v_pred, _ = model(
                            x_state,
                            cond_state,
                            t_state,
                            lead_idx=lead_state,
                            compute_variance=False,
                        )
                        return v_pred
                    v_pred = checkpoint(
                        velocity_only,
                        x_t,
                        x_cond,
                        t,
                        lead_idx,
                        use_reentrant=False,
                    )
                else:
                    def velocity_only(x_state, cond_state, t_state, lead_state, global_state):
                        v_pred, _ = model(
                            x_state,
                            cond_state,
                            t_state,
                            lead_idx=lead_state,
                            compute_variance=False,
                            global_context=global_state,
                        )
                        return v_pred
                    v_pred = checkpoint(
                        velocity_only,
                        x_t,
                        x_cond,
                        t,
                        lead_idx,
                        global_context,
                        use_reentrant=False,
                    )
            else:
                v_pred, _ = model(
                    x_t,
                    x_cond,
                    t,
                    lead_idx=lead_idx,
                    compute_variance=False,
                    global_context=global_context,
                )
            x_t = x_t + v_pred * dt
        return x_t
