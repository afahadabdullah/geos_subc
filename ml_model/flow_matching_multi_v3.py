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
UNET_BLOCK_OUT_CHANNELS = (128, 256, 512, 640)


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


class FlowMatchingModel(nn.Module):
    """
    Rectified-flow UNet with lightweight FiLM conditioning and richer week heads.

    v3 keeps the same external interface as multiv1, but improves how lead and
    slow-varying context influence the shared backbone.
    """

    def __init__(self, in_channels=37, out_channels=2):
        super().__init__()

        self.in_channels = in_channels
        self.cond_channels = in_channels - out_channels
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
        self.lead_embedding = nn.Embedding(4, CONTEXT_HIDDEN)
        self.input_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.input_film = FiLMModulation(in_channels, CONTEXT_HIDDEN)

        # Shared UNet backbone (outputs intermediate features, NOT final prediction)
        self.unet = UNet2DModel(
            sample_size=(181, 360),
            in_channels=in_channels,
            out_channels=HEAD_FEATURES,  # Intermediate feature space
            layers_per_block=2,
            # A modest v3 widening: only the deepest stage grows beyond v1.
            # This adds capacity where the feature maps are smallest, which is
            # the safest place to spend extra VRAM.
            block_out_channels=UNET_BLOCK_OUT_CHANNELS,
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
        )

        self.feature_film = FiLMModulation(HEAD_FEATURES, CONTEXT_HIDDEN)
        self.global_context = nn.Linear(CONTEXT_HIDDEN, HEAD_FEATURES)

        self.heads = nn.ModuleList([
            WeekHead(HEAD_FEATURES, out_channels, CONTEXT_HIDDEN) for _ in range(4)
        ])
        self.var_heads = nn.ModuleList([
            WeekHead(HEAD_FEATURES, out_channels, CONTEXT_HIDDEN) for _ in range(4)
        ])
        self.out_channels = out_channels

    def _build_context(self, x_cond, lead_idx=None):
        cond_mean = x_cond.mean(dim=(2, 3))
        cond_var = x_cond.var(dim=(2, 3), unbiased=False)
        cond_std = torch.sqrt(cond_var + 1e-6)
        context = self.context_mlp(torch.cat([cond_mean, cond_std], dim=1))

        if lead_idx is None:
            lead_emb = self.lead_embedding.weight[0].unsqueeze(0).expand(x_cond.shape[0], -1)
        else:
            lead_emb = self.lead_embedding(lead_idx.long())
        return context + lead_emb

    def forward(self, x_t, x_cond, t, lead_idx=None, compute_variance=True):
        """
        x_t:      [B, out_channels, H, W] - State at time t
        x_cond:   [B, cond_channels, H, W] - Conditioning variables
        t:        [B] or scalar in [0, 1]. Representing continuous flow time.
        lead_idx: [B] tensor with values in {0, 1, 2, 3} indicating forecast week.
                  If None, defaults to head 0 (backward compat).
        compute_variance: If False, skip the variance heads entirely and return
                          ``(velocity, None)``.
        """
        context = self._build_context(x_cond, lead_idx=lead_idx)

        # Spatial concatenation (x_t is [B, 2, H, W], x_cond is [B, cond_channels, H, W]).
        x = torch.cat([x_t, x_cond], dim=1)

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

        # Shared UNet forward pass -> intermediate features
        features = self.unet(x, t_scaled).sample

        # Crop back to original resolution
        features = features[..., :orig_H, :orig_W]
        features = self.feature_film(features, context)
        features = F.silu(features + self.global_context(context).unsqueeze(-1).unsqueeze(-1))

        # Route through dedicated per-week output heads
        if lead_idx is None:
            output = self.heads[0](features, context)
            if not compute_variance:
                return output, None
            return output, F.softplus(self.var_heads[0](features, context))

        B = features.shape[0]
        output = torch.zeros(B, self.out_channels, orig_H, orig_W, device=features.device, dtype=features.dtype)
        var_output = None
        if compute_variance:
            var_output = torch.zeros(B, self.out_channels, orig_H, orig_W, device=features.device, dtype=features.dtype)

        for week_idx in range(4):
            mask = (lead_idx == week_idx)
            if mask.any():
                week_context = context[mask]
                output[mask] = self.heads[week_idx](features[mask], week_context)
                if compute_variance:
                    var_output[mask] = F.softplus(
                        self.var_heads[week_idx](features[mask], week_context)
                    ).to(features.dtype)

        return output, var_output


class CustomFlowMatcher:
    """
    ODE solver and interpolation logic for Rectified Flow / Flow Matching.
    Calculates straight paths from Noise (t=0) to Data (t=1).
    """
    def __init__(self, device="cpu"):
        self.device = device

    def sample_time_batch(self, batch_size):
        """
        Samples t ~ U[0, 1] for training.
        """
        return torch.rand((batch_size,), device=self.device)

    def eof_sample(self, eof_bases, mjo_phases, num_samples, H, W, lead_ids=None):
        """
        Sample physically structured noise from MJO-phase × lead-week EOF subspace.
        
        Args:
            eof_bases: dict mapping (phase, lead) or phase -> {eofs, eigenvalues}
            mjo_phases: [B] tensor/list of MJO phases (0-8)
            num_samples: total noise fields to generate (B * num_ensemble)
            H, W: spatial dimensions (181, 360)
            lead_ids: [B] tensor/list of lead indices (0-3). If None, falls back to phase-only.
        
        Returns:
            noise: [num_samples, 2, H, W] structured noise tensor (PR and T2M channels)
        """
        noise = torch.zeros((num_samples, 2, H, W), device=self.device)
        
        for i in range(num_samples):
            b_idx = i % len(mjo_phases)
            phase = int(mjo_phases[b_idx])
            lead = int(lead_ids[b_idx]) if lead_ids is not None else None
            
            # Try (phase, lead) key first, then phase-only, then fallback
            key = (phase, lead) if lead is not None else phase
            if key not in eof_bases:
                key = phase  # Backward compat with phase-only format
            if key not in eof_bases:
                key = (0, lead) if lead is not None else 0  # Weak MJO fallback
            
            if key in eof_bases and 'eofs' in eof_bases[key]:
                eofs = eof_bases[key]['eofs'].to(self.device)
                eigenvals = eof_bases[key]['eigenvalues'].to(self.device)
                K = eofs.shape[0]
                for c in range(2):
                    alpha = torch.randn(K, device=self.device) * torch.sqrt(eigenvals)
                    noise_field = torch.einsum('k,khw->hw', alpha, eofs)
                    
                    # Normalize to unit variance
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
        x_t = t * target + (1 - t) * noise
        t should be shape [B] or broadcastable.
        """
        t = t.view(-1, 1, 1, 1).to(target.device)
        x_t = t * target + (1.0 - t) * noise
        # The true target velocity for loss is purely (target - noise)
        v_target = target - noise
        return x_t, v_target

    @torch.no_grad()
    def prepare_initial_state(self, model, noise, x_cond, lead_idx=None, apply_flow_variance=False,
                              variance_channels=None, variance_beta=1.0,
                              variance_coarse_kernel=None):
        """
        Build the solver initial state x_0.
        If apply_flow_variance is enabled, query the variance head at t=0 and
        temper its effect toward unit scale using variance_beta. Optionally
        replace the raw pixel-scale std field with a coarse averaged mask.
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
        _, var_pred = model(noise, x_cond, t_zero, lead_idx=lead_idx)

        # Standard deviation from predicted variance, clamped for stability.
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

        # beta=0 keeps unit scale, beta=1 applies the full variance head.
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
                    variance_coarse_kernel=None, return_debug=False):
        """
        Inference routine using explicit Euler integration.
        Solves the ODE dx/dt = v(x, t) from t=0 to t=1.
        noise is x_0 standard normal initialization.
        lead_idx: integer or [B] tensor indicating which week head to use.
        apply_flow_variance: If True, queries the model's var_head at t=0 and 
                             scales the initial noise by sqrt(var_pred).
        variance_channels: Optional iterable of booleans with length equal to channel count.
                           If provided, apply variance scaling only to selected channels.
        variance_beta: Temper variance scaling back toward 1.0. 0 disables the
                       variance effect, 1 applies the full predicted scale.
        variance_coarse_kernel: If provided and > 1, average-pool the predicted
                       std field onto a coarser grid, then upsample it back.
        return_debug: If True, also return initial-state diagnostics.
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
        )
            
        dt = 1.0 / num_steps
        
        # Only show inner progress bar for very long samplings
        pbar = tqdm(range(num_steps), desc="ODE Solve", leave=False, disable=num_steps < 20)
        for step in pbar:
            # Current time t
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            
            # Predict velocity through the correct per-week head (we ignore variance during integration)
            v_pred, _ = model(x_t, x_cond, t, lead_idx=lead_idx, compute_variance=False)
            
            # Euler step
            x_t = x_t + v_pred * dt
            
        if return_debug:
            debug["x_t_final"] = x_t
            return x_t, debug
        return x_t  # This is the estimated x_1 (Data)

    def euler_solve_differentiable(self, model, noise, x_cond, num_steps=10, lead_idx=None, use_checkpoint=False):
        """
        Training-time Euler integration with gradients enabled.
        This always starts from pure Gaussian noise and skips variance-head work.
        """
        x_t = noise
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            if use_checkpoint:
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
                v_pred, _ = model(x_t, x_cond, t, lead_idx=lead_idx, compute_variance=False)
            x_t = x_t + v_pred * dt
        return x_t
