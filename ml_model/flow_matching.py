import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
from tqdm.auto import tqdm

# Number of intermediate feature channels between shared UNet and per-week heads
HEAD_FEATURES = 64

class FlowMatchingModel(nn.Module):
    """
    Wrapper for diffusers.UNet2DModel adapted for Rectified Flow / Flow Matching.
    
    Multi-Head Architecture:
      - Shared UNet encoder/decoder outputs [B, HEAD_FEATURES, H, W] intermediate features.
      - 4 dedicated Conv2d(HEAD_FEATURES, 1) output heads, one per forecast week.
      - Each head specializes in its own timescale without competing for filter capacity.
    """
    def __init__(self, in_channels=37, out_channels=2):
        super().__init__()
        
        # Shared UNet backbone (outputs intermediate features, NOT final prediction)
        self.unet = UNet2DModel(
            sample_size=(181, 360),
            in_channels=in_channels,
            out_channels=HEAD_FEATURES,  # Intermediate feature space
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
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
        
        # 4 dedicated mean output heads (Week 1, Week 2, Week 3, Week 4)
        # Each is a lightweight 1x1 conv: [B, 64, H, W] -> [B, out_channels, H, W]
        self.heads = nn.ModuleList([
            nn.Conv2d(HEAD_FEATURES, out_channels, kernel_size=1) for _ in range(4)
        ])
        
        # 4 dedicated variance output heads
        self.var_heads = nn.ModuleList([
            nn.Conv2d(HEAD_FEATURES, out_channels, kernel_size=1) for _ in range(4)
        ])
        
        self.out_channels = out_channels

    def forward(self, x_t, x_cond, t, lead_idx=None):
        """
        x_t:      [B, out_channels, H, W] - State at time t
        x_cond:   [B, 35, H, W] - Conditioning variables
        t:        [B] or scalar in [0, 1]. Representing continuous flow time.
        lead_idx: [B] tensor with values in {0, 1, 2, 3} indicating forecast week.
                  If None, defaults to head 0 (backward compat).
        """
        # Spatial concatenation (x_t is [B, 2, H, W], x_cond is [B, 35, H, W] -> [B, 37, H, W])
        x = torch.cat([x_t, x_cond], dim=1)
        
        # Pad to multiple of 16 for 4 down-blocks
        orig_H, orig_W = x.shape[2], x.shape[3]
        target_H = ((orig_H + 15) // 16) * 16
        target_W = ((orig_W + 15) // 16) * 16
        pad_h = target_H - orig_H
        pad_w = target_W - orig_W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # Scale time for diffusers embedding
        t_scaled = t * 1000.0

        # Shared UNet forward pass -> intermediate features
        features = self.unet(x, t_scaled).sample  # [B, 64, H, W]
        
        # Crop back to original resolution
        features = features[..., :orig_H, :orig_W]
        
        # Route through dedicated per-week output heads
        if lead_idx is None:
            # Backward compatibility: use head 0
            return self.heads[0](features), F.softplus(self.var_heads[0](features))
        
        B = features.shape[0]
        output = torch.zeros(B, self.out_channels, orig_H, orig_W, device=features.device, dtype=features.dtype)
        var_output = torch.zeros(B, self.out_channels, orig_H, orig_W, device=features.device, dtype=features.dtype)
        
        for week_idx in range(4):
            mask = (lead_idx == week_idx)
            if mask.any():
                output[mask] = self.heads[week_idx](features[mask])
                # Softplus ensures strict positive variance predictions, but usually upcasts to Float32.
                # We force cast it back to the original mixed-precision format (fp16) to avoid assignment crashes.
                var_output[mask] = F.softplus(self.var_heads[week_idx](features[mask])).to(features.dtype)
        
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
    def euler_solve(self, model, noise, x_cond, num_steps=10, lead_idx=None, apply_flow_variance=False):
        """
        Inference routine using explicit Euler integration.
        Solves the ODE dx/dt = v(x, t) from t=0 to t=1.
        noise is x_0 standard normal initialization.
        lead_idx: integer or [B] tensor indicating which week head to use.
        apply_flow_variance: If True, queries the model's var_head at t=0 and 
                             scales the initial noise by sqrt(var_pred).
        """
        if apply_flow_variance:
            # Query variance at t=0
            t_zero = torch.zeros((noise.shape[0],), device=noise.device, dtype=torch.float32)
            _, var_pred = model(noise, x_cond, t_zero, lead_idx=lead_idx)
            # Standard Deviation = sqrt(Variance). Small epsilon for numerical stability.
            std_pred = torch.sqrt(var_pred + 1e-6)
            # Clamp to prevent runaway ensemble divergence or collapse
            # min=0.1 prevents ensemble collapse, max=2.0 prevents explosive noise
            std_pred = torch.clamp(std_pred, min=0.1, max=2.0)
            # Flow-Dependent scaling of initial condition
            x_t = noise * std_pred
        else:
            x_t = noise.clone()
            
        dt = 1.0 / num_steps
        
        # Only show inner progress bar for very long samplings
        pbar = tqdm(range(num_steps), desc="ODE Solve", leave=False, disable=num_steps < 20)
        for step in pbar:
            # Current time t
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            
            # Predict velocity through the correct per-week head (we ignore variance during integration)
            v_pred, _ = model(x_t, x_cond, t, lead_idx=lead_idx)
            
            # Euler step
            x_t = x_t + v_pred * dt
            
        return x_t  # This is the estimated x_1 (Data)

