import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

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
    def __init__(self, in_channels=34, out_channels=1):
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
        
        # 4 dedicated output heads (Week 1, Week 2, Week 3, Week 4)
        # Each is a lightweight 1x1 conv: [B, 64, H, W] -> [B, 1, H, W]
        self.heads = nn.ModuleList([
            nn.Conv2d(HEAD_FEATURES, 1, kernel_size=1) for _ in range(4)
        ])

    def forward(self, x_t, x_cond, t, lead_idx=None):
        """
        x_t:      [B, 1, H, W] - State at time t
        x_cond:   [B, 33, H, W] - Conditioning variables
        t:        [B] or scalar in [0, 1]. Representing continuous flow time.
        lead_idx: [B] tensor with values in {0, 1, 2, 3} indicating forecast week.
                  If None, defaults to head 0 (backward compat).
        """
        # Spatial concatenation
        x = torch.cat([x_t, x_cond], dim=1)  # [B, 34, H, W]
        
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
            return self.heads[0](features)
        
        B = features.shape[0]
        output = torch.zeros(B, 1, orig_H, orig_W, device=features.device, dtype=features.dtype)
        
        for week_idx in range(4):
            mask = (lead_idx == week_idx)
            if mask.any():
                output[mask] = self.heads[week_idx](features[mask])
        
        return output


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
    def euler_solve(self, model, noise, x_cond, num_steps=10, lead_idx=None):
        """
        Inference routine using explicit Euler integration.
        Solves the ODE dx/dt = v(x, t) from t=0 to t=1.
        noise is x_0 at t=0.
        lead_idx: integer or [B] tensor indicating which week head to use.
        """
        x_t = noise.clone()
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            # Current time t
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            
            # Predict velocity through the correct per-week head
            v_pred = model(x_t, x_cond, t, lead_idx=lead_idx)
            
            # Euler step
            x_t = x_t + v_pred * dt
            
        return x_t  # This is the estimated x_1 (Data)

