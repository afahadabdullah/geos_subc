import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

class FlowMatchingModel(nn.Module):
    """
    Wrapper for diffusers.UNet2DModel adapted for Rectified Flow / Flow Matching.
    Predicts the velocity vector field pushing noise towards target precipitation.
    """
    def __init__(self, in_channels=32, out_channels=1):
        super().__init__()
        
        # Professional UNet backbone
        # We use UNet2DModel because the conditioning is spatial (concatenated)
        # in_channels=32: 1 noisy target + 31 conditioning (24 Obs/Dev + 4 GEOS + 2 Month + 1 LeadIdx)
        self.unet = UNet2DModel(
            sample_size=(181, 360),
            in_channels=in_channels,
            out_channels=out_channels,
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

    def forward(self, x_t, x_cond, t):
        """
        x_t: [B, 1, H, W] - State at time t
        x_cond:  [B, 31, H, W] - Conditioning variables
        t: [B] or scalar in [0, 1]. Representing continuous flow time.
        """
        # Maintain spatial concatenation logic exactly as before
        x = torch.cat([x_t, x_cond], dim=1) # [B, 32, H, W]
        
        # Pad bounds to handle division by 16 (since we have 4 down-blocks)
        orig_H, orig_W = x.shape[2], x.shape[3]
        target_H = ((orig_H + 15) // 16) * 16
        target_W = ((orig_W + 15) // 16) * 16
        pad_h = target_H - orig_H
        pad_w = target_W - orig_W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # Diffusers UNet2DModel expects a timestep scaling. 
        # For flow matching (t in [0,1]), we scale to [0, 1000] for the time embedding.
        t_scaled = t * 1000.0

        # Predict velocity via diffusers backbone
        velocity_pred = self.unet(x, t_scaled).sample
        
        # Crop back down to original resolution
        return velocity_pred[..., :orig_H, :orig_W]


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
    def euler_solve(self, model, noise, x_cond, num_steps=10):
        """
        Inference routine using explicit Euler integration.
        Solves the ODE dx/dt = v(x, t) from t=0 to t=1.
        noise is x_0 at t=0.
        """
        x_t = noise.clone()
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            # Current time t
            t_val = step * dt
            t = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.float32)
            
            # Predict velocity
            v_pred = model(x_t, x_cond, t)
            
            # Euler step
            x_t = x_t + v_pred * dt
            
        return x_t  # This is the estimated x_1 (Data)

