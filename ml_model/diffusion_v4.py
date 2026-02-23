import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler

class DiffusionModelV4(nn.Module):
    """
    Wrapper for diffusers.UNet2DModel that maintains the V4 spatial conditioning API.
    Professional backbone replacing the scratch-built UNet.
    """
    def __init__(self, in_channels=34, out_channels=4):
        super().__init__()
        
        # Professional UNet backbone with industry-standard config
        # We use UNet2DModel because the conditioning is spatial (concatenated)
        self.unet = UNet2DModel(
            sample_size=(181, 360),
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",       # 181x360
                "DownBlock2D",       # 90x180 (Optimized: Removed expensive high-res attention)
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

    def forward(self, x_noisy, x_cond, timestep):
        """
        x_noisy: [B, 4, H, W]
        x_cond:  [B, 30, H, W]
        """
        # 1. Maintain spatial concatenation logic exactly as before
        x = torch.cat([x_noisy, x_cond], dim=1) # [B, 34, H, W]
        
        # 2. Pad bounds to handle division by 16 (since we have 4 down-blocks)
        orig_H, orig_W = x.shape[2], x.shape[3]
        target_H = ((orig_H + 15) // 16) * 16
        target_W = ((orig_W + 15) // 16) * 16
        pad_h = target_H - orig_H
        pad_w = target_W - orig_W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # 3. Predict noise via diffusers backbone
        # UNet2DModel expects (sample, timestep)
        noise_pred = self.unet(x, timestep).sample
        
        # 4. Crop back down to original resolution
        return noise_pred[..., :orig_H, :orig_W]

class CustomDiffusionScheduler:
    """
    Wrapper for diffusers.DDPMScheduler to maintain the scratch-built API.
    Provides mandatory x0-clipping for physical manifold stability.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=True, # MANDATORY for physical stability
            thresholding=True, # Prevent drift by clipping predicted x0
            dynamic_thresholding_ratio=0.995,
            prediction_type="epsilon"
        )
        
    def add_noise(self, original_samples, noise, timesteps):
        return self.scheduler.add_noise(original_samples, noise, timesteps)
        
    @torch.no_grad()
    def step(self, model_output, timestep, sample, clip_x0=True):
        # Diffusers typically indexes CPU-based scheduler tensors.
        # Ensure t is either an integer or a CPU tensor.
        t = timestep
        if torch.is_tensor(t):
            t = t.to("cpu")
            
        return self.scheduler.step(model_output, t, sample).prev_sample

    @torch.no_grad()
    def reconstruct_x0(self, model_output, timestep, sample):
        """
        Calculates the x0 estimate (denoised image) from the noise prediction.
        """
        t = timestep
        # If t is a tensor and all values are the same, use a scalar to avoid
        # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
        # in some diffusers scheduler versions.
        if torch.is_tensor(t):
            if t.dim() > 0 and torch.all(t == t[0]):
                t = t[0].item()
            else:
                t = t.to("cpu")
                
        return self.scheduler.step(model_output, t, sample).pred_original_sample
