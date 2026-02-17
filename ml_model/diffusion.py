import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler

class ConditionalDiffusion(nn.Module):
    """
    Conditional Diffusion Model using diffusers.UNet2DModel.
    Condition (GEOS + Obs) is concatenated with the noisy target along the channel dimension.
    """
    def __init__(self, 
                 in_channels=1, 
                 condition_channels=7, 
                 out_channels=1, 
                 block_out_channels=(64, 128, 256, 512), 
                 layers_per_block=2,
                 num_train_timesteps=1000):
        super().__init__()
        
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.total_channels = in_channels + condition_channels
        
        self.model = UNet2DModel(
            sample_size=None, # Flexible size
            in_channels=self.total_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        
        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        
    def forward(self, x, condition, timesteps):
        """
        Predict noise for diffusion step t.
        Args:
            x: Noisy input (B, C_in, H, W)
            condition: Conditioning data (B, C_cond, H, W)
            timesteps: Time step indices (B,)
        Returns:
            noise_pred: Predicted noise (B, C_out, H, W)
        """
        # Handle odd dimensions by padding to multiple of 8 (3 downsamples)
        # Input is (181, 360). 181 is not divisible by 8.
        # Pad H to 184 (181 + 3). W (360) is fine.
        
        H, W = x.shape[2], x.shape[3]
        target_H = ((H // 8) + 1) * 8 if H % 8 != 0 else H
        target_W = ((W // 8) + 1) * 8 if W % 8 != 0 else W
        
        pad_h = target_H - H
        pad_w = target_W - W
        
        # Pad (left, right, top, bottom)
        # We pad bottom and right
        padding = (0, pad_w, 0, pad_h) # (W_left, W_right, H_top, H_bottom)
        
        if pad_h > 0 or pad_w > 0:
            x_padded = torch.nn.functional.pad(x, padding, mode='replicate')
            cond_padded = torch.nn.functional.pad(condition, padding, mode='replicate')
        else:
            x_padded = x
            cond_padded = condition
            
        # Concatenate condition
        # (B, C_in + C_cond, H, W)
        model_input = torch.cat([x_padded, cond_padded], dim=1)
        
        # Forward pass
        # sample is noise prediction (or v-prediction depending on config)
        out = self.model(model_input, timesteps).sample
        
        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[..., :H, :W]
            
        return out

    @torch.no_grad()
    def sample(self, condition, num_inference_steps=50, generator=None):
        """
        Generate samples from noise conditioned on input.
        Args:
            condition: (B, C_cond, H, W)
            num_inference_steps: Number of steps for sampling
        Returns:
            samples: (B, C_out, H, W)
        """
        batch_size = condition.shape[0]
        device = condition.device
        H, W = condition.shape[2], condition.shape[3]
        
        # Padding
        target_H = ((H // 8) + 1) * 8 if H % 8 != 0 else H
        target_W = ((W // 8) + 1) * 8 if W % 8 != 0 else W
        
        pad_h = target_H - H
        pad_w = target_W - W
        padding = (0, pad_w, 0, pad_h)
        
        if pad_h > 0 or pad_w > 0:
            cond_padded = torch.nn.functional.pad(condition, padding, mode='replicate')
            final_H, final_W = target_H, target_W
        else:
            cond_padded = condition
            final_H, final_W = H, W
        
        # Initial noise matches PADDED size
        latents = torch.randn(
            (batch_size, self.in_channels, final_H, final_W),
            device=device,
            generator=generator
        )
        
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict noise (using padded inputs)
            # We need to manually concatenate here or call a modified forward?
            # Calling self.forward would pad AGAIN if we passed unpadded tensors.
            # But here we are working in PADDED space.
            # So we should call model directly or handle concatenation manually here.
            
            # Since latents and cond_padded are ALREADY padded, we just concat and run model.
            
            model_input = torch.cat([latents, cond_padded], dim=1)
            noise_pred = self.model(model_input, t).sample
            
            # Step
            latents = self.noise_scheduler.step(noise_pred, t.item(), latents).prev_sample
            
        # Crop at the end
        if pad_h > 0 or pad_w > 0:
            latents = latents[..., :H, :W]
            
        return latents
