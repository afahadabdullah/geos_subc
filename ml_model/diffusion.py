"""
Conditional Diffusion Model for S2S Precipitation Forecasting.

Uses diffusers.UNet2DModel as the backbone.
Scheduler: DDIMScheduler (for stable fast sampling, but training is noise prediction).
Condition (GEOS + Obs + MJO + Seasonality) is concatenated with the
noisy target along the channel dimension. CMDE adds reduced noise
to the condition to prevent trivial copying.

Input:  noisy_target (B, 4, H, W) + condition (B, 49, H, W) = 53 channels
Output: predicted noise ε (B, 4, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDIMScheduler
from tqdm import tqdm


class ConditionalDiffusion(nn.Module):
    """
    Conditional Diffusion Model using diffusers.UNet2DModel.
    
    Condition (GEOS + Obs + Seasonality + MJO + Topo) is concatenated with the 
    noisy target along the channel dimension.
    
    CMDE: During training, reduced noise (cmde_ratio * sqrt(1-α_t)) is 
    added to the condition channels to prevent the model from trivially
    copying the condition to the output.
    """
    def __init__(self, 
                 in_channels=4, 
                 condition_channels=49, 
                 out_channels=4, 
                 block_out_channels=(64, 128, 256, 512), 
                 layers_per_block=2,
                 num_train_timesteps=1000,
                 cmde_ratio=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.out_channels = out_channels
        self.total_channels = in_channels + condition_channels
        self.cmde_ratio = cmde_ratio
        
        self.model = UNet2DModel(
            sample_size=None,  # Flexible spatial size
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
        
        # Noise Scheduler: DDIM for fast sampling steps
        # clip_sample=False is crucial as our z-scored data exceeds [-1, 1]
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps, 
            clip_sample=False,
            prediction_type="epsilon"
        )
        
    def _pad_to_multiple(self, x, multiple=8):
        """Pad spatial dims to next multiple of 8 (needed for 3 downsamples)."""
        H, W = x.shape[2], x.shape[3]
        target_H = ((H + multiple - 1) // multiple) * multiple
        target_W = ((W + multiple - 1) // multiple) * multiple
        pad_h = target_H - H
        pad_w = target_W - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x, H, W  # Return original dims for cropping
        
    def forward(self, x, condition, timesteps):
        """
        Predict noise for diffusion step t.
        Args:
            x: Noisy target (B, in_channels, H, W)
            condition: Conditioning data (B, condition_channels, H, W)
            timesteps: Timestep indices (B,)
        Returns:
            noise_pred: Predicted noise (B, out_channels, H, W)
        """
        # Pad to multiple of 8
        x_padded, orig_H, orig_W = self._pad_to_multiple(x)
        cond_padded, _, _ = self._pad_to_multiple(condition)
        
        # CMDE: Add noise to condition during training
        if self.training and self.cmde_ratio > 0:
            noise = torch.randn_like(cond_padded)
            # Mix clean condition with noise
            cond_padded = (1 - self.cmde_ratio) * cond_padded + self.cmde_ratio * noise
            
        # Concatenate condition
        model_input = torch.cat([x_padded, cond_padded], dim=1)
        
        # Forward pass
        out = self.model(model_input, timesteps).sample
        
        # Crop back to original size
        out = out[..., :orig_H, :orig_W]
            
        return out

    @torch.no_grad()
    def sample(self, condition, num_inference_steps=50, generator=None, verbose=False):
        """
        Generate samples from noise conditioned on input using DDIM.
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        orig_H, orig_W = condition.shape[2], condition.shape[3]
        
        # Initial noise on the valid UNPADDED domain
        latents = torch.randn(
            (batch_size, self.in_channels, orig_H, orig_W),
            device=device,
            generator=generator
        )
        
        # Set scheduler steps (DDIM handles strided sampling)
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        timesteps = self.noise_scheduler.timesteps
        if verbose:
            from tqdm.auto import tqdm
            timesteps = tqdm(timesteps, desc="Sampling", leave=False)
            
        for t in timesteps:
            # forward() automatically handles padding latents and condition,
            # runs the UNet, and crops the predicted noise back to orig_H, orig_W
            noise_pred = self.forward(latents, condition, t)
            
            # DDIM step strictly on the valid domain
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents
