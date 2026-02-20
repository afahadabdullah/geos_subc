"""
Conditional Diffusion Model v2 for S2S Precipitation Forecasting.

Uses diffusers.UNet2DModel as the backbone.
Scheduler: DDIMScheduler (for stable fast sampling, but training is noise prediction).

Differences from v1:
- Predicts exactly ONE lead week at a time.
- Uses `class_labels` to pass `lead_index` into the UNet using `num_class_embeds=4`.

Input:  noisy target (B, 1, H, W) + condition (B, 48, H, W) = 49 channels
Output: predicted noise ε (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDIMScheduler


class ConditionalDiffusionV2(nn.Module):
    """
    Conditional Diffusion Model predicting one lead week at a time.
    """
    def __init__(self, 
                 in_channels=1, 
                 condition_channels=48, 
                 out_channels=1, 
                 block_out_channels=(64, 128, 256, 512), 
                 layers_per_block=2,
                 num_train_timesteps=1000):
        super().__init__()
        
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.out_channels = out_channels
        self.total_channels = in_channels + condition_channels
        
        self.model = UNet2DModel(
            sample_size=None,  
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
            class_embed_type="timestep",
            num_class_embeds=4
        )
        
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps, 
            clip_sample=False,
            prediction_type="epsilon"
        )
        
    def _pad_to_multiple(self, x, multiple=8):
        """Pad spatial dims to next multiple of 8."""
        H, W = x.shape[2], x.shape[3]
        target_H = ((H + multiple - 1) // multiple) * multiple
        target_W = ((W + multiple - 1) // multiple) * multiple
        pad_h = target_H - H
        pad_w = target_W - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x, H, W
        
    def forward(self, x, condition, lead_indices, timesteps):
        """
        Predict noise.
        Args:
            x: Noisy target for ONE lead (B, 1, H, W)
            condition: Conditioning data (B, 48, H, W)
            lead_indices: Timestep index of lead [0, 1, 2, 3] (B,)
            timesteps: Diffusion timestep indices (B,)
        """
        x_padded, orig_H, orig_W = self._pad_to_multiple(x)
        cond_padded, _, _ = self._pad_to_multiple(condition)
        
        model_input = torch.cat([x_padded, cond_padded], dim=1)
        
        out = self.model(model_input, timesteps, class_labels=lead_indices).sample
        
        out = out[..., :orig_H, :orig_W]
        return out

    @torch.no_grad()
    def sample(self, condition, lead_indices, num_inference_steps=50, generator=None, verbose=False):
        """
        Generate samples from noise conditioned on input using DDIM.
        Needs condition (B, 48, H, W) and corresponding lead_indices (B,).
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        orig_H, orig_W = condition.shape[2], condition.shape[3]
        
        latents = torch.randn(
            (batch_size, self.in_channels, orig_H, orig_W),
            device=device,
            generator=generator
        )
        
        self.noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.noise_scheduler.timesteps
        
        if verbose:
            from tqdm.auto import tqdm
            timesteps = tqdm(timesteps, desc="Sampling", leave=False)
            
        for t in timesteps:
            t_batched = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.forward(latents, condition, lead_indices, t_batched)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents
