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
        # Concatenate condition
        # (B, C_in + C_cond, H, W)
        model_input = torch.cat([x, condition], dim=1)
        
        # Forward pass
        # sample is noise prediction (or v-prediction depending on config)
        return self.model(model_input, timesteps).sample

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
        
        # Initial noise
        latents = torch.randn(
            (batch_size, self.in_channels, condition.shape[2], condition.shape[3]),
            device=device,
            generator=generator
        )
        
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.forward(latents, condition, t)
            
            # Step
            latents = self.noise_scheduler.step(noise_pred, t.item(), latents).prev_sample
            
        return latents
