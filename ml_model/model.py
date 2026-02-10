import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# MODELS (Copied from trainv9.py)
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, emb):
        h = self.norm1(x)
        h = F.leaky_relu(h)
        h = self.conv1(h)
        # Broadcast embedding (B, C) -> (B, C, 1, 1)
        emb_out = self.emb_proj(emb)[:, :, None, None]
        h = h + emb_out
        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    """
    V8/V9 UNet Architecture from user's trainv9.py.
    """
    def __init__(self, in_channels, out_channels, base_filters=64, emb_dim=256):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(dim=128)
        # One-hot month projection (12 months -> 128 dim)
        self.month_proj = nn.Linear(12, 128)
        self.cond_mlp = nn.Sequential(nn.Linear(256, emb_dim), nn.LeakyReLU())
        
        # In trainv9, input to conv_in is cat([noisy_target, condition])
        # So in_channels should be sum of both.
        self.conv_in = nn.Conv2d(in_channels, base_filters, 3, padding=1)
        
        self.down1 = ResBlock(base_filters, base_filters, emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(base_filters, base_filters * 2, emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResBlock(base_filters * 2, base_filters * 4, emb_dim)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck1 = ResBlock(base_filters * 4, base_filters * 8, emb_dim)
        self.bottleneck2 = ResBlock(base_filters * 8, base_filters * 8, emb_dim)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ResBlock(base_filters * 8 + base_filters * 4, base_filters * 4, emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ResBlock(base_filters * 4 + base_filters * 2, base_filters * 2, emb_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ResBlock(base_filters * 2 + base_filters, base_filters, emb_dim)
        
        self.conv_out = nn.Conv2d(base_filters, out_channels, 1)
    
    def forward(self, x_input, timesteps, month_onehot):
        # x_input is already cat([noisy_target, condition])
        
        t_emb = self.time_emb(timesteps)
        m_emb = self.month_proj(month_onehot)
        cond_emb = self.cond_mlp(torch.cat([t_emb, m_emb], dim=1))
        
        x = self.conv_in(x_input)
        
        s1 = self.down1(x, cond_emb); x = self.pool1(s1)
        s2 = self.down2(x, cond_emb); x = self.pool2(s2)
        s3 = self.down3(x, cond_emb); x = self.pool3(s3)
        
        x = self.bottleneck1(x, cond_emb)
        x = self.bottleneck2(x, cond_emb)
        
        x = self.up3(x)
        # Handle shape mismatch due to pooling/padding (same as V9 logic)
        if x.shape[2:] != s3.shape[2:]:
            x = F.interpolate(x, size=s3.shape[2:], mode='bilinear')
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x, cond_emb)
        
        x = self.up2(x)
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:], mode='bilinear')
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x, cond_emb)
        
        x = self.up1(x)
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:], mode='bilinear')
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x, cond_emb)
        
        return self.conv_out(x)

# ==============================================================================
# DIFFUSION
# ==============================================================================

class GaussianDiffusion:
    """Manages the noise schedule for the diffusion process."""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alpha_hats = torch.sqrt(self.alpha_hats)
        self.sqrt_one_minus_alpha_hats = torch.sqrt(1.0 - self.alpha_hats)
    
    def add_noise(self, original_images, timesteps):
        """Add noise to images at given timesteps. Returns (noisy_images, noise)."""
        # Expand for broadcasting [B, 1, 1, 1]
        sqrt_alpha_hat_t = self.sqrt_alpha_hats[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_images)
        noisy_images = sqrt_alpha_hat_t * original_images + sqrt_one_minus_alpha_hat_t * noise
        
        return noisy_images, noise
    
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)
