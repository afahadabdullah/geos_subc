import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(in_channels, 4, batch_first=True)
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        norm_x = self.norm(x).view(B, C, -1).permute(0, 2, 1)
        
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        out = x_flat + attn_out
        return out.permute(0, 2, 1).view(B, C, H, W)

class Block(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        if up:
            self.conv1 = nn.Conv2d(2 * in_c, out_c, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
            self.transform = nn.Conv2d(out_c, out_c, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        # Replacing BatchNorm with GroupNorm for smaller batch sizes and stabler diffusion mapping
        self.gnorm1 = nn.GroupNorm(8, out_c)
        self.gnorm2 = nn.GroupNorm(8, out_c)
        # Adding Dropout for high-variance linear stabilization
        self.dropout = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        
        # Conv block + Time inject
        h = self.gnorm1(self.relu(self.conv1(x)))
        h = h + time_emb
        h = self.dropout(h)
        h = self.gnorm2(self.relu(self.conv2(h)))
        
        # Transform (Up or Down)
        return self.transform(h)

class DiffusionModelV4(nn.Module):
    """
    Custom Unet built from scratch.
    Takes concatenated inputs: Noisy Target [B, 4, H, W] + Conditionals [B, 30, H, W]
    Predicts: Noise epsilon [B, 4, H, W]
    """
    def __init__(self, in_channels=34, out_channels=4, time_emb_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection mapping (maintains spatial dim)
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Downsample
        self.down1 = Block(64, 128, time_emb_dim)  # H/2, W/2
        self.down2 = Block(128, 256, time_emb_dim) # H/4, W/4
        self.down3 = Block(256, 512, time_emb_dim) # H/8, W/8

        # Bottleneck
        self.bottleneck_time = nn.Linear(time_emb_dim, 512)
        self.bottleneck_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.bottleneck_gn = nn.GroupNorm(8, 512)
        self.attn = SelfAttention(512)

        # Upsample (c is multiplied by 2 due to skip connections)
        self.up1 = Block(512, 256, time_emb_dim, up=True) # H/4, W/4
        self.up2 = Block(256, 128, time_emb_dim, up=True) # H/2, W/2
        self.up3 = Block(128, 64, time_emb_dim, up=True)  # H, W
        
        # Final output projection
        self.out = nn.Conv2d(64, out_channels, 1)

    def _pad_to_multiple(self, x, multiple=8):
        """Pad spatial dims to next multiple of 8 to handle UNet pooling."""
        H, W = x.shape[2], x.shape[3]
        target_H = ((H + multiple - 1) // multiple) * multiple
        target_W = ((W + multiple - 1) // multiple) * multiple
        pad_h = target_H - H
        pad_w = target_W - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x, orig_H, orig_W, pad_h, pad_w

    def forward(self, x_noisy, x_cond, timestep):
        """
        x_noisy: [B, 1, H, W]
        x_cond:  [B, 48, H, W]
        """
        # Embed time
        t = self.time_mlp(timestep)
        
        # Combine inputs
        x = torch.cat([x_noisy, x_cond], dim=1) # [B, 49, H, W]
        
        # Pad bounds to handle division by 8
        orig_H, orig_W = x.shape[2], x.shape[3]
        target_H = ((orig_H + 7) // 8) * 8
        target_W = ((orig_W + 7) // 8) * 8
        pad_h = target_H - orig_H
        pad_w = target_W - orig_W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # Initial Layer
        x0 = self.conv0(x)
        
        # Encoder (Down path)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        x3 = self.down3(x2, t)

        # Bottleneck
        t_bottle = F.relu(self.bottleneck_time(t))
        t_bottle = t_bottle[(...,) + (None,) * 2]
        x_bottle = self.bottleneck_gn(F.relu(self.bottleneck_conv(x3))) + t_bottle
        x_bottle = self.attn(x_bottle)

        # Decoder (Up path with skip connections)
        # Note: We concatenate the skip connection BEFORE the up block,
        # so up1 expects (512 + 512) -> 256.
        x_up1 = self.up1(torch.cat([x_bottle, x3], dim=1), t)
        
        # Resize if dimensions misaligned after transpose conv
        if x_up1.shape[2:] != x2.shape[2:]:
            x_up1 = F.interpolate(x_up1, size=x2.shape[2:], mode='nearest')
            
        x_up2 = self.up2(torch.cat([x_up1, x2], dim=1), t)
        
        if x_up2.shape[2:] != x1.shape[2:]:
            x_up2 = F.interpolate(x_up2, size=x1.shape[2:], mode='nearest')
            
        x_up3 = self.up3(torch.cat([x_up2, x1], dim=1), t)
        
        if x_up3.shape[2:] != x0.shape[2:]:
            x_up3 = F.interpolate(x_up3, size=x0.shape[2:], mode='nearest')

        # Final projection
        # Add the original feature map to the final prediction
        out = self.out(x_up3)
        
        # Crop back down
        out = out[..., :orig_H, :orig_W]
        
        return out

class CustomDiffusionScheduler:
    """
    Custom DDPM Scheduler implementation built completely from scratch. 
    Matches standard Gaussian Variance schedules (linear).
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear Variance Schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # Precompute square roots for mapping
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        """
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_alpha_prod = sqrt_alpha_prod[(...,) + (None,) * (original_samples.ndim - 1)]
        
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[(...,) + (None,) * (original_samples.ndim - 1)]
        
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def step(self, model_output, timestep, sample):
        """
        Single DDPM Reverse step to reconstruct sample from t to t-1
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - ((1-alpha_t) / sqrt(1-alpha_bar_t)) * eps_theta) + sigma_t * z
        """
        t = timestep
        
        # Extract constants for this step
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Mean calculation
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            sample - (beta_t / sqrt_one_minus_alpha_cumprod_t) * model_output
        )

        if t > 0:
            # Add posterior variance (noise) for all steps except the final t=0 step
            noise = torch.randn_like(sample)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * noise
        else:
            # At t=0, just return the mean
            return mean
