"""
Deterministic UNet with Temporal Attention for Subseasonal Bias Correction

Architecture: Conv2D UNet encoder/decoder with temporal self-attention at the
bottleneck. The 4 lead weeks are treated as channels throughout the encoder,
then reshaped into a temporal dimension at the bottleneck for cross-week
attention, then merged back for the decoder.

No diffusion — direct residual regression in a single forward pass.

Key Features:
- ResBlock with FiLM conditioning (month embedding)
- Spatial Self-Attention at bottleneck
- Temporal Self-Attention across 4 lead weeks at bottleneck
- Huber + MSE combined loss helper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# BUILDING BLOCKS
# ==============================================================================

class ResBlock(nn.Module):
    """Residual block with FiLM conditioning from month embedding."""
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.norm1(x)
        h = F.leaky_relu(h)
        h = self.conv1(h)
        # FiLM: broadcast embedding (B, C) -> (B, C, 1, 1)
        emb_out = self.emb_proj(emb)[:, :, None, None]
        h = h + emb_out
        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over spatial dimensions (H*W tokens)."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W).unbind(1)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj_out(out)


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention across T temporal slots.
    
    Input:  (B, T, D)  where T=4 lead weeks, D=feature dim per week
    Output: (B, T, D)  with cross-week information mixed
    """
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embedding for 4 weeks
        self.pos_emb = nn.Parameter(torch.randn(1, 4, dim) * 0.02)

    def forward(self, x):
        """
        x: (B, T, D) where T=4 lead weeks
        """
        B, T, D = x.shape
        
        # Add positional embedding
        x = x + self.pos_emb[:, :T, :]
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, heads, T, head_dim)
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.proj_out(out)
        
        return x + out


# ==============================================================================
# MAIN MODEL
# ==============================================================================

class TemporalAttentionUNet(nn.Module):
    """
    Deterministic Conv2D UNet with Temporal Attention at the bottleneck.
    
    Architecture:
        - Conv2D encoder (3 downsampling levels)
        - Bottleneck with spatial attention + temporal attention across 4 weeks
        - Conv2D decoder (3 upsampling levels with skip connections)
    
    Conditioning:
        - Month one-hot (12-dim) → projected to embedding → FiLM in every ResBlock
    
    The 4 lead weeks flow through the encoder as channels. At the bottleneck,
    channels are split into 4 temporal slots, temporal attention is applied,
    then they are merged back for the decoder.
    
    Args:
        in_channels: Number of input channels (18 for ocean: 4F + 4O + 4SST + 4SSS + 2MJO)
        out_channels: Number of output channels (4 for 4 lead weeks of residual)
        base_filters: Base number of filters (default 128)
        emb_dim: Dimension of the conditioning embedding (default 256)
        n_weeks: Number of lead-time weeks (default 4)
        temporal_heads: Number of attention heads for temporal attention (default 4)
    """
    def __init__(self, in_channels, out_channels, base_filters=128, emb_dim=256,
                 n_weeks=4, temporal_heads=4):
        super().__init__()
        self.n_weeks = n_weeks
        
        # Month conditioning: 12 → emb_dim
        self.month_proj = nn.Sequential(
            nn.Linear(12, 128),
            nn.LeakyReLU(),
            nn.Linear(128, emb_dim),
            nn.LeakyReLU(),
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, base_filters, 3, padding=1)
        
        self.down1 = ResBlock(base_filters, base_filters, emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(base_filters, base_filters * 2, emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResBlock(base_filters * 2, base_filters * 4, emb_dim)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck1 = ResBlock(base_filters * 4, base_filters * 8, emb_dim)
        self.spatial_attn = SpatialSelfAttention(base_filters * 8, num_heads=8)
        
        # Temporal Attention at bottleneck
        # base_filters * 8 channels split into n_weeks slots → each slot has (base_filters * 8 / n_weeks) dims
        # But that requires base_filters * 8 % n_weeks == 0
        # With base_filters=128, bottleneck= 1024 channels, 1024 / 4 = 256 per week ✓
        self.temporal_dim = (base_filters * 8) // n_weeks
        self.temporal_attn = TemporalAttention(
            dim=self.temporal_dim,  # Feature dim per week
            num_heads=temporal_heads,
            dropout=0.1
        )
        
        self.bottleneck2 = ResBlock(base_filters * 8, base_filters * 8, emb_dim)
        
        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ResBlock(base_filters * 8 + base_filters * 4, base_filters * 4, emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ResBlock(base_filters * 4 + base_filters * 2, base_filters * 2, emb_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ResBlock(base_filters * 2 + base_filters, base_filters, emb_dim)
        
        self.conv_out = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x_input, month_onehot):
        """
        Args:
            x_input: (B, C_in, H, W) — concatenated conditioning inputs
            month_onehot: (B, 12) — one-hot month encoding
        
        Returns:
            (B, 4, H, W) — predicted residual for each lead week
        """
        # Conditioning embedding (month only — no timestep)
        cond_emb = self.month_proj(month_onehot)
        
        # Encoder
        x = self.conv_in(x_input)
        
        s1 = self.down1(x, cond_emb);  x = self.pool1(s1)
        s2 = self.down2(x, cond_emb);  x = self.pool2(s2)
        s3 = self.down3(x, cond_emb);  x = self.pool3(s3)
        
        # Bottleneck
        x = self.bottleneck1(x, cond_emb)
        x = self.spatial_attn(x)
        
        # ---- Temporal Attention ----
        # x shape: (B, C_bottleneck, H', W') where C_bottleneck = base_filters * 8
        B, C, H_b, W_b = x.shape
        
        # Reshape: (B, n_weeks, temporal_dim, H', W') → flatten spatial → (B * H' * W', n_weeks, temporal_dim)
        x = x.reshape(B, self.n_weeks, self.temporal_dim, H_b, W_b)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H_b * W_b, self.n_weeks, self.temporal_dim)
        
        # Apply temporal attention across weeks
        x = self.temporal_attn(x)
        
        # Reshape back: (B, C_bottleneck, H', W')
        x = x.reshape(B, H_b, W_b, self.n_weeks, self.temporal_dim)
        x = x.permute(0, 3, 4, 1, 2).reshape(B, C, H_b, W_b)
        # ---- End Temporal Attention ----
        
        x = self.bottleneck2(x, cond_emb)
        
        # Decoder with skip connections
        x = self.up3(x)
        if x.shape[2:] != s3.shape[2:]:
            x = F.interpolate(x, size=s3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x, cond_emb)
        
        x = self.up2(x)
        if x.shape[2:] != s2.shape[2:]:
            x = F.interpolate(x, size=s2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x, cond_emb)
        
        x = self.up1(x)
        if x.shape[2:] != s1.shape[2:]:
            x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x, cond_emb)
        
        return self.conv_out(x)


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class ConservationMSELoss(nn.Module):
    """
    Conservation + Pixel MSE Loss.
    
    Combines:
    1. Pixel-wise MSE (Area + Intensity Weighted).
    2. Mass Conservation Loss (Squared difference of Global Means).
    
    Total = 0.5 * Pixel_MSE + 0.5 * Mass_MSE
    
    This forces the model to predict the correct TOTAL volume of rain globally
    (correcting the dry bias) even if valid local placement is difficult.
    
    Args:
        norm_stats: Dict with keys {'min', 'max', 'res_min', 'res_max'}
        n_lat, n_lon, lat_range: Grid details
        intensity_scale: Weight scaling for Pixel MSE (default 0.1)
    """
    def __init__(self, norm_stats, n_lat=181, n_lon=360, lat_range=(90, -90),
                 intensity_scale=0.1):
        super().__init__()
        self.intensity_scale = intensity_scale
        
        # Norm stats for denormalization
        self.vmin = norm_stats['min']
        self.vmax = norm_stats['max']
        self.rmin = norm_stats['res_min']
        self.rmax = norm_stats['res_max']
        self.denom = self.vmax - self.vmin if self.vmax != self.vmin else 1.0
        self.res_denom = self.rmax - self.rmin if self.rmax != self.rmin else 1.0
        
        # Compute cos(lat) weights
        import numpy as np
        lats = np.linspace(lat_range[0], lat_range[1], n_lat)
        cos_weights = np.cos(np.deg2rad(lats)).astype(np.float32)
        cos_weights = np.maximum(cos_weights, 0.0)
        # For global mean, we need sum of weights to be the normalization factor
        # But for pixel loss, we usually normalize so mean() matches scale.
        # We'll normalize weights so mean(weights) = 1 (approx)
        cos_weights = cos_weights * (n_lat / cos_weights.sum()) 
        
        weights = torch.from_numpy(cos_weights).reshape(1, 1, n_lat, 1)
        self.register_buffer('area_weights', weights)
    
    def denormalize_prediction(self, res_norm, forecast_norm):
        """Reconstruct prediction in mm/day from residual and forecast."""
        res_log = (res_norm + 1.0) / 2.0 * self.res_denom + self.rmin
        forc_log = (forecast_norm + 1.0) / 2.0 * self.denom + self.vmin
        pred_log = forc_log + res_log
        pred_mm = torch.expm1(pred_log)
        return torch.clamp(pred_mm, min=0.0)

    def global_mean(self, x):
        """Compute area-weighted global mean."""
        # x: (B, C, H, W)
        # area_weights: (1, 1, H, 1)
        # We broadcast weights to (B, C, H, W) implicitly
        weighted_sum = (x * self.area_weights).sum(dim=(2, 3))
        # Total weight sum for the batch/channel
        total_weight = self.area_weights.expand_as(x).sum(dim=(2, 3))
        
        return weighted_sum / (total_weight + 1e-6)

    def forward(self, pred_res, target_res, forecast):
        """
        pred_res, target_res: Normalized residuals [-1, 1]
        forecast: Normalized forecast input [-1, 1]
        """
        # 1. Reconstruct Physical Values (mm/day)
        pred_mm = self.denormalize_prediction(pred_res, forecast)
        target_mm = self.denormalize_prediction(target_res, forecast)
        
        # 2. Pixel-wise MSE Loss
        # Weight = Area * (1 + scale * target_mm)
        w_pixel = self.area_weights * (1.0 + self.intensity_scale * target_mm)
        sq_diff = (pred_mm - target_mm) ** 2
        pixel_loss = (sq_diff * w_pixel).mean()
        
        # 3. Mass Conservation Loss (Global Mean Squared Error)
        # Calculate area-weighted means scalar per sample/channel
        mean_pred = self.global_mean(pred_mm)
        mean_target = self.global_mean(target_mm)
        
        # MSE of the means
        mass_loss = ((mean_pred - mean_target) ** 2).mean()
        
        # 4. Total Loss (50/50 split)
        return 0.5 * pixel_loss + 0.5 * mass_loss


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing TemporalAttentionUNet on {device}...")
    
    # Ocean config: 4F + 4O + 4SST + 4SSS + 2MJO = 18 input channels
    model = TemporalAttentionUNet(
        in_channels=18,
        out_channels=4,
        base_filters=128,
        emb_dim=256,
        n_weeks=4,
        temporal_heads=4
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 18, 181, 360, device=device)
    month = torch.zeros(2, 12, device=device)
    month[:, 0] = 1.0  # January
    
    with torch.no_grad():
        out = model(x, month)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}")
    print("✓ Forward pass successful!")
