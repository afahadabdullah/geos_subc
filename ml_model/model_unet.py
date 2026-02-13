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
- Mass Conservation Loss ONLY
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
        # Conditioning embedding
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
        B, C, H_b, W_b = x.shape
        x = x.reshape(B, self.n_weeks, self.temporal_dim, H_b, W_b)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H_b * W_b, self.n_weeks, self.temporal_dim)
        x = self.temporal_attn(x)
        x = x.reshape(B, H_b, W_b, self.n_weeks, self.temporal_dim)
        x = x.permute(0, 3, 4, 1, 2).reshape(B, C, H_b, W_b)
        # ---- End Temporal Attention ----
        
        x = self.bottleneck2(x, cond_emb)
        
        # Decoder
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

class MassConservationLoss(nn.Module):
    """
    Mass Conservation Loss ONLY for DIRECT PREDICTION.
    
    Combines:
    1. Mass Conservation Loss (Area-weighted Global Mean Squared Error).
    
    Total = Mass_Loss
    
    Args:
        norm_stats: Dict with keys {'min', 'max'}
        n_lat, n_lon, lat_range: Grid details
    """
    def __init__(self, norm_stats, n_lat=181, n_lon=360, lat_range=(90, -90)):
        super().__init__()
        
        # Norm stats for denormalization
        self.vmin = norm_stats['min']
        self.vmax = norm_stats['max']
        self.denom = self.vmax - self.vmin if self.vmax != self.vmin else 1.0
        
        # Compute cos(lat) weights
        import numpy as np
        lats = np.linspace(lat_range[0], lat_range[1], n_lat)
        cos_weights = np.cos(np.deg2rad(lats)).astype(np.float32)
        cos_weights = np.maximum(cos_weights, 0.0)
        cos_weights = cos_weights * (n_lat / cos_weights.sum()) 
        
        weights = torch.from_numpy(cos_weights).reshape(1, 1, n_lat, 1)
        self.register_buffer('area_weights', weights)
    
    def denormalize_precip(self, x_norm, forecast=None):
        """Unscale [-1, 1] log1p-norm to mm/day."""
        x_log = (x_norm + 1.0) / 2.0 * self.denom + self.vmin
        x_mm = torch.expm1(x_log)
        return torch.clamp(x_mm, min=0.0)

    def global_mean(self, x):
        """Compute area-weighted global mean."""
        weighted_sum = (x * self.area_weights).sum(dim=(2, 3))
        total_weight = self.area_weights.expand_as(x).sum(dim=(2, 3))
        return weighted_sum / (total_weight + 1e-6)

    def forward(self, pred_norm, target_norm, forecast=None):
        """
        pred_norm: Predicted Precip (normalized [-1, 1])
        target_norm: True Precip (normalized [-1, 1])
        """
        # 1. Denormalize to Physical Values (mm/day)
        pred_mm = self.denormalize_precip(pred_norm)
        target_mm = self.denormalize_precip(target_norm)
        
        # 2. Mass Conservation Loss (Global Mean Squared Error)
        mean_pred = self.global_mean(pred_mm)
        mean_target = self.global_mean(target_mm)
        loss_mass = ((mean_pred - mean_target) ** 2).mean()
        
        return loss_mass


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing TemporalAttentionUNet on {device}...")
    model = TemporalAttentionUNet(18, 4).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
