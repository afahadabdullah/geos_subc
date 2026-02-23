"""
Diffusion Model for Ensemble Weather Forecasting (v8 - Multi-Variable + ERA5 Land)

Purpose: Train a conditional diffusion model to predict the RESIDUAL (Error) of the GEOS forecast.
         Target = ERA5 Land - GEOS.
         Forecast = GEOS + Predicted_Residual.

Key Features V8:
- Multi-Variable: T2M (temperature) + Precipitation
- ERA5 Land Target: 0.1° resolution (~260x590 for CONUS)
- 9-Channel Input: 4 noisy residual + 4 GEOS + 1 init_obs_mean
- 4-Channel Output: T2M_L1, T2M_L2, PREC_L1, PREC_L2
- Separate normalization for T2M vs Precipitation
- CMDE Multi-Speed Diffusion (reduced noise on GEOS condition)

Architecture:
- Input: Noisy Residual (4ch) + GEOS condition (4ch) + Init Obs mean (1ch) = 9 channels
- Conditioning: Timestep + Month embedding
- Output: Predicted noise (epsilon) for the Residual (4 channels)
"""

import argparse
import os
import csv
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("Warning: Cartopy not found. Maps will be plotted without geographic features.")
    ccrs = None
    cfeature = None

from data.dataset_v9 import GEOSDatasetV9
from diagnostics_v9 import run_diagnostics


# ==============================================================================
# UTILITIES
# ==============================================================================

def save_checkpoint(state, filepath):
    """Save checkpoint safely (write to temp, then rename)."""
    temp_path = filepath + ".tmp"
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, filepath)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_state_dict_flexible(model, state_dict):
    """Load state dict with automatic handling of DataParallel prefix mismatch."""
    model_is_parallel = hasattr(model, 'module')
    ckpt_is_parallel = any(k.startswith('module.') for k in state_dict.keys())
    
    if ckpt_is_parallel and not model_is_parallel:
        print("  Stripping 'module.' prefix from checkpoint keys...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    elif not ckpt_is_parallel and model_is_parallel:
        print("  Adding 'module.' prefix to checkpoint keys...")
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)


def plot_loss_curves(csv_path, save_path, epoch):
    """Plot training and validation loss curves from CSV."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss', markersize=4)
        ax.plot(df['epoch'], df['val_loss'], 'r-o', label='Val Loss', markersize=4)
        
        best_idx = df['val_loss'].idxmin()
        best_epoch = df.loc[best_idx, 'epoch']
        best_loss = df.loc[best_idx, 'val_loss']
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        ax.scatter([best_epoch], [best_loss], color='green', s=100, marker='*', zorder=5)
        ax.annotate(f'Best: {best_loss:.6f}', (best_epoch, best_loss),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Progress - V8 Multi-Variable Residual Learning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"loss_curves_epoch_{epoch}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")


# ==============================================================================
# GAUSSIAN DIFFUSION
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
        sqrt_alpha_hat_t = self.sqrt_alpha_hats[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(original_images)
        noisy_images = sqrt_alpha_hat_t * original_images + sqrt_one_minus_alpha_hat_t * noise
        
        return noisy_images, noise
    
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)
    
    @torch.no_grad()
    def denoise_step(self, model, x_t, t, condition, month):
        """Single denoising step for generation."""
        batch_size = x_t.shape[0]
        
        if isinstance(t, int):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        else:
            t_tensor = t
        
        noise_pred = model(x_t, condition, t_tensor, month)
        
        t_idx = t if isinstance(t, int) else t[0].item()
        alpha_t = self.alphas[t_idx]
        alpha_hat_t = self.alpha_hats[t_idx]
        beta_t = self.betas[t_idx]
        
        if t_idx > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        term1 = 1.0 / torch.sqrt(alpha_t)
        term2 = x_t - (beta_t / torch.sqrt(1.0 - alpha_hat_t)) * noise_pred
        term3 = torch.sqrt(beta_t) * noise
        
        return term1 * term2 + term3


# ==============================================================================
# MODELS
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
        emb_out = self.emb_proj(emb)[:, :, None, None]
        h = h + emb_out
        h = self.norm2(h)
        h = F.leaky_relu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

import heapq

class CheckpointManager:
    """
    Manages saving top K checkpoints based on validation loss.
    """
    def __init__(self, save_dir, top_k=4):
        self.save_dir = save_dir
        self.top_k = top_k
        self.best_ckpts = []  # Heap of (-loss, epoch, path) to keep smallest loss
    
    def save(self, model, epoch, val_loss):
        """Save checkpoint if it's among the top K."""
        ckpt_name = f"ckpt_val_loss_{val_loss:.6f}_epoch_{epoch}.pt"
        save_path = os.path.join(self.save_dir, ckpt_name)
        
        # Metadata to store in heap (we use -loss for max-heap behavior to pop worst)
        # Actually min-loss is better, we want to KEEP smallest losses.
        # Python heap is min-heap. So if we store (-val_loss), the smallest will be at top.
        # Wait, we want to keep K smallest losses. A min-heap of size K?
        # If we store (-val_loss), then the "smallest" (most negative, i.e. largest loss) is at root.
        # So we can pop the worst model easily.
        
        item = (-val_loss, epoch, save_path)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, save_path)
        
        if len(self.best_ckpts) < self.top_k:
            heapq.heappush(self.best_ckpts, item)
            print(f"  Saved Top-{self.top_k} Checkpoint: {ckpt_name}")
        else:
            # Check if this new one is better than the worst one in our top-k
            # The worst one is at the root (min value of -loss, which is max loss)
            worst_item = self.best_ckpts[0]
            worst_loss = -worst_item[0]
            
            if val_loss < worst_loss:
                # Remove old file
                _, _, old_path = heapq.heappop(self.best_ckpts)
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"  Removed old checkpoint: {os.path.basename(old_path)}")
                
                # Add new one
                heapq.heappush(self.best_ckpts, item)
                print(f"  Saved New Top-{self.top_k} Checkpoint: {ckpt_name}")
            else:
                # Not good enough, delete the file we just created
                os.remove(save_path)



class ConditionalUNet(nn.Module):
    """
    V8 UNet: 9 input channels, 4 output channels
    Input: Noisy Residual (4ch) + GEOS (4ch) + Init Obs Mean (1ch) = 9 channels
    Output: Predicted noise (4 channels)
    """
    def __init__(self, in_channels=9, out_channels=4, base_filters=64, num_months=12, emb_dim=256):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(dim=128)
        self.month_emb = nn.Embedding(num_months, 128)
        self.cond_mlp = nn.Sequential(nn.Linear(256, emb_dim), nn.LeakyReLU())
        
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
    
    def forward(self, noisy_target, condition, timesteps, month):
        x = torch.cat([noisy_target, condition], dim=1)
        t_emb = self.time_emb(timesteps)
        m_emb = self.month_emb(month)
        cond_emb = self.cond_mlp(torch.cat([t_emb, m_emb], dim=1))
        
        x = self.conv_in(x)
        s1 = self.down1(x, cond_emb); x = self.pool1(s1)
        s2 = self.down2(x, cond_emb); x = self.pool2(s2)
        s3 = self.down3(x, cond_emb); x = self.pool3(s3)
        
        x = self.bottleneck1(x, cond_emb)
        x = self.bottleneck2(x, cond_emb)
        
        x = self.up3(x)
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
# STATISTICS & NORMALIZATION
# ==============================================================================

def compute_dataset_stats(dataset, train_indices, stats_path, num_samples=500):
    """
    Compute DOMAIN-LEVEL min-max statistics for V9 (Lead 1 Only).
    Separate stats for T2M vs Precipitation.
    """
    print(f"\n{'='*60}\nComputing Domain Min-Max Statistics (V9 Lead 1 Only)\n{'='*60}")
    
    if os.path.exists(stats_path):
        print(f"Stats file exists at {stats_path}. Loading...")
        data = np.load(stats_path)
        return {k: torch.tensor(data[k]).float() for k in data.files}
    
    if len(train_indices) > num_samples:
        sample_indices = np.random.choice(train_indices, num_samples, replace=False)
    else:
        sample_indices = train_indices
    
    print(f"Computing stats using {len(sample_indices)} samples...")
    
    # Initialize
    t2m_geos_min, t2m_geos_max = float('inf'), float('-inf')
    t2m_era5_min, t2m_era5_max = float('inf'), float('-inf')
    t2m_resid_min, t2m_resid_max = float('inf'), float('-inf')
    
    prec_geos_min, prec_geos_max = float('inf'), float('-inf')
    prec_era5_min, prec_era5_max = float('inf'), float('-inf')
    prec_resid_min, prec_resid_max = float('inf'), float('-inf')
    
    init_t2m_min, init_t2m_max = float('inf'), float('-inf')
    init_prec_min, init_prec_max = float('inf'), float('-inf')
    
    for idx in tqdm(sample_indices, desc="Computing stats"):
        try:
            geos, era5, month, init_obs = dataset[idx]
            
            # GEOS: [2, H, W] -> T2M (0), PREC (1)
            geos_t2m = geos[0:1]
            geos_prec = geos[1:2]
            
            # ERA5
            era5_t2m = era5[0:1]
            era5_prec = era5[1:2]
            
            # Residuals
            resid_t2m = era5_t2m - geos_t2m
            resid_prec = era5_prec - geos_prec
            
            # Init Obs: [2, H, W] -> T2M (0), PREC (1)
            init_t2m = init_obs[0]
            init_prec = init_obs[1]
            
            # Update T2M stats
            t2m_geos_min = min(t2m_geos_min, geos_t2m.min().item())
            t2m_geos_max = max(t2m_geos_max, geos_t2m.max().item())
            t2m_era5_min = min(t2m_era5_min, era5_t2m.min().item())
            t2m_era5_max = max(t2m_era5_max, era5_t2m.max().item())
            t2m_resid_min = min(t2m_resid_min, resid_t2m.min().item())
            t2m_resid_max = max(t2m_resid_max, resid_t2m.max().item())
            
            # Update Precip stats
            prec_geos_min = min(prec_geos_min, geos_prec.min().item())
            prec_geos_max = max(prec_geos_max, geos_prec.max().item())
            prec_era5_min = min(prec_era5_min, era5_prec.min().item())
            prec_era5_max = max(prec_era5_max, era5_prec.max().item())
            prec_resid_min = min(prec_resid_min, resid_prec.min().item())
            prec_resid_max = max(prec_resid_max, resid_prec.max().item())
            
            # Update Init Obs stats
            init_t2m_min = min(init_t2m_min, init_t2m.min().item())
            init_t2m_max = max(init_t2m_max, init_t2m.max().item())
            init_prec_min = min(init_prec_min, init_prec.min().item())
            init_prec_max = max(init_prec_max, init_prec.max().item())
            
        except Exception as e:
            continue
    
    stats = {
        't2m_geos_min': np.array(t2m_geos_min),
        't2m_geos_max': np.array(t2m_geos_max),
        't2m_resid_min': np.array(t2m_resid_min),
        't2m_resid_max': np.array(t2m_resid_max),
        'prec_geos_min': np.array(prec_geos_min),
        'prec_geos_max': np.array(prec_geos_max),
        'prec_resid_min': np.array(prec_resid_min),
        'prec_resid_max': np.array(prec_resid_max),
        'init_t2m_min': np.array(init_t2m_min),
        'init_t2m_max': np.array(init_t2m_max),
        'init_prec_min': np.array(init_prec_min),
        'init_prec_max': np.array(init_prec_max),
    }
    
    np.savez(stats_path, **stats)
    print(f"Saved stats to {stats_path}")
    print(f"  T2M GEOS: [{t2m_geos_min:.2f}, {t2m_geos_max:.2f}] K")
    print(f"  T2M Resid: [{t2m_resid_min:.2f}, {t2m_resid_max:.2f}] K")
    print(f"  Prec GEOS: [{prec_geos_min:.4f}, {prec_geos_max:.4f}] mm/day")
    print(f"  Prec Resid: [{prec_resid_min:.4f}, {prec_resid_max:.4f}] mm/day")
    print(f"  Init T2M: [{init_t2m_min:.2f}, {init_t2m_max:.2f}] K")
    print(f"  Init Prec: [{init_prec_min:.4f}, {init_prec_max:.4f}] mm")
    
    return {k: torch.tensor(v).float() for k, v in stats.items()}


def load_stats(stats_path):
    """Load domain min-max stats for V8."""
    print(f"Loading stats from {stats_path}...")
    data = np.load(stats_path)
    stats = {}
    for k in data.files:
        stats[k] = torch.tensor(data[k]).float()
    return stats


def minmax_normalize(x, x_min, x_max):
    """Normalize to [-1, 1] using domain min-max."""
    return 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0


def minmax_denormalize(x_norm, x_min, x_max):
    """Denormalize from [-1, 1] to original scale."""
    return (x_norm + 1.0) / 2.0 * (x_max - x_min) + x_min


def normalize_batch_v9(geos, era5, init_obs, stats, device):
    """
    Normalize using domain min-max for V9 (Elevation Support).
    
    GEOS/ERA5: [B, 2, H, W] -> (T2M_L1, PREC_L1)
    init_obs: [B, 3, H, W] -> (T2M, PREC, ELEV)
    
    Returns:
    - geos_norm: [B, 2, H, W]
    - resid_norm: [B, 2, H, W]
    - condition: [B, 2, H, W] (Init_Mean + Elev)
    """
    # T2M stats
    t2m_geos_min = stats['t2m_geos_min'].to(device)
    t2m_geos_max = stats['t2m_geos_max'].to(device)
    t2m_resid_min = stats['t2m_resid_min'].to(device)
    t2m_resid_max = stats['t2m_resid_max'].to(device)
    
    # Precip stats
    prec_geos_min = stats['prec_geos_min'].to(device)
    prec_geos_max = stats['prec_geos_max'].to(device)
    prec_resid_min = stats['prec_resid_min'].to(device)
    prec_resid_max = stats['prec_resid_max'].to(device)
    
    # Init Obs stats
    init_t2m_min = stats['init_t2m_min'].to(device)
    init_t2m_max = stats['init_t2m_max'].to(device)
    init_prec_min = stats['init_prec_min'].to(device)
    init_prec_max = stats['init_prec_max'].to(device)
    
    # Split by variable (Lead 1 Only: 2 channels [T2M, PREC])
    geos_t2m = geos[:, 0:1]
    geos_prec = geos[:, 1:2]
    era5_t2m = era5[:, 0:1]
    era5_prec = era5[:, 1:2]
    
    # Normalize GEOS
    geos_t2m_norm = minmax_normalize(geos_t2m, t2m_geos_min, t2m_geos_max)
    geos_prec_norm = minmax_normalize(geos_prec, prec_geos_min, prec_geos_max)
    geos_norm = torch.cat([geos_t2m_norm, geos_prec_norm], dim=1)  # [B, 2, H, W]
    
    # Compute and normalize residuals
    resid_t2m = era5_t2m - geos_t2m
    resid_prec = era5_prec - geos_prec
    resid_t2m_norm = minmax_normalize(resid_t2m, t2m_resid_min, t2m_resid_max)
    resid_prec_norm = minmax_normalize(resid_prec, prec_resid_min, prec_resid_max)
    resid_norm = torch.cat([resid_t2m_norm, resid_prec_norm], dim=1)  # [B, 2, H, W]
    
    # Normalize init_obs
    init_t2m = init_obs[:, 0:1]
    init_prec = init_obs[:, 1:2]
    elev = init_obs[:, 2:3] # [B, 1, H, W] - Already [0, 1] from Dataset
    
    # Map Elev [0, 1] -> [-1, 1]
    elev_norm = (elev * 2.0) - 1.0
    
    init_t2m_norm = minmax_normalize(init_t2m, init_t2m_min, init_t2m_max)
    init_prec_norm = minmax_normalize(init_prec, init_prec_min, init_prec_max)
    
    # Combine init obs: Mean(T2M, Prec) + Elev
    # Output: [B, 2, H, W]
    init_mean = 0.5 * (init_t2m_norm + init_prec_norm)
    
    init_obs_out = torch.cat([init_mean, elev_norm], dim=1) # [B, 2, H, W]
    
    return geos_norm, resid_norm, init_obs_out


# ==============================================================================
# TRAINING & VALIDATION
# ==============================================================================

def train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch, stats, cmde_ratio=0.1):
    """Train with CMDE conditioning for V9 multi-variable + elevation."""
    model.train()
    total_loss = 0.0
    count = 0
    
    stats_dev = {k: v.to(device) for k, v in stats.items()}
    
    if epoch == 0:
        print("\n=== DEBUG: Stats Check (V9 Multi-Variable) ===")
        for k, v in stats_dev.items():
            print(f"  {k}: {v.item():.4f}")
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        geos, era5, month, init_obs = batch
        geos = geos.to(device)
        era5 = era5.to(device)
        month = month.to(device).long()
        init_obs = init_obs.to(device)
        
        if epoch == 0 and batch_idx == 0:
            print("\n=== DEBUG: First Batch Raw Inputs ===")
            print(f"  geos shape: {geos.shape}")
            print(f"  geos T2M: min={geos[:,0:1].min():.2f}, max={geos[:,0:1].max():.2f}")
            print(f"  geos Prec: min={geos[:,1:2].min():.4f}, max={geos[:,1:2].max():.4f}")
            print(f"  era5 T2M: min={era5[:,0:1].min():.2f}, max={era5[:,0:1].max():.2f}")
            print(f"  era5 Prec: min={era5[:,1:2].min():.4f}, max={era5[:,1:2].max():.4f}")
            print(f"  init_obs: min={init_obs.min():.2f}, max={init_obs.max():.2f}")
        
        # Normalize with V9 multi-variable normalization
        geos_norm, resid_norm, init_obs_norm = normalize_batch_v9(
            geos, era5, init_obs, stats_dev, device
        )
        
        if epoch == 0 and batch_idx == 0:
            print("\n=== DEBUG: After Normalization ===")
            print(f"  geos_norm: min={geos_norm.min():.2f}, max={geos_norm.max():.2f}")
            print(f"  resid_norm: min={resid_norm.min():.2f}, max={resid_norm.max():.2f}")
            print(f"  init_obs_norm: min={init_obs_norm.min():.2f}, max={init_obs_norm.max():.2f}")
        
        # Diffusion setup
        bs = geos.shape[0]
        timesteps = diffusion.sample_timesteps(bs)
        
        # Add FULL noise to RESIDUAL (signal)
        noisy_resid, noise = diffusion.add_noise(resid_norm, timesteps)
        
        # CMDE: Add REDUCED noise to GEOS condition
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
        geos_noise = cmde_ratio * sqrt_one_minus_alpha * torch.randn_like(geos_norm)
        noisy_geos = geos_norm + geos_noise
        
        # Condition = noisy_GEOS (4ch) + init_obs (1ch) = 5 channels
        condition = torch.cat([noisy_geos, init_obs_norm], dim=1)
        
        # Predict noise (model expects 9 input channels: 4 noisy_resid + 5 condition)
        noise_pred = model(noisy_resid, condition, timesteps, month)
        
        if epoch == 0 and batch_idx == 0:
            print("\n=== DEBUG: Model Output ===")
            print(f"  noise_pred shape: {noise_pred.shape}")
            print(f"  noise_pred: min={noise_pred.min():.2f}, max={noise_pred.max():.2f}")
            print(f"  noise: min={noise.min():.2f}, max={noise.max():.2f}")
        
        loss = F.mse_loss(noise_pred, noise)
        
        if epoch == 0 and batch_idx < 5:
            print(f"  Batch {batch_idx} loss: {loss.item():.6f}")
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        pbar.set_postfix(loss=loss.item())
    
    return total_loss / count


@torch.no_grad()
def validate(model, diffusion, val_loader, device, stats, cmde_ratio=0.1):
    """Validate with CMDE conditioning for V8."""
    model.eval()
    total_loss = 0.0
    count = 0
    stats_dev = {k: v.to(device) for k, v in stats.items()}
    
    for batch in tqdm(val_loader, desc="Validating"):
        geos, era5, month, init_obs = batch
        geos = geos.to(device)
        era5 = era5.to(device)
        month = month.to(device).long()
        init_obs = init_obs.to(device)
        
        geos_norm, resid_norm, init_obs_norm = normalize_batch_v9(
            geos, era5, init_obs, stats_dev, device
        )
        
        bs = geos.shape[0]
        timesteps = diffusion.sample_timesteps(bs)
        
        noisy_resid, noise = diffusion.add_noise(resid_norm, timesteps)
        
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
        geos_noise = cmde_ratio * sqrt_one_minus_alpha * torch.randn_like(geos_norm)
        noisy_geos = geos_norm + geos_noise
        
        condition = torch.cat([noisy_geos, init_obs_norm], dim=1)
        
        noise_pred = model(noisy_resid, condition, timesteps, month)
        loss = F.mse_loss(noise_pred, noise)
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count


# ==============================================================================
# ENSEMBLE GENERATION
# ==============================================================================

@torch.no_grad()
def generate_ensemble_forecast(model, diffusion, geos_norm, geos_raw, init_obs_norm, month, stats, device, n_members=10, cmde_ratio=0.1, use_log_precip=True):
    """
    Generate forecast (V9) with single-lead output.
    Returns: [N_members, B, 2, H, W] forecasts (T2M, PREC)
    """
    model.eval()
    B, C, H, W = geos_norm.shape  # C=2
    
    stats_dev = {k: v.to(device) for k, v in stats.items()}
    
    # Separate residual denorm stats
    t2m_resid_min = stats_dev['t2m_resid_min']
    t2m_resid_max = stats_dev['t2m_resid_max']
    prec_resid_min = stats_dev['prec_resid_min']
    prec_resid_max = stats_dev['prec_resid_max']
    
    ensemble = []
    
    for i in range(n_members):
        # Sample latent residual (2 channels)
        x_t = torch.randn(B, 2, H, W, device=device)
        
        # Denoise
        timesteps = range(diffusion.timesteps - 1, -1, -1)
        for t in timesteps:
            t_tensor = torch.tensor([t], device=device).long()
            sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[t_tensor].view(1, 1, 1, 1)
            
            # Add noise to GEOS
            # geos_norm is 2ch
            geos_noise = cmde_ratio * sqrt_one_minus_alpha * torch.randn_like(geos_norm)
            noisy_geos = geos_norm + geos_noise
            
            # Condition: GEOS (2) + Init (2) = 4 channels
            condition = torch.cat([noisy_geos, init_obs_norm], dim=1)
            
            x_t = diffusion.denoise_step(model, x_t, t, condition, month)
        
        # x_t is now Predicted Normalized Residual in [-1, 1]
        pred_resid_norm = x_t
        
        # Denormalize Residual (separate for T2M and Precip)
        pred_resid_t2m_norm = pred_resid_norm[:, 0:1]
        pred_resid_prec_norm = pred_resid_norm[:, 1:2]
        
        pred_resid_t2m = minmax_denormalize(pred_resid_t2m_norm, t2m_resid_min, t2m_resid_max)
        pred_resid_prec = minmax_denormalize(pred_resid_prec_norm, prec_resid_min, prec_resid_max)
        
        pred_resid = torch.cat([pred_resid_t2m, pred_resid_prec], dim=1)
        
        # Add to GEOS (both are in same space - log for precip)
        forecast = geos_raw + pred_resid
        
        # Convert precipitation from log space to linear space
        if use_log_precip:
            # T2M stays as-is, Precip needs expm1
            forecast_t2m = forecast[:, 0:1]
            forecast_prec_log = forecast[:, 1:2]
            forecast_prec = torch.expm1(forecast_prec_log)
            # Clamp to non-negative (safety)
            forecast_prec = torch.clamp(forecast_prec, min=0.0)
            forecast = torch.cat([forecast_t2m, forecast_prec], dim=1)
        
        ensemble.append(forecast.unsqueeze(0))
    
    return torch.cat(ensemble, dim=0)  # [N, B, 2, H, W]


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_samples(model, diffusion, dataset, device, epoch, save_path, stats, n_members=5):
    """Visualize samples using V8 normalization (multi-variable)."""
    model.eval()
    
    idx = np.random.randint(0, len(dataset))
    geos_raw, era5_raw, month, init_obs_raw = dataset[idx]
    
    geos_raw = geos_raw.unsqueeze(0).to(device)
    era5_raw = era5_raw.unsqueeze(0).to(device)
    init_obs_raw = init_obs_raw.unsqueeze(0).to(device)
    month_t = torch.tensor([month], device=device).long()
    
    stats_dev = {k: v.to(device) for k, v in stats.items()}
    
    geos_norm, _, init_obs_norm = normalize_batch_v9(
        geos_raw, era5_raw, init_obs_raw, stats_dev, device
    )
    
    # Generate ensemble
    ensemble = generate_ensemble_forecast(
        model, diffusion, geos_norm, geos_raw, init_obs_norm, month_t,
        stats, device, n_members=n_members
    )
    
    # Resample GEOS for "Blocky" visualization
    g_raw_sample = geos_raw.cpu() # [1, 4, H, W]
    g_down = F.interpolate(g_raw_sample, scale_factor=0.2, mode='bilinear', align_corners=False)
    g_blocky = F.interpolate(g_down, size=(geos_raw.shape[2], geos_raw.shape[3]), mode='nearest').squeeze(0) # [4, H, W]
    
    ens_mean = ensemble.mean(dim=0).cpu()  # [1, 4, H, W]
    era5_cpu = era5_raw.cpu()
    
    # Prepare data arrays [2, H, W]
    # Prepare data arrays [H, W] (Single Lead - Index 0=T2M, Index 1=Prec)
    g_t2m = g_blocky[0].numpy()
    e_t2m = era5_cpu[0, 0].numpy()
    p_t2m = ens_mean[0, 0].numpy()
    
    g_prec = torch.expm1(g_blocky[1]).numpy()
    e_prec = torch.expm1(era5_cpu[0, 1]).numpy()
    p_prec = ens_mean[0, 1].numpy()
    
    # DEBUG: Print value ranges
    print(f"\\n=== VIS DEBUG: Value Ranges ===")
    print(f"  GEOS T2M: {g_t2m.min():.1f} to {g_t2m.max():.1f} K")
    print(f"  ERA5 T2M: {e_t2m.min():.1f} to {e_t2m.max():.1f} K")
    print(f"  Pred T2M: {p_t2m.min():.1f} to {p_t2m.max():.1f} K")
    print(f"  GEOS Prec: {g_prec.min():.2f} to {g_prec.max():.2f} mm/d")
    print(f"  ERA5 Prec: {e_prec.min():.2f} to {e_prec.max():.2f} mm/d")
    print(f"  Pred Prec: {p_prec.min():.2f} to {p_prec.max():.2f} mm/d")
    
    # Metrics for title
    t2m_rmse = np.sqrt(((p_t2m - e_t2m)**2).mean())
    prec_rmse = np.sqrt(((p_prec - e_prec)**2).mean())
    
    # Grid for Cartopy
    if ccrs is not None:
        H, W = geos_raw.shape[2], geos_raw.shape[3]
        lats = np.linspace(dataset.lat_max, dataset.lat_min, H)
        lons = np.linspace(dataset.lon_min, dataset.lon_max, W)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Plot
    if ccrs is not None:
        fig, axes = plt.subplots(4, 4, figsize=(24, 20), subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    def plot_ax(ax, data, title, cmap_name, vmin=None, vmax=None, levels=None):
        cmap = plt.get_cmap(cmap_name)
        norm = None
        if levels is not None:
            norm = mcolors.BoundaryNorm(levels, cmap.N)
            
        if ccrs is not None:
            im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=101)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=102)
            ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=103)
            ax.add_feature(cfeature.OCEAN, zorder=100, facecolor='white', edgecolor='none')
        else:
            im = ax.imshow(data, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, interpolation='none')
        
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Dynamic Levels (Min/Max from ERA5 Target)
    t2m_min = np.floor(e_t2m.min())
    t2m_max = np.ceil(e_t2m.max())
    prec_max_dyn = np.ceil(e_prec.max())
    if prec_max_dyn < 1.0: prec_max_dyn = 1.0
    
    levels_t2m = np.linspace(t2m_min, t2m_max, 15)
    levels_prec = np.linspace(0, prec_max_dyn, 11)
    levels_err_t2m = np.linspace(-10, 10, 21)
    levels_err_prec = np.linspace(-20, 20, 21)
    
    # Row 1: T2M Lead 1
    plot_ax(axes[0, 0], g_t2m, "GEOS T2M (K)", 'RdBu_r', levels=levels_t2m)
    plot_ax(axes[0, 1], e_t2m, "ERA5 T2M (K)", 'RdBu_r', levels=levels_t2m)
    plot_ax(axes[0, 2], p_t2m, "Pred T2M (K)", 'RdBu_r', levels=levels_t2m)
    plot_ax(axes[0, 3], e_t2m - p_t2m, f"Error (RMSE={t2m_rmse:.2f})", 'bwr', levels=levels_err_t2m)
    
    # Row 2: Prec Lead 1 (Dynamic Range)
    g_p1 = np.clip(g_prec, 0, prec_max_dyn)
    e_p1 = np.clip(e_prec, 0, prec_max_dyn)
    p_p1 = np.clip(p_prec, 0, prec_max_dyn)
    err_p1 = np.clip(e_prec - p_prec, -20, 20)
    
    plot_ax(axes[1, 0], g_p1, "GEOS Prec (mm/d)", 'Blues', levels=levels_prec)
    plot_ax(axes[1, 1], e_p1, "ERA5 Prec (mm/d)", 'Blues', levels=levels_prec)
    plot_ax(axes[1, 2], p_p1, "Pred Prec (mm/d)", 'Blues', levels=levels_prec)
    plot_ax(axes[1, 3], err_p1, f"Error (RMSE={prec_rmse:.2f})", 'bwr', levels=levels_err_prec)
    
    # Row 2: Prec Lead 1 (Dynamic Range)
    g_p1 = np.clip(g_prec, 0, prec_max_dyn)
    e_p1 = np.clip(e_prec, 0, prec_max_dyn)
    p_p1 = np.clip(p_prec, 0, prec_max_dyn)
    err_p1 = np.clip(e_prec - p_prec, -20, 20)
    
    plot_ax(axes[1, 0], g_p1, "GEOS Prec (mm/d)", 'Blues', levels=levels_prec)
    plot_ax(axes[1, 1], e_p1, "ERA5 Prec (mm/d)", 'Blues', levels=levels_prec)
    plot_ax(axes[1, 2], p_p1, "Pred Prec (mm/d)", 'Blues', levels=levels_prec)
    plot_ax(axes[1, 3], err_p1, f"Error (RMSE={prec_rmse:.2f})", 'bwr', levels=levels_err_prec)
    
    # Row 3: Bias and Model Error
    plot_ax(axes[2, 0], e_t2m - g_t2m, f"T2M Bias (ERA5-GEOS)\nRMSE: {np.sqrt(((e_t2m-g_t2m)**2).mean()):.2f} K", 'coolwarm', -5, 5)
    plot_ax(axes[2, 1], e_prec - g_prec, f"Prec Bias (ERA5-GEOS)\nRMSE: {np.sqrt(((e_prec-g_prec)**2).mean()):.2f}", 'BrBG', -5, 5)
    plot_ax(axes[2, 2], p_t2m - e_t2m, f"Model Error (Pred-ERA5)\nRMSE: {t2m_rmse:.2f} K", 'coolwarm', -5, 5)
    plot_ax(axes[2, 3], p_prec - e_prec, f"Model Error (Pred-ERA5)\nRMSE: {prec_rmse:.2f}", 'BrBG', -5, 5)
    
    # Row 4: Empty or Elevation debug?
    # Plot Elevation if available
    # init_obs_raw has 3 channels: T2M, Prec, Elev. Elev is index 2.
    elev = init_obs_raw[0, 2].cpu().numpy()
    plot_ax(axes[3, 0], elev, "Elevation (Meters)", 'terrain')
    
    axes[3, 1].axis('off')
    axes[3, 2].axis('off')
    axes[3, 3].axis('off')
    
    plt.suptitle(f"Sample Visualisation (V9 Lead 1 Only) - Epoch {epoch}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_epoch_{epoch}.png"), dpi=150)
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main(args):
    device = torch.device(args.device)
    print(f"Train V8 (Multi-Variable + ERA5 Land) on {device}")
    
    os.makedirs(args.save_path, exist_ok=True)
    plots_path = os.path.join(args.save_path, "plots")
    os.makedirs(plots_path, exist_ok=True)
    
    stats_path = os.path.join(args.save_path, "dataset_stats_v8.npz")
    
    # Load Dataset
    print("Loading dataset...")
    full_dataset = GEOSDatasetV9(
        geos_root=args.data_path,
        era5_land_path_1=args.era5_land_path_1,
        era5_land_path_2=args.era5_land_path_2,
        elevation_path=args.elevation_path,
        include_init_obs=True,
        load_stats_files=False,
        cache_to_ram=args.cache_data
    )
    
    # Split
    train_indices, val_indices = [], []
    for idx, p1 in enumerate(full_dataset.file_paths):
        year = int(p1.name.split('.')[0][:4])
        if year <= 2017:
            train_indices.append(idx)
        elif year <= 2019:
            val_indices.append(idx)
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    
    # Stats
    if args.stats_file and os.path.exists(args.stats_file):
        print(f"Loading stats from provided file: {args.stats_file}")
        stats = load_stats(args.stats_file)
    elif os.path.exists(stats_path) and not args.fresh:
        stats = load_stats(stats_path)
    else:
        stats_indices = [i for i in train_indices
                         if 1990 <= int(full_dataset.file_paths[i].name.split('.')[0][:4]) <= 2012]
        print(f"Computing stats on {len(stats_indices)} samples (1990-2012)...")
        stats = compute_dataset_stats(full_dataset, stats_indices, stats_path, num_samples=len(stats_indices))
    
    print(f"Stats loaded.")
    
    # Run Diagnostics if requested
    if args.diag:
        run_diagnostics(full_dataset, stats, device, normalize_fn=normalize_batch_v9, save_path=plots_path)
        print("Diagnostics complete. Exiting.")
        return
    
    # DataLoaders
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Fast DataLoader settings for Multi-GPU
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers // 2 if args.num_workers > 1 else args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Model (6 channels input: 2 noisy + 2 GEOS + 1 init + 1 elev = 6 channels)
    model = ConditionalUNet(
        in_channels=6, 
        out_channels=2, 
        base_filters=args.base_filters
    ).to(device)
    
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / (args.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training Loop
    best_loss = float('inf')
    start_epoch = 0
    
    # Resume from latest checkpoint if available
    latest_ckpt_path = os.path.join(args.save_path, "ckpt_latest.pt")
    if os.path.exists(latest_ckpt_path) and not args.fresh:
        print(f"Resuming from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        load_state_dict_flexible(model, ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('val_loss', float('inf'))
        
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}")
    
    # Initialize CheckpointManager
    ckpt_manager = CheckpointManager(args.save_path, top_k=4)
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, diffusion, train_loader, optimizer, device, epoch, stats, cmde_ratio=args.cmde_ratio)
        val_loss = validate(model, diffusion, val_loader, device, stats, cmde_ratio=args.cmde_ratio)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        # Save Latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss
        }, os.path.join(args.save_path, "ckpt_latest.pt"))
        
        # Save Best and Top K
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"*** New Best: {best_loss:.6f} ***")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, os.path.join(args.save_path, "ckpt_best.pt"))
            
            # Visualize only on new best
            visualize_samples(model, diffusion, val_subset.dataset, device, epoch, plots_path, stats)
        
        # Manage Top K Checkpoints
        ckpt_manager.save(model, epoch, val_loss)

        
        # Logging
        with open(os.path.join(args.save_path, "log.csv"), "a") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])


def run_test(args):
    """Run evaluation on test set (2017+)."""
    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print("V9 Test Mode: Evaluating on Test Set (2017+)")
    print(f"{'='*60}\n")
    
    # Load Dataset
    full_dataset = GEOSDatasetV9(
        geos_root=args.data_path,
        era5_land_path_1=args.era5_land_path_1,
        era5_land_path_2=args.era5_land_path_2,
        elevation_path=args.elevation_path,
        include_init_obs=True
    )
    
    # Get test indices (2017+) - V9 has single paths (Lead 1 only)
    test_indices = []
    for idx, p in enumerate(full_dataset.file_paths):
        year = int(p.name.split('.')[0][:4])
        if year >= 2017:
            test_indices.append(idx)
    
    print(f"Found {len(test_indices)} test samples (2017+)")
    
    # Limit to first 5 samples for quick testing
    test_indices = test_indices[:5]
    print(f"Limiting to first {len(test_indices)} samples for testing")
    
    test_subset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load stats
    stats_path = os.path.join(args.save_path, "dataset_stats_v8.npz")
    stats = load_stats(stats_path)
    stats_dev = {k: v.to(device) for k, v in stats.items()}
    
    # Load model
    ckpt_path = os.path.join(args.save_path, "ckpt_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        return
    
    model = ConditionalUNet(
        in_channels=9,     # 4 (x_t) + 4 (GEOS) + 1 (Init Obs) = 9 total
        out_channels=4,
        base_filters=args.base_filters
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")
    
    diffusion = GaussianDiffusion(timesteps=100, device=device)
    
    # Evaluate
    t2m_errors = []
    prec_errors = []
    n_samples = 0
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            geos_raw, era5_raw, months, init_obs_raw = batch
            geos_raw = geos_raw.to(device)
            era5_raw = era5_raw.to(device)
            init_obs_raw = init_obs_raw.to(device)
            months = months.to(device)  # Fix: Move months to device!
            
            geos_norm, _, init_norm = normalize_batch_v8(geos_raw, era5_raw, init_obs_raw, stats_dev, device)
            
            # Generate ensemble forecast
            ensemble = generate_ensemble_forecast(
                model, diffusion, geos_norm, geos_raw, init_norm, months,
                stats, device, n_members=5
            )
            
            ens_mean = ensemble.mean(dim=0)  # [B, 4, H, W]
            
            # Convert ERA5 from log to linear for precip comparison
            era5_t2m = era5_raw[:, :2]  # Already in K
            era5_prec = torch.expm1(era5_raw[:, 2:])  # Convert from log
            
            pred_t2m = ens_mean[:, :2]
            pred_prec = ens_mean[:, 2:]  # Already in linear mm/day
            
            # Compute RMSE per sample
            t2m_rmse = torch.sqrt(((pred_t2m - era5_t2m) ** 2).mean(dim=(1, 2, 3)))
            prec_rmse = torch.sqrt(((pred_prec - era5_prec) ** 2).mean(dim=(1, 2, 3)))
            
            t2m_errors.extend(t2m_rmse.cpu().numpy())
            prec_errors.extend(prec_rmse.cpu().numpy())
            n_samples += geos_raw.shape[0]
            
            # Plot test samples
            plot_dir = os.path.join(args.save_path, "test_plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            # Prepare Grid for Cartopy
            if ccrs is not None:
                H, W = geos_raw.shape[2], geos_raw.shape[3]
                lats = np.linspace(full_dataset.lat_max, full_dataset.lat_min, H)
                lons = np.linspace(full_dataset.lon_min, full_dataset.lon_max, W)
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            for i in range(geos_raw.shape[0]):
                # Resample GEOS to 0.5 degree "Blocky" view for visualization
                # Dataset upsamples 0.5 -> 0.1 bilinearly. 
                # We want to show the user the effective input resolution.
                # Downsample (0.2x) then Upsample Nearest (5x) to get blocky look at 0.1 grid
                g_raw_sample = geos_raw[i:i+1].cpu() # [1, 4, H, W]
                g_down = F.interpolate(g_raw_sample, scale_factor=0.2, mode='bilinear', align_corners=False)
                g_blocky = F.interpolate(g_down, size=(H, W), mode='nearest')
                
                # Prepare data for plotting
                g_t2m = g_blocky[0, :2].numpy()
                e_t2m = era5_t2m[i].cpu().numpy()
                p_t2m = pred_t2m[i].cpu().numpy()
                
                g_prec = torch.expm1(g_blocky[0, 2:]).numpy() # Convert GEOS log->linear

                e_prec = era5_prec[i].cpu().numpy()
                p_prec = pred_prec[i].cpu().numpy()
                
                # Setup Figure
                if ccrs is not None:
                    fig, axes = plt.subplots(4, 4, figsize=(24, 20), subplot_kw={'projection': ccrs.PlateCarree()})
                else:
                    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
                
                def plot_ax(ax, data, title, cmap_name, vmin=None, vmax=None, levels=None):
                    cmap = plt.get_cmap(cmap_name)
                    norm = None
                    if levels is not None:
                        norm = mcolors.BoundaryNorm(levels, cmap.N)
                        
                    if ccrs is not None:
                        im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, transform=ccrs.PlateCarree())
                        ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=101)
                        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=102)
                        ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=103)
                        # Mask Ocean
                        ax.add_feature(cfeature.OCEAN, zorder=100, facecolor='white', edgecolor='none')
                    else:
                        im = ax.imshow(data, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, interpolation='none')
                        
                    ax.set_title(title, fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Dynamic Levels (Min/Max from ERA5 Target)
                t2m_min = np.floor(e_t2m.min())
                t2m_max = np.ceil(e_t2m.max())
                prec_max_dyn = np.ceil(e_prec.max())
                
                if prec_max_dyn < 1.0: prec_max_dyn = 1.0 # Ensure at least some range
                
                # Create levels
                levels_t2m = np.linspace(t2m_min, t2m_max, 15)
                levels_prec = np.linspace(0, prec_max_dyn, 11)
                
                # Keep error levels fixed for consistency, or maybe dynamic?
                # Fixed is better for error comparison across samples usually, 
                # but if errors are huge, we miss details. Let's keep fixed for now.
                levels_err_t2m = np.linspace(-10, 10, 21) 
                levels_err_prec = np.linspace(-20, 20, 21)
                
                # Row 1: T2M Lead 1
                plot_ax(axes[0, 0], g_t2m[0], "GEOS T2M L1 (K)", 'RdBu_r', levels=levels_t2m)
                plot_ax(axes[0, 1], e_t2m[0], "ERA5 T2M L1 (K)", 'RdBu_r', levels=levels_t2m)
                plot_ax(axes[0, 2], p_t2m[0], "Pred T2M L1 (K)", 'RdBu_r', levels=levels_t2m)
                plot_ax(axes[0, 3], e_t2m[0] - p_t2m[0], f"Error (RMSE={t2m_rmse[i]:.2f})", 'bwr', levels=levels_err_t2m)
                
                # Row 2: T2M Lead 2
                plot_ax(axes[1, 0], g_t2m[1], "GEOS T2M L2 (K)", 'RdBu_r', levels=levels_t2m)
                plot_ax(axes[1, 1], e_t2m[1], "ERA5 T2M L2 (K)", 'RdBu_r', levels=levels_t2m)
                plot_ax(axes[1, 2], p_t2m[1], "Pred T2M L2 (K)", 'RdBu_r', levels=levels_t2m)
                plot_ax(axes[1, 3], e_t2m[1] - p_t2m[1], "Error T2M L2", 'bwr', levels=levels_err_t2m)
                
                # Row 3: Prec Lead 1 (Dynamic Range)
                # Clip data to dynamic max to avoid coloring issues if outliers exist (though levels handle this)
                g_p1 = np.clip(g_prec[0], 0, prec_max_dyn)
                e_p1 = np.clip(e_prec[0], 0, prec_max_dyn)
                p_p1 = np.clip(p_prec[0], 0, prec_max_dyn)
                err_p1 = np.clip(e_prec[0] - p_prec[0], -20, 20)
                
                plot_ax(axes[2, 0], g_p1, "GEOS Prec L1 (mm/d)", 'Blues', levels=levels_prec)
                plot_ax(axes[2, 1], e_p1, "ERA5 Prec L1 (mm/d)", 'Blues', levels=levels_prec)
                plot_ax(axes[2, 2], p_p1, "Pred Prec L1 (mm/d)", 'Blues', levels=levels_prec)
                plot_ax(axes[2, 3], err_p1, f"Error (RMSE={prec_rmse[i]:.2f})", 'bwr', levels=levels_err_prec)
                
                # Row 4: Prec Lead 2
                g_p2 = np.clip(g_prec[1], 0, prec_max_dyn)
                e_p2 = np.clip(e_prec[1], 0, prec_max_dyn)
                p_p2 = np.clip(p_prec[1], 0, prec_max_dyn)
                err_p2 = np.clip(e_prec[1] - p_prec[1], -20, 20)
                
                plot_ax(axes[3, 0], g_p2, "GEOS Prec L2 (mm/d)", 'Blues', levels=levels_prec)
                plot_ax(axes[3, 1], e_p2, "ERA5 Prec L2 (mm/d)", 'Blues', levels=levels_prec)
                plot_ax(axes[3, 2], p_p2, "Pred Prec L2 (mm/d)", 'Blues', levels=levels_prec)
                plot_ax(axes[3, 3], err_p2, "Error Prec L2", 'bwr', levels=levels_err_prec)
                
                plt.suptitle(f"Test Sample {n_samples - geos_raw.shape[0] + i} - Month {months[i].item()+1}", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"test_sample_{n_samples - geos_raw.shape[0] + i}.png"), dpi=150)
                plt.close()
                print(f"Saved plot for test sample {n_samples - geos_raw.shape[0] + i}")

    
    # Compute metrics
    t2m_rmse_mean = np.mean(t2m_errors)
    prec_rmse_mean = np.mean(prec_errors)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Samples evaluated: {n_samples}")
    print(f"T2M RMSE:  {t2m_rmse_mean:.3f} K")
    print(f"Prec RMSE: {prec_rmse_mean:.3f} mm/day")
    
    # Save results
    results_path = os.path.join(args.save_path, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"V8 Test Results\n")
        f.write(f"Samples: {n_samples}\n")
        f.write(f"T2M RMSE: {t2m_rmse_mean:.4f} K\n")
        f.write(f"Prec RMSE: {prec_rmse_mean:.4f} mm/day\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/afahad/project/GEOSS2S3_highres/data/")
    parser.add_argument('--era5_land_path_1', type=str, default="/home/afahad/nb/data/era_land/ERA_land_1990_2005.nc4")
    parser.add_argument('--era5_land_path_2', type=str, default="/home/afahad/nb/data/era_land/ERA_land_2006_2024.nc4")
    parser.add_argument('--elevation_path', type=str, default="/home/afahad/nb/data/era5_land_gepotential.nc", help='Path to Elevation/Geopotential NetCDF')
    parser.add_argument('--save_path', type=str, default="./checkpoints_v9")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--cmde_ratio', type=float, default=0.1, help='CMDE noise ratio for GEOS condition')
    parser.add_argument('--fresh', action='store_true', help='Force re-computation of dataset stats')
    parser.add_argument('--stats_file', type=str, default=None, help='Path to existing stats file (e.g. from V8)')
    parser.add_argument('--diag', action='store_true', help='Run diagnostics on data and exit')
    parser.add_argument('--test', action='store_true', help='Run evaluation on test set (2017+)')
    parser.add_argument('--cache_data', action='store_true')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if args.test:
        run_test(args)
    else:
        main(args)

