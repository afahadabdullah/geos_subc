import os
import sys
import ctypes

# --- TACC/Remote Fix: Preload Conda libstdc++ ---
try:
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        libstd = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
        if os.path.exists(libstd):
            ctypes.CDLL(libstd, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
# ------------------------------------------------

import torch
import numpy as np


import matplotlib.pyplot as plt

def denormalize(x):
    """
    Inverse min-max normalization for log1p-transformed precipitation.
    Inverse of (log1p(x) - min) / (max - min).
    Automatically loads stats from norm_stats.json.
    """
    # Load stats (MUST exist — run calculate_stats.py first)
    import json
    stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"norm_stats.json not found at {stats_path}. "
            f"Run `python ml_model/calculate_stats.py` first to generate it."
        )
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    vmin = stats["log1p_min"]
    vmax = stats["log1p_max"]
    
    # x is (..., H, W)
    # Undo [-1, 1] scaling -> [0, 1]
    x = (x + 1.0) / 2.0
    
    # Undo min-max scaling: x * (max - min) + min
    denom = vmax - vmin if vmax != vmin else 1.0
    x = x * denom + vmin
    
    # Undo log1p with safety clamp to prevent overflow (inf)
    # log1p(22000) ~ 10.0. Max float32 exp is ~88.
    if isinstance(x, torch.Tensor):
        x = torch.clamp(x, max=15.0)  # Safety clamp
        x = torch.expm1(x)
        x = torch.clamp(x, min=0.0)
    else:
        x = np.clip(x, a_min=None, a_max=15.0) # Safety clamp
        x = np.expm1(x)
        x = np.maximum(x, 0.0)
    return x


def denormalize_residual(residual_norm, forecast_norm):
    """
    Convert a normalized residual back to physical precipitation (mm/day).
    
    Steps:
    1. Undo [-1, 1] scaling on residual using residual_min/max
    2. Undo [-1, 1] scaling on forecast using global min/max  
    3. Combine: log1p(prediction) = residual_log + forecast_log
    4. expm1 to get mm/day
    
    Args:
        residual_norm: Normalized residual in [-1, 1] (from DDIM output)
        forecast_norm: Normalized forecast in [-1, 1] (input condition)
    Returns:
        Physical precipitation in mm/day
    """
    import json
    stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.json")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    vmin = stats["log1p_min"]
    vmax = stats["log1p_max"]
    res_min = stats.get("residual_min", -5.0)
    res_max = stats.get("residual_max", 5.0)
    
    # 1. Undo [-1, 1] on residual -> log-space residual
    res_denom = res_max - res_min if res_max != res_min else 1.0
    residual_log = (residual_norm + 1.0) / 2.0 * res_denom + res_min
    
    # 2. Undo [-1, 1] on forecast -> log-space forecast
    denom = vmax - vmin if vmax != vmin else 1.0
    forecast_log = (forecast_norm + 1.0) / 2.0 * denom + vmin
    
    # 3. Combine: prediction_log = forecast_log + residual_log
    prediction_log = forecast_log + residual_log
    
    # 4. Undo log1p
    # Clamp to avoid overflow issues with expm1
    # Max value for log1p is ~10 (precip ~22000 mm/day)
    # Max value for float32 exp is ~88
    # Clamp to 10.0, which is ~22000 mm/day (physically extreme but safe)
    if isinstance(prediction_log, torch.Tensor):
        prediction_log = torch.clamp(prediction_log, max=10.0) 
        predicted_physical = torch.expm1(prediction_log)
        predicted_physical = torch.clamp(predicted_physical, min=0.0) # Ensure non-negative precip
    else:
        prediction_log = np.clip(prediction_log, a_min=None, a_max=10.0) 
        predicted_physical = np.expm1(prediction_log)
        predicted_physical = np.maximum(predicted_physical, 0.0) # Ensure non-negative precip
    
    return predicted_physical

def denormalize_zscore(x, mean_map, std_map):
    """
    Inverse Z-Score normalization.
    x: Normalized input (C, H, W) or (B, C, H, W)
    mean_map: (H, W)
    std_map: (H, W)
    """
    # Inverse: x * std + mean
    # Ensure maps are broadcastable
    # If x is Tensor, maps should be Tensor.
    if isinstance(x, torch.Tensor):
        if not isinstance(mean_map, torch.Tensor):
             mean_map = torch.tensor(mean_map, device=x.device, dtype=x.dtype)
             std_map = torch.tensor(std_map, device=x.device, dtype=x.dtype)
        # Reshape maps to (1, 1, H, W) or (1, H, W) depending on x
        if x.ndim == 4: # (B, C, H, W)
            mu = mean_map.view(1, 1, *x.shape[-2:])
            sigma = std_map.view(1, 1, *x.shape[-2:])
        else: # (C, H, W)
            mu = mean_map.view(1, *x.shape[-2:])
            sigma = std_map.view(1, *x.shape[-2:])
        
        x = x * sigma + mu
        x = torch.clamp(x, max=15.0) # Safety
        x = torch.expm1(x)
        x = torch.clamp(x, min=0.0)
    else:
        # Numpy
        mu = mean_map
        sigma = std_map
        x = x * sigma + mu
        x = np.clip(x, a_min=None, a_max=15.0)
        x = np.expm1(x)
        x = np.maximum(x, 0.0)
    return x

def denormalize_residual_zscore(residual_norm, forecast_norm, 
                                resid_mean, resid_std, 
                                geos_mean, geos_std):
    """
    Inverse Z-Score for residual.
    """
    # 1. Residual log
    if isinstance(residual_norm, torch.Tensor):
        if not isinstance(resid_mean, torch.Tensor):
             resid_mean = torch.tensor(resid_mean, device=residual_norm.device, dtype=residual_norm.dtype)
             resid_std = torch.tensor(resid_std, device=residual_norm.device, dtype=residual_norm.dtype)
             geos_mean = torch.tensor(geos_mean, device=forecast_norm.device, dtype=forecast_norm.dtype)
             geos_std = torch.tensor(geos_std, device=forecast_norm.device, dtype=forecast_norm.dtype)
             
        if residual_norm.ndim == 4:
            r_mu = resid_mean.view(1, 1, *residual_norm.shape[-2:])
            r_sigma = resid_std.view(1, 1, *residual_norm.shape[-2:])
            f_mu = geos_mean.view(1, 1, *forecast_norm.shape[-2:])
            f_sigma = geos_std.view(1, 1, *forecast_norm.shape[-2:])
        else:
            r_mu = resid_mean.view(1, *residual_norm.shape[-2:])
            r_sigma = resid_std.view(1, *residual_norm.shape[-2:])
            f_mu = geos_mean.view(1, *forecast_norm.shape[-2:])
            f_sigma = geos_std.view(1, *forecast_norm.shape[-2:])
            
        res_log = residual_norm * r_sigma + r_mu
        fore_log = forecast_norm * f_sigma + f_mu
        
        pred_log = fore_log + res_log
        pred_log = torch.clamp(pred_log, max=10.0)
        phy = torch.expm1(pred_log)
        phy = torch.clamp(phy, min=0.0)
    else:
        # Numpy
        res_log = residual_norm * resid_std + resid_mean
        fore_log = forecast_norm * geos_std + geos_mean
        pred_log = fore_log + res_log
        pred_log = np.clip(pred_log, a_min=None, a_max=10.0)
        phy = np.expm1(pred_log)
        phy = np.maximum(phy, 0.0)
        
    return phy
    """
    Compute Continuous Ranked Probability Score (CRPS) for an ensemble forecast.
    observations: (B, ...) denormalized
    forecasts: (B, M, ...) denormalized
    """
    obs = observations.unsqueeze(1)
    term_1 = torch.abs(forecasts - obs).mean(dim=1)
    f1 = forecasts.unsqueeze(2)
    f2 = forecasts.unsqueeze(1)
    # Handle possible OOM for very large grids by avoiding full expansion if needed
    term_2 = torch.abs(f1 - f2).mean(dim=(1, 2))
    crps = term_1 - 0.5 * term_2
    return crps.mean()

def plot_comparison(input_geos, target_gpcp, ens_mean, save_path, title="Validation Comparison"):
    """
    Comprehensive validation visualization suite.
    Shows all 4 lead weeks with GEOS Input, GPCP Truth,
    Ensemble Mean, GEOS Bias (GPCP−GEOS), and Model Residual (GPCP−EnsMean).
    
    All inputs are denormalized numpy arrays: (4, H, W)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import TwoSlopeNorm
    
    n_weeks = input_geos.shape[0]
    week_labels = [f"LW{i+1}" for i in range(n_weeks)]
    
    # Clean NaNs
    def clean(arr):
        return np.nan_to_num(arr, nan=0.0)
    
    geos = clean(input_geos)
    gpcp = clean(target_gpcp)
    ens  = clean(ens_mean)
    
    # Difference maps
    bias     = gpcp - geos  # GPCP − GEOS (positive = GEOS underestimates)
    residual = gpcp - ens   # GPCP − Ensemble Mean (positive = model underestimates)
    
    # --- Layout: 4 rows (weeks) × 5 columns ---
    fig = plt.figure(figsize=(25, 20))
    gs = gridspec.GridSpec(n_weeks, 5, wspace=0.15, hspace=0.3)
    
    col_titles = ["GEOS Input", "GPCP Truth",
                   "Ensemble Mean", "Bias (GPCP−GEOS)", "Residual (GPCP−EnsMean)"]
    
    # Precip colormap limits (99.9th percentile of truth/input for natural scaling)
    all_precip = np.concatenate([geos.flatten(), gpcp.flatten()])
    vmax_precip = float(np.percentile(all_precip, 99.9)) + 1.0
    if vmax_precip <= 0:
        vmax_precip = 10.0
    
    # Difference colormap limits (symmetric) - Determined by GEOS bias only
    diff_abs_max = max(
        float(np.percentile(np.abs(bias), 99)),
        0.5
    )
    
    for row in range(n_weeks):
        panels = [geos[row], gpcp[row], ens[row], bias[row], residual[row]]
        
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            data = panels[col]
            
            if col < 3:
                # Precipitation panels (sequential colormap)
                im = ax.imshow(data, cmap='YlGnBu', vmin=0, vmax=vmax_precip, origin='upper')
            else:
                # Difference panels (diverging colormap)
                norm = TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max)
                im = ax.imshow(data, cmap='BrBG', norm=norm, origin='upper')
            
            # Stats annotation
            rmse_val = np.sqrt(np.mean(data**2)) if col >= 3 else np.mean(data)
            stat_label = f"RMSE={rmse_val:.2f}" if col >= 3 else f"Mean={rmse_val:.2f}"
            ax.text(0.02, 0.98, stat_label, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Row labels (left side)
            if col == 0:
                ax.set_ylabel(week_labels[row], fontsize=12, fontweight='bold')
            
            # Column titles (top row only)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Colorbars
    cbar_ax1 = fig.add_axes([0.08, 0.04, 0.50, 0.015])
    sm1 = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(0, vmax_precip))
    sm1.set_array([])
    fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal', label='Precipitation (mm/day)')
    
    # Difference colorbar
    cbar_ax2 = fig.add_axes([0.63, 0.04, 0.30, 0.015])
    sm2 = plt.cm.ScalarMappable(cmap='BrBG', norm=TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max))
    sm2.set_array([])
    fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal', label='Difference (mm/day)')
    
    fig.suptitle(f"{title}  |  Precip vmax={vmax_precip:.1f}  |  Diff range=±{diff_abs_max:.1f}",
                 fontsize=14, fontweight='bold', y=0.98)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def compute_mse(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

def compute_mjo_rmse(pred_rmm, target_rmm):
    return torch.sqrt(torch.mean((pred_rmm - target_rmm)**2))


def plot_normalization_diagnostic(forecast, target, observed, save_path):
    """
    Plot histograms and stats of normalized inputs to verify they are in [-1, 1].
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    data_list = [
        ("Forecast", forecast.flatten()),
        ("Target", target.flatten()),
        ("Observed", observed.flatten())
    ]
    
    for ax, (name, data) in zip(axes, data_list):
        ax.hist(data, bins=50, alpha=0.7, color='blue', density=True)
        ax.set_title(f"{name}\nMin:{data.min():.2f} Max:{data.max():.2f}\nMean:{data.mean():.2f} Std:{data.std():.2f}")
        ax.axvline(-1, color='r', linestyle='--')
        ax.axvline(1, color='r', linestyle='--')
        ax.set_xlim(-1.5, 1.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
