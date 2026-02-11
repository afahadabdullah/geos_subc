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
    Inverse of (log1p(x) - mean) / std.
    Automatically loads stats from norm_stats.json if available.
    """
    # Load stats just like dataset.py
    import json
    stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        mean = stats.get("log1p_mean", 0.0)
        std = stats.get("log1p_std", 4.0)
    else:
        mean = 0.0
        std = 4.0
    
    # x is (..., H, W)
    # Undo scaling
    val = x * std + mean
    # expm1 and clip to 0 for physical consistency
    res = torch.expm1(val)
    return torch.clamp(res, min=0.0)

def crps_ensemble(observations, forecasts):
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

def plot_comparison(input_geos, target_gpcp, prediction, ens_mean, save_path, title="Validation Comparison"):
    """
    Comprehensive validation visualization suite.
    Shows all 4 lead weeks with GEOS Input, GPCP Truth, Model Prediction,
    Ensemble Mean, GEOS Bias (GPCP−GEOS), and Model Residual (GPCP−Pred).
    
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
    pred = clean(prediction)
    ens  = clean(ens_mean)
    
    # Difference maps
    bias     = gpcp - geos  # GPCP − GEOS (positive = GEOS underestimates)
    residual = gpcp - ens   # GPCP − Ensemble Mean (positive = model underestimates)
    
    # --- Layout: 4 rows (weeks) × 6 columns ---
    fig = plt.figure(figsize=(30, 20))
    gs = gridspec.GridSpec(n_weeks, 6, wspace=0.15, hspace=0.3)
    
    col_titles = ["GEOS Input", "GPCP Truth", "Prediction (1 sample)", 
                   "Ensemble Mean (3)", "Bias (GPCP−GEOS)", "Residual (GPCP−EnsMean)"]
    
    # Precip colormap limits (99.9th percentile of truth/input for natural scaling)
    all_precip = np.concatenate([geos.flatten(), gpcp.flatten()])
    vmax_precip = float(np.percentile(all_precip, 99.9)) + 1.0
    if vmax_precip <= 0:
        vmax_precip = 10.0
    
    # Difference colormap limits (symmetric)
    diff_abs_max = max(
        float(np.percentile(np.abs(bias), 99)),
        float(np.percentile(np.abs(residual), 99)),
        0.5
    )
    
    for row in range(n_weeks):
        panels = [geos[row], gpcp[row], pred[row], ens[row], bias[row], residual[row]]
        
        for col in range(6):
            ax = fig.add_subplot(gs[row, col])
            data = panels[col]
            
            if col < 4:
                # Precipitation panels (sequential colormap)
                im = ax.imshow(data, cmap='Blues', vmin=0, vmax=vmax_precip, origin='upper')
            else:
                # Difference panels (diverging colormap)
                norm = TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max)
                im = ax.imshow(data, cmap='RdBu', norm=norm, origin='upper')
            
            # Stats annotation
            rmse_val = np.sqrt(np.mean(data**2)) if col >= 4 else np.mean(data)
            stat_label = f"RMSE={rmse_val:.2f}" if col >= 4 else f"Mean={rmse_val:.2f}"
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
    # Precipitation colorbar (spans columns 0-3)
    cbar_ax1 = fig.add_axes([0.08, 0.04, 0.55, 0.015])
    sm1 = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, vmax_precip))
    sm1.set_array([])
    fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal', label='Precipitation (mm/day)')
    
    # Difference colorbar (spans columns 4-5)
    cbar_ax2 = fig.add_axes([0.68, 0.04, 0.25, 0.015])
    sm2 = plt.cm.ScalarMappable(cmap='RdBu', norm=TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max))
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
