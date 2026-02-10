import torch
import numpy as np
import matplotlib.pyplot as plt
import os

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
    Plots Lead Week 4 comparison.
    input_geos: (4, H, W) Lead Week 4 is index 3
    target_gpcp: (4, H, W)
    prediction: (4, H, W)
    ens_mean: (4, H, W)
    """
    # Use Lead Week 4 (Index 3) for visualization
    # If the input has only 1 lead week or a different shape, handle it gracefully
    idx = min(3, input_geos.shape[0] - 1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Handle NaNs and outliers for visualization robustness
    def clean(arr):
        return np.nan_to_num(arr, nan=0.0)

    p0 = clean(input_geos[idx])
    p1 = clean(target_gpcp[idx])
    p2 = clean(prediction[idx])
    p3 = clean(ens_mean[idx])
    
    # Dynamic vmax but capped to avoid washing out due to extreme predictions
    # 99th percentile of truth/input is a good "natural" cap for the colormap
    natural_max = np.percentile(np.concatenate([p0.flatten(), p1.flatten()]), 99.9) + 1.0
    vmax = min(max(p0.max(), p1.max(), p2.max(), p3.max()), natural_max * 2.0)
    if vmax <= 0: vmax = 10.0 # Safety
    
    vmin = 0
    
    im0 = axes[0].imshow(p0, cmap='Blues', vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_title("GEOS Input (LW4)")
    
    im1 = axes[1].imshow(p1, cmap='Blues', vmin=vmin, vmax=vmax, origin='upper')
    axes[1].set_title("GPCP Ground Truth")
    
    im2 = axes[2].imshow(p2, cmap='Blues', vmin=vmin, vmax=vmax, origin='upper')
    axes[2].set_title("Model Prediction (1 sample)")
    
    im3 = axes[3].imshow(p3, cmap='Blues', vmin=vmin, vmax=vmax, origin='upper')
    axes[3].set_title("Ensemble Mean (3 samples)")
    
    plt.colorbar(im3, ax=axes, orientation='horizontal', label='Precipitation (mm/day)', shrink=0.6)
    fig.suptitle(f"{title} (vmax={vmax:.1f})")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def compute_mse(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

def compute_mjo_rmse(pred_rmm, target_rmm):
    return torch.sqrt(torch.mean((pred_rmm - target_rmm)**2))
