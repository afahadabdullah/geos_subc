import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def denormalize(x):
    """Inverse of log1p(x) / 4.0"""
    return torch.expm1(x * 4.0)

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
    idx = 3
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Common colorbar limits for comparison
    vmax = max(input_geos[idx].max(), target_gpcp[idx].max(), prediction[idx].max(), ens_mean[idx].max())
    vmin = 0
    
    im0 = axes[0].imshow(input_geos[idx], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[0].set_title("GEOS Input (LW4)")
    
    im1 = axes[1].imshow(target_gpcp[idx], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[1].set_title("GPCP Ground Truth")
    
    im2 = axes[2].imshow(prediction[idx], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[2].set_title("Model Prediction (1 sample)")
    
    im3 = axes[3].imshow(ens_mean[idx], cmap='Blues', vmin=vmin, vmax=vmax)
    axes[3].set_title("Ensemble Mean (3 samples)")
    
    plt.colorbar(im3, ax=axes, orientation='horizontal', label='Precipitation (mm/day)', shrink=0.6)
    fig.suptitle(title)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def compute_mse(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

def compute_mjo_rmse(pred_rmm, target_rmm):
    return torch.sqrt(torch.mean((pred_rmm - target_rmm)**2))
