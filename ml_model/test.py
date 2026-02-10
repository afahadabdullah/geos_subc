"""
CMDE Diffusion Model - Test / Evaluation Mode
================================================
Generates ensemble predictions for random validation samples across all lead times.
Uses cartopy for publication-quality map visualizations with coastlines, borders, and lat/lon.

Usage:
    python ml_model/test.py --checkpoint ml_output_cmde/best_model_epoch_10 --n_samples 5 --n_ensemble 5
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from ml_model.dataset import GeosSubCDataset
    from ml_model.model import ConditionalUNet, GaussianDiffusion
    from ml_model.utils import denormalize
except ImportError:
    from dataset import GeosSubCDataset
    from model import ConditionalUNet, GaussianDiffusion
    from utils import denormalize

# Matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_args():
    parser = argparse.ArgumentParser(description="CMDE Test Mode: Ensemble Prediction Visualization")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint directory. If None, auto-detects best model.")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of random validation samples to visualize")
    parser.add_argument("--n_ensemble", type=int, default=5,
                        help="Number of ensemble members per sample")
    parser.add_argument("--output_dir", type=str, default="ml_output_cmde/test_plots",
                        help="Directory to save test plots")
    parser.add_argument("--data_root", type=str, default="dataprocess",
                        help="Root directory for data")
    parser.add_argument("--val_years", type=int, nargs=2, default=[2015, 2016],
                        help="Validation year range")
    parser.add_argument("--ddpm_steps", type=int, default=50,
                        help="Number of DDPM reverse steps (50 is fast, 200 is high quality)")
    return parser.parse_args()


def find_best_checkpoint(output_dir="ml_output_cmde"):
    """Find the best checkpoint directory automatically."""
    best_models = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("best_model_epoch_")],
        key=lambda x: int(x.split("_")[-1])
    )
    if best_models:
        return os.path.join(output_dir, best_models[-1])
    
    latest = os.path.join(output_dir, "latest_checkpoint")
    if os.path.exists(latest):
        return latest
    
    raise FileNotFoundError(f"No checkpoints found in {output_dir}")


@torch.no_grad()
def ddpm_sample(model, diffusion, forecast, mjo_map, month_onehot, n_steps=50, image_size=(181, 360)):
    """
    Full DDPM reverse diffusion sampling (accelerated with step skipping).
    Returns a single denoised prediction: (4, H, W)
    """
    device = forecast.device
    bs = forecast.shape[0]
    
    # Start from pure noise
    x_t = torch.randn(bs, 4, image_size[0], image_size[1], device=device)
    
    # Evenly spaced timesteps for acceleration
    timestep_indices = np.linspace(diffusion.timesteps - 1, 0, n_steps, dtype=int)
    
    for i, t_val in enumerate(timestep_indices):
        t = torch.tensor([t_val], device=device).expand(bs)
        
        # Build model input
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[t].view(-1, 1, 1, 1)
        cond_noise = torch.randn_like(forecast)
        noisy_forecast = forecast + (0.1 * sqrt_one_minus_alpha * cond_noise)
        
        model_input = torch.cat([x_t, noisy_forecast, mjo_map], dim=1)
        pred_noise = model(model_input, t, month_onehot)
        
        # DDPM update step
        alpha_t = diffusion.alphas[t_val]
        alpha_hat_t = diffusion.alpha_hats[t_val]
        beta_t = diffusion.betas[t_val]
        
        # Predicted x_0
        sqrt_alpha_hat = diffusion.sqrt_alpha_hats[t_val]
        sqrt_one_minus_alpha_hat = diffusion.sqrt_one_minus_alpha_hats[t_val]
        x0_pred = (x_t - sqrt_one_minus_alpha_hat * pred_noise) / (sqrt_alpha_hat + 1e-8)
        
        # Compute x_{t-1}
        if i < len(timestep_indices) - 1:
            # Not the last step: add noise
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            # Simplified DDPM step
            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / sqrt_one_minus_alpha_hat) * pred_noise
            ) + sigma * noise
        else:
            # Last step: deterministic
            x_t = x0_pred
    
    return x_t


def plot_test_sample(geos_input, gpcp_truth, ensemble_preds, ens_mean,
                     sample_idx, init_date, save_dir, lats, lons):
    """
    Publication-quality cartopy plot for a single test sample.
    
    Layout: 4 rows (LW1-LW4) × 5 columns:
       GEOS Input | GPCP Truth | Ensemble Mean | Bias (GPCP-GEOS) | Residual (GPCP-EnsMean)
    
    All inputs are denormalized numpy arrays: (4, H, W)
    """
    n_weeks = geos_input.shape[0]
    week_labels = [f"Lead Week {i+1}" for i in range(n_weeks)]
    
    # Clean NaNs
    geos = np.nan_to_num(geos_input, nan=0.0)
    gpcp = np.nan_to_num(gpcp_truth, nan=0.0)
    ens  = np.nan_to_num(ens_mean, nan=0.0)
    
    bias = gpcp - geos      # GPCP - GEOS
    residual = gpcp - ens   # GPCP - Ensemble Mean
    
    # Colormap limits
    all_precip = np.concatenate([geos.flatten(), gpcp.flatten()])
    vmax_precip = float(np.percentile(all_precip, 99.5)) + 0.5
    if vmax_precip <= 0:
        vmax_precip = 10.0
    
    diff_abs_max = max(
        float(np.percentile(np.abs(bias), 99)),
        float(np.percentile(np.abs(residual), 99)),
        0.5
    )
    
    # Map projection
    proj = ccrs.PlateCarree()
    
    fig = plt.figure(figsize=(28, 22))
    gs = gridspec.GridSpec(n_weeks, 5, wspace=0.08, hspace=0.12,
                           left=0.05, right=0.95, top=0.93, bottom=0.08)
    
    col_titles = ["GEOS Input", "GPCP Truth", "Ensemble Mean",
                  "Bias (GPCP−GEOS)", "Residual (GPCP−EnsMean)"]
    
    for row in range(n_weeks):
        panels = [geos[row], gpcp[row], ens[row], bias[row], residual[row]]
        
        for col in range(5):
            ax = fig.add_subplot(gs[row, col], projection=proj)
            data = panels[col]
            
            if col < 3:
                # Precipitation (sequential)
                im = ax.pcolormesh(lons, lats, data, cmap='Blues',
                                   vmin=0, vmax=vmax_precip,
                                   transform=ccrs.PlateCarree(), shading='auto')
            else:
                # Difference (diverging)
                norm = TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max)
                im = ax.pcolormesh(lons, lats, data, cmap='RdBu',
                                   norm=norm, transform=ccrs.PlateCarree(), shading='auto')
            
            # Map features
            ax.coastlines(linewidth=0.5, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
            ax.set_global()
            
            # Gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                              alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if col > 0:
                gl.left_labels = False
            if row < n_weeks - 1:
                gl.bottom_labels = False
            gl.xlabel_style = {'fontsize': 7}
            gl.ylabel_style = {'fontsize': 7}
            
            # Per-panel stats
            if col < 3:
                stat_text = f"Mean={np.mean(data):.2f}"
            else:
                stat_text = f"RMSE={np.sqrt(np.mean(data**2)):.2f}"
            ax.text(0.02, 0.97, stat_text, transform=ax.transAxes, fontsize=7,
                    va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
            
            # Row labels
            if col == 0:
                ax.text(-0.08, 0.5, week_labels[row], transform=ax.transAxes,
                        fontsize=11, fontweight='bold', va='center', rotation=90)
            
            # Column titles
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=8)
    
    # Colorbars
    cbar_ax1 = fig.add_axes([0.05, 0.04, 0.52, 0.015])
    sm1 = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, vmax_precip))
    sm1.set_array([])
    fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal', label='Precipitation (mm/day)')
    
    cbar_ax2 = fig.add_axes([0.62, 0.04, 0.33, 0.015])
    sm2 = plt.cm.ScalarMappable(cmap='RdBu',
                                norm=TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max))
    sm2.set_array([])
    fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal', label='Difference (mm/day)')
    
    fig.suptitle(
        f"CMDE Test | Sample {sample_idx} | Init: {init_date} | "
        f"{len(ensemble_preds)}-Member Ensemble | Precip vmax={vmax_precip:.1f}",
        fontsize=14, fontweight='bold', y=0.97
    )
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"test_sample_{sample_idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def run_test(args):
    """Main test function."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- Find Checkpoint ---
    if args.checkpoint is None:
        args.checkpoint = find_best_checkpoint()
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # --- Dataset ---
    print(f"Loading validation data ({args.val_years[0]}-{args.val_years[1]})...")
    val_dataset = GeosSubCDataset(
        data_root=args.data_root,
        start_year=args.val_years[0],
        end_year=args.val_years[1],
        mjo_file="mjo_processed.csv",
        preload=True
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    # --- Model ---
    image_size = (181, 360)
    in_channels = 10
    out_channels = 4
    
    model = ConditionalUNet(in_channels=in_channels, out_channels=out_channels, base_filters=64)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # Load checkpoint via accelerate state
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model, dummy_optimizer, dummy_dataloader = accelerator.prepare(
        model, dummy_optimizer, dummy_dataloader
    )
    
    accelerator.load_state(args.checkpoint)
    
    # Move diffusion to device
    diffusion.device = accelerator.device
    diffusion.betas = diffusion.betas.to(accelerator.device)
    diffusion.alphas = diffusion.alphas.to(accelerator.device)
    diffusion.alpha_hats = diffusion.alpha_hats.to(accelerator.device)
    diffusion.sqrt_alpha_hats = diffusion.sqrt_alpha_hats.to(accelerator.device)
    diffusion.sqrt_one_minus_alpha_hats = diffusion.sqrt_one_minus_alpha_hats.to(accelerator.device)
    
    model.eval()
    print("Model loaded successfully.")
    
    # --- Lat/Lon Grid (1° global) ---
    lats = np.linspace(90, -90, image_size[0])
    lons = np.linspace(0, 359, image_size[1])
    
    # --- Random Sample Selection ---
    np.random.seed(42)
    n_total = len(val_dataset)
    sample_indices = np.random.choice(n_total, size=min(args.n_samples, n_total), replace=False)
    sample_indices.sort()
    
    print(f"\nGenerating {args.n_ensemble}-member ensemble predictions for {len(sample_indices)} samples...")
    print(f"DDPM reverse steps: {args.ddpm_steps}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    for i, s_idx in enumerate(sample_indices):
        batch = val_dataset[s_idx]
        init_date = batch["S"]
        
        print(f"\n[{i+1}/{len(sample_indices)}] Sample {s_idx} | Init: {init_date}")
        
        # Move to device
        forecast = batch["input_forecast"].unsqueeze(0).to(accelerator.device)
        target = batch["target_truth"].unsqueeze(0).to(accelerator.device)
        mjo = batch["mjo_conditioning"].unsqueeze(0).to(accelerator.device)
        month_oh = batch["month_onehot"].unsqueeze(0).to(accelerator.device)
        
        mjo_map = mjo.view(1, 2, 1, 1).expand(-1, -1, image_size[0], image_size[1])
        
        # Generate ensemble
        ensemble_preds = []
        for m in range(args.n_ensemble):
            print(f"  Ensemble member {m+1}/{args.n_ensemble}...", end=" ", flush=True)
            pred = ddpm_sample(model, diffusion, forecast, mjo_map, month_oh,
                              n_steps=args.ddpm_steps, image_size=image_size)
            pred_denorm = denormalize(pred[0]).detach().cpu().numpy()
            ensemble_preds.append(pred_denorm)
            print(f"Done (mean={pred_denorm.mean():.2f})")
        
        ens_mean = np.mean(ensemble_preds, axis=0)
        
        # Denormalize inputs
        geos_raw = denormalize(forecast[0]).detach().cpu().numpy()
        gpcp_raw = denormalize(target[0]).detach().cpu().numpy()
        
        # Plot
        plot_test_sample(
            geos_input=geos_raw,
            gpcp_truth=gpcp_raw,
            ensemble_preds=ensemble_preds,
            ens_mean=ens_mean,
            sample_idx=s_idx,
            init_date=init_date,
            save_dir=args.output_dir,
            lats=lats,
            lons=lons
        )
    
    print(f"\n{'=' * 60}")
    print(f"Test complete! {len(sample_indices)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
