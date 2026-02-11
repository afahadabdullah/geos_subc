"""
CMDE Diffusion Model - Test / Evaluation (Z-Score Experiment)
==============================================================
Generates ensemble predictions for validation samples across all lead times.
Uses per-grid Z-Score denormalization (stats_z.nc maps).

Usage:
    python ml_model/testv2.py --n_samples 6 --n_ensemble 5
    python ml_model/testv2.py --checkpoint ml_output_zscore/best_model_epoch_10 --ddpm_steps 50
"""
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xarray as xr
import pandas as pd

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from ml_model.dataset import GeosSubCDataset
    from ml_model.model import ConditionalUNet, GaussianDiffusion
    from ml_model.utils import denormalize_zscore, denormalize_residual_zscore
except ImportError:
    from dataset import GeosSubCDataset
    from model import ConditionalUNet, GaussianDiffusion
    from utils import denormalize_zscore, denormalize_residual_zscore

# Matplotlib backend for headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---- Helpers ----

def get_geos_ens_mean(data_root, init_date_str):
    """Load full GEOS ensemble for the given date and compute the mean (mm/day)."""
    date = pd.to_datetime(init_date_str)
    year = date.year
    zarr_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        precip = ds['pr'].sel(S=date).values
        if precip.ndim == 4:
            n_members = precip.shape[0]
            ens_mean = np.mean(precip, axis=0)
        else:
            n_members = 1
            ens_mean = precip
        return ens_mean, n_members
    except Exception as e:
        print(f"Warning: Could not load GEOS ensemble for {init_date_str}: {e}")
        return None, 0


def select_seasonal_samples(dataset, n_samples=6):
    """Select 6 samples spaced throughout the year (Jan, Mar, May, Jul, Sep, Nov)."""
    target_months = [1, 3, 5, 7, 9, 11]
    indices = []
    found_months = set()
    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        s_date = sample['S']
        month = s_date.month
        m_idx = sample['M']
        if month in target_months and month not in found_months and m_idx == 0:
            indices.append(idx)
            found_months.add(month)
            if len(found_months) == len(target_months):
                break
    if len(indices) < 6:
        print(f"  Warning: Only found {len(indices)} seasonal samples. Filling remaining randomly.")
        remaining = n_samples - len(indices)
        pool = [i for i in range(len(dataset)) if i not in indices]
        if pool:
            fill = np.random.choice(pool, size=min(remaining, len(pool)), replace=False)
            indices.extend(list(fill))
    indices.sort()
    return np.array(indices)


# ---- CLI ----

def parse_args():
    parser = argparse.ArgumentParser(description="CMDE Test Mode (Z-Score Experiment)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. If None, auto-detects best model in ml_output_zscore/.")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of validation samples to visualize")
    parser.add_argument("--n_ensemble", type=int, default=5,
                        help="Number of ensemble members per sample")
    parser.add_argument("--output_dir", type=str, default="ml_output_zscore/test_plots",
                        help="Directory to save test plots")
    parser.add_argument("--data_root", type=str, default="dataprocess",
                        help="Root directory for data")
    parser.add_argument("--val_years", type=int, nargs=2, default=[2015, 2016],
                        help="Validation year range")
    parser.add_argument("--ddpm_steps", type=int, default=1000,
                        help="Number of reverse steps (1000 = Full DDPM, <1000 = Accelerated DDIM)")
    return parser.parse_args()


def find_best_checkpoint(output_dir="ml_output_zscore"):
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


# ---- Samplers ----

@torch.no_grad()
def ddpm_sample_full(model, diffusion, forecast, observed, mjo_map, month_onehot, image_size=(181, 360), cmde_ratio=0.1):
    """Standard DDPM sampling (Full 1000 steps)."""
    device = forecast.device
    bs = forecast.shape[0]
    x = torch.randn(bs, 4, image_size[0], image_size[1], device=device)
    
    for t in tqdm(reversed(range(diffusion.timesteps)), desc="DDPM Sampling", total=diffusion.timesteps, leave=False):
        t_tensor = torch.tensor([t], device=device).expand(bs)
        
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[t_tensor].view(-1, 1, 1, 1)
        cond_noise = torch.randn_like(forecast)
        noisy_forecast = forecast + (cmde_ratio * sqrt_one_minus_alpha * cond_noise)
        
        model_input = torch.cat([x, noisy_forecast, observed, mjo_map], dim=1)
        pred_noise = model(model_input, t_tensor, month_onehot)
        
        alpha_t = diffusion.alphas[t]
        alpha_hat_t = diffusion.alpha_hats[t]
        beta_t = diffusion.betas[t]
        sqrt_one_minus_alpha_hat_t = diffusion.sqrt_one_minus_alpha_hats[t]
        
        pred_x0 = (x - sqrt_one_minus_alpha_hat_t * pred_noise) / (torch.sqrt(alpha_hat_t) + 1e-8)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        pred_noise_clamped = (x - torch.sqrt(alpha_hat_t) * pred_x0) / (sqrt_one_minus_alpha_hat_t + 1e-8)
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alpha_hat_t) * pred_noise_clamped
        ) + torch.sqrt(beta_t) * noise
        
    return x


@torch.no_grad()
def ddim_sample(model, diffusion, forecast, observed, mjo_map, month_onehot, n_steps=50, image_size=(181, 360), eta=0.0, cmde_ratio=0.1):
    """DDIM sampler with accelerated stepping."""
    device = forecast.device
    bs = forecast.shape[0]
    timestep_indices = np.linspace(diffusion.timesteps - 1, 0, n_steps, dtype=int)
    x = torch.randn(bs, 4, image_size[0], image_size[1], device=device)
    
    for i, t_curr in enumerate(timestep_indices):
        t_tensor = torch.tensor([t_curr], device=device).expand(bs)
        
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[t_tensor].view(-1, 1, 1, 1)
        cond_noise = torch.randn_like(forecast)
        noisy_forecast = forecast + (cmde_ratio * sqrt_one_minus_alpha * cond_noise)
        
        model_input = torch.cat([x, noisy_forecast, observed, mjo_map], dim=1)
        pred_noise = model(model_input, t_tensor, month_onehot)
        
        alpha_bar_t = diffusion.alpha_hats[t_curr]
        
        if i < len(timestep_indices) - 1:
            t_prev = timestep_indices[i+1]
            alpha_bar_t_prev = diffusion.alpha_hats[t_prev]
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)
            
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t + 1e-8)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        
        term_1 = torch.sqrt(alpha_bar_t_prev) * pred_x0
        term_2 = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * pred_noise
        x = term_1 + term_2
        
        if eta > 0:
            noise = torch.randn_like(x)
            x += sigma_t * noise
            
    return x


# ---- Plot ----

def plot_test_sample(geos_ens_mean, gpcp_truth, ensemble_preds, ens_mean,
                     sample_idx, init_date, save_dir, lats, lons, n_geos_ens=4):
    """Publication-quality cartopy plot for a single test sample."""
    n_weeks = geos_ens_mean.shape[0]
    n_ens = len(ensemble_preds)
    week_labels = [f"Lead Week {i+1}" for i in range(n_weeks)]
    
    geos = np.nan_to_num(geos_ens_mean, nan=0.0)
    gpcp = np.nan_to_num(gpcp_truth, nan=0.0)
    ens  = np.nan_to_num(ens_mean, nan=0.0)
    
    geos_bias = gpcp - geos
    model_bias = gpcp - ens
    
    ens_stack = np.stack([np.nan_to_num(e, nan=0.0) for e in ensemble_preds], axis=0)
    spread = np.std(ens_stack, axis=0)
    improvement = np.abs(geos_bias) - np.abs(model_bias)
    
    all_precip = np.concatenate([geos.flatten(), gpcp.flatten()])
    vmax_precip = float(np.percentile(all_precip, 99.5)) + 0.5
    if vmax_precip <= 0:
        vmax_precip = 10.0
    
    diff_abs_max = max(
        float(np.percentile(np.abs(geos_bias), 99)),
        float(np.percentile(np.abs(model_bias), 99)),
        0.5
    )
    vmax_spread = max(float(np.percentile(spread, 99.5)), 0.1)
    imp_abs_max = max(float(np.percentile(np.abs(improvement), 99)), 0.5)
    
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(42, 22))
    gs = gridspec.GridSpec(n_weeks, 7, wspace=0.06, hspace=0.10,
                           left=0.04, right=0.96, top=0.93, bottom=0.08)
    
    col_titles = [
        "GPCP Target (Truth)",
        f"GEOS Ens Mean ({n_geos_ens})",
        f"Model Ens Mean ({n_ens})",
        "GEOS Bias (GPCP−GeosMean)",
        "Model Bias (GPCP−ModMean)",
        "Ensemble Spread (Std)",
        "Improvement (vs GEOS Mean)"
    ]
    
    for row in range(n_weeks):
        panels = [gpcp[row], geos[row], ens[row], geos_bias[row], model_bias[row], spread[row], improvement[row]]
        
        for col in range(7):
            ax = fig.add_subplot(gs[row, col], projection=proj)
            data = panels[col]
            
            if col < 3:
                im = ax.pcolormesh(lons, lats, data, cmap='YlGnBu',
                                   vmin=0, vmax=vmax_precip,
                                   transform=ccrs.PlateCarree(), shading='auto')
            elif col < 5:
                norm = TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max)
                im = ax.pcolormesh(lons, lats, data, cmap='BrBG',
                                   norm=norm, transform=ccrs.PlateCarree(), shading='auto')
            elif col == 5:
                im = ax.pcolormesh(lons, lats, data, cmap='YlOrRd',
                                   vmin=0, vmax=vmax_spread,
                                   transform=ccrs.PlateCarree(), shading='auto')
            else:
                norm = TwoSlopeNorm(vcenter=0, vmin=-imp_abs_max, vmax=imp_abs_max)
                im = ax.pcolormesh(lons, lats, data, cmap='RdBu',
                                   norm=norm, transform=ccrs.PlateCarree(), shading='auto')
            
            ax.coastlines(linewidth=0.5, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
            ax.set_global()
            
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
            
            if col < 3:
                stat_text = f"Mean={np.mean(data):.2f}"
            elif col < 5:
                stat_text = f"RMSE={np.sqrt(np.mean(data**2)):.2f}"
            elif col == 5:
                stat_text = f"Avg Std={np.mean(data):.3f}"
            else:
                pct_improved = float(np.sum(data > 0)) / data.size * 100
                stat_text = f"Improved={pct_improved:.0f}%"
            
            ax.text(0.02, 0.97, stat_text, transform=ax.transAxes, fontsize=7,
                    va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
            
            if col == 0:
                ax.text(-0.08, 0.5, week_labels[row], transform=ax.transAxes,
                        fontsize=11, fontweight='bold', va='center', rotation=90)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold', pad=8)
    
    # Colorbars
    cbar_ax1 = fig.add_axes([0.04, 0.04, 0.35, 0.012])
    sm1 = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(0, vmax_precip))
    sm1.set_array([])
    fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal', label='Precipitation (mm/day)')
    
    cbar_ax2 = fig.add_axes([0.43, 0.04, 0.22, 0.012])
    sm2 = plt.cm.ScalarMappable(cmap='BrBG',
                                norm=TwoSlopeNorm(vcenter=0, vmin=-diff_abs_max, vmax=diff_abs_max))
    sm2.set_array([])
    fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal', label='Difference (mm/day)')
    
    cbar_ax3 = fig.add_axes([0.69, 0.04, 0.10, 0.012])
    sm3 = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, vmax_spread))
    sm3.set_array([])
    fig.colorbar(sm3, cax=cbar_ax3, orientation='horizontal', label='Ensemble Spread (Std)')
    
    cbar_ax4 = fig.add_axes([0.82, 0.04, 0.14, 0.012])
    sm4 = plt.cm.ScalarMappable(cmap='RdBu',
                                norm=TwoSlopeNorm(vcenter=0, vmin=-imp_abs_max, vmax=imp_abs_max))
    sm4.set_array([])
    fig.colorbar(sm4, cax=cbar_ax4, orientation='horizontal', label='Improvement (Blue=Model Better)')
    
    fig.suptitle(
        f"CMDE Z-Score Test | Sample {sample_idx} | Init: {init_date} | "
        f"{n_ens}-Member Ensemble",
        fontsize=14, fontweight='bold', y=0.97
    )
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"test_sample_{sample_idx}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


# ---- Main ----

def run_test(args):
    """Main test function using Z-Score normalization."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --- Find Checkpoint ---
    if args.checkpoint is None:
        args.checkpoint = find_best_checkpoint("ml_output_zscore")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # --- Dataset (Z-Score) ---
    print(f"Loading validation data ({args.val_years[0]}-{args.val_years[1]}) with Z-Score normalization...")
    val_dataset = GeosSubCDataset(
        data_root=args.data_root,
        start_year=args.val_years[0],
        end_year=args.val_years[1],
        mjo_file="mjo_processed.csv",
        preload=True,
        normalization="zscore"
    )
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load Z-Score maps for denormalization
    maps = val_dataset.maps
    
    # --- Model ---
    image_size = (181, 360)
    model = ConditionalUNet(in_channels=14, out_channels=4, base_filters=128)
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # Load checkpoint via accelerate
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    
    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model, dummy_optimizer, dummy_dataloader = accelerator.prepare(
        model, dummy_optimizer, dummy_dataloader
    )
    
    accelerator.load_state(args.checkpoint)
    
    diffusion.device = accelerator.device
    diffusion.betas = diffusion.betas.to(accelerator.device)
    diffusion.alphas = diffusion.alphas.to(accelerator.device)
    diffusion.alpha_hats = diffusion.alpha_hats.to(accelerator.device)
    diffusion.sqrt_alpha_hats = diffusion.sqrt_alpha_hats.to(accelerator.device)
    diffusion.sqrt_one_minus_alpha_hats = diffusion.sqrt_one_minus_alpha_hats.to(accelerator.device)
    
    model.eval()
    print("Model loaded successfully.")
    
    # Move maps to GPU for denormalization
    gpu_maps = {}
    for k, v in maps.items():
        gpu_maps[k] = torch.tensor(v, device=accelerator.device, dtype=torch.float32)
    
    # --- Lat/Lon Grid ---
    lats = np.linspace(90, -90, image_size[0])
    lons = np.linspace(0, 359, image_size[1])
    
    # --- Sample Selection ---
    if args.n_samples == 6:
        print("Selecting seasonal samples (Jan, Mar, May, Jul, Sep, Nov)...")
        sample_indices = select_seasonal_samples(val_dataset)
    else:
        np.random.seed(42)
        n_total = len(val_dataset)
        sample_indices = np.random.choice(n_total, size=min(args.n_samples, n_total), replace=False)
        sample_indices.sort()
    
    print(f"\nGenerating {args.n_ensemble}-member ensemble for {len(sample_indices)} samples...")
    print(f"DDPM reverse steps: {args.ddpm_steps}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    for i, s_idx in enumerate(sample_indices):
        batch = val_dataset[s_idx]
        init_date = batch["S"]
        
        print(f"\n[{i+1}/{len(sample_indices)}] Sample {s_idx} | Init: {init_date}")
        
        forecast = batch["input_forecast"].unsqueeze(0).to(accelerator.device)
        target = batch["target_truth"].unsqueeze(0).to(accelerator.device)
        observed = batch["observed_state"].unsqueeze(0).to(accelerator.device)
        mjo = batch["mjo_conditioning"].unsqueeze(0).to(accelerator.device)
        month_oh = batch["month_onehot"].unsqueeze(0).to(accelerator.device)
        
        mjo_map = mjo.view(1, 2, 1, 1).expand(-1, -1, image_size[0], image_size[1])
        
        # Generate ensemble
        ensemble_preds = []
        for m in range(args.n_ensemble):
            print(f"  Ensemble member {m+1}/{args.n_ensemble}...", end=" ", flush=True)
            
            if args.ddpm_steps >= 1000:
                pred = ddpm_sample_full(model, diffusion, forecast, observed, mjo_map, month_oh, image_size=image_size)
            else:
                pred = ddim_sample(model, diffusion, forecast, observed, mjo_map, month_oh,
                                   n_steps=args.ddpm_steps, image_size=image_size)
            
            # Z-Score denormalization: residual + forecast -> mm/day
            pred_denorm = denormalize_residual_zscore(
                pred[0], forecast[0],
                gpu_maps["resid_mean"], gpu_maps["resid_std"],
                gpu_maps["geos_mean"], gpu_maps["geos_std"]
            ).detach().cpu().numpy()
            
            ensemble_preds.append(pred_denorm)
            print(f"Done (mean={pred_denorm.mean():.2f})")
        
        ens_mean = np.mean(ensemble_preds, axis=0)
        
        # Denormalize inputs for plotting
        geos_raw_single = denormalize_zscore(
            forecast[0], gpu_maps["geos_mean"], gpu_maps["geos_std"]
        ).detach().cpu().numpy()
        
        gpcp_raw = denormalize_zscore(
            target[0], gpu_maps["gpcp_mean"], gpu_maps["gpcp_std"]
        ).detach().cpu().numpy()
        
        # Fetch full GEOS ensemble mean
        geos_ens_mean_full, n_geos = get_geos_ens_mean(args.data_root, init_date)
        
        if geos_ens_mean_full is None:
             print("  Warning: Using single member GEOS as mean (fallback).")
             geos_final = geos_raw_single
             n_geos_final = 1
        else:
             geos_final = geos_ens_mean_full
             n_geos_final = n_geos
        
        # Plot
        plot_test_sample(
            geos_ens_mean=geos_final,
            gpcp_truth=gpcp_raw,
            ensemble_preds=ensemble_preds,
            ens_mean=ens_mean,
            sample_idx=s_idx,
            init_date=init_date,
            save_dir=args.output_dir,
            lats=lats,
            lons=lons,
            n_geos_ens=n_geos_final
        )
    
    print(f"\n{'=' * 60}")
    print(f"Test complete! {len(sample_indices)} plots saved to {args.output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
