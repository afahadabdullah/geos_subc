"""
Conditional Diffusion Model Training for S2S Precipitation Forecasting

Architecture: diffusers.UNet2DModel wrapped in ConditionalDiffusion
Input: 52 channels (4 noisy target + 28 obs + 16 GEOS + 2 seasonality + 2 MJO)
Output: 4 channels (predicted noise ε for 4 lead weeks)
Loss: Area-weighted MSE on noise prediction
Training: CMDE (10% noise on condition) + DDPM noise schedule

Same data pipeline as train_UNET_updated.py (S2SHybridDataset + MJO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os
import argparse
import yaml
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.diffusion import ConditionalDiffusion

import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# UTILITIES
# ==============================================================================

def get_area_weights(lats, device):
    """Area weights based on cosine of latitude, normalized to mean=1."""
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights_tensor.view(1, 1, -1, 1)  # (1, 1, H, 1) broadcasts to (B, 4, H, W)


def load_topography(data_dir):
    """
    Load ERA5 geopotential and interpolate to GEOS 1°x1° grid (181x360).
    Returns z-score normalized tensor of shape (181, 360).
    """
    topo_path = os.path.join(data_dir, "era5_geopotential.nc")
    ds = xr.open_dataset(topo_path)
    varname = list(ds.data_vars)[0]
    da = ds[varname].squeeze()
    
    # Get ERA5 native lat/lon
    era5_lat = da.coords['latitude'].values if 'latitude' in da.coords else da.coords['lat'].values
    era5_lon = da.coords['longitude'].values if 'longitude' in da.coords else da.coords['lon'].values
    era5_data = da.values  # (lat, lon)
    
    # ERA5 lat is often descending (90 to -90) — flip if needed
    if era5_lat[0] > era5_lat[-1]:
        era5_lat = era5_lat[::-1]
        era5_data = era5_data[::-1, :]
    
    # GEOS target grid: 181 x 360 (1° x 1°)
    geos_lat = np.linspace(-90, 90, 181)
    geos_lon = np.linspace(0, 359, 360)
    
    # Handle ERA5 lon convention (-180 to 180 vs 0 to 360)
    if era5_lon.min() < 0:
        # Convert -180..180 to 0..360
        sort_idx = np.argsort((era5_lon % 360))
        era5_lon = era5_lon[sort_idx] % 360
        era5_data = era5_data[:, sort_idx]
    
    # Interpolate
    interp = RegularGridInterpolator(
        (era5_lat, era5_lon), era5_data, method='linear', bounds_error=False, fill_value=None
    )
    geos_lon_grid, geos_lat_grid = np.meshgrid(geos_lon, geos_lat)
    points = np.stack([geos_lat_grid.ravel(), geos_lon_grid.ravel()], axis=-1)
    topo_geos = interp(points).reshape(181, 360)
    
    # Z-score normalize
    topo_mean, topo_std = topo_geos.mean(), topo_geos.std()
    topo_norm = (topo_geos - topo_mean) / (topo_std + 1e-8)
    
    topo_tensor = torch.tensor(topo_norm, dtype=torch.float32)
    print(f"Loaded topography from {topo_path}: ERA5 ({len(era5_lat)}x{len(era5_lon)}) → GEOS ({topo_tensor.shape})")
    return topo_tensor


# ==============================================================================
# TRAINING
# ==============================================================================

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_diffusion.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Accelerator
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # --- Datasets (same as UNet) ---
    preload = config.get("preload", False)
    train_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["train_start_year"],
        end_year=config["train_end_year"],
        normalize=True,
        preload=preload
    )
    loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True
    )

    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=preload
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )

    # Fixed validation batch for consistent plotting
    fixed_val_batch = next(iter(val_loader))

    # --- Target Normalization Constants (Derived from GPCP log1p) ---
    TARGET_MEAN = 0.82
    TARGET_STD = 0.79

    # --- Load Topography (constant, interpolated to GEOS grid) ---
    topo_tensor = load_topography(config["data_dir"])

    # --- Model ---
    # Target: 4 lead weeks of precipitation
    # Condition: 28 obs + 16 GEOS + 2 sin/cos + 2 MJO + 1 topo = 49
    CMDE_RATIO = 0.0  # Completely disabled to prevent any possible mismatch natively
    N_SAMPLES_VAL = 1  # Single reconstruction prediction instead of an ensemble
    INFERENCE_STEPS_VAL = 50  # Faster for validation
    INFERENCE_STEPS_TEST = 1000  # Full schedule for test

    model = ConditionalDiffusion(
        in_channels=4,           # 4 lead weeks (noisy target)
        condition_channels=49,   # 28 obs + 16 GEOS + 2 seasonality + 2 MJO + 1 topo
        out_channels=4,          # predicted noise for 4 leads
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000,
        cmde_ratio=0.0
    )

    total_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"ConditionalDiffusion Parameters: {total_params:,}")
        print(f"Input: 4 noisy + 49 condition = 53 channels")
        print(f"Output: 4 channels (predicted noise)")
        print(f"CMDE ratio: {CMDE_RATIO}, Val members: {N_SAMPLES_VAL}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(config["learning_rate"]),
        steps_per_epoch=len(loader),
        epochs=config["epochs"]
    )

    # Prepare with accelerator
    model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, lr_scheduler
    )

    # Area weights (181 lat points)
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # Output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    log_file = os.path.join(config["output_dir"], "training_log_diffusion.csv")
    if accelerator.is_main_process:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_Loss", "Val_Noise_Loss", "Val_RMSE"])

    # Load checkpoint
    start_epoch = 0
    latest_ckpt = os.path.join(config["output_dir"], "latest_diffusion_ckpt.pt")
    top_k_ckpts = []
    save_top_k = config.get("save_top_k", 4)

    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'top_k_ckpts' in checkpoint:
            top_k_ckpts = checkpoint['top_k_ckpts']
        print(f"Resumed from epoch {start_epoch}")

    best_val_rmse = float('inf')
    if top_k_ckpts:
        best_val_rmse = top_k_ckpts[0][0]
        print(f"Resumed Best Val RMSE: {best_val_rmse:.4f}")

    # ==================================================================
    # TRAINING LOOP
    # ==================================================================
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0.0
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for batch in pbar:
            x_obs = batch['x_obs']           # (B, 28, H, W)
            x_geos = batch['x_geos']         # (B, 4, 1, 4, H, W)
            y_target = batch['y_target']     # (B, 4, H, W) - RAW mm/day
            months = batch['month']          # (B,)
            mjo = batch['mjo']              # (B, 2)

            B, _, H, W = x_obs.shape

            # Flatten GEOS: (B, 4, 1, 4, H, W) -> (B, 16, H, W)
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)

            # Month embeddings: sin/cos spatial maps
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)

            # MJO: broadcast (B, 2) -> (B, 2, H, W)
            mjo_map = mjo.view(B, 2, 1, 1).expand(B, 2, H, W).to(device)

            # Topography: (H, W) -> (B, 1, H, W)
            topo_batch = topo_tensor.to(device).unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

            # Condition: (B, 49, H, W)
            condition = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, mjo_map, topo_batch], dim=1)

            # Convert RAW target to log1p space before normalization
            y_log = torch.log1p(y_target.clamp(min=0.0))

            # Normalize perfectly to ~N(0, 1) using GPCP specific stats
            target_norm = (y_log - TARGET_MEAN) / TARGET_STD

            # Sample random timesteps
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # Add FULL noise to target
            noise = torch.randn_like(target_norm)
            noisy_target = model.noise_scheduler.add_noise(target_norm, noise, timesteps)

            # CMDE: Add REDUCED noise to condition
            alpha_hat = model.noise_scheduler.alphas_cumprod.to(device)[timesteps]
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_hat).view(-1, 1, 1, 1)
            cond_noise = CMDE_RATIO * sqrt_one_minus_alpha * torch.randn_like(condition)
            noisy_condition = condition + cond_noise

            # Predict noise
            noise_pred = model(noisy_target, noisy_condition, timesteps)

            # Area-weighted MSE loss
            loss = (area_weights * (noise_pred - noise)**2).mean()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(loader)

        # ==============================================================
        # VALIDATION: Noise MSE
        # ==============================================================
        model.eval()
        val_loss_sum = 0
        val_count = 0

        v_rmse_sum = 0.0
        v_rmse_count = 0
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                vx_obs = val_batch['x_obs'].to(device)
                vx_geos = val_batch['x_geos'].to(device)
                vy_target = val_batch['y_target'].to(device)
                v_months = val_batch['month'].to(device)
                v_mjo = val_batch['mjo'].to(device)

                vB, _, vH, vW = vx_obs.shape
                vx_geos_flat = vx_geos.squeeze(2).reshape(vB, 16, vH, vW)

                v_sin = torch.sin(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_cos = torch.cos(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_mjo_map = v_mjo.view(vB, 2, 1, 1).expand(vB, 2, vH, vW).to(device)
                v_topo = topo_tensor.to(device).unsqueeze(0).unsqueeze(0).expand(vB, 1, vH, vW)
                v_condition = torch.cat([vx_obs, vx_geos_flat, v_sin, v_cos, v_mjo_map, v_topo], dim=1)

                # 1. Global Noise Loss (for convergence tracking)
                vy_log = torch.log1p(vy_target.clamp(min=0.0))
                vtarget_norm = (vy_log - TARGET_MEAN) / TARGET_STD

                v_timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (vB,), device=device).long()
                v_noise = torch.randn_like(vtarget_norm)
                v_noisy = model.noise_scheduler.add_noise(vtarget_norm, v_noise, v_timesteps)
                v_pred = model(v_noisy, v_condition, v_timesteps)

                v_loss = (area_weights * (v_pred - v_noise)**2).mean()
                val_loss_sum += v_loss.item()
                val_count += 1

                # 2. Sampled Weighted RMSE (for physical fidelity tracking, first 10 batches)
                if i < 10:
                    v_sample_norm = unwrapped_model.sample(v_condition, num_inference_steps=INFERENCE_STEPS_VAL)
                    
                    # Denormalize using GPCP target constants
                    v_sample_denorm = (v_sample_norm * TARGET_STD) + TARGET_MEAN
                        
                    v_pred_mm = torch.expm1(v_sample_denorm.clamp(min=0.0, max=6.0))
                    
                    v_diff_sq = (v_pred_mm - vy_target.to(device))**2
                    v_weighted_mse = (v_diff_sq * area_weights).mean()
                    v_rmse_sum += torch.sqrt(v_weighted_mse).item()
                    v_rmse_count += 1
                    
                    if i == 0 and accelerator.is_main_process:
                        print(f"DEBUG | Val Batch 0 Stats | Target: {vy_target.min().item():.2f}/{vy_target.max().item():.2f} "
                              f"| Pred: {v_pred_mm.min().item():.2f}/{v_pred_mm.max().item():.2f} "
                              f"| LogPred: {v_sample_denorm.min().item():.2f}/{v_sample_denorm.max().item():.2f}")

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
        val_rmse = v_rmse_sum / v_rmse_count if v_rmse_count > 0 else 0

        # ==============================================================
        # FIXED BATCH: Generate Ensemble + RMSE
        # ==============================================================
        fb_obs = fixed_val_batch['x_obs'].to(device)
        fb_geos = fixed_val_batch['x_geos'].to(device)
        fb_target = fixed_val_batch['y_target'].to(device)
        fb_months = fixed_val_batch['month'].to(device)
        fb_mjo = fixed_val_batch['mjo'].to(device)

        fb_B = fb_obs.shape[0]
        _, _, H, W = fb_obs.shape
        fb_geos_flat = fb_geos.squeeze(2).reshape(fb_B, 16, H, W)

        fb_sin = torch.sin(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_cos = torch.cos(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_mjo_map = fb_mjo.view(fb_B, 2, 1, 1).expand(fb_B, 2, H, W).to(device)
        fb_topo = topo_tensor.to(device).unsqueeze(0).unsqueeze(0).expand(fb_B, 1, H, W)

        fb_cond = torch.cat([fb_obs, fb_geos_flat, fb_sin, fb_cos, fb_mjo_map, fb_topo], dim=1)

        # Generate N_SAMPLES_VAL ensemble members
        with torch.no_grad():
            ensemble_samples = []
            for i_ens in range(N_SAMPLES_VAL):
                sample_norm = unwrapped_model.sample(fb_cond, num_inference_steps=INFERENCE_STEPS_VAL)

                # Denormalize
                sample_denorm = (sample_norm * TARGET_STD) + TARGET_MEAN

                # Convert from log1p space to mm/day (clamp to prevent overflow)
                sample_precip = torch.expm1(sample_denorm.clamp(min=0.0, max=6.0))
                sample_precip = sample_precip.clamp(min=0.0)
                ensemble_samples.append(sample_precip)

            # Stack: list of (B, 4, H, W) -> (N, B, 4, H, W)
            ens_stack = torch.stack(ensemble_samples, dim=0)
            ens_mean = ens_stack.mean(dim=0)  # (B, 4, H, W)

        # Target is already in mm/day (raw from dataset)
        fb_target_mm = fb_target

        # RMSE of ensemble mean
        val_rmse = torch.sqrt(torch.mean((ens_mean - fb_target_mm)**2)).item()

        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Noise Loss: {avg_val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, avg_val_loss, val_rmse])

            # ==============================================================
            # PLOT: Validation Plot (Either New Best or every 5 Epochs)
            # ==============================================================
            if val_rmse < best_val_rmse or epoch % 5 == 0:
                if val_rmse < best_val_rmse:
                    print(f"New Best! RMSE improved from {best_val_rmse:.4f} to {val_rmse:.4f}. Plotting...")
                    best_val_rmse = val_rmse
                else:
                    print(f"Periodic Plotting at Epoch {epoch}...")

                # First sample only, all 4 leads
                t_img_all = fb_target_mm[0].cpu().numpy()         # (4, H, W) mm/day
                ens_mean_img = ens_mean[0].cpu().numpy()          # (4, H, W)

                # One individual sample for display
                sample_img = ens_stack[0, 0].cpu().numpy()  # (4, H, W)

                # GEOS Mean (denormalize)
                g_flat = fb_geos_flat[0]  # (16, H, W)
                g_ens = g_flat.view(4, 4, H, W)  # (Members, Leads, H, W)
                g_mean_log = g_ens.mean(dim=0)    # (4, H, W)
                g_img_all = np.expm1(np.maximum(g_mean_log.cpu().numpy(), 0.0))

                # Plot: 4 rows (leads) x 6 cols: Target | GEOS | Sample | EnsMean | GEOS Bias | Diff Bias
                fig, axes = plt.subplots(4, 6, figsize=(30, 16))

                for l_idx in range(4):
                    t_img = t_img_all[l_idx]
                    g_img = g_img_all[l_idx]
                    s_img = np.clip(sample_img[l_idx], 0, None)
                    m_img = ens_mean_img[l_idx]
                    geos_bias = g_img - t_img
                    diff_bias = m_img - t_img
                    
                    g_rmse = np.sqrt(np.mean(geos_bias**2))
                    d_rmse = np.sqrt(np.mean(diff_bias**2))

                    if l_idx == 0: axes[l_idx, 0].set_title("Target GPCP")
                    axes[l_idx, 0].imshow(t_img, cmap='Blues', vmin=0, vmax=50)
                    axes[l_idx, 0].set_ylabel(f"Week {l_idx+1}")

                    if l_idx == 0: axes[l_idx, 1].set_title("GEOS Mean")
                    axes[l_idx, 1].imshow(g_img, cmap='Blues', vmin=0, vmax=50)

                    if l_idx == 0: axes[l_idx, 2].set_title("Diff Sample")
                    axes[l_idx, 2].imshow(s_img, cmap='Blues', vmin=0, vmax=50)

                    if l_idx == 0: axes[l_idx, 3].set_title("Reconstruction")
                    axes[l_idx, 3].imshow(m_img, cmap='Blues', vmin=0, vmax=50)

                    if l_idx == 0: axes[l_idx, 4].set_title("GEOS Bias")
                    axes[l_idx, 4].imshow(geos_bias, cmap='RdBu_r', vmin=-20, vmax=20)
                    axes[l_idx, 4].set_ylabel(f"RMSE: {g_rmse:.2f}")

                    if l_idx == 0: axes[l_idx, 5].set_title("Diff Bias")
                    axes[l_idx, 5].imshow(diff_bias, cmap='RdBu_r', vmin=-20, vmax=20)
                    axes[l_idx, 5].set_ylabel(f"RMSE: {d_rmse:.2f}")

                os.makedirs(os.path.join(config["output_dir"], "plots_diffusion"), exist_ok=True)
                plt.suptitle(f"Diffusion - Epoch {epoch} | RMSE: {val_rmse:.2f} | Noise Loss: {avg_val_loss:.4f}", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(config["output_dir"], f"plots_diffusion/epoch_{epoch}_rmse_{val_rmse:.2f}.png"), dpi=150)
                plt.close()
            else:
                print(f"Val RMSE ({val_rmse:.4f}) did not improve over best ({best_val_rmse:.4f}).")

            # ==============================================================
            # SAVE CHECKPOINTS
            # ==============================================================
            ckpt_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'top_k_ckpts': top_k_ckpts
            }
            torch.save(ckpt_state, latest_ckpt)

            # Top K logic
            current_path = os.path.join(config["output_dir"], f"diffusion_epoch_{epoch}_rmse_{val_rmse:.4f}.pt")
            top_k_ckpts.append((val_rmse, epoch, current_path))
            top_k_ckpts.sort(key=lambda x: x[0])

            if len(top_k_ckpts) > save_top_k:
                worst = top_k_ckpts.pop()
                if worst[2] != current_path and os.path.exists(worst[2]):
                    os.remove(worst[2])
                    print(f"Removed worse checkpoint: {worst[2]}")

            is_in_top = any(x[2] == current_path for x in top_k_ckpts)
            if is_in_top:
                print(f"New Top Model! RMSE: {val_rmse:.4f}")
                torch.save(ckpt_state, current_path)

            ckpt_state['top_k_ckpts'] = top_k_ckpts
            torch.save(ckpt_state, latest_ckpt)


# ==============================================================================
# TEST SUITE
# ==============================================================================

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_diffusion.yaml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config["num_workers"], pin_memory=True
    )

    # --- Load Topography (constant, interpolated to GEOS grid) ---
    topo_tensor = load_topography(config["data_dir"])

    # Model (same architecture as training)
    model = ConditionalDiffusion(
        in_channels=4,
        condition_channels=49,
        out_channels=4,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000,
        cmde_ratio=0.0
    ).to(device)
    # Load best checkpoint
    latest_ckpt = os.path.join(config["output_dir"], "latest_diffusion_ckpt.pt")
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
        top_k = checkpoint.get('top_k_ckpts', [])

        if top_k:
            best_ckpt_path = top_k[0][2]
            print(f"Loading best model from {best_ckpt_path} (RMSE: {top_k[0][0]:.4f})")
            if os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model'])
            else:
                print("Best model file missing, loading latest.")
                model.load_state_dict(checkpoint['model'])
        else:
            print("No top_k info, loading latest.")
            model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found!")
        return

    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()

    import cartopy.crs as ccrs

    test_indices = [0, 10, 20, 30, 40]
    output_dir = os.path.join(config["output_dir"], "plots_test_suite")
    os.makedirs(output_dir, exist_ok=True)

    N_MEMBERS_TEST = 5
    INFERENCE_STEPS_TEST = 1000

    print(f"Test Suite: indices {test_indices}, {N_MEMBERS_TEST} members, {INFERENCE_STEPS_TEST} steps")

    # Accumulators for summary
    rmse_diff = {l: [] for l in range(4)}
    rmse_geos = {l: [] for l in range(4)}

    current_idx = 0
    samples_processed = 0

    with torch.no_grad():
        for batch in val_loader:
            if current_idx not in test_indices:
                current_idx += 1
                continue

            print(f"Processing sample {current_idx}...")

            x_obs = batch['x_obs'].to(device)
            x_geos = batch['x_geos'].to(device)
            y_target = batch['y_target'].to(device)
            t_months = batch['month'].to(device)
            t_mjo = batch['mjo'].to(device)

            B, _, H, W = x_obs.shape
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)

            t_sin = torch.sin(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            t_cos = torch.cos(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            t_mjo_map = t_mjo.view(B, 2, 1, 1).expand(B, 2, H, W).to(device)
            t_topo = topo_tensor.to(device).unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

            condition = torch.cat([x_obs, x_geos_flat, t_sin, t_cos, t_mjo_map, t_topo], dim=1)

            unwrapped = accelerator.unwrap_model(model)

            # Generate ensemble
            ensemble = []
            for i_ens in range(N_MEMBERS_TEST):
                print(f"  Member {i_ens+1}/{N_MEMBERS_TEST}")
                gen_norm = unwrapped.sample(condition, num_inference_steps=INFERENCE_STEPS_TEST, verbose=True)

                # Denormalize
                if val_dataset.geos_mean is not None:
                    gm = val_dataset.geos_mean.to(device)
                    gs = val_dataset.geos_std.to(device)
                    gen = (gen_norm * gs) + gm
                else:
                    gen = gen_norm

                # log1p -> mm/day
                gen_mm = torch.expm1(gen.clamp(min=0.0, max=6.0)).clamp(min=0.0)
                ensemble.append(gen_mm)

            # Stack and compute mean
            ens_all = torch.stack(ensemble, dim=0)  # (N, 1, 4, H, W)
            ens_mean = ens_all.mean(dim=0)           # (1, 4, H, W)

            # Target is already in mm/day
            target_mm = y_target.clamp(min=0.0)
            target_np = target_mm.squeeze(0).cpu().numpy()    # (4, H, W)
            ens_mean_np = ens_mean.squeeze(0).cpu().numpy()   # (4, H, W)

            # GEOS Mean (denormalize)
            geos_ens = x_geos_flat.view(4, 4, H, W)
            geos_mean_log = geos_ens.mean(dim=0)
            geos_np = torch.expm1(geos_mean_log.clamp(min=0.0, max=6.0)).cpu().numpy()

            # RMSE
            for lead in range(4):
                rmse_diff[lead].append(np.sqrt(np.mean((ens_mean_np[lead] - target_np[lead])**2)))
                rmse_geos[lead].append(np.sqrt(np.mean((geos_np[lead] - target_np[lead])**2)))

            # --- Per-Sample Plot: 6 cols with cartopy ---
            lats = np.linspace(-90, 90, H)
            lons = np.linspace(0, 360, W)

            # One sample for display
            sample_np = ens_all[0, 0].cpu().numpy()  # (4, H, W)

            fig = plt.figure(figsize=(30, 16))

            for lead in range(4):
                g_img = geos_np[lead]
                t_img = target_np[lead]
                s_img = np.clip(sample_np[lead], 0, None)
                m_img = ens_mean_np[lead]
                geos_bias = g_img - t_img
                diff_bias = m_img - t_img

                g_rmse = np.sqrt(np.mean(geos_bias**2))
                d_rmse = np.sqrt(np.mean(diff_bias**2))

                # Target
                ax = fig.add_subplot(4, 6, lead * 6 + 1, projection=ccrs.PlateCarree())
                ax.imshow(t_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Target", fontsize=9)

                # GEOS
                ax = fig.add_subplot(4, 6, lead * 6 + 2, projection=ccrs.PlateCarree())
                ax.imshow(g_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: GEOS", fontsize=9)

                # Diff Sample
                ax = fig.add_subplot(4, 6, lead * 6 + 3, projection=ccrs.PlateCarree())
                ax.imshow(s_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Diff Sample", fontsize=9)

                # Ens Mean
                ax = fig.add_subplot(4, 6, lead * 6 + 4, projection=ccrs.PlateCarree())
                ax.imshow(m_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Ens Mean", fontsize=9)

                # GEOS Bias
                ax = fig.add_subplot(4, 6, lead * 6 + 5, projection=ccrs.PlateCarree())
                ax.imshow(geos_bias, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-20, vmax=20)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: GEOS Bias\nRMSE: {g_rmse:.2f}", fontsize=9)

                # Diff Bias
                ax = fig.add_subplot(4, 6, lead * 6 + 6, projection=ccrs.PlateCarree())
                ax.imshow(diff_bias, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-20, vmax=20)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Diff Bias\nRMSE: {d_rmse:.2f}", fontsize=9)

            plt.suptitle(f"Diffusion Test — Sample {current_idx}", fontsize=14)
            plt.savefig(os.path.join(output_dir, f"test_sample_{current_idx}.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()
            print(f"  Saved plot for sample {current_idx}.")

            samples_processed += 1
            current_idx += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Test Suite Summary ({samples_processed} samples)")
    print(f"{'='*60}")
    for lead in range(4):
        avg_d = np.mean(rmse_diff[lead])
        avg_g = np.mean(rmse_geos[lead])
        print(f"  Week {lead+1}: Diffusion RMSE={avg_d:.2f}, GEOS RMSE={avg_g:.2f}")
    print(f"{'='*60}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_diffusion.yaml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        test()
    else:
        train()
