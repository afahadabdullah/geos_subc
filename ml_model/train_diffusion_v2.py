"""
Conditional Diffusion Model v2 Training for S2S Precipitation Forecasting

Architecture: diffusers.UNet2DModel wrapped in ConditionalDiffusionV2
Input: 49 channels (1 noisy target lead + 28 obs + 16 GEOS flat + 2 seasonality + 2 MJO)
Conditioned on class_labels = lead_index [0, 1, 2, 3]
Output: 1 channel (predicted noise ε for the target lead)
Loss: Area-weighted MSE on noise prediction
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
from ml_model.diffusion_v2 import ConditionalDiffusionV2

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
    return weights_tensor.view(1, 1, -1, 1)  # (1, 1, H, 1) broadcasts


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

    # --- Datasets ---
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

    # Fixed validation batch
    fixed_val_batch = next(iter(val_loader))

    # --- Target Normalization Constants (SymLog bounds for residual) ---
    TARGET_SYMLOG_MIN = -6.55
    TARGET_SYMLOG_MAX = 6.55

    # --- Model ---
    # Target: 1 lead week
    # Condition: 28 obs + 16 GEOS + 2 sin/cos + 2 MJO = 48
    N_SAMPLES_VAL = 1
    INFERENCE_STEPS_VAL = 50

    model = ConditionalDiffusionV2(
        in_channels=1,           
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000
    ).to(device)

    if accelerator.is_main_process:
        print("--- Training Dataset Bounds ---")
        print(f"GEOS  : min={train_dataset.geos_min:.2f}, max={train_dataset.geos_max:.2f}")
        print(f"SST   : min={train_dataset.sst_min:.2f}, max={train_dataset.sst_max:.2f}")
        print(f"SSS   : min={train_dataset.sss_min:.2f}, max={train_dataset.sss_max:.2f}")
        print(f"SM    : min={train_dataset.sm_min:.2f}, max={train_dataset.sm_max:.2f}")
        print(f"IVT   : min={train_dataset.ivt_min:.2f}, max={train_dataset.ivt_max:.2f}")
        print(f"Z500  : min={train_dataset.z500_min:.2f}, max={train_dataset.z500_max:.2f}")
        print(f"U250  : min={train_dataset.u250_min:.2f}, max={train_dataset.u250_max:.2f}")
        print("-------------------------------")

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"ConditionalDiffusionV2 Parameters: {total_params:,}")
        print(f"Input: 1 noisy + 48 cond = 49 channels")
        print(f"Output: 1 channel (predicted noise)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] * len(loader),
        eta_min=1e-6
    )

    model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, lr_scheduler
    )

    # Area weights (181 lat points)
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # Output directory
    output_dir = config["output_dir"] + "_v2"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log_diffusion.csv")
    if accelerator.is_main_process:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_Loss", "Val_Noise_Loss", "Val_RMSE"])

    # Load checkpoint
    start_epoch = 0
    latest_ckpt = os.path.join(output_dir, "latest_diffusion_ckpt.pt")
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

        for i, batch in enumerate(pbar):
            x_obs = batch['x_obs']           
            x_geos = batch['x_geos']         
            y_target = batch['y_target']     
            months = batch['month']          
            mjo = batch['mjo']              

            B, _, H, W = x_obs.shape

            # Flatten GEOS: (B, 4, 1, 4, H, W) -> (B, 16, H, W)
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
            
            # Normalize GEOS condition
            if train_dataset.geos_min is not None:
                x_geos_flat = 2.0 * (x_geos_flat - train_dataset.geos_min.to(device)) / (train_dataset.geos_max.to(device) - train_dataset.geos_min.to(device)) - 1.0
            
            # (Prev-GPCP channels 12-15 are already min-max normalized to [-1, 1] in dataset_hybrid.py)

            # Month embeddings
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)

            # MJO
            mjo_map = mjo.view(B, 2, 1, 1).expand(B, 2, H, W).to(device)

            # Condition: (B, 48, H, W)
            condition = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, mjo_map], dim=1)

            # Single Lead Selection
            target_lead = torch.randint(0, 4, (1,)).item()
            lead_indices = torch.full((B,), target_lead, dtype=torch.long, device=device)
            y_target_lead = y_target[:, target_lead:target_lead+1, :, :]

            # Compute GEOS ensemble mean in raw mm/day for the target lead
            x_geos_raw = torch.expm1(x_geos.clamp(max=6.55))
            geos_raw_mean = x_geos_raw.mean(dim=1).squeeze(1) # (B, 4, H, W)
            geos_mean_lead = geos_raw_mean[:, target_lead:target_lead+1, :, :]

            # Compute raw residual target: GPCP - GEOS
            y_residual = y_target_lead - geos_mean_lead

            # SymLog normalize the residual
            y_symlog = torch.sign(y_residual) * torch.log1p(torch.abs(y_residual))
            target_norm = 2.0 * (y_symlog - TARGET_SYMLOG_MIN) / (TARGET_SYMLOG_MAX - TARGET_SYMLOG_MIN) - 1.0

            # Sample random timesteps
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            if epoch == start_epoch and i == 0 and accelerator.is_main_process:
                print(f"\\nDEBUG INITIAL BATCH {epoch}")
                print(f"Raw Target lead:   min={y_target_lead.min():.4f}, max={y_target_lead.max():.4f}")
                print(f"Residual Target:   min={y_residual.min():.4f}, max={y_residual.max():.4f}")
                print(f"SymLog Target  :   min={y_symlog.min():.4f}, max={y_symlog.max():.4f}")
                print(f"Norm Target [-1,1]: min={target_norm.min():.4f}, max={target_norm.max():.4f}")
                print(f"GEOS Cond [-1, 1] : min={x_geos_flat.min():.4f}, max={x_geos_flat.max():.4f}")
                print(f"Obs Cond  [-1, 1] : min={x_obs.min():.4f}, max={x_obs.max():.4f}")
                print(f"Prev-GPCP [-1, 1] : min={x_obs[:,12:16].min():.4f}, max={x_obs[:,12:16].max():.4f}")

            noise = torch.randn_like(target_norm)
            noisy_target = model.noise_scheduler.add_noise(target_norm, noise, timesteps)

            # Predict noise
            noise_pred = model(noisy_target, condition, lead_indices, timesteps)

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
        # VALIDATION: Iterative per-lead reconstruction
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

                if train_dataset.geos_min is not None:
                    vx_geos_flat = 2.0 * (vx_geos_flat - train_dataset.geos_min.to(device)) / (train_dataset.geos_max.to(device) - train_dataset.geos_min.to(device)) - 1.0
                # (Prev-GPCP channels already normalized in dataset)

                v_sin = torch.sin(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_cos = torch.cos(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_mjo_map = v_mjo.view(vB, 2, 1, 1).expand(vB, 2, vH, vW).to(device)
                v_condition = torch.cat([vx_obs, vx_geos_flat, v_sin, v_cos, v_mjo_map], dim=1)

                v_pred_leads_mm = []

                # Reconstruct each lead
                vx_geos_raw = torch.expm1(vx_geos.clamp(max=6.55))
                vgeos_raw_mean = vx_geos_raw.mean(dim=1).squeeze(1)

                for lead in range(4):
                    lead_indices = torch.full((vB,), lead, dtype=torch.long, device=device)
                    vy_target_lead = vy_target[:, lead:lead+1, :, :]
                    vgeos_mean_lead = vgeos_raw_mean[:, lead:lead+1, :, :]
                    
                    # 1. Global Noise Loss
                    vy_residual = vy_target_lead - vgeos_mean_lead
                    vy_symlog = torch.sign(vy_residual) * torch.log1p(torch.abs(vy_residual))
                    vtarget_norm = 2.0 * (vy_symlog - TARGET_SYMLOG_MIN) / (TARGET_SYMLOG_MAX - TARGET_SYMLOG_MIN) - 1.0

                    v_timesteps = torch.randint(0, unwrapped_model.noise_scheduler.config.num_train_timesteps, (vB,), device=device).long()
                    v_noise = torch.randn_like(vtarget_norm)
                    v_noisy = unwrapped_model.noise_scheduler.add_noise(vtarget_norm, v_noise, v_timesteps)
                    v_pred_noise = model(v_noisy, v_condition, lead_indices, v_timesteps)
                    v_loss = (area_weights * (v_pred_noise - v_noise)**2).mean()
                    val_loss_sum += v_loss.item()
                    
                    # 2. Sampled RMSE (First 5 batches for performance)
                    if i < 5:
                        v_sample_norm = unwrapped_model.sample(v_condition, lead_indices, num_inference_steps=INFERENCE_STEPS_VAL)
                        v_sample_denorm = (v_sample_norm + 1.0) / 2.0 * (TARGET_SYMLOG_MAX - TARGET_SYMLOG_MIN) + TARGET_SYMLOG_MIN
                        v_pred_res_mm = torch.sign(v_sample_denorm) * torch.expm1(torch.abs(v_sample_denorm).clamp(max=6.55))
                        v_pred_mm = (vgeos_mean_lead + v_pred_res_mm).clamp(min=0.0)
                        v_pred_leads_mm.append(v_pred_mm)
                        
                val_count += 4

                if i < 5:
                    v_pred_full_mm = torch.cat(v_pred_leads_mm, dim=1) # (B, 4, H, W)
                    v_diff_sq = (v_pred_full_mm - vy_target.to(device))**2
                    v_weighted_mse = (v_diff_sq * area_weights).mean()
                    v_rmse_sum += torch.sqrt(v_weighted_mse).item()
                    v_rmse_count += 1
                    
                    if i == 0 and accelerator.is_main_process:
                        print(f"DEBUG | Val Batch 0 Stats | Target: {vy_target.min().item():.2f}/{vy_target.max().item():.2f} "
                              f"| Pred: {v_pred_full_mm.min().item():.2f}/{v_pred_full_mm.max().item():.2f} ")

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

        if train_dataset.geos_min is not None:
            fb_geos_flat = 2.0 * (fb_geos_flat - train_dataset.geos_min.to(device)) / (train_dataset.geos_max.to(device) - train_dataset.geos_min.to(device)) - 1.0
        # (Prev-GPCP channels already normalized in dataset)

        fb_sin = torch.sin(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_cos = torch.cos(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_mjo_map = fb_mjo.view(fb_B, 2, 1, 1).expand(fb_B, 2, H, W).to(device)

        fb_cond = torch.cat([fb_obs, fb_geos_flat, fb_sin, fb_cos, fb_mjo_map], dim=1)

        # Generate ensemble
        with torch.no_grad():
            fb_geos_raw = torch.expm1(fb_geos.clamp(max=6.55))
            fb_geos_raw_mean = fb_geos_raw.mean(dim=1).squeeze(1)

            ensemble_samples = []
            for i_ens in range(N_SAMPLES_VAL):
                fb_pred_leads_mm = []
                for lead in range(4):
                    lead_indices = torch.full((fb_B,), lead, dtype=torch.long, device=device)
                    fb_geos_mean_lead = fb_geos_raw_mean[:, lead:lead+1, :, :]

                    sample_norm = unwrapped_model.sample(fb_cond, lead_indices, num_inference_steps=INFERENCE_STEPS_VAL)
                    sample_denorm = (sample_norm + 1.0) / 2.0 * (TARGET_SYMLOG_MAX - TARGET_SYMLOG_MIN) + TARGET_SYMLOG_MIN
                    sample_res_mm = torch.sign(sample_denorm) * torch.expm1(torch.abs(sample_denorm).clamp(max=6.55))
                    sample_precip = (fb_geos_mean_lead + sample_res_mm).clamp(min=0.0)
                    fb_pred_leads_mm.append(sample_precip)
                ens_member_full = torch.cat(fb_pred_leads_mm, dim=1)
                ensemble_samples.append(ens_member_full)

            ens_stack = torch.stack(ensemble_samples, dim=0)
            ens_mean = ens_stack.mean(dim=0)  # (B, 4, H, W)

        fb_target_mm = fb_target
        fb_rmse = torch.sqrt(torch.mean((ens_mean - fb_target_mm)**2)).item()

        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Noise: {avg_val_loss:.4f} | Val FB RMSE: {fb_rmse:.4f} | Val Sampled RMSE: {val_rmse:.4f}")

            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, avg_val_loss, fb_rmse])

            # PLOT (best based on Fixed Batch RMSE)
            if fb_rmse < best_val_rmse or epoch % 5 == 0:
                if fb_rmse < best_val_rmse:
                    print(f"New Best! RMSE improved from {best_val_rmse:.4f} to {fb_rmse:.4f}. Plotting...")
                    best_val_rmse = fb_rmse

                t_img_all = fb_target_mm[0].cpu().numpy()         
                ens_mean_img = ens_mean[0].cpu().numpy()          
                sample_img = ens_stack[0, 0].cpu().numpy()  

                g_flat = fb_geos_flat[0]  
                if train_dataset.geos_min is not None:
                    g_flat = (g_flat + 1.0) / 2.0 * (train_dataset.geos_max.to(device) - train_dataset.geos_min.to(device)) + train_dataset.geos_min.to(device)
                g_ens = g_flat.view(4, 4, H, W)  
                g_mean_log = g_ens.mean(dim=0)    
                g_img_all = np.expm1(np.maximum(g_mean_log.cpu().numpy(), 0.0))

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

                os.makedirs(os.path.join(output_dir, "plots_diffusion"), exist_ok=True)
                plt.suptitle(f"Diffusion v2 - Epoch {epoch} | RMSE: {fb_rmse:.2f} | Noise Loss: {avg_val_loss:.4f}", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"plots_diffusion/epoch_{epoch}_rmse_{fb_rmse:.2f}.png"), dpi=150)
                plt.close()

            # SAVE CHECKPOINTS
            ckpt_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'top_k_ckpts': top_k_ckpts
            }
            torch.save(ckpt_state, latest_ckpt)

            current_path = os.path.join(output_dir, f"diffusion_epoch_{epoch}_rmse_{fb_rmse:.4f}.pt")
            top_k_ckpts.append((fb_rmse, epoch, current_path))
            top_k_ckpts.sort(key=lambda x: x[0])

            if len(top_k_ckpts) > save_top_k:
                worst = top_k_ckpts.pop()
                if worst[2] != current_path and os.path.exists(worst[2]):
                    os.remove(worst[2])

            is_in_top = any(x[2] == current_path for x in top_k_ckpts)
            if is_in_top:
                print(f"New Top Model! RMSE: {fb_rmse:.4f}")
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

    TARGET_LOG_MIN = 0.0
    TARGET_LOG_MAX = 6.55

    model = ConditionalDiffusionV2(
        in_channels=1,
        condition_channels=48,
        out_channels=1,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000
    ).to(device)

    output_dir = config["output_dir"] + "_v2"
    latest_ckpt = os.path.join(output_dir, "latest_diffusion_ckpt.pt")
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
                model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found!")
        return

    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()

    import cartopy.crs as ccrs

    test_indices = [0, 10, 20, 30, 40]
    output_dir_test = os.path.join(output_dir, "plots_test_suite")
    os.makedirs(output_dir_test, exist_ok=True)

    N_MEMBERS_TEST = 5
    INFERENCE_STEPS_TEST = 1000

    print(f"Test Suite: indices {test_indices}, {N_MEMBERS_TEST} members, {INFERENCE_STEPS_TEST} steps")

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

            if val_dataset.geos_min is not None:
                x_geos_flat = 2.0 * (x_geos_flat - val_dataset.geos_min.to(device)) / (val_dataset.geos_max.to(device) - val_dataset.geos_min.to(device)) - 1.0

            # (Prev-GPCP channels already normalized in dataset)

            t_sin = torch.sin(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            t_cos = torch.cos(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            t_mjo_map = t_mjo.view(B, 2, 1, 1).expand(B, 2, H, W).to(device)

            condition = torch.cat([x_obs, x_geos_flat, t_sin, t_cos, t_mjo_map], dim=1)

            unwrapped = accelerator.unwrap_model(model)

            t_geos_raw = torch.expm1(x_geos.clamp(max=6.55))
            t_geos_raw_mean = t_geos_raw.mean(dim=1).squeeze(1)

            ensemble = []
            for i_ens in range(N_MEMBERS_TEST):
                print(f"  Member {i_ens+1}/{N_MEMBERS_TEST}")
                fb_pred_leads_mm = []
                for lead in range(4):
                    lead_indices = torch.full((B,), lead, dtype=torch.long, device=device)
                    t_geos_mean_lead = t_geos_raw_mean[:, lead:lead+1, :, :]
                    
                    gen_norm = unwrapped.sample(condition, lead_indices, num_inference_steps=INFERENCE_STEPS_TEST, verbose=False)
                    gen_denorm = (gen_norm + 1.0) / 2.0 * (TARGET_SYMLOG_MAX - TARGET_SYMLOG_MIN) + TARGET_SYMLOG_MIN
                    gen_res_mm = torch.sign(gen_denorm) * torch.expm1(torch.abs(gen_denorm).clamp(max=6.55))
                    gen_mm = (t_geos_mean_lead + gen_res_mm).clamp(min=0.0)
                    fb_pred_leads_mm.append(gen_mm)
                ens_member_full = torch.cat(fb_pred_leads_mm, dim=1)
                ensemble.append(ens_member_full)

            ens_all = torch.stack(ensemble, dim=0)  # (N, B, 4, H, W)
            ens_mean = ens_all.mean(dim=0)           # (B, 4, H, W)

            target_np = y_target.clamp(min=0.0).squeeze(0).cpu().numpy()    # (4, H, W)
            ens_mean_np = ens_mean.squeeze(0).cpu().numpy()   # (4, H, W)

            # GEOS Mean
            geos_ens = x_geos_flat.view(4, 4, H, W)
            if val_dataset.geos_min is not None:
                geos_ens = (geos_ens + 1.0) / 2.0 * (val_dataset.geos_max.to(device) - val_dataset.geos_min.to(device)) + val_dataset.geos_min.to(device)
            geos_mean_log = geos_ens.mean(dim=0)
            geos_np = torch.expm1(geos_mean_log.clamp(min=0.0, max=6.55)).cpu().numpy()

            for lead in range(4):
                rmse_diff[lead].append(np.sqrt(np.mean((ens_mean_np[lead] - target_np[lead])**2)))
                rmse_geos[lead].append(np.sqrt(np.mean((geos_np[lead] - target_np[lead])**2)))

            lats = np.linspace(-90, 90, H)
            lons = np.linspace(0, 360, W)
            sample_np = ens_all[0, 0].cpu().numpy()  

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

                ax = fig.add_subplot(4, 6, lead * 6 + 1, projection=ccrs.PlateCarree())
                ax.imshow(t_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Target", fontsize=9)

                ax = fig.add_subplot(4, 6, lead * 6 + 2, projection=ccrs.PlateCarree())
                ax.imshow(g_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: GEOS", fontsize=9)

                ax = fig.add_subplot(4, 6, lead * 6 + 3, projection=ccrs.PlateCarree())
                ax.imshow(s_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Diff Sample", fontsize=9)

                ax = fig.add_subplot(4, 6, lead * 6 + 4, projection=ccrs.PlateCarree())
                ax.imshow(m_img, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=50)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Ens Mean", fontsize=9)

                ax = fig.add_subplot(4, 6, lead * 6 + 5, projection=ccrs.PlateCarree())
                ax.imshow(geos_bias, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-20, vmax=20)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: GEOS Bias\nRMSE: {g_rmse:.2f}", fontsize=9)

                ax = fig.add_subplot(4, 6, lead * 6 + 6, projection=ccrs.PlateCarree())
                ax.imshow(diff_bias, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-20, vmax=20)
                ax.coastlines()
                ax.set_title(f"W{lead+1}: Diff Bias\nRMSE: {d_rmse:.2f}", fontsize=9)

            plt.suptitle(f"Diffusion Test — Sample {current_idx}", fontsize=14)
            plt.savefig(os.path.join(output_dir_test, f"test_sample_{current_idx}.png"), bbox_inches='tight', dpi=150)
            plt.close()
            print(f"  Saved plot for sample {current_idx}.")

            samples_processed += 1
            current_idx += 1

    print(f"\n{'='*60}\nTest Suite Summary ({samples_processed} samples)\n{'='*60}")
    for lead in range(4):
        avg_d = np.mean(rmse_diff[lead])
        avg_g = np.mean(rmse_geos[lead])
        print(f"  Week {lead+1}: Diffusion RMSE={avg_d:.2f}, GEOS RMSE={avg_g:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_diffusion.yaml")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        test()
    else:
        train()
