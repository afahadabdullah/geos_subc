import os
import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import csv
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import argparse
from accelerate import Accelerator

# Local Modules
from dataset_hybrid import S2SHybridDataset
from diffusion_v4 import DiffusionModelV4, CustomDiffusionScheduler

def get_area_weights(lats, device):
    lats_rad = np.deg2rad(lats)
    weights = np.cos(lats_rad)
    weights = weights / np.mean(weights)
    weights_tensor = torch.from_numpy(weights).float().to(device)
    weights_tensor = weights_tensor.view(1, 1, len(lats), 1)
    return weights_tensor

def train(force_new_stats=False):
    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    # Load config
    config_path = "ml_model/config_diffusion_v4.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    epochs = config.get("epochs", 500)
    batch_size = config.get("batch_size", 4)
    lr = float(config.get("learning_rate", 1e-4))
    
    # ---------------------------------------------------------
    # 1. Dataset Initialization & Global Stats Calculation
    # ---------------------------------------------------------
    train_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["train_start_year"],
        end_year=config["train_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )
    
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=config.get("num_workers", 4), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=config.get("num_workers", 4), pin_memory=True
    )

    # Calculate Global Min-Max for Target GPCP Precipitation
    stats_file = "v4_global_stats.pt"
    if not force_new_stats and os.path.exists(stats_file):
        if accelerator.is_main_process:
            print(f"Loading cached Global Bounds from {stats_file}")
        stats = torch.load(stats_file)
        global_min = stats['global_min'].to(device)
        global_max = stats['global_max'].to(device)
    else:
        if accelerator.is_main_process:
            print("Computing Global GPCP Targets Bounds (Min/Max)...")
            g_min, g_max = float('inf'), float('-inf')
            
            # Subsample random batches for speed, parsing the max/min of targets
            import sys
            sys.set_int_max_str_digits(100000)
            
            for i, batch in enumerate(tqdm(loader, desc="Scanning Bounds")):
                y = batch['y_target']
                x_geos = batch['x_geos'].to(y.device) # [B, 4, M, H, W] expected, though m=1 is typically 4.
                
                # S2SHybridDataset transforms GEOS to log1p mapping inherently.
                # We calculate the residual strictly in the log-space to compress massive outliers.
                geos_mean_log = x_geos.mean(dim=1).squeeze(1) # [B, L, H, W]
                y_log = torch.log1p(y.clamp(min=0.0))
                
                residual_log = y_log - geos_mean_log
                
                b_min = residual_log.min().item()
                b_max = residual_log.max().item()
                if b_min < g_min: g_min = b_min
                if b_max > g_max: g_max = b_max

            print(f"Calculated Bounds -> Min: {g_min:.4f}, Max: {g_max:.4f}")
            torch.save({'global_min': torch.tensor(g_min), 'global_max': torch.tensor(g_max)}, stats_file)
            global_min = torch.tensor(g_min).to(device)
            global_max = torch.tensor(g_max).to(device)
            
    # Broadcast bounds to all processes if using Multi-GPU
    # Safe fallback if not multi-GPU
    if "global_min" not in locals():
        stats = torch.load(stats_file)
        global_min = stats['global_min'].to(device)
        global_max = stats['global_max'].to(device)

    # ---------------------------------------------------------
    # 2. Model & Scheduler Setup
    # ---------------------------------------------------------
    model = DiffusionModelV4(in_channels=34, out_channels=4).to(device)
    scheduler = CustomDiffusionScheduler(num_timesteps=1000, device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader), eta_min=1e-6
    )

    model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, lr_scheduler
    )

    # Area weights
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # Output directory
    output_dir = config.get("output_dir", "ml_output_diffusion_v4")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log_v4.csv")
    
    if accelerator.is_main_process and not os.path.exists(log_file):
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Noise", "Val_RMSE"])

    # Fixed Val Batch for continuous plotting
    fixed_val_batch = next(iter(val_loader))

    start_epoch = 0
    best_val_rmse = float('inf')
    
    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for i, batch in enumerate(pbar):    
            # Conditionals: [B, 48, H, W]
            x_geos = batch['x_geos'].to(device) 
            x_obs  = batch['x_obs'].to(device)
            
            B, M, C, L, H, W = x_geos.shape
            x_geos_flat = x_geos.view(B, -1, H, W)
            
            months = batch['month'].to(device)
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)

            x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month], dim=1) # [B, 30, H, W]

            # Targets: [B, 4, H, W]
            y_target = batch['y_target'].to(device)
            
            # Extract GEOS mean in log space directly from dataset_hybrid mapped variable
            geos_mean_log = x_geos_flat
            y_target_log = torch.log1p(y_target.clamp(min=0.0))

            # Target all 4 lead weeks simultaneously
            y_residual_log = y_target_log - geos_mean_log

            # Linear Min-Max Normalization [-1, 1] applied to Log Residuals
            target_norm = 2.0 * ((y_residual_log - global_min) / (global_max - global_min)) - 1.0

            if i == 0 and accelerator.is_main_process:
                print(f"DEBUG | Train Batch 0 | Log Residual Target Bounds: {y_residual_log.min().item():.2f} to {y_residual_log.max().item():.2f}")
                print(f"DEBUG | Train Batch 0 | Normalized Target Bounds: {target_norm.min().item():.2f} to {target_norm.max().item():.2f}")

            # Forward Diffusion (Noise injection)
            timesteps = torch.randint(0, scheduler.num_timesteps, (B,), device=device).long()
            noise = torch.randn_like(target_norm)
            noisy_target = scheduler.add_noise(target_norm, noise, timesteps)

            # Predict the noise
            noise_pred = model(noisy_target, x_cond, timesteps)

            # Loss scaling with spatial priority
            loss = (area_weights * (noise_pred - noise)**2).mean()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(loader)

        # ---------------------------------------------------------
        # 4. Validation Loop (Inference begins after epoch 20 to save compute)
        # ---------------------------------------------------------
        if epoch >= 20:
            model.eval()
            val_loss_sum = 0
            val_count = 0
            rmse_sum = 0
            
            unwrapped_model = accelerator.unwrap_model(model)
            
            with torch.no_grad():
                # Process strictly the fixed evaluation batch to gauge qualitative reconstruction
                fb_target = fixed_val_batch['y_target'].to(device)
                vB = fb_target.shape[0]
                
                fx_geos = fixed_val_batch['x_geos'].to(device) 
                fx_obs  = fixed_val_batch['x_obs'].to(device)
                fx_geos_flat = fx_geos.view(vB, -1, H, W)
                
                f_months = fixed_val_batch['month'].to(device)
                fsin_month = torch.sin(2 * np.pi * (f_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
                fcos_month = torch.cos(2 * np.pi * (f_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
                
                fx_cond = torch.cat([fx_obs, fx_geos_flat, fsin_month, fcos_month], dim=1)
                
                # Extract fixed batch GEOS mean in log space
                fx_geos_mean_log = fx_geos_flat
                fb_target_log = torch.log1p(fb_target.clamp(min=0.0))
                
                # Diagnostic target log residual calculations for reference
                f_y_residual_log = fb_target_log - fx_geos_mean_log

                # Single Pass Reverse Sampling predicting all 4 Leads simultaneously
                latents = torch.randn((vB, 4, H, W), device=device)
                
                # Iterative reverse diffusion loop
                for t in tqdm(reversed(range(0, scheduler.num_timesteps)), desc="Evaluating generation", leave=False):
                    t_batched = torch.full((vB,), t, device=device, dtype=torch.long)
                    pred_noise = unwrapped_model(latents, fx_cond, t_batched)
                    latents = scheduler.step(pred_noise, t, latents)
                    
                # Reverse Normalization Linear Mapping back to Log Residual space
                denorm_residual_log = ((latents + 1.0) / 2.0) * (global_max - global_min) + global_min
                
                # Reconstruct GPCP by adding generated log residual back onto GEOS log mean
                final_precip_log = fx_geos_mean_log + denorm_residual_log
                final_precip = torch.expm1(final_precip_log)
                full_pred = final_precip.clamp(min=0.0) # Physical limit
                
                # Accuracy tracking
                v_diff_sq = (full_pred - fb_target)**2
                v_weighted_mse = (v_diff_sq * area_weights).mean()
                fb_rmse = torch.sqrt(v_weighted_mse).item()
                
            if accelerator.is_main_process:
                print(f"DEBUG | Val Batch 0 | Target Log Residual: {f_y_residual_log.min().item():.2f} to {f_y_residual_log.max().item():.2f}")
                
                # Check how big the absolute predicted residual is vs expected
                pred_residual_log = torch.log1p(full_pred) - fx_geos_mean_log
                print(f"DEBUG | Val Batch 0 | Generated Log Residual: {pred_residual_log.min().item():.2f} to {pred_residual_log.max().item():.2f}")
                print(f"DEBUG | Val Batch 0 | Final Reconstructed Precip: {full_pred.min().item():.2f} to {full_pred.max().item():.2f}")
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val FB RMSE: {fb_rmse:.4f}")

                with open(log_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, avg_train_loss, 0.0, fb_rmse])
                    
                # Plot
                if fb_rmse < best_val_rmse or epoch % 5 == 0:
                    if fb_rmse < best_val_rmse: best_val_rmse = fb_rmse
                    
                    t_img = fb_target[0].cpu().numpy()
                    p_img = full_pred[0].cpu().numpy()
                    
                    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
                    for l in range(4):
                        diff = p_img[l] - t_img[l]
                        
                        axes[l, 0].imshow(t_img[l], cmap='Blues', vmin=0, vmax=50)
                        axes[l, 0].set_ylabel(f"Week {l+1}")
                        if l == 0: axes[l, 0].set_title("Target GPCP")
                        
                        axes[l, 1].imshow(p_img[l], cmap='Blues', vmin=0, vmax=50)
                        if l == 0: axes[l, 1].set_title("Predicted GPCP")
                        
                        axes[l, 2].imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20)
                        if l == 0: axes[l, 2].set_title("Diff Bias")
                        
                    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"plots/epoch_{epoch}_rmse_{fb_rmse:.2f}.png"))
                    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", action="store_true", help="Force recalculation of global stats")
    args = parser.parse_args()
    
    train(force_new_stats=args.new)
