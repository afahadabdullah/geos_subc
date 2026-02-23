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

@torch.no_grad()
def run_val_inference(epoch, model, val_loader, scheduler, device, accelerator, output_dir, log_file, 
                      residual_min, residual_max, geos_min, geos_max, area_weights, global_bounds, is_test=False, is_fast_recon=True):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # We'll evaluate on the first batch of the val_loader for consistent plotting/metrics
    batch = next(iter(val_loader))
    fb_target = batch['y_target'].to(device)
    vB, _, H, W = fb_target.shape
    
    fx_geos = batch['x_geos'].to(device) 
    fx_obs  = batch['x_obs'].to(device)
    fx_geos_flat = fx_geos.view(vB, -1, H, W)
    
    f_months = batch['month'].to(device)
    fsin_month = torch.sin(2 * np.pi * (f_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fcos_month = torch.cos(2 * np.pi * (f_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fx_cond = torch.cat([fx_obs, fx_geos_flat, fsin_month, fcos_month], dim=1) # [vB, 30, H, W]
    
    if is_fast_recon and not is_test:
        # Ensemble-based Fast Reconstruction at t=500
        num_ensemble = 3
        t_recon = 500
        t_batched = torch.full((vB,), t_recon, device=device, dtype=torch.long)
        
        ensemble_preds = []
        for _ in range(num_ensemble):
            noise = torch.randn_like(fb_target)
            x_t = scheduler.add_noise(fb_target, noise, t_batched)
            pred_noise = unwrapped_model(x_t, fx_cond, t_batched)
            pred_x0 = scheduler.reconstruct_x0(pred_noise, t_batched, x_t)
            ensemble_preds.append(pred_x0)
            
        # Average the ensemble members in the latent space
        pred_target_norm = torch.stack(ensemble_preds).mean(dim=0)
        recon_type = f"FastEnsemble (n={num_ensemble}, t={500})"
    else:
        # Full Reverse Sampling (1000 steps)
        latents = torch.randn((vB, 4, H, W), device=device)
        for t in tqdm(reversed(range(0, scheduler.num_timesteps)), desc="Reverse Sampling", leave=False, disable=not accelerator.is_main_process):
            t_batched = torch.full((vB,), t, device=device, dtype=torch.long)
            pred_noise = unwrapped_model(latents, fx_cond, t_batched)
            latents = scheduler.step(pred_noise, t, latents)
            
        pred_target_norm = latents
        recon_type = "FullSampling"
    
    # Denormalization
    denorm_residual_raw = ((pred_target_norm + 1.0) / 2.0) * (residual_max - residual_min) + residual_min
    fx_geos_norm = fx_geos.squeeze(1).squeeze(1)
    fx_geos_raw = ((fx_geos_norm + 1.0) / 2.0) * (geos_max - geos_min) + geos_min
    
    pred_gpcp_raw = fx_geos_raw + denorm_residual_raw
    full_pred = torch.clamp(pred_gpcp_raw, min=0.0)
    
    demap_target_residual_raw = ((fb_target + 1.0) / 2.0) * (residual_max - residual_min) + residual_min
    true_target_raw = fx_geos_raw + demap_target_residual_raw
    true_target_precip = torch.clamp(true_target_raw, min=0.0)
    
    # Metrics
    v_diff_sq = (full_pred - true_target_precip)**2
    v_weighted_mse = (v_diff_sq * area_weights).mean()
    fb_rmse = torch.sqrt(v_weighted_mse).item()
    
    if accelerator.is_main_process:
        print(f"Epoch {epoch} | Val FB Physical RMSE [{recon_type}]: {fb_rmse:.4f}")
        
    return fb_rmse, full_pred, true_target_precip

def train(args, accelerator):
    device = accelerator.device

    # Load config
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    epochs = config.get("epochs", 500)
    batch_size = config.get("batch_size", 4)
    lr = float(config.get("learning_rate", 1e-4))
    
    # ---------------------------------------------------------
    # 1. Dataset Initialization & Global Stats Calculation
    # ---------------------------------------------------------
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )

    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=config.get("num_workers", 4), pin_memory=True
    )

    train_dataset = None
    loader = None
    if not args.test:
        train_dataset = S2SHybridDataset(
            data_root=config["data_dir"],
            start_year=config["train_start_year"],
            end_year=config["train_end_year"],
            normalize=True,
            preload=config.get("preload", False)
        )
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=config.get("num_workers", 4), pin_memory=True
        )

    # Calculate Global Min-Max for Target GPCP Precipitation
    stats_file = "ml_model/v4_global_stats.pt"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"CRITICAL: {stats_file} missing. Please run calculate_global_stats_v4.py first!")
    
    global_bounds = torch.load(stats_file, weights_only=True)
    residual_min = global_bounds["residual_raw"]["min"]
    residual_max = global_bounds["residual_raw"]["max"]
    
    geos_min = global_bounds["geos_raw"]["min"]
    geos_max = global_bounds["geos_raw"]["max"]
    
    if accelerator.is_main_process:
        print("\n=======================================================")
        print(f"✅ Loaded Strict Global Stats: {stats_file}")
        print(f"   [Residual Raw Bounds]: Min = {residual_min:.4f}, Max = {residual_max:.4f}")
        print(f"   [GEOS Raw Bounds]    : Min = {geos_min:.4f}, Max = {geos_max:.4f}")
        print("=======================================================\n")

    # ---------------------------------------------------------
    # 2. Model & Scheduler Setup
    # ---------------------------------------------------------
    model = DiffusionModelV4(in_channels=34, out_channels=4).to(device)
    scheduler = CustomDiffusionScheduler(num_timesteps=1000, device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if not args.test:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(loader), eta_min=1e-6
        )
        model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, loader, val_loader, lr_scheduler
        )
    else:
        # Test mode: only prepare model and val_loader
        model, val_loader = accelerator.prepare(model, val_loader)
        optimizer = None
        lr_scheduler = None

    if accelerator.is_main_process:
        print(f"\n--- ACCELERATOR DIAGNOSTICS ---")
        print(f"   Accelerator Device: {device}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Current Device: {torch.cuda.current_device()}")
            print(f"   CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   Mixed Precision: {accelerator.mixed_precision}")
        
        # Check model device
        model_device = next(model.parameters()).device
        print(f"   Model Parameter Device: {model_device}")
        print(f"---------------------------------\n")

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
    
    # Load latest checkpoint if it exists
    if args.test:
        ckpt_path = os.path.join(output_dir, args.ckpt)
    else:
        ckpt_path = os.path.join(output_dir, "latest_diffusion_ckpt_v4.pt")

    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            # Unwrap for loading
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(checkpoint['model'])
            
            if not args.test:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                if 'best_val_rmse' in checkpoint:
                    best_val_rmse = checkpoint['best_val_rmse']
                
            if accelerator.is_main_process:
                print(f"\n🔄 Loaded checkpoint: {ckpt_path}")
                if not args.test:
                    print(f"   Starting at Epoch: {start_epoch}")
                    print(f"   Best Val RMSE so far: {best_val_rmse:.4f}\n")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ Failed to load checkpoint {ckpt_path}: {e}")
    else:
        if args.test:
            raise FileNotFoundError(f"CRITICAL: Checkpoint {ckpt_path} not found for testing!")
        if accelerator.is_main_process:
            print(f"\n🚀 Starting fresh training from Epoch 0\n")
        
    # ---------------------------------------------------------
    # 3. Execution Mode: Train or Test
    # ---------------------------------------------------------
    if args.test:
        if accelerator.is_main_process:
            print(f"\n🧪 RUNNING TEST MODE: Evaluating {ckpt_path}\n")
        
        # Test mode runs a single full-range validation inference
        run_val_inference(start_epoch, model, val_loader, scheduler, device, accelerator, output_dir, log_file, 
                         residual_min, residual_max, geos_min, geos_max, area_weights, global_bounds, 
                         is_test=True, is_fast_recon=False)
        return

    # ---------------------------------------------------------
    # Pre-Training Diagnostics (Raw vs Normalized Bounds)
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        print("\n--- PRE-TRAINING DIAGNOSTICS: RAW vs NORMALIZED ---")
        
        # Disable normalization briefly to fetch pure values
        train_dataset.normalize = False
        raw_sample = train_dataset[0]
        
        # Re-enable normalization to fetch mapped values
        train_dataset.normalize = True
        norm_sample = train_dataset[0]
        
        def print_bounds(name, raw_t, norm_t):
            print(f"{name:<12} | RAW: [{raw_t.min():>8.4f}, {raw_t.max():>8.4f}] --> NORM: [{norm_t.min():>8.4f}, {norm_t.max():>8.4f}]")
            
        print_bounds("OBS Arrays", raw_sample['x_obs'], norm_sample['x_obs'])
        print_bounds("GEOS Raw", raw_sample['x_geos'], norm_sample['x_geos'])
        print_bounds("Target Raw", raw_sample['y_target'], norm_sample['y_target'])
        print("---------------------------------------------------\n")
    
    # ---------------------------------------------------------
    # Pre-Flight NaN Integrity Scan
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        print("\n--- INITIATING PRE-FLIGHT NaN SCAN (Checking entire dataset) ---")
        nan_found = False
        for batch_idx, batch in enumerate(tqdm(loader, desc="Scanning for NaNs/Infs")):
            if torch.isnan(batch['x_geos']).any() or torch.isinf(batch['x_geos']).any():
                print(f"CRITICAL: NaN/Inf detected in GEOS array at batch {batch_idx}")
                nan_found = True
            if torch.isnan(batch['x_obs']).any() or torch.isinf(batch['x_obs']).any():
                print(f"CRITICAL: NaN/Inf detected in OBS array at batch {batch_idx}")
                nan_found = True
            if torch.isnan(batch['y_target']).any() or torch.isinf(batch['y_target']).any():
                print(f"CRITICAL: NaN/Inf detected in TARGET array at batch {batch_idx}")
                nan_found = True
                
            if nan_found:
                raise ValueError(f"Pre-flight scan failed! NaNs detected in training data. Check dataset limits.")
                
        print("✅ Pre-flight scan complete. Zero NaNs detected across all batches.")
        print("----------------------------------------------------------------\n")
        
    # Wait for all processes to finish checking before starting training
    accelerator.wait_for_everyone()

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

            # Conditionals are already normalized [-1, 1] by dataset_hybrid
            x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month], dim=1) # [B, 30, H, W]

            # Targets are already log-residual normalized [-1, 1] by dataset_hybrid
            target_norm = batch['y_target'].to(device) # [B, 4, H, W]

            if epoch == start_epoch and i == 0 and accelerator.is_main_process:
                print(f"\n--- DEBUG | Train Batch 0 Diagnostics ---")
                print(f"x_obs bounds         : {x_obs.min().item():.2f} to {x_obs.max().item():.2f}")
                print(f"x_geos bounds        : {x_geos.min().item():.2f} to {x_geos.max().item():.2f}")
                print(f"Final x_cond shape   : {x_cond.shape}")
                print(f"Final x_cond bounds  : {x_cond.min().item():.2f} to {x_cond.max().item():.2f}")
                print(f"Pure DataLoader Target Bounds: {target_norm.min().item():.2f} to {target_norm.max().item():.2f}")
                print(f"-----------------------------------------\n")

                # --- Create Before/After Normalization Diagnostic Plot ---
                # Take index 0, lead 1 for visualization
                sample_idx = 0
                lead_idx = 0
                
                # 1. Reverse Normalize GEOS
                geos_norm_sample = x_geos[sample_idx, 0, 0, lead_idx].cpu().numpy()
                geos_raw_sample = ((geos_norm_sample + 1.0) / 2.0) * (geos_max - geos_min) + geos_min
                
                # 2. Reverse Normalize SST (Channel 0 of x_obs)
                sst_norm_sample = x_obs[sample_idx, 0].cpu().numpy()
                sst_raw_sample = ((sst_norm_sample + 1.0) / 2.0) * (global_bounds["sst"]["max"] - global_bounds["sst"]["min"]) + global_bounds["sst"]["min"]
                
                # 3. Reverse Normalize Target Residual (Lead 0)
                res_norm_sample = target_norm[sample_idx, lead_idx].cpu().numpy()
                res_raw_sample = ((res_norm_sample + 1.0) / 2.0) * (residual_max - residual_min) + residual_min

                # 4. Reverse Normalize Observational States
                # Channel 0: SST
                sst_norm_sample = x_obs[sample_idx, 0].cpu().numpy()
                sst_raw_sample = ((sst_norm_sample + 1.0) / 2.0) * (global_bounds["sst"]["max"] - global_bounds["sst"]["min"]) + global_bounds["sst"]["min"]
                
                # Channel 4: SSS
                sss_norm_sample = x_obs[sample_idx, 4].cpu().numpy()
                sss_raw_sample = ((sss_norm_sample + 1.0) / 2.0) * (global_bounds["sss"]["max"] - global_bounds["sss"]["min"]) + global_bounds["sss"]["min"]
                
                # Channel 8: Soil Moisture
                sm_norm_sample = x_obs[sample_idx, 8].cpu().numpy()
                sm_raw_sample = ((sm_norm_sample + 1.0) / 2.0) * (global_bounds["sm"]["max"] - global_bounds["sm"]["min"]) + global_bounds["sm"]["min"]
                
                # Channel 12: IVT
                ivt_norm_sample = x_obs[sample_idx, 12].cpu().numpy()
                ivt_raw_sample = ((ivt_norm_sample + 1.0) / 2.0) * (global_bounds["ivt"]["max"] - global_bounds["ivt"]["min"]) + global_bounds["ivt"]["min"]
                
                # Channel 16: Z500
                z500_norm_sample = x_obs[sample_idx, 16].cpu().numpy()
                z500_raw_sample = ((z500_norm_sample + 1.0) / 2.0) * (global_bounds["z500"]["max"] - global_bounds["z500"]["min"]) + global_bounds["z500"]["min"]
                
                # Channel 20: U250
                u250_norm_sample = x_obs[sample_idx, 20].cpu().numpy()
                u250_raw_sample = ((u250_norm_sample + 1.0) / 2.0) * (global_bounds["u250"]["max"] - global_bounds["u250"]["min"]) + global_bounds["u250"]["min"]

                fig, axes = plt.subplots(8, 2, figsize=(14, 32))
                # Row 1: GEOS
                im1 = axes[0, 0].imshow(geos_raw_sample, cmap='Blues')
                axes[0, 0].set_title(f"Raw GEOS (Lead {lead_idx+1})")
                fig.colorbar(im1, ax=axes[0, 0])
                
                im2 = axes[0, 1].imshow(geos_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[0, 1].set_title("Normalized GEOS [-1, 1]")
                fig.colorbar(im2, ax=axes[0, 1])
                
                # Row 2: Target Residual
                im3 = axes[1, 0].imshow(res_raw_sample, cmap='RdBu_r')
                axes[1, 0].set_title("Raw Residual (GPCP - GEOS)")
                fig.colorbar(im3, ax=axes[1, 0])
                
                im4 = axes[1, 1].imshow(res_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[1, 1].set_title("Normalized Residual [-1, 1]")
                fig.colorbar(im4, ax=axes[1, 1])
                
                # Row 3: SST
                im5 = axes[2, 0].imshow(sst_raw_sample, cmap='viridis')
                axes[2, 0].set_title("Raw SST")
                fig.colorbar(im5, ax=axes[2, 0])
                
                im6 = axes[2, 1].imshow(sst_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[2, 1].set_title("Normalized SST [-1, 1]")
                fig.colorbar(im6, ax=axes[2, 1])

                # Row 4: SSS
                im7 = axes[3, 0].imshow(sss_raw_sample, cmap='YlGnBu')
                axes[3, 0].set_title("Raw SSS")
                fig.colorbar(im7, ax=axes[3, 0])
                
                im8 = axes[3, 1].imshow(sss_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[3, 1].set_title("Normalized SSS [-1, 1]")
                fig.colorbar(im8, ax=axes[3, 1])

                # Row 5: Soil Moisture
                im9 = axes[4, 0].imshow(sm_raw_sample, cmap='YlOrBr')
                axes[4, 0].set_title("Raw SM")
                fig.colorbar(im9, ax=axes[4, 0])
                
                im10 = axes[4, 1].imshow(sm_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[4, 1].set_title("Normalized SM [-1, 1]")
                fig.colorbar(im10, ax=axes[4, 1])

                # Row 6: IVT
                im11 = axes[5, 0].imshow(ivt_raw_sample, cmap='cubehelix')
                axes[5, 0].set_title("Raw IVT")
                fig.colorbar(im11, ax=axes[5, 0])
                
                im12 = axes[5, 1].imshow(ivt_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[5, 1].set_title("Normalized IVT [-1, 1]")
                fig.colorbar(im12, ax=axes[5, 1])

                # Row 7: Z500
                im13 = axes[6, 0].imshow(z500_raw_sample, cmap='magma')
                axes[6, 0].set_title("Raw Z500")
                fig.colorbar(im13, ax=axes[6, 0])
                
                im14 = axes[6, 1].imshow(z500_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[6, 1].set_title("Normalized Z500 [-1, 1]")
                fig.colorbar(im14, ax=axes[6, 1])

                # Row 8: U250
                im15 = axes[7, 0].imshow(u250_raw_sample, cmap='coolwarm')
                axes[7, 0].set_title("Raw U250")
                fig.colorbar(im15, ax=axes[7, 0])
                
                im16 = axes[7, 1].imshow(u250_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[7, 1].set_title("Normalized U250 [-1, 1]")
                fig.colorbar(im16, ax=axes[7, 1])

                plt.tight_layout()
                diag_path = os.path.join(output_dir, "normalization_check.png")
                plt.savefig(diag_path)
                plt.close()
                print(f"✅ Normalization diagnostic plot saved to {diag_path}!")

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
        # Unconditional Epoch-End Resume Checkpoint
        # ---------------------------------------------------------
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_rmse': best_val_rmse
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_diffusion_ckpt_v4.pt"))

        # ---------------------------------------------------------
        # 4. Validation & Checkpointing (Fast Reconstruction RMSE)
        # ---------------------------------------------------------
        # Use fast reconstruction for the RMSE check to keep epochs moving quickly
        is_plot_epoch = (epoch % config.get("plot_epochs", 20) == 0) or args.full_val
        
        fb_rmse, full_pred, true_target_precip = run_val_inference(
            epoch, model, val_loader, scheduler, device, accelerator, output_dir, log_file, 
            residual_min, residual_max, geos_min, geos_max, area_weights, global_bounds,
            is_test=is_plot_epoch, 
            is_fast_recon=not is_plot_epoch
        )
        
        if is_plot_epoch and accelerator.is_main_process:
            print(f"📊 Full Sampling Validation complete for Epoch {epoch}. Plotting results...")
        
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_rmse': best_val_rmse
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_diffusion_ckpt_v4.pt"))

            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, 0.0, fb_rmse])
                    
            if fb_rmse < best_val_rmse:
                best_val_rmse = fb_rmse
                print(f"🌟 New best Validation RMSE: {best_val_rmse:.4f}! Saving best model...")
                ckpt['best_val_rmse'] = best_val_rmse
                torch.save(ckpt, os.path.join(output_dir, "best_diffusion_ckpt_v4.pt"))
                
                # Plot
                t_img = true_target_precip[0].cpu().numpy()
                p_img = full_pred[0].cpu().numpy()
                fig, axes = plt.subplots(4, 3, figsize=(15, 16))
                for l in range(4):
                    diff = p_img[l] - t_img[l]
                    t_min, t_max = t_img[l].min(), t_img[l].max()
                    
                    im0 = axes[l, 0].imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
                    fig.colorbar(im0, ax=axes[l, 0], fraction=0.046, pad=0.04)
                    if l == 0: axes[l, 0].set_title("Target GPCP")
                    
                    im1 = axes[l, 1].imshow(p_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
                    fig.colorbar(im1, ax=axes[l, 1], fraction=0.046, pad=0.04)
                    if l == 0: axes[l, 1].set_title("Predicted GPCP")
                    
                    im2 = axes[l, 2].imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50)
                    fig.colorbar(im2, ax=axes[l, 2], fraction=0.046, pad=0.04)
                    if l == 0: axes[l, 2].set_title("Diff Bias [-50, 50]")
                
                os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"plots/epoch_{epoch}_rmse_{fb_rmse:.2f}.png"))
                plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train or Test Diffusion Model V4")
    parser.add_argument("--config", type=str, default="ml_model/config_diffusion_v4.yaml")
    parser.add_argument("--test", action="store_true", help="Run in inference/test mode only")
    parser.add_argument("--ckpt", type=str, default="best_diffusion_ckpt_v4.pt", 
                        help="Checkpoint filename in output_dir to load for testing (default: best_diffusion_ckpt_v4.pt)")
    parser.add_argument("--full-val", action="store_true", help="Force full reverse sampling validation (1000 steps) for all validation epochs.")
    args = parser.parse_args()

    accelerator = Accelerator(split_batches=True)
    train(args, accelerator)

if __name__ == "__main__":
    main()
