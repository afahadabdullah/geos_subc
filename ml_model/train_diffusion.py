import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os
import argparse
import yaml # pyyaml
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.diffusion import ConditionalDiffusion

def get_area_weights(lats, device):
    """
    Calculates area weights based on cosine of latitude.
    Normalizes weights to have a mean of 1.
    """
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    # Shape: (1, 1, H, 1) for broadcasting with (B*L, 1, H, W)
    weights_tensor = torch.from_numpy(weights).float().to(device)
    return weights_tensor.view(1, 1, -1, 1)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Accelerator
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Dataset
    preload = config.get("preload", False)
    train_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["train_start_year"],
        end_year=config["train_end_year"],
        normalize=True,
        preload=preload
    )
    
    loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Validation Dataset
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=preload
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    # Validation Fixed Batch for consistent plotting
    fixed_val_batch = next(iter(val_loader))
    
    # Model: Conditional Diffusion
    # In: 1 (Target Noisy)
    # Cond: 4 (GEOS) + 4 (Obs) = 8
    model = ConditionalDiffusion(
        in_channels=1,
        condition_channels=8,
        out_channels=1,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    
    # Scheduler
    # Standard Cosine or Linear warmup
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        steps_per_epoch=len(loader), 
        epochs=config["epochs"]
    )

    # Prepare
    # Note: accelerator.prepare handles moving data to device
    model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, lr_scheduler
    )

    # Area Weights for Loss
    # Latitude range: -90 to 90 (181 points)
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # Output Dir
    os.makedirs(config["output_dir"], exist_ok=True)
    log_file = os.path.join(config["output_dir"], "training_log_diffusion.csv")
    if accelerator.is_main_process:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_Loss", "Val_RMSE"])

    # Load Checkpoint?
    start_epoch = 0
    latest_ckpt = os.path.join(config["output_dir"], "latest_diffusion_ckpt.pt")
    
    # Top K Checkpoints
    top_k_ckpts = [] # List of (rmse, epoch, path)
    save_top_k = config.get("save_top_k", 4)
    
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Scheduler might need state dict? 
        start_epoch = checkpoint['epoch'] + 1
    best_val_rmse = float('inf')
    if top_k_ckpts:
        best_val_rmse = top_k_ckpts[0][0] # RMSE is the first element
        print(f"Resumed Best Val RMSE: {best_val_rmse:.4f}")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0.0
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Data Info:
            # x_obs: (B, 3, L, H, W) -> Obs Var logic? 
            # In dataset_hybrid, x_obs is (B, 3, L, H, W) [SST, SSS, Soil]
            # x_geos: (B, 4, 1, L, H, W) [Members]
            # y_target: (B, L, H, W)
            
            x_obs = batch['x_obs']
            x_geos = batch['x_geos']
            y_target = batch['y_target']
            
            B, C_obs, H, W = x_obs.shape
            L = 4 # Hardcoded for now (4 weeks)
            
            # Reshape: Treat Lead Time as independent samples
            # x_obs: (B, 16, H, W). 
            # Channels 0-3: SST (w1, w2, w3, w4), 4-7: SSS, 8-11: SM, 12-15: Prev
            # We want (B*4, 4, H, W) where first dim is sample (B x Lead)
            
            # 1. Reshape to (B, 4, 4, H, W) -> (B, Var, Lead, H, W)
            x_obs_reshaped = x_obs.view(B, 4, 4, H, W)
            
            # 2. Permute to (B, Lead, Var, H, W)
            x_obs_lead = x_obs_reshaped.permute(0, 2, 1, 3, 4)
            
            # 3. Flatten (B*Lead, Var, H, W)
            x_obs_flat = x_obs_lead.reshape(B * L, 4, H, W)
            
            # GEOS: (B, 4, 1, L, H, W) -> (B*L, 4, H, W)
            # x_geos has (Members, 1, Lead)
            # We want (B, Members, Lead, H, W)
            # x_geos: (B, 4, 1, 4, H, W)
            x_geos_lead = x_geos.squeeze(2).permute(0, 2, 1, 3, 4) # (B, 4, 4, H, W)
            x_geos_flat = x_geos_lead.reshape(B * L, 4, H, W)
            
            # Target: (B, L, H, W) -> (B*L, 1, H, W)
            y_target_flat = y_target.reshape(B * L, 1, H, W)
            
            # NORMALIZE TARGET
            # Diffusion learns noise on normalized data ideally N(0,1)
            # Use GEOS stats for consistency
            if train_dataset.geos_mean is not None:
                gm = train_dataset.geos_mean.to(device)
                gs = train_dataset.geos_std.to(device)
                # Careful: Check shape of gm/gs
                # S2SHybridDataset: gm can be (1, L, H, W) or (1,)
                # Let's handle generic case
                if gm.numel() > 1:
                    # Generic implementation: if (L, H, W) or similar
                    # Expand B times
                    if gm.ndim == 3: gm = gm.unsqueeze(0) # (1, L, H, W)
                    gm_full = gm.expand(B, -1, -1, -1).reshape(B * L, 1, H, W)
                    gs_full = gs.expand(B, -1, -1, -1).reshape(B * L, 1, H, W)
                    target_normalized = (y_target_flat - gm_full) / gs_full
                else:
                    target_normalized = (y_target_flat - gm) / gs
            else:
                 target_normalized = y_target_flat # Fallback
            
            # Condition: (B*L, 8, H, W) -> 4 GEOS + 4 Obs
            # NEW: Add Lead Time Map (9th channel)
            # Create Lead Map: (B, 4, 1, H, W)
            # Values: 0.25, 0.50, 0.75, 1.0 for leads 0, 1, 2, 3
            lead_map = torch.tensor([0.25, 0.50, 0.75, 1.0], device=device).view(1, 4, 1, 1, 1)
            lead_map = lead_map.expand(B, -1, 1, H, W) # (B, 4, 1, H, W)
            lead_map_flat = lead_map.reshape(B * L, 1, H, W)
            
            condition = torch.cat([x_obs_flat, x_geos_flat, lead_map_flat], dim=1)
            
            # Sample Timesteps
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps, 
                (x_obs_flat.shape[0],), device=device
            ).long()
            
            # Add Noise
            noise = torch.randn_like(y_target_flat)
            noisy_target = model.noise_scheduler.add_noise(target_normalized, noise, timesteps)
            
            # Predict Noise
            # Inputs: noisy_target, condition, timesteps
            noise_pred = model(noisy_target, condition, timesteps)
            
            # Area-Weighted MSE Loss
            loss = (area_weights * (noise_pred - noise)**2).mean()
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss_sum = 0
        val_count = 0
        
        # We will compute RMSE on normalized space for speed 
        # (or denormalize if we want true RMSE, but normalized is fine for model selection)
        # Let's compute Normalized MSE Loss as proxy for selection
        
        with torch.no_grad():
             for val_batch in val_loader:
                vx_obs = val_batch['x_obs']
                vx_geos = val_batch['x_geos']
                vy_target = val_batch['y_target']
                
                vB, _, vH, vW = vx_obs.shape
                # Preprocess same as train
                vx_obs_flat = vx_obs.view(vB, 4, 4, vH, vW).permute(0, 2, 1, 3, 4).reshape(vB * 4, 4, vH, vW)
                vx_geos_flat = vx_geos.squeeze(2).permute(0, 2, 1, 3, 4).reshape(vB * 4, 4, vH, vW)
                vy_target_flat = vy_target.reshape(vB * 4, 1, vH, vW)
                
                # Normalize Target
                if train_dataset.geos_mean is not None:
                    # Re-use gm/gs logic from above if valid
                    gm = train_dataset.geos_mean.to(device)
                    gs = train_dataset.geos_std.to(device)
                    if gm.numel() > 1:
                        if gm.ndim == 3: gm = gm.unsqueeze(0)
                        gm_v = gm.expand(vB, -1, -1, -1).reshape(vB * 4, 1, vH, vW)
                        gs_v = gs.expand(vB, -1, -1, -1).reshape(vB * 4, 1, vH, vW)
                        vtarget_norm = (vy_target_flat - gm_v) / gs_v
                    else:
                        vtarget_norm = (vy_target_flat - gm) / gs
                else:
                    vtarget_norm = vy_target_flat
                
                # Lead Map for Validation
                v_lead_map = torch.tensor([0.25, 0.50, 0.75, 1.0], device=device).view(1, 4, 1, 1, 1)
                v_lead_map = v_lead_map.expand(vB, -1, 1, vH, vW).reshape(vB * 4, 1, vH, vW)
                
                v_condition = torch.cat([vx_obs_flat, vx_geos_flat, v_lead_map], dim=1)
                
                # We can't easily compute full generation RMSE for every batch (too slow)
                # Instead, compute one-step denoising error or simple MSE loss on noise
                # OR generated sample for subset. 
                # Standard practice: Validation Loss (MSE on noise)
                # If user wants "Best Model" based on generation quality, we should sample a subset.
                # Let's stick to Noise MSE for speed and stability, 
                # AND compute generation RMSE on the FIXED BATCH.
                
                # Validation Loss (Noise MSE)
                v_timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (vx_obs_flat.shape[0],), device=device).long()
                v_noise = torch.randn_like(vy_target_flat)
                v_noisy = model.noise_scheduler.add_noise(vtarget_norm, v_noise, v_timesteps)
                v_pred = model(v_noisy, v_condition, v_timesteps)
                
                # Area-Weighted MSE Loss
                v_loss = (area_weights * (v_pred - v_noise)**2).mean()
                val_loss_sum += v_loss.item()
                val_count += 1

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
        
        # --- FIXED BATCH VISUALIZATION & RMSE ---
        # Compute "Real" RMSE on fixed batch for checking generation quality
        # This is what we will use for "Best Model" check to align with user request
        
        # Prepare Fixed Batch
        fb_obs = fixed_val_batch['x_obs'].to(device)
        fb_geos = fixed_val_batch['x_geos'].to(device)
        fb_target = fixed_val_batch['y_target'].to(device)
        
        # Process Fixed Batch
        # Just take first sample (Lead 0) for visualization
        # But compute RMSE on whole batch
        fb_B = fb_obs.shape[0]
        fb_obs_flat = fb_obs.view(fb_B, 4, 4, H, W).permute(0, 2, 1, 3, 4).reshape(fb_B * 4, 4, H, W)
        fb_geos_flat = fb_geos.squeeze(2).permute(0, 2, 1, 3, 4).reshape(fb_B * 4, 4, H, W)
        fb_target_flat = fb_target.reshape(fb_B * 4, 1, H, W)
        
        # Normalize FB
        if train_dataset.geos_mean is not None:
            gm = train_dataset.geos_mean.to(device)
            gs = train_dataset.geos_std.to(device)
            if gm.numel() > 1:
                if gm.ndim == 3: gm = gm.unsqueeze(0)
                gm_fb = gm.expand(fb_B, -1, -1, -1).reshape(fb_B * 4, 1, H, W)
                gs_fb = gs.expand(fb_B, -1, -1, -1).reshape(fb_B * 4, 1, H, W)
                fb_target_norm = (fb_target_flat - gm_fb) / gs_fb
            else:
                fb_target_norm = (fb_target_flat - gm) / gs
        else:
            fb_target_norm = fb_target_flat
            
        fb_lead_map = torch.tensor([0.25, 0.50, 0.75, 1.0], device=device).view(1, 4, 1, 1, 1)
        fb_lead_map = fb_lead_map.expand(fb_B, -1, 1, H, W).reshape(fb_B * 4, 1, H, W)

        fb_cond = torch.cat([fb_obs_flat, fb_geos_flat, fb_lead_map], dim=1)
        
        unwrapped_model = accelerator.unwrap_model(model)
        # Sample (Input: Condition)
        # Using 20 steps for faster val check
        fb_samples_norm = unwrapped_model.sample(fb_cond, num_inference_steps=20)
        
        # Denormalize for RMSE
        if train_dataset.geos_mean is not None:
             if gm.numel() > 1:
                 fb_samples = (fb_samples_norm * gs_fb) + gm_fb
             else:
                 fb_samples = (fb_samples_norm * gs) + gm
        else:
             fb_samples = fb_samples_norm
             
        # Compute RMSE on Fixed Batch
        val_rmse = torch.sqrt(torch.mean((fb_samples - fb_target_flat)**2)).item()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val Noise Loss: {avg_val_loss:.4f} | Val RMSE (Fixed): {val_rmse:.4f}")
            
            # Log
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, val_rmse])
            
            # Plot ONLY if a new best model is found
            if val_rmse < best_val_rmse:
                print(f"New Best Model Found! RMSE improved from {best_val_rmse:.4f} to {val_rmse:.4f}. Plotting...")
                best_val_rmse = val_rmse
                
                # Plot (First Sample of Fixed Batch)
                # Move to CPU
                s_img = fb_samples[0].cpu().numpy().squeeze()
                t_img = fb_target_flat[0].cpu().numpy().squeeze()
                g_mean = fb_geos_flat[0].mean(dim=0).cpu().numpy().squeeze() 
                
                if train_dataset.geos_mean is not None:
                     if gm.numel() > 1:
                         g_scalar_m = gm[0,0,0,0].item() if gm.ndim==4 else gm.mean().item()
                         g_scalar_s = gs[0,0,0,0].item() if gs.ndim==4 else gs.mean().item()
                     else:
                         g_scalar_m = gm.item()
                         g_scalar_s = gs.item()
                     g_img = (g_mean * g_scalar_s) + g_scalar_m
                else:
                     g_img = g_mean

                diff_img = s_img - t_img
                geos_bias = g_img - t_img
                
                fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                im0 = ax[0].imshow(g_img, cmap='Blues', vmin=0, vmax=50); ax[0].set_title(f"GEOS Mean")
                plt.colorbar(im0, ax=ax[0])
                
                im1 = ax[1].imshow(t_img, cmap='Blues', vmin=0, vmax=50); ax[1].set_title("Target GPCP")
                plt.colorbar(im1, ax=ax[1])
                
                im2 = ax[2].imshow(s_img, cmap='Blues', vmin=0, vmax=50); ax[2].set_title(f"Diff Sample\nRMSE: {val_rmse:.2f}")
                plt.colorbar(im2, ax=ax[2])
                
                im3 = ax[3].imshow(diff_img, cmap='RdBu_r', vmin=-20, vmax=20); ax[3].set_title("Diff (Sample - Target)")
                plt.colorbar(im3, ax=ax[3])

                im4 = ax[4].imshow(geos_bias, cmap='RdBu_r', vmin=-20, vmax=20); ax[4].set_title("GEOS Bias (GEOS - Target)")
                plt.colorbar(im4, ax=ax[4])
                
                os.makedirs(os.path.join(config["output_dir"], "plots_diffusion"), exist_ok=True)
                plt.savefig(os.path.join(config["output_dir"], f"plots_diffusion/epoch_{epoch}_rmse_{val_rmse:.2f}.png"))
                plt.close()
            else:
                print(f"Validation RMSE ({val_rmse:.4f}) did not improve over current best ({best_val_rmse:.4f}). Skipping plot.")
            
            # SAVE LATEST
            ckpt_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'top_k_ckpts': top_k_ckpts
            }
            torch.save(ckpt_state, latest_ckpt)
            
            # SAVE TOP K logic
            # Add current
            current_path = os.path.join(config["output_dir"], f"model_epoch_{epoch}_rmse_{val_rmse:.4f}.pt")
            
            top_k_ckpts.append((val_rmse, epoch, current_path))
            top_k_ckpts.sort(key=lambda x: x[0]) # Ascending RMSE
            
            if len(top_k_ckpts) > save_top_k:
                worst = top_k_ckpts.pop() # Remove largest RMSE
                if worst[2] != current_path:
                    if os.path.exists(worst[2]):
                        os.remove(worst[2])
                        print(f"Removed worse checkpoint: {worst[2]}")
            
            # Now check if current is still in list
            is_in_top = any(x[2] == current_path for x in top_k_ckpts)
            if is_in_top:
                print(f"New Top Model! RMSE: {val_rmse:.4f}")
                torch.save(ckpt_state, current_path)
                
            # Update latest with new list
            ckpt_state['top_k_ckpts'] = top_k_ckpts
            torch.save(ckpt_state, latest_ckpt) # Update latest again with correct list

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Validation Dataset (2015-2017)
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1, # Process one by one for detailed plotting
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Model
    model = ConditionalDiffusion(
        in_channels=1,
        condition_channels=8,
        out_channels=1,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000
    )

    # Load Best or Latest Model
    # Try to find best model from top_k_ckpts in latest_ckpt
    latest_ckpt = os.path.join(config["output_dir"], "latest_diffusion_ckpt.pt")
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        top_k = checkpoint.get('top_k_ckpts', [])
        
        if top_k:
            best_ckpt_path = top_k[0][2] # (rmse, epoch, path) - sorted ascending
            print(f"Loading best model from {best_ckpt_path} (RMSE: {top_k[0][0]:.4f})")
            if os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
            else:
                 print(f"Best model file missing, loading latest instead.")
                 model.load_state_dict(checkpoint['model'])
        else:
            print("No top_k info, loading latest.")
            model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found.")
        return

    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()

    # Indices to test: 0, 10, 20, 30, 40
    test_indices = [0, 10, 20, 30, 40]
    output_dir = os.path.join(config["output_dir"], "plots_test_suite")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running Test Suite on indices {test_indices}...")

    # Iterate and select
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    current_idx = 0
    samples_processed = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if current_idx in test_indices:
                print(f"Processing sample {current_idx}...")
                
                x_obs = batch['x_obs']
                x_geos = batch['x_geos']
                y_target = batch['y_target']
                
                # Assume B=1
                B = x_obs.shape[0]
                _, _, H, W = x_obs.shape
                
                # Preprocess (Same as train)
                # x_obs: (B, 16, H, W)
                # x_geos: (B, 4, 1, 4, H, W)
                # y_target: (B, 4, H, W)
                
                # Reshape logic from train:
                x_obs_reshaped = x_obs.view(B, 4, 4, H, W)
                x_obs_lead = x_obs_reshaped.permute(0, 2, 1, 3, 4) # (B, Lead, Var, H, W)
                x_obs_flat = x_obs_lead.reshape(B * 4, 4, H, W) # (4, 4, H, W)
                
                x_geos_lead = x_geos.squeeze(2).permute(0, 2, 1, 3, 4) # (B, 4, 4, H, W)
                x_geos_flat = x_geos_lead.reshape(B * 4, 4, H, W)
                
                y_target_flat = y_target.reshape(B * 4, 1, H, W) 
                
                # Normalize Target for comparison/noise
                if val_dataset.geos_mean is not None:
                    gm = val_dataset.geos_mean.to(device)
                    gs = val_dataset.geos_std.to(device)
                    # Handle shapes
                    if gm.numel() > 1:
                        if gm.ndim == 3: gm = gm.unsqueeze(0)
                        gm_full = gm.expand(B, -1, -1, -1).reshape(B * 4, 1, H, W)
                        gs_full = gs.expand(B, -1, -1, -1).reshape(B * 4, 1, H, W)
                    else:
                         gm_full = gm
                         gs_full = gs
                
                # Lead Map for Test
                lead_map = torch.tensor([0.25, 0.50, 0.75, 1.0], device=device).view(1, 4, 1, 1, 1)
                lead_map = lead_map.expand(B, -1, 1, H, W).reshape(B * 4, 1, H, W)
                
                condition = torch.cat([x_obs_flat, x_geos_flat, lead_map], dim=1) # (4, 9, H, W)
                
                # Plot Setup: 4 Rows (Leads), 5 Columns (GEOS, Target, Diffusion, Diff Bias, GEOS Bias)
                fig = plt.figure(figsize=(25, 20))
                unwrapped_model = accelerator.unwrap_model(model)
                
                lats = np.linspace(-90, 90, H)
                lons = np.linspace(0, 360, W)
                
                def plot_panel(fig, row, col, data, title, cmap, vmin, vmax):
                    ax = fig.add_subplot(4, 5, row * 5 + col + 1, projection=ccrs.PlateCarree())
                    im = ax.imshow(data, origin='lower', extent=[lons.min(), lons.max(), lats.min(), lats.max()], 
                                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.coastlines()
                    ax.set_title(title, fontsize=10)
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    if col > 0: gl.left_labels = False
                    if row < 3: gl.bottom_labels = False
                    return im

                # Generate 5 Diffusion Members for ALL leads (Batched)
                # condition is (4, 8, H, W) -> Output (4, 1, H, W)
                diff_generations = []
                
                print(f"  Generating 5 ensemble members for Sample {current_idx}...")
                for i_ens in range(5):
                     print(f"    Member {i_ens+1}/5")
                     # Verbose=True shows the 1000 steps progress bar
                     gen = unwrapped_model.sample(condition, num_inference_steps=1000, verbose=True)
                     diff_generations.append(gen)
                
                # Stack to (5, 4, 1, H, W) then mean over ensemble -> (4, 1, H, W)
                diff_ens = torch.stack(diff_generations, dim=0) 
                diff_mean_norm_all = diff_ens.mean(dim=0)

                for lead_idx in range(4):
                    # print(f"  Plotting Lead Week {lead_idx+1}...")
                    
                    target_l = y_target_flat[lead_idx:lead_idx+1] # (1, 1, H, W)
                    geos_l = x_geos_flat[lead_idx:lead_idx+1] # (1, 4, H, W) - 4 members
                    
                    # GEOS Ensemble Mean (Normalized)
                    geos_mean_norm = geos_l.mean(dim=1, keepdim=True) # (1, 1, H, W)
                    
                    # Diffusion Mean (Normalized)
                    diff_mean_norm = diff_mean_norm_all[lead_idx:lead_idx+1] # (1, 1, H, W)
                    
                    # DENORMALIZE
                    if val_dataset.geos_mean is not None:
                        if gm.numel() > 1:
                             g_s = gm[0, lead_idx] if gm.ndim == 4 else gm[lead_idx]
                             s_s = gs[0, lead_idx] if gs.ndim == 4 else gs[lead_idx]
                             g_s = g_s.unsqueeze(0).unsqueeze(0)
                             s_s = s_s.unsqueeze(0).unsqueeze(0)
                        else:
                             g_s = gm_full
                             s_s = gs_full
                             
                        geos_mean = (geos_mean_norm * s_s) + g_s
                        diff_mean = (diff_mean_norm * s_s) + g_s
                        target = target_l # Targeted normalized or raw? usually gpcp is raw
                    else:
                        geos_mean = geos_mean_norm
                        diff_mean = diff_mean_norm
                        target = target_l
                    
                    # Calculate RMSEs
                    geos_rmse = torch.sqrt(torch.mean((geos_mean - target)**2)).item()
                    diff_rmse = torch.sqrt(torch.mean((diff_mean - target)**2)).item()
                    
                    # Data for plotting
                    g_img = geos_mean.cpu().numpy().squeeze()
                    t_img = target.cpu().numpy().squeeze()
                    d_img = diff_mean.cpu().numpy().squeeze()
                    diff_map = d_img - t_img
                    geos_diff_map = g_img - t_img
                    
                    # Plot Row
                    im0 = plot_panel(fig, lead_idx, 0, g_img, f"W{lead_idx+1}: GEOS Ens Mean\nRMSE: {geos_rmse:.2f}", 'Blues', 0, 50)
                    im1 = plot_panel(fig, lead_idx, 1, t_img, f"W{lead_idx+1}: Target GPCP", 'Blues', 0, 50)
                    im2 = plot_panel(fig, lead_idx, 2, d_img, f"W{lead_idx+1}: Diffusion Mean\nRMSE: {diff_rmse:.2f}", 'Blues', 0, 50)
                    im3 = plot_panel(fig, lead_idx, 3, diff_map, f"W{lead_idx+1}: Diff Bias (Diff-Target)", 'RdBu_r', -20, 20)
                    im4 = plot_panel(fig, lead_idx, 4, geos_diff_map, f"W{lead_idx+1}: GEOS Bias (GEOS-Target)", 'RdBu_r', -20, 20)
                    
                    # Add colorbars to the right end
                    if lead_idx == 0:
                        cax1 = fig.add_axes([0.92, 0.6, 0.015, 0.25])
                        fig.colorbar(im0, cax=cax1, label='mm/day')
                        cax2 = fig.add_axes([0.92, 0.15, 0.015, 0.25])
                        fig.colorbar(im3, cax=cax2, label='mm/day')

                plt.suptitle(f"Sample Index {current_idx} (Val Set) - All Lead Weeks", fontsize=16)
                plt.savefig(os.path.join(output_dir, f"test_sample_{current_idx}_all_leads.png"), bbox_inches='tight', dpi=150)
                plt.close()
                print(f"Saved multi-lead plot for sample {current_idx}.")
                
                samples_processed += 1
            
            current_idx += 1
                
    print("Test Suite Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        train()
