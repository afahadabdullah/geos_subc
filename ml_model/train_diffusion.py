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
        top_k_ckpts = checkpoint.get('top_k_ckpts', [])
        print(f"Resuming from epoch {start_epoch}")

    best_val_rmse = float('inf')

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
            condition = torch.cat([x_obs_flat, x_geos_flat], dim=1)
            
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
            
            loss = F.mse_loss(noise_pred, noise)
            
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
                
                v_condition = torch.cat([vx_obs_flat, vx_geos_flat], dim=1)
                
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
                val_loss_sum += F.mse_loss(v_pred, v_noise).item()
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
            
        fb_cond = torch.cat([fb_obs_flat, fb_geos_flat], dim=1)
        
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
            
            # Plot (First Sample of Fixed Batch)
            # Move to CPU
            s_img = fb_samples[0].cpu().numpy().squeeze()
            t_img = fb_target_flat[0].cpu().numpy().squeeze()
            g_mean = fb_geos_flat[0].mean(dim=0).cpu().numpy().squeeze() # GEOS Mean (Normalized space?)
            # Wait, fb_geos_flat is normalized? 
            # In dataset, geos is normalized.
            # So we need to denorm GEOS for plot if we want to compare with target
            if train_dataset.geos_mean is not None:
                 # Quick denorm for plot
                 # Using the first element of gm/gs
                 if gm.numel() > 1:
                     g_scalar_m = gm[0,0,0,0].item() if gm.ndim==4 else gm.mean().item()
                     g_scalar_s = gs[0,0,0,0].item() if gs.ndim==4 else gs.mean().item()
                 else:
                     g_scalar_m = gm.item()
                     g_scalar_s = gs.item()
                 g_img = (g_img * g_scalar_s) + g_scalar_m

            diff_img = s_img - t_img
            
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            im0 = ax[0].imshow(g_img, cmap='Blues', vmin=0, vmax=50); ax[0].set_title(f"GEOS Mean")
            plt.colorbar(im0, ax=ax[0])
            
            im1 = ax[1].imshow(t_img, cmap='Blues', vmin=0, vmax=50); ax[1].set_title("Target GPCP")
            plt.colorbar(im1, ax=ax[1])
            
            im2 = ax[2].imshow(s_img, cmap='Blues', vmin=0, vmax=50); ax[2].set_title(f"Diff Sample (RMSE: {val_rmse:.2f})")
            plt.colorbar(im2, ax=ax[2])
            
            im3 = ax[3].imshow(diff_img, cmap='RdBu_r', vmin=-20, vmax=20); ax[3].set_title("Diff (Sample - Target)")
            plt.colorbar(im3, ax=ax[3])
            
            os.makedirs(os.path.join(config["output_dir"], "plots_diffusion"), exist_ok=True)
            plt.savefig(os.path.join(config["output_dir"], f"plots_diffusion/epoch_{epoch}_rmse_{val_rmse:.2f}.png"))
            plt.close()
            
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

if __name__ == "__main__":
    train()
