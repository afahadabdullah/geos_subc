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
    model, optimizer, loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, lr_scheduler
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
        
        # Validation (Sample 1 batch)
        # Use first batch of loader (assuming consistent if shuffle=False? Train uses shuffle=True)
        # Just grab one batch inside this process logic for now
        # Ideally fix validation set.
        
        model.eval()
        val_rmse_sum = 0
        val_count = 0
        
        # Consistent Validation Data (from first batch of Epoch)
        # In a real setup, use val_loader
        # Let's verify on ONE sample from the LAST batch used (easy access)
        # Or recreate small loop
        
        with torch.no_grad():
            # Use last batch data for plotting (approximate val)
            B_val = 1 
            cond_val = condition[:B_val] 
            target_val_norm = target_normalized[:B_val]
            target_val_raw = y_target_flat[:B_val]
            
            # Need GEOS mean for denormalization later, or just use raw geos
            # Just grab raw GEOS mean from x_geos_flat
            geos_val_mean_raw = x_geos_flat[:B_val].mean(dim=1, keepdim=True) # Normalized scale!
            # If dataset.geos_mean is present, x_geos is roughly N(0,1).
            # But wait, dataset DOES normalize GEOS.
            # So `geos_val_mean_raw` is in Normalized Space.
            
            
            # Sample (Denoise)
            unwrapped_model = accelerator.unwrap_model(model)
            # 50 steps for speed in val
            samples_norm = unwrapped_model.sample(cond_val, num_inference_steps=50)
            
            # Denormalize
            if train_dataset.geos_mean is not None:
                gm = train_dataset.geos_mean.to(device)
                gs = train_dataset.geos_std.to(device)
                
                # Reshape for broadcasting just in case
                # samples_norm is (1, 1, H, W)
                 # gm/gs might be (L, H, W)?
                # We selected B_val=1 which is ONE sample from B*L flat batch. 
                # Ideally we know WHICH lead time it is.
                # It is the first one -> Lead 0.
                
                if gm.numel() > 1:
                    # Assume (1, L, H, W) -> slice lead 0
                    gm_s = gm[0, 0] if gm.ndim == 4 else gm[0] # Handle shapes roughly
                    gs_s = gs[0, 0] if gs.ndim == 4 else gs[0]
                    # This is getting hacky.
                    # Best effort denorm
                    # If gm is scalar, easy.
                    samples_denorm = (samples_norm * gs) + gm
                    target_denorm = (target_val_norm * gs) + gm # Should match target_val_raw
                    # geos_denorm = (geos_val_mean_raw * gs) + gm # GEOS was normalized
                else:
                    samples_denorm = (samples_norm * gs) + gm
                    target_denorm = (target_val_norm * gs) + gm
                    geos_denorm = (geos_val_mean_raw * gs) + gm
            else:
                 samples_denorm = samples_norm
                 target_denorm = target_val_norm
                 geos_denorm = geos_val_mean_raw
            
            val_rmse = torch.sqrt(torch.mean((samples_denorm - target_denorm)**2)).item()
            geos_rmse = torch.sqrt(torch.mean((geos_denorm - target_denorm)**2)).item()
            
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | GEOS RMSE: {geos_rmse:.4f}")
            
            # Log
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, val_rmse])
            
            # Plot
            # Move to CPU
            s_img = samples_denorm.cpu().numpy().squeeze()
            t_img = target_denorm.cpu().numpy().squeeze()
            g_img = geos_denorm.cpu().numpy().squeeze()
            diff_img = s_img - t_img
            
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            im0 = ax[0].imshow(g_img, cmap='Blues', vmin=0, vmax=50); ax[0].set_title(f"GEOS Mean (RMSE: {geos_rmse:.2f})")
            plt.colorbar(im0, ax=ax[0])
            
            im1 = ax[1].imshow(t_img, cmap='Blues', vmin=0, vmax=50); ax[1].set_title("Target GPCP")
            plt.colorbar(im1, ax=ax[1])
            
            im2 = ax[2].imshow(s_img, cmap='Blues', vmin=0, vmax=50); ax[2].set_title(f"Diff Sample (RMSE: {val_rmse:.2f})")
            plt.colorbar(im2, ax=ax[2])
            
            im3 = ax[3].imshow(diff_img, cmap='RdBu_r', vmin=-20, vmax=20); ax[3].set_title("Diff (Sample - Target)")
            plt.colorbar(im3, ax=ax[3])
            
            os.makedirs(os.path.join(config["output_dir"], "plots_diffusion"), exist_ok=True)
            plt.savefig(os.path.join(config["output_dir"], f"plots_diffusion/epoch_{epoch}.png"))
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
            
            # We want to check if this is worthy
            # Algorithm:
            # 1. Append (rmse, epoch, current_path)
            # 2. Sort
            # 3. If len > K, remove worst
            # 4. If current survived, SAVE IT.
            
            top_k_ckpts.append((val_rmse, epoch, current_path))
            top_k_ckpts.sort(key=lambda x: x[0]) # Ascending RMSE
            
            if len(top_k_ckpts) > save_top_k:
                worst = top_k_ckpts.pop() # Remove largest RMSE
                # If worst was the one we just added (current), then we don't save.
                # If worst was an old one, delete file.
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
