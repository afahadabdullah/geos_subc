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
    train_dataset = S2SHybridDataset(
        root_dir=config["data_dir"],
        year_range=config["train_years"],
        normalize=True
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
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Scheduler might need state dict? 
        start_epoch = checkpoint['epoch'] + 1
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
            
            B, _, L, H, W = x_obs.shape
            
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
                if gm.ndim == 3: gm = gm.unsqueeze(0).unsqueeze(0) # (1, 1, L, H, W) but here flat target?
                # y_target_flat corresponds to B*L flattened.
                # gm is (1, L, H, W)?
                # If gm is (1, L, H, W), we need to repeat B times then flatten?
                # Or reshape gm/gs to compatible shape.
                
                # Careful: Check shape of gm/gs
                # S2SHybridDataset: gm can be (1, L, H, W) or (1,)
                # Let's handle generic case
                if gm.numel() > 1:
                    # Assume (1, L, H, W) or (L, H, W)
                    # Expand B times -> (B, 1, L, H, W)
                    # Then reshape to (B*L, 1, H, W)
                    
                    # Ensure dimensions match target_flat (B*L, 1, H, W)
                    # target was (B, L, H, W) -> (B*L, 1, H, W)
                    # So expand gm to (B, L, H, W)
                    if gm.ndim == 2: gm = gm.unsqueeze(0).unsqueeze(0) # (1,1,H,W)? No dataset says (1,L,H,W)
                    elif gm.ndim == 3: gm = gm.unsqueeze(0) # (1, L, H, W)
                    
                    # We need to replicate for batch
                    # (1, L, H, W) -> (B, L, H, W)
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
            noisy_target = model.noise_scheduler.add_noise(y_target_flat, noise, timesteps)
            
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
            # Use last batch data for plotting
            B_val = 1 # Just sample 1 image
            # Take index 0 from flat batch
            cond_val = condition[:B_val] # (1, 7, H, W)
            target_val = y_target_flat[:B_val] # (1, 1, H, W)
            geos_val_mean = x_geos_flat[:B_val].mean(dim=1, keepdim=True) # (1, 1, H, W)
            
            # Sample (Denoise)
            # Use module directly if wrapped by DDP
            unwrapped_model = accelerator.unwrap_model(model)
            # 50 steps for speed in val
            samples = unwrapped_model.sample(cond_val, num_inference_steps=50)
            
            # Stats (Denormalize for RMSE)
            rmse_val = 0
            if train_dataset.geos_mean is not None:
                gm = train_dataset.geos_mean.to(device)
                gs = train_dataset.geos_std.to(device)
                om = train_dataset.obs_mean.to(device) # Precip is obs var? No target is y_target
                # Target is Precip (index 0 of obs? No, obs is vars 0,1,2: SST, SSS, Soil? Wait)
                # In dataset_hybrid, y_target is loaded separately.
                # Assuming y_target is normalized using geos stats? Or obs stats?
                # dataset_hybrid.py:
                # self.obs_mean = global_stats['obs_mean'] # (5,)
                # target is index 0 of obs vars (Precip) -> Use obs_mean[0], obs_std[0]
                
                # Careful: obs_mean is (5,) = [Precip, SST, SSS, Soil?, GPCP?]
                # load_stats says: obs_vars = ['precip', 'sst', 'sss', 'soil_moisture', 'gpcp'?]
                # Actually calculate_global_stats.py:
                # obs_vars = ['gpcp', 'sst', 'sss', 'soil_moisture'] -> 4 vars?
                # dataset_hybrid.py:
                # self.obs_vars = ['gpcp', 'sst', 'sss', 'soil_moisture']
                # target is gpcp -> index 0
                
                tm = train_dataset.obs_mean[0].to(device)
                ts = train_dataset.obs_std[0].to(device)
                
                samples_denorm = (samples * ts) + tm
                target_denorm = (target_val * ts) + tm
                geos_denorm = (geos_val_mean * gs) + gm # GEOS uses its own stats
                
            else:
                samples_denorm = samples
                target_denorm = target_val
                geos_denorm = geos_val_mean
                
            val_rmse = torch.sqrt(torch.mean((samples_denorm - target_denorm)**2)).item()
            geos_rmse = torch.sqrt(torch.mean((geos_denorm - target_denorm)**2)).item()
            
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | GEOS RMSE: {geos_rmse:.4f}")
            
            # Log
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, val_rmse])
            
            # Save Checkpoint
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, latest_ckpt)
            
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

if __name__ == "__main__":
    train()
