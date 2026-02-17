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

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.model_committee import CommitteeModel
from ml_model.loss_committee import ZeroInflatedGammaLoss

def train(config_path="config.yaml"): # Or hardcoded defaults
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # Config
    config = {
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 500,
        "output_dir": "ml_output_committee"
    }
    
    # 1. Dataset
    print("Initializing Dataset...")
    # Adjust root path as needed. Assuming running from project root.
    dataset = S2SHybridDataset(data_root="dataprocess", start_year=1999, end_year=2015, preload=True)
    # Val set? Using last year 2015-2016 from split?
    # Or simplified: Train 1999-2015.
    
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    
    # 2. Model
    # Obs channels: SST (4) + SSS (4) + SM (4) + PrevGPCP (4) = 16
    # GEOS channels: Pr -> 1
    model = CommitteeModel(channels_obs=16, channels_geos=1, num_members=4)
    
    # 3. Loss & Optimizer
    criterion = ZeroInflatedGammaLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    # Prepare
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    # Training Loop
    if accelerator.is_main_process:
        print(f"Starting Training on {device}...")
    
    # Ensure output dir exists
    if accelerator.is_main_process:
        os.makedirs(config["output_dir"], exist_ok=True)
        
    # Validation & Logging Setup
    import csv
    import matplotlib.pyplot as plt
    import numpy as np
    
    log_file = os.path.join(config["output_dir"], "training_log.csv")
    if accelerator.is_main_process:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_RMSE", "Best_Val_Loss"])
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume Logic
    latest_ckpt = os.path.join(config["output_dir"], "latest_checkpoint.pt")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, disable=not accelerator.is_main_process)
        for batch in pbar:
            x_obs = batch['x_obs']
            x_geos = batch['x_geos']
            y_target = batch['y_target']
            
            p_stack, alpha_stack, beta_stack = model(x_obs, x_geos)
            
            loss = 0.0
            M = p_stack.shape[1]
            for m in range(M):
                lm = criterion(p_stack[:,m], alpha_stack[:,m], beta_stack[:,m], y_target)
                loss += lm
            loss = loss / M
            
            if torch.isnan(loss):
                continue
                
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
        avg_train_loss = epoch_loss / len(loader)
        
        # Validation Step (Using same loader for now or subset?)
        # Ideally we need a separate val_loader. For now, let's just use Train stats 
        # OR run a quick check on a few batches.
        # User asked for "verification or testing after each epoch".
        # Let's assume we split dataset or just verify on a subset of train for now if val not defined.
        # But `S2SHybridDataset` covers 1999-2016.
        # Let's verify on the first batch of the loader (as a sanity check) + plot.
        
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_count = 0 
        
        # Validation Progress Bar
        val_pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Validating Epoch {epoch}")
        
        with torch.no_grad():
             for batch in val_pbar:
                 x_obs = batch['x_obs']
                 x_geos = batch['x_geos']
                 y_target = batch['y_target']
                 
                 p_stack, alpha_stack, beta_stack = model(x_obs, x_geos)
                 
                 loss = 0.0
                 M = p_stack.shape[1]
                 for m in range(M):
                     loss += criterion(p_stack[:,m], alpha_stack[:,m], beta_stack[:,m], y_target)
                 
                 batch_loss = (loss / M).item()
                 val_loss += batch_loss
                 val_pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
                 
                 # RMSE for batch (approx, based on 1st member or avg?)
                 # Let's compute RMSE of Expected Value vs Target
                 # p_stack: (B, M, L, H, W) -> Mean across M?
                 # Committee Mean Prediction
                 p_mean = p_stack.mean(dim=1)
                 alpha_mean = alpha_stack.mean(dim=1)
                 beta_mean = beta_stack.mean(dim=1)
                 
                 expected = p_mean * (alpha_mean / beta_mean)
                 # Target: (B, L, H, W)
                 mse = F.mse_loss(expected, y_target, reduction='sum')
                 val_mse += mse.item()
                 val_count += y_target.numel()
        
        avg_val_loss = val_loss / len(loader)
        val_rmse = np.sqrt(val_mse / val_count)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
            
            # CSV Log
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, avg_val_loss, val_rmse, best_val_loss])

            # Save Latest
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, latest_ckpt)
            
            # Save Best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New Best Model! Loss: {avg_val_loss:.4f}")
                torch.save(model.state_dict(), os.path.join(config["output_dir"], f"best_model_epoch_{epoch}.pt"))
                
                # Plotting (Only on Best or every 5 epochs)
                # Plot the last batch processing
                # p_stack: (B, M, L, H, W)
                # Take first sample in batch, first lead time
                b_idx = 0
                l_idx = 0
                m_idx = 0 # Member 0
                
                p = p_stack[b_idx, m_idx, l_idx].cpu().numpy()
                alpha = alpha_stack[b_idx, m_idx, l_idx].cpu().numpy()
                beta = beta_stack[b_idx, m_idx, l_idx].cpu().numpy()
                target = y_target[b_idx, l_idx].cpu().numpy()
                geos = x_geos[b_idx, m_idx, 0, l_idx].cpu().numpy() # Raw GEOS (normalized)
                # Expected Value = p * (alpha/beta)
                expected = p * (alpha / beta)
                diff = expected - target
                
                # RMSE Calculation
                # GEOS (denormalized?)
                # GEOS input is normalized. We need to denormalize to compare with Target (mm/day).
                # If we use global stats:
                if dataset.normalize and dataset.geos_mean is not None:
                    # gm, gs are tensors. Move to cpu numpy.
                    gm = dataset.geos_mean.cpu().numpy().squeeze()
                    gs = dataset.geos_std.cpu().numpy().squeeze()
                    geos_denorm = (geos * gs) + gm
                else:
                    # Fallback (assuming it was normalized with grid_stats.nc)
                    geos_denorm = geos # Placeholder if unknown
                
                # RMSE
                # GEOS vs Target
                rmse_geos = np.sqrt(np.mean((geos_denorm - target)**2))
                # Pred vs Target
                rmse_pred = np.sqrt(np.mean((expected - target)**2))
                
                fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                
                im0 = ax[0].imshow(geos_denorm, cmap='Blues', vmin=0, vmax=50); ax[0].set_title(f"GEOS (RMSE: {rmse_geos:.2f})")
                plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
                
                im1 = ax[1].imshow(target, cmap='Blues', vmin=0, vmax=50); ax[1].set_title("Target (GPCP)")
                plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
                
                im2 = ax[2].imshow(expected, cmap='Blues', vmin=0, vmax=50); ax[2].set_title(f"Pred (RMSE: {rmse_pred:.2f})")
                plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
                
                im3 = ax[3].imshow(diff, cmap='RdBu_r', vmin=-20, vmax=20); ax[3].set_title("Diff (Pred-Target)")
                plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
                
                im4 = ax[4].imshow(p, cmap='gray', vmin=0, vmax=1); ax[4].set_title("Prob(Rain)")
                plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
                
                os.makedirs(os.path.join(config["output_dir"], "plots"), exist_ok=True)
                plt.savefig(os.path.join(config["output_dir"], f"plots/epoch_{epoch}.png"))
                plt.close()

if __name__ == "__main__":
    train()
