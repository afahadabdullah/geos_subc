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
    # Since loader is sequential, we can just iterate and pick
    
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
                B, C_obs, H, W = x_obs.shape # (1, 3, L, H, W) in dataset? 
                # Wait, dataset_hybrid returns x_obs as (16, H, W).
                # Batch dim added by loader -> (1, 16, H, W)
                
                # Preprocess (Same as train)
                # x_obs: (1, 16, H, W)
                # x_geos: (1, 4, 1, 4, H, W)
                # y_target: (1, 4, H, W)
                
                # Reshape logic from train:
                # x_obs_reshaped = x_obs.view(B, 4, 4, H, W) -> (B, Var, Lead, H, W)
                x_obs_reshaped = x_obs.view(B, 4, 4, H, W)
                x_obs_lead = x_obs_reshaped.permute(0, 2, 1, 3, 4) # (B, Lead, Var, H, W)
                x_obs_flat = x_obs_lead.reshape(B * 4, 4, H, W) # (4, 4, H, W)
                
                x_geos_lead = x_geos.squeeze(2).permute(0, 2, 1, 3, 4) # (B, 4, 4, H, W)
                x_geos_flat = x_geos_lead.reshape(B * 4, 4, H, W)
                
                y_target_flat = y_target.reshape(B * 4, 1, H, W) 
                # y_target_flat is (4, 1, H, W) - 4 lead times
                
                # Normalize Target for comparison/noise
                if val_dataset.geos_mean is not None:
                    gm = val_dataset.geos_mean.to(device)
                    gs = val_dataset.geos_std.to(device)
                    # Handle shapes
                    if gm.numel() > 1:
                        if gm.ndim == 3: gm = gm.unsqueeze(0)
                        gm_full = gm.expand(B, -1, -1, -1).reshape(B * 4, 1, H, W)
                        gs_full = gs.expand(B, -1, -1, -1).reshape(B * 4, 1, H, W)
                        # For plotting denorm, we need scalar/broadcastable keys
                    else:
                         gm_full = gm
                         gs_full = gs
                
                condition = torch.cat([x_obs_flat, x_geos_flat], dim=1) # (4, 8, H, W)
                
                # FOCUS ON LEAD TIME 0 (First Week) for Simplicity in specific plot?
                # Or plot all 4? User said "5 samples".
                # Let's plot Lead Time 0 for each of the 5 samples.
                
                cond_l0 = condition[0:1] # (1, 8, H, W)
                target_l0 = y_target_flat[0:1] # (1, 1, H, W)
                geos_l0 = x_geos_flat[0:1] # (1, 4, H, W) - 4 members
                
                # GEOS Ensemble Mean (Normalized)
                geos_mean_norm = geos_l0.mean(dim=1, keepdim=True) # (1, 1, H, W)
                
                # Generate 10 Diffusion Members
                diff_generations = []
                unwrapped_model = accelerator.unwrap_model(model)
                
                for i_ens in range(10):
                    # Sample (1000 steps)
                    # cond_l0 is (1, 8, H, W)
                    gen = unwrapped_model.sample(cond_l0, num_inference_steps=1000)
                    diff_generations.append(gen)
                
                diff_ens = torch.cat(diff_generations, dim=0) # (10, 1, H, W)
                diff_mean_norm = diff_ens.mean(dim=0, keepdim=True) # (1, 1, H, W)
                
                # DENORMALIZE ALL
                if val_dataset.geos_mean is not None:
                    # Need appropriate gm/gs for lead 0
                    if gm.numel() > 1:
                        # gm is (1, L, H, W) -> (1, 4, H, W)
                         g_s = gm[0, 0] if gm.ndim == 4 else gm[0]
                         s_s = gs[0, 0] if gs.ndim == 4 else gs[0]
                         # Add dims (1, 1, H, W) if needed
                         g_s = g_s.unsqueeze(0).unsqueeze(0)
                         s_s = s_s.unsqueeze(0).unsqueeze(0)
                    else:
                         g_s = gm_full
                         s_s = gs_full
                         
                    geos_mean = (geos_mean_norm * s_s) + g_s
                    diff_mean = (diff_mean_norm * s_s) + g_s
                    target = target_l0 # Target was already raw from dataset_hybrid? 
                    # Wait, in train loop:
                    # y_target = batch['y_target'] -> raw
                    # target_normalized = ...
                    # So y_target_flat is RAW.
                else:
                    geos_mean = geos_mean_norm
                    diff_mean = diff_mean_norm
                    target = target_l0
                
                # Calculate RMSEs
                geos_rmse = torch.sqrt(torch.mean((geos_mean - target)**2)).item()
                diff_rmse = torch.sqrt(torch.mean((diff_mean - target)**2)).item()
                
                # Plot
                # Move to CPU
                g_img = geos_mean.cpu().numpy().squeeze()
                t_img = target.cpu().numpy().squeeze()
                d_img = diff_mean.cpu().numpy().squeeze()
                diff_map = d_img - t_img
                
                fig, ax = plt.subplots(1, 4, figsize=(24, 6))
                
                im0 = ax[0].imshow(g_img, cmap='Blues', vmin=0, vmax=50)
                ax[0].set_title(f"GEOS Ens Mean\nRMSE: {geos_rmse:.2f}")
                plt.colorbar(im0, ax=ax[0])
                
                im1 = ax[1].imshow(t_img, cmap='Blues', vmin=0, vmax=50)
                ax[1].set_title("Target GPCP")
                plt.colorbar(im1, ax=ax[1])
                
                im2 = ax[2].imshow(d_img, cmap='Blues', vmin=0, vmax=50)
                ax[2].set_title(f"Diffusion Ens Mean (10)\nRMSE: {diff_rmse:.2f}")
                plt.colorbar(im2, ax=ax[2])
                
                im3 = ax[3].imshow(diff_map, cmap='RdBu_r', vmin=-20, vmax=20)
                ax[3].set_title("Diff (Model - Target)")
                plt.colorbar(im3, ax=ax[3])
                
                plt.suptitle(f"Sample Index {current_idx} (Val)")
                plt.savefig(os.path.join(output_dir, f"test_sample_{current_idx}.png"))
                plt.close()
                print(f"Saved plot for sample {current_idx}. G_RMSE={geos_rmse:.2f}, D_RMSE={diff_rmse:.2f}")
                
                samples_processed += 1
            
            current_idx += 1
            if current_idx > max(test_indices):
                break
                
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
