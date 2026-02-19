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
    # In: 4 (Target Noisy - 4 Leads)
    # Cond: 16 (Obs) + 16 (GEOS) + 2 (Month) = 34
    model = ConditionalDiffusion(
        in_channels=4,
        condition_channels=34,
        out_channels=4,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        num_train_timesteps=1000
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        steps_per_epoch=len(loader), 
        epochs=config["epochs"]
    )

    # Prepare
    model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, lr_scheduler
    )

    # Area Weights for Loss
    # Latitude range: -90 to 90 (181 points)
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device) # (1, 1, H, 1) -> Broadcasts to (B, 4, H, W)

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
        if 'top_k_ckpts' in checkpoint:
            top_k_ckpts = checkpoint['top_k_ckpts']
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
            # x_obs: (B, 16, H, W) [Stacked Obs]
            # x_geos: (B, 4, 1, 4, H, W) [Members, 1, Leads, H, W]
            # y_target: (B, 4, H, W) [4 Leads]
            
            x_obs = batch['x_obs']
            x_geos = batch['x_geos']
            y_target = batch['y_target']
            months = batch['month'] # (B,)
            
            B, _, H, W = x_obs.shape
            
            # GEOS: (B, 4, 1, 4, H, W) -> (B, 16, H, W)
            # Flatten Members(4) and Leads(4) into Channels(16)
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
            
            # Target is already (B, 4, H, W)
            
            # NORMALIZE TARGET
            # Use GEOS stats for consistency
            if train_dataset.geos_mean is not None:
                gm = train_dataset.geos_mean.to(device)
                gs = train_dataset.geos_std.to(device)
                # gm shape check. Dataset loads it. 
                # If global scalar: (1,) or ()
                # If grid: (1, 4, H, W) hopefully? Dataset loads grid_stats.nc
                # If grid_stats.nc matches (L, H, W), then gm is (1, 4, H, W).
                # Broadcasting (1,4,H,W) to (B,4,H,W) works.
                
                # Careful: If gm is (1,), it works.
                # If gm is (1, L, H, W) == (1, 4, H, W), it works.
                target_normalized = (y_target - gm) / (gs * 3.0)
            else:
                 target_normalized = y_target # Fallback
            
            # Month Embeddings (Seasonality)
            # months is (B,)
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            
            # Condition: (B, 34, H, W)
            condition = torch.cat([x_obs, x_geos_flat, sin_month, cos_month], dim=1)
            
            # Sample Timesteps
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps, 
                (B,), device=device
            ).long()
            
            # Add Noise
            noise = torch.randn_like(target_normalized)
            noisy_target = model.noise_scheduler.add_noise(target_normalized, noise, timesteps)
            
            # Predict Noise
            # Inputs: noisy_target(B,4,H,W), condition(B,34,H,W), timesteps(B)
            # Output: noise_pred(B,4,H,W)
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
        
        with torch.no_grad():
             for val_batch in val_loader:
                vx_obs = val_batch['x_obs']
                vx_geos = val_batch['x_geos']
                vy_target = val_batch['y_target']
                v_months = val_batch['month']
                
                vB, _, vH, vW = vx_obs.shape
                
                # Reshape GEOS
                vx_geos_flat = vx_geos.squeeze(2).reshape(vB, 16, vH, vW)

                # Normalize Target
                if train_dataset.geos_mean is not None:
                    gm = train_dataset.geos_mean.to(device)
                    gs = train_dataset.geos_std.to(device)
                    vtarget_norm = (vy_target - gm) / (gs * 3.0)
                else:
                    vtarget_norm = vy_target
                
                # Month Embeddings
                v_sin_month = torch.sin(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_cos_month = torch.cos(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                
                # Condition
                v_condition = torch.cat([vx_obs, vx_geos_flat, v_sin_month, v_cos_month], dim=1)
                
                # Validation Loss (Noise MSE)
                v_timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (vB,), device=device).long()
                v_noise = torch.randn_like(vtarget_norm)
                v_noisy = model.noise_scheduler.add_noise(vtarget_norm, v_noise, v_timesteps)
                v_pred = model(v_noisy, v_condition, v_timesteps)
                
                v_loss = (area_weights * (v_pred - v_noise)**2).mean()
                val_loss_sum += v_loss.item()
                val_count += 1

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
        
        # --- FIXED BATCH VISUALIZATION & RMSE ---
        # Prepare Fixed Batch
        fb_obs = fixed_val_batch['x_obs'].to(device)
        fb_geos = fixed_val_batch['x_geos'].to(device)
        fb_target = fixed_val_batch['y_target'].to(device)
        fb_months = fixed_val_batch['month'].to(device)
        
        fb_B = fb_obs.shape[0]
        fb_geos_flat = fb_geos.squeeze(2).reshape(fb_B, 16, H, W)
        
        # Month
        fb_sin_month = torch.sin(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_cos_month = torch.cos(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        
        # Condition
        fb_cond = torch.cat([fb_obs, fb_geos_flat, fb_sin_month, fb_cos_month], dim=1)
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Sample (Output: B, 4, H, W)
        fb_samples_norm = unwrapped_model.sample(fb_cond, num_inference_steps=20)
        
        # Denormalize
        if train_dataset.geos_mean is not None:
             gm = train_dataset.geos_mean.to(device)
             gs = train_dataset.geos_std.to(device)
             fb_samples = (fb_samples_norm * gs * 3.0) + gm
        else:
             fb_samples = fb_samples_norm
             
        # Compute RMSE on Fixed Batch (All Leads)
        val_rmse = torch.sqrt(torch.mean((fb_samples - fb_target)**2)).item()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val Noise Loss: {avg_val_loss:.4f} | Val RMSE (Fixed): {val_rmse:.4f}")
            
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, val_rmse])
            
            if val_rmse < best_val_rmse:
                print(f"New Best Model Found! RMSE improved from {best_val_rmse:.4f} to {val_rmse:.4f}. Plotting...")
                best_val_rmse = val_rmse
                
                # Plot First Sample, All 4 Leads
                # s_img: (4, H, W)
                s_img_all = fb_samples[0].cpu().numpy()
                t_img_all = fb_target[0].cpu().numpy()
                
                # GEOS Mean (4 leads)
                # geos_flat: (16, H, W) -> Reshape to (4, 4, H, W) to avg over members
                g_flat = fb_geos_flat[0] # (16, H, W)
                g_ens = g_flat.view(4, 4, H, W) # (Members, Leads, H, W)
                g_mean_norm = g_ens.mean(dim=0) # (Leads, H, W) -> (4, H, W)
                
                # Denormalize GEOS Mean
                if train_dataset.geos_mean is not None:
                    # Need CPU stats
                    gm_cpu = train_dataset.geos_mean.squeeze().cpu().numpy() # (4, H, W) or (1,)
                    gs_cpu = train_dataset.geos_std.squeeze().cpu().numpy()
                    g_img_all = (g_mean_norm.cpu().numpy() * gs_cpu * 3.0) + gm_cpu
                else:
                    g_img_all = g_mean_norm.cpu().numpy()

                fig, axes = plt.subplots(4, 5, figsize=(25, 20))
                
                for l_idx in range(4):
                    g_img = g_img_all[l_idx]
                    t_img = t_img_all[l_idx]
                    s_img = s_img_all[l_idx]
                    diff_img = s_img - t_img
                    geos_bias = g_img - t_img
                    
                    rmse_l = np.sqrt(np.mean((s_img - t_img)**2))
                    
                    if l_idx == 0: axes[l_idx, 0].set_title("GEOS Mean")
                    axes[l_idx, 0].imshow(g_img, cmap='Blues', vmin=0, vmax=50)
                    
                    if l_idx == 0: axes[l_idx, 1].set_title("Target GPCP")
                    axes[l_idx, 1].imshow(t_img, cmap='Blues', vmin=0, vmax=50)
                    
                    if l_idx == 0: axes[l_idx, 2].set_title("Diffusion")
                    axes[l_idx, 2].imshow(s_img, cmap='Blues', vmin=0, vmax=50)
                    axes[l_idx, 2].set_ylabel(f"Week {l_idx+1}\nRMSE: {rmse_l:.2f}")
                    
                    if l_idx == 0: axes[l_idx, 3].set_title("Diff Bias")
                    axes[l_idx, 3].imshow(diff_img, cmap='RdBu_r', vmin=-20, vmax=20)
                    
                    if l_idx == 0: axes[l_idx, 4].set_title("GEOS Bias")
                    axes[l_idx, 4].imshow(geos_bias, cmap='RdBu_r', vmin=-20, vmax=20)

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
            current_path = os.path.join(config["output_dir"], f"model_epoch_{epoch}_rmse_{val_rmse:.4f}.pt")
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

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Validation Dataset (TEST MODE usually uses Val set or separate Test set)
    # Re-using Val parameters for consistency with user request
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
        in_channels=4,
        condition_channels=34,
        out_channels=4,
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
                
                x_obs = batch['x_obs'].to(device)
                x_geos = batch['x_geos'].to(device)
                y_target = batch['y_target'].to(device)
                t_months = batch['month'].to(device)
                
                # Assume B=1
                B = x_obs.shape[0]
                _, _, H, W = x_obs.shape
                
                # Reshape GEOS
                x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
                
                # Month Embeddings (Test)
                t_sin_month = torch.sin(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
                t_cos_month = torch.cos(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
                
                condition = torch.cat([x_obs, x_geos_flat, t_sin_month, t_cos_month], dim=1)
                
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

                # Generate 5 Diffusion Members (Batched)
                # Output: (5, 4, H, W)
                diff_generations = []
                
                print(f"  Generating 5 ensemble members for Sample {current_idx}...")
                for i_ens in range(5):
                     print(f"    Member {i_ens+1}/5")
                     gen_norm = unwrapped_model.sample(condition, num_inference_steps=1000, verbose=True)
                     
                     # Denormalize
                     if val_dataset.geos_mean is not None:
                        gm = val_dataset.geos_mean.to(device)
                        gs = val_dataset.geos_std.to(device)
                        gen = (gen_norm * gs * 3.0) + gm
                     else:
                        gen = gen_norm
                        
                     diff_generations.append(gen)
                
                # Stack: (5, B, 4, H, W). B=1 -> (5, 4, H, W)
                diff_ens = torch.cat(diff_generations, dim=0) 
                diff_mean_all = diff_ens.mean(dim=0) # (4, H, W)

                # Prepare Target & GEOS for plotting
                # Target: (1, 4, H, W) -> (4, H, W)
                target_all = y_target.squeeze(0)
                
                # GEOS: x_geos_flat is (1, 16, H, W) -> (4, 4, H, W) [Members, Leads, H, W]
                geos_ens = x_geos_flat.view(4, 4, H, W)
                geos_mean_norm = geos_ens.mean(dim=0) # (4, H, W)
                
                # Denormalize GEOS
                if val_dataset.geos_mean is not None:
                    # Need stats on device
                     gm = val_dataset.geos_mean.to(device)
                     gs = val_dataset.geos_std.to(device)
                     # Handle broadcasting if needed
                     # If gm is (1, 4, H, W) -> squeeze(0) -> (4, H, W) matches
                     # If gm is (1,) -> matches
                     if gm.ndim == 4:
                         gm_sq = gm.squeeze(0)
                         gs_sq = gs.squeeze(0)
                     else:
                         gm_sq = gm
                         gs_sq = gs
                         
                     geos_mean_all = (geos_mean_norm * gs_sq * 3.0) + gm_sq
                else:
                    geos_mean_all = geos_mean_norm


                for lead_idx in range(4):
                    # Data for plotting
                    g_img = geos_mean_all[lead_idx].cpu().numpy().squeeze()
                    t_img = target_all[lead_idx].cpu().numpy().squeeze()
                    d_img = diff_mean_all[lead_idx].cpu().numpy().squeeze()
                    
                    diff_map = d_img - t_img
                    geos_diff_map = g_img - t_img
                    
                    geos_rmse = np.sqrt(np.mean((g_img - t_img)**2))
                    diff_rmse = np.sqrt(np.mean((d_img - t_img)**2))
                    
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
