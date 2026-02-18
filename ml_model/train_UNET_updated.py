"""
UNet-Based Precipitation Prediction with Zero-Inflated Log-Normal (ZILN) Distribution

Architecture: TemporalAttentionUNet from model_unet.py
Output: 12 channels (3 ZILN params x 4 lead weeks) -> p, mu, sigma per grid cell
Loss: CRPS (Continuous Ranked Probability Score) for ZILN distribution, area-weighted
Target: Raw GPCP precipitation (mm/day), no normalization

The ZILN parameterization handles precipitation's extreme dynamic range:
  - p (rain probability):  sigmoid(raw_p)  -> [0, 1]
  - mu (log-space mean):   raw_mu          -> unbounded
  - sigma (log-space std): softplus(raw_s) + 1e-4 -> positive

Expected Value: E[Rain] = p * exp(mu + sigma^2/2)
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
from ml_model.model_unet import TemporalAttentionUNet


# ==============================================================================
# ZILN PARAMETERIZATION & CRPS LOSS
# ==============================================================================

def parameterize_ziln(raw_output):
    """
    Convert raw UNet output (B, 12, H, W) to ZILN parameters.
    
    Args:
        raw_output: (B, 12, H, W) - Raw UNet output
        
    Returns:
        p:     (B, 4, H, W) - Rain probability [0, 1]
        mu:    (B, 4, H, W) - Log-space mean (unbounded)
        sigma: (B, 4, H, W) - Log-space std (positive)
    """
    B, C, H, W = raw_output.shape
    # Reshape: (B, 12, H, W) -> (B, 4, 3, H, W)
    params = raw_output.view(B, 4, 3, H, W)
    
    raw_p = params[:, :, 0, :, :]     # (B, 4, H, W)
    raw_mu = params[:, :, 1, :, :]    # (B, 4, H, W)
    raw_sigma = params[:, :, 2, :, :] # (B, 4, H, W)
    
    p = torch.sigmoid(raw_p)
    mu = raw_mu
    sigma = F.softplus(raw_sigma) + 1e-4
    
    return p, mu, sigma


def ziln_expected_value(p, mu, sigma):
    """
    Compute E[Rain] = p * exp(mu + sigma^2 / 2)
    
    This is the mean of the ZILN distribution, used for deterministic validation.
    """
    return p * torch.exp(mu + 0.5 * sigma**2)


def crps_ziln_loss(p, mu, sigma, target, area_weights=None):
    """
    CRPS loss for Zero-Inflated Log-Normal distribution.
    
    The CRPS for a mixture: (1-p)*Dirac(0) + p*LogNormal(mu, sigma)
    
    For observation y and ZILN forecast F:
      CRPS = E|X - y| - 0.5 * E|X - X'|
    
    Using the analytical CRPS of log-normal:
      CRPS_LN(y; mu, sigma) = y*(2*Phi(z) - 1) - 2*exp(mu + sigma^2/2) * 
                               (Phi(z - sigma) + Phi(-sigma/sqrt(2)) - 1)
      where z = (ln(y) - mu) / sigma, Phi is the standard normal CDF.
    
    For the zero-inflated case:
      CRPS_ZILN = (1-p)^2 * y                              (if y > 0, dry component)
                + p * CRPS_LN(y; mu, sigma)                 (wet component)
                + p*(1-p)*E_LN                              (cross-term)
                - 0.5*p^2 * (correction for E|X-X'|)
    
    We use a simplified but correct formulation:
    
    Args:
        p:      (B, 4, H, W) - Rain probability [0, 1]
        mu:     (B, 4, H, W) - Log-space mean
        sigma:  (B, 4, H, W) - Log-space std
        target: (B, 4, H, W) - Raw GPCP (mm/day), >= 0
        area_weights: (1, 1, H, 1) - cos(lat) weights
        
    Returns:
        Scalar CRPS loss (lower is better).
    """
    # Numerical stability
    eps = 1e-6
    y = target.clamp(min=0.0)
    
    # Standard normal CDF helper
    Phi = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    # --- Log-Normal CRPS component ---
    # For y > 0: z = (ln(y) - mu) / sigma
    # For y = 0: we handle separately
    
    y_safe = y.clamp(min=eps)  # Avoid log(0)
    z = (torch.log(y_safe) - mu) / sigma
    
    # E_LN = exp(mu + sigma^2/2), the log-normal mean
    E_LN = torch.exp(mu + 0.5 * sigma**2)
    
    # CRPS of LogNormal(mu, sigma) for observation y:
    # crps_ln = y * (2*Phi(z) - 1) - 2*E_LN * (Phi(z - sigma) + Phi(-sigma/sqrt(2)) - 1)
    crps_ln_wet = y_safe * (2 * Phi(z) - 1) \
                  - 2 * E_LN * (Phi(z - sigma) + Phi(-sigma / math.sqrt(2)) - 1)
    
    # For y == 0 (dry observation), the pure log-normal CRPS simplifies:
    # crps_ln_dry = 2*E_LN * Phi(-sigma/sqrt(2)) - E_LN (from sign changes)
    # Actually: crps_ln(0) = -y*(2*Phi(z)-1) becomes 0, and we get:
    # crps_ln(0) = 2*E_LN * (1 - Phi(-sigma/sqrt(2)) - (Phi(z-sigma) at y=0))
    # But for y=0, z -> -inf, Phi(z) -> 0, Phi(z-sigma) -> 0
    crps_ln_dry = 2 * E_LN * (1 - Phi(-sigma / math.sqrt(2)))
    # Simplification: crps_ln_dry = 2*E_LN * Phi(sigma/sqrt(2))
    
    is_wet = (y > eps).float()
    crps_ln = is_wet * crps_ln_wet + (1 - is_wet) * crps_ln_dry
    
    # --- Zero-Inflated CRPS ---
    # Full ZILN CRPS decomposition:
    # CRPS = (1-p)*|y| + p*crps_ln - p*(1-p)*E_LN + 0.5*p^2 * (...)
    # 
    # Following Scheuerer & Hamill (2015) style:
    # CRPS_ZILN(y) = (1-p)^2 * y * is_wet        (penalty: predicted dry but was wet)
    #              + p * crps_ln                   (log-normal CRPS contribution)
    #              + p*(1-p) * E_LN               (cross-component variance penalty)
    #              - 0.5 * p^2 * E_LN * (2*Phi(sigma/sqrt(2)) - 1)  (self-spread)
    
    # Spread of log-normal: E|X-X'| for LogNormal
    # For two iid LogNormal(mu,sigma): E|X-X'| = 2*E_LN*(2*Phi(sigma/sqrt(2)) - 1)
    ln_spread = 2 * E_LN * (2 * Phi(sigma / math.sqrt(2)) - 1)
    
    crps = (1 - p)**2 * y \
         + p * crps_ln \
         + p * (1 - p) * E_LN \
         - 0.5 * p**2 * ln_spread
    
    # Area weighting
    if area_weights is not None:
        crps = crps * area_weights
    
    return crps.mean()


def get_area_weights(lats, device):
    """
    Calculates area weights based on cosine of latitude.
    Normalizes weights to have a mean of 1.
    """
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    # Shape: (1, 1, H, 1) for broadcasting with (B, 4, H, W)
    weights_tensor = torch.from_numpy(weights).float().to(device)
    return weights_tensor.view(1, 1, -1, 1)


# ==============================================================================
# TRAIN
# ==============================================================================

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_unet.yaml", help="Path to config file")
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
    
    # Model: TemporalAttentionUNet
    # Input:  34 channels (16 Obs + 16 GEOS flat + 2 Month sin/cos)
    # Output: 12 channels (3 ZILN params x 4 leads: p, mu, sigma)
    in_channels = 34
    out_channels = 12   # 3 params * 4 leads

    model = TemporalAttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=128,
        emb_dim=256,
        n_weeks=4,
        temporal_heads=4
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
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # Output Dir
    os.makedirs(config["output_dir"], exist_ok=True)
    log_file = os.path.join(config["output_dir"], "training_log_unet.csv")
    if accelerator.is_main_process:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_CRPS", "Val_CRPS", "Val_RMSE"])

    # Load Checkpoint?
    start_epoch = 0
    latest_ckpt = os.path.join(config["output_dir"], "latest_unet_ckpt.pt")
    
    # Top K Checkpoints
    top_k_ckpts = []  # List of (rmse, epoch, path)
    save_top_k = config.get("save_top_k", 4)
    
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'top_k_ckpts' in checkpoint:
            top_k_ckpts = checkpoint['top_k_ckpts']
    best_val_rmse = float('inf')
    if top_k_ckpts:
        best_val_rmse = top_k_ckpts[0][0]
        print(f"Resumed Best Val RMSE: {best_val_rmse:.4f}")

    # Print Model Info
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TemporalAttentionUNet Parameters: {n_params:,}")
        print(f"Input Channels: {in_channels}, Output Channels: {out_channels}")
        print(f"ZILN Parameterization: 3 params (p, mu, sigma) x 4 leads = 12 output channels")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0.0
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Data Info:
            # x_obs: (B, 16, H, W) [Stacked Obs]
            # x_geos: (B, 4, 1, 4, H, W) [Members, 1, Leads, H, W]
            # y_target: (B, 4, H, W) [4 Leads] - RAW GPCP (mm/day)
            
            x_obs = batch['x_obs']
            x_geos = batch['x_geos']
            y_target = batch['y_target']
            months = batch['month']  # (B,)
            
            B, _, H, W = x_obs.shape
            
            # GEOS: (B, 4, 1, 4, H, W) -> (B, 16, H, W)
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
            
            # Target: RAW GPCP (mm/day), clamp >= 0
            y_target = y_target.clamp(min=0.0)
            
            # Month Embeddings (Seasonality)
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            
            # Model Input: (B, 34, H, W)
            x_input = torch.cat([x_obs, x_geos_flat, sin_month, cos_month], dim=1)
            
            # Month one-hot for FiLM conditioning
            month_onehot = F.one_hot(months.long() - 1, num_classes=12).float().to(device)
            
            # Forward
            raw_output = model(x_input, month_onehot)  # (B, 12, H, W)
            
            # ZILN Parameterization
            p, mu, sigma = parameterize_ziln(raw_output)
            
            # CRPS Loss
            loss = crps_ziln_loss(p, mu, sigma, y_target, area_weights=area_weights)
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            pbar.set_postfix({"crps": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_crps_sum = 0
        val_count = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                vx_obs = val_batch['x_obs']
                vx_geos = val_batch['x_geos']
                vy_target = val_batch['y_target'].clamp(min=0.0)
                v_months = val_batch['month']
                
                vB, _, vH, vW = vx_obs.shape
                
                # Reshape GEOS
                vx_geos_flat = vx_geos.squeeze(2).reshape(vB, 16, vH, vW)

                # Month Embeddings
                v_sin_month = torch.sin(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_cos_month = torch.cos(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                
                # Model Input
                v_input = torch.cat([vx_obs, vx_geos_flat, v_sin_month, v_cos_month], dim=1)
                v_month_onehot = F.one_hot(v_months.long() - 1, num_classes=12).float().to(device)
                
                # Forward
                v_raw_output = model(v_input, v_month_onehot)
                v_p, v_mu, v_sigma = parameterize_ziln(v_raw_output)
                
                # Validation CRPS
                v_crps = crps_ziln_loss(v_p, v_mu, v_sigma, vy_target, area_weights=area_weights)
                val_crps_sum += v_crps.item()
                val_count += 1

        avg_val_crps = val_crps_sum / val_count if val_count > 0 else 0
        
        # --- FIXED BATCH VISUALIZATION & RMSE ---
        fb_obs = fixed_val_batch['x_obs'].to(device)
        fb_geos = fixed_val_batch['x_geos'].to(device)
        fb_target = fixed_val_batch['y_target'].to(device).clamp(min=0.0)
        fb_months = fixed_val_batch['month'].to(device)
        
        fb_B = fb_obs.shape[0]
        _, _, H, W = fb_obs.shape
        fb_geos_flat = fb_geos.squeeze(2).reshape(fb_B, 16, H, W)
        
        # Month
        fb_sin_month = torch.sin(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_cos_month = torch.cos(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        
        fb_input = torch.cat([fb_obs, fb_geos_flat, fb_sin_month, fb_cos_month], dim=1)
        fb_month_onehot = F.one_hot(fb_months.long() - 1, num_classes=12).float().to(device)
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        with torch.no_grad():
            fb_raw = unwrapped_model(fb_input, fb_month_onehot)  # (B, 12, H, W)
            fb_p, fb_mu, fb_sigma = parameterize_ziln(fb_raw)
            
            # Expected value: E[Rain] = p * exp(mu + sigma^2/2)
            fb_pred = ziln_expected_value(fb_p, fb_mu, fb_sigma)  # (B, 4, H, W)
        
        # Compute RMSE on Fixed Batch (All Leads)
        val_rmse = torch.sqrt(torch.mean((fb_pred - fb_target)**2)).item()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | CRPS: {avg_train_loss:.4f} | Val CRPS: {avg_val_crps:.4f} | Val RMSE (Fixed): {val_rmse:.4f}")
            
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, avg_val_crps, val_rmse])
            
            if val_rmse < best_val_rmse:
                print(f"New Best Model Found! RMSE improved from {best_val_rmse:.4f} to {val_rmse:.4f}. Plotting...")
                best_val_rmse = val_rmse
                
                # Plot First Sample, All 4 Leads
                s_img_all = fb_pred[0].cpu().numpy()    # (4, H, W)
                t_img_all = fb_target[0].cpu().numpy()  # (4, H, W)
                p_img_all = fb_p[0].cpu().numpy()       # (4, H, W) rain prob
                
                # GEOS Mean (4 leads) - Denormalize
                g_flat = fb_geos_flat[0]  # (16, H, W)
                g_ens = g_flat.view(4, 4, H, W)  # (Members, Leads, H, W)
                g_mean_norm = g_ens.mean(dim=0)   # (Leads, H, W) -> (4, H, W)
                
                # Denormalize GEOS Mean
                if train_dataset.geos_mean is not None:
                    gm_cpu = train_dataset.geos_mean.squeeze().cpu().numpy()
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
                    
                    if l_idx == 0: axes[l_idx, 2].set_title("UNet E[Rain]")
                    axes[l_idx, 2].imshow(s_img, cmap='Blues', vmin=0, vmax=50)
                    axes[l_idx, 2].set_ylabel(f"Week {l_idx+1}\nRMSE: {rmse_l:.2f}")
                    
                    if l_idx == 0: axes[l_idx, 3].set_title("UNet Bias")
                    axes[l_idx, 3].imshow(diff_img, cmap='RdBu_r', vmin=-20, vmax=20)
                    
                    if l_idx == 0: axes[l_idx, 4].set_title("GEOS Bias")
                    axes[l_idx, 4].imshow(geos_bias, cmap='RdBu_r', vmin=-20, vmax=20)

                os.makedirs(os.path.join(config["output_dir"], "plots_unet"), exist_ok=True)
                plt.suptitle(f"UNet ZILN - Epoch {epoch} | RMSE: {val_rmse:.2f} | CRPS: {avg_val_crps:.4f}", fontsize=14)
                plt.savefig(os.path.join(config["output_dir"], f"plots_unet/epoch_{epoch}_rmse_{val_rmse:.2f}.png"))
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
            current_path = os.path.join(config["output_dir"], f"unet_epoch_{epoch}_rmse_{val_rmse:.4f}.pt")
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


# ==============================================================================
# TEST
# ==============================================================================

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_unet.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Validation Dataset
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Model
    model = TemporalAttentionUNet(
        in_channels=34,
        out_channels=12,
        base_filters=128,
        emb_dim=256,
        n_weeks=4,
        temporal_heads=4
    )

    # Load Best or Latest Model
    latest_ckpt = os.path.join(config["output_dir"], "latest_unet_ckpt.pt")
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        top_k = checkpoint.get('top_k_ckpts', [])
        
        if top_k:
            best_ckpt_path = top_k[0][2]
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

    # Indices to test
    test_indices = [0, 10, 20, 30, 40]
    output_dir = os.path.join(config["output_dir"], "plots_test_suite")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running Test Suite on indices {test_indices}...")

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
                y_target = batch['y_target'].to(device).clamp(min=0.0)
                t_months = batch['month'].to(device)
                
                B = x_obs.shape[0]
                _, _, H, W = x_obs.shape
                
                # Reshape GEOS
                x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
                
                # Month Embeddings
                t_sin_month = torch.sin(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
                t_cos_month = torch.cos(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
                
                x_input = torch.cat([x_obs, x_geos_flat, t_sin_month, t_cos_month], dim=1)
                month_onehot = F.one_hot(t_months.long() - 1, num_classes=12).float().to(device)
                
                # Forward
                unwrapped_model = accelerator.unwrap_model(model)
                raw_output = unwrapped_model(x_input, month_onehot)
                pred_p, pred_mu, pred_sigma = parameterize_ziln(raw_output)
                pred_mean = ziln_expected_value(pred_p, pred_mu, pred_sigma)  # (1, 4, H, W)
                
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

                fig = plt.figure(figsize=(25, 20))
                
                # Prepare data
                target_all = y_target.squeeze(0)  # (4, H, W)
                pred_all = pred_mean.squeeze(0)    # (4, H, W)
                
                # GEOS Mean Denormalized
                geos_ens = x_geos_flat.view(4, 4, H, W)
                geos_mean_norm = geos_ens.mean(dim=0)  # (4, H, W)
                
                if val_dataset.geos_mean is not None:
                    gm = val_dataset.geos_mean.to(device)
                    gs = val_dataset.geos_std.to(device)
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
                    g_img = geos_mean_all[lead_idx].cpu().numpy().squeeze()
                    t_img = target_all[lead_idx].cpu().numpy().squeeze()
                    d_img = pred_all[lead_idx].cpu().numpy().squeeze()
                    
                    diff_map = d_img - t_img
                    geos_diff_map = g_img - t_img
                    
                    geos_rmse = np.sqrt(np.mean((g_img - t_img)**2))
                    unet_rmse = np.sqrt(np.mean((d_img - t_img)**2))
                    
                    im0 = plot_panel(fig, lead_idx, 0, g_img, f"W{lead_idx+1}: GEOS Ens Mean\nRMSE: {geos_rmse:.2f}", 'Blues', 0, 50)
                    im1 = plot_panel(fig, lead_idx, 1, t_img, f"W{lead_idx+1}: Target GPCP", 'Blues', 0, 50)
                    im2 = plot_panel(fig, lead_idx, 2, d_img, f"W{lead_idx+1}: UNet E[Rain]\nRMSE: {unet_rmse:.2f}", 'Blues', 0, 50)
                    im3 = plot_panel(fig, lead_idx, 3, diff_map, f"W{lead_idx+1}: UNet Bias", 'RdBu_r', -20, 20)
                    im4 = plot_panel(fig, lead_idx, 4, geos_diff_map, f"W{lead_idx+1}: GEOS Bias", 'RdBu_r', -20, 20)
                    
                    if lead_idx == 0:
                        cax1 = fig.add_axes([0.92, 0.6, 0.015, 0.25])
                        fig.colorbar(im0, cax=cax1, label='mm/day')
                        cax2 = fig.add_axes([0.92, 0.15, 0.015, 0.25])
                        fig.colorbar(im3, cax=cax2, label='mm/day')

                plt.suptitle(f"UNet ZILN - Sample {current_idx} (Val Set) - All Lead Weeks", fontsize=16)
                plt.savefig(os.path.join(output_dir, f"test_sample_{current_idx}_all_leads.png"), bbox_inches='tight', dpi=150)
                plt.close()
                print(f"Saved multi-lead plot for sample {current_idx}.")
                
                samples_processed += 1
            
            current_idx += 1
                
    print("Test Suite Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_unet.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        train()
