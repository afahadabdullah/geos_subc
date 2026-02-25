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
from dataset_flow import S2SHybridDataset
from flow_matching import FlowMatchingModel, CustomFlowMatcher

def get_area_weights(lats, device):
    lats_rad = np.deg2rad(lats)
    weights = np.cos(lats_rad)
    weights = weights / np.mean(weights)
    weights_tensor = torch.from_numpy(weights).float().to(device)
    weights_tensor = weights_tensor.view(1, 1, len(lats), 1)
    return weights_tensor

def compute_crps(ensemble_preds, target, area_weights):
    """
    Computes CRPS for a small ensemble.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    area_weights: [1, 1, H, 1]
    """
    # Mask NaNs to avoid CRPS:nan
    # ensemble_preds: [E, B, C, H, W], target: [B, C, H, W]
    mask = ~torch.isnan(target) # [B, C, H, W]
    if not mask.any():
        return 0.0
    
    # 1. Mean Absolute Error Term: 1/E sum |x_i - y|
    E = ensemble_preds.shape[0]
    diff = torch.abs(ensemble_preds - target.unsqueeze(0)) # [E, B, C, H, W]
    mae_term = diff.mean(dim=0) # [B, C, H, W]
    
    # 2. Ensemble Spread Term: 1/(2E^2) sum sum |x_i - x_j|
    spread_term = torch.zeros_like(mae_term)
    if E > 1:
        for i in range(E):
            for j in range(E):
                spread_term += torch.abs(ensemble_preds[i] - ensemble_preds[j])
        spread_term = spread_term / (2 * E * E)
    
    crps_map = mae_term - spread_term
    # Zero out NaNs in map for weighted mean
    crps_map_clean = torch.where(mask, crps_map, torch.zeros_like(crps_map))
    weights_clean = torch.where(mask, area_weights, torch.zeros_like(area_weights))
    
    weighted_crps = (crps_map_clean * weights_clean).sum() / (weights_clean.sum() + 1e-8)
    return weighted_crps.item()

def save_val_plot(epoch, full_pred, true_target_precip, model_crps, model_rmse, geos_pred, geos_crps, geos_rmse, output_dir, ai_residual=None, suffix=""):
    """
    Standardizes plotting logic for validation results (5-column layout).
    """
    t_img = true_target_precip[0].cpu().numpy()
    p_img = full_pred[0].cpu().numpy()
    g_img = geos_pred[0].cpu().numpy()
    res_img = ai_residual[0].cpu().numpy() if ai_residual is not None else None
    
    fig, axes = plt.subplots(4, 5, figsize=(25, 16))
    for l in range(4):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        
        # Col 1: Target
        im0 = axes[l, 0].imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im0, ax=axes[l, 0], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 0].set_title("Target GPCP")
        
        # Col 2: Model Pred
        im1 = axes[l, 1].imshow(p_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im1, ax=axes[l, 1], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 1].set_title("Model Pred (Mean)")
        
        # Col 3: Model Diff
        diff_model = p_img[l] - t_img[l]
        im2 = axes[l, 2].imshow(diff_model, cmap='RdBu_r', vmin=-50, vmax=50)
        fig.colorbar(im2, ax=axes[l, 2], fraction=0.046, pad=0.04)
        if l == 0: 
            axes[l, 2].set_title(f"Model Diff (Skill vs GEOS)\nCRPS:{model_crps:.2f}, RMSE:{model_rmse:.2f}")
        
        # Col 4: GEOS Diff
        diff_geos = g_img[l] - t_img[l]
        im3 = axes[l, 3].imshow(diff_geos, cmap='RdBu_r', vmin=-50, vmax=50)
        fig.colorbar(im3, ax=axes[l, 3], fraction=0.046, pad=0.04)
        if l == 0:
            axes[l, 3].set_title(f"GEOS Baseline Diff\nCRPS:{geos_crps:.2f}, RMSE:{geos_rmse:.2f}")

        # Col 5: AI Predicted Residual (Innovation)
        if res_img is not None:
            im4 = axes[l, 4].imshow(res_img[l], cmap='RdBu_r', vmin=-30, vmax=30)
            fig.colorbar(im4, ax=axes[l, 4], fraction=0.046, pad=0.04)
            if l == 0: axes[l, 4].set_title("AI Predicted Residual\n(Model - GEOS)")
        else:
            axes[l, 4].axis('off')
    
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.tight_layout()
    filename = f"epoch_{epoch}_{suffix}_score_{model_crps:.4f}.png" if suffix else f"epoch_{epoch}_score_{model_crps:.4f}.png"
    plt.savefig(os.path.join(output_dir, "plots", filename))
    plt.close()

@torch.no_grad()
def run_val_inference(epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                      target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, is_test=False, is_fast_recon=True):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Take the first batch (BatchSize=4 expected). These represent leads 1, 2, 3, 4 of the first val date.
    batch = next(iter(val_loader))
    fb_target_norm = batch['y_target'].to(device) # [4, 1, H, W]
    vB, _, H, W = fb_target_norm.shape
    
    # Prepare Ground Truth Precip (4, H, W)
    # We use 'target_raw_full' which is (4, H, W) - redundant across batch members but reliable
    true_target_precip = batch['target_raw_full'][0].to(device) # [4, H, W]
    
    # Prepare GEOS Baseline (4, H, W)
    geos_ens_raw = batch['geos_ens_raw'].to(device) # [4, M=4, L=4, H, W]
    # Members for the first sample in batch
    geos_ens_sample = geos_ens_raw[0] # [M=4, L=4, H, W]
    geos_mean_raw = geos_ens_sample.mean(dim=0) # [4, H, W]
    
    # Prepare 4-week prediction buffer
    pred_res_norm_agg = torch.zeros((4, H, W), device=device)
    
    # We collect ensemble members for CRPS if fast_recon
    num_ensemble = 10 if is_fast_recon and not is_test else (5 if is_test else 1)
    ensemble_preds_precip = [] # Will be [E, 4, H, W]

    # Progress bar for internal status during long samplings
    ens_pbar = tqdm(range(num_ensemble), desc="  [Inference Ensemble]", disable=not accelerator.is_main_process, leave=False)
    for eidx in ens_pbar:
        sample_weeks = []
        for lead_idx in range(4):
            # Extract lead-specific conditioning from the batch
            # Index 0-3 in the batch correspond to leads 0-3
            fx_obs = batch['x_obs'][lead_idx].unsqueeze(0).to(device)  # [1, 24, H, W]
            fx_geos = batch['x_geos'][lead_idx].to(device)              # [1, 1, 4, H, W]
            fx_geos_flat = fx_geos.view(1, -1, H, W)                    # [1, 4, H, W]
            
            f_month = batch['month'][lead_idx].to(device).view(1)
            fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(1, 1, 1, 1).expand(1, 1, H, W)
            fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(1, 1, 1, 1).expand(1, 1, H, W)
            
            # Lead Embedding (Match training logic)
            fl_idx = batch['lead_idx'][lead_idx].to(device).view(1)
            f_lead_val = (fl_idx.float() / 1.5) - 1.0 # Scale [0, 3] to [-1, 1]
            f_lead_channel = f_lead_val.view(1, 1, 1, 1).expand(1, 1, H, W)
            
            fx_cat_geos = fx_geos.view(1, -1, H, W)             # [1, 4, H, W]
            
            fx_cond = torch.cat([fx_obs, fx_cat_geos, fsin_month, fcos_month, f_lead_channel], dim=1) # [1, 31, H, W]
            
            target_norm_week = fb_target_norm[lead_idx].unsqueeze(0) # [1, 1, H, W]
            
            if is_fast_recon and not is_test:
                num_steps = 10
            else:
                num_steps = 50
                
            noise = torch.randn((1, 1, H, W), device=device)
            p_x1 = flow_matcher.euler_solve(unwrapped_model, noise, fx_cond, num_steps=num_steps)
            week_pred_norm = p_x1.squeeze(0) # [1, H, W]
            
            # Convert direct GPCP back to physical units
            # Mapping: Y = 2.0 * (sqrt_val - s_min) / (s_max - s_min) - 1.0 => sqrt_val = (Y + 1.0)/2.0 * (s_max - s_min) + s_min
            week_sqrt = ((week_pred_norm + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
            # Square to get precipitation
            week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
            sample_weeks.append(week_precip)
            
        ensemble_preds_precip.append(torch.stack(sample_weeks)) # [4, H, W]

    ensemble_preds_precip = torch.stack(ensemble_preds_precip) # [E, 4, H, W]
    full_pred = ensemble_preds_precip.mean(dim=0) # [4, H, W]
    
    # Metric calculations with NaN handling
    # Calculate CRPS
    val_metric = compute_crps(ensemble_preds_precip.unsqueeze(1), true_target_precip.unsqueeze(0), area_weights)
    
    # Model RMSE (NaN-aware)
    mse_map = (full_pred - true_target_precip)**2
    mask = ~torch.isnan(mse_map)
    if mask.any():
        # area_weights is [1, 1, 181, 1], mse_map is [4, 181, 360]
        # We need to provide a 3D view of area_weights to match mse_map for expand_as
        aw_expanded = area_weights.view(1, 181, 1).expand_as(mse_map)
        model_rmse = torch.sqrt((mse_map[mask] * aw_expanded[mask]).sum() / (aw_expanded[mask].sum() + 1e-8)).item()
    else:
        model_rmse = 0.0
    
    # GEOS Baseline Metrics (NaN-aware)
    geos_crps = compute_crps(geos_ens_sample.unsqueeze(1), true_target_precip.unsqueeze(0), area_weights)
    geos_mse_map = (geos_mean_raw - true_target_precip)**2
    
    # We must use a 2D mask for Geos (since geos_mean is 2D and target is 2D)
    # The mask from above is 3D because full_pred has the ensemble dimension [4, H, W]
    mask_2d = ~torch.isnan(geos_mse_map)

    if mask_2d.any():
        aw_expanded_2d = area_weights.view(181, 1).expand_as(geos_mse_map)
        geos_rmse = torch.sqrt((geos_mse_map[mask_2d] * aw_expanded_2d[mask_2d]).sum() / (aw_expanded_2d[mask_2d].sum() + 1e-8)).item()
    else:
        geos_rmse = 0.0
    
    recon_type = f"SingleLead-Ensemble (n={num_ensemble})"
    if accelerator.is_main_process:
        print(f"Epoch {epoch} | Val CRPS [{recon_type}]: {val_metric:.4f}")
        
    ai_residual = full_pred - geos_mean_raw
    
    # Clean up true_target_precip for save_val_plot (replace NaNs with 0 for imshow)
    true_target_precip_plot = torch.nan_to_num(true_target_precip, nan=0.0)
    
    return val_metric, full_pred.unsqueeze(0), true_target_precip_plot.unsqueeze(0), model_rmse, geos_mean_raw.unsqueeze(0), geos_crps, geos_rmse, ai_residual.unsqueeze(0)

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
        preload=config.get("preload", False),
        stats_file="v5_global_stats.pt"
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
            preload=config.get("preload", False),
            stats_file="v5_global_stats.pt"
        )
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=config.get("num_workers", 4), pin_memory=True
        )

    # Calculate Global Min-Max for Target GPCP Precipitation
    stats_file = "ml_model/v5_global_stats.pt"
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"CRITICAL: {stats_file} missing. Please run calculate_global_stats_v5.py first!")
    
    global_bounds = torch.load(stats_file, weights_only=True)
    # Force robust physical range for Direct Power Transformed GPCP (sqrt)
    # Target raw max is roughly 50 mm/day max in GPCP weekly. sqrt(50) ~= 7.071
    target_sqrt_min = 0.0
    target_sqrt_max = 7.071
    
    geos_min = global_bounds["geos_raw"]["min"]
    geos_max = global_bounds["geos_raw"]["max"]
    
    if accelerator.is_main_process:
        print("\n=======================================================")
        print(f"✅ Loaded Strict Global Stats: {stats_file}")
        print(f"   [Target SQRT Bounds] : Min = {target_sqrt_min:.4f}, Max = {target_sqrt_max:.4f}")
        print(f"   [GEOS Raw Bounds]    : Min = {geos_min:.4f}, Max = {geos_max:.4f}")
        print("=======================================================\n")

    # ---------------------------------------------------------
    # 2. Model & Scheduler Setup
    # ---------------------------------------------------------
    model = FlowMatchingModel(in_channels=32, out_channels=1).to(device)
    flow_matcher = CustomFlowMatcher(device=device)
    
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
    output_dir = config.get("output_dir", "ml_output_diffusion_v5")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log_v5.csv")
    
    if accelerator.is_main_process and not os.path.exists(log_file):
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Noise", "Val_CRPS"])

    # Fixed Val Batch for continuous plotting
    fixed_val_batch = next(iter(val_loader))

    start_epoch = 0
    best_val_crps = float('inf')
    top_models = [] # List of {"path": str, "crps": float, "epoch": int}
    
    # Load latest checkpoint if it exists
    if args.test:
        ckpt_path = os.path.join(output_dir, args.ckpt)
    else:
        ckpt_path = os.path.join(output_dir, "latest_diffusion_ckpt_v5.pt")

    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            # Unwrap for loading
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(checkpoint['model'])
            
            if not args.test:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                if 'best_val_crps' in checkpoint:
                    best_val_crps = checkpoint['best_val_crps']
                elif 'best_val_rmse' in checkpoint:
                    best_val_crps = checkpoint['best_val_rmse'] # Migration fallback
                
                if 'top_models' in checkpoint:
                    top_models = checkpoint['top_models']
                
            if accelerator.is_main_process:
                print(f"\n🔄 Loaded checkpoint: {ckpt_path}")
                if not args.test:
                    print(f"   Starting at Epoch: {start_epoch}")
                    print(f"   Best Val CRPS so far: {best_val_crps:.4f}\n")
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
        
        val_outputs = run_val_inference(
            start_epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
            target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, 
            is_test=True, is_fast_recon=False
        )
        v_met, v_pred, v_target, v_rmse, v_geos_mean, v_geos_crps, v_geos_rmse, v_ai_res = val_outputs
        
        if accelerator.is_main_process:
            save_val_plot(start_epoch, v_pred, v_target, v_met, v_rmse, v_geos_mean, v_geos_crps, v_geos_rmse, output_dir, ai_residual=v_ai_res, suffix="test_mode")
        return

    # ---------------------------------------------------------
    # Pre-Training Diagnostics (Raw vs Normalized Bounds)
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        print("\n--- PRE-TRAINING VISUAL DIAGNOSTICS (Normalization Check) ---")
        # Fetch one batch for diagnostics
        batch = fixed_val_batch
        x_geos = batch['x_geos'].to(device)
        x_obs = batch['x_obs'].to(device)
        target_norm = batch['y_target'].to(device)
        lead_idx_tensor = batch['lead_idx'].to(device)
        
        # Take index 0, lead 1 for visualization
        sample_idx = 0
        lead_idx = 0
        
        # 1. Reverse Normalize GEOS
        geos_norm_sample = x_geos[sample_idx, 0, 0, lead_idx].cpu().numpy()
        geos_raw_sample = ((geos_norm_sample + 1.0) / 2.0) * (geos_max - geos_min) + geos_min
        
        # 2. Reverse Normalize SST (Channel 0 of x_obs)
        sst_norm_sample = x_obs[sample_idx, 0].cpu().numpy()
        sst_raw_sample = ((sst_norm_sample + 1.0) / 2.0) * (global_bounds["sst"]["max"] - global_bounds["sst"]["min"]) + global_bounds["sst"]["min"]
        
        # 3. Reverse Normalize Target SQRT (Lead 0)
        sqrt_norm_sample = target_norm[sample_idx, lead_idx].cpu().numpy()
        sqrt_raw_sample = ((sqrt_norm_sample + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        res_raw_sample = np.square(sqrt_raw_sample) - geos_raw_sample

        # 4. Reverse Normalize Observational States
        # SST(0-3), SSS(4-7), SM(8-11), IVT(12-15), ZDEV(16-19), U250(20-23)
        sss_norm_sample = x_obs[sample_idx, 4].cpu().numpy()
        sss_raw_sample = ((sss_norm_sample + 1.0) / 2.0) * (global_bounds["sss"]["max"] - global_bounds["sss"]["min"]) + global_bounds["sss"]["min"]
        
        sm_norm_sample = x_obs[sample_idx, 8].cpu().numpy()
        sm_raw_sample = ((sm_norm_sample + 1.0) / 2.0) * (global_bounds["sm"]["max"] - global_bounds["sm"]["min"]) + global_bounds["sm"]["min"]
        
        ivt_norm_sample = x_obs[sample_idx, 12].cpu().numpy()
        ivt_raw_sample = ((ivt_norm_sample + 1.0) / 2.0) * (global_bounds["ivt"]["max"] - global_bounds["ivt"]["min"]) + global_bounds["ivt"]["min"]
        
        zdev_norm_sample = x_obs[sample_idx, 16].cpu().numpy()
        zdev_raw_sample = ((zdev_norm_sample + 1.0) / 2.0) * (global_bounds["z500_zonal_dev"]["max"] - global_bounds["z500_zonal_dev"]["min"]) + global_bounds["z500_zonal_dev"]["min"]
        
        u250_norm_sample = x_obs[sample_idx, 20].cpu().numpy()
        u250_raw_sample = ((u250_norm_sample + 1.0) / 2.0) * (global_bounds["u250"]["max"] - global_bounds["u250"]["min"]) + global_bounds["u250"]["min"]

        # 5. Raw GPCP (from dataset)
        gpcp_raw_sample = batch['target_raw'][sample_idx, lead_idx].cpu().numpy()

        fig, axes = plt.subplots(10, 2, figsize=(14, 40))
        # Row 1-8 logic remains same...
        im1 = axes[0, 0].imshow(geos_raw_sample, cmap='Blues')
        axes[0, 0].set_title(f"Raw GEOS (Lead {lead_idx+1})")
        fig.colorbar(im1, ax=axes[0, 0])
        im2 = axes[0, 1].imshow(geos_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 1].set_title("Normalized GEOS [-1, 1]")
        fig.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[1, 0].imshow(np.square(sqrt_raw_sample), cmap='Blues')
        axes[1, 0].set_title("Reconstructed GPCP (Un-SQRT)")
        fig.colorbar(im3, ax=axes[1, 0])
        im4 = axes[1, 1].imshow(sqrt_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 1].set_title("Normalized SQRT Target [-1, 1]")
        fig.colorbar(im4, ax=axes[1, 1])
        
        axes[2, 0].imshow(sst_raw_sample, cmap='viridis')
        axes[2, 0].set_title("Raw SST")
        axes[2, 1].imshow(sst_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, 1].set_title("Normalized SST [-1, 1]")
        
        axes[3, 0].imshow(sss_raw_sample, cmap='YlGnBu')
        axes[3, 0].set_title("Raw SSS")
        axes[3, 1].imshow(sss_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[3, 1].set_title("Normalized SSS [-1, 1]")
        
        axes[4, 0].imshow(sm_raw_sample, cmap='YlOrBr')
        axes[4, 0].set_title("Raw SM")
        axes[4, 1].imshow(sm_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[4, 1].set_title("Normalized SM [-1, 1]")
        
        axes[5, 0].imshow(ivt_raw_sample, cmap='cubehelix')
        axes[5, 1].imshow(ivt_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        
        im13 = axes[6, 0].imshow(zdev_raw_sample, cmap='RdBu_r', vmin=-3000, vmax=3000)
        axes[6, 0].set_title("Raw Z500 Zonal Dev")
        im14 = axes[6, 1].imshow(zdev_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[6, 1].set_title("Normalized Z500 Zonal Dev [-1, 1]")
        
        axes[7, 0].imshow(u250_raw_sample, cmap='coolwarm')
        axes[7, 1].imshow(u250_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)

        im17 = axes[8, 0].imshow(gpcp_raw_sample, cmap='Blues')
        axes[8, 0].set_title("Raw GPCP (Pure Target)")
        axes[8, 1].text(0.5, 0.5, "GPCP is not\nnormalized directly.", ha='center', va='center', transform=axes[8, 1].transAxes)
        axes[8, 1].axis('off')

        flat_geos = geos_raw_sample.flatten()
        flat_gpcp = gpcp_raw_sample.flatten()
        cc = np.corrcoef(flat_geos, flat_gpcp)[0, 1] if np.std(flat_geos) > 1e-6 else 0.0
        
        axes[9, 0].imshow(geos_raw_sample - gpcp_raw_sample, cmap='RdBu_r', vmin=-20, vmax=20)
        axes[9, 0].set_title(f"Spatial Alignment | CC: {cc:.4f}")
        axes[9, 1].scatter(flat_geos[::50], flat_gpcp[::50], alpha=0.3, s=1)
        
        plt.tight_layout()
        diag_path = os.path.join(output_dir, "normalization_check.png")
        plt.savefig(diag_path)
        plt.close()
        print(f"✅ Normalization diagnostic plot saved to {diag_path}!")
        print(f"---------------------------------------------------\n")
    
    # ---------------------------------------------------------
    # Pre-Flight NaN Integrity Scan (SKIP IF RESUMING)
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        if start_epoch > 0:
            print(f"\n✅ Resuming from Epoch {start_epoch}. Skipping Pre-Flight NaN Scan (already verified).")
        else:
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
    epochs_done_this_run = 0
    max_epochs_this_run = getattr(args, "epochs_per_run", float('inf'))

    for epoch in range(start_epoch, epochs):
        if epochs_done_this_run >= max_epochs_this_run:
            if accelerator.is_main_process:
                print(f"\n⚠️ Reached --epochs-per-run limit ({max_epochs_this_run}). Exiting for resubmission.")
            break
        
        model.train()
        train_loss = 0.0
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch}")

        for i, batch in enumerate(pbar):    
            # Conditionals: [B, 31, H, W]
            x_geos = batch['x_geos'].to(device) # [B, 1, 1, 4, H, W]
            x_obs  = batch['x_obs'].to(device)  # [B, 24, H, W]
            
            B, M, C_extra, L, H, W = x_geos.shape
            # Flatten GEOS weeks 1-4 into channels
            x_geos_flat = x_geos.view(B, -1, H, W) # [B, 4, H, W]
            
            months = batch['month'].to(device)
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)

            # --- Lead Embedding (NEW) ---
            lead_idx = batch['lead_idx'].to(device) # [B]
            # Map [0, 1, 2, 3] to [-1.0, -0.33, 0.33, 1.0]
            lead_val = (lead_idx.float() / 1.5) - 1.0 
            lead_channel = lead_val.view(B, 1, 1, 1).expand(B, 1, H, W)

            # Total: 24 (Obs/Dev) + 4 (GEOS) + 2 (Month) + 1 (Lead) = 31 channels
            x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1) 

            # Targets are already residual normalized [-1, 1] by dataset_hybrid
            target_norm = batch['y_target'].to(device) # [B, 1, H, W]

            if False: # [REDUNDANT: Moved to Pre-Training Section]
                print(f"\n--- DEBUG | Train Batch 0 Exhaustive Diagnostics ---")
                # 1. Observational States
                # SST(0-3), SSS(4-7), SM(8-11), IVT(12-15), ZonalDev(16-19), U250(20-23)
                vars = ["SST", "SSS", "SM", "IVT", "ZDEV", "U250"]
                for idx, vname in enumerate(vars):
                    v_ch = idx * 4
                    v_raw = x_obs[0, v_ch : v_ch + 4].min().item()
                    v_max = x_obs[0, v_ch : v_ch + 4].max().item()
                    print(f"  {vname:<4} Bounds (Norm) : {v_raw:>6.2f} to {v_max:>6.2f}")
                
                print(f"  GEOS Bounds (Norm) : {x_geos.min().item():>6.2f} to {x_geos.max().item():>6.2f}")
                print(f"  Lead Index         : {lead_idx[0].item()} (Val: {lead_val[0].item():.2f})")
                print(f"  Final x_cond shape : {x_cond.shape}")
                print(f"  Final x_cond bounds: {x_cond.min().item():>6.2f} to {x_cond.max().item():>6.2f}")
                print(f"  Target Bounds (Norm): {target_norm.min().item():>6.2f} to {target_norm.max().item():>6.2f}")
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
                
                # 3. Reverse Normalize Target SQRT (Lead 0)
                sqrt_norm_sample = target_norm[sample_idx, lead_idx].cpu().numpy()
                sqrt_raw_sample = ((sqrt_norm_sample + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
                res_raw_sample = np.square(sqrt_raw_sample) - geos_raw_sample

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
                
                # Channel 16: Zonal Deviation (Rossby Waves)
                zdev_norm_sample = x_obs[sample_idx, 16].cpu().numpy()
                zdev_raw_sample = ((zdev_norm_sample + 1.0) / 2.0) * (global_bounds["z500_zonal_dev"]["max"] - global_bounds["z500_zonal_dev"]["min"]) + global_bounds["z500_zonal_dev"]["min"]
                
                # Channel 20: U250
                u250_norm_sample = x_obs[sample_idx, 20].cpu().numpy()
                u250_raw_sample = ((u250_norm_sample + 1.0) / 2.0) * (global_bounds["u250"]["max"] - global_bounds["u250"]["min"]) + global_bounds["u250"]["min"]

                # 5. Raw GPCP (from dataset)
                gpcp_raw_sample = batch['target_raw'][sample_idx, lead_idx].cpu().numpy()

                fig, axes = plt.subplots(10, 2, figsize=(14, 40))
                # Row 1: GEOS
                im1 = axes[0, 0].imshow(geos_raw_sample, cmap='Blues')
                axes[0, 0].set_title(f"Raw GEOS (Lead {lead_idx+1})")
                fig.colorbar(im1, ax=axes[0, 0])
                
                im2 = axes[0, 1].imshow(geos_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[0, 1].set_title("Normalized GEOS [-1, 1]")
                fig.colorbar(im2, ax=axes[0, 1])
                
                # Row 2: Target
                im3 = axes[1, 0].imshow(np.square(sqrt_raw_sample), cmap='Blues')
                axes[1, 0].set_title("Reconstructed GPCP (Un-SQRT)")
                fig.colorbar(im3, ax=axes[1, 0])
                
                im4 = axes[1, 1].imshow(sqrt_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[1, 1].set_title("Normalized SQRT Target [-1, 1]")
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

                # Row 7: Zonal Deviation (Rossby Waves)
                im13 = axes[6, 0].imshow(zdev_raw_sample, cmap='RdBu_r', vmin=-3000, vmax=3000)
                axes[6, 0].set_title("Raw Z500 Zonal Dev")
                fig.colorbar(im13, ax=axes[6, 0])
                
                im14 = axes[6, 1].imshow(zdev_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[6, 1].set_title("Normalized Z500 Zonal Dev [-1, 1]")
                fig.colorbar(im14, ax=axes[6, 1])

                # Row 8: U250
                im15 = axes[7, 0].imshow(u250_raw_sample, cmap='coolwarm')
                axes[7, 0].set_title("Raw U250")
                fig.colorbar(im15, ax=axes[7, 0])
                
                im16 = axes[7, 1].imshow(u250_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
                axes[7, 1].set_title("Normalized U250 [-1, 1]")
                fig.colorbar(im16, ax=axes[7, 1])

                # Row 9: GPCP Absolute (The "Real" Target)
                im17 = axes[8, 0].imshow(gpcp_raw_sample, cmap='Blues')
                axes[8, 0].set_title("Raw GPCP (Pure Target)")
                fig.colorbar(im17, ax=axes[8, 0])
                
                # Plot something neutral for normalized GPCP (since we only normalize residual)
                axes[8, 1].text(0.5, 0.5, "GPCP is not\nnormalized directly.\nWe normalize the\nresidual [GPCP - GEOS].", 
                             ha='center', va='center', transform=axes[8, 1].transAxes)
                axes[8, 1].axis('off')

                # Row 10: GEOS vs GPCP Orientation Check
                # Calculate Spatial Correlation
                flat_geos = geos_raw_sample.flatten()
                flat_gpcp = gpcp_raw_sample.flatten()
                
                # Pearson Correlation
                if np.std(flat_geos) > 1e-6 and np.std(flat_gpcp) > 1e-6:
                    cc = np.corrcoef(flat_geos, flat_gpcp)[0, 1]
                else:
                    cc = 0.0
                
                im18 = axes[9, 0].imshow(geos_raw_sample - gpcp_raw_sample, cmap='RdBu_r', vmin=-20, vmax=20)
                axes[9, 0].set_title(f"Spatial Alignment: GEOS - GPCP\nCorrelation: {cc:.4f}")
                fig.colorbar(im18, ax=axes[9, 0])
                
                # Show a scatter plot for orientation verification
                axes[9, 1].scatter(flat_geos[::50], flat_gpcp[::50], alpha=0.3, s=1)
                axes[9, 1].set_xlabel("GEOS Rainfall")
                axes[9, 1].set_ylabel("GPCP Rainfall")
                axes[9, 1].set_title("Orientation Scatter (Subsampled)")

                plt.tight_layout()
                diag_path = os.path.join(output_dir, "normalization_check.png")
                plt.savefig(diag_path)
                plt.close()
                print(f"✅ Normalization diagnostic plot saved to {diag_path}!")
                print(f"✅ Spatial Orientation Check: Correlation = {cc:.4f}")
                if cc < 0:
                    print(f"🚨 CRITICAL WARNING: Negative spatial correlation detected! Data might be flipped (Lat/Lon).")
                elif cc < 0.2:
                    print(f"⚠️ WARNING: Low spatial correlation detected. Check for shifts or misalignment.")

            # Flow Matching Interpolation
            t = flow_matcher.sample_time_batch(B)
            noise = torch.randn_like(target_norm)
            x_t, v_target = flow_matcher.interpolate(target_norm, noise, t)

            # Predict the velocity
            v_pred = model(x_t, x_cond, t)

            # Loss scaling with spatial priority
            loss = (area_weights * (v_pred - v_target)**2).mean()

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
                'best_val_crps': best_val_crps,
                'top_models': top_models
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_flow_ckpt.pt"))

        if accelerator.is_main_process:
            print(f"\n⌛ Epoch {epoch} complete. Starting Validation (Inference)...")
        
        # Validate every epoch, but only do expensive plotting/sampling on new best or if forced.
        # User requested: Fast validation first. After epoch 6, if new best, do full validation.
        is_plot_epoch = args.full_val
        
        val_outputs = run_val_inference(
            epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
            target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds,
            is_test=is_plot_epoch, 
            is_fast_recon=not is_plot_epoch
        )
        current_val_metric, full_pred, true_target_precip, model_rmse, geos_mean, geos_crps, geos_rmse, current_ai_res = val_outputs
        
        if accelerator.is_main_process:
            # 1. Always plot the results from the first validation pass (usually fast ensemble)
            plot_suffix = "fast" if not is_plot_epoch else "full"
            save_val_plot(epoch, full_pred, true_target_precip, current_val_metric, model_rmse, 
                          geos_mean, geos_crps, geos_rmse, output_dir, ai_residual=current_ai_res, suffix=plot_suffix)

            # 2. Check for Top 4 Model Buffer
            is_in_top4 = False
            worst_top_crps = max([m['crps'] for m in top_models]) if len(top_models) == 4 else float('inf')
            
            if current_val_metric < worst_top_crps:
                print(f"🌟 New Top-4 model found! CRPS: {current_val_metric:.4f}")
                is_in_top4 = True
                
                # Absolute Best Check
                is_new_best = (current_val_metric < best_val_crps)
                if is_new_best:
                    print(f"🏆 NEW ABSOLUTE BEST! Previous Best: {best_val_crps:.4f}")
                    best_val_crps = current_val_metric

                # Trigger high-quality sampling if new BEST (absolute) found after epoch 6
                if is_new_best and epoch > 6 and not is_plot_epoch:
                    print(f"📸 Breakthrough! Triggering high-quality 1000-step sampling for diagnostic plots...")
                    best_sampled_metric, best_sampled_pred, best_target, b_rmse, b_geos_mean, b_geos_crps, b_geos_rmse, b_ai_res = run_val_inference(
                        epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                        target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds,
                        is_test=True, is_fast_recon=False
                    )
                    save_val_plot(epoch, best_sampled_pred, best_target, best_sampled_metric, b_rmse, 
                                  b_geos_mean, b_geos_crps, b_geos_rmse, output_dir, ai_residual=b_ai_res, suffix="BEST_sampled")

                # Manage Top 4 Persistence
                new_best_name = f"best_model_epoch_{epoch}_crps_{current_val_metric:.4f}.pt"
                new_best_path = os.path.join(output_dir, new_best_name)
                
                unwrapped_model = accelerator.unwrap_model(model)
                best_ckpt = {
                    'epoch': epoch,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_crps': best_val_crps,
                    'top_models': top_models
                }
                
                if len(top_models) == 4:
                    # Replace worst
                    worst_model = max(top_models, key=lambda x: x['crps'])
                    if os.path.exists(worst_model['path']):
                        os.remove(worst_model['path'])
                    top_models.remove(worst_model)
                
                # Save new and update list
                torch.save(best_ckpt, new_best_path)
                top_models.append({"path": new_best_path, "crps": current_val_metric, "epoch": epoch})
                top_models.sort(key=lambda x: x['crps']) # Best first
                
                # Keep a symlink or redundant copy for 'best_diffusion_ckpt_v5.pt'
                if is_new_best:
                    torch.save(best_ckpt, os.path.join(output_dir, "best_diffusion_ckpt_v5.pt"))

            # Save Latest Checkpoint & Logs
            ckpt = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_crps': best_val_crps,
                'top_models': top_models
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_diffusion_ckpt_v5.pt"))

            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, 0.0, current_val_metric])
                    
            if is_in_top4:
                top_str = ", ".join([f"E{m['epoch']}({m['crps']:.3f})" for m in top_models])
                print(f"📍 Top 4 Models: [{top_str}]")

        # Track progress for this execution session
        epochs_done_this_run += 1

def main():
    parser = argparse.ArgumentParser(description="Train or Test Diffusion Model V5")
    parser.add_argument("--config", type=str, default="ml_model/config_diffusion_v5.yaml")
    parser.add_argument("--test", action="store_true", help="Run in inference/test mode only")
    parser.add_argument("--ckpt", type=str, default="best_diffusion_ckpt_v5.pt", 
                        help="Checkpoint filename in output_dir to load for testing (default: best_diffusion_ckpt_v5.pt)")
    parser.add_argument("--full-val", action="store_true", help="Force full reverse sampling validation (1000 steps) for all validation epochs.")
    parser.add_argument("--epochs-per-run", type=int, default=10000, 
                        help="Number of epochs to run before exiting gracefully (useful for job chaining)")
    args = parser.parse_args()

    # Load config to get mixed_precision setting
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    mixed_precision = config.get("mixed_precision", "no")

    accelerator = Accelerator(
        split_batches=True,
        mixed_precision=mixed_precision
    )
    train(args, accelerator)

if __name__ == "__main__":
    main()
