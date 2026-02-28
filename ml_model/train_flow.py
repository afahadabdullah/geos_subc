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

def save_val_plot(epoch, full_pred, true_target_precip, model_crps, model_rmse, geos_pred, geos_crps, geos_rmse, output_dir, ai_residual=None, suffix="", geos_single=None, model_single=None, model_var=None):
    """
    Standardizes plotting logic for validation results (7-column layout).
    """
    t_img = true_target_precip[0].cpu().numpy()
    p_img = full_pred[0].cpu().numpy()
    g_img = geos_pred[0].cpu().numpy()
    g_sing_img = geos_single[0].cpu().numpy() if geos_single is not None else g_img
    m_sing_img = model_single[0].cpu().numpy() if model_single is not None else p_img
    m_var_img = model_var[0].cpu().numpy() if model_var is not None else np.zeros_like(p_img)
    
    fig, axes = plt.subplots(4, 7, figsize=(35, 16))
    for l in range(4):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        
        # Col 1: Target
        im0 = axes[l, 0].imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im0, ax=axes[l, 0], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 0].set_title("Target GPCP")
        
        # Col 2: Single GEOS Ens Member
        im1 = axes[l, 1].imshow(g_sing_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im1, ax=axes[l, 1], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 1].set_title("GEOS (Single Ens Member)")
        
        # Col 3: Single Model Ens Member
        im2 = axes[l, 2].imshow(m_sing_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im2, ax=axes[l, 2], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 2].set_title("Model (Single Ens Member)")
        
        # Col 4: GEOS ens mean - Target
        diff_geos = g_img[l] - t_img[l]
        im3 = axes[l, 3].imshow(diff_geos, cmap='RdBu_r', vmin=-30, vmax=30)
        fig.colorbar(im3, ax=axes[l, 3], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 3].set_title(f"GEOS Bias (GEOS Mean - Target)\nCRPS:{geos_crps:.2f}, RMSE:{geos_rmse:.2f}")
        
        # Col 5: Model ens mean - Target
        diff_model = p_img[l] - t_img[l]
        im4 = axes[l, 4].imshow(diff_model, cmap='RdBu_r', vmin=-30, vmax=30)
        fig.colorbar(im4, ax=axes[l, 4], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 4].set_title(f"Model Bias (Model Mean - Target)\nCRPS:{model_crps:.2f}, RMSE:{model_rmse:.2f}")
        
        # Col 6: Closeness plot: abs(GEOS Bias) - abs(Model Bias)
        closeness = np.abs(diff_geos) - np.abs(diff_model)
        im5 = axes[l, 5].imshow(closeness, cmap='PiYG', vmin=-25, vmax=25)
        fig.colorbar(im5, ax=axes[l, 5], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 5].set_title("Closeness: |GEOS Bias| - |Model Bias|\nGreen (>0) = Model Better, Pink (<0) = GEOS Better")
        
        # Col 7: Model Ensemble Variance
        # We cap variance visualization at the 99th percentile across all leads to keep colors readable
        var_vmax = np.percentile(m_var_img, 99) if model_var is not None and m_var_img.max() > 0 else 1.0
        im6 = axes[l, 6].imshow(m_var_img[l], cmap='YlGn', vmin=0, vmax=var_vmax)
        fig.colorbar(im6, ax=axes[l, 6], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 6].set_title("Model Ens Variance")

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.tight_layout()
    filename = f"epoch_{epoch}_{suffix}_score_{model_crps:.4f}.png" if suffix else f"epoch_{epoch}_score_{model_crps:.4f}.png"
    plt.savefig(os.path.join(output_dir, "plots", filename))
    plt.close()

@torch.no_grad()
def run_val_inference(epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                      target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, is_test=False, is_fast_recon=True,
                      cached_geos_crps=None, cached_geos_rmse=None):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Sample 6 batches evenly across the validation set for monthly coverage (2 months apart).
    # With 2 years of weekly data (~104 samples, batch_size=4 -> ~26 batches),
    # 6 evenly-spaced batches give us representative seasonal coverage.
    total_val_batches = len(val_loader)
    num_val_samples = 6
    if total_val_batches >= num_val_samples:
        step = total_val_batches / num_val_samples
        target_batches = [int(i * step) for i in range(num_val_samples)]
    else:
        target_batches = list(range(total_val_batches))
    
    total_crps = 0.0
    total_rmse = 0.0
    total_geos_crps = 0.0
    total_geos_rmse = 0.0
    count = 0
    
    # We will only save/return the tensors for the first batch (idx 0) so the plotting remains identical
    saved_tensors = {}
    
    for b_idx, batch in enumerate(val_loader):
        if b_idx not in target_batches:
            if b_idx > max(target_batches):
                break # Stop iterating once we have all target batches
            continue
            
        fb_target_norm = batch['y_target'].to(device) # [vB, 1, H, W]
        vB, _, H, W = fb_target_norm.shape
        num_inits = vB // 4
        
        # Extract unique init dates (every 4th element)
        true_target_precip = batch['target_raw_full'][0::4].to(device) # [num_inits, 4, H, W]
        
        geos_ens_raw = batch['geos_ens_raw'].to(device) 
        geos_ens_sample = geos_ens_raw[0::4] # [num_inits, M=4, L=4, H, W]
        geos_mean_raw = geos_ens_sample.mean(dim=1) # [num_inits, 4, H, W]
    
        # Prepare 4-week prediction buffer
        pred_res_norm_agg = torch.zeros((4, H, W), device=device)
        
        # Fast validation: 8 ensemble members (speed). Full validation: 6 members (quality).
        num_ensemble = 8 if (is_fast_recon and not is_test) else 6
        ensemble_preds_precip = [] # Will be [E, 4, H, W]

        # Progress bar for internal status during long samplings
        # Fast: 10 Euler steps (rapid screening). Full: 50 steps (publication quality)
        num_steps = 10 if is_fast_recon and not is_test else 50
        
        # --- VRAM GPU BATCHING: Solve all Lead Weeks and Ensemble members SIMULTANEOUSLY ---
        # With zombie processes gone, we can fit [vB * 6] through the UNet at once.
        vB = fb_target_norm.shape[0] 
        
        fx_obs = batch['x_obs'].to(device) 
        fx_geos = batch['x_geos'].to(device) 
        fx_geos_cat = fx_geos.view(vB, -1, H, W)
        
        f_month = batch['month'].to(device).float()
        fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
        fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
        
        fl_idx = batch['lead_idx'].to(device).float()
        f_lead_val = (fl_idx / 1.5) - 1.0 
        f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, H, W)
        
        # [vB, 35, H, W]
        fx_cond = torch.cat([fx_obs, fx_geos_cat, fsin_month, fcos_month, f_lead_channel], dim=1) 
        
        # Expand for simultaneous ensemble generation: [vB * num_ensemble, 35, H, W]
        fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W)
        noise_expanded = torch.randn((vB * num_ensemble, 1, H, W), device=device)
        lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
        
        # Single parallel ODE solve for the entire validation batch and ensemble
        p_x1_expanded = flow_matcher.euler_solve(
            unwrapped_model, noise_expanded, fx_cond_expanded, 
            num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=True
        )
        
        p_x1_batch = p_x1_expanded.view(vB, num_ensemble, H, W)

        
        week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        week_precip = torch.clamp(week_sqrt ** 2, min=0.0) # [vB, num_ensemble, H, W]
        
        ensemble_preds_precip = week_precip.transpose(0, 1) # [num_ensemble, vB, H, W]
        
        # Reshape to separate initialization dates and lead weeks
        ensemble_preds_precip = ensemble_preds_precip.view(num_ensemble, num_inits, 4, H, W)
        
        full_pred = ensemble_preds_precip.mean(dim=0) # [num_inits, 4, H, W]
        model_var = ensemble_preds_precip.var(dim=0) # [num_inits, 4, H, W]
    
        # Calculate CRPS across all inits in this batch natively
        b_crps = compute_crps(ensemble_preds_precip, true_target_precip, area_weights)
        
        # Model RMSE (NaN-aware)
        mse_map = (full_pred - true_target_precip)**2
        mask = ~torch.isnan(mse_map)
        if mask.any():
            aw_expanded = area_weights.view(1, 1, 181, 1).expand_as(mse_map)
            b_rmse = torch.sqrt((mse_map[mask] * aw_expanded[mask]).sum() / (aw_expanded[mask].sum() + 1e-8)).item()
        else:
            b_rmse = 0.0
        
        # GEOS Baseline Metrics (NaN-aware)
        if cached_geos_crps is None:
            # Transpose to [Member, Init, Lead, H, W] to match compute_crps standard [E, B, C, H, W]
            g_crps = compute_crps(geos_ens_sample.transpose(0, 1), true_target_precip, area_weights)
            geos_mse_map = (geos_mean_raw - true_target_precip)**2
            mask_2d = ~torch.isnan(geos_mse_map)
        
            if mask_2d.any():
                aw_expanded_2d = area_weights.view(1, 1, 181, 1).expand_as(geos_mse_map)
                g_rmse = torch.sqrt((geos_mse_map[mask_2d] * aw_expanded_2d[mask_2d]).sum() / (aw_expanded_2d[mask_2d].sum() + 1e-8)).item()
            else:
                g_rmse = 0.0
                
            total_geos_crps += g_crps * num_inits # Weight by batch items
            total_geos_rmse += g_rmse * num_inits
            
        total_crps += b_crps * num_inits # Weight by batch items
        total_rmse += b_rmse * num_inits
        count += num_inits
        
        # Save only the first season (Winter) for visual plotting consistency
        if b_idx == 0:
            # We select the first init date [0] for plotting to keep dims matching matplotlib scripts
            true_target_precip_plot = torch.nan_to_num(true_target_precip[0], nan=0.0)
            ai_residual = full_pred[0] - geos_mean_raw[0]
            saved_tensors = {
                'full_pred': full_pred[0].unsqueeze(0),
                'true_target': true_target_precip_plot.unsqueeze(0),
                'geos_mean': geos_mean_raw[0].unsqueeze(0),
                'ai_res': ai_residual.unsqueeze(0),
                'geos_single': geos_ens_sample[0, 0].unsqueeze(0),
                'model_single': ensemble_preds_precip[0, 0].unsqueeze(0),
                'model_var': model_var[0].unsqueeze(0)
            }
            
    # Compute Averages
    avg_crps = total_crps / count
    avg_rmse = total_rmse / count
    if cached_geos_crps is None:
        avg_geos_crps = total_geos_crps / count
        avg_geos_rmse = total_geos_rmse / count
    else:
        avg_geos_crps = cached_geos_crps
        avg_geos_rmse = cached_geos_rmse
    
    recon_type = f"Monthly (N={len(target_batches)}x{num_ensemble})"
    if accelerator.is_main_process:
        print(f"Epoch {epoch} | Val CRPS [{recon_type}]: {avg_crps:.4f} (GEOS baseline: {avg_geos_crps:.4f})")
        
    return avg_crps, saved_tensors['full_pred'], saved_tensors['true_target'], avg_rmse, saved_tensors['geos_mean'], avg_geos_crps, avg_geos_rmse, saved_tensors['ai_res'], saved_tensors['geos_single'], saved_tensors['model_single'], saved_tensors['model_var']

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

    # Process multiple init dates per validation batch for speed (batch_size * 2 since we flattened leads)
    val_batch_size = max(8, batch_size * 2) 
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=True,
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
    model = FlowMatchingModel(in_channels=36, out_channels=1).to(device)
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

        print(f"\n--- FLOW ARCHITECTURE DIAGNOSTICS ---")
        print(f"   Model Base: FlowMatchingModel (UNet2D Structure)")
        print(f"   Total Input Channels: 36")
        print(f"   --- Condition Channels (x_cond = 35) ---")
        print(f"     [00-03] x_obs: SST (L=1 to 4)")
        print(f"     [04-07] x_obs: SSS (L=1 to 4)")
        print(f"     [08-11] x_obs: Soil Moisture (L=1 to 4)")
        print(f"     [12-15] x_obs: IVT (L=1 to 4)")
        print(f"     [16-19] x_obs: Z500 Zonal Dev (L=1 to 4)")
        print(f"     [20-23] x_obs: U250 (L=1 to 4)")
        print(f"     [24-27] x_obs: MJO Spatial Wave (L=1 to 4)")
        print(f"     [28-31] x_geos: GEOS Precipitation Forecast (L=1 to 4)")
        print(f"     [    32] Month: Sine Temporal Embedding")
        print(f"     [    33] Month: Cosine Temporal Embedding")
        print(f"     [    34] Target Lead: Relative Index Tracking [-1 to +1]")
        print(f"   --- Dynamic Flow Channel (x_t = 1) ---")
        print(f"     [    35] x_t: Pure Noise Vector (Solver Substrate)")
        print(f"   --- Optimization Target ---")
        print(f"     Velocity Target (v_theta): SQRT(GPCP) Precipitation, Normalized to [-1, 1]")
        print(f"   --- Dedicated Output Heads (Multi-Task Architecture) ---")
        print(f"     Head 0: Week 1 (Conv2d 64→1)")
        print(f"     Head 1: Week 2 (Conv2d 64→1)")
        print(f"     Head 2: Week 3 (Conv2d 64→1)")
        print(f"     Head 3: Week 4 (Conv2d 64→1)")
        print(f"     Shared UNet features: 64 intermediate channels")
        print(f"-------------------------------------\n")

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
        ckpt_path = os.path.join(output_dir, "latest_flow_ckpt.pt")

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
                    
                # Intelligent Reset: If the loaded CRPS is suspiciously low (< 0.8), it likely came from the old 
                # single-batch (January-only) validation logic. We must reset it to allow the new 4-season metric to save.
                if best_val_crps < 0.8:
                    if accelerator.is_main_process:
                        print(f"⚠️ Detected suspiciously low CRPS ({best_val_crps:.4f}) from old validation logic. Resetting to 1.3000.")
                    best_val_crps = 1.3000
                    top_models = []
                
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
        
        # Channel 20: U250
        u250_norm_sample = x_obs[sample_idx, 20].cpu().numpy()
        u250_raw_sample = ((u250_norm_sample + 1.0) / 2.0) * (global_bounds["u250"]["max"] - global_bounds["u250"]["min"]) + global_bounds["u250"]["min"]

        # Channel 24: MJO Wave Spatial Map
        mjo_norm_sample = x_obs[sample_idx, 24].cpu().numpy()
        if "mjo" in global_bounds:
            m_min, m_max = global_bounds["mjo"]["min"], global_bounds["mjo"]["max"]
        else:
            m_min, m_max = -100.0, 100.0
        mjo_raw_sample = ((mjo_norm_sample + 1.0) / 2.0) * (m_max - m_min) + m_min

        # 5. Raw GPCP (from dataset)
        gpcp_raw_sample = batch['target_raw'][sample_idx, lead_idx].cpu().numpy()

        fig, axes = plt.subplots(11, 2, figsize=(14, 44))
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

        im17 = axes[8, 0].imshow(mjo_raw_sample, cmap='PiYG', vmin=-30, vmax=30)
        axes[8, 0].set_title("Raw MJO Wave Anomaly")
        fig.colorbar(im17, ax=axes[8, 0])
        
        im18 = axes[8, 1].imshow(mjo_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[8, 1].set_title("Normalized MJO Wave [-1, 1]")
        fig.colorbar(im18, ax=axes[8, 1])

        im19 = axes[9, 0].imshow(gpcp_raw_sample, cmap='Blues')
        axes[9, 0].set_title("Raw GPCP (Pure Target)")
        fig.colorbar(im19, ax=axes[9, 0])
        
        axes[9, 1].text(0.5, 0.5, "GPCP is not\nnormalized directly.", ha='center', va='center', transform=axes[9, 1].transAxes)
        axes[9, 1].axis('off')

        flat_geos = geos_raw_sample.flatten()
        flat_gpcp = gpcp_raw_sample.flatten()
        cc = np.corrcoef(flat_geos, flat_gpcp)[0, 1] if np.std(flat_geos) > 1e-6 else 0.0
        
        im20 = axes[10, 0].imshow(geos_raw_sample - gpcp_raw_sample, cmap='RdBu_r', vmin=-20, vmax=20)
        axes[10, 0].set_title(f"Spatial Alignment | CC: {cc:.4f}")
        fig.colorbar(im20, ax=axes[10, 0])
        
        axes[10, 1].scatter(flat_geos[::50], flat_gpcp[::50], alpha=0.3, s=1)
        axes[10, 1].set_xlabel("GEOS Rainfall")
        axes[10, 1].set_ylabel("GPCP Rainfall")
        axes[10, 1].set_title("Orientation Scatter (Subsampled)")
        
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

    global_cached_geos_crps = None
    global_cached_geos_rmse = None
    
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

            # Total: 28 (Obs/Dev/MJO) + 4 (GEOS) + 2 (Month) + 1 (Lead) = 35 channels
            x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1) 

            # Targets are already residual normalized [-1, 1] by dataset_hybrid
            target_norm = batch['y_target'].to(device) # [B, 1, H, W]

            # Flow Matching Interpolation
            t = flow_matcher.sample_time_batch(B)
            noise = torch.randn_like(target_norm)
            x_t, v_target = flow_matcher.interpolate(target_norm, noise, t)

            # Predict the velocity (routed through the correct per-week output head)
            v_pred, var_pred = model(x_t, x_cond, t, lead_idx=lead_idx)

            # --- Target Variance (Gradient Isolated) ---
            # We want the variance head to predict the squared error of the mean head,
            # WITHOUT letting gradients flow backward to disrupt the flow matching trajectory.
            target_var = (v_target - v_pred.detach())**2

            # --- Temporal Loss Weighting ---
            # Prioritize gradient updates for harder long-term leads (Week 4 > Week 1)
            # 0=Week1, 1=Week2, 2=Week3, 3=Week4
            w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
            temp_weights = w_escalation[lead_idx].view(B, 1, 1, 1)

            # Loss computation (Dual Head)
            loss_v = (area_weights * temp_weights * (v_pred - v_target)**2).mean()
            loss_var = (area_weights * temp_weights * (var_pred - target_var)**2).mean()
            
            # Combine losses
            loss = loss_v + loss_var

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix({"loss_v": f"{loss_v.item():.4f}", "loss_var": f"{loss_var.item():.4f}"})

        avg_train_loss = train_loss / len(loader)
        
        if accelerator.is_main_process:
            print(f"📈 Epoch {epoch} Training Loss (Noise MSE): {avg_train_loss:.4f}")

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

        # --- ADAPTIVE VALIDATION SCHEDULE ---
        # Phase 1 (epoch < 20):  No validation (model still warming up)
        # Phase 2 (20-49):       Every 3 epochs (20, 23, 26, ...)
        # Phase 3 (50-99):       Every 2 epochs (50, 52, 54, ...)
        # Phase 4 (100+):        Every epoch (100, 101, 102, ...)
        # Full 50-step val only fires if fast val finds a new absolute best.
        def should_validate(ep):
            if ep < 20:
                return False
            elif ep < 50:
                return (ep % 3 == 0)
            elif ep < 100:
                return (ep % 2 == 0)
            else:
                return True

        if not should_validate(epoch):
            if accelerator.is_main_process:
                next_val = next((e for e in range(epoch+1, epoch+200) if should_validate(e)), epoch+1)
                print(f"⏭️  Epoch {epoch}: Skipping validation (schedule: next at {next_val}).")
            epochs_done_this_run += 1
            continue

        if accelerator.is_main_process:
            print(f"\n⌛ Epoch {epoch} complete. Starting Validation (Inference)...")
        
        # Always do fast validation first. Only do expensive full sampling if new best is found.
        is_plot_epoch = args.full_val
        
        val_outputs = run_val_inference(
            epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
            target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds,
            is_test=is_plot_epoch, 
            is_fast_recon=not is_plot_epoch,
            cached_geos_crps=global_cached_geos_crps,
            cached_geos_rmse=global_cached_geos_rmse
        )
        current_val_metric, full_pred, true_target_precip, model_rmse, geos_mean, geos_crps, geos_rmse, current_ai_res, geos_single, model_single, model_var = val_outputs
        
        if global_cached_geos_crps is None:
            global_cached_geos_crps = geos_crps
            global_cached_geos_rmse = geos_rmse
            
        if accelerator.is_main_process:
            # 1. Always plot the results from the first validation pass (usually fast ensemble)
            plot_suffix = "fast" if not is_plot_epoch else "full"
            save_val_plot(epoch, full_pred, true_target_precip, current_val_metric, model_rmse, 
                          geos_mean, geos_crps, geos_rmse, output_dir, ai_residual=current_ai_res, suffix=plot_suffix,
                          geos_single=geos_single, model_single=model_single, model_var=model_var)

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
                    num_steps = 50 
                    print(f"📸 Breakthrough! Triggering high-quality {num_steps}-step sampling for diagnostic plots...")
                    best_sampled_metric, best_sampled_pred, best_target, b_rmse, b_geos_mean, b_geos_crps, b_geos_rmse, b_ai_res, b_gs, b_ms, b_mv = run_val_inference(
                        epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                        target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds,
                        is_test=True, is_fast_recon=False
                    )
                    save_val_plot(epoch, best_sampled_pred, best_target, best_sampled_metric, b_rmse, 
                                  b_geos_mean, b_geos_crps, b_geos_rmse, output_dir, ai_residual=b_ai_res, suffix="BEST_sampled",
                                  geos_single=b_gs, model_single=b_ms, model_var=b_mv)

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
                
                # Keep a symlink or redundant copy for 'best_flow_ckpt.pt'
                if is_new_best:
                    torch.save(best_ckpt, os.path.join(output_dir, "best_flow_ckpt.pt"))

            # Save Latest Checkpoint & Logs
            ckpt = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_crps': best_val_crps,
                'top_models': top_models
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_flow_ckpt.pt"))

            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, 0.0, current_val_metric])
                    
            if is_in_top4:
                top_str = ", ".join([f"E{m['epoch']}({m['crps']:.3f})" for m in top_models])
                print(f"📍 Top 4 Models: [{top_str}]")

        # Track progress for this execution session
        epochs_done_this_run += 1

def main():
    parser = argparse.ArgumentParser(description="Train or Test Flow Matching Model")
    parser.add_argument("--config", type=str, default="ml_model/config_flow.yaml")
    parser.add_argument("--test", action="store_true", help="Run in inference/test mode only")
    parser.add_argument("--ckpt", type=str, default="best_flow_ckpt.pt", 
                        help="Checkpoint filename in output_dir to load for testing (default: best_flow_ckpt.pt)")
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
