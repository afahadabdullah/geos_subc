import os
import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import csv
import json
import xarray as xr
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import argparse
from accelerate import Accelerator

import pandas as pd

# Local Modules
from dataset_flow import S2SHybridDataset
from flow_matching import FlowMatchingModel, CustomFlowMatcher
import noise_utils

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

def compute_rmse(pred: torch.Tensor, target: torch.Tensor, area_weights: torch.Tensor):
    """
    Computes area-weighted RMSE.
    pred: [B, H, W]
    target: [B, H, W]
    area_weights: [1, 1, H, 1]
    """
    mse_map = (pred - target)**2
    mask = ~torch.isnan(mse_map)
    if mask.any():
        aw_expanded = area_weights.view(1, 1, 181, 1).expand_as(mse_map)
        rmse = torch.sqrt((mse_map[mask] * aw_expanded[mask]).sum() / (aw_expanded[mask].sum() + 1e-8)).item()
    else:
        rmse = 0.0
    return rmse

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
                      cached_geos_crps=None, cached_geos_rmse=None, use_flow_variance=False, eof_bases=None,
                      nao_bases=None, nao_lookup=None, enso_bases=None, oni_lookup=None, mjo_df=None):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # Sample exactly 6 specific months from 2021 (Jan, Mar, May, Jul, Sep, Nov)
    target_months = [1, 3, 5, 7, 9, 11]
    target_batches = []
    found_months = set()
    
    # We iterate once over the loader to map out the batch indices that correspond to our requested 2021 months
    for b_idx, batch in enumerate(val_loader):
        year = int(batch['year'][0].item()) if 'year' in batch else 2021
        month = int(batch['month'][0].item())
        
        if year == 2021 and month in target_months and month not in found_months:
            target_batches.append(b_idx)
            found_months.add(month)
            
        if len(target_batches) == len(target_months):
            break
            
    # Fallback just in case the dataloader is much smaller or doesn't have 2021
    if len(target_batches) < 6:
        target_batches = list(range(min(6, len(val_loader))))
    
    total_crps = 0.0
    total_rmse = 0.0
    total_geos_crps = 0.0
    total_geos_rmse = 0.0
    count = 0
    
    # We will only save/return the tensors for the first batch (idx 0) so the plotting remains identical
    saved_tensors = {}
    
    for b_idx, batch in enumerate(val_loader):
        if b_idx not in target_batches:
            if b_idx > max(target_batches) if len(target_batches) > 0 else 0:
                break # Stop iterating once we have all target batches
            continue
            
        fb_target_norm = batch['y_target'].to(device) # [vB, 2, H, W]
        vB, _, H, W = fb_target_norm.shape
        num_inits = vB // 4
        
        # Extract unique init dates (every 4th element)
        true_target_raw = batch['target_raw_full'][0::4].to(device) # [num_inits, 2, 4, H, W]
        true_target_precip = true_target_raw[:, 0] # [num_inits, 4, H, W]
        true_target_t2m = true_target_raw[:, 1] # [num_inits, 4, H, W]
        
        geos_ens_raw = batch['geos_ens_raw'].to(device) 
        geos_ens_sample = geos_ens_raw[0::4] # [num_inits, M=4, C=2, L=4, H, W]
        
        geos_mean_raw = geos_ens_sample.mean(dim=1) # [num_inits, 2, 4, H, W]
        geos_mean_precip = geos_mean_raw[:, 0]
        geos_mean_t2m = geos_mean_raw[:, 1]
    
        # Prepare 4-week prediction buffer
        pred_res_norm_agg = torch.zeros((4, H, W), device=device)
        
        # Fast validation: 12 ensemble members (speed). Full validation: 12 members (quality).
        num_ensemble = 12
        ensemble_preds_precip = [] # Will be [num_ensemble, num_inits, 4, H, W]
        ensemble_preds_t2m = []

        # Progress bar for internal status during long samplings
        # Fast: 10 Euler steps (rapid screening). Full: 50 steps (publication quality)
        num_steps = 10 if is_fast_recon and not is_test else 50
        
        # --- VRAM GPU BATCHING: Solve all Lead Weeks and Ensemble members SIMULTANEOUSLY ---
        # With zombie processes gone, we can fit [vB * 12] through the UNet at once.
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
        # Generate antithetical dynamic multimodal noise (z and -z)
        num_base_members = num_ensemble // 2
        
        # Try to dynamically fetch validation year from batch if available, fallback to 2021
        val_year = 2021
        if 'year' in batch:
            val_year = int(batch['year'][0].item())
            
        if eof_bases is not None:
            noise_base = noise_utils.generate_dynamic_multimodal_noise(
                batch, num_base_members, device, eof_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, val_year
            )
            # noise_base is [vB*num_base_members, 2, H, W] from updated flow_matcher.eof_sample
            noise_anti = -noise_base
            
            noise_base_reshaped = noise_base.view(vB, num_base_members, 2, H, W)
            noise_anti_reshaped = noise_anti.view(vB, num_base_members, 2, H, W)
            noise_expanded = torch.cat([noise_base_reshaped, noise_anti_reshaped], dim=1).reshape(vB * num_ensemble, 2, H, W)
        else:
            noise_base = torch.randn((vB * num_base_members, 2, H, W), device=device)
            noise_anti = -noise_base
            noise_base_reshaped = noise_base.view(vB, num_base_members, 2, H, W)
            noise_anti_reshaped = noise_anti.view(vB, num_base_members, 2, H, W)
            noise_expanded = torch.cat([noise_base_reshaped, noise_anti_reshaped], dim=1).reshape(vB * num_ensemble, 2, H, W)
            
        lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
        
        # Single parallel ODE solve for the entire validation batch and ensemble
        p_x1_expanded = flow_matcher.euler_solve(
            unwrapped_model, noise_expanded, fx_cond_expanded, 
            num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=use_flow_variance
        )
        
        p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)

        # Reverse PR (Channel 0)
        p_x1_pr = p_x1_batch[:, :, 0] # [vB, E, H, W]
        week_sqrt = ((p_x1_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        week_precip = torch.clamp(week_sqrt ** 2, min=0.0) # [vB, num_ensemble, H, W]
        
        # Reverse T2M (Channel 1)
        # Using placeholder min/max of 200/320 for T2M as setup in the dataloader until stats hit
        p_x1_t2m = p_x1_batch[:, :, 1]
        t2m_min, t2m_max = 200.0, 320.0 
        week_t2m = ((p_x1_t2m + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min
        
        ensemble_preds_precip = week_precip.transpose(0, 1) # [num_ensemble, vB, H, W]
        ensemble_preds_t2m = week_t2m.transpose(0, 1)
        
        # Reshape to separate initialization dates and lead weeks
        ensemble_preds_precip = ensemble_preds_precip.view(num_ensemble, num_inits, 4, H, W)
        ensemble_preds_t2m = ensemble_preds_t2m.view(num_ensemble, num_inits, 4, H, W)
        
        full_pred_precip = ensemble_preds_precip.mean(dim=0) # [num_inits, 4, H, W]
        model_var_precip = ensemble_preds_precip.var(dim=0) # [num_inits, 4, H, W]
        
        full_pred_t2m = ensemble_preds_t2m.mean(dim=0)
        
        # Calculate Natively
        b_crps = compute_crps(ensemble_preds_precip, true_target_precip, area_weights)
        b_rmse = compute_rmse(full_pred_precip, true_target_precip, area_weights)
        
        b_crps_t2m = compute_crps(ensemble_preds_t2m, true_target_t2m, area_weights)
        b_rmse_t2m = compute_rmse(full_pred_t2m, true_target_t2m, area_weights)
        
        # GEOS Baseline Metrics (NaN-aware)
        if cached_geos_crps is None:
            g_crps = compute_crps(geos_ens_sample[:, :, 0].transpose(0, 1), true_target_precip, area_weights)
            g_rmse = compute_rmse(geos_mean_precip, true_target_precip, area_weights)
            
            total_geos_crps += g_crps * num_inits # Weight by batch items
            total_geos_rmse += g_rmse * num_inits
            
        total_crps += b_crps * num_inits # Weight by batch items
        total_rmse += b_rmse * num_inits
        count += num_inits
        
        # Save only the first season (Winter) for visual plotting consistency
        if b_idx == 0:
            # We select the first init date [0] for plotting to keep dims matching matplotlib scripts
            true_target_precip_plot = torch.nan_to_num(true_target_precip[0], nan=0.0)
            ai_residual = full_pred_precip[0] - geos_mean_precip[0]
            saved_tensors = {
                'full_pred': full_pred_precip[0].unsqueeze(0),
                'true_target': true_target_precip_plot.unsqueeze(0),
                'geos_mean': geos_mean_precip[0].unsqueeze(0),
                'ai_res': ai_residual.unsqueeze(0),
                'geos_single': torch.nan_to_num(geos_ens_sample[0, 0, 0], nan=0.0).unsqueeze(0),
                'model_single': ensemble_preds_precip[0, 0].unsqueeze(0),
                'model_var': model_var_precip[0].unsqueeze(0)
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
        print(f"Epoch {epoch} | Val CRPS [PR]: {avg_crps:.4f} (GEOS baseline: {avg_geos_crps:.4f})")
        print(f"Epoch {epoch} | Val CRPS [T2M]: {b_crps_t2m:.4f}  | RMSE: {b_rmse_t2m:.4f}")
        
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

    # Area weights (needed early for diagnostic plot)
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # ─── Land-Ocean Mask (V6: 65% land / 35% ocean) ───
    # Derive from SSS data: NaN pixels = land, valid pixels = ocean
    # Cache to .pt file so we only need SSS once.
    land_ocean_weights = torch.ones(1, 1, 181, 360, device=device)  # Default: uniform
    mask_cache_path = os.path.join(os.path.dirname(__file__), "land_ocean_mask_v6.pt")
    
    if os.path.exists(mask_cache_path):
        # ── Load cached mask ──
        cached = torch.load(mask_cache_path, map_location=device, weights_only=True)
        land_ocean_weights = cached['weights'].to(device)
        if accelerator.is_main_process:
            print(f"  ✅ V6 Land-Ocean Mask loaded from cache: {mask_cache_path}")
            print(f"     Land pixels: {cached['n_land']}, weight = {cached['land_w']:.4f}")
            print(f"     Ocean pixels: {cached['n_ocean']}, weight = {cached['ocean_w']:.4f}")
    else:
        # ── Create mask from SSS ──
        sss_sample_path = os.path.join(config["data_dir"], "sss_weekly_2020.zarr")
        if os.path.exists(sss_sample_path):
            try:
                ds_sss = xr.open_zarr(sss_sample_path, consolidated=False)
                sss_arr = ds_sss['sss'].isel(S=0, L=0).values  # [Y, X]
                is_land = np.isnan(sss_arr)  # True = land
                n_land = int(is_land.sum())
                n_ocean = int((~is_land).sum())
                n_total = n_land + n_ocean
                
                if n_land > 0 and n_ocean > 0:
                    land_w = 0.65 * n_total / n_land
                    ocean_w = 0.35 * n_total / n_ocean
                    
                    mask_np = np.where(is_land, land_w, ocean_w).astype(np.float32)
                    land_ocean_weights = torch.from_numpy(mask_np).to(device).view(1, 1, 181, 360)
                    
                    # Save cache for future runs
                    if accelerator.is_main_process:
                        torch.save({
                            'weights': land_ocean_weights.cpu(),
                            'is_land': torch.from_numpy(is_land),
                            'n_land': n_land, 'n_ocean': n_ocean,
                            'land_w': land_w, 'ocean_w': ocean_w,
                        }, mask_cache_path)
                        print(f"  ✅ V6 Land-Ocean Mask created from {sss_sample_path}")
                        print(f"     Land pixels: {n_land} ({n_land/n_total*100:.1f}%), weight = {land_w:.4f}")
                        print(f"     Ocean pixels: {n_ocean} ({n_ocean/n_total*100:.1f}%), weight = {ocean_w:.4f}")
                        print(f"     💾 Cached to {mask_cache_path}")
                        
                        # ── Diagnostic Plot ──
                        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
                        
                        im0 = axes[0].imshow(is_land.astype(float), cmap='RdYlGn', vmin=0, vmax=1,
                                            extent=[-180, 180, -90, 90], aspect='auto')
                        axes[0].set_title(f"Land-Ocean Mask (Green=Land, Red=Ocean)\nLand: {n_land} px ({n_land/n_total*100:.1f}%)")
                        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                        
                        im1 = axes[1].imshow(mask_np, cmap='hot_r',
                                            extent=[-180, 180, -90, 90], aspect='auto')
                        axes[1].set_title(f"Loss Weight Map\nLand w={land_w:.3f}, Ocean w={ocean_w:.3f}")
                        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                        
                        combined = (area_weights.cpu().numpy().reshape(181, 1) * mask_np)
                        im2 = axes[2].imshow(combined, cmap='magma',
                                            extent=[-180, 180, -90, 90], aspect='auto')
                        axes[2].set_title("Combined: Area Weight × Land-Ocean Weight")
                        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                        
                        plt.suptitle("V6 Land-Ocean Loss Weighting Diagnostic", fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        diag_dir = config.get("output_dir", "ml_output_flow6")
                        os.makedirs(diag_dir, exist_ok=True)
                        plot_path = os.path.join(diag_dir, "land_ocean_mask_diagnostic.png")
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"     📸 Diagnostic plot saved: {plot_path}")
                else:
                    if accelerator.is_main_process:
                        print(f"  ⚠️ SSS mask has no land/ocean split. Using uniform weights.")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"  ⚠️ Failed to load SSS for land mask: {e}. Using uniform weights.")
        else:
            if accelerator.is_main_process:
                print(f"  ⚠️ SSS file not found at {sss_sample_path}. Using uniform land-ocean weights.")
    
    # Get stats file from config (no fallback - must be specified)
    stats_filename = config.get("stats_file", "v1_multi_global_stats.pt")
    
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False),
        stats_file=stats_filename,
        subsample_monthly=True
    )

    # Process multiple init dates per validation batch for speed (batch_size * 2 since we flattened leads)
    val_batch_size = max(8, batch_size * 2) 
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False,
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
            stats_file=stats_filename
        )
        loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=config.get("num_workers", 4), pin_memory=True
        )

    # Calculate Global Min-Max for Target GPCP Precipitation
    stats_file = os.path.join("ml_model", stats_filename)
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"CRITICAL: {stats_file} missing. Please run calculate_global_stats_multi_v1.py first!")
    
    global_bounds = torch.load(stats_file, weights_only=True)
    # Force robust physical range for Direct Power Transformed GPCP (sqrt)
    # Target raw max is roughly 50 mm/day max in GPCP weekly. sqrt(50) ~= 7.071
    target_sqrt_min = 0.0
    target_sqrt_max = 7.071
    
    geos_min = global_bounds["geos_pr_raw"]["min"] if "geos_pr_raw" in global_bounds else global_bounds["geos_raw"]["min"]
    geos_max = global_bounds["geos_pr_raw"]["max"] if "geos_pr_raw" in global_bounds else global_bounds["geos_raw"]["max"]
    
    if accelerator.is_main_process:
        print("\n=======================================================")
        print(f"✅ Loaded Strict Global Stats: {stats_file}")
        print(f"   [Target SQRT Bounds] : Min = {target_sqrt_min:.4f}, Max = {target_sqrt_max:.4f}")
        print(f"   [GEOS Raw Bounds]    : Min = {geos_min:.4f}, Max = {geos_max:.4f}")
        print("=======================================================\n")

    # ---------------------------------------------------------
    # 2. Model & Scheduler Setup
    # ---------------------------------------------------------
    model = FlowMatchingModel(in_channels=37, out_channels=2).to(device)
    flow_matcher = CustomFlowMatcher(device=device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if not args.test:
        model, optimizer, loader, val_loader = accelerator.prepare(
            model, optimizer, loader, val_loader
        )
    else:
        # Test mode: only prepare model and val_loader
        model, val_loader = accelerator.prepare(model, val_loader)
        optimizer = None

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
        print(f"   Total Input Channels: 37")
        print(f"   --- Condition Channels (x_cond = 35) ---")
        print(f"     [00-03] x_obs: SST (L=1 to 4)")
        print(f"     [04-07] x_obs: SSS (L=1 to 4)")
        print(f"     [08-11] x_obs: Soil Moisture (L=1 to 4)")
        print(f"     [12-15] x_obs: IVT (L=1 to 4)")
        print(f"     [16-19] x_obs: Z500 Zonal Dev (L=1 to 4)")
        print(f"     [20-23] x_obs: U250 (L=1 to 4)")
        print(f"     [24-27] x_obs: MJO Spatial Wave (L=1 to 4)")
        print(f"     [28-31] x_geos: GEOS Precipitation Forecast (L=1 to 4)")
        print(f"     [32-35] x_geos: GEOS T2M Forecast (L=1 to 4)")
        print(f"     [    36] Month: Sine Temporal Embedding")
        print(f"     [    37] Month: Cosine Temporal Embedding")
        print(f"     [    38] Target Lead: Relative Index Tracking [-1 to +1]")
        print(f"   --- Dynamic Flow Channel (x_t = 2) ---")
        print(f"     [ 39,40] x_t: Pure Noise Vector (Solver Substrate) PR & T2M")
        print(f"   --- Dedicated Output Heads (Multi-Task Architecture) ---")
        print(f"     Head 0: Week 1 (Conv2d 64→2)")
        print(f"     Head 1: Week 2 (Conv2d 64→2)")
        print(f"     Head 2: Week 3 (Conv2d 64→2)")
        print(f"     Head 3: Week 4 (Conv2d 64→2)")
        print(f"     Shared UNet features: 64 intermediate channels")
        print(f"-------------------------------------\n")


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

    # Load Dynamic Multi-Modal Bases
    eof_bases_path = os.path.join(config["data_dir"], "mjo_eof_bases.pt")
    nao_bases_path = os.path.join(config["data_dir"], "nao_eof_bases.pt")
    enso_bases_path = os.path.join(config["data_dir"], "enso_eof_bases.pt")
    
    eof_bases = torch.load(eof_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(eof_bases_path) else None
    nao_bases = torch.load(nao_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(nao_bases_path) else None
    enso_bases = torch.load(enso_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(enso_bases_path) else None
    
    try:
        nao_lookup = noise_utils.parse_nao_index(os.path.join(config["data_dir"], "norm.daily.nao.index.b500101.current.ascii"))
        oni_lookup = noise_utils.parse_oni_index(os.path.join(config["data_dir"], "oni.ascii.txt"))
        mjo_df = pd.read_csv(os.path.join(config["data_dir"], "mjo_processed.csv"), parse_dates=['S']).set_index(pd.to_datetime(pd.read_csv(os.path.join(config["data_dir"], "mjo_processed.csv"), parse_dates=['S'])['S']).dt.strftime('%Y-%m-%d'))
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ Teleconnection index loading failed: {e}. Falling back to default amplitudes.")
        nao_lookup, oni_lookup, mjo_df = None, None, None
        
    if accelerator.is_main_process and eof_bases is not None:
        print("✅ Loaded Multi-Modal EOF bases & Teleconnection Indices for dynamic validation noise.")
    
    start_epoch = 0
    best_val_loss = float('inf')
    top_models = [] # List of {"path": str, "val_loss": float, "epoch": int}
    
    # Load latest checkpoint if it exists
    if args.test:
        # Resolve checkpoint from --ckpt-rank using JSON registry, or fallback to --ckpt
        if hasattr(args, 'ckpt_rank') and args.ckpt_rank is not None:
            registry_path = os.path.join(output_dir, "model_registry.json")
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                rank = args.ckpt_rank
                if rank < 1 or rank > len(registry):
                    raise ValueError(f"--ckpt-rank {rank} out of range. Registry has {len(registry)} models.")
                ckpt_entry = registry[rank - 1]  # 0-indexed
                ckpt_path = ckpt_entry['path']
                if accelerator.is_main_process:
                    print(f"🎯 Loading checkpoint rank #{rank}: Epoch {ckpt_entry['epoch']}, Val Loss {ckpt_entry['val_loss']:.4f}")
                    print(f"   Path: {ckpt_path}")
            else:
                raise FileNotFoundError(f"model_registry.json not found in {output_dir}. Cannot use --ckpt-rank.")
        else:
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
                start_epoch = checkpoint['epoch'] + 1
                
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except (ValueError, RuntimeError) as e:
                    if accelerator.is_main_process:
                        print(f"   ⚠️ Optimizer state mismatch. Starting with fresh optimizer.")
                        print(f"      Error: {e}")
                
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                elif 'best_val_crps' in checkpoint:
                    if accelerator.is_main_process:
                        print("   ⚠️ Legacy CRPS checkpoint detected. Migrating tracking to MSE Loss scale.")
                    best_val_loss = float('inf')
                    checkpoint['top_models'] = [] # Clear legacy CRPS top_models so we don't mix scales
                
                if 'top_models' in checkpoint:
                    top_models = checkpoint['top_models']
                    # Ensure backward compatible keys
                    for m in top_models:
                        if 'crps' in m:
                            m['val_loss'] = m.pop('crps')
                    
                if config.get("reset_validation_history", False):
                    if accelerator.is_main_process:
                        print(f"⚠️ Resetting validation metrics as requested by config (reset_validation_history: True).")
                    best_val_loss = float('inf')
                    top_models = []
                
            if accelerator.is_main_process:
                print(f"\n🔄 Loaded checkpoint: {ckpt_path}")
                if not args.test:
                    print(f"   Starting at Epoch: {start_epoch}")
                    print(f"   Best Val Loss so far: {best_val_loss:.4f}\n")
                    

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
            is_test=True, is_fast_recon=False,
            use_flow_variance=True,
            eof_bases=eof_bases, nao_bases=nao_bases, nao_lookup=nao_lookup,
            enso_bases=enso_bases, oni_lookup=oni_lookup, mjo_df=mjo_df
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
        geos_norm_sample = np.nan_to_num(x_geos[sample_idx, 0, 0, lead_idx].cpu().numpy(), nan=-1.0)
        geos_raw_sample = ((geos_norm_sample + 1.0) / 2.0) * (geos_max - geos_min) + geos_min
        
        # 2. Reverse Normalize SST (Channel 0 of x_obs)
        sst_norm_sample = np.nan_to_num(x_obs[sample_idx, 0].cpu().numpy(), nan=0.0)
        sst_raw_sample = ((sst_norm_sample + 1.0) / 2.0) * (global_bounds["sst"]["max"] - global_bounds["sst"]["min"]) + global_bounds["sst"]["min"]
        
        # 3. Reverse Normalize Target SQRT (Lead 0)
        sqrt_norm_sample = np.nan_to_num(target_norm[sample_idx, lead_idx].cpu().numpy(), nan=-1.0)
        sqrt_raw_sample = ((sqrt_norm_sample + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        res_raw_sample = np.square(sqrt_raw_sample) - geos_raw_sample

        # 4. Reverse Normalize Observational States
        # SST(0-3), SSS(4-7), SM(8-11), IVT(12-15), ZDEV(16-19), U250(20-23)
        sss_norm_sample = np.nan_to_num(x_obs[sample_idx, 4].cpu().numpy(), nan=0.0)
        sss_raw_sample = ((sss_norm_sample + 1.0) / 2.0) * (global_bounds["sss"]["max"] - global_bounds["sss"]["min"]) + global_bounds["sss"]["min"]
        
        sm_norm_sample = np.nan_to_num(x_obs[sample_idx, 8].cpu().numpy(), nan=0.0)
        sm_raw_sample = ((sm_norm_sample + 1.0) / 2.0) * (global_bounds["sm"]["max"] - global_bounds["sm"]["min"]) + global_bounds["sm"]["min"]
        
        ivt_norm_sample = np.nan_to_num(x_obs[sample_idx, 12].cpu().numpy(), nan=0.0)
        ivt_raw_sample = ((ivt_norm_sample + 1.0) / 2.0) * (global_bounds["ivt"]["max"] - global_bounds["ivt"]["min"]) + global_bounds["ivt"]["min"]
        
        zdev_norm_sample = np.nan_to_num(x_obs[sample_idx, 16].cpu().numpy(), nan=0.0)
        zdev_raw_sample = ((zdev_norm_sample + 1.0) / 2.0) * (global_bounds["z500_zonal_dev"]["max"] - global_bounds["z500_zonal_dev"]["min"]) + global_bounds["z500_zonal_dev"]["min"]
        
        # Channel 20: U250
        u250_norm_sample = np.nan_to_num(x_obs[sample_idx, 20].cpu().numpy(), nan=0.0)
        u250_raw_sample = ((u250_norm_sample + 1.0) / 2.0) * (global_bounds["u250"]["max"] - global_bounds["u250"]["min"]) + global_bounds["u250"]["min"]

        # Channel 24: MJO Wave Spatial Map
        mjo_norm_sample = np.nan_to_num(x_obs[sample_idx, 24].cpu().numpy(), nan=0.0)
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
        for i, batch in enumerate(pbar):    
            # Conditionals
            x_geos = batch['x_geos'].to(device)
            x_obs  = batch['x_obs'].to(device)
            
            B = x_geos.shape[0]
            H, W = x_obs.shape[-2], x_obs.shape[-1]
            # Flatten GEOS variables and leads into channels
            x_geos_flat = x_geos.contiguous().view(B, -1, H, W)
            
            months = batch['month'].to(device)
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)

            # --- Lead Embedding (NEW) ---
            lead_idx = batch['lead_idx'].to(device) # [B]
            # Map [0, 1, 2, 3] to [-1.0, -0.33, 0.33, 1.0]
            lead_val = (lead_idx.float() / 1.5) - 1.0 
            lead_channel = lead_val.view(B, 1, 1, 1).expand(B, 1, H, W)

            # Total: 24 (Obs) + 4 (MJO) + 8 (GEOS [2 vars * 4 leads]) + 2 (Month) + 1 (Lead) = 39 channels out of which Obs/MJO combined is 28.
            # So 28 + 8 + 2 + 1 = 39 ? Wait. x_obs earlier was 28. 28 + 8 + 2 + 1 = 39. Let's trace.
            # x_obs shape in dataset_flow.py gives: SST(4)+SSS(4)+SM(4)+IVT(4)+Z500(4)+U250(4)+MJO(4) = 28.
            x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1) 

            # Targets are already residual normalized [-1, 1] by dataset_hybrid
            target_norm = batch['y_target'].to(device) # [B, 2, H, W] (PR, T2M)

            # Flow Matching Interpolation
            t = flow_matcher.sample_time_batch(B)
            noise = torch.randn_like(target_norm)
            x_t, v_target = flow_matcher.interpolate(target_norm, noise, t)

            # Predict the velocity (routed through the correct per-week output head)
            v_pred, _ = model(x_t, x_cond, t, lead_idx=lead_idx)

            # --- Temporal Loss Weighting ---
            # Prioritize gradient updates for harder long-term leads (Week 4 > Week 1)
            # 0=Week1, 1=Week2, 2=Week3, 3=Week4
            w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
            temp_weights = w_escalation[lead_idx].view(B, 1, 1, 1)

            # Loss computation (V6: area-weighted + land-ocean weighted + temporal weighted MSE)
            # Both PR (Channel 0) and T2M (Channel 1) are treated with the MSE loss
            # For PR, this is the transformed velocity. For T2M, it's the 200-320K standardized velocity.
            # Expanding temp_weights for 2 channels
            temp_weights_expanded = temp_weights.expand(-1, 2, -1, -1)
            loss = (area_weights * land_ocean_weights * temp_weights_expanded * (v_pred - v_target)**2).mean()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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
                'best_val_loss': best_val_loss,
                'top_models': top_models
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_flow_ckpt.pt"))

        # --- ADAPTIVE VALIDATION SCHEDULE ---
        # Phase 1 (epoch < 5):   No validation (model still warming up)
        # Phase 2 (5-19):        Every 5 epochs
        # Phase 3 (20-99):       Every 3 epochs
        # Phase 4 (100+):        Every epoch
        # MSE-only for first 100 epochs. Plotting starts at epoch 20 on new best.
        def should_validate(ep):
            if ep < 5:
                return False
            elif ep < 20:
                return (ep % 5 == 0)
            elif ep < 100:
                return (ep % 3 == 0)
            else:
                return True

        if not should_validate(epoch):
            if accelerator.is_main_process:
                next_val = next((e for e in range(epoch+1, epoch+200) if should_validate(e)), epoch+1)
                print(f"⏭️  Epoch {epoch}: Skipping validation (schedule: next at {next_val}).")
            epochs_done_this_run += 1
            continue

        if accelerator.is_main_process:
            print(f"\n⌛ Epoch {epoch} complete. Starting Fast Validation (Noise MSE)...")
        
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_geos = batch['x_geos'].to(device)
                x_obs  = batch['x_obs'].to(device)
                
                B = x_geos.shape[0]
                H, W = x_obs.shape[-2], x_obs.shape[-1]
                x_geos_flat = x_geos.contiguous().view(B, -1, H, W)
                
                months = batch['month'].to(device)
                sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
                cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
                
                lead_idx = batch['lead_idx'].to(device)
                lead_val = (lead_idx.float() / 1.5) - 1.0 
                lead_channel = lead_val.view(B, 1, 1, 1).expand(B, 1, H, W)
                
                x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1)
                target_norm = batch['y_target'].to(device)  # [B, 2, H, W]
                
                t = flow_matcher.sample_time_batch(B)
                noise = torch.randn_like(target_norm)
                x_t, v_target = flow_matcher.interpolate(target_norm, noise, t)
                
                v_pred, _ = model(x_t, x_cond, t, lead_idx=lead_idx)
                
                w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
                temp_weights = w_escalation[lead_idx].view(B, 1, 1, 1).expand(-1, 2, -1, -1)
                
                loss_val = (area_weights * land_ocean_weights * temp_weights * (v_pred - v_target)**2).mean()
                val_loss_total += loss_val.item()
                val_steps += 1
                
        current_val_metric = val_loss_total / max(1, val_steps)
        
        if accelerator.is_main_process:
            print(f"✅ Validation Complete. Avg Noise MSE Loss: {current_val_metric:.4f}")

            # 2. Check if this is a new best model
            is_new_best = (current_val_metric < best_val_loss)
            
            if is_new_best:
                print(f"🏆 NEW BEST MODEL! Val Loss: {current_val_metric:.4f} (Previous: {best_val_loss:.4f})")
                best_val_loss = current_val_metric

                # Save checkpoint with descriptive name
                new_best_name = f"best_model_epoch_{epoch}_loss_{current_val_metric:.4f}.pt"
                new_best_path = os.path.join(output_dir, new_best_name)
                
                unwrapped_model = accelerator.unwrap_model(model)
                best_ckpt = {
                    'epoch': epoch,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                
                torch.save(best_ckpt, new_best_path)
                torch.save(best_ckpt, os.path.join(output_dir, "best_flow_ckpt.pt"))
                
                # Update JSON registry (append + re-sort)
                registry_path = os.path.join(output_dir, "model_registry.json")
                if os.path.exists(registry_path):
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                else:
                    registry = []
                
                registry.append({
                    "rank": 0,  # Will be recalculated
                    "path": new_best_path,
                    "val_loss": current_val_metric,
                    "epoch": epoch
                })
                # Sort by val_loss (best first) and assign ranks
                registry.sort(key=lambda x: x['val_loss'])
                for i, entry in enumerate(registry):
                    entry['rank'] = i + 1
                
                with open(registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
                
                print(f"📋 Model Registry updated: {len(registry)} models tracked.")
                top_str = ", ".join([f"c{e['rank']}:E{e['epoch']}({e['val_loss']:.4f})" for e in registry[:5]])
                print(f"📍 Top 5: [{top_str}]")
                
                # Generate validation plot only after epoch 20 on new absolute best
                if epoch >= 20:
                    print(f"📊 Generating validation plot for new best at Epoch {epoch}...")
                    val_outputs = run_val_inference(
                        epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                        target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, 
                        is_test=False, is_fast_recon=True,
                        use_flow_variance=False,
                        eof_bases=eof_bases, nao_bases=nao_bases, nao_lookup=nao_lookup,
                        enso_bases=enso_bases, oni_lookup=oni_lookup, mjo_df=mjo_df
                    )
                    v_met, v_pred, v_target, v_rmse, v_geos_mean, v_geos_crps, v_geos_rmse, v_ai_res, v_geos_single, v_model_single, v_model_var = val_outputs
                    save_val_plot(epoch, v_pred, v_target, v_met, v_rmse, v_geos_mean, v_geos_crps, v_geos_rmse, output_dir, 
                                  ai_residual=v_ai_res, suffix="best", geos_single=v_geos_single, model_single=v_model_single, model_var=v_model_var)
                    print(f"📸 Validation plot saved for Epoch {epoch}.")

            # Save Latest Checkpoint & Logs
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt = {
                'epoch': epoch,
                'model': unwrapped_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_flow_ckpt.pt"))

            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, 0.0, current_val_metric])

        # Track progress for this execution session
        epochs_done_this_run += 1

def main():
    parser = argparse.ArgumentParser(description="Train or Test Flow Matching Model")
    parser.add_argument("--config", type=str, default="ml_model/config_flow.yaml")
    parser.add_argument("--test", action="store_true", help="Run in inference/test mode only")
    parser.add_argument("--ckpt", type=str, default="best_flow_ckpt.pt", 
                        help="Checkpoint filename in output_dir to load for testing (default: best_flow_ckpt.pt)")
    parser.add_argument("--ckpt-rank", type=int, default=None,
                        help="Load the Nth best model from model_registry.json (e.g., --ckpt-rank 1 for best, --ckpt-rank 3 for 3rd best)")
    parser.add_argument("--full-val", action="store_true", help="Force full reverse sampling validation (1000 steps) for all validation epochs.")
    parser.add_argument("--epochs-per-run", type=int, default=10000, 
                        help="Number of epochs to run before exiting gracefully (useful for job chaining)")
    parser.add_argument("--reset-variance", action="store_true", help="Force wipe the Variance Heads back to 0.0 (1.0x multiplier target) on load.")
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
