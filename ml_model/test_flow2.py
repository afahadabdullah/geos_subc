import os
import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import csv
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
from datetime import datetime

# Local Modules
from dataset_flow import S2SHybridDataset
from flow_matching import FlowMatchingModel, CustomFlowMatcher

def get_area_weights(lats, device):
    lats_rad = np.deg2rad(lats)
    weights = np.cos(lats_rad)
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()  # Normalize to sum to 1 (matches compare_noise.py)
    weights_tensor = torch.from_numpy(weights).float().to(device)
    weights_tensor = weights_tensor.view(1, 1, len(lats), 1)
    return weights_tensor

def compute_crps(ensemble_preds, target, area_weights):
    """
    Computes CRPS for a small ensemble.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    """
    mask = ~torch.isnan(target) # [B, C, H, W]
    if not mask.any():
        return 0.0
    
    E = ensemble_preds.shape[0]
    diff = torch.abs(ensemble_preds - target.unsqueeze(0))
    mae_term = diff.mean(dim=0)
    
    spread_term = torch.zeros_like(mae_term)
    if E > 1:
        for i in range(E):
            for j in range(E):
                spread_term += torch.abs(ensemble_preds[i] - ensemble_preds[j])
        spread_term = spread_term / (2 * E * E)
    
    crps_map = mae_term - spread_term
    crps_map_clean = torch.where(mask, crps_map, torch.zeros_like(crps_map))
    weights_clean = torch.where(mask, area_weights, torch.zeros_like(area_weights))
    
    weighted_crps = (crps_map_clean * weights_clean).sum() / (weights_clean.sum() + 1e-8)
    return weighted_crps.item(), crps_map_clean

def save_test_plot(batch_idx, full_pred, true_target_precip, model_crps, model_rmse, geos_pred, geos_crps, geos_rmse, output_dir, geos_single, model_single, lats, lons):
    """
    Standardizes plotting logic for testing results (6-column layout with Cartopy).
    """
    t_img = true_target_precip[0].cpu().numpy()
    p_img = full_pred[0].cpu().numpy()
    g_img = geos_pred[0].cpu().numpy()
    g_sing_img = geos_single[0].cpu().numpy()
    m_sing_img = model_single[0].cpu().numpy()
    
    # We use PlateCarree for flat equirectangular projection, or use Robinson for something fancy
    proj = ccrs.PlateCarree()
    
    fig, axes = plt.subplots(4, 6, figsize=(32, 18), subplot_kw={'projection': proj})
    
    # Pre-calculate extent to keep things bounded [lon_min, lon_max, lat_min, lat_max]
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    for l in range(4):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        
        # Helper to style the axes
        def style_ax(ax, title):
            ax.set_title(title, fontsize=10)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if ax.get_subplotspec().colspan.start > 0:
                gl.left_labels = False # Only show y-axis on far left
        
        # Col 1: Target
        ax0 = axes[l, 0]
        im0 = ax0.imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        style_ax(ax0, "Target GPCP" if l == 0 else "")
        if l == 0: ax0.set_title("Target GPCP")
        
        # Col 2: Single GEOS Ens Member
        ax1 = axes[l, 1]
        im1 = ax1.imshow(g_sing_img[l], cmap='Blues', vmin=t_min, vmax=t_max, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        style_ax(ax1, "GEOS (Single Ens Member)" if l == 0 else "")
        
        # Col 3: Single Model Ens Member
        ax2 = axes[l, 2]
        im2 = ax2.imshow(m_sing_img[l], cmap='Blues', vmin=t_min, vmax=t_max, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        style_ax(ax2, "Model (Single Ens Member)" if l == 0 else "")
        
        # Col 4: GEOS ens mean - Target
        ax3 = axes[l, 3]
        diff_geos = g_img[l] - t_img[l]
        im3 = ax3.imshow(diff_geos, cmap='RdBu_r', vmin=-30, vmax=30, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        style_ax(ax3, f"GEOS Bias (GEOS Mean - Target)\nCRPS:{geos_crps:.2f}, RMSE:{geos_rmse:.2f}" if l == 0 else "")
        
        # Col 5: Model ens mean - Target
        ax4 = axes[l, 4]
        diff_model = p_img[l] - t_img[l]
        im4 = ax4.imshow(diff_model, cmap='RdBu_r', vmin=-30, vmax=30, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        style_ax(ax4, f"Model Bias (Model Mean - Target)\nCRPS:{model_crps:.2f}, RMSE:{model_rmse:.2f}" if l == 0 else "")
        
        # Col 6: Closeness plot: abs(GEOS Bias) - abs(Model Bias)
        ax5 = axes[l, 5]
        closeness = np.abs(diff_geos) - np.abs(diff_model)
        im5 = ax5.imshow(closeness, cmap='PiYG', vmin=-25, vmax=25, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        style_ax(ax5, "Closeness: |GEOS Bias| - |Model Bias|\nGreen (>0) = Model Better, Pink (<0) = GEOS Better" if l == 0 else "")

    os.makedirs(os.path.join(output_dir, "test_plots2"), exist_ok=True)
    plt.tight_layout()
    # Cartopy gridlines can complain if layout is excessively tight
    fig.subplots_adjust(hspace=0.1, wspace=0.1) 
    filename = f"test_M3_2015_idx{batch_idx}_score_{model_crps:.4f}.png"
    plt.savefig(os.path.join(output_dir, "test_plots2", filename), bbox_inches='tight', dpi=150)
    plt.close()

import noise_utils
def run_test_inference(batch_idx, batch, model, flow_matcher, device, output_dir, log_file, 
                      target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, lons, lats, num_ensemble=20, num_steps=50, save_plot=True, 
                      eof_bases=None, nao_bases=None, nao_lookup=None, enso_bases=None, oni_lookup=None, mjo_df=None, year=None):
    model.eval()
    
    fb_target_norm = batch['y_target'].to(device) # [vB, 1, H, W]
    vB, _, H, W = fb_target_norm.shape
    num_inits = vB // 4
    
    true_target_precip = batch['target_raw_full'][0::4].to(device) # [num_inits, 4, H, W]
    
    geos_ens_raw = batch['geos_ens_raw'].to(device) # [vB, M, L, H, W]
    geos_ens_sample = geos_ens_raw[0::4] # [num_inits, M, L, H, W]
    geos_mean_raw = geos_ens_sample.mean(dim=1) # [num_inits, 4, H, W]
    
    # ---------------------------------------------------------------------------------
    # STRATEGY 3: GEOS-INFORMED COVARIANCE NOISE SCALING
    # Instead of drawing pure N(0,1) noise, we scale the noise by the spatial variance
    # of the GEOS ensemble to force the model to explore uncertainty exactly where
    # the physics model is uncertain.
    # ---------------------------------------------------------------------------------
    # Calculate GEOS structural variance across the 4 members, for all leads [num_inits, 4, H, W]
    geos_struct_var = geos_ens_sample.var(dim=1) # [num_inits, 4, H, W]
    
    # --- VRAM GPU BATCHING: Solve all Lead Weeks and Ensemble members simultaneously ---
    vB = batch['x_obs'].shape[0] # Should be num_inits * 4
    
    geos_struct_var = geos_struct_var.view(vB, H, W) # [vB, H, W]
    
    fx_obs = batch['x_obs'].to(device) # [vB, C, H, W]
    fx_geos = batch['x_geos'].to(device) # [vB, 1, 4, H, W]
    fx_geos_cat = fx_geos.view(vB, -1, H, W)
    
    f_month = batch['month'].to(device).float()
    fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fl_idx = batch['lead_idx'].to(device).float()
    f_lead_val = (fl_idx / 1.5) - 1.0 
    f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fx_cond = torch.cat([fx_obs, fx_geos_cat, fsin_month, fcos_month, f_lead_channel], dim=1) # [vB, 35, H, W]
    
    # Build GEOS variance scaled noise over the entire vB sequence
    lead_var = geos_struct_var.unsqueeze(1) 
    var_mean = lead_var.view(vB, -1).mean(dim=1).view(vB, 1, 1, 1)
    var_norm = lead_var / (var_mean + 1e-6)
    var_scaled = torch.clamp(var_norm, min=0.5, max=2.0) 

    # --- VRAM GPU BATCHING: Solve all Lead Weeks and Ensemble members SIMULTANEOUSLY ---
    # With zombie processes gone, we can fit [vB * num_ensemble] through the UNet at once.
    
    # Expand condition to [vB, num_ensemble, 35, H, W] -> [vB * num_ensemble, 35, H, W]
    fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W)
    
    # Expand variance scalar for base noise
    var_scaled_expanded = var_scaled.unsqueeze(1).expand(vB, num_ensemble, 1, H, W).reshape(vB * num_ensemble, 1, H, W)
    
    # Generate noise: use EOF-structured noise if available, otherwise GEOS-variance-scaled
    if eof_bases is not None:
        smart_noise_expanded = noise_utils.generate_dynamic_multimodal_noise(
            batch, num_ensemble, device, eof_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, year
        )
    else:
        base_noise_expanded = torch.randn((vB * num_ensemble, 1, H, W), device=device)
        smart_noise_expanded = base_noise_expanded * var_scaled_expanded
        
    lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
    
    # Single parallel ODE solve for the entire test batch and ensemble
    p_x1_expanded = flow_matcher.euler_solve(
        model, smart_noise_expanded, fx_cond_expanded, 
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=True
    )
    
    # Reshape back to [vB, num_ensemble, H, W]
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, H, W)
    
    week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0) # [vB, num_ensemble, H, W]
    
    ensemble_preds_precip = week_precip.transpose(0, 1) # [num_ensemble, vB, H, W]
    
    # Reshape to separate initialization dates and lead weeks
    ensemble_preds_precip = ensemble_preds_precip.view(num_ensemble, num_inits, 4, H, W)
    full_pred = ensemble_preds_precip.mean(dim=0) # [num_inits, 4, H, W]
    
    val_metric, model_crps_map = compute_crps(ensemble_preds_precip, true_target_precip, area_weights)
    
    mse_map = (full_pred - true_target_precip)**2
    mask = ~torch.isnan(mse_map)
    if mask.any():
        aw_expanded = area_weights.view(1, 181, 1).expand_as(mse_map)
        model_rmse = torch.sqrt((mse_map[mask] * aw_expanded[mask]).sum() / (aw_expanded[mask].sum() + 1e-8)).item()
    else:
        model_rmse = 0.0
    
    geos_crps, geos_crps_map = compute_crps(geos_ens_sample.transpose(0, 1), true_target_precip, area_weights)
    geos_mse_map = (geos_mean_raw - true_target_precip)**2
    mask_2d = ~torch.isnan(geos_mse_map)

    if mask_2d.any():
        aw_expanded_2d = area_weights.view(1, 1, 181, 1).expand_as(geos_mse_map)
        geos_rmse = torch.sqrt((geos_mse_map[mask_2d] * aw_expanded_2d[mask_2d]).sum() / (aw_expanded_2d[mask_2d].sum() + 1e-8)).item()
    else:
        geos_rmse = 0.0
    
    true_target_precip_plot = torch.nan_to_num(true_target_precip, nan=0.0)
    
    if save_plot:
        # Generate diagnostic plot for this batch
        save_test_plot(batch_idx, full_pred, true_target_precip_plot, val_metric, model_rmse, 
                       geos_mean_raw, geos_crps, geos_rmse, output_dir, 
                       geos_single=geos_ens_sample[:, 0], model_single=ensemble_preds_precip[0],
                       lats=lats, lons=lons)
                   
    tensors = {
        'full_pred': full_pred,
        'true_target_precip': true_target_precip_plot,
        'geos_mean': geos_mean_raw,
        'geos_single': geos_ens_sample[:, 0], # [num_inits, L, H, W]
        'model_single': ensemble_preds_precip[0], # [num_inits, L, H, W]
        'model_crps_map': model_crps_map,
        'model_mse_map': mse_map,
        'geos_crps_map': geos_crps_map,
        'geos_mse_map': geos_mse_map
    }
    return val_metric, model_rmse, geos_crps, geos_rmse, tensors, num_inits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_flow.yaml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint (auto-discovers best if None)")
    parser.add_argument("--year", type=int, default=2015, help="Test year to validate against")
    parser.add_argument("--ensemble-size", type=int, default=4, help="Number of members in smart noise ensemble")
    parser.add_argument("--steps", type=int, default=50, help="Number of Euler steps for ODE solve")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Flow Matcher on {device} using Test Year {args.year}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "ml_output_flow2")
    os.makedirs(output_dir, exist_ok=True)
    
    output_data_dir = os.path.join(output_dir, f"test_data2_{args.year}_N{args.ensemble_size}")
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Provide 12 initialization dates (1 per month)
    rng = random.Random(42)  # Fixed seed for deterministic test samples
    target_batches = set()
    total_expected_batches = 52 # 1 year = 52 weeks
    chunk_size = total_expected_batches / 12.0
    
    for m in range(12):
        start_idx = int(m * chunk_size)
        end_idx = int((m + 1) * chunk_size) - 1
        # Handle decimal boundary cases for the last month
        if m == 11:
            end_idx = total_expected_batches - 1
        # Pick one random initialization batch from this 1-month window
        target_batches.add(rng.randint(start_idx, max(start_idx, end_idx)))

    # Check cache integrity (we just added spatial maps so old caches will crash)
    for b_idx in target_batches:
        df = os.path.join(output_data_dir, f"batch_{b_idx}.npz")
        if os.path.exists(df):
            try:
                with np.load(df) as f:
                    _ = f['model_crps_map']
            except (KeyError, Exception):
                print(f"Purging outdated cache: {df}")
                os.remove(df)
                
    all_cached = all([os.path.exists(os.path.join(output_data_dir, f"batch_{b_idx}.npz")) for b_idx in target_batches])

    # Load lat values for area weights
    import xarray as xr
    geos_sample_path = os.path.join(config["data_dir"], f"geos_subc_{args.year}.zarr")
    ds_geos = xr.open_zarr(geos_sample_path, consolidated=False)
    lats = ds_geos.Y.values
    lons = ds_geos.X.values
    area_weights = get_area_weights(lats, device)

    if all_cached:
        print(f"\n✅ Found all 12 pre-computed test batches in {output_data_dir}.")
        print("Skipping dataset load and model weights. Proceeding directly to analysis!")
        test_iterator = range(12)
        model = None
        flow_matcher = None
        target_sqrt_min = 0.0
        target_sqrt_max = 7.071
        geos_min = 0.0
        geos_max = 0.0
    else:
        # Force dataset to only load the test year
        test_dataset = S2SHybridDataset(
            data_root=config["data_dir"],
            start_year=args.year,
            end_year=args.year,
            normalize=True,
            preload=config.get("preload", False),
            stats_file="v5_global_stats.pt"
        )
    
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset, batch_size=4, shuffle=False, 
            num_workers=2, pin_memory=True
        )
        test_iterator = test_loader
    
        stats_file = "ml_model/v5_global_stats.pt"
        global_bounds = torch.load(stats_file, weights_only=True)
        target_sqrt_min = 0.0
        target_sqrt_max = 7.071
        geos_min = global_bounds["geos_raw"]["min"]
        geos_max = global_bounds["geos_raw"]["max"]
    
        model = FlowMatchingModel(in_channels=36, out_channels=1).to(device)
        
        if args.ckpt is None:
            import glob
            model_paths = glob.glob(os.path.join(output_dir, "best_model_epoch_*_crps_*.pt"))
            if not model_paths:
                raise FileNotFoundError(f"No best models found in {output_dir}")
            
            def extract_crps(path):
                filename = os.path.basename(path).replace('.pt', '')
                try:
                    return float(filename.split('_crps_')[-1])
                except:
                    return 999.0
                    
            best_ckpt = min(model_paths, key=extract_crps)
            print(f"🔍 Auto-discovered best top model: {os.path.basename(best_ckpt)}")
            args.ckpt = best_ckpt
            
        print(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        flow_matcher = CustomFlowMatcher(device=device)
    
    import noise_utils
    # Load MJO EOF bases for physically structured ensemble noise
    eof_bases_path = os.path.join(os.path.dirname(__file__), "mjo_eof_bases.pt")
    eof_bases = None
    if os.path.exists(eof_bases_path):
        eof_data = torch.load(eof_bases_path, map_location='cpu', weights_only=False)
        eof_bases = eof_data['eof_bases']
        print(f"✅ Loaded MJO EOF bases: {eof_data['n_eofs']} EOFs/phase")
    else:
        print(f"⚠️ MJO EOF bases not found. Using GEOS-variance-scaled noise.")
        
    # NAO
    nao_eof_path = os.path.join(os.path.dirname(__file__), "nao_eof_bases.pt")
    nao_idx_path = os.path.join(config["data_dir"], "norm.daily.nao.index.b500101.current.ascii")
    nao_bases, nao_lookup = None, None
    if os.path.exists(nao_eof_path) and os.path.exists(nao_idx_path):
        nao_data = torch.load(nao_eof_path, map_location='cpu', weights_only=False)
        nao_bases = nao_data['eof_bases']
        nao_lookup = noise_utils.parse_nao_index(nao_idx_path)
        print(f"✅ Loaded NAO EOFs")
        
    # ENSO
    enso_eof_path = os.path.join(os.path.dirname(__file__), "enso_eof_bases.pt")
    oni_idx_path = os.path.join(config["data_dir"], "oni.ascii.txt")
    enso_bases, oni_lookup = None, None
    if os.path.exists(enso_eof_path) and os.path.exists(oni_idx_path):
        enso_data = torch.load(enso_eof_path, map_location='cpu', weights_only=False)
        enso_bases = enso_data['eof_bases']
        oni_lookup = noise_utils.parse_oni_index(oni_idx_path)
        print(f"✅ Loaded ENSO EOFs")
        
    # MJO RMM CSV
    mjo_df = None
    mjo_csv_path = os.path.join(config["data_dir"], "mjo_processed.csv")
    if os.path.exists(mjo_csv_path):
        import pandas as pd
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['S'])
        mjo_df['date_str'] = mjo_df['S'].dt.strftime('%Y-%m-%d')
        mjo_df = mjo_df.set_index('date_str')
        print(f"✅ Loaded MJO RMM CSV")

    csv_file = os.path.join(output_dir, f"test_metrics_50step_{args.year}_N{args.ensemble_size}.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Batch_Idx", "Model_CRPS", "Model_RMSE", "GEOS_CRPS", "GEOS_RMSE"])
        
    # ------------------ TESTING LOOP ------------------
    all_model_crps, all_model_rmse = [], []
    all_geos_crps, all_geos_rmse = [], []
    
    # Storage for temporal correlation & spatial error maps
    all_full_preds = []
    all_targets = []
    all_geos_means = []
    
    all_model_crps_maps = []
    all_model_mse_maps = []
    all_geos_crps_maps = []
    all_geos_mse_maps = []
    pbar = tqdm(test_iterator, desc=f"Testing {args.year}", leave=True)
    for batch_idx, batch in enumerate(pbar):
        if len(all_model_crps) >= 12:
            break
            
        if not all_cached and batch_idx not in target_batches:
            if batch['y_target'].shape[0] != 4:
                continue
            continue
            
        # When all_cached=True, test_iterator is range(12), so batch_idx is 0..11
        # We map these sequential indices back to their true randomized batch names to load the right files
        if all_cached:
            actual_b_idx = sorted(list(target_batches))[batch_idx]
            batch_idx = actual_b_idx
            # Dummy batch
            batch = {'y_target': torch.zeros((4, 1, 1, 1))}
            
        if not all_cached:
            # We process tests sample by sample for accurate ensemble aggregation
            # Ensure we only pass one distinct init date (which is flattened into 4 leads by the dataset batcher)
            if batch['y_target'].shape[0] != 4:
                continue
            
        data_file = os.path.join(output_data_dir, f"batch_{batch_idx}.npz")
        
        # Check if we already ran this batch
        if os.path.exists(data_file):
            # Load from disk
            data = np.load(data_file)
            m_crps = float(data['model_crps'])
            m_rmse = float(data['model_rmse'])
            g_crps = float(data['geos_crps'])
            g_rmse = float(data['geos_rmse'])
            
            f_pred = data['full_pred']
            t_target = data['true_target_precip']
            g_mean = data['geos_mean']
            
            m_crps_map = data['model_crps_map']
            m_mse_map = data['model_mse_map']
            g_crps_map = data['geos_crps_map']
            g_mse_map = data['geos_mse_map']
            
            # Reconstruct tensors for plotting first few batches
            if batch_idx < 5:
                # Need spatial maps for debugging if desired, but plotting tests mostly need images
                full_pred = torch.from_numpy(f_pred)
                true_target_precip_plot = torch.from_numpy(t_target)
                geos_mean_raw = torch.from_numpy(g_mean)
                geos_single = torch.from_numpy(data['geos_single'])
                model_single = torch.from_numpy(data['model_single'])
                
                save_test_plot(batch_idx, full_pred, true_target_precip_plot, m_crps, m_rmse, 
                               geos_mean_raw, g_crps, g_rmse, output_dir, 
                               geos_single=geos_single, model_single=model_single,
                               lats=lats, lons=lons)
        else:
            # Run inference
            m_crps, m_rmse, g_crps, g_rmse, tensors, n_inits = run_test_inference(
                batch_idx, batch, model, flow_matcher, device, output_dir, None,
                target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, lons, lats, 
                num_ensemble=args.ensemble_size, num_steps=args.steps, save_plot=(batch_idx < 5),
                eof_bases=eof_bases, nao_bases=nao_bases, nao_lookup=nao_lookup, 
                enso_bases=enso_bases, oni_lookup=oni_lookup, mjo_df=mjo_df, year=args.year
            )
            
            f_pred = tensors['full_pred'].cpu().numpy()
            t_target = tensors['true_target_precip'].cpu().numpy()
            g_mean = tensors['geos_mean'].cpu().numpy()
            
            m_crps_map = tensors['model_crps_map'].cpu().numpy()
            m_mse_map = tensors['model_mse_map'].cpu().numpy()
            g_crps_map = tensors['geos_crps_map'].cpu().numpy()
            g_mse_map = tensors['geos_mse_map'].cpu().numpy()
            
            # Save to disk to avoid rerunning
            np.savez_compressed(
                data_file,
                model_crps=m_crps / n_inits,
                model_rmse=m_rmse / n_inits,
                geos_crps=g_crps / n_inits,
                geos_rmse=g_rmse / n_inits,
                num_inits=n_inits,
                full_pred=f_pred,
                true_target_precip=t_target,
                geos_mean=g_mean,
                geos_single=tensors['geos_single'].cpu().numpy(),
                model_single=tensors['model_single'].cpu().numpy(),
                model_crps_map=m_crps_map,
                model_mse_map=m_mse_map,
                geos_crps_map=g_crps_map,
                geos_mse_map=g_mse_map
            )
            
        all_full_preds.append(f_pred)
        all_targets.append(t_target)
        all_geos_means.append(g_mean)
        all_model_crps_maps.append(m_crps_map)
        all_model_mse_maps.append(m_mse_map)
        all_geos_crps_maps.append(g_crps_map)
        all_geos_mse_maps.append(g_mse_map)
        
        all_model_crps.append(m_crps)
        all_model_rmse.append(m_rmse)
        all_geos_crps.append(g_crps)
        all_geos_rmse.append(g_rmse)
        
        # Simple progress tracking: count how many batches we've processed
        num_batches_done = len(all_model_crps)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([batch_idx, m_crps, m_rmse, g_crps, g_rmse])
            
        pbar.set_postfix({
            "M_CRPS": f"{sum(all_model_crps) / num_batches_done:.3f}",
            "G_CRPS": f"{sum(all_geos_crps) / num_batches_done:.3f}"
        })

    print("\nCalculating Temporal Correlation Maps...")
    all_full_preds_arr = np.concatenate(all_full_preds, axis=0) # [T, H, W]
    all_targets_arr = np.concatenate(all_targets, axis=0)
    all_geos_means_arr = np.concatenate(all_geos_means, axis=0)
    
    def temporal_correlation(x, y):
        x_mean = np.mean(x, axis=0, keepdims=True)
        y_mean = np.mean(y, axis=0, keepdims=True)
        x_anom = x - x_mean
        y_anom = y - y_mean
        cov = np.sum(x_anom * y_anom, axis=0)
        var_x = np.sum(x_anom**2, axis=0)
        var_y = np.sum(y_anom**2, axis=0)
        corr = cov / (np.sqrt(var_x * var_y) + 1e-8)
        return corr
        
    model_corr_maps_4wk = temporal_correlation(all_full_preds_arr, all_targets_arr) # [4, H, W]
    geos_corr_maps_4wk = temporal_correlation(all_geos_means_arr, all_targets_arr)  # [4, H, W]
    
    # Area-Weighted Correlation Averages
    aw_np = area_weights.squeeze().cpu().numpy() # [H]
    aw_2d = np.broadcast_to(aw_np[:, np.newaxis], (model_corr_maps_4wk.shape[1], model_corr_maps_4wk.shape[2])) # [H, W]
    
    proj = ccrs.PlateCarree()
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    def style_ax_corr(ax, title):
        ax.set_title(title, fontsize=12)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # --- Plot Weekly Correlation Maps ---
    for wk in range(4):
        model_corr_map = model_corr_maps_4wk[wk]
        geos_corr_map = geos_corr_maps_4wk[wk]
        
        mask_model = ~np.isnan(model_corr_map)
        model_avg_corr = np.nansum(model_corr_map[mask_model] * aw_2d[mask_model]) / (np.nansum(aw_2d[mask_model]) + 1e-8)
        
        mask_geos = ~np.isnan(geos_corr_map)
        geos_avg_corr = np.nansum(geos_corr_map[mask_geos] * aw_2d[mask_geos]) / (np.nansum(aw_2d[mask_geos]) + 1e-8)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={'projection': proj})
        
        im0 = axes[0].imshow(geos_corr_map, cmap='RdYlGn', vmin=-1, vmax=1, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        style_ax_corr(axes[0], f"Week {wk+1} GEOS Correlation (Avg: {geos_avg_corr:.3f})")
        
        im1 = axes[1].imshow(model_corr_map, cmap='RdYlGn', vmin=-1, vmax=1, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        style_ax_corr(axes[1], f"Week {wk+1} Model Correlation (Avg: {model_avg_corr:.3f})")
        
        diff_corr = model_corr_map - geos_corr_map
        im2 = axes[2].imshow(diff_corr, cmap='PuOr', vmin=-0.4, vmax=0.4, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        style_ax_corr(axes[2], f"Week {wk+1} Difference: Model - GEOS")
        
        plt.tight_layout()
        corr_filename = os.path.join(output_dir, "test_plots", f"correlation_map_{args.year}_N{args.ensemble_size}_wk{wk+1}.png")
        plt.savefig(corr_filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    # --- Generate Weekly CRPS & RMSE Map Plots ---
    print("Calculating and Plotting Spatial CRPS and RMSE Maps per lead week...")
    all_model_crps_maps_arr = np.concatenate(all_model_crps_maps, axis=0) # [T, 4, H, W]
    all_geos_crps_maps_arr = np.concatenate(all_geos_crps_maps, axis=0)
    all_model_mse_maps_arr = np.concatenate(all_model_mse_maps, axis=0)
    all_geos_mse_maps_arr = np.concatenate(all_geos_mse_maps, axis=0)

    avg_model_crps_maps_4wk = np.mean(all_model_crps_maps_arr, axis=0) # [4, H, W]
    avg_geos_crps_maps_4wk = np.mean(all_geos_crps_maps_arr, axis=0) 
    
    avg_model_rmse_maps_4wk = np.sqrt(np.mean(all_model_mse_maps_arr, axis=0))
    avg_geos_rmse_maps_4wk = np.sqrt(np.mean(all_geos_mse_maps_arr, axis=0))
    
    def plot_error_map(geos_map, model_map, metric_name, vmin, vmax, diff_vmax, filename, title_prefix):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={'projection': proj})
        
        im0 = axes[0].imshow(geos_map, cmap='OrRd', vmin=vmin, vmax=vmax, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        style_ax_corr(axes[0], f"{title_prefix} GEOS {metric_name} Error")
        
        im1 = axes[1].imshow(model_map, cmap='OrRd', vmin=vmin, vmax=vmax, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        style_ax_corr(axes[1], f"{title_prefix} Model {metric_name} Error")
        
        diff_err = geos_map - model_map
        im2 = axes[2].imshow(diff_err, cmap='PiYG', vmin=-diff_vmax, vmax=diff_vmax, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        style_ax_corr(axes[2], f"{title_prefix} {metric_name} Impr: GEOS-Model (>0 Model Better)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "test_plots", filename), bbox_inches='tight', dpi=150)
        plt.close()
        
    for wk in range(4):
        plot_error_map(avg_geos_crps_maps_4wk[wk], avg_model_crps_maps_4wk[wk], "CRPS", 0, 8, 3, f"crps_map_{args.year}_N{args.ensemble_size}_wk{wk+1}.png", f"Week {wk+1}")
        plot_error_map(avg_geos_rmse_maps_4wk[wk], avg_model_rmse_maps_4wk[wk], "RMSE", 0, 15, 5, f"rmse_map_{args.year}_N{args.ensemble_size}_wk{wk+1}.png", f"Week {wk+1}")

    # Final Area-Weighted Global Error Scorings from the multi-week average Maps
    avg_model_crps_map = np.mean(avg_model_crps_maps_4wk, axis=0) # [H, W]
    avg_geos_crps_map = np.mean(avg_geos_crps_maps_4wk, axis=0)
    avg_model_rmse_map = np.sqrt(np.mean(np.mean(all_model_mse_maps_arr, axis=0), axis=0))
    avg_geos_rmse_map = np.sqrt(np.mean(np.mean(all_geos_mse_maps_arr, axis=0), axis=0))

    m_crps_mask = ~np.isnan(avg_model_crps_map)
    avg_model_crps_test = np.nansum(avg_model_crps_map[m_crps_mask] * aw_2d[m_crps_mask]) / (np.nansum(aw_2d[m_crps_mask]) + 1e-8)
    
    g_crps_mask = ~np.isnan(avg_geos_crps_map)
    avg_geos_crps_test = np.nansum(avg_geos_crps_map[g_crps_mask] * aw_2d[g_crps_mask]) / (np.nansum(aw_2d[g_crps_mask]) + 1e-8)
    
    m_rmse_mask = ~np.isnan(avg_model_rmse_map)
    avg_model_rmse_test = np.nansum(avg_model_rmse_map[m_rmse_mask] * aw_2d[m_rmse_mask]) / (np.nansum(aw_2d[m_rmse_mask]) + 1e-8)
    
    g_rmse_mask = ~np.isnan(avg_geos_rmse_map)
    avg_geos_rmse_test = np.nansum(avg_geos_rmse_map[g_rmse_mask] * aw_2d[g_rmse_mask]) / (np.nansum(aw_2d[g_rmse_mask]) + 1e-8)
    
    crps_skill = (1.0 - (avg_model_crps_test / avg_geos_crps_test)) * 100.0
    rmse_skill = (1.0 - (avg_model_rmse_test / avg_geos_rmse_test)) * 100.0

    print("\n==================================")
    print(f"      TEST YEAR {args.year} RESULTS      ")
    print("==================================")
    print(f"GEOS Baseline Average CRPS: {avg_geos_crps_test:.4f}")
    print(f"Model Smart-Ens Average CRPS: {avg_model_crps_test:.4f}")
    print(f"--> CRPS Skill Score: {crps_skill:.2f}% improvement")
    print(f"GEOS Baseline Average RMSE: {avg_geos_rmse_test:.4f}")
    print(f"Model Smart-Ens Average RMSE: {avg_model_rmse_test:.4f}")
    print(f"--> RMSE Skill Score: {rmse_skill:.2f}% improvement")
    print(f"GEOS Baseline Average Corr: {geos_avg_corr:.4f}")
    print(f"Model Smart-Ens Average Corr: {model_avg_corr:.4f}")
    print("==================================")

if __name__ == "__main__":
    main()
