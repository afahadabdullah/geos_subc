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
import xarray as xr
import pandas as pd

# Local Modules
from dataset_flow import S2SHybridDataset
from flow_matching import FlowMatchingModel, CustomFlowMatcher
import noise_utils

def get_area_weights(lats, device):
    lats_rad = np.deg2rad(lats)
    weights = np.cos(lats_rad)
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    weights_tensor = torch.from_numpy(weights).float().to(device)
    weights_tensor = weights_tensor.view(1, 1, len(lats), 1)
    return weights_tensor

def compute_crps(ensemble_preds, target, area_weights, mask_2d=None):
    """
    Computes CRPS for a small ensemble.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    mask_2d: Optional boolean mask [H, W] to filter spatial regions (e.g. Land/Ocean)
    """
    nan_mask = ~torch.isnan(target) # [B, C, H, W]
    if not nan_mask.any():
        return 0.0, None
    
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
    
    # Combine NaN mask with optional spatial mask
    final_mask = nan_mask
    if mask_2d is not None:
        final_mask = final_mask & mask_2d.view(1, 1, *mask_2d.shape)
        
    if not final_mask.any():
        return 0.0, crps_map
        
    crps_map_clean = torch.where(final_mask, crps_map, torch.zeros_like(crps_map))
    weights_clean = torch.where(final_mask, area_weights, torch.zeros_like(area_weights))
    
    weighted_crps = (crps_map_clean * weights_clean).sum() / (weights_clean.sum() + 1e-8)
    return weighted_crps.item(), crps_map

def save_test_plot(batch_idx, full_pred, true_target_precip, model_crps, model_rmse, geos_pred, geos_crps, geos_rmse, output_dir, geos_single, model_single, lats, lons):
    t_img = true_target_precip[0].cpu().numpy()
    p_img = full_pred[0].cpu().numpy()
    g_img = geos_pred[0].cpu().numpy()
    g_sing_img = geos_single[0].cpu().numpy()
    m_sing_img = model_single[0].cpu().numpy()
    
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(4, 6, figsize=(32, 18), subplot_kw={'projection': proj})
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    for l in range(4):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        
        def style_ax(ax, title):
            ax.set_title(title, fontsize=10)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            if ax.get_subplotspec().colspan.start > 0:
                gl.left_labels = False
        
        # Col 1: Target
        ax0 = axes[l, 0]
        im0 = ax0.imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        style_ax(ax0, "Target GPCP" if l == 0 else "")
        
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
        
        # Col 4: GEOS Bias
        ax3 = axes[l, 3]
        diff_geos = g_img[l] - t_img[l]
        im3 = ax3.imshow(diff_geos, cmap='RdBu_r', vmin=-30, vmax=30, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        style_ax(ax3, f"GEOS Bias\nCRPS:{geos_crps:.2f}, RMSE:{geos_rmse:.2f}" if l == 0 else "")
        
        # Col 5: Model Bias
        ax4 = axes[l, 4]
        diff_model = p_img[l] - t_img[l]
        im4 = ax4.imshow(diff_model, cmap='RdBu_r', vmin=-30, vmax=30, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        style_ax(ax4, f"Model Bias\nCRPS:{model_crps:.2f}, RMSE:{model_rmse:.2f}" if l == 0 else "")
        
        # Col 6: Closeness
        ax5 = axes[l, 5]
        closeness = np.abs(diff_geos) - np.abs(diff_model)
        im5 = ax5.imshow(closeness, cmap='PiYG', vmin=-25, vmax=25, origin='lower', extent=extent, transform=ccrs.PlateCarree())
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        style_ax(ax5, "Closeness (|GEOS| - |Model|)\nGreen=Model Better" if l == 0 else "")

    os.makedirs(os.path.join(output_dir, "test_plots_v6"), exist_ok=True)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1) 
    filename = f"test_v6_{args.year}_idx{batch_idx}_score_{model_crps:.4f}.png"
    plt.savefig(os.path.join(output_dir, "test_plots_v6", filename), bbox_inches='tight', dpi=150)
    plt.close()

def run_test_inference(batch_idx, batch, model, flow_matcher, device, output_dir, 
                      target_sqrt_min, target_sqrt_max, area_weights, lons, lats, num_ensemble=15, num_steps=50, save_plot=True, 
                      eof_bases=None, nao_bases=None, nao_lookup=None, enso_bases=None, oni_lookup=None, mjo_df=None, year=None,
                      is_land_mask=None):
    model.eval()
    fb_target_norm = batch['y_target'].to(device)
    vB, _, H, W = fb_target_norm.shape
    num_inits = vB // 4
    true_target_precip = batch['target_raw_full'][0::4].to(device)
    geos_ens_raw = batch['geos_ens_raw'].to(device)
    geos_ens_sample = geos_ens_raw[0::4]
    geos_mean_raw = geos_ens_sample.mean(dim=1)
    
    fx_obs = batch['x_obs'].to(device)
    fx_geos = batch['x_geos'].to(device)
    fx_geos_cat = fx_geos.view(vB, -1, H, W)
    f_month = batch['month'].to(device).float()
    fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fl_idx = batch['lead_idx'].to(device).float()
    f_lead_val = (fl_idx / 1.5) - 1.0 
    f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fx_cond = torch.cat([fx_obs, fx_geos_cat, fsin_month, fcos_month, f_lead_channel], dim=1)
    
    total_ensemble = num_ensemble * 2
    fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, total_ensemble, -1, H, W).reshape(vB * total_ensemble, -1, H, W)
    
    # Generate Noise
    noise_base = noise_utils.generate_dynamic_multimodal_noise(
        batch, num_ensemble, device, eof_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, year
    )
    noise_anti = -noise_base
    noise_base_reshaped = noise_base.view(vB, num_ensemble, 1, H, W)
    noise_anti_reshaped = noise_anti.view(vB, num_ensemble, 1, H, W)
    smart_noise_expanded = torch.cat([noise_base_reshaped, noise_anti_reshaped], dim=1).reshape(vB * total_ensemble, 1, H, W)
    
    lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, total_ensemble).reshape(-1).long()
    
    p_x1_expanded = flow_matcher.euler_solve(
        model, smart_noise_expanded, fx_cond_expanded, 
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=True
    )
    
    p_x1_batch = p_x1_expanded.view(vB, total_ensemble, H, W)
    week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
    ensemble_preds_precip = week_precip.transpose(0, 1).view(total_ensemble, num_inits, 4, H, W)
    full_pred = ensemble_preds_precip.mean(dim=0)
    
    def get_metrics(preds, target, weights, mask=None):
        crps_val, crps_map = compute_crps(preds, target, weights, mask_2d=mask)
        mse_map = (preds.mean(dim=0) - target)**2
        nan_mask = ~torch.isnan(mse_map)
        if mask is not None:
            nan_mask = nan_mask & mask.view(1, 1, *mask.shape)
        if nan_mask.any():
            aw_expanded = weights.view(1, 1, 181, 1).expand_as(mse_map)
            rmse = torch.sqrt((mse_map[nan_mask] * aw_expanded[nan_mask]).sum() / (aw_expanded[nan_mask].sum() + 1e-8)).item()
        else:
            rmse = 0.0
        return crps_val, rmse, crps_map, mse_map

    # Overall Global Metrics
    m_crps, m_rmse, m_crps_map, m_mse_map = get_metrics(ensemble_preds_precip, true_target_precip, area_weights)
    g_crps, g_rmse, g_crps_map, g_mse_map = get_metrics(geos_ens_sample.transpose(0, 1), true_target_precip, area_weights)
    
    # Category Breakdowns (Land vs Ocean if Mask available)
    land_m_crps, land_m_rmse = 0.0, 0.0
    ocean_m_crps, ocean_m_rmse = 0.0, 0.0
    land_g_crps, land_g_rmse = 0.0, 0.0
    ocean_g_crps, ocean_g_rmse = 0.0, 0.0
    
    if is_land_mask is not None:
        land_m_crps, land_m_rmse, _, _ = get_metrics(ensemble_preds_precip, true_target_precip, area_weights, mask=is_land_mask)
        ocean_m_crps, ocean_m_rmse, _, _ = get_metrics(ensemble_preds_precip, true_target_precip, area_weights, mask=~is_land_mask)
        land_g_crps, land_g_rmse, _, _ = get_metrics(geos_ens_sample.transpose(0, 1), true_target_precip, area_weights, mask=is_land_mask)
        ocean_g_crps, ocean_g_rmse, _, _ = get_metrics(geos_ens_sample.transpose(0, 1), true_target_precip, area_weights, mask=~is_land_mask)

    true_target_precip_plot = torch.nan_to_num(true_target_precip, nan=0.0)
    if save_plot:
        save_test_plot(batch_idx, full_pred, true_target_precip_plot, m_crps, m_rmse, 
                       geos_mean_raw, g_crps, g_rmse, output_dir, 
                       geos_single=geos_ens_sample[:, 0], model_single=ensemble_preds_precip[0],
                       lats=lats, lons=lons)
                   
    tensors = {
        'full_pred': full_pred,
        'true_target_precip': true_target_precip_plot,
        'geos_mean': geos_mean_raw,
        'geos_single': geos_ens_sample[:, 0],
        'model_single': ensemble_preds_precip[0],
        'model_crps_map': m_crps_map,
        'model_mse_map': m_mse_map,
        'geos_crps_map': g_crps_map,
        'geos_mse_map': g_mse_map
    }
    metrics = {
        'model_crps': m_crps, 'model_rmse': m_rmse,
        'geos_crps': g_crps, 'geos_rmse': g_rmse,
        'land_model_crps': land_m_crps, 'land_model_rmse': land_m_rmse,
        'ocean_model_crps': ocean_m_crps, 'ocean_model_rmse': ocean_m_rmse,
        'land_geos_crps': land_g_crps, 'land_geos_rmse': land_g_rmse,
        'ocean_geos_crps': ocean_g_crps, 'ocean_geos_rmse': ocean_g_rmse
    }
    return metrics, tensors, num_inits

def main():
    parser = argparse.ArgumentParser(description="V6 Land-Ocean Weighted Evaluation")
    parser.add_argument("--config", type=str, default="ml_model/config_flow.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--ensemble-size", type=int, default=15)
    parser.add_argument("--steps", type=int, default=50)
    global args
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "ml_output_flow6")
    os.makedirs(output_dir, exist_ok=True)
    
    test_data_dir = os.path.join(output_dir, f"test_v6_data_{args.year}_N{args.ensemble_size}")
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Select 12 weeks (1 per month)
    rng = random.Random(42)
    target_batches = set()
    for m in range(12):
        target_batches.add(rng.randint(int(m*4.33), int((m+1)*4.33)-1))

    # Mask
    mask_path = os.path.join(os.path.dirname(__file__), "land_ocean_mask_v6.pt")
    is_land_mask = None
    if os.path.exists(mask_path):
        cached = torch.load(mask_path, map_location=device, weights_only=True)
        is_land_mask = cached['is_land'].to(device) # [181, 360]
        print(f"✅ Loaded Land-Ocean Mask for metric breakdown.")

    # Data
    test_dataset = S2SHybridDataset(
        data_root=config["data_dir"], start_year=args.year, end_year=args.year, 
        normalize=True, stats_file="v5_global_stats.pt"
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    lats = np.linspace(-90, 90, 181)
    lons = np.linspace(-180, 179, 360) # Approx
    area_weights = get_area_weights(lats, device)

    # Model
    model = FlowMatchingModel(in_channels=36, out_channels=1).to(device)
    if args.ckpt is None:
        import glob
        cpts = glob.glob(os.path.join(output_dir, "best_model_epoch_*_crps_*.pt"))
        args.ckpt = min(cpts, key=lambda p: float(p.split('_crps_')[-1].replace('.pt', '')))
    
    print(f"🚀 Evaluating: {os.path.basename(args.ckpt)}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    flow_matcher = CustomFlowMatcher(device=device)

    # Teleconnections
    eof_bases = torch.load("ml_model/mjo_eof_bases.pt", map_location='cpu')['eof_bases']
    nao_bases = torch.load("ml_model/nao_eof_bases.pt", map_location='cpu')['eof_bases']
    nao_lookup = noise_utils.parse_nao_index(os.path.join(config["data_dir"], "norm.daily.nao.index.b500101.current.ascii"))
    enso_bases = torch.load("ml_model/enso_eof_bases.pt", map_location='cpu')['eof_bases']
    oni_lookup = noise_utils.parse_oni_index(os.path.join(config["data_dir"], "oni.ascii.txt"))
    mjo_df = pd.read_csv(os.path.join(config["data_dir"], "mjo_processed.csv"), parse_dates=['S']).set_index(pd.to_datetime(pd.read_csv(os.path.join(config["data_dir"], "mjo_processed.csv"), parse_dates=['S'])['S']).dt.strftime('%Y-%m-%d'))

    # Loop
    csv_file = os.path.join(output_dir, f"test_v6_metrics_{args.year}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Batch", "M_CRPS", "G_CRPS", "Land_M_CRPS", "Land_G_CRPS", "Ocean_M_CRPS", "Ocean_G_CRPS"])

    all_m = []
    pbar = tqdm(test_loader, desc=f"Testing V6 {args.year}", total=52//4)
    for b_idx, batch in enumerate(pbar):
        if b_idx not in target_batches or batch['y_target'].shape[0] != 4: continue
        
        m, tens, n_in = run_test_inference(
            b_idx, batch, model, flow_matcher, device, output_dir, 0.0, 7.071, area_weights, lons, lats,
            num_ensemble=args.ensemble_size, num_steps=args.steps, save_plot=(b_idx < 40),
            eof_bases=eof_bases, nao_bases=nao_bases, nao_lookup=nao_lookup, enso_bases=enso_bases, oni_lookup=oni_lookup, 
            mjo_df=mjo_df, year=args.year, is_land_mask=is_land_mask
        )
        all_m.append(m)
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([b_idx, m['model_crps'], m['geos_crps'], m['land_model_crps'], m['land_geos_crps'], m['ocean_model_crps'], m['ocean_geos_crps']])

    # Final Stats
    def avg_key(key): return np.mean([x[key] for x in all_m])
    
    print("\n" + "="*40)
    print(f"      V6 TEST RESULTS ({args.year})      ")
    print("="*40)
    print(f"OVERALL Global CRPS:  Model={avg_key('model_crps'):.4f}, GEOS={avg_key('geos_crps'):.4f}")
    if is_land_mask is not None:
        print(f"LAND CRPS:           Model={avg_key('land_model_crps'):.4f}, GEOS={avg_key('land_geos_crps'):.4f}")
        print(f"OCEAN CRPS:          Model={avg_key('ocean_model_crps'):.4f}, GEOS={avg_key('ocean_geos_crps'):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
