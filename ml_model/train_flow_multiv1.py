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
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False

import argparse
from accelerate import Accelerator

import pandas as pd

# Local Modules
from dataset_flow_multi import S2SHybridDataset
from flow_matching_multi import FlowMatchingModel, CustomFlowMatcher
import noise_utils
import noise_utils_multi

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
    weights_expanded = area_weights.expand_as(crps_map_clean)
    weights_clean = torch.where(mask, weights_expanded, torch.zeros_like(weights_expanded))
    
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


@torch.no_grad()
def euler_solve_chunked(
    flow_matcher,
    model,
    noise,
    x_cond,
    num_steps,
    lead_idx,
    apply_flow_variance=False,
    variance_beta=1.0,
    chunk_size=None,
):
    """
    Memory-safe wrapper for validation/test inference.
    Splits the expanded ensemble batch into smaller chunks so ODE solves do not
    blow up GPU memory when using large validation batches or ensembles.
    """
    if chunk_size is None or chunk_size <= 0 or noise.shape[0] <= chunk_size:
        return flow_matcher.euler_solve(
            model,
            noise,
            x_cond,
            num_steps=num_steps,
            lead_idx=lead_idx,
            apply_flow_variance=apply_flow_variance,
            variance_beta=variance_beta,
        )

    outputs = []
    for start in range(0, noise.shape[0], chunk_size):
        end = min(start + chunk_size, noise.shape[0])
        outputs.append(
            flow_matcher.euler_solve(
                model,
                noise[start:end],
                x_cond[start:end],
                num_steps=num_steps,
                lead_idx=lead_idx[start:end] if lead_idx is not None else None,
                apply_flow_variance=apply_flow_variance,
                variance_beta=variance_beta,
            )
        )
    return torch.cat(outputs, dim=0)


def compute_crps_with_map(ensemble_preds, target, area_weights):
    """
    Computes CRPS and returns both scalar and spatial CRPS map.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    area_weights: [1, 1, H, 1]
    """
    mask = ~torch.isnan(target)
    if not mask.any():
        return 0.0, torch.zeros_like(target)

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
    weights_expanded = area_weights.expand_as(crps_map_clean)
    weights_clean = torch.where(mask, weights_expanded, torch.zeros_like(weights_expanded))
    weighted_crps = (crps_map_clean * weights_clean).sum() / (weights_clean.sum() + 1e-8)
    return weighted_crps.item(), crps_map


def temporal_correlation_maps(x, y):
    """
    x, y: numpy arrays [T, 4, H, W]
    returns: [4, H, W]
    """
    x_mean = np.mean(x, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    x_anom = x - x_mean
    y_anom = y - y_mean
    cov = np.sum(x_anom * y_anom, axis=0)
    var_x = np.sum(x_anom ** 2, axis=0)
    var_y = np.sum(y_anom ** 2, axis=0)
    return cov / (np.sqrt(var_x * var_y) + 1e-8)


def weighted_mean_2d(metric_map, aw_2d):
    mask = ~np.isnan(metric_map)
    if not np.any(mask):
        return 0.0
    return float(np.nansum(metric_map[mask] * aw_2d[mask]) / (np.nansum(aw_2d[mask]) + 1e-8))


def style_cartopy_ax(ax, title, extent, show_left_labels=True):
    ax.set_title(title, fontsize=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.bottom_labels = True


def save_test_plot_cartopy_multi(
    batch_idx,
    init_label,
    output_dir,
    lats,
    lons,
    full_pred_pr,
    true_target_pr,
    geos_mean_pr,
    geos_single_pr,
    model_single_pr,
    model_crps_pr,
    model_rmse_pr,
    geos_crps_pr,
    geos_rmse_pr,
    full_pred_t2m,
    true_target_t2m,
    geos_mean_t2m,
    geos_single_t2m,
    model_single_t2m,
    model_crps_t2m,
    model_rmse_t2m,
    geos_crps_t2m,
    geos_rmse_t2m,
    plot_subdir="test_plots_multi",
):
    if not HAS_CARTOPY:
        return

    proj = ccrs.PlateCarree()
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    fig, axes = plt.subplots(8, 6, figsize=(31, 24), subplot_kw={"projection": proj})

    def draw_panel(row, col, img, cmap, vmin, vmax, title, show_left, ylabel):
        ax = axes[row, col]
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_cartopy_ax(ax, title, extent, show_left_labels=show_left)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")

    for l in range(4):
        pr_target = true_target_pr[l]
        pr_geos_mean = geos_mean_pr[l]
        pr_model_mean = full_pred_pr[l]
        pr_geos_single = geos_single_pr[l]
        pr_model_single = model_single_pr[l]
        pr_title_suffix = f"Wk{l + 1}"
        pr_vmin = float(np.nanmin(pr_target))
        pr_vmax = float(np.nanmax(pr_target))

        draw_panel(l, 0, pr_target, "Blues", pr_vmin, pr_vmax, "PR Target" if l == 0 else "", True, pr_title_suffix)
        draw_panel(l, 1, pr_geos_single, "Blues", pr_vmin, pr_vmax, "PR GEOS Single" if l == 0 else "", False, "")
        draw_panel(l, 2, pr_model_single, "Blues", pr_vmin, pr_vmax, "PR Model Single" if l == 0 else "", False, "")
        draw_panel(
            l,
            3,
            pr_geos_mean - pr_target,
            "RdBu_r",
            -30,
            30,
            f"PR GEOS Bias\nCRPS:{geos_crps_pr:.2f} RMSE:{geos_rmse_pr:.2f}" if l == 0 else "",
            False,
            "",
        )
        draw_panel(
            l,
            4,
            pr_model_mean - pr_target,
            "RdBu_r",
            -30,
            30,
            f"PR Model Bias\nCRPS:{model_crps_pr:.2f} RMSE:{model_rmse_pr:.2f}" if l == 0 else "",
            False,
            "",
        )
        draw_panel(
            l,
            5,
            np.abs(pr_geos_mean - pr_target) - np.abs(pr_model_mean - pr_target),
            "PiYG",
            -25,
            25,
            "PR Closeness\nGreen=Model Better" if l == 0 else "",
            False,
            "",
        )

    for l in range(4):
        row = l + 4
        t2m_target = true_target_t2m[l]
        t2m_geos_mean = geos_mean_t2m[l]
        t2m_model_mean = full_pred_t2m[l]
        t2m_geos_single = geos_single_t2m[l]
        t2m_model_single = model_single_t2m[l]
        t2m_title_suffix = f"Wk{l + 1}"
        t2m_vmin = float(np.nanmin(t2m_target))
        t2m_vmax = float(np.nanmax(t2m_target))

        draw_panel(row, 0, t2m_target, "RdYlBu_r", t2m_vmin, t2m_vmax, "T2M Target" if l == 0 else "", True, t2m_title_suffix)
        draw_panel(row, 1, t2m_geos_single, "RdYlBu_r", t2m_vmin, t2m_vmax, "T2M GEOS Single" if l == 0 else "", False, "")
        draw_panel(row, 2, t2m_model_single, "RdYlBu_r", t2m_vmin, t2m_vmax, "T2M Model Single" if l == 0 else "", False, "")
        draw_panel(
            row,
            3,
            t2m_geos_mean - t2m_target,
            "RdBu_r",
            -10,
            10,
            f"T2M GEOS Bias\nCRPS:{geos_crps_t2m:.2f} RMSE:{geos_rmse_t2m:.2f}" if l == 0 else "",
            False,
            "",
        )
        draw_panel(
            row,
            4,
            t2m_model_mean - t2m_target,
            "RdBu_r",
            -10,
            10,
            f"T2M Model Bias\nCRPS:{model_crps_t2m:.2f} RMSE:{model_rmse_t2m:.2f}" if l == 0 else "",
            False,
            "",
        )
        draw_panel(
            row,
            5,
            np.abs(t2m_geos_mean - t2m_target) - np.abs(t2m_model_mean - t2m_target),
            "PiYG",
            -5,
            5,
            "T2M Closeness\nGreen=Model Better" if l == 0 else "",
            False,
            "",
        )

    os.makedirs(os.path.join(output_dir, plot_subdir), exist_ok=True)
    fig.suptitle(
        f"Multi Test Plot | Init {init_label} | PR CRPS {model_crps_pr:.4f} | T2M CRPS {model_crps_t2m:.4f}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    filename = f"test_multi_idx{batch_idx:03d}_{init_label}_pr{model_crps_pr:.4f}_t2m{model_crps_t2m:.4f}.png"
    plt.savefig(os.path.join(output_dir, plot_subdir, filename), bbox_inches="tight", dpi=150)
    plt.close()


def save_test_metric_map_triplet(
    geos_map,
    model_map,
    title_prefix,
    metric_name,
    filename,
    output_dir,
    lats,
    lons,
    vmin,
    vmax,
    diff_vmax,
    plot_subdir="test_plots_multi",
):
    if not HAS_CARTOPY:
        return

    proj = ccrs.PlateCarree()
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": proj})

    diff_map = geos_map - model_map
    panels = [
        (geos_map, f"{title_prefix} GEOS {metric_name}", "OrRd", vmin, vmax),
        (model_map, f"{title_prefix} Model {metric_name}", "OrRd", vmin, vmax),
        (
            diff_map,
            f"{title_prefix} {metric_name} Diff: GEOS-Model\nGreen (+) = Model Better | Magenta (-) = GEOS Better",
            "PiYG",
            -diff_vmax,
            diff_vmax,
        ),
    ]
    for i, (img, title, cmap, pmin, pmax) in enumerate(panels):
        ax = axes[i]
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=pmin,
            vmax=pmax,
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_cartopy_ax(ax, title, extent, show_left_labels=(i == 0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_subdir, filename), bbox_inches="tight", dpi=150)
    plt.close()


def save_test_correlation_triplet(
    geos_map,
    model_map,
    title_prefix,
    filename,
    output_dir,
    lats,
    lons,
    geos_avg,
    model_avg,
    plot_subdir="test_plots_multi",
):
    if not HAS_CARTOPY:
        return

    proj = ccrs.PlateCarree()
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": proj})
    diff_map = model_map - geos_map
    panels = [
        (geos_map, f"{title_prefix} GEOS Corr (Avg: {geos_avg:.3f})", "RdYlGn", -1, 1),
        (model_map, f"{title_prefix} Model Corr (Avg: {model_avg:.3f})", "RdYlGn", -1, 1),
        (
            diff_map,
            f"{title_prefix} Corr Diff: Model-GEOS\nOrange (+) = Model Better | Purple (-) = GEOS Better",
            "PuOr",
            -0.4,
            0.4,
        ),
    ]
    for i, (img, title, cmap, pmin, pmax) in enumerate(panels):
        ax = axes[i]
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=pmin,
            vmax=pmax,
            origin="lower",
            extent=extent,
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        style_cartopy_ax(ax, title, extent, show_left_labels=(i == 0))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, plot_subdir, filename), bbox_inches="tight", dpi=150)
    plt.close()


def load_plot_coords(data_dir, year_hint):
    """
    Match the older test scripts by reading the GEOS grid coordinates directly
    from the source data. Fall back to a 0..359 longitude convention, which is
    still safer than the incorrect -180..179 assumption for these files.
    """
    candidate_years = []
    if year_hint is not None:
        candidate_years.append(int(year_hint))
    candidate_years.extend([2021, 2020, 2019])

    seen = set()
    for year in candidate_years:
        if year in seen:
            continue
        seen.add(year)
        geos_sample_path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
        if not os.path.exists(geos_sample_path):
            continue
        try:
            ds_geos = xr.open_zarr(geos_sample_path, consolidated=False)
            lats = ds_geos.Y.values
            lons = ds_geos.X.values
            ds_geos.close()
            return lats, lons
        except Exception:
            continue

    return np.linspace(-90, 90, 181), np.arange(360)


def compute_multi_variance_loss(
    var_pred: torch.Tensor,
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    spatial_weights: torch.Tensor,
    temp_weights: torch.Tensor,
):
    """
    Match the v4 variance-head objective more closely:
    learn a relative standard-deviation multiplier rather than a squared variance target.
    """
    abs_err = torch.abs(v_target - v_pred.detach())
    target_scale = abs_err / (abs_err.mean(dim=(2, 3), keepdim=True) + 1e-6)

    std_mult = torch.sqrt(var_pred + 1e-6)
    loss_mse_var = (spatial_weights * temp_weights * (std_mult - target_scale) ** 2).mean()

    # Keep the multiplier near a physically reasonable range while gently pulling toward 1.0.
    var_penalty = torch.relu(std_mult - 2.5) ** 2 + torch.relu(0.5 - std_mult) ** 2
    identity_pull = (std_mult - 1.0) ** 2
    loss_reg = (var_penalty * 10.0 + identity_pull * 0.5).mean()

    return loss_mse_var + loss_reg

def save_val_plot(epoch, full_pred, true_target_precip, model_crps, model_rmse, geos_pred, geos_crps, geos_rmse, output_dir, 
                  ai_residual=None, suffix="", geos_single=None, model_single=None, model_var=None,
                  full_pred_t2m=None, true_target_t2m=None, geos_pred_t2m=None, model_var_t2m=None,
                  model_crps_t2m=0.0, model_rmse_t2m=0.0, geos_crps_t2m=0.0, geos_rmse_t2m=0.0):
    """
    Standardizes plotting logic for validation results (7-column layout).
    8 rows: 4 for PR (Weeks 1-4), 4 for T2M (Weeks 1-4).
    """
    t_img = true_target_precip[0].cpu().numpy()
    p_img = full_pred[0].cpu().numpy()
    g_img = geos_pred[0].cpu().numpy()
    g_sing_img = geos_single[0].cpu().numpy() if geos_single is not None else g_img
    m_sing_img = model_single[0].cpu().numpy() if model_single is not None else p_img
    m_var_img = model_var[0].cpu().numpy() if model_var is not None else np.zeros_like(p_img)
    
    # T2M arrays (may be None if not available)
    has_t2m = full_pred_t2m is not None and true_target_t2m is not None
    if has_t2m:
        t_img_t2m = true_target_t2m[0].cpu().numpy()
        p_img_t2m = full_pred_t2m[0].cpu().numpy()
        g_img_t2m = geos_pred_t2m[0].cpu().numpy() if geos_pred_t2m is not None else np.zeros_like(t_img_t2m)
        m_var_img_t2m = model_var_t2m[0].cpu().numpy() if model_var_t2m is not None else np.zeros_like(t_img_t2m)
    
    num_rows = 8 if has_t2m else 4
    fig, axes = plt.subplots(num_rows, 7, figsize=(35, 4 * num_rows))
    
    # --- PR Rows (0-3) ---
    for l in range(4):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        
        im0 = axes[l, 0].imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im0, ax=axes[l, 0], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 0].set_title("Target GPCP")
        axes[l, 0].set_ylabel(f"PR Wk{l+1}", fontsize=12, fontweight='bold')
        
        im1 = axes[l, 1].imshow(g_sing_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im1, ax=axes[l, 1], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 1].set_title("GEOS (Single Ens)")
        
        im2 = axes[l, 2].imshow(m_sing_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im2, ax=axes[l, 2], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 2].set_title("Model (Single Ens)")
        
        diff_geos = g_img[l] - t_img[l]
        im3 = axes[l, 3].imshow(diff_geos, cmap='RdBu_r', vmin=-30, vmax=30)
        fig.colorbar(im3, ax=axes[l, 3], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 3].set_title(f"GEOS Bias\nCRPS:{geos_crps:.2f}, RMSE:{geos_rmse:.2f}")
        
        diff_model = p_img[l] - t_img[l]
        im4 = axes[l, 4].imshow(diff_model, cmap='RdBu_r', vmin=-30, vmax=30)
        fig.colorbar(im4, ax=axes[l, 4], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 4].set_title(f"Model Bias\nCRPS:{model_crps:.2f}, RMSE:{model_rmse:.2f}")
        
        closeness = np.abs(diff_geos) - np.abs(diff_model)
        im5 = axes[l, 5].imshow(closeness, cmap='PiYG', vmin=-25, vmax=25)
        fig.colorbar(im5, ax=axes[l, 5], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 5].set_title("Closeness\nGreen=Model Better")
        
        var_vmax = np.percentile(m_var_img, 99) if m_var_img.max() > 0 else 1.0
        im6 = axes[l, 6].imshow(m_var_img[l], cmap='YlGn', vmin=0, vmax=var_vmax)
        fig.colorbar(im6, ax=axes[l, 6], fraction=0.046, pad=0.04)
        if l == 0: axes[l, 6].set_title("Model Ens Variance")

    # --- T2M Rows (4-7) ---
    if has_t2m:
        for l in range(4):
            row = l + 4
            t_min_t, t_max_t = t_img_t2m[l].min(), t_img_t2m[l].max()
            
            im0 = axes[row, 0].imshow(t_img_t2m[l], cmap='RdYlBu_r', vmin=t_min_t, vmax=t_max_t)
            fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 0].set_title("Target ERA5 T2M")
            axes[row, 0].set_ylabel(f"T2M Wk{l+1}", fontsize=12, fontweight='bold')
            
            im1 = axes[row, 1].imshow(g_img_t2m[l], cmap='RdYlBu_r', vmin=t_min_t, vmax=t_max_t)
            fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 1].set_title("GEOS TAS Mean")
            
            im2 = axes[row, 2].imshow(p_img_t2m[l], cmap='RdYlBu_r', vmin=t_min_t, vmax=t_max_t)
            fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 2].set_title("Model T2M Mean")
            
            diff_geos_t = g_img_t2m[l] - t_img_t2m[l]
            im3 = axes[row, 3].imshow(diff_geos_t, cmap='RdBu_r', vmin=-10, vmax=10)
            fig.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 3].set_title(f"GEOS T2M Bias\nCRPS:{geos_crps_t2m:.2f}, RMSE:{geos_rmse_t2m:.2f}")
            
            diff_model_t = p_img_t2m[l] - t_img_t2m[l]
            im4 = axes[row, 4].imshow(diff_model_t, cmap='RdBu_r', vmin=-10, vmax=10)
            fig.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 4].set_title(f"Model T2M Bias\nCRPS:{model_crps_t2m:.2f}, RMSE:{model_rmse_t2m:.2f}")
            
            closeness_t = np.abs(diff_geos_t) - np.abs(diff_model_t)
            im5 = axes[row, 5].imshow(closeness_t, cmap='PiYG', vmin=-5, vmax=5)
            fig.colorbar(im5, ax=axes[row, 5], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 5].set_title("Closeness T2M\nGreen=Model Better")
            
            var_vmax_t = np.percentile(m_var_img_t2m, 99) if m_var_img_t2m.max() > 0 else 1.0
            im6 = axes[row, 6].imshow(m_var_img_t2m[l], cmap='YlGn', vmin=0, vmax=var_vmax_t)
            fig.colorbar(im6, ax=axes[row, 6], fraction=0.046, pad=0.04)
            if l == 0: axes[row, 6].set_title("T2M Ens Variance")

    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    combined_crps = (model_crps + model_crps_t2m) / 2.0 if has_t2m else model_crps
    title = f"Epoch {epoch} | Combined CRPS: {combined_crps:.4f}  |  PR CRPS: {model_crps:.4f}  |  T2M CRPS: {model_crps_t2m:.4f}" if has_t2m else f"Epoch {epoch} | PR CRPS: {model_crps:.4f}"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    filename = f"epoch_{epoch}_{suffix}_score_{model_crps:.4f}.png" if suffix else f"epoch_{epoch}_score_{model_crps:.4f}.png"
    plt.savefig(os.path.join(output_dir, "plots", filename), bbox_inches='tight')
    plt.close()

@torch.no_grad()
def run_val_inference(epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                      target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, is_test=False, is_fast_recon=True,
                      cached_geos_crps=None, cached_geos_rmse=None, 
                      cached_geos_crps_t2m=None, cached_geos_rmse_t2m=None,
                      use_flow_variance=False, eof_bases=None,
                      nao_bases=None, nao_lookup=None, enso_bases=None, oni_lookup=None, mjo_df=None,
                      t2m_eof_bases=None, t2m_nao_bases=None, t2m_enso_bases=None,
                      use_eof_lhs_noise=False, validation_noise_cache=None,
                      print_validation_noise_diag=False,
                      validation_num_ensemble=15,
                      validation_num_steps=None,
                      validation_ode_batch_size=None,
                      validation_rho_pr=1.0,
                      validation_rho_t2m=None,
                      validation_var_beta_pr=1.0,
                      validation_var_beta_t2m=None):
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
    total_crps_t2m = 0.0
    total_rmse_t2m = 0.0
    total_geos_crps = 0.0
    total_geos_rmse = 0.0
    total_geos_crps_t2m = 0.0
    total_geos_rmse_t2m = 0.0
    count = 0
    did_print_noise_diag = False
    if validation_rho_t2m is None:
        validation_rho_t2m = validation_rho_pr
    if validation_var_beta_t2m is None:
        validation_var_beta_t2m = validation_var_beta_pr
    
    # Save tensors for the first validation batch we actually process. The
    # selected validation months may skip dataloader batch 0 entirely.
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
        
        num_ensemble = int(validation_num_ensemble)
        ensemble_preds_precip = [] # Will be [num_ensemble, num_inits, 4, H, W]
        ensemble_preds_t2m = []

        num_steps = int(validation_num_steps) if validation_num_steps is not None else (10 if is_fast_recon and not is_test else 50)
        
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
        
        current_year = int(batch['year'][0].item()) if 'year' in batch else 2021
        current_month = int(batch['month'][0].item())
        current_day = int(batch['day'][0].item()) if 'day' in batch else 15
        mode_tag = "pure_random"
        if use_eof_lhs_noise:
            mode_tag = (
                f"pr_t2m_eof_lhs_rho_pr{validation_rho_pr:.2f}_rho_t2m{validation_rho_t2m:.2f}"
            )
        if use_flow_variance:
            mode_tag += f"_beta_pr{validation_var_beta_pr:.2f}_beta_t2m{validation_var_beta_t2m:.2f}"
        cache_key = (mode_tag, b_idx, current_year, current_month, current_day, num_ensemble, vB, H, W)
        cache_hit = False

        if validation_noise_cache is not None and cache_key in validation_noise_cache:
            noise_expanded = validation_noise_cache[cache_key].to(device)
            cache_hit = True
        else:
            if use_eof_lhs_noise:
                eof_noise = noise_utils_multi.generate_dynamic_multimodal_noise_multi(
                    batch=batch,
                    E=num_ensemble,
                    device=device,
                    pr_mjo_bases=eof_bases,
                    pr_nao_bases=nao_bases,
                    pr_enso_bases=enso_bases,
                    t2m_mjo_bases=t2m_eof_bases,
                    t2m_nao_bases=t2m_nao_bases,
                    t2m_enso_bases=t2m_enso_bases,
                    nao_lookup=nao_lookup,
                    oni_lookup=oni_lookup,
                    mjo_df=mjo_df,
                    year=current_year,
                    use_lhs=True,
                    orthogonalize_lhs=True,
                )
                noise_expanded = noise_utils_multi.mix_noise_with_random_multi(
                    eof_noise, validation_rho_pr, validation_rho_t2m
                )
            else:
                noise_expanded = torch.randn((vB * num_ensemble, 2, H, W), device=device)

            if validation_noise_cache is not None:
                validation_noise_cache[cache_key] = noise_expanded.detach().cpu()
            
        lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
        
        # Single parallel ODE solve for the entire validation batch and ensemble
        p_x1_expanded = euler_solve_chunked(
            flow_matcher,
            unwrapped_model,
            noise_expanded,
            fx_cond_expanded,
            num_steps=num_steps,
            lead_idx=lead_idx_expanded,
            apply_flow_variance=use_flow_variance,
            variance_beta=(validation_var_beta_pr, validation_var_beta_t2m),
            chunk_size=validation_ode_batch_size,
        )

        if print_validation_noise_diag and not did_print_noise_diag and accelerator.is_main_process:
            if use_eof_lhs_noise:
                mode_label = (
                    f"PR EOF-LHS rho={validation_rho_pr:.2f} beta={validation_var_beta_pr:.2f} / "
                    f"T2M EOF-LHS rho={validation_rho_t2m:.2f} beta={validation_var_beta_t2m:.2f}"
                )
            else:
                mode_label = "Pure Random"
            source_label = "cache-hit" if cache_hit else ("cache-build" if validation_noise_cache is not None else "fresh")
            print(
                f"    📊 [Val Noise Debug] Epoch={epoch} Batch={b_idx} "
                f"Init={current_year:04d}-{current_month:02d}-{current_day:02d} "
                f"Mode={mode_label} Source={source_label}"
            )
            noise_utils_multi.print_noise_channel_stats(noise_expanded.float(), prefix="Val Noise")
            noise_utils_multi.print_noise_channel_stats(p_x1_expanded.float(), prefix="Val ODE Output")
            did_print_noise_diag = True
        
        p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)

        # Reverse PR (Channel 0)
        p_x1_pr = torch.clamp(p_x1_batch[:, :, 0], min=-1.0, max=1.0) # [vB, E, H, W]
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
            g_crps_t2m = compute_crps(geos_ens_sample[:, :, 1].transpose(0, 1), true_target_t2m, area_weights)
            g_rmse_t2m = compute_rmse(geos_mean_t2m, true_target_t2m, area_weights)
            
            total_geos_crps += g_crps * num_inits
            total_geos_rmse += g_rmse * num_inits
            total_geos_crps_t2m += g_crps_t2m * num_inits
            total_geos_rmse_t2m += g_rmse_t2m * num_inits
            
        total_crps += b_crps * num_inits
        total_rmse += b_rmse * num_inits
        total_crps_t2m += b_crps_t2m * num_inits
        total_rmse_t2m += b_rmse_t2m * num_inits
        count += num_inits
        
        # Save only the first processed validation batch for visual plotting consistency
        if not saved_tensors:
            true_target_precip_plot = torch.nan_to_num(true_target_precip[0], nan=0.0)
            true_target_t2m_plot = torch.nan_to_num(true_target_t2m[0], nan=280.0)
            ai_residual = full_pred_precip[0] - geos_mean_precip[0]
            model_var_t2m = ensemble_preds_t2m.var(dim=0)
            saved_tensors = {
                'full_pred': full_pred_precip[0].unsqueeze(0),
                'true_target': true_target_precip_plot.unsqueeze(0),
                'geos_mean': geos_mean_precip[0].unsqueeze(0),
                'ai_res': ai_residual.unsqueeze(0),
                'geos_single': torch.nan_to_num(geos_ens_sample[0, 0, 0], nan=0.0).unsqueeze(0),
                'model_single': ensemble_preds_precip[0, 0].unsqueeze(0),
                'model_var': model_var_precip[0].unsqueeze(0),
                # T2M plotting tensors
                'full_pred_t2m': full_pred_t2m[0].unsqueeze(0),
                'true_target_t2m': true_target_t2m_plot.unsqueeze(0),
                'geos_mean_t2m': geos_mean_t2m[0].unsqueeze(0),
                'model_var_t2m': model_var_t2m[0].unsqueeze(0),
            }
            
    # Compute Averages
    avg_crps = total_crps / count
    avg_rmse = total_rmse / count
    avg_crps_t2m = total_crps_t2m / count
    avg_rmse_t2m = total_rmse_t2m / count
    
    if cached_geos_crps is None:
        avg_geos_crps = total_geos_crps / count
        avg_geos_rmse = total_geos_rmse / count
        avg_geos_crps_t2m = total_geos_crps_t2m / count
        avg_geos_rmse_t2m = total_geos_rmse_t2m / count
    else:
        avg_geos_crps = cached_geos_crps
        avg_geos_rmse = cached_geos_rmse
        avg_geos_crps_t2m = cached_geos_crps_t2m if cached_geos_crps_t2m is not None else 0.0
        avg_geos_rmse_t2m = cached_geos_rmse_t2m if cached_geos_rmse_t2m is not None else 0.0
    
    recon_type = f"Monthly (N={len(target_batches)}x{num_ensemble})"
    combined_crps = (avg_crps + avg_crps_t2m) / 2.0
    if accelerator.is_main_process:
        print(f"Epoch {epoch} | Val CRPS [PR]: {avg_crps:.4f} (GEOS: {avg_geos_crps:.4f})")
        print(f"Epoch {epoch} | Val CRPS [T2M]: {avg_crps_t2m:.4f} (GEOS: {avg_geos_crps_t2m:.4f})")
        print(f"Epoch {epoch} | Combined CRPS: {combined_crps:.4f}")
        
    return {
        'combined_crps': combined_crps,
        'avg_crps_pr': avg_crps,
        'avg_rmse_pr': avg_rmse,
        'avg_crps_t2m': avg_crps_t2m,
        'avg_rmse_t2m': avg_rmse_t2m,
        'avg_geos_crps_pr': avg_geos_crps,
        'avg_geos_rmse_pr': avg_geos_rmse,
        'avg_geos_crps_t2m': avg_geos_crps_t2m,
        'avg_geos_rmse_t2m': avg_geos_rmse_t2m,
        'tensors': saved_tensors,
    }


@torch.no_grad()
def run_full_test_suite_multi(
    epoch,
    model,
    val_loader,
    flow_matcher,
    device,
    accelerator,
    output_dir,
    target_sqrt_min,
    target_sqrt_max,
    area_weights,
    lats=None,
    lons=None,
    use_flow_variance=False,
    eof_bases=None,
    nao_bases=None,
    nao_lookup=None,
    enso_bases=None,
    oni_lookup=None,
    mjo_df=None,
    t2m_eof_bases=None,
    t2m_nao_bases=None,
    t2m_enso_bases=None,
    use_eof_lhs_noise=False,
    validation_noise_cache=None,
    validation_num_ensemble=15,
    validation_num_steps=10,
    validation_ode_batch_size=None,
    validation_max_ensemble_per_chunk=None,
    validation_rho_pr=1.0,
    validation_rho_t2m=None,
    validation_var_beta_pr=1.0,
    validation_var_beta_t2m=None,
    sample_plot_limit=None,
    plot_subdir="test_plots_multi",
):
    if not HAS_CARTOPY:
        if accelerator.is_main_process:
            print("⚠️ Cartopy is unavailable. Skipping full multi-target test plot suite.")
        return

    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    if validation_rho_t2m is None:
        validation_rho_t2m = validation_rho_pr
    if validation_var_beta_t2m is None:
        validation_var_beta_t2m = validation_var_beta_pr

    if lats is None:
        lats = np.linspace(-90, 90, 181)
    if lons is None:
        lons = np.arange(360)
    aw_np = area_weights.squeeze().detach().cpu().numpy()
    aw_2d = np.broadcast_to(aw_np[:, np.newaxis], (len(lats), len(lons)))

    plot_dir = os.path.join(output_dir, plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)

    all_pr_preds = []
    all_pr_targets = []
    all_pr_geos = []
    all_pr_model_crps_maps = []
    all_pr_model_mse_maps = []
    all_pr_geos_crps_maps = []
    all_pr_geos_mse_maps = []

    all_t2m_preds = []
    all_t2m_targets = []
    all_t2m_geos = []
    all_t2m_model_crps_maps = []
    all_t2m_model_mse_maps = []
    all_t2m_geos_crps_maps = []
    all_t2m_geos_mse_maps = []

    total_pr_crps = 0.0
    total_pr_rmse = 0.0
    total_pr_geos_crps = 0.0
    total_pr_geos_rmse = 0.0
    total_t2m_crps = 0.0
    total_t2m_rmse = 0.0
    total_t2m_geos_crps = 0.0
    total_t2m_geos_rmse = 0.0
    total_inits = 0

    pbar = tqdm(val_loader, desc="Full Test Suite", disable=not accelerator.is_main_process)
    sample_plots_written = 0

    for b_idx, batch in enumerate(pbar):
        fb_target_norm = batch['y_target'].to(device)
        vB, _, H, W = fb_target_norm.shape
        num_inits = vB // 4

        true_target_raw = batch['target_raw_full'][0::4].to(device)
        true_target_precip = true_target_raw[:, 0]
        true_target_t2m = true_target_raw[:, 1]

        geos_ens_raw = batch['geos_ens_raw'].to(device)
        geos_ens_sample = geos_ens_raw[0::4]
        geos_mean_raw = geos_ens_sample.mean(dim=1)
        geos_mean_precip = geos_mean_raw[:, 0]
        geos_mean_t2m = geos_mean_raw[:, 1]

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
        num_ensemble = int(validation_num_ensemble)
        fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W)

        current_year = int(batch['year'][0].item()) if 'year' in batch else 2021
        current_month = int(batch['month'][0].item())
        current_day = int(batch['day'][0].item()) if 'day' in batch else 15
        mode_tag = "pure_random"
        if use_eof_lhs_noise:
            mode_tag = f"pr_t2m_eof_lhs_rho_pr{validation_rho_pr:.2f}_rho_t2m{validation_rho_t2m:.2f}"
        if use_flow_variance:
            mode_tag += f"_beta_pr{validation_var_beta_pr:.2f}_beta_t2m{validation_var_beta_t2m:.2f}"
        cache_key = ("testsuite", mode_tag, b_idx, current_year, current_month, current_day, num_ensemble, vB, H, W)

        if validation_noise_cache is not None and cache_key in validation_noise_cache:
            noise_expanded = validation_noise_cache[cache_key].to(device)
        else:
            if use_eof_lhs_noise:
                eof_noise = noise_utils_multi.generate_dynamic_multimodal_noise_multi(
                    batch=batch,
                    E=num_ensemble,
                    device=device,
                    pr_mjo_bases=eof_bases,
                    pr_nao_bases=nao_bases,
                    pr_enso_bases=enso_bases,
                    t2m_mjo_bases=t2m_eof_bases,
                    t2m_nao_bases=t2m_nao_bases,
                    t2m_enso_bases=t2m_enso_bases,
                    nao_lookup=nao_lookup,
                    oni_lookup=oni_lookup,
                    mjo_df=mjo_df,
                    year=current_year,
                    use_lhs=True,
                    orthogonalize_lhs=True,
                )
                noise_expanded = noise_utils_multi.mix_noise_with_random_multi(
                    eof_noise, validation_rho_pr, validation_rho_t2m
                )
            else:
                noise_expanded = torch.randn((vB * num_ensemble, 2, H, W), device=device)

            if validation_noise_cache is not None:
                validation_noise_cache[cache_key] = noise_expanded.detach().cpu()

        lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
        chunk_size = validation_ode_batch_size
        if validation_max_ensemble_per_chunk is not None:
            max_ensemble_chunk = max(1, int(validation_max_ensemble_per_chunk))
            max_state_chunk = vB * max_ensemble_chunk
            chunk_size = max_state_chunk if chunk_size is None else min(chunk_size, max_state_chunk)
        p_x1_expanded = euler_solve_chunked(
            flow_matcher,
            unwrapped_model,
            noise_expanded,
            fx_cond_expanded,
            num_steps=int(validation_num_steps),
            lead_idx=lead_idx_expanded,
            apply_flow_variance=use_flow_variance,
            variance_beta=(validation_var_beta_pr, validation_var_beta_t2m),
            chunk_size=chunk_size,
        )

        p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)
        p_x1_pr = torch.clamp(p_x1_batch[:, :, 0], min=-1.0, max=1.0)
        week_sqrt = ((p_x1_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        week_precip = torch.clamp(week_sqrt ** 2, min=0.0)

        p_x1_t2m = p_x1_batch[:, :, 1]
        t2m_min, t2m_max = 200.0, 320.0
        week_t2m = ((p_x1_t2m + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min

        ensemble_preds_precip = week_precip.transpose(0, 1).view(num_ensemble, num_inits, 4, H, W)
        ensemble_preds_t2m = week_t2m.transpose(0, 1).view(num_ensemble, num_inits, 4, H, W)
        full_pred_precip = ensemble_preds_precip.mean(dim=0)
        full_pred_t2m = ensemble_preds_t2m.mean(dim=0)

        pr_crps, pr_crps_map = compute_crps_with_map(ensemble_preds_precip, true_target_precip, area_weights)
        pr_rmse = compute_rmse(full_pred_precip, true_target_precip, area_weights)
        pr_geos_crps, pr_geos_crps_map = compute_crps_with_map(geos_ens_sample[:, :, 0].transpose(0, 1), true_target_precip, area_weights)
        pr_geos_rmse = compute_rmse(geos_mean_precip, true_target_precip, area_weights)
        pr_model_mse_map = (full_pred_precip - true_target_precip) ** 2
        pr_geos_mse_map = (geos_mean_precip - true_target_precip) ** 2

        t2m_crps, t2m_crps_map = compute_crps_with_map(ensemble_preds_t2m, true_target_t2m, area_weights)
        t2m_rmse = compute_rmse(full_pred_t2m, true_target_t2m, area_weights)
        t2m_geos_crps, t2m_geos_crps_map = compute_crps_with_map(geos_ens_sample[:, :, 1].transpose(0, 1), true_target_t2m, area_weights)
        t2m_geos_rmse = compute_rmse(geos_mean_t2m, true_target_t2m, area_weights)
        t2m_model_mse_map = (full_pred_t2m - true_target_t2m) ** 2
        t2m_geos_mse_map = (geos_mean_t2m - true_target_t2m) ** 2

        total_pr_crps += pr_crps * num_inits
        total_pr_rmse += pr_rmse * num_inits
        total_pr_geos_crps += pr_geos_crps * num_inits
        total_pr_geos_rmse += pr_geos_rmse * num_inits
        total_t2m_crps += t2m_crps * num_inits
        total_t2m_rmse += t2m_rmse * num_inits
        total_t2m_geos_crps += t2m_geos_crps * num_inits
        total_t2m_geos_rmse += t2m_geos_rmse * num_inits
        total_inits += num_inits

        all_pr_preds.append(full_pred_precip.detach().cpu().numpy())
        all_pr_targets.append(torch.nan_to_num(true_target_precip, nan=0.0).detach().cpu().numpy())
        all_pr_geos.append(torch.nan_to_num(geos_mean_precip, nan=0.0).detach().cpu().numpy())
        all_pr_model_crps_maps.append(pr_crps_map.detach().cpu().numpy())
        all_pr_model_mse_maps.append(pr_model_mse_map.detach().cpu().numpy())
        all_pr_geos_crps_maps.append(pr_geos_crps_map.detach().cpu().numpy())
        all_pr_geos_mse_maps.append(pr_geos_mse_map.detach().cpu().numpy())

        all_t2m_preds.append(full_pred_t2m.detach().cpu().numpy())
        all_t2m_targets.append(torch.nan_to_num(true_target_t2m, nan=280.0).detach().cpu().numpy())
        all_t2m_geos.append(torch.nan_to_num(geos_mean_t2m, nan=280.0).detach().cpu().numpy())
        all_t2m_model_crps_maps.append(t2m_crps_map.detach().cpu().numpy())
        all_t2m_model_mse_maps.append(t2m_model_mse_map.detach().cpu().numpy())
        all_t2m_geos_crps_maps.append(t2m_geos_crps_map.detach().cpu().numpy())
        all_t2m_geos_mse_maps.append(t2m_geos_mse_map.detach().cpu().numpy())

        if accelerator.is_main_process:
            init_years = batch['year'][0::4].detach().cpu().numpy().astype(int) if 'year' in batch else np.full(num_inits, current_year, dtype=int)
            init_months = batch['month'][0::4].detach().cpu().numpy().astype(int)
            init_days = batch['day'][0::4].detach().cpu().numpy().astype(int) if 'day' in batch else np.full(num_inits, current_day, dtype=int)

            for init_idx in range(num_inits):
                if sample_plot_limit is not None and sample_plots_written >= sample_plot_limit:
                    break

                single_pr_ens = ensemble_preds_precip[:, init_idx:init_idx + 1]
                single_pr_target = true_target_precip[init_idx:init_idx + 1]
                single_pr_full = full_pred_precip[init_idx:init_idx + 1]
                single_pr_geos_ens = geos_ens_sample[init_idx:init_idx + 1, :, 0].transpose(0, 1)
                single_pr_geos_mean = geos_mean_precip[init_idx:init_idx + 1]

                single_t2m_ens = ensemble_preds_t2m[:, init_idx:init_idx + 1]
                single_t2m_target = true_target_t2m[init_idx:init_idx + 1]
                single_t2m_full = full_pred_t2m[init_idx:init_idx + 1]
                single_t2m_geos_ens = geos_ens_sample[init_idx:init_idx + 1, :, 1].transpose(0, 1)
                single_t2m_geos_mean = geos_mean_t2m[init_idx:init_idx + 1]

                pr_crps_init, _ = compute_crps_with_map(single_pr_ens, single_pr_target, area_weights)
                pr_rmse_init = compute_rmse(single_pr_full, single_pr_target, area_weights)
                pr_geos_crps_init, _ = compute_crps_with_map(single_pr_geos_ens, single_pr_target, area_weights)
                pr_geos_rmse_init = compute_rmse(single_pr_geos_mean, single_pr_target, area_weights)

                t2m_crps_init, _ = compute_crps_with_map(single_t2m_ens, single_t2m_target, area_weights)
                t2m_rmse_init = compute_rmse(single_t2m_full, single_t2m_target, area_weights)
                t2m_geos_crps_init, _ = compute_crps_with_map(single_t2m_geos_ens, single_t2m_target, area_weights)
                t2m_geos_rmse_init = compute_rmse(single_t2m_geos_mean, single_t2m_target, area_weights)

                init_label = f"{init_years[init_idx]:04d}-{init_months[init_idx]:02d}-{init_days[init_idx]:02d}"
                save_test_plot_cartopy_multi(
                    batch_idx=sample_plots_written,
                    init_label=init_label,
                    output_dir=output_dir,
                    lats=lats,
                    lons=lons,
                    full_pred_pr=full_pred_precip[init_idx].detach().cpu().numpy(),
                    true_target_pr=torch.nan_to_num(true_target_precip[init_idx], nan=0.0).detach().cpu().numpy(),
                    geos_mean_pr=torch.nan_to_num(geos_mean_precip[init_idx], nan=0.0).detach().cpu().numpy(),
                    geos_single_pr=torch.nan_to_num(geos_ens_sample[init_idx, 0, 0], nan=0.0).detach().cpu().numpy(),
                    model_single_pr=ensemble_preds_precip[0, init_idx].detach().cpu().numpy(),
                    model_crps_pr=pr_crps_init,
                    model_rmse_pr=pr_rmse_init,
                    geos_crps_pr=pr_geos_crps_init,
                    geos_rmse_pr=pr_geos_rmse_init,
                    full_pred_t2m=full_pred_t2m[init_idx].detach().cpu().numpy(),
                    true_target_t2m=torch.nan_to_num(true_target_t2m[init_idx], nan=280.0).detach().cpu().numpy(),
                    geos_mean_t2m=torch.nan_to_num(geos_mean_t2m[init_idx], nan=280.0).detach().cpu().numpy(),
                    geos_single_t2m=torch.nan_to_num(geos_ens_sample[init_idx, 0, 1], nan=280.0).detach().cpu().numpy(),
                    model_single_t2m=ensemble_preds_t2m[0, init_idx].detach().cpu().numpy(),
                    model_crps_t2m=t2m_crps_init,
                    model_rmse_t2m=t2m_rmse_init,
                    geos_crps_t2m=t2m_geos_crps_init,
                    geos_rmse_t2m=t2m_geos_rmse_init,
                    plot_subdir=plot_subdir,
                )
                sample_plots_written += 1
                print(
                    f"📸 Saved month plot immediately for {init_label} "
                    f"(PR CRPS {pr_crps_init:.4f}, T2M CRPS {t2m_crps_init:.4f})"
                )

        if accelerator.is_main_process:
            done = max(1, total_inits)
            pbar.set_postfix({
                "PR_CRPS": f"{total_pr_crps / done:.3f}",
                "T2M_CRPS": f"{total_t2m_crps / done:.3f}",
            })

    if total_inits == 0:
        if accelerator.is_main_process:
            print("⚠️ Full test suite found no batches to process.")
        return

    pr_preds = np.concatenate(all_pr_preds, axis=0)
    pr_targets = np.concatenate(all_pr_targets, axis=0)
    pr_geos = np.concatenate(all_pr_geos, axis=0)
    pr_model_crps_maps = np.concatenate(all_pr_model_crps_maps, axis=0)
    pr_model_mse_maps = np.concatenate(all_pr_model_mse_maps, axis=0)
    pr_geos_crps_maps = np.concatenate(all_pr_geos_crps_maps, axis=0)
    pr_geos_mse_maps = np.concatenate(all_pr_geos_mse_maps, axis=0)

    t2m_preds = np.concatenate(all_t2m_preds, axis=0)
    t2m_targets = np.concatenate(all_t2m_targets, axis=0)
    t2m_geos = np.concatenate(all_t2m_geos, axis=0)
    t2m_model_crps_maps = np.concatenate(all_t2m_model_crps_maps, axis=0)
    t2m_model_mse_maps = np.concatenate(all_t2m_model_mse_maps, axis=0)
    t2m_geos_crps_maps = np.concatenate(all_t2m_geos_crps_maps, axis=0)
    t2m_geos_mse_maps = np.concatenate(all_t2m_geos_mse_maps, axis=0)

    pr_model_corr = temporal_correlation_maps(pr_preds, pr_targets)
    pr_geos_corr = temporal_correlation_maps(pr_geos, pr_targets)
    t2m_model_corr = temporal_correlation_maps(t2m_preds, t2m_targets)
    t2m_geos_corr = temporal_correlation_maps(t2m_geos, t2m_targets)

    pr_avg_model_crps_maps = np.nanmean(pr_model_crps_maps, axis=0)
    pr_avg_geos_crps_maps = np.nanmean(pr_geos_crps_maps, axis=0)
    pr_avg_model_rmse_maps = np.sqrt(np.nanmean(pr_model_mse_maps, axis=0))
    pr_avg_geos_rmse_maps = np.sqrt(np.nanmean(pr_geos_mse_maps, axis=0))

    t2m_avg_model_crps_maps = np.nanmean(t2m_model_crps_maps, axis=0)
    t2m_avg_geos_crps_maps = np.nanmean(t2m_geos_crps_maps, axis=0)
    t2m_avg_model_rmse_maps = np.sqrt(np.nanmean(t2m_model_mse_maps, axis=0))
    t2m_avg_geos_rmse_maps = np.sqrt(np.nanmean(t2m_geos_mse_maps, axis=0))

    if accelerator.is_main_process:
        for wk in range(4):
            save_test_correlation_triplet(
                pr_geos_corr[wk],
                pr_model_corr[wk],
                f"PR Week {wk + 1}",
                f"corr_pr_wk{wk + 1}.png",
                output_dir,
                lats,
                lons,
                geos_avg=weighted_mean_2d(pr_geos_corr[wk], aw_2d),
                model_avg=weighted_mean_2d(pr_model_corr[wk], aw_2d),
                plot_subdir=plot_subdir,
            )
            save_test_correlation_triplet(
                t2m_geos_corr[wk],
                t2m_model_corr[wk],
                f"T2M Week {wk + 1}",
                f"corr_t2m_wk{wk + 1}.png",
                output_dir,
                lats,
                lons,
                geos_avg=weighted_mean_2d(t2m_geos_corr[wk], aw_2d),
                model_avg=weighted_mean_2d(t2m_model_corr[wk], aw_2d),
                plot_subdir=plot_subdir,
            )
            save_test_metric_map_triplet(
                pr_avg_geos_crps_maps[wk],
                pr_avg_model_crps_maps[wk],
                f"PR Week {wk + 1}",
                "CRPS",
                f"crps_pr_wk{wk + 1}.png",
                output_dir,
                lats,
                lons,
                0,
                8,
                3,
                plot_subdir=plot_subdir,
            )
            save_test_metric_map_triplet(
                pr_avg_geos_rmse_maps[wk],
                pr_avg_model_rmse_maps[wk],
                f"PR Week {wk + 1}",
                "RMSE",
                f"rmse_pr_wk{wk + 1}.png",
                output_dir,
                lats,
                lons,
                0,
                15,
                5,
                plot_subdir=plot_subdir,
            )
            save_test_metric_map_triplet(
                t2m_avg_geos_crps_maps[wk],
                t2m_avg_model_crps_maps[wk],
                f"T2M Week {wk + 1}",
                "CRPS",
                f"crps_t2m_wk{wk + 1}.png",
                output_dir,
                lats,
                lons,
                0,
                4,
                1.5,
                plot_subdir=plot_subdir,
            )
            save_test_metric_map_triplet(
                t2m_avg_geos_rmse_maps[wk],
                t2m_avg_model_rmse_maps[wk],
                f"T2M Week {wk + 1}",
                "RMSE",
                f"rmse_t2m_wk{wk + 1}.png",
                output_dir,
                lats,
                lons,
                0,
                8,
                3,
                plot_subdir=plot_subdir,
            )

        summary = {
            "epoch": int(epoch),
            "num_inits": int(total_inits),
            "avg_crps_pr": total_pr_crps / total_inits,
            "avg_rmse_pr": total_pr_rmse / total_inits,
            "avg_geos_crps_pr": total_pr_geos_crps / total_inits,
            "avg_geos_rmse_pr": total_pr_geos_rmse / total_inits,
            "avg_crps_t2m": total_t2m_crps / total_inits,
            "avg_rmse_t2m": total_t2m_rmse / total_inits,
            "avg_geos_crps_t2m": total_t2m_geos_crps / total_inits,
            "avg_geos_rmse_t2m": total_t2m_geos_rmse / total_inits,
        }
        with open(os.path.join(plot_dir, "test_summary_multi.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\n==================================")
        print("   FULL MULTI TEST SUITE DONE")
        print("==================================")
        print(f"PR   GEOS CRPS: {summary['avg_geos_crps_pr']:.4f} | Model CRPS: {summary['avg_crps_pr']:.4f}")
        print(f"PR   GEOS RMSE: {summary['avg_geos_rmse_pr']:.4f} | Model RMSE: {summary['avg_rmse_pr']:.4f}")
        print(f"T2M GEOS CRPS: {summary['avg_geos_crps_t2m']:.4f} | Model CRPS: {summary['avg_crps_t2m']:.4f}")
        print(f"T2M GEOS RMSE: {summary['avg_geos_rmse_t2m']:.4f} | Model RMSE: {summary['avg_rmse_t2m']:.4f}")
        print(f"📸 Cartopy test plots saved to {plot_dir}")
        print("==================================")

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
    plot_lats, plot_lons = load_plot_coords(config["data_dir"], config.get("val_end_year"))

    variance_phase_lr = float(config.get("variance_phase_lr", 1e-4))
    force_variance_phase = bool(config.get("force_variance_phase", False))
    validation_num_ensemble = int(config.get("validation_num_ensemble", 15))
    validation_num_steps = int(config.get("validation_num_steps", 10))
    validation_ode_batch_size = int(config.get("validation_ode_batch_size", 120))
    test_num_ensemble = int(config.get("test_num_ensemble", 90))
    test_num_steps = int(config.get("test_num_steps", 10))
    test_max_ensemble_per_chunk = int(config.get("test_max_ensemble_per_chunk", 30))
    validation_rho_pr = float(config.get("validation_rho_pr", 1.0))
    validation_rho_t2m = float(config.get("validation_rho_t2m", validation_rho_pr))
    validation_var_beta_pr = float(config.get("validation_var_beta_pr", 1.0))
    validation_var_beta_t2m = float(config.get("validation_var_beta_t2m", validation_var_beta_pr))
    
    if accelerator.is_main_process:
        print("\n=======================================================")
        print(f"✅ Loaded Strict Global Stats: {stats_file}")
        print(f"   [Target SQRT Bounds] : Min = {target_sqrt_min:.4f}, Max = {target_sqrt_max:.4f}")
        print(f"   [GEOS Raw Bounds]    : Min = {geos_min:.4f}, Max = {geos_max:.4f}")
        print(f"   [Plot Lon Range]     : {float(plot_lons.min()):.2f} .. {float(plot_lons.max()):.2f}")
        if force_variance_phase:
            print(f"   [Training Mode]      : Variance-only (lr={variance_phase_lr:.2e})")
        else:
            print(f"   [Training Mode]      : Velocity-only")
        print(f"   [Validation Ens]     : {validation_num_ensemble}")
        print(f"   [Validation Steps]   : {validation_num_steps}")
        print(f"   [Validation Chunk]   : {validation_ode_batch_size}")
        print(f"   [Validation Rho]     : PR={validation_rho_pr:.2f}, T2M={validation_rho_t2m:.2f}")
        print(f"   [Validation Beta]    : PR={validation_var_beta_pr:.2f}, T2M={validation_var_beta_t2m:.2f}")
        print(f"   [Test Ens]           : {test_num_ensemble}")
        print(f"   [Test Steps]         : {test_num_steps}")
        print(f"   [Test Ens/Chunk]     : {test_max_ensemble_per_chunk}")
        print("=======================================================\n")

    # ---------------------------------------------------------
    # 2. Model & Scheduler Setup
    # ---------------------------------------------------------
    model = FlowMatchingModel(in_channels=41, out_channels=2).to(device)
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
        print(f"   Total Input Channels: 41")
        print(f"   --- Condition Channels (x_cond = 39) ---")
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

    def resolve_eof_path(filename):
        for base_dir in (
            os.path.join(config["data_dir"], "eof"),
            config["data_dir"],
            os.path.dirname(__file__),
        ):
            candidate = os.path.join(base_dir, filename)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(config["data_dir"], "eof", filename)

    # Load Dynamic Multi-Modal Bases
    eof_bases_path = resolve_eof_path("mjo_eof_bases.pt")
    nao_bases_path = resolve_eof_path("nao_eof_bases.pt")
    enso_bases_path = resolve_eof_path("enso_eof_bases.pt")
    
    eof_bases = torch.load(eof_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(eof_bases_path) else None
    nao_bases = torch.load(nao_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(nao_bases_path) else None
    enso_bases = torch.load(enso_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(enso_bases_path) else None

    # Load T2M Dynamic Multi-Modal Bases
    t2m_eof_bases_path = resolve_eof_path("mjo_t2m_eof_bases.pt")
    t2m_nao_bases_path = resolve_eof_path("nao_t2m_eof_bases.pt")
    t2m_enso_bases_path = resolve_eof_path("enso_t2m_eof_bases.pt")
    
    t2m_eof_bases = torch.load(t2m_eof_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(t2m_eof_bases_path) else None
    t2m_nao_bases = torch.load(t2m_nao_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(t2m_nao_bases_path) else None
    t2m_enso_bases = torch.load(t2m_enso_bases_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(t2m_enso_bases_path) else None
    
    try:
        nao_lookup = noise_utils.parse_nao_index(os.path.join(config["data_dir"], "norm.daily.nao.index.b500101.current.ascii"))
        oni_lookup = noise_utils.parse_oni_index(os.path.join(config["data_dir"], "oni.ascii.txt"))
        mjo_df = pd.read_csv(os.path.join(config["data_dir"], "mjo_processed.csv"), parse_dates=['S']).set_index(pd.to_datetime(pd.read_csv(os.path.join(config["data_dir"], "mjo_processed.csv"), parse_dates=['S'])['S']).dt.strftime('%Y-%m-%d'))
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ Teleconnection index loading failed: {e}. Falling back to default amplitudes.")
        nao_lookup, oni_lookup, mjo_df = None, None, None
        
    if accelerator.is_main_process and eof_bases is not None:
        print("✅ Loaded Multi-Modal EOF bases & Teleconnection Indices (pure noise in velocity mode, EOF-LHS in variance-only mode).")
        print(f"   PR EOF files : {eof_bases_path}, {nao_bases_path}, {enso_bases_path}")
        print(f"   T2M EOF files: {t2m_eof_bases_path}, {t2m_nao_bases_path}, {t2m_enso_bases_path}")

    validation_noise_cache = {}
    
    start_epoch = 1
    loaded_checkpoint_epoch = 0
    loaded_is_variance_phase = False
    best_val_loss = float('inf')
    crps_phase_reset = False  # Will be set True after MSE→CRPS transition (or if resuming from Phase 2)
    top_models = []
    is_variance_phase = False
    
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
            loaded_checkpoint_epoch = int(checkpoint.get('epoch', 0))
            loaded_is_variance_phase = bool(checkpoint.get('is_variance_phase', False))
            # Unwrap for loading
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(checkpoint['model'])
            if args.test:
                start_epoch = loaded_checkpoint_epoch
            
            if not args.test:
                start_epoch = loaded_checkpoint_epoch + 1
                resume_into_variance_phase = force_variance_phase
                
                if resume_into_variance_phase:
                    if accelerator.is_main_process:
                        print(
                            "   ℹ️ Variance-only resume detected (forced by config). "
                            "Skipping optimizer state load and rebuilding a var-heads optimizer."
                        )
                else:
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
                    print(f"   Best Val Loss so far: {best_val_loss:.4f}")
                    if start_epoch > 101:
                        print(f"   ✅ Resuming deep in CRPS Phase (>101). best_val_loss is already CRPS-scale.")
                    elif start_epoch == 101:
                        print(f"   🔀 Resuming exactly at Phase Boundary (Epoch 101). Will reset best_val_loss.")
                    else:
                        print(f"   📋 Resuming in MSE Phase (≤100). Will reset best_val_loss later.")
                    print()
            
            # If resuming AFTER the boundary (Phase 2), skip the MSE→CRPS reset
            # But if we start exactly at 101, we MUST trigger the reset inside the loop!
            if start_epoch > 101:
                crps_phase_reset = True
                    

        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ Failed to load checkpoint {ckpt_path}: {e}")
    else:
        if args.test:
            raise FileNotFoundError(f"CRITICAL: Checkpoint {ckpt_path} not found for testing!")
        if accelerator.is_main_process:
            print(f"\n🚀 Starting fresh training from Epoch 0\n")

    if not args.test and start_epoch > epochs:
        if accelerator.is_main_process:
            print(
                f"✅ Checkpoint already reached configured max epochs "
                f"({start_epoch - 1} / {epochs}). Exiting without training."
            )
        return

    start_in_variance_phase = force_variance_phase

    if not args.test and start_in_variance_phase:
        if accelerator.is_main_process:
            if force_variance_phase:
                print("🔒 Force-enabling variance phase from config. Freezing UNet + mean heads.")
                if loaded_checkpoint_epoch == 0:
                    print("   ⚠️ No checkpoint was loaded. This will train only var_heads from scratch.")

        unwrapped = accelerator.unwrap_model(model)
        for param in unwrapped.unet.parameters():
            param.requires_grad_(False)
        for param in unwrapped.heads.parameters():
            param.requires_grad_(False)
        for param in unwrapped.var_heads.parameters():
            param.requires_grad_(True)

        optimizer = torch.optim.AdamW(
            [p for p in unwrapped.var_heads.parameters() if p.requires_grad],
            lr=variance_phase_lr
        )
        model, optimizer, loader, val_loader = accelerator.prepare(
            model, optimizer, loader, val_loader
        )
        is_variance_phase = True
        
    # ---------------------------------------------------------
    # 3. Execution Mode: Train or Test
    # ---------------------------------------------------------
    if args.test:
        if accelerator.is_main_process:
            print(f"\n🧪 RUNNING TEST MODE: Full multi-target test suite for {ckpt_path}")
            print(
                f"   Using {test_num_ensemble} ensemble members, {test_num_steps} ODE steps, "
                f"max {test_max_ensemble_per_chunk} ensembles/forward chunk.\n"
            )

        run_full_test_suite_multi(
            start_epoch,
            model,
            val_loader,
            flow_matcher,
            device,
            accelerator,
            output_dir,
            target_sqrt_min,
            target_sqrt_max,
            area_weights,
            lats=plot_lats,
            lons=plot_lons,
            use_flow_variance=(force_variance_phase or loaded_is_variance_phase),
            eof_bases=eof_bases,
            nao_bases=nao_bases,
            nao_lookup=nao_lookup,
            enso_bases=enso_bases,
            oni_lookup=oni_lookup,
            mjo_df=mjo_df,
            t2m_eof_bases=t2m_eof_bases,
            t2m_nao_bases=t2m_nao_bases,
            t2m_enso_bases=t2m_enso_bases,
            use_eof_lhs_noise=(force_variance_phase or loaded_is_variance_phase),
            validation_noise_cache=validation_noise_cache,
            validation_num_ensemble=test_num_ensemble,
            validation_num_steps=test_num_steps,
            validation_ode_batch_size=validation_ode_batch_size,
            validation_max_ensemble_per_chunk=test_max_ensemble_per_chunk,
            validation_rho_pr=validation_rho_pr,
            validation_rho_t2m=validation_rho_t2m,
            validation_var_beta_pr=validation_var_beta_pr,
            validation_var_beta_t2m=validation_var_beta_t2m,
            sample_plot_limit=24,
            plot_subdir=f"test_plots_multi_e{test_num_ensemble}_s{test_num_steps}",
        )
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

        gpcp_raw_sample = batch['target_raw'][sample_idx, lead_idx].cpu().numpy()

        # 6. GEOS TAS (conditioning channel, index 1 in geos channels)
        # x_geos shape: [B, M=1, C=2, L, H, W] -> M=0 (only member), C=1 (TAS), L=lead_idx
        geos_tas_norm_sample = np.nan_to_num(x_geos[sample_idx, 0, 1, lead_idx].cpu().numpy(), nan=0.0)
        if "geos_tas_raw" in global_bounds:
            tas_gmin, tas_gmax = global_bounds["geos_tas_raw"]["min"], global_bounds["geos_tas_raw"]["max"]
        else:
            tas_gmin, tas_gmax = 200.0, 320.0
        geos_tas_raw_sample = ((geos_tas_norm_sample + 1.0) / 2.0) * (tas_gmax - tas_gmin) + tas_gmin

        # 7. ERA5 T2M target (channel 1 of y_target)
        # target_norm shape: [B, C=2, H, W] -> C=1 is T2M for this lead
        t2m_norm_sample = np.nan_to_num(target_norm[sample_idx, 1].cpu().numpy(), nan=0.0)
        if "target_t2m_raw" in global_bounds:
            t2m_tmin, t2m_tmax = global_bounds["target_t2m_raw"]["min"], global_bounds["target_t2m_raw"]["max"]
        else:
            t2m_tmin, t2m_tmax = 200.0, 320.0
        t2m_raw_sample = ((t2m_norm_sample + 1.0) / 2.0) * (t2m_tmax - t2m_tmin) + t2m_tmin

        fig, axes = plt.subplots(13, 2, figsize=(14, 52))
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

        # Row 10: GEOS TAS
        im_tas1 = axes[10, 0].imshow(geos_tas_raw_sample, cmap='RdYlBu_r')
        axes[10, 0].set_title("Raw GEOS TAS (Conditioning)")
        fig.colorbar(im_tas1, ax=axes[10, 0])
        im_tas2 = axes[10, 1].imshow(geos_tas_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[10, 1].set_title("Normalized GEOS TAS [-1, 1]")
        fig.colorbar(im_tas2, ax=axes[10, 1])

        # Row 11: ERA5 T2M Target
        im_t2m1 = axes[11, 0].imshow(t2m_raw_sample, cmap='RdYlBu_r')
        axes[11, 0].set_title("Raw ERA5 T2M (Target)")
        fig.colorbar(im_t2m1, ax=axes[11, 0])
        im_t2m2 = axes[11, 1].imshow(t2m_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[11, 1].set_title("Normalized ERA5 T2M [-1, 1]")
        fig.colorbar(im_t2m2, ax=axes[11, 1])

        flat_geos = geos_raw_sample.flatten()
        flat_gpcp = gpcp_raw_sample.flatten()
        cc = np.corrcoef(flat_geos, flat_gpcp)[0, 1] if np.std(flat_geos) > 1e-6 else 0.0
        
        im20 = axes[12, 0].imshow(geos_raw_sample - gpcp_raw_sample, cmap='RdBu_r', vmin=-20, vmax=20)
        axes[12, 0].set_title(f"Spatial Alignment | CC: {cc:.4f}")
        fig.colorbar(im20, ax=axes[12, 0])
        
        axes[12, 1].scatter(flat_geos[::50], flat_gpcp[::50], alpha=0.3, s=1)
        axes[12, 1].set_xlabel("GEOS Rainfall")
        axes[12, 1].set_ylabel("GPCP Rainfall")
        axes[12, 1].set_title("Orientation Scatter (Subsampled)")
        
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
    global_cached_geos_crps_t2m = None
    global_cached_geos_rmse_t2m = None
    spatial_weights = area_weights * land_ocean_weights
    
    for epoch in range(start_epoch, epochs + 1):
        if epochs_done_this_run >= max_epochs_this_run:
            if accelerator.is_main_process:
                print(f"\n⚠️ Reached --epochs-per-run limit ({max_epochs_this_run}). Exiting for resubmission.")
            break

        model.train()
        train_loss = 0.0
        phase_label = "VarOnly" if is_variance_phase else "VelOnly"
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase_label}]", disable=not accelerator.is_main_process)
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

            # Predict the velocity AND variance (routed through the correct per-week output head)
            v_pred, var_pred = model(x_t, x_cond, t, lead_idx=lead_idx)

            # --- Temporal Loss Weighting ---
            # Prioritize gradient updates for harder long-term leads (Week 4 > Week 1)
            # 0=Week1, 1=Week2, 2=Week3, 3=Week4
            w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
            temp_weights = w_escalation[lead_idx].view(B, 1, 1, 1)

            # --- Combined Loss Computation ---
            # 1. Velocity MSE Loss
            temp_weights_expanded = temp_weights.expand(-1, 2, -1, -1)
            loss_vel = (spatial_weights * temp_weights_expanded * (v_pred - v_target)**2).mean()

            # 2. Variance loss in relative std-multiplier space (v4-style)
            loss_var = compute_multi_variance_loss(
                var_pred=var_pred,
                v_pred=v_pred,
                v_target=v_target,
                spatial_weights=spatial_weights,
                temp_weights=temp_weights_expanded,
            )

            if is_variance_phase:
                loss = loss_var
            else:
                loss = loss_vel

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
                'top_models': top_models,
                'is_variance_phase': is_variance_phase,
            }
            torch.save(ckpt, os.path.join(output_dir, "latest_flow_ckpt.pt"))

        # --- ADAPTIVE VALIDATION SCHEDULE ---
        # Phase 1 (epoch 1-100):  MSE-based validation & best model tracking
        #   Epoch 1-4:   No validation
        #   Epoch 5-19:  Every 5 epochs
        #   Epoch 20-69: Every 3 epochs
        #   Epoch 70+:   Every epoch
        # CRPS-based validation (15 ens, 10 steps)
        #   Every epoch: run_val_inference, periodic ckpt every 5 epochs
        
        use_crps_phase = (epoch > 100)
        
        # One-time reset: when transitioning from MSE (Phase 1) to CRPS (Phase 2),
        # best_val_loss must be reset because MSE values (~0.06) are on a completely
        # different scale than CRPS values (~5-20). Without this, no model would
        # ever be saved as "new best" in Phase 2.
        if use_crps_phase and not crps_phase_reset:
            if accelerator.is_main_process:
                print(f"\n🔄 PHASE TRANSITION: Epoch {epoch} → Switching to CRPS-based validation.")
                print(f"   Resetting best_val_loss from {best_val_loss:.4f} (MSE) → inf (CRPS)")
            best_val_loss = float('inf')
            crps_phase_reset = True
        
        def should_validate(ep):
            if ep < 5:
                return False
            elif ep < 20:
                return (ep % 5 == 0)
            elif ep < 70:
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
            if use_crps_phase:
                print(f"\n⌛ Epoch {epoch} complete. Starting CRPS Validation ({validation_num_ensemble} ens, {validation_num_steps} steps)...")
            else:
                print(f"\n⌛ Epoch {epoch} complete. Starting Fast Validation (Noise MSE)...")
        
        # ============================================================
        #  CRPS-based validation
        # ============================================================
        if use_crps_phase:
            use_flow_variance = is_variance_phase
            val_result = run_val_inference(
                epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, 
                is_test=False, is_fast_recon=True, use_flow_variance=use_flow_variance,
                use_eof_lhs_noise=is_variance_phase,
                validation_noise_cache=validation_noise_cache,
                print_validation_noise_diag=True,
                cached_geos_crps=global_cached_geos_crps, cached_geos_rmse=global_cached_geos_rmse,
                cached_geos_crps_t2m=global_cached_geos_crps_t2m, cached_geos_rmse_t2m=global_cached_geos_rmse_t2m,
                eof_bases=eof_bases, nao_bases=nao_bases, nao_lookup=nao_lookup,
                enso_bases=enso_bases, oni_lookup=oni_lookup, mjo_df=mjo_df,
                t2m_eof_bases=t2m_eof_bases, t2m_nao_bases=t2m_nao_bases, t2m_enso_bases=t2m_enso_bases,
                validation_num_ensemble=validation_num_ensemble,
                validation_num_steps=validation_num_steps,
                validation_ode_batch_size=validation_ode_batch_size,
                validation_rho_pr=validation_rho_pr,
                validation_rho_t2m=validation_rho_t2m,
                validation_var_beta_pr=validation_var_beta_pr,
                validation_var_beta_t2m=validation_var_beta_t2m,
            )
            current_val_metric = val_result['combined_crps']
            if global_cached_geos_crps is None:
                global_cached_geos_crps = val_result['avg_geos_crps_pr']
                global_cached_geos_rmse = val_result['avg_geos_rmse_pr']
                global_cached_geos_crps_t2m = val_result['avg_geos_crps_t2m']
                global_cached_geos_rmse_t2m = val_result['avg_geos_rmse_t2m']
            if accelerator.is_main_process:
                print(f"✅ CRPS Done. Combined: {current_val_metric:.4f} | PR: {val_result['avg_crps_pr']:.4f} | T2M: {val_result['avg_crps_t2m']:.4f}")
                is_new_best = (current_val_metric < best_val_loss)
                if is_new_best:
                    print(f"🏆 NEW BEST (CRPS)! {current_val_metric:.4f} (Prev: {best_val_loss:.4f})")
                    best_val_loss = current_val_metric
                    new_best_name = f"best_model_epoch_{epoch}_crps_{current_val_metric:.4f}.pt"
                    new_best_path = os.path.join(output_dir, new_best_name)
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_ckpt = {
                        'epoch': epoch,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'is_variance_phase': is_variance_phase,
                    }
                    torch.save(best_ckpt, new_best_path)
                    torch.save(best_ckpt, os.path.join(output_dir, "best_flow_ckpt.pt"))
                    registry_path = os.path.join(output_dir, "model_registry.json")
                    registry = json.load(open(registry_path)) if os.path.exists(registry_path) else []
                    registry.append({"rank": 0, "path": new_best_path, "val_loss": current_val_metric, "epoch": epoch, "metric": "combined_crps"})
                    registry.sort(key=lambda x: x['val_loss'])
                    for i, entry in enumerate(registry): entry['rank'] = i + 1
                    with open(registry_path, 'w') as f: json.dump(registry, f, indent=2)
                    print(f"📋 Registry: {len(registry)} models tracked.")
                    vt = val_result['tensors']
                    save_val_plot(epoch, vt['full_pred'], vt['true_target'], val_result['avg_crps_pr'], val_result['avg_rmse_pr'],
                                  vt['geos_mean'], val_result['avg_geos_crps_pr'], val_result['avg_geos_rmse_pr'], output_dir,
                                  ai_residual=vt['ai_res'], suffix="best_crps", geos_single=vt['geos_single'],
                                  model_single=vt['model_single'], model_var=vt['model_var'],
                                  full_pred_t2m=vt['full_pred_t2m'], true_target_t2m=vt['true_target_t2m'],
                                  geos_pred_t2m=vt['geos_mean_t2m'], model_var_t2m=vt['model_var_t2m'],
                                  model_crps_t2m=val_result['avg_crps_t2m'], model_rmse_t2m=val_result['avg_rmse_t2m'],
                                  geos_crps_t2m=val_result['avg_geos_crps_t2m'], geos_rmse_t2m=val_result['avg_geos_rmse_t2m'])
                    print(f"📸 Validation plot saved for Epoch {epoch}.")
                if epoch % 5 == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save({
                        'epoch': epoch,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'is_variance_phase': is_variance_phase,
                    },
                               os.path.join(output_dir, f"periodic_ckpt_epoch_{epoch}.pt"))
                    print(f"💾 Periodic checkpoint: periodic_ckpt_epoch_{epoch}.pt")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'is_variance_phase': is_variance_phase,
                },
                           os.path.join(output_dir, "latest_flow_ckpt.pt"))
                with open(log_file, "a") as f: csv.writer(f).writerow([epoch, avg_train_loss, current_val_metric, val_result['avg_crps_pr']])
        # ============================================================
        #  PHASE 1 (epoch <= 100): MSE-based validation
        # ============================================================
        else:
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
                    target_norm = batch['y_target'].to(device)
                    t = flow_matcher.sample_time_batch(B)
                    noise = torch.randn_like(target_norm)
                    x_t, v_target = flow_matcher.interpolate(target_norm, noise, t)
                    v_pred, var_pred = model(x_t, x_cond, t, lead_idx=lead_idx)
                    w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
                    temp_weights = w_escalation[lead_idx].view(B, 1, 1, 1).expand(-1, 2, -1, -1)
                    
                    loss_vel = (spatial_weights * temp_weights * (v_pred - v_target)**2).mean()
                    loss_var = compute_multi_variance_loss(
                        var_pred=var_pred,
                        v_pred=v_pred,
                        v_target=v_target,
                        spatial_weights=spatial_weights,
                        temp_weights=temp_weights,
                    )

                    if is_variance_phase:
                        loss_val = loss_var
                    else:
                        loss_val = loss_vel
                    val_loss_total += loss_val.item()
                    val_steps += 1
        
            current_val_metric = val_loss_total / max(1, val_steps)
            
            if accelerator.is_main_process:
                print(f"✅ MSE Validation Complete. Avg Loss: {current_val_metric:.4f}")

                is_new_best = (current_val_metric < best_val_loss)
                
                if is_new_best:
                    print(f"🏆 NEW BEST (MSE)! {current_val_metric:.4f} (Prev: {best_val_loss:.4f})")
                    best_val_loss = current_val_metric

                    new_best_name = f"best_model_epoch_{epoch}_loss_{current_val_metric:.4f}.pt"
                    new_best_path = os.path.join(output_dir, new_best_name)
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_ckpt = {
                        'epoch': epoch,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'is_variance_phase': is_variance_phase,
                    }
                    torch.save(best_ckpt, new_best_path)
                    torch.save(best_ckpt, os.path.join(output_dir, "best_flow_ckpt.pt"))
                    registry_path = os.path.join(output_dir, "model_registry.json")
                    registry = json.load(open(registry_path)) if os.path.exists(registry_path) else []
                    registry.append({"rank": 0, "path": new_best_path, "val_loss": current_val_metric, "epoch": epoch, "metric": "mse"})
                    registry.sort(key=lambda x: x['val_loss'])
                    for i, entry in enumerate(registry): entry['rank'] = i + 1
                    with open(registry_path, 'w') as f: json.dump(registry, f, indent=2)
                    print(f"📋 Registry: {len(registry)} models tracked.")
                
                    if epoch >= 20:
                        print(f"📊 Generating PR+T2M validation plot for Epoch {epoch}...")
                        val_result = run_val_inference(
                            epoch, model, val_loader, flow_matcher, device, accelerator, output_dir, log_file, 
                            target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, 
                            is_test=False, is_fast_recon=True,
                            use_flow_variance=is_variance_phase,
                            use_eof_lhs_noise=is_variance_phase,
                            validation_noise_cache=validation_noise_cache,
                            print_validation_noise_diag=False,
                            eof_bases=eof_bases, nao_bases=nao_bases, nao_lookup=nao_lookup,
                            enso_bases=enso_bases, oni_lookup=oni_lookup, mjo_df=mjo_df,
                            t2m_eof_bases=t2m_eof_bases, t2m_nao_bases=t2m_nao_bases, t2m_enso_bases=t2m_enso_bases,
                            validation_num_ensemble=validation_num_ensemble,
                            validation_num_steps=validation_num_steps,
                            validation_ode_batch_size=validation_ode_batch_size,
                            validation_rho_pr=validation_rho_pr,
                            validation_rho_t2m=validation_rho_t2m,
                            validation_var_beta_pr=validation_var_beta_pr,
                            validation_var_beta_t2m=validation_var_beta_t2m,
                        )
                        vt = val_result['tensors']
                        save_val_plot(epoch, vt['full_pred'], vt['true_target'], 
                                      val_result['avg_crps_pr'], val_result['avg_rmse_pr'], 
                                      vt['geos_mean'], val_result['avg_geos_crps_pr'], val_result['avg_geos_rmse_pr'], 
                                      output_dir, ai_residual=vt['ai_res'], suffix="best",
                                      geos_single=vt['geos_single'], model_single=vt['model_single'], model_var=vt['model_var'],
                                      full_pred_t2m=vt['full_pred_t2m'], true_target_t2m=vt['true_target_t2m'],
                                      geos_pred_t2m=vt['geos_mean_t2m'], model_var_t2m=vt['model_var_t2m'],
                                      model_crps_t2m=val_result['avg_crps_t2m'], model_rmse_t2m=val_result['avg_rmse_t2m'],
                                      geos_crps_t2m=val_result['avg_geos_crps_t2m'], geos_rmse_t2m=val_result['avg_geos_rmse_t2m'])
                        print(f"📸 Validation plot saved for Epoch {epoch}.")

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'is_variance_phase': is_variance_phase,
                },
                           os.path.join(output_dir, "latest_flow_ckpt.pt"))
                with open(log_file, "a") as f:
                    csv.writer(f).writerow([epoch, avg_train_loss, 0.0, current_val_metric])

        # Track progress for this execution session
        epochs_done_this_run += 1

def main():
    parser = argparse.ArgumentParser(description="Train or Test Flow Matching Model")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
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
