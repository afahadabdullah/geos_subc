import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from dataset_flow_multi import (
    S2SHybridDataset,
    crop_spatial_to_domain,
    get_target_domain_coords,
    resolve_target_domain,
)
from flow_matching_multi_v7 import FlowMatchingModel, CustomFlowMatcher
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


def compute_crps_tensor(ensemble_preds, target, spatial_weights):
    """
    Differentiable CRPS reduction for training.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    spatial_weights: [1, 1, H, W] or broadcastable equivalent
    """
    mask = ~torch.isnan(target)
    if not mask.any():
        return ensemble_preds.sum() * 0.0

    E = ensemble_preds.shape[0]
    diff = torch.abs(ensemble_preds - target.unsqueeze(0))
    mae_term = diff.mean(dim=0)

    spread_term = torch.zeros_like(mae_term)
    if E > 1:
        for i in range(E - 1):
            pairwise = torch.abs(ensemble_preds[i + 1:] - ensemble_preds[i:i + 1]).sum(dim=0)
            spread_term = spread_term + pairwise
        spread_term = spread_term / (E * E)

    crps_map = mae_term - spread_term
    crps_map_clean = torch.where(mask, crps_map, torch.zeros_like(crps_map))
    weights_expanded = spatial_weights.expand_as(crps_map_clean)
    weights_clean = torch.where(mask, weights_expanded, torch.zeros_like(weights_expanded))
    return (crps_map_clean * weights_clean).sum() / (weights_clean.sum() + 1e-8)

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
        aw_expanded = area_weights.expand_as(mse_map)
        rmse = torch.sqrt((mse_map[mask] * aw_expanded[mask]).sum() / (aw_expanded[mask].sum() + 1e-8)).item()
    else:
        rmse = 0.0
    return rmse


def sample_deterministic_validation_state(target_norm: torch.Tensor, batch_index: int, base_seed: int, device):
    """
    Build a deterministic validation interpolation state for fast MSE validation.
    The same validation batch always receives the same sampled t and Gaussian noise,
    which removes Monte Carlo jitter from epoch-to-epoch comparisons.
    """
    batch_size = target_norm.shape[0]
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(int(base_seed) + int(batch_index))
    t = torch.rand((batch_size,), generator=cpu_gen, dtype=torch.float32).to(device)
    noise = torch.randn(target_norm.shape, generator=cpu_gen, dtype=target_norm.dtype).to(device)
    return t, noise


def get_batch_global_context(batch, device):
    global_context = batch.get("x_global_context")
    if global_context is None or global_context.shape[1] == 0:
        return None
    return global_context.to(device)


def expand_for_ensemble(tensor, num_ensemble):
    if tensor is None:
        return None
    batch_size = tensor.shape[0]
    return (
        tensor.unsqueeze(1)
        .expand(batch_size, num_ensemble, *tensor.shape[1:])
        .reshape(batch_size * num_ensemble, *tensor.shape[1:])
    )


def generate_compare_eof_lhs_noise(
    batch,
    num_ensemble,
    device,
    eof_bases,
    nao_bases,
    enso_bases,
    t2m_eof_bases,
    t2m_nao_bases,
    t2m_enso_bases,
    nao_lookup,
    oni_lookup,
    mjo_df,
    rho_pr,
    rho_t2m,
):
    current_year = int(batch["year"][0].item()) if "year" in batch else 2021
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
    return noise_utils_multi.mix_noise_with_random_multi(eof_noise, rho_pr, rho_t2m)


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
    variance_coarse_kernel=None,
    chunk_size=None,
    global_context=None,
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
            variance_coarse_kernel=variance_coarse_kernel,
            global_context=global_context,
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
                variance_coarse_kernel=variance_coarse_kernel,
                global_context=global_context[start:end] if global_context is not None else None,
            )
        )
    return torch.cat(outputs, dim=0)


def euler_solve_train_chunked(
    flow_matcher,
    model,
    noise,
    x_cond,
    num_steps,
    lead_idx,
    chunk_size=None,
    use_checkpoint=False,
    global_context=None,
):
    """
    Gradient-enabled chunked Euler solve for CRPS fine-tuning.
    Chunking reduces per-forward memory pressure, but total activation memory
    still scales with ensemble size because all trajectories participate in CRPS.
    """
    if chunk_size is None or chunk_size <= 0 or noise.shape[0] <= chunk_size:
        return flow_matcher.euler_solve_differentiable(
            model,
            noise,
            x_cond,
            num_steps=num_steps,
            lead_idx=lead_idx,
            use_checkpoint=use_checkpoint,
            global_context=global_context,
        )

    outputs = []
    for start in range(0, noise.shape[0], chunk_size):
        end = min(start + chunk_size, noise.shape[0])
        outputs.append(
            flow_matcher.euler_solve_differentiable(
                model,
                noise[start:end],
                x_cond[start:end],
                num_steps=num_steps,
                lead_idx=lead_idx[start:end] if lead_idx is not None else None,
                use_checkpoint=use_checkpoint,
                global_context=global_context[start:end] if global_context is not None else None,
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


def decode_multi_forecast_raw(pred_norm, target_sqrt_min, target_sqrt_max, t2m_min=200.0, t2m_max=320.0):
    """
    Reverse the normalized flow output back to physical PR/T2M units.
    pred_norm: [..., 2, H, W]
    returns:   [..., 2, H, W]
    """
    pred_pr = torch.clamp(pred_norm[..., 0:1, :, :], min=-1.0, max=1.0)
    pred_pr_sqrt = ((pred_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    pred_pr_raw = torch.clamp(pred_pr_sqrt ** 2, min=0.0)

    pred_t2m = torch.clamp(pred_norm[..., 1:2, :, :], min=-1.0, max=1.0)
    pred_t2m_raw = ((pred_t2m + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min
    return torch.cat([pred_pr_raw, pred_t2m_raw], dim=-3)


def decode_t2m_forecast_from_norm(
    pred_t2m_norm,
    batch,
    geos_ens_raw,
    t2m_target_mode="absolute",
    t2m_residual_min=-20.0,
    t2m_residual_max=20.0,
):
    """
    Decode normalized T2M model output to absolute Kelvin.

    v7 can train channel 1 as a GEOS-residual target:
        ERA5_T2M - GEOS_ensemble_mean_T2M
    In that mode, inference must add the decoded residual back to the same
    GEOS lead used by the flattened sample.
    """
    pred_t2m_norm = torch.clamp(pred_t2m_norm, min=-1.0, max=1.0)
    if t2m_target_mode == "geos_residual":
        residual = ((pred_t2m_norm + 1.0) / 2.0) * (t2m_residual_max - t2m_residual_min) + t2m_residual_min
        lead_idx = batch["lead_idx"].to(pred_t2m_norm.device).long()
        geos_t2m_all = geos_ens_raw[:, :, 1]
        _, members, _, height, width = geos_t2m_all.shape
        gather_idx = lead_idx.view(-1, 1, 1, 1, 1).expand(-1, members, 1, height, width)
        geos_t2m_lead = geos_t2m_all.gather(2, gather_idx).squeeze(2).mean(dim=1)
        return geos_t2m_lead.unsqueeze(1) + residual

    t2m_min, t2m_max = 200.0, 320.0
    return ((pred_t2m_norm + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min


def resolve_t2m_residual_bounds(config, global_bounds):
    t2m_target_mode = str(config.get("t2m_target_mode", "absolute")).lower()
    if t2m_target_mode not in {"absolute", "geos_residual"}:
        raise ValueError(f"Unsupported t2m_target_mode={t2m_target_mode!r}")

    explicit_min = config.get("t2m_residual_min")
    explicit_max = config.get("t2m_residual_max")
    if explicit_min is not None and explicit_max is not None:
        return t2m_target_mode, float(explicit_min), float(explicit_max)

    residual_bounds = global_bounds.get("target_t2m_residual_raw")
    if t2m_target_mode == "geos_residual":
        if residual_bounds is None:
            raise KeyError(
                "t2m_target_mode=geos_residual requires target_t2m_residual_raw "
                "in the stats file. Run calculate_global_local_stats_multi_v7.py."
            )
        return t2m_target_mode, float(residual_bounds["min"]), float(residual_bounds["max"])

    return t2m_target_mode, -20.0, 20.0


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


def crop_eof_bases_to_domain(eof_bases, domain_info):
    if eof_bases is None or domain_info is None:
        return eof_bases

    target_shape = (len(domain_info["lats"]), len(domain_info["lons"]))
    cropped = {}
    for key, value in eof_bases.items():
        item = dict(value)
        if "eofs" in item:
            eof_shape = tuple(item["eofs"].shape[-2:])
            if eof_shape != target_shape:
                item["eofs"] = crop_spatial_to_domain(item["eofs"], domain_info)
        cropped[key] = item
    return cropped


def compute_multi_variance_loss(
    var_pred: torch.Tensor,
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    spatial_weights: torch.Tensor,
    temp_weights: torch.Tensor,
    variance_coarse_kernel=None,
):
    """
    Match the v4 variance-head objective more closely:
    learn a relative standard-deviation multiplier rather than a squared variance target.
    """
    abs_err = torch.abs(v_target - v_pred.detach())
    target_scale = abs_err / (abs_err.mean(dim=(2, 3), keepdim=True) + 1e-6)

    std_mult = torch.sqrt(var_pred + 1e-6)
    std_for_loss = std_mult
    if variance_coarse_kernel is not None:
        kernel = int(variance_coarse_kernel)
        if kernel > 1:
            coarse = F.avg_pool2d(
                std_for_loss.float(),
                kernel_size=kernel,
                stride=kernel,
                ceil_mode=True,
                count_include_pad=False,
            )
            std_for_loss = F.interpolate(
                coarse,
                size=std_for_loss.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(std_for_loss.dtype)
            std_for_loss = torch.clamp(std_for_loss, min=0.1, max=2.0)

    loss_mse_var = (spatial_weights * temp_weights * (std_for_loss - target_scale) ** 2).mean()

    # Keep the multiplier near a physically reasonable range while gently pulling toward 1.0.
    var_penalty = torch.relu(std_mult - 2.5) ** 2 + torch.relu(0.5 - std_mult) ** 2
    identity_pull = (std_mult - 1.0) ** 2
    loss_reg = (var_penalty * 10.0 + identity_pull * 0.5).mean()

    return loss_mse_var + loss_reg


def compute_multi_crps_training_loss(
    flow_matcher,
    model,
    x_cond,
    lead_idx,
    target_raw,
    num_ensemble,
    num_steps,
    target_sqrt_min,
    target_sqrt_max,
    spatial_weights,
    ode_chunk_size=None,
    max_ensemble_per_chunk=None,
    pr_weight=1.0,
    t2m_weight=1.0,
    use_checkpoint=False,
    global_context=None,
):
    """
    Fine-tuning loss that backpropagates through a short Euler rollout and
    scores the resulting ensemble with differentiable CRPS in raw PR/T2M units.
    Uses pure Gaussian noise only.
    """
    if num_ensemble < 1:
        raise ValueError(f"num_ensemble must be >= 1, got {num_ensemble}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    total_weight = float(pr_weight) + float(t2m_weight)
    if total_weight <= 0.0:
        raise ValueError("At least one of pr_weight or t2m_weight must be > 0.")

    B, _, H, W = target_raw.shape
    device = target_raw.device
    dtype = target_raw.dtype

    x_cond_expanded = x_cond.unsqueeze(1).expand(B, num_ensemble, -1, H, W).reshape(B * num_ensemble, -1, H, W)
    lead_idx_expanded = lead_idx.unsqueeze(1).expand(B, num_ensemble).reshape(-1).long()
    if global_context is not None:
        gc, gh, gw = global_context.shape[1], global_context.shape[2], global_context.shape[3]
        global_context_expanded = (
            global_context.unsqueeze(1)
            .expand(B, num_ensemble, gc, gh, gw)
            .reshape(B * num_ensemble, gc, gh, gw)
        )
    else:
        global_context_expanded = None
    noise = torch.randn((B * num_ensemble, 2, H, W), device=device, dtype=dtype)

    chunk_size = ode_chunk_size
    if max_ensemble_per_chunk is not None:
        max_ensemble_chunk = max(1, int(max_ensemble_per_chunk))
        max_state_chunk = B * max_ensemble_chunk
        chunk_size = max_state_chunk if chunk_size is None else min(chunk_size, max_state_chunk)

    pred_norm = euler_solve_train_chunked(
        flow_matcher,
        model,
        noise,
        x_cond_expanded,
        num_steps=num_steps,
        lead_idx=lead_idx_expanded,
        chunk_size=chunk_size,
        use_checkpoint=use_checkpoint,
        global_context=global_context_expanded,
    )

    pred_raw = decode_multi_forecast_raw(pred_norm, target_sqrt_min, target_sqrt_max)
    pred_raw = pred_raw.view(B, num_ensemble, 2, H, W).transpose(0, 1).float()
    target_raw = target_raw.float()
    spatial_weights = spatial_weights.float()

    pr_loss = compute_crps_tensor(pred_raw[:, :, 0:1], target_raw[:, 0:1], spatial_weights)
    t2m_loss = compute_crps_tensor(pred_raw[:, :, 1:2], target_raw[:, 1:2], spatial_weights)
    combined_loss = (float(pr_weight) * pr_loss + float(t2m_weight) * t2m_loss) / total_weight

    return combined_loss, {
        "crps_pr": pr_loss.detach(),
        "crps_t2m": t2m_loss.detach(),
        "ensemble_size": num_ensemble,
        "num_steps": num_steps,
        "chunk_size": chunk_size if chunk_size is not None else B * num_ensemble,
    }

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
                      validation_var_beta_t2m=None,
                      validation_variance_coarse_kernel=None,
                      t2m_target_mode="absolute",
                      t2m_residual_min=-20.0,
                      t2m_residual_max=20.0):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    total_crps = 0.0
    total_rmse = 0.0
    total_crps_t2m = 0.0
    total_rmse_t2m = 0.0
    total_geos_crps = 0.0
    total_geos_rmse = 0.0
    total_geos_crps_t2m = 0.0
    total_geos_rmse_t2m = 0.0
    count = 0
    processed_batches = 0
    did_print_noise_diag = False
    if validation_rho_t2m is None:
        validation_rho_t2m = validation_rho_pr
    if validation_var_beta_t2m is None:
        validation_var_beta_t2m = validation_var_beta_pr
    
    # Save tensors for the first validation batch we actually process.
    saved_tensors = {}
    
    for b_idx, batch in enumerate(val_loader):
        processed_batches += 1
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
        fx_global_context = get_batch_global_context(batch, device)
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
        if fx_global_context is not None:
            gc, gh, gw = fx_global_context.shape[1:]
            fx_global_context_expanded = (
                fx_global_context.unsqueeze(1)
                .expand(vB, num_ensemble, gc, gh, gw)
                .reshape(vB * num_ensemble, gc, gh, gw)
            )
        else:
            fx_global_context_expanded = None
        
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
            if validation_variance_coarse_kernel is not None:
                mode_tag += f"_coarse{int(validation_variance_coarse_kernel)}"
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
            variance_coarse_kernel=validation_variance_coarse_kernel,
            chunk_size=validation_ode_batch_size,
            global_context=fx_global_context_expanded,
        )

        if print_validation_noise_diag and not did_print_noise_diag and accelerator.is_main_process:
            if use_eof_lhs_noise:
                mode_label = (
                    f"PR EOF-LHS rho={validation_rho_pr:.2f} beta={validation_var_beta_pr:.2f} / "
                    f"T2M EOF-LHS rho={validation_rho_t2m:.2f} beta={validation_var_beta_t2m:.2f}"
                )
                if validation_variance_coarse_kernel is not None:
                    mode_label += f" + coarse var{int(validation_variance_coarse_kernel)}"
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
        
        p_x1_t2m = torch.clamp(p_x1_batch[:, :, 1], min=-1.0, max=1.0)
        week_t2m = decode_t2m_forecast_from_norm(
            p_x1_t2m,
            batch,
            geos_ens_raw,
            t2m_target_mode=t2m_target_mode,
            t2m_residual_min=t2m_residual_min,
            t2m_residual_max=t2m_residual_max,
        )
        
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
    
    recon_type = f"Monthly2021 (batches={processed_batches}, inits={count}, ens={num_ensemble})"
    combined_crps = (avg_crps + avg_crps_t2m) / 2.0
    if accelerator.is_main_process:
        print(f"Epoch {epoch} | Val Loader Summary: {processed_batches} batches, {count} init dates")
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
    validation_variance_coarse_kernel=None,
    sample_plot_limit=None,
    plot_subdir="test_plots_multi",
    t2m_target_mode="absolute",
    t2m_residual_min=-20.0,
    t2m_residual_max=20.0,
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
        fx_global_context = get_batch_global_context(batch, device)
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
        if fx_global_context is not None:
            gc, gh, gw = fx_global_context.shape[1:]
            fx_global_context_expanded = (
                fx_global_context.unsqueeze(1)
                .expand(vB, num_ensemble, gc, gh, gw)
                .reshape(vB * num_ensemble, gc, gh, gw)
            )
        else:
            fx_global_context_expanded = None

        current_year = int(batch['year'][0].item()) if 'year' in batch else 2021
        current_month = int(batch['month'][0].item())
        current_day = int(batch['day'][0].item()) if 'day' in batch else 15
        mode_tag = "pure_random"
        if use_eof_lhs_noise:
            mode_tag = f"pr_t2m_eof_lhs_rho_pr{validation_rho_pr:.2f}_rho_t2m{validation_rho_t2m:.2f}"
        if use_flow_variance:
            mode_tag += f"_beta_pr{validation_var_beta_pr:.2f}_beta_t2m{validation_var_beta_t2m:.2f}"
            if validation_variance_coarse_kernel is not None:
                mode_tag += f"_coarse{int(validation_variance_coarse_kernel)}"
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
            variance_coarse_kernel=validation_variance_coarse_kernel,
            chunk_size=chunk_size,
            global_context=fx_global_context_expanded,
        )

        p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)
        p_x1_pr = torch.clamp(p_x1_batch[:, :, 0], min=-1.0, max=1.0)
        week_sqrt = ((p_x1_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        week_precip = torch.clamp(week_sqrt ** 2, min=0.0)

        p_x1_t2m = torch.clamp(p_x1_batch[:, :, 1], min=-1.0, max=1.0)
        week_t2m = decode_t2m_forecast_from_norm(
            p_x1_t2m,
            batch,
            geos_ens_raw,
            t2m_target_mode=t2m_target_mode,
            t2m_residual_min=t2m_residual_min,
            t2m_residual_max=t2m_residual_max,
        )

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

    data_dir_override = os.environ.get("DATA_DIR_OVERRIDE")
    if data_dir_override:
        config["data_dir"] = data_dir_override

    epochs = config.get("epochs", 500)
    batch_size = config.get("batch_size", 4)
    lr = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    reset_optimizer_state = bool(config.get("reset_optimizer_state", False))

    target_domain = config.get("target_domain")
    target_domain_bounds = config.get("target_domain_bounds")
    t2m_target_mode = str(config.get("t2m_target_mode", "absolute")).lower()
    if t2m_target_mode not in {"absolute", "geos_residual"}:
        raise ValueError(f"Unsupported t2m_target_mode={t2m_target_mode!r}")
    t2m_residual_min_cfg = config.get("t2m_residual_min")
    t2m_residual_max_cfg = config.get("t2m_residual_max")
    domain_info = resolve_target_domain(target_domain, target_domain_bounds)
    lats, lons = get_target_domain_coords(target_domain, target_domain_bounds)
    grid_h, grid_w = len(lats), len(lons)

    # Area weights (needed early for diagnostic plot)
    area_weights = get_area_weights(lats, device)

    # ─── Land-Ocean Mask (V6: 65% land / 35% ocean) ───
    # Derive from SSS data: NaN pixels = land, valid pixels = ocean
    # Cache to .pt file so we only need SSS once.
    land_ocean_weights = torch.ones(1, 1, grid_h, grid_w, device=device)  # Default: uniform
    domain_slug = "global" if domain_info is None else "".join(
        c if c.isalnum() else "_" for c in str(target_domain or domain_info["label"]).lower()
    ).strip("_")
    mask_cache_name = "land_ocean_mask_v6.pt" if domain_info is None else f"land_ocean_mask_v6_{domain_slug}.pt"
    mask_cache_path = os.path.join(os.path.dirname(__file__), mask_cache_name)
    needs_mask_build = True
    
    if os.path.exists(mask_cache_path):
        # ── Load cached mask ──
        cached = torch.load(mask_cache_path, map_location=device, weights_only=True)
        cached_weights = cached['weights'].to(device)
        if tuple(cached_weights.shape[-2:]) == (grid_h, grid_w):
            land_ocean_weights = cached_weights
            needs_mask_build = False
            if accelerator.is_main_process:
                print(f"  ✅ V6 Land-Ocean Mask loaded from cache: {mask_cache_path}")
                print(f"     Land pixels: {cached['n_land']}, weight = {cached['land_w']:.4f}")
                print(f"     Ocean pixels: {cached['n_ocean']}, weight = {cached['ocean_w']:.4f}")
        elif accelerator.is_main_process:
            print(
                f"  ⚠️ Cached land-ocean mask shape {tuple(cached_weights.shape[-2:])} "
                f"does not match target grid {(grid_h, grid_w)}. Rebuilding."
            )

    if needs_mask_build:
        # ── Create mask from SSS ──
        sss_sample_path = os.path.join(config["data_dir"], "sss_weekly_2020.zarr")
        if os.path.exists(sss_sample_path):
            try:
                ds_sss = xr.open_zarr(sss_sample_path, consolidated=False)
                sss_arr = ds_sss['sss'].isel(S=0, L=0).values  # [Y, X]
                ds_sss.close()
                if domain_info is not None:
                    sss_arr = crop_spatial_to_domain(sss_arr, domain_info)
                is_land = np.isnan(sss_arr)  # True = land
                n_land = int(is_land.sum())
                n_ocean = int((~is_land).sum())
                n_total = n_land + n_ocean
                
                if n_land > 0 and n_ocean > 0:
                    land_w = 0.65 * n_total / n_land
                    ocean_w = 0.35 * n_total / n_ocean
                    
                    mask_np = np.where(is_land, land_w, ocean_w).astype(np.float32)
                    land_ocean_weights = torch.from_numpy(mask_np).to(device).view(1, 1, grid_h, grid_w)
                    
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
                        plot_extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
                        
                        im0 = axes[0].imshow(is_land.astype(float), cmap='RdYlGn', vmin=0, vmax=1,
                                            extent=plot_extent, aspect='auto')
                        axes[0].set_title(f"Land-Ocean Mask (Green=Land, Red=Ocean)\nLand: {n_land} px ({n_land/n_total*100:.1f}%)")
                        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
                        
                        im1 = axes[1].imshow(mask_np, cmap='hot_r',
                                            extent=plot_extent, aspect='auto')
                        axes[1].set_title(f"Loss Weight Map\nLand w={land_w:.3f}, Ocean w={ocean_w:.3f}")
                        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                        
                        combined = (area_weights.cpu().numpy().reshape(grid_h, 1) * mask_np)
                        im2 = axes[2].imshow(combined, cmap='magma',
                                            extent=plot_extent, aspect='auto')
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
    
    val_dataset_full = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False),
        stats_file=stats_filename,
        subsample_monthly=False,
        target_domain=target_domain,
        target_domain_bounds=target_domain_bounds,
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min_cfg,
        t2m_residual_max=t2m_residual_max_cfg,
    )

    val_dataset_monthly = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config.get("crps_val_start_year", config["val_start_year"]),
        end_year=config.get("crps_val_end_year", config["val_end_year"]),
        normalize=True,
        preload=config.get("preload", False),
        stats_file=stats_filename,
        subsample_monthly=True,
        target_domain=target_domain,
        target_domain_bounds=target_domain_bounds,
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min_cfg,
        t2m_residual_max=t2m_residual_max_cfg,
    )

    # Process multiple init dates per validation batch for speed (batch_size * 2 since we flattened leads)
    val_batch_size = max(8, batch_size * 2) 
    
    from torch.utils.data import DataLoader
    val_loader_full = DataLoader(
        val_dataset_full, batch_size=val_batch_size, shuffle=False, drop_last=False,
        num_workers=config.get("num_workers", 4), pin_memory=True
    )
    val_loader_monthly = DataLoader(
        val_dataset_monthly, batch_size=val_batch_size, shuffle=False, drop_last=False,
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
            stats_file=stats_filename,
            target_domain=target_domain,
            target_domain_bounds=target_domain_bounds,
            local_obs_variables=config.get("local_obs_variables"),
            global_context_variables=config.get("global_context_variables"),
            t2m_target_mode=t2m_target_mode,
            t2m_residual_min=t2m_residual_min_cfg,
            t2m_residual_max=t2m_residual_max_cfg,
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
    t2m_target_mode, t2m_residual_min, t2m_residual_max = resolve_t2m_residual_bounds(config, global_bounds)
    # Force robust physical range for Direct Power Transformed GPCP (sqrt)
    # Target raw max is roughly 50 mm/day max in GPCP weekly. sqrt(50) ~= 7.071
    target_sqrt_min = 0.0
    target_sqrt_max = 7.071
    
    geos_min = global_bounds["geos_pr_raw"]["min"] if "geos_pr_raw" in global_bounds else global_bounds["geos_raw"]["min"]
    geos_max = global_bounds["geos_pr_raw"]["max"] if "geos_pr_raw" in global_bounds else global_bounds["geos_raw"]["max"]
    if domain_info is None:
        plot_lats, plot_lons = load_plot_coords(config["data_dir"], config.get("val_end_year"))
    else:
        plot_lats, plot_lons = lats, lons

    variance_phase_lr = float(config.get("variance_phase_lr", 1e-4))
    force_variance_phase = bool(config.get("force_variance_phase", False))
    variance_phase_start_epoch = int(config.get("variance_phase_start_epoch", 0))
    crps_loss = bool(config.get("crps_loss", False))
    crps_loss_num_ensemble = int(config.get("crps_loss_num_ensemble", 4))
    crps_loss_num_steps = int(config.get("crps_loss_num_steps", 10))
    crps_loss_ode_batch_size = int(config.get("crps_loss_ode_batch_size", max(1, batch_size * crps_loss_num_ensemble)))
    crps_loss_max_ensemble_per_chunk = int(config.get("crps_loss_max_ensemble_per_chunk", crps_loss_num_ensemble))
    crps_loss_pr_weight = float(config.get("crps_loss_pr_weight", 1.0))
    crps_loss_t2m_weight = float(config.get("crps_loss_t2m_weight", 1.0))
    crps_loss_use_land_ocean_weights = bool(config.get("crps_loss_use_land_ocean_weights", False))
    crps_loss_use_gradient_checkpointing = bool(config.get("crps_loss_use_gradient_checkpointing", True))
    if crps_loss and t2m_target_mode == "geos_residual":
        raise ValueError(
            "crps_loss=True is not wired for v7 residual T2M targets yet. "
            "Use velocity/variance training or add residual-aware CRPS decoding."
        )
    validation_num_ensemble = int(config.get("validation_num_ensemble", 15))
    validation_num_steps = int(config.get("validation_num_steps", 10))
    validation_ode_batch_size = int(config.get("validation_ode_batch_size", 120))
    mse_validation_seed = int(config.get("mse_validation_seed", 1234))
    crps_val_start_year = int(config.get("crps_val_start_year", config["val_start_year"]))
    crps_val_end_year = int(config.get("crps_val_end_year", config["val_end_year"]))
    validation_variance_coarse_kernel = config.get("validation_variance_coarse_kernel", None)
    if validation_variance_coarse_kernel is not None:
        validation_variance_coarse_kernel = int(validation_variance_coarse_kernel)
    test_num_ensemble = int(config.get("test_num_ensemble", 90))
    test_num_steps = int(config.get("test_num_steps", 10))
    test_max_ensemble_per_chunk = int(config.get("test_max_ensemble_per_chunk", 30))
    test_sample_plot_limit_cfg = config.get("test_sample_plot_limit", 24)
    if test_sample_plot_limit_cfg is None:
        test_sample_plot_limit = None
    else:
        test_sample_plot_limit = int(test_sample_plot_limit_cfg)
        if test_sample_plot_limit <= 0:
            test_sample_plot_limit = None
    validation_rho_pr = float(config.get("validation_rho_pr", 1.0))
    validation_rho_t2m = float(config.get("validation_rho_t2m", validation_rho_pr))
    validation_var_beta_pr = float(config.get("validation_var_beta_pr", 1.0))
    validation_var_beta_t2m = float(config.get("validation_var_beta_t2m", validation_var_beta_pr))
    variance_training_num_ensemble = int(config.get("variance_training_num_ensemble", validation_num_ensemble))
    if variance_training_num_ensemble < 1:
        raise ValueError(f"variance_training_num_ensemble must be >= 1, got {variance_training_num_ensemble}")
    dense_mse_validation_until = int(config.get("dense_mse_validation_until", 0))
    plot_validation_every = int(config.get("plot_validation_every", 0))
    mse_early_stop_patience = int(config.get("mse_early_stop_patience", 0))
    mse_early_stop_start_epoch = int(config.get("mse_early_stop_start_epoch", 0))
    mse_early_stop_min_delta = float(config.get("mse_early_stop_min_delta", 0.0))
    mse_plateau_patience = int(config.get("mse_plateau_patience", 0))
    mse_plateau_factor = float(config.get("mse_plateau_factor", 0.5))
    mse_min_lr = float(config.get("mse_min_lr", 0.0))
    unet_block_out_channels = tuple(int(v) for v in config.get("unet_block_out_channels", [128, 256, 512, 768]))
    if len(unet_block_out_channels) != 4:
        raise ValueError(f"unet_block_out_channels must have length 4, got {unet_block_out_channels}")
    obs_channel_count = int(val_dataset_full.obs_channel_count)
    geos_channel_count = 2 * 4
    temporal_channel_count = 3
    cond_channel_count = obs_channel_count + geos_channel_count + temporal_channel_count
    model_in_channels = cond_channel_count + 2
    global_context_channel_count = int(val_dataset_full.global_context_channel_count)

    if crps_loss and (force_variance_phase or variance_phase_start_epoch > 0):
        if accelerator.is_main_process:
            print("   ℹ️ crps_loss=True overrides variance-head-only phase settings. Using CRPS fine-tuning mode.")
        force_variance_phase = False
        variance_phase_start_epoch = 0
    
    if accelerator.is_main_process:
        print("\n=======================================================")
        print(f"✅ Loaded Strict Global Stats: {stats_file}")
        print(f"   [Target SQRT Bounds] : Min = {target_sqrt_min:.4f}, Max = {target_sqrt_max:.4f}")
        print(f"   [T2M Target Mode]    : {t2m_target_mode}")
        if t2m_target_mode == "geos_residual":
            print(f"   [T2M Residual Bounds]: Min = {t2m_residual_min:.4f}, Max = {t2m_residual_max:.4f}")
        print(f"   [GEOS Raw Bounds]    : Min = {geos_min:.4f}, Max = {geos_max:.4f}")
        if domain_info is not None:
            print(f"   [Target Domain]      : {domain_info['label']} ({grid_h}x{grid_w})")
            print(f"   [Target Lat Range]   : {float(lats.min()):.2f} .. {float(lats.max()):.2f}")
            print(f"   [Target Lon Range]   : {float(lons.min()):.2f} .. {float(lons.max()):.2f}")
        else:
            print(f"   [Target Domain]      : Global ({grid_h}x{grid_w})")
        print(f"   [Plot Lon Range]     : {float(plot_lons.min()):.2f} .. {float(plot_lons.max()):.2f}")
        if crps_loss:
            print(f"   [Training Mode]      : CRPS fine-tune (pure Gaussian rollout)")
        elif force_variance_phase:
            print(f"   [Training Mode]      : Variance-only (lr={variance_phase_lr:.2e})")
        else:
            print(f"   [Training Mode]      : Velocity-only")
        if variance_phase_start_epoch > 0:
            print(f"   [Variance Phase]     : starts at epoch {variance_phase_start_epoch}, lr={variance_phase_lr:.2e}")
            print(
                f"   [Variance Train]     : EOF-LHS compare noise, ens={variance_training_num_ensemble}, "
                f"rho PR/T2M={validation_rho_pr:.2f}/{validation_rho_t2m:.2f}, "
                f"beta PR/T2M={validation_var_beta_pr:.2f}/{validation_var_beta_t2m:.2f}, "
                f"coarse={validation_variance_coarse_kernel}"
            )
        if crps_loss:
            train_weight_mode = "area x land-ocean" if crps_loss_use_land_ocean_weights else "area-only"
            rollout_states = batch_size * crps_loss_num_ensemble
            print(f"   [CRPS Train Ens]     : {crps_loss_num_ensemble}")
            print(f"   [CRPS Train Steps]   : {crps_loss_num_steps}")
            print(f"   [CRPS Train Chunk]   : {crps_loss_ode_batch_size}")
            print(f"   [CRPS Ens/Chunk]     : {crps_loss_max_ensemble_per_chunk}")
            print(f"   [CRPS Weights]       : PR={crps_loss_pr_weight:.2f}, T2M={crps_loss_t2m_weight:.2f}")
            print(f"   [CRPS Spatial Wt]    : {train_weight_mode}")
            print(f"   [CRPS Grad Ckpt]     : {crps_loss_use_gradient_checkpointing}")
            print(f"   [CRPS Cost Hint]     : ~{rollout_states} states x {crps_loss_num_steps} Euler steps per batch")
        print(f"   [Validation Ens]     : {validation_num_ensemble}")
        print(f"   [Validation Steps]   : {validation_num_steps}")
        print(f"   [Validation Chunk]   : {validation_ode_batch_size}")
        print(f"   [MSE Val Seed]       : {mse_validation_seed}")
        print(f"   [MSE Val Dataset]    : Full {config['val_start_year']}-{config['val_end_year']}")
        print(f"   [CRPS Val Dataset]   : Monthly subset {crps_val_start_year}-{crps_val_end_year}")
        print(f"   [Validation Rho]     : PR={validation_rho_pr:.2f}, T2M={validation_rho_t2m:.2f}")
        print(f"   [Validation Beta]    : PR={validation_var_beta_pr:.2f}, T2M={validation_var_beta_t2m:.2f}")
        print(f"   [Validation Coarse]  : {validation_variance_coarse_kernel}")
        print(f"   [Dense MSE Val To]   : {dense_mse_validation_until if dense_mse_validation_until > 0 else 'disabled'}")
        print(f"   [Val Plot Every]     : {plot_validation_every if plot_validation_every > 0 else 'best-only'}")
        print(
            f"   [MSE Plateau]       : patience={mse_plateau_patience}, "
            f"factor={mse_plateau_factor:.2f}, min_lr={mse_min_lr:.2e}"
        )
        print(
            f"   [MSE Early Stop]    : patience={mse_early_stop_patience}, "
            f"start={mse_early_stop_start_epoch}, min_delta={mse_early_stop_min_delta:.2e}"
        )
        print(f"   [UNet Widths]        : {' -> '.join(str(v) for v in unet_block_out_channels)}")
        print(f"   [Local Predictors]   : {list(val_dataset_full.local_obs_variables)} ({obs_channel_count} channels)")
        print(f"   [Global Context]     : {list(val_dataset_full.global_context_variables)} ({global_context_channel_count} channels)")
        print(f"   [Test Ens]           : {test_num_ensemble}")
        print(f"   [Test Steps]         : {test_num_steps}")
        print(f"   [Test Ens/Chunk]     : {test_max_ensemble_per_chunk}")
        print(f"   [Test Plot Limit]    : {test_sample_plot_limit if test_sample_plot_limit is not None else 'all'}")
        print(f"   [Optimizer]          : AdamW lr={lr:.2e}, wd={weight_decay:.2e}")
        print(f"   [Reset Optimizer]    : {reset_optimizer_state}")
        print("=======================================================\n")

    # ---------------------------------------------------------
    # 2. Model & Scheduler Setup
    # ---------------------------------------------------------
    model = FlowMatchingModel(
        in_channels=model_in_channels,
        out_channels=2,
        block_out_channels=unet_block_out_channels,
        sample_size=(grid_h, grid_w),
        global_context_channels=global_context_channel_count,
    ).to(device)
    flow_matcher = CustomFlowMatcher(device=device)
    if crps_loss and crps_loss_use_gradient_checkpointing:
        if hasattr(model.unet, "enable_gradient_checkpointing"):
            model.unet.enable_gradient_checkpointing()
        if accelerator.is_main_process:
            print("   ✅ Enabled UNet gradient checkpointing for CRPS fine-tuning.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if not args.test:
        model, optimizer, loader, val_loader_full, val_loader_monthly = accelerator.prepare(
            model, optimizer, loader, val_loader_full, val_loader_monthly
        )
    else:
        # Test mode: only prepare model and validation loaders
        model, val_loader_full, val_loader_monthly = accelerator.prepare(model, val_loader_full, val_loader_monthly)
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
        print(f"   Spatial Grid: {grid_h} x {grid_w}")
        print(f"   Total Input Channels: {model_in_channels}")
        print(f"   --- Condition Channels (x_cond = {cond_channel_count}) ---")
        offset = 0
        for var_name in val_dataset_full.local_obs_variables:
            print(f"     [{offset:02d}-{offset + 3:02d}] x_obs: {var_name} (L=1 to 4, local target grid)")
            offset += 4
        print(f"     [{offset:02d}-{offset + 3:02d}] x_geos: GEOS Precipitation Forecast (L=1 to 4, local)")
        offset += 4
        print(f"     [{offset:02d}-{offset + 3:02d}] x_geos: GEOS T2M Forecast (L=1 to 4, local)")
        offset += 4
        print(f"     [{offset:02d}] Month: Sine Temporal Embedding")
        print(f"     [{offset + 1:02d}] Month: Cosine Temporal Embedding")
        print(f"     [{offset + 2:02d}] Target Lead: Relative Index Tracking [-1 to +1]")
        if val_dataset_full.global_context_variables:
            print(
                f"   --- Global Context Encoder ({global_context_channel_count} channels, full 181x360 grid) ---"
            )
            for var_name in val_dataset_full.global_context_variables:
                print(f"     global: {var_name} (L=1 to 4)")
        print(f"   --- Dynamic Flow Channel (x_t = 2) ---")
        print(f"     [{cond_channel_count:02d},{cond_channel_count + 1:02d}] x_t: Pure Noise Vector (Solver Substrate) PR & T2M")
        print(f"   --- v7 Conditioning Upgrades ---")
        print(f"     Input FiLM: pooled x_cond summary modulates the {model_in_channels}-channel backbone input")
        print(f"     Lead Embedding: discrete week embedding injected into FiLM context")
        print(f"     UNet block widths: {' -> '.join(str(v) for v in unet_block_out_channels)}")
        print(f"   --- Dedicated Output Heads (Per-Week Mini-Decoders) ---")
        print(f"     Head 0: Week 1 (3x3 -> 3x3 -> 1x1)")
        print(f"     Head 1: Week 2 (3x3 -> 3x3 -> 1x1)")
        print(f"     Head 2: Week 3 (3x3 -> 3x3 -> 1x1)")
        print(f"     Head 3: Week 4 (3x3 -> 3x3 -> 1x1)")
        print(f"     Shared UNet features: 64 intermediate channels")
        print(f"-------------------------------------\n")


    # Output directory
    output_dir = config.get("output_dir", "ml_output_diffusion_v5")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log_v7.csv")
    early_stop_marker = os.path.join(output_dir, "EARLY_STOPPED")
    
    if accelerator.is_main_process and not os.path.exists(log_file):
        with open(log_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Noise", "Val_CRPS"])

    # Fixed Val Batch for continuous plotting
    fixed_val_batch = next(iter(val_loader_full))

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

    eof_bases = crop_eof_bases_to_domain(eof_bases, domain_info)
    nao_bases = crop_eof_bases_to_domain(nao_bases, domain_info)
    enso_bases = crop_eof_bases_to_domain(enso_bases, domain_info)
    t2m_eof_bases = crop_eof_bases_to_domain(t2m_eof_bases, domain_info)
    t2m_nao_bases = crop_eof_bases_to_domain(t2m_nao_bases, domain_info)
    t2m_enso_bases = crop_eof_bases_to_domain(t2m_enso_bases, domain_info)
    
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
    best_val_epoch = 0
    mse_bad_val_checks = 0
    variance_phase_best_reset_done = False
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
                resume_into_variance_phase = (
                    force_variance_phase
                    or loaded_is_variance_phase
                    or (variance_phase_start_epoch > 0 and start_epoch >= variance_phase_start_epoch)
                )
                
                if resume_into_variance_phase:
                    if accelerator.is_main_process:
                        print(
                            "   ℹ️ Variance-only resume detected (forced by config). "
                            "Skipping optimizer state load and rebuilding a var-heads optimizer."
                        )
                else:
                    try:
                        optimizer_state = checkpoint['optimizer']
                        param_groups = optimizer_state.get('param_groups', [])
                        loaded_lr = float(param_groups[0].get('lr', lr)) if param_groups else lr
                        loaded_weight_decay = float(param_groups[0].get('weight_decay', weight_decay)) if param_groups else weight_decay
                        if reset_optimizer_state:
                            if accelerator.is_main_process:
                                print(
                                    "   ℹ️ reset_optimizer_state=True. "
                                    f"Starting fresh optimizer at config lr={lr:.2e}."
                                )
                        elif not np.isclose(loaded_weight_decay, weight_decay):
                            if accelerator.is_main_process:
                                print(
                                    "   ℹ️ Optimizer weight_decay changed since the checkpoint "
                                    f"(ckpt wd={loaded_weight_decay:.2e}; config wd={weight_decay:.2e}). "
                                    "Starting with fresh optimizer state."
                                )
                        else:
                            optimizer.load_state_dict(optimizer_state)
                            print(
                                "   ✅ Loaded optimizer state from checkpoint "
                                f"(current lr={loaded_lr:.2e}, config initial lr={lr:.2e})."
                            )
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
                best_val_epoch = int(checkpoint.get('best_val_epoch', 0))
                mse_bad_val_checks = int(checkpoint.get('mse_bad_val_checks', 0))
                variance_phase_best_reset_done = bool(checkpoint.get('variance_phase_best_reset_done', False))
                
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
                    best_val_epoch = 0
                    mse_bad_val_checks = 0
                    variance_phase_best_reset_done = False
                    top_models = []
                
            if accelerator.is_main_process:
                print(f"\n🔄 Loaded checkpoint: {ckpt_path}")
                if not args.test:
                    print(f"   Starting at Epoch: {start_epoch}")
                    print(f"   Best Val Loss so far: {best_val_loss:.4f}")
                    print(f"   Best Val Epoch so far: {best_val_epoch}")
                    print(f"   MSE plateau counter: {mse_bad_val_checks}")
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

    def enable_variance_phase(reason, reset_best_for_phase=False):
        nonlocal optimizer, is_variance_phase, best_val_loss, best_val_epoch
        nonlocal mse_bad_val_checks, top_models, variance_phase_best_reset_done
        if is_variance_phase:
            return
        if accelerator.is_main_process:
            print(f"🔒 Enabling variance-head-only phase: {reason}")
            print("   Freezing backbone, conditioning layers, and velocity heads; training only var_heads.")
        unwrapped = accelerator.unwrap_model(model)
        for param in unwrapped.parameters():
            param.requires_grad_(False)
        for param in unwrapped.var_heads.parameters():
            param.requires_grad_(True)

        var_params = [p for p in unwrapped.var_heads.parameters() if p.requires_grad]
        if not var_params:
            raise RuntimeError("Variance phase requested, but no trainable var_heads parameters were found.")
        optimizer = torch.optim.AdamW(
            var_params,
            lr=variance_phase_lr,
            weight_decay=weight_decay,
        )
        optimizer = accelerator.prepare(optimizer)
        is_variance_phase = True
        if reset_best_for_phase and not variance_phase_best_reset_done:
            if accelerator.is_main_process:
                print("   Resetting best validation tracker so best_flow_ckpt.pt captures the variance-head phase.")
            best_val_loss = float('inf')
            best_val_epoch = 0
            mse_bad_val_checks = 0
            top_models = []
            variance_phase_best_reset_done = True

    start_in_variance_phase = (
        not args.test
        and not crps_loss
        and (
            force_variance_phase
            or loaded_is_variance_phase
            or (variance_phase_start_epoch > 0 and start_epoch >= variance_phase_start_epoch)
        )
    )

    if start_in_variance_phase:
        if force_variance_phase:
            reason = "force_variance_phase=True"
        elif loaded_is_variance_phase:
            reason = "resuming a variance-phase checkpoint"
        else:
            reason = f"start_epoch={start_epoch} >= variance_phase_start_epoch={variance_phase_start_epoch}"
        if loaded_checkpoint_epoch == 0 and accelerator.is_main_process:
            print("   ⚠️ No checkpoint was loaded. This will train only var_heads from scratch.")
        enable_variance_phase(reason, reset_best_for_phase=not loaded_is_variance_phase)
        
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
            val_loader_full,
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
            validation_variance_coarse_kernel=validation_variance_coarse_kernel,
            sample_plot_limit=test_sample_plot_limit,
            plot_subdir=f"test_plots_multi_e{test_num_ensemble}_s{test_num_steps}",
            t2m_target_mode=t2m_target_mode,
            t2m_residual_min=t2m_residual_min,
            t2m_residual_max=t2m_residual_max,
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
        x_global_context = get_batch_global_context(batch, device)
        target_norm = batch['y_target'].to(device)
        lead_idx_tensor = batch['lead_idx'].to(device)
        
        # Take index 0, lead 1 for visualization
        sample_idx = 0
        lead_idx = 0
        
        # 1. Reverse Normalize GEOS
        geos_norm_sample = np.nan_to_num(x_geos[sample_idx, 0, 0, lead_idx].cpu().numpy(), nan=-1.0)
        geos_raw_sample = ((geos_norm_sample + 1.0) / 2.0) * (geos_max - geos_min) + geos_min
        
        local_obs_vars = list(val_dataset_full.local_obs_variables)
        global_context_vars = list(val_dataset_full.global_context_variables)

        def predictor_norm_sample(var_name):
            if var_name in local_obs_vars:
                channel = local_obs_vars.index(var_name) * 4 + lead_idx
                return np.nan_to_num(x_obs[sample_idx, channel].cpu().numpy(), nan=0.0), "local"
            if x_global_context is not None and var_name in global_context_vars:
                channel = global_context_vars.index(var_name) * 4 + lead_idx
                return np.nan_to_num(x_global_context[sample_idx, channel].cpu().numpy(), nan=0.0), "global"
            return np.zeros((grid_h, grid_w), dtype=np.float32), "missing"

        def reverse_minmax(norm_sample, bounds_key, fallback=None):
            if bounds_key in global_bounds:
                vmin, vmax = global_bounds[bounds_key]["min"], global_bounds[bounds_key]["max"]
            elif fallback is not None:
                vmin, vmax = fallback
            else:
                vmin, vmax = -1.0, 1.0
            return ((norm_sample + 1.0) / 2.0) * (vmax - vmin) + vmin

        # 2. Reverse Normalize SST
        sst_norm_sample, sst_scope = predictor_norm_sample("sst")
        sst_raw_sample = reverse_minmax(sst_norm_sample, "sst")
        
        # 3. Reverse Normalize Target SQRT (Lead 0)
        sqrt_norm_sample = np.nan_to_num(target_norm[sample_idx, lead_idx].cpu().numpy(), nan=-1.0)
        sqrt_raw_sample = ((sqrt_norm_sample + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
        res_raw_sample = np.square(sqrt_raw_sample) - geos_raw_sample

        # 4. Reverse Normalize Observational States
        sss_norm_sample, sss_scope = predictor_norm_sample("sss")
        sss_raw_sample = reverse_minmax(sss_norm_sample, "sss")

        sm_norm_sample, sm_scope = predictor_norm_sample("sm")
        sm_raw_sample = reverse_minmax(sm_norm_sample, "sm")

        ivt_norm_sample, ivt_scope = predictor_norm_sample("ivt")
        ivt_raw_sample = reverse_minmax(ivt_norm_sample, "ivt")

        zdev_norm_sample, zdev_scope = predictor_norm_sample("z500_zonal_dev")
        zdev_raw_sample = reverse_minmax(zdev_norm_sample, "z500_zonal_dev")

        u250_norm_sample, u250_scope = predictor_norm_sample("u250")
        u250_raw_sample = reverse_minmax(u250_norm_sample, "u250")

        mjo_norm_sample, mjo_scope = predictor_norm_sample("mjo")
        mjo_raw_sample = reverse_minmax(mjo_norm_sample, "mjo", fallback=(-100.0, 100.0))

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
        # In v7 residual mode, y_target C=1 is normalized ERA5 minus GEOS-mean.
        t2m_norm_sample = np.nan_to_num(target_norm[sample_idx, 1].cpu().numpy(), nan=0.0)
        if t2m_target_mode == "geos_residual":
            t2m_res_sample = ((t2m_norm_sample + 1.0) / 2.0) * (t2m_residual_max - t2m_residual_min) + t2m_residual_min
            geos_t2m_mean_sample = batch["geos_ens_raw"][sample_idx, :, 1, lead_idx].mean(dim=0).cpu().numpy()
            t2m_raw_sample = geos_t2m_mean_sample + t2m_res_sample
            t2m_norm_title = "Normalized T2M Residual [-1, 1]"
            t2m_raw_title = "Raw ERA5 T2M (GEOS Mean + Residual)"
        else:
            if "target_t2m_raw" in global_bounds:
                t2m_tmin, t2m_tmax = global_bounds["target_t2m_raw"]["min"], global_bounds["target_t2m_raw"]["max"]
            else:
                t2m_tmin, t2m_tmax = 200.0, 320.0
            t2m_raw_sample = ((t2m_norm_sample + 1.0) / 2.0) * (t2m_tmax - t2m_tmin) + t2m_tmin
            t2m_norm_title = "Normalized ERA5 T2M [-1, 1]"
            t2m_raw_title = "Raw ERA5 T2M (Target)"

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
        axes[2, 0].set_title(f"Raw SST ({sst_scope})")
        axes[2, 1].imshow(sst_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2, 1].set_title(f"Normalized SST [{sst_scope}]")
        
        axes[3, 0].imshow(sss_raw_sample, cmap='YlGnBu')
        axes[3, 0].set_title(f"Raw SSS ({sss_scope})")
        axes[3, 1].imshow(sss_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[3, 1].set_title(f"Normalized SSS [{sss_scope}]")
        
        axes[4, 0].imshow(sm_raw_sample, cmap='YlOrBr')
        axes[4, 0].set_title(f"Raw SM ({sm_scope})")
        axes[4, 1].imshow(sm_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[4, 1].set_title(f"Normalized SM [{sm_scope}]")
        
        axes[5, 0].imshow(ivt_raw_sample, cmap='cubehelix')
        axes[5, 1].imshow(ivt_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        
        im13 = axes[6, 0].imshow(zdev_raw_sample, cmap='RdBu_r', vmin=-3000, vmax=3000)
        axes[6, 0].set_title(f"Raw Z500 Zonal Dev ({zdev_scope})")
        im14 = axes[6, 1].imshow(zdev_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[6, 1].set_title(f"Normalized Z500 Zonal Dev [{zdev_scope}]")
        
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
        axes[11, 0].set_title(t2m_raw_title)
        fig.colorbar(im_t2m1, ax=axes[11, 0])
        im_t2m2 = axes[11, 1].imshow(t2m_norm_sample, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[11, 1].set_title(t2m_norm_title)
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
        if start_epoch > 1:
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
    stop_training_after_validation = False
    
    for epoch in range(start_epoch, epochs + 1):
        if epochs_done_this_run >= max_epochs_this_run:
            if accelerator.is_main_process:
                print(f"\n⚠️ Reached --epochs-per-run limit ({max_epochs_this_run}). Exiting for resubmission.")
            break
        if (
            not is_variance_phase
            and not crps_loss
            and variance_phase_start_epoch > 0
            and epoch >= variance_phase_start_epoch
        ):
            enable_variance_phase(
                f"epoch {epoch} reached variance_phase_start_epoch={variance_phase_start_epoch}",
                reset_best_for_phase=True,
            )

        model.train()
        train_loss = 0.0
        train_crps_pr_total = 0.0
        train_crps_t2m_total = 0.0
        train_crps_steps = 0
        if crps_loss:
            phase_label = "CRPS"
        else:
            phase_label = "VarOnly" if is_variance_phase else "VelOnly"
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase_label}]", disable=not accelerator.is_main_process)
        for i, batch in enumerate(pbar):    
            # Conditionals
            x_geos = batch['x_geos'].to(device)
            x_obs  = batch['x_obs'].to(device)
            global_context = get_batch_global_context(batch, device)
            
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

            # x_obs is dynamic: local_obs_variables are cropped to the target grid,
            # while global_context_variables are sent separately through the context encoder.
            x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1) 

            if crps_loss:
                target_raw = batch['target_raw'].to(device)
                train_spatial_weights = spatial_weights if crps_loss_use_land_ocean_weights else area_weights
                loss, crps_diag = compute_multi_crps_training_loss(
                    flow_matcher=flow_matcher,
                    model=model,
                    x_cond=x_cond,
                    lead_idx=lead_idx,
                    target_raw=target_raw,
                    num_ensemble=crps_loss_num_ensemble,
                    num_steps=crps_loss_num_steps,
                    target_sqrt_min=target_sqrt_min,
                    target_sqrt_max=target_sqrt_max,
                    spatial_weights=train_spatial_weights,
                    ode_chunk_size=crps_loss_ode_batch_size,
                    max_ensemble_per_chunk=crps_loss_max_ensemble_per_chunk,
                    pr_weight=crps_loss_pr_weight,
                    t2m_weight=crps_loss_t2m_weight,
                    use_checkpoint=crps_loss_use_gradient_checkpointing,
                    global_context=global_context,
                )
                train_crps_pr_total += float(crps_diag["crps_pr"].item())
                train_crps_t2m_total += float(crps_diag["crps_t2m"].item())
                train_crps_steps += 1
            else:
                # Targets are already residual normalized [-1, 1] by dataset_hybrid
                target_norm = batch['y_target'].to(device) # [B, 2, H, W] (PR, T2M)

                if is_variance_phase:
                    # Match compare-noise: EOF-LHS dynamic multimodal noise, rho mix,
                    # and the same coarse variance objective used during inference.
                    E_var = variance_training_num_ensemble
                    noise = generate_compare_eof_lhs_noise(
                        batch=batch,
                        num_ensemble=E_var,
                        device=device,
                        eof_bases=eof_bases,
                        nao_bases=nao_bases,
                        enso_bases=enso_bases,
                        t2m_eof_bases=t2m_eof_bases,
                        t2m_nao_bases=t2m_nao_bases,
                        t2m_enso_bases=t2m_enso_bases,
                        nao_lookup=nao_lookup,
                        oni_lookup=oni_lookup,
                        mjo_df=mjo_df,
                        rho_pr=validation_rho_pr,
                        rho_t2m=validation_rho_t2m,
                    ).to(dtype=target_norm.dtype)
                    target_for_flow = expand_for_ensemble(target_norm, E_var)
                    x_cond_for_flow = expand_for_ensemble(x_cond, E_var)
                    lead_idx_for_flow = lead_idx.unsqueeze(1).expand(B, E_var).reshape(-1).long()
                    global_context_for_flow = expand_for_ensemble(global_context, E_var)
                    flow_batch_size = target_for_flow.shape[0]
                else:
                    noise = torch.randn_like(target_norm)
                    target_for_flow = target_norm
                    x_cond_for_flow = x_cond
                    lead_idx_for_flow = lead_idx
                    global_context_for_flow = global_context
                    flow_batch_size = B

                # Flow Matching Interpolation. The variance head is used at
                # t=0 during inference, so train VarOnly on the same initial
                # EOF-noise state.
                if is_variance_phase:
                    t = torch.zeros((flow_batch_size,), device=device, dtype=torch.float32)
                else:
                    t = flow_matcher.sample_time_batch(flow_batch_size)
                x_t, v_target = flow_matcher.interpolate(target_for_flow, noise, t)

                # Predict the velocity AND variance (routed through the correct per-week output head)
                v_pred, var_pred = model(
                    x_t,
                    x_cond_for_flow,
                    t,
                    lead_idx=lead_idx_for_flow,
                    global_context=global_context_for_flow,
                )

                # --- Temporal Loss Weighting ---
                # Prioritize gradient updates for harder long-term leads (Week 4 > Week 1)
                # 0=Week1, 1=Week2, 2=Week3, 3=Week4
                w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
                temp_weights = w_escalation[lead_idx_for_flow].view(flow_batch_size, 1, 1, 1)

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
                    variance_coarse_kernel=validation_variance_coarse_kernel if is_variance_phase else None,
                )

                if is_variance_phase:
                    loss = loss_var
                else:
                    loss = loss_vel

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            if crps_loss:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "pr": f"{crps_diag['crps_pr'].item():.3f}",
                    "t2m": f"{crps_diag['crps_t2m'].item():.3f}",
                })
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(loader)
        
        if accelerator.is_main_process:
            if crps_loss and train_crps_steps > 0:
                avg_train_crps_pr = train_crps_pr_total / train_crps_steps
                avg_train_crps_t2m = train_crps_t2m_total / train_crps_steps
                print(
                    f"📈 Epoch {epoch} Training Loss (CRPS): {avg_train_loss:.4f} "
                    f"| PR: {avg_train_crps_pr:.4f} | T2M: {avg_train_crps_t2m:.4f}"
                )
            elif is_variance_phase:
                print(f"📈 Epoch {epoch} Training Loss (Variance): {avg_train_loss:.4f}")
            else:
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
                'best_val_epoch': best_val_epoch,
                'mse_bad_val_checks': mse_bad_val_checks,
                'top_models': top_models,
                'is_variance_phase': is_variance_phase,
                'variance_phase_best_reset_done': variance_phase_best_reset_done,
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
            best_val_epoch = 0
            mse_bad_val_checks = 0
            crps_phase_reset = True
        
        def should_plot_validation(ep):
            return plot_validation_every > 0 and ep >= 20 and (ep % plot_validation_every == 0)

        def should_validate(ep):
            if ep < 5:
                return False
            elif should_plot_validation(ep):
                return True
            elif (not use_crps_phase) and dense_mse_validation_until > 0 and ep <= dense_mse_validation_until:
                return True
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
            use_flow_variance = True
            val_result = run_val_inference(
                epoch, model, val_loader_monthly, flow_matcher, device, accelerator, output_dir, log_file, 
                target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds, 
                is_test=False, is_fast_recon=True, use_flow_variance=use_flow_variance,
                use_eof_lhs_noise=True,
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
                validation_variance_coarse_kernel=validation_variance_coarse_kernel,
                t2m_target_mode=t2m_target_mode,
                t2m_residual_min=t2m_residual_min,
                t2m_residual_max=t2m_residual_max,
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
                    best_val_epoch = epoch
                    new_best_name = f"best_model_epoch_{epoch}_crps_{current_val_metric:.4f}.pt"
                    new_best_path = os.path.join(output_dir, new_best_name)
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_ckpt = {
                        'epoch': epoch,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_epoch': best_val_epoch,
                        'mse_bad_val_checks': mse_bad_val_checks,
                        'is_variance_phase': is_variance_phase,
                        'variance_phase_best_reset_done': variance_phase_best_reset_done,
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
                elif should_plot_validation(epoch):
                    vt = val_result['tensors']
                    save_val_plot(epoch, vt['full_pred'], vt['true_target'], val_result['avg_crps_pr'], val_result['avg_rmse_pr'],
                                  vt['geos_mean'], val_result['avg_geos_crps_pr'], val_result['avg_geos_rmse_pr'], output_dir,
                                  ai_residual=vt['ai_res'], suffix="periodic_crps", geos_single=vt['geos_single'],
                                  model_single=vt['model_single'], model_var=vt['model_var'],
                                  full_pred_t2m=vt['full_pred_t2m'], true_target_t2m=vt['true_target_t2m'],
                                  geos_pred_t2m=vt['geos_mean_t2m'], model_var_t2m=vt['model_var_t2m'],
                                  model_crps_t2m=val_result['avg_crps_t2m'], model_rmse_t2m=val_result['avg_rmse_t2m'],
                                  geos_crps_t2m=val_result['avg_geos_crps_t2m'], geos_rmse_t2m=val_result['avg_geos_rmse_t2m'])
                    print(f"📸 Periodic validation plot saved for Epoch {epoch}.")
                if epoch % 5 == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save({
                        'epoch': epoch,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_epoch': best_val_epoch,
                        'mse_bad_val_checks': mse_bad_val_checks,
                        'is_variance_phase': is_variance_phase,
                        'variance_phase_best_reset_done': variance_phase_best_reset_done,
                    },
                               os.path.join(output_dir, f"periodic_ckpt_epoch_{epoch}.pt"))
                    print(f"💾 Periodic checkpoint: periodic_ckpt_epoch_{epoch}.pt")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_epoch': best_val_epoch,
                    'mse_bad_val_checks': mse_bad_val_checks,
                    'is_variance_phase': is_variance_phase,
                    'variance_phase_best_reset_done': variance_phase_best_reset_done,
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
                for b_idx, batch in enumerate(val_loader_full):
                    x_geos = batch['x_geos'].to(device)
                    x_obs  = batch['x_obs'].to(device)
                    global_context = get_batch_global_context(batch, device)
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
                    if is_variance_phase:
                        E_var = variance_training_num_ensemble
                        noise = generate_compare_eof_lhs_noise(
                            batch=batch,
                            num_ensemble=E_var,
                            device=device,
                            eof_bases=eof_bases,
                            nao_bases=nao_bases,
                            enso_bases=enso_bases,
                            t2m_eof_bases=t2m_eof_bases,
                            t2m_nao_bases=t2m_nao_bases,
                            t2m_enso_bases=t2m_enso_bases,
                            nao_lookup=nao_lookup,
                            oni_lookup=oni_lookup,
                            mjo_df=mjo_df,
                            rho_pr=validation_rho_pr,
                            rho_t2m=validation_rho_t2m,
                        ).to(dtype=target_norm.dtype)
                        target_for_flow = expand_for_ensemble(target_norm, E_var)
                        x_cond_for_flow = expand_for_ensemble(x_cond, E_var)
                        lead_idx_for_flow = lead_idx.unsqueeze(1).expand(B, E_var).reshape(-1).long()
                        global_context_for_flow = expand_for_ensemble(global_context, E_var)
                        t = torch.zeros((target_for_flow.shape[0],), device=device, dtype=torch.float32)
                    else:
                        t, noise = sample_deterministic_validation_state(
                            target_norm, b_idx, mse_validation_seed, device
                        )
                        target_for_flow = target_norm
                        x_cond_for_flow = x_cond
                        lead_idx_for_flow = lead_idx
                        global_context_for_flow = global_context

                    x_t, v_target = flow_matcher.interpolate(target_for_flow, noise, t)
                    v_pred, var_pred = model(
                        x_t,
                        x_cond_for_flow,
                        t,
                        lead_idx=lead_idx_for_flow,
                        global_context=global_context_for_flow,
                    )
                    w_escalation = torch.tensor([1.0, 1.1, 1.2, 1.3], device=device)
                    temp_weights = w_escalation[lead_idx_for_flow].view(-1, 1, 1, 1).expand(-1, 2, -1, -1)
                    
                    loss_vel = (spatial_weights * temp_weights * (v_pred - v_target)**2).mean()
                    loss_var = compute_multi_variance_loss(
                        var_pred=var_pred,
                        v_pred=v_pred,
                        v_target=v_target,
                        spatial_weights=spatial_weights,
                        temp_weights=temp_weights,
                        variance_coarse_kernel=validation_variance_coarse_kernel if is_variance_phase else None,
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
                print(f"📉 MSE Gap (val - train): {current_val_metric - avg_train_loss:+.4f}")

                is_new_best = (current_val_metric < (best_val_loss - mse_early_stop_min_delta))
                
                if is_new_best:
                    print(f"🏆 NEW BEST (MSE)! {current_val_metric:.4f} (Prev: {best_val_loss:.4f})")
                    best_val_loss = current_val_metric
                    best_val_epoch = epoch
                    mse_bad_val_checks = 0

                    new_best_name = f"best_model_epoch_{epoch}_loss_{current_val_metric:.4f}.pt"
                    new_best_path = os.path.join(output_dir, new_best_name)
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_ckpt = {
                        'epoch': epoch,
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'best_val_epoch': best_val_epoch,
                        'mse_bad_val_checks': mse_bad_val_checks,
                        'is_variance_phase': is_variance_phase,
                        'variance_phase_best_reset_done': variance_phase_best_reset_done,
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
                else:
                    mse_bad_val_checks += 1
                    print(
                        f"📊 No MSE improvement for {mse_bad_val_checks} validation check(s). "
                        f"Best={best_val_loss:.4f} at epoch {best_val_epoch}."
                    )
                    if mse_plateau_patience > 0 and mse_bad_val_checks % mse_plateau_patience == 0:
                        old_lr = float(optimizer.param_groups[0]["lr"])
                        new_lr = max(old_lr * mse_plateau_factor, mse_min_lr)
                        if new_lr < old_lr:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = new_lr
                            print(
                                f"🔻 MSE plateau detected: reducing LR from {old_lr:.2e} to {new_lr:.2e}."
                            )
                        else:
                            print(f"ℹ️ MSE plateau detected, but LR is already at min_lr={mse_min_lr:.2e}.")
                    if (
                        mse_early_stop_patience > 0
                        and epoch >= mse_early_stop_start_epoch
                        and mse_bad_val_checks >= mse_early_stop_patience
                    ):
                        stop_training_after_validation = True
                        reason = (
                            f"Stopped at epoch {epoch}: no MSE improvement greater than "
                            f"{mse_early_stop_min_delta:.4g} for {mse_bad_val_checks} validation checks. "
                            f"Best epoch={best_val_epoch}, best_val_loss={best_val_loss:.6f}."
                        )
                        with open(early_stop_marker, "w") as f:
                            f.write(reason + "\n")
                        print(f"🛑 {reason}")

                if epoch >= 20 and (is_new_best or should_plot_validation(epoch)):
                    plot_suffix = "best" if is_new_best else "periodic"
                    plot_label = "best" if is_new_best else f"periodic every {plot_validation_every} epochs"
                    print(f"📊 Generating PR+T2M validation plot for Epoch {epoch} ({plot_label})...")
                    val_result = run_val_inference(
                        epoch, model, val_loader_monthly, flow_matcher, device, accelerator, output_dir, log_file,
                        target_sqrt_min, target_sqrt_max, geos_min, geos_max, area_weights, global_bounds,
                        is_test=False, is_fast_recon=True,
                        use_flow_variance=True,
                        use_eof_lhs_noise=True,
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
                        validation_variance_coarse_kernel=validation_variance_coarse_kernel,
                        t2m_target_mode=t2m_target_mode,
                        t2m_residual_min=t2m_residual_min,
                        t2m_residual_max=t2m_residual_max,
                    )
                    vt = val_result['tensors']
                    save_val_plot(
                        epoch, vt['full_pred'], vt['true_target'],
                        val_result['avg_crps_pr'], val_result['avg_rmse_pr'],
                        vt['geos_mean'], val_result['avg_geos_crps_pr'], val_result['avg_geos_rmse_pr'],
                        output_dir, ai_residual=vt['ai_res'], suffix=plot_suffix,
                        geos_single=vt['geos_single'], model_single=vt['model_single'], model_var=vt['model_var'],
                        full_pred_t2m=vt['full_pred_t2m'], true_target_t2m=vt['true_target_t2m'],
                        geos_pred_t2m=vt['geos_mean_t2m'], model_var_t2m=vt['model_var_t2m'],
                        model_crps_t2m=val_result['avg_crps_t2m'], model_rmse_t2m=val_result['avg_rmse_t2m'],
                        geos_crps_t2m=val_result['avg_geos_crps_t2m'], geos_rmse_t2m=val_result['avg_geos_rmse_t2m']
                    )
                    print(f"📸 Validation plot saved for Epoch {epoch}.")

                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_epoch': best_val_epoch,
                    'mse_bad_val_checks': mse_bad_val_checks,
                    'is_variance_phase': is_variance_phase,
                    'variance_phase_best_reset_done': variance_phase_best_reset_done,
                },
                           os.path.join(output_dir, "latest_flow_ckpt.pt"))
                with open(log_file, "a") as f:
                    csv.writer(f).writerow([epoch, avg_train_loss, 0.0, current_val_metric])

        # Track progress for this execution session
        epochs_done_this_run += 1
        if stop_training_after_validation:
            if accelerator.is_main_process:
                print("🏁 Stopping training loop after validation control trigger.")
            break

def main():
    parser = argparse.ArgumentParser(description="Train or Test Flow Matching Model")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv7.yaml")
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
