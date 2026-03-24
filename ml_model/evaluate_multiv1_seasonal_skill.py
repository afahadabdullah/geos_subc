#!/usr/bin/env python3
"""
Evaluate held-out 2020-2021 seasonal multiv1 skill for ML and GEOS.

Metrics are computed by season and weekly lead for both T2M and precipitation:
- anomaly RMSE (against obs weekly climatology)
- anomaly correlation
- upper-tercile Brier Skill Score
- tercile RPSS

Outputs for each season / variable include:
- domain-mean summary CSV and text report
- seasonal mean-state map triplets (Obs / GEOS / ML)
- lead-wise metric map triplets using Cartopy
- domain-mean lead-summary figures
"""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False


PR_MAX_VALID_MM_DAY = 100.0
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

VAR_SPECS = {
    "tas": {
        "label": "T2M",
        "file_prefix": "t2m",
        "unit": "K",
        "obs_template": "t2m_weekly_{year}.zarr",
        "var_candidates": ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"],
        "clim_var_candidates": ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"],
        "cmap": "RdYlBu_r",
        "diff_cmap": "PuOr",
        "mean_vrange": None,
    },
    "pr": {
        "label": "PR",
        "file_prefix": "pr",
        "unit": "mm/day",
        "obs_template": "gpcp_weekly_{year}.zarr",
        "var_candidates": ["pr", "precip", "PRECTOT", "flux_precip", "target", "total_precipitation"],
        "clim_var_candidates": ["pr", "precip", "PRECTOT", "flux_precip", "target", "total_precipitation"],
        "cmap": "Blues",
        "diff_cmap": "PiYG",
        "mean_vrange": (0.0, 20.0),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Seasonal held-out skill evaluation for multiv1 ML and GEOS.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing geos_subc_<year>.zarr, t2m_weekly_<year>.zarr, and gpcp_weekly_<year>.zarr.",
    )
    parser.add_argument(
        "--ml_dir",
        type=str,
        default="dataprocess/gen_multiv1",
        help="Directory containing generated ML yearly zarr stores like 2020.zarr and 2021.zarr.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_output_flowmulti/seasonal_skill_2020_2021",
        help="Directory to save seasonal reports, CSVs, and plots.",
    )
    parser.add_argument("--start_year", type=int, default=None, help="First held-out year. Defaults to config val_start_year.")
    parser.add_argument("--end_year", type=int, default=None, help="Last held-out year. Defaults to config val_end_year.")
    parser.add_argument(
        "--threshold_start_year",
        type=int,
        default=1999,
        help="First year used to build observed tercile thresholds and climatological probabilities.",
    )
    parser.add_argument(
        "--threshold_end_year",
        type=int,
        default=2019,
        help="Last year used to build observed tercile thresholds and climatological probabilities.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=["tas", "pr"],
        default=["tas", "pr"],
        help="Variables to evaluate. Defaults to both T2M (`tas`) and precipitation (`pr`).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=["DJF", "MAM", "JJA", "SON"],
        default=["DJF", "MAM", "JJA", "SON"],
        help="Initialization seasons to evaluate.",
    )
    parser.add_argument(
        "--sample_chunk_size",
        type=int,
        default=2,
        help="Number of init dates to load per evaluation chunk.",
    )
    parser.add_argument(
        "--obs_clim_path",
        type=str,
        default="dataprocess/clim/obs_weekly_clim_1999_2021.zarr",
        help="Obs weekly climatology path used for anomaly RMSE / correlation.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def choose_name(items: Iterable[str], candidates: Sequence[str], label: str) -> str:
    item_set = set(items)
    for name in candidates:
        if name in item_set:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {sorted(item_set)}")


def choose_data_var(ds: xr.Dataset, candidates: Sequence[str], label: str) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {list(ds.data_vars)}")


def open_zarr_required(path: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_zarr(path, consolidated=False, chunks=None)


def sanitize_pr_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)
    values = np.where(values < 0.0, 0.0, values)
    values = np.where(values > PR_MAX_VALID_MM_DAY, np.nan, values)
    return values


def infer_layout(ds: xr.Dataset, kind: str) -> Dict[str, str]:
    s_dim = choose_name(ds.dims, ["S", "time", "init_time"], f"{kind} init dimension")
    lead_dim = choose_name(ds.dims, ["L", "lead", "lead_time"], f"{kind} lead dimension")
    y_dim = choose_name(set(ds.dims) | set(ds.coords), ["Y", "latitude", "lat", "y"], f"{kind} latitude dimension")
    x_dim = choose_name(set(ds.dims) | set(ds.coords), ["X", "longitude", "lon", "x"], f"{kind} longitude dimension")
    member_dim = None
    for candidate in ["M", "member", "ensemble", "ensemble_member"]:
        if candidate in ds.dims:
            member_dim = candidate
            break
    return {
        "s_dim": s_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "member_dim": member_dim,
    }


def infer_clim_layout(ds: xr.Dataset, kind: str, var_key: str) -> Dict[str, str]:
    week_dim = choose_name(ds.dims, ["init_week", "week", "W"], f"{kind} init-week dimension")
    lead_dim = choose_name(ds.dims, ["L", "lead", "lead_time"], f"{kind} lead dimension")
    y_dim = choose_name(set(ds.dims) | set(ds.coords), ["Y", "latitude", "lat", "y"], f"{kind} latitude dimension")
    x_dim = choose_name(set(ds.dims) | set(ds.coords), ["X", "longitude", "lon", "x"], f"{kind} longitude dimension")
    var_name = choose_data_var(ds, VAR_SPECS[var_key]["clim_var_candidates"], f"{kind} variable")
    return {
        "week_dim": week_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "var_name": var_name,
    }


def load_obs_weekly_climatology(path: str, var_key: str) -> Dict[str, np.ndarray]:
    ds = open_zarr_required(path)
    try:
        layout = infer_clim_layout(ds, f"OBS CLIM {var_key}", var_key)
        da = ds[layout["var_name"]].transpose(layout["week_dim"], layout["lead_dim"], layout["y_dim"], layout["x_dim"])
        values = np.asarray(da.values, dtype=np.float64)
        if var_key == "pr":
            values = sanitize_pr_values(values)
        return {"values": values}
    finally:
        ds.close()


def prepare_common_dates(
    ml_ds: xr.Dataset,
    ml_layout: Dict[str, str],
    geos_ds: xr.Dataset,
    geos_layout: Dict[str, str],
    obs_ds: xr.Dataset,
    obs_layout: Dict[str, str],
) -> List[pd.Timestamp]:
    ml_dates = pd.to_datetime(ml_ds[ml_layout["s_dim"]].values).normalize()
    geos_dates = pd.to_datetime(geos_ds[geos_layout["s_dim"]].values).normalize()
    obs_dates = pd.to_datetime(obs_ds[obs_layout["s_dim"]].values).normalize()
    common = sorted(set(ml_dates) & set(geos_dates) & set(obs_dates))
    return [pd.Timestamp(item) for item in common]


def exact_indices_for_dates(s_values: np.ndarray, dates: Sequence[pd.Timestamp]) -> List[int]:
    s_dates = pd.to_datetime(s_values).normalize()
    index_map = {pd.Timestamp(date): idx for idx, date in enumerate(s_dates)}
    missing = [date for date in dates if pd.Timestamp(date) not in index_map]
    if missing:
        missing_str = ", ".join(pd.Timestamp(date).strftime("%Y-%m-%d") for date in missing[:5])
        raise ValueError(f"Missing requested dates in dataset: {missing_str}")
    return [int(index_map[pd.Timestamp(date)]) for date in dates]


def extract_var_chunk(
    ds: xr.Dataset,
    layout: Dict[str, str],
    var_name: str,
    s_indices: Sequence[int],
    lead_idx: int,
    var_key: str,
) -> np.ndarray:
    da = ds[var_name].isel({layout["s_dim"]: list(s_indices), layout["lead_dim"]: int(lead_idx)})
    if layout["member_dim"] is None:
        da = da.transpose(layout["s_dim"], layout["y_dim"], layout["x_dim"])
    else:
        da = da.transpose(layout["s_dim"], layout["member_dim"], layout["y_dim"], layout["x_dim"])
    values = np.asarray(da.values, dtype=np.float64)
    if var_key == "pr":
        values = sanitize_pr_values(values)
    return values


def weighted_mean_2d(metric_map: np.ndarray, aw_2d: np.ndarray) -> float:
    mask = np.isfinite(metric_map)
    if not np.any(mask):
        return float("nan")
    return float(np.nansum(metric_map[mask] * aw_2d[mask]) / (np.nansum(aw_2d[mask]) + 1e-8))


def get_plot_coords(data_dir: str, year_hint: int = 2021) -> Tuple[np.ndarray, np.ndarray]:
    candidate_years = [year_hint, 2021, 2020, 2019]
    seen = set()
    for year in candidate_years:
        if year in seen:
            continue
        seen.add(year)
        path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
        if not os.path.exists(path):
            continue
        try:
            ds = open_zarr_required(path)
            lats = np.asarray(ds["Y"].values if "Y" in ds.coords else ds["lat"].values)
            lons = np.asarray(ds["X"].values if "X" in ds.coords else ds["lon"].values)
            ds.close()
            return lats, lons
        except Exception:
            continue
    return np.linspace(-90.0, 90.0, 181), np.arange(360.0)


def style_cartopy_ax(ax, title: str, extent: Sequence[float], show_left_labels: bool = True):
    ax.set_title(title, fontsize=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_labels
    gl.bottom_labels = True


def save_mean_triplet(
    obs_map: np.ndarray,
    geos_map: np.ndarray,
    ml_map: np.ndarray,
    title_prefix: str,
    filename: str,
    output_dir: str,
    lats: np.ndarray,
    lons: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    obs_avg: float,
    geos_avg: float,
    ml_avg: float,
):
    if not HAS_CARTOPY:
        return
    proj = ccrs.PlateCarree()
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": proj})
    panels = [
        (obs_map, f"{title_prefix} Obs Mean ({obs_avg:.3f})"),
        (geos_map, f"{title_prefix} GEOS Mean ({geos_avg:.3f})"),
        (ml_map, f"{title_prefix} ML Mean ({ml_avg:.3f})"),
    ]
    for idx, (img, title) in enumerate(panels):
        ax = axes[idx]
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
        style_cartopy_ax(ax, title, extent, show_left_labels=(idx == 0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_lower_better_triplet(
    geos_map: np.ndarray,
    ml_map: np.ndarray,
    title_prefix: str,
    metric_name: str,
    filename: str,
    output_dir: str,
    lats: np.ndarray,
    lons: np.ndarray,
    vmin: float,
    vmax: float,
    diff_vmax: float,
    geos_avg: float,
    ml_avg: float,
):
    if not HAS_CARTOPY:
        return
    proj = ccrs.PlateCarree()
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": proj})
    diff_map = geos_map - ml_map
    panels = [
        (geos_map, f"{title_prefix} GEOS {metric_name} ({geos_avg:.3f})", "OrRd", vmin, vmax),
        (ml_map, f"{title_prefix} ML {metric_name} ({ml_avg:.3f})", "OrRd", vmin, vmax),
        (
            diff_map,
            f"{title_prefix} {metric_name} Diff: GEOS-ML\nGreen (+) = ML Better",
            "PiYG",
            -diff_vmax,
            diff_vmax,
        ),
    ]
    for idx, (img, title, cmap, pmin, pmax) in enumerate(panels):
        ax = axes[idx]
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
        style_cartopy_ax(ax, title, extent, show_left_labels=(idx == 0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_higher_better_triplet(
    geos_map: np.ndarray,
    ml_map: np.ndarray,
    title_prefix: str,
    metric_name: str,
    filename: str,
    output_dir: str,
    lats: np.ndarray,
    lons: np.ndarray,
    vmin: float,
    vmax: float,
    diff_vmax: float,
    geos_avg: float,
    ml_avg: float,
):
    if not HAS_CARTOPY:
        return
    proj = ccrs.PlateCarree()
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": proj})
    diff_map = ml_map - geos_map
    panels = [
        (geos_map, f"{title_prefix} GEOS {metric_name} ({geos_avg:.3f})", "RdYlGn", vmin, vmax),
        (ml_map, f"{title_prefix} ML {metric_name} ({ml_avg:.3f})", "RdYlGn", vmin, vmax),
        (
            diff_map,
            f"{title_prefix} {metric_name} Diff: ML-GEOS\nOrange (+) = ML Better",
            "PuOr",
            -diff_vmax,
            diff_vmax,
        ),
    ]
    for idx, (img, title, cmap, pmin, pmax) in enumerate(panels):
        ax = axes[idx]
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
        style_cartopy_ax(ax, title, extent, show_left_labels=(idx == 0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=160)
    plt.close(fig)


def make_domain_metric_figure(rows: List[Dict[str, object]], output_path: str, title: str):
    metrics = [
        ("domain_rmse", "Anomaly RMSE", None),
        ("domain_corr", "Anomaly Corr", 0.0),
        ("domain_bss_upper", "Upper-Tercile BSS", 0.0),
        ("domain_rpss", "Tercile RPSS", 0.0),
    ]
    model_order = ["GEOS", "ML"]
    color_map = {"GEOS": "#d94801", "ML": "#08306b"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, label, ref) in zip(axes, metrics):
        for model_name in model_order:
            model_rows = [row for row in rows if row["model"] == model_name]
            model_rows = sorted(model_rows, key=lambda item: int(item["lead_week"]))
            if not model_rows:
                continue
            lead_vals = [int(row["lead_week"]) for row in model_rows]
            metric_vals = [float(row[key]) for row in model_rows]
            ax.plot(lead_vals, metric_vals, marker="o", linewidth=2.0, color=color_map[model_name], label=model_name)
        if ref is not None:
            ax.axhline(ref, color="#7f7f7f", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Lead Week")
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=14, y=1.04)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_domain_mean_figure(rows: List[Dict[str, object]], output_path: str, title: str, unit: str):
    rows = sorted(rows, key=lambda item: int(item["lead_week"]))
    lead_vals = [int(row["lead_week"]) for row in rows if row["model"] == "ML"]
    obs_vals = [float(row["domain_obs_mean"]) for row in rows if row["model"] == "ML"]
    geos_vals = [float(row["domain_forecast_mean"]) for row in rows if row["model"] == "GEOS"]
    ml_vals = [float(row["domain_forecast_mean"]) for row in rows if row["model"] == "ML"]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(lead_vals, obs_vals, marker="o", linewidth=2.0, color="#1b7837", label="Obs")
    ax.plot(lead_vals, geos_vals, marker="o", linewidth=2.0, color="#d94801", label="GEOS")
    ax.plot(lead_vals, ml_vals, marker="o", linewidth=2.0, color="#08306b", label="ML")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Lead Week")
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str, fieldnames: Sequence[str], rows: List[Dict[str, object]]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_observed_thresholds(
    data_dir: str,
    var_key: str,
    months: Sequence[int],
    start_year: int,
    end_year: int,
) -> Dict[str, np.ndarray]:
    spec = VAR_SPECS[var_key]
    chunks = []
    print(f"  [Thresholds][{spec['label']}] Building observed terciles from {start_year}-{end_year} for months {list(months)}")
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_dir, spec["obs_template"].format(year=year))
        if not os.path.exists(path):
            continue
        ds = open_zarr_required(path)
        try:
            layout = infer_layout(ds, f"OBS {spec['label']} threshold")
            var_name = choose_data_var(ds, spec["var_candidates"], f"OBS {spec['label']} threshold variable")
            s_values = pd.to_datetime(ds[layout["s_dim"]].values).normalize()
            indices = [idx for idx, date in enumerate(s_values) if int(pd.Timestamp(date).month) in months]
            if not indices:
                continue
            da = ds[var_name].isel({layout["s_dim"]: indices}).transpose(layout["s_dim"], layout["lead_dim"], layout["y_dim"], layout["x_dim"])
            values = np.asarray(da.values, dtype=np.float32)
            if var_key == "pr":
                values = sanitize_pr_values(values).astype(np.float32)
            chunks.append(values)
        finally:
            ds.close()
    if not chunks:
        raise ValueError(f"No observed samples found for {spec['label']} thresholds in months {months}")
    hist = np.concatenate(chunks, axis=0).astype(np.float64)
    with np.errstate(invalid="ignore"):
        low = np.nanpercentile(hist, 33.333333, axis=0)
        high = np.nanpercentile(hist, 66.666667, axis=0)
    finite = np.isfinite(hist) & np.isfinite(low[None, :, :, :]) & np.isfinite(high[None, :, :, :])
    cat0 = finite & (hist < low[None, :, :, :])
    cat2 = finite & (hist > high[None, :, :, :])
    cat1 = finite & ~(cat0 | cat2)
    count = finite.sum(axis=0)
    ref_probs = np.full(low.shape + (3,), np.nan, dtype=np.float64)
    np.divide(cat0.sum(axis=0), count, out=ref_probs[..., 0], where=count > 0)
    np.divide(cat1.sum(axis=0), count, out=ref_probs[..., 1], where=count > 0)
    np.divide(cat2.sum(axis=0), count, out=ref_probs[..., 2], where=count > 0)
    return {
        "low": low.astype(np.float32),
        "high": high.astype(np.float32),
        "ref_probs": ref_probs.astype(np.float32),
        "sample_count": count.astype(np.int32),
    }


def init_model_acc(lead_count: int, y_count: int, x_count: int) -> Dict[str, np.ndarray]:
    shape = (lead_count, y_count, x_count)
    return {
        "mean_sum": np.zeros(shape, dtype=np.float64),
        "mean_count": np.zeros(shape, dtype=np.float64),
        "rmse_sq_sum": np.zeros(shape, dtype=np.float64),
        "det_count": np.zeros(shape, dtype=np.float64),
        "corr_sum_x": np.zeros(shape, dtype=np.float64),
        "corr_sum_y": np.zeros(shape, dtype=np.float64),
        "corr_sum_x2": np.zeros(shape, dtype=np.float64),
        "corr_sum_y2": np.zeros(shape, dtype=np.float64),
        "corr_sum_xy": np.zeros(shape, dtype=np.float64),
        "bs_sum": np.zeros(shape, dtype=np.float64),
        "bs_ref_sum": np.zeros(shape, dtype=np.float64),
        "bs_count": np.zeros(shape, dtype=np.float64),
        "rps_sum": np.zeros(shape, dtype=np.float64),
        "rps_ref_sum": np.zeros(shape, dtype=np.float64),
        "rps_count": np.zeros(shape, dtype=np.float64),
    }


def init_obs_acc(lead_count: int, y_count: int, x_count: int) -> Dict[str, np.ndarray]:
    shape = (lead_count, y_count, x_count)
    return {
        "mean_sum": np.zeros(shape, dtype=np.float64),
        "mean_count": np.zeros(shape, dtype=np.float64),
    }


def update_mean_acc(acc: Dict[str, np.ndarray], values: np.ndarray, lead_idx: int):
    valid = np.isfinite(values)
    acc["mean_sum"][lead_idx] += np.where(valid, values, 0.0).sum(axis=0)
    acc["mean_count"][lead_idx] += valid.sum(axis=0)


def update_det_acc(acc: Dict[str, np.ndarray], pred_anom: np.ndarray, obs_anom: np.ndarray, lead_idx: int):
    valid = np.isfinite(pred_anom) & np.isfinite(obs_anom)
    if not np.any(valid):
        return
    diff = pred_anom - obs_anom
    acc["rmse_sq_sum"][lead_idx] += np.where(valid, diff ** 2, 0.0).sum(axis=0)
    acc["det_count"][lead_idx] += valid.sum(axis=0)
    acc["corr_sum_x"][lead_idx] += np.where(valid, pred_anom, 0.0).sum(axis=0)
    acc["corr_sum_y"][lead_idx] += np.where(valid, obs_anom, 0.0).sum(axis=0)
    acc["corr_sum_x2"][lead_idx] += np.where(valid, pred_anom ** 2, 0.0).sum(axis=0)
    acc["corr_sum_y2"][lead_idx] += np.where(valid, obs_anom ** 2, 0.0).sum(axis=0)
    acc["corr_sum_xy"][lead_idx] += np.where(valid, pred_anom * obs_anom, 0.0).sum(axis=0)


def update_prob_acc(
    acc: Dict[str, np.ndarray],
    forecast_raw: np.ndarray,
    obs_raw: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    ref_probs: np.ndarray,
    lead_idx: int,
):
    finite_members = np.isfinite(forecast_raw)
    member_count = finite_members.sum(axis=1)
    threshold_valid = np.isfinite(low) & np.isfinite(high) & np.all(np.isfinite(ref_probs), axis=-1)
    base_valid = (member_count > 0) & np.isfinite(obs_raw) & threshold_valid[None, :, :]
    if not np.any(base_valid):
        return

    safe_member_count = np.where(member_count > 0, member_count, 1)
    p_low = np.sum(forecast_raw < low[None, None, :, :], axis=1) / safe_member_count
    p_high = np.sum(forecast_raw > high[None, None, :, :], axis=1) / safe_member_count
    p_mid = np.clip(1.0 - p_low - p_high, 0.0, 1.0)

    obs_low = obs_raw < low[None, :, :]
    obs_high = obs_raw > high[None, :, :]
    obs_mid = base_valid & ~(obs_low | obs_high)

    upper_ref = ref_probs[:, :, 2][None, :, :]
    bs = (p_high - obs_high.astype(np.float64)) ** 2
    bs_ref = (upper_ref - obs_high.astype(np.float64)) ** 2
    acc["bs_sum"][lead_idx] += np.where(base_valid, bs, 0.0).sum(axis=0)
    acc["bs_ref_sum"][lead_idx] += np.where(base_valid, bs_ref, 0.0).sum(axis=0)
    acc["bs_count"][lead_idx] += base_valid.sum(axis=0)

    cdf_f1 = p_low
    cdf_f2 = p_low + p_mid
    cdf_o1 = obs_low.astype(np.float64)
    cdf_o2 = (obs_low | obs_mid).astype(np.float64)
    cdf_r1 = ref_probs[:, :, 0][None, :, :]
    cdf_r2 = (ref_probs[:, :, 0] + ref_probs[:, :, 1])[None, :, :]
    rps = 0.5 * ((cdf_f1 - cdf_o1) ** 2 + (cdf_f2 - cdf_o2) ** 2)
    rps_ref = 0.5 * ((cdf_r1 - cdf_o1) ** 2 + (cdf_r2 - cdf_o2) ** 2)
    acc["rps_sum"][lead_idx] += np.where(base_valid, rps, 0.0).sum(axis=0)
    acc["rps_ref_sum"][lead_idx] += np.where(base_valid, rps_ref, 0.0).sum(axis=0)
    acc["rps_count"][lead_idx] += base_valid.sum(axis=0)


def finalize_mean_map(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    out = np.full_like(sum_arr, np.nan, dtype=np.float64)
    np.divide(sum_arr, count_arr, out=out, where=count_arr > 0.0)
    return out


def finalize_rmse_map(acc: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.full_like(acc["rmse_sq_sum"], np.nan, dtype=np.float64)
    np.divide(acc["rmse_sq_sum"], acc["det_count"], out=out, where=acc["det_count"] > 0.0)
    return np.sqrt(out)


def finalize_corr_map(acc: Dict[str, np.ndarray]) -> np.ndarray:
    count = acc["det_count"]
    mean_x = np.full_like(acc["corr_sum_x"], np.nan, dtype=np.float64)
    mean_y = np.full_like(acc["corr_sum_y"], np.nan, dtype=np.float64)
    np.divide(acc["corr_sum_x"], count, out=mean_x, where=count > 0.0)
    np.divide(acc["corr_sum_y"], count, out=mean_y, where=count > 0.0)
    var_x = np.full_like(mean_x, np.nan, dtype=np.float64)
    var_y = np.full_like(mean_y, np.nan, dtype=np.float64)
    cov_xy = np.full_like(mean_x, np.nan, dtype=np.float64)
    np.divide(acc["corr_sum_x2"], count, out=var_x, where=count > 0.0)
    np.divide(acc["corr_sum_y2"], count, out=var_y, where=count > 0.0)
    np.divide(acc["corr_sum_xy"], count, out=cov_xy, where=count > 0.0)
    var_x = var_x - mean_x ** 2
    var_y = var_y - mean_y ** 2
    cov_xy = cov_xy - mean_x * mean_y
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    corr = np.full_like(mean_x, np.nan, dtype=np.float64)
    np.divide(cov_xy, denom, out=corr, where=denom > 1e-12)
    return corr


def finalize_skill_map(sum_arr: np.ndarray, ref_sum_arr: np.ndarray) -> np.ndarray:
    out = np.full_like(sum_arr, np.nan, dtype=np.float64)
    valid = ref_sum_arr > 1e-12
    out[valid] = 1.0 - (sum_arr[valid] / ref_sum_arr[valid])
    return out


def evaluate_season_variable(
    args,
    config: Dict,
    season_name: str,
    months: Sequence[int],
    var_key: str,
    obs_clim: Dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> List[Dict[str, object]]:
    spec = VAR_SPECS[var_key]
    label = spec["label"]
    unit = spec["unit"]
    season_dir = os.path.join(args.output_dir, season_name, spec["file_prefix"])
    os.makedirs(season_dir, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"[{season_name}][{label}] Starting seasonal evaluation")
    print(f"  Months       : {list(months)}")
    print(f"  Threshold yrs: {args.threshold_start_year}-{args.threshold_end_year}")
    thresholds = build_observed_thresholds(args.data_dir, var_key, months, args.threshold_start_year, args.threshold_end_year)

    obs_acc = None
    model_accs = None
    year_common_counts: Dict[int, int] = {}
    total_common_inits = 0

    for year in range(args.start_year, args.end_year + 1):
        ml_path = os.path.join(args.ml_dir, f"{year}.zarr")
        geos_path = os.path.join(args.data_dir, f"geos_subc_{year}.zarr")
        obs_path = os.path.join(args.data_dir, spec["obs_template"].format(year=year))
        print(f"\n[{season_name}][{label}][{year}] Opening datasets")
        print(f"  ML  : {ml_path}")
        print(f"  GEOS: {geos_path}")
        print(f"  OBS : {obs_path}")

        ml_ds = open_zarr_required(ml_path)
        geos_ds = open_zarr_required(geos_path)
        obs_ds = open_zarr_required(obs_path)
        try:
            ml_layout = infer_layout(ml_ds, f"ML {label}")
            geos_layout = infer_layout(geos_ds, f"GEOS {label}")
            obs_layout = infer_layout(obs_ds, f"OBS {label}")
            ml_var = choose_data_var(ml_ds, spec["var_candidates"], f"ML {label} variable")
            geos_var = choose_data_var(geos_ds, spec["var_candidates"], f"GEOS {label} variable")
            obs_var = choose_data_var(obs_ds, spec["var_candidates"], f"OBS {label} variable")

            common_dates = prepare_common_dates(ml_ds, ml_layout, geos_ds, geos_layout, obs_ds, obs_layout)
            common_dates = [date for date in common_dates if int(date.month) in months]
            if not common_dates:
                print(f"[{season_name}][{label}][{year}] No common init dates for this season")
                continue

            year_common_counts[year] = len(common_dates)
            total_common_inits += len(common_dates)
            print(f"[{season_name}][{label}][{year}] Found {len(common_dates)} common init dates")

            ml_idx = exact_indices_for_dates(ml_ds[ml_layout["s_dim"]].values, common_dates)
            geos_idx = exact_indices_for_dates(geos_ds[geos_layout["s_dim"]].values, common_dates)
            obs_idx = exact_indices_for_dates(obs_ds[obs_layout["s_dim"]].values, common_dates)

            lead_count = min(
                int(ml_ds.sizes[ml_layout["lead_dim"]]),
                int(geos_ds.sizes[geos_layout["lead_dim"]]),
                int(obs_ds.sizes[obs_layout["lead_dim"]]),
                4,
            )
            y_count = int(obs_ds.sizes[obs_layout["y_dim"]])
            x_count = int(obs_ds.sizes[obs_layout["x_dim"]])
            if obs_acc is None:
                obs_acc = init_obs_acc(lead_count, y_count, x_count)
                model_accs = {"GEOS": init_model_acc(lead_count, y_count, x_count), "ML": init_model_acc(lead_count, y_count, x_count)}

            total_chunks = (len(common_dates) + args.sample_chunk_size - 1) // args.sample_chunk_size
            print(f"[{season_name}][{label}][{year}] Evaluating {lead_count} leads with chunk_size={args.sample_chunk_size} ({total_chunks} chunks)")

            for chunk_start in range(0, len(common_dates), args.sample_chunk_size):
                chunk_end = min(len(common_dates), chunk_start + args.sample_chunk_size)
                chunk_number = chunk_start // args.sample_chunk_size + 1
                if total_chunks <= 10 or chunk_number == 1 or chunk_number == total_chunks or (chunk_number % 5 == 0):
                    lo = common_dates[chunk_start].strftime("%Y-%m-%d")
                    hi = common_dates[chunk_end - 1].strftime("%Y-%m-%d")
                    print(f"[{season_name}][{label}][{year}] Chunk {chunk_number}/{total_chunks}: {lo} .. {hi}")

                ml_chunk_idx = ml_idx[chunk_start:chunk_end]
                geos_chunk_idx = geos_idx[chunk_start:chunk_end]
                obs_chunk_idx = obs_idx[chunk_start:chunk_end]
                chunk_weeks = np.asarray([int(date.isocalendar().week) for date in common_dates[chunk_start:chunk_end]], dtype=np.int32)
                week_indices = np.clip(chunk_weeks - 1, 0, 52)

                for lead_idx in range(lead_count):
                    ml_chunk = extract_var_chunk(ml_ds, ml_layout, ml_var, ml_chunk_idx, lead_idx, var_key)
                    geos_chunk = extract_var_chunk(geos_ds, geos_layout, geos_var, geos_chunk_idx, lead_idx, var_key)
                    obs_chunk = extract_var_chunk(obs_ds, obs_layout, obs_var, obs_chunk_idx, lead_idx, var_key)

                    if ml_chunk.ndim != 4 or geos_chunk.ndim != 4 or obs_chunk.ndim != 3:
                        raise ValueError(
                            f"Unexpected chunk ranks for {season_name}/{label}/W{lead_idx+1}: "
                            f"ML {ml_chunk.shape}, GEOS {geos_chunk.shape}, OBS {obs_chunk.shape}"
                        )

                    with np.errstate(invalid="ignore"):
                        ml_mean_raw = np.nanmean(ml_chunk, axis=1)
                        geos_mean_raw = np.nanmean(geos_chunk, axis=1)

                    update_mean_acc(obs_acc, obs_chunk, lead_idx)
                    update_mean_acc(model_accs["ML"], ml_mean_raw, lead_idx)
                    update_mean_acc(model_accs["GEOS"], geos_mean_raw, lead_idx)

                    obs_clim_slice = obs_clim["values"][week_indices, lead_idx]
                    obs_anom = obs_chunk - obs_clim_slice
                    ml_anom = ml_mean_raw - obs_clim_slice
                    geos_anom = geos_mean_raw - obs_clim_slice

                    update_det_acc(model_accs["ML"], ml_anom, obs_anom, lead_idx)
                    update_det_acc(model_accs["GEOS"], geos_anom, obs_anom, lead_idx)

                    update_prob_acc(
                        model_accs["ML"],
                        ml_chunk,
                        obs_chunk,
                        thresholds["low"][lead_idx],
                        thresholds["high"][lead_idx],
                        thresholds["ref_probs"][lead_idx],
                        lead_idx,
                    )
                    update_prob_acc(
                        model_accs["GEOS"],
                        geos_chunk,
                        obs_chunk,
                        thresholds["low"][lead_idx],
                        thresholds["high"][lead_idx],
                        thresholds["ref_probs"][lead_idx],
                        lead_idx,
                    )
            print(f"[{season_name}][{label}][{year}] Done")
        finally:
            ml_ds.close()
            geos_ds.close()
            obs_ds.close()

    if obs_acc is None or model_accs is None:
        raise ValueError(f"No evaluation samples found for {season_name} / {label}")

    lead_count = obs_acc["mean_sum"].shape[0]
    obs_mean_map = finalize_mean_map(obs_acc["mean_sum"], obs_acc["mean_count"])
    geos_mean_map = finalize_mean_map(model_accs["GEOS"]["mean_sum"], model_accs["GEOS"]["mean_count"])
    ml_mean_map = finalize_mean_map(model_accs["ML"]["mean_sum"], model_accs["ML"]["mean_count"])

    geos_rmse_map = finalize_rmse_map(model_accs["GEOS"])
    ml_rmse_map = finalize_rmse_map(model_accs["ML"])
    geos_corr_map = finalize_corr_map(model_accs["GEOS"])
    ml_corr_map = finalize_corr_map(model_accs["ML"])
    geos_bss_map = finalize_skill_map(model_accs["GEOS"]["bs_sum"], model_accs["GEOS"]["bs_ref_sum"])
    ml_bss_map = finalize_skill_map(model_accs["ML"]["bs_sum"], model_accs["ML"]["bs_ref_sum"])
    geos_rpss_map = finalize_skill_map(model_accs["GEOS"]["rps_sum"], model_accs["GEOS"]["rps_ref_sum"])
    ml_rpss_map = finalize_skill_map(model_accs["ML"]["rps_sum"], model_accs["ML"]["rps_ref_sum"])

    aw = np.cos(np.deg2rad(lats))
    aw = np.clip(aw, 0.0, None)
    aw_2d = np.broadcast_to(aw[:, None], obs_mean_map.shape[1:])

    season_rows: List[Dict[str, object]] = []
    report_lines = [
        f"{season_name} {label} seasonal skill evaluation",
        f"Eval years: {args.start_year}-{args.end_year}",
        f"Season months: {list(months)}",
        f"Threshold years: {args.threshold_start_year}-{args.threshold_end_year}",
        f"Total common init dates: {total_common_inits}",
        "Common init dates by year: " + ", ".join(f"{year}={count}" for year, count in sorted(year_common_counts.items())),
        f"Obs weekly climatology: {args.obs_clim_path}",
        "BSS event: upper tercile relative to observed historical thresholds",
        "RPSS categories: terciles relative to observed historical thresholds",
        "",
    ]

    for lead_idx in range(lead_count):
        mean_triplet_vmin, mean_triplet_vmax = spec["mean_vrange"] if spec["mean_vrange"] is not None else (
            float(np.nanmin([obs_mean_map[lead_idx], geos_mean_map[lead_idx], ml_mean_map[lead_idx]])),
            float(np.nanmax([obs_mean_map[lead_idx], geos_mean_map[lead_idx], ml_mean_map[lead_idx]])),
        )

        obs_avg = weighted_mean_2d(obs_mean_map[lead_idx], aw_2d)
        geos_avg = weighted_mean_2d(geos_mean_map[lead_idx], aw_2d)
        ml_avg = weighted_mean_2d(ml_mean_map[lead_idx], aw_2d)

        save_mean_triplet(
            obs_mean_map[lead_idx],
            geos_mean_map[lead_idx],
            ml_mean_map[lead_idx],
            f"{season_name} {label} W{lead_idx + 1}",
            f"mean_state_wk{lead_idx + 1}.png",
            season_dir,
            lats,
            lons,
            spec["cmap"],
            mean_triplet_vmin,
            mean_triplet_vmax,
            obs_avg,
            geos_avg,
            ml_avg,
        )

        geos_rmse_avg = weighted_mean_2d(geos_rmse_map[lead_idx], aw_2d)
        ml_rmse_avg = weighted_mean_2d(ml_rmse_map[lead_idx], aw_2d)
        rmse_vmax = max(
            1e-6,
            float(np.nanpercentile(np.concatenate([geos_rmse_map[lead_idx].ravel(), ml_rmse_map[lead_idx].ravel()]), 99)),
        )
        save_lower_better_triplet(
            geos_rmse_map[lead_idx],
            ml_rmse_map[lead_idx],
            f"{season_name} {label} W{lead_idx + 1}",
            "RMSE",
            f"rmse_wk{lead_idx + 1}.png",
            season_dir,
            lats,
            lons,
            0.0,
            rmse_vmax,
            max(0.25 * rmse_vmax, 1e-6),
            geos_rmse_avg,
            ml_rmse_avg,
        )

        geos_corr_avg = weighted_mean_2d(geos_corr_map[lead_idx], aw_2d)
        ml_corr_avg = weighted_mean_2d(ml_corr_map[lead_idx], aw_2d)
        save_higher_better_triplet(
            geos_corr_map[lead_idx],
            ml_corr_map[lead_idx],
            f"{season_name} {label} W{lead_idx + 1}",
            "Corr",
            f"corr_wk{lead_idx + 1}.png",
            season_dir,
            lats,
            lons,
            -1.0,
            1.0,
            0.4,
            geos_corr_avg,
            ml_corr_avg,
        )

        geos_bss_avg = weighted_mean_2d(geos_bss_map[lead_idx], aw_2d)
        ml_bss_avg = weighted_mean_2d(ml_bss_map[lead_idx], aw_2d)
        save_higher_better_triplet(
            geos_bss_map[lead_idx],
            ml_bss_map[lead_idx],
            f"{season_name} {label} W{lead_idx + 1}",
            "BSS Upper",
            f"bss_upper_wk{lead_idx + 1}.png",
            season_dir,
            lats,
            lons,
            -1.0,
            1.0,
            0.5,
            geos_bss_avg,
            ml_bss_avg,
        )

        geos_rpss_avg = weighted_mean_2d(geos_rpss_map[lead_idx], aw_2d)
        ml_rpss_avg = weighted_mean_2d(ml_rpss_map[lead_idx], aw_2d)
        save_higher_better_triplet(
            geos_rpss_map[lead_idx],
            ml_rpss_map[lead_idx],
            f"{season_name} {label} W{lead_idx + 1}",
            "RPSS",
            f"rpss_wk{lead_idx + 1}.png",
            season_dir,
            lats,
            lons,
            -1.0,
            1.0,
            0.5,
            geos_rpss_avg,
            ml_rpss_avg,
        )

        for model_name, acc in [("GEOS", model_accs["GEOS"]), ("ML", model_accs["ML"])]:
            row = {
                "season": season_name,
                "variable": var_key,
                "variable_label": label,
                "lead_week": lead_idx + 1,
                "model": model_name,
                "domain_obs_mean": obs_avg,
                "domain_forecast_mean": geos_avg if model_name == "GEOS" else ml_avg,
                "domain_rmse": geos_rmse_avg if model_name == "GEOS" else ml_rmse_avg,
                "domain_corr": geos_corr_avg if model_name == "GEOS" else ml_corr_avg,
                "domain_bss_upper": geos_bss_avg if model_name == "GEOS" else ml_bss_avg,
                "domain_rpss": geos_rpss_avg if model_name == "GEOS" else ml_rpss_avg,
                "sample_count_det": int(np.nansum(acc["det_count"][lead_idx])),
                "sample_count_prob": int(np.nansum(acc["rps_count"][lead_idx])),
            }
            season_rows.append(row)

            report_lines.append(
                f"W{lead_idx + 1} {model_name}: "
                f"obs_mean={row['domain_obs_mean']:.3f} {unit}, "
                f"fcst_mean={row['domain_forecast_mean']:.3f} {unit}, "
                f"rmse={row['domain_rmse']:.3f}, corr={row['domain_corr']:.3f}, "
                f"bss_upper={row['domain_bss_upper']:.3f}, rpss={row['domain_rpss']:.3f}"
            )
        report_lines.append("")

    summary_csv = os.path.join(season_dir, "summary.csv")
    report_txt = os.path.join(season_dir, "report.txt")
    write_csv(
        summary_csv,
        [
            "season",
            "variable",
            "variable_label",
            "lead_week",
            "model",
            "domain_obs_mean",
            "domain_forecast_mean",
            "domain_rmse",
            "domain_corr",
            "domain_bss_upper",
            "domain_rpss",
            "sample_count_det",
            "sample_count_prob",
        ],
        season_rows,
    )
    with open(report_txt, "w") as f:
        f.write("\n".join(report_lines).rstrip() + "\n")

    make_domain_metric_figure(
        season_rows,
        os.path.join(season_dir, "domain_metric_summary.png"),
        f"{season_name} {label} Domain-Mean Skill Summary",
    )
    make_domain_mean_figure(
        season_rows,
        os.path.join(season_dir, "domain_mean_state.png"),
        f"{season_name} {label} Domain-Mean Mean State",
        unit,
    )

    print(f"[{season_name}][{label}] ✅ Saved summary CSV: {summary_csv}")
    print(f"[{season_name}][{label}] ✅ Saved report: {report_txt}")
    print(f"[{season_name}][{label}] ✅ Saved plot suite under {season_dir}")
    return season_rows


def main():
    args = parse_args()
    config = load_config(args.config)
    args.data_dir = args.data_dir or config["data_dir"]
    args.start_year = int(args.start_year if args.start_year is not None else config.get("val_start_year", 2020))
    args.end_year = int(args.end_year if args.end_year is not None else config.get("val_end_year", 2021))
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 88)
    print("MULTIV1 SEASONAL SKILL EVALUATION")
    print(f"Eval Years      : {args.start_year}-{args.end_year}")
    print(f"Threshold Years : {args.threshold_start_year}-{args.threshold_end_year}")
    print(f"Seasons         : {args.seasons}")
    print(f"Variables       : {args.variables}")
    print(f"Data Dir        : {args.data_dir}")
    print(f"ML Dir          : {args.ml_dir}")
    print(f"Output Dir      : {os.path.abspath(args.output_dir)}")
    print(f"Obs Climatology : {args.obs_clim_path}")
    print("=" * 88)
    if not HAS_CARTOPY:
        print("⚠️ Cartopy is unavailable. Map plots will be skipped.")

    lats, lons = get_plot_coords(args.data_dir, year_hint=args.end_year)
    all_rows: List[Dict[str, object]] = []

    for season_name in args.seasons:
        months = SEASON_MONTHS[season_name]
        for var_key in args.variables:
            obs_clim = load_obs_weekly_climatology(args.obs_clim_path, var_key)
            season_rows = evaluate_season_variable(args, config, season_name, months, var_key, obs_clim, lats, lons)
            all_rows.extend(season_rows)

    combined_csv = os.path.join(args.output_dir, "combined_domain_summary.csv")
    write_csv(
        combined_csv,
        [
            "season",
            "variable",
            "variable_label",
            "lead_week",
            "model",
            "domain_obs_mean",
            "domain_forecast_mean",
            "domain_rmse",
            "domain_corr",
            "domain_bss_upper",
            "domain_rpss",
            "sample_count_det",
            "sample_count_prob",
        ],
        all_rows,
    )
    print(f"\n✅ Combined domain summary written to {combined_csv}")
    print("✅ Seasonal skill evaluation completed successfully.")


if __name__ == "__main__":
    main()
