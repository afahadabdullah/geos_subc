#!/usr/bin/env python3
"""Build manuscript figures from FIMr1p1-FlowMatch evaluation products.

This script is intentionally downstream of the expensive evaluators. It reads
the CSV/NetCDF products written by:

  - ml_model/evaluate_matrix_suite_flow_finalv1_global.py
  - ml_model/compare_noise_flow_finalv1_global.py
  - ml_model/evaluate_event_catalog_flow_finalv1_global.py
  - ml_model/evaluate_event_quantile_forecast_flow_finalv1_global.py

When an expected evaluation artifact is missing, the relevant panel is rendered
as a clear missing-data note so the full figure set can still be regenerated
while final runs are pending.

Figure layout (approved):
  1  Combined framework, architecture, and sampling schematic
  2  Global PR skill: lead bars + season-lead heatmaps + spatial maps (3x2)
  3  Global T2M skill: same structure as Fig 2 (3x2)
  4  Extreme-event subset: PR + T2M CRPS/RMSE skill bars (2x2)
  5  California AR flood PR case study (2x3 cropped panels)
  6  UK heat event T2M case study (2x3 cropped panels)
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD = "FIMr1p1-FlowMatch"
BASELINE = "FIMr1p1"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
LEADS = [1, 2, 3, 4]
VARIABLE_LABELS = {"pr": "Precipitation", "t2m": "2 m temperature"}
VARIABLE_SHORT = {"pr": "PR", "t2m": "T2M"}
METRIC_LABELS = {
    "crps": "CRPS",
    "rmse": "RMSE",
    "crps_skill_pct": "CRPS skill (%)",
    "rmse_skill_pct": "RMSE skill (%)",
    "calibrated_bss_diff": "Calibrated BSS gain",
    "bss_diff": "Raw BSS gain",
}

PRIMARY_EVENT_IDS = [
    "europe_t2m_202207_uk_heatwave",
    "conus_pr_202301_california_atmospheric_rivers",
    "bangladesh_pr_202206_meghalaya_sylhet_downpours",
]
PRIMARY_EVENT_LABELS = {
    "europe_t2m_202207_uk_heatwave": "UK July 2022 heatwave (T2M)",
    "conus_pr_202301_california_atmospheric_rivers": "California Jan 2023 Atmospheric Rivers (PR)",
    "bangladesh_pr_202206_meghalaya_sylhet_downpours": "Bangladesh/Sylhet June 2022 flood (PR)",
}
PRIMARY_EVENT_CONTEXT = {
    "europe_t2m_202207_uk_heatwave": {
        "event_window": "18-19 Jul 2022 target week",
        "domain": "United Kingdom / western Europe",
        "diagnostic": "hot-tail T2M exceedance",
    },
    "conus_pr_202301_california_atmospheric_rivers": {
        "event_window": "Jan 2023 target week",
        "domain": "California / West Coast",
        "diagnostic": "heavy-rain PR exceedance",
    },
    "bangladesh_pr_202206_meghalaya_sylhet_downpours": {
        "event_window": "17-23 Jun 2022 target week",
        "domain": "NE Bangladesh / Meghalaya-Assam",
        "diagnostic": "heavy-rain PR exceedance",
    },
}

EVENT_PLOT_LAYOUTS = {
    "spatial_event_focus": (2, 4),
    "spatial_risk": (3, 4),
    "spatial_verification": (4, 4),
    "quantile_spatial": (4, 4),
}

COLOR_FIM = "#b43c30"
COLOR_MODEL = "#202124"
COLOR_PR = "#2a6fbb"
COLOR_T2M = "#b07021"
COLOR_POS = "#2c7a4b"
COLOR_NEG = "#a33a3a"
TEXT_DARK = "#1f2933"
TEXT_MUTED = "#5b6770"
BOX_EDGE = "#2d4658"

VARIABLE_COLORS = {"pr": COLOR_PR, "t2m": COLOR_T2M}

DEFAULT_MATRIX_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_land_obsclim_chunked",
    "ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_e90_s50",
]
DEFAULT_EVENT_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023",
]
DEFAULT_QUANTILE_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/event_quantile_eval_global_2021_2023",
]
NOISE_CSV_PATTERNS = [
    "ml_output_noise_compare_global_flow_finalv1/noise_comparison_global_*.csv",
    "ml_output_flow_finalv1_global_noisectx_t2mres/noise_comparison_global_*.csv",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures from existing evaluation outputs."
    )
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--matrix-dir", default=None, help="Directory containing matrix_summary_metrics.csv.")
    parser.add_argument("--event-dir", default=None, help="Directory containing event_selected_lead_metrics.csv.")
    parser.add_argument("--quantile-dir", default=None, help="Directory containing event quantile CSV products.")
    parser.add_argument("--noise-csv", default=None, help="Optional explicit noise comparison CSV.")
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="pdf",
        help="Figure file format to write.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--matrix-subset", default="all_data", choices=("all_data", "extreme_events"))
    parser.add_argument("--spatial-subset", default="all_data", choices=("all_data", "extreme_events"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def first_existing_dir(explicit: str | None, candidates: list[str]) -> Path:
    if explicit:
        return Path(explicit)
    for item in candidates:
        path = Path(item)
        if path.exists():
            return path
    return Path(candidates[0])


def newest_matching(patterns: list[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(pattern))
    matches = [path for path in matches if path.exists()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_csv_or_none(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def output_formats(fmt: str) -> list[str]:
    return ["pdf", "png"] if fmt == "both" else [fmt]


def matrix_summary_path(matrix_dir: Path) -> Path:
    return matrix_dir / "matrix_summary_metrics.csv"


def matrix_spatial_path(matrix_dir: Path) -> Path:
    return matrix_dir / "matrix_spatial_metrics.nc"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def wrap(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def clean_label(value: object) -> str:
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def style_figure(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.patch.set_facecolor("white")


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#d9dee3", linewidth=0.6, alpha=0.75)
    ax.tick_params(labelsize=8)


def missing_panel(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    box = FancyBboxPatch(
        (0.08, 0.18),
        0.84,
        0.58,
        transform=ax.transAxes,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor="#f4f6f8",
        edgecolor="#a9b4bf",
        linewidth=1.1,
    )
    ax.add_patch(box)
    ax.text(
        0.50,
        0.47,
        wrap(message, 46),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=TEXT_MUTED,
    )


# ---------------------------------------------------------------------------
# Data aggregation helpers
# ---------------------------------------------------------------------------

def weighted_average(series: pd.Series, weights: pd.Series | None) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if weights is None:
        weights_arr = np.ones_like(values, dtype=float)
    else:
        weights_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights_arr) & (weights_arr > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights_arr[valid]))


def group_weights(df: pd.DataFrame) -> pd.Series | None:
    for name in ("weight_sum", "n_cases", "n_forecasts", "n_samples"):
        if name in df:
            return df[name]
    return None


def skill_pct(model_value: float, geos_value: float) -> float:
    if not np.isfinite(model_value) or not np.isfinite(geos_value) or abs(geos_value) <= 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - model_value / geos_value))


def aggregate_matrix_by_lead(summary: pd.DataFrame | None, subset: str = "all_data") -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    df = summary.copy()
    if "subset" in df:
        df = df[df["subset"].eq(subset)]
    if "group_type" in df:
        season_df = df[df["group_type"].eq("valid_season_lead")]
        month_df = df[df["group_type"].eq("valid_month_lead")]
        df = season_df if not season_df.empty else month_df
    if df.empty:
        return pd.DataFrame()

    metric_cols = [
        "model_crps",
        "geos_crps",
        "model_rmse",
        "geos_rmse",
        "model_spread",
        "geos_spread",
        "model_bss",
        "geos_bss",
        "model_calibrated_bss",
        "geos_calibrated_bss",
        "bss_diff",
        "calibrated_bss_diff",
        "crps_skill_pct",
        "rmse_skill_pct",
    ]
    rows = []
    for (variable, lead), group in df.groupby(["variable", "lead"], dropna=False):
        row = {"variable": str(variable), "lead": int(lead)}
        weights = group_weights(group)
        for col in metric_cols:
            if col in group:
                row[col] = weighted_average(group[col], weights)
        if "crps_skill_pct" not in row and {"model_crps", "geos_crps"} <= set(row):
            row["crps_skill_pct"] = skill_pct(row["model_crps"], row["geos_crps"])
        if "rmse_skill_pct" not in row and {"model_rmse", "geos_rmse"} <= set(row):
            row["rmse_skill_pct"] = skill_pct(row["model_rmse"], row["geos_rmse"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variable", "lead"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Season-lead heatmap helpers
# ---------------------------------------------------------------------------

def season_lead_values(summary: pd.DataFrame | None, variable: str, metric_col: str, subset: str) -> np.ndarray | None:
    if summary is None or summary.empty:
        return None
    required = {"subset", "variable", "group_type", "group_value", "lead", metric_col}
    if not required <= set(summary.columns):
        return None
    df = summary[
        summary["subset"].eq(subset)
        & summary["variable"].eq(variable)
        & summary["group_type"].eq("valid_season_lead")
    ]
    if df.empty:
        return None
    arr = np.full((len(SEASONS), len(LEADS)), np.nan, dtype=float)
    for i, season in enumerate(SEASONS):
        for j, lead in enumerate(LEADS):
            group = df[df["group_value"].astype(str).eq(season) & df["lead"].astype(int).eq(lead)]
            if not group.empty:
                arr[i, j] = weighted_average(group[metric_col], group_weights(group))
    return arr


def heatmap_limits(arrays: list[np.ndarray | None], fallback: float = 20.0) -> tuple[float, float]:
    values = []
    for arr in arrays:
        if arr is not None:
            values.extend(np.asarray(arr, dtype=float)[np.isfinite(arr)].ravel().tolist())
    if not values:
        return -fallback, fallback
    lim = max(float(np.nanpercentile(np.abs(values), 95)), 1e-6)
    lim = min(max(lim, 5.0), 100.0)
    return -lim, lim


def plot_heatmap_panel(
    ax: plt.Axes,
    arr: np.ndarray | None,
    raw_model_arr: np.ndarray | None,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "RdYlGn",
    is_change: bool = True,
):
    if arr is None or not np.isfinite(arr).any():
        missing_panel(ax, title, "Missing valid_season_lead matrix rows.")
        return None
        
    if is_change:
        from matplotlib.colors import TwoSlopeNorm
        vlim = max(abs(vmin), abs(vmax))
        vlim = max(vlim, 0.1)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
        im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xticks(np.arange(len(LEADS)))
    ax.set_xticklabels([str(lead) for lead in LEADS], fontsize=8, fontweight="bold")
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_yticklabels(SEASONS, fontsize=8, fontweight="bold")
    ax.set_xlabel("Lead week", fontsize=8, fontweight="bold")
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                if is_change:
                    raw_val = raw_model_arr[i, j] if raw_model_arr is not None else np.nan
                    if np.isfinite(raw_val):
                        if raw_val > 10.0:
                            label = f"{val:+.0f}%\n({raw_val:.1f})"
                        else:
                            label = f"{val:+.0f}%\n({raw_val:.2f})"
                    else:
                        label = f"{val:+.0f}%"
                else:
                    if val > 10.0:
                        label = f"{val:.1f}"
                    else:
                        label = f"{val:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.0, color="#101820", fontweight="bold")
    return im


# ---------------------------------------------------------------------------
# Spatial map helpers
# ---------------------------------------------------------------------------

def load_xarray_dataset(path: Path):
    if not path.exists():
        return None
    try:
        import xarray as xr
    except Exception as exc:
        print(f"xarray is required for spatial maps but is unavailable: {exc}")
        return None
    try:
        return xr.open_dataset(path)
    except Exception as exc:
        print(f"Could not open {path}: {exc}")
        return None


def coord_values_for(ds, dim: str, desired: list[object]) -> list[object]:
    desired_text = {str(item) for item in desired}
    return [value for value in ds[dim].values.tolist() if str(value) in desired_text]


def weighted_dataarray_mean(values, weights, dims: list[str]):
    weights = weights.where(np.isfinite(weights) & (weights > 0), 0.0)
    values = values.where(np.isfinite(values))
    numerator = (values * weights).sum(dim=dims, skipna=True)
    denominator = weights.where(np.isfinite(values)).sum(dim=dims, skipna=True)
    return numerator / denominator


def spatial_metric_map(ds, variable: str, metric_name: str, subset: str):
    if ds is None:
        return None, None, None
    needed_dims = {"subset", "variable", "group_type", "group_value", "lead", "lat", "lon"}
    if not needed_dims <= set(ds.dims):
        return None, None, None
    selector = {
        "subset": coord_values_for(ds, "subset", [subset]),
        "variable": coord_values_for(ds, "variable", [variable]),
        "group_type": coord_values_for(ds, "group_type", ["valid_season_lead"]),
        "group_value": coord_values_for(ds, "group_value", SEASONS),
        "lead": coord_values_for(ds, "lead", LEADS),
    }
    if any(not values for values in selector.values()):
        return None, None, None
    sel = {dim: values for dim, values in selector.items()}
    count = ds["sample_count"].sel(sel)
    reduce_dims = ["group_value", "lead"]
    if metric_name in ("crps_skill_pct", "rmse_skill_pct", "mae_skill_pct"):
        metric = metric_name.replace("_skill_pct", "")
        model = weighted_dataarray_mean(ds[f"model_{metric}"].sel(sel), count, reduce_dims)
        geos = weighted_dataarray_mean(ds[f"geos_{metric}"].sel(sel), count, reduce_dims)
        field = 100.0 * (1.0 - model / geos)
    elif metric_name == "calibrated_bss_diff":
        field = weighted_dataarray_mean(ds["calibrated_bss_diff"].sel(sel), count, reduce_dims)
    elif metric_name == "bss_diff":
        field = weighted_dataarray_mean(ds["bss_diff"].sel(sel), count, reduce_dims)
    else:
        field = weighted_dataarray_mean(ds[metric_name].sel(sel), count, reduce_dims)
    field = field.squeeze(drop=True)
    return np.asarray(ds["lon"].values), np.asarray(ds["lat"].values), np.asarray(field.values, dtype=float)


def spatial_limits(fields: list[np.ndarray | None], fallback: float = 30.0) -> tuple[float, float]:
    vals = []
    for field in fields:
        if field is not None:
            finite = np.asarray(field)[np.isfinite(field)]
            if finite.size:
                vals.extend(finite.tolist())
    if not vals:
        return -fallback, fallback
    lim = max(float(np.nanpercentile(np.abs(vals), 95)), 1e-6)
    lim = min(max(lim, 5.0), 100.0)
    return -lim, lim


def plot_plain_map(
    ax: plt.Axes,
    lons: np.ndarray | None,
    lats: np.ndarray | None,
    field: np.ndarray | None,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "RdYlGn",
    is_change: bool = True,
):
    if lons is None or lats is None or field is None or not np.isfinite(field).any():
        missing_panel(ax, title, "Missing matrix_spatial_metrics.nc data for this map.")
        return None
        
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if is_change:
        from matplotlib.colors import TwoSlopeNorm
        vlim = max(abs(vmin), abs(vmax))
        vlim = max(vlim, 0.1)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
        levels = np.linspace(-vlim, vlim, 21)
        mesh = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, norm=norm, extend="neither", transform=ccrs.PlateCarree())
    else:
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        levels = np.linspace(vmin, vmax, 21)
        mesh = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, extend="neither", transform=ccrs.PlateCarree())
        
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)

    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=1.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#222222", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#555555", linestyle=":", zorder=2)

    if np.nanmax(lons) - np.nanmin(lons) > 340:
        ax.set_global()
    else:
        ax.set_extent([float(np.nanmin(lons)), float(np.nanmax(lons)), float(np.nanmin(lats)), float(np.nanmax(lats))], crs=ccrs.PlateCarree())

    ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, color="#d9dee3", linewidth=0.25, alpha=0.7)
    return mesh


# ---------------------------------------------------------------------------
# Drawing helpers (for schematics)
# ---------------------------------------------------------------------------

def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#667788") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    face: str,
    edge: str = BOX_EDGE,
    title_size: float = 9.5,
    body_size: float = 8.0,
    body_wrap: int = 30,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02 * w, y + h - 0.23 * h, title, transform=ax.transAxes, ha="left", va="center",
            fontsize=title_size, fontweight="bold", color=TEXT_DARK)
    ax.text(x + 0.02 * w, y + 0.40 * h, wrap(body, body_wrap), transform=ax.transAxes, ha="left", va="center",
            fontsize=body_size, color=TEXT_DARK, linespacing=1.18)


# ===================================================================
# Figure 1 — Combined framework overview
# ===================================================================

def figure_1_framework_overview(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    style_figure(
        fig,
        "Figure 1. Forecast-refinement framework",
        "FIMr1p1 guidance and observed context condition a flow-matching ensemble generator.",
    )
    ax.set_axis_off()

    col_x = [0.035, 0.285, 0.535, 0.785]
    col_w = 0.185
    top_y = 0.62
    box_h = 0.24

    draw_box(
        ax,
        (col_x[0], top_y),
        (col_w, box_h),
        "Forecast inputs",
        f"{BASELINE} PR/T2M ensemble summaries, recent observed predictors, static geography, season, and lead week.",
        "#eef6fb",
        body_wrap=27,
    )
    draw_box(
        ax,
        (col_x[1], top_y),
        (col_w, box_h),
        "Physical ensemble prior",
        "Gaussian baseline or EOF-LHS perturbations conditioned on MJO, NAO, ENSO, and lead.",
        "#f3eefe",
        body_wrap=27,
    )
    draw_box(
        ax,
        (col_x[2], top_y),
        (col_w, box_h),
        "Conditional flow model",
        "Shared U-Net representation with global context tokens and separate PR/T2M lead heads.",
        "#edf1fb",
        body_wrap=27,
    )
    draw_box(
        ax,
        (col_x[3], top_y),
        (col_w, box_h),
        "Weekly ensemble outputs",
        "90-member PR and T2M forecasts for weeks 1-4, evaluated with CRPS, RMSE, BSS, spatial maps, and event diagnostics.",
        "#eaf5ef",
        body_wrap=28,
    )

    for i in range(3):
        add_arrow(ax, (col_x[i] + col_w + 0.015, top_y + box_h / 2), (col_x[i + 1] - 0.015, top_y + box_h / 2))

    draw_box(
        ax,
        (0.13, 0.24),
        (0.27, 0.20),
        "Why the prior matters",
        "Structured perturbations give the ODE coherent large-scale covariance at t=0 instead of asking the model to build it from white noise.",
        "#fff6db",
        body_wrap=39,
    )
    draw_box(
        ax,
        (0.47, 0.24),
        (0.27, 0.20),
        "Sampling",
        "Euler integration maps x0 to forecast fields; variance tempering adjusts spread without changing the learned velocity path.",
        "#fef1df",
        body_wrap=39,
    )

    add_arrow(ax, (col_x[1] + col_w / 2, top_y), (0.265, 0.44), color="#8a69a6")
    add_arrow(ax, (0.40, 0.34), (0.47, 0.34), color="#8a69a6")
    add_arrow(ax, (0.605, 0.44), (col_x[2] + col_w / 2, top_y), color="#8a69a6")

    ax.text(
        0.13,
        0.12,
        "Noise ablation: EOF-LHS higher-spread prior improves mean CRPS over Gaussian "
        "(PR 28.4% vs 24.9%; T2M 43.2% vs 41.7% relative to FIMr1p1).",
        transform=ax.transAxes,
        fontsize=9.2,
        color=TEXT_MUTED,
    )
    return save_figure(fig, output_dir, "fig1_framework_overview", formats, dpi)


# ===================================================================
# Figure 2 — Architecture and sampling detail (schematic, unchanged)
# ===================================================================

def figure_2_architecture(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.2, 6.9))
    style_figure(
        fig,
        "Figure 2. Architecture and sampling detail",
        "Deterministic velocity pathway is separated from stochastic initialization and variance tempering.",
    )
    ax.set_axis_off()

    draw_box(ax, (0.04, 0.63), (0.16, 0.13), "State x_t", "PR + T2M residual field at flow time t.", "#f6f7fb")
    draw_box(
        ax,
        (0.04, 0.38),
        (0.16, 0.15),
        "Condition c",
        f"{BASELINE} stats, observed predictors, static geography, season, lead.",
        "#eef7fb",
        body_wrap=22,
    )
    draw_box(ax, (0.25, 0.55), (0.18, 0.16), "Shared encoder", "Multi-scale convolutional feature hierarchy.", "#edf1fb")
    draw_box(ax, (0.48, 0.55), (0.18, 0.16), "Bottleneck attention", "Local U-Net cells attend to global context tokens.", "#f0eefb")
    draw_box(ax, (0.71, 0.55), (0.18, 0.16), "Shared decoder", "Upsampling path returns full-grid features.", "#edf1fb")

    draw_box(
        ax,
        (0.72, 0.31),
        (0.13, 0.13),
        "PR heads",
        "Week 1-4 velocity and variance.",
        "#eaf3f8",
        body_wrap=18,
    )
    draw_box(
        ax,
        (0.86, 0.31),
        (0.13, 0.13),
        "T2M heads",
        "Week 1-4 velocity and variance.",
        "#f8ece6",
        body_wrap=18,
    )
    draw_box(ax, (0.04, 0.10), (0.20, 0.15), "x0 prior", "EOF modes conditioned on MJO/NAO/ENSO + random perturbations.", "#f3eefe")
    draw_box(ax, (0.31, 0.10), (0.21, 0.15), "ODE integration", "Explicit Euler updates x_{t+dt}=x_t+dt v_theta.", "#fff6db")
    draw_box(ax, (0.59, 0.10), (0.22, 0.15), "Variance tempering", "sigma_eff = 1 + beta(sigma_theta - 1); coarse and clipped.", "#fef1df")
    draw_box(ax, (0.84, 0.10), (0.13, 0.15), "Ensemble", "PR/T2M samples for weeks 1-4.", "#eaf5ef", body_wrap=18)

    add_arrow(ax, (0.20, 0.70), (0.25, 0.64))
    add_arrow(ax, (0.20, 0.46), (0.25, 0.61))
    add_arrow(ax, (0.43, 0.63), (0.48, 0.63))
    add_arrow(ax, (0.66, 0.63), (0.71, 0.63))
    add_arrow(ax, (0.80, 0.55), (0.785, 0.44))
    add_arrow(ax, (0.82, 0.55), (0.925, 0.44))
    add_arrow(ax, (0.24, 0.18), (0.31, 0.18), color="#8a69a6")
    add_arrow(ax, (0.52, 0.18), (0.59, 0.18), color="#8a69a6")
    add_arrow(ax, (0.81, 0.18), (0.84, 0.18), color="#8a69a6")
    add_arrow(ax, (0.87, 0.31), (0.84, 0.24), color="#8a69a6")
    add_arrow(ax, (0.73, 0.31), (0.72, 0.24), color="#8a69a6")

    ax.text(0.03, 0.82, "Velocity pathway", transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT_MUTED)
    ax.text(0.03, 0.29, "Stochastic sampling pathway", transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT_MUTED)
    return save_figure(fig, output_dir, "fig2_architecture_sampling", formats, dpi)


# ===================================================================
# Plot panel helpers
# ===================================================================

def plot_skill_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    variable: str,
    metric: str,
    color: str | None = None,
    title: str | None = None,
) -> None:
    """Grouped bar chart of skill (%) by lead week for a single variable."""
    skill_col = f"{metric}_skill_pct"
    if data.empty or skill_col not in data:
        # Try to compute from raw columns
        model_col = f"model_{metric}"
        geos_col = f"geos_{metric}"
        if data.empty or model_col not in data or geos_col not in data:
            missing_panel(ax, f"{VARIABLE_SHORT.get(variable, variable)} {metric.upper()} skill",
                          "Missing matrix summary data.")
            return
        sub = data[data["variable"].eq(variable)].sort_values("lead")
        if sub.empty:
            missing_panel(ax, f"{VARIABLE_SHORT.get(variable, variable)} {metric.upper()} skill",
                          f"No rows for variable {variable}.")
            return
        skill_values = [skill_pct(float(r[model_col]), float(r[geos_col])) for _, r in sub.iterrows()]
    else:
        sub = data[data["variable"].eq(variable)].sort_values("lead")
        if sub.empty:
            missing_panel(ax, f"{VARIABLE_SHORT.get(variable, variable)} {metric.upper()} skill",
                          f"No rows for variable {variable}.")
            return
        skill_values = sub[skill_col].to_numpy(dtype=float)

    leads = sub["lead"].to_numpy(dtype=int)
    bar_color = color or VARIABLE_COLORS.get(variable, COLOR_MODEL)

    bars = ax.bar(leads, skill_values, width=0.55, color=bar_color, alpha=0.88, edgecolor="white", linewidth=0.8)

    # Numeric labels on bars
    for bar_obj, val in zip(bars, skill_values):
        if np.isfinite(val):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=TEXT_DARK,
            )

    # Mean-skill reference line
    finite_skills = [v for v in skill_values if np.isfinite(v)]
    if finite_skills:
        mean_skill = np.mean(finite_skills)
        ax.axhline(mean_skill, color=bar_color, linewidth=1.0, linestyle="--", alpha=0.5)
        ax.text(
            4.4, mean_skill, f"mean {mean_skill:.1f}%",
            va="center", fontsize=6.5, color=TEXT_MUTED,
        )

    style_axis(ax)
    ax.set_xticks(LEADS)
    ax.set_xticklabels([f"W{l}" for l in LEADS])
    ax.set_xlabel("Lead week", fontsize=8.5)
    ax.set_ylabel(f"{metric.upper()} skill (%)", fontsize=8.5)
    bar_title = title or f"{VARIABLE_SHORT.get(variable, variable)} {metric.upper()} skill vs {BASELINE}"
    ax.set_title(bar_title, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)


def add_colorbar(fig: plt.Figure, im, ax, label: str, is_change: bool = True, shrink: float = 0.8, pad: float = 0.03):
    from matplotlib import ticker
    if im is None:
        return None
    cbar = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad, extend="neither")
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if is_change:
        cbar.locator = ticker.MaxNLocator(nbins=7, symmetric=True)
    else:
        cbar.locator = ticker.MaxNLocator(nbins=6)
    cbar.update_ticks()
    return cbar


# ===================================================================
# Figure 3 — Global PR skill (3x2)
# ===================================================================

def figure_variable_skill(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    summary: pd.DataFrame | None,
    matrix_dir: Path,
    variable: str,
    fig_num: int,
    stem: str,
    subset: str,
    spatial_subset: str,
) -> list[Path]:
    """Shared builder for the per-variable multi-panel skill figure (2x4).

    Row 1: Season-lead skill heatmaps (GEOS, Skill Change)
    Row 2: Spatial skill maps (GEOS, Skill Change)
    """
    var_label = VARIABLE_LABELS.get(variable, variable)
    var_short = VARIABLE_SHORT.get(variable, variable.upper())

    fig = plt.figure(figsize=(18.0, 7.8))
    style_figure(
        fig,
        f"Figure {fig_num}. {var_label} forecast skill vs {BASELINE}",
        f"Season-lead matrices (top) and spatial maps (bottom). Positive = ML improves on {BASELINE}.",
    )
    
    gs = fig.add_gridspec(2, 4, height_ratios=[0.70, 1.30], hspace=0.15, wspace=0.12, left=0.02, right=0.98, bottom=0.03, top=0.97)

    # --- Row 1: Season-lead heatmaps ---
    geos_crps_arr = season_lead_values(summary, variable, "geos_crps", subset)
    model_crps_arr = season_lead_values(summary, variable, "model_crps", subset)
    crps_change_arr = season_lead_values(summary, variable, "crps_skill_pct", subset)
    
    geos_rmse_arr = season_lead_values(summary, variable, "geos_rmse", subset)
    model_rmse_arr = season_lead_values(summary, variable, "model_rmse", subset)
    rmse_change_arr = season_lead_values(summary, variable, "rmse_skill_pct", subset)

    ax_crps_geos_hm = fig.add_subplot(gs[0, 0])
    ax_crps_change_hm = fig.add_subplot(gs[0, 1])
    ax_rmse_geos_hm = fig.add_subplot(gs[0, 2])
    ax_rmse_change_hm = fig.add_subplot(gs[0, 3])

    gc_vmin, gc_vmax = (np.nanmin(geos_crps_arr), np.nanmax(geos_crps_arr)) if geos_crps_arr is not None else (0.0, 1.0)
    gr_vmin, gr_vmax = (np.nanmin(geos_rmse_arr), np.nanmax(geos_rmse_arr)) if geos_rmse_arr is not None else (0.0, 1.0)
    
    c_vmin, c_vmax = heatmap_limits([crps_change_arr])
    r_vmin, r_vmax = heatmap_limits([rmse_change_arr])

    im_c_g = plot_heatmap_panel(ax_crps_geos_hm, geos_crps_arr, None, f"(a) {var_short} {BASELINE} CRPS", gc_vmin, gc_vmax, cmap="viridis", is_change=False)
    im_c_c = plot_heatmap_panel(ax_crps_change_hm, crps_change_arr, model_crps_arr, f"(b) {var_short} CRPS Skill vs {BASELINE}", c_vmin, c_vmax, cmap="RdYlGn", is_change=True)
    im_r_g = plot_heatmap_panel(ax_rmse_geos_hm, geos_rmse_arr, None, f"(c) {var_short} {BASELINE} RMSE", gr_vmin, gr_vmax, cmap="viridis", is_change=False)
    im_r_c = plot_heatmap_panel(ax_rmse_change_hm, rmse_change_arr, model_rmse_arr, f"(d) {var_short} RMSE Skill vs {BASELINE}", r_vmin, r_vmax, cmap="RdYlGn", is_change=True)

    add_colorbar(fig, im_c_g, ax_crps_geos_hm, "CRPS", is_change=False)
    add_colorbar(fig, im_c_c, ax_crps_change_hm, "Skill (%)", is_change=True)
    add_colorbar(fig, im_r_g, ax_rmse_geos_hm, "RMSE", is_change=False)
    add_colorbar(fig, im_r_c, ax_rmse_change_hm, "Skill (%)", is_change=True)

    # --- Row 2: Spatial maps ---
    ds = load_xarray_dataset(matrix_spatial_path(matrix_dir))
    map_geos_crps = spatial_metric_map(ds, variable, "geos_crps", spatial_subset)
    map_crps_change = spatial_metric_map(ds, variable, "crps_skill_pct", spatial_subset)
    map_geos_rmse = spatial_metric_map(ds, variable, "geos_rmse", spatial_subset)
    map_rmse_change = spatial_metric_map(ds, variable, "rmse_skill_pct", spatial_subset)

    # Use geos_crps as mask to set ocean points to NaN across all maps
    if map_geos_crps[2] is not None:
        ocean_mask = np.isnan(map_geos_crps[2])
        map_geos_crps = (map_geos_crps[0], map_geos_crps[1], np.where(ocean_mask, np.nan, map_geos_crps[2]))
        map_crps_change = (map_crps_change[0], map_crps_change[1], np.where(ocean_mask, np.nan, map_crps_change[2]))
        map_geos_rmse = (map_geos_rmse[0], map_geos_rmse[1], np.where(ocean_mask, np.nan, map_geos_rmse[2]))
        map_rmse_change = (map_rmse_change[0], map_rmse_change[1], np.where(ocean_mask, np.nan, map_rmse_change[2]))

    import cartopy.crs as ccrs
    ax_crps_geos_map = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_crps_change_map = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax_rmse_geos_map = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
    ax_rmse_change_map = fig.add_subplot(gs[1, 3], projection=ccrs.PlateCarree())

    if map_geos_crps[2] is not None:
        gc_map_finite = map_geos_crps[2][np.isfinite(map_geos_crps[2])]
        g_crps_vmin = float(np.nanpercentile(gc_map_finite, 2)) if gc_map_finite.size else 0.0
        g_crps_vmax = float(np.nanpercentile(gc_map_finite, 98)) if gc_map_finite.size else 1.0
    else:
        g_crps_vmin, g_crps_vmax = 0.0, 1.0

    if map_geos_rmse[2] is not None:
        gr_map_finite = map_geos_rmse[2][np.isfinite(map_geos_rmse[2])]
        g_rmse_vmin = float(np.nanpercentile(gr_map_finite, 2)) if gr_map_finite.size else 0.0
        g_rmse_vmax = float(np.nanpercentile(gr_map_finite, 98)) if gr_map_finite.size else 1.0
    else:
        g_rmse_vmin, g_rmse_vmax = 0.0, 1.0

    s_crps_vmin, s_crps_vmax = spatial_limits([map_crps_change[2]])
    s_rmse_vmin, s_rmse_vmax = spatial_limits([map_rmse_change[2]])

    im_c_map_g = plot_plain_map(ax_crps_geos_map, *map_geos_crps, f"(e) {var_short} {BASELINE} CRPS map", g_crps_vmin, g_crps_vmax, cmap="viridis", is_change=False)
    im_c_map_c = plot_plain_map(ax_crps_change_map, *map_crps_change, f"(f) {var_short} ML CRPS skill improvement (%)", s_crps_vmin, s_crps_vmax, cmap="RdYlGn", is_change=True)
    im_r_map_g = plot_plain_map(ax_rmse_geos_map, *map_geos_rmse, f"(g) {var_short} {BASELINE} RMSE map", g_rmse_vmin, g_rmse_vmax, cmap="viridis", is_change=False)
    im_r_map_c = plot_plain_map(ax_rmse_change_map, *map_rmse_change, f"(h) {var_short} ML RMSE skill improvement (%)", s_rmse_vmin, s_rmse_vmax, cmap="RdYlGn", is_change=True)

    if ds is not None:
        ds.close()

    add_colorbar(fig, im_c_map_g, ax_crps_geos_map, "CRPS", is_change=False)
    add_colorbar(fig, im_c_map_c, ax_crps_change_map, "Skill Improvement (%)", is_change=True)
    add_colorbar(fig, im_r_map_g, ax_rmse_geos_map, "RMSE", is_change=False)
    add_colorbar(fig, im_r_map_c, ax_rmse_change_map, "Skill Improvement (%)", is_change=True)

    return save_figure(fig, output_dir, stem, formats, dpi)


def figure_3_pr_skill(
    output_dir: Path, formats: list[str], dpi: int,
    summary: pd.DataFrame | None, matrix_dir: Path,
    subset: str, spatial_subset: str,
) -> list[Path]:
    return figure_variable_skill(
        output_dir, formats, dpi, summary, matrix_dir,
        variable="pr", fig_num=3, stem="fig3_pr_skill",
        subset=subset, spatial_subset=spatial_subset,
    )


# ===================================================================
# Figure 4 — Global T2M skill (3x2)
# ===================================================================

def figure_4_t2m_skill(
    output_dir: Path, formats: list[str], dpi: int,
    summary: pd.DataFrame | None, matrix_dir: Path,
    subset: str, spatial_subset: str,
) -> list[Path]:
    return figure_variable_skill(
        output_dir, formats, dpi, summary, matrix_dir,
        variable="t2m", fig_num=4, stem="fig4_t2m_skill",
        subset=subset, spatial_subset=spatial_subset,
    )


# ===================================================================
# Figure 5 — Extreme-event subset skill (2x2)
# ===================================================================

def figure_5_extreme_subset(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    summary: pd.DataFrame | None,
) -> list[Path]:
    """Skill bars using only the extreme-event subset from the matrix summary."""
    agg = aggregate_matrix_by_lead(summary, subset="extreme_events")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))
    style_figure(
        fig,
        f"Figure 5. Forecast skill on observed extreme events (p95 threshold)",
        f"CRPS and RMSE skill (%) vs {BASELINE} on the extreme-event subset. Positive = ML improves.",
    )

    plot_skill_bars(axes[0, 0], agg, "pr", "crps", title=f"(a) PR CRPS skill vs {BASELINE}")
    plot_skill_bars(axes[0, 1], agg, "pr", "rmse", title=f"(b) PR RMSE skill vs {BASELINE}")
    plot_skill_bars(axes[1, 0], agg, "t2m", "crps", title=f"(c) T2M CRPS skill vs {BASELINE}")
    plot_skill_bars(axes[1, 1], agg, "t2m", "rmse", title=f"(d) T2M RMSE skill vs {BASELINE}")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return save_figure(fig, output_dir, "fig5_extreme_subset_skill", formats, dpi)


# ===================================================================
# Figure 6 & 7 — Event case studies (Cropped PNG Embeddings)
# ===================================================================

def event_glob_ids(event_id: str) -> list[str]:
    if event_id.startswith("bangladesh_pr_202206"):
        return [event_id, "bangladesh_pr_202206*"]
    return [event_id]


def event_plot_sort_key(path: Path) -> tuple[int, str]:
    text = path.name
    match = re.search(r"lead(\d+)", text)
    lead = int(match.group(1)) if match else 99
    lead_order = {4: 0, 3: 1, 2: 2, 1: 3}.get(lead, 9)
    return lead_order, text


def resolve_recorded_plot_path(path_text: object, base_dir: Path) -> Path | None:
    if path_text is None or (isinstance(path_text, float) and math.isnan(path_text)):
        return None
    path = Path(str(path_text))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(base_dir / path)
        candidates.append(base_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def plot_index_candidates(
    plot_index: pd.DataFrame | None,
    event_id: str,
    plot_types: list[str],
    base_dir: Path,
) -> list[Path]:
    if plot_index is None or plot_index.empty or not {"event_id", "plot_type", "path"} <= set(plot_index.columns):
        return []
    event_ids = plot_index["event_id"].astype(str)
    event_mask = event_ids.eq(event_id)
    if not event_mask.any() and event_id.startswith("bangladesh_pr_202206"):
        event_mask = event_ids.str.startswith("bangladesh_pr_202206")
    subset = plot_index[event_mask & plot_index["plot_type"].astype(str).isin(plot_types)]
    paths = []
    for value in subset["path"]:
        path = resolve_recorded_plot_path(value, base_dir)
        if path is not None:
            paths.append(path)
    return sorted(paths, key=event_plot_sort_key)


def first_event_plot(
    event_id: str,
    plot_type: str,
    base_dir: Path,
    plot_index: pd.DataFrame | None,
    glob_dir: Path,
    glob_suffix: str,
) -> Path | None:
    candidates = plot_index_candidates(plot_index, event_id, [plot_type], base_dir)
    for glob_id in event_glob_ids(event_id):
        candidates.extend(Path(path) for path in glob.glob(str(glob_dir / f"{glob_id}{glob_suffix}")))
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=event_plot_sort_key)[0]


def find_primary_event_plots(
    event_id: str,
    event_dir: Path,
    quantile_dir: Path,
    event_plot_index: pd.DataFrame | None,
    quantile_plot_index: pd.DataFrame | None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    focus = first_event_plot(
        event_id,
        "spatial_event_focus",
        event_dir,
        event_plot_index,
        event_dir / "plots" / "spatial_maps",
        "_lead*_spatial_event_focus.png",
    )
    risk = first_event_plot(
        event_id,
        "spatial_risk",
        event_dir,
        event_plot_index,
        event_dir / "plots" / "spatial_maps",
        "_lead*_spatial_risk.png",
    )
    verification = first_event_plot(
        event_id,
        "spatial_verification",
        event_dir,
        event_plot_index,
        event_dir / "plots" / "spatial_maps",
        "_lead*_spatial_verification.png",
    )
    quantile = first_event_plot(
        event_id,
        "quantile_spatial",
        quantile_dir,
        quantile_plot_index,
        quantile_dir / "plots" / "quantile_spatial_maps",
        "_*quantile_spatial.png",
    )
    if risk is not None:
        paths["spatial_risk"] = risk
    if focus is not None:
        paths["spatial_event_focus"] = focus
    if verification is not None:
        paths["spatial_verification"] = verification
    if quantile is not None:
        paths["quantile_spatial"] = quantile
    return paths


def crop_event_panel(image: np.ndarray, nrows: int, ncols: int, panel_index: int) -> np.ndarray:
    image = np.asarray(image)
    h, w = image.shape[:2]
    top = int(round(h * 0.080))
    bottom = int(round(h * 0.018))
    left = int(round(w * 0.020))
    right = int(round(w * 0.012))
    usable_w = max(1, w - left - right)
    usable_h = max(1, h - top - bottom)
    row = int(panel_index) // ncols
    col = int(panel_index) % ncols
    cell_w = usable_w / float(ncols)
    cell_h = usable_h / float(nrows)
    x0 = int(round(left + col * cell_w + 0.020 * cell_w))
    x1 = int(round(left + (col + 1) * cell_w - 0.075 * cell_w))
    y0 = int(round(top + row * cell_h + 0.060 * cell_h))
    y1 = int(round(top + (row + 1) * cell_h - 0.180 * cell_h))
    x0, x1 = max(0, x0), min(w, max(x0 + 1, x1))
    y0, y1 = max(0, y0), min(h, max(y0 + 1, y1))
    return image[y0:y1, x0:x1]


def plot_cropped_event_panel(ax: plt.Axes, panel: np.ndarray, title: str) -> None:
    ax.imshow(panel)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color("#c8d1d9")
    ax.text(
        0.015,
        0.965,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=TEXT_DARK,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "pad": 1.8},
    )


def plot_small_missing_panel(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.03,
        0.92,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=TEXT_DARK,
    )
    ax.text(
        0.50,
        0.48,
        wrap(message, 32),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        color=TEXT_MUTED,
    )


def mean_column(rows: pd.DataFrame, candidates: list[str]) -> tuple[str | None, float]:
    for col in candidates:
        if col in rows:
            values = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                return col, float(np.nanmean(values))
    return None, float("nan")


def signed_metric_text(value: float, digits: int = 2, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "pending"
    return f"{value:+.{digits}f}{suffix}"


def unique_pair_count(rows: pd.DataFrame) -> int | None:
    if rows.empty:
        return None
    pair_cols = [col for col in ["init_time", "lead"] if col in rows]
    if pair_cols:
        return int(len(rows.drop_duplicates(pair_cols)))
    return int(len(rows))


def compact_pair_text(rows: pd.DataFrame) -> str:
    if rows.empty or not {"init_time", "lead"} <= set(rows.columns):
        return "pending"
    pairs = []
    for _, row in rows.drop_duplicates(["init_time", "lead"]).head(3).iterrows():
        try:
            init = pd.Timestamp(row["init_time"]).strftime("%Y-%m-%d")
        except Exception:
            init = str(row["init_time"])
        try:
            lead = int(row["lead"])
            pairs.append(f"{init} W{lead}")
        except Exception:
            pairs.append(init)
    more = len(rows.drop_duplicates(["init_time", "lead"])) - len(pairs)
    if more > 0:
        pairs.append(f"+{more} more")
    return ", ".join(pairs) if pairs else "pending"


def draw_summary_value(ax: plt.Axes, x: float, y: float, label: str, value: str, color: str = TEXT_DARK) -> None:
    ax.text(x, y + 0.12, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.9, color=TEXT_MUTED)
    ax.text(x, y - 0.05, value, transform=ax.transAxes, ha="left", va="center", fontsize=8.2,
            fontweight="bold", color=color)


def event_rows_for_id(event_df: pd.DataFrame | None, event_id: str) -> pd.DataFrame:
    if event_df is None or event_df.empty or "event_id" not in event_df:
        return pd.DataFrame()
    rows = event_df[event_df["event_id"].astype(str).eq(event_id)].copy()
    if rows.empty and event_id.startswith("bangladesh_pr_202206"):
        rows = event_df[event_df["event_id"].astype(str).str.startswith("bangladesh_pr_202206")].copy()
    if rows.empty:
        return rows
    sort_cols = [col for col in ["init_time", "lead", "valid_time"] if col in rows]
    if sort_cols:
        rows = rows.sort_values(sort_cols)
    return rows.reset_index(drop=True)


def quantile_rows_for_id(quantile_df: pd.DataFrame | None, event_id: str) -> pd.DataFrame:
    if quantile_df is None or quantile_df.empty or "event_id" not in quantile_df:
        return pd.DataFrame()
    rows = quantile_df[quantile_df["event_id"].astype(str).eq(event_id)].copy()
    if rows.empty and event_id.startswith("bangladesh_pr_202206"):
        rows = quantile_df[quantile_df["event_id"].astype(str).str.startswith("bangladesh_pr_202206")].copy()
    if rows.empty:
        return rows
    sort_cols = [col for col in ["sample_kind", "init_time", "lead", "metric"] if col in rows]
    if sort_cols:
        rows = rows.sort_values(sort_cols)
    return rows.reset_index(drop=True)


def plot_primary_event_summary_strip(
    ax: plt.Axes,
    event_id: str,
    event_df: pd.DataFrame | None,
    quantile_df: pd.DataFrame | None,
    image_path: Path | None,
) -> None:
    event_rows = event_rows_for_id(event_df, event_id)
    quantile_rows = quantile_rows_for_id(quantile_df, event_id)
    context = PRIMARY_EVENT_CONTEXT.get(event_id, {})
    ax.set_axis_off()
    box = FancyBboxPatch(
        (0.002, 0.04),
        0.996,
        0.88,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="#f7f9fb",
        edgecolor="#c8d1d9",
        linewidth=0.8,
    )
    ax.add_patch(box)

    _, crps_value = mean_column(event_rows, ["crps_on_obs_extreme_skill_pct", "crps_skill_pct"])
    _, raw_bss_value = mean_column(event_rows, ["bss_diff"])
    _, bss_value = mean_column(event_rows, ["calibrated_bss_diff", "bss_diff"])
    _, raw_prob_value = mean_column(event_rows, ["event_probability_on_obs_extreme_diff", "event_probability_top_tail_diff"])
    _, neighborhood_prob_value = mean_column(event_rows, [
        "event_probability_neighborhood_on_obs_extreme_diff",
        "event_probability_neighborhood_top_tail_diff",
        "event_probability_neighborhood_diff",
    ])
    _, calibrated_prob_value = mean_column(event_rows, [
        "cal_event_probability_on_obs_extreme_diff",
        "event_probability_calibrated_top_tail_diff",
        "event_probability_calibrated_diff",
    ])
    _, q95_skill = mean_column(quantile_rows, ["q95_error_skill"])
    _, qprob_value = mean_column(quantile_rows, ["prob_threshold_or_more_diff_model_minus_geos"])
    _, percentile_value = mean_column(quantile_rows, ["obs_percentile_diff_model_minus_geos"])

    n_pairs = unique_pair_count(event_rows)
    if n_pairs is None:
        n_pairs = unique_pair_count(quantile_rows)

    source_status = "spatial maps loaded" if image_path is not None else "spatial maps pending"
    metrics_status = "metrics loaded" if not event_rows.empty or not quantile_rows.empty else "metrics pending"
    if not event_rows.empty:
        row0 = event_rows.iloc[0]
    elif not quantile_rows.empty:
        row0 = quantile_rows.iloc[0]
    else:
        row0 = {}
    event_name = clean_label(row0.get("event_name", PRIMARY_EVENT_LABELS.get(event_id, event_id)))
    variable = VARIABLE_SHORT.get(str(row0.get("variable", "")).lower(), "")
    variable_text = f"{variable} | " if variable else ""
    pair_text = compact_pair_text(event_rows if not event_rows.empty else quantile_rows)
    ax.text(
        0.018,
        0.72,
        event_name,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=TEXT_DARK,
    )
    ax.text(
        0.018,
        0.39,
        (
            f"{variable_text}{context.get('diagnostic', 'tail-risk diagnostic')} | "
            f"{context.get('event_window', 'event window pending')} | "
            f"{context.get('domain', 'domain pending')}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.6,
        color=TEXT_MUTED,
    )
    ax.text(
        0.018,
        0.16,
        f"{source_status}; {metrics_status}. Init/lead: {pair_text}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.1,
        color=TEXT_MUTED,
    )
    if event_rows.empty and quantile_rows.empty:
        ax.text(
            0.42,
            0.62,
            "Metric summary pending",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=TEXT_DARK,
        )
        ax.text(
            0.42,
            0.31,
            "Run the catalog and quantile event evaluators to fill CRPS, BSS, probability, and quantile summaries.",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.3,
            color=TEXT_MUTED,
        )
        return
    x0 = 0.42
    x1 = 0.61
    x2 = 0.80
    if not event_rows.empty:
        draw_summary_value(ax, x0, 0.66, "event-mask CRPS skill", signed_metric_text(crps_value, 1, "%"),
                           COLOR_POS if crps_value >= 0 else COLOR_NEG)
        draw_summary_value(ax, x1, 0.66, "BSS gain", signed_metric_text(raw_bss_value, 3),
                           COLOR_POS if raw_bss_value >= 0 else COLOR_NEG)
        draw_summary_value(ax, x2, 0.66, "cal. BSS gain", signed_metric_text(bss_value, 3),
                           COLOR_POS if bss_value >= 0 else COLOR_NEG)
        draw_summary_value(ax, x0, 0.27, "event-prob gain", signed_metric_text(raw_prob_value, 3),
                           COLOR_POS if raw_prob_value >= 0 else COLOR_NEG)
        draw_summary_value(ax, x1, 0.27, "neighborhood-prob gain", signed_metric_text(neighborhood_prob_value, 3),
                           COLOR_POS if neighborhood_prob_value >= 0 else COLOR_NEG)
        draw_summary_value(ax, x2, 0.27, "cal. event-prob gain", signed_metric_text(calibrated_prob_value, 3),
                           COLOR_POS if calibrated_prob_value >= 0 else COLOR_NEG)
    else:
        draw_summary_value(ax, x0, 0.66, "q95 error skill", signed_metric_text(q95_skill, 2),
                           COLOR_POS if q95_skill >= 0 else COLOR_NEG)
        draw_summary_value(ax, x1, 0.66, "P(threshold) gain", signed_metric_text(qprob_value, 3),
                           COLOR_POS if qprob_value >= 0 else COLOR_NEG)
        draw_summary_value(ax, x2, 0.66, "obs-percentile gain", signed_metric_text(percentile_value, 3),
                           COLOR_POS if percentile_value >= 0 else COLOR_NEG)
    if n_pairs is not None:
        ax.text(0.985, 0.18, f"n={n_pairs}", transform=ax.transAxes, ha="right", va="center",
                fontsize=7.3, color=TEXT_MUTED)


def first_spatial_path(plot_paths: dict[str, Path]) -> Path | None:
    for key in ("spatial_event_focus", "spatial_risk", "spatial_verification", "quantile_spatial"):
        if key in plot_paths:
            return plot_paths[key]
    return None


def plot_contoured_panel(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: str = "both",
):
    from matplotlib.colors import TwoSlopeNorm
    finite = field[np.isfinite(field)]
    if not finite.size:
        missing_panel(ax, title, "All values are NaN.")
        return None
    
    if vmin is None:
    	vmin = float(np.nanpercentile(finite, 2))
    if vmax is None:
    	vmax = float(np.nanpercentile(finite, 98))
        
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    if norm is None and vmin < 0 and vmax > 0 and ("diff" in title.lower() or "gain" in title.lower() or "closeness" in title.lower() or "improvement" in title.lower()):
        vlim = max(abs(vmin), abs(vmax))
        vlim = max(vlim, 0.01)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)
        levels = np.linspace(-vlim, vlim, 21)
    elif norm is None:
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        levels = np.linspace(vmin, vmax, 21)
    else:
        levels = np.linspace(norm.vmin, norm.vmax, 21)

    mesh = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, norm=norm, extend=extend, transform=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#222222", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#444444", linestyle=":", zorder=2)
    
    if np.nanmin(lons) > -135 and np.nanmax(lons) < -110:
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#777777", linestyle=":", zorder=2)

    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    
    ax.set_extent([float(np.nanmin(lons)), float(np.nanmax(lons)), float(np.nanmin(lats)), float(np.nanmax(lats))], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, color="#d9dee3", linewidth=0.25, alpha=0.7)
    return mesh


def plot_contoured_panel_with_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    field: np.ndarray,
    title: str,
    cmap: str,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    is_change: bool = False,
    extend: str = "both",
):
    im = plot_contoured_panel(ax, lons, lats, field, title, cmap, norm=norm, vmin=vmin, vmax=vmax, extend=extend)
    if im is not None:
        from matplotlib import ticker
        cbar_extend = "neither" if ("bss" in title.lower() and not "improvement" in title.lower()) else extend
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.03, spacing="proportional", extend=cbar_extend)
        cbar.ax.tick_params(labelsize=6)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=7)
        if is_change:
            cbar.locator = ticker.MaxNLocator(nbins=6, symmetric=True)
        else:
            cbar.locator = ticker.MaxNLocator(nbins=6)
        cbar.update_ticks()
    return im


def figure_event_cropped(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    event_dir: Path,
    quantile_dir: Path,
    event_df: pd.DataFrame | None,
    quantile_df: pd.DataFrame | None,
    event_id: str,
    fig_num: int,
    stem: str,
    fig_title: str,
    fig_subtitle: str,
) -> list[Path]:
    """Figure 6 & 7: 3x3 grid of contoured panels from NetCDF data, or cropped PNG fallback."""
    nc_path = event_dir / "plots" / "spatial_maps" / f"{event_id}_lead4_spatial_data.nc"
    plot_paths = find_primary_event_plots(event_id, event_dir, quantile_dir, None, None)
    image_path = first_spatial_path(plot_paths)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Set figure size based on event to remove empty margins
    is_t2m = "t2m" in event_id.lower()
    if is_t2m:
        fig = plt.figure(figsize=(15.0, 9.8))
    else:
        fig = plt.figure(figsize=(14.0, 11.5))
        
    fig.patch.set_facecolor("white")
    style_figure(fig, fig_title, fig_subtitle)

    # 3x3 grid of maps filling the entire figure
    gs = fig.add_gridspec(3, 3, hspace=0.18, wspace=0.18, left=0.03, right=0.97, bottom=0.03, top=0.97)

    field_cmap = "RdYlBu_r" if is_t2m else "YlGnBu"
    bss_cmap = "viridis"

    if nc_path.exists():
        try:
            import xarray as xr
            ds = xr.open_dataset(nc_path)
            lons = ds["lon"].values
            lats = ds["lat"].values
            
            obs = ds["obs_plot"].values + (273.15 if is_t2m else 0.0)
            geos_q95 = ds["geos_upper_quantile"].values
            model_q95 = ds["model_upper_quantile"].values
            
            geos_crps = ds["geos_crps"].values
            model_crps = ds["model_crps"].values
            
            geos_bss = ds["geos_bss"].values
            model_bss = ds["model_bss"].values
            ds.close()

            # Clean up ocean points using obs mask to ensure perfectly white oceans on all panels
            ocean_mask = np.isnan(obs)
            geos_q95 = np.where(ocean_mask, np.nan, geos_q95)
            model_q95 = np.where(ocean_mask, np.nan, model_q95)
            geos_crps = np.where(ocean_mask, np.nan, geos_crps)
            model_crps = np.where(ocean_mask, np.nan, model_crps)
            geos_bss = np.where(ocean_mask, np.nan, geos_bss)
            model_bss = np.where(ocean_mask, np.nan, model_bss)

            # Compute percentage CRPS skill improvement instead of raw diff
            denom = np.where(geos_crps == 0.0, 1e-5, geos_crps)
            crps_diff = 100.0 * (1.0 - model_crps / denom)
            crps_diff = np.where(ocean_mask, np.nan, crps_diff)

            # Compute percentage BSS skill improvement and mask out values outside [-30%, 30%]
            bss_diff_pct = 100.0 * (model_bss - geos_bss)
            bss_diff = np.where((bss_diff_pct >= -30.0) & (bss_diff_pct <= 30.0), bss_diff_pct, np.nan)
            bss_diff = np.where(ocean_mask, np.nan, bss_diff)

            # Shared limits for the top row (obs & q95)
            field_vals = np.concatenate([obs.ravel(), geos_q95.ravel(), model_q95.ravel()])
            finite_field = field_vals[np.isfinite(field_vals)]
            fvmin = float(np.nanpercentile(finite_field, 2)) if finite_field.size else 0.0
            fvmax = float(np.nanpercentile(finite_field, 98)) if finite_field.size else 1.0

            # Shared limits for the middle row (CRPS)
            crps_vals = np.concatenate([geos_crps.ravel(), model_crps.ravel()])
            finite_crps = crps_vals[np.isfinite(crps_vals)]
            cvmin = 0.0
            cvmax = float(np.nanpercentile(finite_crps, 98)) if finite_crps.size else 1.0

            # Diverging limits for CRPS percentage improvement
            finite_diff = crps_diff[np.isfinite(crps_diff)]
            fd_lim = float(np.nanpercentile(np.abs(finite_diff), 95)) if finite_diff.size else 30.0
            fd_lim = min(max(fd_lim, 10.0), 100.0)
            fd_vmin, fd_vmax = -fd_lim, fd_lim

            # Shared limits for the bottom row (BSS maps)
            bvmin = 0.0
            bvmax = 0.6

            # Labels for colorbars
            phys_label = "Temperature (K)" if is_t2m else "Precipitation (mm/day)"

            # Plot top row (Observed, Baseline q95, ML q95)
            ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_a, lons, lats, obs, "(a) Observed", field_cmap, vmin=fvmin, vmax=fvmax, cbar_label=phys_label, extend="both")
            
            ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_b, lons, lats, geos_q95, f"(b) {BASELINE} q95", field_cmap, vmin=fvmin, vmax=fvmax, cbar_label=phys_label, extend="both")
            
            ax_c = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_c, lons, lats, model_q95, f"(c) {METHOD} q95", field_cmap, vmin=fvmin, vmax=fvmax, cbar_label=phys_label, extend="both")

            # Plot middle row (Baseline CRPS, ML CRPS, CRPS Skill Improvement %)
            ax_d = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_d, lons, lats, geos_crps, f"(d) {BASELINE} CRPS", "Purples", vmin=cvmin, vmax=cvmax, cbar_label="CRPS", extend="max")
            
            ax_e = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_e, lons, lats, model_crps, f"(e) {METHOD} CRPS", "Purples", vmin=cvmin, vmax=cvmax, cbar_label="CRPS", extend="max")
            
            ax_f = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_f, lons, lats, crps_diff, "(f) CRPS Skill Improvement (%)", "RdYlGn", vmin=fd_vmin, vmax=fd_vmax, cbar_label="Improvement (%)", is_change=True, extend="both")
            # Subtle scatter stippling where ML CRPS improvement is robust (> 10%)
            lon2d, lat2d = np.meshgrid(lons, lats)
            crps_sig = crps_diff > 10.0
            # Thin the mask to plot clean grid stippling (every 2nd point)
            thin_mask = np.zeros_like(crps_sig, dtype=bool)
            thin_mask[::2, ::2] = True
            crps_sig_plot = crps_sig & thin_mask
            if crps_sig_plot.any():
                ax_f.scatter(lon2d[crps_sig_plot], lat2d[crps_sig_plot], color="black", s=3.0, alpha=0.7, marker="o", edgecolors="none", transform=ccrs.PlateCarree())

            # Plot bottom row (Baseline BSS, ML BSS, BSS Skill Improvement %)
            ax_g = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_g, lons, lats, geos_bss, f"(g) {BASELINE} BSS", "Blues", vmin=bvmin, vmax=bvmax, cbar_label="BSS", extend="both")
            
            ax_h = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_h, lons, lats, model_bss, f"(h) {METHOD} BSS", "Blues", vmin=bvmin, vmax=bvmax, cbar_label="BSS", extend="both")
            
            ax_i = fig.add_subplot(gs[2, 2], projection=ccrs.PlateCarree())
            plot_contoured_panel_with_colorbar(fig, ax_i, lons, lats, bss_diff, "(i) BSS Skill Improvement (%)", "RdBu", vmin=-30.0, vmax=30.0, cbar_label="Improvement (%)", is_change=True, extend="both")
            # Subtle scatter stippling where ML BSS improvement is robust (> 5%)
            bss_sig = bss_diff > 5.0
            bss_sig_plot = bss_sig & thin_mask
            if bss_sig_plot.any():
                ax_i.scatter(lon2d[bss_sig_plot], lat2d[bss_sig_plot], color="black", s=3.0, alpha=0.7, marker="o", edgecolors="none", transform=ccrs.PlateCarree())

        except Exception as exc:
            ax_big = fig.add_subplot(gs[:, :])
            missing_panel(ax_big, "Error loading NetCDF event data", f"Could not load data from {nc_path.name}: {exc}")
    else:
        ax_big = fig.add_subplot(gs[:, :])
        missing_panel(
            ax_big,
            "Spatial maps pending",
            (
                f"NetCDF data {nc_path.name} not found.\n"
                f"Please run paper/scripts/make_contoured_event_plots.py first."
            ),
        )

    return save_figure(fig, output_dir, stem, formats, dpi)


def figure_6_event_pr(
    output_dir: Path, formats: list[str], dpi: int,
    event_dir: Path, quantile_dir: Path,
    event_df: pd.DataFrame | None, quantile_df: pd.DataFrame | None,
) -> list[Path]:
    """California atmospheric river flood — PR case study."""
    return figure_event_cropped(
        output_dir, formats, dpi, event_dir, quantile_dir, event_df, quantile_df,
        event_id="conus_pr_202301_california_atmospheric_rivers",
        fig_num=6,
        stem="fig6_event_pr_california",
        fig_title="Figure 6. California atmospheric river flood — precipitation",
        fig_subtitle=(
            f"Observed PR, {BASELINE} and {METHOD} ensemble features, exceedance probabilities, "
            f"and gain map. Positive gain = ML forecast closer to observations."
        ),
    )


def figure_7_event_t2m(
    output_dir: Path, formats: list[str], dpi: int,
    event_dir: Path, quantile_dir: Path,
    event_df: pd.DataFrame | None, quantile_df: pd.DataFrame | None,
) -> list[Path]:
    """UK July 2022 heatwave — T2M case study."""
    return figure_event_cropped(
        output_dir, formats, dpi, event_dir, quantile_dir, event_df, quantile_df,
        event_id="europe_t2m_202207_uk_heatwave",
        fig_num=7,
        stem="fig7_event_t2m_uk_heatwave",
        fig_title="Figure 7. UK heatwave July 2022 — 2 m temperature",
        fig_subtitle=(
            f"Observed T2M, {BASELINE} and {METHOD} ensemble features, exceedance probabilities, "
            f"and gain map. Positive gain = ML forecast closer to observations."
        ),
    )


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    formats = output_formats(args.format)
    output_dir = Path(args.output_dir)
    matrix_dir = first_existing_dir(args.matrix_dir, DEFAULT_MATRIX_DIR_CANDIDATES)
    event_dir = first_existing_dir(args.event_dir, DEFAULT_EVENT_DIR_CANDIDATES)
    quantile_dir = first_existing_dir(args.quantile_dir, DEFAULT_QUANTILE_DIR_CANDIDATES)

    print("Figure input locations")
    print(f"  matrix_dir     : {matrix_dir}")
    print(f"  event_dir      : {event_dir}")
    print(f"  quantile_dir   : {quantile_dir}")
    print(f"  output_dir     : {output_dir}")

    summary = read_csv_or_none(matrix_summary_path(matrix_dir))
    event_df = read_csv_or_none(event_dir / "event_selected_lead_metrics.csv")
    quantile_df = read_csv_or_none(quantile_dir / "event_quantile_selected_comparison.csv")

    written: list[Path] = []

    # Fig 1 — Combined framework, architecture, and sampling schematic
    written.extend(figure_1_framework_overview(output_dir, formats, args.dpi))

    # Fig 2 — Global PR skill (3x2)
    written.extend(figure_3_pr_skill(
        output_dir, formats, args.dpi, summary, matrix_dir,
        args.matrix_subset, args.spatial_subset,
    ))

    # Fig 3 — Global T2M skill (3x2)
    written.extend(figure_4_t2m_skill(
        output_dir, formats, args.dpi, summary, matrix_dir,
        args.matrix_subset, args.spatial_subset,
    ))

    # Fig 4 — Extreme-event subset skill (2x2)
    written.extend(figure_5_extreme_subset(output_dir, formats, args.dpi, summary))

    # Fig 5 — California AR flood case study (2x3 cropped panels)
    written.extend(figure_6_event_pr(output_dir, formats, args.dpi, event_dir, quantile_dir, event_df, quantile_df))

    # Fig 6 — UK heat event case study (2x3 cropped panels)
    written.extend(figure_7_event_t2m(output_dir, formats, args.dpi, event_dir, quantile_dir, event_df, quantile_df))

    print(f"\nWrote {len(written)} figure files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
