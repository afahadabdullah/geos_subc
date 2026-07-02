#!/usr/bin/env python3
"""Build manuscript figures from FIMr1p1-FlowMatch evaluation products.

This script is intentionally downstream of the expensive evaluators. It reads
the CSV/NetCDF products written by:

  - ml_model/evaluate_matrix_suite_flow_finalv1_global.py
  - ml_model/compare_noise_flow_finalv1_global.py
  - ml_model/evaluate_event_catalog_flow_finalv1_global.py

When an expected evaluation artifact is missing, the relevant panel is rendered
as a clear missing-data note so the full figure set can still be regenerated
while final runs are pending.

Figure layout (approved):
  1  System overview schematic
  2  Architecture and sampling detail schematic
  3  Global PR skill: lead bars + season-lead heatmaps + spatial maps (3x2)
  4  Global T2M skill: same structure as Fig 3 (3x2)
  5  Extreme-event subset: PR + T2M CRPS/RMSE skill bars (2x2)
  6  California AR flood PR case study (2x3)
  7  UK heat event T2M case study (2x3)
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
    parser.add_argument("--quantile-dir", default=None, help="Directory containing event quantile NetCDF products.")
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
    parser.add_argument("--event-limit", type=int, default=8, help="Maximum event rows shown in Figure 5.")
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
    fig.suptitle(title, x=0.015, y=0.995, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT_DARK)
    if subtitle:
        fig.text(0.015, 0.962, subtitle, ha="left", va="top", fontsize=9.5, color=TEXT_MUTED)


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


def plot_skill_heatmap(ax: plt.Axes, arr: np.ndarray | None, title: str, vmin: float, vmax: float):
    if arr is None or not np.isfinite(arr).any():
        missing_panel(ax, title, "Missing valid_season_lead matrix rows.")
        return None
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(arr, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(LEADS)), [str(lead) for lead in LEADS], fontsize=8)
    ax.set_yticks(np.arange(len(SEASONS)), SEASONS, fontsize=8)
    ax.set_xlabel("Lead week", fontsize=8.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=7, color="#101820")
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
):
    if lons is None or lats is None or field is None or not np.isfinite(field).any():
        missing_panel(ax, title, "Missing matrix_spatial_metrics.nc data for this map.")
        return None
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(lons, lats, field, shading="auto", cmap="RdYlGn", norm=norm)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    ax.grid(True, color="#d9dee3", linewidth=0.25, alpha=0.7)
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
# Figure 1 — System overview (schematic, unchanged)
# ===================================================================

def figure_1_system_overview(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    style_figure(
        fig,
        "Figure 1. Forecast-system overview",
        "Dynamical guidance, observed context, stochastic flow sampling, and verification outputs.",
    )
    ax.set_axis_off()

    draw_box(ax, (0.03, 0.66), (0.22, 0.18), "Recent observed context",
             "SST, SSS, soil moisture, IVT, Z500 zonal deviation, U250, MJO wave; four pre-init weeks.",
             "#eaf3f8")
    draw_box(ax, (0.03, 0.42), (0.22, 0.17), f"{BASELINE} ensemble guidance",
             "PR and T2M weekly lead summaries: mean, spread, q10, q90, member count.",
             "#f8ece6")
    draw_box(ax, (0.03, 0.20), (0.22, 0.15), "Static + calendar",
             "Elevation, land mask, coordinates, season phase, and lead encoding.",
             "#eef4ea")

    draw_box(ax, (0.34, 0.57), (0.23, 0.20), "Conditioning builder",
             "Local target-grid channels plus global context tokens for large-scale circulation.",
             "#f7f7f7")
    draw_box(ax, (0.34, 0.28), (0.23, 0.20), "Flow-matching model",
             "Shared U-Net backbone with global-token cross-attention and PR/T2M lead-specific heads.",
             "#edf1fb")
    draw_box(ax, (0.34, 0.07), (0.23, 0.13), "Structured prior",
             "EOF-conditioned perturbations blended with random noise.",
             "#f3eefe")

    draw_box(ax, (0.66, 0.51), (0.23, 0.20), "Probabilistic sampler",
             "Euler ODE integration from x0 to x1 with variance-tempered spread control.",
             "#fff6db")
    draw_box(ax, (0.66, 0.26), (0.23, 0.16), "Weekly ensemble forecast",
             "90-member PR and T2M forecasts for lead weeks 1-4.",
             "#eaf5ef")
    draw_box(ax, (0.66, 0.06), (0.23, 0.13), "Verification products",
             "CRPS, RMSE, BSS, calibration, spatial maps, and event tail-risk metrics.",
             "#f7eeee")

    for y in (0.75, 0.505, 0.275):
        add_arrow(ax, (0.25, y), (0.34, 0.66 if y > 0.6 else 0.40))
    add_arrow(ax, (0.57, 0.67), (0.66, 0.61))
    add_arrow(ax, (0.57, 0.38), (0.66, 0.58))
    add_arrow(ax, (0.57, 0.14), (0.66, 0.54), color="#8a69a6")
    add_arrow(ax, (0.775, 0.51), (0.775, 0.42))
    add_arrow(ax, (0.775, 0.26), (0.775, 0.19))

    ax.text(0.08, 0.88, "Inputs", transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT_MUTED)
    ax.text(0.39, 0.88, "Conditional model", transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT_MUTED)
    ax.text(0.70, 0.88, "Outputs", transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT_MUTED)
    return save_figure(fig, output_dir, "fig1_system_overview", formats, dpi)


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
# New panel helpers
# ===================================================================

def plot_skill_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    variable: str,
    metric: str,
    color: str | None = None,
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
    ax.set_title(
        f"{VARIABLE_SHORT.get(variable, variable)} {metric.upper()} skill vs {BASELINE}",
        loc="left", fontsize=10, fontweight="bold",
    )
    ax.set_ylim(bottom=0)


def plot_closeness_map(
    ax: plt.Axes,
    lons: np.ndarray | None,
    lats: np.ndarray | None,
    obs: np.ndarray | None,
    baseline_mean: np.ndarray | None,
    model_mean: np.ndarray | None,
    title: str,
) -> None:
    """Plot |obs - baseline| - |obs - model|. Positive = ML closer to reality."""
    if any(arr is None for arr in (lons, lats, obs, baseline_mean, model_mean)):
        missing_panel(ax, title, "Missing event spatial data for closeness map.")
        return None
    closeness = np.abs(obs - baseline_mean) - np.abs(obs - model_mean)
    if not np.isfinite(closeness).any():
        missing_panel(ax, title, "All closeness values are NaN.")
        return None
    finite = closeness[np.isfinite(closeness)]
    lim = max(float(np.nanpercentile(np.abs(finite), 95)), 0.1)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    mesh = ax.pcolormesh(lons, lats, closeness, shading="auto", cmap="RdYlGn", norm=norm)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    ax.grid(True, color="#d9dee3", linewidth=0.25, alpha=0.7)
    return mesh


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
    """Shared builder for the per-variable multi-panel skill figure (3x2).

    Row 1: Lead-dependent skill bars (CRPS, RMSE)
    Row 2: Season-lead skill heatmaps (CRPS, RMSE)
    Row 3: Spatial skill maps (CRPS, RMSE)
    """
    var_label = VARIABLE_LABELS.get(variable, variable)
    var_short = VARIABLE_SHORT.get(variable, variable.upper())

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.5))
    style_figure(
        fig,
        f"Figure {fig_num}. {var_label} forecast skill vs {BASELINE}",
        f"Lead bars (top), season-lead matrices (middle), and spatial maps (bottom). Positive = ML improves on {BASELINE}.",
    )

    # --- Row 1: Lead-dependent skill bars ---
    agg = aggregate_matrix_by_lead(summary, subset=subset)
    plot_skill_bars(axes[0, 0], agg, variable, "crps")
    plot_skill_bars(axes[0, 1], agg, variable, "rmse")

    # --- Row 2: Season-lead heatmaps ---
    crps_arr = season_lead_values(summary, variable, "crps_skill_pct", subset)
    rmse_arr = season_lead_values(summary, variable, "rmse_skill_pct", subset)
    vmin, vmax = heatmap_limits([crps_arr, rmse_arr])
    im = None
    maybe_im = plot_skill_heatmap(axes[1, 0], crps_arr, f"{var_short} CRPS skill by season & lead", vmin, vmax)
    if maybe_im is not None:
        im = maybe_im
    maybe_im = plot_skill_heatmap(axes[1, 1], rmse_arr, f"{var_short} RMSE skill by season & lead", vmin, vmax)
    if maybe_im is not None:
        im = maybe_im

    # --- Row 3: Spatial maps ---
    ds = load_xarray_dataset(matrix_spatial_path(matrix_dir))
    map_crps = spatial_metric_map(ds, variable, "crps_skill_pct", spatial_subset)
    map_rmse = spatial_metric_map(ds, variable, "rmse_skill_pct", spatial_subset)
    fields = [map_crps[2], map_rmse[2]]
    svmin, svmax = spatial_limits(fields)
    mesh = None
    maybe_mesh = plot_plain_map(axes[2, 0], *map_crps, f"{var_short} CRPS skill map", svmin, svmax)
    if maybe_mesh is not None:
        mesh = maybe_mesh
    maybe_mesh = plot_plain_map(axes[2, 1], *map_rmse, f"{var_short} RMSE skill map", svmin, svmax)
    if maybe_mesh is not None:
        mesh = maybe_mesh

    if ds is not None:
        ds.close()

    # Colorbars
    if im is not None:
        cbar_hm = fig.colorbar(im, ax=axes[1, :].ravel().tolist(), shrink=0.82, pad=0.03)
        cbar_hm.set_label("Skill vs " + BASELINE + " (%)", fontsize=8)
        cbar_hm.ax.tick_params(labelsize=7)
    if mesh is not None:
        cbar_sp = fig.colorbar(mesh, ax=axes[2, :].ravel().tolist(), shrink=0.82, pad=0.03)
        cbar_sp.set_label("Skill vs " + BASELINE + " (%)", fontsize=8)
        cbar_sp.ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 0.93, 0.93], h_pad=3.0)
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

    plot_skill_bars(axes[0, 0], agg, "pr", "crps")
    plot_skill_bars(axes[0, 1], agg, "pr", "rmse")
    plot_skill_bars(axes[1, 0], agg, "t2m", "crps")
    plot_skill_bars(axes[1, 1], agg, "t2m", "rmse")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return save_figure(fig, output_dir, "fig5_extreme_subset_skill", formats, dpi)


# ===================================================================
# Figures 6 & 7 — Event case studies (2x3)
# ===================================================================

def load_event_spatial_data(
    quantile_dir: Path,
    event_name: str,
    variable: str,
) -> dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]]:
    """Try to load event spatial products from NetCDF files in quantile_dir.

    Returns a dict mapping field names to (lons, lats, field) tuples.
    Falls back to (None, None, None) when data is unavailable.
    """
    empty = (None, None, None)
    result = {
        "observed": empty,
        "geos_mean": empty,
        "model_mean": empty,
        "geos_exceedance": empty,
        "model_exceedance": empty,
    }

    if not quantile_dir.exists():
        return result

    # Search for matching NetCDF files
    candidates = list(quantile_dir.glob(f"*{event_name}*{variable}*.nc"))
    if not candidates:
        candidates = list(quantile_dir.glob(f"*{variable}*event*.nc"))
    if not candidates:
        candidates = list(quantile_dir.glob("*.nc"))
    if not candidates:
        return result

    try:
        import xarray as xr
    except ImportError:
        return result

    for nc_path in candidates:
        try:
            ds = xr.open_dataset(nc_path)
        except Exception:
            continue

        lons = np.asarray(ds["lon"].values) if "lon" in ds else None
        lats = np.asarray(ds["lat"].values) if "lat" in ds else None
        if lons is None or lats is None:
            ds.close()
            continue

        # Map expected variable names to result keys
        field_map = {
            "observed": ["observed", "obs", f"obs_{variable}", "target"],
            "geos_mean": ["geos_mean", "geos_ensemble_mean", f"geos_{variable}_mean", "baseline_mean"],
            "model_mean": ["model_mean", "model_ensemble_mean", f"model_{variable}_mean", "ml_mean"],
            "geos_exceedance": ["geos_exceedance_prob", "geos_exceed_p95", f"geos_{variable}_exceed"],
            "model_exceedance": ["model_exceedance_prob", "model_exceed_p95", f"model_{variable}_exceed"],
        }

        for key, name_options in field_map.items():
            for name in name_options:
                if name in ds:
                    field = ds[name].squeeze(drop=True)
                    result[key] = (lons, lats, np.asarray(field.values, dtype=float))
                    break

        ds.close()
        # If we found at least an observed field, stop searching
        if result["observed"][0] is not None:
            break

    return result


def plot_field_map(
    ax: plt.Axes,
    lons: np.ndarray | None,
    lats: np.ndarray | None,
    field: np.ndarray | None,
    title: str,
    cmap: str = "YlGnBu",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a non-diverging spatial field (observed values, ensemble means, probabilities)."""
    if lons is None or lats is None or field is None or not np.isfinite(field).any():
        missing_panel(ax, title, "Missing event spatial data.")
        return None
    if vmin is None:
        vmin = float(np.nanpercentile(field[np.isfinite(field)], 2))
    if vmax is None:
        vmax = float(np.nanpercentile(field[np.isfinite(field)], 98))
    mesh = ax.pcolormesh(lons, lats, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    ax.grid(True, color="#d9dee3", linewidth=0.25, alpha=0.7)
    return mesh


def figure_event_case_study(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    quantile_dir: Path,
    event_name: str,
    variable: str,
    fig_num: int,
    stem: str,
    fig_title: str,
    fig_subtitle: str,
    field_cmap: str = "YlGnBu",
    prob_cmap: str = "YlOrRd",
) -> list[Path]:
    """Shared builder for event case-study figures (2x3).

    Panel layout:
      (a) Observed field         (b) FIMr1p1 ensemble mean   (c) ML ensemble mean
      (d) FIMr1p1 exceedance     (e) ML exceedance prob      (f) Closeness map
    """
    var_short = VARIABLE_SHORT.get(variable, variable.upper())
    data = load_event_spatial_data(quantile_dir, event_name, variable)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5))
    style_figure(fig, fig_title, fig_subtitle)

    # Determine shared color limits for the field panels (obs + means)
    field_arrays = [data["observed"][2], data["geos_mean"][2], data["model_mean"][2]]
    finite_vals = []
    for arr in field_arrays:
        if arr is not None:
            finite_vals.extend(arr[np.isfinite(arr)].ravel().tolist())
    if finite_vals:
        fvmin = float(np.nanpercentile(finite_vals, 2))
        fvmax = float(np.nanpercentile(finite_vals, 98))
    else:
        fvmin, fvmax = None, None

    # Row 1: Observed, baseline mean, ML mean
    plot_field_map(axes[0, 0], *data["observed"], f"(a) Observed {var_short}", cmap=field_cmap, vmin=fvmin, vmax=fvmax)
    plot_field_map(axes[0, 1], *data["geos_mean"], f"(b) {BASELINE} ensemble mean", cmap=field_cmap, vmin=fvmin, vmax=fvmax)
    plot_field_map(axes[0, 2], *data["model_mean"], f"(c) {METHOD} ensemble mean", cmap=field_cmap, vmin=fvmin, vmax=fvmax)

    # Row 2: Exceedance probabilities and closeness
    plot_field_map(axes[1, 0], *data["geos_exceedance"], f"(d) {BASELINE} P(exceed p95)", cmap=prob_cmap, vmin=0, vmax=1)
    plot_field_map(axes[1, 1], *data["model_exceedance"], f"(e) {METHOD} P(exceed p95)", cmap=prob_cmap, vmin=0, vmax=1)

    # Closeness map
    obs_lons, obs_lats, obs_field = data["observed"]
    _, _, geos_field = data["geos_mean"]
    _, _, model_field = data["model_mean"]
    plot_closeness_map(
        axes[1, 2], obs_lons, obs_lats, obs_field, geos_field, model_field,
        f"(f) Closeness (green = ML closer)",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_figure(fig, output_dir, stem, formats, dpi)


def figure_6_event_pr(
    output_dir: Path, formats: list[str], dpi: int,
    quantile_dir: Path,
) -> list[Path]:
    """California atmospheric river flood — PR case study."""
    return figure_event_case_study(
        output_dir, formats, dpi, quantile_dir,
        event_name="california_ar",
        variable="pr",
        fig_num=6,
        stem="fig6_event_pr_california",
        fig_title="Figure 6. California atmospheric river flood — precipitation",
        fig_subtitle=(
            f"Observed PR, {BASELINE} and {METHOD} ensemble means, exceedance probabilities, "
            f"and closeness map. Positive closeness (green) = ML forecast closer to observations."
        ),
        field_cmap="YlGnBu",
        prob_cmap="YlOrRd",
    )


def figure_7_event_t2m(
    output_dir: Path, formats: list[str], dpi: int,
    quantile_dir: Path,
) -> list[Path]:
    """UK July 2022 heatwave — T2M case study."""
    return figure_event_case_study(
        output_dir, formats, dpi, quantile_dir,
        event_name="uk_heatwave",
        variable="t2m",
        fig_num=7,
        stem="fig7_event_t2m_uk_heatwave",
        fig_title="Figure 7. UK heatwave July 2022 — 2 m temperature",
        fig_subtitle=(
            f"Observed T2M, {BASELINE} and {METHOD} ensemble means, exceedance probabilities, "
            f"and closeness map. Positive closeness (green) = ML forecast closer to observations."
        ),
        field_cmap="RdYlBu_r",
        prob_cmap="YlOrRd",
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

    written: list[Path] = []

    # Fig 1 — System overview (schematic)
    written.extend(figure_1_system_overview(output_dir, formats, args.dpi))

    # Fig 2 — Architecture (schematic)
    written.extend(figure_2_architecture(output_dir, formats, args.dpi))

    # Fig 3 — Global PR skill (3x2)
    written.extend(figure_3_pr_skill(
        output_dir, formats, args.dpi, summary, matrix_dir,
        args.matrix_subset, args.spatial_subset,
    ))

    # Fig 4 — Global T2M skill (3x2)
    written.extend(figure_4_t2m_skill(
        output_dir, formats, args.dpi, summary, matrix_dir,
        args.matrix_subset, args.spatial_subset,
    ))

    # Fig 5 — Extreme-event subset skill (2x2)
    written.extend(figure_5_extreme_subset(output_dir, formats, args.dpi, summary))

    # Fig 6 — California AR flood case study (2x3)
    written.extend(figure_6_event_pr(output_dir, formats, args.dpi, quantile_dir))

    # Fig 7 — UK heat event case study (2x3)
    written.extend(figure_7_event_t2m(output_dir, formats, args.dpi, quantile_dir))

    print(f"\nWrote {len(written)} figure files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
