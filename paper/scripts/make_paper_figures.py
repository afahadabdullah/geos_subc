#!/usr/bin/env python3
"""Build publication figures for the FIMr1p1-FlowMatch manuscript.

This script is intentionally downstream of the expensive evaluators. It reads
the CSV/NetCDF products written by:

  - ml_model/evaluate_matrix_suite_flow_finalv1_global.py
  - ml_model/compare_noise_flow_finalv1_global.py
  - ml_model/evaluate_event_catalog_flow_finalv1_global.py  (+ contoured NetCDF
    products from paper/scripts/make_contoured_event_plots.py)

Missing evaluation products render as clearly labeled "pending" panels so the
full figure set always regenerates end-to-end.

Figure set (see paper/FIGURE_PLAN.md):
  1  fig1_framework_overview   Framework schematic (self-contained)
  2  fig2_pr_skill             PR season-lead heatmaps + Robinson skill maps
  3  fig3_t2m_skill            T2M, same layout
  4  fig4_noise_ablation       Gaussian vs EOF-LHS stochastic-prior ablation
  5  fig5_extreme_skill        Extreme-subset skill vs all-case skill by lead
  6  fig6_event_pr_california  California AR precipitation case study (3x3)
  7  fig7_event_t2m_uk_heatwave UK July 2022 heatwave T2M case study (3x3)

Usage:
  python paper/scripts/make_paper_figures.py --format both
"""

from __future__ import annotations

import argparse
import glob
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD = "FIMr1p1-FlowMatch"
BASELINE = "FIMr1p1"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
LEADS = [1, 2, 3, 4]
VARIABLE_LABELS = {"pr": "Precipitation", "t2m": "2 m temperature"}
VARIABLE_SHORT = {"pr": "PR", "t2m": "T2M"}

# Palette (colorblind-aware)
C_BASELINE = "#9a3b3b"   # muted red — raw dynamical baseline
C_MODEL = "#1f5fa8"      # blue — ML system
C_PR = "#1f5fa8"
C_T2M = "#c07a2b"
C_ACCENT = "#6a4c93"     # purple — stochastic pathway
TEXT_DARK = "#22303c"
TEXT_MUTED = "#5b6770"
EDGE = "#39536b"
CMAP_MAG = "viridis"     # magnitudes (CRPS, RMSE)
CMAP_SKILL = "RdBu"      # diverging: blue = improvement, red = degradation

DEFAULT_MATRIX_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_land_obsclim_chunked",
    "ml_output_flow_finalv1_global_noisectx_t2mres/paper_matrix_eval_global_2021_2023",
    "ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_e90_s50",
]
DEFAULT_EVENT_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023",
]
DEFAULT_ENSEMBLE_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_extreme_t2m30_pr30_regions_2021_2023_wk3wk4_memberboot50_caseboot15_pub",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2024_e90_s50",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2023_e90_s50",
]
NOISE_CSV_PATTERNS = [
    "ml_output_noise_compare_global_flow_finalv1/noise_comparison_global_*.csv",
    "ml_output_flow_finalv1_global_noisectx_t2mres/noise_comparison_global_*.csv",
]

# Frozen 2021 full-year noise-ablation headline values (52 batches); used only
# as a clearly annotated fallback when no noise_comparison CSV is found.
NOISE_FALLBACK = {
    "Gaussian $\\mathcal{N}(0,I)$": {"pr": 24.9, "t2m": 41.7},
    "EOF-LHS structured": {"pr": 28.4, "t2m": 43.2},
}

EVENT_PR_ID = "conus_pr_202301_california_atmospheric_rivers"
EVENT_T2M_ID = "europe_t2m_202207_uk_heatwave"


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#7a8794",
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_DARK,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": TEXT_DARK,
        "ytick.color": TEXT_DARK,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    })


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9.5, fontweight="bold", color=TEXT_DARK, pad=4)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#dde3e8", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def missing_panel(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_axis_off()
    panel_title(ax, title)
    box = FancyBboxPatch(
        (0.06, 0.16), 0.88, 0.62, transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#f4f6f8", edgecolor="#aab5c0", linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(0.5, 0.47, "\n".join(textwrap.wrap(message, 44)), transform=ax.transAxes,
            ha="center", va="center", fontsize=8.5, color=TEXT_MUTED)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# CLI / path helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript figures from existing evaluation outputs.")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--matrix-dir", default=None, help="Directory containing matrix_summary_metrics.csv.")
    parser.add_argument("--event-dir", default=None, help="Directory containing event spatial NetCDF products.")
    parser.add_argument("--ensemble-dir", default=None,
                        help="Directory containing ensemble_size_summary.csv from the ensemble tests.")
    parser.add_argument("--noise-csv", default=None, help="Optional explicit noise comparison CSV.")
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--matrix-subset", default="all_data", choices=("all_data", "extreme_events"))
    parser.add_argument("--spatial-subset", default="all_data", choices=("all_data", "extreme_events"))
    return parser.parse_args()


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


def output_formats(fmt: str) -> list[str]:
    return ["pdf", "png"] if fmt == "both" else [fmt]


def load_xarray_dataset(path: Path):
    if not path.exists():
        return None
    try:
        import xarray as xr
        return xr.open_dataset(path)
    except Exception as exc:
        print(f"Could not open {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Matrix aggregation helpers (schema of matrix_summary_metrics.csv)
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
        "model_crps", "geos_crps", "model_rmse", "geos_rmse",
        "crps_skill_pct", "rmse_skill_pct",
        "bss_diff", "calibrated_bss_diff",
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


# ---------------------------------------------------------------------------
# Spatial helpers (schema of matrix_spatial_metrics.nc)
# ---------------------------------------------------------------------------

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
    else:
        field = weighted_dataarray_mean(ds[metric_name].sel(sel), count, reduce_dims)
    field = field.squeeze(drop=True)
    return (
        np.asarray(ds["lon"].values),
        np.asarray(ds["lat"].values),
        np.asarray(field.values, dtype=float),
    )


def symmetric_limit(fields: list[np.ndarray | None], fallback: float = 30.0,
                    lo: float = 5.0, hi: float = 100.0) -> float:
    vals = []
    for field in fields:
        if field is not None:
            finite = np.asarray(field, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                vals.extend(np.abs(finite).ravel().tolist())
    if not vals:
        return fallback
    lim = float(np.nanpercentile(vals, 95))
    return min(max(lim, lo), hi)


# ---------------------------------------------------------------------------
# Shared plotting pieces
# ---------------------------------------------------------------------------

def annotated_heatmap(ax: plt.Axes, arr: np.ndarray | None, sub_arr: np.ndarray | None,
                      title: str, cmap: str, norm=None, vmin=None, vmax=None,
                      value_fmt: str = "{:.2f}", sub_fmt: str = "({:.2f})") -> object:
    """Season x lead heatmap with per-cell annotations and contrast-aware text."""
    if arr is None or not np.isfinite(arr).any():
        missing_panel(ax, title, "Missing valid_season_lead rows in matrix_summary_metrics.csv.")
        return None
    if norm is not None:
        im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    panel_title(ax, title)
    ax.set_xticks(np.arange(len(LEADS)))
    ax.set_xticklabels([f"W{lead}" for lead in LEADS])
    ax.set_yticks(np.arange(len(SEASONS)))
    ax.set_yticklabels(SEASONS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                continue
            rgba = im.cmap(im.norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "#101820" if luminance > 0.55 else "white"
            label = value_fmt.format(val)
            if sub_arr is not None and np.isfinite(sub_arr[i, j]):
                label += "\n" + sub_fmt.format(sub_arr[i, j])
            ax.text(j, i, label, ha="center", va="center", fontsize=7.2,
                    color=color, fontweight="bold", linespacing=1.25)
    return im


def robinson_map(fig: plt.Figure, gridspec_slot, lons, lats, field, title: str,
                 cmap: str, norm=None, vmin=None, vmax=None) -> tuple[plt.Axes, object]:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax = fig.add_subplot(gridspec_slot, projection=ccrs.Robinson(central_longitude=0))
    if lons is None or lats is None or field is None or not np.isfinite(field).any():
        missing_panel(ax, title, "Missing matrix_spatial_metrics.nc data for this map.")
        return ax, None
    kwargs = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    mesh = ax.pcolormesh(lons, lats, field, cmap=cmap, shading="auto",
                         transform=ccrs.PlateCarree(), rasterized=True, **kwargs)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=1.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#333333", zorder=2)
    ax.set_global()
    ax.spines["geo"].set_linewidth(0.6)
    panel_title(ax, title)
    return ax, mesh


def slim_colorbar(fig: plt.Figure, mesh, ax, label: str, symmetric: bool = False,
                  shrink: float = 0.85, pad: float = 0.02, aspect: float = 22):
    if mesh is None:
        return None
    cbar = fig.colorbar(mesh, ax=ax, shrink=shrink, pad=pad, aspect=aspect)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    nbins = 7 if symmetric else 6
    cbar.locator = ticker.MaxNLocator(nbins=nbins, symmetric=symmetric)
    cbar.update_ticks()
    return cbar


# ===========================================================================
# Figure 1 — Framework overview (redesigned schematic)
# ===========================================================================

def _correlated_field(shape: tuple[int, int], length_scale: float, seed: int) -> np.ndarray:
    """Smooth spatially correlated random field via FFT low-pass (numpy only)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    k2 = kx ** 2 + ky ** 2
    filt = np.exp(-2.0 * (np.pi * length_scale) ** 2 * k2)
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
    return (field - field.mean()) / (field.std() + 1e-9)


def _white_field(shape: tuple[int, int], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


def _card(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str,
          face: str, edge: str = EDGE, lw: float = 1.1, title_size: float = 9.0) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
        linewidth=lw, edgecolor=edge, facecolor=face, mutation_aspect=0.6,
    ))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=TEXT_DARK)


def _arrow(ax: plt.Axes, start, end, color="#5c6f80", lw=1.5, rad=0.0, style="-|>", mut=14):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mut, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=6, shrinkA=2, shrinkB=2,
    ))


def _badge(ax: plt.Axes, x: float, y: float, num: str, color: str = EDGE) -> None:
    ax.add_patch(Circle((x, y), 1.35, facecolor=color, edgecolor="none", zorder=8))
    ax.text(x, y, num, ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=9)


def _texture_inset(fig: plt.Figure, ax: plt.Axes, xy: tuple[float, float],
                   wh: tuple[float, float], data: np.ndarray, cmap: str,
                   label: str | None = None, label_size: float = 6.8) -> None:
    """Small rendered field inside the schematic canvas (data coordinates)."""
    ins = ax.inset_axes([xy[0], xy[1], wh[0], wh[1]], transform=ax.transData)
    ins.imshow(data, cmap=cmap, aspect="auto", interpolation="bilinear")
    ins.set_xticks([])
    ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#8595a4")
    if label:
        ins.set_title(label, fontsize=label_size, color=TEXT_DARK, pad=1.5)


def _unet_glyph(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    """Compact encoder-bottleneck-decoder glyph with skip arcs + attention node."""
    depths = [1.00, 0.72, 0.46]
    bar_w = w * 0.085
    gap = w * 0.045
    enc_x = []
    # Encoder bars
    cx = x0
    for i, d in enumerate(depths):
        bh = h * d
        ax.add_patch(FancyBboxPatch(
            (cx, y0 + (h - bh) / 2), bar_w, bh,
            boxstyle="round,pad=0.05,rounding_size=0.25",
            facecolor="#c7d8ec", edgecolor=EDGE, linewidth=0.8, mutation_aspect=0.6))
        enc_x.append(cx)
        cx += bar_w + gap
    # Bottleneck with cross-attention node
    bh = h * 0.30
    bx = cx
    ax.add_patch(FancyBboxPatch(
        (bx, y0 + (h - bh) / 2), bar_w * 1.25, bh,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        facecolor="#e4d9f2", edgecolor=C_ACCENT, linewidth=1.0, mutation_aspect=0.6))
    ax.add_patch(Circle((bx + bar_w * 0.62, y0 + h / 2 + bh * 0.95), h * 0.075,
                        facecolor=C_ACCENT, edgecolor="none", zorder=7))
    ax.text(bx + bar_w * 0.62, y0 + h / 2 + bh * 2.1, "global\ncross-attn",
            ha="center", va="bottom", fontsize=6.4, color=C_ACCENT, linespacing=1.1)
    cx = bx + bar_w * 1.25 + gap
    # Decoder bars
    dec_x = []
    for i, d in enumerate(reversed(depths)):
        bh = h * d
        ax.add_patch(FancyBboxPatch(
            (cx, y0 + (h - bh) / 2), bar_w, bh,
            boxstyle="round,pad=0.05,rounding_size=0.25",
            facecolor="#c7d8ec", edgecolor=EDGE, linewidth=0.8, mutation_aspect=0.6))
        dec_x.append(cx)
        cx += bar_w + gap
    # Skip connections
    for i in range(len(depths) - 1):
        d = depths[i]
        y_arc = y0 + h / 2 + h * d / 2 + 0.4
        _arrow(ax, (enc_x[i] + bar_w, y_arc), (dec_x[len(depths) - 2 - i], y_arc),
               color="#93a6b8", lw=0.9, rad=-0.25, style="-|>", mut=7)
    # Output heads
    head_x = cx + gap * 0.3
    for k, (lbl, col) in enumerate([("PR heads\nW1-4", C_PR), ("T2M heads\nW1-4", C_T2M)]):
        hy = y0 + h * (0.60 - 0.42 * k)
        ax.add_patch(FancyBboxPatch(
            (head_x, hy), bar_w * 2.4, h * 0.30,
            boxstyle="round,pad=0.05,rounding_size=0.25",
            facecolor="white", edgecolor=col, linewidth=1.0, mutation_aspect=0.6))
        ax.text(head_x + bar_w * 1.2, hy + h * 0.15, lbl, ha="center", va="center",
                fontsize=6.2, color=col, fontweight="bold", linespacing=1.1)
        _arrow(ax, (cx - gap * 0.4, y0 + h / 2), (head_x, hy + h * 0.15),
               color="#93a6b8", lw=0.8, mut=7)


def figure_1_framework_overview(output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.set_axis_off()

    # ---------------- Stage 1: conditioning inputs (left column) ----------------
    in_x, in_w = 2.0, 20.5
    cards = [
        (44.5, 11.0, "#eef5fb", f"{BASELINE} ensemble guidance",
         "PR + T2M weekly summaries:\nmean, spread, q10, q90, count"),
        (31.0, 11.0, "#eef8f2", "Observed predictors",
         "SST, SSS, soil moisture, IVT,\nZ500$^{*}$, U250, MJO (4 prior weeks)"),
        (17.5, 11.0, "#f6f4ee", "Static + calendar",
         "elevation, land mask, lat/lon\nencodings, season, lead week"),
    ]
    for y, h, face, title, body in cards:
        _card(ax, (in_x, y), (in_w, h), title, face)
        ax.text(in_x + in_w / 2, y + h / 2 - 1.6, body, ha="center", va="center",
                fontsize=7.2, color=TEXT_MUTED, linespacing=1.35)
    _badge(ax, in_x + 1.2, 56.4, "1")
    ax.text(in_x + 3.2, 56.4, "Conditioning  $c$", ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=TEXT_DARK)

    # ---------------- Stage 2: generator (center top) ----------------
    gen_x, gen_y, gen_w, gen_h = 27.5, 30.5, 34.0, 25.0
    _card(ax, (gen_x, gen_y), (gen_w, gen_h), "Conditional flow-matching generator", "#f2f5fa")
    _unet_glyph(ax, gen_x + 3.0, gen_y + 4.5, gen_w - 6.0, gen_h - 11.0)
    ax.text(gen_x + gen_w / 2, gen_y + 2.2,
            r"$v_\theta(x_t,\,c,\,t,\,\ell)$ — lead-aware velocity field",
            ha="center", va="center", fontsize=8.0, color=TEXT_DARK)
    _badge(ax, gen_x + 1.2, gen_y + gen_h + 1.4, "2")

    for y_src in (50.0, 36.5, 23.0):
        _arrow(ax, (in_x + in_w + 0.6, y_src), (gen_x - 0.6, gen_y + gen_h * 0.5),
               rad=0.08 if y_src > 40 else (-0.08 if y_src < 30 else 0.0), lw=1.3)

    # ---------------- Stage 3: stochastic prior + ODE sampling (bottom center) ----------------
    pr_x, pr_y, pr_w, pr_h = 27.5, 3.5, 24.0, 21.5
    _card(ax, (pr_x, pr_y), (pr_w, pr_h), "Physically informed stochastic prior  $x_0$", "#f7f3fc",
          edge=C_ACCENT)
    tex_w, tex_h = 8.6, 8.6
    _texture_inset(fig, ax, (pr_x + 1.8, pr_y + 6.4), (tex_w, tex_h),
                   _white_field((36, 54), seed=7), "RdBu_r",
                   label=r"Gaussian $\mathcal{N}(0,I)$")
    _texture_inset(fig, ax, (pr_x + pr_w - tex_w - 1.8, pr_y + 6.4), (tex_w, tex_h),
                   _correlated_field((36, 54), length_scale=7.0, seed=11), "RdBu_r",
                   label="EOF-LHS (MJO/NAO/ENSO)")
    ax.text(pr_x + pr_w / 2, pr_y + 10.7, "vs", ha="center", va="center",
            fontsize=8.5, color=TEXT_MUTED, fontstyle="italic")
    ax.text(pr_x + pr_w / 2, pr_y + 4.9,
            r"$x_0=\rho\,x_{\mathrm{EOF}}+\sqrt{1-\rho^{2}}\,x_{\mathrm{rand}}$",
            ha="center", va="center", fontsize=8.0, color=TEXT_DARK)
    ax.text(pr_x + pr_w / 2, pr_y + 2.5,
            r"$\rho_{\mathrm{PR}}=0.25,\ \rho_{\mathrm{T2M}}=0.08$"
            "\nregime- and lead-conditioned",
            ha="center", va="center", fontsize=6.1, color=TEXT_MUTED, linespacing=1.05)
    _badge(ax, pr_x + 1.2, pr_y + pr_h + 1.4, "3", color=C_ACCENT)

    # ODE trajectory inset
    ode_x, ode_y, ode_w, ode_h = 55.5, 3.5, 19.0, 21.5
    _card(ax, (ode_x, ode_y), (ode_w, ode_h), "ODE sampling (50 Euler steps)", "#fdf8ee",
          edge="#b08a2e")
    ins = ax.inset_axes([ode_x + 1.8, ode_y + 4.8, ode_w - 3.6, ode_h - 9.2],
                        transform=ax.transData)
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 51)
    n_members = 7
    x0s = rng.standard_normal(n_members) * 1.05
    x1s = 0.55 * rng.standard_normal(n_members) + 0.15
    member_cmap = plt.get_cmap("viridis")
    for m in range(n_members):
        wiggle = 0.22 * np.sin(np.pi * t) * rng.standard_normal()
        path = (1 - t) * x0s[m] + t * x1s[m] + wiggle * (1 - t)
        color = member_cmap(0.15 + 0.7 * m / max(n_members - 1, 1))
        ins.plot(t, path, color=color, lw=1.3, alpha=0.9, zorder=3)
    ins.axvspan(-0.06, 0.02, color=C_ACCENT, alpha=0.12, lw=0)
    ins.axvspan(0.98, 1.06, color="#2c7a4b", alpha=0.10, lw=0)
    ins.set_xlim(-0.06, 1.06)
    ins.set_xticks([0, 0.5, 1.0])
    ins.set_xticklabels(["$t{=}0$\n$x_0$", "$t$", "$t{=}1$\nforecast"], fontsize=6.4)
    ins.set_yticks([])
    ins.tick_params(length=0, pad=1.5)
    for side in ("top", "right", "left"):
        ins.spines[side].set_visible(False)
    ins.spines["bottom"].set_linewidth(0.6)
    ax.text(ode_x + ode_w / 2, ode_y + 2.6,
            r"$x_{t+\Delta t}=x_t+\Delta t\,v_\theta$"
            "\n"
            r"$\sigma_{\mathrm{eff}}=1+\beta(\sigma_\theta-1)$",
            ha="center", va="center", fontsize=6.0, color=TEXT_MUTED, linespacing=1.08)

    _arrow(ax, (pr_x + pr_w + 0.6, pr_y + pr_h * 0.55), (ode_x - 0.6, ode_y + ode_h * 0.55),
           color=C_ACCENT, lw=1.5)
    _arrow(ax, (gen_x + gen_w * 0.62, gen_y - 0.6), (ode_x + ode_w * 0.35, ode_y + ode_h + 0.6),
           color="#5c6f80", lw=1.3, rad=-0.12)
    ax.text(gen_x + gen_w * 0.72, gen_y - 3.4, r"velocity $v_\theta$", fontsize=7.0,
            color=TEXT_MUTED, ha="left")

    # ---------------- Stage 4: outputs + verification (right column) ----------------
    out_x, out_w = 65.5, 32.0
    _card(ax, (out_x, 30.5), (out_w, 25.0), "Weekly probabilistic forecasts", "#eef7f0",
          edge="#2c7a4b")
    # Stacked ensemble member textures
    base_x, base_y = out_x + 2.6, 34.2
    for k in range(3):
        off = (2 - k) * 1.5
        _texture_inset(fig, ax, (base_x + off, base_y + off), (9.6, 9.0),
                       _correlated_field((36, 54), length_scale=6.0, seed=20 + k), "BrBG")
    ax.text(base_x + 6.6, 32.4, "90-member PR + T2M\nensembles, weeks 1-4",
            ha="center", va="center", fontsize=7.0, color=TEXT_MUTED, linespacing=1.3)
    ax.text(out_x + out_w - 8.6, 46.5, "calibrated\ntail risk", ha="center", va="center",
            fontsize=7.0, color=TEXT_MUTED, linespacing=1.3)
    # Simple pdf sketch: baseline vs refined
    pdf_ins = ax.inset_axes([out_x + out_w - 14.4, 34.0, 12.0, 11.0], transform=ax.transData)
    xs = np.linspace(-3.6, 3.6, 200)
    pdf_ins.fill_between(xs, np.exp(-((xs + 0.9) ** 2) / (2 * 1.35 ** 2)), color=C_BASELINE,
                         alpha=0.25, lw=0)
    pdf_ins.plot(xs, np.exp(-((xs + 0.9) ** 2) / (2 * 1.35 ** 2)), color=C_BASELINE, lw=1.2)
    pdf_ins.fill_between(xs, np.exp(-((xs - 0.25) ** 2) / (2 * 0.8 ** 2)), color=C_MODEL,
                         alpha=0.25, lw=0)
    pdf_ins.plot(xs, np.exp(-((xs - 0.25) ** 2) / (2 * 0.8 ** 2)), color=C_MODEL, lw=1.2)
    pdf_ins.axvline(0.55, color="#2c7a4b", lw=1.1, ls="--")
    pdf_ins.text(0.62, 0.96, "obs", fontsize=6.2, color="#2c7a4b", rotation=90, va="top")
    pdf_ins.text(-3.3, 0.92, BASELINE, fontsize=6.2, color=C_BASELINE)
    pdf_ins.text(-3.3, 0.78, "refined", fontsize=6.2, color=C_MODEL)
    pdf_ins.set_xticks([])
    pdf_ins.set_yticks([])
    for side in ("top", "right", "left"):
        pdf_ins.spines[side].set_visible(False)
    pdf_ins.spines["bottom"].set_linewidth(0.6)
    _badge(ax, out_x + 1.2, 56.9, "4", color="#2c7a4b")

    # Verification card
    ver_y, ver_h = 3.5, 21.5
    _card(ax, (out_x + 13.0, ver_y), (out_w - 13.0, ver_h), "Verification", "#f6f7f9")
    chips = ["CRPS", "RMSE / bias\n/ corr", "calibrated\nBSS",
             "spread skill", "season x lead", "event tails\n(q95/q99)"]
    chip_y = ver_y + ver_h - 6.4
    for idx, chip in enumerate(chips):
        row, col = divmod(idx, 2)
        cx = out_x + 14.4 + col * 8.8
        cy = chip_y - row * 4.6
        ax.add_patch(FancyBboxPatch(
            (cx, cy), 8.0, 3.2, boxstyle="round,pad=0.25,rounding_size=0.7",
            facecolor="white", edgecolor="#9aa8b5", linewidth=0.8, mutation_aspect=0.6))
        ax.text(cx + 4.0, cy + 1.6, chip, ha="center", va="center", fontsize=5.9,
                color=TEXT_DARK, linespacing=1.05)

    _arrow(ax, (ode_x + ode_w + 0.6, ode_y + ode_h * 0.62), (out_x - 0.6 + 1.0, 40.0),
           color="#2c7a4b", lw=1.6, rad=-0.15)
    _arrow(ax, (out_x + out_w * 0.55, 29.9), (out_x + out_w * 0.62, ver_y + ver_h + 0.6),
           color="#5c6f80", lw=1.2)

    ax.set_ylim(2.8, 58)
    return save_figure(fig, output_dir, "fig1_framework_overview", formats, dpi)


# ===========================================================================
# Figures 2 & 3 — Per-variable skill (heatmaps + Robinson maps)
# ===========================================================================

def figure_variable_skill(output_dir: Path, formats: list[str], dpi: int,
                          summary: pd.DataFrame | None, matrix_dir: Path,
                          variable: str, stem: str, subset: str,
                          spatial_subset: str) -> list[Path]:
    var_short = VARIABLE_SHORT.get(variable, variable.upper())
    unit = "mm day$^{-1}$" if variable == "pr" else "K"

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.58, 1.55], hspace=0.20, wspace=0.14,
                          left=0.025, right=0.975, bottom=0.02, top=0.96)

    # Row 1 — season x lead heatmaps
    geos_crps = season_lead_values(summary, variable, "geos_crps", subset)
    model_crps = season_lead_values(summary, variable, "model_crps", subset)
    crps_skill = season_lead_values(summary, variable, "crps_skill_pct", subset)
    geos_rmse = season_lead_values(summary, variable, "geos_rmse", subset)
    model_rmse = season_lead_values(summary, variable, "model_rmse", subset)
    rmse_skill = season_lead_values(summary, variable, "rmse_skill_pct", subset)

    hlim_c = symmetric_limit([crps_skill])
    hlim_r = symmetric_limit([rmse_skill])

    axes_hm = [fig.add_subplot(gs[0, k]) for k in range(4)]
    im_a = annotated_heatmap(axes_hm[0], geos_crps, None, f"(a) {BASELINE} CRPS", CMAP_MAG)
    im_b = annotated_heatmap(axes_hm[1], crps_skill, model_crps,
                             "(b) CRPS skill (%)", CMAP_SKILL,
                             norm=TwoSlopeNorm(vcenter=0.0, vmin=-hlim_c, vmax=hlim_c),
                             value_fmt="{:+.0f}%")
    im_c = annotated_heatmap(axes_hm[2], geos_rmse, None, f"(c) {BASELINE} RMSE", CMAP_MAG)
    im_d = annotated_heatmap(axes_hm[3], rmse_skill, model_rmse,
                             "(d) RMSE skill (%)", CMAP_SKILL,
                             norm=TwoSlopeNorm(vcenter=0.0, vmin=-hlim_r, vmax=hlim_r),
                             value_fmt="{:+.0f}%")
    slim_colorbar(fig, im_a, axes_hm[0], f"CRPS ({unit})")
    slim_colorbar(fig, im_b, axes_hm[1], "skill (%)", symmetric=True)
    slim_colorbar(fig, im_c, axes_hm[2], f"RMSE ({unit})")
    slim_colorbar(fig, im_d, axes_hm[3], "skill (%)", symmetric=True)

    # Row 2 — Robinson spatial maps
    ds = load_xarray_dataset(matrix_dir / "matrix_spatial_metrics.nc")
    map_gc = spatial_metric_map(ds, variable, "geos_crps", spatial_subset)
    map_cs = spatial_metric_map(ds, variable, "crps_skill_pct", spatial_subset)
    map_gr = spatial_metric_map(ds, variable, "geos_rmse", spatial_subset)
    map_rs = spatial_metric_map(ds, variable, "rmse_skill_pct", spatial_subset)
    if ds is not None:
        ds.close()

    # Land-only display: use baseline-CRPS NaN pattern as common mask.
    if map_gc[2] is not None:
        ocean = ~np.isfinite(map_gc[2])
        map_cs = (map_cs[0], map_cs[1], np.where(ocean, np.nan, map_cs[2]) if map_cs[2] is not None else None)
        map_gr = (map_gr[0], map_gr[1], np.where(ocean, np.nan, map_gr[2]) if map_gr[2] is not None else None)
        map_rs = (map_rs[0], map_rs[1], np.where(ocean, np.nan, map_rs[2]) if map_rs[2] is not None else None)

    def mag_limits(field):
        if field is None:
            return 0.0, 1.0
        finite = field[np.isfinite(field)]
        if not finite.size:
            return 0.0, 1.0
        return float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))

    gc_lo, gc_hi = mag_limits(map_gc[2])
    gr_lo, gr_hi = mag_limits(map_gr[2])
    slim_cs = symmetric_limit([map_cs[2]])
    slim_rs = symmetric_limit([map_rs[2]])

    ax_e, im_e = robinson_map(fig, gs[1, 0], *map_gc, f"(e) {BASELINE} CRPS",
                              CMAP_MAG, vmin=gc_lo, vmax=gc_hi)
    ax_f, im_f = robinson_map(fig, gs[1, 1], *map_cs, f"(f) {var_short} CRPS skill (%)",
                              CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-slim_cs, vmax=slim_cs))
    ax_g, im_g = robinson_map(fig, gs[1, 2], *map_gr, f"(g) {BASELINE} RMSE",
                              CMAP_MAG, vmin=gr_lo, vmax=gr_hi)
    ax_h, im_h = robinson_map(fig, gs[1, 3], *map_rs, f"(h) {var_short} RMSE skill (%)",
                              CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-slim_rs, vmax=slim_rs))
    map_cbar = {"shrink": 0.62, "pad": 0.012, "aspect": 30}
    slim_colorbar(fig, im_e, ax_e, f"CRPS ({unit})", **map_cbar)
    slim_colorbar(fig, im_f, ax_f, "skill (%)", symmetric=True, **map_cbar)
    slim_colorbar(fig, im_g, ax_g, f"RMSE ({unit})", **map_cbar)
    slim_colorbar(fig, im_h, ax_h, "skill (%)", symmetric=True, **map_cbar)

    return save_figure(fig, output_dir, stem, formats, dpi)


# ===========================================================================
# Figure 4 — Noise-prior ablation
# ===========================================================================

def load_noise_ablation(noise_csv: Path | None) -> tuple[dict[str, dict[str, float]], str]:
    """Return {strategy_label: {variable: crps_skill_pct}} plus a provenance note."""
    df = read_csv_or_none(noise_csv)
    if df is None or df.empty:
        return NOISE_FALLBACK, "frozen 2021 full-year ablation (CSV not found at plot time)"

    strat_col = next((c for c in ("strategy", "noise_mode", "noise", "sampler", "mode",
                                  "label", "config") if c in df.columns), None)
    var_col = next((c for c in ("variable", "var", "target") if c in df.columns), None)
    if strat_col is None or var_col is None:
        return NOISE_FALLBACK, "frozen 2021 full-year ablation (unrecognized CSV schema)"

    result: dict[str, dict[str, float]] = {}
    for (strategy, variable), group in df.groupby([strat_col, var_col]):
        variable = str(variable).lower()
        weights = group_weights(group)
        if "crps_skill_pct" in group:
            value = weighted_average(group["crps_skill_pct"], weights)
        elif {"model_crps", "geos_crps"} <= set(group.columns):
            value = skill_pct(weighted_average(group["model_crps"], weights),
                              weighted_average(group["geos_crps"], weights))
        else:
            continue
        result.setdefault(str(strategy), {})[variable] = value
    if not result:
        return NOISE_FALLBACK, "frozen 2021 full-year ablation (no usable rows in CSV)"
    return result, f"from {noise_csv.name}"


def figure_4_noise_ablation(output_dir: Path, formats: list[str], dpi: int,
                            noise_csv: Path | None) -> list[Path]:
    data, provenance = load_noise_ablation(noise_csv)
    strategies = list(data.keys())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                     gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.30})

    # (a) CRPS skill vs raw baseline for each prior
    n = len(strategies)
    width = 0.34
    xs = np.arange(n)
    for vi, variable in enumerate(("pr", "t2m")):
        vals = [data[s].get(variable, np.nan) for s in strategies]
        offs = xs + (vi - 0.5) * width
        bars = ax_a.bar(offs, vals, width * 0.92,
                        color=C_PR if variable == "pr" else C_T2M,
                        edgecolor="white", linewidth=0.8,
                        label=VARIABLE_SHORT[variable])
        for bar_obj, val in zip(bars, vals):
            if np.isfinite(val):
                ax_a.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.6,
                          f"{val:.1f}", ha="center", va="bottom", fontsize=7.6,
                          fontweight="bold", color=TEXT_DARK)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(["\n".join(textwrap.wrap(s, 18)) for s in strategies], fontsize=8)
    ax_a.set_ylabel(f"CRPS skill vs raw {BASELINE} (%)")
    ax_a.axhline(0.0, color="#7a8794", lw=0.8)
    ax_a.legend(loc="upper left", ncols=2)
    style_axis(ax_a)
    panel_title(ax_a, "(a) CRPS skill by stochastic prior")

    # (b) relative CRPS reduction of each structured prior vs the Gaussian prior
    gauss_key = next((s for s in strategies if "gauss" in s.lower()), strategies[0])
    others = [s for s in strategies if s != gauss_key]
    if others:
        labels, pr_vals, t2m_vals = [], [], []
        for s in others:
            labels.append(s)
            rel = {}
            for variable in ("pr", "t2m"):
                skill_s = data[s].get(variable, np.nan)
                skill_g = data[gauss_key].get(variable, np.nan)
                # CRPS ratio to baseline: (1 - skill/100); relative reduction vs Gaussian:
                if np.isfinite(skill_s) and np.isfinite(skill_g) and (100.0 - skill_g) > 1e-9:
                    rel[variable] = 100.0 * (1.0 - (100.0 - skill_s) / (100.0 - skill_g))
                else:
                    rel[variable] = np.nan
            pr_vals.append(rel["pr"])
            t2m_vals.append(rel["t2m"])
        ys = np.arange(len(others))
        hbar_h = 0.34
        ax_b.barh(ys + hbar_h / 2, pr_vals, hbar_h * 0.92, color=C_PR, label="PR",
                  edgecolor="white", linewidth=0.8)
        ax_b.barh(ys - hbar_h / 2, t2m_vals, hbar_h * 0.92, color=C_T2M, label="T2M",
                  edgecolor="white", linewidth=0.8)
        for y, v in list(zip(ys + hbar_h / 2, pr_vals)) + list(zip(ys - hbar_h / 2, t2m_vals)):
            if np.isfinite(v):
                ax_b.text(v + 0.08, y, f"{v:+.1f}%", va="center", ha="left",
                          fontsize=7.6, fontweight="bold", color=TEXT_DARK)
        ax_b.set_yticks(ys)
        ax_b.set_yticklabels(["\n".join(textwrap.wrap(s, 16)) for s in others], fontsize=8)
        ax_b.axvline(0.0, color="#7a8794", lw=0.8)
        ax_b.set_xlabel("CRPS reduction vs Gaussian prior (%)")
        ax_b.legend(loc="lower right")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.grid(True, axis="x", color="#dde3e8", linewidth=0.6, alpha=0.8)
        ax_b.set_axisbelow(True)
        upper = np.nanmax([v for v in pr_vals + t2m_vals if np.isfinite(v)] or [1.0])
        ax_b.set_xlim(right=upper * 1.35)
        panel_title(ax_b, "(b) Added value of the physical prior")
    else:
        missing_panel(ax_b, "(b) Added value of the physical prior",
                      "Only one sampling strategy found in the ablation CSV.")

    print(f"fig4 noise-ablation values: {provenance}")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig4_noise_ablation", formats, dpi)


# ===========================================================================
# Figure 5 — Ensemble size and skill convergence
# ===========================================================================

LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}
LEAD_LINESTYLES = {3: "--", 4: "-"}
FIG5_LEADS = (3, 4)


def figure_5_member_convergence(output_dir: Path, formats: list[str], dpi: int,
                                ensemble_dir: Path) -> list[Path]:
    """Skill versus number of generated ensemble members.

    Reads ensemble_size_summary.csv written by
    ml_model/evaluate_ensemble_tests_flow_finalv1_global.py (columns
    {metric}_mean/_p05/_p95 grouped by variable, lead, member_count).
    """
    df = read_csv_or_none(ensemble_dir / "ensemble_size_summary.csv")
    specs = [
        ("crps_skill_pct", "CRPS skill (%)", 0.0),
        ("rmse_skill_pct", "RMSE skill (%)", 0.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)
    letters = iter("abcd")

    for vi, variable in enumerate(("pr", "t2m")):
        for mi, (metric, label, refline) in enumerate(specs):
            ax = axes[vi, mi]
            letter = next(letters)
            title = f"({letter}) {VARIABLE_SHORT[variable]} {label}"
            if df is None or df.empty or f"{metric}_mean" not in df.columns:
                missing_panel(ax, title,
                              "Missing ensemble_size_summary.csv; run "
                              "ml_model/evaluate_ensemble_tests_flow_finalv1_global.py.")
                continue
            sub = df[df["variable"].astype(str).str.lower().eq(variable)]
            if sub.empty:
                missing_panel(ax, title, f"No rows for variable {variable}.")
                continue
            available_leads = sorted(set(sub["lead"].astype(int).unique()))
            plot_leads = [lead for lead in FIG5_LEADS if lead in available_leads] or available_leads
            for lead in plot_leads:
                grp = sub[sub["lead"].astype(int).eq(lead)].sort_values("member_count")
                x = grp["member_count"].to_numpy(dtype=float)
                mean = grp[f"{metric}_mean"].to_numpy(dtype=float)
                color = LEAD_COLORS.get(int(lead), C_MODEL)
                lo_col, hi_col = f"{metric}_p05", f"{metric}_p95"
                if lo_col in grp and hi_col in grp:
                    lo = grp[lo_col].to_numpy(dtype=float)
                    hi = grp[hi_col].to_numpy(dtype=float)
                    ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
                ax.plot(x, mean, color=color, lw=1.7, ls=LEAD_LINESTYLES.get(int(lead), "-"),
                        marker="o", ms=3.6,
                        label=f"W{int(lead)}")
            ax.axhline(refline, color="#7a8794", lw=0.9, ls="--")
            style_axis(ax)
            panel_title(ax, title)
            if vi == 1:
                ax.set_xlabel("Generated members")
            if mi == 0:
                ax.set_ylabel(VARIABLE_LABELS[variable])
            if vi == 0 and mi == len(specs) - 1:
                ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig5_member_convergence", formats, dpi)


# ===========================================================================
# Figure 6 — Extreme-event subset skill (with all-case reference)
# ===========================================================================

def figure_6_extreme_skill(output_dir: Path, formats: list[str], dpi: int,
                           summary: pd.DataFrame | None) -> list[Path]:
    agg_ext = aggregate_matrix_by_lead(summary, subset="extreme_events")
    agg_all = aggregate_matrix_by_lead(summary, subset="all_data")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
    metrics = [("crps", "(a) CRPS skill"), ("rmse", "(b) RMSE skill")]
    width = 0.34

    for ax, (metric, title) in zip(axes, metrics):
        col = f"{metric}_skill_pct"
        if agg_ext.empty or col not in agg_ext:
            missing_panel(ax, title, "Missing extreme_events subset in matrix_summary_metrics.csv.")
            continue
        for vi, variable in enumerate(("pr", "t2m")):
            sub = agg_ext[agg_ext["variable"].eq(variable)].sort_values("lead")
            if sub.empty:
                continue
            xs = sub["lead"].to_numpy(dtype=float) + (vi - 0.5) * width
            vals = sub[col].to_numpy(dtype=float)
            color = C_PR if variable == "pr" else C_T2M
            bars = ax.bar(xs, vals, width * 0.92, color=color, edgecolor="white",
                          linewidth=0.8, label=f"{VARIABLE_SHORT[variable]} extremes")
            for bar_obj, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            bar_obj.get_height() + 0.5, f"{val:.1f}",
                            ha="center", va="bottom", fontsize=7.2,
                            fontweight="bold", color=TEXT_DARK)
            # All-case reference markers
            if not agg_all.empty and col in agg_all:
                ref = agg_all[agg_all["variable"].eq(variable)].sort_values("lead")
                if not ref.empty:
                    ax.scatter(ref["lead"].to_numpy(dtype=float) + (vi - 0.5) * width,
                               ref[col].to_numpy(dtype=float),
                               facecolor="white", edgecolor=color, s=26, zorder=5,
                               linewidth=1.3,
                               label=f"{VARIABLE_SHORT[variable]} all-case" if metric == "crps" else None)
        ax.axhline(0.0, color="#7a8794", lw=0.8)
        ax.set_xticks(LEADS)
        ax.set_xticklabels([f"W{l}" for l in LEADS])
        ax.set_xlabel("Lead week")
        ax.set_ylabel(f"{metric.upper()} skill vs {BASELINE} (%)")
        style_axis(ax)
        panel_title(ax, title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
                   ncols=4, fontsize=8)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig6_extreme_skill", formats, dpi)


# ===========================================================================
# Figures 7 & 8 — Event case studies (3x3 contoured NetCDF panels)
# ===========================================================================

def regional_panel(ax, lons, lats, field, title: str, cmap: str, norm=None,
                   vmin=None, vmax=None, extend: str = "both"):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    finite = field[np.isfinite(field)]
    if not finite.size:
        missing_panel(ax, title, "All values are NaN.")
        return None
    if vmin is None:
        vmin = float(np.nanpercentile(finite, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 98))
    if norm is not None:
        levels = np.linspace(norm.vmin, norm.vmax, 21)
    else:
        if vmin == vmax:
            vmin, vmax = vmin - 0.1, vmax + 0.1
        levels = np.linspace(vmin, vmax, 21)
    mesh = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, norm=norm,
                       extend=extend, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#222222", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#555555", linestyle=":", zorder=2)
    if np.nanmin(lons) > -135 and np.nanmax(lons) < -105:
        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="#777777", linestyle=":", zorder=2)
    ax.set_extent([float(np.nanmin(lons)), float(np.nanmax(lons)),
                   float(np.nanmin(lats)), float(np.nanmax(lats))], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False, color="#d9dee3", linewidth=0.25, alpha=0.7)
    panel_title(ax, title)
    return mesh


def row_colorbar(fig, mesh, axes, label: str, symmetric: bool = False, extend: str = "both"):
    if mesh is None:
        return None
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.86, pad=0.015, aspect=26, extend=extend)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    cbar.locator = ticker.MaxNLocator(nbins=7 if symmetric else 6, symmetric=symmetric)
    cbar.update_ticks()
    return cbar


def figure_event_case(output_dir: Path, formats: list[str], dpi: int, event_dir: Path,
                      event_id: str, stem: str, is_t2m: bool) -> list[Path]:
    import cartopy.crs as ccrs

    nc_path = event_dir / "plots" / "spatial_maps" / f"{event_id}_lead4_spatial_data.nc"
    fig = plt.figure(figsize=(13.4, 9.6) if is_t2m else (12.6, 10.8))
    gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.06,
                          left=0.02, right=0.90, bottom=0.02, top=0.97)

    if not nc_path.exists():
        ax_big = fig.add_subplot(gs[:, :])
        missing_panel(ax_big, "Spatial maps pending",
                      f"NetCDF product {nc_path.name} not found; run "
                      "paper/scripts/make_contoured_event_plots.py first.")
        return save_figure(fig, output_dir, stem, formats, dpi)

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
    except Exception as exc:
        ax_big = fig.add_subplot(gs[:, :])
        missing_panel(ax_big, "Error loading NetCDF event data",
                      f"Could not load {nc_path.name}: {exc}")
        return save_figure(fig, output_dir, stem, formats, dpi)

    ocean = np.isnan(obs)
    fields = {}
    for name, arr in [("geos_q95", geos_q95), ("model_q95", model_q95),
                      ("geos_crps", geos_crps), ("model_crps", model_crps),
                      ("geos_bss", geos_bss), ("model_bss", model_bss)]:
        fields[name] = np.where(ocean, np.nan, arr)

    denom = np.where(fields["geos_crps"] == 0.0, 1e-5, fields["geos_crps"])
    crps_gain = np.where(ocean, np.nan, 100.0 * (1.0 - fields["model_crps"] / denom))
    bss_gain = 100.0 * (fields["model_bss"] - fields["geos_bss"])
    bss_gain = np.where((bss_gain >= -30.0) & (bss_gain <= 30.0), bss_gain, np.nan)
    bss_gain = np.where(ocean, np.nan, bss_gain)

    # Shared row limits
    row1_vals = np.concatenate([obs.ravel(), fields["geos_q95"].ravel(), fields["model_q95"].ravel()])
    row1_finite = row1_vals[np.isfinite(row1_vals)]
    f_lo = float(np.nanpercentile(row1_finite, 2)) if row1_finite.size else 0.0
    f_hi = float(np.nanpercentile(row1_finite, 98)) if row1_finite.size else 1.0
    crps_vals = np.concatenate([fields["geos_crps"].ravel(), fields["model_crps"].ravel()])
    crps_finite = crps_vals[np.isfinite(crps_vals)]
    c_hi = float(np.nanpercentile(crps_finite, 98)) if crps_finite.size else 1.0
    gain_lim = symmetric_limit([crps_gain], fallback=30.0, lo=10.0)

    field_cmap = "RdYlBu_r" if is_t2m else "YlGnBu"
    phys_label = "temperature (K)" if is_t2m else "precipitation (mm day$^{-1}$)"

    axes_r1 = [fig.add_subplot(gs[0, k], projection=ccrs.PlateCarree()) for k in range(3)]
    m1 = regional_panel(axes_r1[0], lons, lats, obs, "(a) Observed", field_cmap, vmin=f_lo, vmax=f_hi)
    regional_panel(axes_r1[1], lons, lats, fields["geos_q95"], f"(b) {BASELINE} q95",
                   field_cmap, vmin=f_lo, vmax=f_hi)
    regional_panel(axes_r1[2], lons, lats, fields["model_q95"], f"(c) {METHOD} q95",
                   field_cmap, vmin=f_lo, vmax=f_hi)
    row_colorbar(fig, m1, axes_r1, phys_label)

    axes_r2 = [fig.add_subplot(gs[1, k], projection=ccrs.PlateCarree()) for k in range(3)]
    m2 = regional_panel(axes_r2[0], lons, lats, fields["geos_crps"], f"(d) {BASELINE} CRPS",
                        "magma_r", vmin=0.0, vmax=c_hi, extend="max")
    regional_panel(axes_r2[1], lons, lats, fields["model_crps"], f"(e) {METHOD} CRPS",
                   "magma_r", vmin=0.0, vmax=c_hi, extend="max")
    row_colorbar(fig, m2, axes_r2[:2], "CRPS", extend="max")
    m2c = regional_panel(axes_r2[2], lons, lats, crps_gain, "(f) CRPS skill (%)",
                         CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-gain_lim, vmax=gain_lim))
    row_colorbar(fig, m2c, [axes_r2[2]], "skill (%)", symmetric=True)
    # Stipple robust CRPS improvement (> 10%), thinned
    lon2d, lat2d = np.meshgrid(lons, lats)
    thin = np.zeros_like(crps_gain, dtype=bool)
    thin[::2, ::2] = True
    sig = (crps_gain > 10.0) & thin
    if sig.any():
        axes_r2[2].scatter(lon2d[sig], lat2d[sig], color="black", s=2.6, alpha=0.65,
                           marker="o", edgecolors="none", transform=ccrs.PlateCarree())

    axes_r3 = [fig.add_subplot(gs[2, k], projection=ccrs.PlateCarree()) for k in range(3)]
    m3 = regional_panel(axes_r3[0], lons, lats, fields["geos_bss"], f"(g) {BASELINE} BSS",
                        "Blues", vmin=0.0, vmax=0.6)
    regional_panel(axes_r3[1], lons, lats, fields["model_bss"], f"(h) {METHOD} BSS",
                   "Blues", vmin=0.0, vmax=0.6)
    row_colorbar(fig, m3, axes_r3[:2], "BSS")
    m3c = regional_panel(axes_r3[2], lons, lats, bss_gain, "(i) BSS gain (x100)",
                         CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-30.0, vmax=30.0))
    row_colorbar(fig, m3c, [axes_r3[2]], "gain (x100)", symmetric=True)
    sig_b = (bss_gain > 5.0) & thin
    if sig_b.any():
        axes_r3[2].scatter(lon2d[sig_b], lat2d[sig_b], color="black", s=2.6, alpha=0.65,
                           marker="o", edgecolors="none", transform=ccrs.PlateCarree())

    return save_figure(fig, output_dir, stem, formats, dpi)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    set_style()
    args = parse_args()
    formats = output_formats(args.format)
    output_dir = Path(args.output_dir)
    matrix_dir = first_existing_dir(args.matrix_dir, DEFAULT_MATRIX_DIR_CANDIDATES)
    event_dir = first_existing_dir(args.event_dir, DEFAULT_EVENT_DIR_CANDIDATES)
    ensemble_dir = first_existing_dir(args.ensemble_dir, DEFAULT_ENSEMBLE_DIR_CANDIDATES)
    noise_csv = Path(args.noise_csv) if args.noise_csv else newest_matching(NOISE_CSV_PATTERNS)

    print("Figure input locations")
    print(f"  matrix_dir   : {matrix_dir}")
    print(f"  event_dir    : {event_dir}")
    print(f"  ensemble_dir : {ensemble_dir}")
    print(f"  noise_csv    : {noise_csv}")
    print(f"  output_dir   : {output_dir}")

    summary = read_csv_or_none(matrix_dir / "matrix_summary_metrics.csv")

    written: list[Path] = []
    written.extend(figure_1_framework_overview(output_dir, formats, args.dpi))
    written.extend(figure_variable_skill(
        output_dir, formats, args.dpi, summary, matrix_dir,
        variable="pr", stem="fig2_pr_skill",
        subset=args.matrix_subset, spatial_subset=args.spatial_subset))
    written.extend(figure_variable_skill(
        output_dir, formats, args.dpi, summary, matrix_dir,
        variable="t2m", stem="fig3_t2m_skill",
        subset=args.matrix_subset, spatial_subset=args.spatial_subset))
    written.extend(figure_4_noise_ablation(output_dir, formats, args.dpi, noise_csv))
    written.extend(figure_5_member_convergence(output_dir, formats, args.dpi, ensemble_dir))
    written.extend(figure_6_extreme_skill(output_dir, formats, args.dpi, summary))
    written.extend(figure_event_case(output_dir, formats, args.dpi, event_dir,
                                     EVENT_PR_ID, "fig7_event_pr_california", is_t2m=False))
    written.extend(figure_event_case(output_dir, formats, args.dpi, event_dir,
                                     EVENT_T2M_ID, "fig8_event_t2m_uk_heatwave", is_t2m=True))

    print(f"\nWrote {len(written)} figure files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
