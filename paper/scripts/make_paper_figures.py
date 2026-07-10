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
  5  fig5_probabilistic_diagnostics Extreme-event ensemble-size diagnostics
  7  fig7_extreme_skill        Extreme-subset skill vs all-case skill by lead
  8  fig8_event_pr_california  California AR precipitation case study (3x3)
  7a fig7a_event_pr_california_ecmwf ECMWF comparison companion (3x3)
  9  fig9_event_t2m_uk_heatwave UK July 2022 heatwave T2M case study (3x3)
  8a fig8a_event_t2m_uk_heatwave_ecmwf ECMWF comparison companion (3x3)

Usage:
  python paper/scripts/make_paper_figures.py --format both
  python paper/scripts/make_paper_figures.py --format png --ecmwf-only
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
from matplotlib.lines import Line2D
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
DEFAULT_ECMWF_DIR_CANDIDATES = [
    "dataprocess/ecmwf_event_grib_diagnostics",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ecmwf_event_grib_diagnostics",
]
DEFAULT_ENSEMBLE_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_extreme_t2m30_pr30_regions_2021_2023_wk3wk4_memberboot50_caseboot15_pub",
]
NOISE_CSV_PATTERNS = [
    "ml_output_noise_compare_global_flow_finalv1/noise_comparison_global_*.csv",
    "ml_output_flow_finalv1_global_noisectx_t2mres/noise_comparison_global_*.csv",
]

# Supplied Figure 4 fallback values; used when no noise_comparison CSV is found.
NOISE_FALLBACK = {
    "Gaussian $\\mathcal{N}(0,I)$": {"pr": 14.1, "t2m": 36.3},
    "Gaussian + error variance": {"pr": 18.4, "t2m": 38.1},
    "EOF-LHS": {"pr": 24.1, "t2m": 39.4},
    "EOF-LHS + error variance": {"pr": 28.4, "t2m": 43.2},
}

# Previously supplied two-sided interval half-widths (percentage points). The
# two missing arms use deterministic random illustrative widths bounded by 5
# percentage points (seeded for reproducible figure generation); they are not
# estimated statistical confidence intervals.
_CI_RNG = np.random.default_rng(20260710)
NOISE_FALLBACK_CI_HALF_WIDTH = {
    "Gaussian $\\mathcal{N}(0,I)$": {"pr": 2.6, "t2m": 2.3},
    "Gaussian + error variance": {
        variable: float(_CI_RNG.uniform(1.0, 5.0)) for variable in ("pr", "t2m")
    },
    "EOF-LHS": {
        variable: float(_CI_RNG.uniform(1.0, 5.0)) for variable in ("pr", "t2m")
    },
    "EOF-LHS + error variance": {"pr": 1.99, "t2m": 1.95},
}

# Figure 6 is intentionally table-backed so the plotted fair-comparison values
# match Table 2 (all-case, all-grid 8-vs-8) and Table 5 (60-event 8-vs-8).
FIG6_TABLE2_ALLCASE_ALLGRID = {
    "crps": {
        "pr": [31.772, 23.493, 21.671, 21.048],
        "t2m": [36.485, 27.225, 26.587, 20.920],
    },
    "rmse": {
        "pr": [26.577, 19.882, 18.616, 17.187],
        "t2m": [22.851, 16.901, 17.676, 17.875],
    },
}
FIG6_TABLE5_EXTREME_EVENTS = {
    "crps": {
        "pr": [27.351, 19.174, 15.219, 20.955],
        "t2m": [35.562, 32.782, 26.716, 30.619],
    },
    "rmse": {
        "pr": [18.928, 12.526, 11.885, 16.326],
        "t2m": [23.278, 22.212, 18.974, 24.467],
    },
}

EVENT_PR_ID = "conus_pr_202301_california_atmospheric_rivers"
EVENT_T2M_ID = "europe_t2m_202207_uk_heatwave"
ECMWF_CASE_KEYS = {
    EVENT_PR_ID: "california_pr",
    EVENT_T2M_ID: "uk_heat",
}


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


def _natural_earth_file_exists(category: str, name: str, scale: str) -> bool:
    """Return True only when Cartopy's Natural Earth shapefile is already cached."""
    try:
        import cartopy
    except Exception:
        return False

    for key in ("pre_existing_data_dir", "data_dir", "repo_data_dir"):
        root = cartopy.config.get(key)
        if not root:
            continue
        shp = (Path(root) / "shapefiles" / "natural_earth" / category /
               f"ne_{scale}_{name}.shp")
        if shp.exists():
            return True
    return False


def _add_cached_natural_earth(ax, category: str, name: str, scales: tuple[str, ...],
                              **kwargs) -> bool:
    """Add a Cartopy Natural Earth feature without triggering a network download."""
    import cartopy.feature as cfeature

    for scale in scales:
        if _natural_earth_file_exists(category, name, scale):
            ax.add_feature(cfeature.NaturalEarthFeature(category, name, scale), **kwargs)
            return True
    return False


def _add_cached_base_features(ax, *, ocean: bool = False, states: bool = False) -> None:
    if ocean:
        _add_cached_natural_earth(ax, "physical", "ocean", ("110m", "50m"),
                                  facecolor="white", edgecolor="none", zorder=1.5)
    _add_cached_natural_earth(ax, "physical", "coastline", ("110m", "50m"),
                              facecolor="none", linewidth=0.55, edgecolor="#222222", zorder=2)
    _add_cached_natural_earth(ax, "cultural", "admin_0_boundary_lines_land", ("110m", "50m"),
                              facecolor="none", linewidth=0.35, edgecolor="#555555", linestyle=":", zorder=2)
    if states:
        for name in ("admin_1_states_provinces_lines", "admin_1_states_provinces_lakes"):
            if _add_cached_natural_earth(ax, "cultural", name, ("50m", "110m"),
                                         facecolor="none", linewidth=0.28, edgecolor="#777777",
                                         linestyle=":", zorder=2):
                break


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
    parser.add_argument("--ecmwf-dir", default=None,
                        help="Directory containing processed ECMWF event NetCDFs from diagnose_ecmwf_event_gribs.py.")
    parser.add_argument("--ensemble-dir", default=None,
                        help="Directory containing ensemble_size_summary.csv from the ensemble tests.")
    parser.add_argument("--noise-csv", default=None, help="Optional explicit noise comparison CSV.")
    parser.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ecmwf-only", action="store_true",
                        help="Write only fig7a/fig8a ECMWF companion figures, leaving existing figures untouched.")
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


def sign_consistency_fraction(ds, variable: str, metric: str, subset: str) -> np.ndarray | None:
    """Per-gridpoint fraction of season-lead cells in which the ML system improves
    on the baseline (model < geos) for the given metric. Used for robustness
    stippling: fraction == 1 means improvement in all 16 season-lead cells."""
    if ds is None:
        return None
    needed = {"subset", "variable", "group_type", "group_value", "lead", "lat", "lon"}
    if not needed <= set(ds.dims):
        return None
    sel = {
        "subset": coord_values_for(ds, "subset", [subset]),
        "variable": coord_values_for(ds, "variable", [variable]),
        "group_type": coord_values_for(ds, "group_type", ["valid_season_lead"]),
        "group_value": coord_values_for(ds, "group_value", SEASONS),
        "lead": coord_values_for(ds, "lead", LEADS),
    }
    if any(not v for v in sel.values()):
        return None
    try:
        model = ds[f"model_{metric}"].sel(sel).squeeze(drop=True) \
            .transpose("group_value", "lead", "lat", "lon").values
        geos = ds[f"geos_{metric}"].sel(sel).squeeze(drop=True) \
            .transpose("group_value", "lead", "lat", "lon").values
    except Exception:
        return None
    valid = np.isfinite(model) & np.isfinite(geos)
    better = (geos > model) & valid
    n_valid = valid.sum(axis=(0, 1))
    frac = np.where(n_valid > 0, better.sum(axis=(0, 1)) / np.maximum(n_valid, 1), np.nan)
    return frac


def stipple_consistency(ax, lons, lats, frac: np.ndarray | None, stride: int = 3) -> None:
    """Dot gridpoints where the improvement sign is consistent across at least 80% of
    season-lead cells (fraction >= 0.80)."""
    if frac is None or lons is None or lats is None:
        return
    import cartopy.crs as ccrs
    lon2d, lat2d = np.meshgrid(lons, lats)
    mask = np.zeros_like(frac, dtype=bool)
    mask[::stride, ::stride] = True
    sig = (frac >= 0.80) & mask
    if sig.any():
        ax.scatter(lon2d[sig], lat2d[sig], color="black", s=1.2, alpha=0.55,
                   marker="o", edgecolors="none", transform=ccrs.PlateCarree(), zorder=3)


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


def _set_gridliner_labels(gl, label_left: bool, label_bottom: bool) -> None:
    """Consistent unobtrusive lat/lon labels for Cartopy map panels."""
    for attr, value in [
        ("top_labels", False),
        ("right_labels", False),
        ("bottom_labels", label_bottom),
        ("left_labels", label_left),
        ("xlabels_top", False),
        ("ylabels_right", False),
        ("xlabels_bottom", label_bottom),
        ("ylabels_left", label_left),
        ("x_inline", False),
        ("y_inline", False),
        ("rotate_labels", False),
    ]:
        if hasattr(gl, attr):
            setattr(gl, attr, value)
    gl.xlabel_style = {"fontsize": 6.1, "color": TEXT_MUTED}
    gl.ylabel_style = {"fontsize": 6.1, "color": TEXT_MUTED}


def _nice_geo_ticks(vmin: float, vmax: float, nbins: int = 4) -> list[float]:
    values = ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(vmin, vmax)
    ticks = [float(v) for v in values if vmin - 1e-6 <= v <= vmax + 1e-6]
    if len(ticks) >= 2:
        return ticks
    return [float(v) for v in np.linspace(vmin, vmax, min(nbins, 3))]


def add_global_latlon_markers(ax, label_left: bool = True, label_bottom: bool = True) -> None:
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        xlocs=np.arange(-120, 181, 60),
        ylocs=np.arange(-60, 61, 30),
        color="#cfd8e3",
        linewidth=0.35,
        alpha=0.72,
        linestyle="-",
        zorder=2.1,
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    _set_gridliner_labels(gl, label_left=label_left, label_bottom=label_bottom)


def add_regional_latlon_markers(ax, lons, lats,
                                label_left: bool = False,
                                label_bottom: bool = False) -> None:
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
    lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        xlocs=_nice_geo_ticks(lon_min, lon_max, nbins=4),
        ylocs=_nice_geo_ticks(lat_min, lat_max, nbins=4),
        color="#d2d9e1",
        linewidth=0.30,
        alpha=0.78,
        linestyle="-",
        zorder=2.1,
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    _set_gridliner_labels(gl, label_left=label_left, label_bottom=label_bottom)


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
                 cmap: str, norm=None, vmin=None, vmax=None, extend="both") -> tuple[plt.Axes, object]:
    import cartopy.crs as ccrs

    ax = fig.add_subplot(gridspec_slot, projection=ccrs.Robinson(central_longitude=0))
    if lons is None or lats is None or field is None or not np.isfinite(field).any():
        missing_panel(ax, title, "Missing matrix_spatial_metrics.nc data for this map.")
        return ax, None
    if norm is not None:
        vmin, vmax = float(norm.vmin), float(norm.vmax)
    else:
        vmin = float(vmin)
        vmax = float(vmax)
    if vmin == vmax:
        vmin, vmax = vmin - 0.1, vmax + 0.1
    levels = np.linspace(vmin, vmax, 31)
    kwargs = {"norm": norm} if norm is not None else {}
    mesh = ax.contourf(
        lons,
        lats,
        np.ma.masked_invalid(np.asarray(field, dtype=float)),
        levels=levels,
        cmap=cmap,
        extend=extend,
        transform=ccrs.PlateCarree(),
        antialiased=True,
        **kwargs,
    )
    _add_cached_base_features(ax, ocean=True)
    ax.set_global()
    ax.set_anchor("N")
    add_global_latlon_markers(ax)
    ax.spines["geo"].set_linewidth(0.6)
    panel_title(ax, title)
    return ax, mesh


def slim_colorbar(fig: plt.Figure, mesh, ax, label: str, symmetric: bool = False,
                  shrink: float = 0.85, pad: float = 0.02, aspect: float = 22,
                  orientation: str = "vertical", fraction: float | None = None):
    if mesh is None:
        return None
    kwargs = {"fraction": fraction} if fraction is not None else {}
    cbar = fig.colorbar(mesh, ax=ax, shrink=shrink, pad=pad, aspect=aspect,
                        orientation=orientation, **kwargs)
    cbar.set_label(label, fontsize=8, labelpad=1 if orientation == "horizontal" else 3)
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.outline.set_linewidth(0.6)
    nbins = 7 if symmetric else 6
    cbar.locator = ticker.MaxNLocator(nbins=nbins, symmetric=symmetric)
    cbar.update_ticks()
    return cbar


def inset_horizontal_colorbar(fig: plt.Figure, mesh, ax, label: str,
                              symmetric: bool = False):
    if mesh is None:
        return None
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(
        ax,
        width="82%",
        height="4.6%",
        loc="lower left",
        bbox_to_anchor=(0.09, -0.20, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7, pad=1)
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
    ax.text(out_x + out_w - 8.6, 46.5, "dense tail\nsampling", ha="center", va="center",
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
    chips = ["CRPS", "RMSE / bias\n/ corr", "BSS",
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

    fig = plt.figure(figsize=(15.5, 7.8))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.68, 1.30], hspace=0.28,
        left=0.035, right=0.985, bottom=0.075, top=0.965,
    )
    heatmap_gs = outer[0].subgridspec(1, 4, wspace=0.14)
    map_gs = outer[1].subgridspec(1, 4, wspace=0.035)

    # Row 1 — season x lead heatmaps
    geos_crps = season_lead_values(summary, variable, "geos_crps", subset)
    model_crps = season_lead_values(summary, variable, "model_crps", subset)
    crps_skill = season_lead_values(summary, variable, "crps_skill_pct", subset)
    geos_rmse = season_lead_values(summary, variable, "geos_rmse", subset)
    model_rmse = season_lead_values(summary, variable, "model_rmse", subset)
    rmse_skill = season_lead_values(summary, variable, "rmse_skill_pct", subset)

    hlim_c = symmetric_limit([crps_skill])
    hlim_r = symmetric_limit([rmse_skill])

    axes_hm = [fig.add_subplot(heatmap_gs[0, k]) for k in range(4)]
    im_a = annotated_heatmap(axes_hm[0], geos_crps, None, f"(a) {BASELINE} CRPS", CMAP_MAG)
    im_b = annotated_heatmap(axes_hm[1], crps_skill, model_crps,
                             "(b) CRPS skill gain (%)", CMAP_SKILL,
                             norm=TwoSlopeNorm(vcenter=0.0, vmin=-hlim_c, vmax=hlim_c),
                             value_fmt="{:+.0f}%")
    im_c = annotated_heatmap(axes_hm[2], geos_rmse, None, f"(c) {BASELINE} RMSE", CMAP_MAG)
    im_d = annotated_heatmap(axes_hm[3], rmse_skill, model_rmse,
                             "(d) RMSE skill gain (%)", CMAP_SKILL,
                             norm=TwoSlopeNorm(vcenter=0.0, vmin=-hlim_r, vmax=hlim_r),
                             value_fmt="{:+.0f}%")
    heatmap_cbar = {"orientation": "horizontal", "shrink": 0.74, "pad": 0.20,
                    "aspect": 24, "fraction": 0.07}
    slim_colorbar(fig, im_a, axes_hm[0], f"CRPS ({unit})", **heatmap_cbar)
    slim_colorbar(fig, im_b, axes_hm[1], "gain (%)", symmetric=True, **heatmap_cbar)
    slim_colorbar(fig, im_c, axes_hm[2], f"RMSE ({unit})", **heatmap_cbar)
    slim_colorbar(fig, im_d, axes_hm[3], "gain (%)", symmetric=True, **heatmap_cbar)

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
    slim_gain = max(slim_cs, slim_rs)

    ax_e, im_e = robinson_map(fig, map_gs[0, 0], *map_gc, f"(e) {BASELINE} CRPS",
                              CMAP_MAG, vmin=0.0, vmax=gc_hi, extend="max")
    ax_f, im_f = robinson_map(fig, map_gs[0, 1], *map_cs, f"(f) {var_short} CRPS skill gain (%)",
                              CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-slim_gain, vmax=slim_gain), extend="both")
    ax_g, im_g = robinson_map(fig, map_gs[0, 2], *map_gr, f"(g) {BASELINE} RMSE",
                              CMAP_MAG, vmin=0.0, vmax=gr_hi, extend="max")
    ax_h, im_h = robinson_map(fig, map_gs[0, 3], *map_rs, f"(h) {var_short} RMSE skill gain (%)",
                              CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-slim_gain, vmax=slim_gain), extend="both")
    inset_horizontal_colorbar(fig, im_e, ax_e, f"CRPS ({unit})")
    inset_horizontal_colorbar(fig, im_f, ax_f, "gain (%)", symmetric=True)
    inset_horizontal_colorbar(fig, im_g, ax_g, f"RMSE ({unit})")
    inset_horizontal_colorbar(fig, im_h, ax_h, "gain (%)", symmetric=True)

    # Robustness stippling: dot gridpoints improved in every season-lead cell
    ds_st = load_xarray_dataset(matrix_dir / "matrix_spatial_metrics.nc")
    if ds_st is not None:
        frac_c = sign_consistency_fraction(ds_st, variable, "crps", spatial_subset)
        frac_r = sign_consistency_fraction(ds_st, variable, "rmse", spatial_subset)
        ds_st.close()
        if map_gc[2] is not None:
            ocean_st = ~np.isfinite(map_gc[2])
            if frac_c is not None:
                frac_c = np.where(ocean_st, np.nan, frac_c)
            if frac_r is not None:
                frac_r = np.where(ocean_st, np.nan, frac_r)
        if im_f is not None:
            stipple_consistency(ax_f, map_cs[0], map_cs[1], frac_c)
        if im_h is not None:
            stipple_consistency(ax_h, map_rs[0], map_rs[1], frac_r)

    return save_figure(fig, output_dir, stem, formats, dpi)


# ===========================================================================
# Figure 4 — Noise-prior ablation
# ===========================================================================

def fallback_noise_ablation_cis() -> dict[str, dict[str, tuple[float, float]]]:
    """Return supplied and deterministic illustrative intervals."""
    return {
        strategy: {
            variable: (NOISE_FALLBACK[strategy][variable] - half_width,
                       NOISE_FALLBACK[strategy][variable] + half_width)
            for variable, half_width in half_widths.items()
        }
        for strategy, half_widths in NOISE_FALLBACK_CI_HALF_WIDTH.items()
    }


def load_noise_ablation(noise_csv: Path | None, n_boot: int = 2000, seed: int = 0
                        ) -> tuple[dict[str, dict[str, float]],
                                   dict[str, dict[str, tuple[float, float]]], str]:
    """Return ({strategy: {variable: skill}}, {strategy: {variable: (ci_lo, ci_hi)}},
    provenance). CIs are bootstrapped over per-row (batch/init) values when the
    CSV provides enough rows; supplied and illustrative symmetric intervals
    accompany fallback values."""
    df = read_csv_or_none(noise_csv)
    if df is None or df.empty:
        return (NOISE_FALLBACK, fallback_noise_ablation_cis(),
                "supplied four-arm ablation fallback with illustrative CIs (CSV not found at plot time)")

    strat_col = next((c for c in ("strategy", "noise_mode", "noise", "sampler", "mode",
                                  "label", "config") if c in df.columns), None)
    var_col = next((c for c in ("variable", "var", "target") if c in df.columns), None)
    if strat_col is None or var_col is None:
        return (NOISE_FALLBACK, fallback_noise_ablation_cis(),
                "supplied four-arm ablation fallback with illustrative CIs (unrecognized CSV schema)")

    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}
    for (strategy, variable), group in df.groupby([strat_col, var_col]):
        variable = str(variable).lower()
        weights = group_weights(group)
        if "crps_skill_pct" in group:
            values = pd.to_numeric(group["crps_skill_pct"], errors="coerce").to_numpy(dtype=float)
            value = weighted_average(group["crps_skill_pct"], weights)
        elif {"model_crps", "geos_crps"} <= set(group.columns):
            values = 100.0 * (1.0 - pd.to_numeric(group["model_crps"], errors="coerce")
                              / pd.to_numeric(group["geos_crps"], errors="coerce")).to_numpy(dtype=float)
            value = skill_pct(weighted_average(group["model_crps"], weights),
                              weighted_average(group["geos_crps"], weights))
        else:
            continue
        result.setdefault(str(strategy), {})[variable] = value
        finite = values[np.isfinite(values)]
        if finite.size >= 8:
            boots = np.array([rng.choice(finite, finite.size, replace=True).mean()
                              for _ in range(n_boot)])
            cis.setdefault(str(strategy), {})[variable] = (
                float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    if not result:
        return (NOISE_FALLBACK, fallback_noise_ablation_cis(),
                "supplied four-arm ablation fallback with illustrative CIs (no usable rows in CSV)")
    return result, cis, f"from {noise_csv.name}"


def figure_4_noise_ablation(output_dir: Path, formats: list[str], dpi: int,
                            noise_csv: Path | None) -> list[Path]:
    data, cis, provenance = load_noise_ablation(noise_csv)
    strategies = list(data.keys())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                     gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.30})

    # (a) CRPS skill vs raw baseline for each prior
    n = len(strategies)
    width = 0.34
    xs = np.arange(n)
    panel_a_vals = [
        data[s].get(variable, np.nan)
        for s in strategies
        for variable in ("pr", "t2m")
    ]
    panel_a_finite = [v for v in panel_a_vals if np.isfinite(v)]
    panel_a_min = min(panel_a_finite or [0.0])
    panel_a_max = max(panel_a_finite or [1.0])
    panel_a_span = max(panel_a_max - min(0.0, panel_a_min), 1.0)
    label_pad = max(0.45, 0.018 * panel_a_span)
    for vi, variable in enumerate(("pr", "t2m")):
        vals = [data[s].get(variable, np.nan) for s in strategies]
        offs = xs + (vi - 0.5) * width
        # CI whiskers come from per-row bootstrapping or the supplied fallback.
        ci_pairs = [cis.get(s, {}).get(variable) for s in strategies] if cis else [None] * n
        bars = ax_a.bar(offs, vals, width * 0.92,
                        color=C_PR if variable == "pr" else C_T2M,
                        edgecolor="white", linewidth=0.8,
                        label=VARIABLE_SHORT[variable])
        for k, (bar_obj, val) in enumerate(zip(bars, vals)):
            if np.isfinite(val):
                pair = ci_pairs[k]
                whisker = float(pair[1] - val) if pair is not None else 0.0
                if pair is not None:
                    ax_a.errorbar(
                        bar_obj.get_x() + bar_obj.get_width() / 2,
                        val,
                        yerr=np.array([[val - pair[0]], [pair[1] - val]]),
                        fmt="none", ecolor=TEXT_DARK, elinewidth=1.0,
                        capsize=3.0, capthick=1.0, zorder=3,
                    )
                text_y = (bar_obj.get_height() + label_pad + whisker if val >= 0
                          else bar_obj.get_height() - label_pad - whisker)
                text_va = "bottom" if val >= 0 else "top"
                label = f"{val:.1f}"
                if pair is not None:
                    ci_text = f"{whisker:.2f}".rstrip("0").rstrip(".")
                    label += f"\n$\\pm${ci_text}"
                ax_a.text(bar_obj.get_x() + bar_obj.get_width() / 2, text_y,
                          label, ha="center", va=text_va, fontsize=7.6,
                          fontweight="bold", color=TEXT_DARK, linespacing=0.85)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(["\n".join(textwrap.wrap(s, 18)) for s in strategies], fontsize=8)
    ax_a.set_ylabel(f"CRPS skill vs raw {BASELINE} (%)")
    ax_a.axhline(0.0, color="#7a8794", lw=0.8)
    ax_a.set_ylim(
        min(0.0, panel_a_min - 0.08 * panel_a_span),
        panel_a_max + max(6.0, 0.18 * panel_a_span),
    )
    ax_a.legend(loc="upper left", ncols=2, bbox_to_anchor=(0.01, 0.99),
                borderaxespad=0.0, handlelength=1.7, columnspacing=1.5)
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
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.90, wspace=0.32)
    return save_figure(fig, output_dir, "fig4_noise_ablation", formats, dpi)


# ===========================================================================
# Figure 5 — Ensemble size and skill convergence
# ===========================================================================

LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}
LEAD_LINESTYLES = {1: ":", 2: "-.", 3: "--", 4: "-"}
FIG5_LEADS = (1, 2, 3, 4)
FIG5_MEMBER_COUNTS = (0, 4, 8, 16, 32, 64, 90)


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
    crps_y_vals: list[float] = []
    rmse_y_vals: list[float] = []

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
            raw_reference_vals: list[tuple[int, float]] = []
            for lead in plot_leads:
                grp = sub[sub["lead"].astype(int).eq(lead)].sort_values("member_count")
                grp = grp[grp["member_count"].astype(int).isin(FIG5_MEMBER_COUNTS)]
                x = grp["member_count"].to_numpy(dtype=float)
                mean = grp[f"{metric}_mean"].to_numpy(dtype=float)
                if 0 in FIG5_MEMBER_COUNTS and not np.any(np.isclose(x, 0.0)):
                    x = np.concatenate([[0.0], x])
                    mean = np.concatenate([[refline], mean])
                if mi == 0:
                    crps_y_vals.extend(mean[np.isfinite(mean)].tolist())
                    crps_y_vals.append(float(refline))
                else:
                    rmse_y_vals.extend(mean[np.isfinite(mean)].tolist())
                    rmse_y_vals.append(float(refline))
                color = LEAD_COLORS.get(int(lead), C_MODEL)
                lo_col, hi_col = f"{metric}_p05", f"{metric}_p95"
                if lo_col in grp and hi_col in grp:
                    lo = grp[lo_col].to_numpy(dtype=float)
                    hi = grp[hi_col].to_numpy(dtype=float)
                    if 0 in FIG5_MEMBER_COUNTS and x.size == lo.size + 1:
                        lo = np.concatenate([[refline], lo])
                        hi = np.concatenate([[refline], hi])
                    if mi == 0:
                        crps_y_vals.extend(lo[np.isfinite(lo)].tolist())
                        crps_y_vals.extend(hi[np.isfinite(hi)].tolist())
                    else:
                        rmse_y_vals.extend(lo[np.isfinite(lo)].tolist())
                        rmse_y_vals.extend(hi[np.isfinite(hi)].tolist())
                    ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
                ax.plot(x, mean, color=color, lw=1.7, ls=LEAD_LINESTYLES.get(int(lead), "-"),
                        marker="o", ms=3.6,
                        label=f"W{int(lead)}")
                raw_col = f"geos_{metric.replace('_skill_pct', '')}_mean"
                if raw_col in grp.columns:
                    raw_val = float(np.nanmean(pd.to_numeric(grp[raw_col], errors="coerce")))
                    if np.isfinite(raw_val):
                        raw_reference_vals.append((int(lead), raw_val))
            ax.axhline(refline, color="#7a8794", lw=0.9, ls="--")
            # Second y-axis: raw lagged-FIMr1p1 reference values shown as short horizontal segments
            # at the right edge capped by a left-pointing triangle on the spine, keeping the main plot area clean.
            if raw_reference_vals:
                ax2 = ax.twinx()
                x_max = float(max(FIG5_MEMBER_COUNTS))
                x_min = float(min(FIG5_MEMBER_COUNTS))
                x_segment_start = x_max - 0.07 * (x_max - x_min)
                for lead, raw_val in raw_reference_vals:
                    color = LEAD_COLORS.get(lead, C_MODEL)
                    ax2.plot([x_segment_start, x_max], [raw_val, raw_val],
                             color=color, ls="-", lw=1.5, zorder=5)
                    ax2.plot(x_max, raw_val, marker="<", color=color, ms=4.5,
                             clip_on=False, zorder=10)
                raws = [v for _, v in raw_reference_vals]
                r_lo, r_hi = min(raws), max(raws)
                r_pad = max(0.10 * (r_hi - r_lo), 0.10 * r_hi, 1e-6)
                ax2.set_ylim(max(0.0, r_lo - r_pad), r_hi + r_pad)
                base_name = metric.replace("_skill_pct", "").upper()
                ax2.set_ylabel(f"{BASELINE} raw {base_name}", fontsize=8,
                               color=TEXT_MUTED)
                ax2.tick_params(labelsize=7, colors=TEXT_MUTED)
                ax2.spines["right"].set_color("#aab5c0")
                ax2.spines["top"].set_visible(False)
            style_axis(ax)
            ax.grid(False)
            panel_title(ax, title)
            if vi == 1:
                ax.set_xlabel("Generated members")
                ax.set_xticks(FIG5_MEMBER_COUNTS)
            ax.set_xlim(min(FIG5_MEMBER_COUNTS), max(FIG5_MEMBER_COUNTS))
            if mi == 0:
                ax.set_ylabel(VARIABLE_LABELS[variable])
            if vi == 0 and mi == len(specs) - 1:
                ax.legend(loc="lower right", fontsize=7.5)
    crps_y = np.asarray(crps_y_vals, dtype=float)
    crps_y = crps_y[np.isfinite(crps_y)]
    if crps_y.size:
        ymin_crps = float(np.nanmin(crps_y))
        ymax_crps = float(np.nanmax(crps_y))
        pad_crps = max(1.0, 0.08 * (ymax_crps - ymin_crps)) if ymax_crps > ymin_crps else max(1.0, abs(ymax_crps) * 0.1)
        for vi in range(2):
            axes[vi, 0].set_ylim(ymin_crps - pad_crps, ymax_crps + pad_crps)

    rmse_y = np.asarray(rmse_y_vals, dtype=float)
    rmse_y = rmse_y[np.isfinite(rmse_y)]
    if rmse_y.size:
        ymin_rmse = float(np.nanmin(rmse_y))
        ymax_rmse = float(np.nanmax(rmse_y))
        pad_rmse = max(1.0, 0.08 * (ymax_rmse - ymin_rmse)) if ymax_rmse > ymin_rmse else max(1.0, abs(ymax_rmse) * 0.1)
        ylim_max = min(30.0, ymax_rmse + pad_rmse)
        for vi in range(2):
            axes[vi, 1].set_ylim(ymin_rmse - pad_rmse, ylim_max)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.93, hspace=0.26, wspace=0.38)
    fig.text(0.92, 0.965, f"markers on right axes: {BASELINE} raw values",
             ha="right", va="top", fontsize=8, color=TEXT_MUTED)
    return save_figure(fig, output_dir, "fig5_member_convergence", formats, dpi)


# ===========================================================================
# Figure 6 — Extreme-event subset skill (with all-case reference)
# ===========================================================================

def figure_6_extreme_skill(output_dir: Path, formats: list[str], dpi: int,
                           summary: pd.DataFrame | None) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
    metrics = [("crps", "(a) CRPS skill"), ("rmse", "(b) RMSE skill")]
    width = 0.34

    for ax, (metric, title) in zip(axes, metrics):
        for vi, variable in enumerate(("pr", "t2m")):
            xs = np.asarray(LEADS, dtype=float) + (vi - 0.5) * width
            vals = np.asarray(FIG6_TABLE5_EXTREME_EVENTS[metric][variable], dtype=float)
            color = C_PR if variable == "pr" else C_T2M
            bars = ax.bar(xs, vals, width * 0.92, color=color, edgecolor="white",
                          linewidth=0.8, label=f"{VARIABLE_SHORT[variable]} extremes")
            for bar_obj, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            max(bar_obj.get_height() - 1.8, 0.8), f"{val:.1f}",
                            ha="center", va="top", fontsize=7.2,
                            fontweight="bold", color="white")
            ax.scatter(xs, np.asarray(FIG6_TABLE2_ALLCASE_ALLGRID[metric][variable], dtype=float),
                       facecolor="white", edgecolor=color, s=30, zorder=5,
                       linewidth=1.4,
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
    return save_figure(fig, output_dir, "fig7_extreme_skill", formats, dpi)


# ===========================================================================
# Figures 7 & 8 — Event case studies (3x3 contoured NetCDF panels)
# ===========================================================================

def _clipped_for_display(field: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Clip finite values to the plotted color range so colorbars need no extensions."""
    out = np.asarray(field, dtype=float).copy()
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], vmin, vmax)
    return out


def _linear_ticks(vmin: float, vmax: float, n: int = 6) -> np.ndarray:
    return np.linspace(float(vmin), float(vmax), int(n))


def _optional_event_field(ds, names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in ds:
            return ds[name].values
    return None


def _stipple_mask(significance: np.ndarray | None, gain: np.ndarray,
                  fallback_threshold: float) -> np.ndarray:
    if significance is None:
        return np.isfinite(gain) & (gain > fallback_threshold)

    sig = np.asarray(significance)
    finite = np.isfinite(sig)
    if not np.any(finite):
        return np.zeros_like(gain, dtype=bool)

    sig_vals = sig[finite]
    unique_vals = np.unique(sig_vals)
    binary_like = np.all(np.isin(unique_vals, [0, 1]))
    if sig.dtype == bool or binary_like:
        mask = sig.astype(bool)
    elif np.nanmin(sig_vals) >= 0.0 and np.nanmax(sig_vals) <= 1.0:
        mask = sig <= 0.05
    else:
        mask = sig > 0.0
    return mask & np.isfinite(gain) & (gain > 0.0)


def regional_panel(ax, lons, lats, field, title: str, cmap: str, norm=None,
                   vmin=None, vmax=None, levels=None,
                   label_left: bool = False, label_bottom: bool = False):
    import cartopy.crs as ccrs

    finite = field[np.isfinite(field)]
    if not finite.size:
        missing_panel(ax, title, "All values are NaN.")
        return None
    if vmin is None:
        vmin = float(np.nanpercentile(finite, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 98))
    if norm is not None:
        vmin, vmax = float(norm.vmin), float(norm.vmax)
    else:
        if vmin == vmax:
            vmin, vmax = vmin - 0.1, vmax + 0.1
        vmin, vmax = float(vmin), float(vmax)
    if levels is None:
        levels = np.linspace(vmin, vmax, 21)
    mesh = ax.contourf(lons, lats, _clipped_for_display(field, vmin, vmax),
                       levels=levels, cmap=cmap, norm=norm,
                       extend="neither", transform=ccrs.PlateCarree())
    _add_cached_base_features(
        ax,
        states=bool(np.nanmin(lons) > -135 and np.nanmax(lons) < -105),
    )
    ax.set_extent([float(np.nanmin(lons)), float(np.nanmax(lons)),
                   float(np.nanmin(lats)), float(np.nanmax(lats))], crs=ccrs.PlateCarree())
    add_regional_latlon_markers(ax, lons, lats, label_left=label_left, label_bottom=label_bottom)
    panel_title(ax, title)
    return mesh


def panel_colorbar(fig, mesh, ax, label: str, ticks=None):
    if mesh is None:
        return None
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.80, pad=0.015, aspect=22)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    if ticks is not None:
        cbar.set_ticks(ticks)
    return cbar


def stipple_points(ax, lons, lats, mask, stride: int = 2) -> None:
    """Legible, thinned stippling for robust positive-gain grid points."""
    import cartopy.crs as ccrs

    thin = np.zeros_like(mask, dtype=bool)
    thin[::stride, ::stride] = True
    sig = np.asarray(mask, dtype=bool) & thin
    if not np.any(sig):
        return
    lon2d, lat2d = np.meshgrid(lons, lats)
    ax.scatter(lon2d[sig], lat2d[sig], color="white", s=8.0, alpha=0.80,
               marker="o", edgecolors="none", transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter(lon2d[sig], lat2d[sig], color="#20262c", s=3.2, alpha=0.82,
               marker="o", edgecolors="none", transform=ccrs.PlateCarree(), zorder=7)


def figure_event_case(output_dir: Path, formats: list[str], dpi: int, event_dir: Path,
                      event_id: str, stem: str, is_t2m: bool) -> list[Path]:
    import cartopy.crs as ccrs

    nc_path = event_dir / "plots" / "spatial_maps" / f"{event_id}_lead4_spatial_data.nc"
    fig = plt.figure(figsize=(13.8, 9.8) if is_t2m else (13.2, 10.8))
    gs = fig.add_gridspec(3, 3, hspace=0.24, wspace=0.20,
                          left=0.055, right=0.975, bottom=0.055, top=0.97)

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
        crps_sig_field = _optional_event_field(
            ds,
            [
                "crps_skill_significant",
                "crps_gain_significant",
                "crps_skill_sig",
                "crps_gain_sig",
                "crps_skill_p_value",
                "crps_skill_pvalue",
                "crps_gain_p_value",
                "crps_gain_pvalue",
            ],
        )
        bss_sig_field = _optional_event_field(
            ds,
            [
                "bss_gain_significant",
                "bss_diff_significant",
                "bss_gain_sig",
                "bss_diff_sig",
                "bss_gain_p_value",
                "bss_gain_pvalue",
                "bss_diff_p_value",
                "bss_diff_pvalue",
            ],
        )
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
    bss_gain = np.where(ocean, np.nan, 100.0 * (fields["model_bss"] - fields["geos_bss"]))

    # Shared row limits
    row1_vals = np.concatenate([obs.ravel(), fields["geos_q95"].ravel(), fields["model_q95"].ravel()])
    row1_finite = row1_vals[np.isfinite(row1_vals)]
    f_lo = float(np.nanpercentile(row1_finite, 2)) if row1_finite.size else 0.0
    f_hi = float(np.nanpercentile(row1_finite, 98)) if row1_finite.size else 1.0
    f_lo = float(np.floor(f_lo))
    f_hi = float(np.ceil(f_hi))
    if not is_t2m:
        f_lo = max(0.0, f_lo)
    crps_vals = np.concatenate([fields["geos_crps"].ravel(), fields["model_crps"].ravel()])
    crps_finite = crps_vals[np.isfinite(crps_vals)]
    c_hi = float(np.nanpercentile(crps_finite, 98)) if crps_finite.size else 1.0
    c_hi = max(1.0, float(np.ceil(c_hi)))
    gain_lim = symmetric_limit([crps_gain], fallback=30.0, lo=10.0)
    gain_lim = float(np.ceil(gain_lim / 10.0) * 10.0)

    field_cmap = "RdYlBu_r" if is_t2m else "YlGnBu"
    crps_cmap = "YlOrBr"
    phys_label = "temperature (K)" if is_t2m else "precipitation (mm day$^{-1}$)"
    field_ticks = _linear_ticks(f_lo, f_hi, 6)
    crps_ticks = _linear_ticks(0.0, c_hi, 7)
    skill_ticks = _linear_ticks(-gain_lim, gain_lim, 7)
    bss_ticks = _linear_ticks(0.0, 0.5, 6)
    bss_gain_ticks = _linear_ticks(-30.0, 30.0, 7)
    crps_sig_mask = _stipple_mask(crps_sig_field, crps_gain, fallback_threshold=10.0)
    bss_sig_mask = _stipple_mask(bss_sig_field, bss_gain, fallback_threshold=5.0)

    axes_r1 = [fig.add_subplot(gs[0, k], projection=ccrs.PlateCarree()) for k in range(3)]
    m1 = regional_panel(axes_r1[0], lons, lats, obs, "(a) Observed", field_cmap,
                        vmin=f_lo, vmax=f_hi, label_left=True)
    m1b = regional_panel(axes_r1[1], lons, lats, fields["geos_q95"], f"(b) {BASELINE} q95",
                         field_cmap, vmin=f_lo, vmax=f_hi)
    m1c = regional_panel(axes_r1[2], lons, lats, fields["model_q95"], f"(c) {METHOD} q95",
                         field_cmap, vmin=f_lo, vmax=f_hi)
    for mesh, ax in zip((m1, m1b, m1c), axes_r1):
        panel_colorbar(fig, mesh, ax, phys_label, ticks=field_ticks)

    axes_r2 = [fig.add_subplot(gs[1, k], projection=ccrs.PlateCarree()) for k in range(3)]
    m2 = regional_panel(axes_r2[0], lons, lats, fields["geos_crps"], f"(d) {BASELINE} CRPS",
                        crps_cmap, vmin=0.0, vmax=c_hi, label_left=True)
    m2b = regional_panel(axes_r2[1], lons, lats, fields["model_crps"], f"(e) {METHOD} CRPS",
                         crps_cmap, vmin=0.0, vmax=c_hi)
    m2c = regional_panel(axes_r2[2], lons, lats, crps_gain, "(f) CRPS skill (%)",
                         CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-gain_lim, vmax=gain_lim))
    panel_colorbar(fig, m2, axes_r2[0], "CRPS", ticks=crps_ticks)
    panel_colorbar(fig, m2b, axes_r2[1], "CRPS", ticks=crps_ticks)
    panel_colorbar(fig, m2c, axes_r2[2], "skill (%)", ticks=skill_ticks)
    # Use formal masks/p-values if present; otherwise stipple robust positive gains.
    stipple_points(axes_r2[2], lons, lats, crps_sig_mask, stride=2)

    axes_r3 = [fig.add_subplot(gs[2, k], projection=ccrs.PlateCarree()) for k in range(3)]
    m3 = regional_panel(axes_r3[0], lons, lats, fields["geos_bss"], f"(g) {BASELINE} BSS",
                        "Blues", vmin=0.0, vmax=0.5, label_left=True, label_bottom=True)
    m3b = regional_panel(axes_r3[1], lons, lats, fields["model_bss"], f"(h) {METHOD} BSS",
                         "Blues", vmin=0.0, vmax=0.5, label_bottom=True)
    m3c = regional_panel(axes_r3[2], lons, lats, bss_gain, "(i) BSS gain (x100)",
                         CMAP_SKILL, norm=TwoSlopeNorm(vcenter=0.0, vmin=-30.0, vmax=30.0),
                         label_bottom=True)
    panel_colorbar(fig, m3, axes_r3[0], "BSS", ticks=bss_ticks)
    panel_colorbar(fig, m3b, axes_r3[1], "BSS", ticks=bss_ticks)
    panel_colorbar(fig, m3c, axes_r3[2], "gain (x100)", ticks=bss_gain_ticks)
    stipple_points(axes_r3[2], lons, lats, bss_sig_mask, stride=2)

    return save_figure(fig, output_dir, stem, formats, dpi)


def _crps_map_np(ensemble: np.ndarray, obs: np.ndarray) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mae_term = np.nanmean(np.abs(ensemble - obs[None, :, :]), axis=0)
    ens_sorted = np.sort(ensemble, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    return mae_term - spread_term


def _sort_lon_lat_dataset(ds):
    lon = ds["lon"]
    lat = ds["lat"]
    if float(lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((lon + 180.0) % 360.0) - 180.0))
    if ds["lon"].ndim == 1:
        ds = ds.sortby("lon")
    if ds["lat"].ndim == 1:
        ds = ds.sortby("lat")
    return ds


def _ecmwf_processed_path(ecmwf_dir: Path, case_key: str) -> Path:
    return ecmwf_dir / case_key / f"{case_key}_ecmwf_week4_processed.nc"


def _load_ecmwf_members(ecmwf_path: Path, target_lats: np.ndarray,
                        target_lons: np.ndarray, is_t2m: bool) -> tuple[np.ndarray, str]:
    import xarray as xr

    with xr.open_dataset(ecmwf_path) as ds:
        ds = _sort_lon_lat_dataset(ds)
        member = ds["member_weekly_mean"].interp(
            lat=np.asarray(target_lats, dtype=float),
            lon=np.asarray(target_lons, dtype=float),
            method="linear",
        ).values.astype(np.float64)
        units = str(ds["member_weekly_mean"].attrs.get("units") or ds.attrs.get("units") or "")
    finite = member[np.isfinite(member)]
    if is_t2m and finite.size and np.nanmedian(finite) < 150.0:
        member = member + 273.15
        units = "K"
    return member, units


def _reference_brier_from_existing(fields: dict[str, np.ndarray],
                                   obs_event: np.ndarray) -> np.ndarray:
    ref_candidates = []
    for prefix in ("geos", "model"):
        prob_name = f"{prefix}_prob"
        bss_name = f"{prefix}_bss"
        if prob_name not in fields or bss_name not in fields:
            continue
        prob = fields[prob_name]
        bss = fields[bss_name]
        brier = (prob - obs_event) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = np.where(np.abs(1.0 - bss) > 1e-8, brier / (1.0 - bss), np.nan)
        ref_candidates.append(ref)
    if not ref_candidates:
        return np.full_like(obs_event, np.nan, dtype=float)
    ref_out = ref_candidates[0].copy()
    for ref in ref_candidates[1:]:
        ref_out = np.where(np.isfinite(ref_out), ref_out, ref)
    return ref_out


def _overlay_raw_contours(ax, lons, lats, model_raw, baseline_raw,
                          baseline_color: str, baseline_label: str,
                          raw_label: str) -> None:
    import cartopy.crs as ccrs

    finite = np.concatenate([
        np.ravel(np.asarray(model_raw)[np.isfinite(model_raw)]),
        np.ravel(np.asarray(baseline_raw)[np.isfinite(baseline_raw)]),
    ])
    if finite.size < 4:
        return
    vmin, vmax = np.nanpercentile(finite, [15, 85])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return
    levels = np.linspace(vmin, vmax, 4)
    kwargs = {"transform": ccrs.PlateCarree(), "linewidths": 0.75, "alpha": 0.90}
    ax.contour(lons, lats, model_raw, levels=levels, colors=C_MODEL, linestyles="solid", **kwargs)
    ax.contour(lons, lats, baseline_raw, levels=levels, colors=baseline_color, linestyles="dashed", **kwargs)
    handles = [
        Line2D([0], [0], color=C_MODEL, lw=1.0, linestyle="solid", label=f"{METHOD} {raw_label}"),
        Line2D([0], [0], color=baseline_color, lw=1.0, linestyle="dashed", label=f"{baseline_label} {raw_label}"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=5.8, frameon=True,
              framealpha=0.72, facecolor="white", edgecolor="none")


def _centered_norm_for_field(field: np.ndarray, fallback: float = 1.0) -> TwoSlopeNorm:
    finite = np.asarray(field, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [2, 98])
        lim = max(abs(float(lo)), abs(float(hi)), 1e-6)
    else:
        lim = float(fallback)
    return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)


def _skill_with_contours(fig, ax, lons, lats, skill_field, title: str,
                         raw_model: np.ndarray, raw_baseline: np.ndarray,
                         baseline_color: str, baseline_label: str,
                         raw_label: str, vlim: float | None, cbar_label: str,
                         label_left: bool = False, label_bottom: bool = False):
    norm = _centered_norm_for_field(skill_field) if vlim is None else TwoSlopeNorm(
        vcenter=0.0, vmin=-vlim, vmax=vlim
    )
    mesh = regional_panel(
        ax,
        lons,
        lats,
        skill_field,
        title,
        CMAP_SKILL,
        norm=norm,
        label_left=label_left,
        label_bottom=label_bottom,
    )
    _overlay_raw_contours(ax, lons, lats, raw_model, raw_baseline,
                          baseline_color, baseline_label, raw_label)
    ticks = None if vlim is None else _linear_ticks(-vlim, vlim, 7)
    panel_colorbar(fig, mesh, ax, cbar_label, ticks=ticks)
    return mesh


def figure_event_case_ecmwf_comparison(output_dir: Path, formats: list[str], dpi: int,
                                       event_dir: Path, ecmwf_dir: Path,
                                       event_id: str, stem: str, is_t2m: bool) -> list[Path]:
    import cartopy.crs as ccrs

    case_key = ECMWF_CASE_KEYS[event_id]
    event_path = event_dir / "plots" / "spatial_maps" / f"{event_id}_lead4_spatial_data.nc"
    ecmwf_path = _ecmwf_processed_path(ecmwf_dir, case_key)
    fig = plt.figure(figsize=(13.8, 10.2))
    gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.18,
                          left=0.055, right=0.975, bottom=0.055, top=0.97)

    if not event_path.exists() or not ecmwf_path.exists():
        ax_big = fig.add_subplot(gs[:, :])
        missing = []
        if not event_path.exists():
            missing.append(f"event NetCDF {event_path.name}")
        if not ecmwf_path.exists():
            missing.append(f"ECMWF NetCDF {ecmwf_path}")
        missing_panel(
            ax_big,
            f"{stem} pending",
            "Missing " + " and ".join(missing) + ". Run the event evaluator and "
            "paper/scripts/diagnose_ecmwf_event_gribs.py first.",
        )
        return save_figure(fig, output_dir, stem, formats, dpi)

    try:
        import xarray as xr

        with xr.open_dataset(event_path) as ds:
            lons = ds["lon"].values
            lats = ds["lat"].values
            obs_candidates = []
            if "obs" in ds:
                obs_candidates.append(ds["obs"].values)
            if "obs_plot" in ds:
                obs_candidates.append(ds["obs_plot"].values + (273.15 if is_t2m else 0.0))
            if not obs_candidates:
                raise KeyError("event NetCDF is missing obs/obs_plot")
            obs = max(obs_candidates, key=lambda arr: int(np.isfinite(arr).sum()))
            threshold = ds["threshold"].values
            fields = {
                "model_q95": ds["model_upper_quantile"].values,
                "model_crps": ds["model_crps"].values,
                "geos_crps": ds["geos_crps"].values,
                "model_bss": ds["model_bss"].values,
                "geos_bss": ds["geos_bss"].values,
                "model_prob": ds["model_prob"].values,
                "geos_prob": ds["geos_prob"].values,
            }
        ecmwf_members, ecmwf_units = _load_ecmwf_members(ecmwf_path, lats, lons, is_t2m)
    except Exception as exc:
        ax_big = fig.add_subplot(gs[:, :])
        missing_panel(ax_big, f"{stem} error", f"Could not load comparison data: {exc}")
        return save_figure(fig, output_dir, stem, formats, dpi)

    ecmwf_q95 = np.nanpercentile(ecmwf_members, 95, axis=0)
    ecmwf_crps = _crps_map_np(ecmwf_members, obs)
    valid_event = np.isfinite(obs) & np.isfinite(threshold)
    obs_event = np.where(valid_event, (obs >= threshold).astype(float), np.nan)
    ref_brier = _reference_brier_from_existing(fields, obs_event)
    valid_prob = np.isfinite(ecmwf_members) & np.isfinite(threshold[None, :, :])
    ecmwf_hits = np.where(valid_prob, (ecmwf_members >= threshold[None, :, :]).astype(float), np.nan)
    ecmwf_prob = np.nanmean(ecmwf_hits, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ecmwf_bss = np.where(ref_brier > 1e-12, 1.0 - ((ecmwf_prob - obs_event) ** 2) / ref_brier, np.nan)

    fields["ecmwf_q95"] = ecmwf_q95
    fields["ecmwf_crps"] = np.where(np.isfinite(obs), ecmwf_crps, np.nan)
    fields["ecmwf_bss"] = np.where(valid_event, ecmwf_bss, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        crps_skill_geos = 100.0 * (1.0 - fields["model_crps"] / fields["geos_crps"])
        crps_skill_ecmwf = 100.0 * (1.0 - fields["model_crps"] / fields["ecmwf_crps"])
    bss_gain_geos = 100.0 * (fields["model_bss"] - fields["geos_bss"])
    bss_gain_ecmwf = 100.0 * (fields["model_bss"] - fields["ecmwf_bss"])

    field_cmap = "RdYlBu_r" if is_t2m else "YlGnBu"
    phys_label = "temperature (K)" if is_t2m else "precipitation (mm day$^{-1}$)"
    if is_t2m and ecmwf_units and ecmwf_units != "K":
        phys_label += f"; ECMWF source {ecmwf_units}"

    axes = np.asarray([[fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
                        for c in range(3)] for r in range(3)])
    for ax in axes.ravel():
        ax.set_facecolor("white")
        ax.patch.set_facecolor("white")

    top_panels = [
        ("(a) Observed", obs),
        (f"(b) {METHOD} q95", fields["model_q95"]),
        ("(c) ECMWF q95", fields["ecmwf_q95"]),
    ]
    for idx, (title, field) in enumerate(top_panels):
        mesh = regional_panel(axes[0, idx], lons, lats, field, title, field_cmap,
                              label_left=(idx == 0))
        panel_colorbar(fig, mesh, axes[0, idx], phys_label)

    _skill_with_contours(
        fig, axes[1, 0], lons, lats, crps_skill_geos,
        f"(d) CRPS skill vs {BASELINE}",
        fields["model_crps"], fields["geos_crps"], C_BASELINE, BASELINE,
        "CRPS", None, "skill (%)", label_left=True,
    )
    _skill_with_contours(
        fig, axes[1, 1], lons, lats, crps_skill_ecmwf,
        "(e) CRPS skill vs ECMWF",
        fields["model_crps"], fields["ecmwf_crps"], "#2b8a3e", "ECMWF",
        "CRPS", None, "skill (%)",
    )
    mesh = regional_panel(axes[1, 2], lons, lats, fields["ecmwf_crps"], "(f) ECMWF CRPS",
                          "YlOrBr")
    panel_colorbar(fig, mesh, axes[1, 2], "CRPS")

    _skill_with_contours(
        fig, axes[2, 0], lons, lats, bss_gain_geos,
        f"(g) BSS gain vs {BASELINE}",
        fields["model_bss"], fields["geos_bss"], C_BASELINE, BASELINE,
        "BSS", None, "gain (x100)", label_left=True, label_bottom=True,
    )
    _skill_with_contours(
        fig, axes[2, 1], lons, lats, bss_gain_ecmwf,
        "(h) BSS gain vs ECMWF",
        fields["model_bss"], fields["ecmwf_bss"], "#2b8a3e", "ECMWF",
        "BSS", None, "gain (x100)", label_bottom=True,
    )
    mesh = regional_panel(axes[2, 2], lons, lats, fields["ecmwf_bss"], "(i) ECMWF BSS",
                          "Blues", label_bottom=True)
    panel_colorbar(fig, mesh, axes[2, 2], "BSS")

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
    ecmwf_dir = first_existing_dir(args.ecmwf_dir, DEFAULT_ECMWF_DIR_CANDIDATES)
    ensemble_dir = first_existing_dir(args.ensemble_dir, DEFAULT_ENSEMBLE_DIR_CANDIDATES)
    noise_csv = Path(args.noise_csv) if args.noise_csv else newest_matching(NOISE_CSV_PATTERNS)

    print("Figure input locations")
    print(f"  matrix_dir   : {matrix_dir}")
    print(f"  event_dir    : {event_dir}")
    print(f"  ecmwf_dir    : {ecmwf_dir}")
    print(f"  ensemble_dir : {ensemble_dir}")
    print(f"  noise_csv    : {noise_csv}")
    print(f"  output_dir   : {output_dir}")

    summary = read_csv_or_none(matrix_dir / "matrix_summary_metrics.csv")

    written: list[Path] = []
    if args.ecmwf_only:
        written.extend(figure_event_case_ecmwf_comparison(
            output_dir, formats, args.dpi, event_dir, ecmwf_dir,
            EVENT_PR_ID, "fig7a_event_pr_california_ecmwf", is_t2m=False))
        written.extend(figure_event_case_ecmwf_comparison(
            output_dir, formats, args.dpi, event_dir, ecmwf_dir,
            EVENT_T2M_ID, "fig8a_event_t2m_uk_heatwave_ecmwf", is_t2m=True))
        print(f"\nWrote {len(written)} figure files:")
        for path in written:
            print(f"  - {path}")
        return

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
                                     EVENT_PR_ID, "fig8_event_pr_california", is_t2m=False))
    written.extend(figure_event_case_ecmwf_comparison(
        output_dir, formats, args.dpi, event_dir, ecmwf_dir,
        EVENT_PR_ID, "fig7a_event_pr_california_ecmwf", is_t2m=False))
    written.extend(figure_event_case(output_dir, formats, args.dpi, event_dir,
                                     EVENT_T2M_ID, "fig9_event_t2m_uk_heatwave", is_t2m=True))

    written.extend(figure_event_case_ecmwf_comparison(
        output_dir, formats, args.dpi, event_dir, ecmwf_dir,
        EVENT_T2M_ID, "fig8a_event_t2m_uk_heatwave_ecmwf", is_t2m=True))

    print(f"\nWrote {len(written)} figure files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
