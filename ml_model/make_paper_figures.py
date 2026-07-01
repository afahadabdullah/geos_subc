#!/usr/bin/env python3
"""Build manuscript figures from GEOS-FlowMatch evaluation products.

This script is intentionally downstream of the expensive evaluators. It reads
the CSV/NetCDF products written by:

  - ml_model/evaluate_matrix_suite_flow_finalv1_global.py
  - ml_model/compare_noise_flow_finalv1_global.py
  - ml_model/compare_checkpoints_flow_finalv1_global.py
  - ml_model/evaluate_event_catalog_flow_finalv1_global.py
  - ml_model/evaluate_event_quantile_forecast_flow_finalv1_global.py

When an expected evaluation artifact is missing, the relevant panel is rendered
as a clear missing-data note so the full figure set can still be regenerated
while final runs are pending.
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


METHOD = "GEOS-FlowMatch"
BASELINE = "GEOS"
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
}

PRIMARY_EVENT_IDS = [
    "europe_t2m_202207_uk_heatwave",
    "bangladesh_pr_202206_meghalaya_sylhet_downpours",
    "bangladesh_pr_202206_sylhet_floods",
]
PRIMARY_EVENT_LABELS = {
    "europe_t2m_202207_uk_heatwave": "UK July 2022 heatwave (T2M)",
    "bangladesh_pr_202206_meghalaya_sylhet_downpours": "Bangladesh/Sylhet June 2022 flood (PR)",
    "bangladesh_pr_202206_sylhet_floods": "Bangladesh/Sylhet June 2022 flood (PR)",
}

COLOR_GEOS = "#b43c30"
COLOR_MODEL = "#202124"
COLOR_PR = "#2a6fbb"
COLOR_T2M = "#b07021"
COLOR_POS = "#2c7a4b"
COLOR_NEG = "#a33a3a"
TEXT_DARK = "#1f2933"
TEXT_MUTED = "#5b6770"
BOX_EDGE = "#2d4658"

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
CHECKPOINT_CSV_PATTERNS = [
    "ml_output_checkpoint_compare_global_flow_finalv1/checkpoint_pure_noise_global_*_summary.csv",
    "ml_output_flow_finalv1_global_noisectx_t2mres/checkpoint_pure_noise_global_*_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures from existing evaluation outputs."
    )
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--matrix-dir", default=None, help="Directory containing matrix_summary_metrics.csv.")
    parser.add_argument("--event-dir", default=None, help="Directory containing event_selected_lead_metrics.csv.")
    parser.add_argument("--quantile-dir", default=None, help="Directory containing event quantile CSV products.")
    parser.add_argument("--noise-csv", default=None, help="Optional explicit noise comparison CSV.")
    parser.add_argument("--checkpoint-csv", default=None, help="Optional explicit checkpoint sweep summary CSV.")
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "both"),
        default="pdf",
        help="Figure file format to write.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--matrix-subset", default="all_data", choices=("all_data", "extreme_events"))
    parser.add_argument("--spatial-subset", default="all_data", choices=("all_data", "extreme_events"))
    parser.add_argument(
        "--spatial-leads",
        default="2,3,4",
        help="Comma-separated lead weeks for the main spatial atlas. Default follows the manuscript text.",
    )
    parser.add_argument("--event-limit", type=int, default=8, help="Maximum event rows shown in Figure 7.")
    parser.add_argument(
        "--primary-event-ids",
        default="europe_t2m_202207_uk_heatwave,bangladesh_pr_202206_meghalaya_sylhet_downpours",
        help="Comma-separated event IDs for Figure 7 rows.",
    )
    parser.add_argument(
        "--write-legacy-aliases",
        action="store_true",
        help="Also write old placeholder-style names for figures 2, 4, and 6.",
    )
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


def parse_int_list(text: str | None, default: list[int]) -> list[int]:
    if text is None or str(text).strip() == "":
        return list(default)
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or list(default)


def parse_str_list(text: str | None, default: list[str]) -> list[str]:
    if text is None or str(text).strip() == "":
        return list(default)
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    return values or list(default)


def wrap(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def clean_label(value: object) -> str:
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def matrix_summary_path(matrix_dir: Path) -> Path:
    return matrix_dir / "matrix_summary_metrics.csv"


def matrix_spatial_path(matrix_dir: Path) -> Path:
    return matrix_dir / "matrix_spatial_metrics.nc"


def calibration_path(matrix_dir: Path) -> Path:
    return matrix_dir / "bss_calibration_params.csv"


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


def skill_pct(model_value: float, geos_value: float) -> float:
    if not np.isfinite(model_value) or not np.isfinite(geos_value) or abs(geos_value) <= 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - model_value / geos_value))


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
    draw_box(ax, (0.03, 0.42), (0.22, 0.17), "GEOS ensemble guidance",
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
        "GEOS stats, observed predictors, static geography, season, lead.",
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


def plot_metric_lines(ax: plt.Axes, data: pd.DataFrame, variable: str, metric: str) -> None:
    model_col = f"model_{metric}"
    geos_col = f"geos_{metric}"
    if data.empty or model_col not in data or geos_col not in data:
        missing_panel(ax, f"{VARIABLE_SHORT[variable]} {metric.upper()}", "Missing scalar matrix summary rows.")
        return
    sub = data[data["variable"].eq(variable)].sort_values("lead")
    if sub.empty:
        missing_panel(ax, f"{VARIABLE_SHORT[variable]} {metric.upper()}", f"No rows for variable {variable}.")
        return
    x = sub["lead"].to_numpy(dtype=float)
    ax.plot(x, sub[geos_col], marker="o", linewidth=1.8, color=COLOR_GEOS, label=BASELINE)
    ax.plot(x, sub[model_col], marker="o", linewidth=1.8, color=COLOR_MODEL, label=METHOD)
    style_axis(ax)
    ax.set_xticks(LEADS)
    ax.set_xlabel("Lead week", fontsize=8.5)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.upper()), fontsize=8.5)
    ax.set_title(f"{VARIABLE_SHORT[variable]} {metric.upper()}", loc="left", fontsize=10, fontweight="bold")
    skill_col = f"{metric}_skill_pct"
    if skill_col in sub:
        for _, row in sub.iterrows():
            value = row.get(skill_col, np.nan)
            if np.isfinite(value):
                ax.annotate(
                    f"{value:+.0f}%",
                    (float(row["lead"]), float(row[model_col])),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=COLOR_POS if value >= 0 else COLOR_NEG,
                )


def figure_3_lead_skill(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    summary: pd.DataFrame | None,
    subset: str,
) -> list[Path]:
    agg = aggregate_matrix_by_lead(summary, subset=subset)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6), sharex=True)
    style_figure(
        fig,
        "Figure 3. Lead-dependent skill",
        f"Scalar metrics from matrix_summary_metrics.csv, subset={subset}. Point labels show ML skill vs GEOS.",
    )
    for ax, variable, metric in [
        (axes[0, 0], "pr", "crps"),
        (axes[0, 1], "pr", "rmse"),
        (axes[1, 0], "t2m", "crps"),
        (axes[1, 1], "t2m", "rmse"),
    ]:
        plot_metric_lines(ax, agg, variable, metric)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.965), frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return save_figure(fig, output_dir, "fig3_lead_skill", formats, dpi)


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


def plot_skill_heatmap(ax: plt.Axes, arr: np.ndarray | None, title: str, vmin: float, vmax: float) -> None:
    if arr is None or not np.isfinite(arr).any():
        missing_panel(ax, title, "Missing valid_season_lead matrix rows.")
        return
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


def figure_4_season_lead(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    summary: pd.DataFrame | None,
    subset: str,
) -> list[Path]:
    panels = [
        ("pr", "crps_skill_pct", "PR CRPS skill"),
        ("pr", "rmse_skill_pct", "PR RMSE skill"),
        ("t2m", "crps_skill_pct", "T2M CRPS skill"),
        ("t2m", "rmse_skill_pct", "T2M RMSE skill"),
    ]
    arrays = [season_lead_values(summary, var, metric, subset) for var, metric, _ in panels]
    vmin, vmax = heatmap_limits(arrays)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    style_figure(
        fig,
        "Figure 4. Season-by-lead skill matrices",
        f"Positive values indicate lower ML error than raw GEOS; subset={subset}.",
    )
    im = None
    for ax, arr, (_, _, title) in zip(axes.ravel(), arrays, panels):
        maybe_im = plot_skill_heatmap(ax, arr, title, vmin, vmax)
        if maybe_im is not None:
            im = maybe_im
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
        cbar.set_label("Skill vs GEOS (%)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 0.94, 0.91])
    return save_figure(fig, output_dir, "fig4_season_lead_matrices", formats, dpi)


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


def spatial_metric_map(
    ds,
    variable: str,
    metric_name: str,
    subset: str,
    lead_values: list[int] | None = None,
    group_values: list[str] | None = None,
):
    if ds is None:
        return None, None, None
    needed_dims = {"subset", "variable", "group_type", "group_value", "lead", "lat", "lon"}
    if not needed_dims <= set(ds.dims):
        return None, None, None
    lead_values = lead_values or LEADS
    group_values = group_values or SEASONS
    selector = {
        "subset": coord_values_for(ds, "subset", [subset]),
        "variable": coord_values_for(ds, "variable", [variable]),
        "group_type": coord_values_for(ds, "group_type", ["valid_season_lead"]),
        "group_value": coord_values_for(ds, "group_value", group_values),
        "lead": coord_values_for(ds, "lead", lead_values),
    }
    if any(not values for values in selector.values()):
        return None, None, None
    sel = {dim: values for dim, values in selector.items()}
    count = ds["sample_count"].sel(sel)
    reduce_dims = ["group_value"]
    if len(selector["lead"]) > 1:
        reduce_dims.append("lead")
    if metric_name in ("crps_skill_pct", "rmse_skill_pct", "mae_skill_pct"):
        metric = metric_name.replace("_skill_pct", "")
        model = weighted_dataarray_mean(ds[f"model_{metric}"].sel(sel), count, reduce_dims)
        geos = weighted_dataarray_mean(ds[f"geos_{metric}"].sel(sel), count, reduce_dims)
        field = 100.0 * (1.0 - model / geos)
    elif metric_name == "bias_change":
        model = weighted_dataarray_mean(ds["model_bias"].sel(sel), count, reduce_dims)
        geos = weighted_dataarray_mean(ds["geos_bias"].sel(sel), count, reduce_dims)
        field = model - geos
    elif metric_name == "calibrated_bss_diff":
        field = weighted_dataarray_mean(ds["calibrated_bss_diff"].sel(sel), count, reduce_dims)
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
    center_zero: bool = True,
):
    if lons is None or lats is None or field is None or not np.isfinite(field).any():
        missing_panel(ax, title, "Missing matrix_spatial_metrics.nc data for this map.")
        return None
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if center_zero else None
    mesh = ax.pcolormesh(lons, lats, field, shading="auto", cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    ax.grid(True, color="#d9dee3", linewidth=0.25, alpha=0.7)
    return mesh


def figure_5_spatial_skill(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    matrix_dir: Path,
    subset: str,
    spatial_leads: list[int],
) -> list[Path]:
    ds = load_xarray_dataset(matrix_spatial_path(matrix_dir))
    row_specs = [
        ("pr", "crps_skill_pct", "PR CRPS skill"),
        ("t2m", "crps_skill_pct", "T2M CRPS skill"),
        ("pr", "rmse_skill_pct", "PR RMSE skill"),
        ("t2m", "rmse_skill_pct", "T2M RMSE skill"),
    ]
    specs = [
        (variable, metric, f"{title} W{lead}", lead)
        for variable, metric, title in row_specs
        for lead in spatial_leads
    ]
    maps = [
        spatial_metric_map(ds, variable, metric, subset, lead_values=[lead])
        for variable, metric, _, lead in specs
    ]
    fields = [field for _, _, field in maps]
    vmin, vmax = spatial_limits(fields)

    fig, axes = plt.subplots(len(row_specs), len(spatial_leads), figsize=(3.6 * len(spatial_leads), 9.6))
    style_figure(
        fig,
        "Figure 5. Spatial skill maps",
        (
            "Valid-season pooled, lead-specific spatial skill from matrix_spatial_metrics.nc. "
            f"Positive means ML improves on GEOS; subset={subset}."
        ),
    )
    mesh = None
    for ax, (lons, lats, field), (_, _, title, _) in zip(np.asarray(axes).ravel(), maps, specs):
        maybe_mesh = plot_plain_map(ax, lons, lats, field, title, vmin, vmax, cmap="RdYlGn", center_zero=True)
        if maybe_mesh is not None:
            mesh = maybe_mesh
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=np.asarray(axes).ravel().tolist(), shrink=0.80, pad=0.02)
        cbar.set_label("Skill vs GEOS (%)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 0.94, 0.91])
    written = save_figure(fig, output_dir, "fig5_spatial_skill_maps", formats, dpi)

    written.extend(figure_5b_spatial_bias_event(output_dir, formats, dpi, ds, spatial_leads))
    if ds is not None:
        ds.close()
    return written


def figure_5b_spatial_bias_event(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    ds,
    spatial_leads: list[int],
) -> list[Path]:
    row_specs = [
        ("pr", "bias_change", "PR bias change", "all_data", "RdBu_r", "ML - GEOS bias"),
        ("t2m", "bias_change", "T2M bias change", "all_data", "RdBu_r", "ML - GEOS bias"),
        ("pr", "crps_skill_pct", "PR extreme CRPS skill", "extreme_events", "RdYlGn", "Skill vs GEOS (%)"),
        ("t2m", "calibrated_bss_diff", "T2M extreme calibrated BSS gain", "extreme_events", "RdYlGn", "ML - GEOS"),
    ]
    specs = [
        (variable, metric, f"{title} W{lead}", subset, cmap, label, lead)
        for variable, metric, title, subset, cmap, label in row_specs
        for lead in spatial_leads
    ]
    maps = [
        spatial_metric_map(ds, variable, metric, subset, lead_values=[lead])
        for variable, metric, _, subset, _, _, lead in specs
    ]
    fields_by_label: dict[str, list[np.ndarray | None]] = {}
    for (_, _, _, _, _, label, _), (_, _, field) in zip(specs, maps):
        fields_by_label.setdefault(label, []).append(field)
    limits_by_label = {
        label: spatial_limits(fields, fallback=1.0 if "bias" in label.lower() else 30.0)
        for label, fields in fields_by_label.items()
    }

    fig, axes = plt.subplots(len(row_specs), len(spatial_leads), figsize=(3.6 * len(spatial_leads), 9.6))
    style_figure(
        fig,
        "Figure 5b. Bias and event-subset spatial diagnostics",
        "Companion atlas for bias changes and observed-extreme verification maps requested in the manuscript.",
    )
    meshes = {}
    for ax, (lons, lats, field), (_, _, title, _, cmap, label, _) in zip(np.asarray(axes).ravel(), maps, specs):
        vmin, vmax = limits_by_label[label]
        maybe_mesh = plot_plain_map(ax, lons, lats, field, title, vmin, vmax, cmap=cmap, center_zero=True)
        if maybe_mesh is not None and label not in meshes:
            meshes[label] = maybe_mesh
    if meshes:
        first_mesh = next(iter(meshes.values()))
        cbar = fig.colorbar(first_mesh, ax=np.asarray(axes).ravel().tolist(), shrink=0.80, pad=0.02)
        cbar.set_label("Centered diagnostic value", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(rect=[0, 0, 0.94, 0.91])
    return save_figure(fig, output_dir, "fig5b_spatial_bias_event_maps", formats, dpi)


def plot_noise_ablation(ax: plt.Axes, noise_df: pd.DataFrame | None) -> None:
    if noise_df is None or noise_df.empty or not {"strategy", "lead", "pr_crps", "t2m_crps"} <= set(noise_df.columns):
        missing_panel(
            ax,
            "Sampling ablation",
            "Missing noise comparison CSV. Run compare_noise_flow_finalv1_global.py.",
        )
        return
    df = noise_df[noise_df["lead"].astype(str).eq("total")].copy()
    if df.empty:
        df = noise_df.copy()
    df = df.head(6)
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df["pr_crps"], width, label="PR", color=COLOR_PR, alpha=0.85)
    ax.bar(x + width / 2, df["t2m_crps"], width, label="T2M", color=COLOR_T2M, alpha=0.85)
    ax.set_xticks(x, [wrap(clean_label(v), 14) for v in df["strategy"]], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("CRPS", fontsize=8.5)
    ax.set_title("Sampling ablation", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7)


def plot_spread_skill(ax: plt.Axes, lead_df: pd.DataFrame) -> None:
    needed = {"variable", "lead", "model_spread", "geos_spread", "model_rmse", "geos_rmse"}
    if lead_df.empty or not needed <= set(lead_df.columns):
        missing_panel(ax, "Spread-error ratio", "Missing spread/RMSE columns in matrix summary.")
        return
    for variable, color, marker in [("pr", COLOR_PR, "o"), ("t2m", COLOR_T2M, "s")]:
        sub = lead_df[lead_df["variable"].eq(variable)].sort_values("lead")
        if sub.empty:
            continue
        model_ratio = sub["model_spread"].to_numpy(float) / sub["model_rmse"].to_numpy(float)
        geos_ratio = sub["geos_spread"].to_numpy(float) / sub["geos_rmse"].to_numpy(float)
        ax.plot(sub["lead"], geos_ratio, color=color, linestyle="--", marker=marker, label=f"{VARIABLE_SHORT[variable]} GEOS")
        ax.plot(sub["lead"], model_ratio, color=color, linestyle="-", marker=marker, label=f"{VARIABLE_SHORT[variable]} ML")
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_xticks(LEADS)
    ax.set_xlabel("Lead week", fontsize=8.5)
    ax.set_ylabel("Spread / RMSE", fontsize=8.5)
    ax.set_title("Spread-error ratio", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)


def plot_bss_gain(ax: plt.Axes, lead_df: pd.DataFrame) -> None:
    if lead_df.empty or not {"variable", "lead", "calibrated_bss_diff"} <= set(lead_df.columns):
        missing_panel(ax, "Calibrated BSS gain", "Missing calibrated BSS columns in matrix summary.")
        return
    for variable, color, marker in [("pr", COLOR_PR, "o"), ("t2m", COLOR_T2M, "s")]:
        sub = lead_df[lead_df["variable"].eq(variable)].sort_values("lead")
        if sub.empty:
            continue
        ax.plot(sub["lead"], sub["calibrated_bss_diff"], color=color, marker=marker, linewidth=1.8,
                label=VARIABLE_SHORT[variable])
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_xticks(LEADS)
    ax.set_xlabel("Lead week", fontsize=8.5)
    ax.set_ylabel("ML - GEOS", fontsize=8.5)
    ax.set_title("Calibrated BSS gain", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7)


def plot_checkpoint_sweep(ax: plt.Axes, checkpoint_df: pd.DataFrame | None) -> None:
    if checkpoint_df is None or checkpoint_df.empty:
        missing_panel(
            ax,
            "Checkpoint sweep",
            "Missing checkpoint sweep CSV. Run compare_checkpoints_flow_finalv1_global.py.",
        )
        return
    df = checkpoint_df.copy()
    if "lead" in df:
        total = df[df["lead"].astype(str).eq("total")]
        if not total.empty:
            df = total
    metric = "combined_crps" if "combined_crps" in df else "pr_crps" if "pr_crps" in df else None
    if metric is None:
        missing_panel(ax, "Checkpoint sweep", "Checkpoint CSV lacks combined_crps/pr_crps.")
        return
    if "checkpoint_epoch" in df:
        df["_epoch"] = pd.to_numeric(df["checkpoint_epoch"], errors="coerce")
        df = df.sort_values(["_epoch", metric], na_position="last")
        labels = [str(int(v)) if np.isfinite(v) else clean_label(c)[:12] for v, c in zip(df["_epoch"], df["checkpoint"])]
    else:
        df = df.sort_values(metric)
        labels = [clean_label(v)[:12] for v in df.get("checkpoint", np.arange(len(df)))]
    df = df.head(12)
    ax.plot(np.arange(len(df)), df[metric], marker="o", color=COLOR_MODEL, linewidth=1.8)
    ax.set_xticks(np.arange(len(df)), labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=8.5)
    ax.set_title("Checkpoint sweep", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)


def plot_calibration_params(ax: plt.Axes, calibration_df: pd.DataFrame | None) -> None:
    if calibration_df is None or calibration_df.empty or not {"variable", "source", "lead", "slope"} <= set(calibration_df.columns):
        missing_panel(
            ax,
            "Reliability calibration",
            "Missing bss_calibration_params.csv. The evaluator writes this during logistic_cv BSS calibration.",
        )
        return
    df = calibration_df.copy()
    df = df[df["source"].astype(str).isin(["model", "geos"])]
    for variable, color, marker in [("pr", COLOR_PR, "o"), ("t2m", COLOR_T2M, "s")]:
        for source, linestyle in [("model", "-"), ("geos", "--")]:
            sub = df[df["variable"].eq(variable) & df["source"].eq(source)]
            if sub.empty:
                continue
            grouped = sub.groupby("lead")["slope"].mean().reset_index()
            label = f"{VARIABLE_SHORT[variable]} {'ML' if source == 'model' else 'GEOS'}"
            ax.plot(grouped["lead"], grouped["slope"], color=color, linestyle=linestyle, marker=marker, label=label)
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_xticks(LEADS)
    ax.set_xlabel("Lead week", fontsize=8.5)
    ax.set_ylabel("Logistic calibration slope", fontsize=8.5)
    ax.set_title("Reliability calibration", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=6.2, ncol=2)


def figure_6_probabilistic(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    summary: pd.DataFrame | None,
    subset: str,
    noise_df: pd.DataFrame | None,
    checkpoint_df: pd.DataFrame | None,
    calibration_df: pd.DataFrame | None,
) -> list[Path]:
    lead_df = aggregate_matrix_by_lead(summary, subset=subset)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.1))
    style_figure(
        fig,
        "Figure 6. Probabilistic diagnostics and ablations",
        "Ablation and calibration panels use whichever downstream evaluator products are available.",
    )
    plot_noise_ablation(axes[0, 0], noise_df)
    plot_spread_skill(axes[0, 1], lead_df)
    plot_bss_gain(axes[0, 2], lead_df)
    plot_calibration_params(axes[1, 0], calibration_df)
    missing_panel(
        axes[1, 1],
        "Rank/PIT diagnostic",
        "Current matrix evaluator does not write member-rank or PIT histograms. Add this product to populate the panel.",
    )
    plot_checkpoint_sweep(axes[1, 2], checkpoint_df)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return save_figure(fig, output_dir, "fig6_probabilistic_diagnostics", formats, dpi)


def choose_event_rows(event_df: pd.DataFrame | None, limit: int) -> pd.DataFrame:
    if event_df is None or event_df.empty:
        return pd.DataFrame()
    df = event_df.copy()
    if "selection_mode" in df:
        selected = df[df["selection_mode"].astype(str).str.contains("selected|event", case=False, na=False)]
        if not selected.empty:
            df = selected
    sort_cols = [col for col in ["variable", "region", "event_id", "lead"] if col in df]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.head(max(1, int(limit))).reset_index(drop=True)


def event_axis_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in df.iterrows():
        name = row.get("event_name", row.get("event_id", "event"))
        lead = row.get("lead", "")
        variable = VARIABLE_SHORT.get(str(row.get("variable", "")).lower(), str(row.get("variable", "")).upper())
        labels.append(wrap(f"{variable} W{lead} {name}", 22))
    return labels


def plot_event_probability(ax: plt.Axes, rows: pd.DataFrame) -> None:
    cols = {"model_event_probability_on_obs_extreme", "geos_event_probability_on_obs_extreme"}
    if rows.empty or not cols <= set(rows.columns):
        missing_panel(ax, "Probability on observed extremes", "Missing event_selected_lead_metrics.csv.")
        return
    y = np.arange(len(rows))
    ax.barh(y - 0.18, rows["geos_event_probability_on_obs_extreme"], height=0.32, color=COLOR_GEOS, label=BASELINE)
    ax.barh(y + 0.18, rows["model_event_probability_on_obs_extreme"], height=0.32, color=COLOR_MODEL, label=METHOD)
    ax.set_yticks(y, event_axis_labels(rows), fontsize=6.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean exceedance probability", fontsize=8.5)
    ax.set_title("Probability on observed extremes", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    style_axis(ax)


def plot_event_gain_bars(ax: plt.Axes, rows: pd.DataFrame, col: str, title: str, xlabel: str) -> None:
    if rows.empty or col not in rows:
        missing_panel(ax, title, f"Missing {col} in event metrics.")
        return
    y = np.arange(len(rows))
    values = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=float)
    colors = [COLOR_POS if value >= 0 else COLOR_NEG for value in values]
    ax.barh(y, values, color=colors, alpha=0.86)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, event_axis_labels(rows), fontsize=6.7)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)


def plot_event_skill_scatter(ax: plt.Axes, rows: pd.DataFrame) -> None:
    needed = {"crps_skill_pct", "rmse_skill_pct", "variable"}
    if rows.empty or not needed <= set(rows.columns):
        missing_panel(ax, "Event CRPS/RMSE skill", "Missing event skill columns.")
        return
    for variable, color, marker in [("pr", COLOR_PR, "o"), ("t2m", COLOR_T2M, "s")]:
        sub = rows[rows["variable"].eq(variable)]
        if sub.empty:
            continue
        ax.scatter(sub["crps_skill_pct"], sub["rmse_skill_pct"], s=42, color=color, marker=marker,
                   label=VARIABLE_SHORT[variable], alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("CRPS skill (%)", fontsize=8.5)
    ax.set_ylabel("RMSE skill (%)", fontsize=8.5)
    ax.set_title("Event CRPS/RMSE skill", loc="left", fontsize=10, fontweight="bold")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7)


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


def event_plot_sort_key(path: Path) -> tuple[int, str]:
    text = path.name
    match = re.search(r"lead(\d+)", text)
    lead = int(match.group(1)) if match else 99
    lead_order = {3: 0, 4: 1, 2: 2, 1: 3}.get(lead, 9)
    return lead_order, text


def plot_index_candidates(
    plot_index: pd.DataFrame | None,
    event_id: str,
    plot_types: list[str],
    base_dir: Path,
) -> list[Path]:
    if plot_index is None or plot_index.empty or not {"event_id", "plot_type", "path"} <= set(plot_index.columns):
        return []
    subset = plot_index[
        plot_index["event_id"].astype(str).eq(event_id)
        & plot_index["plot_type"].astype(str).isin(plot_types)
    ]
    paths = []
    for value in subset["path"]:
        path = resolve_recorded_plot_path(value, base_dir)
        if path is not None:
            paths.append(path)
    return sorted(paths, key=event_plot_sort_key)


def find_primary_event_plot(
    event_id: str,
    event_dir: Path,
    quantile_dir: Path,
    event_plot_index: pd.DataFrame | None,
    quantile_plot_index: pd.DataFrame | None,
) -> Path | None:
    candidates = []
    candidates.extend(plot_index_candidates(quantile_plot_index, event_id, ["quantile_spatial"], quantile_dir))
    candidates.extend(plot_index_candidates(event_plot_index, event_id, ["spatial_risk"], event_dir))
    candidates.extend(plot_index_candidates(event_plot_index, event_id, ["spatial_verification"], event_dir))
    candidates.extend(Path(path) for path in glob.glob(str(quantile_dir / "plots" / "quantile_spatial_maps" / f"{event_id}_*quantile_spatial.png")))
    candidates.extend(Path(path) for path in glob.glob(str(event_dir / "plots" / "spatial_maps" / f"{event_id}_lead*_spatial_risk.png")))
    candidates.extend(Path(path) for path in glob.glob(str(event_dir / "plots" / "spatial_maps" / f"{event_id}_lead*_spatial_verification.png")))
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=event_plot_sort_key)[0]


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


def mean_column(rows: pd.DataFrame, candidates: list[str]) -> tuple[str | None, float]:
    for col in candidates:
        if col in rows:
            values = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                return col, float(np.nanmean(values))
    return None, float("nan")


def format_pair_list(rows: pd.DataFrame, max_items: int = 5) -> str:
    if rows.empty or not {"init_time", "lead"} <= set(rows.columns):
        return "not available"
    pairs = []
    for _, row in rows.drop_duplicates(["init_time", "lead"]).iterrows():
        pairs.append(f"{row['init_time']} W{int(row['lead'])}")
    if len(pairs) > max_items:
        pairs = pairs[:max_items] + [f"+{len(pairs) - max_items} more"]
    return "\n".join(pairs) if pairs else "not available"


def plot_primary_event_summary(ax: plt.Axes, event_id: str, event_df: pd.DataFrame | None) -> None:
    rows = event_rows_for_id(event_df, event_id)
    ax.set_axis_off()
    ax.set_title("Selected init/lead summary", loc="left", fontsize=10, fontweight="bold")
    if rows.empty:
        missing_panel(ax, "Selected init/lead summary", "Missing event_selected_lead_metrics.csv rows for this primary event.")
        return

    _, crps_value = mean_column(rows, ["crps_on_obs_extreme_skill_pct", "crps_skill_pct"])
    _, bss_value = mean_column(rows, ["calibrated_bss_diff", "bss_diff"])
    _, prob_value = mean_column(
        rows,
        [
            "event_probability_neighborhood_on_obs_extreme_diff",
            "event_probability_on_obs_extreme_diff",
            "event_probability_neighborhood_top_tail_diff",
        ],
    )
    pair_cols = [col for col in ["init_time", "lead"] if col in rows]
    n_pairs = int(len(rows.drop_duplicates(pair_cols))) if pair_cols else int(len(rows))
    event_name = clean_label(rows.iloc[0].get("event_name", event_id))
    variable = VARIABLE_SHORT.get(str(rows.iloc[0].get("variable", "")).lower(), str(rows.iloc[0].get("variable", "")).upper())
    text = (
        f"{event_name}\n"
        f"Variable: {variable}\n"
        f"Selected pairs: {n_pairs}\n\n"
        f"Mean event-mask CRPS skill: {crps_value:+.2f}%\n"
        f"Mean calibrated BSS gain: {bss_value:+.3f}\n"
        f"Mean event-probability gain: {prob_value:+.3f}\n\n"
        f"Init/lead pairs:\n{format_pair_list(rows)}"
    )
    box = FancyBboxPatch(
        (0.03, 0.05),
        0.94,
        0.86,
        transform=ax.transAxes,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor="#f7f8fa",
        edgecolor="#a9b4bf",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(0.08, 0.84, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color=TEXT_DARK)


def plot_event_image(ax: plt.Axes, image_path: Path | None, event_id: str) -> None:
    ax.set_axis_off()
    label = PRIMARY_EVENT_LABELS.get(event_id, event_id)
    ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
    if image_path is None:
        missing_panel(
            ax,
            label,
            (
                "Missing spatial event map. Run evaluate_event_catalog_flow_finalv1_global.py "
                "or evaluate_event_quantile_forecast_flow_finalv1_global.py with --make_plots."
            ),
        )
        return
    try:
        image = plt.imread(image_path)
    except Exception as exc:
        missing_panel(ax, label, f"Could not read event map {image_path}: {exc}")
        return
    ax.imshow(image)
    ax.text(
        0.01,
        0.01,
        str(image_path),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
    )


def figure_7_extremes(
    output_dir: Path,
    formats: list[str],
    dpi: int,
    event_df: pd.DataFrame | None,
    event_dir: Path,
    quantile_dir: Path,
    event_plot_index: pd.DataFrame | None,
    quantile_plot_index: pd.DataFrame | None,
    primary_event_ids: list[str],
) -> list[Path]:
    primary_event_ids = primary_event_ids[:2] if primary_event_ids else PRIMARY_EVENT_IDS[:2]
    fig, axes = plt.subplots(
        len(primary_event_ids),
        2,
        figsize=(15.0, 5.3 * max(1, len(primary_event_ids))),
        gridspec_kw={"width_ratios": [3.4, 1.0]},
    )
    axes = np.asarray(axes).reshape(len(primary_event_ids), 2)
    style_figure(
        fig,
        "Figure 7. Extreme-event tail-risk diagnostics",
        "Primary spatial case studies requested in main.tex: UK July 2022 T2M heatwave and Bangladesh/Sylhet June 2022 PR flood.",
    )
    for row_idx, event_id in enumerate(primary_event_ids):
        image_path = find_primary_event_plot(event_id, event_dir, quantile_dir, event_plot_index, quantile_plot_index)
        plot_event_image(axes[row_idx, 0], image_path, event_id)
        plot_primary_event_summary(axes[row_idx, 1], event_id, event_df)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save_figure(fig, output_dir, "fig7_extreme_tail_risk", formats, dpi)


def write_legacy_aliases(output_dir: Path, formats: list[str], dpi: int) -> None:
    aliases = {
        "fig2_architecture_sampling": "fig2_method_detail",
        "fig5_spatial_skill_maps": "fig4_spatial_maps",
        "fig6_probabilistic_diagnostics": "fig5_probabilistic_diagnostics",
    }
    for src_stem, dst_stem in aliases.items():
        for fmt in formats:
            src = output_dir / f"{src_stem}.{fmt}"
            dst = output_dir / f"{dst_stem}.{fmt}"
            if not src.exists():
                continue
            data = src.read_bytes()
            dst.write_bytes(data)


def main() -> None:
    args = parse_args()
    formats = output_formats(args.format)
    output_dir = Path(args.output_dir)
    spatial_leads = parse_int_list(args.spatial_leads, default=[2, 3, 4])
    primary_event_ids = parse_str_list(args.primary_event_ids, default=PRIMARY_EVENT_IDS[:2])
    matrix_dir = first_existing_dir(args.matrix_dir, DEFAULT_MATRIX_DIR_CANDIDATES)
    event_dir = first_existing_dir(args.event_dir, DEFAULT_EVENT_DIR_CANDIDATES)
    quantile_dir = first_existing_dir(args.quantile_dir, DEFAULT_QUANTILE_DIR_CANDIDATES)
    noise_csv = Path(args.noise_csv) if args.noise_csv else newest_matching(NOISE_CSV_PATTERNS)
    checkpoint_csv = Path(args.checkpoint_csv) if args.checkpoint_csv else newest_matching(CHECKPOINT_CSV_PATTERNS)

    print("Figure input locations")
    print(f"  matrix_dir     : {matrix_dir}")
    print(f"  event_dir      : {event_dir}")
    print(f"  quantile_dir   : {quantile_dir}")
    print(f"  noise_csv      : {noise_csv or 'not found'}")
    print(f"  checkpoint_csv : {checkpoint_csv or 'not found'}")
    print(f"  spatial_leads  : {spatial_leads}")
    print(f"  primary_events : {primary_event_ids}")
    print(f"  output_dir     : {output_dir}")

    summary = read_csv_or_none(matrix_summary_path(matrix_dir))
    noise_df = read_csv_or_none(noise_csv)
    checkpoint_df = read_csv_or_none(checkpoint_csv)
    calibration_df = read_csv_or_none(calibration_path(matrix_dir))
    event_df = read_csv_or_none(event_dir / "event_selected_lead_metrics.csv")
    event_plot_index = read_csv_or_none(event_dir / "event_plot_index.csv")
    quantile_plot_index = read_csv_or_none(quantile_dir / "event_quantile_plot_index.csv")

    written: list[Path] = []
    written.extend(figure_1_system_overview(output_dir, formats, args.dpi))
    written.extend(figure_2_architecture(output_dir, formats, args.dpi))
    written.extend(figure_3_lead_skill(output_dir, formats, args.dpi, summary, args.matrix_subset))
    written.extend(figure_4_season_lead(output_dir, formats, args.dpi, summary, args.matrix_subset))
    written.extend(figure_5_spatial_skill(output_dir, formats, args.dpi, matrix_dir, args.spatial_subset, spatial_leads))
    written.extend(
        figure_6_probabilistic(
            output_dir,
            formats,
            args.dpi,
            summary,
            args.matrix_subset,
            noise_df,
            checkpoint_df,
            calibration_df,
        )
    )
    written.extend(
        figure_7_extremes(
            output_dir,
            formats,
            args.dpi,
            event_df,
            event_dir,
            quantile_dir,
            event_plot_index,
            quantile_plot_index,
            primary_event_ids,
        )
    )

    if args.write_legacy_aliases:
        write_legacy_aliases(output_dir, formats, args.dpi)

    print("Wrote figures:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
