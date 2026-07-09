#!/usr/bin/env python3
"""Extreme-event ensemble-size BSS diagnostics.

This is a standalone companion to evaluate_ensemble_correlation_extremes_*
and evaluate_ensemble_tests_flow_finalv1_global.py. It reuses the same
regional extreme-event selection, then scores probabilistic q95 exceedance
forecasts as generated ensemble size increases.

BSS event definition:
  observation >= local observed threshold map selected by valid time.

The Brier reference is the local observed event frequency from the same
threshold source, matching the matrix/event-map evaluation scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import evaluate_ensemble_tests_flow_finalv1_global as ens
    import evaluate_matrix_suite_flow_finalv1_global as matrix_eval
except ModuleNotFoundError:  # Allows `python -m ml_model...` from repo root.
    from ml_model import evaluate_ensemble_tests_flow_finalv1_global as ens
    from ml_model import evaluate_matrix_suite_flow_finalv1_global as matrix_eval


DEFAULT_THRESHOLD_FILE = (
    "ml_output_flow_finalv1_global_noisectx_t2mres/"
    "matrix_eval_global_2021_2023_land_obsclim_chunked/"
    "event_thresholds_and_frequencies.nc"
)

LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}
SUM_COLUMNS = [
    "model_bs_sum",
    "geos_bs_sum",
    "ref_bs_sum",
    "weight_sum",
    "obs_event_weight_sum",
    "model_prob_weight_sum",
    "geos_prob_weight_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast_dir",
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Directory containing YEAR.zarr forecast stores.",
    )
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--skip_years", default="", help="Comma-separated years to skip.")
    parser.add_argument(
        "--out_dir",
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "ensemble_bss_extreme_t2m30_pr30_regions_2021_2023_w1w4_memberboot50"
        ),
    )
    parser.add_argument("--variables", default="pr,t2m", help="Comma-separated subset of pr,t2m.")
    parser.add_argument(
        "--sample_sizes",
        default="4,8,16,32,64,90",
        help="Comma-separated generated ensemble sizes. Zero is ignored because BSS needs probabilities.",
    )
    parser.add_argument("--member_bootstrap_repeats", type=int, default=50)
    parser.add_argument("--case_bootstrap_repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval_mask", choices=("all", "land", "ocean"), default="all")
    parser.add_argument("--land_mask_file", default=None, help="Optional .pt land mask for land/ocean masks.")
    parser.add_argument("--lead_values", default="1,2,3,4", help="Comma-separated lead weeks to keep.")
    parser.add_argument("--init_dates", default="", help="Optional comma-separated init dates, YYYY-MM-DD.")
    parser.add_argument("--valid_dates", default="", help="Optional comma-separated valid dates, YYYY-MM-DD.")
    parser.add_argument(
        "--init_months",
        default="",
        help="Optional init months or season aliases, e.g. 6,7,8 or JJA.",
    )
    parser.add_argument(
        "--threshold_file",
        default="",
        help=(
            "Observed threshold/frequency NetCDF. If omitted, the script uses the standard matrix "
            f"threshold file when present ({DEFAULT_THRESHOLD_FILE}); otherwise it builds thresholds "
            "from --threshold_forecast_dir over the selected threshold years."
        ),
    )
    parser.add_argument(
        "--threshold_forecast_dir",
        default=None,
        help="Forecast Zarr directory used only to build fallback observed thresholds; defaults to --forecast_dir.",
    )
    parser.add_argument("--threshold_start_year", type=int, default=None)
    parser.add_argument("--threshold_end_year", type=int, default=None)
    parser.add_argument("--threshold_skip_years", default="")
    parser.add_argument(
        "--threshold_grouping",
        choices=("pooled", "monthly", "seasonal"),
        default="monthly",
        help="Fallback threshold grouping if --threshold_file is unavailable.",
    )
    parser.add_argument("--extreme_quantile_pr", type=float, default=0.95)
    parser.add_argument("--extreme_quantile_t2m", type=float, default=0.95)
    parser.add_argument("--pr_min_threshold", type=float, default=5.0)
    parser.add_argument(
        "--extreme_event_count",
        type=int,
        default=30,
        help="Number of observed regional extremes to select.",
    )
    parser.add_argument(
        "--extreme_event_variable",
        default="t2m,pr",
        help="Observed variable(s) used to rank extremes, e.g. t2m, pr, or t2m,pr.",
    )
    parser.add_argument(
        "--extreme_event_regions",
        default="global_extremes",
        help="Comma-separated region names or a region set name such as heatwave/global_extremes/precip.",
    )
    parser.add_argument("--extreme_event_max_per_region", type=int, default=2)
    parser.add_argument(
        "--extreme_event_count_per_lead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select --extreme_event_count cases separately for each requested lead week.",
    )
    parser.add_argument("--allow_missing_years", action="store_true")
    parser.add_argument("--report_member_count", type=int, default=8)
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def timestamp_now_utc() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def resolve_threshold_file(args: argparse.Namespace) -> str | None:
    explicit = str(args.threshold_file or "").strip()
    if explicit:
        return explicit
    if os.path.exists(DEFAULT_THRESHOLD_FILE):
        return DEFAULT_THRESHOLD_FILE
    return None


def load_threshold_bundles(
    args: argparse.Namespace,
    years: list[int],
    variables: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], np.ndarray, np.ndarray]:
    args.threshold_file = resolve_threshold_file(args)
    if args.threshold_file:
        return matrix_eval.load_thresholds_from_file(args.threshold_file, variables, args)
    return matrix_eval.collect_obs_thresholds(args.forecast_dir, years, variables, args)


def bss_sums(
    ensemble: np.ndarray,
    obs: np.ndarray,
    threshold: np.ndarray,
    obs_event_freq: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(threshold)
        & np.isfinite(obs_event_freq)
    )
    if not np.any(valid):
        return {
            f"{prefix}_bs_sum": 0.0,
            "ref_bs_sum": 0.0,
            "weight_sum": 0.0,
            "obs_event_weight_sum": 0.0,
            f"{prefix}_prob_weight_sum": 0.0,
        }
    ens_valid = np.asarray(ensemble, dtype=np.float32)[:, valid]
    threshold_valid = np.asarray(threshold, dtype=np.float32)[valid]
    obs_valid = np.asarray(obs, dtype=np.float32)[valid]
    clim_valid = np.clip(np.asarray(obs_event_freq, dtype=np.float64)[valid], 0.0, 1.0)
    weights_valid = np.asarray(weights, dtype=np.float64)[valid]
    with np.errstate(invalid="ignore"):
        prob = np.nanmean(ens_valid >= threshold_valid[None, :], axis=0).astype(np.float64, copy=False)
    event = (obs_valid >= threshold_valid).astype(np.float64, copy=False)
    prob_finite = np.isfinite(prob)
    if not np.any(prob_finite):
        return {
            f"{prefix}_bs_sum": 0.0,
            "ref_bs_sum": 0.0,
            "weight_sum": 0.0,
            "obs_event_weight_sum": 0.0,
            f"{prefix}_prob_weight_sum": 0.0,
        }
    prob = prob[prob_finite]
    event = event[prob_finite]
    clim_valid = clim_valid[prob_finite]
    weights_valid = weights_valid[prob_finite]
    brier = (prob - event) ** 2
    ref = (clim_valid - event) ** 2
    return {
        f"{prefix}_bs_sum": float(np.sum(weights_valid * brier)),
        "ref_bs_sum": float(np.sum(weights_valid * ref)),
        "weight_sum": float(np.sum(weights_valid)),
        "obs_event_weight_sum": float(np.sum(weights_valid * event)),
        f"{prefix}_prob_weight_sum": float(np.sum(weights_valid * prob)),
    }


def row_metrics(row: dict[str, float] | pd.Series) -> dict[str, float]:
    out = dict(row)
    ref = float(out.get("ref_bs_sum", np.nan))
    weight = float(out.get("weight_sum", np.nan))
    model_bs = float(out.get("model_bs_sum", np.nan))
    geos_bs = float(out.get("geos_bs_sum", np.nan))
    out["model_bs"] = model_bs / weight if np.isfinite(weight) and weight > 0.0 else np.nan
    out["geos_bs"] = geos_bs / weight if np.isfinite(weight) and weight > 0.0 else np.nan
    out["ref_bs"] = ref / weight if np.isfinite(weight) and weight > 0.0 else np.nan
    out["model_bss"] = 1.0 - model_bs / ref if np.isfinite(ref) and ref > 1e-12 else np.nan
    out["geos_bss"] = 1.0 - geos_bs / ref if np.isfinite(ref) and ref > 1e-12 else np.nan
    out["bss_gain"] = out["model_bss"] - out["geos_bss"]
    out["bss_gain_x100"] = 100.0 * out["bss_gain"] if np.isfinite(out["bss_gain"]) else np.nan
    out["obs_event_fraction"] = (
        float(out.get("obs_event_weight_sum", np.nan)) / weight
        if np.isfinite(weight) and weight > 0.0
        else np.nan
    )
    out["model_event_probability"] = (
        float(out.get("model_prob_weight_sum", np.nan)) / weight
        if np.isfinite(weight) and weight > 0.0
        else np.nan
    )
    out["geos_event_probability"] = (
        float(out.get("geos_prob_weight_sum", np.nan)) / weight
        if np.isfinite(weight) and weight > 0.0
        else np.nan
    )
    return out


def aggregate_sums(case_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame()
    grouped = case_df.groupby(group_cols, dropna=False)
    sums = grouped[SUM_COLUMNS].sum(min_count=1).reset_index()
    sums["n_rows"] = grouped.size().to_numpy(dtype=int)
    sums["n_cases"] = grouped["case_id"].nunique().to_numpy(dtype=int)
    rows = [row_metrics(row) for row in sums.to_dict("records")]
    return pd.DataFrame(rows)


def summarize_member_repeats(repeat_summary: pd.DataFrame) -> pd.DataFrame:
    if repeat_summary.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "member_count"]
    metrics = (
        "model_bs",
        "geos_bs",
        "ref_bs",
        "model_bss",
        "geos_bss",
        "bss_gain",
        "bss_gain_x100",
        "obs_event_fraction",
        "model_event_probability",
        "geos_event_probability",
    )
    for key, group in repeat_summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["member_repeat_count"] = int(group["member_repeat"].nunique())
        row["n_cases"] = int(group["n_cases"].max())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_p05"] = np.nan
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p95"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_p05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            row[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def bootstrap_case_intervals(case_df: pd.DataFrame, repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    if repeats <= 0 or case_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "member_count"]
    metrics = ("model_bss", "geos_bss", "bss_gain", "bss_gain_x100")
    for key, group in case_df.groupby(group_cols, dropna=False):
        cases = sorted(group["case_id"].astype(str).unique())
        if not cases:
            continue
        rows_by_case = {
            case: group[group["case_id"].astype(str).eq(case)].reset_index(drop=True)
            for case in cases
        }
        boot = {metric: [] for metric in metrics}
        for _ in range(repeats):
            sampled_cases = rng.choice(cases, size=len(cases), replace=True)
            state = {col: 0.0 for col in SUM_COLUMNS}
            for case in sampled_cases:
                case_rows = rows_by_case[str(case)]
                picked = case_rows.iloc[int(rng.integers(0, len(case_rows)))]
                for col in SUM_COLUMNS:
                    state[col] += float(picked[col])
            scores = row_metrics(state)
            for metric in boot:
                boot[metric].append(scores[metric])
        row = dict(zip(group_cols, key))
        row["case_bootstrap_repeats"] = int(repeats)
        row["n_cases"] = int(len(cases))
        for metric, values in boot.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                row[f"{metric}_p025"] = np.nan
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p975"] = np.nan
                if metric == "bss_gain":
                    row["bss_gain_p_gt0"] = np.nan
                    row["bss_gain_p_le0"] = np.nan
                continue
            row[f"{metric}_p025"] = float(np.quantile(arr, 0.025))
            row[f"{metric}_p50"] = float(np.quantile(arr, 0.50))
            row[f"{metric}_p975"] = float(np.quantile(arr, 0.975))
            if metric == "bss_gain":
                row["bss_gain_p_gt0"] = float(np.mean(arr > 0.0))
                row["bss_gain_p_le0"] = float(np.mean(arr <= 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_member_report(summary: pd.DataFrame, out_dir: Path, member_count: int) -> str | None:
    if member_count <= 0 or summary.empty:
        return None
    sub = summary[summary["member_count"].astype(int).eq(int(member_count))].copy()
    if sub.empty:
        print(f"No BSS rows found for ensemble size {member_count}.")
        return None
    cols = [
        "variable",
        "lead",
        "member_count",
        "model_bss_mean",
        "geos_bss_mean",
        "bss_gain_x100_mean",
        "obs_event_fraction_mean",
        "model_event_probability_mean",
        "geos_event_probability_mean",
        "n_cases",
    ]
    sub = sub[[col for col in cols if col in sub.columns]].sort_values(["variable", "lead"])
    path = out_dir / f"ensemble_bss_member{int(member_count)}_report.csv"
    write_csv(sub, path)
    print(f"\nEnsemble size {int(member_count)} BSS report:")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(sub.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    return str(path)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _robust_ylim(values: list[float], fallback: tuple[float, float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    lo = float(np.nanquantile(arr, 0.02))
    hi = float(np.nanquantile(arr, 0.98))
    pad = max(0.08 * (hi - lo), 0.04 * max(abs(lo), abs(hi), 1.0))
    lo -= pad
    hi += pad
    if hi <= lo:
        hi = lo + max(0.1, abs(lo) * 0.1)
    return lo, hi


def plot_bss_dashboard(summary: pd.DataFrame, bootstrap: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    plt = _import_matplotlib()
    variables = [var for var in ("pr", "t2m") if var in set(summary["variable"].astype(str))]
    if not variables:
        variables = sorted(summary["variable"].astype(str).unique())

    fig, axes = plt.subplots(
        2,
        len(variables),
        figsize=(5.4 * len(variables), 5.8),
        sharex="col",
        gridspec_kw={"hspace": 0.25, "wspace": 0.18, "height_ratios": [1.08, 0.92]},
    )
    axes = np.asarray(axes).reshape(2, len(variables))
    letters = "abcdefghijklmnopqrstuvwxyz"
    bss_values: list[float] = []
    gain_values: list[float] = []

    def style_axis(ax, zero_line: bool = False) -> None:
        ax.grid(True, axis="y", color="#dde3e8", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if zero_line:
            ax.axhline(0.0, color="#7a8794", lw=0.9, ls="--")

    for vi, variable in enumerate(variables):
        sub = summary[summary["variable"].astype(str).eq(variable)]
        ax_bss = axes[0, vi]
        ax_gain = axes[1, vi]
        ax_bss.set_title(f"({letters[vi]}) {variable.upper()} q95-exceedance BSS", loc="left", fontweight="bold")
        ax_gain.set_title(
            f"({letters[len(variables) + vi]}) {variable.upper()} BSS gain (x100)",
            loc="left",
            fontweight="bold",
        )
        if sub.empty:
            for ax in (ax_bss, ax_gain):
                ax.text(0.5, 0.5, "No rows", transform=ax.transAxes, ha="center", va="center")
            continue
        for lead in sorted(sub["lead"].astype(int).unique()):
            line = sub[sub["lead"].astype(int).eq(lead)].sort_values("member_count")
            x = line["member_count"].to_numpy(dtype=float)
            color = LEAD_COLORS.get(int(lead), "#2f6f9f")
            geos_bss = line["geos_bss_mean"].to_numpy(dtype=float)
            model_bss = line["model_bss_mean"].to_numpy(dtype=float)
            bss_values.extend(geos_bss[np.isfinite(geos_bss)].tolist())
            bss_values.extend(model_bss[np.isfinite(model_bss)].tolist())
            ax_bss.plot(x, geos_bss, color=color, lw=1.15, ls="--", alpha=0.50)
            ax_bss.plot(x, model_bss, color=color, lw=1.85, marker="o", ms=3.5, label=f"W{int(lead)}")

            gain = line["bss_gain_x100_mean"].to_numpy(dtype=float)
            gain_values.extend(gain[np.isfinite(gain)].tolist())
            lo = line["bss_gain_x100_p05"].to_numpy(dtype=float)
            hi = line["bss_gain_x100_p95"].to_numpy(dtype=float)
            if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                ax_gain.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
                gain_values.extend(lo[np.isfinite(lo)].tolist())
                gain_values.extend(hi[np.isfinite(hi)].tolist())
            if not bootstrap.empty:
                ci = bootstrap[
                    bootstrap["variable"].astype(str).eq(variable)
                    & bootstrap["lead"].astype(int).eq(int(lead))
                ].sort_values("member_count")
                if not ci.empty and {"bss_gain_x100_p025", "bss_gain_x100_p975"} <= set(ci.columns):
                    ax_gain.fill_between(
                        ci["member_count"].to_numpy(dtype=float),
                        ci["bss_gain_x100_p025"].to_numpy(dtype=float),
                        ci["bss_gain_x100_p975"].to_numpy(dtype=float),
                        color=color,
                        alpha=0.08,
                        lw=0,
                    )
            ax_gain.plot(x, gain, color=color, lw=1.8, marker="o", ms=3.8)

        style_axis(ax_bss, zero_line=True)
        style_axis(ax_gain, zero_line=True)
        ax_gain.set_xlabel("Generated members")
        if vi == 0:
            ax_bss.set_ylabel("BSS")
            ax_gain.set_ylabel("FlowMatch - GEOS (x100)")

    bss_ylim = _robust_ylim(bss_values, (-1.0, 1.0))
    gain_ylim = _robust_ylim(gain_values, (-50.0, 50.0))
    for ax in axes[0, :].ravel():
        if ax.has_data():
            ax.set_ylim(*bss_ylim)
    for ax in axes[1, :].ravel():
        if ax.has_data():
            ax.set_ylim(*gain_ylim)

    from matplotlib.lines import Line2D

    lead_handles = [
        Line2D([0], [0], color=LEAD_COLORS.get(int(lead), "#2f6f9f"), lw=2.0, marker="o", ms=3.5, label=f"W{int(lead)}")
        for lead in sorted(set(summary["lead"].astype(int)))
    ]
    style_handles = [
        Line2D([0], [0], color="#4d5b68", lw=1.9, marker="o", ms=3.4, label="FlowMatch"),
        Line2D([0], [0], color="#4d5b68", lw=1.2, ls="--", alpha=0.55, label="GEOS"),
    ]
    fig.legend(
        lead_handles,
        [handle.get_label() for handle in lead_handles],
        loc="upper center",
        ncol=min(4, len(lead_handles)),
        frameon=False,
        bbox_to_anchor=(0.43, 0.98),
        fontsize=8.5,
    )
    fig.legend(
        style_handles,
        [handle.get_label() for handle in style_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.70, 0.98),
        fontsize=8.5,
    )
    fig.suptitle("Extreme-event spatial BSS and FlowMatch gain", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "ensemble_bss_extreme_dashboard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def read_existing_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def make_plots(out_dir: Path) -> list[str]:
    summary = read_existing_csv(out_dir / "ensemble_bss_summary.csv")
    bootstrap = read_existing_csv(out_dir / "ensemble_bss_case_bootstrap_ci.csv")
    path = plot_bss_dashboard(summary, bootstrap, out_dir)
    return [str(path)] if path is not None else []


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    variables = ens.parse_variables(args.variables)
    sample_sizes = [size for size in ens.parse_int_list(args.sample_sizes) if int(size) > 0]
    skip_years = ens.parse_int_set(args.skip_years)
    lead_filter = ens.parse_int_set(args.lead_values) if args.lead_values else None
    init_date_filter = ens.parse_date_filter(args.init_dates)
    valid_date_filter = ens.parse_date_filter(args.valid_dates)
    init_months = ens.parse_month_filter(args.init_months)
    event_regions = ens.parse_region_list(args.extreme_event_regions)
    event_variables = ens.parse_variables(args.extreme_event_variable)
    years = [year for year in range(args.start_year, args.end_year + 1) if year not in skip_years]
    years = ens.validate_forecast_stores(args.forecast_dir, years, args.allow_missing_years)
    rng = np.random.default_rng(args.seed)

    selected_events = ens.select_extreme_event_cases(
        args,
        years,
        lead_filter,
        init_months,
        init_date_filter,
        valid_date_filter,
        event_regions,
        event_variables,
    )
    events_by_year: dict[int, list[dict[str, object]]] = {}
    for event in selected_events:
        events_by_year.setdefault(int(event["year"]), []).append(event)

    thresholds, obs_clim, threshold_lats, threshold_lons = load_threshold_bundles(args, years, variables)

    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "skip_years": sorted(skip_years),
        "variables": variables,
        "sample_sizes_requested": sample_sizes,
        "member_bootstrap_repeats": int(args.member_bootstrap_repeats),
        "case_bootstrap_repeats": int(args.case_bootstrap_repeats),
        "seed": int(args.seed),
        "eval_mask": args.eval_mask,
        "land_mask_file": os.path.abspath(args.land_mask_file) if args.land_mask_file else None,
        "lead_values": sorted(lead_filter) if lead_filter is not None else [],
        "init_dates": sorted(init_date_filter),
        "valid_dates": sorted(valid_date_filter),
        "init_months": sorted(init_months),
        "threshold_file": os.path.abspath(args.threshold_file) if args.threshold_file else None,
        "threshold_grouping": {
            variable: thresholds[variable].get("grouping", "pooled")
            for variable in variables
        },
        "extreme_event_count_requested": int(args.extreme_event_count),
        "extreme_event_count_per_lead": bool(args.extreme_event_count_per_lead),
        "extreme_event_variables": event_variables,
        "extreme_event_regions": event_regions,
        "extreme_event_max_per_region": int(args.extreme_event_max_per_region),
        "selected_extreme_events": selected_events,
        "bss_event_definition": "obs >= local observed q95 threshold selected by valid time",
        "bss_reference": "local observed event frequency from the threshold source",
        "started_at": timestamp_now_utc(),
    }

    case_rows: list[dict[str, object]] = []
    weights = None
    base_mask = None
    processed_cases = 0
    start_time = time.time()

    for year in years:
        year_events = events_by_year.get(int(year), [])
        if not year_events:
            continue
        path = ens.store_path(args.forecast_dir, year)
        print(f"Opening {path}")
        ds = xr.open_zarr(path, consolidated=False, chunks=None)
        try:
            lats, lons = ens.get_lat_lon(ds, ens.VARIABLES[variables[0]]["model"])
            if not (np.allclose(lats, threshold_lats) and np.allclose(lons, threshold_lons)):
                raise ValueError(
                    "Threshold grid does not match forecast grid. "
                    "Rebuild thresholds on the same grid before evaluating BSS."
                )
            if weights is None or base_mask is None:
                weights = ens.area_weights_from_lats(lats, len(lons))
                base_mask = ens.load_eval_mask(args, (len(lats), len(lons)))
                print(f"Evaluation mask: {args.eval_mask}; kept {int(np.sum(base_mask))}/{base_mask.size} grid cells")
            for event in year_events:
                variable = str(event.get("event_score_variable", ""))
                if variable not in variables:
                    continue
                spec = ens.VARIABLES[variable]
                init_idx = int(event["init_idx"])
                lead_idx = int(event["lead_idx"])
                lead_value = int(event["lead"])
                init_time, valid_time = ens.case_times(ds, init_idx, lead_idx, lead_value)
                if init_date_filter and ens.date_key(init_time) not in init_date_filter:
                    continue
                if valid_date_filter and ens.date_key(valid_time) not in valid_date_filter:
                    continue
                region = str(event.get("region", ""))
                region_mask = ens.region_mask_from_bounds(lats, lons, ens.REGIONS[region])
                case_mask = base_mask & region_mask
                if int(np.sum(case_mask)) <= 1:
                    raise ValueError(f"Event region {region!r} has too few valid cells for BSS.")

                threshold = matrix_eval.select_grouped_map(thresholds[variable], valid_time)
                obs_event_freq = matrix_eval.select_grouped_map(obs_clim[variable], valid_time)
                obs = ens.load_obs_array(ds, spec["obs"], init_idx, lead_idx)
                model = ens.load_forecast_array(ds, spec["model"], init_idx, lead_idx)
                geos = ens.load_forecast_array(ds, spec["geos"], init_idx, lead_idx)
                model_members = int(model.shape[0])
                usable_sizes = [size for size in sample_sizes if int(size) <= model_members]
                if not usable_sizes:
                    usable_sizes = [model_members]
                case_id = f"{year}_{init_idx:04d}_lead{lead_value}_{region}_{variable}"

                geos_state = bss_sums(geos, obs, threshold, obs_event_freq, weights, case_mask, "geos")
                for size in usable_sizes:
                    repeats = 1 if int(size) >= model_members else max(1, int(args.member_bootstrap_repeats))
                    for member_repeat in range(repeats):
                        if int(size) >= model_members:
                            member_idx = np.arange(model_members)
                        else:
                            member_idx = rng.choice(model_members, size=int(size), replace=False)
                        model_state = bss_sums(
                            model[member_idx, :, :],
                            obs,
                            threshold,
                            obs_event_freq,
                            weights,
                            case_mask,
                            "model",
                        )
                        row = {
                            "case_id": case_id,
                            "year": int(year),
                            "init_index": init_idx,
                            "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
                            "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
                            "lead": lead_value,
                            "variable": variable,
                            "member_count": int(size),
                            "member_repeat": int(member_repeat),
                            "model_members_available": model_members,
                            "geos_members_available": int(geos.shape[0]),
                            "eval_mask": args.eval_mask,
                            "region": region,
                            "region_name": str(ens.REGIONS[region]["name"]),
                            "event_rank": event.get("event_rank", ""),
                            "event_selection_lead": event.get("event_selection_lead", ""),
                            "event_score": event.get("event_score", np.nan),
                            "event_score_variable": variable,
                        }
                        row.update(model_state)
                        row["geos_bs_sum"] = geos_state["geos_bs_sum"]
                        row["geos_prob_weight_sum"] = geos_state["geos_prob_weight_sum"]
                        row["ref_bs_sum"] = geos_state["ref_bs_sum"]
                        row["weight_sum"] = geos_state["weight_sum"]
                        row["obs_event_weight_sum"] = geos_state["obs_event_weight_sum"]
                        row.update(row_metrics(row))
                        case_rows.append(row)
                processed_cases += 1
                if processed_cases % 20 == 0:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Processed {processed_cases} extreme cases in {elapsed:.1f} min")
        finally:
            ds.close()

    if not case_rows:
        raise RuntimeError("No extreme-event BSS rows were processed.")

    case_df = pd.DataFrame(case_rows)
    repeat_summary = aggregate_sums(case_df, ["variable", "lead", "member_count", "member_repeat"])
    summary = summarize_member_repeats(repeat_summary)
    bootstrap = bootstrap_case_intervals(case_df, int(args.case_bootstrap_repeats), rng)

    write_csv(case_df, out_dir / "case_member_bss_sums.csv")
    write_csv(repeat_summary, out_dir / "ensemble_bss_member_repeat_summary.csv")
    write_csv(summary, out_dir / "ensemble_bss_summary.csv")
    if not bootstrap.empty:
        write_csv(bootstrap, out_dir / "ensemble_bss_case_bootstrap_ci.csv")
    report_path = write_member_report(summary, out_dir, int(args.report_member_count))

    metadata["processed_extreme_cases"] = int(processed_cases)
    metadata["case_member_rows"] = int(len(case_df))
    metadata["completed_at"] = timestamp_now_utc()
    metadata["outputs"] = {
        "case_member_bss_sums": str(out_dir / "case_member_bss_sums.csv"),
        "ensemble_bss_member_repeat_summary": str(out_dir / "ensemble_bss_member_repeat_summary.csv"),
        "ensemble_bss_summary": str(out_dir / "ensemble_bss_summary.csv"),
        "ensemble_bss_case_bootstrap_ci": (
            str(out_dir / "ensemble_bss_case_bootstrap_ci.csv") if not bootstrap.empty else None
        ),
        "ensemble_bss_member_report": report_path,
    }
    metadata_path = out_dir / "ensemble_bss_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {metadata_path}")

    if args.make_plots:
        plot_paths = make_plots(out_dir)
        metadata["outputs"]["plots"] = plot_paths
        metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.plot_only:
        write_member_report(read_existing_csv(out_dir / "ensemble_bss_summary.csv"), out_dir, int(args.report_member_count))
        if args.make_plots:
            make_plots(out_dir)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
