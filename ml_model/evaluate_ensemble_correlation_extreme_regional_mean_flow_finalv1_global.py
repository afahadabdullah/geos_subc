#!/usr/bin/env python3
"""Extreme-event regional-mean ensemble correlation diagnostics.

This standalone script answers a different question from
evaluate_ensemble_correlation_extremes_flow_finalv1_global.py:

  * spatial-correlation script: correlation across grid cells within event regions
  * this script: correlation across event regional means

For every selected extreme case, each ensemble member is first reduced to a
weighted regional mean over the event region. Ensemble-size sampling is then
applied to those member regional means, and Pearson correlation is computed
across the selected events.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import evaluate_ensemble_tests_flow_finalv1_global as ens
except ModuleNotFoundError:
    from ml_model import evaluate_ensemble_tests_flow_finalv1_global as ens


TARGETS = ("ens_mean", "q95")
TARGET_LABELS = {"ens_mean": "ensemble mean", "q95": "q95"}
LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback only used on minimal envs.
    scipy_stats = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast_dir", default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50")
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--skip_years", default="", help="Comma-separated years to skip.")
    parser.add_argument(
        "--out_dir",
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "ensemble_corr_extreme_regional_mean_t2m30_pr30_regions_2021_2023_w1w4_memberboot50"
        ),
    )
    parser.add_argument("--variables", default="pr,t2m", help="Comma-separated subset of pr,t2m.")
    parser.add_argument("--sample_sizes", default="4,8,16,32,64,90")
    parser.add_argument("--member_bootstrap_repeats", type=int, default=50)
    parser.add_argument("--case_bootstrap_repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval_mask", choices=("all", "land", "ocean"), default="all")
    parser.add_argument("--land_mask_file", default=None)
    parser.add_argument("--lead_values", default="1,2,3,4")
    parser.add_argument("--init_dates", default="")
    parser.add_argument("--valid_dates", default="")
    parser.add_argument("--init_months", default="")
    parser.add_argument("--extreme_event_count", type=int, default=30)
    parser.add_argument("--extreme_event_variable", default="t2m,pr")
    parser.add_argument("--extreme_event_regions", default="global_extremes")
    parser.add_argument("--extreme_event_max_per_region", type=int, default=2)
    parser.add_argument("--extreme_event_count_per_lead", action=argparse.BooleanOptionalAction, default=True)
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


def read_existing_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def weighted_region_mean(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(field)[mask]
    w = np.asarray(weights)[mask]
    finite = np.isfinite(values) & np.isfinite(w) & (w > 0)
    if not np.any(finite):
        return np.nan
    return float(np.sum(values[finite] * w[finite]) / np.sum(w[finite]))


def member_region_means(members: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(members)[:, mask]
    w = np.asarray(weights)[mask].astype(np.float64, copy=False)
    finite = np.isfinite(values) & np.isfinite(w)[None, :] & (w[None, :] > 0)
    denom = np.sum(np.where(finite, w[None, :], 0.0), axis=1)
    numer = np.sum(np.where(finite, values * w[None, :], 0.0), axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(numer, denom, out=out, where=denom > 1e-12)
    return out


def target_value(member_values: np.ndarray, target: str) -> float:
    values = np.asarray(member_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if target == "ens_mean":
        return float(np.mean(values))
    if target == "q95":
        return float(np.quantile(values, 0.95))
    raise KeyError(f"Unknown target: {target}")


def pearson_corr_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, int, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    n = int(np.sum(finite))
    if n < 2:
        return np.nan, n, np.nan
    x = x[finite]
    y = y[finite]
    x_anom = x - np.mean(x)
    y_anom = y - np.mean(y)
    denom = np.sqrt(np.sum(x_anom * x_anom) * np.sum(y_anom * y_anom))
    if denom <= 1e-12:
        return np.nan, n, np.nan
    corr = float(np.sum(x_anom * y_anom) / denom)
    corr = float(np.clip(corr, -1.0, 1.0))
    if n <= 2:
        return corr, n, np.nan
    if abs(corr) >= 1.0:
        return corr, n, 0.0
    t_stat = abs(corr) * math.sqrt((n - 2.0) / max(1.0 - corr * corr, 1e-15))
    if scipy_stats is not None:
        p_value = float(2.0 * scipy_stats.t.sf(t_stat, df=n - 2))
    else:
        # Large-sample normal approximation if SciPy is not available.
        p_value = float(math.erfc(t_stat / math.sqrt(2.0)))
    return corr, n, p_value


def row_metrics(model_values: np.ndarray, geos_values: np.ndarray, obs_values: np.ndarray) -> dict[str, float]:
    model_corr, model_n, model_p = pearson_corr_stats(model_values, obs_values)
    geos_corr, geos_n, geos_p = pearson_corr_stats(geos_values, obs_values)
    corr_diff = model_corr - geos_corr
    corr_gain_pct = 100.0 * corr_diff / abs(geos_corr) if np.isfinite(geos_corr) and abs(geos_corr) > 1e-12 else np.nan
    return {
        "model_corr": model_corr,
        "model_corr_n": model_n,
        "model_corr_p_value": model_p,
        "geos_corr": geos_corr,
        "geos_corr_n": geos_n,
        "geos_corr_p_value": geos_p,
        "corr_diff": corr_diff,
        "corr_gain_pct": corr_gain_pct,
    }


def aggregate_repeats(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["variable", "lead", "target", "member_count", "member_repeat"]
    for key, group in case_df.groupby(group_cols, dropna=False):
        metrics = row_metrics(
            group["model_value"].to_numpy(dtype=float),
            group["geos_value"].to_numpy(dtype=float),
            group["obs_value"].to_numpy(dtype=float),
        )
        row = dict(zip(group_cols, key))
        row.update(metrics)
        row["n_cases"] = int(group["case_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_member_repeats(repeat_df: pd.DataFrame) -> pd.DataFrame:
    if repeat_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "target", "member_count"]
    for key, group in repeat_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["member_repeat_count"] = int(group["member_repeat"].nunique())
        row["n_cases"] = int(group["n_cases"].max())
        for metric in ("model_corr", "geos_corr", "corr_diff", "corr_gain_pct"):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_p05"] = np.nan
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p95"] = np.nan
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_p05"] = float(np.quantile(values, 0.05))
                row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
                row[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        for metric in ("model_corr_p_value", "geos_corr_p_value"):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def bootstrap_case_intervals(case_df: pd.DataFrame, repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    if repeats <= 0 or case_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "target", "member_count"]
    for key, group in case_df.groupby(group_cols, dropna=False):
        case_ids = sorted(group["case_id"].astype(str).unique())
        rows_by_case = {
            case_id: group[group["case_id"].astype(str).eq(case_id)].reset_index(drop=True)
            for case_id in case_ids
        }
        boot = {metric: [] for metric in ("model_corr", "geos_corr", "corr_diff", "corr_gain_pct")}
        for _ in range(repeats):
            sampled_cases = rng.choice(case_ids, size=len(case_ids), replace=True)
            model_values = []
            geos_values = []
            obs_values = []
            for case_id in sampled_cases:
                choices = rows_by_case[str(case_id)]
                picked = choices.iloc[int(rng.integers(0, len(choices)))]
                model_values.append(float(picked["model_value"]))
                geos_values.append(float(picked["geos_value"]))
                obs_values.append(float(picked["obs_value"]))
            metrics = row_metrics(np.asarray(model_values), np.asarray(geos_values), np.asarray(obs_values))
            for metric, value in metrics.items():
                if metric in boot:
                    boot[metric].append(value)
        row = dict(zip(group_cols, key))
        row["case_bootstrap_repeats"] = int(repeats)
        row["n_cases"] = int(len(case_ids))
        for metric, values in boot.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                row[f"{metric}_p025"] = np.nan
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p975"] = np.nan
            else:
                row[f"{metric}_p025"] = float(np.quantile(arr, 0.025))
                row[f"{metric}_p50"] = float(np.quantile(arr, 0.50))
                row[f"{metric}_p975"] = float(np.quantile(arr, 0.975))
        diff_arr = np.asarray(boot["corr_diff"], dtype=float)
        diff_arr = diff_arr[np.isfinite(diff_arr)]
        if diff_arr.size:
            row["corr_diff_positive_fraction"] = float(np.mean(diff_arr > 0.0))
            row["corr_diff_bootstrap_p_value_two_sided"] = float(
                2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0))
            )
        else:
            row["corr_diff_positive_fraction"] = np.nan
            row["corr_diff_bootstrap_p_value_two_sided"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_member_report(summary: pd.DataFrame, out_dir: Path, member_count: int) -> str | None:
    if member_count <= 0 or summary.empty:
        return None
    sub = summary[summary["member_count"].astype(int).eq(int(member_count))].copy()
    if sub.empty:
        print(f"No regional-mean correlation rows found for ensemble size {member_count}.")
        return None
    cols = [
        "variable",
        "lead",
        "target",
        "member_count",
        "model_corr_mean",
        "geos_corr_mean",
        "corr_diff_mean",
        "corr_gain_pct_mean",
        "model_corr_p_value_p50",
        "geos_corr_p_value_p50",
        "n_cases",
    ]
    sub = sub[[col for col in cols if col in sub.columns]].sort_values(["variable", "target", "lead"])
    path = out_dir / f"ensemble_regional_mean_correlation_member{int(member_count)}_report.csv"
    write_csv(sub, path)
    print(f"\nEnsemble size {int(member_count)} regional-mean correlation report:")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(sub.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    return str(path)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return plt, Line2D


def plot_regional_mean_dashboard(summary: pd.DataFrame, bootstrap: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    plt, Line2D = _import_matplotlib()
    variables = [var for var in ("pr", "t2m") if var in set(summary["variable"].astype(str))]
    if not variables:
        variables = sorted(summary["variable"].astype(str).unique())
    fig, axes = plt.subplots(
        2 * len(variables),
        2,
        figsize=(11.0, 4.2 * len(variables)),
        sharex="col",
        gridspec_kw={"height_ratios": [1.1, 0.9] * len(variables), "hspace": 0.22, "wspace": 0.18},
    )
    axes = np.asarray(axes).reshape(2 * len(variables), 2)
    letters = "abcdefghijklmnopqrstuvwxyz"
    corr_values_by_variable = {variable: [] for variable in variables}
    gain_values_by_variable = {variable: [] for variable in variables}

    def style_axis(ax, zero_line: bool = False) -> None:
        ax.grid(True, axis="y", color="#dde3e8", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if zero_line:
            ax.axhline(0.0, color="#7a8794", lw=0.9, ls="--")

    def robust_range(values, lower_floor, upper_floor, lower_cap=None, upper_cap=None):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return lower_floor, upper_floor
        lo = float(np.nanquantile(arr, 0.02))
        hi = float(np.nanquantile(arr, 0.98))
        pad = max(0.03 * (hi - lo), 0.015 * max(abs(hi), abs(lo), 1.0))
        lo = min(lower_floor, lo - pad)
        hi = max(upper_floor, hi + pad)
        if lower_cap is not None:
            lo = max(lower_cap, lo)
        if upper_cap is not None:
            hi = min(upper_cap, hi)
        return lo, hi

    for vi, variable in enumerate(variables):
        for ti, target in enumerate(TARGETS):
            ax_corr = axes[2 * vi, ti]
            ax_gain = axes[2 * vi + 1, ti]
            corr_letter = letters[(2 * vi) * 2 + ti]
            gain_letter = letters[(2 * vi + 1) * 2 + ti]
            title_corr = f"({corr_letter}) {variable.upper()} {TARGET_LABELS[target]} event-mean correlation"
            title_gain = f"({gain_letter}) {variable.upper()} {TARGET_LABELS[target]} event-mean gain"
            sub = summary[
                summary["variable"].astype(str).eq(variable)
                & summary["target"].astype(str).eq(target)
            ]
            if sub.empty:
                for ax, title in ((ax_corr, title_corr), (ax_gain, title_gain)):
                    ax.set_axis_off()
                    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
                    ax.text(0.5, 0.5, "No rows", transform=ax.transAxes, ha="center", va="center")
                continue
            for lead in sorted(sub["lead"].astype(int).unique()):
                line = sub[sub["lead"].astype(int).eq(lead)].sort_values("member_count")
                x = line["member_count"].to_numpy(dtype=float)
                color = LEAD_COLORS.get(int(lead), "#2f6f9f")
                geos_corr = line["geos_corr_mean"].to_numpy(dtype=float)
                model_corr = line["model_corr_mean"].to_numpy(dtype=float)
                corr_values_by_variable[variable].extend(geos_corr[np.isfinite(geos_corr)].tolist())
                corr_values_by_variable[variable].extend(model_corr[np.isfinite(model_corr)].tolist())
                ax_corr.plot(x, geos_corr, color=color, lw=1.15, ls="--", alpha=0.46)
                ax_corr.plot(x, model_corr, color=color, lw=1.85, marker="o", ms=3.4)

                gain = line["corr_gain_pct_mean"].to_numpy(dtype=float)
                gain_values_by_variable[variable].extend(gain[np.isfinite(gain)].tolist())
                lo = line["corr_gain_pct_p05"].to_numpy(dtype=float)
                hi = line["corr_gain_pct_p95"].to_numpy(dtype=float)
                if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                    ax_gain.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
                    gain_values_by_variable[variable].extend(lo[np.isfinite(lo)].tolist())
                    gain_values_by_variable[variable].extend(hi[np.isfinite(hi)].tolist())
                if not bootstrap.empty:
                    ci = bootstrap[
                        bootstrap["variable"].astype(str).eq(variable)
                        & bootstrap["target"].astype(str).eq(target)
                        & bootstrap["lead"].astype(int).eq(int(lead))
                    ].sort_values("member_count")
                    if not ci.empty and {"corr_gain_pct_p025", "corr_gain_pct_p975"} <= set(ci.columns):
                        ax_gain.fill_between(
                            ci["member_count"].to_numpy(dtype=float),
                            ci["corr_gain_pct_p025"].to_numpy(dtype=float),
                            ci["corr_gain_pct_p975"].to_numpy(dtype=float),
                            color=color,
                            alpha=0.08,
                            lw=0,
                        )
                ax_gain.plot(x, gain, color=color, lw=1.8, marker="o", ms=3.8)

            ax_corr.set_title(title_corr, loc="left", fontweight="bold", fontsize=10)
            ax_gain.set_title(title_gain, loc="left", fontweight="bold", fontsize=10)
            style_axis(ax_corr)
            style_axis(ax_gain, zero_line=True)
            if vi == len(variables) - 1:
                ax_gain.set_xlabel("Generated members")
            if ti == 0:
                ax_corr.set_ylabel("Correlation")
                ax_gain.set_ylabel("FlowMatch gain (%)")

    for vi, variable in enumerate(variables):
        if variable == "t2m":
            corr_ylim = robust_range(corr_values_by_variable[variable], 0.80, 1.00, -1.0, 1.0)
            gain_ylim = robust_range(gain_values_by_variable[variable], -10.0, 25.0, -50.0, 100.0)
        else:
            corr_ylim = robust_range(corr_values_by_variable[variable], -0.20, 0.80, -1.0, 1.0)
            gain_ylim = robust_range(gain_values_by_variable[variable], -25.0, 80.0, -100.0, 200.0)
        for ax in axes[2 * vi, :].ravel():
            if ax.has_data():
                ax.set_ylim(*corr_ylim)
        for ax in axes[2 * vi + 1, :].ravel():
            if ax.has_data():
                ax.set_ylim(*gain_ylim)

    leads = sorted(set(summary["lead"].astype(int)))
    lead_handles = [
        Line2D([0], [0], color=LEAD_COLORS.get(int(lead), "#2f6f9f"), lw=2.0, marker="o", ms=3.5,
               label=f"W{int(lead)}")
        for lead in leads
    ]
    style_handles = [
        Line2D([0], [0], color="#4d5b68", lw=1.9, marker="o", ms=3.4, label="FlowMatch"),
        Line2D([0], [0], color="#4d5b68", lw=1.2, ls="--", alpha=0.55, label="GEOS"),
    ]
    fig.legend(lead_handles, [h.get_label() for h in lead_handles], loc="upper center",
               ncol=min(4, len(lead_handles)), frameon=False, bbox_to_anchor=(0.42, 0.965), fontsize=8.5)
    fig.legend(style_handles, [h.get_label() for h in style_handles], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.70, 0.965), fontsize=8.5)
    fig.suptitle("Extreme-event regional-mean correlation and FlowMatch gain", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "ensemble_regional_mean_correlation_dashboard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def make_plots(out_dir: Path) -> list[str]:
    summary = read_existing_csv(out_dir / "ensemble_regional_mean_correlation_summary.csv")
    bootstrap = read_existing_csv(out_dir / "ensemble_regional_mean_correlation_case_bootstrap_ci.csv")
    path = plot_regional_mean_dashboard(summary, bootstrap, out_dir)
    return [str(path)] if path is not None else []


def collect_event_member_values(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
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

    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "skip_years": sorted(skip_years),
        "variables": variables,
        "sample_sizes_requested": sample_sizes,
        "member_bootstrap_repeats": int(args.member_bootstrap_repeats),
        "case_bootstrap_repeats": int(args.case_bootstrap_repeats),
        "seed": int(args.seed),
        "eval_mask": args.eval_mask,
        "land_mask_file": os.path.abspath(args.land_mask_file) if args.land_mask_file else None,
        "lead_values": sorted(lead_filter) if lead_filter is not None else [],
        "extreme_event_count_requested": int(args.extreme_event_count),
        "extreme_event_count_per_lead": bool(args.extreme_event_count_per_lead),
        "extreme_event_variables": event_variables,
        "extreme_event_regions": event_regions,
        "extreme_event_max_per_region": int(args.extreme_event_max_per_region),
        "selected_extreme_events": selected_events,
        "targets": list(TARGETS),
        "correlation_axis": "event regional means",
        "pearson_p_value": (
            "two-sided t-test for nonzero Pearson correlation; SciPy t distribution "
            "or normal approximation if SciPy is unavailable"
        ),
        "corr_diff_bootstrap_p_value": "two-sided bootstrap sign p-value for FlowMatch-GEOS correlation difference",
        "started_at": timestamp_now_utc(),
    }

    event_records: list[dict[str, object]] = []
    weights = None
    base_mask = None
    start_time = time.time()
    processed_cases = 0
    for year in years:
        year_events = events_by_year.get(int(year), [])
        if not year_events:
            continue
        path = ens.store_path(args.forecast_dir, year)
        print(f"Opening {path}")
        ds = xr.open_zarr(path, consolidated=False, chunks=None)
        try:
            lats, lons = ens.get_lat_lon(ds, ens.VARIABLES[variables[0]]["model"])
            if weights is None or base_mask is None:
                weights = ens.area_weights_from_lats(lats, len(lons))
                base_mask = ens.load_eval_mask(args, (len(lats), len(lons)))
                print(f"Evaluation mask: {args.eval_mask}; kept {int(np.sum(base_mask))}/{base_mask.size} grid cells")
            for event in year_events:
                variable = str(event.get("event_score_variable", ""))
                if variable not in variables:
                    continue
                init_idx = int(event["init_idx"])
                lead_idx = int(event["lead_idx"])
                lead_value = int(event["lead"])
                init_time, valid_time = ens.case_times(ds, init_idx, lead_idx, lead_value)
                if init_date_filter and ens.date_key(init_time) not in init_date_filter:
                    continue
                if valid_date_filter and ens.date_key(valid_time) not in valid_date_filter:
                    continue
                region = str(event.get("region", ""))
                case_mask = base_mask & ens.region_mask_from_bounds(lats, lons, ens.REGIONS[region])
                if int(np.sum(case_mask)) <= 1:
                    raise ValueError(f"Event region {region!r} has too few valid cells.")
                spec = ens.VARIABLES[variable]
                obs = ens.load_obs_array(ds, spec["obs"], init_idx, lead_idx)
                model = ens.load_forecast_array(ds, spec["model"], init_idx, lead_idx)
                geos = ens.load_forecast_array(ds, spec["geos"], init_idx, lead_idx)
                case_id = f"{year}_{init_idx:04d}_lead{lead_value}_{region}_{variable}"
                event_records.append(
                    {
                        "case_id": case_id,
                        "year": int(year),
                        "init_index": init_idx,
                        "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
                        "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
                        "lead": lead_value,
                        "variable": variable,
                        "eval_mask": args.eval_mask,
                        "region": region,
                        "region_name": str(ens.REGIONS[region]["name"]),
                        "event_rank": event.get("event_rank", ""),
                        "event_selection_lead": event.get("event_selection_lead", ""),
                        "event_score": event.get("event_score", np.nan),
                        "event_score_variable": variable,
                        "obs_value": weighted_region_mean(obs, weights, case_mask),
                        "model_member_values": member_region_means(model, weights, case_mask),
                        "geos_member_values": member_region_means(geos, weights, case_mask),
                    }
                )
                processed_cases += 1
                if processed_cases % 20 == 0:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Processed {processed_cases} event regional means in {elapsed:.1f} min")
        finally:
            ds.close()
    metadata["processed_extreme_cases"] = int(processed_cases)
    return event_records, metadata


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sample_sizes = [size for size in ens.parse_int_list(args.sample_sizes) if int(size) > 0]

    event_records, metadata = collect_event_member_values(args)
    if not event_records:
        raise RuntimeError("No event regional means were processed.")

    case_rows = []
    for event in event_records:
        model_values = np.asarray(event["model_member_values"], dtype=np.float64)
        geos_values = np.asarray(event["geos_member_values"], dtype=np.float64)
        model_members = int(model_values.size)
        geos_targets = {target: target_value(geos_values, target) for target in TARGETS}
        usable_sizes = [size for size in sample_sizes if int(size) <= model_members] or [model_members]
        for size in usable_sizes:
            repeats = 1 if int(size) >= model_members else max(1, int(args.member_bootstrap_repeats))
            for member_repeat in range(repeats):
                if int(size) >= model_members:
                    member_idx = np.arange(model_members)
                else:
                    member_idx = rng.choice(model_members, size=int(size), replace=False)
                sample = model_values[member_idx]
                for target in TARGETS:
                    row = {
                        key: event[key]
                        for key in (
                            "case_id",
                            "year",
                            "init_index",
                            "init_time",
                            "valid_time",
                            "lead",
                            "variable",
                            "eval_mask",
                            "region",
                            "region_name",
                            "event_rank",
                            "event_selection_lead",
                            "event_score",
                            "event_score_variable",
                        )
                    }
                    row.update(
                        {
                            "target": target,
                            "member_count": int(size),
                            "member_repeat": int(member_repeat),
                            "model_members_available": int(model_members),
                            "geos_members_available": int(geos_values.size),
                            "obs_value": float(event["obs_value"]),
                            "model_value": target_value(sample, target),
                            "geos_value": geos_targets[target],
                        }
                    )
                    case_rows.append(row)

    case_df = pd.DataFrame(case_rows)
    repeat_summary = aggregate_repeats(case_df)
    summary = summarize_member_repeats(repeat_summary)
    bootstrap = bootstrap_case_intervals(case_df, int(args.case_bootstrap_repeats), rng)

    write_csv(case_df, out_dir / "case_member_regional_mean_values.csv")
    write_csv(repeat_summary, out_dir / "ensemble_regional_mean_correlation_member_repeat_summary.csv")
    write_csv(summary, out_dir / "ensemble_regional_mean_correlation_summary.csv")
    if not bootstrap.empty:
        write_csv(bootstrap, out_dir / "ensemble_regional_mean_correlation_case_bootstrap_ci.csv")
    report_path = write_member_report(summary, out_dir, int(args.report_member_count))

    metadata["case_member_rows"] = int(len(case_df))
    metadata["completed_at"] = timestamp_now_utc()
    metadata["outputs"] = {
        "case_member_regional_mean_values": str(out_dir / "case_member_regional_mean_values.csv"),
        "ensemble_regional_mean_correlation_member_repeat_summary": str(
            out_dir / "ensemble_regional_mean_correlation_member_repeat_summary.csv"
        ),
        "ensemble_regional_mean_correlation_summary": str(out_dir / "ensemble_regional_mean_correlation_summary.csv"),
        "ensemble_regional_mean_correlation_case_bootstrap_ci": (
            str(out_dir / "ensemble_regional_mean_correlation_case_bootstrap_ci.csv")
            if not bootstrap.empty
            else None
        ),
        "ensemble_regional_mean_member_report": report_path,
    }
    metadata_path = out_dir / "ensemble_regional_mean_correlation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {metadata_path}")
    if args.make_plots:
        metadata["outputs"]["plots"] = make_plots(out_dir)
        metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.plot_only:
        write_member_report(
            read_existing_csv(out_dir / "ensemble_regional_mean_correlation_summary.csv"),
            out_dir,
            int(args.report_member_count),
        )
        if args.make_plots:
            make_plots(out_dir)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
