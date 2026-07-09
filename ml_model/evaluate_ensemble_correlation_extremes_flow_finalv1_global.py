#!/usr/bin/env python3
"""Extreme-event ensemble-size correlation diagnostics.

This is a standalone companion to evaluate_ensemble_tests_flow_finalv1_global.py.
It keeps the existing evaluator and paper figure scripts untouched, but reuses
their forecast-store and extreme-event selection helpers.

For each selected regional extreme event, the script computes weighted spatial
Pearson correlation against observations as generated ensemble size increases.
Two FlowMatch summaries are evaluated:

  - ens_mean: mean over the sampled generated members
  - q95:      95th percentile over the sampled generated members

The GEOS/FIMr1p1 baseline is evaluated with the matching ensemble summary
(GEOS ensemble mean or GEOS q95), and plots show correlation gain
(FlowMatch minus GEOS) by member count.
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
except ModuleNotFoundError:  # Allows `python -m ml_model...` from repo root.
    from ml_model import evaluate_ensemble_tests_flow_finalv1_global as ens


TARGETS = ("ens_mean", "q95")
TARGET_LABELS = {
    "ens_mean": "ensemble mean",
    "q95": "q95",
}
LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}
LEAD_LINESTYLES = {1: ":", 2: "-.", 3: "--", 4: "-"}
SUM_COLUMNS = [
    "model_weight_sum",
    "model_x_sum",
    "model_y_sum",
    "model_x2_sum",
    "model_y2_sum",
    "model_xy_sum",
    "geos_weight_sum",
    "geos_x_sum",
    "geos_y_sum",
    "geos_x2_sum",
    "geos_y2_sum",
    "geos_xy_sum",
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
            "ensemble_corr_extreme_t2m30_pr30_regions_2021_2023_w1w4_memberboot50"
        ),
    )
    parser.add_argument("--variables", default="pr,t2m", help="Comma-separated subset of pr,t2m.")
    parser.add_argument(
        "--sample_sizes",
        default="4,8,16,32,64,90",
        help="Comma-separated generated ensemble sizes. Zero is ignored because correlation is undefined.",
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


def target_vector(members_by_point: np.ndarray, target: str) -> np.ndarray:
    if target == "ens_mean":
        return np.nanmean(members_by_point, axis=0)
    if target == "q95":
        return np.nanquantile(members_by_point, 0.95, axis=0)
    raise KeyError(f"Unknown target summary: {target}")


def weighted_corr_sums(field: np.ndarray, obs: np.ndarray, weights: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.isfinite(field) & np.isfinite(obs) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return {
            f"{prefix}_weight_sum": 0.0,
            f"{prefix}_x_sum": 0.0,
            f"{prefix}_y_sum": 0.0,
            f"{prefix}_x2_sum": 0.0,
            f"{prefix}_y2_sum": 0.0,
            f"{prefix}_xy_sum": 0.0,
        }
    x = field[finite].astype(np.float64, copy=False)
    y = obs[finite].astype(np.float64, copy=False)
    w = weights[finite].astype(np.float64, copy=False)
    return {
        f"{prefix}_weight_sum": float(np.sum(w)),
        f"{prefix}_x_sum": float(np.sum(w * x)),
        f"{prefix}_y_sum": float(np.sum(w * y)),
        f"{prefix}_x2_sum": float(np.sum(w * x * x)),
        f"{prefix}_y2_sum": float(np.sum(w * y * y)),
        f"{prefix}_xy_sum": float(np.sum(w * x * y)),
    }


def corr_from_sums(row: dict[str, float] | pd.Series, prefix: str) -> float:
    w = float(row.get(f"{prefix}_weight_sum", np.nan))
    if not np.isfinite(w) or w <= 1e-12:
        return np.nan
    mx = float(row[f"{prefix}_x_sum"]) / w
    my = float(row[f"{prefix}_y_sum"]) / w
    cov = float(row[f"{prefix}_xy_sum"]) / w - mx * my
    vx = float(row[f"{prefix}_x2_sum"]) / w - mx * mx
    vy = float(row[f"{prefix}_y2_sum"]) / w - my * my
    denom = np.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom <= 1e-12:
        return np.nan
    corr = cov / denom
    return float(corr) if np.isfinite(corr) else np.nan


def add_corr_metrics(row: dict[str, float] | pd.Series) -> dict[str, float]:
    out = dict(row)
    out["model_corr"] = corr_from_sums(out, "model")
    out["geos_corr"] = corr_from_sums(out, "geos")
    out["corr_diff"] = out["model_corr"] - out["geos_corr"]
    geos = out["geos_corr"]
    out["corr_gain_pct"] = (
        100.0 * out["corr_diff"] / abs(geos)
        if np.isfinite(geos) and abs(geos) > 1e-12
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
    rows = [add_corr_metrics(row) for row in sums.to_dict("records")]
    return pd.DataFrame(rows)


def summarize_member_repeats(repeat_summary: pd.DataFrame) -> pd.DataFrame:
    if repeat_summary.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "target", "member_count"]
    for key, group in repeat_summary.groupby(group_cols, dropna=False):
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
    group_cols = ["variable", "lead", "target", "member_count"]
    for key, group in case_df.groupby(group_cols, dropna=False):
        cases = sorted(group["case_id"].astype(str).unique())
        if not cases:
            continue
        rows_by_case = {
            case: group[group["case_id"].astype(str).eq(case)].reset_index(drop=True)
            for case in cases
        }
        boot = {metric: [] for metric in ("model_corr", "geos_corr", "corr_diff", "corr_gain_pct")}
        for _ in range(repeats):
            sampled_cases = rng.choice(cases, size=len(cases), replace=True)
            state = {col: 0.0 for col in SUM_COLUMNS}
            for case in sampled_cases:
                case_rows = rows_by_case[str(case)]
                picked = case_rows.iloc[int(rng.integers(0, len(case_rows)))]
                for col in SUM_COLUMNS:
                    state[col] += float(picked[col])
            metrics = add_corr_metrics(state)
            for metric in boot:
                boot[metric].append(metrics[metric])
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
            else:
                row[f"{metric}_p025"] = float(np.quantile(arr, 0.025))
                row[f"{metric}_p50"] = float(np.quantile(arr, 0.50))
                row[f"{metric}_p975"] = float(np.quantile(arr, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_member_report(summary: pd.DataFrame, out_dir: Path, member_count: int) -> str | None:
    if member_count <= 0 or summary.empty:
        return None
    sub = summary[summary["member_count"].astype(int).eq(int(member_count))].copy()
    if sub.empty:
        print(f"No correlation rows found for ensemble size {member_count}.")
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
        "n_cases",
    ]
    sub = sub[[col for col in cols if col in sub.columns]].sort_values(["variable", "target", "lead"])
    path = out_dir / f"ensemble_correlation_member{int(member_count)}_report.csv"
    write_csv(sub, path)
    print(f"\nEnsemble size {int(member_count)} correlation report:")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(sub.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    return str(path)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_correlation_dashboard(summary: pd.DataFrame, bootstrap: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    plt = _import_matplotlib()
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
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    corr_values: list[float] = []
    gain_values: list[float] = []

    def style_axis(ax, zero_line: bool = False) -> None:
        ax.grid(True, axis="y", color="#dde3e8", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if zero_line:
            ax.axhline(0.0, color="#7a8794", lw=0.9, ls="--")

    def robust_symmetric_limit(values: list[float], minimum: float, maximum: float | None = None) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return minimum
        limit = float(np.nanquantile(np.abs(arr), 0.98))
        limit = max(minimum, 1.10 * limit)
        if maximum is not None:
            limit = min(maximum, limit)
        return limit

    for vi, variable in enumerate(variables):
        for ti, target in enumerate(TARGETS):
            ax_corr = axes[2 * vi, ti]
            ax_gain = axes[2 * vi + 1, ti]
            title_corr = f"({next(letters)}) {variable.upper()} {TARGET_LABELS[target]} correlation"
            title_gain = f"({next(letters)}) {variable.upper()} {TARGET_LABELS[target]} gain"
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
                corr_values.extend(geos_corr[np.isfinite(geos_corr)].tolist())
                corr_values.extend(model_corr[np.isfinite(model_corr)].tolist())
                ax_corr.plot(
                    x,
                    geos_corr,
                    color=color,
                    lw=1.5,
                    ls=LEAD_LINESTYLES.get(int(lead), "--"),
                    alpha=0.78,
                    label=f"W{int(lead)} GEOS",
                )
                ax_corr.plot(
                    x,
                    model_corr,
                    color=color,
                    lw=2.0,
                    marker="o",
                    ms=3.7,
                    label=f"W{int(lead)} FlowMatch",
                )

                gain = line["corr_gain_pct_mean"].to_numpy(dtype=float)
                gain_values.extend(gain[np.isfinite(gain)].tolist())
                lo = line["corr_gain_pct_p05"].to_numpy(dtype=float)
                hi = line["corr_gain_pct_p95"].to_numpy(dtype=float)
                if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                    ax_gain.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
                    gain_values.extend(lo[np.isfinite(lo)].tolist())
                    gain_values.extend(hi[np.isfinite(hi)].tolist())
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
                ax_gain.plot(
                    x,
                    gain,
                    color=color,
                    lw=1.8,
                    ls=LEAD_LINESTYLES.get(int(lead), "-"),
                    marker="o",
                    ms=3.8,
                    label=f"W{int(lead)}",
                )

            ax_corr.set_title(title_corr, loc="left", fontweight="bold", fontsize=10)
            ax_gain.set_title(title_gain, loc="left", fontweight="bold", fontsize=10)
            style_axis(ax_corr)
            style_axis(ax_gain, zero_line=True)
            if vi == len(variables) - 1:
                ax_gain.set_xlabel("Generated members")
            if ti == 0:
                ax_corr.set_ylabel("Correlation")
                ax_gain.set_ylabel("FlowMatch gain (%)")
            if vi == 0 and ti == 1:
                ax_corr.text(
                    0.99,
                    0.04,
                    "solid: FlowMatch\nmatching dashed: GEOS",
                    transform=ax_corr.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="#40515f",
                )

    corr_arr = np.asarray(corr_values, dtype=float)
    corr_arr = corr_arr[np.isfinite(corr_arr)]
    if corr_arr.size:
        ymin = max(-1.0, float(np.nanmin(corr_arr)) - 0.04)
        ymax = min(1.0, float(np.nanmax(corr_arr)) + 0.04)
        if ymax <= ymin:
            ymin, ymax = -0.1, 1.0
        for ax in axes[0::2, :].ravel():
            if ax.has_data():
                ax.set_ylim(ymin, ymax)
    gain_limit = robust_symmetric_limit(gain_values, minimum=5.0)
    for ax in axes[1::2, :].ravel():
        if ax.has_data():
            ax.set_ylim(-gain_limit, gain_limit)

    lead_handles = []
    for lead in sorted(set(summary["lead"].astype(int))):
        handle, = axes[0, 0].plot([], [], color=LEAD_COLORS.get(int(lead), "#2f6f9f"), lw=2.0,
                                  marker="o", ms=3.5, label=f"W{int(lead)}")
        lead_handles.append(handle)
    fig.legend(lead_handles, [handle.get_label() for handle in lead_handles],
               loc="upper center", ncol=min(4, len(lead_handles)), frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=8.5)
    fig.suptitle("Extreme-event correlation and FlowMatch gain as generated ensemble size increases", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / "ensemble_correlation_extreme_dashboard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def read_existing_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def make_plots(out_dir: Path) -> list[str]:
    summary = read_existing_csv(out_dir / "ensemble_correlation_summary.csv")
    bootstrap = read_existing_csv(out_dir / "ensemble_correlation_case_bootstrap_ci.csv")
    path = plot_correlation_dashboard(summary, bootstrap, out_dir)
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
        "extreme_event_count_requested": int(args.extreme_event_count),
        "extreme_event_count_per_lead": bool(args.extreme_event_count_per_lead),
        "extreme_event_variables": event_variables,
        "extreme_event_regions": event_regions,
        "extreme_event_max_per_region": int(args.extreme_event_max_per_region),
        "selected_extreme_events": selected_events,
        "targets": list(TARGETS),
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
                    raise ValueError(f"Event region {region!r} has too few valid cells for correlation.")

                obs = ens.load_obs_array(ds, spec["obs"], init_idx, lead_idx)
                model = ens.load_forecast_array(ds, spec["model"], init_idx, lead_idx)
                geos = ens.load_forecast_array(ds, spec["geos"], init_idx, lead_idx)
                obs_vec = obs[case_mask]
                weight_vec = weights[case_mask]
                model_points = model[:, case_mask]
                geos_points = geos[:, case_mask]
                model_members = int(model_points.shape[0])
                geos_targets = {
                    target: target_vector(geos_points, target)
                    for target in TARGETS
                }
                usable_sizes = [size for size in sample_sizes if int(size) <= model_members]
                if not usable_sizes:
                    usable_sizes = [model_members]
                case_id = f"{year}_{init_idx:04d}_lead{lead_value}_{region}_{variable}"
                geos_sums = {
                    target: weighted_corr_sums(geos_targets[target], obs_vec, weight_vec, "geos")
                    for target in TARGETS
                }

                for size in usable_sizes:
                    repeats = 1 if int(size) >= model_members else max(1, int(args.member_bootstrap_repeats))
                    for member_repeat in range(repeats):
                        if int(size) >= model_members:
                            member_idx = np.arange(model_members)
                        else:
                            member_idx = rng.choice(model_members, size=int(size), replace=False)
                        sample_points = model_points[member_idx, :]
                        model_targets = {
                            target: target_vector(sample_points, target)
                            for target in TARGETS
                        }
                        for target in TARGETS:
                            row = {
                                "case_id": case_id,
                                "year": int(year),
                                "init_index": init_idx,
                                "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
                                "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
                                "lead": lead_value,
                                "variable": variable,
                                "target": target,
                                "member_count": int(size),
                                "member_repeat": int(member_repeat),
                                "model_members_available": model_members,
                                "geos_members_available": int(geos_points.shape[0]),
                                "eval_mask": args.eval_mask,
                                "region": region,
                                "region_name": str(ens.REGIONS[region]["name"]),
                                "event_rank": event.get("event_rank", ""),
                                "event_selection_lead": event.get("event_selection_lead", ""),
                                "event_score": event.get("event_score", np.nan),
                                "event_score_variable": variable,
                            }
                            row.update(weighted_corr_sums(model_targets[target], obs_vec, weight_vec, "model"))
                            row.update(geos_sums[target])
                            row.update(add_corr_metrics(row))
                            case_rows.append(row)
                processed_cases += 1
                if processed_cases % 20 == 0:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Processed {processed_cases} extreme cases in {elapsed:.1f} min")
        finally:
            ds.close()

    if not case_rows:
        raise RuntimeError("No extreme-event correlation rows were processed.")

    case_df = pd.DataFrame(case_rows)
    repeat_summary = aggregate_sums(case_df, ["variable", "lead", "target", "member_count", "member_repeat"])
    summary = summarize_member_repeats(repeat_summary)
    bootstrap = bootstrap_case_intervals(case_df, int(args.case_bootstrap_repeats), rng)

    write_csv(case_df, out_dir / "case_member_correlation_sums.csv")
    write_csv(repeat_summary, out_dir / "ensemble_correlation_member_repeat_summary.csv")
    write_csv(summary, out_dir / "ensemble_correlation_summary.csv")
    if not bootstrap.empty:
        write_csv(bootstrap, out_dir / "ensemble_correlation_case_bootstrap_ci.csv")
    report_path = write_member_report(summary, out_dir, int(args.report_member_count))

    metadata["processed_extreme_cases"] = int(processed_cases)
    metadata["case_member_rows"] = int(len(case_df))
    metadata["completed_at"] = timestamp_now_utc()
    metadata["outputs"] = {
        "case_member_correlation_sums": str(out_dir / "case_member_correlation_sums.csv"),
        "ensemble_correlation_member_repeat_summary": str(out_dir / "ensemble_correlation_member_repeat_summary.csv"),
        "ensemble_correlation_summary": str(out_dir / "ensemble_correlation_summary.csv"),
        "ensemble_correlation_case_bootstrap_ci": (
            str(out_dir / "ensemble_correlation_case_bootstrap_ci.csv") if not bootstrap.empty else None
        ),
        "ensemble_correlation_member_report": report_path,
    }
    metadata_path = out_dir / "ensemble_correlation_metadata.json"
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
        write_member_report(read_existing_csv(out_dir / "ensemble_correlation_summary.csv"),
                            out_dir, int(args.report_member_count))
        if args.make_plots:
            make_plots(out_dir)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
