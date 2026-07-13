#!/usr/bin/env python3
"""Tail-quantile ensemble-size diagnostics for observed extreme events.

This standalone companion to evaluate_ensemble_tests_flow_finalv1_global.py
keeps the existing CRPS/ensemble-mean RMSE products unchanged. For each
generated-member subset it verifies the ensemble q95 and q99 forecasts using:

  * quantile (pinball) score, the proper score for a forecast quantile; and
  * RMSE of the forecast quantile, a tail-amplitude error diagnostic.

Both metrics are reported as skill relative to the corresponding quantile from
the saved lagged-FIMr1p1 ensemble. Extreme cases and regional masks are selected
with the same routines used by the existing Fig. 5 evaluator.
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


BASELINE_LABEL = "FIMr1p1"
MODEL_LABEL = "FIMr1p1-FlowMatch"
LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}
LEAD_LINESTYLES = {1: ":", 2: "-.", 3: "--", 4: "-"}
SUM_COLUMNS = [
    "model_weight_sum",
    "model_quantile_score_sum",
    "model_sse_sum",
    "model_error_sum",
    "model_forecast_sum",
    "geos_weight_sum",
    "geos_quantile_score_sum",
    "geos_sse_sum",
    "geos_error_sum",
    "geos_forecast_sum",
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
            "ensemble_quantile_extreme_t2m30_pr30_regions_2021_2023_"
            "w1w4_q95q99_memberboot90_caseboot15"
        ),
    )
    parser.add_argument("--variables", default="pr,t2m")
    parser.add_argument("--quantiles", default="0.95,0.99", help="Comma-separated forecast quantiles.")
    parser.add_argument("--sample_sizes", default="6,10,20,30,60,90")
    parser.add_argument("--member_bootstrap_repeats", type=int, default=90)
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
    parser.add_argument("--extreme_event_max_per_region", type=int, default=10)
    parser.add_argument(
        "--extreme_event_count_per_lead",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow_missing_years", action="store_true")
    parser.add_argument("--report_member_count", type=int, default=6)
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_quantiles(text: str) -> list[float]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value < 1.0:
            raise ValueError(f"Quantiles must be between 0 and 1; got {value}.")
        values.append(value)
    if not values:
        raise ValueError("At least one quantile is required.")
    return sorted(set(values))


def quantile_name(quantile: float) -> str:
    return f"q{int(round(100.0 * quantile)):02d}"


def timestamp_now_utc() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def forecast_quantiles(ensemble: np.ndarray, quantiles: list[float]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanquantile(np.asarray(ensemble, dtype=np.float64), quantiles, axis=0)


def paired_quantile_sums(
    model_q: np.ndarray,
    geos_q: np.ndarray,
    obs: np.ndarray,
    quantile: float,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    model_q = np.asarray(model_q, dtype=np.float64)
    geos_q = np.asarray(geos_q, dtype=np.float64)
    obs64 = np.asarray(obs, dtype=np.float64)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(obs64)
        & np.isfinite(model_q)
        & np.isfinite(geos_q)
    )
    weight = np.where(valid, np.asarray(weights, dtype=np.float64), 0.0)
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0.0:
        return {column: 0.0 for column in SUM_COLUMNS}

    row: dict[str, float] = {}
    for prefix, forecast in (("model", model_q), ("geos", geos_q)):
        error = forecast - obs64
        # L_tau(q, y) = tau * (y-q) for underprediction and
        # (1-tau) * (q-y) for overprediction.
        loss = np.where(obs64 >= forecast, quantile * (obs64 - forecast), (1.0 - quantile) * error)
        row[f"{prefix}_weight_sum"] = weight_sum
        row[f"{prefix}_quantile_score_sum"] = float(np.sum(np.where(valid, loss, 0.0) * weight))
        row[f"{prefix}_sse_sum"] = float(np.sum(np.where(valid, error * error, 0.0) * weight))
        row[f"{prefix}_error_sum"] = float(np.sum(np.where(valid, error, 0.0) * weight))
        row[f"{prefix}_forecast_sum"] = float(np.sum(np.where(valid, forecast, 0.0) * weight))
    return row


def row_metrics(row: dict[str, float] | pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in ("model", "geos"):
        weight = float(row.get(f"{prefix}_weight_sum", 0.0))
        if weight <= 0.0:
            out[f"{prefix}_quantile_score"] = np.nan
            out[f"{prefix}_quantile_rmse"] = np.nan
            out[f"{prefix}_quantile_bias"] = np.nan
            out[f"{prefix}_forecast_quantile_mean"] = np.nan
            continue
        out[f"{prefix}_quantile_score"] = float(row[f"{prefix}_quantile_score_sum"] / weight)
        out[f"{prefix}_quantile_rmse"] = float(np.sqrt(row[f"{prefix}_sse_sum"] / weight))
        out[f"{prefix}_quantile_bias"] = float(row[f"{prefix}_error_sum"] / weight)
        out[f"{prefix}_forecast_quantile_mean"] = float(row[f"{prefix}_forecast_sum"] / weight)

    model_score = out["model_quantile_score"]
    geos_score = out["geos_quantile_score"]
    model_rmse = out["model_quantile_rmse"]
    geos_rmse = out["geos_quantile_rmse"]
    out["quantile_skill_pct"] = (
        100.0 * (1.0 - model_score / geos_score)
        if np.isfinite(model_score) and np.isfinite(geos_score) and geos_score > 1e-12
        else np.nan
    )
    out["quantile_rmse_skill_pct"] = (
        100.0 * (1.0 - model_rmse / geos_rmse)
        if np.isfinite(model_rmse) and np.isfinite(geos_rmse) and geos_rmse > 1e-12
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
    rows = []
    for row in sums.to_dict("records"):
        row.update(row_metrics(row))
        rows.append(row)
    return pd.DataFrame(rows)


SUMMARY_METRICS = (
    "model_quantile_score",
    "geos_quantile_score",
    "quantile_skill_pct",
    "model_quantile_rmse",
    "geos_quantile_rmse",
    "quantile_rmse_skill_pct",
    "model_quantile_bias",
    "geos_quantile_bias",
    "model_forecast_quantile_mean",
    "geos_forecast_quantile_mean",
)


def summarize_member_repeats(repeat_summary: pd.DataFrame) -> pd.DataFrame:
    if repeat_summary.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "quantile", "member_count"]
    for key, group in repeat_summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_member_repeats"] = int(group["member_repeat"].nunique())
        row["n_cases"] = int(group["n_cases"].max())
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_p05"] = float(np.quantile(values, 0.05)) if values.size else np.nan
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50)) if values.size else np.nan
            row[f"{metric}_p95"] = float(np.quantile(values, 0.95)) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def bootstrap_case_intervals(
    case_df: pd.DataFrame,
    repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if repeats <= 0 or case_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["variable", "lead", "quantile", "member_count"]
    metrics = ("quantile_skill_pct", "quantile_rmse_skill_pct")
    for key, group in case_df.groupby(group_cols, dropna=False):
        cases = sorted(group["case_id"].astype(str).unique())
        rows_by_case = {
            case: group[group["case_id"].astype(str).eq(case)].reset_index(drop=True)
            for case in cases
        }
        boot = {metric: [] for metric in metrics}
        for _ in range(repeats):
            state = {column: 0.0 for column in SUM_COLUMNS}
            for case in rng.choice(cases, size=len(cases), replace=True):
                candidates = rows_by_case[str(case)]
                picked = candidates.iloc[int(rng.integers(0, len(candidates)))]
                for column in SUM_COLUMNS:
                    state[column] += float(picked[column])
            scores = row_metrics(state)
            for metric in metrics:
                boot[metric].append(scores[metric])
        row = dict(zip(group_cols, key))
        row["case_bootstrap_repeats"] = int(repeats)
        row["n_cases"] = int(len(cases))
        for metric, values in boot.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            row[f"{metric}_p025"] = float(np.quantile(arr, 0.025)) if arr.size else np.nan
            row[f"{metric}_p50"] = float(np.quantile(arr, 0.50)) if arr.size else np.nan
            row[f"{metric}_p975"] = float(np.quantile(arr, 0.975)) if arr.size else np.nan
            row[f"{metric}_p_gt0"] = float(np.mean(arr > 0.0)) if arr.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_member_report(summary: pd.DataFrame, out_dir: Path, member_count: int) -> str | None:
    if member_count <= 0 or summary.empty:
        return None
    sub = summary[summary["member_count"].astype(int).eq(int(member_count))].copy()
    if sub.empty:
        print(f"No tail-quantile rows found for ensemble size {member_count}.")
        return None
    columns = [
        "variable",
        "lead",
        "quantile",
        "member_count",
        "quantile_skill_pct_mean",
        "quantile_skill_pct_p05",
        "quantile_skill_pct_p95",
        "quantile_rmse_skill_pct_mean",
        "quantile_rmse_skill_pct_p05",
        "quantile_rmse_skill_pct_p95",
        "model_quantile_score_mean",
        "geos_quantile_score_mean",
        "model_quantile_rmse_mean",
        "geos_quantile_rmse_mean",
        "n_cases",
    ]
    sub = sub[[column for column in columns if column in sub.columns]].sort_values(
        ["variable", "quantile", "lead"]
    )
    path = out_dir / f"ensemble_quantile_member{int(member_count)}_report.csv"
    write_csv(sub, path)
    print(f"\nEnsemble size {int(member_count)} q95/q99 report:")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(sub.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    return str(path)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _metric_range(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return -5.0, 5.0
    low = min(0.0, float(np.nanmin(array)))
    high = max(0.0, float(np.nanmax(array)))
    pad = max(1.0, 0.08 * (high - low))
    return low - pad, high + pad


def plot_quantile_figure(summary: pd.DataFrame, out_dir: Path, quantile: float) -> Path | None:
    sub_q = summary[np.isclose(summary["quantile"].astype(float), float(quantile))].copy()
    if sub_q.empty:
        return None
    plt = _import_matplotlib()
    specs = (
        ("quantile_skill_pct", "quantile-score skill (%)"),
        ("quantile_rmse_skill_pct", "quantile RMSE skill (%)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)
    letters = iter("abcd")
    values_by_column: list[list[float]] = [[], []]
    qlabel = quantile_name(quantile)

    for variable_index, variable in enumerate(("pr", "t2m")):
        variable_rows = sub_q[sub_q["variable"].astype(str).str.lower().eq(variable)]
        for metric_index, (metric, label) in enumerate(specs):
            ax = axes[variable_index, metric_index]
            letter = next(letters)
            variable_label = "PR" if variable == "pr" else "T2M"
            ax.set_title(f"({letter}) {variable_label} {qlabel} {label}", loc="left", fontweight="bold")
            raw_reference = []
            for lead in sorted(variable_rows["lead"].astype(int).unique()):
                line = variable_rows[variable_rows["lead"].astype(int).eq(lead)].sort_values("member_count")
                x = line["member_count"].to_numpy(dtype=float)
                mean = line[f"{metric}_mean"].to_numpy(dtype=float)
                low = line[f"{metric}_p05"].to_numpy(dtype=float)
                high = line[f"{metric}_p95"].to_numpy(dtype=float)
                color = LEAD_COLORS.get(int(lead), "#2f6f9f")
                values_by_column[metric_index].extend(mean[np.isfinite(mean)].tolist())
                values_by_column[metric_index].extend(low[np.isfinite(low)].tolist())
                values_by_column[metric_index].extend(high[np.isfinite(high)].tolist())
                ax.fill_between(x, low, high, color=color, alpha=0.16, lw=0)
                ax.plot(
                    x,
                    mean,
                    color=color,
                    lw=1.7,
                    ls=LEAD_LINESTYLES.get(int(lead), "-"),
                    marker="o",
                    ms=3.6,
                    label=f"W{int(lead)}",
                )
                raw_metric = "geos_quantile_score_mean" if metric_index == 0 else "geos_quantile_rmse_mean"
                raw_value = float(np.nanmean(pd.to_numeric(line[raw_metric], errors="coerce")))
                if np.isfinite(raw_value):
                    raw_reference.append((int(lead), raw_value))

            ax.axhline(0.0, color="#7a8794", lw=0.9, ls="--")
            ax.grid(True, axis="y", color="#dde3e8", linewidth=0.6, alpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(0.0, 90.0)
            if variable_index == 1:
                ax.set_xlabel("Generated members")
                ax.set_xticks(np.arange(0, 91, 10))
            if metric_index == 0:
                ax.set_ylabel("Precipitation" if variable == "pr" else "2 m temperature")
            if variable_index == 0 and metric_index == 1:
                ax.legend(loc="best", frameon=False, fontsize=7.5)

            if raw_reference:
                ax2 = ax.twinx()
                for lead, raw_value in raw_reference:
                    color = LEAD_COLORS.get(int(lead), "#2f6f9f")
                    ax2.plot([84.0, 90.0], [raw_value, raw_value], color=color, lw=1.4)
                    ax2.plot(90.0, raw_value, marker="<", color=color, ms=4.5, clip_on=False)
                raw_values = [value for _, value in raw_reference]
                raw_low, raw_high = min(raw_values), max(raw_values)
                raw_pad = max(0.10 * (raw_high - raw_low), 0.08 * max(abs(raw_high), 1e-6))
                ax2.set_ylim(max(0.0, raw_low - raw_pad), raw_high + raw_pad)
                raw_label = "quantile score" if metric_index == 0 else "quantile RMSE"
                ax2.set_ylabel(f"{BASELINE_LABEL} raw {qlabel} {raw_label}", fontsize=8, color="#66727e")
                ax2.tick_params(labelsize=7, colors="#66727e")
                ax2.spines["right"].set_color("#aab5c0")
                ax2.spines["top"].set_visible(False)

    for metric_index in range(2):
        ylim = _metric_range(values_by_column[metric_index])
        for variable_index in range(2):
            axes[variable_index, metric_index].set_ylim(*ylim)

    fig.suptitle(
        f"Extreme-event {qlabel} forecast convergence: {MODEL_LABEL} vs {BASELINE_LABEL}",
        fontsize=11.5,
    )
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.10, top=0.90, hspace=0.28, wspace=0.40)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"fig5_member_convergence_{qlabel}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def read_existing_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def make_plots(out_dir: Path, quantiles: list[float]) -> list[str]:
    summary = read_existing_csv(out_dir / "ensemble_quantile_size_summary.csv")
    paths = [plot_quantile_figure(summary, out_dir, quantile) for quantile in quantiles]
    return [str(path) for path in paths if path is not None]


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    variables = ens.parse_variables(args.variables)
    quantiles = parse_quantiles(args.quantiles)
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

    print(
        "Selecting observed extreme-event cases; this scans observed fields before "
        "forecast verification starts.",
        flush=True,
    )
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
    print(f"Selected {len(selected_events)} total extreme-event cases.", flush=True)
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
        "quantiles": quantiles,
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
        "probabilistic_metric": "pinball/quantile score",
        "tail_amplitude_metric": "RMSE of forecast quantile",
        "reference": BASELINE_LABEL,
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
        print(f"Opening {path} for {len(year_events)} selected events", flush=True)
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
                if int(np.sum(case_mask)) <= 0:
                    raise ValueError(f"Event region {region!r} kept zero grid cells.")

                # Restrict the quantile calculation to the verification cells.
                # This is equivalent to full-grid quantiles followed by masking,
                # but avoids doing q95/q99 work for unrelated grid cells.
                obs_full = ens.load_obs_array(ds, spec["obs"], init_idx, lead_idx)
                model_full = ens.load_forecast_array(ds, spec["model"], init_idx, lead_idx)
                geos_full = ens.load_forecast_array(ds, spec["geos"], init_idx, lead_idx)
                obs = np.asarray(obs_full[case_mask], dtype=np.float32)
                model = np.asarray(model_full[:, case_mask], dtype=np.float32)
                geos = np.asarray(geos_full[:, case_mask], dtype=np.float32)
                case_weights = np.asarray(weights[case_mask], dtype=np.float64)
                case_mask_vector = np.ones(obs.shape, dtype=bool)
                model_members = int(model.shape[0])
                usable_sizes = [size for size in sample_sizes if int(size) <= model_members]
                if not usable_sizes:
                    usable_sizes = [model_members]
                case_id = f"{year}_{init_idx:04d}_lead{lead_value}_{region}_{variable}"
                geos_quantile_fields = forecast_quantiles(geos, quantiles)

                for size in usable_sizes:
                    repeats = 1 if int(size) >= model_members else max(1, int(args.member_bootstrap_repeats))
                    for member_repeat in range(repeats):
                        if int(size) >= model_members:
                            member_idx = np.arange(model_members)
                        else:
                            member_idx = rng.choice(model_members, size=int(size), replace=False)
                        model_sample = model[member_idx, ...]
                        model_quantile_fields = forecast_quantiles(model_sample, quantiles)
                        for quantile_index, quantile in enumerate(quantiles):
                            sums = paired_quantile_sums(
                                model_quantile_fields[quantile_index],
                                geos_quantile_fields[quantile_index],
                                obs,
                                quantile,
                                case_weights,
                                case_mask_vector,
                            )
                            row = {
                                "case_id": case_id,
                                "year": int(year),
                                "init_index": init_idx,
                                "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
                                "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
                                "lead": lead_value,
                                "variable": variable,
                                "quantile": float(quantile),
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
                            row.update(sums)
                            row.update(row_metrics(row))
                            case_rows.append(row)
                processed_cases += 1
                if processed_cases == 1 or processed_cases % 20 == 0:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Processed {processed_cases} extreme cases in {elapsed:.1f} min", flush=True)
        finally:
            ds.close()

    if not case_rows:
        raise RuntimeError("No extreme-event quantile rows were processed.")

    case_df = pd.DataFrame(case_rows)
    group_cols = ["variable", "lead", "quantile", "member_count", "member_repeat"]
    repeat_summary = aggregate_sums(case_df, group_cols)
    summary = summarize_member_repeats(repeat_summary)
    bootstrap = bootstrap_case_intervals(case_df, int(args.case_bootstrap_repeats), rng)

    write_csv(case_df, out_dir / "case_member_quantile_metrics.csv")
    write_csv(repeat_summary, out_dir / "ensemble_quantile_member_repeat_summary.csv")
    write_csv(summary, out_dir / "ensemble_quantile_size_summary.csv")
    if not bootstrap.empty:
        write_csv(bootstrap, out_dir / "ensemble_quantile_case_bootstrap_ci.csv")
    report_path = write_member_report(summary, out_dir, int(args.report_member_count))

    metadata["processed_extreme_cases"] = int(processed_cases)
    metadata["case_member_rows"] = int(len(case_df))
    metadata["completed_at"] = timestamp_now_utc()
    metadata["outputs"] = {
        "case_member_quantile_metrics": str(out_dir / "case_member_quantile_metrics.csv"),
        "ensemble_quantile_member_repeat_summary": str(out_dir / "ensemble_quantile_member_repeat_summary.csv"),
        "ensemble_quantile_size_summary": str(out_dir / "ensemble_quantile_size_summary.csv"),
        "ensemble_quantile_case_bootstrap_ci": (
            str(out_dir / "ensemble_quantile_case_bootstrap_ci.csv") if not bootstrap.empty else None
        ),
        "ensemble_quantile_member_report": report_path,
    }
    metadata_path = out_dir / "ensemble_quantile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {metadata_path}")

    if args.make_plots:
        metadata["outputs"]["plots"] = make_plots(out_dir, quantiles)
        metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    quantiles = parse_quantiles(args.quantiles)
    if args.plot_only:
        summary = read_existing_csv(out_dir / "ensemble_quantile_size_summary.csv")
        write_member_report(summary, out_dir, int(args.report_member_count))
        if args.make_plots:
            make_plots(out_dir, quantiles)
        return
    evaluate(args)


if __name__ == "__main__":
    main()
