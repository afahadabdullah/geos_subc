#!/usr/bin/env python3
"""
Evaluate generated flow_finalv1_global forecast Zarr stores against observations.

Each yearly Zarr store is expected to contain:
  model_pr/model_t2m: (init, ensemble, lead, lat, lon)
  geos_pr/geos_t2m:   (init, geos_member, lead, lat, lon)
  obs_pr/obs_t2m:     (init, lead, lat, lon)

The evaluator writes:
  - one per-year per-init/per-lead metrics CSV, used for resume
  - one combined per-init/per-lead metrics CSV
  - one grouped summary CSV with all, lead, month, and season aggregations
  - optional diagnostic skill plots
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = {
    "pr": {
        "model": "model_pr",
        "geos": "geos_pr",
        "obs": "obs_pr",
        "units": "mm/day",
    },
    "t2m": {
        "model": "model_t2m",
        "geos": "geos_t2m",
        "obs": "obs_t2m",
        "units": "K",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated global flow_finalv1 forecast Zarr stores."
    )
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="dataprocess/gen_flow_finalv1_global_junjul_2021_2024_e90_s50",
        help="Directory containing YEAR.zarr stores.",
    )
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--skip_years", type=str, default="", help="Comma-separated years to skip.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ml_output_flow_finalv1_global_noisectx_t2mres/zarr_eval_global_2021_2024_e90_s50",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="pr,t2m",
        help="Comma-separated subset of pr,t2m.",
    )
    parser.add_argument(
        "--max_runtime_minutes",
        type=float,
        default=None,
        help="Stop cleanly before starting a new year after this many minutes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--make_plots", action="store_true")
    return parser.parse_args()


def parse_years(text):
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def parse_variables(text):
    variables = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    bad = [v for v in variables if v not in VARIABLES]
    if bad:
        raise ValueError(f"Unknown variables {bad}; valid options are {sorted(VARIABLES)}")
    if not variables:
        raise ValueError("--variables cannot be empty")
    return variables


def season_name(month):
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def month_label(month):
    return f"{int(month):02d}"


def deadline_reached(deadline):
    return deadline is not None and time.monotonic() >= deadline


def area_weights_from_lats(lats):
    weights = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64)))
    weights = np.clip(weights, 0.0, None)
    return weights[:, None].astype(np.float64)


def weighted_mean(field, weights, mask):
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    weighted_mask = np.where(mask, weights, 0.0)
    denom = np.sum(weighted_mask)
    if denom <= 0:
        return np.nan
    return float(np.sum(np.where(mask, field, 0.0) * weighted_mask) / denom)


def crps_ensemble(ensemble, obs, weights):
    """
    Exact empirical CRPS for an ensemble at every grid point, then area-weighted.

    CRPS = mean(|x_i-y|) - (1 / (2E^2)) * sum_ij |x_i-x_j|
    The second term is computed from sorted samples to avoid E^2 memory/time.
    """
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    if ensemble.ndim != 3:
        raise ValueError(f"Expected ensemble [E,H,W], got shape {ensemble.shape}")
    if obs.ndim != 2:
        raise ValueError(f"Expected obs [H,W], got shape {obs.shape}")

    finite = np.isfinite(obs) & np.all(np.isfinite(ensemble), axis=0)
    if not finite.any():
        return np.nan

    ens64 = ensemble.astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    mae_term = np.mean(np.abs(ens64 - obs64[None, :, :]), axis=0)

    ens_sorted = np.sort(ens64, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    crps_map = mae_term - spread_term
    return weighted_mean(crps_map, weights, finite)


def deterministic_metrics(ensemble, obs, weights):
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mean = np.nanmean(ensemble, axis=0).astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    finite = np.isfinite(obs64) & np.isfinite(mean)
    if not finite.any():
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "spread": np.nan,
        }
    err = mean - obs64
    mse = weighted_mean(err * err, weights, finite)
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    mae = weighted_mean(np.abs(err), weights, finite)
    bias = weighted_mean(err, weights, finite)
    spread_map = np.nanstd(ensemble.astype(np.float64, copy=False), axis=0)
    spread = weighted_mean(spread_map, weights, finite)
    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "spread": spread,
    }


def evaluate_ensemble(ensemble, obs, weights):
    out = deterministic_metrics(ensemble, obs, weights)
    out["crps"] = crps_ensemble(ensemble, obs, weights)
    out["ensemble_members"] = int(np.asarray(ensemble).shape[0])
    return out


def safe_ratio_skill(model_value, geos_value):
    if not np.isfinite(model_value) or not np.isfinite(geos_value) or abs(geos_value) < 1e-12:
        return np.nan
    return 1.0 - (model_value / geos_value)


def add_skill_columns(df):
    df = df.copy()
    for metric in ("crps", "rmse", "mae"):
        df[f"{metric}_skill_vs_geos"] = [
            safe_ratio_skill(m, g)
            for m, g in zip(df[f"model_{metric}"], df[f"geos_{metric}"])
        ]
        df[f"{metric}_improvement_pct"] = 100.0 * df[f"{metric}_skill_vs_geos"]
    df["abs_bias_skill_vs_geos"] = [
        safe_ratio_skill(abs(m), abs(g))
        for m, g in zip(df["model_bias"], df["geos_bias"])
    ]
    df["abs_bias_improvement_pct"] = 100.0 * df["abs_bias_skill_vs_geos"]
    return df


def evaluate_year(year, zarr_path, out_csv, variables, overwrite=False):
    if os.path.exists(out_csv) and not overwrite:
        print(f"✅ {year}: using existing per-init metrics {out_csv}")
        return pd.read_csv(out_csv, parse_dates=["init", "valid_time"])

    print(f"🔎 {year}: evaluating {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
    try:
        lats = ds["lat"].values
        weights = area_weights_from_lats(lats)
        init_values = pd.to_datetime(ds["init"].values).normalize()
        lead_values = ds["lead"].values
        rows = []

        for init_idx, init_time in enumerate(init_values):
            init_month = int(init_time.month)
            init_year = int(init_time.year)
            for lead_idx, lead_value in enumerate(lead_values):
                if "valid_time" in ds:
                    valid_time = pd.Timestamp(ds["valid_time"].isel(init=init_idx, lead=lead_idx).values)
                else:
                    valid_time = init_time + pd.to_timedelta(int(lead_value) * 7, unit="D")
                valid_month = int(valid_time.month)
                for variable in variables:
                    names = VARIABLES[variable]
                    obs = ds[names["obs"]].isel(init=init_idx, lead=lead_idx).values
                    model = ds[names["model"]].isel(init=init_idx, lead=lead_idx).values
                    geos = ds[names["geos"]].isel(init=init_idx, lead=lead_idx).values

                    model_metrics = evaluate_ensemble(model, obs, weights)
                    geos_metrics = evaluate_ensemble(geos, obs, weights)
                    row = {
                        "year": year,
                        "init": init_time,
                        "init_year": init_year,
                        "init_month": init_month,
                        "init_month_label": month_label(init_month),
                        "init_season": season_name(init_month),
                        "lead": int(lead_value),
                        "lead_label": f"week{int(lead_value)}",
                        "valid_time": valid_time,
                        "valid_year": int(valid_time.year),
                        "valid_month": valid_month,
                        "valid_month_label": month_label(valid_month),
                        "valid_season": season_name(valid_month),
                        "variable": variable,
                        "units": names["units"],
                        "model_members": model_metrics["ensemble_members"],
                        "geos_members": geos_metrics["ensemble_members"],
                    }
                    for metric in ("crps", "rmse", "mae", "bias", "spread"):
                        row[f"model_{metric}"] = model_metrics[metric]
                        row[f"geos_{metric}"] = geos_metrics[metric]
                    rows.append(row)

        out = add_skill_columns(pd.DataFrame(rows))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"✅ {year}: wrote {len(out)} rows to {out_csv}")
        return out
    finally:
        ds.close()


def aggregate_group(df, group_type, group_cols):
    value_cols = [
        "model_crps",
        "geos_crps",
        "model_rmse",
        "geos_rmse",
        "model_mae",
        "geos_mae",
        "model_bias",
        "geos_bias",
        "model_spread",
        "geos_spread",
        "model_members",
        "geos_members",
    ]
    if group_cols:
        grouped = df.groupby(["variable"] + group_cols, dropna=False)
        out = grouped.agg(
            n_samples=("model_crps", "count"),
            n_init_dates=("init", "nunique"),
            **{col: (col, "mean") for col in value_cols},
        ).reset_index()
    else:
        grouped = df.groupby(["variable"], dropna=False)
        out = grouped.agg(
            n_samples=("model_crps", "count"),
            n_init_dates=("init", "nunique"),
            **{col: (col, "mean") for col in value_cols},
        ).reset_index()
    out.insert(0, "group_type", group_type)
    for col in (
        "lead",
        "lead_label",
        "init_month",
        "init_month_label",
        "init_season",
        "valid_month",
        "valid_month_label",
        "valid_season",
    ):
        if col not in out.columns:
            out[col] = "all" if col.endswith("label") or col.endswith("season") else np.nan
    return add_skill_columns(out)


def build_summary(df):
    groups = [
        ("all", []),
        ("lead", ["lead", "lead_label"]),
        ("init_month", ["init_month", "init_month_label"]),
        ("init_month_lead", ["init_month", "init_month_label", "lead", "lead_label"]),
        ("init_season", ["init_season"]),
        ("init_season_lead", ["init_season", "lead", "lead_label"]),
        ("valid_month", ["valid_month", "valid_month_label"]),
        ("valid_month_lead", ["valid_month", "valid_month_label", "lead", "lead_label"]),
        ("valid_season", ["valid_season"]),
        ("valid_season_lead", ["valid_season", "lead", "lead_label"]),
    ]
    summary = pd.concat([aggregate_group(df, name, cols) for name, cols in groups], ignore_index=True)
    ordered_cols = [
        "group_type",
        "variable",
        "lead",
        "lead_label",
        "init_month",
        "init_month_label",
        "init_season",
        "valid_month",
        "valid_month_label",
        "valid_season",
        "n_samples",
        "n_init_dates",
        "model_members",
        "geos_members",
        "model_crps",
        "geos_crps",
        "crps_skill_vs_geos",
        "crps_improvement_pct",
        "model_rmse",
        "geos_rmse",
        "rmse_skill_vs_geos",
        "rmse_improvement_pct",
        "model_mae",
        "geos_mae",
        "mae_skill_vs_geos",
        "mae_improvement_pct",
        "model_bias",
        "geos_bias",
        "abs_bias_skill_vs_geos",
        "abs_bias_improvement_pct",
        "model_spread",
        "geos_spread",
    ]
    return summary[ordered_cols]


def maybe_make_plots(summary, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def plot_group(group_type, x_col, label_col, filename):
        sub = summary[summary["group_type"].eq(group_type)].copy()
        if sub.empty:
            return
        labels = list(dict.fromkeys(sub[label_col].astype(str).tolist()))
        variables = [v for v in ("pr", "t2m") if v in set(sub["variable"])]
        fig, axes = plt.subplots(len(variables), 2, figsize=(12, 4 * len(variables)), squeeze=False)
        for row_idx, variable in enumerate(variables):
            vdf = sub[sub["variable"].eq(variable)].copy()
            vdf[label_col] = vdf[label_col].astype(str)
            vdf = vdf.set_index(label_col).reindex(labels).reset_index()
            x = np.arange(len(labels))
            axes[row_idx, 0].bar(x, vdf["crps_improvement_pct"])
            axes[row_idx, 0].axhline(0, color="k", linewidth=0.8)
            axes[row_idx, 0].set_title(f"{variable.upper()} CRPS skill vs GEOS")
            axes[row_idx, 0].set_ylabel("Improvement (%)")
            axes[row_idx, 0].set_xticks(x)
            axes[row_idx, 0].set_xticklabels(labels, rotation=45, ha="right")

            axes[row_idx, 1].bar(x, vdf["rmse_improvement_pct"])
            axes[row_idx, 1].axhline(0, color="k", linewidth=0.8)
            axes[row_idx, 1].set_title(f"{variable.upper()} RMSE skill vs GEOS")
            axes[row_idx, 1].set_ylabel("Improvement (%)")
            axes[row_idx, 1].set_xticks(x)
            axes[row_idx, 1].set_xticklabels(labels, rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

    plot_group("lead", "lead", "lead_label", "skill_by_lead.png")
    plot_group("init_month", "init_month", "init_month_label", "skill_by_init_month.png")
    plot_group("init_season", "init_season", "init_season", "skill_by_init_season.png")
    plot_group("valid_season", "valid_season", "valid_season", "skill_by_valid_season.png")


def print_headline(summary):
    headline = summary[summary["group_type"].eq("all")].copy()
    if headline.empty:
        return
    cols = [
        "variable",
        "n_init_dates",
        "model_crps",
        "geos_crps",
        "crps_improvement_pct",
        "model_rmse",
        "geos_rmse",
        "rmse_improvement_pct",
    ]
    print("\nHeadline all-init/all-lead skill:")
    print(headline[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    args = parse_args()
    variables = parse_variables(args.variables)
    skip_years = parse_years(args.skip_years)
    expected_years = [year for year in range(args.start_year, args.end_year + 1) if year not in skip_years]
    os.makedirs(args.out_dir, exist_ok=True)
    yearly_dir = os.path.join(args.out_dir, "yearly_metrics")
    os.makedirs(yearly_dir, exist_ok=True)

    deadline = None
    if args.max_runtime_minutes is not None and args.max_runtime_minutes > 0:
        deadline = time.monotonic() + float(args.max_runtime_minutes) * 60.0

    print("\n" + "=" * 88)
    print("Global flow_finalv1 forecast Zarr evaluation")
    print(f"  Forecast dir : {args.forecast_dir}")
    print(f"  Years        : {args.start_year}-{args.end_year}")
    print(f"  Skip years   : {args.skip_years or 'none'}")
    print(f"  Variables    : {variables}")
    print(f"  Out dir      : {args.out_dir}")
    print(f"  Soft runtime : {args.max_runtime_minutes if args.max_runtime_minutes else 'disabled'} minutes")
    print("=" * 88 + "\n")

    year_frames = []
    all_complete = True
    for year in expected_years:
        year_csv = os.path.join(yearly_dir, f"{year}_per_init_lead_metrics.csv")
        if deadline_reached(deadline) and not os.path.exists(year_csv):
            print(f"⏸️ Soft runtime reached before starting {year}.")
            all_complete = False
            break
        zarr_path = os.path.join(args.forecast_dir, f"{year}.zarr")
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"Missing forecast Zarr store for {year}: {zarr_path}")
        year_df = evaluate_year(
            year=year,
            zarr_path=zarr_path,
            out_csv=year_csv,
            variables=variables,
            overwrite=args.overwrite,
        )
        year_frames.append(year_df)

    missing_metrics = [
        year
        for year in expected_years
        if not os.path.exists(os.path.join(yearly_dir, f"{year}_per_init_lead_metrics.csv"))
    ]
    if missing_metrics:
        print(f"⏸️ Missing per-year metrics: {missing_metrics}")
        all_complete = False

    if not all_complete:
        print("⏸️ Evaluation incomplete; rerun this script to resume.")
        return 0

    if not year_frames:
        year_frames = [
            pd.read_csv(os.path.join(yearly_dir, f"{year}_per_init_lead_metrics.csv"), parse_dates=["init", "valid_time"])
            for year in expected_years
        ]

    combined = pd.concat(year_frames, ignore_index=True)
    combined = add_skill_columns(combined)
    combined_csv = os.path.join(args.out_dir, "per_init_lead_metrics.csv")
    combined.to_csv(combined_csv, index=False, float_format="%.6f")

    summary = build_summary(combined)
    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    summary.to_csv(summary_csv, index=False, float_format="%.6f")

    if args.make_plots:
        maybe_make_plots(summary, args.out_dir)

    overview = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "skip_years": sorted(skip_years),
        "years_evaluated": expected_years,
        "variables": variables,
        "n_rows": int(len(combined)),
        "n_init_dates": int(combined["init"].nunique()),
        "summary_csv": os.path.abspath(summary_csv),
        "per_init_lead_csv": os.path.abspath(combined_csv),
    }
    with open(os.path.join(args.out_dir, "evaluation_overview.json"), "w") as f:
        json.dump(overview, f, indent=2)

    print_headline(summary)
    print(f"\n✅ Wrote per-init/per-lead metrics: {combined_csv}")
    print(f"✅ Wrote grouped summary metrics: {summary_csv}")
    if args.make_plots:
        print(f"✅ Wrote plots under: {os.path.join(args.out_dir, 'plots')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
