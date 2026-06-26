#!/usr/bin/env python3
"""
Evaluate generated flow_finalv1_global forecast Zarr stores against observations.

Each yearly Zarr store is expected to contain:
  model_pr/model_t2m: (init, ensemble, lead, lat, lon)
  geos_pr/geos_t2m:   (init, geos_member, lead, lat, lon)
  obs_pr/obs_t2m:     (init, lead, lat, lon)

The evaluator writes:
  - one per-year per-init/per-lead metrics CSV, used for resume
  - one per-year direct-reduction metric-state CSV, used for resume
  - one combined per-init/per-lead metrics CSV
  - one grouped direct-reduction summary CSV with all, lead, month, year,
    and season aggregations
  - one grouped scalar-mean summary CSV for diagnostics/backward comparison
  - optional diagnostic skill plots and spatial skill maps
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


GROUP_SPECS = [
    ("all", []),
    ("year", ["year"]),
    ("year_lead", ["year", "lead", "lead_label"]),
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

GROUP_COLUMNS = [
    "year",
    "lead",
    "lead_label",
    "init_month",
    "init_month_label",
    "init_season",
    "valid_month",
    "valid_month_label",
    "valid_season",
]

ORDERED_SUMMARY_COLS = [
    "group_type",
    "variable",
    "year",
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

DIRECT_STATE_SUM_COLUMNS = [
    "n_samples",
    "n_init_dates",
    "model_weight_sum",
    "geos_weight_sum",
    "model_crps_weighted_sum",
    "geos_crps_weighted_sum",
    "model_sse_weighted_sum",
    "geos_sse_weighted_sum",
    "model_ae_weighted_sum",
    "geos_ae_weighted_sum",
    "model_bias_weighted_sum",
    "geos_bias_weighted_sum",
    "model_spread_weighted_sum",
    "geos_spread_weighted_sum",
    "model_member_sum",
    "geos_member_sum",
]


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


def weighted_sums_for_ensemble(ensemble, obs, weights):
    """
    Return weighted sums for direct group reduction.

    This mirrors the training/test-mode reduction more closely than averaging
    already-reduced per-init/per-lead scalar RMSE values:
      RMSE = sqrt(sum(w * squared_error) / sum(w))
    CRPS/MAE/bias/spread are also reduced from their weighted gridpoint sums.
    """
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    if ensemble.ndim != 3:
        raise ValueError(f"Expected ensemble [E,H,W], got shape {ensemble.shape}")
    if obs.ndim != 2:
        raise ValueError(f"Expected obs [H,W], got shape {obs.shape}")

    finite = np.isfinite(obs) & np.all(np.isfinite(ensemble), axis=0)
    weighted_mask = np.where(finite, weights, 0.0)
    weight_sum = float(np.sum(weighted_mask))
    if weight_sum <= 0:
        return {
            "weight_sum": 0.0,
            "crps_weighted_sum": 0.0,
            "sse_weighted_sum": 0.0,
            "ae_weighted_sum": 0.0,
            "bias_weighted_sum": 0.0,
            "spread_weighted_sum": 0.0,
            "members": int(ensemble.shape[0]),
        }

    ens64 = ensemble.astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    mean = np.mean(ens64, axis=0)
    err = mean - obs64
    spread_map = np.std(ens64, axis=0)

    mae_term = np.mean(np.abs(ens64 - obs64[None, :, :]), axis=0)
    ens_sorted = np.sort(ens64, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    crps_map = mae_term - spread_term

    return {
        "weight_sum": weight_sum,
        "crps_weighted_sum": float(np.sum(np.where(finite, crps_map, 0.0) * weighted_mask)),
        "sse_weighted_sum": float(np.sum(np.where(finite, err * err, 0.0) * weighted_mask)),
        "ae_weighted_sum": float(np.sum(np.where(finite, np.abs(err), 0.0) * weighted_mask)),
        "bias_weighted_sum": float(np.sum(np.where(finite, err, 0.0) * weighted_mask)),
        "spread_weighted_sum": float(np.sum(np.where(finite, spread_map, 0.0) * weighted_mask)),
        "members": int(e),
    }


def grid_diagnostics_for_ensemble(ensemble, obs):
    """Per-grid CRPS, squared error, bias, and spread for one ensemble forecast."""
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    if ensemble.ndim != 3:
        raise ValueError(f"Expected ensemble [E,H,W], got shape {ensemble.shape}")
    if obs.ndim != 2:
        raise ValueError(f"Expected obs [H,W], got shape {obs.shape}")

    finite = np.isfinite(obs) & np.all(np.isfinite(ensemble), axis=0)
    ens64 = ensemble.astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    mean = np.mean(ens64, axis=0)
    err = mean - obs64
    spread = np.std(ens64, axis=0)

    mae_term = np.mean(np.abs(ens64 - obs64[None, :, :]), axis=0)
    ens_sorted = np.sort(ens64, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    crps = mae_term - spread_term

    return {
        "finite": finite,
        "crps": np.where(finite, crps, 0.0),
        "sse": np.where(finite, err * err, 0.0),
        "bias": np.where(finite, err, 0.0),
        "spread": np.where(finite, spread, 0.0),
    }


def _spatial_state_template(height, width):
    return {
        "count": np.zeros((height, width), dtype=np.float64),
        "model_crps_sum": np.zeros((height, width), dtype=np.float64),
        "geos_crps_sum": np.zeros((height, width), dtype=np.float64),
        "model_sse_sum": np.zeros((height, width), dtype=np.float64),
        "geos_sse_sum": np.zeros((height, width), dtype=np.float64),
        "model_bias_sum": np.zeros((height, width), dtype=np.float64),
        "geos_bias_sum": np.zeros((height, width), dtype=np.float64),
        "model_spread_sum": np.zeros((height, width), dtype=np.float64),
        "geos_spread_sum": np.zeros((height, width), dtype=np.float64),
    }


def _update_spatial_group(accumulators, group, variable, model_maps, geos_maps):
    finite = model_maps["finite"] & geos_maps["finite"]
    height, width = finite.shape
    key = (group, variable)
    if key not in accumulators:
        accumulators[key] = _spatial_state_template(height, width)
    state = accumulators[key]
    finite_f = finite.astype(np.float64)
    state["count"] += finite_f
    for metric in ("crps", "sse", "bias", "spread"):
        state[f"model_{metric}_sum"] += np.where(finite, model_maps[metric], 0.0)
        state[f"geos_{metric}_sum"] += np.where(finite, geos_maps[metric], 0.0)


def _update_spatial_accumulators(accumulators, row, variable, model, geos, obs):
    model_maps = grid_diagnostics_for_ensemble(model, obs)
    geos_maps = grid_diagnostics_for_ensemble(geos, obs)
    groups = [
        "all",
        f"year_{int(row['year'])}",
        f"lead_{row['lead_label']}",
        f"init_season_{row['init_season']}",
        f"init_season_lead_{row['init_season']}_{row['lead_label']}",
        f"valid_season_{row['valid_season']}",
        f"valid_season_lead_{row['valid_season']}_{row['lead_label']}",
    ]
    for group in groups:
        _update_spatial_group(accumulators, group, variable, model_maps, geos_maps)


def _spatial_accumulators_to_dataset(accumulators, lats, lons):
    if not accumulators:
        raise ValueError("No spatial diagnostics were accumulated.")
    groups = sorted({key[0] for key in accumulators})
    variables = [v for v in ("pr", "t2m") if any(key[1] == v for key in accumulators)]
    shape = (len(groups), len(variables), len(lats), len(lons))

    data = {
        "sample_count": np.full(shape, np.nan, dtype=np.float32),
        "model_crps": np.full(shape, np.nan, dtype=np.float32),
        "geos_crps": np.full(shape, np.nan, dtype=np.float32),
        "crps_skill_pct": np.full(shape, np.nan, dtype=np.float32),
        "model_rmse": np.full(shape, np.nan, dtype=np.float32),
        "geos_rmse": np.full(shape, np.nan, dtype=np.float32),
        "rmse_skill_pct": np.full(shape, np.nan, dtype=np.float32),
        "model_bias": np.full(shape, np.nan, dtype=np.float32),
        "geos_bias": np.full(shape, np.nan, dtype=np.float32),
        "bias_delta": np.full(shape, np.nan, dtype=np.float32),
        "model_spread": np.full(shape, np.nan, dtype=np.float32),
        "geos_spread": np.full(shape, np.nan, dtype=np.float32),
        "spread_delta": np.full(shape, np.nan, dtype=np.float32),
    }

    for g_idx, group in enumerate(groups):
        for v_idx, variable in enumerate(variables):
            state = accumulators.get((group, variable))
            if state is None:
                continue
            count = state["count"]
            valid = count > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                model_crps = np.where(valid, state["model_crps_sum"] / count, np.nan)
                geos_crps = np.where(valid, state["geos_crps_sum"] / count, np.nan)
                model_rmse = np.where(valid, np.sqrt(state["model_sse_sum"] / count), np.nan)
                geos_rmse = np.where(valid, np.sqrt(state["geos_sse_sum"] / count), np.nan)
                model_bias = np.where(valid, state["model_bias_sum"] / count, np.nan)
                geos_bias = np.where(valid, state["geos_bias_sum"] / count, np.nan)
                model_spread = np.where(valid, state["model_spread_sum"] / count, np.nan)
                geos_spread = np.where(valid, state["geos_spread_sum"] / count, np.nan)
                crps_skill = 100.0 * (1.0 - model_crps / geos_crps)
                rmse_skill = 100.0 * (1.0 - model_rmse / geos_rmse)
            crps_skill = np.where(np.isfinite(crps_skill) & (geos_crps > 1e-12), crps_skill, np.nan)
            rmse_skill = np.where(np.isfinite(rmse_skill) & (geos_rmse > 1e-12), rmse_skill, np.nan)

            index = (g_idx, v_idx)
            data["sample_count"][index] = count.astype(np.float32)
            data["model_crps"][index] = model_crps.astype(np.float32)
            data["geos_crps"][index] = geos_crps.astype(np.float32)
            data["crps_skill_pct"][index] = crps_skill.astype(np.float32)
            data["model_rmse"][index] = model_rmse.astype(np.float32)
            data["geos_rmse"][index] = geos_rmse.astype(np.float32)
            data["rmse_skill_pct"][index] = rmse_skill.astype(np.float32)
            data["model_bias"][index] = model_bias.astype(np.float32)
            data["geos_bias"][index] = geos_bias.astype(np.float32)
            data["bias_delta"][index] = (model_bias - geos_bias).astype(np.float32)
            data["model_spread"][index] = model_spread.astype(np.float32)
            data["geos_spread"][index] = geos_spread.astype(np.float32)
            data["spread_delta"][index] = (model_spread - geos_spread).astype(np.float32)

    coords = {
        "group": np.asarray(groups, dtype=object),
        "variable": np.asarray(variables, dtype=object),
        "lat": np.asarray(lats, dtype=np.float32),
        "lon": np.asarray(lons, dtype=np.float32),
    }
    ds = xr.Dataset(
        {name: (("group", "variable", "lat", "lon"), values) for name, values in data.items()},
        coords=coords,
        attrs={
            "description": "Spatial model-vs-GEOS skill diagnostics from generated forecast Zarr stores",
            "skill_definition": "100 * (1 - model_metric / geos_metric); positive means model improves over GEOS",
        },
    )
    ds["crps_skill_pct"].attrs["units"] = "%"
    ds["rmse_skill_pct"].attrs["units"] = "%"
    ds["sample_count"].attrs["description"] = "Number of init/lead samples contributing at each grid cell"
    return ds


def build_spatial_metric_maps(forecast_dir, years, variables, out_dir, overwrite=False):
    map_path = os.path.join(out_dir, "spatial_metric_maps.nc")
    if os.path.exists(map_path) and not overwrite:
        existing = xr.open_dataset(map_path)
        existing_groups = {str(g) for g in existing["group"].values}
        required_groups = {"all", "lead_week1", "lead_week2", "lead_week3", "lead_week4"}
        required_groups.update(
            f"init_season_lead_{season}_week{lead}"
            for season in ("DJF", "MAM", "JJA", "SON")
            for lead in range(1, 5)
        )
        if required_groups.issubset(existing_groups):
            print(f"✅ Using existing spatial metric maps: {map_path}")
            return existing
        existing.close()
        print(f"♻️ Existing spatial metric maps are missing new diagnostic groups; rebuilding {map_path}")

    print("🗺️ Building spatial metric maps from forecast Zarr stores...")
    accumulators = {}
    saved_lats = None
    saved_lons = None
    for year in years:
        zarr_path = os.path.join(forecast_dir, f"{year}.zarr")
        ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
        try:
            lats = ds["lat"].values
            lons = ds["lon"].values
            if saved_lats is None:
                saved_lats = lats
                saved_lons = lons
            init_values = pd.to_datetime(ds["init"].values).normalize()
            lead_values = ds["lead"].values
            for init_idx, init_time in enumerate(init_values):
                init_month = int(init_time.month)
                for lead_idx, lead_value in enumerate(lead_values):
                    if "valid_time" in ds:
                        valid_time = pd.Timestamp(ds["valid_time"].isel(init=init_idx, lead=lead_idx).values)
                    else:
                        valid_time = init_time + pd.to_timedelta(int(lead_value) * 7, unit="D")
                    row = {
                        "year": year,
                        "lead": int(lead_value),
                        "lead_label": f"week{int(lead_value)}",
                        "init_season": season_name(init_month),
                        "valid_season": season_name(int(valid_time.month)),
                    }
                    for variable in variables:
                        names = VARIABLES[variable]
                        obs = ds[names["obs"]].isel(init=init_idx, lead=lead_idx).values
                        model = ds[names["model"]].isel(init=init_idx, lead=lead_idx).values
                        geos = ds[names["geos"]].isel(init=init_idx, lead=lead_idx).values
                        _update_spatial_accumulators(accumulators, row, variable, model, geos, obs)
        finally:
            ds.close()

    map_ds = _spatial_accumulators_to_dataset(accumulators, saved_lats, saved_lons)
    os.makedirs(out_dir, exist_ok=True)
    map_ds.to_netcdf(map_path)
    print(f"✅ Wrote spatial metric maps: {map_path}")
    return map_ds


def _default_group_value(column):
    if column in {"lead", "init_month", "valid_month", "year"}:
        return np.nan
    return "all"


def _direct_state_template(group_type, variable, row, group_cols):
    state = {
        "group_type": group_type,
        "variable": variable,
        "n_samples": 0,
        "init_dates": set(),
        "model_weight_sum": 0.0,
        "geos_weight_sum": 0.0,
        "model_crps_weighted_sum": 0.0,
        "geos_crps_weighted_sum": 0.0,
        "model_sse_weighted_sum": 0.0,
        "geos_sse_weighted_sum": 0.0,
        "model_ae_weighted_sum": 0.0,
        "geos_ae_weighted_sum": 0.0,
        "model_bias_weighted_sum": 0.0,
        "geos_bias_weighted_sum": 0.0,
        "model_spread_weighted_sum": 0.0,
        "geos_spread_weighted_sum": 0.0,
        "model_member_sum": 0.0,
        "geos_member_sum": 0.0,
    }
    for column in GROUP_COLUMNS:
        state[column] = row[column] if column in group_cols else _default_group_value(column)
    return state


def update_direct_accumulators(accumulators, row, variable, model, geos, obs, weights):
    model_sums = weighted_sums_for_ensemble(model, obs, weights)
    geos_sums = weighted_sums_for_ensemble(geos, obs, weights)
    for group_type, group_cols in GROUP_SPECS:
        key = (group_type, variable, tuple(row[col] for col in group_cols))
        state = accumulators.get(key)
        if state is None:
            state = _direct_state_template(group_type, variable, row, group_cols)
            accumulators[key] = state

        state["n_samples"] += 1
        state["init_dates"].add(pd.Timestamp(row["init"]).strftime("%Y-%m-%d"))
        state["model_weight_sum"] += model_sums["weight_sum"]
        state["geos_weight_sum"] += geos_sums["weight_sum"]
        state["model_crps_weighted_sum"] += model_sums["crps_weighted_sum"]
        state["geos_crps_weighted_sum"] += geos_sums["crps_weighted_sum"]
        state["model_sse_weighted_sum"] += model_sums["sse_weighted_sum"]
        state["geos_sse_weighted_sum"] += geos_sums["sse_weighted_sum"]
        state["model_ae_weighted_sum"] += model_sums["ae_weighted_sum"]
        state["geos_ae_weighted_sum"] += geos_sums["ae_weighted_sum"]
        state["model_bias_weighted_sum"] += model_sums["bias_weighted_sum"]
        state["geos_bias_weighted_sum"] += geos_sums["bias_weighted_sum"]
        state["model_spread_weighted_sum"] += model_sums["spread_weighted_sum"]
        state["geos_spread_weighted_sum"] += geos_sums["spread_weighted_sum"]
        state["model_member_sum"] += model_sums["members"]
        state["geos_member_sum"] += geos_sums["members"]


def direct_accumulators_to_state_df(accumulators):
    rows = []
    for state in accumulators.values():
        row = {k: v for k, v in state.items() if k != "init_dates"}
        row["n_init_dates"] = len(state["init_dates"])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["group_type", "variable", *GROUP_COLUMNS, *DIRECT_STATE_SUM_COLUMNS])
    out = pd.DataFrame(rows)
    for column in ["group_type", "variable", *GROUP_COLUMNS, *DIRECT_STATE_SUM_COLUMNS]:
        if column not in out.columns:
            out[column] = np.nan
    return out[["group_type", "variable", *GROUP_COLUMNS, *DIRECT_STATE_SUM_COLUMNS]]


def direct_state_to_summary(state_df):
    if state_df.empty:
        return pd.DataFrame(columns=ORDERED_SUMMARY_COLS)
    group_cols = ["group_type", "variable", *GROUP_COLUMNS]
    numeric_cols = [col for col in DIRECT_STATE_SUM_COLUMNS if col in state_df.columns]
    grouped = state_df.groupby(group_cols, dropna=False)[numeric_cols].sum().reset_index()
    rows = []
    for _, row in grouped.iterrows():
        model_w = float(row["model_weight_sum"])
        geos_w = float(row["geos_weight_sum"])
        n_samples = max(float(row["n_samples"]), 1.0)
        out = {col: row[col] for col in group_cols}
        out["n_samples"] = int(row["n_samples"])
        out["n_init_dates"] = int(row["n_init_dates"])
        out["model_members"] = float(row["model_member_sum"]) / n_samples
        out["geos_members"] = float(row["geos_member_sum"]) / n_samples

        for prefix, weight in (("model", model_w), ("geos", geos_w)):
            if weight <= 0:
                out[f"{prefix}_crps"] = np.nan
                out[f"{prefix}_rmse"] = np.nan
                out[f"{prefix}_mae"] = np.nan
                out[f"{prefix}_bias"] = np.nan
                out[f"{prefix}_spread"] = np.nan
                continue
            out[f"{prefix}_crps"] = float(row[f"{prefix}_crps_weighted_sum"]) / weight
            out[f"{prefix}_rmse"] = np.sqrt(float(row[f"{prefix}_sse_weighted_sum"]) / weight)
            out[f"{prefix}_mae"] = float(row[f"{prefix}_ae_weighted_sum"]) / weight
            out[f"{prefix}_bias"] = float(row[f"{prefix}_bias_weighted_sum"]) / weight
            out[f"{prefix}_spread"] = float(row[f"{prefix}_spread_weighted_sum"]) / weight
        rows.append(out)

    summary = add_skill_columns(pd.DataFrame(rows))
    for col in ORDERED_SUMMARY_COLS:
        if col not in summary.columns:
            summary[col] = np.nan
    return summary[ORDERED_SUMMARY_COLS]


def evaluate_year(year, zarr_path, out_csv, direct_state_csv, variables, overwrite=False):
    if os.path.exists(out_csv) and os.path.exists(direct_state_csv) and not overwrite:
        print(f"✅ {year}: using existing metrics {out_csv} and {direct_state_csv}")
        return (
            pd.read_csv(out_csv, parse_dates=["init", "valid_time"]),
            pd.read_csv(direct_state_csv),
        )

    print(f"🔎 {year}: evaluating {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
    try:
        lats = ds["lat"].values
        weights = area_weights_from_lats(lats)
        init_values = pd.to_datetime(ds["init"].values).normalize()
        lead_values = ds["lead"].values
        rows = []
        direct_accumulators = {}

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
                    update_direct_accumulators(
                        direct_accumulators,
                        row,
                        variable,
                        model,
                        geos,
                        obs,
                        weights,
                    )

        out = add_skill_columns(pd.DataFrame(rows))
        direct_state = direct_accumulators_to_state_df(direct_accumulators)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out.to_csv(out_csv, index=False, float_format="%.6f")
        direct_state.to_csv(direct_state_csv, index=False, float_format="%.10f")
        print(f"✅ {year}: wrote {len(out)} rows to {out_csv}")
        print(f"✅ {year}: wrote direct metric state to {direct_state_csv}")
        return out, direct_state
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
    for col in GROUP_COLUMNS:
        if col not in out.columns:
            out[col] = _default_group_value(col)
    return add_skill_columns(out)


def build_summary(df):
    summary = pd.concat([aggregate_group(df, name, cols) for name, cols in GROUP_SPECS], ignore_index=True)
    for col in ORDERED_SUMMARY_COLS:
        if col not in summary.columns:
            summary[col] = np.nan
    return summary[ORDERED_SUMMARY_COLS]


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
    plot_group("year", "year", "year", "skill_by_year.png")
    plot_group("init_month", "init_month", "init_month_label", "skill_by_init_month.png")
    plot_group("init_season", "init_season", "init_season", "skill_by_init_season.png")
    plot_group("valid_season", "valid_season", "valid_season", "skill_by_valid_season.png")

    def plot_init_season_lead(variable):
        sub = summary[
            summary["group_type"].eq("init_season_lead")
            & summary["variable"].eq(variable)
        ].copy()
        if sub.empty:
            return
        seasons = [s for s in ("DJF", "MAM", "JJA", "SON") if s in set(sub["init_season"].astype(str))]
        leads = [f"week{i}" for i in range(1, 5)]
        fig, axes = plt.subplots(len(seasons), 2, figsize=(11, 3.2 * len(seasons)), squeeze=False)
        for row_idx, season in enumerate(seasons):
            sdf = sub[sub["init_season"].astype(str).eq(season)].copy()
            sdf["lead_label"] = sdf["lead_label"].astype(str)
            sdf = sdf.set_index("lead_label").reindex(leads).reset_index()
            x = np.arange(len(leads))
            axes[row_idx, 0].bar(x, sdf["crps_improvement_pct"])
            axes[row_idx, 0].axhline(0, color="k", linewidth=0.8)
            axes[row_idx, 0].set_title(f"{variable.upper()} {season} init CRPS skill by lead")
            axes[row_idx, 0].set_ylabel("Improvement (%)")
            axes[row_idx, 0].set_xticks(x)
            axes[row_idx, 0].set_xticklabels(leads, rotation=30, ha="right")

            axes[row_idx, 1].bar(x, sdf["rmse_improvement_pct"])
            axes[row_idx, 1].axhline(0, color="k", linewidth=0.8)
            axes[row_idx, 1].set_title(f"{variable.upper()} {season} init RMSE skill by lead")
            axes[row_idx, 1].set_ylabel("Improvement (%)")
            axes[row_idx, 1].set_xticks(x)
            axes[row_idx, 1].set_xticklabels(leads, rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{variable}_skill_by_init_season_lead.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    for variable in [v for v in ("pr", "t2m") if v in set(summary["variable"])]:
        plot_init_season_lead(variable)


def _finite_percentile_limits(field, lower=2, upper=98, symmetric=False):
    values = np.asarray(field)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    lo, hi = np.nanpercentile(values, [lower, upper])
    if symmetric:
        vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
        return -vmax, vmax
    if abs(float(hi) - float(lo)) < 1e-12:
        pad = max(abs(float(hi)) * 0.05, 1e-6)
        return float(lo) - pad, float(hi) + pad
    return float(lo), float(hi)


def _plot_spatial_panel(ax, lons, lats, field, title, cmap, vmin=None, vmax=None):
    mesh = ax.pcolormesh(lons, lats, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    return mesh


def _get_spatial_field(spatial_ds, variable, group, metric):
    return spatial_ds[metric].sel(variable=variable, group=group).values


def _available_groups(spatial_ds, prefix=None):
    groups = [str(g) for g in spatial_ds["group"].values]
    if prefix is None:
        return groups
    return [g for g in groups if g.startswith(prefix)]


def plot_spatial_diagnostics(spatial_ds, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots", "spatial")
    os.makedirs(plot_dir, exist_ok=True)
    lats = spatial_ds["lat"].values
    lons = spatial_ds["lon"].values
    variables = [str(v) for v in spatial_ds["variable"].values]
    groups = _available_groups(spatial_ds)

    def plot_all_panels(variable):
        if "all" not in groups:
            return
        panel_specs = [
            ("model_crps", "Model CRPS", "viridis", False, None),
            ("geos_crps", "GEOS CRPS", "viridis", False, None),
            ("crps_skill_pct", "CRPS skill (%)", "RdBu", True, (-50, 50)),
            ("model_rmse", "Model RMSE", "viridis", False, None),
            ("geos_rmse", "GEOS RMSE", "viridis", False, None),
            ("rmse_skill_pct", "RMSE skill (%)", "RdBu", True, (-50, 50)),
            ("model_bias", "Model bias", "RdBu_r", True, None),
            ("geos_bias", "GEOS bias", "RdBu_r", True, None),
            ("bias_delta", "Model-GEOS bias", "RdBu_r", True, None),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(18, 10), constrained_layout=True)
        for ax, (metric, title, cmap, symmetric, fixed_limits) in zip(axes.flat, panel_specs):
            field = _get_spatial_field(spatial_ds, variable, "all", metric)
            if fixed_limits is None:
                vmin, vmax = _finite_percentile_limits(field, symmetric=symmetric)
            else:
                vmin, vmax = fixed_limits
            mesh = _plot_spatial_panel(
                ax,
                lons,
                lats,
                field,
                f"{variable.upper()} {title}",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            fig.colorbar(mesh, ax=ax, shrink=0.75)
        fig.suptitle(f"{variable.upper()} spatial diagnostics | all samples", fontsize=14)
        fig.savefig(os.path.join(plot_dir, f"{variable}_spatial_all_metrics.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_skill_grid(variable, group_list, labels, filename, title):
        present = [(g, label) for g, label in zip(group_list, labels) if g in groups]
        if not present:
            return
        fig, axes = plt.subplots(2, len(present), figsize=(4.5 * len(present), 7), squeeze=False, constrained_layout=True)
        for col, (group, label) in enumerate(present):
            for row, metric in enumerate(("crps_skill_pct", "rmse_skill_pct")):
                field = _get_spatial_field(spatial_ds, variable, group, metric)
                mesh = _plot_spatial_panel(
                    axes[row, col],
                    lons,
                    lats,
                    field,
                    f"{label} {'CRPS' if row == 0 else 'RMSE'} skill",
                    cmap="RdBu",
                    vmin=-50,
                    vmax=50,
                )
                fig.colorbar(mesh, ax=axes[row, col], shrink=0.75)
        fig.suptitle(f"{variable.upper()} spatial skill vs GEOS | {title}", fontsize=14)
        fig.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_model_geos_metric_grid(variable, group_list, labels, metric, filename, title):
        present = [(g, label) for g, label in zip(group_list, labels) if g in groups]
        if not present:
            return
        model_metric = f"model_{metric}"
        geos_metric = f"geos_{metric}"
        combined = []
        for group, _ in present:
            combined.append(_get_spatial_field(spatial_ds, variable, group, model_metric))
            combined.append(_get_spatial_field(spatial_ds, variable, group, geos_metric))
        vmin, vmax = _finite_percentile_limits(np.stack(combined), lower=1, upper=99)
        fig, axes = plt.subplots(2, len(present), figsize=(4.5 * len(present), 7), squeeze=False, constrained_layout=True)
        for col, (group, label) in enumerate(present):
            for row, source in enumerate(("model", "geos")):
                field = _get_spatial_field(spatial_ds, variable, group, f"{source}_{metric}")
                mesh = _plot_spatial_panel(
                    axes[row, col],
                    lons,
                    lats,
                    field,
                    f"{label} {'ML' if source == 'model' else 'GEOS'} {metric.upper()}",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(mesh, ax=axes[row, col], shrink=0.75)
        fig.suptitle(f"{variable.upper()} ML vs GEOS {metric.upper()} | {title}", fontsize=14)
        fig.savefig(os.path.join(plot_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)

    for variable in variables:
        plot_all_panels(variable)
        for metric in ("crps", "rmse"):
            plot_model_geos_metric_grid(
                variable,
                [f"lead_week{i}" for i in range(1, 5)],
                [f"week{i}" for i in range(1, 5)],
                metric,
                f"{variable}_spatial_{metric}_by_lead_ml_geos.png",
                "by lead",
            )
        plot_skill_grid(
            variable,
            [f"lead_week{i}" for i in range(1, 5)],
            [f"week{i}" for i in range(1, 5)],
            f"{variable}_spatial_skill_by_lead.png",
            "by lead",
        )
        year_groups = [g for g in _available_groups(spatial_ds, "year_")]
        year_groups = sorted(year_groups, key=lambda x: int(x.split("_", 1)[1]))
        plot_skill_grid(
            variable,
            year_groups,
            [g.split("_", 1)[1] for g in year_groups],
            f"{variable}_spatial_skill_by_year.png",
            "by year",
        )
        for prefix, label in (("init_season", "init season"), ("valid_season", "valid season")):
            season_groups = [f"{prefix}_{s}" for s in ("DJF", "MAM", "JJA", "SON")]
            plot_skill_grid(
                variable,
                season_groups,
                ["DJF", "MAM", "JJA", "SON"],
                f"{variable}_spatial_skill_by_{prefix}.png",
                label,
            )
        for season in ("DJF", "MAM", "JJA", "SON"):
            season_lead_groups = [f"init_season_lead_{season}_week{i}" for i in range(1, 5)]
            plot_skill_grid(
                variable,
                season_lead_groups,
                [f"{season} week{i}" for i in range(1, 5)],
                f"{variable}_spatial_skill_init_{season}_by_lead.png",
                f"{season} init by lead",
            )
            for metric in ("crps", "rmse"):
                plot_model_geos_metric_grid(
                    variable,
                    season_lead_groups,
                    [f"{season} week{i}" for i in range(1, 5)],
                    metric,
                    f"{variable}_spatial_{metric}_init_{season}_by_lead_ml_geos.png",
                    f"{season} init by lead",
                )

    print(f"✅ Wrote spatial diagnostic plots under: {plot_dir}")


def plot_orientation_diagnostics(forecast_dir, years, variables, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots", "orientation")
    os.makedirs(plot_dir, exist_ok=True)
    first_year = years[0]
    zarr_path = os.path.join(forecast_dir, f"{first_year}.zarr")
    ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
    try:
        lats = np.asarray(ds["lat"].values)
        lons = np.asarray(ds["lon"].values)
        init_time = pd.Timestamp(ds["init"].values[0])
        lead_value = int(ds["lead"].values[0])
        report = {
            "checked_zarr": os.path.abspath(zarr_path),
            "sample_init": init_time.strftime("%Y-%m-%d"),
            "sample_lead": lead_value,
            "lat_first": float(lats[0]),
            "lat_last": float(lats[-1]),
            "lat_min": float(np.nanmin(lats)),
            "lat_max": float(np.nanmax(lats)),
            "lat_strictly_increasing": bool(np.all(np.diff(lats) > 0)),
            "lon_first": float(lons[0]),
            "lon_last": float(lons[-1]),
            "lon_min": float(np.nanmin(lons)),
            "lon_max": float(np.nanmax(lons)),
            "lon_strictly_increasing": bool(np.all(np.diff(lons) > 0)),
            "plotting_method": "pcolormesh(lon, lat, field) with y limits set to lat min/max; if lat is increasing, south is bottom and north is top.",
            "expected_global_orientation": "lat[0] should be -90/South Pole and lat[-1] should be +90/North Pole.",
        }
        for variable in variables:
            names = VARIABLES[variable]
            obs = ds[names["obs"]].isel(init=0, lead=0).values
            model = ds[names["model"]].isel(init=0, lead=0).mean("ensemble").values
            geos = ds[names["geos"]].isel(init=0, lead=0).mean("geos_member").values
            report[f"{variable}_shape_obs"] = list(obs.shape)
            report[f"{variable}_shape_model_mean"] = list(model.shape)
            report[f"{variable}_shape_geos_mean"] = list(geos.shape)
            report[f"{variable}_south_band_obs_mean"] = float(np.nanmean(obs[: max(1, min(10, obs.shape[0])), :]))
            report[f"{variable}_north_band_obs_mean"] = float(np.nanmean(obs[-max(1, min(10, obs.shape[0])) :, :]))

            fields = [
                (obs, "Obs", "viridis", False),
                (geos, "GEOS mean", "viridis", False),
                (model, "ML mean", "viridis", False),
                (geos - obs, "GEOS - Obs", "RdBu_r", True),
                (model - obs, "ML - Obs", "RdBu_r", True),
                (model - geos, "ML - GEOS", "RdBu_r", True),
            ]
            fig, axes = plt.subplots(2, 3, figsize=(17, 7), constrained_layout=True)
            for ax, (field, title, cmap, symmetric) in zip(axes.flat, fields):
                if symmetric:
                    vmin, vmax = _finite_percentile_limits(field, lower=2, upper=98, symmetric=True)
                else:
                    vmin, vmax = _finite_percentile_limits(field, lower=2, upper=98)
                mesh = _plot_spatial_panel(
                    ax,
                    lons,
                    lats,
                    field,
                    f"{variable.upper()} {title}",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(mesh, ax=ax, shrink=0.75)
            fig.suptitle(
                f"Orientation check | {variable.upper()} | init {init_time:%Y-%m-%d} lead week{lead_value}\n"
                f"lat[0]={lats[0]:.1f}, lat[-1]={lats[-1]:.1f}; plotted with latitude coordinate on y-axis",
                fontsize=13,
            )
            fig.savefig(os.path.join(plot_dir, f"{variable}_orientation_sample_fields.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 8))
            ax.plot(np.nanmean(obs, axis=1), lats, label="Obs")
            ax.plot(np.nanmean(geos, axis=1), lats, label="GEOS mean")
            ax.plot(np.nanmean(model, axis=1), lats, label="ML mean")
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_xlabel(f"{variable.upper()} zonal mean")
            ax.set_ylabel("latitude")
            ax.set_title(f"{variable.upper()} zonal mean orientation check")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{variable}_orientation_zonal_mean.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        report_path = os.path.join(plot_dir, "orientation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Wrote orientation diagnostics under: {plot_dir}")
    finally:
        ds.close()


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
    year_direct_states = []
    all_complete = True
    for year in expected_years:
        year_csv = os.path.join(yearly_dir, f"{year}_per_init_lead_metrics.csv")
        year_state_csv = os.path.join(yearly_dir, f"{year}_direct_metric_state.csv")
        if deadline_reached(deadline) and not (os.path.exists(year_csv) and os.path.exists(year_state_csv)):
            print(f"⏸️ Soft runtime reached before starting {year}.")
            all_complete = False
            break
        zarr_path = os.path.join(args.forecast_dir, f"{year}.zarr")
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"Missing forecast Zarr store for {year}: {zarr_path}")
        year_df, year_direct_state = evaluate_year(
            year=year,
            zarr_path=zarr_path,
            out_csv=year_csv,
            direct_state_csv=year_state_csv,
            variables=variables,
            overwrite=args.overwrite,
        )
        year_frames.append(year_df)
        year_direct_states.append(year_direct_state)

    missing_metrics = [
        year
        for year in expected_years
        if (
            not os.path.exists(os.path.join(yearly_dir, f"{year}_per_init_lead_metrics.csv"))
            or not os.path.exists(os.path.join(yearly_dir, f"{year}_direct_metric_state.csv"))
        )
    ]
    if missing_metrics:
        print(f"⏸️ Missing per-year metric/detail-state files: {missing_metrics}")
        all_complete = False

    if not all_complete:
        print("⏸️ Evaluation incomplete; rerun this script to resume.")
        return 0

    if not year_frames:
        year_frames = [
            pd.read_csv(os.path.join(yearly_dir, f"{year}_per_init_lead_metrics.csv"), parse_dates=["init", "valid_time"])
            for year in expected_years
        ]
    if not year_direct_states:
        year_direct_states = [
            pd.read_csv(os.path.join(yearly_dir, f"{year}_direct_metric_state.csv"))
            for year in expected_years
        ]

    combined = pd.concat(year_frames, ignore_index=True)
    combined = add_skill_columns(combined)
    combined_csv = os.path.join(args.out_dir, "per_init_lead_metrics.csv")
    combined.to_csv(combined_csv, index=False, float_format="%.6f")

    direct_state = pd.concat(year_direct_states, ignore_index=True)
    direct_state_csv = os.path.join(args.out_dir, "direct_metric_state.csv")
    direct_state.to_csv(direct_state_csv, index=False, float_format="%.10f")

    summary = direct_state_to_summary(direct_state)
    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    summary.to_csv(summary_csv, index=False, float_format="%.6f")

    scalar_summary = build_summary(combined)
    scalar_summary_csv = os.path.join(args.out_dir, "summary_metrics_scalar_mean.csv")
    scalar_summary.to_csv(scalar_summary_csv, index=False, float_format="%.6f")

    spatial_map_path = os.path.join(args.out_dir, "spatial_metric_maps.nc")
    orientation_report_path = os.path.join(args.out_dir, "plots", "orientation", "orientation_report.json")
    if args.make_plots:
        maybe_make_plots(summary, args.out_dir)
        spatial_ds = build_spatial_metric_maps(
            forecast_dir=args.forecast_dir,
            years=expected_years,
            variables=variables,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
        )
        try:
            plot_spatial_diagnostics(spatial_ds, args.out_dir)
        finally:
            spatial_ds.close()
        plot_orientation_diagnostics(args.forecast_dir, expected_years, variables, args.out_dir)

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
        "summary_reduction": "direct_gridpoint_weighted",
        "summary_csv": os.path.abspath(summary_csv),
        "scalar_mean_summary_csv": os.path.abspath(scalar_summary_csv),
        "direct_metric_state_csv": os.path.abspath(direct_state_csv),
        "spatial_metric_maps": os.path.abspath(spatial_map_path) if args.make_plots else None,
        "orientation_report": os.path.abspath(orientation_report_path) if args.make_plots else None,
        "per_init_lead_csv": os.path.abspath(combined_csv),
    }
    with open(os.path.join(args.out_dir, "evaluation_overview.json"), "w") as f:
        json.dump(overview, f, indent=2)

    print_headline(summary)
    print(f"\n✅ Wrote per-init/per-lead metrics: {combined_csv}")
    print(f"✅ Wrote direct grouped summary metrics: {summary_csv}")
    print(f"✅ Wrote scalar-mean diagnostic summary: {scalar_summary_csv}")
    print(f"✅ Wrote direct metric state: {direct_state_csv}")
    if args.make_plots:
        print(f"✅ Wrote plots under: {os.path.join(args.out_dir, 'plots')}")
        print(f"✅ Wrote spatial metric maps: {spatial_map_path}")
        print(f"✅ Wrote orientation report: {orientation_report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
