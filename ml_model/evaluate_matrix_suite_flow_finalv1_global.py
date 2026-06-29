#!/usr/bin/env python3
"""
Matrix-style evaluation suite for generated flow_finalv1_global forecast Zarrs.

This is intentionally separate from evaluate_forecast_zarr_flow_finalv1_global.py.
It focuses on 2021-2023 verification matrices with lead weeks kept explicit:

  - valid-season x lead matrices
  - valid-month x lead matrices
  - all-data and observed-extreme conditional subsets
  - scalar CSV summaries and spatial map NetCDF/PNG products

Metrics:
  RMSE, MAE, bias, Pearson correlation, CRPS, BSS, calibrated BSS, spread.

BSS event definition:
  observation >= local observed threshold map. By default thresholds are the
  gridpoint 95th percentile over the evaluation period. For precipitation, the
  threshold is also constrained to be at least --pr_min_threshold mm/day.

Calibrated BSS:
  forecast event probabilities are adjusted with a simple logit climatological
  frequency correction at each grid point:
    logit(p_cal) = logit(p_raw) + logit(obs_event_freq) - logit(fcst_event_freq)
  This is a light-weight diagnostic, not a full cross-validated calibration.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = {
    "pr": {
        "model": "model_pr",
        "geos": "geos_pr",
        "obs": "obs_pr",
        "units": "mm/day",
        "extreme_quantile_arg": "extreme_quantile_pr",
        "min_threshold_arg": "pr_min_threshold",
    },
    "t2m": {
        "model": "model_t2m",
        "geos": "geos_t2m",
        "obs": "obs_t2m",
        "units": "K",
        "extreme_quantile_arg": "extreme_quantile_t2m",
        "min_threshold_arg": None,
    },
}

SEASONS = ["DJF", "MAM", "JJA", "SON"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
LEADS = [1, 2, 3, 4]
SUBSETS = ["all_data", "extreme_events"]
GROUP_TYPES = ["valid_season_lead", "valid_month_lead"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 2021-2023 matrix skill from generated global Zarrs.")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Directory containing YEAR.zarr forecast stores.",
    )
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--skip_years", type=str, default="")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_e90_s50",
    )
    parser.add_argument("--variables", type=str, default="pr,t2m")
    parser.add_argument("--extreme_quantile_pr", type=float, default=0.95)
    parser.add_argument("--extreme_quantile_t2m", type=float, default=0.95)
    parser.add_argument("--pr_min_threshold", type=float, default=5.0)
    parser.add_argument("--epsilon_probability", type=float, default=1e-4)
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_runtime_minutes",
        type=float,
        default=None,
        help="Stop before starting a new major pass after this many minutes.",
    )
    return parser.parse_args()


def parse_years(text):
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def parse_variables(text):
    variables = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    bad = [v for v in variables if v not in VARIABLES]
    if bad:
        raise ValueError(f"Unknown variables {bad}; expected subset of {sorted(VARIABLES)}")
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


def deadline_reached(deadline):
    return deadline is not None and time.monotonic() >= deadline


def safe_divide(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return out


def area_weights_from_lats(lats):
    weights = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64)))
    weights = np.clip(weights, 0.0, None)
    return weights[:, None]


def logit(p, eps):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_probability(prob, obs_event_freq, forecast_event_freq, eps=1e-4):
    offset = logit(obs_event_freq, eps) - logit(forecast_event_freq, eps)
    return np.clip(inv_logit(logit(prob, eps) + offset), 0.0, 1.0)


def crps_map(ensemble, obs):
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    ens64 = ensemble.astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    mae_term = np.nanmean(np.abs(ens64 - obs64[None, :, :]), axis=0)
    ens_sorted = np.sort(ens64, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    return mae_term - spread_term


def ensemble_diagnostics(ensemble, obs, threshold, obs_event_freq, fcst_event_freq, eps):
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mean = np.nanmean(ensemble, axis=0).astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    err = mean - obs64
    finite = np.isfinite(obs64) & np.isfinite(mean) & np.isfinite(threshold)
    prob = np.nanmean(ensemble >= threshold[None, :, :], axis=0).astype(np.float64, copy=False)
    event = obs64 >= threshold
    prob_cal = calibrate_probability(prob, obs_event_freq, fcst_event_freq, eps=eps)
    return {
        "finite": finite,
        "mean": mean,
        "obs": obs64,
        "err": err,
        "abs_err": np.abs(err),
        "sse": err * err,
        "spread": np.nanstd(ensemble.astype(np.float64, copy=False), axis=0),
        "crps": crps_map(ensemble, obs64),
        "prob": prob,
        "prob_cal": prob_cal,
        "event": event.astype(np.float64),
        "brier": (prob - event) ** 2,
        "brier_cal": (prob_cal - event) ** 2,
        "brier_ref": (obs_event_freq - event) ** 2,
    }


def scalar_state():
    return {
        "n_forecasts": 0,
        "weight_sum": 0.0,
        "model_sse": 0.0,
        "geos_sse": 0.0,
        "model_ae": 0.0,
        "geos_ae": 0.0,
        "model_bias": 0.0,
        "geos_bias": 0.0,
        "model_crps": 0.0,
        "geos_crps": 0.0,
        "model_spread": 0.0,
        "geos_spread": 0.0,
        "model_bs": 0.0,
        "geos_bs": 0.0,
        "model_bs_cal": 0.0,
        "geos_bs_cal": 0.0,
        "ref_bs": 0.0,
        "model_x": 0.0,
        "model_x2": 0.0,
        "model_xy": 0.0,
        "geos_x": 0.0,
        "geos_x2": 0.0,
        "geos_xy": 0.0,
        "obs_y": 0.0,
        "obs_y2": 0.0,
    }


def spatial_state(shape):
    return {
        "count": np.zeros(shape, dtype=np.float32),
        "model_sse": np.zeros(shape, dtype=np.float32),
        "geos_sse": np.zeros(shape, dtype=np.float32),
        "model_ae": np.zeros(shape, dtype=np.float32),
        "geos_ae": np.zeros(shape, dtype=np.float32),
        "model_bias": np.zeros(shape, dtype=np.float32),
        "geos_bias": np.zeros(shape, dtype=np.float32),
        "model_crps": np.zeros(shape, dtype=np.float32),
        "geos_crps": np.zeros(shape, dtype=np.float32),
        "model_spread": np.zeros(shape, dtype=np.float32),
        "geos_spread": np.zeros(shape, dtype=np.float32),
        "model_bs": np.zeros(shape, dtype=np.float32),
        "geos_bs": np.zeros(shape, dtype=np.float32),
        "model_bs_cal": np.zeros(shape, dtype=np.float32),
        "geos_bs_cal": np.zeros(shape, dtype=np.float32),
        "ref_bs": np.zeros(shape, dtype=np.float32),
        "model_x": np.zeros(shape, dtype=np.float32),
        "model_x2": np.zeros(shape, dtype=np.float32),
        "model_xy": np.zeros(shape, dtype=np.float32),
        "geos_x": np.zeros(shape, dtype=np.float32),
        "geos_x2": np.zeros(shape, dtype=np.float32),
        "geos_xy": np.zeros(shape, dtype=np.float32),
        "obs_y": np.zeros(shape, dtype=np.float32),
        "obs_y2": np.zeros(shape, dtype=np.float32),
    }


def update_scalar(state, model, geos, weights, mask):
    finite = mask & model["finite"] & geos["finite"]
    if not finite.any():
        return
    wm = np.where(finite, weights, 0.0)
    wsum = float(np.sum(wm))
    if wsum <= 0:
        return
    state["n_forecasts"] += 1
    state["weight_sum"] += wsum
    for prefix, diag in (("model", model), ("geos", geos)):
        state[f"{prefix}_sse"] += float(np.sum(np.where(finite, diag["sse"], 0.0) * wm))
        state[f"{prefix}_ae"] += float(np.sum(np.where(finite, diag["abs_err"], 0.0) * wm))
        state[f"{prefix}_bias"] += float(np.sum(np.where(finite, diag["err"], 0.0) * wm))
        state[f"{prefix}_crps"] += float(np.sum(np.where(finite, diag["crps"], 0.0) * wm))
        state[f"{prefix}_spread"] += float(np.sum(np.where(finite, diag["spread"], 0.0) * wm))
        state[f"{prefix}_bs"] += float(np.sum(np.where(finite, diag["brier"], 0.0) * wm))
        state[f"{prefix}_bs_cal"] += float(np.sum(np.where(finite, diag["brier_cal"], 0.0) * wm))
        x = diag["mean"]
        y = diag["obs"]
        state[f"{prefix}_x"] += float(np.sum(np.where(finite, x, 0.0) * wm))
        state[f"{prefix}_x2"] += float(np.sum(np.where(finite, x * x, 0.0) * wm))
        state[f"{prefix}_xy"] += float(np.sum(np.where(finite, x * y, 0.0) * wm))
    y = model["obs"]
    state["obs_y"] += float(np.sum(np.where(finite, y, 0.0) * wm))
    state["obs_y2"] += float(np.sum(np.where(finite, y * y, 0.0) * wm))
    state["ref_bs"] += float(np.sum(np.where(finite, model["brier_ref"], 0.0) * wm))


def update_spatial(state, model, geos, mask):
    finite = mask & model["finite"] & geos["finite"]
    if not finite.any():
        return
    f = finite.astype(np.float32)
    state["count"] += f
    for prefix, diag in (("model", model), ("geos", geos)):
        state[f"{prefix}_sse"] += np.where(finite, diag["sse"], 0.0).astype(np.float32)
        state[f"{prefix}_ae"] += np.where(finite, diag["abs_err"], 0.0).astype(np.float32)
        state[f"{prefix}_bias"] += np.where(finite, diag["err"], 0.0).astype(np.float32)
        state[f"{prefix}_crps"] += np.where(finite, diag["crps"], 0.0).astype(np.float32)
        state[f"{prefix}_spread"] += np.where(finite, diag["spread"], 0.0).astype(np.float32)
        state[f"{prefix}_bs"] += np.where(finite, diag["brier"], 0.0).astype(np.float32)
        state[f"{prefix}_bs_cal"] += np.where(finite, diag["brier_cal"], 0.0).astype(np.float32)
        x = diag["mean"]
        y = diag["obs"]
        state[f"{prefix}_x"] += np.where(finite, x, 0.0).astype(np.float32)
        state[f"{prefix}_x2"] += np.where(finite, x * x, 0.0).astype(np.float32)
        state[f"{prefix}_xy"] += np.where(finite, x * y, 0.0).astype(np.float32)
    y = model["obs"]
    state["obs_y"] += np.where(finite, y, 0.0).astype(np.float32)
    state["obs_y2"] += np.where(finite, y * y, 0.0).astype(np.float32)
    state["ref_bs"] += np.where(finite, model["brier_ref"], 0.0).astype(np.float32)


def correlation_from_sums(x_sum, y_sum, x2_sum, y2_sum, xy_sum, weight_sum):
    if np.isscalar(weight_sum):
        if weight_sum <= 1e-12:
            return np.nan
    else:
        weight_sum = np.asarray(weight_sum, dtype=np.float64)
    mx = safe_divide(x_sum, weight_sum)
    my = safe_divide(y_sum, weight_sum)
    cov = safe_divide(xy_sum, weight_sum) - mx * my
    vx = safe_divide(x2_sum, weight_sum) - mx * mx
    vy = safe_divide(y2_sum, weight_sum) - my * my
    corr = safe_divide(cov, np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0)))
    return np.where(np.isfinite(corr), corr, np.nan)


def scalar_rows_from_states(states):
    rows = []
    for key, state in sorted(states.items()):
        subset, variable, group_type, group_value, lead = key
        w = state["weight_sum"]
        if w <= 0:
            continue
        row = {
            "subset": subset,
            "variable": variable,
            "group_type": group_type,
            "group_value": group_value,
            "lead": int(lead),
            "lead_label": f"week{int(lead)}",
            "n_forecasts": state["n_forecasts"],
            "weight_sum": w,
        }
        for prefix in ("model", "geos"):
            row[f"{prefix}_rmse"] = float(np.sqrt(state[f"{prefix}_sse"] / w))
            row[f"{prefix}_mae"] = float(state[f"{prefix}_ae"] / w)
            row[f"{prefix}_bias"] = float(state[f"{prefix}_bias"] / w)
            row[f"{prefix}_crps"] = float(state[f"{prefix}_crps"] / w)
            row[f"{prefix}_spread"] = float(state[f"{prefix}_spread"] / w)
            row[f"{prefix}_brier"] = float(state[f"{prefix}_bs"] / w)
            row[f"{prefix}_brier_calibrated"] = float(state[f"{prefix}_bs_cal"] / w)
            row[f"{prefix}_bss"] = 1.0 - state[f"{prefix}_bs"] / state["ref_bs"] if state["ref_bs"] > 1e-12 else np.nan
            row[f"{prefix}_calibrated_bss"] = (
                1.0 - state[f"{prefix}_bs_cal"] / state["ref_bs"] if state["ref_bs"] > 1e-12 else np.nan
            )
            row[f"{prefix}_corr"] = float(
                correlation_from_sums(
                    state[f"{prefix}_x"],
                    state["obs_y"],
                    state[f"{prefix}_x2"],
                    state["obs_y2"],
                    state[f"{prefix}_xy"],
                    w,
                )
            )
        for metric in ("rmse", "mae", "crps"):
            geos_value = row[f"geos_{metric}"]
            row[f"{metric}_skill_pct"] = (
                100.0 * (1.0 - row[f"model_{metric}"] / geos_value)
                if np.isfinite(geos_value) and geos_value > 1e-12
                else np.nan
            )
        row["corr_diff"] = row["model_corr"] - row["geos_corr"]
        row["bss_diff"] = row["model_bss"] - row["geos_bss"]
        row["calibrated_bss_diff"] = row["model_calibrated_bss"] - row["geos_calibrated_bss"]
        row["abs_bias_skill_pct"] = (
            100.0 * (1.0 - abs(row["model_bias"]) / abs(row["geos_bias"]))
            if np.isfinite(row["geos_bias"]) and abs(row["geos_bias"]) > 1e-12
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def spatial_dataset_from_states(states, lats, lons):
    group_values = SEASONS + MONTHS
    shape = (len(SUBSETS), len(VARIABLES), len(GROUP_TYPES), len(group_values), len(LEADS), len(lats), len(lons))
    coords = {
        "subset": SUBSETS,
        "variable": list(VARIABLES),
        "group_type": GROUP_TYPES,
        "group_value": group_values,
        "lead": LEADS,
        "lat": np.asarray(lats, dtype=np.float32),
        "lon": np.asarray(lons, dtype=np.float32),
    }
    out_vars = [
        "sample_count",
        "model_rmse",
        "geos_rmse",
        "rmse_skill_pct",
        "model_mae",
        "geos_mae",
        "mae_skill_pct",
        "model_bias",
        "geos_bias",
        "model_corr",
        "geos_corr",
        "corr_diff",
        "model_crps",
        "geos_crps",
        "crps_skill_pct",
        "model_bss",
        "geos_bss",
        "bss_diff",
        "model_calibrated_bss",
        "geos_calibrated_bss",
        "calibrated_bss_diff",
        "model_spread",
        "geos_spread",
    ]
    data = {name: np.full(shape, np.nan, dtype=np.float32) for name in out_vars}
    idx = {
        "subset": {v: i for i, v in enumerate(SUBSETS)},
        "variable": {v: i for i, v in enumerate(VARIABLES)},
        "group_type": {v: i for i, v in enumerate(GROUP_TYPES)},
        "group_value": {v: i for i, v in enumerate(group_values)},
        "lead": {v: i for i, v in enumerate(LEADS)},
    }
    for key, state in states.items():
        subset, variable, group_type, group_value, lead = key
        if group_type not in idx["group_type"] or group_value not in idx["group_value"]:
            continue
        count = state["count"].astype(np.float64)
        valid = count > 0
        pos = (
            idx["subset"][subset],
            idx["variable"][variable],
            idx["group_type"][group_type],
            idx["group_value"][group_value],
            idx["lead"][int(lead)],
        )
        data["sample_count"][pos] = count.astype(np.float32)
        for prefix in ("model", "geos"):
            with np.errstate(divide="ignore", invalid="ignore"):
                rmse = np.where(valid, np.sqrt(state[f"{prefix}_sse"] / count), np.nan)
                mae = np.where(valid, state[f"{prefix}_ae"] / count, np.nan)
                bias = np.where(valid, state[f"{prefix}_bias"] / count, np.nan)
                crps = np.where(valid, state[f"{prefix}_crps"] / count, np.nan)
                spread = np.where(valid, state[f"{prefix}_spread"] / count, np.nan)
                bss = np.where(
                    valid & (state["ref_bs"] > 1e-12),
                    1.0 - state[f"{prefix}_bs"] / state["ref_bs"],
                    np.nan,
                )
                cbss = np.where(
                    valid & (state["ref_bs"] > 1e-12),
                    1.0 - state[f"{prefix}_bs_cal"] / state["ref_bs"],
                    np.nan,
                )
            corr = correlation_from_sums(
                state[f"{prefix}_x"],
                state["obs_y"],
                state[f"{prefix}_x2"],
                state["obs_y2"],
                state[f"{prefix}_xy"],
                count,
            )
            data[f"{prefix}_rmse"][pos] = rmse.astype(np.float32)
            data[f"{prefix}_mae"][pos] = mae.astype(np.float32)
            data[f"{prefix}_bias"][pos] = bias.astype(np.float32)
            data[f"{prefix}_crps"][pos] = crps.astype(np.float32)
            data[f"{prefix}_spread"][pos] = spread.astype(np.float32)
            data[f"{prefix}_bss"][pos] = bss.astype(np.float32)
            data[f"{prefix}_calibrated_bss"][pos] = cbss.astype(np.float32)
            data[f"{prefix}_corr"][pos] = corr.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            data["rmse_skill_pct"][pos] = (100.0 * (1.0 - data["model_rmse"][pos] / data["geos_rmse"][pos])).astype(np.float32)
            data["mae_skill_pct"][pos] = (100.0 * (1.0 - data["model_mae"][pos] / data["geos_mae"][pos])).astype(np.float32)
            data["crps_skill_pct"][pos] = (100.0 * (1.0 - data["model_crps"][pos] / data["geos_crps"][pos])).astype(np.float32)
        data["corr_diff"][pos] = (data["model_corr"][pos] - data["geos_corr"][pos]).astype(np.float32)
        data["bss_diff"][pos] = (data["model_bss"][pos] - data["geos_bss"][pos]).astype(np.float32)
        data["calibrated_bss_diff"][pos] = (
            data["model_calibrated_bss"][pos] - data["geos_calibrated_bss"][pos]
        ).astype(np.float32)
    ds = xr.Dataset(
        {name: (("subset", "variable", "group_type", "group_value", "lead", "lat", "lon"), values) for name, values in data.items()},
        coords=coords,
        attrs={
            "description": "Season/month x lead spatial matrix diagnostics for flow_finalv1_global.",
            "skill_definition": "100 * (1 - ML metric / GEOS metric); positive means ML improves over GEOS.",
            "bss_reference": "local observed event climatology from selected evaluation years.",
        },
    )
    return ds


def collect_obs_thresholds(forecast_dir, years, variables, args):
    thresholds = {}
    climatology = {}
    saved_lats = None
    saved_lons = None
    print("📏 Building observed extreme thresholds...")
    for variable in variables:
        spec = VARIABLES[variable]
        obs_chunks = []
        for year in years:
            zarr_path = os.path.join(forecast_dir, f"{year}.zarr")
            ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
            try:
                if saved_lats is None:
                    saved_lats = ds["lat"].values
                    saved_lons = ds["lon"].values
                obs = ds[spec["obs"]].values.astype(np.float32, copy=False)
                obs_chunks.append(obs.reshape(-1, obs.shape[-2], obs.shape[-1]))
            finally:
                ds.close()
        stack = np.concatenate(obs_chunks, axis=0)
        q = float(getattr(args, spec["extreme_quantile_arg"]))
        threshold = np.nanquantile(stack, q, axis=0).astype(np.float32)
        min_arg = spec["min_threshold_arg"]
        if min_arg is not None:
            threshold = np.maximum(threshold, float(getattr(args, min_arg))).astype(np.float32)
        events = stack >= threshold[None, :, :]
        event_freq = np.nanmean(events, axis=0).astype(np.float32)
        event_freq = np.where(np.isfinite(event_freq), event_freq, np.nan).astype(np.float32)
        thresholds[variable] = threshold
        climatology[variable] = event_freq
        print(
            f"   {variable}: q={q:.3f}, threshold mean={float(np.nanmean(threshold)):.3f} "
            f"{spec['units']}, event freq mean={float(np.nanmean(event_freq)):.4f}"
        )
    return thresholds, climatology, saved_lats, saved_lons


def collect_forecast_event_climatology(forecast_dir, years, variables, thresholds, deadline=None):
    out = {}
    print("🎯 Building forecast event-probability climatology for calibrated BSS...")
    for variable in variables:
        threshold = thresholds[variable]
        shape = threshold.shape
        sums = {"model": np.zeros(shape, dtype=np.float64), "geos": np.zeros(shape, dtype=np.float64)}
        count = np.zeros(shape, dtype=np.float64)
        spec = VARIABLES[variable]
        for year in years:
            if deadline_reached(deadline):
                raise TimeoutError("Soft runtime limit reached while building forecast event climatology.")
            ds = xr.open_zarr(os.path.join(forecast_dir, f"{year}.zarr"), consolidated=False, chunks=None)
            try:
                n_init = ds.sizes["init"]
                n_lead = ds.sizes["lead"]
                for init_idx in range(n_init):
                    for lead_idx in range(n_lead):
                        obs = ds[spec["obs"]].isel(init=init_idx, lead=lead_idx).values
                        finite = np.isfinite(obs) & np.isfinite(threshold)
                        if not finite.any():
                            continue
                        model = ds[spec["model"]].isel(init=init_idx, lead=lead_idx).values
                        geos = ds[spec["geos"]].isel(init=init_idx, lead=lead_idx).values
                        sums["model"] += np.where(finite, np.nanmean(model >= threshold[None, :, :], axis=0), 0.0)
                        sums["geos"] += np.where(finite, np.nanmean(geos >= threshold[None, :, :], axis=0), 0.0)
                        count += finite.astype(np.float64)
            finally:
                ds.close()
        out[variable] = {
            "model": np.where(count > 0, sums["model"] / count, np.nan).astype(np.float32),
            "geos": np.where(count > 0, sums["geos"] / count, np.nan).astype(np.float32),
        }
        print(
            f"   {variable}: mean model event p={float(np.nanmean(out[variable]['model'])):.4f}, "
            f"GEOS event p={float(np.nanmean(out[variable]['geos'])):.4f}"
        )
    return out


def group_keys_for_sample(subset, variable, valid_time, lead_value):
    valid_month = f"{int(valid_time.month):02d}"
    valid_season = season_name(int(valid_time.month))
    return [
        (subset, variable, "valid_season_lead", valid_season, int(lead_value)),
        (subset, variable, "valid_month_lead", valid_month, int(lead_value)),
    ]


def evaluate(forecast_dir, years, variables, thresholds, obs_clim, fcst_clim, lats, lons, args, deadline=None):
    weights = area_weights_from_lats(lats)
    scalar_states = {}
    spatial_states = {}
    shape = (len(lats), len(lons))
    eps = float(args.epsilon_probability)
    print("🧮 Evaluating matrix metrics...")
    for year in years:
        if deadline_reached(deadline):
            raise TimeoutError("Soft runtime limit reached while evaluating.")
        zarr_path = os.path.join(forecast_dir, f"{year}.zarr")
        print(f"   Year {year}: {zarr_path}")
        ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
        try:
            init_values = pd.to_datetime(ds["init"].values).normalize()
            lead_values = ds["lead"].values
            for init_idx, init_time in enumerate(init_values):
                if "valid_time" in ds:
                    valid_values = pd.to_datetime(ds["valid_time"].isel(init=init_idx).values).normalize()
                else:
                    valid_values = pd.to_datetime(
                        [init_time + pd.to_timedelta(int(lead) * 7, unit="D") for lead in lead_values]
                    ).normalize()
                for lead_idx, lead_value in enumerate(lead_values):
                    valid_time = pd.Timestamp(valid_values[lead_idx])
                    for variable in variables:
                        spec = VARIABLES[variable]
                        threshold = thresholds[variable]
                        obs_event_freq = obs_clim[variable]
                        obs = ds[spec["obs"]].isel(init=init_idx, lead=lead_idx).values
                        model_ens = ds[spec["model"]].isel(init=init_idx, lead=lead_idx).values
                        geos_ens = ds[spec["geos"]].isel(init=init_idx, lead=lead_idx).values
                        model = ensemble_diagnostics(
                            model_ens,
                            obs,
                            threshold,
                            obs_event_freq,
                            fcst_clim[variable]["model"],
                            eps,
                        )
                        geos = ensemble_diagnostics(
                            geos_ens,
                            obs,
                            threshold,
                            obs_event_freq,
                            fcst_clim[variable]["geos"],
                            eps,
                        )
                        event_mask = obs >= threshold
                        masks = {
                            "all_data": np.ones(shape, dtype=bool),
                            "extreme_events": event_mask,
                        }
                        for subset, subset_mask in masks.items():
                            for key in group_keys_for_sample(subset, variable, valid_time, lead_value):
                                if key not in scalar_states:
                                    scalar_states[key] = scalar_state()
                                if key not in spatial_states:
                                    spatial_states[key] = spatial_state(shape)
                                update_scalar(scalar_states[key], model, geos, weights, subset_mask)
                                update_spatial(spatial_states[key], model, geos, subset_mask)
        finally:
            ds.close()
    summary = scalar_rows_from_states(scalar_states)
    spatial = spatial_dataset_from_states(spatial_states, lats, lons)
    return summary, spatial


def save_threshold_dataset(thresholds, obs_clim, fcst_clim, lats, lons, out_dir):
    data_vars = {}
    for variable in thresholds:
        data_vars[f"{variable}_threshold"] = (("lat", "lon"), thresholds[variable].astype(np.float32))
        data_vars[f"{variable}_obs_event_frequency"] = (("lat", "lon"), obs_clim[variable].astype(np.float32))
        data_vars[f"{variable}_model_event_frequency"] = (("lat", "lon"), fcst_clim[variable]["model"].astype(np.float32))
        data_vars[f"{variable}_geos_event_frequency"] = (("lat", "lon"), fcst_clim[variable]["geos"].astype(np.float32))
    ds = xr.Dataset(
        data_vars,
        coords={"lat": np.asarray(lats, dtype=np.float32), "lon": np.asarray(lons, dtype=np.float32)},
        attrs={"description": "Event thresholds and event frequencies used by matrix evaluation suite."},
    )
    path = os.path.join(out_dir, "event_thresholds_and_frequencies.nc")
    ds.to_netcdf(path)
    print(f"✅ Wrote event thresholds/frequencies: {path}")
    return path


def plot_heatmap(data, row_labels, col_labels, title, out_path, cmap="viridis", center_zero=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(data, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.1), max(4, len(row_labels) * 0.45)))
    if center_zero:
        vmax = np.nanpercentile(np.abs(arr), 95) if np.isfinite(arr).any() else 1.0
        vmin = -vmax
    else:
        vmin, vmax = np.nanpercentile(arr[np.isfinite(arr)], [5, 95]) if np.isfinite(arr).any() else (0, 1)
    mesh = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel("lead")
    fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_scalar_matrix_plots(summary, out_dir):
    plot_dir = os.path.join(out_dir, "plots", "scalar_matrices")
    os.makedirs(plot_dir, exist_ok=True)
    families = {
        "rmse": ["model_rmse", "geos_rmse", "rmse_skill_pct"],
        "corr": ["model_corr", "geos_corr", "corr_diff"],
        "crps": ["model_crps", "geos_crps", "crps_skill_pct"],
        "bss": ["model_bss", "geos_bss", "bss_diff"],
        "calibrated_bss": ["model_calibrated_bss", "geos_calibrated_bss", "calibrated_bss_diff"],
        "mae": ["model_mae", "geos_mae", "mae_skill_pct"],
        "bias": ["model_bias", "geos_bias", "abs_bias_skill_pct"],
        "spread": ["model_spread", "geos_spread"],
    }
    for subset in SUBSETS:
        for variable in sorted(summary["variable"].unique()):
            for group_type, rows in (("valid_season_lead", SEASONS), ("valid_month_lead", MONTHS)):
                df = summary[(summary["subset"] == subset) & (summary["variable"] == variable) & (summary["group_type"] == group_type)]
                if df.empty:
                    continue
                for family, columns in families.items():
                    for column in columns:
                        matrix = np.full((len(rows), len(LEADS)), np.nan, dtype=np.float64)
                        for r_idx, group_value in enumerate(rows):
                            for c_idx, lead in enumerate(LEADS):
                                match = df[(df["group_value"] == group_value) & (df["lead"] == lead)]
                                if not match.empty and column in match:
                                    matrix[r_idx, c_idx] = float(match.iloc[0][column])
                        center_zero = column.endswith("_diff") or "skill" in column or "bias" in column
                        cmap = "RdBu" if center_zero else "viridis"
                        out_path = os.path.join(plot_dir, f"{subset}_{variable}_{group_type}_{column}.png")
                        plot_heatmap(
                            matrix,
                            rows,
                            [f"week{lead}" for lead in LEADS],
                            f"{subset} | {variable} | {group_type} | {column}",
                            out_path,
                            cmap=cmap,
                            center_zero=center_zero,
                        )
    print(f"✅ Wrote scalar matrix plots under: {plot_dir}")


def plot_spatial_grid(ds, subset, variable, group_type, metric, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = SEASONS if group_type == "valid_season_lead" else MONTHS
    lats = ds["lat"].values
    lons = ds["lon"].values
    fig, axes = plt.subplots(
        len(rows),
        len(LEADS),
        figsize=(4.4 * len(LEADS), max(2.0 * len(rows), 6.0)),
        squeeze=False,
        constrained_layout=True,
    )
    values = []
    for group_value in rows:
        for lead in LEADS:
            arr = ds[metric].sel(subset=subset, variable=variable, group_type=group_type, group_value=group_value, lead=lead).values
            values.append(arr)
    combined = np.concatenate([np.ravel(v[np.isfinite(v)]) for v in values if np.isfinite(v).any()]) if values else np.array([])
    center_zero = metric.endswith("_diff") or metric.endswith("_skill_pct") or metric.endswith("_bias")
    if combined.size:
        if center_zero:
            vmax = np.nanpercentile(np.abs(combined), 95)
            vmin = -vmax
        else:
            vmin, vmax = np.nanpercentile(combined, [5, 95])
    else:
        vmin, vmax = (-1, 1) if center_zero else (0, 1)
    cmap = "RdBu" if center_zero else "viridis"
    last_mesh = None
    for r_idx, group_value in enumerate(rows):
        for c_idx, lead in enumerate(LEADS):
            ax = axes[r_idx, c_idx]
            arr = ds[metric].sel(subset=subset, variable=variable, group_type=group_type, group_value=group_value, lead=lead).values
            last_mesh = ax.pcolormesh(lons, lats, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{group_value} week{lead}", fontsize=9)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
    fig.suptitle(f"{subset} | {variable} | {group_type} | {metric}", fontsize=14)
    if last_mesh is not None:
        fig.colorbar(last_mesh, ax=axes.ravel().tolist(), shrink=0.75)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_spatial_plots(spatial, out_dir):
    plot_dir = os.path.join(out_dir, "plots", "spatial_matrices")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = [
        "model_rmse",
        "geos_rmse",
        "rmse_skill_pct",
        "model_corr",
        "geos_corr",
        "corr_diff",
        "model_crps",
        "geos_crps",
        "crps_skill_pct",
        "model_bss",
        "geos_bss",
        "bss_diff",
        "model_calibrated_bss",
        "geos_calibrated_bss",
        "calibrated_bss_diff",
    ]
    for subset in SUBSETS:
        for variable in spatial["variable"].values:
            for group_type in GROUP_TYPES:
                for metric in metrics:
                    out_path = os.path.join(plot_dir, f"{subset}_{variable}_{group_type}_{metric}.png")
                    plot_spatial_grid(spatial, subset, str(variable), group_type, metric, out_path)
    print(f"✅ Wrote spatial matrix plots under: {plot_dir}")


def main():
    args = parse_args()
    years = [year for year in range(args.start_year, args.end_year + 1) if year not in parse_years(args.skip_years)]
    variables = parse_variables(args.variables)
    os.makedirs(args.out_dir, exist_ok=True)
    deadline = time.monotonic() + args.max_runtime_minutes * 60.0 if args.max_runtime_minutes else None

    summary_path = os.path.join(args.out_dir, "matrix_summary_metrics.csv")
    spatial_path = os.path.join(args.out_dir, "matrix_spatial_metrics.nc")
    metadata_path = os.path.join(args.out_dir, "matrix_eval_metadata.json")
    if os.path.exists(summary_path) and os.path.exists(spatial_path) and not args.overwrite:
        print(f"✅ Existing matrix evaluation found: {summary_path}")
        if args.make_plots:
            summary = pd.read_csv(summary_path)
            spatial = xr.open_dataset(spatial_path)
            make_scalar_matrix_plots(summary, args.out_dir)
            make_spatial_plots(spatial, args.out_dir)
            spatial.close()
        return

    thresholds, obs_clim, lats, lons = collect_obs_thresholds(args.forecast_dir, years, variables, args)
    if deadline_reached(deadline):
        raise TimeoutError("Soft runtime limit reached after threshold pass.")
    fcst_clim = collect_forecast_event_climatology(args.forecast_dir, years, variables, thresholds, deadline=deadline)
    save_threshold_dataset(thresholds, obs_clim, fcst_clim, lats, lons, args.out_dir)
    summary, spatial = evaluate(
        args.forecast_dir,
        years,
        variables,
        thresholds,
        obs_clim,
        fcst_clim,
        lats,
        lons,
        args,
        deadline=deadline,
    )

    summary.to_csv(summary_path, index=False, float_format="%.6f")
    spatial.to_netcdf(spatial_path)
    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "years": years,
        "variables": variables,
        "subsets": SUBSETS,
        "group_types": GROUP_TYPES,
        "extreme_definition": "obs >= local observed threshold map",
        "extreme_quantile_pr": args.extreme_quantile_pr,
        "extreme_quantile_t2m": args.extreme_quantile_t2m,
        "pr_min_threshold": args.pr_min_threshold,
        "calibrated_bss": "logit climatological frequency correction, diagnostic/not cross-validated",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Wrote matrix summary: {summary_path}")
    print(f"✅ Wrote matrix spatial maps: {spatial_path}")
    print(f"✅ Wrote metadata: {metadata_path}")
    if args.make_plots:
        make_scalar_matrix_plots(summary, args.out_dir)
        make_spatial_plots(spatial, args.out_dir)
    spatial.close()


if __name__ == "__main__":
    main()
