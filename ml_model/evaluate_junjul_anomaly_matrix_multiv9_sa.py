#!/usr/bin/env python3
"""
Evaluate June/July v9 SA anomalies for ML and GEOS against observed anomalies.

Inputs are anomaly Zarrs produced by build_junjul_climatology_anomalies_multiv9_sa.py.
Outputs are CSV matrices separated by init month, variable, system, and lead.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_ANOM_PATH = (
    "dataprocess/clim_anom_multiv9_sa_55e100e_0n40n_junjul/"
    "v9_junjul_anomalies_2021_2023.zarr"
)
DEFAULT_CLIM_PATH = (
    "dataprocess/clim_anom_multiv9_sa_55e100e_0n40n_junjul/"
    "v9_junjul_climatology_2005_2024.zarr"
)
DEFAULT_FORECAST_DIR = "dataprocess/gen_multiv9_sa_55e100e_0n40n_junjul_e10clim_e100eval_s50"
DEFAULT_OUTPUT_DIR = "ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/anomaly_matrix_junjul"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v9 SA June/July anomaly matrices.")
    parser.add_argument("--anomaly_path", type=str, default=DEFAULT_ANOM_PATH)
    parser.add_argument("--climatology_path", type=str, default=DEFAULT_CLIM_PATH)
    parser.add_argument("--forecast_dir", type=str, default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--months", type=str, default="6,7")
    parser.add_argument("--baseline_start_year", type=int, default=2005)
    parser.add_argument("--baseline_end_year", type=int, default=2024)
    parser.add_argument("--skip_years", type=str, default="2017")
    parser.add_argument("--extreme_quantiles", type=str, default="0.90,0.95")
    parser.add_argument("--decision_thresholds", type=str, default="0.1,0.25,0.5")
    return parser.parse_args()


def parse_float_list(text):
    values = [float(item.strip()) for item in str(text or "").split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def parse_int_list(text):
    values = [int(item.strip()) for item in str(text or "").split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_int_set(text):
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def area_weights(lat):
    weights = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    weights = weights / np.nanmean(weights)
    return weights[:, None]


def broadcast_weights(values, weights_2d):
    spatial = np.broadcast_to(weights_2d, values.shape[-2:])
    return np.broadcast_to(spatial, values.shape)


def weighted_mean(values, weights_2d):
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.nan
    weights = broadcast_weights(arr, weights_2d)
    return float(np.nansum(np.where(mask, arr * weights, 0.0)) / (np.nansum(np.where(mask, weights, 0.0)) + 1e-12))


def weighted_sum_bool(mask, weights_2d):
    arr = np.asarray(mask, dtype=bool)
    weights = broadcast_weights(arr, weights_2d)
    return float(np.sum(np.where(arr, weights, 0.0)))


def open_year(forecast_dir, year):
    path = os.path.join(forecast_dir, f"{year}.zarr")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_zarr(path, consolidated=False, chunks=None)


def init_month_mask(ds, month):
    return pd.to_datetime(ds["init"].values).month == int(month)


def build_obs_anomaly_thresholds(args, months, quantiles):
    if not os.path.exists(args.climatology_path):
        raise FileNotFoundError(f"Climatology Zarr not found: {args.climatology_path}")

    skip_years = parse_int_set(args.skip_years)
    clim = xr.open_zarr(args.climatology_path, consolidated=False, chunks=None)
    thresholds = {int(month): {"pr": {}, "t2m": {}} for month in months}
    metadata = {
        "baseline_years_requested": [args.baseline_start_year, args.baseline_end_year],
        "baseline_years_loaded": [],
        "baseline_years_missing": [],
        "skip_years": sorted(skip_years),
        "threshold_source": "observed anomalies from baseline forecast_dir minus observed climatology",
    }
    obs_stacks = {int(month): {"pr": [], "t2m": []} for month in months}

    try:
        for year in range(args.baseline_start_year, args.baseline_end_year + 1):
            if year in skip_years:
                continue
            path = os.path.join(args.forecast_dir, f"{year}.zarr")
            if not os.path.exists(path):
                print(f"⚠️ Missing threshold baseline year {year}: {path}. Skipping.")
                metadata["baseline_years_missing"].append(year)
                continue
            ds = open_year(args.forecast_dir, year)
            try:
                metadata["baseline_years_loaded"].append(year)
                for month in months:
                    mask = init_month_mask(ds, month)
                    if not np.any(mask):
                        continue
                    sub = ds.isel(init=np.where(mask)[0])
                    month_clim = clim.sel(month=int(month))
                    obs_stacks[int(month)]["pr"].append(
                        sub["obs_pr"].values.astype(np.float32, copy=False)
                        - month_clim["obs_pr_clim"].values[None]
                    )
                    obs_stacks[int(month)]["t2m"].append(
                        sub["obs_t2m"].values.astype(np.float32, copy=False)
                        - month_clim["obs_t2m_clim"].values[None]
                    )
            finally:
                ds.close()
    finally:
        clim.close()

    for month in months:
        for variable in ("pr", "t2m"):
            arrays = obs_stacks[int(month)][variable]
            if not arrays:
                raise RuntimeError(f"No baseline observed anomalies found for month={month}, variable={variable}")
            stack = np.concatenate(arrays, axis=0)
            for quantile in quantiles:
                thresholds[int(month)][variable][float(quantile)] = np.nanquantile(stack, quantile, axis=0)
    return thresholds, metadata


def weighted_std(values, weights_2d):
    mean = weighted_mean(values, weights_2d)
    if not np.isfinite(mean):
        return np.nan
    return float(np.sqrt(max(weighted_mean((np.asarray(values) - mean) ** 2, weights_2d), 0.0)))


def weighted_corr(x, y, weights_2d):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(mask):
        return np.nan
    weights = broadcast_weights(x_arr, weights_2d)
    weights = np.where(mask, weights, 0.0)
    w_sum = weights.sum()
    if w_sum <= 0:
        return np.nan
    x_mean = np.sum(np.where(mask, x_arr * weights, 0.0)) / w_sum
    y_mean = np.sum(np.where(mask, y_arr * weights, 0.0)) / w_sum
    x_anom = np.where(mask, x_arr - x_mean, 0.0)
    y_anom = np.where(mask, y_arr - y_mean, 0.0)
    cov = np.sum(weights * x_anom * y_anom) / w_sum
    x_var = np.sum(weights * x_anom ** 2) / w_sum
    y_var = np.sum(weights * y_anom ** 2) / w_sum
    if x_var <= 0 or y_var <= 0:
        return np.nan
    return float(cov / np.sqrt(x_var * y_var))


def crps_ensemble(ensemble, obs):
    # ensemble [init, member, lead, lat, lon], obs [init, lead, lat, lon]
    n_members = ensemble.shape[1]
    mae = np.nanmean(np.abs(ensemble - obs[:, None]), axis=1)
    sorted_ens = np.sort(ensemble, axis=1)
    coeff = (2 * np.arange(n_members, dtype=np.float64) - n_members + 1).reshape(1, n_members, 1, 1, 1)
    spread_term = np.nansum(sorted_ens.astype(np.float64) * coeff, axis=1) / (n_members ** 2)
    return mae - spread_term


def select_lead(arr, lead_idx):
    if lead_idx is None:
        return arr
    if arr.ndim == 5:
        return arr[:, :, lead_idx:lead_idx + 1]
    return arr[:, lead_idx:lead_idx + 1]


def overall_metrics(month, variable, system, ensemble, obs, weights, lead_idx=None):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    ens_mean = np.nanmean(ens, axis=1)
    spread = np.nanstd(ens, axis=1, ddof=1) if ens.shape[1] > 1 else np.zeros_like(ens_mean)
    err = ens_mean - target
    rmse = float(np.sqrt(max(weighted_mean(err ** 2, weights), 0.0)))
    spread_mean = weighted_mean(spread, weights)
    return {
        "month": int(month),
        "variable": variable,
        "system": system,
        "lead": "all" if lead_idx is None else int(lead_idx + 1),
        "n_init": int(target.shape[0]),
        "bias": weighted_mean(err, weights),
        "mae": weighted_mean(np.abs(err), weights),
        "rmse": rmse,
        "acc": weighted_corr(ens_mean, target, weights),
        "crps": weighted_mean(crps_ensemble(ens, target), weights),
        "ensemble_spread_mean": spread_mean,
        "spread_error_ratio": spread_mean / rmse if rmse > 0 else np.nan,
        "forecast_anom_std": weighted_std(ens_mean, weights),
        "obs_anom_std": weighted_std(target, weights),
    }


def event_probability(ensemble, threshold):
    return np.nanmean(ensemble > threshold[None, None], axis=1)


def event_mask(obs, threshold):
    return obs > threshold[None]


def extreme_metrics(month, variable, system, ensemble, obs, weights, quantile, threshold, lead_idx=None):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    if lead_idx is not None:
        threshold = threshold[lead_idx:lead_idx + 1]
    prob = event_probability(ens, threshold)
    event = event_mask(target, threshold)
    ens_mean = np.nanmean(ens, axis=1)
    spread = np.nanstd(ens, axis=1, ddof=1) if ens.shape[1] > 1 else np.zeros_like(ens_mean)
    event_err = np.where(event, ens_mean - target, np.nan)
    event_rmse = float(np.sqrt(max(weighted_mean(event_err ** 2, weights), 0.0)))
    obs_rate = weighted_mean(event.astype(np.float32), weights)
    forecast_rate = weighted_mean(prob, weights)
    return {
        "month": int(month),
        "variable": variable,
        "system": system,
        "lead": "all" if lead_idx is None else int(lead_idx + 1),
        "tail": "upper",
        "quantile": float(quantile),
        "threshold_area_mean": weighted_mean(threshold, weights),
        "obs_event_rate": obs_rate,
        "forecast_event_probability_mean": forecast_rate,
        "frequency_bias": forecast_rate / obs_rate if obs_rate > 0 else np.nan,
        "brier_score": weighted_mean((prob - event.astype(np.float32)) ** 2, weights),
        "event_rmse": event_rmse,
        "event_bias": weighted_mean(event_err, weights),
        "ensemble_spread_on_obs_events": weighted_mean(np.where(event, spread, np.nan), weights),
        "forecast_anomaly_on_obs_events": weighted_mean(np.where(event, ens_mean, np.nan), weights),
        "obs_anomaly_on_events": weighted_mean(np.where(event, target, np.nan), weights),
    }, prob, event


def decision_rows(base, prob, event, weights, decision_thresholds):
    rows = []
    for decision in decision_thresholds:
        yes = prob >= decision
        hits = weighted_sum_bool(yes & event, weights)
        false_alarms = weighted_sum_bool(yes & ~event, weights)
        misses = weighted_sum_bool(~yes & event, weights)
        row = {
            key: base[key]
            for key in ("month", "variable", "system", "lead", "tail", "quantile")
        }
        row.update({
            "decision_probability": float(decision),
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "pod": hits / (hits + misses) if (hits + misses) > 0 else np.nan,
            "far": false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan,
            "csi": hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else np.nan,
        })
        rows.append(row)
    return rows


def reliability_rows(base, prob, event, weights):
    rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            in_bin = (prob >= left) & (prob <= right)
        else:
            in_bin = (prob >= left) & (prob < right)
        weighted_count = weighted_sum_bool(in_bin, weights)
        if weighted_count <= 0:
            continue
        row = {
            key: base[key]
            for key in ("month", "variable", "system", "lead", "tail", "quantile")
        }
        row.update({
            "prob_bin_left": float(left),
            "prob_bin_right": float(right),
            "weighted_count": weighted_count,
            "mean_forecast_probability": weighted_mean(np.where(in_bin, prob, np.nan), weights),
            "observed_frequency": weighted_mean(np.where(in_bin, event.astype(np.float32), np.nan), weights),
        })
        rows.append(row)
    return rows


def data_for(ds, variable, system):
    if system == "ml":
        return ds[f"ml_{variable}_anom"].values.astype(np.float32, copy=False)
    if system == "geos":
        return ds[f"geos_{variable}_anom"].values.astype(np.float32, copy=False)
    raise ValueError(system)


def obs_for(ds, variable):
    return ds[f"obs_{variable}_anom"].values.astype(np.float32, copy=False)


def main():
    args = parse_args()
    months = parse_int_list(args.months)
    quantiles = parse_float_list(args.extreme_quantiles)
    decision_thresholds = parse_float_list(args.decision_thresholds)
    os.makedirs(args.output_dir, exist_ok=True)
    thresholds, threshold_meta = build_obs_anomaly_thresholds(args, months, quantiles)

    ds = xr.open_zarr(args.anomaly_path, consolidated=False, chunks=None)
    try:
        weights = area_weights(ds["lat"].values)
        overall_rows = []
        extreme_rows = []
        decision_matrix_rows = []
        reliability_matrix_rows = []

        init_month = ds["init_month"].values.astype(int)
        for month in months:
            indices = np.where(init_month == int(month))[0]
            if len(indices) == 0:
                print(f"⚠️ No anomaly init dates for month={month}.")
                continue
            sub = ds.isel(init=indices)
            for variable in ("pr", "t2m"):
                obs = obs_for(sub, variable)
                for system in ("ml", "geos"):
                    ensemble = data_for(sub, variable, system)
                    overall_rows.append(overall_metrics(month, variable, system, ensemble, obs, weights))
                    for lead_idx in range(obs.shape[1]):
                        overall_rows.append(overall_metrics(month, variable, system, ensemble, obs, weights, lead_idx=lead_idx))

                    for quantile in quantiles:
                        threshold = thresholds[int(month)][variable][float(quantile)]
                        row, prob, event = extreme_metrics(
                            month, variable, system, ensemble, obs, weights, quantile, threshold=threshold
                        )
                        extreme_rows.append(row)
                        decision_matrix_rows.extend(decision_rows(row, prob, event, weights, decision_thresholds))
                        reliability_matrix_rows.extend(reliability_rows(row, prob, event, weights))
                        for lead_idx in range(obs.shape[1]):
                            lead_row, lead_prob, lead_event = extreme_metrics(
                                month,
                                variable,
                                system,
                                ensemble,
                                obs,
                                weights,
                                quantile,
                                threshold=threshold,
                                lead_idx=lead_idx,
                            )
                            extreme_rows.append(lead_row)
                            decision_matrix_rows.extend(decision_rows(lead_row, lead_prob, lead_event, weights, decision_thresholds))
                            reliability_matrix_rows.extend(reliability_rows(lead_row, lead_prob, lead_event, weights))
    finally:
        ds.close()

    overall_path = os.path.join(args.output_dir, "anomaly_overall_matrix.csv")
    extreme_path = os.path.join(args.output_dir, "anomaly_extreme_matrix.csv")
    decision_path = os.path.join(args.output_dir, "anomaly_extreme_decision_matrix.csv")
    reliability_path = os.path.join(args.output_dir, "anomaly_extreme_reliability_matrix.csv")
    metadata_path = os.path.join(args.output_dir, "anomaly_matrix_metadata.json")

    pd.DataFrame(overall_rows).to_csv(overall_path, index=False)
    pd.DataFrame(extreme_rows).to_csv(extreme_path, index=False)
    pd.DataFrame(decision_matrix_rows).to_csv(decision_path, index=False)
    pd.DataFrame(reliability_matrix_rows).to_csv(reliability_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "anomaly_path": args.anomaly_path,
                "climatology_path": args.climatology_path,
                "forecast_dir": args.forecast_dir,
                "months": months,
                "extreme_quantiles": quantiles,
                "decision_thresholds": decision_thresholds,
                "threshold_metadata": threshold_meta,
                "notes": "Extreme thresholds are month/lead/grid-specific observed anomaly quantiles from the baseline years.",
            },
            f,
            indent=2,
        )

    print("\nJune/July anomaly matrix evaluation complete")
    print(f"  Overall matrix : {overall_path}")
    print(f"  Extreme matrix : {extreme_path}")
    print(f"  Decision matrix: {decision_path}")
    print(f"  Reliability   : {reliability_path}")
    print(f"  Metadata       : {metadata_path}")


if __name__ == "__main__":
    main()
