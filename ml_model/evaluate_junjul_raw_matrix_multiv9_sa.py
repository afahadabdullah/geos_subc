#!/usr/bin/env python3
"""
Evaluate raw June/July v9 SA forecasts for ML and GEOS against observations.

This intentionally avoids the 2005-2024 ML climatology/anomaly path. Raw
forecast skill uses the generated evaluation-year Zarrs. Physical extreme
thresholds use independent raw observations from 2001-2020 by default, while
a separate diagnostic uses evaluation-period OBS/ML/GEOS self-thresholds.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from dataset_flow_multi import S2SHybridDataset


DEFAULT_FORECAST_DIR = "dataprocess/gen_multiv9_sa_55e100e_0n40n_junjul_testmode_e100_s50"
DEFAULT_OUTPUT_DIR = (
    "ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/"
    "raw_matrix_junjul_testmode_2021_2024_obsclim2001_2020"
)
DEFAULT_CONFIG = "ml_model/config_flow_multiv9.yaml"
DEFAULT_OBS_CLIM_START_YEAR = 2001
DEFAULT_OBS_CLIM_END_YEAR = 2020


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate raw v9 SA June/July forecasts.")
    parser.add_argument("--forecast_dir", type=str, default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--months", type=str, default="6,7")
    parser.add_argument("--skip_years", type=str, default="")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--obs_clim_data_dir", type=str, default=None)
    parser.add_argument("--obs_clim_start_year", type=int, default=DEFAULT_OBS_CLIM_START_YEAR)
    parser.add_argument("--obs_clim_end_year", type=int, default=DEFAULT_OBS_CLIM_END_YEAR)
    parser.add_argument("--obs_clim_skip_years", type=str, default="")
    parser.add_argument(
        "--threshold_source",
        type=str,
        default="obs_climatology",
        choices=("obs_climatology", "obs_eval"),
        help="Main physical-event thresholds: independent obs climatology or eval-period obs.",
    )
    parser.add_argument("--extreme_quantiles", type=str, default="0.90,0.95")
    parser.add_argument("--decision_thresholds", type=str, default="0.1,0.25,0.5")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_obs_clim_data_dir(args, config):
    return (
        args.obs_clim_data_dir
        or os.environ.get("DATA_DIR_OVERRIDE")
        or config.get("data_dir")
        or "dataprocess"
    )


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


def month_name(month):
    return {6: "Jun", 7: "Jul"}.get(int(month), str(month))


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
    num = np.nansum(np.where(mask, arr * weights, 0.0))
    den = np.nansum(np.where(mask, weights, 0.0))
    return float(num / (den + 1e-12))


def weighted_sum_bool(mask, weights_2d):
    arr = np.asarray(mask, dtype=bool)
    weights = broadcast_weights(arr, weights_2d)
    return float(np.sum(np.where(arr, weights, 0.0)))


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
    x_dev = np.where(mask, x_arr - x_mean, 0.0)
    y_dev = np.where(mask, y_arr - y_mean, 0.0)
    cov = np.sum(weights * x_dev * y_dev) / w_sum
    x_var = np.sum(weights * x_dev ** 2) / w_sum
    y_var = np.sum(weights * y_dev ** 2) / w_sum
    if x_var <= 0 or y_var <= 0:
        return np.nan
    return float(cov / np.sqrt(x_var * y_var))


def crps_ensemble(ensemble, obs):
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


def open_year(forecast_dir, year):
    path = os.path.join(forecast_dir, f"{year}.zarr")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_zarr(path, consolidated=False, chunks=None)


def load_eval_dataset(args, months):
    skip_years = parse_int_set(args.skip_years)
    datasets = []
    loaded_years = []
    missing_years = []
    for year in range(args.start_year, args.end_year + 1):
        if year in skip_years:
            continue
        path = os.path.join(args.forecast_dir, f"{year}.zarr")
        if not os.path.exists(path):
            print(f"⚠️ Missing eval year {year}: {path}. Skipping.")
            missing_years.append(year)
            continue
        ds = open_year(args.forecast_dir, year)
        mask = np.isin(pd.to_datetime(ds["init"].values).month, months)
        if not np.any(mask):
            print(f"⚠️ Eval year {year}: no init dates for months={months}.")
            ds.close()
            continue
        datasets.append(ds.isel(init=np.where(mask)[0]).load())
        ds.close()
        loaded_years.append(year)

    if not datasets:
        raise RuntimeError("No evaluation Zarrs were loaded.")
    combined = xr.concat(datasets, dim="init")
    for ds in datasets:
        ds.close()
    combined.attrs["loaded_years"] = ",".join(str(year) for year in loaded_years)
    combined.attrs["missing_years"] = ",".join(str(year) for year in missing_years)
    combined.attrs["skip_years"] = ",".join(str(year) for year in sorted(skip_years))
    return combined, loaded_years, missing_years, sorted(skip_years)


def load_obs_climatology(args, config, months, variables, expected_shape):
    """Load independent raw observed PR/T2M climatology from dataset files."""
    data_dir = resolve_obs_clim_data_dir(args, config)
    skip_years = parse_int_set(args.obs_clim_skip_years)
    obs_by_month = {month: {variable: [] for variable in variables} for month in months}
    loaded_years = []
    empty_years = []
    failed_years = {}
    init_counts = {}

    print(
        "  Obs climatology: raw observations "
        f"{args.obs_clim_start_year}-{args.obs_clim_end_year} "
        f"from {data_dir}"
    )

    for year in range(args.obs_clim_start_year, args.obs_clim_end_year + 1):
        if year in skip_years:
            continue
        try:
            clim_ds = S2SHybridDataset(
                data_root=data_dir,
                start_year=year,
                end_year=year,
                preload=False,
                normalize=False,
                stats_file=config.get("stats_file", ""),
                subsample_monthly=False,
                target_domain=config.get("target_domain"),
                target_domain_bounds=config.get("target_domain_bounds"),
                local_obs_variables=config.get("local_obs_variables"),
                global_context_variables=config.get("global_context_variables"),
                t2m_target_mode=config.get("t2m_target_mode", "absolute"),
            )
        except Exception as exc:
            print(f"  ⚠️ Obs climatology year {year}: failed to index ({exc}). Skipping.")
            failed_years[str(year)] = str(exc)
            continue

        count = 0
        for sample_idx, sample in enumerate(clim_ds.samples):
            if int(sample.get("lead_idx", -1)) != 0:
                continue
            init_date = pd.Timestamp(sample["date"])
            month = int(init_date.month)
            if month not in months:
                continue
            try:
                item = clim_ds[sample_idx]
                target_raw = item["target_raw_full"].detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception as exc:
                print(f"  ⚠️ Obs climatology {year} {init_date.date()}: failed to load ({exc}). Skipping init.")
                continue
            if target_raw.shape[1:] != expected_shape:
                raise RuntimeError(
                    f"Obs climatology shape mismatch for {year} {init_date.date()}: "
                    f"got {target_raw.shape[1:]}, expected {expected_shape}"
                )
            obs_by_month[month]["pr"].append(target_raw[0])
            obs_by_month[month]["t2m"].append(target_raw[1])
            count += 1

        if count > 0:
            loaded_years.append(year)
            init_counts[str(year)] = count
        else:
            empty_years.append(year)

    obs_clim = {}
    for month in months:
        for variable in variables:
            values = obs_by_month[month][variable]
            if not values:
                raise RuntimeError(
                    f"No observed climatology samples for month={month}, variable={variable}. "
                    "Check obs climatology years/data_dir."
                )
            obs_clim[(month, variable)] = np.stack(values, axis=0)

    metadata = {
        "data_dir": data_dir,
        "start_year": args.obs_clim_start_year,
        "end_year": args.obs_clim_end_year,
        "loaded_years": loaded_years,
        "empty_years": empty_years,
        "failed_years": failed_years,
        "skip_years": sorted(skip_years),
        "init_counts": init_counts,
    }
    print(f"  Obs climatology loaded years: {loaded_years}")
    if empty_years:
        print(f"  Obs climatology empty years: {empty_years}")
    if failed_years:
        print(f"  Obs climatology failed years: {sorted(failed_years)}")
    return obs_clim, metadata


def data_for(ds, variable, system):
    prefix = "model" if system == "ml" else "geos"
    return ds[f"{prefix}_{variable}"].values.astype(np.float32, copy=False)


def obs_for(ds, variable):
    return ds[f"obs_{variable}"].values.astype(np.float32, copy=False)


def overall_metrics(scope, year, init, month, variable, system, ensemble, obs, weights, lead_idx=None):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    ens_mean = np.nanmean(ens, axis=1)
    spread = np.nanstd(ens, axis=1, ddof=1) if ens.shape[1] > 1 else np.zeros_like(ens_mean)
    err = ens_mean - target
    rmse = float(np.sqrt(max(weighted_mean(err ** 2, weights), 0.0)))
    spread_mean = weighted_mean(spread, weights)
    return {
        "scope": scope,
        "year": year,
        "init": init,
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
        "forecast_raw_std": weighted_std(ens_mean, weights),
        "obs_raw_std": weighted_std(target, weights),
    }


def event_probability(ensemble, threshold, tail):
    if tail == "upper":
        return np.nanmean(ensemble > threshold[None, None], axis=1)
    if tail == "lower":
        return np.nanmean(ensemble < threshold[None, None], axis=1)
    raise ValueError(tail)


def event_mask(obs, threshold, tail):
    if tail == "upper":
        return obs > threshold[None]
    if tail == "lower":
        return obs < threshold[None]
    raise ValueError(tail)


def raw_threshold(obs, quantile, tail):
    q = float(quantile) if tail == "upper" else 1.0 - float(quantile)
    return np.nanquantile(obs, q, axis=0)


def tails_for_variable(variable):
    if variable == "t2m":
        return ("upper", "lower")
    return ("upper",)


def ensemble_threshold(ensemble, quantile, tail):
    arr = np.asarray(ensemble, dtype=np.float32)
    if arr.ndim != 5:
        raise ValueError(f"Expected ensemble shape [init, member, lead, lat, lon], got {arr.shape}")
    n_init, n_member, n_lead, n_lat, n_lon = arr.shape
    return raw_threshold(arr.reshape(n_init * n_member, n_lead, n_lat, n_lon), quantile, tail)


def build_obs_thresholds(obs_by_month, months, quantiles):
    thresholds = {}
    for month in months:
        thresholds[month] = {}
        for variable in ("pr", "t2m"):
            obs = obs_by_month[(month, variable)]
            thresholds[month][variable] = {
                (tail, quantile): raw_threshold(obs, quantile, tail)
                for tail in tails_for_variable(variable)
                for quantile in quantiles
            }
    return thresholds


def build_eval_obs_thresholds(ds, months, quantiles):
    init_dates = pd.to_datetime(ds["init"].values).normalize()
    thresholds = {}
    for month in months:
        indices = np.where(init_dates.month == int(month))[0]
        if len(indices) == 0:
            continue
        month_sub = ds.isel(init=indices)
        thresholds[month] = {}
        for variable in ("pr", "t2m"):
            month_obs = obs_for(month_sub, variable)
            thresholds[month][variable] = {
                (tail, quantile): raw_threshold(month_obs, quantile, tail)
                for tail in tails_for_variable(variable)
                for quantile in quantiles
            }
    return thresholds


def build_self_thresholds(month_sub, quantiles):
    """Build 2021-2024 relative-tail thresholds for OBS, ML, and GEOS separately."""
    thresholds = {}
    for system in ("obs", "ml", "geos"):
        thresholds[system] = {}
        for variable in ("pr", "t2m"):
            values = obs_for(month_sub, variable) if system == "obs" else data_for(month_sub, variable, system)
            thresholds[system][variable] = {
                (tail, quantile): (
                    raw_threshold(values, quantile, tail)
                    if system == "obs"
                    else ensemble_threshold(values, quantile, tail)
                )
                for tail in tails_for_variable(variable)
                for quantile in quantiles
            }
    return thresholds


def extreme_metrics(
    scope,
    year,
    init,
    month,
    variable,
    system,
    ensemble,
    obs,
    weights,
    quantile,
    tail,
    threshold,
    lead_idx=None,
    threshold_source="obs_climatology",
):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    if lead_idx is not None:
        threshold = threshold[lead_idx:lead_idx + 1]
    prob = event_probability(ens, threshold, tail)
    event = event_mask(target, threshold, tail)
    ens_mean = np.nanmean(ens, axis=1)
    spread = np.nanstd(ens, axis=1, ddof=1) if ens.shape[1] > 1 else np.zeros_like(ens_mean)
    event_err = np.where(event, ens_mean - target, np.nan)
    event_rmse = float(np.sqrt(max(weighted_mean(event_err ** 2, weights), 0.0)))
    obs_rate = weighted_mean(event.astype(np.float32), weights)
    forecast_rate = weighted_mean(prob, weights)
    return {
        "scope": scope,
        "year": year,
        "init": init,
        "month": int(month),
        "variable": variable,
        "system": system,
        "lead": "all" if lead_idx is None else int(lead_idx + 1),
        "tail": tail,
        "quantile": float(quantile),
        "threshold_source": threshold_source,
        "threshold_area_mean": weighted_mean(threshold, weights),
        "obs_event_rate": obs_rate,
        "forecast_event_probability_mean": forecast_rate,
        "frequency_bias": forecast_rate / obs_rate if obs_rate > 0 else np.nan,
        "brier_score": weighted_mean((prob - event.astype(np.float32)) ** 2, weights),
        "event_rmse": event_rmse,
        "event_bias": weighted_mean(event_err, weights),
        "ensemble_spread_on_obs_events": weighted_mean(np.where(event, spread, np.nan), weights),
        "forecast_raw_on_obs_events": weighted_mean(np.where(event, ens_mean, np.nan), weights),
        "obs_raw_on_events": weighted_mean(np.where(event, target, np.nan), weights),
    }, prob, event


def self_threshold_extreme_metrics(
    scope,
    year,
    init,
    month,
    variable,
    system,
    ensemble,
    obs,
    weights,
    quantile,
    tail,
    obs_threshold,
    forecast_threshold,
    lead_idx=None,
):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    if lead_idx is not None:
        obs_threshold = obs_threshold[lead_idx:lead_idx + 1]
        forecast_threshold = forecast_threshold[lead_idx:lead_idx + 1]
    prob = event_probability(ens, forecast_threshold, tail)
    event = event_mask(target, obs_threshold, tail)
    ens_mean = np.nanmean(ens, axis=1)
    spread = np.nanstd(ens, axis=1, ddof=1) if ens.shape[1] > 1 else np.zeros_like(ens_mean)
    event_err = np.where(event, ens_mean - target, np.nan)
    event_rmse = float(np.sqrt(max(weighted_mean(event_err ** 2, weights), 0.0)))
    obs_rate = weighted_mean(event.astype(np.float32), weights)
    forecast_rate = weighted_mean(prob, weights)
    brier = weighted_mean((prob - event.astype(np.float32)) ** 2, weights)
    return {
        "scope": scope,
        "year": year,
        "init": init,
        "month": int(month),
        "variable": variable,
        "system": system,
        "lead": "all" if lead_idx is None else int(lead_idx + 1),
        "tail": tail,
        "quantile": float(quantile),
        "threshold_source": "self_eval_period",
        "obs_threshold_area_mean": weighted_mean(obs_threshold, weights),
        "forecast_threshold_area_mean": weighted_mean(forecast_threshold, weights),
        "obs_event_rate": obs_rate,
        "forecast_event_probability_mean": forecast_rate,
        "forecast_probability_bias_vs_obs": forecast_rate - obs_rate,
        "frequency_bias": forecast_rate / obs_rate if obs_rate > 0 else np.nan,
        "brier_score": brier,
        "event_rmse": event_rmse,
        "event_bias": weighted_mean(event_err, weights),
        "ensemble_spread_on_obs_events": weighted_mean(np.where(event, spread, np.nan), weights),
        "forecast_raw_on_obs_events": weighted_mean(np.where(event, ens_mean, np.nan), weights),
        "obs_raw_on_events": weighted_mean(np.where(event, target, np.nan), weights),
        "notes": "Diagnostic only: forecast probability uses each system's own eval-period threshold; event truth uses obs eval-period threshold.",
    }


def weighted_decision_rows(base, prob, event, weights, decision_thresholds):
    rows = []
    for decision in decision_thresholds:
        yes = prob >= decision
        hits = weighted_sum_bool(yes & event, weights)
        false_alarms = weighted_sum_bool(yes & ~event, weights)
        misses = weighted_sum_bool(~yes & event, weights)
        base_keys = ("scope", "year", "init", "month", "variable", "system", "lead", "tail", "quantile", "threshold_source")
        row = {key: base[key] for key in base_keys if key in base}
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
        base_keys = ("scope", "year", "init", "month", "variable", "system", "lead", "tail", "quantile", "threshold_source")
        row = {key: base[key] for key in base_keys if key in base}
        row.update({
            "prob_bin_left": float(left),
            "prob_bin_right": float(right),
            "weighted_count": weighted_count,
            "mean_forecast_probability": weighted_mean(np.where(in_bin, prob, np.nan), weights),
            "observed_frequency": weighted_mean(np.where(in_bin, event.astype(np.float32), np.nan), weights),
        })
        rows.append(row)
    return rows


def lead_all(df):
    return df[df["lead"].astype(str) == "all"].copy()


def all_years_lead_all(df):
    return df[(df["scope"] == "all_years") & (df["lead"].astype(str) == "all")].copy()


def ml_vs_geos_improvement(overall_df):
    key_cols = ["scope", "year", "init", "month", "variable", "lead"]
    ml = overall_df[overall_df["system"] == "ml"].set_index(key_cols)
    geos = overall_df[overall_df["system"] == "geos"].set_index(key_cols)
    rows = []
    for idx in ml.index:
        if idx not in geos.index:
            continue
        geos_crps = geos.loc[idx, "crps"]
        geos_rmse = geos.loc[idx, "rmse"]
        row = dict(zip(key_cols, idx))
        row.update({
            "ml_crps": ml.loc[idx, "crps"],
            "geos_crps": geos_crps,
            "crps_improve_pct": 100.0 * (geos_crps - ml.loc[idx, "crps"]) / geos_crps if geos_crps else np.nan,
            "ml_rmse": ml.loc[idx, "rmse"],
            "geos_rmse": geos_rmse,
            "rmse_improve_pct": 100.0 * (geos_rmse - ml.loc[idx, "rmse"]) / geos_rmse if geos_rmse else np.nan,
            "ml_acc": ml.loc[idx, "acc"],
            "geos_acc": geos.loc[idx, "acc"],
            "acc_delta": ml.loc[idx, "acc"] - geos.loc[idx, "acc"],
            "ml_spread_error_ratio": ml.loc[idx, "spread_error_ratio"],
            "geos_spread_error_ratio": geos.loc[idx, "spread_error_ratio"],
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key_cols)


def print_table(title, df, columns, sort_by=None, float_format="{:.4f}"):
    if df.empty:
        print(f"\n{title}: no rows")
        return
    table = df.copy()
    if sort_by:
        table = table.sort_values(sort_by)
    numeric_cols = table.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in {"month", "n_init"}:
            table[col] = table[col].map(lambda value: str(int(value)) if pd.notna(value) else "nan")
        elif col in {"quantile", "decision_probability"}:
            table[col] = table[col].map(lambda value: f"{value:.2f}" if pd.notna(value) else "nan")
        else:
            table[col] = table[col].map(lambda value: float_format.format(value) if pd.notna(value) else "nan")
    print(f"\n{title}")
    print(table[columns].to_string(index=False))


def print_evaluation_summaries(overall_df, extreme_df, self_extreme_df, decision_df):
    total = all_years_lead_all(overall_df)
    print_table(
        "Raw forecast metrics (lead=all; lower CRPS/RMSE better, higher ACC better)",
        total,
        [
            "month",
            "variable",
            "system",
            "crps",
            "rmse",
            "bias",
            "acc",
            "ensemble_spread_mean",
            "spread_error_ratio",
            "forecast_raw_std",
            "obs_raw_std",
        ],
        sort_by=["month", "variable", "system"],
    )

    improvement = ml_vs_geos_improvement(overall_df)
    improvement_total = all_years_lead_all(improvement)
    print_table(
        "Raw ML improvement vs GEOS (lead=all)",
        improvement_total,
        [
            "month",
            "variable",
            "crps_improve_pct",
            "rmse_improve_pct",
            "acc_delta",
            "ml_spread_error_ratio",
            "geos_spread_error_ratio",
        ],
        sort_by=["month", "variable"],
    )

    extreme_total = all_years_lead_all(extreme_df)
    print_table(
        "Raw physical-threshold extreme metrics (lead=all; thresholds from obs climatology unless overridden)",
        extreme_total,
        [
            "month",
            "variable",
            "tail",
            "quantile",
            "threshold_source",
            "system",
            "obs_event_rate",
            "forecast_event_probability_mean",
            "frequency_bias",
            "brier_score",
            "event_rmse",
            "ensemble_spread_on_obs_events",
        ],
        sort_by=["month", "variable", "tail", "quantile", "system"],
    )

    self_extreme_total = all_years_lead_all(self_extreme_df)
    print_table(
        "Raw relative-tail self-threshold metrics (diagnostic only; q from OBS/ML/GEOS evaluation period separately)",
        self_extreme_total,
        [
            "month",
            "variable",
            "tail",
            "quantile",
            "system",
            "obs_threshold_area_mean",
            "forecast_threshold_area_mean",
            "obs_event_rate",
            "forecast_event_probability_mean",
            "forecast_probability_bias_vs_obs",
            "frequency_bias",
            "brier_score",
        ],
        sort_by=["month", "variable", "tail", "quantile", "system"],
    )

    decision_total = all_years_lead_all(decision_df)
    if not decision_total.empty:
        decision_focus = decision_total[decision_total["decision_probability"].isin([0.25, 0.5])]
        print_table(
            "Raw extreme event decision scores (lead=all)",
            decision_focus,
            [
                "month",
                "variable",
                "tail",
                "quantile",
                "system",
                "decision_probability",
                "pod",
                "far",
                "csi",
            ],
            sort_by=["month", "variable", "tail", "quantile", "decision_probability", "system"],
        )
    return improvement


def print_final_evaluation_summary(overall_df, extreme_df, self_extreme_df, improvement_df):
    print("\n" + "=" * 88)
    print("FINAL RAW EVALUATION SUMMARY")
    print("=" * 88)

    total = all_years_lead_all(overall_df)
    print_table(
        "Final raw metrics by month and variable (all years, all leads)",
        total,
        [
            "month",
            "variable",
            "system",
            "crps",
            "rmse",
            "bias",
            "acc",
            "ensemble_spread_mean",
            "spread_error_ratio",
        ],
        sort_by=["month", "variable", "system"],
    )

    improvement_total = all_years_lead_all(improvement_df)
    print_table(
        "Final ML improvement vs GEOS (all years, all leads)",
        improvement_total,
        [
            "month",
            "variable",
            "crps_improve_pct",
            "rmse_improve_pct",
            "acc_delta",
            "ml_spread_error_ratio",
            "geos_spread_error_ratio",
        ],
        sort_by=["month", "variable"],
    )

    weekly = improvement_df[
        (improvement_df["scope"] == "all_years") & (improvement_df["lead"].astype(str) != "all")
    ].copy()
    print_table(
        "Final weekly lead ML improvement vs GEOS (all years)",
        weekly,
        [
            "month",
            "variable",
            "lead",
            "crps_improve_pct",
            "rmse_improve_pct",
            "acc_delta",
        ],
        sort_by=["month", "variable", "lead"],
    )

    extreme_total = all_years_lead_all(extreme_df)
    if not extreme_total.empty:
        focus_quantile = float(extreme_total["quantile"].max())
        extreme_focus = extreme_total[extreme_total["quantile"] == focus_quantile].copy()
        print_table(
            f"Final extreme-event metrics (q={focus_quantile:.2f}, all years, all leads)",
            extreme_focus,
            [
                "month",
                "variable",
                "tail",
                "system",
                "threshold_source",
                "obs_event_rate",
                "forecast_event_probability_mean",
                "frequency_bias",
                "brier_score",
                "event_rmse",
            ],
            sort_by=["month", "variable", "tail", "system"],
        )

    self_extreme_total = all_years_lead_all(self_extreme_df)
    if not self_extreme_total.empty:
        focus_quantile = float(self_extreme_total["quantile"].max())
        self_focus = self_extreme_total[self_extreme_total["quantile"] == focus_quantile].copy()
        print_table(
            f"Final relative-tail self-threshold metrics (q={focus_quantile:.2f}, diagnostic only)",
            self_focus,
            [
                "month",
                "variable",
                "tail",
                "system",
                "obs_event_rate",
                "forecast_event_probability_mean",
                "frequency_bias",
                "brier_score",
            ],
            sort_by=["month", "variable", "tail", "system"],
        )

    print("=" * 88)


def add_value_labels(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -3
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_grouped_metric(ax, total, metric, ylabel, title, labels):
    x = np.arange(len(labels))
    width = 0.36
    for idx, system in enumerate(("ml", "geos")):
        values = []
        for month, variable in labels:
            row = total[(total["month"] == month) & (total["variable"] == variable) & (total["system"] == system)]
            values.append(row[metric].iloc[0] if not row.empty else np.nan)
        bars = ax.bar(x + (idx - 0.5) * width, values, width=width, label=system.upper())
        add_value_labels(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{month_name(m)} {v.upper()}" for m, v in labels], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)


def make_plots(overall_df, extreme_df, self_extreme_df, reliability_df, improvement_df, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    mpl_config_dir = os.path.join(output_dir, ".matplotlib_cache")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"⚠️ Matplotlib unavailable; skipping plots ({exc})")
        return []

    plot_paths = []
    total = all_years_lead_all(overall_df)
    labels = sorted(total[["month", "variable"]].drop_duplicates().itertuples(index=False, name=None))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plot_grouped_metric(axes[0, 0], total, "crps", "CRPS", "Raw CRPS", labels)
    plot_grouped_metric(axes[0, 1], total, "rmse", "RMSE", "Raw RMSE", labels)
    plot_grouped_metric(axes[1, 0], total, "bias", "Bias", "Raw bias", labels)
    plot_grouped_metric(axes[1, 1], total, "spread_error_ratio", "Spread / RMSE", "Spread calibration", labels)
    axes[0, 0].legend(loc="best")
    path = os.path.join(plot_dir, "raw_overall_skill.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels_text = [f"{month_name(row.month)} {row.variable.upper()}" for row in improvement_df.itertuples()]
    x = np.arange(len(labels_text))
    width = 0.36
    bars1 = ax.bar(x - width / 2, improvement_df["crps_improve_pct"], width=width, label="CRPS improvement")
    bars2 = ax.bar(x + width / 2, improvement_df["rmse_improve_pct"], width=width, label="RMSE improvement")
    add_value_labels(ax, bars1)
    add_value_labels(ax, bars2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_text, rotation=25, ha="right")
    ax.set_ylabel("Improvement vs GEOS (%)")
    ax.set_title("Raw ML skill improvement vs GEOS")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    path = os.path.join(plot_dir, "raw_ml_vs_geos_improvement.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    extreme_total = all_years_lead_all(extreme_df)
    focus = extreme_total[
        ((extreme_total["variable"] == "pr") & (extreme_total["tail"] == "upper"))
        | ((extreme_total["variable"] == "t2m") & (extreme_total["tail"].isin(["upper", "lower"])))
    ]
    if not focus.empty:
        groups = sorted(focus[["variable", "tail", "quantile"]].drop_duplicates().itertuples(index=False, name=None))
        fig, axes = plt.subplots(len(groups), 2, figsize=(13, 3.8 * len(groups)), squeeze=False, constrained_layout=True)
        for row_idx, (variable, tail, quantile) in enumerate(groups):
            gdf = focus[
                (focus["variable"] == variable)
                & (focus["tail"] == tail)
                & np.isclose(focus["quantile"], quantile)
            ]
            group_labels = sorted(gdf[["month", "variable"]].drop_duplicates().itertuples(index=False, name=None))
            plot_grouped_metric(
                axes[row_idx, 0],
                gdf,
                "brier_score",
                "Brier score",
                f"{variable.upper()} {tail} q={quantile:.2f}",
                group_labels,
            )
            plot_grouped_metric(
                axes[row_idx, 1],
                gdf,
                "frequency_bias",
                "Frequency bias",
                f"{variable.upper()} {tail} frequency q={quantile:.2f}",
                group_labels,
            )
            axes[row_idx, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axes[0, 0].legend(loc="best")
        path = os.path.join(plot_dir, "raw_extreme_skill.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    self_total = all_years_lead_all(self_extreme_df)
    if not self_total.empty:
        qmax = float(self_total["quantile"].max())
        self_focus = self_total[np.isclose(self_total["quantile"], qmax)].copy()
        self_focus["label"] = (
            self_focus["month"].map(month_name)
            + " "
            + self_focus["variable"].str.upper()
            + " "
            + self_focus["tail"]
        )
        labels_self = list(self_focus[self_focus["system"] == "ml"]["label"])
        x = np.arange(len(labels_self))
        width = 0.36
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
        for ax, metric, title in (
            (axes[0], "brier_score", f"Relative-tail Brier score q={qmax:.2f}"),
            (axes[1], "frequency_bias", f"Relative-tail frequency bias q={qmax:.2f}"),
        ):
            for offset, system in ((-width / 2, "ml"), (width / 2, "geos")):
                values = (
                    self_focus[self_focus["system"] == system]
                    .set_index("label")
                    .reindex(labels_self)[metric]
                    .values
                )
                bars = ax.bar(x + offset, values, width=width, label=system.upper())
                add_value_labels(ax, bars, fmt="{:.2f}")
            if metric == "frequency_bias":
                ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_self, rotation=25, ha="right")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].legend(loc="best")
        fig.suptitle(
            "Diagnostic only: each system uses its own 2021-2024 threshold",
            fontsize=11,
        )
        path = os.path.join(plot_dir, "raw_self_threshold_extreme_skill_qmax.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    rel_total = all_years_lead_all(reliability_df)
    if not rel_total.empty:
        qmax = rel_total["quantile"].max()
        rel_q = rel_total[np.isclose(rel_total["quantile"], qmax)]
        rel_q = rel_q[((rel_q["variable"] == "pr") & (rel_q["tail"] == "upper")) | (rel_q["variable"] == "t2m")]
        panels = sorted(rel_q[["month", "variable", "tail"]].drop_duplicates().itertuples(index=False, name=None))
        if panels:
            ncols = 2
            nrows = int(np.ceil(len(panels) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False, constrained_layout=True)
            for ax, (month, variable, tail) in zip(axes.ravel(), panels):
                subset = rel_q[(rel_q["month"] == month) & (rel_q["variable"] == variable) & (rel_q["tail"] == tail)]
                for system in ("ml", "geos"):
                    s = subset[subset["system"] == system].sort_values("mean_forecast_probability")
                    if s.empty:
                        continue
                    ax.plot(
                        s["mean_forecast_probability"],
                        s["observed_frequency"],
                        marker="o",
                        linewidth=1.8,
                        label=system.upper(),
                    )
                ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"{month_name(month)} {variable.upper()} {tail} q={qmax:.2f}")
                ax.set_xlabel("Forecast probability")
                ax.set_ylabel("Observed frequency")
                ax.grid(True, alpha=0.25)
            for ax in axes.ravel()[len(panels):]:
                ax.axis("off")
            axes[0, 0].legend(loc="best")
            path = os.path.join(plot_dir, "raw_extreme_reliability_qmax.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plot_paths.append(path)

    return plot_paths


def evaluate_subset(
    sub,
    scope,
    year_label,
    init_label,
    month,
    weights,
    quantiles,
    decision_thresholds,
    thresholds_by_variable,
    threshold_source,
    overall_rows,
    extreme_rows,
    decision_rows,
    reliability_matrix_rows,
):
    for variable in ("pr", "t2m"):
        obs = obs_for(sub, variable)
        thresholds = thresholds_by_variable[variable]
        for system in ("ml", "geos"):
            ensemble = data_for(sub, variable, system)
            overall_rows.append(overall_metrics(scope, year_label, init_label, month, variable, system, ensemble, obs, weights))
            for lead_idx in range(obs.shape[1]):
                overall_rows.append(
                    overall_metrics(scope, year_label, init_label, month, variable, system, ensemble, obs, weights, lead_idx=lead_idx)
                )

            for tail in tails_for_variable(variable):
                for quantile in quantiles:
                    threshold = thresholds[(tail, quantile)]
                    row, prob, event = extreme_metrics(
                        scope,
                        year_label,
                        init_label,
                        month,
                        variable,
                        system,
                        ensemble,
                        obs,
                        weights,
                        quantile,
                        tail,
                        threshold,
                        threshold_source=threshold_source,
                    )
                    extreme_rows.append(row)
                    decision_rows.extend(weighted_decision_rows(row, prob, event, weights, decision_thresholds))
                    reliability_matrix_rows.extend(reliability_rows(row, prob, event, weights))
                    for lead_idx in range(obs.shape[1]):
                        lead_row, lead_prob, lead_event = extreme_metrics(
                            scope,
                            year_label,
                            init_label,
                            month,
                            variable,
                            system,
                            ensemble,
                            obs,
                            weights,
                            quantile,
                            tail,
                            threshold,
                            lead_idx=lead_idx,
                            threshold_source=threshold_source,
                        )
                        extreme_rows.append(lead_row)
                        decision_rows.extend(weighted_decision_rows(lead_row, lead_prob, lead_event, weights, decision_thresholds))
                        reliability_matrix_rows.extend(reliability_rows(lead_row, lead_prob, lead_event, weights))


def evaluate_self_threshold_subset(
    sub,
    scope,
    year_label,
    init_label,
    month,
    weights,
    quantiles,
    self_thresholds,
    self_extreme_rows,
):
    for variable in ("pr", "t2m"):
        obs = obs_for(sub, variable)
        obs_thresholds = self_thresholds["obs"][variable]
        for system in ("ml", "geos"):
            ensemble = data_for(sub, variable, system)
            forecast_thresholds = self_thresholds[system][variable]
            for tail in tails_for_variable(variable):
                for quantile in quantiles:
                    obs_threshold = obs_thresholds[(tail, quantile)]
                    forecast_threshold = forecast_thresholds[(tail, quantile)]
                    self_extreme_rows.append(
                        self_threshold_extreme_metrics(
                            scope,
                            year_label,
                            init_label,
                            month,
                            variable,
                            system,
                            ensemble,
                            obs,
                            weights,
                            quantile,
                            tail,
                            obs_threshold,
                            forecast_threshold,
                        )
                    )
                    for lead_idx in range(obs.shape[1]):
                        self_extreme_rows.append(
                            self_threshold_extreme_metrics(
                                scope,
                                year_label,
                                init_label,
                                month,
                                variable,
                                system,
                                ensemble,
                                obs,
                                weights,
                                quantile,
                                tail,
                                obs_threshold,
                                forecast_threshold,
                                lead_idx=lead_idx,
                            )
                        )


def main():
    args = parse_args()
    months = parse_int_list(args.months)
    quantiles = parse_float_list(args.extreme_quantiles)
    decision_thresholds = parse_float_list(args.decision_thresholds)
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)

    ds, loaded_years, missing_years, skip_years = load_eval_dataset(args, months)
    try:
        weights = area_weights(ds["lat"].values)
        overall_rows = []
        extreme_rows = []
        self_extreme_rows = []
        decision_rows = []
        reliability_matrix_rows = []
        init_dates = pd.to_datetime(ds["init"].values).normalize()
        init_month = init_dates.month
        init_year = init_dates.year
        expected_shape = obs_for(ds, "pr").shape[1:]
        obs_clim_metadata = None
        if args.threshold_source == "obs_climatology":
            obs_clim, obs_clim_metadata = load_obs_climatology(
                args, config, months, ("pr", "t2m"), expected_shape
            )
            physical_thresholds = build_obs_thresholds(obs_clim, months, quantiles)
        else:
            physical_thresholds = build_eval_obs_thresholds(ds, months, quantiles)

        for month in months:
            indices = np.where(init_month == int(month))[0]
            if len(indices) == 0:
                print(f"⚠️ No raw init dates for month={month}.")
                continue
            month_sub = ds.isel(init=indices)
            thresholds_by_variable = physical_thresholds[int(month)]
            self_thresholds = build_self_thresholds(month_sub, quantiles)

            evaluate_subset(
                month_sub,
                "all_years",
                "all",
                "all",
                month,
                weights,
                quantiles,
                decision_thresholds,
                thresholds_by_variable,
                args.threshold_source,
                overall_rows,
                extreme_rows,
                decision_rows,
                reliability_matrix_rows,
            )
            evaluate_self_threshold_subset(
                month_sub,
                "all_years",
                "all",
                "all",
                month,
                weights,
                quantiles,
                self_thresholds,
                self_extreme_rows,
            )

            for year in sorted(np.unique(init_year[indices])):
                year_indices = indices[init_year[indices] == year]
                evaluate_subset(
                    ds.isel(init=year_indices),
                    "year",
                    str(int(year)),
                    "all",
                    month,
                    weights,
                    quantiles,
                    decision_thresholds,
                    thresholds_by_variable,
                    args.threshold_source,
                    overall_rows,
                    extreme_rows,
                    decision_rows,
                    reliability_matrix_rows,
                )
                evaluate_self_threshold_subset(
                    ds.isel(init=year_indices),
                    "year",
                    str(int(year)),
                    "all",
                    month,
                    weights,
                    quantiles,
                    self_thresholds,
                    self_extreme_rows,
                )

            for idx in indices:
                init_label = init_dates[idx].strftime("%Y-%m-%d")
                evaluate_subset(
                    ds.isel(init=[idx]),
                    "init",
                    init_dates[idx].strftime("%Y"),
                    init_label,
                    month,
                    weights,
                    quantiles,
                    decision_thresholds,
                    thresholds_by_variable,
                    args.threshold_source,
                    overall_rows,
                    extreme_rows,
                    decision_rows,
                    reliability_matrix_rows,
                )
                evaluate_self_threshold_subset(
                    ds.isel(init=[idx]),
                    "init",
                    init_dates[idx].strftime("%Y"),
                    init_label,
                    month,
                    weights,
                    quantiles,
                    self_thresholds,
                    self_extreme_rows,
                )
    finally:
        ds.close()

    overall_df = pd.DataFrame(overall_rows)
    extreme_df = pd.DataFrame(extreme_rows)
    self_extreme_df = pd.DataFrame(self_extreme_rows)
    decision_df = pd.DataFrame(decision_rows)
    reliability_df = pd.DataFrame(reliability_matrix_rows)
    improvement_df = print_evaluation_summaries(overall_df, extreme_df, self_extreme_df, decision_df)
    plot_paths = make_plots(
        overall_df,
        extreme_df,
        self_extreme_df,
        reliability_df,
        all_years_lead_all(improvement_df),
        args.output_dir,
    )

    paths = {
        "overall": os.path.join(args.output_dir, "raw_overall_matrix.csv"),
        "overall_yearly": os.path.join(args.output_dir, "raw_overall_yearly_matrix.csv"),
        "overall_init": os.path.join(args.output_dir, "raw_overall_init_matrix.csv"),
        "overall_all_year_weekly": os.path.join(args.output_dir, "raw_overall_all_year_weekly_matrix.csv"),
        "extreme": os.path.join(args.output_dir, "raw_extreme_matrix.csv"),
        "extreme_yearly": os.path.join(args.output_dir, "raw_extreme_yearly_matrix.csv"),
        "extreme_init": os.path.join(args.output_dir, "raw_extreme_init_matrix.csv"),
        "extreme_all_year_weekly": os.path.join(args.output_dir, "raw_extreme_all_year_weekly_matrix.csv"),
        "self_extreme": os.path.join(args.output_dir, "raw_self_threshold_extreme_matrix.csv"),
        "self_extreme_yearly": os.path.join(args.output_dir, "raw_self_threshold_extreme_yearly_matrix.csv"),
        "self_extreme_init": os.path.join(args.output_dir, "raw_self_threshold_extreme_init_matrix.csv"),
        "self_extreme_all_year_weekly": os.path.join(args.output_dir, "raw_self_threshold_extreme_all_year_weekly_matrix.csv"),
        "decision": os.path.join(args.output_dir, "raw_extreme_decision_matrix.csv"),
        "reliability": os.path.join(args.output_dir, "raw_extreme_reliability_matrix.csv"),
        "improvement": os.path.join(args.output_dir, "raw_ml_vs_geos_improvement.csv"),
        "metadata": os.path.join(args.output_dir, "raw_matrix_metadata.json"),
    }
    overall_df.to_csv(paths["overall"], index=False)
    overall_df[overall_df["scope"] == "year"].to_csv(paths["overall_yearly"], index=False)
    overall_df[overall_df["scope"] == "init"].to_csv(paths["overall_init"], index=False)
    overall_df[(overall_df["scope"] == "all_years") & (overall_df["lead"].astype(str) != "all")].to_csv(
        paths["overall_all_year_weekly"], index=False
    )
    extreme_df.to_csv(paths["extreme"], index=False)
    extreme_df[extreme_df["scope"] == "year"].to_csv(paths["extreme_yearly"], index=False)
    extreme_df[extreme_df["scope"] == "init"].to_csv(paths["extreme_init"], index=False)
    extreme_df[(extreme_df["scope"] == "all_years") & (extreme_df["lead"].astype(str) != "all")].to_csv(
        paths["extreme_all_year_weekly"], index=False
    )
    self_extreme_df.to_csv(paths["self_extreme"], index=False)
    self_extreme_df[self_extreme_df["scope"] == "year"].to_csv(paths["self_extreme_yearly"], index=False)
    self_extreme_df[self_extreme_df["scope"] == "init"].to_csv(paths["self_extreme_init"], index=False)
    self_extreme_df[
        (self_extreme_df["scope"] == "all_years") & (self_extreme_df["lead"].astype(str) != "all")
    ].to_csv(paths["self_extreme_all_year_weekly"], index=False)
    decision_df.to_csv(paths["decision"], index=False)
    reliability_df.to_csv(paths["reliability"], index=False)
    improvement_df.to_csv(paths["improvement"], index=False)
    with open(paths["metadata"], "w") as f:
        json.dump(
            {
                "forecast_dir": args.forecast_dir,
                "output_dir": args.output_dir,
                "years_requested": [args.start_year, args.end_year],
                "years_loaded": loaded_years,
                "years_missing": missing_years,
                "skip_years": skip_years,
                "months": months,
                "config": args.config,
                "threshold_source": args.threshold_source,
                "obs_climatology": obs_clim_metadata,
                "physical_threshold_definition": (
                    "Separate init-month, lead, and gridpoint quantiles from raw observed "
                    "2001-2020 targets by default. The same observed threshold is applied "
                    "to OBS truth, ML members, and GEOS members."
                ),
                "self_threshold_definition": (
                    "Separate init-month, lead, and gridpoint quantiles from the evaluation "
                    "period. OBS uses observed values; ML and GEOS each use their own pooled "
                    "init-by-member distribution. These scores are diagnostic and not an "
                    "independent physical-event verification."
                ),
                "extreme_quantiles": quantiles,
                "decision_thresholds": decision_thresholds,
                "plots": plot_paths,
                "notes": (
                    "Main extreme matrices use physical thresholds from observed climatology by default. "
                    "Self-threshold matrices are diagnostic relative-tail scores using separate evaluation-period OBS/ML/GEOS thresholds."
                ),
            },
            f,
            indent=2,
        )

    print("\nJune/July raw matrix evaluation complete")
    print(f"  Forecast dir : {args.forecast_dir}")
    print(f"  Years loaded : {loaded_years}")
    print(f"  Overall      : {paths['overall']}")
    print(f"  Year overall : {paths['overall_yearly']}")
    print(f"  Init overall : {paths['overall_init']}")
    print(f"  Weekly mean  : {paths['overall_all_year_weekly']}")
    print(f"  Extreme      : {paths['extreme']}")
    print(f"  Year extreme : {paths['extreme_yearly']}")
    print(f"  Init extreme : {paths['extreme_init']}")
    print(f"  Weekly event : {paths['extreme_all_year_weekly']}")
    print(f"  Self extreme : {paths['self_extreme']}")
    print(f"  Self yearly  : {paths['self_extreme_yearly']}")
    print(f"  Self init    : {paths['self_extreme_init']}")
    print(f"  Self weekly  : {paths['self_extreme_all_year_weekly']}")
    print(f"  Decision     : {paths['decision']}")
    print(f"  Reliability  : {paths['reliability']}")
    print(f"  Improvement  : {paths['improvement']}")
    if plot_paths:
        print("  Plots        :")
        for path in plot_paths:
            print(f"    {path}")
    print(f"  Metadata     : {paths['metadata']}")
    print_final_evaluation_summary(overall_df, extreme_df, self_extreme_df, improvement_df)


if __name__ == "__main__":
    main()
