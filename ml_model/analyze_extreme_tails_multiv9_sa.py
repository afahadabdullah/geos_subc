#!/usr/bin/env python3
"""
Analyze June/July extreme-tail behavior from saved multi-v9 SA forecast Zarrs.

Baseline thresholds are computed from observed 2005-2020 June/July init dates.
Evaluation scores are computed for 2021-2024 model and GEOS ensembles.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description="Extreme-tail analysis for saved multi-v9 SA Zarr forecasts.")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="dataprocess/gen_multiv9_sa_55e100e_0n40n_junjul_e10clim_e100eval_s50",
    )
    parser.add_argument("--baseline_start_year", type=int, default=2005)
    parser.add_argument("--baseline_end_year", type=int, default=2020)
    parser.add_argument("--eval_start_year", type=int, default=2021)
    parser.add_argument("--eval_end_year", type=int, default=2024)
    parser.add_argument("--quantiles", type=str, default="0.90,0.95,0.99")
    parser.add_argument("--decision_thresholds", type=str, default="0.1,0.25,0.5")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/extreme_tail_junjul",
    )
    return parser.parse_args()


def parse_float_list(text):
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def open_year(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_zarr(path, consolidated=False, chunks=None)


def load_obs_stack(forecast_dir, years, var_name, skip_missing=False):
    arrays = []
    init_values = []
    loaded_years = []
    missing_years = []
    for year in years:
        path = os.path.join(forecast_dir, f"{year}.zarr")
        if skip_missing and not os.path.exists(path):
            print(f"⚠️ Missing baseline year {year}: {path}. Skipping.")
            missing_years.append(year)
            continue
        ds = open_year(path)
        try:
            arrays.append(ds[var_name].values.astype(np.float32, copy=False))
            init_values.extend(pd.to_datetime(ds["init"].values).strftime("%Y-%m-%d").tolist())
            loaded_years.append(year)
        finally:
            ds.close()
    if not arrays:
        raise RuntimeError(f"No baseline arrays were loaded for {var_name}.")
    return np.concatenate(arrays, axis=0), init_values, loaded_years, missing_years


def load_eval_stack(forecast_dir, years):
    datasets = [open_year(os.path.join(forecast_dir, f"{year}.zarr")) for year in years]
    try:
        combined = xr.concat(datasets, dim="init")
        out = {
            "model_pr": combined["model_pr"].values.astype(np.float32, copy=False),
            "model_t2m": combined["model_t2m"].values.astype(np.float32, copy=False),
            "geos_pr": combined["geos_pr"].values.astype(np.float32, copy=False),
            "geos_t2m": combined["geos_t2m"].values.astype(np.float32, copy=False),
            "obs_pr": combined["obs_pr"].values.astype(np.float32, copy=False),
            "obs_t2m": combined["obs_t2m"].values.astype(np.float32, copy=False),
            "lat": combined["lat"].values.astype(np.float64),
            "lon": combined["lon"].values.astype(np.float64),
            "init": pd.to_datetime(combined["init"].values).strftime("%Y-%m-%d").tolist(),
        }
    finally:
        for ds in datasets:
            ds.close()
    return out


def area_weights(lat):
    weights = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights = weights / np.nanmean(weights)
    return weights[:, None]


def weighted_mean(arr, weights_2d):
    values = np.asarray(arr, dtype=np.float64)
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.nan
    weights = np.broadcast_to(weights_2d, values.shape[-2:])
    weights = np.broadcast_to(weights, values.shape)
    return float(np.nansum(np.where(mask, values * weights, 0.0)) / (np.nansum(np.where(mask, weights, 0.0)) + 1e-12))


def weighted_sum_bool(mask, weights_2d):
    values = np.asarray(mask, dtype=bool)
    weights = np.broadcast_to(weights_2d, values.shape[-2:])
    weights = np.broadcast_to(weights, values.shape)
    return float(np.sum(np.where(values, weights, 0.0)))


def event_probability(ensemble, threshold, upper=True):
    if upper:
        return np.mean(ensemble > threshold[None, None, :, :, :], axis=1)
    return np.mean(ensemble < threshold[None, None, :, :, :], axis=1)


def event_mask(obs, threshold, upper=True):
    if upper:
        return obs > threshold[None, :, :, :]
    return obs < threshold[None, :, :, :]


def exceedance_amount(values, threshold, upper=True):
    if upper:
        return np.maximum(values - threshold, 0.0)
    return np.maximum(threshold - values, 0.0)


def summarize_system(system_name, prob, ensemble, obs, obs_event, threshold, weights, variable, tail, quantile):
    brier = (prob - obs_event.astype(np.float32)) ** 2
    obs_rate = weighted_mean(obs_event.astype(np.float32), weights)
    forecast_rate = weighted_mean(prob, weights)
    if ensemble is not None:
        ens_excess = exceedance_amount(ensemble, threshold[None, None, :, :, :], upper=(tail == "upper"))
        forecast_excess = weighted_mean(np.mean(ens_excess, axis=1), weights)
    else:
        forecast_excess = np.nan
    obs_excess = weighted_mean(exceedance_amount(obs, threshold[None, :, :, :], upper=(tail == "upper")), weights)
    return {
        "variable": variable,
        "tail": tail,
        "quantile": quantile,
        "system": system_name,
        "obs_event_rate": obs_rate,
        "forecast_event_probability_mean": forecast_rate,
        "frequency_bias": forecast_rate / obs_rate if obs_rate > 0 else np.nan,
        "brier_score": weighted_mean(brier, weights),
        "obs_mean_excess": obs_excess,
        "forecast_mean_excess": forecast_excess,
    }


def lead_rows(base_row, prob, ensemble, obs, obs_event, threshold, weights):
    rows = []
    for lead_idx in range(obs.shape[1]):
        row = dict(base_row)
        row["lead"] = lead_idx + 1
        row["obs_event_rate"] = weighted_mean(obs_event[:, lead_idx], weights)
        row["forecast_event_probability_mean"] = weighted_mean(prob[:, lead_idx], weights)
        row["frequency_bias"] = (
            row["forecast_event_probability_mean"] / row["obs_event_rate"]
            if row["obs_event_rate"] > 0
            else np.nan
        )
        row["brier_score"] = weighted_mean((prob[:, lead_idx] - obs_event[:, lead_idx].astype(np.float32)) ** 2, weights)
        row["obs_mean_excess"] = weighted_mean(
            exceedance_amount(
                obs[:, lead_idx],
                threshold[lead_idx][None, :, :],
                upper=(row["tail"] == "upper"),
            ),
            weights,
        )
        if ensemble is not None:
            row["forecast_mean_excess"] = weighted_mean(
                np.mean(exceedance_amount(
                    ensemble[:, :, lead_idx],
                    threshold[lead_idx][None, None, :, :],
                    upper=(row["tail"] == "upper"),
                ), axis=1),
                weights,
            )
        rows.append(row)
    return rows


def reliability_rows(system, prob, obs_event, weights, variable, tail, quantile):
    rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            in_bin = (prob >= lo) & (prob <= hi)
        else:
            in_bin = (prob >= lo) & (prob < hi)
        weight_sum = weighted_sum_bool(in_bin, weights)
        if weight_sum <= 0:
            continue
        rows.append({
            "variable": variable,
            "tail": tail,
            "quantile": quantile,
            "system": system,
            "prob_bin_left": lo,
            "prob_bin_right": hi,
            "weighted_count": weight_sum,
            "mean_forecast_probability": weighted_mean(np.where(in_bin, prob, np.nan), weights),
            "observed_frequency": weighted_mean(np.where(in_bin, obs_event.astype(np.float32), np.nan), weights),
        })
    return rows


def decision_rows(system, prob, obs_event, weights, variable, tail, quantile, thresholds):
    rows = []
    for decision in thresholds:
        yes = prob >= decision
        obs_yes = obs_event.astype(bool)
        hits = weighted_sum_bool(yes & obs_yes, weights)
        false_alarms = weighted_sum_bool(yes & ~obs_yes, weights)
        misses = weighted_sum_bool(~yes & obs_yes, weights)
        rows.append({
            "variable": variable,
            "tail": tail,
            "quantile": quantile,
            "system": system,
            "decision_probability": decision,
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "pod": hits / (hits + misses) if (hits + misses) > 0 else np.nan,
            "far": false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan,
            "csi": hits / (hits + false_alarms + misses) if (hits + false_alarms + misses) > 0 else np.nan,
        })
    return rows


def quantile_rows(eval_data, baseline_thresholds, weights, variable, quantile):
    if variable == "pr":
        model = eval_data["model_pr"]
        geos = eval_data["geos_pr"]
        obs = eval_data["obs_pr"]
    else:
        model = eval_data["model_t2m"]
        geos = eval_data["geos_t2m"]
        obs = eval_data["obs_t2m"]
    model_q = np.nanquantile(model, quantile, axis=(0, 1))
    geos_q = np.nanquantile(geos, quantile, axis=(0, 1))
    obs_q = np.nanquantile(obs, quantile, axis=0)
    baseline = baseline_thresholds[variable][quantile]
    return [
        {
            "variable": variable,
            "quantile": quantile,
            "system": "model",
            "eval_quantile_mean": weighted_mean(model_q, weights),
            "obs_eval_quantile_mean": weighted_mean(obs_q, weights),
            "baseline_threshold_mean": weighted_mean(baseline, weights),
            "system_minus_obs_eval_quantile": weighted_mean(model_q - obs_q, weights),
        },
        {
            "variable": variable,
            "quantile": quantile,
            "system": "geos",
            "eval_quantile_mean": weighted_mean(geos_q, weights),
            "obs_eval_quantile_mean": weighted_mean(obs_q, weights),
            "baseline_threshold_mean": weighted_mean(baseline, weights),
            "system_minus_obs_eval_quantile": weighted_mean(geos_q - obs_q, weights),
        },
    ]


def main():
    args = parse_args()
    quantiles = parse_float_list(args.quantiles)
    decision_thresholds = parse_float_list(args.decision_thresholds)
    baseline_years = list(range(args.baseline_start_year, args.baseline_end_year + 1))
    eval_years = list(range(args.eval_start_year, args.eval_end_year + 1))
    os.makedirs(args.output_dir, exist_ok=True)

    base_pr, base_init, loaded_baseline_years, missing_baseline_years = load_obs_stack(
        args.forecast_dir, baseline_years, "obs_pr", skip_missing=True
    )
    base_t2m, _, _, _ = load_obs_stack(args.forecast_dir, baseline_years, "obs_t2m", skip_missing=True)
    eval_data = load_eval_stack(args.forecast_dir, eval_years)
    weights = area_weights(eval_data["lat"])

    baseline_thresholds = {"pr": {}, "t2m": {}}
    for q in quantiles:
        baseline_thresholds["pr"][q] = np.nanquantile(base_pr, q, axis=0)
        baseline_thresholds["t2m"][q] = np.nanquantile(base_t2m, q, axis=0)

    summary_rows = []
    reliability = []
    decisions = []
    quantile_summary = []

    for variable in ("pr", "t2m"):
        obs = eval_data[f"obs_{variable}"]
        model = eval_data[f"model_{variable}"]
        geos = eval_data[f"geos_{variable}"]
        tail = "upper"
        for q in quantiles:
            threshold = baseline_thresholds[variable][q]
            obs_event = event_mask(obs, threshold, upper=True)
            systems = [
                ("model", event_probability(model, threshold, upper=True), model),
                ("geos", event_probability(geos, threshold, upper=True), geos),
            ]
            for system_name, prob, ensemble in systems:
                total = summarize_system(system_name, prob, ensemble, obs, obs_event, threshold, weights, variable, tail, q)
                total["lead"] = "all"
                summary_rows.append(total)
                summary_rows.extend(lead_rows(total, prob, ensemble, obs, obs_event, threshold, weights))
                reliability.extend(reliability_rows(system_name, prob, obs_event, weights, variable, tail, q))
                decisions.extend(decision_rows(system_name, prob, obs_event, weights, variable, tail, q, decision_thresholds))
            quantile_summary.extend(quantile_rows(eval_data, baseline_thresholds, weights, variable, q))

    summary_path = os.path.join(args.output_dir, "extreme_tail_summary.csv")
    reliability_path = os.path.join(args.output_dir, "extreme_tail_reliability.csv")
    decision_path = os.path.join(args.output_dir, "extreme_tail_decision_scores.csv")
    quantile_path = os.path.join(args.output_dir, "extreme_tail_quantile_comparison.csv")
    meta_path = os.path.join(args.output_dir, "extreme_tail_metadata.json")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(reliability).to_csv(reliability_path, index=False)
    pd.DataFrame(decisions).to_csv(decision_path, index=False)
    pd.DataFrame(quantile_summary).to_csv(quantile_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "forecast_dir": args.forecast_dir,
                "baseline_years_requested": baseline_years,
                "baseline_years_loaded": loaded_baseline_years,
                "baseline_years_missing": sorted(set(missing_baseline_years)),
                "eval_years": eval_years,
                "baseline_init_count": len(base_init),
                "eval_init_count": len(eval_data["init"]),
                "quantiles": quantiles,
                "decision_thresholds": decision_thresholds,
                "notes": "Thresholds are gridpoint/lead-specific observed climatological quantiles from baseline years.",
            },
            f,
            indent=2,
        )

    print("\nExtreme-tail analysis complete")
    print(f"  Forecast dir : {args.forecast_dir}")
    print(
        f"  Baseline     : {args.baseline_start_year}-{args.baseline_end_year} "
        f"({len(base_init)} init dates, loaded years={loaded_baseline_years})"
    )
    if missing_baseline_years:
        print(f"  Missing base : {sorted(set(missing_baseline_years))}")
    print(f"  Evaluation   : {args.eval_start_year}-{args.eval_end_year} ({len(eval_data['init'])} init dates)")
    print(f"  Summary      : {summary_path}")
    print(f"  Reliability  : {reliability_path}")
    print(f"  Decisions    : {decision_path}")
    print(f"  Quantiles    : {quantile_path}")


if __name__ == "__main__":
    main()
