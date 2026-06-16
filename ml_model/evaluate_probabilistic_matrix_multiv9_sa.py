#!/usr/bin/env python3
"""
Additional probabilistic forecast diagnostics for South Asia v9 generated Zarrs.

This complements evaluate_junjul_raw_matrix_multiv9_sa.py with calibration and
probabilistic event-skill matrices: interval coverage, rank histograms, Brier
skill, ROC AUC, and precision-recall AUC.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from evaluate_junjul_raw_matrix_multiv9_sa import (
    DEFAULT_FORECAST_DIR,
    area_weights,
    data_for,
    load_eval_dataset,
    obs_for,
    parse_float_list,
    parse_int_list,
    parse_int_set,
    select_lead,
    tails_for_variable,
    raw_threshold,
    event_mask,
    event_probability,
    weighted_mean,
)
from dataset_flow_multi import S2SHybridDataset


DEFAULT_OUTPUT_DIR = (
    "ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/"
    "probabilistic_matrix_junjul_testmode_2021_2024_obsclim2001_2020"
)
DEFAULT_CONFIG = "ml_model/config_flow_multiv9.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v9 SA probabilistic forecast diagnostics.")
    parser.add_argument("--forecast_dir", type=str, default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--months", type=str, default="6,7")
    parser.add_argument("--skip_years", type=str, default="")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--obs_clim_data_dir", type=str, default=None)
    parser.add_argument("--obs_clim_start_year", type=int, default=2001)
    parser.add_argument("--obs_clim_end_year", type=int, default=2020)
    parser.add_argument("--obs_clim_skip_years", type=str, default="")
    parser.add_argument("--extreme_quantiles", type=str, default="0.90,0.95")
    parser.add_argument("--interval_levels", type=str, default="0.50,0.80,0.90,0.95")
    parser.add_argument("--rank_bins", type=int, default=10)
    return parser.parse_args()


def month_name(month):
    return {6: "Jun", 7: "Jul"}.get(int(month), str(month))


def broadcast_weights(values, weights_2d):
    spatial = np.broadcast_to(weights_2d, values.shape[-2:])
    return np.broadcast_to(spatial, values.shape)


def weighted_binary_sum(mask, weights_2d):
    arr = np.asarray(mask, dtype=bool)
    weights = broadcast_weights(arr, weights_2d)
    return float(np.sum(np.where(arr, weights, 0.0)))


def finite_flat(values, weights_2d):
    arr = np.asarray(values, dtype=np.float64)
    weights = broadcast_weights(arr, weights_2d).astype(np.float64, copy=False)
    mask = np.isfinite(arr) & np.isfinite(weights) & (weights > 0)
    return arr[mask], weights[mask]


def finite_pair_flat(prob, event, weights_2d):
    p = np.asarray(prob, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    weights = broadcast_weights(p, weights_2d).astype(np.float64, copy=False)
    mask = np.isfinite(p) & np.isfinite(weights) & (weights > 0)
    return p[mask], e[mask], weights[mask]


def auc_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    dx = np.diff(x)
    return float(np.sum(dx * (y[:-1] + y[1:]) * 0.5))


def weighted_roc_auc(prob, event, weights_2d):
    p, e, w = finite_pair_flat(prob, event, weights_2d)
    if p.size == 0 or not np.any(e) or np.all(e):
        return np.nan
    order = np.argsort(-p)
    e = e[order]
    w = w[order]
    pos = float(np.sum(w[e]))
    neg = float(np.sum(w[~e]))
    if pos <= 0 or neg <= 0:
        return np.nan
    tpr = np.concatenate([[0.0], np.cumsum(np.where(e, w, 0.0)) / pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(np.where(~e, w, 0.0)) / neg, [1.0]])
    return auc_trapezoid(tpr, fpr)


def weighted_average_precision(prob, event, weights_2d):
    p, e, w = finite_pair_flat(prob, event, weights_2d)
    if p.size == 0 or not np.any(e):
        return np.nan
    order = np.argsort(-p)
    e = e[order]
    w = w[order]
    tp = np.cumsum(np.where(e, w, 0.0))
    fp = np.cumsum(np.where(~e, w, 0.0))
    pos = float(tp[-1])
    if pos <= 0:
        return np.nan
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum(np.where(e, precision * (recall - recall_prev), 0.0)))


def load_config(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_obs_clim_data_dir(args, config):
    return (
        args.obs_clim_data_dir
        or os.environ.get("DATA_DIR_OVERRIDE")
        or config.get("data_dir")
        or "dataprocess"
    )


def load_obs_climatology(args, config, months, variables, expected_shape):
    """Load independent raw observed PR/T2M climatology from the dataset files."""
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


def interval_rows(scope, year, month, variable, system, ensemble, obs, weights, interval_levels, lead_idx=None):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    rows = []
    for level in interval_levels:
        lower_q = (1.0 - float(level)) / 2.0
        upper_q = 1.0 - lower_q
        lower = np.nanquantile(ens, lower_q, axis=1)
        upper = np.nanquantile(ens, upper_q, axis=1)
        covered = (target >= lower) & (target <= upper)
        rows.append({
            "scope": scope,
            "year": year,
            "month": int(month),
            "month_name": month_name(month),
            "variable": variable,
            "system": system,
            "lead": "all" if lead_idx is None else int(lead_idx + 1),
            "interval": float(level),
            "coverage": weighted_mean(covered.astype(np.float32), weights),
            "mean_width": weighted_mean(upper - lower, weights),
            "lower_miss_rate": weighted_mean((target < lower).astype(np.float32), weights),
            "upper_miss_rate": weighted_mean((target > upper).astype(np.float32), weights),
        })
    return rows


def rank_histogram_rows(scope, year, month, variable, system, ensemble, obs, weights, rank_bins, lead_idx=None):
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    rank_fraction = np.nanmean(ens <= target[:, None], axis=1)
    finite = np.isfinite(rank_fraction)
    weights_full = broadcast_weights(rank_fraction, weights)
    total_weight = float(np.sum(np.where(finite, weights_full, 0.0)))
    rows = []
    bins = np.linspace(0.0, 1.0, rank_bins + 1)
    for bin_id, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        in_bin = (rank_fraction >= left) & (rank_fraction <= right if bin_id == rank_bins - 1 else rank_fraction < right)
        count = float(np.sum(np.where(finite & in_bin, weights_full, 0.0)))
        rows.append({
            "scope": scope,
            "year": year,
            "month": int(month),
            "month_name": month_name(month),
            "variable": variable,
            "system": system,
            "lead": "all" if lead_idx is None else int(lead_idx + 1),
            "bin": bin_id,
            "bin_left": float(left),
            "bin_right": float(right),
            "weighted_count": count,
            "fraction": count / total_weight if total_weight > 0 else np.nan,
            "ideal_fraction": 1.0 / rank_bins,
        })
    return rows


def event_rows(
    scope,
    year,
    month,
    variable,
    system,
    ensemble,
    obs,
    weights,
    thresholds,
    climatology_event_rates,
    threshold_area_means,
    quantiles,
    lead_idx=None,
):
    rows = []
    ens = select_lead(ensemble, lead_idx)
    target = select_lead(obs, lead_idx)
    lead_key = "all" if lead_idx is None else int(lead_idx + 1)
    for quantile in quantiles:
        for tail in tails_for_variable(variable):
            base_key = (month, variable, float(quantile), tail)
            threshold = thresholds[base_key]
            if lead_idx is not None:
                threshold = threshold[lead_idx:lead_idx + 1]
            prob = event_probability(ens, threshold, tail)
            event = event_mask(target, threshold, tail)
            event_rate = weighted_mean(event.astype(np.float32), weights)
            brier = weighted_mean((prob - event.astype(np.float32)) ** 2, weights)
            forecast_prob_mean = weighted_mean(prob, weights)
            clim_event_rate = climatology_event_rates[base_key + (lead_key,)]
            clim_brier = weighted_mean((clim_event_rate - event.astype(np.float32)) ** 2, weights)
            rows.append({
                "scope": scope,
                "year": year,
                "month": int(month),
                "month_name": month_name(month),
                "variable": variable,
                "system": system,
                "lead": "all" if lead_idx is None else int(lead_idx + 1),
                "tail": tail,
                "quantile": float(quantile),
                "threshold_area_mean": threshold_area_means[base_key + (lead_key,)],
                "obs_event_rate": event_rate,
                "climatology_event_rate": clim_event_rate,
                "forecast_event_probability_mean": forecast_prob_mean,
                "frequency_bias": forecast_prob_mean / event_rate if event_rate > 0 else np.nan,
                "frequency_bias_vs_climatology": (
                    forecast_prob_mean / clim_event_rate if clim_event_rate > 0 else np.nan
                ),
                "brier_score": brier,
                "brier_skill_vs_climatology": 1.0 - (brier / clim_brier) if clim_brier > 0 else np.nan,
                "roc_auc": weighted_roc_auc(prob, event, weights),
                "pr_auc": weighted_average_precision(prob, event, weights),
            })
    return rows


def make_thresholds(obs_clim, months, variables, quantiles):
    thresholds = {}
    for month in months:
        for variable in variables:
            obs = obs_clim[(month, variable)]
            for quantile in quantiles:
                for tail in tails_for_variable(variable):
                    thresholds[(month, variable, float(quantile), tail)] = raw_threshold(obs, quantile, tail)
    return thresholds


def make_climatology_event_rates(obs_clim, thresholds, months, variables, quantiles, weights):
    rates = {}
    threshold_area_means = {}
    for month in months:
        for variable in variables:
            obs = obs_clim[(month, variable)]
            for quantile in quantiles:
                for tail in tails_for_variable(variable):
                    base_key = (month, variable, float(quantile), tail)
                    threshold = thresholds[base_key]
                    event = event_mask(obs, threshold, tail)
                    rates[base_key + ("all",)] = weighted_mean(event.astype(np.float32), weights)
                    threshold_area_means[base_key + ("all",)] = weighted_mean(threshold, weights)
                    for lead_idx in range(4):
                        lead_key = int(lead_idx + 1)
                        rates[base_key + (lead_key,)] = weighted_mean(
                            event[:, lead_idx:lead_idx + 1].astype(np.float32),
                            weights,
                        )
                        threshold_area_means[base_key + (lead_key,)] = weighted_mean(
                            threshold[lead_idx:lead_idx + 1],
                            weights,
                        )
    return rates, threshold_area_means


def subset_by(ds, init_indices):
    return ds.isel(init=np.asarray(init_indices, dtype=np.int64))


def add_scope_rows(
    ds,
    scope,
    year_label,
    months,
    variables,
    systems,
    weights,
    thresholds,
    climatology_event_rates,
    threshold_area_means,
    quantiles,
    interval_levels,
    rank_bins,
):
    interval = []
    ranks = []
    events = []
    init_dates = pd.to_datetime(ds["init"].values)
    for month in months:
        month_idx = np.where(init_dates.month == month)[0]
        if month_idx.size == 0:
            continue
        month_ds = subset_by(ds, month_idx)
        for variable in variables:
            obs = obs_for(month_ds, variable)
            for system in systems:
                ensemble = data_for(month_ds, variable, system)
                for lead_idx in [None, 0, 1, 2, 3]:
                    interval.extend(
                        interval_rows(
                            scope, year_label, month, variable, system, ensemble, obs,
                            weights, interval_levels, lead_idx=lead_idx,
                        )
                    )
                    ranks.extend(
                        rank_histogram_rows(
                            scope, year_label, month, variable, system, ensemble, obs,
                            weights, rank_bins, lead_idx=lead_idx,
                        )
                    )
                    events.extend(
                        event_rows(
                            scope, year_label, month, variable, system, ensemble, obs,
                            weights, thresholds, climatology_event_rates,
                            threshold_area_means, quantiles, lead_idx=lead_idx,
                        )
                    )
    return interval, ranks, events


def print_table(df, title, cols, sort_by):
    print("\n" + title)
    if df.empty:
        print("  <empty>")
        return
    out = df.sort_values(sort_by)[cols].copy()
    float_cols = out.select_dtypes(include=[np.floating]).columns
    out[float_cols] = out[float_cols].round(4)
    print(out.to_string(index=False))


def make_plots(interval_df, event_df, rank_df, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_interval = interval_df[(interval_df["scope"] == "all_years") & (interval_df["lead"].astype(str) == "all")]
    for variable in ["pr", "t2m"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, month in zip(axes, [6, 7]):
            subset = total_interval[(total_interval["variable"] == variable) & (total_interval["month"] == month)]
            for system in ["ml", "geos"]:
                s = subset[subset["system"] == system].sort_values("interval")
                ax.plot(s["interval"], s["coverage"], marker="o", label=system.upper())
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_title(f"{month_name(month)} {variable.upper()} interval coverage")
            ax.set_xlabel("Nominal interval")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Observed coverage")
        axes[0].legend()
        path = os.path.join(plot_dir, f"prob_interval_coverage_{variable}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    total_event = event_df[
        (event_df["scope"] == "all_years")
        & (event_df["lead"].astype(str) == "all")
        & (np.isclose(event_df["quantile"], 0.95))
    ].copy()
    if not total_event.empty:
        total_event["label"] = total_event["month_name"] + " " + total_event["variable"].str.upper() + " " + total_event["tail"]
        labels = list(total_event[total_event["system"] == "ml"]["label"])
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, metric, title in [
            (axes[0], "brier_skill_vs_climatology", "Brier skill vs climatology"),
            (axes[1], "roc_auc", "ROC AUC"),
        ]:
            width = 0.36
            for offset, system in [(-width / 2, "ml"), (width / 2, "geos")]:
                s = total_event[total_event["system"] == system].set_index("label").reindex(labels)
                ax.bar(x + offset, s[metric].values, width=width, label=system.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend()
        path = os.path.join(plot_dir, "prob_event_skill_q95.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    total_rank = rank_df[(rank_df["scope"] == "all_years") & (rank_df["lead"].astype(str) == "all")]
    for variable in ["pr", "t2m"]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        for ax, month, system in zip(axes.ravel(), [6, 6, 7, 7], ["ml", "geos", "ml", "geos"]):
            s = total_rank[
                (total_rank["variable"] == variable)
                & (total_rank["month"] == month)
                & (total_rank["system"] == system)
            ].sort_values("bin")
            ax.bar(s["bin"], s["fraction"], color="tab:blue" if system == "ml" else "tab:orange")
            ax.axhline(float(s["ideal_fraction"].iloc[0]) if not s.empty else 0.1, color="k", linestyle="--", linewidth=1)
            ax.set_title(f"{month_name(month)} {variable.upper()} {system.upper()} rank")
            ax.grid(axis="y", alpha=0.3)
        path = os.path.join(plot_dir, f"prob_rank_histogram_{variable}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    months = parse_int_list(args.months)
    quantiles = parse_float_list(args.extreme_quantiles)
    interval_levels = parse_float_list(args.interval_levels)
    ds, loaded_years, missing_years, skip_years = load_eval_dataset(args, months)
    weights = area_weights(ds["lat"].values)
    print("\nSouth Asia v9 probabilistic matrix evaluation")
    print(f"  Forecast dir : {args.forecast_dir}")
    print(f"  Years        : {args.start_year}-{args.end_year}")
    print("  Target mode  : raw observed PR/T2M, no anomaly transform")
    print(f"  Months       : {months}")
    print(f"  Config       : {args.config}")

    variables = ["pr", "t2m"]
    systems = ["ml", "geos"]
    expected_shape = obs_for(ds, "pr").shape[1:]
    obs_clim, obs_clim_metadata = load_obs_climatology(
        args, config, months, variables, expected_shape
    )
    thresholds = make_thresholds(obs_clim, months, variables, quantiles)
    climatology_event_rates, threshold_area_means = make_climatology_event_rates(
        obs_clim, thresholds, months, variables, quantiles, weights
    )

    interval_rows_all = []
    rank_rows_all = []
    event_rows_all = []

    interval, ranks, events = add_scope_rows(
        ds, "all_years", "all", months, variables, systems, weights,
        thresholds, climatology_event_rates, threshold_area_means,
        quantiles, interval_levels, args.rank_bins,
    )
    interval_rows_all.extend(interval)
    rank_rows_all.extend(ranks)
    event_rows_all.extend(events)

    init_years = pd.to_datetime(ds["init"].values).year
    for year in sorted(set(init_years)):
        idx = np.where(init_years == year)[0]
        interval, ranks, events = add_scope_rows(
            subset_by(ds, idx), "year", int(year), months, variables, systems, weights,
            thresholds, climatology_event_rates, threshold_area_means,
            quantiles, interval_levels, args.rank_bins,
        )
        interval_rows_all.extend(interval)
        rank_rows_all.extend(ranks)
        event_rows_all.extend(events)

    interval_df = pd.DataFrame(interval_rows_all)
    rank_df = pd.DataFrame(rank_rows_all)
    event_df = pd.DataFrame(event_rows_all)

    paths = {
        "interval": os.path.join(args.output_dir, "prob_interval_coverage_matrix.csv"),
        "rank_histogram": os.path.join(args.output_dir, "prob_rank_histogram_matrix.csv"),
        "event": os.path.join(args.output_dir, "prob_event_skill_matrix.csv"),
        "metadata": os.path.join(args.output_dir, "probabilistic_matrix_metadata.json"),
    }
    interval_df.to_csv(paths["interval"], index=False)
    rank_df.to_csv(paths["rank_histogram"], index=False)
    event_df.to_csv(paths["event"], index=False)

    make_plots(interval_df, event_df, rank_df, args.output_dir)

    with open(paths["metadata"], "w") as f:
        json.dump(
            {
                "forecast_dir": args.forecast_dir,
                "output_dir": args.output_dir,
                "loaded_years": loaded_years,
                "missing_years": missing_years,
                "skip_years": skip_years,
                "months": months,
                "extreme_quantiles": quantiles,
                "interval_levels": interval_levels,
                "rank_bins": args.rank_bins,
                "config": args.config,
                "obs_climatology": obs_clim_metadata,
            },
            f,
            indent=2,
        )

    lead_all_interval = interval_df[
        (interval_df["scope"] == "all_years")
        & (interval_df["lead"].astype(str) == "all")
        & (np.isclose(interval_df["interval"], 0.9))
    ]
    print_table(
        lead_all_interval,
        "90% interval coverage (all years, all leads; ideal is 0.90)",
        ["month", "variable", "system", "coverage", "mean_width", "lower_miss_rate", "upper_miss_rate"],
        ["month", "variable", "system"],
    )

    q95_event = event_df[
        (event_df["scope"] == "all_years")
        & (event_df["lead"].astype(str) == "all")
        & (np.isclose(event_df["quantile"], 0.95))
    ]
    print_table(
        q95_event,
        "q=0.95 event probabilistic skill (all years, all leads)",
        [
            "month", "variable", "tail", "system", "threshold_area_mean",
            "obs_event_rate", "climatology_event_rate",
            "forecast_event_probability_mean", "frequency_bias",
            "frequency_bias_vs_climatology",
            "brier_score", "brier_skill_vs_climatology", "roc_auc", "pr_auc",
        ],
        ["month", "variable", "tail", "system"],
    )

    print("\nProbabilistic matrix evaluation complete")
    for label, path in paths.items():
        print(f"  {label:14s}: {path}")


if __name__ == "__main__":
    main()
