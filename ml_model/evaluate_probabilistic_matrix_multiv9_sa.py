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
    DEFAULT_OBS_CLIM_END_YEAR,
    DEFAULT_OBS_CLIM_START_YEAR,
    area_weights,
    data_for,
    load_eval_dataset,
    obs_for,
    parse_float_list,
    parse_int_list,
    select_lead,
    tails_for_variable,
    raw_threshold,
    event_mask,
    event_probability,
    build_self_thresholds,
    load_obs_climatology,
    self_threshold_extreme_metrics,
    weighted_mean,
)


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
    parser.add_argument("--obs_clim_start_year", type=int, default=DEFAULT_OBS_CLIM_START_YEAR)
    parser.add_argument("--obs_clim_end_year", type=int, default=DEFAULT_OBS_CLIM_END_YEAR)
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
    climatology_event_probability_maps,
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
            clim_prob_map = climatology_event_probability_maps[base_key + (lead_key,)]
            clim_brier = weighted_mean((clim_prob_map - event.astype(np.float32)) ** 2, weights)
            scalar_clim_brier = weighted_mean((clim_event_rate - event.astype(np.float32)) ** 2, weights)
            eval_rate_brier = weighted_mean((event_rate - event.astype(np.float32)) ** 2, weights)
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
                "threshold_source": "obs_climatology",
                "threshold_area_mean": threshold_area_means[base_key + (lead_key,)],
                "obs_event_rate": event_rate,
                "climatology_event_rate": clim_event_rate,
                "forecast_event_probability_mean": forecast_prob_mean,
                "forecast_probability_bias_vs_obs": forecast_prob_mean - event_rate,
                "obs_minus_climatology_event_rate": event_rate - clim_event_rate,
                "frequency_bias": forecast_prob_mean / event_rate if event_rate > 0 else np.nan,
                "frequency_bias_vs_climatology": (
                    forecast_prob_mean / clim_event_rate if clim_event_rate > 0 else np.nan
                ),
                "brier_score": brier,
                "climatology_brier_score": clim_brier,
                "scalar_climatology_brier_score": scalar_clim_brier,
                "eval_event_rate_brier_score": eval_rate_brier,
                "brier_skill_vs_climatology": 1.0 - (brier / clim_brier) if clim_brier > 0 else np.nan,
                "brier_skill_vs_scalar_climatology": (
                    1.0 - (brier / scalar_clim_brier) if scalar_clim_brier > 0 else np.nan
                ),
                "brier_skill_vs_eval_event_rate": (
                    1.0 - (brier / eval_rate_brier) if eval_rate_brier > 0 else np.nan
                ),
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
    probability_maps = {}
    threshold_area_means = {}
    for month in months:
        for variable in variables:
            obs = obs_clim[(month, variable)]
            for quantile in quantiles:
                for tail in tails_for_variable(variable):
                    base_key = (month, variable, float(quantile), tail)
                    threshold = thresholds[base_key]
                    event = event_mask(obs, threshold, tail)
                    event_probability_map = np.nanmean(event.astype(np.float32), axis=0)
                    rates[base_key + ("all",)] = weighted_mean(event.astype(np.float32), weights)
                    probability_maps[base_key + ("all",)] = event_probability_map
                    threshold_area_means[base_key + ("all",)] = weighted_mean(threshold, weights)
                    for lead_idx in range(4):
                        lead_key = int(lead_idx + 1)
                        rates[base_key + (lead_key,)] = weighted_mean(
                            event[:, lead_idx:lead_idx + 1].astype(np.float32),
                            weights,
                        )
                        probability_maps[base_key + (lead_key,)] = event_probability_map[lead_idx:lead_idx + 1]
                        threshold_area_means[base_key + (lead_key,)] = weighted_mean(
                            threshold[lead_idx:lead_idx + 1],
                            weights,
                        )
    return rates, probability_maps, threshold_area_means


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
    climatology_event_probability_maps,
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
                            climatology_event_probability_maps, threshold_area_means,
                            quantiles, lead_idx=lead_idx,
                        )
                    )
    return interval, ranks, events


def add_self_threshold_scope_rows(ds, scope, year_label, months, variables, systems, weights, self_thresholds_by_month, quantiles):
    rows = []
    init_dates = pd.to_datetime(ds["init"].values)
    for month in months:
        month_idx = np.where(init_dates.month == month)[0]
        if month_idx.size == 0:
            continue
        month_ds = subset_by(ds, month_idx)
        self_thresholds = self_thresholds_by_month[int(month)]
        for variable in variables:
            obs = obs_for(month_ds, variable)
            obs_thresholds = self_thresholds["obs"][variable]
            for system in systems:
                ensemble = data_for(month_ds, variable, system)
                forecast_thresholds = self_thresholds[system][variable]
                for quantile in quantiles:
                    for tail in tails_for_variable(variable):
                        obs_threshold = obs_thresholds[(tail, quantile)]
                        forecast_threshold = forecast_thresholds[(tail, quantile)]
                        rows.append(
                            self_threshold_extreme_metrics(
                                scope,
                                year_label,
                                "all",
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
                            rows.append(
                                self_threshold_extreme_metrics(
                                    scope,
                                    year_label,
                                    "all",
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
    return rows


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

        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        width = 0.24
        for offset, col, label, color in [
            (-width, "obs_event_rate", "Observed event rate", "tab:green"),
            (0.0, "climatology_event_rate", "2001-2020 obs climatology", "tab:gray"),
            (width, "forecast_event_probability_mean", "Forecast probability", "tab:blue"),
        ]:
            s = total_event[total_event["system"] == "ml"].set_index("label").reindex(labels)
            axes[0].bar(x + offset, s[col].values, width=width, label=label, color=color)
        axes[0].set_title("ML event-rate diagnostics")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=25, ha="right")
        axes[0].set_ylabel("Probability")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend()

        for offset, system in [(-width / 2, "ml"), (width / 2, "geos")]:
            s = total_event[total_event["system"] == system].set_index("label").reindex(labels)
            axes[1].bar(x + offset, s["forecast_probability_bias_vs_obs"].values, width=width, label=system.upper())
        axes[1].axhline(0.0, color="k", linewidth=1)
        axes[1].set_title("Forecast probability bias vs observed event rate")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=25, ha="right")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend()
        path = os.path.join(plot_dir, "prob_event_rate_diagnostics_q95.png")
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


def make_self_threshold_plots(self_event_df, output_dir):
    if self_event_df.empty:
        return
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = self_event_df[
        (self_event_df["scope"] == "all_years")
        & (self_event_df["lead"].astype(str) == "all")
        & (np.isclose(self_event_df["quantile"], 0.95))
    ].copy()
    if total.empty:
        return
    total["label"] = total["month"].map(month_name) + " " + total["variable"].str.upper() + " " + total["tail"]
    labels = list(total[total["system"] == "ml"]["label"])
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for ax, metric, title in [
        (axes[0], "brier_score", "Relative-tail self-threshold Brier score"),
        (axes[1], "frequency_bias", "Relative-tail self-threshold frequency bias"),
    ]:
        for offset, system in [(-width / 2, "ml"), (width / 2, "geos")]:
            s = total[total["system"] == system].set_index("label").reindex(labels)
            ax.bar(x + offset, s[metric].values, width=width, label=system.upper())
        if metric == "frequency_bias":
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        "Diagnostic only: each system uses its own evaluation-period threshold",
        fontsize=11,
    )
    path = os.path.join(plot_dir, "prob_self_threshold_event_q95.png")
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
    climatology_event_rates, climatology_event_probability_maps, threshold_area_means = make_climatology_event_rates(
        obs_clim, thresholds, months, variables, quantiles, weights
    )
    init_dates = pd.to_datetime(ds["init"].values)
    self_thresholds_by_month = {}
    for month in months:
        month_idx = np.where(init_dates.month == int(month))[0]
        if month_idx.size > 0:
            self_thresholds_by_month[int(month)] = build_self_thresholds(ds.isel(init=month_idx), quantiles)

    interval_rows_all = []
    rank_rows_all = []
    event_rows_all = []
    self_event_rows_all = []

    interval, ranks, events = add_scope_rows(
        ds, "all_years", "all", months, variables, systems, weights,
        thresholds, climatology_event_rates, climatology_event_probability_maps, threshold_area_means,
        quantiles, interval_levels, args.rank_bins,
    )
    interval_rows_all.extend(interval)
    rank_rows_all.extend(ranks)
    event_rows_all.extend(events)
    self_event_rows_all.extend(
        add_self_threshold_scope_rows(
            ds, "all_years", "all", months, variables, systems, weights, self_thresholds_by_month, quantiles
        )
    )

    init_years = init_dates.year
    for year in sorted(set(init_years)):
        idx = np.where(init_years == year)[0]
        interval, ranks, events = add_scope_rows(
            subset_by(ds, idx), "year", int(year), months, variables, systems, weights,
            thresholds, climatology_event_rates, climatology_event_probability_maps, threshold_area_means,
            quantiles, interval_levels, args.rank_bins,
        )
        interval_rows_all.extend(interval)
        rank_rows_all.extend(ranks)
        event_rows_all.extend(events)
        self_event_rows_all.extend(
            add_self_threshold_scope_rows(
                subset_by(ds, idx),
                "year",
                int(year),
                months,
                variables,
                systems,
                weights,
                self_thresholds_by_month,
                quantiles,
            )
        )

    interval_df = pd.DataFrame(interval_rows_all)
    rank_df = pd.DataFrame(rank_rows_all)
    event_df = pd.DataFrame(event_rows_all)
    self_event_df = pd.DataFrame(self_event_rows_all)

    paths = {
        "interval": os.path.join(args.output_dir, "prob_interval_coverage_matrix.csv"),
        "rank_histogram": os.path.join(args.output_dir, "prob_rank_histogram_matrix.csv"),
        "event": os.path.join(args.output_dir, "prob_event_skill_matrix.csv"),
        "self_event": os.path.join(args.output_dir, "prob_self_threshold_event_matrix.csv"),
        "metadata": os.path.join(args.output_dir, "probabilistic_matrix_metadata.json"),
    }
    interval_df.to_csv(paths["interval"], index=False)
    rank_df.to_csv(paths["rank_histogram"], index=False)
    event_df.to_csv(paths["event"], index=False)
    self_event_df.to_csv(paths["self_event"], index=False)

    make_plots(interval_df, event_df, rank_df, args.output_dir)
    make_self_threshold_plots(self_event_df, args.output_dir)

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
                "physical_threshold_definition": (
                    "Separate init-month, lead, and gridpoint quantiles from raw observed "
                    "2001-2020 targets. The same observed threshold is applied to OBS truth, "
                    "ML members, and GEOS members."
                ),
                "self_threshold_definition": (
                    "Separate init-month, lead, and gridpoint quantiles from the evaluation "
                    "period. OBS uses observed values; ML and GEOS each use their own pooled "
                    "init-by-member distribution."
                ),
                "self_threshold_notes": (
                    "prob_self_threshold_event_matrix.csv is diagnostic only: forecast probabilities use "
                    "each system's own evaluation-period threshold, while event truth uses the OBS evaluation-period threshold."
                ),
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
            "month", "variable", "tail", "system", "threshold_source", "threshold_area_mean",
            "obs_event_rate", "climatology_event_rate",
            "forecast_event_probability_mean", "frequency_bias",
            "frequency_bias_vs_climatology",
            "forecast_probability_bias_vs_obs", "obs_minus_climatology_event_rate",
            "brier_score", "climatology_brier_score", "scalar_climatology_brier_score",
            "brier_skill_vs_climatology", "brier_skill_vs_scalar_climatology",
            "brier_skill_vs_eval_event_rate", "roc_auc", "pr_auc",
        ],
        ["month", "variable", "tail", "system"],
    )

    q95_self = self_event_df[
        (self_event_df["scope"] == "all_years")
        & (self_event_df["lead"].astype(str) == "all")
        & (np.isclose(self_event_df["quantile"], 0.95))
    ]
    print_table(
        q95_self,
        "q=0.95 relative-tail self-threshold skill (diagnostic only)",
        [
            "month", "variable", "tail", "system",
            "obs_threshold_area_mean", "forecast_threshold_area_mean",
            "obs_event_rate", "forecast_event_probability_mean",
            "forecast_probability_bias_vs_obs", "frequency_bias",
            "brier_score",
        ],
        ["month", "variable", "tail", "system"],
    )

    print("\nProbabilistic matrix evaluation complete")
    for label, path in paths.items():
        print(f"  {label:14s}: {path}")


if __name__ == "__main__":
    main()
