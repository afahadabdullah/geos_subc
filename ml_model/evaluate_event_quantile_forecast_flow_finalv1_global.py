#!/usr/bin/env python3
"""
Probabilistic quantile evaluation for historical extreme-event forecasts.

This is a companion to evaluate_event_catalog_flow_finalv1_global.py.  It uses
the same event catalog and event-window sample selection, but reframes each
case as a distributional forecast:

  * regional ensemble distributions for area mean, top-tail intensity,
    spatial upper quantile, and event-area fraction;
  * observed percentile rank under ML and GEOS;
  * P(forecast >= observed), P(forecast >= obs-climatology threshold);
  * quantile/pinball losses for upper-tail quantiles;
  * compact distribution and fixed-init progression plots.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from evaluate_event_catalog_flow_finalv1_global import (
    DEFAULT_TAIL_FRACTION,
    VARIABLES,
    bbox_mask,
    choose_event_samples,
    fixed_init_progression_samples,
    find_candidate_samples,
    load_event_catalog,
    load_land_mask,
    load_thresholds_from_file,
    normalize_catalog,
    open_forecast_grid,
    parse_bbox,
    parse_list,
    region_weights,
    regional_member_tail_values,
    regional_member_values,
    select_grouped_map,
    weighted_mean,
    weighted_top_mean,
)


INTENSITY_METRICS = {"area_mean", "top_tail_intensity", "spatial_q95_intensity"}
METRIC_LABELS = {
    "area_mean": "regional area mean",
    "top_tail_intensity": "top-tail intensity",
    "spatial_q95_intensity": "spatial q95 intensity",
    "event_area_fraction": "event-area fraction",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate event forecasts as probabilistic quantile distributions."
    )
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
    )
    parser.add_argument(
        "--threshold_file",
        type=str,
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "matrix_eval_global_2021_2023_land_obsclim_chunked/event_thresholds_and_frequencies.nc"
        ),
    )
    parser.add_argument("--land_mask_file", type=str, default="ml_model/land_ocean_mask_v6.pt")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ml_output_flow_finalv1_global_noisectx_t2mres/event_quantile_eval_global_2021_2023",
    )
    parser.add_argument("--event_catalog", type=str, default="default", help="default or path to CSV/JSON catalog.")
    parser.add_argument("--regions", type=str, default="all")
    parser.add_argument("--variables", type=str, default="pr,t2m")
    parser.add_argument("--leads", type=str, default="3,4")
    parser.add_argument("--progression_leads", type=str, default="1,2,3,4")
    parser.add_argument("--regional_weighting", choices=("uniform", "area"), default="uniform")
    parser.add_argument("--tail_fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument(
        "--spatial_quantile",
        type=float,
        default=0.95,
        help="Within-region spatial quantile computed for each ensemble member.",
    )
    parser.add_argument(
        "--ensemble_quantiles",
        type=str,
        default="0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99",
        help="Across-member forecast quantiles to save.",
    )
    parser.add_argument(
        "--loss_quantiles",
        type=str,
        default="0.90,0.95,0.99",
        help="Across-member quantiles evaluated with pinball loss.",
    )
    parser.add_argument(
        "--event_area_fraction_threshold",
        type=float,
        default=0.10,
        help="Event-area-fraction threshold used for P(event-area fraction >= threshold).",
    )
    parser.add_argument("--extreme_quantile_pr", type=float, default=0.95)
    parser.add_argument("--extreme_quantile_t2m", type=float, default=0.95)
    parser.add_argument("--pr_min_threshold", type=float, default=5.0)
    parser.add_argument("--timeseries_window_days", type=int, default=42)
    parser.add_argument("--event_tolerance_days", type=int, default=10)
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2023)
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--write_member_values", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def qcol(q):
    return f"q{int(round(float(q) * 100)):02d}"


def safe_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.nanquantile(values, float(q)))


def weighted_quantile(field, weights, quantile):
    field = np.asarray(field, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    values = field[finite]
    w = weights[finite]
    order = np.argsort(values)
    values = values[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 0:
        return np.nan
    return float(np.interp(float(quantile) * total, cumulative, values))


def regional_member_spatial_quantile_values(ensemble, weights, quantile):
    values = []
    for member in np.asarray(ensemble):
        values.append(weighted_quantile(member, weights, quantile))
    return np.asarray(values, dtype=np.float64)


def regional_member_event_area_fraction_values(ensemble, threshold, weights):
    values = []
    threshold = np.asarray(threshold, dtype=np.float32)
    for member in np.asarray(ensemble, dtype=np.float32):
        event = (member >= threshold).astype(np.float32)
        values.append(weighted_mean(event, weights))
    return np.asarray(values, dtype=np.float64)


def scalar_crps(values, obs_value):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(obs_value):
        return np.nan
    n = values.size
    mean_abs = float(np.mean(np.abs(values - obs_value)))
    sorted_values = np.sort(values)
    i = np.arange(1, n + 1, dtype=np.float64)
    half_pairwise_mean = float(np.sum((2.0 * i - n - 1.0) * sorted_values) / (n * n))
    return mean_abs - half_pairwise_mean


def pinball_loss(obs_value, forecast_quantile, tau):
    if not np.isfinite(obs_value) or not np.isfinite(forecast_quantile):
        return np.nan
    diff = obs_value - forecast_quantile
    return float(max(float(tau) * diff, (float(tau) - 1.0) * diff))


def metric_units(variable, metric_name):
    if metric_name == "event_area_fraction":
        return "fraction"
    return VARIABLES[variable]["units"]


def plot_value(value, variable, metric_name):
    if metric_name in INTENSITY_METRICS:
        return np.asarray(value, dtype=np.float64) + float(VARIABLES[variable]["offset"])
    return np.asarray(value, dtype=np.float64)


def plot_units(variable, metric_name):
    if metric_name == "event_area_fraction":
        return "fraction"
    return VARIABLES[variable]["plot_units"]


def compute_member_metrics(ensemble, threshold, weights, tail_fraction, spatial_quantile):
    return {
        "area_mean": regional_member_values(ensemble, weights),
        "top_tail_intensity": regional_member_tail_values(ensemble, weights, fraction=tail_fraction),
        "spatial_q95_intensity": regional_member_spatial_quantile_values(ensemble, weights, spatial_quantile),
        "event_area_fraction": regional_member_event_area_fraction_values(ensemble, threshold, weights),
    }


def compute_reference_metrics(obs, threshold, weights, tail_fraction, spatial_quantile, event_area_fraction_threshold):
    obs_event = (np.asarray(obs, dtype=np.float32) >= np.asarray(threshold, dtype=np.float32)).astype(np.float32)
    obs_metrics = {
        "area_mean": weighted_mean(obs, weights),
        "top_tail_intensity": weighted_top_mean(obs, weights, fraction=tail_fraction),
        "spatial_q95_intensity": weighted_quantile(obs, weights, spatial_quantile),
        "event_area_fraction": weighted_mean(obs_event, weights),
    }
    threshold_metrics = {
        "area_mean": weighted_mean(threshold, weights),
        "top_tail_intensity": weighted_top_mean(threshold, weights, fraction=tail_fraction),
        "spatial_q95_intensity": weighted_quantile(threshold, weights, spatial_quantile),
        "event_area_fraction": float(event_area_fraction_threshold),
    }
    return obs_metrics, threshold_metrics


def summarize_distribution(values, obs_value, threshold_value, ensemble_quantiles, loss_quantiles):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    row = {
        "n_members": int(finite.size),
        "obs_value": float(obs_value) if np.isfinite(obs_value) else np.nan,
        "threshold_value": float(threshold_value) if np.isfinite(threshold_value) else np.nan,
        "ens_mean": float(np.nanmean(finite)) if finite.size else np.nan,
        "ens_std": float(np.nanstd(finite)) if finite.size else np.nan,
        "ens_min": float(np.nanmin(finite)) if finite.size else np.nan,
        "ens_max": float(np.nanmax(finite)) if finite.size else np.nan,
        "obs_percentile": float(np.nanmean(finite <= obs_value)) if finite.size and np.isfinite(obs_value) else np.nan,
        "prob_obs_or_more": float(np.nanmean(finite >= obs_value)) if finite.size and np.isfinite(obs_value) else np.nan,
        "prob_threshold_or_more": (
            float(np.nanmean(finite >= threshold_value)) if finite.size and np.isfinite(threshold_value) else np.nan
        ),
        "obs_below_ens_min": bool(np.isfinite(obs_value) and finite.size and obs_value < np.nanmin(finite)),
        "obs_above_ens_max": bool(np.isfinite(obs_value) and finite.size and obs_value > np.nanmax(finite)),
        "scalar_crps": scalar_crps(finite, obs_value),
        "abs_error_ens_mean": (
            abs(float(np.nanmean(finite)) - float(obs_value)) if finite.size and np.isfinite(obs_value) else np.nan
        ),
        "abs_error_ens_median": (
            abs(safe_quantile(finite, 0.50) - float(obs_value)) if finite.size and np.isfinite(obs_value) else np.nan
        ),
    }
    for q in ensemble_quantiles:
        row[qcol(q)] = safe_quantile(finite, q)
    for tau in loss_quantiles:
        forecast_q = safe_quantile(finite, tau)
        row[f"pinball_{qcol(tau)}"] = pinball_loss(obs_value, forecast_q, tau)
        row[f"abs_error_{qcol(tau)}"] = (
            abs(float(forecast_q) - float(obs_value)) if np.isfinite(forecast_q) and np.isfinite(obs_value) else np.nan
        )
    return row


def sample_metadata(event, sample, sample_kind):
    valid_time = pd.Timestamp(sample["valid_time"])
    return {
        "sample_id": (
            f"{event['event_id']}_init{pd.Timestamp(sample['init_time']).strftime('%Y%m%d')}"
            f"_valid{valid_time.strftime('%Y%m%d')}_lead{int(sample['lead'])}_{sample_kind}"
        ),
        "sample_kind": sample_kind,
        "event_id": event["event_id"],
        "region": event["region"],
        "region_label": event["region_label"],
        "variable": event["variable"],
        "event_name": event["event_name"],
        "event_start": str(pd.Timestamp(event["event_start"]).date()),
        "event_end": str(pd.Timestamp(event["event_end"]).date()),
        "init_time": str(pd.Timestamp(sample["init_time"]).date()),
        "valid_time": str(valid_time.date()),
        "target_window_start": str(pd.Timestamp(sample["target_window_start"]).date()),
        "target_window_end": str(pd.Timestamp(sample["target_window_end"]).date()),
        "lead": int(sample["lead"]),
        "event_overlap_days": int(sample.get("event_overlap_days", 0)),
        "event_overlap_fraction": float(sample.get("event_overlap_fraction", 0.0)),
        "event_distance_days": int(sample.get("event_distance_days", 0)),
        "selection_mode": sample.get("selection_mode", sample_kind),
    }


def evaluate_quantile_sample(
    sample,
    event,
    thresholds,
    weights,
    ensemble_quantiles,
    loss_quantiles,
    tail_fraction,
    spatial_quantile,
    event_area_fraction_threshold,
    sample_kind,
    keep_member_rows=True,
):
    variable = str(event["variable"])
    spec = VARIABLES[variable]
    ds = xr.open_zarr(sample["zarr_path"], consolidated=False, chunks=None)
    try:
        obs = ds[spec["obs"]].isel(init=sample["init_idx"], lead=sample["lead_idx"]).values.astype(np.float32)
        model_ens = ds[spec["model"]].isel(init=sample["init_idx"], lead=sample["lead_idx"]).values.astype(np.float32)
        geos_ens = ds[spec["geos"]].isel(init=sample["init_idx"], lead=sample["lead_idx"]).values.astype(np.float32)
    finally:
        ds.close()

    threshold = select_grouped_map(thresholds[variable], pd.Timestamp(sample["valid_time"]))
    valid_weights = np.where(np.isfinite(obs) & np.isfinite(threshold), weights, 0.0)
    obs_metrics, threshold_metrics = compute_reference_metrics(
        obs,
        threshold,
        valid_weights,
        tail_fraction,
        spatial_quantile,
        event_area_fraction_threshold,
    )
    model_metrics = compute_member_metrics(model_ens, threshold, valid_weights, tail_fraction, spatial_quantile)
    geos_metrics = compute_member_metrics(geos_ens, threshold, valid_weights, tail_fraction, spatial_quantile)
    meta = sample_metadata(event, sample, sample_kind)

    summary_rows = []
    member_rows = []
    for source, label, metric_dict in [("model", "ML", model_metrics), ("geos", "GEOS", geos_metrics)]:
        for metric_name, values in metric_dict.items():
            row = {
                **meta,
                "source": source,
                "source_label": label,
                "metric": metric_name,
                "metric_label": METRIC_LABELS[metric_name],
                "units": metric_units(variable, metric_name),
            }
            row.update(
                summarize_distribution(
                    values,
                    obs_metrics[metric_name],
                    threshold_metrics[metric_name],
                    ensemble_quantiles,
                    loss_quantiles,
                )
            )
            summary_rows.append(row)
            if keep_member_rows:
                for member_index, value in enumerate(np.asarray(values, dtype=np.float64)):
                    member_rows.append(
                        {
                            **meta,
                            "source": source,
                            "source_label": label,
                            "metric": metric_name,
                            "member": int(member_index),
                            "value": float(value) if np.isfinite(value) else np.nan,
                            "obs_value": float(obs_metrics[metric_name])
                            if np.isfinite(obs_metrics[metric_name])
                            else np.nan,
                            "threshold_value": float(threshold_metrics[metric_name])
                            if np.isfinite(threshold_metrics[metric_name])
                            else np.nan,
                            "units": metric_units(variable, metric_name),
                        }
                    )
    return summary_rows, member_rows


def comparison_table(summary, loss_quantiles):
    if summary.empty:
        return pd.DataFrame()
    index_cols = [
        "sample_id",
        "sample_kind",
        "event_id",
        "region",
        "region_label",
        "variable",
        "event_name",
        "event_start",
        "event_end",
        "init_time",
        "valid_time",
        "target_window_start",
        "target_window_end",
        "lead",
        "event_overlap_days",
        "event_overlap_fraction",
        "metric",
        "metric_label",
        "units",
    ]
    rows = []
    for key, group in summary.groupby(index_cols, dropna=False):
        by_source = {row.source: row for row in group.itertuples(index=False)}
        if "model" not in by_source or "geos" not in by_source:
            continue
        model = by_source["model"]._asdict()
        geos = by_source["geos"]._asdict()
        row = dict(zip(index_cols, key))
        row.update(
            {
                "obs_value": model["obs_value"],
                "threshold_value": model["threshold_value"],
                "model_obs_percentile": model["obs_percentile"],
                "geos_obs_percentile": geos["obs_percentile"],
                "obs_percentile_diff_model_minus_geos": model["obs_percentile"] - geos["obs_percentile"],
                "model_prob_obs_or_more": model["prob_obs_or_more"],
                "geos_prob_obs_or_more": geos["prob_obs_or_more"],
                "prob_obs_or_more_diff_model_minus_geos": model["prob_obs_or_more"] - geos["prob_obs_or_more"],
                "model_prob_threshold_or_more": model["prob_threshold_or_more"],
                "geos_prob_threshold_or_more": geos["prob_threshold_or_more"],
                "prob_threshold_or_more_diff_model_minus_geos": (
                    model["prob_threshold_or_more"] - geos["prob_threshold_or_more"]
                ),
                "model_scalar_crps": model["scalar_crps"],
                "geos_scalar_crps": geos["scalar_crps"],
                "scalar_crps_skill_pct": (
                    100.0 * (1.0 - model["scalar_crps"] / geos["scalar_crps"])
                    if np.isfinite(model["scalar_crps"])
                    and np.isfinite(geos["scalar_crps"])
                    and abs(geos["scalar_crps"]) > 1e-12
                    else np.nan
                ),
                "model_q95": model.get("q95", np.nan),
                "geos_q95": geos.get("q95", np.nan),
                "q95_error_skill": geos.get("abs_error_q95", np.nan) - model.get("abs_error_q95", np.nan),
                "model_mean_error": model["abs_error_ens_mean"],
                "geos_mean_error": geos["abs_error_ens_mean"],
                "mean_error_skill": geos["abs_error_ens_mean"] - model["abs_error_ens_mean"],
            }
        )
        for tau in loss_quantiles:
            col = f"pinball_{qcol(tau)}"
            row[f"model_{col}"] = model.get(col, np.nan)
            row[f"geos_{col}"] = geos.get(col, np.nan)
            row[f"{col}_skill"] = geos.get(col, np.nan) - model.get(col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_distribution(sample_summary, sample_members, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sample_summary.empty or sample_members.empty:
        return None
    first = sample_summary.iloc[0]
    variable = str(first["variable"])
    plot_dir = os.path.join(out_dir, "plots", "quantile_distributions")
    os.makedirs(plot_dir, exist_ok=True)

    metrics = ["area_mean", "top_tail_intensity", "spatial_q95_intensity", "event_area_fraction"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    axes = axes.ravel()
    for ax, metric_name in zip(axes, metrics):
        metric_members = sample_members[sample_members["metric"].eq(metric_name)]
        metric_summary = sample_summary[sample_summary["metric"].eq(metric_name)]
        if metric_members.empty or metric_summary.empty:
            ax.set_visible(False)
            continue
        ml_values = metric_members[metric_members["source"].eq("model")]["value"].to_numpy(dtype=np.float64)
        geos_values = metric_members[metric_members["source"].eq("geos")]["value"].to_numpy(dtype=np.float64)
        obs_value = float(metric_summary.iloc[0]["obs_value"])
        threshold_value = float(metric_summary.iloc[0]["threshold_value"])
        ml_plot = plot_value(ml_values, variable, metric_name)
        geos_plot = plot_value(geos_values, variable, metric_name)
        obs_plot = float(plot_value(obs_value, variable, metric_name))
        threshold_plot = float(plot_value(threshold_value, variable, metric_name))
        finite = np.concatenate([ml_plot[np.isfinite(ml_plot)], geos_plot[np.isfinite(geos_plot)], [obs_plot]])
        if finite.size:
            bins = np.linspace(np.nanpercentile(finite, 1), np.nanpercentile(finite, 99), 24)
            if np.nanmax(bins) <= np.nanmin(bins):
                bins = 20
        else:
            bins = 20
        ax.hist(geos_plot[np.isfinite(geos_plot)], bins=bins, density=True, alpha=0.32, color="#ff7f0e", label="GEOS")
        ax.hist(ml_plot[np.isfinite(ml_plot)], bins=bins, density=True, alpha=0.32, color="#1f77b4", label="ML")
        ax.axvline(obs_plot, color="black", linewidth=2.0, label="Obs")
        ax.axvline(threshold_plot, color="0.35", linestyle="--", linewidth=1.5, label="Obs-clim threshold")
        ml_row = metric_summary[metric_summary["source"].eq("model")].iloc[0]
        geos_row = metric_summary[metric_summary["source"].eq("geos")].iloc[0]
        ax.set_title(
            f"{METRIC_LABELS[metric_name]}\n"
            f"ML obs pct={ml_row['obs_percentile']:.2f}, GEOS={geos_row['obs_percentile']:.2f} | "
            f"P(obs+) ML={ml_row['prob_obs_or_more']:.2f}, GEOS={geos_row['prob_obs_or_more']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel(f"{plot_units(variable, metric_name)}")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.99), fontsize=8)
    fig.suptitle(
        f"{first['region_label']} | {first['event_name']} | {variable.upper()} quantile forecast | "
        f"init {first['init_time']} valid {first['valid_time']} lead week {first['lead']}",
        y=0.94,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(
        plot_dir,
        f"{first['event_id']}_init{str(first['init_time']).replace('-', '')}_lead{int(first['lead'])}_quantile_dist.png",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_progression(event, progression_summary, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if progression_summary.empty:
        return []
    plot_dir = os.path.join(out_dir, "plots", "quantile_progression")
    os.makedirs(plot_dir, exist_ok=True)
    out_paths = []
    variable = str(event["variable"])
    metrics = ["area_mean", "top_tail_intensity", "spatial_q95_intensity", "event_area_fraction"]
    for init_time, init_group in progression_summary.groupby("init_time", sort=True):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)
        axes = axes.ravel()
        for ax, metric_name in zip(axes, metrics):
            metric_group = init_group[init_group["metric"].eq(metric_name)].copy()
            if metric_group.empty:
                ax.set_visible(False)
                continue
            for source, color, label in [("geos", "#ff7f0e", "GEOS"), ("model", "#1f77b4", "ML")]:
                sub = metric_group[metric_group["source"].eq(source)].sort_values("lead")
                if sub.empty:
                    continue
                x = sub["lead"].to_numpy(dtype=float)
                q10 = plot_value(sub["q10"].to_numpy(dtype=np.float64), variable, metric_name)
                q50 = plot_value(sub["q50"].to_numpy(dtype=np.float64), variable, metric_name)
                q90 = plot_value(sub["q90"].to_numpy(dtype=np.float64), variable, metric_name)
                q95 = plot_value(sub["q95"].to_numpy(dtype=np.float64), variable, metric_name)
                ax.fill_between(x, q10, q90, color=color, alpha=0.16, label=f"{label} p10-p90")
                ax.plot(x, q50, color=color, marker="o", label=f"{label} p50")
                ax.plot(x, q95, color=color, linestyle="--", linewidth=1.3, label=f"{label} p95")
            obs = metric_group.drop_duplicates("lead").sort_values("lead")
            x_obs = obs["lead"].to_numpy(dtype=float)
            obs_plot = plot_value(obs["obs_value"].to_numpy(dtype=np.float64), variable, metric_name)
            threshold_plot = plot_value(obs["threshold_value"].to_numpy(dtype=np.float64), variable, metric_name)
            ax.plot(x_obs, obs_plot, color="black", marker="o", linewidth=2.0, label="Obs")
            ax.plot(x_obs, threshold_plot, color="0.35", linestyle=":", linewidth=1.4, label="threshold")
            ax.set_title(METRIC_LABELS[metric_name], fontsize=9)
            ax.set_ylabel(plot_units(variable, metric_name))
            ax.grid(alpha=0.25)
            ax.set_xticks(x_obs)
            ax.set_xticklabels([f"week{int(v)}" for v in x_obs])
        handles = []
        labels = []
        seen = set()
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label and label not in seen:
                    handles.append(handle)
                    labels.append(label)
                    seen.add(label)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.99), fontsize=7)
        fig.suptitle(
            f"{event['region_label']} | {event['event_name']} | {variable.upper()} fixed-init quantile progression | "
            f"init {init_time}",
            y=0.94,
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_path = os.path.join(
            plot_dir,
            f"{event['event_id']}_init{str(init_time).replace('-', '')}_quantile_progression.png",
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    leads = parse_list(args.leads, int)
    progression_leads = parse_list(args.progression_leads, int)
    regions = parse_list(args.regions)
    if regions == ["all"]:
        regions = ["all"]
    variables = parse_list(args.variables)
    ensemble_quantiles = sorted(set(parse_list(args.ensemble_quantiles, float)) | {0.10, 0.50, 0.90, 0.95})
    loss_quantiles = sorted(set(parse_list(args.loss_quantiles, float)) | {0.95})
    if not (0.0 < float(args.tail_fraction) <= 1.0):
        raise ValueError("--tail_fraction must be >0 and <=1")
    if not (0.0 < float(args.spatial_quantile) < 1.0):
        raise ValueError("--spatial_quantile must be >0 and <1")
    if not (0.0 <= float(args.event_area_fraction_threshold) <= 1.0):
        raise ValueError("--event_area_fraction_threshold must be between 0 and 1")

    catalog = normalize_catalog(load_event_catalog(args.event_catalog), regions, variables, args.start_year, args.end_year)
    catalog_path = os.path.join(args.out_dir, "event_catalog_used.csv")
    catalog.to_csv(catalog_path, index=False)
    print(f"✅ Wrote event catalog used: {catalog_path}")

    lats, lons = open_forecast_grid(args.forecast_dir, args.start_year, args.end_year)
    land_mask, land_source = load_land_mask(args.land_mask_file, (len(lats), len(lons)))
    thresholds, _, threshold_lats, threshold_lons = load_thresholds_from_file(args.threshold_file, variables, args)
    if not (np.allclose(lats, threshold_lats) and np.allclose(lons, threshold_lons)):
        raise ValueError("Threshold grid does not match forecast grid.")

    selected_summary_rows = []
    selected_member_rows = []
    progression_summary_rows = []
    progression_member_rows = []
    plot_records = []

    for _, event_row in catalog.iterrows():
        event = event_row.to_dict()
        event["bbox"] = parse_bbox(event["bbox"])
        mask = bbox_mask(lons, lats, event["bbox"]) & land_mask
        if not mask.any():
            print(f"⚠️ {event['event_id']}: empty land mask; skipping.")
            continue
        weights = region_weights(lats, mask, args.regional_weighting)
        samples = find_candidate_samples(
            args.forecast_dir,
            event,
            leads,
            args.timeseries_window_days,
            args.event_tolerance_days,
        )
        if not samples:
            print(f"⚠️ {event['event_id']}: no lead {leads} samples found near event.")
            continue
        selected = choose_event_samples(samples, leads)
        event_selected_summary = []
        event_selected_members = []
        for sample in selected:
            summary_rows, member_rows = evaluate_quantile_sample(
                sample,
                event,
                thresholds,
                weights,
                ensemble_quantiles,
                loss_quantiles,
                float(args.tail_fraction),
                float(args.spatial_quantile),
                float(args.event_area_fraction_threshold),
                sample_kind="selected_event_lead",
                keep_member_rows=True,
            )
            selected_summary_rows.extend(summary_rows)
            selected_member_rows.extend(member_rows)
            event_selected_summary.extend(summary_rows)
            event_selected_members.extend(member_rows)
            if args.make_plots:
                summary_df = pd.DataFrame(summary_rows)
                member_df = pd.DataFrame(member_rows)
                plot_path = plot_distribution(summary_df, member_df, args.out_dir)
                if plot_path:
                    plot_records.append(
                        {
                            "event_id": event["event_id"],
                            "lead": int(sample["lead"]),
                            "plot_type": "quantile_distribution",
                            "path": plot_path,
                        }
                    )

        fixed_samples = fixed_init_progression_samples(selected, event, progression_leads)
        event_progression_summary = []
        for sample in fixed_samples:
            summary_rows, member_rows = evaluate_quantile_sample(
                sample,
                event,
                thresholds,
                weights,
                ensemble_quantiles,
                loss_quantiles,
                float(args.tail_fraction),
                float(args.spatial_quantile),
                float(args.event_area_fraction_threshold),
                sample_kind="fixed_init_progression",
                keep_member_rows=args.write_member_values,
            )
            progression_summary_rows.extend(summary_rows)
            progression_member_rows.extend(member_rows)
            event_progression_summary.extend(summary_rows)
        if args.make_plots and event_progression_summary:
            for path in plot_progression(event, pd.DataFrame(event_progression_summary), args.out_dir):
                plot_records.append({"event_id": event["event_id"], "plot_type": "quantile_progression", "path": path})

        print(
            f"✅ {event['event_id']}: selected samples={len(selected)}, "
            f"fixed progression samples={len(fixed_samples)}"
        )

    selected_summary = pd.DataFrame(selected_summary_rows)
    progression_summary = pd.DataFrame(progression_summary_rows)
    selected_comparison = comparison_table(selected_summary, loss_quantiles)
    progression_comparison = comparison_table(progression_summary, loss_quantiles)

    selected_summary_path = os.path.join(args.out_dir, "event_quantile_selected_summary.csv")
    selected_comparison_path = os.path.join(args.out_dir, "event_quantile_selected_comparison.csv")
    progression_summary_path = os.path.join(args.out_dir, "event_quantile_progression_summary.csv")
    progression_comparison_path = os.path.join(args.out_dir, "event_quantile_progression_comparison.csv")
    selected_summary.to_csv(selected_summary_path, index=False, float_format="%.6f")
    selected_comparison.to_csv(selected_comparison_path, index=False, float_format="%.6f")
    progression_summary.to_csv(progression_summary_path, index=False, float_format="%.6f")
    progression_comparison.to_csv(progression_comparison_path, index=False, float_format="%.6f")
    print(f"✅ Wrote selected quantile summary: {selected_summary_path}")
    print(f"✅ Wrote selected ML-vs-GEOS comparison: {selected_comparison_path}")
    print(f"✅ Wrote fixed-init progression summary: {progression_summary_path}")
    print(f"✅ Wrote fixed-init progression comparison: {progression_comparison_path}")

    if args.write_member_values:
        selected_member_path = os.path.join(args.out_dir, "event_quantile_selected_member_values.csv")
        progression_member_path = os.path.join(args.out_dir, "event_quantile_progression_member_values.csv")
        pd.DataFrame(selected_member_rows).to_csv(selected_member_path, index=False, float_format="%.6f")
        pd.DataFrame(progression_member_rows).to_csv(progression_member_path, index=False, float_format="%.6f")
        print(f"✅ Wrote selected member values: {selected_member_path}")
        print(f"✅ Wrote progression member values: {progression_member_path}")

    if not selected_comparison.empty:
        overall = (
            selected_comparison.groupby(["region", "region_label", "variable", "metric", "metric_label"], dropna=False)
            .agg(
                n=("event_id", "count"),
                mean_scalar_crps_skill_pct=("scalar_crps_skill_pct", "mean"),
                mean_q95_error_skill=("q95_error_skill", "mean"),
                mean_prob_obs_or_more_diff=("prob_obs_or_more_diff_model_minus_geos", "mean"),
                mean_prob_threshold_diff=("prob_threshold_or_more_diff_model_minus_geos", "mean"),
                mean_obs_percentile_diff=("obs_percentile_diff_model_minus_geos", "mean"),
            )
            .reset_index()
        )
        overall_path = os.path.join(args.out_dir, "event_quantile_overall_by_region_variable_metric.csv")
        overall.to_csv(overall_path, index=False, float_format="%.6f")
        print(f"✅ Wrote overall quantile table: {overall_path}")

    plot_index_path = os.path.join(args.out_dir, "event_quantile_plot_index.csv")
    pd.DataFrame(plot_records).to_csv(plot_index_path, index=False)
    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "threshold_file": os.path.abspath(args.threshold_file),
        "land_mask_file": land_source,
        "event_catalog": args.event_catalog,
        "leads": leads,
        "progression_leads": progression_leads,
        "tail_fraction": float(args.tail_fraction),
        "spatial_quantile": float(args.spatial_quantile),
        "ensemble_quantiles": ensemble_quantiles,
        "loss_quantiles": loss_quantiles,
        "event_area_fraction_threshold": float(args.event_area_fraction_threshold),
        "regional_weighting": args.regional_weighting,
        "timeseries_window_days": int(args.timeseries_window_days),
        "event_tolerance_days": int(args.event_tolerance_days),
        "event_count": int(len(catalog)),
        "note": (
            "This quantile evaluator treats each event forecast as a regional predictive distribution. "
            "Observed percentile near 1 means the observed high-end event was at or above the forecast upper tail; "
            "prob_obs_or_more is the ensemble probability of an outcome at least as extreme as observed; "
            "q95_error_skill and pinball-loss skill are positive when ML is better than GEOS."
        ),
    }
    metadata_path = os.path.join(args.out_dir, "event_quantile_eval_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
