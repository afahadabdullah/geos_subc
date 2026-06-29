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
  observation >= local observed threshold map. By default thresholds are local
  gridpoint 95th percentiles. Prefer building/providing long-term observed
  thresholds; evaluation-year thresholds are a fallback. For precipitation, the
  threshold is also constrained to be at least --pr_min_threshold mm/day.

Calibrated BSS:
  by default, forecast event probabilities are adjusted with leave-one-year-out
  logistic reliability calibration:
    logit(p_cal) = a + b * logit(p_raw)
  Coefficients are fit from area-weighted binned reliability counts, grouped by
  variable/source/lead/valid-season, with broader fallbacks for sparse groups.
"""

import argparse
import json
import os
import time
import warnings

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

MAP_CONTEXT = {
    "enabled": False,
    "ccrs": None,
    "plate_carree": None,
    "data_crs": None,
    "features": [],
    "lon_formatter": None,
    "lat_formatter": None,
}


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
    parser.add_argument(
        "--threshold_file",
        type=str,
        default=None,
        help=(
            "Optional observed-threshold NetCDF from build_observed_extreme_thresholds_flow_finalv1_global.py. "
            "If omitted, thresholds are built from --threshold_forecast_dir/years."
        ),
    )
    parser.add_argument(
        "--threshold_forecast_dir",
        type=str,
        default=None,
        help="Forecast Zarr directory used only for observed thresholds; defaults to --forecast_dir.",
    )
    parser.add_argument("--threshold_start_year", type=int, default=None)
    parser.add_argument("--threshold_end_year", type=int, default=None)
    parser.add_argument("--threshold_skip_years", type=str, default="")
    parser.add_argument(
        "--threshold_grouping",
        choices=("pooled", "monthly", "seasonal"),
        default="monthly",
        help="Observed threshold grouping. monthly is usually best for long-term observed climatology.",
    )
    parser.add_argument(
        "--eval_mask",
        choices=("all", "land", "ocean"),
        default="all",
        help="Spatial mask for scalar/spatial evaluation. Use land to prevent ocean-dominated maps/tables.",
    )
    parser.add_argument(
        "--land_mask_file",
        type=str,
        default=None,
        help="Optional .pt land mask with is_land or land_mask. Required for --eval_mask land/ocean.",
    )
    parser.add_argument("--epsilon_probability", type=float, default=1e-4)
    parser.add_argument(
        "--bss_calibration",
        choices=("logistic_cv", "base_rate", "none"),
        default="logistic_cv",
        help=(
            "Probability calibration for calibrated BSS. logistic_cv uses leave-one-year-out "
            "Platt/logistic calibration from binned reliability counts; base_rate is the old "
            "local logit climatology correction; none uses raw ensemble probabilities."
        ),
    )
    parser.add_argument(
        "--bss_calibration_grouping",
        choices=("lead_season", "lead", "global"),
        default="lead_season",
        help="Grouping used for logistic_cv calibration before fallback broadening.",
    )
    parser.add_argument("--bss_calibration_bins", type=int, default=41)
    parser.add_argument("--bss_calibration_ridge", type=float, default=1.0)
    parser.add_argument("--bss_calibration_min_weight", type=float, default=100.0)
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument(
        "--map_features",
        choices=("auto", "cartopy", "plain"),
        default="auto",
        help="Use Cartopy coastlines/borders for spatial matrix plots when available.",
    )
    parser.add_argument(
        "--county_boundaries",
        choices=("auto", "on", "off"),
        default="off",
        help="Add US county boundaries to spatial maps when cached and requested.",
    )
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


def _cached_natural_earth_feature(cfeature, ccrs, shapereader, download_warning, resolution, category, name, **kwargs):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", download_warning)
            path = shapereader.natural_earth(resolution=resolution, category=category, name=name)
        geometries = list(shapereader.Reader(path).geometries())
        if not geometries:
            return None
        return cfeature.ShapelyFeature(geometries, ccrs.PlateCarree(), **kwargs)
    except Exception:
        return None


def configure_map_context(args):
    MAP_CONTEXT.update(
        {
            "enabled": False,
            "ccrs": None,
            "plate_carree": None,
            "data_crs": None,
            "features": [],
            "lon_formatter": None,
            "lat_formatter": None,
        }
    )
    if args.map_features == "plain":
        print("🗺️ Spatial maps: plain matplotlib axes (--map_features plain).")
        return
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.io import DownloadWarning, shapereader
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
    except Exception as exc:
        label = "required" if args.map_features == "cartopy" else "requested"
        print(f"⚠️ Cartopy spatial maps {label}, but Cartopy is unavailable ({exc}). Falling back to plain axes.")
        return

    feature_specs = [
        ("50m", "physical", "coastline", {"edgecolor": "black", "facecolor": "none", "linewidth": 0.45}),
        (
            "50m",
            "cultural",
            "admin_0_boundary_lines_land",
            {"edgecolor": "black", "facecolor": "none", "linewidth": 0.35},
        ),
        (
            "50m",
            "cultural",
            "admin_1_states_provinces_lines",
            {"edgecolor": "0.35", "facecolor": "none", "linewidth": 0.25, "alpha": 0.75},
        ),
    ]
    if args.county_boundaries in ("on", "auto"):
        feature_specs.append(
            (
                "10m",
                "cultural",
                "admin_2_counties",
                {"edgecolor": "0.35", "facecolor": "none", "linewidth": 0.15, "alpha": 0.45},
            )
        )

    features = []
    skipped = []
    for resolution, category, name, kwargs in feature_specs:
        feature = _cached_natural_earth_feature(
            cfeature,
            ccrs,
            shapereader,
            DownloadWarning,
            resolution,
            category,
            name,
            **kwargs,
        )
        if feature is None:
            skipped.append(f"{resolution}/{name}")
        else:
            features.append(feature)

    if skipped:
        print("🗺️ Cartopy spatial maps enabled. Skipped uncached layers: " + ", ".join(skipped))
    else:
        print("🗺️ Cartopy spatial maps enabled with Natural Earth overlays.")
    if not features:
        print("⚠️ No cached Natural Earth overlays found; using Cartopy GeoAxes without coast/border layers.")

    plate_carree = ccrs.PlateCarree()
    MAP_CONTEXT.update(
        {
            "enabled": True,
            "ccrs": ccrs,
            "plate_carree": plate_carree,
            "data_crs": plate_carree,
            "features": features,
            "lon_formatter": LongitudeFormatter(),
            "lat_formatter": LatitudeFormatter(),
        }
    )


def _add_cyclic_if_global(lons, field):
    lons = np.asarray(lons, dtype=np.float64)
    field = np.asarray(field)
    if lons.size < 3 or field.shape[-1] != lons.size:
        return lons, field
    finite_lons = lons[np.isfinite(lons)]
    if finite_lons.size < 3:
        return lons, field
    diffs = np.diff(lons)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if positive_diffs.size == 0:
        return lons, field
    dx = float(np.nanmedian(positive_diffs))
    span = float(np.nanmax(lons) - np.nanmin(lons) + dx)
    if span < 350.0:
        return lons, field
    cyclic_lons = np.concatenate([lons, [lons[-1] + dx]])
    cyclic_field = np.concatenate([field, field[..., :1]], axis=-1)
    return cyclic_lons, cyclic_field


def prepare_spatial_field_for_plot(lons, lats, field):
    plot_lons = np.asarray(lons, dtype=np.float64)
    plot_lats = np.asarray(lats, dtype=np.float64)
    plot_field = np.asarray(field)

    if plot_lats.ndim == 1 and plot_field.shape[-2] == plot_lats.size and plot_lats[0] > plot_lats[-1]:
        plot_lats = plot_lats[::-1]
        plot_field = plot_field[..., ::-1, :]

    if MAP_CONTEXT["enabled"] and plot_lons.ndim == 1 and plot_field.shape[-1] == plot_lons.size:
        if np.nanmax(plot_lons) > 180.0:
            plot_lons = ((plot_lons + 180.0) % 360.0) - 180.0
        order = np.argsort(plot_lons)
        plot_lons = plot_lons[order]
        plot_field = plot_field[..., order]
        unique_mask = np.concatenate([[True], np.diff(np.round(plot_lons, 7)) > 0])
        if unique_mask.size == plot_lons.size and not np.all(unique_mask):
            plot_lons = plot_lons[unique_mask]
            plot_field = plot_field[..., unique_mask]
        plot_lons, plot_field = _add_cyclic_if_global(plot_lons, plot_field)

    return plot_lons, plot_lats, plot_field


def make_map_subplots(nrows, ncols, figsize, **kwargs):
    import matplotlib.pyplot as plt

    if MAP_CONTEXT["enabled"]:
        kwargs.setdefault("subplot_kw", {"projection": MAP_CONTEXT["plate_carree"]})
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def add_map_overlays(ax, lons, lats):
    if not MAP_CONTEXT["enabled"]:
        return
    plate_carree = MAP_CONTEXT["plate_carree"]
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))
    try:
        if lon_max - lon_min >= 350.0 and lat_max - lat_min >= 170.0:
            ax.set_global()
        else:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=plate_carree)
    except Exception:
        pass
    for feature in MAP_CONTEXT["features"]:
        try:
            ax.add_feature(feature, zorder=3)
        except Exception:
            continue
    try:
        ax.gridlines(crs=plate_carree, linewidth=0.2, color="0.55", alpha=0.35, linestyle="-")
    except Exception:
        pass
    try:
        if lon_max - lon_min >= 300.0:
            xticks = np.arange(-180.0, 181.0, 60.0)
        else:
            xticks = np.linspace(lon_min, lon_max, 5)
        if lat_max - lat_min >= 120.0:
            yticks = np.arange(-90.0, 91.0, 30.0)
        else:
            yticks = np.linspace(lat_min, lat_max, 5)
        ax.set_xticks(xticks, crs=plate_carree)
        ax.set_yticks(yticks, crs=plate_carree)
        if MAP_CONTEXT["lon_formatter"] is not None:
            ax.xaxis.set_major_formatter(MAP_CONTEXT["lon_formatter"])
        if MAP_CONTEXT["lat_formatter"] is not None:
            ax.yaxis.set_major_formatter(MAP_CONTEXT["lat_formatter"])
        ax.tick_params(labelsize=6)
    except Exception:
        pass


def season_name(month):
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def threshold_group_values(grouping):
    if grouping == "monthly":
        return MONTHS
    if grouping == "seasonal":
        return SEASONS
    return ["pooled"]


def group_label_for_time(grouping, valid_time):
    if grouping == "monthly":
        return f"{int(pd.Timestamp(valid_time).month):02d}"
    if grouping == "seasonal":
        return season_name(int(pd.Timestamp(valid_time).month))
    return "pooled"


def make_threshold_bundle(values, grouping="pooled", group_values=None, source=None):
    values = np.asarray(values, dtype=np.float32)
    grouping = str(grouping or "pooled")
    if values.ndim == 2:
        grouping = "pooled" if grouping not in ("monthly", "seasonal") else grouping
        group_values = ["pooled"] if group_values is None else list(group_values)
    elif values.ndim == 3:
        if group_values is None:
            group_values = threshold_group_values(grouping)
        group_values = [str(v) for v in group_values]
        if len(group_values) != values.shape[0]:
            raise ValueError(
                f"Threshold group count mismatch: values has {values.shape[0]} groups, "
                f"but group_values={group_values}"
            )
    else:
        raise ValueError(f"Expected 2D or 3D threshold/climatology values, got shape {values.shape}")
    return {
        "values": values,
        "grouping": grouping,
        "group_values": group_values,
        "source": source,
    }


def _as_threshold_bundle(obj, default_grouping="pooled"):
    if isinstance(obj, dict) and "values" in obj:
        return obj
    return make_threshold_bundle(obj, grouping=default_grouping)


def select_grouped_map(bundle, valid_time):
    bundle = _as_threshold_bundle(bundle)
    values = np.asarray(bundle["values"], dtype=np.float32)
    if values.ndim == 2:
        return values
    label = group_label_for_time(bundle.get("grouping", "pooled"), valid_time)
    group_values = [str(v) for v in bundle.get("group_values", [])]
    if label not in group_values:
        raise KeyError(f"Threshold group {label!r} not available in {group_values}")
    return values[group_values.index(label)]


def first_threshold_map(bundle):
    values = np.asarray(_as_threshold_bundle(bundle)["values"], dtype=np.float32)
    return values[0] if values.ndim == 3 else values


def bundle_shape(bundle):
    values = np.asarray(_as_threshold_bundle(bundle)["values"])
    return tuple(values.shape[-2:])


def group_values_from_coord(coord_values, grouping):
    if grouping == "monthly":
        out = []
        for value in coord_values:
            if np.issubdtype(np.asarray(value).dtype, np.number):
                out.append(f"{int(value):02d}")
            else:
                text = str(value)
                out.append(f"{int(text):02d}" if text.isdigit() else text)
        return out
    if grouping == "seasonal":
        return [str(v) for v in coord_values]
    return ["pooled"]


def infer_threshold_grouping(data_array):
    dims = tuple(data_array.dims)
    for dim in dims:
        lower = dim.lower()
        if lower in ("month", "valid_month"):
            return "monthly", dim
        if lower in ("season", "valid_season"):
            return "seasonal", dim
        if "threshold_group" in lower or lower in ("group", "clim_group"):
            values = [str(v) for v in data_array[dim].values]
            if set(values).issubset(set(MONTHS)) or all(v.isdigit() and 1 <= int(v) <= 12 for v in values):
                return "monthly", dim
            if set(values).issubset(set(SEASONS)):
                return "seasonal", dim
            return str(data_array.attrs.get("threshold_grouping", "pooled")), dim
    return str(data_array.attrs.get("threshold_grouping", "pooled")), None


def valid_times_for_dataset(ds, init_idx, init_time, lead_values):
    if "valid_time" in ds:
        return pd.to_datetime(ds["valid_time"].isel(init=init_idx).values).normalize()
    return pd.to_datetime(
        [init_time + pd.to_timedelta(int(lead) * 7, unit="D") for lead in lead_values]
    ).normalize()


def load_evaluation_mask(args, lats, lons):
    shape = (len(lats), len(lons))
    if args.eval_mask == "all":
        print("🌍 Evaluation mask: all grid points.")
        return np.ones(shape, dtype=bool), "all"
    if not args.land_mask_file:
        raise ValueError("--land_mask_file is required when --eval_mask is land or ocean.")
    import torch

    print(f"🌍 Loading evaluation land/ocean mask from: {args.land_mask_file}")
    cached = torch.load(args.land_mask_file, map_location="cpu", weights_only=True)
    if "is_land" in cached:
        land_mask = np.asarray(cached["is_land"], dtype=bool).squeeze()
    elif "land_mask" in cached:
        land_mask = np.asarray(cached["land_mask"], dtype=bool).squeeze()
    else:
        raise ValueError(f"Mask file {args.land_mask_file} is missing 'is_land' or 'land_mask'.")
    if land_mask.shape != shape:
        raise ValueError(f"Evaluation mask shape {land_mask.shape} does not match forecast grid {shape}.")
    eval_mask = land_mask if args.eval_mask == "land" else ~land_mask
    print(
        f"🌍 Evaluation mask active: {args.eval_mask}; "
        f"kept={int(eval_mask.sum())} masked={int(eval_mask.size - eval_mask.sum())}"
    )
    return eval_mask.astype(bool), os.path.abspath(args.land_mask_file)


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


def apply_probability_calibration(
    prob,
    obs_event_freq=None,
    forecast_event_freq=None,
    eps=1e-4,
    method="base_rate",
    calibrator=None,
):
    prob = np.asarray(prob, dtype=np.float64)
    if method == "none":
        return np.clip(prob, 0.0, 1.0)
    if method == "logistic_cv" and calibrator is not None:
        intercept = float(calibrator.get("intercept", 0.0))
        slope = float(calibrator.get("slope", 1.0))
        return np.clip(inv_logit(intercept + slope * logit(prob, eps)), 0.0, 1.0)
    if forecast_event_freq is None or obs_event_freq is None:
        return np.clip(prob, 0.0, 1.0)
    return calibrate_probability(prob, obs_event_freq, forecast_event_freq, eps=eps)


def empty_calibration_count_state(num_bins):
    return {
        "total": np.zeros(num_bins, dtype=np.float64),
        "event": np.zeros(num_bins, dtype=np.float64),
        "prob": np.zeros(num_bins, dtype=np.float64),
    }


def update_calibration_count_state(state, prob, event, finite, weights, num_bins):
    prob = np.asarray(prob, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    finite = np.asarray(finite, dtype=bool) & np.isfinite(prob)
    if not finite.any():
        return
    weight_field = np.broadcast_to(weights, prob.shape).astype(np.float64, copy=False)
    flat_prob = np.clip(prob[finite], 0.0, 1.0)
    flat_event = event[finite].astype(np.float64)
    flat_weight = weight_field[finite]
    bin_idx = np.floor(flat_prob * num_bins).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    state["total"] += np.bincount(bin_idx, weights=flat_weight, minlength=num_bins)
    state["event"] += np.bincount(bin_idx, weights=flat_weight * flat_event, minlength=num_bins)
    state["prob"] += np.bincount(bin_idx, weights=flat_weight * flat_prob, minlength=num_bins)


def merge_calibration_count_states(states, num_bins):
    merged = empty_calibration_count_state(num_bins)
    for state in states:
        merged["total"] += state["total"]
        merged["event"] += state["event"]
        merged["prob"] += state["prob"]
    return merged


def fit_logistic_from_binned_counts(count_state, eps=1e-4, ridge=1.0, min_weight=100.0):
    total = np.asarray(count_state["total"], dtype=np.float64)
    event = np.asarray(count_state["event"], dtype=np.float64)
    prob_sum = np.asarray(count_state["prob"], dtype=np.float64)
    total_weight = float(np.sum(total))
    event_weight = float(np.sum(event))
    if total_weight < float(min_weight):
        return None
    event_rate = event_weight / total_weight
    if not np.isfinite(event_rate) or event_rate <= eps or event_rate >= 1.0 - eps:
        return None

    valid = total > 0.0
    if not valid.any():
        return None
    p_mean = np.where(valid, prob_sum / np.maximum(total, 1e-12), np.nan)
    x = logit(p_mean[valid], eps)
    n = total[valid]
    y = event[valid]
    unique_x = np.unique(np.round(x[np.isfinite(x)], 8))
    if unique_x.size < 2:
        return {
            "method": "climatology",
            "intercept": float(logit(event_rate, eps)),
            "slope": 0.0,
            "total_weight": total_weight,
            "event_rate": float(event_rate),
            "bins_used": int(valid.sum()),
        }

    beta = np.array([logit(event_rate, eps), 0.0], dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    design = np.column_stack([np.ones_like(x), x])
    ridge_matrix = np.diag([0.0, float(ridge)])
    for _ in range(50):
        eta = np.clip(design @ beta, -30.0, 30.0)
        mu = inv_logit(eta)
        score = design.T @ (y - n * mu) - ridge_matrix @ beta
        info = (design.T * (n * mu * (1.0 - mu))) @ design + ridge_matrix
        try:
            delta = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            return {
                "method": "climatology",
                "intercept": float(logit(event_rate, eps)),
                "slope": 0.0,
                "total_weight": total_weight,
                "event_rate": float(event_rate),
                "bins_used": int(valid.sum()),
            }
        beta += delta
        if float(np.max(np.abs(delta))) < 1e-6:
            break

    beta[0] = float(np.clip(beta[0], -20.0, 20.0))
    beta[1] = float(np.clip(beta[1], 0.0, 10.0))
    if beta[1] <= 1e-8:
        beta[0] = float(logit(event_rate, eps))
        beta[1] = 0.0
        method = "climatology"
    else:
        method = "logistic"
    return {
        "method": method,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "total_weight": total_weight,
        "event_rate": float(event_rate),
        "bins_used": int(valid.sum()),
    }


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


def ensemble_diagnostics(
    ensemble,
    obs,
    threshold,
    obs_event_freq,
    fcst_event_freq,
    eps,
    calibration_method="base_rate",
    calibrator=None,
):
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    mean = np.nanmean(ensemble, axis=0).astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    err = mean - obs64
    finite = np.isfinite(obs64) & np.isfinite(mean) & np.isfinite(threshold)
    prob = np.nanmean(ensemble >= threshold[None, :, :], axis=0).astype(np.float64, copy=False)
    event = obs64 >= threshold
    prob_cal = apply_probability_calibration(
        prob,
        obs_event_freq=obs_event_freq,
        forecast_event_freq=fcst_event_freq,
        eps=eps,
        method=calibration_method,
        calibrator=calibrator,
    )
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
            "bss_reference": "local observed event climatology matching the selected observed threshold source/group.",
        },
    )
    return ds


def _find_coord_name(ds, candidates):
    for candidate in candidates:
        if candidate in ds.coords:
            return candidate
        if candidate in ds.dims:
            return candidate
    lowered = {str(name).lower(): name for name in list(ds.coords) + list(ds.dims)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _threshold_array_to_bundle(data_array, variable, args, fallback_source, apply_min_threshold=False):
    lat_dim = next((dim for dim in data_array.dims if str(dim).lower() in ("lat", "latitude", "y")), None)
    lon_dim = next((dim for dim in data_array.dims if str(dim).lower() in ("lon", "longitude", "x")), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"{data_array.name} is missing recognizable lat/lon dims; dims={data_array.dims}")

    grouping, group_dim = infer_threshold_grouping(data_array)
    if group_dim is None:
        values = data_array.transpose(lat_dim, lon_dim).values.astype(np.float32, copy=False)
        group_values = ["pooled"]
        grouping = "pooled"
    else:
        values = data_array.transpose(group_dim, lat_dim, lon_dim).values.astype(np.float32, copy=False)
        group_values = group_values_from_coord(data_array[group_dim].values, grouping)

    min_arg = VARIABLES[variable]["min_threshold_arg"]
    if apply_min_threshold and min_arg is not None:
        values = np.maximum(values, float(getattr(args, min_arg))).astype(np.float32)
    return make_threshold_bundle(values, grouping=grouping, group_values=group_values, source=fallback_source)


def load_thresholds_from_file(path, variables, args):
    print(f"📏 Loading observed extreme thresholds from: {path}")
    ds = xr.open_dataset(path)
    try:
        lat_name = _find_coord_name(ds, ("lat", "latitude", "Y"))
        lon_name = _find_coord_name(ds, ("lon", "longitude", "X"))
        if lat_name is None or lon_name is None:
            raise ValueError(f"Threshold file {path} is missing lat/lon coordinates.")
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        thresholds = {}
        climatology = {}
        for variable in variables:
            threshold_name = f"{variable}_threshold"
            if threshold_name not in ds:
                raise ValueError(f"Threshold file {path} is missing variable {threshold_name}.")
            thresholds[variable] = _threshold_array_to_bundle(
                ds[threshold_name],
                variable,
                args,
                os.path.abspath(path),
                apply_min_threshold=True,
            )

            freq_name = f"{variable}_obs_event_frequency"
            if freq_name in ds:
                climatology[variable] = _threshold_array_to_bundle(
                    ds[freq_name],
                    variable,
                    args,
                    os.path.abspath(path),
                    apply_min_threshold=False,
                )
            else:
                q = float(getattr(args, VARIABLES[variable]["extreme_quantile_arg"]))
                threshold_values = thresholds[variable]["values"]
                frequency = np.full_like(threshold_values, 1.0 - q, dtype=np.float32)
                frequency = np.where(np.isfinite(threshold_values), frequency, np.nan).astype(np.float32)
                print(
                    f"⚠️ {freq_name} missing in threshold file; using nominal frequency={1.0 - q:.3f}. "
                    "For precipitation with a minimum threshold, build a threshold file with observed frequencies."
                )
                climatology[variable] = make_threshold_bundle(
                    frequency,
                    grouping=thresholds[variable]["grouping"],
                    group_values=thresholds[variable]["group_values"],
                    source=os.path.abspath(path),
                )

            values = thresholds[variable]["values"]
            freq = climatology[variable]["values"]
            group_info = (
                f"{thresholds[variable]['grouping']} groups={thresholds[variable]['group_values']}"
                if values.ndim == 3
                else "pooled"
            )
            print(
                f"   {variable}: {group_info}; threshold mean={float(np.nanmean(values)):.3f} "
                f"{VARIABLES[variable]['units']}, obs event freq mean={float(np.nanmean(freq)):.4f}"
            )
        return thresholds, climatology, lats, lons
    finally:
        ds.close()


def collect_obs_thresholds(forecast_dir, years, variables, args):
    if args.threshold_file:
        return load_thresholds_from_file(args.threshold_file, variables, args)

    threshold_dir = args.threshold_forecast_dir or forecast_dir
    threshold_start = args.threshold_start_year if args.threshold_start_year is not None else min(years)
    threshold_end = args.threshold_end_year if args.threshold_end_year is not None else max(years)
    threshold_skip = parse_years(args.threshold_skip_years)
    threshold_years = [
        year for year in range(int(threshold_start), int(threshold_end) + 1) if year not in threshold_skip
    ]
    if not threshold_years:
        raise ValueError("No threshold years selected.")

    thresholds = {}
    climatology = {}
    saved_lats = None
    saved_lons = None
    group_values = threshold_group_values(args.threshold_grouping)
    print(
        "📏 Building observed extreme thresholds from forecast-store observations: "
        f"{threshold_dir}, years={threshold_years}, grouping={args.threshold_grouping}"
    )
    for variable in variables:
        spec = VARIABLES[variable]
        obs_by_group = {group_value: [] for group_value in group_values}
        for year in threshold_years:
            zarr_path = os.path.join(threshold_dir, f"{year}.zarr")
            ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
            try:
                if saved_lats is None:
                    saved_lats = ds["lat"].values
                    saved_lons = ds["lon"].values
                init_values = pd.to_datetime(ds["init"].values).normalize()
                lead_values = ds["lead"].values
                n_lead = ds.sizes["lead"]
                for init_idx, init_time in enumerate(init_values):
                    valid_values = valid_times_for_dataset(ds, init_idx, init_time, lead_values)
                    for lead_idx in range(n_lead):
                        valid_time = pd.Timestamp(valid_values[lead_idx])
                        group_value = group_label_for_time(args.threshold_grouping, valid_time)
                        obs = ds[spec["obs"]].isel(init=init_idx, lead=lead_idx).values.astype(np.float32, copy=False)
                        obs_by_group[group_value].append(obs)
            finally:
                ds.close()

        q = float(getattr(args, spec["extreme_quantile_arg"]))
        threshold_maps = []
        frequency_maps = []
        shape = (len(saved_lats), len(saved_lons))
        min_arg = spec["min_threshold_arg"]
        for group_value in group_values:
            chunks = obs_by_group[group_value]
            if not chunks:
                threshold = np.full(shape, np.nan, dtype=np.float32)
                event_freq = np.full(shape, np.nan, dtype=np.float32)
            else:
                stack = np.stack(chunks, axis=0).astype(np.float32, copy=False)
                threshold = np.nanquantile(stack, q, axis=0).astype(np.float32)
                if min_arg is not None:
                    threshold = np.maximum(threshold, float(getattr(args, min_arg))).astype(np.float32)
                events = stack >= threshold[None, :, :]
                event_freq = np.nanmean(events, axis=0).astype(np.float32)
                event_freq = np.where(np.isfinite(event_freq), event_freq, np.nan).astype(np.float32)
            threshold_maps.append(threshold)
            frequency_maps.append(event_freq)

        if args.threshold_grouping == "pooled":
            threshold_values = threshold_maps[0]
            frequency_values = frequency_maps[0]
        else:
            threshold_values = np.stack(threshold_maps, axis=0)
            frequency_values = np.stack(frequency_maps, axis=0)
        thresholds[variable] = make_threshold_bundle(
            threshold_values,
            grouping=args.threshold_grouping,
            group_values=group_values,
            source=os.path.abspath(threshold_dir),
        )
        climatology[variable] = make_threshold_bundle(
            frequency_values,
            grouping=args.threshold_grouping,
            group_values=group_values,
            source=os.path.abspath(threshold_dir),
        )
        print(
            f"   {variable}: q={q:.3f}, grouping={args.threshold_grouping}, "
            f"threshold mean={float(np.nanmean(threshold_values)):.3f} {spec['units']}, "
            f"obs event freq mean={float(np.nanmean(frequency_values)):.4f}"
        )
    return thresholds, climatology, saved_lats, saved_lons


def collect_forecast_event_climatology(forecast_dir, years, variables, thresholds, deadline=None):
    out = {}
    print("🎯 Building forecast event-probability climatology for calibrated BSS...")
    for variable in variables:
        shape = bundle_shape(thresholds[variable])
        sums = {"model": np.zeros(shape, dtype=np.float64), "geos": np.zeros(shape, dtype=np.float64)}
        count = np.zeros(shape, dtype=np.float64)
        spec = VARIABLES[variable]
        for year in years:
            if deadline_reached(deadline):
                raise TimeoutError("Soft runtime limit reached while building forecast event climatology.")
            ds = xr.open_zarr(os.path.join(forecast_dir, f"{year}.zarr"), consolidated=False, chunks=None)
            try:
                init_values = pd.to_datetime(ds["init"].values).normalize()
                lead_values = ds["lead"].values
                n_init = ds.sizes["init"]
                n_lead = ds.sizes["lead"]
                for init_idx, init_time in enumerate(init_values):
                    valid_values = valid_times_for_dataset(ds, init_idx, init_time, lead_values)
                    for lead_idx in range(n_lead):
                        threshold = select_grouped_map(thresholds[variable], pd.Timestamp(valid_values[lead_idx]))
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


def collect_forecast_calibration_inputs(
    forecast_dir,
    years,
    variables,
    thresholds,
    lats,
    lons,
    args,
    evaluation_mask=None,
    deadline=None,
):
    num_bins = int(args.bss_calibration_bins)
    if num_bins < 3:
        raise ValueError("--bss_calibration_bins must be >= 3")
    weights = area_weights_from_lats(lats)
    if evaluation_mask is None:
        evaluation_mask = np.ones((len(lats), len(lons)), dtype=bool)
    fcst_clim = {}
    calibration_counts = {}
    print("🎯 Building forecast event climatology and BSS calibration reliability counts...")
    for variable in variables:
        shape = bundle_shape(thresholds[variable])
        sums = {"model": np.zeros(shape, dtype=np.float64), "geos": np.zeros(shape, dtype=np.float64)}
        count = np.zeros(shape, dtype=np.float64)
        spec = VARIABLES[variable]
        for year in years:
            if deadline_reached(deadline):
                raise TimeoutError("Soft runtime limit reached while building BSS calibration inputs.")
            ds = xr.open_zarr(os.path.join(forecast_dir, f"{year}.zarr"), consolidated=False, chunks=None)
            try:
                init_values = pd.to_datetime(ds["init"].values).normalize()
                lead_values = ds["lead"].values
                n_init = ds.sizes["init"]
                n_lead = ds.sizes["lead"]
                for init_idx, init_time in enumerate(init_values):
                    valid_values = valid_times_for_dataset(ds, init_idx, init_time, lead_values)
                    for lead_idx in range(n_lead):
                        lead_value = int(lead_values[lead_idx])
                        valid_time = pd.Timestamp(valid_values[lead_idx])
                        valid_season = season_name(int(valid_time.month))
                        threshold = select_grouped_map(thresholds[variable], valid_time)
                        obs = ds[spec["obs"]].isel(init=init_idx, lead=lead_idx).values
                        finite = np.isfinite(obs) & np.isfinite(threshold) & evaluation_mask
                        if not finite.any():
                            continue
                        event = obs >= threshold
                        count += finite.astype(np.float64)
                        for source, member_dim in (("model", "ensemble"), ("geos", "geos_member")):
                            ensemble = ds[spec[source]].isel(init=init_idx, lead=lead_idx).values
                            prob = np.nanmean(ensemble >= threshold[None, :, :], axis=0).astype(np.float64, copy=False)
                            sums[source] += np.where(finite, prob, 0.0)
                            key = (variable, source, int(year), lead_value, valid_season)
                            if key not in calibration_counts:
                                calibration_counts[key] = empty_calibration_count_state(num_bins)
                            update_calibration_count_state(
                                calibration_counts[key],
                                prob,
                                event,
                                finite,
                                weights,
                                num_bins,
                            )
            finally:
                ds.close()
        fcst_clim[variable] = {
            "model": np.where(count > 0, sums["model"] / count, np.nan).astype(np.float32),
            "geos": np.where(count > 0, sums["geos"] / count, np.nan).astype(np.float32),
        }
        print(
            f"   {variable}: mean model event p={float(np.nanmean(fcst_clim[variable]['model'])):.4f}, "
            f"GEOS event p={float(np.nanmean(fcst_clim[variable]['geos'])):.4f}"
        )
    return fcst_clim, calibration_counts


def calibration_candidate_groups(grouping, lead, season):
    if grouping == "global":
        return [(None, None, "global")]
    if grouping == "lead":
        return [(lead, None, "lead"), (None, None, "global")]
    return [
        (lead, season, "lead_season"),
        (lead, None, "lead"),
        (None, season, "season"),
        (None, None, "global"),
    ]


def aggregate_calibration_counts(calibration_counts, variable, source, train_years, lead, season, num_bins):
    selected = []
    train_years = set(int(y) for y in train_years)
    for (v, s, year, key_lead, key_season), state in calibration_counts.items():
        if v != variable or s != source or int(year) not in train_years:
            continue
        if lead is not None and int(key_lead) != int(lead):
            continue
        if season is not None and str(key_season) != str(season):
            continue
        selected.append(state)
    if not selected:
        return None
    return merge_calibration_count_states(selected, num_bins)


def fit_bss_calibration_models(years, variables, calibration_counts, args):
    models = {}
    rows = []
    method = args.bss_calibration
    if method != "logistic_cv":
        return models, pd.DataFrame(rows)

    num_bins = int(args.bss_calibration_bins)
    eps = float(args.epsilon_probability)
    all_years = [int(y) for y in years]
    for variable in variables:
        for source in ("model", "geos"):
            for holdout_year in all_years:
                train_years = [year for year in all_years if year != holdout_year]
                cv_mode = "leave_one_year_out"
                if not train_years:
                    train_years = all_years
                    cv_mode = "in_sample_single_year"
                for lead in LEADS:
                    for season in SEASONS:
                        fitted = None
                        group_used = None
                        for candidate_lead, candidate_season, candidate_group in calibration_candidate_groups(
                            args.bss_calibration_grouping,
                            lead,
                            season,
                        ):
                            counts = aggregate_calibration_counts(
                                calibration_counts,
                                variable,
                                source,
                                train_years,
                                candidate_lead,
                                candidate_season,
                                num_bins,
                            )
                            if counts is None:
                                continue
                            fitted = fit_logistic_from_binned_counts(
                                counts,
                                eps=eps,
                                ridge=float(args.bss_calibration_ridge),
                                min_weight=float(args.bss_calibration_min_weight),
                            )
                            if fitted is not None:
                                group_used = candidate_group
                                break
                        if fitted is None:
                            fitted = {
                                "method": "identity",
                                "intercept": 0.0,
                                "slope": 1.0,
                                "total_weight": 0.0,
                                "event_rate": np.nan,
                                "bins_used": 0,
                            }
                            group_used = "identity_fallback"
                        fitted = dict(fitted)
                        fitted.update(
                            {
                                "variable": variable,
                                "source": source,
                                "holdout_year": int(holdout_year),
                                "lead": int(lead),
                                "season": season,
                                "train_years": ",".join(str(y) for y in train_years),
                                "cv_mode": cv_mode,
                                "group_used": group_used,
                            }
                        )
                        models[(variable, source, int(holdout_year), int(lead), season)] = fitted
                        rows.append(fitted)

    table = pd.DataFrame(rows)
    if not table.empty:
        print(
            "🎚️ Fitted logistic BSS calibrators: "
            f"{len(table)} groups; method counts={table['method'].value_counts().to_dict()}"
        )
    return models, table


def get_bss_calibrator(calibration_models, variable, source, year, lead, season):
    return calibration_models.get((variable, source, int(year), int(lead), str(season)))


def group_keys_for_sample(subset, variable, valid_time, lead_value):
    valid_month = f"{int(valid_time.month):02d}"
    valid_season = season_name(int(valid_time.month))
    return [
        (subset, variable, "valid_season_lead", valid_season, int(lead_value)),
        (subset, variable, "valid_month_lead", valid_month, int(lead_value)),
    ]


def evaluate(
    forecast_dir,
    years,
    variables,
    thresholds,
    obs_clim,
    fcst_clim,
    calibration_models,
    lats,
    lons,
    args,
    evaluation_mask=None,
    deadline=None,
):
    weights = area_weights_from_lats(lats)
    scalar_states = {}
    spatial_states = {}
    shape = (len(lats), len(lons))
    if evaluation_mask is None:
        evaluation_mask = np.ones(shape, dtype=bool)
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
                valid_values = valid_times_for_dataset(ds, init_idx, init_time, lead_values)
                for lead_idx, lead_value in enumerate(lead_values):
                    valid_time = pd.Timestamp(valid_values[lead_idx])
                    valid_season = season_name(int(valid_time.month))
                    for variable in variables:
                        spec = VARIABLES[variable]
                        threshold = select_grouped_map(thresholds[variable], valid_time)
                        obs_event_freq = select_grouped_map(obs_clim[variable], valid_time)
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
                            calibration_method=args.bss_calibration,
                            calibrator=get_bss_calibrator(
                                calibration_models,
                                variable,
                                "model",
                                year,
                                int(lead_value),
                                valid_season,
                            ),
                        )
                        geos = ensemble_diagnostics(
                            geos_ens,
                            obs,
                            threshold,
                            obs_event_freq,
                            fcst_clim[variable]["geos"],
                            eps,
                            calibration_method=args.bss_calibration,
                            calibrator=get_bss_calibrator(
                                calibration_models,
                                variable,
                                "geos",
                                year,
                                int(lead_value),
                                valid_season,
                            ),
                        )
                        event_mask = obs >= threshold
                        masks = {
                            "all_data": evaluation_mask,
                            "extreme_events": event_mask & evaluation_mask,
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


def _add_bundle_to_dataset_parts(data_vars, coords, name, bundle, group_dim):
    bundle = _as_threshold_bundle(bundle)
    values = np.asarray(bundle["values"], dtype=np.float32)
    if values.ndim == 2:
        data_vars[name] = (("lat", "lon"), values.astype(np.float32))
        return
    group_values = [str(v) for v in bundle.get("group_values", threshold_group_values(bundle.get("grouping", "pooled")))]
    coords[group_dim] = group_values
    data_vars[name] = ((group_dim, "lat", "lon"), values.astype(np.float32))


def save_threshold_dataset(thresholds, obs_clim, fcst_clim, lats, lons, out_dir):
    data_vars = {}
    coords = {"lat": np.asarray(lats, dtype=np.float32), "lon": np.asarray(lons, dtype=np.float32)}
    for variable in thresholds:
        group_dim = f"{variable}_threshold_group"
        _add_bundle_to_dataset_parts(data_vars, coords, f"{variable}_threshold", thresholds[variable], group_dim)
        _add_bundle_to_dataset_parts(
            data_vars,
            coords,
            f"{variable}_obs_event_frequency",
            obs_clim[variable],
            group_dim,
        )
        data_vars[f"{variable}_model_event_frequency"] = (("lat", "lon"), fcst_clim[variable]["model"].astype(np.float32))
        data_vars[f"{variable}_geos_event_frequency"] = (("lat", "lon"), fcst_clim[variable]["geos"].astype(np.float32))
    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "description": "Event thresholds and event frequencies used by matrix evaluation suite.",
            "threshold_grouping_note": (
                "Grouped thresholds are selected by valid time before scoring. "
                "Forecast event frequencies are pooled over evaluated init/lead samples."
            ),
        },
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
    panel_width = 5.2 if MAP_CONTEXT["enabled"] else 4.4
    panel_height = 2.55 if MAP_CONTEXT["enabled"] else 2.0
    fig, axes = make_map_subplots(
        len(rows),
        len(LEADS),
        figsize=(panel_width * len(LEADS), max(panel_height * len(rows), 6.5)),
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
            plot_lons, plot_lats, plot_arr = prepare_spatial_field_for_plot(lons, lats, arr)
            mesh_kwargs = {}
            if MAP_CONTEXT["enabled"]:
                mesh_kwargs["transform"] = MAP_CONTEXT["data_crs"]
            last_mesh = ax.pcolormesh(
                plot_lons,
                plot_lats,
                plot_arr,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
                **mesh_kwargs,
            )
            add_map_overlays(ax, plot_lons, plot_lats)
            ax.set_title(f"{group_value} week{lead}", fontsize=9)
            ax.set_xlabel("lon", fontsize=8)
            ax.set_ylabel("lat", fontsize=8)
            if not MAP_CONTEXT["enabled"]:
                ax.set_xlim(float(np.nanmin(plot_lons)), float(np.nanmax(plot_lons)))
                ax.set_ylim(float(np.nanmin(plot_lats)), float(np.nanmax(plot_lats)))
                ax.tick_params(labelsize=7)
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


def build_lead_season_skill_table(summary):
    required = {
        "subset",
        "variable",
        "group_type",
        "group_value",
        "lead",
        "lead_label",
        "n_forecasts",
        "model_crps",
        "geos_crps",
        "crps_skill_pct",
        "model_rmse",
        "geos_rmse",
        "rmse_skill_pct",
        "model_calibrated_bss",
        "geos_calibrated_bss",
        "calibrated_bss_diff",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Cannot build lead-season skill table; summary is missing columns: {missing}")

    table = summary[summary["group_type"].eq("valid_season_lead")].copy()
    if table.empty:
        return table
    season_order = {season: i for i, season in enumerate(SEASONS)}
    subset_order = {subset: i for i, subset in enumerate(SUBSETS)}
    table["season"] = table["group_value"].astype(str)
    table["lead_week"] = table["lead"].astype(int)
    table["subset_order"] = table["subset"].map(subset_order).fillna(999).astype(int)
    table["season_order"] = table["season"].map(season_order).fillna(999).astype(int)
    table["ml_better_crps"] = table["crps_skill_pct"] > 0.0
    table["ml_better_rmse"] = table["rmse_skill_pct"] > 0.0
    table["ml_better_calibrated_bss"] = table["calibrated_bss_diff"] > 0.0
    table["ml_better_all_three"] = (
        table["ml_better_crps"] & table["ml_better_rmse"] & table["ml_better_calibrated_bss"]
    )
    table = table.rename(
        columns={
            "n_forecasts": "n_cases",
            "model_crps": "ml_crps",
            "model_rmse": "ml_rmse",
            "model_calibrated_bss": "ml_calibrated_bss",
            "geos_calibrated_bss": "geos_calibrated_bss",
            "crps_skill_pct": "crps_improvement_pct",
            "rmse_skill_pct": "rmse_improvement_pct",
            "calibrated_bss_diff": "calibrated_bss_gain",
        }
    )
    ordered_columns = [
        "variable",
        "subset",
        "season",
        "lead_week",
        "lead_label",
        "n_cases",
        "geos_crps",
        "ml_crps",
        "crps_improvement_pct",
        "geos_rmse",
        "ml_rmse",
        "rmse_improvement_pct",
        "geos_calibrated_bss",
        "ml_calibrated_bss",
        "calibrated_bss_gain",
        "ml_better_crps",
        "ml_better_rmse",
        "ml_better_calibrated_bss",
        "ml_better_all_three",
        "subset_order",
        "season_order",
    ]
    table = table[ordered_columns]
    table = table.sort_values(["variable", "subset_order", "season_order", "lead_week"]).reset_index(drop=True)
    return table.drop(columns=["subset_order", "season_order"])


def write_lead_season_skill_table(summary, out_dir):
    table = build_lead_season_skill_table(summary)
    table_path = os.path.join(out_dir, "lead_season_skill_table.csv")
    table.to_csv(table_path, index=False, float_format="%.6f")
    print(f"✅ Wrote lead-season skill table: {table_path}")
    if table.empty:
        print("⚠️ Lead-season skill table is empty.")
        return table_path

    print("\nLead-season skill summary")
    print(
        "  CRPS/RMSE improvement (%) = 100 * (1 - ML/GEOS), positive means ML lower error.\n"
        "  Calibrated BSS gain = ML calibrated BSS - GEOS calibrated BSS, positive means ML better."
    )
    display_columns = [
        "variable",
        "subset",
        "season",
        "lead_label",
        "n_cases",
        "geos_crps",
        "ml_crps",
        "crps_improvement_pct",
        "geos_rmse",
        "ml_rmse",
        "rmse_improvement_pct",
        "geos_calibrated_bss",
        "ml_calibrated_bss",
        "calibrated_bss_gain",
        "ml_better_all_three",
    ]
    display = table[display_columns].copy()
    with pd.option_context("display.max_rows", 200, "display.width", 220):
        print(display.to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    print()
    return table_path


def main():
    args = parse_args()
    years = [year for year in range(args.start_year, args.end_year + 1) if year not in parse_years(args.skip_years)]
    variables = parse_variables(args.variables)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.make_plots:
        configure_map_context(args)
    deadline = time.monotonic() + args.max_runtime_minutes * 60.0 if args.max_runtime_minutes else None

    summary_path = os.path.join(args.out_dir, "matrix_summary_metrics.csv")
    spatial_path = os.path.join(args.out_dir, "matrix_spatial_metrics.nc")
    metadata_path = os.path.join(args.out_dir, "matrix_eval_metadata.json")
    calibration_path = os.path.join(args.out_dir, "bss_calibration_params.csv")
    if os.path.exists(summary_path) and (not args.make_plots or os.path.exists(spatial_path)) and not args.overwrite:
        print(f"✅ Existing matrix evaluation found: {summary_path}")
        summary = pd.read_csv(summary_path)
        write_lead_season_skill_table(summary, args.out_dir)
        if args.make_plots:
            spatial = xr.open_dataset(spatial_path)
            make_scalar_matrix_plots(summary, args.out_dir)
            make_spatial_plots(spatial, args.out_dir)
            spatial.close()
        return
    if os.path.exists(summary_path) and args.make_plots and not os.path.exists(spatial_path) and not args.overwrite:
        print(
            f"♻️ Existing summary found, but spatial plots were requested and {spatial_path} is missing. "
            "Recomputing full matrix evaluation."
        )

    thresholds, obs_clim, lats, lons = collect_obs_thresholds(args.forecast_dir, years, variables, args)
    first_forecast = os.path.join(args.forecast_dir, f"{years[0]}.zarr")
    grid_ds = xr.open_zarr(first_forecast, consolidated=False, chunks=None)
    try:
        forecast_lats = np.asarray(grid_ds["lat"].values)
        forecast_lons = np.asarray(grid_ds["lon"].values)
    finally:
        grid_ds.close()
    if len(forecast_lats) != len(lats) or len(forecast_lons) != len(lons):
        raise ValueError(
            f"Threshold grid shape {(len(lats), len(lons))} does not match forecast grid "
            f"{(len(forecast_lats), len(forecast_lons))}."
        )
    if not (np.allclose(forecast_lats, lats, equal_nan=True) and np.allclose(forecast_lons, lons, equal_nan=True)):
        raise ValueError(
            "Threshold latitude/longitude coordinates do not match the forecast Zarr grid. "
            "Rebuild thresholds on the same grid before evaluating."
        )
    evaluation_mask, evaluation_mask_source = load_evaluation_mask(args, lats, lons)
    if deadline_reached(deadline):
        raise TimeoutError("Soft runtime limit reached after threshold pass.")
    fcst_clim, calibration_counts = collect_forecast_calibration_inputs(
        args.forecast_dir,
        years,
        variables,
        thresholds,
        lats,
        lons,
        args,
        evaluation_mask=evaluation_mask,
        deadline=deadline,
    )
    calibration_models, calibration_table = fit_bss_calibration_models(
        years,
        variables,
        calibration_counts,
        args,
    )
    if not calibration_table.empty:
        calibration_table.to_csv(calibration_path, index=False, float_format="%.8f")
        print(f"✅ Wrote BSS calibration parameters: {calibration_path}")
    save_threshold_dataset(thresholds, obs_clim, fcst_clim, lats, lons, args.out_dir)
    summary, spatial = evaluate(
        args.forecast_dir,
        years,
        variables,
        thresholds,
        obs_clim,
        fcst_clim,
        calibration_models,
        lats,
        lons,
        args,
        evaluation_mask=evaluation_mask,
        deadline=deadline,
    )

    summary.to_csv(summary_path, index=False, float_format="%.6f")
    spatial.to_netcdf(spatial_path)
    lead_season_table_path = write_lead_season_skill_table(summary, args.out_dir)
    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "years": years,
        "variables": variables,
        "subsets": SUBSETS,
        "group_types": GROUP_TYPES,
        "extreme_definition": (
            "obs >= local observed climatological threshold map selected by valid time; "
            "thresholds should preferably come from long-term observations."
        ),
        "threshold_file": os.path.abspath(args.threshold_file) if args.threshold_file else None,
        "threshold_forecast_dir": os.path.abspath(args.threshold_forecast_dir or args.forecast_dir),
        "threshold_years": None
        if args.threshold_file
        else [
            year
            for year in range(
                int(args.threshold_start_year if args.threshold_start_year is not None else min(years)),
                int(args.threshold_end_year if args.threshold_end_year is not None else max(years)) + 1,
            )
            if year not in parse_years(args.threshold_skip_years)
        ],
        "threshold_grouping": {
            variable: thresholds[variable].get("grouping", "pooled")
            for variable in variables
        },
        "extreme_quantile_pr": args.extreme_quantile_pr,
        "extreme_quantile_t2m": args.extreme_quantile_t2m,
        "pr_min_threshold": args.pr_min_threshold,
        "eval_mask": args.eval_mask,
        "land_mask_file": os.path.abspath(args.land_mask_file) if args.land_mask_file else None,
        "evaluation_mask_source": evaluation_mask_source,
        "calibrated_bss": (
            "leave-one-year-out logistic reliability calibration from area-weighted binned counts"
            if args.bss_calibration == "logistic_cv"
            else args.bss_calibration
        ),
        "bss_calibration": args.bss_calibration,
        "bss_calibration_grouping": args.bss_calibration_grouping,
        "bss_calibration_bins": args.bss_calibration_bins,
        "bss_calibration_ridge": args.bss_calibration_ridge,
        "bss_calibration_min_weight": args.bss_calibration_min_weight,
        "bss_calibration_params": os.path.abspath(calibration_path) if os.path.exists(calibration_path) else None,
        "map_features": args.map_features,
        "county_boundaries": args.county_boundaries,
        "cartopy_enabled": bool(MAP_CONTEXT["enabled"]),
        "cartopy_feature_count": int(len(MAP_CONTEXT["features"])),
        "lead_season_skill_table": os.path.abspath(lead_season_table_path),
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
