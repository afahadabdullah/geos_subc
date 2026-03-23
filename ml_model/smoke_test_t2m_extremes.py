#!/usr/bin/env python3
"""
Smoke-test regional T2M forecasts for a few well-known 2020-2021 heat events.

This script compares three data sources:
1. ML generated ensemble forecasts from ``dataprocess/gen_multiv1/<year>.zarr``
2. Raw GEOS ensemble forecasts from ``geos_subc_<year>.zarr``
3. Weekly observed T2M from ``t2m_weekly_<year>.zarr``

For each event, the script:
- picks the init date nearest to ``event_date - 21 days``
- extracts all four weekly lead forecasts from that init
- computes a latitude-weighted regional mean T2M
- plots ML, GEOS, and observation time series over weeks 1-4
- optionally evaluates anomaly versions using either historical obs climatology
  or saved weekly climatology stores

The goal is quick smoke testing, not a full verification suite.
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml


EVENTS = [
    {
        "name": "pnw_heat_dome_2021",
        "title": "Pacific Northwest Heat Dome",
        "event_date": "2021-06-29",
        "lat_min": 46.0,
        "lat_max": 50.5,
        "lon_min": 239.0,
        "lon_max": 244.0,
        "source_note": "Late-June 2021 Pacific Northwest heat wave",
    },
    {
        "name": "sicily_heatwave_2021",
        "title": "Sicily Heatwave",
        "event_date": "2021-08-11",
        "lat_min": 36.0,
        "lat_max": 38.5,
        "lon_min": 13.0,
        "lon_max": 16.5,
        "source_note": "11 Aug 2021 Sicily / central Mediterranean heat",
    },
    {
        "name": "rajasthan_heatwave_2020",
        "title": "West Rajasthan Heatwave",
        "event_date": "2020-05-26",
        "lat_min": 27.0,
        "lat_max": 29.5,
        "lon_min": 73.0,
        "lon_max": 76.0,
        "source_note": "26 May 2020 Churu / West Rajasthan extreme heat",
    },
]


@dataclass(frozen=True)
class EventSpec:
    name: str
    title: str
    event_date: pd.Timestamp
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    source_note: str


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test T2M extreme-event regional means from ML, GEOS, and obs data.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing geos_subc_<year>.zarr and t2m_weekly_<year>.zarr. Defaults to config data_dir.",
    )
    parser.add_argument(
        "--ml_dir",
        type=str,
        default="dataprocess/gen_multiv1",
        help="Directory containing ML generated yearly zarr stores like 2020.zarr and 2021.zarr.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_output_flowmulti/smoke_t2m_extremes",
        help="Directory to save the smoke-test plot and CSV summary.",
    )
    parser.add_argument(
        "--init_offset_days",
        type=int,
        default=21,
        help="Target init date offset before the event. The script uses the nearest available init date.",
    )
    parser.add_argument(
        "--max_init_slip_days",
        type=int,
        default=10,
        help="Maximum allowed difference between the desired and chosen init date.",
    )
    parser.add_argument(
        "--event_names",
        nargs="*",
        default=None,
        help="Optional subset of built-in event names to run.",
    )
    parser.add_argument(
        "--fair_member_count",
        type=int,
        default=4,
        help="Downsample ML to this many members for fairer spread/quantile comparison against GEOS.",
    )
    parser.add_argument(
        "--clim_start_year",
        type=int,
        default=None,
        help="Start year for regional obs climatology. Defaults to config train_start_year.",
    )
    parser.add_argument(
        "--clim_end_year",
        type=int,
        default=None,
        help="End year for regional obs climatology. Defaults to config train_end_year.",
    )
    parser.add_argument(
        "--clim_max_init_slip_days",
        type=int,
        default=4,
        help="Maximum allowed init-date mismatch when aligning historical climatology years by month-day.",
    )
    parser.add_argument(
        "--anomaly_mode",
        choices=["obs_hist", "obs_store", "system_store"],
        default="obs_hist",
        help="Use raw historical obs climatology (`obs_hist`), saved obs weekly climatology (`obs_store`), or saved system-specific weekly climatologies (`system_store`) for anomaly evaluation.",
    )
    parser.add_argument(
        "--ml_clim_path",
        type=str,
        default="dataprocess/clim/ml_weekly_ensmean_clim_1999_2021.zarr",
        help="Weekly ML climatology path used when anomaly_mode=system_store.",
    )
    parser.add_argument(
        "--geos_clim_path",
        type=str,
        default="dataprocess/clim/geos_weekly_ensmean_clim_1999_2021.zarr",
        help="Weekly GEOS climatology path used when anomaly_mode=system_store.",
    )
    parser.add_argument(
        "--obs_clim_path",
        type=str,
        default="dataprocess/clim/obs_weekly_clim_1999_2021.zarr",
        help="Weekly OBS climatology path used when anomaly_mode=obs_store or anomaly_mode=system_store.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def choose_name(items: Iterable[str], candidates: Sequence[str], label: str) -> str:
    item_set = set(items)
    for name in candidates:
        if name in item_set:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {sorted(item_set)}")


def choose_data_var(ds: xr.Dataset, candidates: Sequence[str], label: str) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {list(ds.data_vars)}")


def normalize_lon_value(lon: float) -> float:
    return lon % 360.0


def normalize_lon_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.mod(arr, 360.0)


def build_event_specs(event_names: Sequence[str] = None) -> List[EventSpec]:
    selected = []
    allowed = {item["name"] for item in EVENTS}
    requested = set(event_names) if event_names else None
    if requested is not None:
        unknown = requested - allowed
        if unknown:
            raise ValueError(f"Unknown --event_names: {sorted(unknown)}")
    for item in EVENTS:
        if requested is not None and item["name"] not in requested:
            continue
        selected.append(
            EventSpec(
                name=item["name"],
                title=item["title"],
                event_date=pd.Timestamp(item["event_date"]),
                lat_min=float(item["lat_min"]),
                lat_max=float(item["lat_max"]),
                lon_min=float(item["lon_min"]),
                lon_max=float(item["lon_max"]),
                source_note=item["source_note"],
            )
        )
    return selected


def open_year_dataset(path: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_zarr(path, consolidated=False, chunks=None)


def infer_layout(ds: xr.Dataset, kind: str) -> Dict[str, str]:
    s_dim = choose_name(ds.dims, ["S", "time", "init_time"], f"{kind} init dimension")
    lead_dim = choose_name(ds.dims, ["L", "lead", "lead_time"], f"{kind} lead dimension")
    y_dim = choose_name(set(ds.dims) | set(ds.coords), ["Y", "latitude", "lat", "y"], f"{kind} latitude dimension")
    x_dim = choose_name(set(ds.dims) | set(ds.coords), ["X", "longitude", "lon", "x"], f"{kind} longitude dimension")
    member_dim = None
    if "M" in ds.dims:
        member_dim = "M"
    else:
        for candidate in ["member", "ensemble", "ensemble_member"]:
            if candidate in ds.dims:
                member_dim = candidate
                break
    tas_var = choose_data_var(ds, ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"], f"{kind} tas variable")
    return {
        "s_dim": s_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "member_dim": member_dim,
        "tas_var": tas_var,
    }


def infer_clim_layout(ds: xr.Dataset, kind: str) -> Dict[str, str]:
    week_dim = choose_name(ds.dims, ["init_week", "week", "W"], f"{kind} init-week dimension")
    lead_dim = choose_name(ds.dims, ["L", "lead", "lead_time"], f"{kind} lead dimension")
    y_dim = choose_name(set(ds.dims) | set(ds.coords), ["Y", "latitude", "lat", "y"], f"{kind} latitude dimension")
    x_dim = choose_name(set(ds.dims) | set(ds.coords), ["X", "longitude", "lon", "x"], f"{kind} longitude dimension")
    tas_var = choose_data_var(ds, ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"], f"{kind} tas variable")
    n_init_var = "n_init_tas" if "n_init_tas" in ds.data_vars else None
    return {
        "week_dim": week_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "tas_var": tas_var,
        "n_init_var": n_init_var,
    }


def nearest_init_index(s_values: np.ndarray, target_date: pd.Timestamp, max_slip_days: int) -> Tuple[int, pd.Timestamp, int]:
    s_dates = pd.to_datetime(s_values).normalize()
    deltas = np.abs((s_dates - target_date.normalize()).days)
    idx = int(np.argmin(deltas))
    chosen = pd.Timestamp(s_dates[idx])
    slip_days = int(deltas[idx])
    if slip_days > max_slip_days:
        raise ValueError(
            f"Nearest init date {chosen.strftime('%Y-%m-%d')} is {slip_days} days away from target "
            f"{target_date.strftime('%Y-%m-%d')}, exceeding the allowed {max_slip_days} days."
        )
    return idx, chosen, slip_days


def exact_init_index(s_values: np.ndarray, target_date: pd.Timestamp) -> int:
    s_dates = pd.to_datetime(s_values).normalize()
    matches = np.where(s_dates == target_date.normalize())[0]
    if len(matches) == 0:
        raise ValueError(f"Could not find init date {target_date.strftime('%Y-%m-%d')} in the dataset.")
    return int(matches[0])


def replace_year_safe(date: pd.Timestamp, year: int) -> pd.Timestamp:
    try:
        return date.replace(year=year)
    except ValueError:
        return date.replace(year=year, day=min(date.day, 28))


def downsample_members(series: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"fair member count must be positive, got {count}")
    n_members = int(series.shape[0])
    if n_members <= count:
        return np.asarray(series, dtype=np.float64)

    raw_idx = np.linspace(0, n_members - 1, count)
    idx = np.round(raw_idx).astype(int)
    idx = np.clip(idx, 0, n_members - 1)

    seen = set()
    unique_idx = []
    for item in idx.tolist():
        if item not in seen:
            unique_idx.append(item)
            seen.add(item)
    if len(unique_idx) < count:
        for item in range(n_members):
            if item not in seen:
                unique_idx.append(item)
                seen.add(item)
            if len(unique_idx) == count:
                break
    return np.asarray(series[unique_idx], dtype=np.float64)


def infer_event_lead(event_date: pd.Timestamp, init_date: pd.Timestamp) -> int:
    delta_days = max(0, int((event_date.normalize() - init_date.normalize()).days))
    return max(1, min(4, delta_days // 7 + 1))


def get_coord_values(da: xr.DataArray, coord_name: str) -> np.ndarray:
    if coord_name in da.coords:
        return np.asarray(da[coord_name].values)
    return np.arange(da.sizes[coord_name], dtype=np.float64)


def region_mask(lat_vals: np.ndarray, lon_vals: np.ndarray, event: EventSpec) -> np.ndarray:
    lats = np.asarray(lat_vals, dtype=np.float64)
    lons = normalize_lon_array(np.asarray(lon_vals, dtype=np.float64))
    lon_min = normalize_lon_value(event.lon_min)
    lon_max = normalize_lon_value(event.lon_max)

    lat_mask_1d = (lats >= event.lat_min) & (lats <= event.lat_max)
    if lon_min <= lon_max:
        lon_mask_1d = (lons >= lon_min) & (lons <= lon_max)
    else:
        lon_mask_1d = (lons >= lon_min) | (lons <= lon_max)

    mask = np.outer(lat_mask_1d, lon_mask_1d)
    if not np.any(mask):
        raise ValueError(
            f"Region for {event.name} selected zero grid cells. "
            f"Lat range=({event.lat_min}, {event.lat_max}), lon range=({event.lon_min}, {event.lon_max})."
        )
    return mask


def weighted_region_mean(data: np.ndarray, lat_vals: np.ndarray, lon_vals: np.ndarray, event: EventSpec) -> np.ndarray:
    """
    data shape:
    - ensemble forecasts: [M, L, Y, X]
    - obs: [L, Y, X]
    """
    lat_vals = np.asarray(lat_vals, dtype=np.float64)
    lon_vals = np.asarray(lon_vals, dtype=np.float64)
    mask = region_mask(lat_vals, lon_vals, event)
    lat_weights = np.cos(np.deg2rad(lat_vals)).astype(np.float64)
    lat_weights = np.clip(lat_weights, 0.0, None)
    weights_2d = lat_weights[:, None] * mask.astype(np.float64)
    valid_weight_sum = float(weights_2d.sum())
    if valid_weight_sum <= 0.0:
        raise ValueError(f"Region weights are zero for {event.name}.")

    if data.ndim == 4:
        weighted = data * weights_2d[None, None, :, :]
        return np.nansum(weighted, axis=(-2, -1)) / valid_weight_sum
    if data.ndim == 3:
        weighted = data * weights_2d[None, :, :]
        return np.nansum(weighted, axis=(-2, -1)) / valid_weight_sum
    raise ValueError(f"Unsupported data rank {data.ndim}; expected 3 or 4.")


def extract_event_series(ds: xr.Dataset, layout: Dict[str, str], event: EventSpec, init_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = ds[layout["tas_var"]].isel({layout["s_dim"]: init_idx})

    if layout["member_dim"] is None:
        da = da.transpose(layout["lead_dim"], layout["y_dim"], layout["x_dim"])
        lat_vals = get_coord_values(da, layout["y_dim"])
        lon_vals = get_coord_values(da, layout["x_dim"])
        series = weighted_region_mean(np.asarray(da.values), lat_vals, lon_vals, event)
        series = np.asarray(series, dtype=np.float64)[None, :]
    else:
        da = da.transpose(layout["member_dim"], layout["lead_dim"], layout["y_dim"], layout["x_dim"])
        lat_vals = get_coord_values(da, layout["y_dim"])
        lon_vals = get_coord_values(da, layout["x_dim"])
        series = weighted_region_mean(np.asarray(da.values), lat_vals, lon_vals, event)

    lead_vals = get_coord_values(da, layout["lead_dim"])
    return np.asarray(series, dtype=np.float64), np.asarray(lead_vals), np.asarray(lat_vals)


def extract_obs_series(ds: xr.Dataset, layout: Dict[str, str], init_idx: int, event: EventSpec) -> np.ndarray:
    da = ds[layout["tas_var"]].isel({layout["s_dim"]: init_idx}).transpose(layout["lead_dim"], layout["y_dim"], layout["x_dim"])
    lat_vals = get_coord_values(da, layout["y_dim"])
    lon_vals = get_coord_values(da, layout["x_dim"])
    return np.asarray(weighted_region_mean(np.asarray(da.values), lat_vals, lon_vals, event), dtype=np.float64)


def compute_obs_climatology(
    ds_cache: Dict[Tuple[str, int], xr.Dataset],
    data_dir: str,
    event: EventSpec,
    reference_init: pd.Timestamp,
    lead_count: int,
    clim_start_year: int,
    clim_end_year: int,
    clim_max_init_slip_days: int,
) -> Dict[str, np.ndarray]:
    clim_samples = []
    slip_days = []
    used_years = []
    skipped_years = []

    for year in range(clim_start_year, clim_end_year + 1):
        obs_path = os.path.join(data_dir, f"t2m_weekly_{year}.zarr")
        if not os.path.exists(obs_path):
            skipped_years.append(f"{year}:missing_file")
            continue
        obs_ds = get_or_open_dataset(ds_cache, ("clim_obs", year), obs_path)
        obs_layout = infer_layout(obs_ds, "CLIM OBS")
        target_init = replace_year_safe(reference_init, year)
        try:
            init_idx, _, slip = nearest_init_index(
                obs_ds[obs_layout["s_dim"]].values,
                target_date=target_init,
                max_slip_days=clim_max_init_slip_days,
            )
        except ValueError:
            skipped_years.append(f"{year}:init_mismatch")
            continue
        obs_series = extract_obs_series(obs_ds, obs_layout, init_idx, event)[:lead_count]
        clim_samples.append(obs_series)
        slip_days.append(slip)
        used_years.append(year)

    if not clim_samples:
        raise ValueError(
            f"Could not build climatology for {event.name}. "
            f"No valid years remained between {clim_start_year} and {clim_end_year}. "
            f"Skipped={skipped_years}"
        )

    clim_arr = np.asarray(clim_samples, dtype=np.float64)
    return {
        "mean": np.nanmean(clim_arr, axis=0),
        "p90": np.nanpercentile(clim_arr, 90, axis=0),
        "samples": clim_arr,
        "used_years": np.asarray(used_years, dtype=np.int32),
        "skipped_years": skipped_years,
        "mean_slip_days": float(np.mean(slip_days)) if slip_days else 0.0,
        "max_slip_days": int(np.max(slip_days)) if slip_days else 0,
    }


def summarize_ensemble(series: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(series, axis=0),
        "p05": np.nanpercentile(series, 5, axis=0),
        "p10": np.nanpercentile(series, 10, axis=0),
        "p25": np.nanpercentile(series, 25, axis=0),
        "p50": np.nanpercentile(series, 50, axis=0),
        "p75": np.nanpercentile(series, 75, axis=0),
        "p90": np.nanpercentile(series, 90, axis=0),
        "p95": np.nanpercentile(series, 95, axis=0),
        "min": np.nanmin(series, axis=0),
        "max": np.nanmax(series, axis=0),
    }


def exceedance_probabilities(series: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    threshold = np.asarray(threshold, dtype=np.float64)
    return 100.0 * np.mean(series > threshold[None, :], axis=0)


def anomaly_mode_label(mode: str) -> str:
    if mode == "obs_hist":
        return "Obs-Historical Anomaly"
    if mode == "obs_store":
        return "Obs-Weekly-Climatology Anomaly"
    if mode == "system_store":
        return "System-Weekly-Climatology Anomaly"
    raise ValueError(f"Unsupported anomaly mode: {mode}")


def extract_store_climatology_mean(ds: xr.Dataset, layout: Dict[str, str], event: EventSpec, init_date: pd.Timestamp, lead_count: int) -> Dict[str, object]:
    week_idx = max(0, min(52, int(init_date.isocalendar().week) - 1))
    da = ds[layout["tas_var"]].isel({layout["week_dim"]: week_idx, layout["lead_dim"]: slice(0, lead_count)})
    da = da.transpose(layout["lead_dim"], layout["y_dim"], layout["x_dim"])
    lat_vals = get_coord_values(da, layout["y_dim"])
    lon_vals = get_coord_values(da, layout["x_dim"])
    mean_series = np.asarray(weighted_region_mean(np.asarray(da.values), lat_vals, lon_vals, event), dtype=np.float64)
    n_init = None
    if layout["n_init_var"] is not None:
        n_init = int(np.asarray(ds[layout["n_init_var"]].isel({layout["week_dim"]: week_idx}).values).item())
    return {
        "mean": mean_series,
        "init_week": week_idx + 1,
        "n_init": n_init,
    }


def compute_anomaly_context(
    args,
    ds_cache: Dict[Tuple[str, object], xr.Dataset],
    data_dir: str,
    event: EventSpec,
    chosen_init: pd.Timestamp,
    lead_count: int,
    clim_start_year: int,
    clim_end_year: int,
) -> Dict[str, object]:
    obs_hist = compute_obs_climatology(
        ds_cache=ds_cache,
        data_dir=data_dir,
        event=event,
        reference_init=chosen_init,
        lead_count=lead_count,
        clim_start_year=clim_start_year,
        clim_end_year=clim_end_year,
        clim_max_init_slip_days=args.clim_max_init_slip_days,
    )

    obs_clim_mean = np.asarray(obs_hist["mean"][:lead_count], dtype=np.float64)
    ml_clim_mean = np.asarray(obs_clim_mean, dtype=np.float64)
    geos_clim_mean = np.asarray(obs_clim_mean, dtype=np.float64)
    obs_store_meta = None
    ml_store_meta = None
    geos_store_meta = None

    if args.anomaly_mode in {"obs_store", "system_store"}:
        obs_clim_ds = get_or_open_dataset(ds_cache, ("obs_clim_store", args.obs_clim_path), args.obs_clim_path)
        obs_clim_layout = infer_clim_layout(obs_clim_ds, "OBS CLIM")
        obs_store_meta = extract_store_climatology_mean(obs_clim_ds, obs_clim_layout, event, chosen_init, lead_count)
        obs_clim_mean = np.asarray(obs_store_meta["mean"], dtype=np.float64)
        ml_clim_mean = np.asarray(obs_clim_mean, dtype=np.float64)
        geos_clim_mean = np.asarray(obs_clim_mean, dtype=np.float64)

    if args.anomaly_mode == "system_store":
        ml_clim_ds = get_or_open_dataset(ds_cache, ("ml_clim_store", args.ml_clim_path), args.ml_clim_path)
        geos_clim_ds = get_or_open_dataset(ds_cache, ("geos_clim_store", args.geos_clim_path), args.geos_clim_path)
        ml_store_meta = extract_store_climatology_mean(ml_clim_ds, infer_clim_layout(ml_clim_ds, "ML CLIM"), event, chosen_init, lead_count)
        geos_store_meta = extract_store_climatology_mean(geos_clim_ds, infer_clim_layout(geos_clim_ds, "GEOS CLIM"), event, chosen_init, lead_count)
        ml_clim_mean = np.asarray(ml_store_meta["mean"], dtype=np.float64)
        geos_clim_mean = np.asarray(geos_store_meta["mean"], dtype=np.float64)

    return {
        "mode": args.anomaly_mode,
        "mode_label": anomaly_mode_label(args.anomaly_mode),
        "obs_hist": obs_hist,
        "obs_clim_mean": obs_clim_mean,
        "ml_clim_mean": ml_clim_mean,
        "geos_clim_mean": geos_clim_mean,
        "obs_clim_p90": np.asarray(obs_hist["p90"][:lead_count], dtype=np.float64),
        "obs_store_meta": obs_store_meta,
        "ml_store_meta": ml_store_meta,
        "geos_store_meta": geos_store_meta,
    }


def plot_event_panel(
    ax,
    event: EventSpec,
    lead_labels: Sequence[str],
    ml_series: np.ndarray,
    geos_series: np.ndarray,
    obs_series: np.ndarray,
    init_date: pd.Timestamp,
    chosen_lead: int,
    panel_title: str,
    ylabel: str,
    clim_mean: np.ndarray = None,
    ml_full_mean: np.ndarray = None,
    zero_line: bool = False,
):
    x = np.arange(1, len(lead_labels) + 1)
    ml_stats = summarize_ensemble(ml_series)
    geos_stats = summarize_ensemble(geos_series)

    ax.fill_between(x, ml_stats["p10"], ml_stats["p90"], color="#c6dbef", alpha=0.45, label="ML-4 p10-p90")
    ax.fill_between(x, ml_stats["p25"], ml_stats["p75"], color="#9ecae1", alpha=0.45, label="ML-4 p25-p75")
    ax.plot(x, ml_stats["mean"], color="#08519c", linewidth=2.5, marker="o", label="ML-4 mean")
    ax.plot(x, ml_stats["p50"], color="#2171b5", linewidth=1.8, marker="o", linestyle="--", label="ML-4 q50")
    if ml_full_mean is not None:
        ax.plot(x, ml_full_mean, color="#08306b", linewidth=1.1, linestyle=":", label="ML-all mean")

    ax.fill_between(x, geos_stats["p10"], geos_stats["p90"], color="#fdd0a2", alpha=0.35, label="GEOS p10-p90")
    ax.fill_between(x, geos_stats["p25"], geos_stats["p75"], color="#fdae6b", alpha=0.35, label="GEOS p25-p75")
    ax.plot(x, geos_stats["mean"], color="#d94801", linewidth=2.0, marker="s", label="GEOS mean")
    ax.plot(x, geos_stats["p50"], color="#f16913", linewidth=1.6, marker="s", linestyle="--", label="GEOS q50")

    ax.plot(x, obs_series, color="#2b2b2b", linewidth=2.2, marker="D", linestyle="--", label="Obs")
    if clim_mean is not None:
        ax.plot(x, clim_mean, color="#636363", linewidth=1.4, linestyle="-.", label="Clim mean")
    if zero_line:
        ax.axhline(0.0, color="#636363", linewidth=1.0, linestyle="--")

    ax.axvline(chosen_lead, color="#7f7f7f", linestyle=":", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lead_labels)
    ax.set_title(
        f"{event.title} ({panel_title})\nEvent={event.event_date.strftime('%Y-%m-%d')}  Init={init_date.strftime('%Y-%m-%d')}  Focus lead=W{chosen_lead}",
        fontsize=11,
    )
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)


def write_summary_csv(path: str, rows: List[Dict[str, object]]):
    fieldnames = [
        "event_name",
        "event_title",
        "source_note",
        "event_date",
        "target_init_date",
        "chosen_init_date",
        "init_slip_days",
        "focus_lead_week",
        "region_box",
        "anomaly_mode",
        "fair_member_count",
        "ml_full_member_count",
        "geos_member_count",
        "clim_year_count",
        "clim_mean_init_slip_days",
        "clim_max_init_slip_days",
        "obs_clim_init_count",
        "ml_clim_init_count",
        "geos_clim_init_count",
        "lead_week",
        "clim_mean_k",
        "obs_clim_mean_k",
        "ml_clim_mean_k",
        "geos_clim_mean_k",
        "clim_p90_k",
        "obs_anom_k",
        "ml_mean_k",
        "ml_p05_k",
        "ml_p10_k",
        "ml_p25_k",
        "ml_p50_k",
        "ml_p75_k",
        "ml_p90_k",
        "ml_p95_k",
        "ml_mean_anom_k",
        "ml_p50_anom_k",
        "geos_mean_k",
        "geos_p05_k",
        "geos_p10_k",
        "geos_p25_k",
        "geos_p50_k",
        "geos_p75_k",
        "geos_p90_k",
        "geos_p95_k",
        "geos_mean_anom_k",
        "geos_p50_anom_k",
        "obs_k",
        "ml_abs_err_mean_k",
        "geos_abs_err_mean_k",
        "ml_abs_err_p50_k",
        "geos_abs_err_p50_k",
        "winner_mean_fair",
        "winner_p50_fair",
        "ml_prob_gt_obs_pct",
        "geos_prob_gt_obs_pct",
        "ml_prob_gt_clim_p90_pct",
        "geos_prob_gt_clim_p90_pct",
        "winner_mean",
        "winner_p50",
        "ml_contains_obs_p10_p90",
        "geos_contains_obs_p10_p90",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pick_winner(value_a: float, value_b: float, label_a: str = "ML", label_b: str = "GEOS", tol: float = 1e-6) -> str:
    if abs(value_a - value_b) <= tol:
        return "Tie"
    return label_a if value_a < value_b else label_b


def within_interval(value: float, lo: float, hi: float) -> bool:
    return bool(lo <= value <= hi)


def build_event_report(
    event: EventSpec,
    target_init: pd.Timestamp,
    chosen_init: pd.Timestamp,
    slip_days: int,
    focus_lead: int,
    region_box: str,
    ml_stats: Dict[str, np.ndarray],
    geos_stats: Dict[str, np.ndarray],
    obs_clim_mean: np.ndarray,
    ml_clim_mean: np.ndarray,
    geos_clim_mean: np.ndarray,
    clim_p90: np.ndarray,
    obs_series: np.ndarray,
    obs_series_anom: np.ndarray,
    ml_stats_anom: Dict[str, np.ndarray],
    geos_stats_anom: Dict[str, np.ndarray],
    ml_prob_gt_obs: np.ndarray,
    geos_prob_gt_obs: np.ndarray,
    ml_prob_gt_p90: np.ndarray,
    geos_prob_gt_p90: np.ndarray,
    ml_full_mean: np.ndarray,
    anomaly_mode_label_text: str,
) -> List[str]:
    lines = [
        f"[{event.name}] {event.title}",
        f"  Event date: {event.event_date.strftime('%Y-%m-%d')} | target init: {target_init.strftime('%Y-%m-%d')} | chosen init: {chosen_init.strftime('%Y-%m-%d')} | slip={slip_days}d | focus=W{focus_lead} | box={region_box} | anomaly={anomaly_mode_label_text}",
    ]
    for lead_idx in range(len(obs_series)):
        obs = float(obs_series[lead_idx])
        obs_clim = float(obs_clim_mean[lead_idx])
        ml_clim = float(ml_clim_mean[lead_idx])
        geos_clim = float(geos_clim_mean[lead_idx])
        obs_anom = float(obs_series_anom[lead_idx])
        ml_mean = float(ml_stats["mean"][lead_idx])
        geos_mean = float(geos_stats["mean"][lead_idx])
        ml_p50 = float(ml_stats["p50"][lead_idx])
        geos_p50 = float(geos_stats["p50"][lead_idx])
        ml_mean_anom = float(ml_stats_anom["mean"][lead_idx])
        geos_mean_anom = float(geos_stats_anom["mean"][lead_idx])
        ml_err_mean = abs(ml_mean - obs)
        geos_err_mean = abs(geos_mean - obs)
        ml_err_p50 = abs(ml_p50 - obs)
        geos_err_p50 = abs(geos_p50 - obs)
        mean_winner = pick_winner(ml_err_mean, geos_err_mean)
        p50_winner = pick_winner(ml_err_p50, geos_err_p50)
        lines.append(
            "  "
            f"W{lead_idx + 1}: obs={obs:.2f} K (anom={obs_anom:+.2f}, obs_clim={obs_clim:.2f}, ml_clim={ml_clim:.2f}, geos_clim={geos_clim:.2f}, p90={float(clim_p90[lead_idx]):.2f}) | "
            f"ML-4 mean={ml_mean:.2f}, q10/q50/q90={float(ml_stats['p10'][lead_idx]):.2f}/{ml_p50:.2f}/{float(ml_stats['p90'][lead_idx]):.2f}, "
            f"mean_anom={ml_mean_anom:+.2f}, "
            f"P>Tobs={float(ml_prob_gt_obs[lead_idx]):.1f}%, P>Tp90={float(ml_prob_gt_p90[lead_idx]):.1f}% | "
            f"GEOS mean={geos_mean:.2f}, q10/q50/q90={float(geos_stats['p10'][lead_idx]):.2f}/{geos_p50:.2f}/{float(geos_stats['p90'][lead_idx]):.2f}, "
            f"mean_anom={geos_mean_anom:+.2f}, "
            f"P>Tobs={float(geos_prob_gt_obs[lead_idx]):.1f}%, P>Tp90={float(geos_prob_gt_p90[lead_idx]):.1f}% | "
            f"winner(mean)={mean_winner}, winner(q50)={p50_winner}"
        )
    focus_idx = focus_lead - 1
    ml_focus_mean_err = abs(float(ml_stats["mean"][focus_idx]) - float(obs_series[focus_idx]))
    geos_focus_mean_err = abs(float(geos_stats["mean"][focus_idx]) - float(obs_series[focus_idx]))
    ml_focus_p50_err = abs(float(ml_stats["p50"][focus_idx]) - float(obs_series[focus_idx]))
    geos_focus_p50_err = abs(float(geos_stats["p50"][focus_idx]) - float(obs_series[focus_idx]))
    lines.append(
        "  "
        f"Focus lead W{focus_lead}: winner(mean)={pick_winner(ml_focus_mean_err, geos_focus_mean_err)} "
        f"({ml_focus_mean_err:.2f}K vs {geos_focus_mean_err:.2f}K), "
        f"winner(q50)={pick_winner(ml_focus_p50_err, geos_focus_p50_err)} "
        f"({ml_focus_p50_err:.2f}K vs {geos_focus_p50_err:.2f}K)"
    )
    lines.append(
        "  "
        f"Focus lead W{focus_lead}: ML-all mean={float(ml_full_mean[focus_idx]):.2f} K "
        f"vs ML-4 mean={float(ml_stats['mean'][focus_idx]):.2f} K vs GEOS mean={float(geos_stats['mean'][focus_idx]):.2f} K"
    )
    lines.append(
        "  "
        f"Focus lead W{focus_lead}: obs in ML q10-q90={within_interval(float(obs_series[focus_idx]), float(ml_stats['p10'][focus_idx]), float(ml_stats['p90'][focus_idx]))}, "
        f"obs in GEOS q10-q90={within_interval(float(obs_series[focus_idx]), float(geos_stats['p10'][focus_idx]), float(geos_stats['p90'][focus_idx]))}"
    )
    lines.append(
        "  "
        f"Focus lead W{focus_lead}: P(ML-4 > obs)={float(ml_prob_gt_obs[focus_idx]):.1f}%, "
        f"P(GEOS > obs)={float(geos_prob_gt_obs[focus_idx]):.1f}%, "
        f"P(ML-4 > clim p90)={float(ml_prob_gt_p90[focus_idx]):.1f}%, "
        f"P(GEOS > clim p90)={float(geos_prob_gt_p90[focus_idx]):.1f}%"
    )
    return lines


def get_or_open_dataset(cache: Dict[Tuple[str, int], xr.Dataset], key: Tuple[str, int], path: str) -> xr.Dataset:
    ds = cache.get(key)
    if ds is None:
        ds = open_year_dataset(path)
        cache[key] = ds
    return ds


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or config["data_dir"]
    clim_start_year = int(args.clim_start_year if args.clim_start_year is not None else config.get("train_start_year", 1999))
    clim_end_year = int(args.clim_end_year if args.clim_end_year is not None else config.get("train_end_year", 2019))
    os.makedirs(args.output_dir, exist_ok=True)

    events = build_event_specs(args.event_names)
    figure_path = os.path.join(args.output_dir, "t2m_extreme_smoke_tests.png")
    csv_path = os.path.join(args.output_dir, "t2m_extreme_smoke_tests.csv")
    report_path = os.path.join(args.output_dir, "t2m_extreme_smoke_tests.txt")

    ds_cache: Dict[Tuple[str, object], xr.Dataset] = {}
    summary_rows: List[Dict[str, object]] = []
    report_lines: List[str] = []
    fig, axes = plt.subplots(len(events), 2, figsize=(18, 4.8 * len(events)), constrained_layout=True)
    if len(events) == 1:
        axes = np.asarray([axes])

    try:
        for axis_row, event in zip(axes, events):
            ax_raw, ax_anom = axis_row
            year = int(event.event_date.year)
            ml_path = os.path.join(args.ml_dir, f"{year}.zarr")
            geos_path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
            obs_path = os.path.join(data_dir, f"t2m_weekly_{year}.zarr")

            ml_ds = get_or_open_dataset(ds_cache, ("ml", year), ml_path)
            geos_ds = get_or_open_dataset(ds_cache, ("geos", year), geos_path)
            obs_ds = get_or_open_dataset(ds_cache, ("obs", year), obs_path)

            ml_layout = infer_layout(ml_ds, "ML")
            geos_layout = infer_layout(geos_ds, "GEOS")
            obs_layout = infer_layout(obs_ds, "OBS")

            target_init = event.event_date - pd.Timedelta(days=args.init_offset_days)
            init_idx, chosen_init, slip_days = nearest_init_index(
                ml_ds[ml_layout["s_dim"]].values,
                target_date=target_init,
                max_slip_days=args.max_init_slip_days,
            )
            geos_init_idx = exact_init_index(geos_ds[geos_layout["s_dim"]].values, chosen_init)
            obs_init_idx = exact_init_index(obs_ds[obs_layout["s_dim"]].values, chosen_init)

            ml_series, ml_leads, _ = extract_event_series(ml_ds, ml_layout, event, init_idx)
            geos_series, geos_leads, _ = extract_event_series(geos_ds, geos_layout, event, geos_init_idx)
            obs_series = extract_obs_series(obs_ds, obs_layout, obs_init_idx, event)

            lead_count = min(ml_series.shape[1], geos_series.shape[1], obs_series.shape[0], 4)
            ml_series_full = ml_series[:, :lead_count]
            geos_series = geos_series[:, :lead_count]
            obs_series = obs_series[:lead_count]

            ml_leads = ml_leads[:lead_count]
            geos_leads = geos_leads[:lead_count]
            if len(ml_leads) != len(geos_leads) or not np.array_equal(np.asarray(ml_leads), np.asarray(geos_leads)):
                raise ValueError(f"Lead coordinate alignment failed for {event.name}.")

            ml_series = downsample_members(ml_series_full, args.fair_member_count)
            ml_full_mean = np.nanmean(ml_series_full, axis=0)

            anomaly_context = compute_anomaly_context(
                args=args,
                ds_cache=ds_cache,
                data_dir=data_dir,
                event=event,
                chosen_init=chosen_init,
                lead_count=lead_count,
                clim_start_year=clim_start_year,
                clim_end_year=clim_end_year,
            )
            climatology = anomaly_context["obs_hist"]
            obs_clim_mean = np.asarray(anomaly_context["obs_clim_mean"], dtype=np.float64)
            ml_clim_mean = np.asarray(anomaly_context["ml_clim_mean"], dtype=np.float64)
            geos_clim_mean = np.asarray(anomaly_context["geos_clim_mean"], dtype=np.float64)
            clim_p90 = np.asarray(anomaly_context["obs_clim_p90"], dtype=np.float64)

            ml_series_anom = ml_series - ml_clim_mean[None, :]
            geos_series_anom = geos_series - geos_clim_mean[None, :]
            obs_series_anom = obs_series - obs_clim_mean
            ml_full_mean_anom = ml_full_mean - ml_clim_mean

            lead_labels = [f"W{i}" for i in range(1, lead_count + 1)]
            focus_lead = infer_event_lead(event.event_date, chosen_init)
            focus_lead = min(focus_lead, lead_count)
            region_box = f"lat[{event.lat_min:.1f},{event.lat_max:.1f}] lon[{event.lon_min:.1f},{event.lon_max:.1f}]"

            plot_event_panel(
                ax=ax_raw,
                event=event,
                lead_labels=lead_labels,
                ml_series=ml_series,
                geos_series=geos_series,
                obs_series=obs_series,
                init_date=chosen_init,
                chosen_lead=focus_lead,
                panel_title="raw",
                ylabel="Regional Mean T2M [K]",
                clim_mean=obs_clim_mean,
                ml_full_mean=ml_full_mean,
            )
            plot_event_panel(
                ax=ax_anom,
                event=event,
                lead_labels=lead_labels,
                ml_series=ml_series_anom,
                geos_series=geos_series_anom,
                obs_series=obs_series_anom,
                init_date=chosen_init,
                chosen_lead=focus_lead,
                panel_title=f"anomaly ({anomaly_context['mode_label']})",
                ylabel="Regional Mean T2M Anomaly [K]",
                ml_full_mean=ml_full_mean_anom,
                zero_line=True,
            )

            ml_stats = summarize_ensemble(ml_series)
            geos_stats = summarize_ensemble(geos_series)
            ml_stats_anom = summarize_ensemble(ml_series_anom)
            geos_stats_anom = summarize_ensemble(geos_series_anom)
            ml_prob_gt_obs = exceedance_probabilities(ml_series, obs_series)
            geos_prob_gt_obs = exceedance_probabilities(geos_series, obs_series)
            ml_prob_gt_p90 = exceedance_probabilities(ml_series, clim_p90)
            geos_prob_gt_p90 = exceedance_probabilities(geos_series, clim_p90)

            event_report = build_event_report(
                event=event,
                target_init=target_init,
                chosen_init=chosen_init,
                slip_days=slip_days,
                focus_lead=focus_lead,
                region_box=region_box,
                ml_stats=ml_stats,
                geos_stats=geos_stats,
                obs_clim_mean=obs_clim_mean,
                ml_clim_mean=ml_clim_mean,
                geos_clim_mean=geos_clim_mean,
                clim_p90=clim_p90,
                obs_series=obs_series,
                obs_series_anom=obs_series_anom,
                ml_stats_anom=ml_stats_anom,
                geos_stats_anom=geos_stats_anom,
                ml_prob_gt_obs=ml_prob_gt_obs,
                geos_prob_gt_obs=geos_prob_gt_obs,
                ml_prob_gt_p90=ml_prob_gt_p90,
                geos_prob_gt_p90=geos_prob_gt_p90,
                ml_full_mean=ml_full_mean,
                anomaly_mode_label_text=anomaly_context["mode_label"],
            )
            for line in event_report:
                print(line)
            report_lines.extend(event_report)
            report_lines.append("")

            for lead_idx in range(lead_count):
                obs_value = float(obs_series[lead_idx])
                ml_err_mean = abs(float(ml_stats["mean"][lead_idx]) - obs_value)
                geos_err_mean = abs(float(geos_stats["mean"][lead_idx]) - obs_value)
                ml_err_p50 = abs(float(ml_stats["p50"][lead_idx]) - obs_value)
                geos_err_p50 = abs(float(geos_stats["p50"][lead_idx]) - obs_value)
                summary_rows.append(
                    {
                        "event_name": event.name,
                        "event_title": event.title,
                        "source_note": event.source_note,
                        "event_date": event.event_date.strftime("%Y-%m-%d"),
                        "target_init_date": target_init.strftime("%Y-%m-%d"),
                        "chosen_init_date": chosen_init.strftime("%Y-%m-%d"),
                        "init_slip_days": slip_days,
                        "focus_lead_week": focus_lead,
                        "region_box": region_box,
                        "anomaly_mode": anomaly_context["mode"],
                        "fair_member_count": int(ml_series.shape[0]),
                        "ml_full_member_count": int(ml_series_full.shape[0]),
                        "geos_member_count": int(geos_series.shape[0]),
                        "clim_year_count": int(len(climatology["used_years"])),
                        "clim_mean_init_slip_days": float(climatology["mean_slip_days"]),
                        "clim_max_init_slip_days": int(climatology["max_slip_days"]),
                        "obs_clim_init_count": anomaly_context["obs_store_meta"]["n_init"] if anomaly_context["obs_store_meta"] is not None else "",
                        "ml_clim_init_count": anomaly_context["ml_store_meta"]["n_init"] if anomaly_context["ml_store_meta"] is not None else "",
                        "geos_clim_init_count": anomaly_context["geos_store_meta"]["n_init"] if anomaly_context["geos_store_meta"] is not None else "",
                        "lead_week": lead_idx + 1,
                        "clim_mean_k": float(obs_clim_mean[lead_idx]),
                        "obs_clim_mean_k": float(obs_clim_mean[lead_idx]),
                        "ml_clim_mean_k": float(ml_clim_mean[lead_idx]),
                        "geos_clim_mean_k": float(geos_clim_mean[lead_idx]),
                        "clim_p90_k": float(clim_p90[lead_idx]),
                        "obs_anom_k": float(obs_series_anom[lead_idx]),
                        "ml_mean_k": float(ml_stats["mean"][lead_idx]),
                        "ml_p05_k": float(ml_stats["p05"][lead_idx]),
                        "ml_p10_k": float(ml_stats["p10"][lead_idx]),
                        "ml_p25_k": float(ml_stats["p25"][lead_idx]),
                        "ml_p50_k": float(ml_stats["p50"][lead_idx]),
                        "ml_p75_k": float(ml_stats["p75"][lead_idx]),
                        "ml_p90_k": float(ml_stats["p90"][lead_idx]),
                        "ml_p95_k": float(ml_stats["p95"][lead_idx]),
                        "ml_mean_anom_k": float(ml_stats_anom["mean"][lead_idx]),
                        "ml_p50_anom_k": float(ml_stats_anom["p50"][lead_idx]),
                        "geos_mean_k": float(geos_stats["mean"][lead_idx]),
                        "geos_p05_k": float(geos_stats["p05"][lead_idx]),
                        "geos_p10_k": float(geos_stats["p10"][lead_idx]),
                        "geos_p25_k": float(geos_stats["p25"][lead_idx]),
                        "geos_p50_k": float(geos_stats["p50"][lead_idx]),
                        "geos_p75_k": float(geos_stats["p75"][lead_idx]),
                        "geos_p90_k": float(geos_stats["p90"][lead_idx]),
                        "geos_p95_k": float(geos_stats["p95"][lead_idx]),
                        "geos_mean_anom_k": float(geos_stats_anom["mean"][lead_idx]),
                        "geos_p50_anom_k": float(geos_stats_anom["p50"][lead_idx]),
                        "obs_k": obs_value,
                        "ml_abs_err_mean_k": ml_err_mean,
                        "geos_abs_err_mean_k": geos_err_mean,
                        "ml_abs_err_p50_k": ml_err_p50,
                        "geos_abs_err_p50_k": geos_err_p50,
                        "winner_mean_fair": pick_winner(ml_err_mean, geos_err_mean),
                        "winner_p50_fair": pick_winner(ml_err_p50, geos_err_p50),
                        "ml_prob_gt_obs_pct": float(ml_prob_gt_obs[lead_idx]),
                        "geos_prob_gt_obs_pct": float(geos_prob_gt_obs[lead_idx]),
                        "ml_prob_gt_clim_p90_pct": float(ml_prob_gt_p90[lead_idx]),
                        "geos_prob_gt_clim_p90_pct": float(geos_prob_gt_p90[lead_idx]),
                        "winner_mean": pick_winner(ml_err_mean, geos_err_mean),
                        "winner_p50": pick_winner(ml_err_p50, geos_err_p50),
                        "ml_contains_obs_p10_p90": within_interval(
                            obs_value,
                            float(ml_stats["p10"][lead_idx]),
                            float(ml_stats["p90"][lead_idx]),
                        ),
                        "geos_contains_obs_p10_p90": within_interval(
                            obs_value,
                            float(geos_stats["p10"][lead_idx]),
                            float(geos_stats["p90"][lead_idx]),
                        ),
                    }
                )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle(f"Smoke Test: Regional Mean T2M for 2020-2021 Extreme Heat Events ({anomaly_mode_label(args.anomaly_mode)})", fontsize=15, y=1.04)
        fig.savefig(figure_path, dpi=170, bbox_inches="tight")
        write_summary_csv(csv_path, summary_rows)
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines).rstrip() + "\n")
        print(f"✅ Saved smoke-test plot: {figure_path}")
        print(f"✅ Saved smoke-test summary: {csv_path}")
        print(f"✅ Saved smoke-test report: {report_path}")
    finally:
        plt.close(fig)
        for ds in ds_cache.values():
            ds.close()


if __name__ == "__main__":
    main()
