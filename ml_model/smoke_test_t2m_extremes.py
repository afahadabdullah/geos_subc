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
        "lat_min": 45.0,
        "lat_max": 53.0,
        "lon_min": 236.0,
        "lon_max": 246.0,
        "source_note": "Late-June 2021 Pacific Northwest heat wave",
    },
    {
        "name": "sicily_heatwave_2021",
        "title": "Sicily Heatwave",
        "event_date": "2021-08-11",
        "lat_min": 35.0,
        "lat_max": 39.5,
        "lon_min": 12.0,
        "lon_max": 18.0,
        "source_note": "11 Aug 2021 Sicily / central Mediterranean heat",
    },
    {
        "name": "rajasthan_heatwave_2020",
        "title": "West Rajasthan Heatwave",
        "event_date": "2020-05-26",
        "lat_min": 26.0,
        "lat_max": 31.0,
        "lon_min": 72.0,
        "lon_max": 78.0,
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
    return xr.open_zarr(path, consolidated=False)


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


def plot_event_panel(ax, event: EventSpec, lead_labels: Sequence[str], ml_series: np.ndarray, geos_series: np.ndarray, obs_series: np.ndarray, init_date: pd.Timestamp, chosen_lead: int):
    x = np.arange(1, len(lead_labels) + 1)
    ml_stats = summarize_ensemble(ml_series)
    geos_stats = summarize_ensemble(geos_series)

    ax.fill_between(x, ml_stats["p10"], ml_stats["p90"], color="#c6dbef", alpha=0.45, label="ML p10-p90")
    ax.fill_between(x, ml_stats["p25"], ml_stats["p75"], color="#9ecae1", alpha=0.45, label="ML p25-p75")
    ax.plot(x, ml_stats["mean"], color="#08519c", linewidth=2.5, marker="o", label="ML mean")
    ax.plot(x, ml_stats["p50"], color="#2171b5", linewidth=1.8, marker="o", linestyle="--", label="ML q50")

    ax.fill_between(x, geos_stats["p10"], geos_stats["p90"], color="#fdd0a2", alpha=0.35, label="GEOS p10-p90")
    ax.fill_between(x, geos_stats["p25"], geos_stats["p75"], color="#fdae6b", alpha=0.35, label="GEOS p25-p75")
    ax.plot(x, geos_stats["mean"], color="#d94801", linewidth=2.0, marker="s", label="GEOS mean")
    ax.plot(x, geos_stats["p50"], color="#f16913", linewidth=1.6, marker="s", linestyle="--", label="GEOS q50")

    ax.plot(x, obs_series, color="#2b2b2b", linewidth=2.2, marker="D", linestyle="--", label="Obs")

    ax.axvline(chosen_lead, color="#7f7f7f", linestyle=":", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lead_labels)
    ax.set_title(
        f"{event.title}\nEvent={event.event_date.strftime('%Y-%m-%d')}  Init={init_date.strftime('%Y-%m-%d')}  Focus lead=W{chosen_lead}",
        fontsize=11,
    )
    ax.set_ylabel("Regional Mean T2M [K]")
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
        "lead_week",
        "ml_mean_k",
        "ml_p05_k",
        "ml_p10_k",
        "ml_p25_k",
        "ml_p50_k",
        "ml_p75_k",
        "ml_p90_k",
        "ml_p95_k",
        "geos_mean_k",
        "geos_p05_k",
        "geos_p10_k",
        "geos_p25_k",
        "geos_p50_k",
        "geos_p75_k",
        "geos_p90_k",
        "geos_p95_k",
        "obs_k",
        "ml_abs_err_mean_k",
        "geos_abs_err_mean_k",
        "ml_abs_err_p50_k",
        "geos_abs_err_p50_k",
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
    ml_stats: Dict[str, np.ndarray],
    geos_stats: Dict[str, np.ndarray],
    obs_series: np.ndarray,
) -> List[str]:
    lines = [
        f"[{event.name}] {event.title}",
        f"  Event date: {event.event_date.strftime('%Y-%m-%d')} | target init: {target_init.strftime('%Y-%m-%d')} | chosen init: {chosen_init.strftime('%Y-%m-%d')} | slip={slip_days}d | focus=W{focus_lead}",
    ]
    for lead_idx in range(len(obs_series)):
        obs = float(obs_series[lead_idx])
        ml_mean = float(ml_stats["mean"][lead_idx])
        geos_mean = float(geos_stats["mean"][lead_idx])
        ml_p50 = float(ml_stats["p50"][lead_idx])
        geos_p50 = float(geos_stats["p50"][lead_idx])
        ml_err_mean = abs(ml_mean - obs)
        geos_err_mean = abs(geos_mean - obs)
        ml_err_p50 = abs(ml_p50 - obs)
        geos_err_p50 = abs(geos_p50 - obs)
        mean_winner = pick_winner(ml_err_mean, geos_err_mean)
        p50_winner = pick_winner(ml_err_p50, geos_err_p50)
        lines.append(
            "  "
            f"W{lead_idx + 1}: obs={obs:.2f} K | "
            f"ML mean={ml_mean:.2f}, q10/q50/q90={float(ml_stats['p10'][lead_idx]):.2f}/{ml_p50:.2f}/{float(ml_stats['p90'][lead_idx]):.2f} | "
            f"GEOS mean={geos_mean:.2f}, q10/q50/q90={float(geos_stats['p10'][lead_idx]):.2f}/{geos_p50:.2f}/{float(geos_stats['p90'][lead_idx]):.2f} | "
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
        f"Focus lead W{focus_lead}: obs in ML q10-q90={within_interval(float(obs_series[focus_idx]), float(ml_stats['p10'][focus_idx]), float(ml_stats['p90'][focus_idx]))}, "
        f"obs in GEOS q10-q90={within_interval(float(obs_series[focus_idx]), float(geos_stats['p10'][focus_idx]), float(geos_stats['p90'][focus_idx]))}"
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
    os.makedirs(args.output_dir, exist_ok=True)

    events = build_event_specs(args.event_names)
    figure_path = os.path.join(args.output_dir, "t2m_extreme_smoke_tests.png")
    csv_path = os.path.join(args.output_dir, "t2m_extreme_smoke_tests.csv")
    report_path = os.path.join(args.output_dir, "t2m_extreme_smoke_tests.txt")

    ds_cache: Dict[Tuple[str, int], xr.Dataset] = {}
    summary_rows: List[Dict[str, object]] = []
    report_lines: List[str] = []
    fig, axes = plt.subplots(len(events), 1, figsize=(12, 4.5 * len(events)), constrained_layout=True)
    if len(events) == 1:
        axes = [axes]

    try:
        for ax, event in zip(axes, events):
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
            ml_series = ml_series[:, :lead_count]
            geos_series = geos_series[:, :lead_count]
            obs_series = obs_series[:lead_count]

            ml_leads = ml_leads[:lead_count]
            geos_leads = geos_leads[:lead_count]
            if len(ml_leads) != len(geos_leads) or not np.array_equal(np.asarray(ml_leads), np.asarray(geos_leads)):
                raise ValueError(f"Lead coordinate alignment failed for {event.name}.")

            lead_labels = [f"W{i}" for i in range(1, lead_count + 1)]
            focus_lead = infer_event_lead(event.event_date, chosen_init)
            focus_lead = min(focus_lead, lead_count)

            plot_event_panel(
                ax=ax,
                event=event,
                lead_labels=lead_labels,
                ml_series=ml_series,
                geos_series=geos_series,
                obs_series=obs_series,
                init_date=chosen_init,
                chosen_lead=focus_lead,
            )

            ml_stats = summarize_ensemble(ml_series)
            geos_stats = summarize_ensemble(geos_series)
            event_report = build_event_report(
                event=event,
                target_init=target_init,
                chosen_init=chosen_init,
                slip_days=slip_days,
                focus_lead=focus_lead,
                ml_stats=ml_stats,
                geos_stats=geos_stats,
                obs_series=obs_series,
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
                        "lead_week": lead_idx + 1,
                        "ml_mean_k": float(ml_stats["mean"][lead_idx]),
                        "ml_p05_k": float(ml_stats["p05"][lead_idx]),
                        "ml_p10_k": float(ml_stats["p10"][lead_idx]),
                        "ml_p25_k": float(ml_stats["p25"][lead_idx]),
                        "ml_p50_k": float(ml_stats["p50"][lead_idx]),
                        "ml_p75_k": float(ml_stats["p75"][lead_idx]),
                        "ml_p90_k": float(ml_stats["p90"][lead_idx]),
                        "ml_p95_k": float(ml_stats["p95"][lead_idx]),
                        "geos_mean_k": float(geos_stats["mean"][lead_idx]),
                        "geos_p05_k": float(geos_stats["p05"][lead_idx]),
                        "geos_p10_k": float(geos_stats["p10"][lead_idx]),
                        "geos_p25_k": float(geos_stats["p25"][lead_idx]),
                        "geos_p50_k": float(geos_stats["p50"][lead_idx]),
                        "geos_p75_k": float(geos_stats["p75"][lead_idx]),
                        "geos_p90_k": float(geos_stats["p90"][lead_idx]),
                        "geos_p95_k": float(geos_stats["p95"][lead_idx]),
                        "obs_k": obs_value,
                        "ml_abs_err_mean_k": ml_err_mean,
                        "geos_abs_err_mean_k": geos_err_mean,
                        "ml_abs_err_p50_k": ml_err_p50,
                        "geos_abs_err_p50_k": geos_err_p50,
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

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.01))
        fig.suptitle("Smoke Test: Regional Mean T2M for 2020-2021 Extreme Heat Events", fontsize=15, y=1.04)
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
