#!/usr/bin/env python3
"""
Single-event case-study plots from generated flow_finalv1_global forecast Zarrs.

Default preset: June 19, 2021 Europe heat-wave event.
Additional presets include major European and western North American heat
events from 2021-2024.

The script reads a saved yearly forecast Zarr, selects init dates around
the preset target dates, then compares lead weeks 1-4 T2M:
  - observed ERA5 T2M
  - GEOS ensemble-mean forecast
  - ML ensemble-mean forecast

Outputs include per-init/lead maps, a composite map, a compact lead-tracking
time series, and a CSV of area-weighted T2M metrics over the requested domain.
The event date/window is used for annotation, so lead weeks outside the target
date are still plotted for manual event tracking.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


EVENT_PRESETS = {
    "europe_jun2021_heatwave": {
        "year": 2021,
        "event_name": "June 19 2021 Europe heat wave",
        "target_inits": "2021-05-27",
        "lead_weeks": "1,2,3,4",
        "event_start": "2021-06-19",
        "event_end": "2021-06-19",
        "lat_min": 35.0,
        "lat_max": 72.0,
        "lon_min": -12.0,
        "lon_max": 45.0,
        "out_dir": (
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "case_jun2021_europe_heatwave_endmay_leads1_4"
        ),
    },
    "med_aug2021_sicily_record_heat": {
        "year": 2021,
        "event_name": "August 11 2021 Mediterranean/Sicily record heat",
        "target_inits": "2021-07-15,2021-07-22,2021-07-29,2021-08-05",
        "lead_weeks": "1,2,3,4",
        "event_start": "2021-08-11",
        "event_end": "2021-08-11",
        "lat_min": 30.0,
        "lat_max": 48.0,
        "lon_min": -10.0,
        "lon_max": 35.0,
        "out_dir": (
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "case_aug2021_mediterranean_sicily_record_heat_leads1_4"
        ),
    },
    "europe_jul2022_heatwave_mortality": {
        "year": 2022,
        "event_name": "July 8-19 2022 Europe heat waves",
        "target_inits": "2022-06-16,2022-06-23,2022-06-30,2022-07-07",
        "lead_weeks": "1,2,3,4",
        "event_start": "2022-07-08",
        "event_end": "2022-07-19",
        "lat_min": 35.0,
        "lat_max": 72.0,
        "lon_min": -12.0,
        "lon_max": 45.0,
        "out_dir": (
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "case_jul2022_europe_heatwave_mortality_leads1_4"
        ),
    },
    "pnw_jun2021_heat_dome": {
        "year": 2021,
        "event_name": "June 28-29 2021 Pacific Northwest heat dome",
        "target_inits": "2021-06-03,2021-06-10,2021-06-17,2021-06-24",
        "lead_weeks": "1,2,3,4",
        "event_start": "2021-06-28",
        "event_end": "2021-06-29",
        "lat_min": 40.0,
        "lat_max": 55.0,
        "lon_min": -130.0,
        "lon_max": -110.0,
        "out_dir": (
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "case_jun2021_pnw_heat_dome_leads1_4"
        ),
    },
    "death_valley_jul2024_extreme_heat": {
        "year": 2024,
        "event_name": "July 7 2024 Death Valley/Furnace Creek extreme heat",
        "target_inits": "2024-06-13,2024-06-20,2024-06-27,2024-07-04",
        "lead_weeks": "1,2,3,4",
        "event_start": "2024-07-07",
        "event_end": "2024-07-07",
        "lat_min": 30.0,
        "lat_max": 42.0,
        "lon_min": -125.0,
        "lon_max": -110.0,
        "out_dir": (
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "case_jul2024_death_valley_extreme_heat_leads1_4"
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one T2M event case from generated global forecast Zarrs.")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Directory containing YEAR.zarr forecast stores.",
    )
    parser.add_argument(
        "--event_preset",
        choices=sorted(EVENT_PRESETS),
        default=None,
        help="Named event preset. Defaults to europe_jun2021_heatwave if event fields are not supplied.",
    )
    parser.add_argument("--list_event_presets", action="store_true", help="Print available event presets and exit.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--event_name", type=str, default=None)
    parser.add_argument(
        "--target_inits",
        type=str,
        default=None,
        help="Comma-separated target init dates. The closest available init is used for each.",
    )
    parser.add_argument(
        "--lead_weeks",
        type=str,
        default=None,
        help="Comma-separated lead weeks to plot for each selected init.",
    )
    parser.add_argument("--event_start", type=str, default=None)
    parser.add_argument("--event_end", type=str, default=None)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument(
        "--lon_min",
        type=float,
        default=None,
        help="Domain western longitude. Use 0..360 or negative degrees; default -12 = 12W.",
    )
    parser.add_argument(
        "--lon_max",
        type=float,
        default=None,
        help="Domain eastern longitude. Use 0..360 or negative degrees; default 45E.",
    )
    parser.add_argument("--max_cases", type=int, default=16)
    parser.add_argument(
        "--temperature_units",
        choices=("C", "K"),
        default="C",
        help="Map/time-series display units. Metrics are unaffected except labels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_event_preset(args):
    if args.list_event_presets:
        print(json.dumps(EVENT_PRESETS, indent=2))
        sys.exit(0)
    preset_name = args.event_preset or "europe_jun2021_heatwave"
    preset = EVENT_PRESETS[preset_name]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.event_preset = preset_name
    return args


def normalize_lon(lon):
    lon = float(lon)
    if lon < 0:
        lon = lon + 360.0
    return lon % 360.0


def select_domain(ds, lat_min, lat_max, lon_min, lon_max):
    lat0, lat1 = sorted([float(lat_min), float(lat_max)])
    raw_lon_min = float(lon_min)
    raw_lon_max = float(lon_max)
    lon0 = normalize_lon(raw_lon_min)
    lon1 = normalize_lon(raw_lon_max)
    use_signed_plot_lons = raw_lon_min < 0.0 or raw_lon_max < 0.0 or lon0 > lon1
    out = ds.sel(lat=slice(lat0, lat1))
    if lon0 <= lon1:
        out = out.sel(lon=slice(lon0, lon1))
    else:
        west = out.sel(lon=slice(lon0, 360.0))
        east = out.sel(lon=slice(0.0, lon1))
        out = xr.concat([west, east], dim="lon")
    if use_signed_plot_lons:
        plot_lons = xr.where(out["lon"] > 180.0, out["lon"] - 360.0, out["lon"])
        out = out.assign_coords(lon=plot_lons).sortby("lon")
    if out.sizes.get("lat", 0) == 0 or out.sizes.get("lon", 0) == 0:
        raise ValueError(
            f"Selected domain is empty: lat={lat_min}..{lat_max}, lon={lon_min}..{lon_max}"
        )
    return out


def area_weights(lats):
    weights = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64)))
    weights = np.clip(weights, 0.0, None)
    return weights[:, None]


def weighted_mean(field, weights):
    field = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(field)
    weighted_mask = np.where(finite, weights, 0.0)
    denom = float(np.sum(weighted_mask))
    if denom <= 0:
        return np.nan
    return float(np.sum(np.where(finite, field, 0.0) * weighted_mask) / denom)


def weighted_rmse(pred, obs, weights):
    return float(np.sqrt(weighted_mean((np.asarray(pred) - np.asarray(obs)) ** 2, weights)))


def weighted_bias(pred, obs, weights):
    return weighted_mean(np.asarray(pred) - np.asarray(obs), weights)


def weighted_crps(ensemble, obs, weights):
    ensemble = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    finite = np.isfinite(obs) & np.all(np.isfinite(ensemble), axis=0)
    weighted_mask = np.where(finite, weights, 0.0)
    denom = float(np.sum(weighted_mask))
    if denom <= 0:
        return np.nan
    mae_term = np.mean(np.abs(ensemble - obs[None, :, :]), axis=0)
    ens_sorted = np.sort(ensemble, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    crps_map = mae_term - spread_term
    return float(np.sum(np.where(finite, crps_map, 0.0) * weighted_mask) / denom)


def field_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": np.nan, "mean": np.nan, "max": np.nan}
    return {
        "min": float(np.nanmin(finite)),
        "mean": float(np.nanmean(finite)),
        "max": float(np.nanmax(finite)),
    }


def to_display_temp(values, units):
    values = np.asarray(values, dtype=np.float64)
    if units == "C":
        return values - 273.15
    return values


def percentile_limits(fields, lower=2, upper=98, symmetric=False):
    finite_chunks = [np.ravel(np.asarray(f)[np.isfinite(f)]) for f in fields if np.isfinite(f).any()]
    if not finite_chunks:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    values = np.concatenate(finite_chunks)
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    lo, hi = np.nanpercentile(values, [lower, upper])
    if symmetric:
        vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
        return -vmax, vmax
    if abs(float(hi) - float(lo)) < 1e-12:
        pad = max(abs(float(hi)) * 0.05, 1e-6)
        return float(lo) - pad, float(hi) + pad
    return float(lo), float(hi)


def plot_panel(ax, lons, lats, field, title, cmap, vmin=None, vmax=None):
    mesh = ax.pcolormesh(lons, lats, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
    ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
    return mesh


def parse_date_list(value):
    dates = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        dates.append(pd.Timestamp(item).normalize())
    if not dates:
        raise ValueError("At least one target init date is required.")
    return dates


def parse_int_list(value):
    ints = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        ints.append(int(item))
    if not ints:
        raise ValueError("At least one lead week is required.")
    return ints


def selected_init_infos(ds, target_init_dates):
    init_values = pd.to_datetime(ds["init"].values).normalize()
    infos = []
    used_indices = set()
    for target in target_init_dates:
        day_offsets = np.abs((init_values - target).days)
        init_idx = int(np.argmin(day_offsets))
        if init_idx in used_indices:
            continue
        used_indices.add(init_idx)
        infos.append(
            {
                "requested_init": pd.Timestamp(target),
                "init_idx": init_idx,
                "init": pd.Timestamp(init_values[init_idx]),
                "init_offset_days": int((init_values[init_idx] - target).days),
            }
        )
    return infos


def find_cases(ds, target_init_dates, lead_weeks, event_start, event_end, max_cases):
    event_start = pd.Timestamp(event_start).normalize()
    event_end = pd.Timestamp(event_end).normalize()
    event_center = event_start + (event_end - event_start) / 2
    lead_values = np.asarray(ds["lead"].values)
    lead_lookup = {int(lead): lead_idx for lead_idx, lead in enumerate(lead_values)}
    missing_leads = [lead for lead in lead_weeks if int(lead) not in lead_lookup]
    if missing_leads:
        raise ValueError(f"Requested lead weeks are not available: {missing_leads}. Available: {lead_values.tolist()}")

    cases = []
    for init_info in selected_init_infos(ds, target_init_dates):
        init_cases = []
        init_idx = init_info["init_idx"]
        init_time = init_info["init"]
        if "valid_time" in ds:
            valid_values = pd.to_datetime(ds["valid_time"].isel(init=init_idx).values).normalize()
        else:
            valid_values = pd.to_datetime(
                [init_time + pd.to_timedelta(int(lead) * 7, unit="D") for lead in lead_values]
            ).normalize()
        for lead_value in lead_weeks:
            lead_idx = int(lead_lookup[int(lead_value)])
            valid_time = pd.Timestamp(valid_values[lead_idx]).normalize()
            event_offset_days = int(round((valid_time - event_center) / pd.Timedelta(days=1)))
            init_cases.append(
                {
                    "init_idx": int(init_idx),
                    "lead_idx": int(lead_idx),
                    "lead": int(lead_value),
                    "init": pd.Timestamp(init_time),
                    "valid": valid_time,
                    "requested_init": init_info["requested_init"],
                    "init_offset_days": init_info["init_offset_days"],
                    "valid_in_event_window": bool(event_start <= valid_time <= event_end),
                    "valid_event_offset_days": event_offset_days,
                    "closest_valid_to_event": False,
                }
            )
        if init_cases:
            closest_idx = int(np.argmin([abs(c["valid_event_offset_days"]) for c in init_cases]))
            init_cases[closest_idx]["closest_valid_to_event"] = True
            cases.extend(init_cases)
    cases = sorted(cases, key=lambda x: (x["init"], x["lead"]))
    if max_cases and max_cases > 0:
        cases = cases[: int(max_cases)]
    return cases


def extract_case_fields(ds_region, case):
    obs = ds_region["obs_t2m"].isel(init=case["init_idx"], lead=case["lead_idx"]).values
    model_ens = ds_region["model_t2m"].isel(init=case["init_idx"], lead=case["lead_idx"]).values
    geos_ens = ds_region["geos_t2m"].isel(init=case["init_idx"], lead=case["lead_idx"]).values
    return {
        "obs": obs,
        "model_ens": model_ens,
        "geos_ens": geos_ens,
        "model_mean": np.nanmean(model_ens, axis=0),
        "geos_mean": np.nanmean(geos_ens, axis=0),
        "model_spread": np.nanstd(model_ens, axis=0),
        "geos_spread": np.nanstd(geos_ens, axis=0),
    }


def plot_case_map(case, fields, lats, lons, weights, units, event_name, event_start, event_end, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = to_display_temp(fields["obs"], units)
    model = to_display_temp(fields["model_mean"], units)
    geos = to_display_temp(fields["geos_mean"], units)
    geos_err = fields["geos_mean"] - fields["obs"]
    model_err = fields["model_mean"] - fields["obs"]
    geos_abs_err = np.abs(geos_err)
    model_abs_err = np.abs(model_err)
    closeness_gain = geos_abs_err - model_abs_err
    closer_to_obs = np.where(model_abs_err < geos_abs_err, 1.0, -1.0)
    closer_to_obs[np.isclose(model_abs_err, geos_abs_err, atol=1e-6)] = 0.0
    model_minus_geos = fields["model_mean"] - fields["geos_mean"]
    model_ens_count = int(fields["model_ens"].shape[0])
    geos_ens_count = int(fields["geos_ens"].shape[0])

    tmin, tmax = percentile_limits([obs, model, geos], lower=1, upper=99)
    diff_min, diff_max = percentile_limits([model_minus_geos], lower=1, upper=99, symmetric=True)
    abs_min, abs_max = percentile_limits([geos_abs_err, model_abs_err], lower=1, upper=99)
    close_min, close_max = percentile_limits([closeness_gain], lower=1, upper=99, symmetric=True)

    model_rmse = weighted_rmse(fields["model_mean"], fields["obs"], weights)
    geos_rmse = weighted_rmse(fields["geos_mean"], fields["obs"], weights)
    model_abs_mean = weighted_mean(model_abs_err, weights)
    geos_abs_mean = weighted_mean(geos_abs_err, weights)
    closeness_mean = weighted_mean(closeness_gain, weights)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    panels = [
        (obs, f"Observed T2M ({units})", "coolwarm", tmin, tmax),
        (geos, f"GEOS ens mean ({units})\nN={geos_ens_count}", "coolwarm", tmin, tmax),
        (model, f"ML ens mean ({units})\nN={model_ens_count}", "coolwarm", tmin, tmax),
        (model_minus_geos, "ML - GEOS ens mean (K)", "RdBu_r", diff_min, diff_max),
        (geos_abs_err, f"|GEOS mean - Obs| (K)\narea mean={geos_abs_mean:.2f}", "YlOrRd", abs_min, abs_max),
        (model_abs_err, f"|ML mean - Obs| (K)\narea mean={model_abs_mean:.2f}", "YlOrRd", abs_min, abs_max),
        (
            closeness_gain,
            f"Closeness gain: |GEOS-Obs|-|ML-Obs| (K)\npositive/blue = ML closer, mean={closeness_mean:+.2f}",
            "RdBu",
            close_min,
            close_max,
        ),
        (closer_to_obs, "Closer forecast\nML=+1/blue, tie=0, GEOS=-1/red", "RdBu", -1.0, 1.0),
    ]
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        mesh = plot_panel(ax, lons, lats, field, title, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, shrink=0.82)

    if case["valid_in_event_window"]:
        event_label = f"inside {event_start} to {event_end} event window"
    elif case["closest_valid_to_event"]:
        event_label = f"closest lead to event ({case['valid_event_offset_days']:+d}d)"
    else:
        event_label = f"{case['valid_event_offset_days']:+d}d from event"
    fig.suptitle(
        f"{event_name} | init {case['init']:%Y-%m-%d} | "
        f"valid {case['valid']:%Y-%m-%d} | lead week{case['lead']} | {event_label} | "
        f"RMSE GEOS={geos_rmse:.2f} ML={model_rmse:.2f}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composite_map(records, lats, lons, units, event_name, composite_label, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = np.nanmean([r["obs"] for r in records], axis=0)
    geos = np.nanmean([r["geos_mean"] for r in records], axis=0)
    model = np.nanmean([r["model_mean"] for r in records], axis=0)
    model_minus_geos = np.nanmean([r["model_mean"] - r["geos_mean"] for r in records], axis=0)
    geos_abs_err = np.nanmean([np.abs(r["geos_mean"] - r["obs"]) for r in records], axis=0)
    model_abs_err = np.nanmean([np.abs(r["model_mean"] - r["obs"]) for r in records], axis=0)
    closeness_gain = geos_abs_err - model_abs_err
    closer_to_obs = np.where(model_abs_err < geos_abs_err, 1.0, -1.0)
    closer_to_obs[np.isclose(model_abs_err, geos_abs_err, atol=1e-6)] = 0.0

    obs_u = to_display_temp(obs, units)
    geos_u = to_display_temp(geos, units)
    model_u = to_display_temp(model, units)
    tmin, tmax = percentile_limits([obs_u, geos_u, model_u], lower=1, upper=99)
    diff_min, diff_max = percentile_limits([model_minus_geos], lower=1, upper=99, symmetric=True)
    abs_min, abs_max = percentile_limits([geos_abs_err, model_abs_err], lower=1, upper=99)
    close_min, close_max = percentile_limits([closeness_gain], lower=1, upper=99, symmetric=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    panels = [
        (obs_u, f"Composite observed T2M ({units})", "coolwarm", tmin, tmax),
        (geos_u, f"Composite GEOS mean ({units})", "coolwarm", tmin, tmax),
        (model_u, f"Composite ML mean ({units})", "coolwarm", tmin, tmax),
        (model_minus_geos, "Composite ML - GEOS mean (K)", "RdBu_r", diff_min, diff_max),
        (geos_abs_err, "Mean |GEOS mean - Obs| across cases (K)", "YlOrRd", abs_min, abs_max),
        (model_abs_err, "Mean |ML mean - Obs| across cases (K)", "YlOrRd", abs_min, abs_max),
        (
            closeness_gain,
            "Mean closeness gain (K)\npositive/blue = ML closer",
            "RdBu",
            close_min,
            close_max,
        ),
        (closer_to_obs, "Mean-error closer forecast\nML=+1/blue, tie=0, GEOS=-1/red", "RdBu", -1.0, 1.0),
    ]
    for ax, (field, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        mesh = plot_panel(ax, lons, lats, field, title, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    fig.suptitle(
        f"{event_name} composite | {composite_label} | "
        f"{len(records)} init/lead cases",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_case_timeseries(metrics_df, units, event_name, event_start, event_end, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metrics_df.empty:
        return
    labels = [
        f"{row.init:%m-%d}\nwk{row.lead}\n{row.valid:%m-%d}"
        f"{'*' if row.valid_in_event_window else '†' if row.closest_valid_to_event else ''}"
        for row in metrics_df.itertuples(index=False)
    ]
    x = np.arange(len(metrics_df))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.8 * len(metrics_df)), 8), constrained_layout=True)

    axes[0].plot(x, metrics_df["obs_domain_mean_display"], marker="o", label="Obs")
    axes[0].plot(x, metrics_df["geos_domain_mean_display"], marker="o", label="GEOS")
    axes[0].plot(x, metrics_df["model_domain_mean_display"], marker="o", label="ML")
    axes[0].set_title(
        f"{event_name}: area-mean T2M ({units}); "
        f"* = inside {event_start} to {event_end}, † = closest lead"
    )
    axes[0].set_ylabel(f"T2M ({units})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, metrics_df["geos_rmse_k"], marker="o", label="GEOS RMSE")
    axes[1].plot(x, metrics_df["model_rmse_k"], marker="o", label="ML RMSE")
    axes[1].plot(x, metrics_df["geos_crps_k"], marker="s", linestyle="--", label="GEOS CRPS")
    axes[1].plot(x, metrics_df["model_crps_k"], marker="s", linestyle="--", label="ML CRPS")
    axes[1].set_title("Area-weighted T2M metrics over event domain")
    axes[1].set_ylabel("K")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = apply_event_preset(parse_args())
    os.makedirs(args.out_dir, exist_ok=True)
    maps_dir = os.path.join(args.out_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    zarr_path = os.path.join(args.forecast_dir, f"{args.year}.zarr")
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(f"Missing yearly forecast Zarr: {zarr_path}")

    ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
    try:
        ds_region = select_domain(ds, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
        lats = ds_region["lat"].values
        lons = ds_region["lon"].values
        weights = area_weights(lats)
        target_init_dates = parse_date_list(args.target_inits)
        lead_weeks = parse_int_list(args.lead_weeks)
        cases = find_cases(
            ds_region,
            target_init_dates=target_init_dates,
            lead_weeks=lead_weeks,
            event_start=args.event_start,
            event_end=args.event_end,
            max_cases=args.max_cases,
        )
        if not cases:
            available = pd.to_datetime(ds_region["init"].values).strftime("%Y-%m-%d").tolist()
            raise RuntimeError(
                "No init/lead cases matched the requested target init/lead selections. "
                f"Available init dates include: {available[:20]}"
            )

        print(
            f"Selected {len(cases)} init/lead cases over "
            f"{len(lats)}x{len(lons)} grid cells. Loading one case at a time."
        )
        print(
            "Target init dates -> closest available: "
            + ", ".join(
                [
                    f"{case['requested_init']:%Y-%m-%d}->{case['init']:%Y-%m-%d}"
                    f" ({case['init_offset_days']:+d}d)"
                    for case in cases
                    if case["lead"] == lead_weeks[0]
                ]
            )
        )
        records = []
        metric_rows = []
        raw_summary_rows = []
        model_ens_count = None
        geos_ens_count = None
        for case in cases:
            fields = extract_case_fields(ds_region, case)
            model_ens_count = int(fields["model_ens"].shape[0])
            geos_ens_count = int(fields["geos_ens"].shape[0])
            records.append({
                "obs": fields["obs"],
                "geos_mean": fields["geos_mean"],
                "model_mean": fields["model_mean"],
                "closest_valid_to_event": case["closest_valid_to_event"],
                "valid_in_event_window": case["valid_in_event_window"],
            })

            obs_mean_k = weighted_mean(fields["obs"], weights)
            geos_mean_k = weighted_mean(fields["geos_mean"], weights)
            model_mean_k = weighted_mean(fields["model_mean"], weights)
            geos_rmse = weighted_rmse(fields["geos_mean"], fields["obs"], weights)
            model_rmse = weighted_rmse(fields["model_mean"], fields["obs"], weights)
            geos_bias = weighted_bias(fields["geos_mean"], fields["obs"], weights)
            model_bias = weighted_bias(fields["model_mean"], fields["obs"], weights)
            geos_crps = weighted_crps(fields["geos_ens"], fields["obs"], weights)
            model_crps = weighted_crps(fields["model_ens"], fields["obs"], weights)
            for field_name, values in (
                ("obs_t2m_k", fields["obs"]),
                ("geos_t2m_k", fields["geos_ens"]),
                ("geos_mean_t2m_k", fields["geos_mean"]),
                ("model_t2m_k", fields["model_ens"]),
                ("model_mean_t2m_k", fields["model_mean"]),
            ):
                summary = field_summary(values)
                raw_summary_rows.append(
                    {
                        "requested_init": case["requested_init"],
                        "init": case["init"],
                        "valid": case["valid"],
                        "lead": case["lead"],
                        "field": field_name,
                        "min": summary["min"],
                        "mean": summary["mean"],
                        "max": summary["max"],
                    }
                )
            row = {
                "requested_init": case["requested_init"],
                "init": case["init"],
                "init_offset_days": case["init_offset_days"],
                "valid": case["valid"],
                "lead": case["lead"],
                "valid_in_event_window": case["valid_in_event_window"],
                "closest_valid_to_event": case["closest_valid_to_event"],
                "valid_event_offset_days": case["valid_event_offset_days"],
                "model_ensemble_members": model_ens_count,
                "geos_ensemble_members": geos_ens_count,
                "obs_domain_mean_k": obs_mean_k,
                "geos_domain_mean_k": geos_mean_k,
                "model_domain_mean_k": model_mean_k,
                "obs_domain_mean_display": obs_mean_k - 273.15 if args.temperature_units == "C" else obs_mean_k,
                "geos_domain_mean_display": geos_mean_k - 273.15 if args.temperature_units == "C" else geos_mean_k,
                "model_domain_mean_display": model_mean_k - 273.15 if args.temperature_units == "C" else model_mean_k,
                "obs_domain_min_k": float(np.nanmin(fields["obs"])),
                "geos_domain_min_k": float(np.nanmin(fields["geos_mean"])),
                "model_domain_min_k": float(np.nanmin(fields["model_mean"])),
                "geos_rmse_k": geos_rmse,
                "model_rmse_k": model_rmse,
                "rmse_skill_vs_geos_pct": 100.0 * (1.0 - model_rmse / geos_rmse) if geos_rmse > 1e-12 else np.nan,
                "geos_crps_k": geos_crps,
                "model_crps_k": model_crps,
                "crps_skill_vs_geos_pct": 100.0 * (1.0 - model_crps / geos_crps) if geos_crps > 1e-12 else np.nan,
                "geos_bias_k": geos_bias,
                "model_bias_k": model_bias,
                "model_minus_geos_domain_mean_k": model_mean_k - geos_mean_k,
            }
            metric_rows.append(row)

            filename = (
                f"case_init{case['init']:%Y%m%d}_valid{case['valid']:%Y%m%d}_"
                f"week{case['lead']}_t2m_maps.png"
            )
            plot_case_map(
                case,
                fields,
                lats,
                lons,
                weights,
                args.temperature_units,
                args.event_name,
                args.event_start,
                args.event_end,
                os.path.join(maps_dir, filename),
            )

        metrics_df = pd.DataFrame(metric_rows).sort_values(["init", "lead"])
        metrics_csv = os.path.join(args.out_dir, "event_case_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False, float_format="%.6f")
        raw_summary_csv = os.path.join(args.out_dir, "event_case_raw_value_summary.csv")
        pd.DataFrame(raw_summary_rows).to_csv(raw_summary_csv, index=False, float_format="%.6f")
        plot_case_timeseries(
            metrics_df,
            args.temperature_units,
            args.event_name,
            args.event_start,
            args.event_end,
            os.path.join(args.out_dir, "case_t2m_timeseries_metrics.png"),
        )
        plot_composite_map(
            records,
            lats,
            lons,
            args.temperature_units,
            args.event_name,
            "all selected inits/leads",
            os.path.join(args.out_dir, "case_composite_t2m_maps.png"),
        )
        closest_records = [r for r in records if r["closest_valid_to_event"]]
        if closest_records:
            plot_composite_map(
                closest_records,
                lats,
                lons,
                args.temperature_units,
                args.event_name,
                "closest valid lead for each target init",
                os.path.join(args.out_dir, "case_closest_event_t2m_maps.png"),
            )

        orientation = {
            "zarr_path": os.path.abspath(zarr_path),
            "event_preset": args.event_preset,
            "event_name": args.event_name,
            "event_start": args.event_start,
            "event_end": args.event_end,
            "target_inits": [d.strftime("%Y-%m-%d") for d in target_init_dates],
            "lead_weeks": lead_weeks,
            "model_ensemble_members": model_ens_count,
            "geos_ensemble_members": geos_ens_count,
            "lat_first": float(ds_region["lat"].values[0]),
            "lat_last": float(ds_region["lat"].values[-1]),
            "lat_strictly_increasing": bool(np.all(np.diff(ds_region["lat"].values) > 0)),
            "lon_first": float(ds_region["lon"].values[0]),
            "lon_last": float(ds_region["lon"].values[-1]),
            "lon_strictly_increasing": bool(np.all(np.diff(ds_region["lon"].values) > 0)),
            "selected_cases": [
                {
                    "requested_init": c["requested_init"].strftime("%Y-%m-%d"),
                    "init": c["init"].strftime("%Y-%m-%d"),
                    "init_offset_days": c["init_offset_days"],
                    "valid": c["valid"].strftime("%Y-%m-%d"),
                    "lead": c["lead"],
                    "valid_in_event_window": c["valid_in_event_window"],
                    "closest_valid_to_event": c["closest_valid_to_event"],
                    "valid_event_offset_days": c["valid_event_offset_days"],
                }
                for c in cases
            ],
            "domain": {
                "lat_min": float(ds_region["lat"].values.min()),
                "lat_max": float(ds_region["lat"].values.max()),
                "lon_min": float(ds_region["lon"].values.min()),
                "lon_max": float(ds_region["lon"].values.max()),
            },
            "plotting_note": "Maps use pcolormesh(lon, lat, field). With increasing lat, south is bottom and north is top.",
        }
        with open(os.path.join(args.out_dir, "case_orientation_and_selection.json"), "w") as f:
            json.dump(orientation, f, indent=2)

        print("\nSelected cases:")
        print(metrics_df[[
            "requested_init",
            "init",
            "init_offset_days",
            "valid",
            "lead",
            "valid_in_event_window",
            "closest_valid_to_event",
            "valid_event_offset_days",
            "model_ensemble_members",
            "geos_ensemble_members",
            "obs_domain_mean_k",
            "geos_rmse_k",
            "model_rmse_k",
            "rmse_skill_vs_geos_pct",
            "geos_crps_k",
            "model_crps_k",
            "crps_skill_vs_geos_pct",
        ]].to_string(index=False))
        print(f"\nEnsemble members used: ML={model_ens_count}, GEOS={geos_ens_count}")
        print(f"\n✅ Wrote metrics: {metrics_csv}")
        print(f"✅ Wrote raw value summary: {raw_summary_csv}")
        print(f"✅ Wrote maps under: {maps_dir}")
        print(f"✅ Wrote composite map: {os.path.join(args.out_dir, 'case_composite_t2m_maps.png')}")
        if closest_records:
            print(f"✅ Wrote closest-event composite map: {os.path.join(args.out_dir, 'case_closest_event_t2m_maps.png')}")
        print(f"✅ Wrote time series: {os.path.join(args.out_dir, 'case_t2m_timeseries_metrics.png')}")
    finally:
        ds.close()


if __name__ == "__main__":
    sys.exit(main())
