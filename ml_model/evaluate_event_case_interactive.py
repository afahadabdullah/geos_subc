#!/usr/bin/env python3
"""
Interactive, non-Slurm evaluation script for extreme events.
Computes RMSE, CRPS, bias, spread, and Brier Skill Score (BSS) over a regional domain
and plots validation time series with ensemble spreads.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Event Presets
EVENT_PRESETS = {
    "southwest_jul2023_heatwave": {
        "year": 2023,
        "event_variable": "t2m",
        "event_name": "July 2023 Southwest US heatwave",
        "target_inits": "2023-06-22,2023-06-29,2023-07-06,2023-07-13,2023-07-20",
        "lead_weeks": "1,2,3,4",
        "event_start": "2023-07-10",
        "event_end": "2023-07-31",
        "lat_min": 30.0,
        "lat_max": 38.0,
        "lon_min": -125.0,
        "lon_max": -105.0,
        "out_dir": "ml_output_flow_finalv1_global_noisectx_t2mres/event_southwest_jul2023_heatwave",
    },
    "pnw_jun2021_heat_dome": {
        "year": 2021,
        "event_variable": "t2m",
        "event_name": "June 28-29 2021 Pacific Northwest heat dome",
        "target_inits": "2021-06-03,2021-06-10,2021-06-17,2021-06-24",
        "lead_weeks": "1,2,3,4",
        "event_start": "2021-06-28",
        "event_end": "2021-06-29",
        "lat_min": 40.0,
        "lat_max": 55.0,
        "lon_min": -130.0,
        "lon_max": -110.0,
        "out_dir": "ml_output_flow_finalv1_global_noisectx_t2mres/event_pnw_jun2021_heat_dome",
    },
    "central_us_aug2023_heatwave": {
        "year": 2023,
        "event_variable": "t2m",
        "event_name": "August 23-24 2023 Central US heatwave",
        "target_inits": "2023-07-27,2023-08-03,2023-08-10,2023-08-17",
        "lead_weeks": "1,2,3,4",
        "event_start": "2023-08-23",
        "event_end": "2023-08-24",
        "lat_min": 35.0,
        "lat_max": 45.0,
        "lon_min": -100.0,
        "lon_max": -85.0,
        "out_dir": "ml_output_flow_finalv1_global_noisectx_t2mres/event_central_us_aug2023_heatwave",
    },
    "europe_jul2022_heatwave": {
        "year": 2022,
        "event_variable": "t2m",
        "event_name": "July 2022 Europe heatwave",
        "target_inits": "2022-06-16,2022-06-23,2022-06-30,2022-07-07,2022-07-14",
        "lead_weeks": "1,2,3,4",
        "event_start": "2022-07-15",
        "event_end": "2022-07-20",
        "lat_min": 35.0,
        "lat_max": 60.0,
        "lon_min": -15.0,
        "lon_max": 25.0,
        "out_dir": "ml_output_flow_finalv1_global_noisectx_t2mres/event_europe_jul2022_heatwave",
    },
    "europe_jul2023_heatwave": {
        "year": 2023,
        "event_variable": "t2m",
        "event_name": "July 2023 Southern Europe heatwave",
        "target_inits": "2023-06-15,2023-06-22,2023-06-29,2023-07-06,2023-07-13,2023-07-20",
        "lead_weeks": "1,2,3,4",
        "event_start": "2023-07-15",
        "event_end": "2023-07-25",
        "lat_min": 35.0,
        "lat_max": 48.0,
        "lon_min": -10.0,
        "lon_max": 30.0,
        "out_dir": "ml_output_flow_finalv1_global_noisectx_t2mres/event_europe_jul2023_heatwave",
    },
    "bangladesh_jun2022_flood": {
        "year": 2022,
        "event_variable": "pr",
        "event_name": "June 2022 Bangladesh India extreme monsoon rain",
        "target_inits": "2022-05-19,2022-05-26,2022-06-02,2022-06-09,2022-06-16",
        "lead_weeks": "1,2,3,4",
        "event_start": "2022-06-15",
        "event_end": "2022-06-22",
        "lat_min": 20.0,
        "lat_max": 30.0,
        "lon_min": 85.0,
        "lon_max": 98.0,
        "out_dir": "ml_output_flow_finalv1_global_noisectx_t2mres/event_bangladesh_jun2022_flood",
    }
}


VARIABLES = {
    "pr": {"obs": "obs_pr", "model": "model_pr", "geos": "geos_pr", "units": "mm/day"},
    "t2m": {"obs": "obs_t2m", "model": "model_t2m", "geos": "geos_t2m", "units": "K"}
}

def normalize_lon(lon):
    lon = float(lon)
    return lon if lon >= 0.0 else lon + 360.0

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

def weighted_brier_score(ensemble, obs, weights, threshold):
    ensemble = np.asarray(ensemble, dtype=np.float64)  # [E, H, W]
    obs = np.asarray(obs, dtype=np.float64)  # [H, W]
    finite = np.isfinite(obs) & np.all(np.isfinite(ensemble), axis=0)
    weighted_mask = np.where(finite, weights, 0.0)
    denom = float(np.sum(weighted_mask))
    if denom <= 0:
        return np.nan
    prob = np.mean(ensemble > threshold, axis=0)
    obs_binary = (obs > threshold).astype(np.float64)
    brier_map = (prob - obs_binary) ** 2
    return float(np.sum(np.where(finite, brier_map, 0.0) * weighted_mask) / denom)

def gridpoint_crps_map(ensemble, obs):
    ensemble = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mae_term = np.mean(np.abs(ensemble - obs[None, :, :]), axis=0)
    ens_sorted = np.sort(ensemble, axis=0)
    e = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, e + 1, dtype=np.float64)) - e - 1.0) / (e * e)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    return mae_term - spread_term

def plot_spatial_skill_maps(rows, lats, lons, weights, threshold, lead_week, spec_names, event_name, out_dir):
    lead_rows = [r for r in rows if r["lead"] == lead_week]
    if not lead_rows:
        print(f"⚠️ No cases for lead week {lead_week} to plot spatial skill maps.")
        return
        
    obs_stack = []
    model_mean_stack = []
    geos_mean_stack = []
    
    for r in lead_rows:
        fields = r["raw_fields"]
        obs_stack.append(fields["obs"])
        model_mean_stack.append(fields["model_mean"])
        geos_mean_stack.append(fields["geos_mean"])
        
    obs_stack = np.asarray(obs_stack)
    model_mean_stack = np.asarray(model_mean_stack)
    geos_mean_stack = np.asarray(geos_mean_stack)
    
    model_rmse_map = np.sqrt(np.mean((model_mean_stack - obs_stack)**2, axis=0))
    geos_rmse_map = np.sqrt(np.mean((geos_mean_stack - obs_stack)**2, axis=0))
    rmse_skill_map = 1.0 - (model_rmse_map / geos_rmse_map)
    rmse_skill_map = np.where(np.isfinite(rmse_skill_map) & (geos_rmse_map > 1e-12), rmse_skill_map, np.nan)
    
    model_crps_cases = []
    geos_crps_cases = []
    for r in lead_rows:
        fields = r["raw_fields"]
        model_crps_cases.append(gridpoint_crps_map(fields["model_ens"], fields["obs"]))
        geos_crps_cases.append(gridpoint_crps_map(fields["geos_ens"], fields["obs"]))
    
    model_crps_map = np.mean(model_crps_cases, axis=0)
    geos_crps_map = np.mean(geos_crps_cases, axis=0)
    crps_skill_map = 1.0 - (model_crps_map / geos_crps_map)
    crps_skill_map = np.where(np.isfinite(crps_skill_map) & (geos_crps_map > 1e-12), crps_skill_map, np.nan)
    
    model_bs_cases = []
    geos_bs_cases = []
    for r in lead_rows:
        fields = r["raw_fields"]
        obs_binary = (fields["obs"] > threshold).astype(np.float64)
        
        model_prob = np.mean(fields["model_ens"] > threshold, axis=0)
        geos_prob = np.mean(fields["geos_ens"] > threshold, axis=0)
        
        model_bs_cases.append((model_prob - obs_binary)**2)
        geos_bs_cases.append((geos_prob - obs_binary)**2)
        
    model_bs_map = np.mean(model_bs_cases, axis=0)
    geos_bs_map = np.mean(geos_bs_cases, axis=0)
    bss_map = 1.0 - (model_bs_map / geos_bs_map)
    bss_map = np.where(np.isfinite(bss_map) & (geos_bs_map > 1e-12), bss_map, np.nan)
    
    cartopy_enabled = False
    ccrs = None
    cfeature = None
    try:
        import cartopy.crs as ccrs_lib
        import cartopy.feature as cfeature_lib
        ccrs = ccrs_lib
        cfeature = cfeature_lib
        cartopy_enabled = True
    except Exception as e:
        print(f"⚠️ Cartopy is not available ({e}). Plotting plain skill maps.")
        
    fig, axes = plt.subplots(
        1, 3, 
        figsize=(18, 5), 
        subplot_kw={"projection": ccrs.PlateCarree()} if cartopy_enabled else None, 
        constrained_layout=True
    )
    
    maps_data = [
        (rmse_skill_map, "RMSE Skill Score", "RdBu_r", -0.5, 0.5),
        (crps_skill_map, "CRPS Skill Score", "RdBu_r", -0.5, 0.5),
        (bss_map, "Brier Skill Score (BSS)", "RdBu_r", -0.5, 0.5)
    ]
    
    for ax, (data_map, title, cmap, vmin, vmax) in zip(axes, maps_data):
        if cartopy_enabled:
            ax.set_extent([np.nanmin(lons), np.nanmax(lons), np.nanmin(lats), np.nanmax(lats)], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.8)
            try:
                states_provinces = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lines',
                    scale='50m',
                    facecolor='none',
                    edgecolor='gray',
                    linewidth=0.6
                )
                ax.add_feature(states_provinces)
            except Exception:
                try:
                    ax.add_feature(cfeature.STATES, edgecolor="gray", linewidth=0.6)
                except Exception:
                    pass
            
            mesh = ax.pcolormesh(lons, lats, data_map, transform=ccrs.PlateCarree(), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            mesh = ax.pcolormesh(lons, lats, data_map, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlim(np.nanmin(lons), np.nanmax(lons))
            ax.set_ylim(np.nanmin(lats), np.nanmax(lats))
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            
        fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05, label="Skill Score (Positive = ML improves over GEOS)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        
    fig.suptitle(f"{event_name}: Spatial Skill Maps vs GEOS (Lead Week {lead_week})", fontsize=15, fontweight="bold", y=1.05)
    
    plot_path = os.path.join(out_dir, f"event_spatial_skill_maps_lead_week_{lead_week}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🗺️ Saved spatial skill maps to: {plot_path}")

def parse_date_list(value):
    dates = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        dates.append(pd.Timestamp(item).normalize())
    return dates

def parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]

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

def find_cases(ds, target_init_dates, lead_weeks, event_start, event_end):
    event_start = pd.Timestamp(event_start).normalize()
    event_end = pd.Timestamp(event_end).normalize()
    event_center = event_start + (event_end - event_start) / 2
    lead_values = np.asarray(ds["lead"].values)
    lead_lookup = {int(lead): lead_idx for lead_idx, lead in enumerate(lead_values)}
    
    cases = []
    for init_info in selected_init_infos(ds, target_init_dates):
        init_idx = init_info["init_idx"]
        init_time = init_info["init"]
        if "valid_time" in ds:
            valid_values = pd.to_datetime(ds["valid_time"].isel(init=init_idx).values).normalize()
        else:
            valid_values = pd.to_datetime(
                [init_time + pd.to_timedelta(int(lead) * 7, unit="D") for lead in lead_values]
            ).normalize()
        for lead_value in lead_weeks:
            if int(lead_value) not in lead_lookup:
                continue
            lead_idx = int(lead_lookup[int(lead_value)])
            valid_time = pd.Timestamp(valid_values[lead_idx]).normalize()
            event_offset_days = int(round((valid_time - event_center) / pd.Timedelta(days=1)))
            cases.append(
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
                }
            )
    return sorted(cases, key=lambda x: (x["init"], x["lead"]))

def extract_case_fields(ds_region, case, spec_names):
    obs = ds_region[spec_names["obs"]].isel(init=case["init_idx"], lead=case["lead_idx"]).values
    model_ens = ds_region[spec_names["model"]].isel(init=case["init_idx"], lead=case["lead_idx"]).values
    geos_ens = ds_region[spec_names["geos"]].isel(init=case["init_idx"], lead=case["lead_idx"]).values
    return {
        "obs": obs,
        "model_ens": model_ens,
        "geos_ens": geos_ens,
        "model_mean": np.nanmean(model_ens, axis=0),
        "geos_mean": np.nanmean(geos_ens, axis=0),
        "model_spread": np.nanstd(model_ens, axis=0),
        "geos_spread": np.nanstd(geos_ens, axis=0),
    }

def apply_preset(args):
    if args.preset and args.preset in EVENT_PRESETS:
        preset = EVENT_PRESETS[args.preset]
        print(f"📌 Applying event preset: {args.preset} ({preset['event_name']})")
        for key, val in preset.items():
            if getattr(args, key) is None:
                setattr(args, key, val)
    return args

def main():
    parser = argparse.ArgumentParser(description="Evaluate extreme event S2S forecast performance interactively.")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Directory containing yearly Zarr forecast stores.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(EVENT_PRESETS.keys()),
        default="southwest_jul2023_heatwave",
        help="Preset event configuration.",
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--event_variable", type=str, default=None, choices=["pr", "t2m"])
    parser.add_argument("--event_name", type=str, default=None)
    parser.add_argument("--target_inits", type=str, default=None, help="Comma-separated target inits.")
    parser.add_argument("--lead_weeks", type=str, default=None, help="Comma-separated lead weeks.")
    parser.add_argument("--event_start", type=str, default=None)
    parser.add_argument("--event_end", type=str, default=None)
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument(
        "--threshold_temp",
        "--threshold",
        dest="threshold_temp",
        type=float,
        default=None,
        help="Custom threshold for Brier score (C/K for temperature, mm/day for precipitation). If None, 90th percentile of observations is used.",
    )
    parser.add_argument(
        "--temperature_units",
        type=str,
        choices=["C", "K"],
        default="C",
        help="Units for plotting and printing.",
    )
    parser.add_argument(
        "--plot_lead",
        type=int,
        default=2,
        help="Lead week to plot in valid-date timeseries.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots/CSV. Defaults to preset folder.",
    )
    parser.add_argument(
        "--dynamic_core",
        action="store_true",
        help="Find epicenter of heat dynamically based on observed temperature during the event, and average over a smaller region around it.",
    )
    parser.add_argument(
        "--core_width",
        type=float,
        default=10.0,
        help="Bounding box width (degrees) to center around dynamic core epicenter.",
    )
    args = parser.parse_args()
    args = apply_preset(args)

    if args.year is None:
        print("❌ Error: --year or --preset is required.")
        sys.exit(1)

    spec_names = VARIABLES[args.event_variable]
    out_dir = args.out_dir or f"ml_output_flow_finalv1_global_noisectx_t2mres/event_{args.preset or 'custom'}"
    if args.dynamic_core:
        out_dir = out_dir + "_dynamic"
    os.makedirs(out_dir, exist_ok=True)

    zarr_path = os.path.join(args.forecast_dir, f"{args.year}.zarr")
    if not os.path.isdir(zarr_path):
        print(f"❌ Error: Zarr store for {args.year} not found at {zarr_path}")
        sys.exit(1)

    print(f"📂 Opening forecast Zarr: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
    
    try:
        target_inits = parse_date_list(args.target_inits)
        lead_weeks = parse_int_list(args.lead_weeks)

        if args.dynamic_core:
            print("🔥 Finding heat epicenter dynamically within search domain...")
            # 1. Slice to broad search domain
            ds_search = select_domain(ds, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
            # 2. Find cases in search domain to identify validation dates
            search_cases = find_cases(ds_search, target_inits, lead_weeks, args.event_start, args.event_end)
            window_cases = [c for c in search_cases if c["valid_in_event_window"]]
            
            if not window_cases:
                print("⚠️ Warning: No cases fall within the event window. Using static domain.")
                ds_region = ds_search
            else:
                # 3. Load observations for all window cases
                obs_list = []
                for case in window_cases:
                    obs_val = ds_search[spec_names["obs"]].isel(init=case["init_idx"], lead=case["lead_idx"]).values
                    obs_list.append(obs_val)
                
                # 4. Average observations over the event window cases
                mean_obs = np.nanmean(np.asarray(obs_list), axis=0) # Shape: (lat, lon)
                
                # 5. Find coordinate of max heat
                lat_idx, lon_idx = np.unravel_index(np.nanargmax(mean_obs), mean_obs.shape)
                lat_peak = float(ds_search["lat"].values[lat_idx])
                lon_peak = float(ds_search["lon"].values[lon_idx])
                
                # Print information
                peak_val = mean_obs[lat_idx, lon_idx]
                if args.event_variable == "t2m" and args.temperature_units == "C":
                    peak_val_display = f"{peak_val - 273.15:.1f}°C"
                else:
                    peak_val_display = f"{peak_val:.1f} {spec_names['units']}"
                print(f"🔥 Epicenter detected at lat={lat_peak:.2f}, lon={lon_peak:.2f} (Peak Avg Temp: {peak_val_display})")
                
                # 6. Calculate dynamic core bounds
                lat_min_core = max(-90.0, lat_peak - args.core_width / 2)
                lat_max_core = min(90.0, lat_peak + args.core_width / 2)
                lon_min_core = lon_peak - args.core_width / 2
                lon_max_core = lon_peak + args.core_width / 2
                
                print(f"🔥 Dynamic core domain (width={args.core_width}°): lat=[{lat_min_core:.2f}..{lat_max_core:.2f}], lon=[{lon_min_core:.2f}..{lon_max_core:.2f}]")
                
                # 7. Slices dataset to the dynamic core region
                ds_region = select_domain(ds, lat_min_core, lat_max_core, lon_min_core, lon_max_core)
        else:
            ds_region = select_domain(ds, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

        lats = ds_region["lat"].values
        lons = ds_region["lon"].values
        weights = area_weights(lats)
        cases = find_cases(ds_region, target_inits, lead_weeks, args.event_start, args.event_end)
        
        if not cases:
            print("❌ Error: No matching init/lead cases found.")
            sys.exit(1)

        print(f"🔎 Selected {len(cases)} cases across domain lat=[{args.lat_min}..{args.lat_max}], lon=[{args.lon_min}..{args.lon_max}]")
        
        # Load and extract all case fields
        case_records = []
        all_obs_values = []
        for case in cases:
            fields = extract_case_fields(ds_region, case, spec_names)
            case_records.append((case, fields))
            all_obs_values.extend(fields["obs"].flatten())
            
        all_obs_values = np.asarray(all_obs_values)
        all_obs_values = all_obs_values[np.isfinite(all_obs_values)]

        # Set threshold
        if args.threshold_temp is not None:
            threshold = args.threshold_temp
            if args.event_variable == "t2m" and args.temperature_units == "C" and threshold < 100:
                threshold += 273.15  # Convert to Kelvin for calculations
        else:
            # Default to 90th percentile of observations
            threshold = float(np.nanpercentile(all_obs_values, 90))
        
        display_threshold = threshold
        if args.event_variable == "t2m" and args.temperature_units == "C":
            display_threshold -= 273.15
        print(f"🎯 Threshold for BSS calculation: {display_threshold:.1f} {args.temperature_units} ({threshold:.2f} K/raw)")

        # Compute case metrics
        rows = []
        for case, fields in case_records:
            obs_mean = weighted_mean(fields["obs"], weights)
            model_mean = weighted_mean(fields["model_mean"], weights)
            geos_mean = weighted_mean(fields["geos_mean"], weights)
            
            # RMSE
            model_rmse = weighted_rmse(fields["model_mean"], fields["obs"], weights)
            geos_rmse = weighted_rmse(fields["geos_mean"], fields["obs"], weights)
            rmse_ss = 1.0 - (model_rmse / geos_rmse) if geos_rmse > 1e-12 else np.nan
            
            # CRPS
            model_crps = weighted_crps(fields["model_ens"], fields["obs"], weights)
            geos_crps = weighted_crps(fields["geos_ens"], fields["obs"], weights)
            crpss = 1.0 - (model_crps / geos_crps) if geos_crps > 1e-12 else np.nan
            
            # Brier Score and BSS
            model_bs = weighted_brier_score(fields["model_ens"], fields["obs"], weights, threshold)
            geos_bs = weighted_brier_score(fields["geos_ens"], fields["obs"], weights, threshold)
            bss = 1.0 - (model_bs / geos_bs) if geos_bs > 1e-12 else np.nan
            
            # Bias and Spread
            model_bias = weighted_bias(fields["model_mean"], fields["obs"], weights)
            geos_bias = weighted_bias(fields["geos_mean"], fields["obs"], weights)
            model_spread = weighted_mean(fields["model_spread"], weights)
            geos_spread = weighted_mean(fields["geos_spread"], weights)
            
            # Units conversion for printing
            obs_mean_d = obs_mean - 273.15 if args.event_variable == "t2m" and args.temperature_units == "C" else obs_mean
            model_mean_d = model_mean - 273.15 if args.event_variable == "t2m" and args.temperature_units == "C" else model_mean
            geos_mean_d = geos_mean - 273.15 if args.event_variable == "t2m" and args.temperature_units == "C" else geos_mean
            model_bias_d = model_bias
            geos_bias_d = geos_bias
            model_rmse_d = model_rmse
            geos_rmse_d = geos_rmse
            
            rows.append({
                "init": case["init"],
                "valid": case["valid"],
                "lead": case["lead"],
                "in_event_window": case["valid_in_event_window"],
                "obs_mean": obs_mean_d,
                "model_mean": model_mean_d,
                "geos_mean": geos_mean_d,
                "model_rmse": model_rmse_d,
                "geos_rmse": geos_rmse_d,
                "rmse_ss": rmse_ss,
                "model_crps": model_crps,
                "geos_crps": geos_crps,
                "crpss": crpss,
                "model_bs": model_bs,
                "geos_bs": geos_bs,
                "bss": bss,
                "model_bias": model_bias_d,
                "geos_bias": geos_bias_d,
                "model_spread": model_spread,
                "geos_spread": geos_spread,
                # Store raw fields for plotting
                "raw_fields": fields
            })
            
        metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != "raw_fields"} for r in rows])
        
        # Save metrics CSV
        csv_path = os.path.join(out_dir, "event_metrics.csv")
        metrics_df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"📊 Wrote metrics table to: {csv_path}")
        
        # Print metrics table to console
        print("\n" + "=" * 120)
        print(f"EVENT METRICS TABLE FOR: {args.event_name}")
        print("=" * 120)
        headers = ["Init", "Valid", "Lead", "Obs Mean", "Model Mean", "GEOS Mean", "Model RMSE", "GEOS RMSE", "RMSE SS", "Model CRPS", "GEOS CRPS", "CRPSS", "BSS", "M Spread", "G Spread"]
        print(f"{headers[0]:<12} {headers[1]:<12} {headers[2]:<4} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10} {headers[6]:<10} {headers[7]:<10} {headers[8]:<8} {headers[9]:<10} {headers[10]:<10} {headers[11]:<8} {headers[12]:<8} {headers[13]:<8} {headers[14]:<8}")
        print("-" * 120)
        for r in rows:
            in_window_marker = "*" if r["in_event_window"] else " "
            print(f"{r['init'].strftime('%Y-%m-%d'):<12} {r['valid'].strftime('%Y-%m-%d'):<12} {r['lead']:<4d} "
                  f"{r['obs_mean']:<10.2f} {r['model_mean']:<10.2f} {r['geos_mean']:<10.2f} "
                  f"{r['model_rmse']:<10.3f} {r['geos_rmse']:<10.3f} {r['rmse_ss']:<8.3f} "
                  f"{r['model_crps']:<10.3f} {r['geos_crps']:<10.3f} {r['crpss']:<8.3f} "
                  f"{r['bss']:<8.3f} {r['model_spread']:<8.3f} {r['geos_spread']:<8.3f} {in_window_marker}")
        print("=" * 120 + "\n")
        
        # Generate Plot 1: Valid-date tracking for each lead week
        for pl in lead_weeks:
            lead_cases = [r for r in rows if r["lead"] == pl]
            if len(lead_cases) >= 2:
                lead_cases = sorted(lead_cases, key=lambda x: x["valid"])
                valid_dates = [r["valid"] for r in lead_cases]
                obs_series = [r["obs_mean"] for r in lead_cases]
                model_means = [r["model_mean"] for r in lead_cases]
                geos_means = [r["geos_mean"] for r in lead_cases]
                
                # Ensemble bounds
                model_low = []
                model_high = []
                geos_low = []
                geos_high = []
                for r in lead_cases:
                    fields = r["raw_fields"]
                    m_ens_means = [weighted_mean(fields["model_ens"][e], weights) for e in range(fields["model_ens"].shape[0])]
                    g_ens_means = [weighted_mean(fields["geos_ens"][e], weights) for e in range(fields["geos_ens"].shape[0])]
                    
                    if args.event_variable == "t2m" and args.temperature_units == "C":
                        m_ens_means = [v - 273.15 for v in m_ens_means]
                        g_ens_means = [v - 273.15 for v in g_ens_means]
                    
                    model_low.append(np.percentile(m_ens_means, 10))
                    model_high.append(np.percentile(m_ens_means, 90))
                    geos_low.append(np.percentile(g_ens_means, 10))
                    geos_high.append(np.percentile(g_ens_means, 90))
                    
                fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
                ax.plot(valid_dates, obs_series, marker="o", color="black", linewidth=2.0, label="Observations")
                ax.plot(valid_dates, model_means, marker="s", color="royalblue", linewidth=1.5, label="ML Forecast (Mean)")
                ax.fill_between(valid_dates, model_low, model_high, color="royalblue", alpha=0.2, label="ML 10th-90th Percentile")
                
                ax.plot(valid_dates, geos_means, marker="^", color="darkorange", linewidth=1.5, label="GEOS Forecast (Mean)")
                ax.fill_between(valid_dates, geos_low, geos_high, color="darkorange", alpha=0.2, label="GEOS 10th-90th Percentile")
                
                unit_label = f"({args.temperature_units})" if args.event_variable == "t2m" else f"({spec_names['units']})"
                ax.set_title(f"{args.event_name}: Valid-date Tracking at Lead Week {pl}")
                ax.set_xlabel("Valid Date")
                ax.set_ylabel(f"Domain Area-weighted Mean {args.event_variable.upper()} {unit_label}")
                ax.grid(alpha=0.3)
                ax.legend(loc="best")
                
                plot_path1 = os.path.join(out_dir, f"event_valid_date_tracking_lead_week_{pl}.png")
                fig.savefig(plot_path1, dpi=150)
                plt.close(fig)
                print(f"📈 Saved valid-date tracking plot to: {plot_path1}")
            else:
                print(f"⚠️ Not enough cases matching lead week {pl} to plot valid-date tracking.")

        # Generate Plot 1B: Spatial skill maps for each lead week
        for pl in lead_weeks:
            plot_spatial_skill_maps(
                rows=rows,
                lats=lats,
                lons=lons,
                weights=weights,
                threshold=threshold,
                lead_week=pl,
                spec_names=spec_names,
                event_name=args.event_name,
                out_dir=out_dir
            )

        # Generate Plot 2: Lead week timeseries for the best forecast initialization date
        # (closest initialization prior to or at the start of the event window)
        event_start_dt = pd.Timestamp(args.event_start).normalize()
        best_init = None
        for case, _ in case_records:
            if case["init"] <= event_start_dt:
                if best_init is None or case["init"] > best_init["init"]:
                    best_init = case
                    
        if best_init is not None:
            init_dt = best_init["init"]
            init_cases = [r for r in rows if r["init"] == init_dt]
            init_cases = sorted(init_cases, key=lambda x: x["lead"])
            
            leads = [r["lead"] for r in init_cases]
            valid_labels = [r["valid"].strftime("%m-%d") for r in init_cases]
            obs_series = [r["obs_mean"] for r in init_cases]
            model_means = [r["model_mean"] for r in init_cases]
            geos_means = [r["geos_mean"] for r in init_cases]
            
            model_low = []
            model_high = []
            geos_low = []
            geos_high = []
            for r in init_cases:
                fields = r["raw_fields"]
                m_ens_means = [weighted_mean(fields["model_ens"][e], weights) for e in range(fields["model_ens"].shape[0])]
                g_ens_means = [weighted_mean(fields["geos_ens"][e], weights) for e in range(fields["geos_ens"].shape[0])]
                
                if args.event_variable == "t2m" and args.temperature_units == "C":
                    m_ens_means = [v - 273.15 for v in m_ens_means]
                    g_ens_means = [v - 273.15 for v in g_ens_means]
                
                model_low.append(np.percentile(m_ens_means, 10))
                model_high.append(np.percentile(m_ens_means, 90))
                geos_low.append(np.percentile(g_ens_means, 10))
                geos_high.append(np.percentile(g_ens_means, 90))
                
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            x_vals = np.arange(len(leads))
            ax.plot(x_vals, obs_series, marker="o", color="black", linewidth=2.0, label="Observations")
            ax.plot(x_vals, model_means, marker="s", color="royalblue", linewidth=1.5, label="ML Forecast (Mean)")
            ax.fill_between(x_vals, model_low, model_high, color="royalblue", alpha=0.2, label="ML 10th-90th Percentile")
            
            ax.plot(x_vals, geos_means, marker="^", color="darkorange", linewidth=1.5, label="GEOS Forecast (Mean)")
            ax.fill_between(x_vals, geos_low, geos_high, color="darkorange", alpha=0.2, label="GEOS 10th-90th Percentile")
            
            unit_label = f"({args.temperature_units})" if args.event_variable == "t2m" else f"({spec_names['units']})"
            ax.set_title(f"{args.event_name}: Forecast Evolution for Init Date {init_dt.strftime('%Y-%m-%d')}")
            ax.set_xlabel("Lead Week (Valid Date)")
            ax.set_ylabel(f"Domain Area-weighted Mean {args.event_variable.upper()} {unit_label}")
            ax.set_xticks(x_vals)
            ax.set_xticklabels([f"Week {l}\n({lbl})" for l, lbl in zip(leads, valid_labels)])
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
            
            plot_path2 = os.path.join(out_dir, f"event_forecast_evolution_init_{init_dt.strftime('%Y-%m-%d')}.png")
            fig.savefig(plot_path2, dpi=150)
            plt.close(fig)
            print(f"📈 Saved forecast evolution plot to: {plot_path2}")
        else:
            print("⚠️ No initialization date found before or at the start of the event window.")

        # Generate Plot 3: Target-Period Forecast Convergence Plot
        # (Aggregates only the forecasts that target valid dates within the event window,
        # comparing different lead times/initializations for the same target event)
        event_cases = [r for r in rows if r["in_event_window"]]
        if len(event_cases) >= 1:
            # Group by lead week
            lead_groups = {}
            for r in event_cases:
                lead = r["lead"]
                if lead not in lead_groups:
                    lead_groups[lead] = []
                lead_groups[lead].append(r)
                
            plot_leads = sorted(lead_groups.keys(), reverse=True) # E.g., [4, 3, 2, 1]
            
            leads_x = []
            obs_vals = []
            model_means = []
            geos_means = []
            model_lows = []
            model_highs = []
            geos_lows = []
            geos_highs = []
            init_dates_labels = []
            
            for pl in plot_leads:
                cases_at_lead = lead_groups[pl]
                # Average metrics over all valid dates inside event window for this lead week
                obs_mean_val = np.mean([c["obs_mean"] for c in cases_at_lead])
                m_mean_val = np.mean([c["model_mean"] for c in cases_at_lead])
                g_mean_val = np.mean([c["geos_mean"] for c in cases_at_lead])
                
                # Retrieve individual member forecasts for spread calculations
                m_ens_all = []
                g_ens_all = []
                for c in cases_at_lead:
                    fields = c["raw_fields"]
                    m_ens_means = [weighted_mean(fields["model_ens"][e], weights) for e in range(fields["model_ens"].shape[0])]
                    g_ens_means = [weighted_mean(fields["geos_ens"][e], weights) for e in range(fields["geos_ens"].shape[0])]
                    if args.event_variable == "t2m" and args.temperature_units == "C":
                        m_ens_means = [v - 273.15 for v in m_ens_means]
                        g_ens_means = [v - 273.15 for v in g_ens_means]
                    m_ens_all.append(m_ens_means)
                    g_ens_all.append(g_ens_means)
                
                # Combine across valid dates for this lead
                m_ens_combined = np.mean(m_ens_all, axis=0) # Average each member across the event window cases
                g_ens_combined = np.mean(g_ens_all, axis=0)
                
                leads_x.append(pl)
                obs_vals.append(obs_mean_val)
                model_means.append(m_mean_val)
                geos_means.append(g_mean_val)
                model_lows.append(np.percentile(m_ens_combined, 10))
                model_highs.append(np.percentile(m_ens_combined, 90))
                geos_lows.append(np.percentile(g_ens_combined, 10))
                geos_highs.append(np.percentile(g_ens_combined, 90))
                
                # Form label showing initialization date range
                inits = sorted(list(set([c["init"].strftime("%m-%d") for c in cases_at_lead])))
                init_dates_labels.append(", ".join(inits))
                
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            x_vals = np.arange(len(leads_x))
            
            # Observations (constant or mean observations over the targeted window)
            obs_mean_event = np.mean(obs_vals)
            ax.axhline(obs_mean_event, color="black", linestyle="--", linewidth=2.0, label=f"Observed Mean ({obs_mean_event:.2f})")
            
            # Plot model and geos means and spreads
            ax.plot(x_vals, model_means, marker="s", color="royalblue", linewidth=2.0, label="ML Forecast (Mean)")
            ax.fill_between(x_vals, model_lows, model_highs, color="royalblue", alpha=0.2, label="ML 10th-90th Percentile")
            
            ax.plot(x_vals, geos_means, marker="^", color="darkorange", linewidth=2.0, label="GEOS Forecast (Mean)")
            ax.fill_between(x_vals, geos_lows, geos_highs, color="darkorange", alpha=0.2, label="GEOS 10th-90th Percentile")
            
            unit_label = f"({args.temperature_units})" if args.event_variable == "t2m" else f"({spec_names['units']})"
            ax.set_title(f"{args.event_name}: Forecast Convergence for Event Window ({args.event_start} to {args.event_end})")
            ax.set_xlabel("Lead Week (Initialization Date)")
            ax.set_ylabel(f"Domain Area-weighted Mean {args.event_variable.upper()} {unit_label}")
            ax.set_xticks(x_vals)
            
            # Label format: "Week 4\n(Init: 05-19)"
            x_labels = [f"Week {l}\n(Init: {lbl})" for l, lbl in zip(leads_x, init_dates_labels)]
            ax.set_xticklabels(x_labels)
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
            
            plot_path3 = os.path.join(out_dir, "event_forecast_convergence.png")
            fig.savefig(plot_path3, dpi=150)
            plt.close(fig)
            print(f"📈 Saved forecast convergence plot to: {plot_path3}")
        else:
            print("⚠️ No cases found falling in the event window for Plot 3.")
            
    finally:
        ds.close()

if __name__ == "__main__":
    main()
