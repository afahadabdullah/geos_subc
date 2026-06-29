#!/usr/bin/env python3
"""
Build observed-climatology extreme threshold maps for flow_finalv1_global.

This produces a small NetCDF that can be passed to
evaluate_matrix_suite_flow_finalv1_global.py --threshold_file.

Default definition:
  - PR event: observed PR >= local monthly 95th percentile, with optional
    minimum threshold in mm/day.
  - T2M event: observed T2M >= local monthly 95th percentile.

Using a long observed period, e.g. 1999-2020, keeps the extreme-event mask
independent of the 2021-2023 model-evaluation years.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr


SEASONS = ["DJF", "MAM", "JJA", "SON"]
MONTHS = [f"{month:02d}" for month in range(1, 13)]

VARIABLES = {
    "pr": {
        "patterns": ("gpcp_weekly_{year}.zarr", "gpcp/{year}.zarr"),
        "candidates": ("precip", "pr", "obs_pr", "gpcp", "precipitation", "total_precipitation"),
        "quantile_arg": "extreme_quantile_pr",
        "min_threshold_arg": "pr_min_threshold",
        "units": "mm/day",
    },
    "t2m": {
        "patterns": ("t2m_weekly_{year}.zarr", "t2m/{year}.zarr"),
        "candidates": ("t2m", "T2M", "obs_t2m", "tas", "TAS", "temperature_2m", "air_temperature"),
        "quantile_arg": "extreme_quantile_t2m",
        "min_threshold_arg": None,
        "units": "K",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build long-term observed extreme threshold maps.")
    parser.add_argument("--data_dir", type=str, default="dataprocess")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--skip_years", type=str, default="")
    parser.add_argument("--variables", type=str, default="pr,t2m")
    parser.add_argument("--grouping", choices=("pooled", "monthly", "seasonal"), default="monthly")
    parser.add_argument("--extreme_quantile_pr", type=float, default=0.95)
    parser.add_argument("--extreme_quantile_t2m", type=float, default=0.95)
    parser.add_argument("--pr_min_threshold", type=float, default=5.0)
    parser.add_argument(
        "--out_file",
        type=str,
        default="ml_model/observed_extreme_thresholds_flow_finalv1_global_1999_2020_monthly.nc",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_years(text):
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def parse_variables(text):
    variables = [item.strip().lower() for item in str(text).split(",") if item.strip()]
    bad = [variable for variable in variables if variable not in VARIABLES]
    if bad:
        raise ValueError(f"Unknown variables {bad}; expected subset of {sorted(VARIABLES)}")
    if not variables:
        raise ValueError("--variables cannot be empty")
    return variables


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


def finite_nanmean(field):
    arr = np.asarray(field, dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def find_year_zarr(data_dir, variable, year):
    for pattern in VARIABLES[variable]["patterns"]:
        path = os.path.join(data_dir, pattern.format(year=year))
        if os.path.exists(path):
            return path
    return None


def find_data_variable(ds, variable):
    for name in VARIABLES[variable]["candidates"]:
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(
        f"Could not find observed {variable} variable. "
        f"Tried {VARIABLES[variable]['candidates']}; available={list(ds.data_vars)}"
    )


def find_dim(data_array, candidates):
    dims = list(data_array.dims)
    lowered = {str(dim).lower(): dim for dim in dims}
    for candidate in candidates:
        if candidate in dims:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def coordinate_values(ds, dim, size, fallback_start=0.0, fallback_stop=None):
    if dim in ds.coords:
        return ds[dim].values
    if fallback_stop is None:
        return np.arange(size, dtype=np.float32)
    return np.linspace(fallback_start, fallback_stop, size, dtype=np.float32)


def infer_grid_and_dims(ds, data_array):
    init_dim = find_dim(data_array, ("S", "init", "time"))
    lead_dim = find_dim(data_array, ("L", "lead"))
    lat_dim = find_dim(data_array, ("lat", "latitude", "Y"))
    lon_dim = find_dim(data_array, ("lon", "longitude", "X"))
    missing = [
        name
        for name, value in (("init", init_dim), ("lead", lead_dim), ("lat", lat_dim), ("lon", lon_dim))
        if value is None
    ]
    if missing:
        raise ValueError(f"Could not identify dims {missing} for {data_array.name}; dims={data_array.dims}")
    lat_size = int(data_array.sizes[lat_dim])
    lon_size = int(data_array.sizes[lon_dim])
    lats = coordinate_values(ds, lat_dim, lat_size, -90.0, 90.0)
    lons = coordinate_values(ds, lon_dim, lon_size, 0.0, 359.0)
    return init_dim, lead_dim, lat_dim, lon_dim, lats, lons


def init_times(ds, init_dim):
    if init_dim in ds.coords:
        values = ds[init_dim].values
        try:
            return pd.to_datetime(values).normalize()
        except Exception as exc:
            raise ValueError(f"Initialization coordinate {init_dim} is not datetime-like.") from exc
    raise ValueError(f"Dataset is missing datetime coordinate for init dim {init_dim}.")


def lead_days(ds, lead_dim, lead_idx):
    if lead_dim in ds.coords:
        value = np.asarray(ds[lead_dim].values)[lead_idx]
        try:
            numeric = int(value)
            if 0 <= numeric <= 3:
                return int(lead_idx + 1) * 7
            if 1 <= numeric <= 4:
                return numeric * 7
        except Exception:
            pass
    return int(lead_idx + 1) * 7


def read_obs_slice(data_array, selectors, lat_dim, lon_dim):
    sliced = data_array.isel(selectors)
    if lat_dim in sliced.dims and lon_dim in sliced.dims:
        return sliced.transpose(lat_dim, lon_dim).values.astype(np.float32, copy=False)
    arr = np.squeeze(sliced.values).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D observed slice after selecting {selectors}, got shape {arr.shape}")
    return arr


def build_variable_threshold(data_dir, years, variable, args):
    spec = VARIABLES[variable]
    group_values = threshold_group_values(args.grouping)
    obs_by_group = {group: [] for group in group_values}
    saved_lats = None
    saved_lons = None

    print(f"📏 {variable}: reading observed data for {years[0]}-{years[-1]} ({args.grouping})")
    for year in years:
        zarr_path = find_year_zarr(data_dir, variable, year)
        if zarr_path is None:
            raise FileNotFoundError(f"No observed {variable} Zarr found for {year} under {data_dir}")
        ds = xr.open_zarr(zarr_path, consolidated=False, chunks=None)
        try:
            data_name = find_data_variable(ds, variable)
            da = ds[data_name]
            init_dim, lead_dim, lat_dim, lon_dim, lats, lons = infer_grid_and_dims(ds, da)
            if saved_lats is None:
                saved_lats = np.asarray(lats)
                saved_lons = np.asarray(lons)
            elif len(saved_lats) != len(lats) or len(saved_lons) != len(lons):
                raise ValueError(f"Grid changed in {zarr_path}; expected {saved_lats.shape}/{saved_lons.shape}")
            init_values = init_times(ds, init_dim)
            for init_idx, init_time in enumerate(init_values):
                for lead_idx in range(int(da.sizes[lead_dim])):
                    valid_time = pd.Timestamp(init_time) + pd.to_timedelta(lead_days(ds, lead_dim, lead_idx), unit="D")
                    group = group_label_for_time(args.grouping, valid_time)
                    obs = read_obs_slice(
                        da,
                        {init_dim: init_idx, lead_dim: lead_idx},
                        lat_dim,
                        lon_dim,
                    )
                    obs_by_group[group].append(obs)
        finally:
            ds.close()

    q = float(getattr(args, spec["quantile_arg"]))
    min_arg = spec["min_threshold_arg"]
    threshold_maps = []
    frequency_maps = []
    shape = (len(saved_lats), len(saved_lons))
    for group in group_values:
        chunks = obs_by_group[group]
        if not chunks:
            threshold = np.full(shape, np.nan, dtype=np.float32)
            frequency = np.full(shape, np.nan, dtype=np.float32)
        else:
            stack = np.stack(chunks, axis=0).astype(np.float32, copy=False)
            threshold = np.nanquantile(stack, q, axis=0).astype(np.float32)
            if min_arg is not None:
                threshold = np.maximum(threshold, float(getattr(args, min_arg))).astype(np.float32)
            frequency = np.nanmean(stack >= threshold[None, :, :], axis=0).astype(np.float32)
        threshold_maps.append(threshold)
        frequency_maps.append(frequency)
        print(
            f"   {variable} {group}: n={len(chunks)}, threshold mean={finite_nanmean(threshold):.3f}, "
            f"event freq mean={finite_nanmean(frequency):.4f}"
        )

    if args.grouping == "pooled":
        thresholds = threshold_maps[0]
        frequencies = frequency_maps[0]
    else:
        thresholds = np.stack(threshold_maps, axis=0)
        frequencies = np.stack(frequency_maps, axis=0)
    return thresholds.astype(np.float32), frequencies.astype(np.float32), saved_lats, saved_lons, group_values


def main():
    args = parse_args()
    variables = parse_variables(args.variables)
    skip = parse_years(args.skip_years)
    years = [year for year in range(args.start_year, args.end_year + 1) if year not in skip]
    if not years:
        raise ValueError("No years selected.")
    if os.path.exists(args.out_file) and not args.overwrite:
        raise FileExistsError(f"{args.out_file} exists. Use --overwrite to replace it.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    data_vars = {}
    coords = {}
    saved_lats = None
    saved_lons = None
    group_values = threshold_group_values(args.grouping)
    if args.grouping != "pooled":
        coords["threshold_group"] = group_values

    for variable in variables:
        thresholds, frequencies, lats, lons, group_values = build_variable_threshold(
            args.data_dir,
            years,
            variable,
            args,
        )
        if saved_lats is None:
            saved_lats = np.asarray(lats, dtype=np.float32)
            saved_lons = np.asarray(lons, dtype=np.float32)
            coords["lat"] = saved_lats
            coords["lon"] = saved_lons
        elif not (np.allclose(saved_lats, lats, equal_nan=True) and np.allclose(saved_lons, lons, equal_nan=True)):
            raise ValueError(f"{variable} grid does not match the first variable grid.")

        if args.grouping == "pooled":
            dims = ("lat", "lon")
        else:
            dims = ("threshold_group", "lat", "lon")
        data_vars[f"{variable}_threshold"] = (dims, thresholds)
        data_vars[f"{variable}_obs_event_frequency"] = (dims, frequencies)

    ds = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "description": "Long-term observed extreme thresholds for flow_finalv1_global matrix evaluation.",
            "data_dir": os.path.abspath(args.data_dir),
            "years": ",".join(str(year) for year in years),
            "threshold_grouping": args.grouping,
            "extreme_quantile_pr": float(args.extreme_quantile_pr),
            "extreme_quantile_t2m": float(args.extreme_quantile_t2m),
            "pr_min_threshold": float(args.pr_min_threshold),
            "event_definition": "obs >= local observed percentile threshold selected by valid time group",
        },
    )
    ds.to_netcdf(args.out_file)
    print(f"✅ Wrote observed threshold file: {args.out_file}")


if __name__ == "__main__":
    main()
