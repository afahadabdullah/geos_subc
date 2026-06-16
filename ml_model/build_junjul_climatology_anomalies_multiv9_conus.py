#!/usr/bin/env python3
"""
Build June/July climatology and anomaly Zarrs from generated multi-v9 SA forecasts.

Climatology:
  - 2005-2024 by default, skipping unavailable years such as 2017.
  - Separate init-month climatologies for June and July.
  - ML and GEOS means are averaged over init dates and ensemble members.
  - Ensemble spread climatology is saved as mean ensemble std over init dates.

Anomalies:
  - 2021-2023 by default.
  - System-specific anomalies: ML - ML climatology, GEOS - GEOS climatology,
    OBS - OBS climatology.
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_FORECAST_DIR = "dataprocess/gen_multiv9_conus_125w66w_24n50n_junjul_e10clim_e100eval_s50"
DEFAULT_OUTPUT_DIR = "dataprocess/clim_anom_multiv9_conus_125w66w_24n50n_junjul"


def parse_args():
    parser = argparse.ArgumentParser(description="Build v9 SA June/July climatology and anomalies.")
    parser.add_argument("--forecast_dir", type=str, default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clim_start_year", type=int, default=2005)
    parser.add_argument("--clim_end_year", type=int, default=2024)
    parser.add_argument("--anom_start_year", type=int, default=2021)
    parser.add_argument("--anom_end_year", type=int, default=2023)
    parser.add_argument("--months", type=str, default="6,7")
    parser.add_argument("--skip_years", type=str, default="2017")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_set(text):
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def parse_months(text):
    months = tuple(sorted(parse_int_set(text)))
    if not months:
        raise ValueError("--months cannot be empty")
    bad = [m for m in months if m < 1 or m > 12]
    if bad:
        raise ValueError(f"Invalid months: {bad}")
    return months


def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def open_year(forecast_dir, year):
    path = os.path.join(forecast_dir, f"{year}.zarr")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_zarr(path, consolidated=False, chunks=None)


def find_template_dataset(forecast_dir, years):
    for year in years:
        path = os.path.join(forecast_dir, f"{year}.zarr")
        if os.path.exists(path):
            return open_year(forecast_dir, year)
    raise FileNotFoundError(f"No yearly Zarr found in {forecast_dir} for years={years}")


def init_month_mask(ds, month):
    return pd.to_datetime(ds["init"].values).month == int(month)


def finite_sum_count(values, axes):
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    return clean.sum(axis=axes, dtype=np.float64), finite.sum(axis=axes).astype(np.float64)


def init_accumulator(months, lead_size, lat_size, lon_size):
    shape = (len(months), lead_size, lat_size, lon_size)
    accum = {
        "sum": {},
        "count": {},
        "spread_sum": {},
        "spread_count": {},
        "obs_init_sum": {},
        "obs_init_sumsq": {},
        "obs_init_count": {},
        "n_init": np.zeros((len(months),), dtype=np.int64),
    }
    for name in ("ml_pr", "ml_t2m", "geos_pr", "geos_t2m", "obs_pr", "obs_t2m"):
        accum["sum"][name] = np.zeros(shape, dtype=np.float64)
        accum["count"][name] = np.zeros(shape, dtype=np.float64)
    for name in ("ml_pr", "ml_t2m", "geos_pr", "geos_t2m"):
        accum["spread_sum"][name] = np.zeros(shape, dtype=np.float64)
        accum["spread_count"][name] = np.zeros(shape, dtype=np.float64)
    for name in ("obs_pr", "obs_t2m"):
        accum["obs_init_sum"][name] = np.zeros(shape, dtype=np.float64)
        accum["obs_init_sumsq"][name] = np.zeros(shape, dtype=np.float64)
        accum["obs_init_count"][name] = np.zeros(shape, dtype=np.float64)
    return accum


def add_member_field(accum, month_idx, name, values):
    # values: [init, member, lead, lat, lon]
    total, count = finite_sum_count(values, axes=(0, 1))
    accum["sum"][name][month_idx] += total
    accum["count"][name][month_idx] += count

    if values.shape[1] > 1:
        spread = np.nanstd(values, axis=1, ddof=1)
    else:
        spread = np.zeros(values.shape[0:1] + values.shape[2:], dtype=np.float32)
    spread_total, spread_count = finite_sum_count(spread, axes=(0,))
    accum["spread_sum"][name][month_idx] += spread_total
    accum["spread_count"][name][month_idx] += spread_count


def add_obs_field(accum, month_idx, name, values):
    # values: [init, lead, lat, lon]
    total, count = finite_sum_count(values, axes=(0,))
    accum["sum"][name][month_idx] += total
    accum["count"][name][month_idx] += count
    accum["obs_init_sum"][name][month_idx] += total
    accum["obs_init_sumsq"][name][month_idx] += np.nansum(values.astype(np.float64) ** 2, axis=0)
    accum["obs_init_count"][name][month_idx] += count


def safe_divide(num, den):
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float64), where=den > 0)


def finalize_climatology(accum, months, lead, lat, lon, attrs):
    data_vars = {}
    for name in ("ml_pr", "ml_t2m", "geos_pr", "geos_t2m", "obs_pr", "obs_t2m"):
        mean = safe_divide(accum["sum"][name], accum["count"][name]).astype(np.float32)
        data_vars[f"{name}_clim"] = (("month", "lead", "lat", "lon"), mean)

    for name in ("ml_pr", "ml_t2m", "geos_pr", "geos_t2m"):
        spread = safe_divide(accum["spread_sum"][name], accum["spread_count"][name]).astype(np.float32)
        data_vars[f"{name}_ens_spread_clim"] = (("month", "lead", "lat", "lon"), spread)

    for name in ("obs_pr", "obs_t2m"):
        count = accum["obs_init_count"][name]
        mean = safe_divide(accum["obs_init_sum"][name], count)
        variance = safe_divide(accum["obs_init_sumsq"][name], count) - mean ** 2
        std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
        data_vars[f"{name}_init_std_clim"] = (("month", "lead", "lat", "lon"), std)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "month": np.asarray(months, dtype=np.int32),
            "lead": np.asarray(lead, dtype=np.int32),
            "lat": np.asarray(lat, dtype=np.float32),
            "lon": np.asarray(lon, dtype=np.float32),
        },
        attrs=attrs,
    )
    ds["n_init"] = ("month", accum["n_init"].astype(np.int32))
    units = {"pr": "mm/day", "t2m": "K"}
    for prefix in ("ml", "geos", "obs"):
        for var in ("pr", "t2m"):
            ds[f"{prefix}_{var}_clim"].attrs["units"] = units[var]
    for prefix in ("ml", "geos"):
        for var in ("pr", "t2m"):
            ds[f"{prefix}_{var}_ens_spread_clim"].attrs["units"] = units[var]
    return ds


def build_climatology(args, months, skip_years, clim_path):
    years = [y for y in range(args.clim_start_year, args.clim_end_year + 1) if y not in skip_years]
    template = find_template_dataset(args.forecast_dir, years)
    try:
        lead = template["lead"].values
        lat = template["lat"].values
        lon = template["lon"].values
    finally:
        template.close()

    accum = init_accumulator(months, len(lead), len(lat), len(lon))
    loaded_years = []
    missing_years = []

    for year in years:
        path = os.path.join(args.forecast_dir, f"{year}.zarr")
        if not os.path.exists(path):
            print(f"⚠️ Missing climatology year {year}: {path}. Skipping.")
            missing_years.append(year)
            continue
        ds = open_year(args.forecast_dir, year)
        try:
            loaded_years.append(year)
            for month_idx, month in enumerate(months):
                mask = init_month_mask(ds, month)
                if not np.any(mask):
                    print(f"⚠️ {year}: no init dates for month={month}.")
                    continue
                sub = ds.isel(init=np.where(mask)[0])
                accum["n_init"][month_idx] += int(sub.sizes["init"])
                add_member_field(accum, month_idx, "ml_pr", sub["model_pr"].values.astype(np.float32, copy=False))
                add_member_field(accum, month_idx, "ml_t2m", sub["model_t2m"].values.astype(np.float32, copy=False))
                add_member_field(accum, month_idx, "geos_pr", sub["geos_pr"].values.astype(np.float32, copy=False))
                add_member_field(accum, month_idx, "geos_t2m", sub["geos_t2m"].values.astype(np.float32, copy=False))
                add_obs_field(accum, month_idx, "obs_pr", sub["obs_pr"].values.astype(np.float32, copy=False))
                add_obs_field(accum, month_idx, "obs_t2m", sub["obs_t2m"].values.astype(np.float32, copy=False))
        finally:
            ds.close()

    attrs = {
        "generated_by": os.path.basename(__file__),
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "climatology_years_requested": f"{args.clim_start_year}-{args.clim_end_year}",
        "climatology_years_loaded": ",".join(str(y) for y in loaded_years),
        "climatology_years_missing": ",".join(str(y) for y in missing_years),
        "skip_years": ",".join(str(y) for y in sorted(skip_years)),
        "months": ",".join(str(m) for m in months),
        "description": "June/July init-month, lead-specific climatology. ML/GEOS means average over init dates and ensemble members.",
    }
    ds_clim = finalize_climatology(accum, months, lead, lat, lon, attrs)
    encoding = {
        name: {"dtype": "float32", "chunks": (1, len(lead), len(lat), len(lon))}
        for name in ds_clim.data_vars
        if name != "n_init"
    }
    encoding["n_init"] = {"dtype": "int32", "chunks": (1,)}
    ds_clim.to_zarr(clim_path, mode="w", encoding=encoding)
    ds_clim.close()
    print(f"✅ Climatology saved: {clim_path}")
    return loaded_years, missing_years


def make_valid_time(init_values, lead_values):
    init_dates = pd.to_datetime(init_values)
    lead_days = pd.to_timedelta(np.asarray(lead_values, dtype=np.int64) * 7, unit="D")
    return init_dates.values[:, None] + lead_days.values[None, :]


def anomaly_dataset_for_subset(sub, clim, month):
    lead = sub["lead"].values
    lat = sub["lat"].values
    lon = sub["lon"].values
    month_clim = clim.sel(month=int(month))

    ml_pr = sub["model_pr"].values.astype(np.float32, copy=False) - month_clim["ml_pr_clim"].values[None, None]
    ml_t2m = sub["model_t2m"].values.astype(np.float32, copy=False) - month_clim["ml_t2m_clim"].values[None, None]
    geos_pr = sub["geos_pr"].values.astype(np.float32, copy=False) - month_clim["geos_pr_clim"].values[None, None]
    geos_t2m = sub["geos_t2m"].values.astype(np.float32, copy=False) - month_clim["geos_t2m_clim"].values[None, None]
    obs_pr = sub["obs_pr"].values.astype(np.float32, copy=False) - month_clim["obs_pr_clim"].values[None]
    obs_t2m = sub["obs_t2m"].values.astype(np.float32, copy=False) - month_clim["obs_t2m_clim"].values[None]

    init_values = sub["init"].values
    ds = xr.Dataset(
        data_vars={
            "ml_pr_anom": (("init", "ensemble", "lead", "lat", "lon"), ml_pr.astype(np.float32, copy=False)),
            "ml_t2m_anom": (("init", "ensemble", "lead", "lat", "lon"), ml_t2m.astype(np.float32, copy=False)),
            "geos_pr_anom": (("init", "geos_member", "lead", "lat", "lon"), geos_pr.astype(np.float32, copy=False)),
            "geos_t2m_anom": (("init", "geos_member", "lead", "lat", "lon"), geos_t2m.astype(np.float32, copy=False)),
            "obs_pr_anom": (("init", "lead", "lat", "lon"), obs_pr.astype(np.float32, copy=False)),
            "obs_t2m_anom": (("init", "lead", "lat", "lon"), obs_t2m.astype(np.float32, copy=False)),
            "ml_pr_anom_mean": (("init", "lead", "lat", "lon"), np.nanmean(ml_pr, axis=1).astype(np.float32)),
            "ml_t2m_anom_mean": (("init", "lead", "lat", "lon"), np.nanmean(ml_t2m, axis=1).astype(np.float32)),
            "geos_pr_anom_mean": (("init", "lead", "lat", "lon"), np.nanmean(geos_pr, axis=1).astype(np.float32)),
            "geos_t2m_anom_mean": (("init", "lead", "lat", "lon"), np.nanmean(geos_t2m, axis=1).astype(np.float32)),
            "ml_pr_anom_spread": (("init", "lead", "lat", "lon"), np.nanstd(ml_pr, axis=1, ddof=1).astype(np.float32)),
            "ml_t2m_anom_spread": (("init", "lead", "lat", "lon"), np.nanstd(ml_t2m, axis=1, ddof=1).astype(np.float32)),
            "geos_pr_anom_spread": (("init", "lead", "lat", "lon"), np.nanstd(geos_pr, axis=1, ddof=1).astype(np.float32)),
            "geos_t2m_anom_spread": (("init", "lead", "lat", "lon"), np.nanstd(geos_t2m, axis=1, ddof=1).astype(np.float32)),
        },
        coords={
            "init": init_values,
            "ensemble": sub["ensemble"].values,
            "geos_member": sub["geos_member"].values,
            "lead": lead,
            "lat": lat,
            "lon": lon,
            "init_month": ("init", np.full((len(init_values),), int(month), dtype=np.int32)),
            "valid_time": (("init", "lead"), make_valid_time(init_values, lead)),
        },
    )
    return ds


def build_anomalies(args, months, skip_years, clim_path, anom_path):
    if os.path.exists(anom_path):
        if not args.overwrite:
            raise FileExistsError(f"Anomaly output already exists: {anom_path}. Use --overwrite.")
        remove_path(anom_path)

    clim = xr.open_zarr(clim_path, consolidated=False, chunks=None)
    wrote_any = False
    written_years = []
    try:
        for year in range(args.anom_start_year, args.anom_end_year + 1):
            if year in skip_years:
                print(f"⏭️ {year}: skipped by --skip_years.")
                continue
            path = os.path.join(args.forecast_dir, f"{year}.zarr")
            if not os.path.exists(path):
                print(f"⚠️ Missing anomaly year {year}: {path}. Skipping.")
                continue
            ds = open_year(args.forecast_dir, year)
            try:
                for month in months:
                    mask = init_month_mask(ds, month)
                    if not np.any(mask):
                        continue
                    sub = ds.isel(init=np.where(mask)[0])
                    ds_anom = anomaly_dataset_for_subset(sub, clim, month)
                    ds_anom.attrs.update({
                        "generated_by": os.path.basename(__file__),
                        "forecast_dir": os.path.abspath(args.forecast_dir),
                        "climatology_path": os.path.abspath(clim_path),
                        "anomaly_years_requested": f"{args.anom_start_year}-{args.anom_end_year}",
                        "description": "System-specific June/July anomaly forecasts and observations.",
                    })
                    encoding = {}
                    for name in ds_anom.data_vars:
                        if name == "init_month":
                            encoding[name] = {"dtype": "int32", "chunks": (max(1, min(8, ds_anom.sizes["init"])),)}
                        elif "ensemble" in ds_anom[name].dims:
                            encoding[name] = {"dtype": "float32", "chunks": (1, min(10, ds_anom.sizes["ensemble"]), 4, len(ds_anom.lat), len(ds_anom.lon))}
                        elif "geos_member" in ds_anom[name].dims:
                            encoding[name] = {"dtype": "float32", "chunks": (1, ds_anom.sizes["geos_member"], 4, len(ds_anom.lat), len(ds_anom.lon))}
                        else:
                            encoding[name] = {"dtype": "float32", "chunks": (1, 4, len(ds_anom.lat), len(ds_anom.lon))}
                    if not wrote_any:
                        ds_anom.to_zarr(anom_path, mode="w", encoding=encoding)
                        wrote_any = True
                    else:
                        ds_anom.to_zarr(anom_path, mode="a", append_dim="init")
                    ds_anom.close()
                written_years.append(year)
            finally:
                ds.close()
    finally:
        clim.close()

    if not wrote_any:
        raise RuntimeError("No anomaly data were written.")
    print(f"✅ Anomalies saved: {anom_path}")
    return written_years


def main():
    args = parse_args()
    months = parse_months(args.months)
    skip_years = parse_int_set(args.skip_years)
    os.makedirs(args.output_dir, exist_ok=True)
    clim_path = os.path.join(args.output_dir, f"v9_junjul_climatology_{args.clim_start_year}_{args.clim_end_year}.zarr")
    anom_path = os.path.join(args.output_dir, f"v9_junjul_anomalies_{args.anom_start_year}_{args.anom_end_year}.zarr")
    metadata_path = os.path.join(args.output_dir, "v9_junjul_climatology_anomaly_metadata.json")

    if os.path.exists(clim_path):
        if args.overwrite:
            remove_path(clim_path)
        else:
            print(f"✅ Climatology exists: {clim_path}")
            loaded_years = []
            missing_years = []
    else:
        loaded_years, missing_years = build_climatology(args, months, skip_years, clim_path)

    written_anom_years = build_anomalies(args, months, skip_years, clim_path, anom_path)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "forecast_dir": args.forecast_dir,
                "output_dir": args.output_dir,
                "climatology_path": clim_path,
                "anomaly_path": anom_path,
                "months": list(months),
                "skip_years": sorted(skip_years),
                "climatology_years": [args.clim_start_year, args.clim_end_year],
                "climatology_years_loaded_when_built": loaded_years,
                "climatology_years_missing_when_built": missing_years,
                "anomaly_years": [args.anom_start_year, args.anom_end_year],
                "anomaly_years_written": written_anom_years,
            },
            f,
            indent=2,
        )

    print("\nJune/July climatology/anomaly build complete")
    print(f"  Climatology : {clim_path}")
    print(f"  Anomalies   : {anom_path}")
    print(f"  Metadata    : {metadata_path}")


if __name__ == "__main__":
    main()
