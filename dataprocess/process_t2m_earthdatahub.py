"""
EarthDataHub ERA5 T2M -> Weekly GEOS-Aligned Zarr
=================================================
Builds ``t2m_weekly_{year}.zarr`` directly from the Earth Data Hub ERA5
single-levels Zarr archive, using the same future lead-week target convention
as ``process_gpcp.py``.

For each GEOS init date ``S`` we compute four future observed weekly means:
    L=0 -> [S,    S+6]
    L=1 -> [S+7,  S+13]
    L=2 -> [S+14, S+20]
    L=3 -> [S+21, S+27]

Authentication
--------------
The script reads a personal access token from an environment variable and
creates a temporary ``.netrc`` file so the official Earth Data Hub access
syntax continues to work:

    machine data.earthdatahub.destine.eu
        password <token>

Environment variables:
    EARTHDATAHUB_PAT           Required. Earth Data Hub personal access token.
    EARTHDATAHUB_URL           Optional override for the source Zarr URL.

Example:
    export EARTHDATAHUB_PAT='...'
    python dataprocess/process_t2m_earthdatahub.py --years 1999 2000 2001
"""

from __future__ import annotations

import argparse
import contextlib
import os
import stat
import tempfile
from typing import Iterable, Tuple

import dask
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


EARTHDATAHUB_PAT_ENV = "EARTHDATAHUB_PAT"
EARTHDATAHUB_URL_ENV = "EARTHDATAHUB_URL"
EARTHDATAHUB_MACHINE = "data.earthdatahub.destine.eu"
DEFAULT_EDH_URL = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
TIME_CANDIDATES = ["time", "valid_time", "forecast_time", "date"]

GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"
DEFAULT_YEARS = list(range(1999, 2026))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GEOS-aligned weekly ERA5 T2M targets directly from Earth Data Hub."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="Years to process. Defaults to 1999 through 2025.",
    )
    parser.add_argument(
        "--geos_dir",
        type=str,
        default=GEOS_DIR,
        help=f"Directory containing geos_subc_<year>.zarr (default: {GEOS_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to write t2m_weekly_<year>.zarr (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing t2m_weekly_<year>.zarr output.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def earthdatahub_netrc_from_env():
    """Create a temporary .netrc from the Earth Data Hub token env var."""
    token = os.environ.get(EARTHDATAHUB_PAT_ENV)
    if not token:
        raise RuntimeError(
            f"Missing required environment variable {EARTHDATAHUB_PAT_ENV}. "
            "Set it to your Earth Data Hub personal access token before running."
        )

    with tempfile.TemporaryDirectory(prefix="edh_netrc_") as tmpdir:
        netrc_path = os.path.join(tmpdir, ".netrc")
        with open(netrc_path, "w", encoding="utf-8") as f:
            f.write(f"machine {EARTHDATAHUB_MACHINE}\n")
            f.write(f"    password {token}\n")
        os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)

        old_home = os.environ.get("HOME")
        old_netrc = os.environ.get("NETRC")
        os.environ["HOME"] = tmpdir
        os.environ["NETRC"] = netrc_path
        try:
            yield
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home

            if old_netrc is None:
                os.environ.pop("NETRC", None)
            else:
                os.environ["NETRC"] = old_netrc


def choose_coord_name(ds: xr.Dataset | xr.DataArray, candidates: Iterable[str], label: str) -> str:
    items = set(ds.coords) | set(ds.dims)
    for name in candidates:
        if name in items:
            return name
    raise KeyError(f"Could not find {label}. Tried {list(candidates)}. Available: {sorted(items)}")


def choose_dim_name(ds: xr.Dataset | xr.DataArray, candidates: Iterable[str], label: str) -> str:
    dims = set(ds.dims)
    for name in candidates:
        if name in dims:
            return name
    return choose_coord_name(ds, candidates, label)


def normalize_time_axis(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    time_name = choose_dim_name(obj, TIME_CANDIDATES, "time dimension")
    if time_name != "time":
        obj = obj.rename({time_name: "time"})
    return obj.assign_coords(time=pd.to_datetime(obj["time"].values)).sortby("time")


def open_remote_era5() -> xr.Dataset:
    url = os.environ.get(EARTHDATAHUB_URL_ENV, DEFAULT_EDH_URL)
    print(f"Opening Earth Data Hub ERA5 store: {url}")
    ds = xr.open_dataset(
        url,
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    if "t2m" not in ds.data_vars:
        raise KeyError(f"'t2m' not found in Earth Data Hub dataset. Available vars: {list(ds.data_vars)}")
    return ds


def load_geos_layout(geos_path: str) -> Tuple[pd.DatetimeIndex, xr.DataArray, xr.DataArray]:
    if not os.path.exists(geos_path):
        raise FileNotFoundError(f"Missing GEOS reference file: {geos_path}")

    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    try:
        s_name = choose_coord_name(ds_geos, ["S", "time", "init_time"], "GEOS init dimension")
        lat_name = choose_coord_name(ds_geos, ["Y", "latitude", "lat", "y"], "GEOS latitude coordinate")
        lon_name = choose_coord_name(ds_geos, ["X", "longitude", "lon", "x"], "GEOS longitude coordinate")

        init_dates = pd.to_datetime(ds_geos[s_name].values)
        target_lat = ds_geos[lat_name].copy(deep=True)
        target_lon = ds_geos[lon_name].copy(deep=True)
    finally:
        ds_geos.close()

    return init_dates, target_lat, target_lon


def normalize_longitudes(da: xr.DataArray, lon_name: str, target_lon: xr.DataArray) -> xr.DataArray:
    lon = da[lon_name]
    target_min = float(np.nanmin(target_lon.values))
    target_max = float(np.nanmax(target_lon.values))
    source_min = float(np.nanmin(lon.values))
    source_max = float(np.nanmax(lon.values))

    if target_min >= 0.0 and source_min < 0.0:
        da = da.assign_coords({lon_name: np.mod(da[lon_name], 360.0)})
        da = da.sortby(lon_name)
    elif target_max <= 180.0 and source_max > 180.0:
        shifted = ((da[lon_name] + 180.0) % 360.0) - 180.0
        da = da.assign_coords({lon_name: shifted})
        da = da.sortby(lon_name)

    return da


def normalize_source_grid(da: xr.DataArray, target_lat: xr.DataArray, target_lon: xr.DataArray) -> xr.DataArray:
    lat_name = choose_coord_name(da, ["latitude", "lat", "Y", "y"], "source latitude coordinate")
    lon_name = choose_coord_name(da, ["longitude", "lon", "X", "x"], "source longitude coordinate")

    da = normalize_longitudes(da, lon_name, target_lon)
    da = da.sortby(lat_name)

    rename_dict = {}
    if lat_name != target_lat.name:
        rename_dict[lat_name] = target_lat.name
    if lon_name != target_lon.name:
        rename_dict[lon_name] = target_lon.name
    if rename_dict:
        da = da.rename(rename_dict)

    return da


def build_daily_regridded_t2m(
    ds_remote: xr.Dataset,
    init_dates: pd.DatetimeIndex,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.DataArray:
    min_init = pd.Timestamp(init_dates.min()).normalize()
    max_init = pd.Timestamp(init_dates.max()).normalize()
    start_date = min_init.strftime("%Y-%m-%d")
    end_date = (max_init + pd.Timedelta(days=27)).strftime("%Y-%m-%d")

    time_name = choose_dim_name(ds_remote["t2m"], TIME_CANDIDATES, "time dimension")
    print(f"  Selecting remote t2m from {start_date} to {end_date} using '{time_name}'")
    da_hourly = ds_remote["t2m"].sel({time_name: slice(start_date, end_date)})
    da_hourly = normalize_time_axis(da_hourly)
    if da_hourly.sizes.get("time", 0) == 0:
        raise RuntimeError(f"No Earth Data Hub t2m data found between {start_date} and {end_date}")

    print("  Resampling hourly data to daily means...")
    da_daily = da_hourly.resample(time="1D").mean()

    print("  Normalizing source coordinate names/orientation...")
    da_daily = normalize_source_grid(da_daily, target_lat, target_lon)

    print(f"  Interpolating daily means to GEOS grid ({len(target_lat)} x {len(target_lon)})...")
    da_daily_interp = da_daily.interp(
        {target_lat.name: target_lat.values, target_lon.name: target_lon.values},
        method="linear",
    )

    print("  Materializing regridded daily data into memory...")
    da_daily_interp = da_daily_interp.astype(np.float32).compute()
    return da_daily_interp


def nan_week(target_lat: xr.DataArray, target_lon: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        np.full((len(target_lat), len(target_lon)), np.nan, dtype=np.float32),
        dims=[target_lat.name, target_lon.name],
        coords={target_lat.name: target_lat, target_lon.name: target_lon},
    )


def build_weekly_targets(
    da_daily: xr.DataArray,
    init_dates: pd.DatetimeIndex,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.Dataset:
    da_daily = normalize_time_axis(da_daily)
    processed = []
    skipped = 0

    for init_date in tqdm(init_dates, desc="  Weekly T2M"):
        weeks = []

        for w in range(4):
            w_start = init_date + pd.Timedelta(days=w * 7)
            w_end = w_start + pd.Timedelta(days=6)
            chunk = da_daily.sel(time=slice(w_start, w_end))

            if chunk.sizes.get("time", 0) < 7:
                weeks.append(nan_week(target_lat, target_lon))
            else:
                weeks.append(chunk.mean(dim="time").astype(np.float32))

        if any(np.isnan(week.values).all() for week in weeks):
            skipped += 1

        sample = xr.concat(weeks, dim="L").assign_coords(L=np.arange(4))
        processed.append(sample)

    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} init dates contain at least one NaN-filled week")

    da_out = xr.concat(processed, dim="S").assign_coords(S=init_dates)
    da_out.name = "t2m"
    ds_out = da_out.to_dataset()
    ds_out = ds_out.transpose("S", "L", target_lat.name, target_lon.name)
    ds_out["t2m"].attrs = {
        "units": "K",
        "long_name": "2 metre temperature (weekly mean)",
        "description": "Four future observed weekly means aligned to GEOS S2S lead weeks",
        "source": "Earth Data Hub ERA5 single levels",
    }
    return ds_out


def process_year(year: int, ds_remote: xr.Dataset, geos_dir: str, output_dir: str, overwrite: bool = False):
    out_path = os.path.join(output_dir, f"t2m_weekly_{year}.zarr")
    if os.path.exists(out_path):
        if not overwrite:
            print(f"Output already exists, skipping {year}: {out_path}")
            return
        print(f"Overwriting existing output: {out_path}")
        import shutil
        shutil.rmtree(out_path)

    geos_path = os.path.join(geos_dir, f"geos_subc_{year}.zarr")
    init_dates, target_lat, target_lon = load_geos_layout(geos_path)

    print(f"\n{'=' * 72}")
    print(f"Processing T2M weekly targets for {year}")
    print(f"  GEOS ref   : {geos_path}")
    print(f"  Init dates : {len(init_dates)} ({init_dates[0].date()} -> {init_dates[-1].date()})")
    print(f"  Output     : {out_path}")

    da_daily = build_daily_regridded_t2m(ds_remote, init_dates, target_lat, target_lon)
    ds_weekly = build_weekly_targets(da_daily, init_dates, target_lat, target_lon)

    os.makedirs(output_dir, exist_ok=True)
    print("  Saving weekly target Zarr...")
    with dask.config.set(scheduler="synchronous"):
        ds_weekly.to_zarr(out_path, mode="w", zarr_format=3)

    print(
        f"  Saved {out_path} with shape "
        f"(S={ds_weekly.sizes['S']}, L={ds_weekly.sizes['L']}, "
        f"{target_lat.name}={ds_weekly.sizes[target_lat.name]}, "
        f"{target_lon.name}={ds_weekly.sizes[target_lon.name]})"
    )


def main():
    args = parse_args()
    with earthdatahub_netrc_from_env():
        ds_remote = open_remote_era5()
        try:
            for year in args.years:
                process_year(
                    year=year,
                    ds_remote=ds_remote,
                    geos_dir=args.geos_dir,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                )
        finally:
            ds_remote.close()

    print("\nAll requested years complete.")


if __name__ == "__main__":
    main()
