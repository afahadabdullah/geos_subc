"""
EarthDataHub ERA5 Pressure Levels -> IVT Weekly GEOS-Aligned Zarr
=================================================================
Builds ``ivt_weekly_{year}.zarr`` directly from the Earth Data Hub ERA5
pressure-levels archive, matching the trailing-week target format used by the
existing IVT weekly processor.

For each GEOS init date ``S`` we compute four observed weekly means before the
forecast start:
    L=0 -> [S-28, S-22]
    L=1 -> [S-21, S-15]
    L=2 -> [S-14, S-8]
    L=3 -> [S-7,  S-1]

Required environment variable:
    EARTHDATAHUB_PAT

Example:
    export EARTHDATAHUB_PAT='...'
    python dataprocess/process_ivt_earthdatahub.py --years 2023 2024 2025
"""

from __future__ import annotations

import argparse
import os

import dask
import numpy as np
import xarray as xr

import earthdatahub_utils as edh


DEFAULT_EDH_URL = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-pressure-levels-v0.zarr"
EDH_URL_ENV = "EARTHDATAHUB_PRESSURE_LEVELS_URL"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"
IVT_LEVELS_HPA = [1000, 925, 850, 700, 600, 500]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GEOS-aligned weekly ERA5 IVT targets directly from Earth Data Hub."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="Years to process. Defaults to 2023 2024 2025.",
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
        help=f"Directory to write ivt_weekly_<year>.zarr (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing ivt_weekly_<year>.zarr output.",
    )
    return parser.parse_args()


def open_remote_pressure_levels() -> xr.Dataset:
    url = os.environ.get(EDH_URL_ENV, DEFAULT_EDH_URL)
    print(f"Opening Earth Data Hub pressure-levels store: {url}")
    ds = xr.open_dataset(
        url,
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
    )
    required = ["specific_humidity", "u_component_of_wind", "v_component_of_wind"]
    missing = [name for name in required if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing required IVT variables in remote dataset: {missing}")
    return ds


def calculate_ivt(ds: xr.Dataset, level_name: str, gravity: float = 9.80665) -> xr.DataArray:
    q = ds["specific_humidity"]
    u = ds["u_component_of_wind"]
    v = ds["v_component_of_wind"]

    p_pa = ds[level_name] * 100.0
    qu = (q * u).assign_coords({level_name: p_pa})
    qv = (q * v).assign_coords({level_name: p_pa})

    ivt_u = -(1.0 / gravity) * qu.integrate(coord=level_name)
    ivt_v = -(1.0 / gravity) * qv.integrate(coord=level_name)
    ivt = np.sqrt(ivt_u**2 + ivt_v**2)
    ivt.name = "ivt"
    ivt.attrs = {
        "units": "kg m-1 s-1",
        "long_name": "Integrated vapor transport magnitude",
        "source": "Earth Data Hub ERA5 pressure levels",
    }
    return ivt


def build_daily_ivt(
    ds_remote: xr.Dataset,
    init_dates,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.Dataset:
    level_name = edh.choose_coord_name(ds_remote, ["level", "pressure_level", "isobaricInhPa"], "pressure level coordinate")
    time_name = edh.choose_dim_name(ds_remote, edh.TIME_CANDIDATES, "time dimension")
    start_date, end_date = edh.build_time_window(init_dates)
    print(f"  Selecting pressure-level data from {start_date} to {end_date} using '{time_name}'")

    ds_sel = ds_remote[["specific_humidity", "u_component_of_wind", "v_component_of_wind"]].sel(
        {time_name: slice(start_date, end_date)},
        **{level_name: IVT_LEVELS_HPA},
    )
    ds_sel = edh.normalize_time_axis(ds_sel)
    if ds_sel.sizes.get("time", 0) == 0:
        raise RuntimeError(f"No pressure-level data found between {start_date} and {end_date}")

    print("  Calculating IVT on native pressure-level grid...")
    da_ivt = calculate_ivt(ds_sel, level_name=level_name)

    print("  Resampling IVT to daily means...")
    da_daily = da_ivt.resample(time="1D").mean()

    print("  Normalizing coordinate names/orientation...")
    da_daily = edh.normalize_source_grid(da_daily, target_lat, target_lon)

    print(f"  Interpolating IVT daily means to GEOS grid ({len(target_lat)} x {len(target_lon)})...")
    da_interp = da_daily.interp(
        {target_lat.name: target_lat.values, target_lon.name: target_lon.values},
        method="linear",
    )

    print("  Materializing regridded daily IVT into memory...")
    ds_daily = da_interp.astype(np.float32).to_dataset(name="ivt").compute()
    return ds_daily


def process_year(year: int, ds_remote: xr.Dataset, geos_dir: str, output_dir: str, overwrite: bool = False):
    out_path = os.path.join(output_dir, f"ivt_weekly_{year}.zarr")
    if os.path.exists(out_path):
        if not overwrite:
            print(f"Output already exists, skipping {year}: {out_path}")
            return
        print(f"Overwriting existing output: {out_path}")
        import shutil
        shutil.rmtree(out_path)

    geos_path = os.path.join(geos_dir, f"geos_subc_{year}.zarr")
    init_dates, target_lat, target_lon = edh.load_geos_layout(geos_path)

    print(f"\n{'=' * 72}")
    print(f"Processing IVT weekly targets for {year}")
    print(f"  GEOS ref   : {geos_path}")
    print(f"  Init dates : {len(init_dates)} ({init_dates[0].date()} -> {init_dates[-1].date()})")
    print(f"  Output     : {out_path}")

    ds_daily = build_daily_ivt(ds_remote, init_dates, target_lat, target_lon)
    ds_weekly = edh.weekly_means_from_daily_dataset(
        ds_daily,
        init_dates=init_dates,
        target_lat=target_lat,
        target_lon=target_lon,
        desc="  Weekly IVT",
    )
    ds_weekly["ivt"].attrs.update(
        {
            "units": "kg m-1 s-1",
            "long_name": "Integrated vapor transport magnitude (weekly mean)",
            "description": "Four trailing observed weekly means before GEOS S2S init dates",
            "source": "Earth Data Hub ERA5 pressure levels",
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    print("  Saving weekly IVT Zarr...")
    with dask.config.set(scheduler="synchronous"):
        ds_weekly.to_zarr(out_path, mode="w")

    print(
        f"  Saved {out_path} with shape "
        f"(S={ds_weekly.sizes['S']}, L={ds_weekly.sizes['L']}, "
        f"{target_lat.name}={ds_weekly.sizes[target_lat.name]}, "
        f"{target_lon.name}={ds_weekly.sizes[target_lon.name]})"
    )


def main():
    args = parse_args()
    with edh.earthdatahub_netrc_from_env():
        ds_remote = open_remote_pressure_levels()
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

    print("\nAll requested IVT years complete.")


if __name__ == "__main__":
    main()
