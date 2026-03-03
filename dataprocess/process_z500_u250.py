"""
Z500 & U250 Weekly Processing Script
======================================
Processes daily ERA5 Z500/U250 Zarr files into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Input:  Daily Zarr files (era5_z500_u250_{year}.zarr) in dataprocess/era5_z500_u250/
        Variables: 'z500' (geopotential at 500hPa), 'u250' (zonal wind at 250hPa)
Output: z500_u250_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid
        Contains both 'z500' and 'u250' variables.

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED data leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]  (oldest)
    L=3 → Week -1: [S-7,  S-1]   (most recent)

Usage:
    python dataprocess/process_z500_u250.py
    python dataprocess/process_z500_u250.py --years 2000 2001
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import warnings

# --- Configuration ---
DAILY_DIR = "/home1/11353/afahad/geos_subc/dataprocess/era5_z500_u250"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"


def process_year(year, daily_dir=DAILY_DIR, output_dir=OUTPUT_DIR):
    """
    Process ERA5 Z500 & U250 for one year:
    1. Load GEOS Zarr to get init dates
    2. Load Daily Z500/U250 files (current + previous year)
    3. Compute 4 weekly means before each init date
    4. Save as Zarr with both variables
    """
    out_path = os.path.join(output_dir, f"z500_u250_weekly_{year}.zarr")
    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping {year}.")
        return

    # 1. Load GEOS to get init dates
    geos_path = os.path.join(GEOS_DIR, f"geos_subc_{year}.zarr")
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping {year}.")
        return

    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    if 'S' not in ds_geos.dims:
        print(f"Dimension 'S' not found in {geos_path}. Skipping.")
        ds_geos.close()
        return

    init_dates = pd.to_datetime(ds_geos['S'].values)

    print(f"\n{'='*60}")
    print(f"Processing Z500 & U250 Weekly for {year}")
    print(f"  GEOS init dates: {len(init_dates)}")

    # 2. Load Daily files (current + previous year for Jan edge cases)
    daily_files = [os.path.join(daily_dir, f"era5_z500_u250_{year}.zarr")]
    prev_file = os.path.join(daily_dir, f"era5_z500_u250_{year-1}.zarr")
    if os.path.exists(prev_file):
        daily_files.insert(0, prev_file)

    valid_files = [f for f in daily_files if os.path.exists(f)]
    if not valid_files:
        print(f"  No daily files found for {year} in {daily_dir}.")
        ds_geos.close()
        return

    print(f"  Loading: {[os.path.basename(f) for f in valid_files]}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_daily = xr.open_mfdataset(valid_files, engine='zarr', combine='by_coords')
    except Exception as e:
        print(f"  Error loading daily files: {e}")
        ds_geos.close()
        return

    # Detect variable names
    z_var = next((v for v in ['z500', 'z', 'geopotential'] if v in ds_daily), None)
    u_var = next((v for v in ['u250', 'u', 'u_component_of_wind'] if v in ds_daily), None)

    if z_var is None or u_var is None:
        print(f"  Variables not found! z_var={z_var}, u_var={u_var}. Available: {list(ds_daily.data_vars)}")
        ds_daily.close()
        ds_geos.close()
        return

    print(f"  Using variables: z500='{z_var}', u250='{u_var}'")

    # 3. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means...")

    z500_samples = []
    u250_samples = []
    skipped = 0

    for init_date in tqdm(init_dates, desc=f"  Z500/U250 {year}"):
        z_weeks = []
        u_weeks = []
        valid = True

        for w in range(4):
            # L=0 (Week -4): [init-28, init-22]
            # L=3 (Week -1): [init-7, init-1]
            days_back_end = (3 - w) * 7 + 1
            w_end = init_date - pd.Timedelta(days=days_back_end)
            w_start = w_end - pd.Timedelta(days=6)

            try:
                z_chunk = ds_daily[z_var].sel(time=slice(w_start, w_end))
                u_chunk = ds_daily[u_var].sel(time=slice(w_start, w_end))

                if len(z_chunk.time) < 4:
                    valid = False
                    break

                z_weeks.append(z_chunk.mean(dim='time').compute())
                u_weeks.append(u_chunk.mean(dim='time').compute())
            except Exception:
                valid = False
                break

        if valid and len(z_weeks) == 4:
            z500_samples.append(xr.concat(z_weeks, dim='L'))
            u250_samples.append(xr.concat(u_weeks, dim='L'))
        else:
            skipped += 1
            z500_samples.append(None)
            u250_samples.append(None)

    # Handle Nones
    if len(z500_samples) <= skipped:
        print(f"  No valid samples for {year}. Skipping.")
        return

    z_template = next(s for s in z500_samples if s is not None)
    u_template = next(s for s in u250_samples if s is not None)

    final_z, final_u = [], []
    for z, u in zip(z500_samples, u250_samples):
        if z is None:
            final_z.append(xr.full_like(z_template, np.nan))
            final_u.append(xr.full_like(u_template, np.nan))
        else:
            final_z.append(z)
            final_u.append(u)

    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing data (filled NaN)")

    # 4. Finalize and Save
    z500_da = xr.concat(final_z, dim='S').assign_coords(S=init_dates, L=np.arange(4))
    u250_da = xr.concat(final_u, dim='S').assign_coords(S=init_dates, L=np.arange(4))

    z500_da.name = 'z500'
    u250_da.name = 'u250'

    ds_out = xr.merge([z500_da.to_dataset(), u250_da.to_dataset()])

    # Rename coords to match GEOS convention
    rename_dict = {}
    if 'latitude' in ds_out.coords and 'Y' not in ds_out.coords:
        rename_dict['latitude'] = 'Y'
    if 'longitude' in ds_out.coords and 'X' not in ds_out.coords:
        rename_dict['longitude'] = 'X'
    if rename_dict:
        ds_out = ds_out.rename(rename_dict)

    out_path = os.path.join(output_dir, f"z500_u250_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")

    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_zarr(out_path, mode='w')

    print(f"  ✓ Finished {year}: {out_path}")

    ds_geos.close()
    ds_daily.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Daily ERA5 Z500/U250 → Weekly Mean Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process.")
    parser.add_argument("--daily_dir", type=str, default=DAILY_DIR,
                        help="Directory containing daily ERA5 Z500/U250 Zarr files.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory.")
    args = parser.parse_args()

    years = args.years if args.years else list(range(1999, 2026))

    for year in years:
        process_year(year, daily_dir=args.daily_dir, output_dir=args.output_dir)

    print("\nAll processing complete.")
