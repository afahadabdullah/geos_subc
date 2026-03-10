"""
ERA5 SLP (Sea Level Pressure) Weekly Processing Script
======================================================
Processes daily ERA5 SLP files into weekly-mean Zarr files aligned with GEOS sequences.

Input:  era5_slp_{year}.zarr
Output: slp_weekly_{year}.zarr
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

DAILY_SLP_DIR = "/home1/11353/afahad/geos_subc/dataprocess/era5_slp"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"

def process_year(year, daily_dir=DAILY_SLP_DIR, output_dir=OUTPUT_DIR):
    geos_path = os.path.join(GEOS_DIR, f"geos_subc_{year}.zarr")
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping.")
        return
    
    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    if 'S' not in ds_geos.dims:
        return
    
    init_dates = pd.to_datetime(ds_geos['S'].values)
    target_lat = ds_geos.coords['Y'] if 'Y' in ds_geos.coords else ds_geos.coords['latitude']
    target_lon = ds_geos.coords['X'] if 'X' in ds_geos.coords else ds_geos.coords['longitude']
    
    print(f"\nProcessing SLP Weekly for {year}")
    
    daily_files = [os.path.join(daily_dir, f"era5_slp_{year}.zarr")]
    prev_year_file = os.path.join(daily_dir, f"era5_slp_{year-1}.zarr")
    if os.path.exists(prev_year_file): daily_files.insert(0, prev_year_file)
        
    try:
        ds_daily = xr.open_mfdataset(daily_files, engine='zarr', combine='by_coords')
    except Exception as e:
        print(f"Error loading daily files: {e}")
        return

    processed_samples = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  SLP {year}"):
        weeks = []
        valid = True
        for w in range(4):
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)
            w_start = w_end - pd.Timedelta(days=6)
            try:
                chunk = ds_daily.sel(time=slice(w_start, w_end))
                if len(chunk.time) < 7:
                    valid = False; break
                w_mean = chunk.mean(dim='time').compute()
                weeks.append(w_mean)
            except:
                valid = False; break
        
        if valid and len(weeks) == 4:
            processed_samples.append(xr.concat(weeks, dim='L'))
        else:
            skipped += 1
            nan_ds = ds_daily.isel(time=0, drop=True).expand_dims(L=4).copy(deep=True)
            for var in nan_ds.data_vars: nan_ds[var].values[:] = np.nan
            processed_samples.append(nan_ds)
            
    if skipped > 0: print(f"  Warning: {skipped} missing dates")

    if len(processed_samples) > 0:
        ds_yearly = xr.concat(processed_samples, dim='S')
        ds_yearly = ds_yearly.assign_coords(S=init_dates, L=np.arange(4))
        
        rename_dict = {}
        if 'latitude' in ds_yearly.coords and 'Y' not in ds_yearly.coords: rename_dict['latitude'] = 'Y'
        if 'longitude' in ds_yearly.coords and 'X' not in ds_yearly.coords: rename_dict['longitude'] = 'X'
        if rename_dict: ds_yearly = ds_yearly.rename(rename_dict)

        out_path = os.path.join(output_dir, f"slp_weekly_{year}.zarr")
        print(f"  Saving to {out_path}...")
        import dask
        with dask.config.set(scheduler='synchronous'):
            ds_yearly.to_zarr(out_path, mode='w', zarr_format=3)
            
    ds_geos.close(); ds_daily.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--daily_dir", type=str, default=DAILY_SLP_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    years = args.years if args.years else list(range(1999, 2022))
    for year in years: process_year(year, args.daily_dir, args.output_dir)
