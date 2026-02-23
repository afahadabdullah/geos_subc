"""
Sea Surface Temperature (SST) Processing Script
=================================================
Processes daily SST NetCDF files (yearly) into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Data Source: NOAA OISST (or similar) — yearly files: sst.day.mean.{year}.nc
Input:  Daily SST in yearly NetCDF files at ~0.25° resolution
Output: sst_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED SST leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]  (oldest)
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]   (most recent)

Usage:
    python dataprocess/process_sst.py
    python dataprocess/process_sst.py --years 2000 2001
    python dataprocess/process_sst.py --sst_dir /path/to/sst/files
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse


# --- Configuration ---
SST_DIR = "dataprocess/SST/noaa_sst_v2"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"


def load_sst_data(year, sst_dir=SST_DIR):
    """
    Load daily SST data for a given year and previous year (for early init dates).
    Files are yearly: sst.day.mean.{year}.nc
    """
    files = []
    
    # Previous year (needed for Jan init dates that look back 28 days)
    prev_file = os.path.join(sst_dir, f"sst.day.mean.{year - 1}.nc")
    if os.path.exists(prev_file):
        files.append(prev_file)
    
    # Current year
    curr_file = os.path.join(sst_dir, f"sst.day.mean.{year}.nc")
    if os.path.exists(curr_file):
        files.append(curr_file)
    else:
        print(f"  SST file not found: {curr_file}")
        return None
    
    ds = xr.open_mfdataset(files, combine='by_coords')
    return ds


def detect_sst_variable(ds):
    """Auto-detect the SST variable name in the dataset."""
    candidates = ['sst', 'SST', 'analysed_sst', 'sea_surface_temperature', 'tos']
    for var in candidates:
        if var in ds.data_vars:
            return var
    # Fallback: first non-coordinate data variable
    data_vars = [v for v in ds.data_vars if v not in ds.coords]
    if data_vars:
        print(f"  Auto-detected SST variable: '{data_vars[0]}'")
        return data_vars[0]
    raise ValueError(f"Cannot find SST variable. Available: {list(ds.data_vars)}")


def detect_coords(ds):
    """Auto-detect lat/lon coordinate names."""
    lat_candidates = ['latitude', 'lat', 'Y', 'y']
    lon_candidates = ['longitude', 'lon', 'X', 'x']
    
    lat_name = None
    lon_name = None
    for c in lat_candidates:
        if c in ds.coords or c in ds.dims:
            lat_name = c
            break
    for c in lon_candidates:
        if c in ds.coords or c in ds.dims:
            lon_name = c
            break
    
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot detect lat/lon. Coords: {list(ds.coords)}")
    return lat_name, lon_name


def process_year(year, sst_dir=SST_DIR, output_dir=OUTPUT_DIR):
    """
    Process SST data for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Load SST yearly files (current + previous year)
    3. Regrid SST to GEOS 1° grid
    4. Compute 4 weekly means before each init date
    5. Save as Zarr
    """
    # 1. Load GEOS to get init dates and grid
    geos_path = os.path.join(GEOS_DIR, f"geos_subc_{year}.zarr")
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping {year}.")
        return
    
    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    
    if 'S' not in ds_geos.dims:
        print(f"Dimension 'S' not found in {geos_path}. Skipping.")
        return
    
    init_dates = pd.to_datetime(ds_geos['S'].values)
    target_lat = ds_geos.coords['Y'] if 'Y' in ds_geos.coords else ds_geos.coords['lat']
    target_lon = ds_geos.coords['X'] if 'X' in ds_geos.coords else ds_geos.coords['lon']
    
    print(f"\n{'='*60}")
    print(f"Processing SST for {year}")
    print(f"  GEOS init dates: {len(init_dates)} ({init_dates[0].date()} → {init_dates[-1].date()})")
    print(f"  Target grid: {len(target_lat)} × {len(target_lon)}")
    
    # 2. Load SST data (current + previous year)
    ds_sst = load_sst_data(year, sst_dir)
    if ds_sst is None:
        return
    
    # Detect variable and coordinate names
    sst_var = detect_sst_variable(ds_sst)
    sst_lat, sst_lon = detect_coords(ds_sst)
    print(f"  SST variable: '{sst_var}', coords: lat='{sst_lat}', lon='{sst_lon}'")
    
    da_sst = ds_sst[sst_var]
    
    # 3. Regrid to GEOS grid
    print(f"  Regridding SST to GEOS 1° grid...")
    
    # Rename coords to match GEOS target for interpolation
    rename_dict = {}
    if sst_lat != target_lat.name:
        rename_dict[sst_lat] = target_lat.name
    if sst_lon != target_lon.name:
        rename_dict[sst_lon] = target_lon.name
    
    if rename_dict:
        da_sst = da_sst.rename(rename_dict)
    
    # Robust coordinate alignment: convert -180/180 to 0/360 range
    if target_lon.name in da_sst.coords:
        print(f"  Aligning longitude convention for {target_lon.name}...")
        da_sst = da_sst.assign_coords({target_lon.name: (da_sst[target_lon.name] % 360)})
        da_sst = da_sst.sortby(target_lon.name)
    
    # Interpolate to GEOS grid
    da_sst_interp = da_sst.interp(
        {target_lat.name: target_lat, target_lon.name: target_lon},
        method='linear'
    )
    
    # 4. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means for {len(init_dates)} init dates...")
    
    processed_data = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  SST {year}"):
        # 4 weeks BEFORE init date:
        #   L=0 → [S-28, S-22]  oldest observed week
        #   L=1 → [S-21, S-15]
        #   L=2 → [S-14, S-8]
        #   L=3 → [S-7,  S-1]   most recent observed week
        weeks = []
        valid = True
        
        for w in range(4):
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)
            w_start = w_end - pd.Timedelta(days=6)
            
            try:
                chunk = da_sst_interp.sel(time=slice(w_start, w_end))
                if len(chunk.time) == 0:
                    valid = False
                    break
                w_mean = chunk.mean(dim='time').squeeze().compute()
                weeks.append(w_mean)
            except Exception:
                valid = False
                break
        
        if valid and len(weeks) == 4:
            sample = xr.concat(weeks, dim='L')
            sample = sample.assign_coords(L=np.arange(4))
            processed_data.append(sample)
        else:
            skipped += 1
            nan_shape = (4, len(target_lat), len(target_lon))
            nan_data = xr.DataArray(
                np.full(nan_shape, np.nan, dtype=np.float32),
                dims=['L', target_lat.name, target_lon.name],
                coords={'L': np.arange(4), target_lat.name: target_lat, target_lon.name: target_lon}
            )
            processed_data.append(nan_data)
    
    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing SST data (filled NaN)")
    
    # 5. Stack and save as Zarr
    ds_out = xr.concat(processed_data, dim='S')
    ds_out = ds_out.assign_coords(S=init_dates)
    
    out_path = os.path.join(output_dir, f"sst_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")
    
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_dataset(name='sst').to_zarr(out_path, mode='w', zarr_format=3)
    
    print(f"  ✓ Finished {year}: {out_path}")
    print(f"    Shape: (S={len(init_dates)}, L=4, Y={len(target_lat)}, X={len(target_lon)})")
    
    ds_sst.close()
    ds_geos.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily SST → weekly Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process. Default: 1999-2016")
    parser.add_argument("--sst_dir", type=str, default=SST_DIR,
                        help=f"Directory containing sst.day.mean.{{year}}.nc files (default: {SST_DIR})")
    args = parser.parse_args()
    
    years = args.years if args.years else list(range(1999, 2017))
    
    print(f"Processing SST for years: {years}")
    print(f"SST source dir: {args.sst_dir}")
    for year in years:
        process_year(year, sst_dir=args.sst_dir)
    
    print(f"\nAll done! SST weekly files saved to {OUTPUT_DIR}/")
