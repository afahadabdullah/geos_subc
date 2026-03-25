"""
Sea Surface Salinity (SSS) Processing Script
=============================================
Processes daily Copernicus SSS NetCDF files into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Data Source: MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013 (Copernicus)
Input:  Daily SSS NetCDF files at ~0.25° resolution
Output: sss_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED SSS leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]  (oldest)
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]   (most recent)

Usage:
    python dataprocess/process_sss.py
    python dataprocess/process_sss.py --years 2000 2001
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import argparse


# --- Configuration ---
SSS_BASE_DIR = "dataprocess/SSS/copernicus_sss_data"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"
DEFAULT_START_YEAR = 2024
DEFAULT_END_YEAR = 2025


def find_sss_files(year, prev_year=None):
    """
    Find all SSS daily NetCDF files for a given year (and optionally previous year).
    Searches recursively through the nested Copernicus directory structure.
    """
    pattern = os.path.join(SSS_BASE_DIR, str(year), "**", "*.nc")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if prev_year is not None:
        prev_pattern = os.path.join(SSS_BASE_DIR, str(prev_year), "**", "*.nc")
        prev_files = sorted(glob.glob(prev_pattern, recursive=True))
        files = prev_files + files
    
    return files


def detect_sss_variable(ds):
    """Auto-detect the SSS variable name in the dataset."""
    candidates = ['sss', 'sos', 'SSS', 'SOS', 'sea_surface_salinity',
                   'ssd', 'SSD', 's_surface']
    for var in candidates:
        if var in ds.data_vars:
            return var
    # Fallback: return first non-coordinate data variable
    data_vars = [v for v in ds.data_vars if v not in ds.coords]
    if data_vars:
        print(f"  Auto-detected SSS variable: '{data_vars[0]}'")
        return data_vars[0]
    raise ValueError(f"Cannot find SSS variable. Available: {list(ds.data_vars)}")


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


def strip_nondim_coords(da, keep_coord_names):
    """
    Drop scalar/non-dimension coordinates that can otherwise make xr.concat fail
    when some samples carry metadata like ``depth`` and NaN fallback samples do not.
    """
    drop_names = [
        name for name in da.coords
        if name not in keep_coord_names and name not in da.dims
    ]
    if drop_names:
        da = da.reset_coords(drop_names, drop=True)
    return da


def process_year(year, output_dir=OUTPUT_DIR):
    """
    Process SSS data for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Load SSS daily files (current + previous year)
    3. Regrid SSS to GEOS 1° grid
    4. Compute 4 weekly means before each init date
    5. Save as Zarr
    """
    out_path = os.path.join(output_dir, f"sss_weekly_{year}.zarr")
    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping {year}.")
        return

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
    print(f"Processing SSS for {year}")
    print(f"  GEOS init dates: {len(init_dates)} ({init_dates[0].date()} → {init_dates[-1].date()})")
    print(f"  Target grid: {len(target_lat)} × {len(target_lon)}")
    
    # 2. Load SSS daily files
    # Need previous year for early init dates (e.g., Jan 7 needs Dec data)
    sss_files = find_sss_files(year, prev_year=year-1)
    
    if not sss_files:
        print(f"  No SSS files found for {year}. Skipping.")
        return
    print(f"  Found {len(sss_files)} SSS daily files (including prev year)")
    
    # Load all into single dataset
    print(f"  Loading SSS data...")
    try:
        ds_sss = xr.open_mfdataset(sss_files, combine='by_coords', chunks={'time': 30})
    except Exception as e:
        print(f"  Error loading SSS files: {e}")
        return
    
    # Detect variable and coordinate names
    sss_var = detect_sss_variable(ds_sss)
    sss_lat, sss_lon = detect_coords(ds_sss)
    print(f"  SSS variable: '{sss_var}', coords: lat='{sss_lat}', lon='{sss_lon}'")
    
    da_sss = ds_sss[sss_var]
    da_sss = da_sss.squeeze(drop=True)
    
    # 3. Regrid to GEOS grid
    print(f"  Regridding SSS to GEOS 1° grid...")
    
    # Rename coords to match GEOS target for interpolation
    rename_dict = {}
    if sss_lat != target_lat.name:
        rename_dict[sss_lat] = target_lat.name
    if sss_lon != target_lon.name:
        rename_dict[sss_lon] = target_lon.name
    
    if rename_dict:
        da_sss = da_sss.rename(rename_dict)

    da_sss = strip_nondim_coords(da_sss, keep_coord_names={target_lat.name, target_lon.name, 'time'})
    
    # Interpolate to GEOS grid
    da_sss_interp = da_sss.interp(
        {target_lat.name: target_lat, target_lon.name: target_lon},
        method='linear'
    )
    
    # 4. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means for {len(init_dates)} init dates...")
    
    processed_data = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  SSS {year}"):
        # 4 weeks BEFORE init date:
        #   L=0 → [S-28, S-22]  oldest observed week
        #   L=1 → [S-21, S-15]
        #   L=2 → [S-14, S-8]
        #   L=3 → [S-7,  S-1]   most recent observed week
        weeks = []
        valid = True
        
        for w in range(4):
            # Week offset from init date (going backwards)
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)  # End of this week
            w_start = w_end - pd.Timedelta(days=6)                    # Start of this week
            
            try:
                chunk = da_sss_interp.sel(time=slice(w_start, w_end))
                if len(chunk.time) == 0:
                    valid = False
                    break
                w_mean = chunk.mean(dim='time').squeeze().compute()
                w_mean = strip_nondim_coords(
                    w_mean,
                    keep_coord_names={target_lat.name, target_lon.name}
                )
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
            # Fill with NaN
            nan_shape = (4, len(target_lat), len(target_lon))
            nan_data = xr.DataArray(
                np.full(nan_shape, np.nan, dtype=np.float32),
                dims=['L', target_lat.name, target_lon.name],
                coords={'L': np.arange(4), target_lat.name: target_lat, target_lon.name: target_lon}
            )
            processed_data.append(nan_data)
    
    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing SSS data (filled NaN)")
    
    # 5. Stack and save as Zarr
    ds_out = xr.concat(processed_data, dim='S')
    ds_out = ds_out.assign_coords(S=init_dates)
    
    out_path = os.path.join(output_dir, f"sss_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")
    
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_dataset(name='sss').to_zarr(out_path, mode='w', zarr_format=3)
    
    print(f"  ✓ Finished {year}: {out_path}")
    print(f"    Shape: (S={len(init_dates)}, L=4, Y={len(target_lat)}, X={len(target_lon)})")
    
    ds_sss.close()
    ds_geos.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Copernicus SSS → weekly Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process. Overrides start/end year.")
    parser.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR,
                        help=f"First year to process when --years is not given. Default: {DEFAULT_START_YEAR}")
    parser.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR,
                        help=f"Last year to process when --years is not given. Default: {DEFAULT_END_YEAR}")
    args = parser.parse_args()

    years = args.years if args.years else list(range(args.start_year, args.end_year + 1))
    
    print(f"Processing SSS for years: {years}")
    for year in years:
        process_year(year)
    
    print(f"\nAll done! SSS weekly files saved to {OUTPUT_DIR}/")
