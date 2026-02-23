"""
Soil Moisture (SoilW) Processing Script
=======================================
Processes daily C3S Soil Moisture NetCDF files into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Data Source: C3S Soil Moisture (Copernicus)
Input:  Daily NetCDF files (C3S-SOILMOISTURE-L3S-SSMV-COMBINED-DAILY-...)
Output: soilw_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED SoilW leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]  (oldest)
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]   (most recent)

Usage:
    python dataprocess/process_soilw.py
    python dataprocess/process_soilw.py --years 2000 2001
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import argparse


# --- Configuration ---
SOIL_BASE_DIR = "dataprocess/soil"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"


def find_soil_files(year, prev_year=None):
    """
    Find all SoilW daily NetCDF files for a given year (and optionally previous year).
    Matches pattern: *DAILY-{YYYY}*.nc
    """
    # Pattern for C3S Soil Moisture: ...DAILY-YYYYMMDD...
    # We search recursively just in case, but user showed flat structure
    
    # Current Year
    pattern = os.path.join(SOIL_BASE_DIR, f"**/*DAILY-{year}*.nc")
    files = sorted(glob.glob(pattern, recursive=True))
    
    # Previous Year (for early Jan inits)
    if prev_year is not None:
        prev_pattern = os.path.join(SOIL_BASE_DIR, f"**/*DAILY-{prev_year}*.nc")
        prev_files = sorted(glob.glob(prev_pattern, recursive=True))
        files = prev_files + files
    
    return files


def detect_soil_variable(ds):
    """Auto-detect the Soil Moisture variable name in the dataset."""
    candidates = ['sm', 'ssmv', 'soil_moisture', 'volumetric_soil_water', 'SOIL_MOISTURE', 'SM']
    
    for var in candidates:
        if var in ds.data_vars:
            return var
            
    # Fallback: return first non-coordinate data variable
    data_vars = [v for v in ds.data_vars if v not in ds.coords]
    if data_vars:
        print(f"  Auto-detected SoilW variable: '{data_vars[0]}'")
        return data_vars[0]
    raise ValueError(f"Cannot find SoilW variable. Available: {list(ds.data_vars)}")


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


def process_year(year, output_dir=OUTPUT_DIR):
    """
    Process SoilW data for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Load SoilW daily files (current + previous year)
    3. Regrid SoilW to GEOS 1° grid
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
    print(f"Processing Soil Moisture for {year}")
    print(f"  GEOS init dates: {len(init_dates)} ({init_dates[0].date()} → {init_dates[-1].date()})")
    print(f"  Target grid: {len(target_lat)} × {len(target_lon)}")
    
    # 2. Load SoilW daily files
    soil_files = find_soil_files(year, prev_year=year-1)
    
    if not soil_files:
        print(f"  No SoilW files found for {year}. Skipping.")
        return
    print(f"  Found {len(soil_files)} SoilW daily files (including prev year)")
    
    # Load all into single dataset
    print(f"  Loading SoilW data...")
    try:
        # Check time encoding issues, sometimes need decode_times=False manually if units weird
        # But usually standard
        # Disable parallel loading to avoid "can't start new thread" error
        ds_soil = xr.open_mfdataset(soil_files, combine='by_coords', chunks={'time': 30}, parallel=False)
    except Exception as e:
        print(f"  Error loading SoilW files: {e}")
        return
    
    # Detect variable and coordinate names
    soil_var = detect_soil_variable(ds_soil)
    s_lat, s_lon = detect_coords(ds_soil)
    print(f"  SoilW variable: '{soil_var}', coords: lat='{s_lat}', lon='{s_lon}'")
    
    da_soil = ds_soil[soil_var]
    
    if rename_dict:
        da_soil = da_soil.rename(rename_dict)
    
    # 3. Regrid to GEOS grid
    print(f"  Regridding SoilW to GEOS 1° grid...")
    
    # Robust coordinate alignment: convert -180/180 to 0/360 range
    # This prevents the "half-world" NaN issue during interpolation.
    if target_lon.name in da_soil.coords:
        print(f"  Aligning longitude convention for {target_lon.name}...")
        da_soil = da_soil.assign_coords({target_lon.name: (da_soil[target_lon.name] % 360)})
        da_soil = da_soil.sortby(target_lon.name)
    
    # Interpolate to GEOS grid
    da_soil_interp = da_soil.interp(
        {target_lat.name: target_lat, target_lon.name: target_lon},
        method='linear'
    )
    
    # 4. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means for {len(init_dates)} init dates...")
    
    processed_data = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  SoilW {year}"):
        weeks = []
        valid = True
        
        for w in range(4):
            # Week offset from init date (going backwards)
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)  # End of this week
            w_start = w_end - pd.Timedelta(days=6)                    # Start of this week
            
            try:
                chunk = da_soil_interp.sel(time=slice(w_start, w_end))
                if len(chunk.time) == 0:
                    valid = False
                    break
                    
                # Soil Moisture often has NaNs (oceans/mask). 
                # Use mean(skipna=True), which is default for xarray
                w_mean = chunk.mean(dim='time', skipna=True).squeeze().compute()
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
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing SoilW data (filled NaN)")
    
    # 5. Stack and save as Zarr
    ds_out = xr.concat(processed_data, dim='S')
    ds_out = ds_out.assign_coords(S=init_dates)
    
    out_path = os.path.join(output_dir, f"soilw_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")
    
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_dataset(name='soilw').to_zarr(out_path, mode='w', zarr_format=3)
    
    print(f"  ✓ Finished {year}: {out_path}")
    print(f"    Shape: (S={len(init_dates)}, L=4, Y={len(target_lat)}, X={len(target_lon)})")
    
    ds_soil.close()
    ds_geos.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process C3S SoilW → weekly Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process. Default: 1999-2016")
    args = parser.parse_args()
    
    years = args.years if args.years else list(range(1999, 2017))
    
    print(f"Processing Soil Moisture for years: {years}")
    for year in years:
        process_year(year)
    
    print(f"\nAll done! SoilW weekly files saved to {OUTPUT_DIR}/")
