"""
Outgoing Longwave Radiation (OLR) Processing Script
===================================================
Processes NOAA PSL Interpolated OLR daily NetCDF into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Data Source: NOAA PSL (olr.day.mean.nc, 2.5 degree)
Input:  Single daily OLR NetCDF file at dataprocess/olr/olr.day.mean.nc
Output: olr_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED OLR leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]  (oldest)
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]   (most recent)

Usage:
    python dataprocess/process_olr.py
    python dataprocess/process_olr.py --years 1999 2000
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse


# --- Configuration ---
OLR_FILE = "dataprocess/olr/olr.day.mean.nc"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"


def process_year(year, ds_olr_full, output_dir=OUTPUT_DIR):
    """
    Process OLR data for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Slice OLR daily data for needed time window (current + previous year)
    3. Regrid OLR to GEOS 1° grid
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
    print(f"Processing OLR for {year}")
    print(f"  GEOS init dates: {len(init_dates)} ({init_dates[0].date()} → {init_dates[-1].date()})")
    print(f"  Target grid: {len(target_lat)} × {len(target_lon)}")
    
    # 2. Slice OLR data for the needed time window
    # We need data from Dec of the previous year up to the end of the current year
    start_date = f"{year-1}-12-01"
    end_date = f"{year}-12-31"
    
    try:
        ds_olr_slice = ds_olr_full.sel(time=slice(start_date, end_date))
    except Exception as e:
        print(f"  Error slicing OLR data for {year}: {e}")
        return
        
    if len(ds_olr_slice.time) == 0:
        print(f"  No OLR data found for {year} in the NetCDF file. Skipping.")
        return
        
    # NOAA uses 'olr' as variable name
    da_olr = ds_olr_slice['olr']
    
    # 3. Regrid to GEOS grid
    print(f"  Regridding OLR to GEOS 1° grid (from 2.5°)...")
    
    # NOAA uses 'lat' and 'lon', rename to match GEOS target
    da_olr = da_olr.rename({'lat': target_lat.name, 'lon': target_lon.name})
    
    # Interpolate to GEOS grid
    da_olr_interp = da_olr.interp(
        {target_lat.name: target_lat, target_lon.name: target_lon},
        method='linear'
    )
    
    # 4. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means for {len(init_dates)} init dates...")
    
    processed_data = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  OLR {year}"):
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
                # Need to use inclusive slicing for xarray datetime
                chunk = da_olr_interp.sel(time=slice(w_start, w_end))
                # For daily data, a full week should have 7 days
                if len(chunk.time) < 1:
                    valid = False
                    break
                # Compute mean over the available days in the week
                w_mean = chunk.mean(dim='time').squeeze()
                
                # Use standard compute if chunk is dask array, otherwise just keep as is
                if hasattr(w_mean, 'compute'):
                    w_mean = w_mean.compute()
                    
                weeks.append(w_mean)
            except Exception:
                valid = False
                break
        
        if valid and len(weeks) == 4:
            # Stack the 4 weeks along a new 'L' dimension
            sample = xr.concat(weeks, dim='L')
            sample = sample.assign_coords(L=np.arange(4))
            processed_data.append(sample)
        else:
            skipped += 1
            # Fill with NaN if missing data
            nan_shape = (4, len(target_lat), len(target_lon))
            nan_data = xr.DataArray(
                np.full(nan_shape, np.nan, dtype=np.float32),
                dims=['L', target_lat.name, target_lon.name],
                coords={'L': np.arange(4), target_lat.name: target_lat, target_lon.name: target_lon}
            )
            processed_data.append(nan_data)
    
    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing OLR data (filled NaN)")
    
    # 5. Stack and save as Zarr
    if len(processed_data) > 0:
        ds_out = xr.concat(processed_data, dim='S')
        ds_out = ds_out.assign_coords(S=init_dates)
        
        # Format as dataset with standard attributes
        ds_final = ds_out.to_dataset(name='olr')
        ds_final['olr'].attrs = {
            'units': 'W/m2',
            'long_name': 'Outgoing Longwave Radiation',
            'description': '4-week trailing means before GEOS S2S init dates',
            'source': 'NOAA PSL Interpolated OLR (olr.day.mean.nc)'
        }
        
        out_path = os.path.join(output_dir, f"olr_weekly_{year}.zarr")
        print(f"  Saving to {out_path}...")
        
        import dask
        with dask.config.set(scheduler='synchronous'):
            # zarr_format=3 can be problematic with some xarray versions, using default
            ds_final.to_zarr(out_path, mode='w')
        
        print(f"  ✓ Finished {year}: {out_path}")
        print(f"    Shape: (S={len(init_dates)}, L=4, Y={len(target_lat)}, X={len(target_lon)})")
    
    ds_geos.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NOAA OLR → GEOS weekly Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process. Default: 1999-2016")
    args = parser.parse_args()
    
    if not os.path.exists(OLR_FILE):
        print(f"ERROR: OLR file not found at {OLR_FILE}")
        print("Please download it first: https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc")
        exit(1)
        
    print(f"Loading NOAA OLR dataset from {OLR_FILE}...")
    # Load the whole file once (lazily) to pass to year processors
    ds_olr_full = xr.open_dataset(OLR_FILE, chunks={'time': 365})
    print(f"  Time range: {ds_olr_full.time.values[0]} to {ds_olr_full.time.values[-1]}")
    
    years = args.years if args.years else list(range(1999, 2017))
    
    print(f"Processing OLR for years: {years}")
    for year in years:
        process_year(year, ds_olr_full)
    
    ds_olr_full.close()
    print(f"\nAll done! OLR weekly files saved to {OUTPUT_DIR}/")
