"""
Soil Moisture (SoilW) Processing Script - Circular Padding Version
===================================================================
Processes daily C3S Soil Moisture NetCDF files into weekly-mean Zarr files.
Uses circular padding to ensure full 0-360 longitudinal coverage.
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import dask
import warnings

# --- Configuration ---
SOIL_BASE_DIR = "dataprocess/soil"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"

def find_soil_files(year, prev_year=None):
    """Find all SoilW daily NetCDF files for current and previous year."""
    pattern = os.path.join(SOIL_BASE_DIR, f"**/*DAILY-{year}*.nc")
    files = sorted(glob.glob(pattern, recursive=True))
    if prev_year is not None:
        prev_pattern = os.path.join(SOIL_BASE_DIR, f"**/*DAILY-{prev_year}*.nc")
        prev_files = sorted(glob.glob(prev_pattern, recursive=True))
        files = prev_files + files
    return files

def detect_soil_variable(ds):
    """Auto-detect the Soil Moisture variable name."""
    candidates = ['sm', 'ssmv', 'soil_moisture', 'volumetric_soil_water']
    for var in candidates:
        if var in ds.data_vars: return var
    data_vars = [v for v in ds.data_vars if v not in ds.coords]
    if data_vars: return data_vars[0]
    raise ValueError(f"Cannot find SoilW variable. Available: {list(ds.data_vars)}")

def detect_coords(ds):
    """Auto-detect lat/lon coordinate names."""
    lat_candidates = ['latitude', 'lat', 'Y', 'y']
    lon_candidates = ['longitude', 'lon', 'X', 'x']
    lat_name = next((c for c in lat_candidates if c in ds.coords or c in ds.dims), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords or c in ds.dims), None)
    if not lat_name or not lon_name:
        raise ValueError(f"Cannot detect lat/lon. Coords: {list(ds.coords)}")
    return lat_name, lon_name

def plot_verification(sample, lon_name, lat_name, out_name="sample_soilw_check.png"):
    """Plot the lead 3 (Week -1) of a sample with detailed stats."""
    try:
        data = sample.isel(L=3).values
        v_min, v_max = np.nanmin(data), np.nanmax(data)
        nan_count = np.isnan(data).sum()
        total_pixels = data.size
        nan_perc = (nan_count / total_pixels) * 100
        
        print(f"  📊 Verification Stats: Min={v_min:.4f}, Max={v_max:.4f}, NaNs={nan_count} ({nan_perc:.1f}%)")
        
        plt.figure(figsize=(12, 6))
        plt.imshow(data, origin='lower', extent=[0, 360, -90, 90], aspect='auto', cmap='terrain')
        plt.colorbar(label='Soil Moisture')
        plt.title(f"Diagnostic Plot: Week -1 Coverage\n(Min={v_min:.3f}, Max={v_max:.3f}, NaNs={nan_perc:.1f}%)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved verification plot to {out_name}")
    except Exception as e:
        print(f"  Warning: Could not save verification plot: {e}")

def check_outputs(years, output_dir=OUTPUT_DIR):
    """Read processed Zarr files and plot for verification."""
    print(f"\n🔬 Checking existing SoilW Zarr files in {output_dir}...")
    for year in years:
        out_path = os.path.join(output_dir, f"soilw_weekly_{year}.zarr")
        if not os.path.exists(out_path): continue
        try:
            ds = xr.open_zarr(out_path, consolidated=False)
            print(f"  Year {year}: {out_path} (Shape: {ds.soilw.shape})")
            sample = ds.soilw.isel(S=0)
            plot_verification(sample, 'X', 'Y', out_name=f"check_soilw_{year}.png")
            ds.close()
            # break # Keep going for all years if needed, but usually first is enough
        except Exception as e:
            print(f"  ❌ Error checking {year}: {e}")

def process_year(year, output_dir=OUTPUT_DIR):
    """Process one year with circular padding to bridge the 0/360 seam."""
    geos_path = os.path.join(GEOS_DIR, f"geos_subc_{year}.zarr")
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping.")
        return
    
    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    init_dates = pd.to_datetime(ds_geos['S'].values)
    target_lat = ds_geos.coords['Y'] if 'Y' in ds_geos.coords else ds_geos.coords['lat']
    target_lon = ds_geos.coords['X'] if 'X' in ds_geos.coords else ds_geos.coords['lon']
    
    print(f"\n{'='*60}")
    print(f"Processing Soil Moisture for {year}")
    print(f"  Target: {len(target_lat)}x{len(target_lon)} grid, {len(init_dates)} dates")
    
    soil_files = find_soil_files(year, prev_year=year-1)
    if not soil_files: return
    
    print(f"  Loading {len(soil_files)} SoilW daily files...")
    ds_soil = xr.open_mfdataset(soil_files, combine='by_coords', chunks={'time': 30}, parallel=False)
    
    soil_var = detect_soil_variable(ds_soil)
    s_lat, s_lon = detect_coords(ds_soil)
    
    da_soil = ds_soil[soil_var]
    
    # 1. Rename coords to match GEOS target
    rename_dict = {}
    if s_lat != target_lat.name: rename_dict[s_lat] = target_lat.name
    if s_lon != target_lon.name: rename_dict[s_lon] = target_lon.name
    if rename_dict:
        da_soil = da_soil.rename(rename_dict)
    
    # 2. Robust Longitude Alignment (0-360)
    print(f"  [1] Aligning longitude convention to 0-360 range...")
    # This maps e.g. [-1, 359] to [359, 359] and [1, 1]
    da_soil = da_soil.assign_coords({target_lon.name: (da_soil[target_lon.name] % 360)})
    da_soil = da_soil.sortby(target_lon.name)
    
    # 3. CIRCULAR PADDING
    # To bridge the 0/360 seam, we pad the data at both ends
    print(f"  [2] Applying circular padding to bridge 0/360 seam...")
    lon_dim = target_lon.name
    # Pad by 2 degrees at each end
    left_pad = da_soil.isel({lon_dim: slice(-5, None)}).assign_coords({lon_dim: da_soil[lon_dim].isel({lon_dim: slice(-5, None)}) - 360})
    right_pad = da_soil.isel({lon_dim: slice(0, 5)}).assign_coords({lon_dim: da_soil[lon_dim].isel({lon_dim: slice(0, 5)}) + 360})
    
    da_soil_padded = xr.concat([left_pad, da_soil, right_pad], dim=lon_dim).sortby(lon_dim)
    
    # Diagnostic: Count valid data points in hemispheres
    # Americas/Western Hemisphere is roughly 180-360
    with dask.config.set(scheduler='synchronous'):
        v_east = (da_soil_padded.sel({lon_dim: slice(0, 180)}) > 0).sum().compute().item()
        v_west = (da_soil_padded.sel({lon_dim: slice(180, 360)}) > 0).sum().compute().item()
    print(f"      Source Check: Valid points in East (0-180): {v_east}, West (180-360): {v_west}")

    # 4. Interpolation to GEOS Grid
    print(f"  [3] Interpolating to GEOS Grid...")
    da_soil_interp = da_soil_padded.interp(
        {target_lat.name: target_lat, target_lon.name: target_lon},
        method='linear'
    )
    
    # Quick check after interpolation
    with dask.config.set(scheduler='synchronous'):
        interp_check = da_soil_interp.isel(time=0).compute().values
    print(f"      Post-Interp Check (Day 0): Min={np.nanmin(interp_check):.4f}, Max={np.nanmax(interp_check):.4f}, NaNs={np.isnan(interp_check).sum()}")
    
    if np.isnan(interp_check).all():
        print("      ⚠️ ERROR: Interpolation resulted in 100% NaNs! Verification required.")

    # 5. Weekly Aggregation
    print(f"  [4] Computing Weekly Means...")
    processed_data = []
    skipped = 0
    plot_done = False
    
    for init_date in tqdm(init_dates, desc=f"  SoilW {year}"):
        weeks = []
        valid = True
        for w in range(4):
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)
            w_start = w_end - pd.Timedelta(days=6)
            try:
                # Use a small buffer to avoid selection floating point issues
                chunk = da_soil_interp.sel(time=slice(w_start, w_end))
                if len(chunk.time) == 0:
                    valid = False; break
                with dask.config.set(scheduler='synchronous'):
                    w_mean = chunk.mean(dim='time', skipna=True).squeeze().compute()
                weeks.append(w_mean)
            except Exception:
                valid = False; break
        
        if valid and len(weeks) == 4:
            sample = xr.concat(weeks, dim='L').assign_coords(L=np.arange(4))
            processed_data.append(sample)
            if not plot_done:
                plot_verification(sample, target_lon.name, target_lat.name)
                plot_done = True
        else:
            skipped += 1
            nan_arr = xr.DataArray(np.full((4, len(target_lat), len(target_lon)), np.nan, dtype=np.float32),
                                    dims=['L', target_lat.name, target_lon.name],
                                    coords={'L': np.arange(4), target_lat.name: target_lat, target_lon.name: target_lon})
            processed_data.append(nan_arr)

    # 6. Save
    ds_out = xr.concat(processed_data, dim='S').assign_coords(S=init_dates)
    out_path = os.path.join(output_dir, f"soilw_weekly_{year}.zarr")
    print(f"  [5] Saving to {out_path}...")
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_dataset(name='soilw').to_zarr(out_path, mode='w', zarr_format=3)
    
    print(f"  ✓ Finished {year}. Skipped {skipped}/{len(init_dates)} dates.")
    ds_soil.close(); ds_geos.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    years = args.years if args.years else list(range(1999, 2017))
    if args.check: check_outputs(years)
    else:
        for year in years: process_year(year)
