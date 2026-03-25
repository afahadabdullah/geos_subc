"""
Extract daily ERA5 IVT on a GEOS-like 1 degree grid.

This is the first stage of the legacy two-step IVT input pipeline:
1. ``extract_era5_ivt.py`` writes daily ``era5_ivt_{year}.zarr``
2. ``process_ivt.py`` converts those daily files into the 4 trailing observed
   weekly means before each GEOS init date.

By default this script uses the newer ARCO ERA5 analysis-ready v3 store:
``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``
"""

import xarray as xr
import os
import numpy as np
import dask
import argparse
import traceback

DEFAULT_ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

def calculate_ivt(ds, levels, gravity=9.80665):
    """
    Calculates Integrated Vapor Transport (IVT) from specific humidity and wind.
    
    Formula:
    IVT_u = (1/g) * integral(q * u) dp
    IVT_v = (1/g) * integral(q * v) dp
    IVT = sqrt(IVT_u^2 + IVT_v^2)
    
    ds: Dataset with 'q', 'u', 'v' on pressure levels
    levels: List of pressure levels in hPa (e.g., [1000, 925, ...])
    """
    
    # Ensure levels are sorted descending (surface to aloft) for proper integration
    # Pressure decreases with height, so integration from surface to top means dp is negative if we do top - bottom
    # Usually we integrate from p_sfc to p_top
    # Integral q*u dp approx sum(q*u * delta_p) / g
    
    # 1. Select variables
    q = ds['specific_humidity'] # kg/kg
    u = ds['u_component_of_wind'] # m/s
    v = ds['v_component_of_wind'] # m/s
    
    # 2. Convert pressure levels to Pascal
    # Levels in dataset coordinate 'level' are typically hPa
    # We need to ensure we are using the correct levels
    # Since we pre-selected levels, we assume ds has them.
    
    # Calculate Vapor Transport components
    qu = q * u
    qv = q * v
    
    # 3. Vertical Integration
    # using xarray integration (trapezoidal rule)
    # integrate along 'level' dimension
    # But 'level' is in hPa, need to convert to Pa for calculation
    
    # Create valid pressure coordinate in Pa
    if 'level' in ds.coords:
        p_pa = ds.coords['level'] * 100.0 # hPa to Pa
        qu = qu.assign_coords(level=p_pa)
        qv = qv.assign_coords(level=p_pa)
    
    # Integrate: int(qu) dp. 
    # Since pressure decreases as index increases often, or we just take absolute value of integration.
    # q*u * dp. 
    
    ivt_u = (1/gravity) * qu.integrate(coord='level')
    ivt_v = (1/gravity) * qv.integrate(coord='level')
    
    # The integrate function handles the sign of dp correctly (if coord is monotonic)
    # However, since pressure decreases with height, integrate might return negative. 
    # We take absolute value of the magnitude anyway, but components signs matter.
    # Wait, IVT direction matters. The integration `int_p1^p2` where p1=1000, p2=500 -> dp is negative.
    # The formula is usually (1/g) int_{p_top}^{p_sfc} ... dp  OR (1/g) int_{p_sfc}^{p_top} ... (-dp)
    # xarray integrate does trapezoidal: 0.5 * (y1+y2) * (x2-x1)
    # If x (pressure) is [1000, 900...], then dx is negative.
    # So the result will be negative of the standard definition (which assumes positive mass).
    # We should perform integration and then probably negate if levels are descending?
    # Actually, standard definition integrates from surface to top w.r.t pressure? 
    # Usually defined as magnitude of vector. 
    # Let's take absolute value of the integral result? No, direction matters for u/v transport.
    # Convention: positive u-IVT means eastward transport.
    # If we integrate [1000, ... 500] (decreasing P), dx is negative, so result is negative.
    # But mass of atmosphere is positive. 
    # So we need to multiply by -1 if integrating from user sorted high->low pressure.
    
    ivt_u = -1.0 * ivt_u
    ivt_v = -1.0 * ivt_v
    
    # 4. Magnitude
    ivt_mag = np.sqrt(ivt_u**2 + ivt_v**2)
    ivt_mag.name = 'ivt'
    
    return ivt_mag, ivt_u, ivt_v

def process_era5_ivt(
    start_year=2023,
    end_year=2025,
    output_base_dir="/home1/11353/afahad/geos_subc/dataprocess/era5_ivt",
    overwrite=False,
    zarr_path=DEFAULT_ZARR_PATH,
):
    
    print(f"Connecting to {zarr_path}...")
    
    # Follow the official ARCO access pattern for the v3 store.
    try:
        ds = xr.open_zarr(
            zarr_path,
            chunks=None,
            storage_options={"token": "anon"},
        )
        if "valid_time_start" in ds.attrs and "valid_time_stop" in ds.attrs and "time" in ds.coords:
            ds = ds.sel(time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"]))
        print("Dataset opened successfully.")
    except Exception as e:
        print(f"Error opening dataset: {type(e).__name__}: {e!r}")
        traceback.print_exc()
        return

    # Define variable names
    q_var = 'specific_humidity'
    u_var = 'u_component_of_wind'
    v_var = 'v_component_of_wind'
    level_coord = 'level'
    
    # Levels for IVT: 1000 to 500 hPa
    # We should select levels that exist in ERA5. 
    # ARCO-ERA5 typically has 37 levels.
    target_levels = [1000, 925, 850, 700, 600, 500]
    
    # Define GEOS 1-degree target grid
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_base_dir, f"era5_ivt_{year}.zarr")
        
        if os.path.exists(output_path):
            if not overwrite:
                print(f"Skipping {year}, file already exists at {output_path}")
                continue
            print(f"Overwriting existing daily file: {output_path}")
            import shutil
            shutil.rmtree(output_path)
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # 1. Select year and variables/levels
            ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            if len(ds_year.time) == 0:
                print(f"Warning: No data found for {year} in ARCO ERA5. Skipping.")
                continue
            
            # Select specific variables and levels
            # We select ALL variables on ALL target levels first to minimize IO calls if efficient,
            # or select variables then levels.
            
            # Subset by levels first? No, subset by var then level is typical. 
            # Note: Checking if levels exist is good practice but we assume standard ERA5.
            
            q = ds_year[q_var].sel({level_coord: target_levels})
            u = ds_year[u_var].sel({level_coord: target_levels})
            v = ds_year[v_var].sel({level_coord: target_levels})
            
            # Merge into dataset for calculation
            ds_subset = xr.Dataset({'specific_humidity': q, 'u_component_of_wind': u, 'v_component_of_wind': v})
            ds_subset = ds_subset.chunk({'time': 48, 'level': len(target_levels), 'latitude': 721, 'longitude': 1440})
            
            # 2. Calculate IVT (preserving 6-hourly or whatever native resolution)
            # Calculation should happen on native grid BEFORE time averaging
            print(f"Calculating IVT for {year}...")
            
            # The calculation reduces the 'level' dimension
            # Result is (time, lat, lon)
            ivt, ivt_u, ivt_v = calculate_ivt(ds_subset, target_levels)
            
            # We only want to save IVT Magnitude per user request ("save ... era5 IVT")
            # User said "save similary daily mean yearly file of era5 IVT". Implicitly magnitude?
            # Or usually better to save components? 
            # User asked "is moisture transport can be used as a proxy for atmospheric river?" -> Magnitude is the proxy.
            # I will save Magnitude.
            
            ds_ivt = ivt.to_dataset(name='ivt')
            
            # 3. Daily Mean Calculation
            print(f"Calculating daily means for {year}...")
            ds_daily = ds_ivt.resample(time='1D').mean()
            
            # 4. Interpolation to GEOS 1-degree grid
            print(f"Interpolating to GEOS 1-degree grid (181x360)...")
            ds_interp = ds_daily.interp(latitude=target_lat, longitude=target_lon, method='linear')
            
            # 5. Save to Zarr
            print(f"Saving to {output_path}...")
            # Rechunk
            ds_interp = ds_interp.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
            
            # Compute immediately or via dask?
            # to_zarr is lazy if data is dask array.
            
            with dask.config.set(scheduler='synchronous'): # Use synchronous to avoid some flaky GCS timeouts/hangs
                ds_interp.to_zarr(output_path, mode='w')
            
            print(f"Successfully saved {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ARCO-ERA5 IVT to daily 1-degree GEOS grid.")
    parser.add_argument("--start_year", type=int, default=2023)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_ivt")
    parser.add_argument("--zarr_path", type=str, default=DEFAULT_ZARR_PATH,
                        help="ARCO ERA5 Zarr source. Defaults to the public v3 full 37-variable hourly store.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing era5_ivt_<year>.zarr files.")
    args = parser.parse_args()
    
    process_era5_ivt(
        start_year=args.start_year,
        end_year=args.end_year,
        output_base_dir=args.output_dir,
        overwrite=args.overwrite,
        zarr_path=args.zarr_path,
    )
