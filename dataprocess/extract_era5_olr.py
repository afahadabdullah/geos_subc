
import xarray as xr
import os
import numpy as np
import argparse

def process_noaa_olr(start_year=1999, end_year=2022, output_base_dir="/home1/11353/afahad/geos_subc/dataprocess/era5_olr"):
    """
    Download and process NOAA PSL Interpolated OLR (Outgoing Longwave Radiation).
    
    Source: NOAA Physical Sciences Laboratory
    URL: https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc
    
    This is the standard daily OLR product (2.5° grid, 1974-2022) widely used
    in MJO research. We interpolate to GEOS 1° grid and save yearly Zarr files.
    
    Pipeline:
       1. Download/open the single NetCDF file from NOAA PSL
       2. Select year range
       3. Interpolate to GEOS 1-degree grid (181x360)
       4. Save as yearly Zarr files
    """
    
    # NOAA PSL Interpolated OLR (single file, ~800 MB)
    olr_url = "https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc"
    local_cache = os.path.join(output_base_dir, "olr.day.mean.nc")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Download if not cached
    if not os.path.exists(local_cache):
        print(f"Downloading NOAA OLR from {olr_url}...")
        print("(~800 MB, this may take a few minutes)")
        import urllib.request
        urllib.request.urlretrieve(olr_url, local_cache)
        print(f"✅ Downloaded to {local_cache}")
    else:
        print(f"✅ Using cached OLR file: {local_cache}")
    
    # Open the dataset
    print("Opening OLR dataset...")
    ds = xr.open_dataset(local_cache)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Grid: {ds.dims}")
    
    # The variable is typically named 'olr'
    olr_var = 'olr'
    if olr_var not in ds:
        # Try alternative names
        for candidate in ['OLR', 'ulwrf', 'olr']:
            if candidate in ds:
                olr_var = candidate
                break
        else:
            print(f"ERROR: OLR variable not found. Available: {list(ds.data_vars)}")
            return
    
    print(f"  Using variable: '{olr_var}' (units: {ds[olr_var].attrs.get('units', 'unknown')})")
    
    # Define GEOS 1-degree target grid
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    # Detect coordinate names (NOAA uses 'lat'/'lon', not 'latitude'/'longitude')
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_base_dir, f"era5_olr_{year}.zarr")
        
        if os.path.exists(output_path):
            print(f"Skipping {year}, file already exists at {output_path}")
            continue
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # 1. Select year
            ds_year = ds[olr_var].sel(time=str(year))
            print(f"  {len(ds_year.time)} days found for {year}")
            
            # 2. Interpolate to GEOS 1-degree grid
            print(f"  Interpolating from 2.5° to 1° GEOS grid (181x360)...")
            ds_interp = ds_year.interp({lat_name: target_lat, lon_name: target_lon}, method='linear')
            
            # Rename coordinates to standard names
            rename_map = {}
            if lat_name != 'latitude':
                rename_map[lat_name] = 'latitude'
            if lon_name != 'longitude':
                rename_map[lon_name] = 'longitude'
            if rename_map:
                ds_interp = ds_interp.rename(rename_map)
            
            # Convert to dataset
            ds_out = ds_interp.to_dataset(name='olr')
            ds_out['olr'].attrs = {
                'units': 'W/m2',
                'long_name': 'Outgoing Longwave Radiation',
                'source': 'NOAA PSL Interpolated OLR (olr.day.mean.nc)',
                'original_resolution': '2.5 degree',
                'interpolated_to': '1.0 degree GEOS grid'
            }
            
            # 3. Rechunk and save to Zarr
            print(f"  Saving to {output_path}...")
            ds_out = ds_out.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
            ds_out.to_zarr(output_path, mode='w')
            
            print(f"  ✅ Successfully saved OLR for {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Done! All years saved to {output_base_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NOAA PSL OLR and interpolate to GEOS 1-degree grid.")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2022)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_olr")
    args = parser.parse_args()
    
    process_noaa_olr(start_year=args.start_year, end_year=args.end_year, output_base_dir=args.output_dir)
