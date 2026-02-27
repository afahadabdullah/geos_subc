from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import dask
import os

def get_base_dataset():
    """
    Connects to ArrayLake and returns the FIMR forecast dataset.
    """
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    print("Opening Zarr store via Xarray (group: esrl-fimr1p1-forecast)...")
    session = repo.writable_session(branch="main")
    ds = xr.open_zarr(session.store, zarr_format=3, group="esrl-fimr1p1-forecast")
    
    # 1. Identify pressure level dimension for ZG
    level_dim = None
    if 'zg' in ds:
        for dim in ds.zg.dims:
            if 'lev' in dim.lower() or 'pressure' in dim.lower():
                level_dim = dim
                break
    
    # 2. Select ZG at 850hPa
    if 'zg' in ds:
        if level_dim:
            print(f"Selecting 'zg' at 850hPa using level dimension: {level_dim}")
            try:
                # Try selecting by coordinate value
                zg_850 = ds.zg.sel({level_dim: 850})
            except Exception:
                # Fallback: if 850 is not an exact value, try nearest or indexing
                print("Exact level 850 not found, attempting nearest match...")
                zg_850 = ds.zg.sel({level_dim: 850}, method='nearest')
        else:
            print("Warning: No level dimension found for 'zg'. Using raw 'zg'.")
            zg_850 = ds.zg
    else:
        print("Error: 'zg' not found in dataset.")
        return None

    # 3. Assemble subset (pr, tas, zg_850)
    # We rename zg_850 to 'zg850' or keep as 'zg'? User said zg at 850.
    # I'll keep it as 'zg' to match downstream expectations if any, or 'zg' for clarity.
    subset = xr.Dataset({
        'pr': ds['pr'],
        'tas': ds['tas'],
        'zg': zg_850
    })
    
    return subset

def save_yearly_data():
    """
    Filters and saves data in yearly Zarr files from 2017 to 2025.
    """
    ds_subset = get_base_dataset()
    if ds_subset is None:
        return
    
    # Ensure output directory exists (local to where the script is run)
    # On TACC, usually dataprocess is in current dir
    os.makedirs("dataprocess", exist_ok=True)
    
    # Identify time dimension
    time_dim = None
    for dim in ['S', 'time', 'init_time']:
        if dim in ds_subset.dims:
            time_dim = dim
            break
    
    if not time_dim:
        # Fallback coordinate search
        time_coords = [c for c in ds_subset.coords if 'time' in c.lower()]
        time_dim = time_coords[0] if time_coords else None

    if not time_dim:
        print("Error: Could not identify time dimension for filtering.")
        return

    print(f"Using dimension '{time_dim}' for filtering.")

    # Loop through years 2017 to 2025
    for year in range(2017, 2026):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        output_path = f"dataprocess/fimr_forecast_{year}.zarr"
        
        print(f"\n--- Processing Year: {year} ---")
        
        if os.path.exists(output_path):
            print(f"Skipping {year}, file already exists.")
            continue

        try:
            ds_year = ds_subset.sel({time_dim: slice(start_date, end_date)})
            
            if ds_year.sizes[time_dim] == 0:
                print(f"No data found for year {year}. Skipping.")
                continue
                
            print(f"Saving {year} data to {output_path} ({ds_year.sizes[time_dim]} timesteps)...")
            
            with ProgressBar(), dask.config.set(scheduler='synchronous'):
                # We use zarr_format=3 if the source is 3, otherwise default.
                # ArrayLake usually uses V3 for newer groups.
                ds_year.to_zarr(output_path, mode='w')
            
            print(f"Successfully saved {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

if __name__ == "__main__":
    try:
        save_yearly_data()
        print("\nAll tasks completed.")
        
    except Exception as e:
        print(f"\nFatal Error: {e}")
        print("\nNote: Make sure you are logged in to ArrayLake on TACC.")
        print("Run: arraylake auth login --no-browser")
