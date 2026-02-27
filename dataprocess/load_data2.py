from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import dask
import os

def get_base_dataset():
    """
    Connects to ArrayLake and returns the FIMR forecast dataset.
    Uses low-level zarr to find keys and drop misaligned variables that break Xarray.
    """
    import zarr
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    print("Creating session for branch 'main'...")
    session = repo.writable_session(branch="main")
    
    print("Inspecting group keys for 'esrl-fimr1p1-forecast'...")
    try:
        # Open group with zarr directly to see what's inside
        z = zarr.open(session.store, mode='r', zarr_format=3)
        group_path = "esrl-fimr1p1-forecast"
        
        # Get all sub-keys (arrays and groups)
        all_keys = list(z[group_path].keys())
        needed_vars = ['pr', 'tas', 'zg']
        to_drop = [k for k in all_keys if k not in needed_vars]
        
        print(f"Found {len(all_keys)} keys. Dropping {len(to_drop)} variables to avoid alignment conflicts.")
        
        # Open with Xarray while dropping everything else
        ds = xr.open_zarr(
            session.store, 
            zarr_format=3, 
            group=group_path, 
            drop_variables=to_drop,
            consolidated=False
        )
    except Exception as e:
        print(f"Fatal Error opening group '{group_path}': {e}")
        return None

    # 1. Identify pressure level dimension for ZG
    level_dim = None
    if 'zg' in ds:
        # FIMR often uses 'P' for pressure level
        for dim in ds.zg.dims:
            if dim.lower() in ['p', 'plev', 'level', 'lev', 'pressure']:
                level_dim = dim
                break
    
    # 2. Fix: Ensure level_dim has an index for .sel(method='nearest')
    if level_dim and level_dim not in ds.coords:
        print(f"Dimension '{level_dim}' has no coordinate. Attempting to load from store...")
        try:
            # Check for the coordinate array at the group root
            lev_ds = xr.open_zarr(session.store, zarr_format=3, group=f"{group_path}/{level_dim}")
            ds = ds.assign_coords({level_dim: lev_ds[level_dim]})
            print(f"Successfully loaded and assigned coordinate '{level_dim}'.")
        except Exception as e:
            print(f"Warning: Could not load coordinate '{level_dim}': {e}")

    # 3. Select ZG at 850hPa
    if 'zg' in ds:
        if level_dim and level_dim in ds.coords:
            print(f"Selecting 'zg' at 850hPa using level dimension: {level_dim}")
            try:
                # Check for Pa vs hPa units
                max_val = ds[level_dim].max().item()
                target_val = 85000 if max_val > 5000 else 850
                zg_850 = ds.zg.sel({level_dim: target_val}, method='nearest')
            except Exception as e:
                print(f"Error slicing zg: {e}. Falling back to isel.")
                # Fallback to a reasonable heuristic (index 10 is often 850hPa)
                zg_850 = ds.zg.isel({level_dim: min(10, ds.sizes[level_dim]-1)})
        else:
            print("Warning: No level coordinate for ZG selection. Using raw.")
            zg_850 = ds.zg
    else:
        print("Error: 'zg' not found in dataset.")
        return None

    # 3. Assemble final subset and rename zg to zg850 as requested
    subset = xr.Dataset({
        'pr': ds['pr'],
        'tas': ds['tas'],
        'zg850': zg_850
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
                # Force Zarr V3 to match source format and avoid metadata interpret errors
                ds_year.to_zarr(output_path, mode='w', zarr_format=3)
            
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
