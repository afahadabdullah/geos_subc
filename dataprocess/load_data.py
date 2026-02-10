from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import dask
import os

def get_base_dataset():
    """
    Connects to ArrayLake and returns the base dataset group.
    """
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    session = repo.writable_session(branch="main")
    
    print("Opening Zarr store via Xarray...")
    ds = xr.open_zarr(session.store, zarr_format=3, group="esrl-fimr1p1-hindcast")
    return ds[['pr', 'tas']]

def save_yearly_data():
    """
    Filters and saves data in yearly Zarr files from 1999 to 2016.
    """
    ds_subset = get_base_dataset()
    
    # Ensure output directory exists
    os.makedirs("dataprocess", exist_ok=True)
    
    # Determine which time dimension to use
    if 'S' in ds_subset.dims:
        time_dim = 'S'
    elif 'time' in ds_subset.dims:
        time_dim = 'time'
    elif 'init_time' in ds_subset.dims:
        time_dim = 'init_time'
    else:
        # Fallback check
        time_coords = [c for c in ds_subset.coords if 'time' in c.lower()]
        time_dim = time_coords[0] if time_coords else None

    if not time_dim:
        print("Error: Could not identify time dimension for filtering.")
        return

    print(f"Using dimension '{time_dim}' for filtering.")

    # Loop through years
    for year in range(1999, 2017):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        output_path = f"dataprocess/geos_subc_{year}.zarr"
        
        print(f"\n--- Processing Year: {year} ---")
        
        # Check if file already exists
        if os.path.exists(output_path):
            print(f"Skipping {year}, file already exists.")
            continue

        try:
            # Filter for the year
            ds_year = ds_subset.sel({time_dim: slice(start_date, end_date)})
            
            if ds_year.sizes[time_dim] == 0:
                print(f"No data found for year {year}. Skipping.")
                continue
                
            print(f"Saving {year} data to {output_path} ({ds_year.sizes[time_dim]} timesteps)...")
            
            # Using synchronous scheduler for memory stability on login/compute nodes
            with ProgressBar(), dask.config.set(scheduler='synchronous'):
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
        print("\nNote: Make sure you are logged in to ArrayLake.")
        print("Run: arraylake auth login --no-browser")
