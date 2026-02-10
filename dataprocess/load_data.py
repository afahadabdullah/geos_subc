from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import dask

def load_geos_subc_data():
    """
    Loads GEOS Subseasonal-to-Seasonal (SubC) hindcast data from Earthmover ArrayLake.
    Filters for years 1999-2016 and selects variables 'pr' and 'tas'.
    """
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    # Using writable_session as per user specification
    session = repo.writable_session(branch="main")
    
    print("Opening Zarr store via Xarray...")
    # Open the dataset group 'esrl-fimr1p1-hindcast'
    ds = xr.open_zarr(session.store, zarr_format=3, group="esrl-fimr1p1-hindcast")
    
    print("Filtering for years 1999-2016 and selecting variables 'pr', 'tas'...")
    # Select variables 'pr' and 'tas'
    ds_subset = ds[['pr', 'tas']]
    
    # Filter for the full period 1999-2016.
    # The dimension 'S' is the initialization time dimension.
    if 'S' in ds_subset.dims:
        print("Using dimension 'S' for filtering.")
        ds_filtered = ds_subset.sel(S=slice('1999-01-01', '2016-12-31'))
    elif 'time' in ds_subset.dims:
        ds_filtered = ds_subset.sel(time=slice('1999-01-01', '2016-12-31'))
    elif 'init_time' in ds_subset.dims:
        ds_filtered = ds_subset.sel(init_time=slice('1999-01-01', '2016-12-31'))
    else:
        # Fallback: check coordinate names if dims don't match exactly
        time_coord = [c for c in ds_subset.coords if 'time' in c.lower()]
        if time_coord:
            ds_filtered = ds_subset.sel({time_coord[0]: slice('1999-01-01', '2016-12-31')})
        else:
            print("Warning: Could not find a 'time' dimension for filtering. Returning full dataset.")
            ds_filtered = ds_subset

    return ds_filtered

if __name__ == "__main__":
    try:
        data = load_geos_subc_data()
        print("\nDataset Summary:")
        print(data)
        
        # Save to Zarr with progress bar and memory optimization
        output_path = "dataprocess/geos_subc_1999_2016.zarr"
        print(f"\nSaving data to {output_path}...")
        print("Note: Using synchronous scheduler to minimize memory usage.")
        
        # Using synchronous scheduler ensures we don't exceed memory limits on login/shared nodes
        with ProgressBar(), dask.config.set(scheduler='synchronous'):
            data.to_zarr(output_path, mode='w', zarr_format=3)
        print("Successfully saved data.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you are logged in to ArrayLake.")
        print("Run: arraylake auth login --no-browser")
