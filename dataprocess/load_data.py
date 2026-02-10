from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

def load_geos_subc_data():
    """
    Loads GEOS Subseasonal-to-Seasonal (SubC) forecast data from Earthmover ArrayLake.
    Filters for year 2000 and selects variables 'pr' and 'tas'.
    """
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    session = repo.writable_session(branch="main")
    
    print("Opening Zarr store via Xarray...")
    # Open the dataset group 'esrl-fimr1p1-hindcast'
    ds = xr.open_zarr(session.store, zarr_format=3, group="esrl-fimr1p1-hindcast")
    
    print("Filtering for year 2000 and selecting variables 'pr', 'tas'...")
    # Select variables 'pr' and 'tas'
    ds_subset = ds[['pr', 'tas']]
    
    # Filter for the year 2000. 
    # The dimension 'S' is the initialization time dimension.
    if 'S' in ds_subset.dims:
        print("Using dimension 'S' for filtering.")
        ds_2000 = ds_subset.sel(S=slice('2000-01-01', '2000-12-31'))
    elif 'time' in ds_subset.dims:
        ds_2000 = ds_subset.sel(time=slice('2000-01-01', '2000-12-31'))
    elif 'init_time' in ds_subset.dims:
        ds_2000 = ds_subset.sel(init_time=slice('2000-01-01', '2000-12-31'))
    else:
        # Fallback: check coordinate names if dims don't match exactly
        time_coord = [c for c in ds_subset.coords if 'time' in c.lower()]
        if time_coord:
            ds_2000 = ds_subset.sel({time_coord[0]: slice('2000-01-01', '2000-12-31')})
        else:
            print("Warning: Could not find a 'time' dimension for filtering. Returning full dataset.")
            ds_2000 = ds_subset

    return ds_2000

if __name__ == "__main__":
    try:
        data = load_geos_subc_data()
        print("\nDataset Summary:")
        print(data)
        
        # Save to Zarr with progress bar
        output_path = "dataprocess/geos_subc_2000.zarr"
        print(f"\nSaving yearly data to {output_path}...")
        with ProgressBar():
            data.to_zarr(output_path, mode='w', zarr_format=3)
        print("Successfully saved data.")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you are logged in to ArrayLake.")
        print("Run: arraylake auth login (if working via SSH/Terminal)")
