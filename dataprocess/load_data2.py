from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import dask
import os

def get_base_dataset():
    """
    Connects to ArrayLake and returns the FIMR forecast dataset.
    Preserves dimensions while dropping misaligned data variables.
    """
    import zarr
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    print("Creating session for branch 'main'...")
    session = repo.writable_session(branch="main")
    
    print("Inspecting group keys for 'esrl-fimr1p1-forecast'...")
    try:
        # Use zarr briefly to identify variables
        z = zarr.open(session.store, mode='r', zarr_format=3)
        group_path = "esrl-fimr1p1-forecast"
        
        all_keys = list(z[group_path].keys())
        # We MUST keep dimensions/coordinates so xarray can resolve them
        data_vars = ['pr', 'tas', 'zg']
        # Typically coordinate names are single letters or common dims
        dim_keys = ['S', 'M', 'L', 'P', 'X', 'Y', 'lat', 'lon', 'latitude', 'longitude', 'time', 'level', 'plev']
        
        keep_keys = data_vars + dim_keys
        to_drop = [k for k in all_keys if k not in keep_keys]
        
        print(f"Found {len(all_keys)} keys. Dropping {len(to_drop)} variables to avoid alignment conflicts.")
        
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
        for dim in ds.zg.dims:
            if dim.lower() in ['p', 'plev', 'level', 'lev', 'pressure']:
                level_dim = dim
                break
    
    # 2. Select ZG at 850hPa
    if 'zg' in ds:
        if level_dim:
            if level_dim not in ds.coords and level_dim in ds.data_vars:
                ds = ds.set_coords(level_dim)
                
            print(f"Selecting 'zg' at 850hPa using level dimension: {level_dim}")
            try:
                # Handle Pa vs hPa units
                max_val = float(ds[level_dim].max())
                target = 85000 if max_val > 5000 else 850
                zg_850 = ds.zg.sel({level_dim: target}, method='nearest')
            except Exception as e:
                print(f"Warning: .sel() failed ({e}), using fallback indexing.")
                zg_850 = ds.zg.isel({level_dim: min(10, ds.sizes[level_dim]-1)})
        else:
            print("Warning: No level dimension found for 'zg'. Using raw 'zg'.")
            zg_850 = ds.zg
    else:
        print("Error: 'zg' not found in dataset.")
        return None

    # 3. Assemble final subset and rename zg to zg850
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
    
    os.makedirs("dataprocess", exist_ok=True)
    
    time_dim = next((d for d in ['S', 'time', 'init_time'] if d in ds_subset.dims), None)
    if not time_dim:
        time_coords = [c for c in ds_subset.coords if 'time' in c.lower()]
        time_dim = time_coords[0] if time_coords else None

    if not time_dim:
        print("Error: Could not identify time dimension.")
        return

    print(f"Using dimension '{time_dim}' for filtering.")

    for year in range(2017, 2026):
        start, end = f"{year}-01-01", f"{year}-12-31"
        out_path = f"dataprocess/fimr_forecast_{year}.zarr"
        
        print(f"\n--- Processing Year: {year} ---")
        if os.path.exists(out_path):
            print(f"Skipping {year}, exists.")
            continue

        try:
            ds_year = ds_subset.sel({time_dim: slice(start, end)})
            if ds_year.sizes[time_dim] == 0:
                print(f"No data for {year}.")
                continue
                
            # CRITICAL: Strip all encoding metadata. 
            # This fixes the "'str' object cannot be interpreted as an integer" 
            # error which happens when stale V3 metadata conflicts with the output backend.
            ds_year.encoding = {}
            for var in ds_year.variables:
                ds_year[var].encoding = {}

            print(f"Saving to {out_path} ({ds_year.sizes[time_dim]} steps)...")
            with ProgressBar(), dask.config.set(scheduler='synchronous'):
                # We use default format for local saving as it's more compatible with older readers
                ds_year.to_zarr(out_path, mode='w')
            
            print(f"Successfully saved {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

if __name__ == "__main__":
    try:
        save_yearly_data()
        print("\nAll tasks completed.")
    except Exception as e:
        print(f"\nFatal Error: {e}")
