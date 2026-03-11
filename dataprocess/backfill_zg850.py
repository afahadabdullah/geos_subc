from arraylake import Client
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
import dask
import os
import shutil

def backfill_zg850():
    """
    Connects to ArrayLake, retrieves zg850 from esrl-fimr1p1-hindcast,
    and appends it to the existing geos_subc_{year}.zarr files.
    """
    print("Connecting to ArrayLake...")
    client = Client()
    repo = client.get_repo("umd/subc")
    print("Creating session for branch 'main'...")
    session = repo.writable_session(branch="main")
    
    group_path = "esrl-fimr1p1-hindcast"
    try:
        print(f"Opening {group_path} from ArrayLake...")
        ds_raw = xr.open_zarr(
            session.store, 
            zarr_format=3, 
            group=group_path, 
            consolidated=False
        )
        print(f"Opened ArrayLake dataset. Keys: {list(ds_raw.data_vars.keys())}")
    except Exception as e:
        return

    # Process zg to zg850
    level_dim = None
    if 'zg' in ds_raw:
        for dim in ds_raw.zg.dims:
            if dim.lower() in ['p', 'plev', 'level', 'lev', 'pressure']:
                level_dim = dim
                break

    if not level_dim:
        print("Warning: Could not identify level dimension for zg. Using raw zg.")
        zg_850 = ds_raw.zg
    else:
        if level_dim not in ds_raw.coords and level_dim in ds_raw.data_vars:
            ds_raw = ds_raw.set_coords(level_dim)
        
        # Determine 850hPa target
        max_val = float(ds_raw[level_dim].max())
        target = 85000 if max_val > 5000 else 850
        print(f"Extracting zg at {target} {level_dim}...")
        
        zg_850 = ds_raw.zg.sel({level_dim: target}, method='nearest')
        if level_dim in zg_850.coords:
            zg_850 = zg_850.drop_vars(level_dim)

    # Rename to zg850
    ds_zg850 = xr.Dataset({'zg850': zg_850})

    time_dim = next((d for d in ['S', 'time', 'init_time'] if d in ds_zg850.dims), None)
    if not time_dim:
        time_coords = [c for c in ds_zg850.coords if 'time' in c.lower()]
        time_dim = time_coords[0] if time_coords else None

    if not time_dim:
        print("Error: Could not identify time dimension.")
        return

    # Debug: Print the min and max dates available
    try:
        t_min, t_max = ds_zg850[time_dim].min().values, ds_zg850[time_dim].max().values
        print(f"Data available in ArrayLake from {t_min} to {t_max}")
    except Exception as e:
        print(f"Failed to check time bounds: {e}")

    # Loop through years 1999 to 2016
    for year in range(1999, 2017):
        target_zarr = f"dataprocess/geos_subc_{year}.zarr"
        if not os.path.exists(target_zarr):
            print(f"Skipping {year}, {target_zarr} not found.")
            continue

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        print(f"\n--- Backfilling Year: {year} ---")
        try:
            # Check if zg850 is already there
            existing_ds = xr.open_zarr(target_zarr, consolidated=False)
            if 'zg850' in existing_ds.data_vars:
                print(f"zg850 already exists in {target_zarr}. Skipping.")
                existing_ds.close()
                continue
            existing_ds.close()

            # Filter remote data
            # Use 'time' or 'S' depending on what ArrayLake has
            # The ArrayLake dataset seems to use 'init_time' or similar for some datasets
            # The dates might also correspond to only a few initializations
            try:
                ds_year = ds_zg850.sel({time_dim: slice(start_date, end_date)})
            except KeyError:
                print(f"Failed to slice {time_dim} for {year}. ArrayLake dims: {ds_zg850.dims}, coords: {list(ds_zg850.coords)}")
                continue

            if ds_year.sizes[time_dim] == 0:
                print(f"No zg850 data found for {year} in ArrayLake (Group: {group_path}).")
                continue
                
            # Process to weekly (L=32 -> L=4)
            if 'L' in ds_year.dims:
                if ds_year.sizes['L'] >= 28:
                    ds_year = ds_year.isel(L=slice(0, 28))
                    ds_year = ds_year.coarsen(L=7, boundary='exact').mean()
                    print(f"Coarsened zg850 to L={ds_year.sizes['L']}")
                else:
                    print(f"Warning: L dimension is {ds_year.sizes['L']}, expected >= 28")

            # Strip encoding (same trick as load_data2.py)
            ds_year.encoding = {}
            for var in ds_year.variables:
                ds_year[var].encoding = {}

            print(f"Merging zg850 ({ds_year.sizes[time_dim]} steps) into {target_zarr}...")
            
            # Load existing
            existing_ds = xr.open_zarr(target_zarr, consolidated=False)
            
            # Merge
            merged_ds = xr.merge([existing_ds, ds_year])
            
            # Write temp
            temp_zarr = f"dataprocess/geos_subc_{year}_temp.zarr"
            with ProgressBar(), dask.config.set(scheduler='synchronous'):
                merged_ds.to_zarr(temp_zarr, mode='w', zarr_format=3)
                
            existing_ds.close()
            
            # Swap
            print(f"Overwriting original file {target_zarr}...")
            shutil.rmtree(target_zarr)
            shutil.move(temp_zarr, target_zarr)
                
            print(f"Successfully backfilled {year}.")

        except Exception as e:
            print(f"Error backfilling {year}: {e}")

if __name__ == "__main__":
    backfill_zg850()
