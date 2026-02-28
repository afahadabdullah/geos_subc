import xarray as xr
import numpy as np
import sys
import os

path = '/scratch/11353/afahad/geossub/geos_subc/dataprocess/geos_subc_2019.zarr'

print(f"Opening {path}...")
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)
    
try:
    ds = xr.open_zarr(path, consolidated=False)
    var = 'pr' if 'pr' in ds else 'precip'
    print(f"Variable: {var}, Full Shape: {ds[var].shape}")
    
    valid, empty = 0, 0
    for i in range(ds.sizes['S']):
        val = ds[var].isel(S=i).values
        if np.isnan(val).all():
            empty += 1
        elif np.isnan(val).any():
            valid += 1  # Partially masked, totally normal
        else:
            valid += 1
            
    print(f"----------------------------------------")
    print(f"Total Initialization Dates: {ds.sizes['S']}")
    print(f"Completely Empty (100% NaN): {empty}")
    print(f"Valid Data Dates (or partial masking): {valid}")
    print(f"----------------------------------------")
    
    if empty > 0:
        print("\nWARNING: It appears the FIMR preprocessing script left most initialization dates completely blank (NaN).")
        print("This causes Flow Matching to receive an input array of zeroes, resulting in a blank plot (normalized to exactly -1.0).")
        
except Exception as e:
    print(f"Error: {e}")
