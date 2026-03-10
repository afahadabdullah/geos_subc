import xarray as xr
import numpy as np

path = "/scratch/11353/afahad/geossub/geos_subc/dataprocess/geos_s2s/2019.zarr"
print(f"Opening {path}...")
try:
    ds = xr.open_zarr(path, consolidated=False)
    var = 'pr' if 'pr' in ds else 'precip'
    print(f"Variable: {var}")
    print(f"S dimension size: {ds.sizes['S']}")
    
    valid_count = 0
    nan_count = 0
    for s_idx in range(ds.sizes['S']):
        val = ds[var].isel(S=s_idx).values
        if np.isnan(val).all():
            nan_count += 1
        elif np.isnan(val).any():
            pass # Partial NaNs due to masks
        else:
            if np.nansum(val) > 0:
                valid_count += 1
                
    print(f"Total Initializations: {ds.sizes['S']}")
    print(f"Completely NaN Initializations: {nan_count}")
    print(f"Valid (Non-zero) Initializations: {valid_count}")
except Exception as e:
    print(f"Error: {e}")
