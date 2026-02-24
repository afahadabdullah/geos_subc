import xarray as xr
import numpy as np
import os

def check_math(year=2000):
    path = f"/scratch/11353/afahad/geossub/geos_subc/dataprocess/z500_u250_weekly_{year}.zarr"
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    print(f"📊 Deep Math Check for {year}")
    ds = xr.open_zarr(path, consolidated=False)
    
    # Simulate _load_sample logic
    # 1. Load raw values from Zarr
    v_raw = ds['z500'].isel(S=0).values # (L, X, Y) -> (4, 360, 181)
    print(f"  Raw V shape from Zarr: {v_raw.shape}")
    print(f"  Raw V Min: {np.nanmin(v_raw):.2f} | Max: {np.nanmax(v_raw):.2f}")

    # 2. Transpose logic from dataset_hybrid.py
    v = v_raw
    if v.ndim == 3:
        if v.shape[1] == 360 and v.shape[2] == 181:
            print("  ✅ Transposing (0, 360, 181) -> (0, 181, 360)")
            v = np.transpose(v, (0, 2, 1))
    
    z500_val = v
    print(f"  z500_val shape: {z500_val.shape}")
    print(f"  z500_val Min: {np.nanmin(z500_val):.2f} | Max: {np.nanmax(z500_val):.2f}")

    # 3. Zonal Deviation logic from dataset_hybrid.py
    zonal_mean = z500_val.mean(axis=2, keepdims=True) # Average over longitude
    print(f"  zonal_mean shape: {zonal_mean.shape}")
    print(f"  zonal_mean Min: {np.nanmin(zonal_mean):.2f} | Max: {np.nanmax(zonal_mean):.2f}")

    zonal_dev = z500_val - zonal_mean
    print(f"  zonal_dev shape: {zonal_dev.shape}")
    print(f"  zonal_dev Min: {np.nanmin(zonal_dev):.2f} | Max: {np.nanmax(zonal_dev):.2f}")
    
    # 4. Check a slice to see if it's flat
    print(f"  zonal_dev sample (L=0, Y=90): {zonal_dev[0, 90, :5]}")

    ds.close()

if __name__ == "__main__":
    check_math()
