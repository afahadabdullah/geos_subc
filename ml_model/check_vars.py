import xarray as xr
import os

def check():
    path = "dataprocess/geos_subc_2000.zarr"
    if not os.path.exists(path):
        print(f"{path} not found.")
        return

    ds = xr.open_zarr(path, consolidated=False)
    print("Variables in GEOS 2000:")
    for v in ds.variables:
        print(f"  - {v}: {ds[v].shape}")
        
    # Check GPCP too
    path_g = "dataprocess/gpcp_weekly_2000.zarr"
    if os.path.exists(path_g):
        ds_g = xr.open_zarr(path_g, consolidated=False)
        print("\nVariables in GPCP 2000:")
        for v in ds_g.variables:
            print(f"  - {v}: {ds_g[v].shape}")

if __name__ == "__main__":
    check()
