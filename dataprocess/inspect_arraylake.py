from arraylake import Client
import xarray as xr

client = Client()
repo = client.get_repo("umd/subc")
session = repo.writable_session(branch="main")

import zarr

print("=== All Groups in umd/subc ===")
try:
    z = zarr.open(session.store, mode='r', zarr_format=3)
    for k in z.group_keys():
        print(f" - {k}")
except Exception as e:
    print(f"Could not list Zarr groups: {e}")

print("\n=== Inspecting esrl-fimr1p1-hindcast ===")
ds = xr.open_zarr(session.store, zarr_format=3, group="esrl-fimr1p1-hindcast", consolidated=False)

if 'pr' in ds:
    print(f"pr time dim size: {ds.pr.sizes}")
if 'tas' in ds:
    print(f"tas time dim size: {ds.tas.sizes}")
if 'zg' in ds:
    print(f"zg time dim size: {ds.zg.sizes}")
else:
    print("zg not found in esrl-fimr1p1-hindcast")

print("\nHint: If you wanted GEOS data, is there a gmao-geos-v2p1 group in the list above?")
