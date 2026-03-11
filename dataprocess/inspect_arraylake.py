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

print("\n=== Inspecting ALL groups for 'pr' dimension 'S' ===")
for group_name in sorted(z.group_keys()):
    try:
        ds = xr.open_zarr(session.store, zarr_format=3, group=group_name, consolidated=False)
        if 'pr' in ds:
            time_dim = next((d for d in ['S', 'time', 'init_time'] if d in ds.dims), 'UNKNOWN')
            if time_dim != 'UNKNOWN':
                size = ds.sizes[time_dim]
                print(f"{group_name}: {time_dim} = {size}")
            else:
                print(f"{group_name}: no known time dim, dims are {ds.dims}")
        else:
            print(f"{group_name}: 'pr' not found")
    except Exception as e:
        print(f"{group_name}: Could not open dataset ({e})")

print("\nHint: If you wanted GEOS data, is there a gmao-geos-v2p1 group in the list above?")
