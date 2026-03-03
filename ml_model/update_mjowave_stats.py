import torch
import os
import argparse
import numpy as np
import xarray as xr
import glob
from tqdm import tqdm

def update_mjo_stats(data_root, stats_path):
    if not os.path.exists(stats_path):
        print(f"❌ Stats file not found: {stats_path}")
        return

    print(f"🔄 Loading existing stats: {stats_path}")
    global_stats = torch.load(stats_path, map_location='cpu', weights_only=True)

    # Glob all MJO zarr files
    mjo_files = sorted(glob.glob(os.path.join(data_root, "mjo_wave_spatial_*.zarr")))
    if not mjo_files:
        print(f"❌ No MJO zarr files found in {data_root}")
        return

    print(f"Loading {len(mjo_files)} continuous MJO wave datasets...")
    ds_list = [xr.open_zarr(f, consolidated=False) for f in mjo_files]
    ds_mjo_full = xr.concat(ds_list, dim='time').sortby('time')

    mjo_min = float('inf')
    mjo_max = float('-inf')

    # Load into RAM for fast scanning
    print(f"Loading MJO array into RAM for fast scanning...")
    da_mjo = ds_mjo_full['mjo_wave'].values
    
    # We ignore NaN values (e.g. land mask if applied, though MJO is global)
    mjo_min = np.nanmin(da_mjo)
    mjo_max = np.nanmax(da_mjo)

    print(f"\n✅ Scan Complete!")
    print(f"  Old MJO bounds: {global_stats.get('mjo', 'Missing')}")
    print(f"  New MJO bounds: {{'min': {mjo_min:.2f}, 'max': {mjo_max:.2f}}}")

    # Update the dictionary
    global_stats["mjo"] = {"min": float(mjo_min), "max": float(mjo_max)}

    # Save back
    torch.save(global_stats, stats_path)
    print(f"💾 Updated stats saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess")
    parser.add_argument("--stats_path", type=str, default="/scratch/11353/afahad/geossub/geos_subc/ml_model/v5_global_stats.pt")
    args = parser.parse_args()

    update_mjo_stats(args.data_root, args.stats_path)
