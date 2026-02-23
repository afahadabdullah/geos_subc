import xarray as xr
import numpy as np
import os
import argparse

def verify_coverage(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n--- Verifying Coverage: {os.path.basename(file_path)} ---")
    try:
        ds = xr.open_zarr(file_path, consolidated=False)
        
        # Detect longitude coordinate
        lon_name = next((c for c in ['X', 'longitude', 'lon', 'x'] if c in ds.coords), None)
        if not lon_name:
            print(f"  Error: Could not find longitude coordinate. Available: {list(ds.coords)}")
            return
            
        lons = ds[lon_name].values
        print(f"  Longitude Range: {lons.min():.2f} to {lons.max():.2f} (Name: {lon_name})")
        
        # Check first variable (non-coordinate)
        var_name = next((v for v in ds.data_vars if v not in ds.coords), None)
        if not var_name:
            print("  No data variables found.")
            return

        print(f"  Checking variable: '{var_name}'")
        data = ds[var_name]
        
        # Split into Western (0-180) and Eastern (180-360) hemispheres if 0-360
        # Or just check for NaNs everywhere
        total_nans = np.isnan(data).sum().values.item()
        total_size = data.size
        nan_perc = (total_nans / total_size) * 100
        print(f"  Total NaNs: {total_nans} ({nan_perc:.2f}%)")
        
        # Check by longitude blocks
        if len(lons) >= 2:
            mid_val = (lons.min() + lons.max()) / 2
            west_mask = ds[lon_name] < mid_val
            east_mask = ds[lon_name] >= mid_val
            
            west_nans = np.isnan(data.where(west_mask)).sum().values.item()
            east_nans = np.isnan(data.where(east_mask)).sum().values.item()
            
            print(f"  Gap Analysis:")
            print(f"    Block 1 (< {mid_val:.1f}): {west_nans} NaNs")
            print(f"    Block 2 (>= {mid_val:.1f}): {east_nans} NaNs")
            
            if abs(west_nans - east_nans) > (total_size / 4):
                print("  ⚠️ WARNING: Significant imbalance detected! One hemisphere is likely empty.")
            else:
                print("  ✅ Coverage looks balanced.")
        
        ds.close()
        
    except Exception as e:
        print(f"  Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Zarr files to check")
    parser.add_argument("--dir", default="dataprocess", help="Directory to scan")
    args = parser.parse_args()
    
    files = args.files
    if not files:
        # Scan directory for common weekly files
        patterns = ["soilw_weekly_*.zarr", "sst_weekly_*.zarr", "sss_weekly_*.zarr", "ivt_weekly_*.zarr"]
        import glob
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(args.dir, p)))
            
    if not files:
        print("No files found to verify.")
    else:
        for f in sorted(files):
            verify_coverage(f)
