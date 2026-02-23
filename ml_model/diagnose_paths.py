import os
import yaml
import torch

def diagnose():
    config_path = "ml_model/config_diffusion_v4.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data_dir = config.get("data_dir", "dataprocess")
    print(f"Checking data_dir: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"ERROR: data_dir does not exist!")
        return
        
    files = os.listdir(data_dir)
    print(f"Found {len(files)} items in data_dir.")
    
    # Check for specific files for Year 1999
    year = 1999
    expected = [
        f"geos_subc_{year}.zarr",
        f"gpcp_weekly_{year}.zarr",
        f"sst_weekly_{year}.zarr",
        f"sss_weekly_{year}.zarr",
        f"soilw_weekly_{year}.zarr",
        f"ivt_weekly_{year}.zarr",
        f"z500_u250_weekly_{year}.zarr"
    ]
    
    for f in expected:
        path = os.path.join(data_dir, f)
        exists = os.path.exists(path)
        print(f"{f:<30} | Exists: {exists}")
        if not exists:
            # Try to find similar files
            base = f.split('_')[0]
            matches = [m for m in files if m.startswith(base) and str(year) in m]
            if matches:
                print(f"  --> Suggestions: {matches}")

    # Check stats file
    stats_file = "ml_model/v4_global_stats.pt"
    if os.path.exists(stats_file):
        stats = torch.load(stats_file, weights_only=True)
        print(f"\nStats Keys: {list(stats.keys())}")
        for k in ['sst', 'sss', 'sm', 'ivt', 'z500', 'u250', 'geos_raw']:
            if k in stats:
                print(f"  {k:<10}: {stats[k]}")
    else:
        print(f"\nERROR: Stats file not found: {stats_file}")

if __name__ == "__main__":
    diagnose()
