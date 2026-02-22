import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Use the dataset directly, but we will temporarily override its normalization 
# behavior so it returns raw values for us to scan.
from dataset_hybrid import S2SHybridDataset

def calculate_global_stats(data_root, out_path="v4_global_stats.pt", batch_size=32):
    # Initialize the dataset with normalize=False so we get the pure physical arrays
    print("Initializing S2SHybridDataset for scanning (1999-2014)...")
    dataset = S2SHybridDataset(data_root=data_root, years=range(1999, 2015), normalize=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Dictionary to keep track of Absolute Min and Max for every variable
    bounds = {
        "sst": {"min": float('inf'), "max": float('-inf')},
        "sss": {"min": float('inf'), "max": float('-inf')},
        "sm": {"min": float('inf'), "max": float('-inf')},
        "ivt": {"min": float('inf'), "max": float('-inf')},
        "z500": {"min": float('inf'), "max": float('-inf')},
        "u250": {"min": float('inf'), "max": float('-inf')},
        "geos_log": {"min": float('inf'), "max": float('-inf')},
        "residual_log": {"min": float('inf'), "max": float('-inf')}
    }

    def update_bounds(key, tensor):
        b_min = tensor.min().item()
        b_max = tensor.max().item()
        if b_min < bounds[key]["min"]: bounds[key]["min"] = b_min
        if b_max > bounds[key]["max"]: bounds[key]["max"] = b_max

    print(f"Scanning {len(dataset)} samples to calculate Global Min-Max limits...")
    for batch in tqdm(loader):
        # 1. Obvservations: [B, 24, H, W]
        # Order from dataset_hybrid: SST(0:4), SSS(4:8), SM(8:12), IVT(12:16), Z500(16:20), U250(20:24)
        x_obs = batch['x_obs'] 
        update_bounds("sst", x_obs[:, 0:4, :, :])
        update_bounds("sss", x_obs[:, 4:8, :, :])
        update_bounds("sm", x_obs[:, 8:12, :, :])
        update_bounds("ivt", x_obs[:, 12:16, :, :])
        update_bounds("z500", x_obs[:, 16:20, :, :])
        update_bounds("u250", x_obs[:, 20:24, :, :])

        # 2. GEOS Forecast [B, 1, 1, L, H, W]
        # Currently, the dataset_hybrid.py already does log1p inside its __getitem__ for GEOS
        # So it's already in log space.
        x_geos = batch['x_geos']
        update_bounds("geos_log", x_geos)

        # 3. GPCP Target [B, L, H, W]
        y_target = batch['y_target']
        y_log = torch.log1p(y_target.clamp(min=0.0))
        
        # Calculate residual log bounds directly
        # x_geos is [B, 1, 1, L, H, W] in the dataset batch output
        geos_log = x_geos.squeeze(1).squeeze(1) # [B, L, H, W]
        residual_log = y_log - geos_log
        update_bounds("residual_log", residual_log)

    print("\n==================================")
    print("Calculated Global Bounds (1999-2014)")
    print("==================================")
    for k, v in bounds.items():
        print(f"{k.upper():<10} | Min: {v['min']:>12.4f} | Max: {v['max']:>12.4f}")

    # Save to dictionary
    torch.save(bounds, out_path)
    print(f"\nSaved strict global limits to {out_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/scratch/11353/afahad/geossub/data")
    parser.add_argument("--out", type=str, default="v4_global_stats.pt")
    args = parser.parse_args()
    
    calculate_global_stats(args.data_root, args.out)
