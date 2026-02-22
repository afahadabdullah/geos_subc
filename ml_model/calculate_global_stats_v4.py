import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Use the dataset directly, but we will temporarily override its normalization 
# behavior so it returns raw values for us to scan.
import gc
from dataset_hybrid import S2SHybridDataset

def calculate_global_stats(data_root, out_path="v4_global_stats.pt", batch_size=32):
    # Initialize the dataset with normalize=False so we get the pure physical arrays
    print("Initializing S2SHybridDataset for scanning (1999-2014)...")
    dataset = S2SHybridDataset(data_root=data_root, start_year=1999, end_year=2014, normalize=False)
    # Using num_workers=0 to prevent multiprocessing memory blowouts on the login node
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            # 1. Obvservations: [B, 24, H, W]
            x_obs = batch['x_obs'] 
            
            sst_b = x_obs[:, 0:4, :, :]
            sss_b = x_obs[:, 4:8, :, :]
            sm_b = x_obs[:, 8:12, :, :]
            ivt_b = x_obs[:, 12:16, :, :]
            z500_b = x_obs[:, 16:20, :, :]
            u250_b = x_obs[:, 20:24, :, :]
            
            update_bounds("sst", sst_b)
            update_bounds("sss", sss_b)
            update_bounds("sm", sm_b)
            update_bounds("ivt", ivt_b)
            update_bounds("z500", z500_b)
            update_bounds("u250", u250_b)

            # 2. GEOS Forecast [B, 1, 1, L, H, W]
            x_geos = batch['x_geos']
            update_bounds("geos_log", x_geos)

            # 3. GPCP Target [B, L, H, W]
            y_target = batch['y_target']
            y_log = torch.log1p(y_target.clamp(min=0.0))
            
            # Calculate residual log bounds directly
            geos_log = x_geos.squeeze(1).squeeze(1) # [B, L, H, W]
            residual_log = y_log - geos_log
            update_bounds("residual_log", residual_log)
            
            if i == 0:
                print(f"\n--- BATCH 0 DIAGNOSTICS ---")
                print(f"SST  Raw  : Min {sst_b.min().item():.4f} | Max {sst_b.max().item():.4f}")
                print(f"SSS  Raw  : Min {sss_b.min().item():.4f} | Max {sss_b.max().item():.4f}")
                print(f"SM   Raw  : Min {sm_b.min().item():.4f} | Max {sm_b.max().item():.4f}")
                print(f"IVT  Raw  : Min {ivt_b.min().item():.4f} | Max {ivt_b.max().item():.4f}")
                print(f"Z500 Raw  : Min {z500_b.min().item():.4f} | Max {z500_b.max().item():.4f}")
                print(f"U250 Raw  : Min {u250_b.min().item():.4f} | Max {u250_b.max().item():.4f}")
                print(f"GEOS Log  : Min {x_geos.min().item():.4f} | Max {x_geos.max().item():.4f}")
                print(f"RESID Log : Min {residual_log.min().item():.4f} | Max {residual_log.max().item():.4f}")
                print(f"---------------------------\n")
            
            # Force Memory Cleanup to prevent OOM
            del x_obs, x_geos, y_target, y_log, geos_log, residual_log, batch
            del sst_b, sss_b, sm_b, ivt_b, z500_b, u250_b
            gc.collect()

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
    parser.add_argument("--out", type=str, default="ml_model/v4_global_stats.pt")
    args = parser.parse_args()
    
    calculate_global_stats(args.data_root, args.out)
