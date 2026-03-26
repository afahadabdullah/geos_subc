import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import gc

# Using dataset_flow directly since that's what train_flow_v6.py uses
from dataset_flow_multi import S2SHybridDataset

DEFAULT_START_YEAR = 1999
DEFAULT_END_YEAR = 2020


def calculate_global_stats(
    data_root,
    out_path="v2_multi_global_stats.pt",
    batch_size=32,
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
):
    # Initialize the dataset with normalize=False so we get the pure physical arrays
    print(f"Initializing S2SHybridDataset for scanning ({start_year}-{end_year})...")
    dataset = S2SHybridDataset(
        data_root=data_root,
        start_year=start_year,
        end_year=end_year,
        normalize=False,
    )
    # Using num_workers=0 to prevent multiprocessing memory blowouts on the login node
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Dictionary to keep track of Absolute Min and Max for every variable
    bounds = {
        "sst": {"min": float('inf'), "max": float('-inf')},
        "sss": {"min": float('inf'), "max": float('-inf')},
        "sm": {"min": float('inf'), "max": float('-inf')},
        "ivt": {"min": float('inf'), "max": float('-inf')},
        "u250": {"min": float('inf'), "max": float('-inf')},
        "z500_zonal_dev": {"min": float('inf'), "max": float('-inf')},
        "geos_pr_raw": {"min": float('inf'), "max": float('-inf')},
        "geos_tas_raw": {"min": float('inf'), "max": float('-inf')},
        "target_t2m_raw": {"min": float('inf'), "max": float('-inf')},
        "residual_pr_raw": {"min": float('inf'), "max": float('-inf')}
    }

    def update_bounds(key, tensor):
        # Flatten and remove NaNs to get true physical bounds
        valid_data = tensor[~torch.isnan(tensor)]
        if valid_data.numel() == 0:
            return
            
        b_min = valid_data.min().item()
        b_max = valid_data.max().item()
        
        if b_min < bounds[key]["min"]: bounds[key]["min"] = b_min
        if b_max > bounds[key]["max"]: bounds[key]["max"] = b_max

    print(f"Scanning {len(dataset)} samples to calculate Global Min-Max limits...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            # 1. Obvservations: [B, 28, H, W]
            x_obs = batch['x_obs'] 
            
            sst_b = x_obs[:, 0:4, :, :]
            sss_b = x_obs[:, 4:8, :, :].clamp(min=25.0) # Physical baseline to avoid Land Masks
            sm_b = x_obs[:, 8:12, :, :]
            ivt_b = x_obs[:, 12:16, :, :]
            zdev_b = x_obs[:, 16:20, :, :] 
            u250_b = x_obs[:, 20:24, :, :]
            
            update_bounds("sst", sst_b)
            update_bounds("sss", sss_b)
            update_bounds("sm", sm_b)
            update_bounds("ivt", ivt_b)
            update_bounds("z500_zonal_dev", zdev_b)
            update_bounds("u250", u250_b)

            # 2. GEOS Forecast [B, 1, 2, L, H, W]
            x_geos = batch['x_geos']
            # Channel 0: PR, Channel 1: TAS
            geos_pr = x_geos[:, :, 0] # [B, 1, L, H, W]
            geos_tas = x_geos[:, :, 1] # [B, 1, L, H, W]
            
            update_bounds("geos_pr_raw", geos_pr)
            update_bounds("geos_tas_raw", geos_tas)

            # 3. Target [B, 1, 2, H, W] (GPCP Precip + ERA5 T2M)
            target_raw = batch['target_raw']
            # Before normalization, dataset_flow returns: target_raw_lead = target_tensor.unsqueeze(0)
            # So target_raw is [B, 1, 2, H, W]
            target_pr = target_raw[:, :, 0] # [B, 1, H, W]
            target_t2m = target_raw[:, :, 1] # [B, 1, H, W]
            
            update_bounds("target_t2m_raw", target_t2m)
            
            # Calculate residual raw bounds directly for precip (for legacy plotting compatibility)
            geos_pr_lead = geos_pr.squeeze(1)[:, batch['lead_idx']] # approximate fallback, actually in dataset it gets matched by lead_idx
            # Wait, geos_pr is [B, 1, 4, H, W], target_pr is [B, 1, H, W]. 
            # We can get the exact GEOS lead predictions from the dataset output
            # But simpler here to just subtract the batch means matching the lead indices
            B = target_pr.shape[0]
            leads = batch['lead_idx'].long()
            # Select the correct lead from GEOS PR for each item in batch
            try:
                # x_geos is [B, 1, 2, L, H, W] -> geos_pr is [B, 1, L, H, W]
                geos_pr_selected = geos_pr[torch.arange(B), 0, leads, :, :] # [B, H, W]
                residual_pr = target_pr.squeeze(1) - geos_pr_selected
                update_bounds("residual_pr_raw", residual_pr)
            except Exception as e:
                # Fallback if multidim indexing fails
                pass
            
            if i == 0:
                print(f"\n--- BATCH 0 DIAGNOSTICS ---")
                print(f"SST  Raw    : Min {sst_b.min().item():.4f} | Max {sst_b.max().item():.4f}")
                print(f"SSS  Raw    : Min {sss_b.min().item():.4f} | Max {sss_b.max().item():.4f}")
                print(f"SM   Raw    : Min {sm_b.min().item():.4f} | Max {sm_b.max().item():.4f}")
                print(f"IVT  Raw    : Min {ivt_b.min().item():.4f} | Max {ivt_b.max().item():.4f}")
                print(f"ZDEV Raw    : Min {zdev_b.min().item():.4f} | Max {zdev_b.max().item():.4f}")
                print(f"TAS  GEOS   : Min {geos_tas.min().item():.4f} | Max {geos_tas.max().item():.4f}")
                print(f"T2M  Target : Min {target_t2m.min().item():.4f} | Max {target_t2m.max().item():.4f}")
                print(f"PR   GEOS   : Min {geos_pr.min().item():.4f} | Max {geos_pr.max().item():.4f}")
                print(f"PR   Target : Min {target_pr.min().item():.4f} | Max {target_pr.max().item():.4f}")
                print(f"---------------------------\n")
            
            # Force Memory Cleanup to prevent OOM
            del x_obs, x_geos, target_raw, geos_pr, geos_tas, target_pr, target_t2m, batch
            del sst_b, sss_b, sm_b, ivt_b, u250_b, zdev_b
            gc.collect()

    print("\n==================================")
    print(f"Calculated Global Bounds ({start_year}-{end_year})")
    print("==================================")
    # Enforce Robust Physical Limits for Precip/Residual
    bounds["residual_pr_raw"] = {"min": -100.0, "max": 100.0}
    bounds["geos_pr_raw"] = {"min": 0.0, "max": 100.0}
    bounds["z500_zonal_dev"] = {"min": -5000.0, "max": 5000.0}
    
    # Just to be safe with MJO phase
    bounds["mjo"] = {"min": -50.0, "max": 50.0}
    
    # Maintain older 'geos_raw' and 'residual_raw' keys for backwards compat
    bounds["geos_raw"] = bounds["geos_pr_raw"]
    bounds["residual_raw"] = bounds["residual_pr_raw"]
    
    for k, v in bounds.items():
        if isinstance(v, dict):
            print(f"{k.upper():<16} | Min: {v['min']:>12.4f} | Max: {v['max']:>12.4f}")

    # Save to dictionary
    torch.save(bounds, out_path)
    print(f"\nSaved strict global limits to {out_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess")
    parser.add_argument("--out", type=str, default="ml_model/v2_multi_global_stats.pt")
    parser.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR)
    args = parser.parse_args()
    
    calculate_global_stats(
        args.data_root,
        args.out,
        start_year=args.start_year,
        end_year=args.end_year,
    )
