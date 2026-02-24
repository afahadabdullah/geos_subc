import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from dataset_hybrid import S2SHybridDataset

def update_zdev_stats(data_root, stats_path):
    if not os.path.exists(stats_path):
        print(f"❌ Stats file not found: {stats_path}")
        return

    print(f"🔄 Loading existing stats: {stats_path}")
    global_stats = torch.load(stats_path, map_location='cpu')

    # Initialize dataset (normalize=False to compute real bounds)
    # We use a subset of years or all as per user preference, but let's do the full V5 range
    dataset = S2SHybridDataset(data_root=data_root, start_year=1999, end_year=2022, normalize=False)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    zdev_min = float('inf')
    zdev_max = float('-inf')

    print(f"🔍 Scanning {len(dataset)} samples for Z500 Zonal Deviation bounds...")
    
    for i, batch in enumerate(tqdm(loader)):
        # In V5.1, Zonal Deviation is indices 16:20 of x_obs
        # x_obs shape: [B, 24, H, W]
        x_obs = batch['x_obs']
        zdev_batch = x_obs[:, 16:20, :, :]
        
        batch_min = zdev_batch.min().item()
        batch_max = zdev_batch.max().item()
        
        if batch_min < zdev_min: zdev_min = batch_min
        if batch_max > zdev_max: zdev_max = batch_max

        if i == 0:
            print(f"  First Batch Min: {batch_min:.2f} | Max: {batch_max:.2f}")

    print(f"\n✅ Scan Complete!")
    print(f"  Old ZDEV bounds: {global_stats.get('z500_zonal_dev', 'Missing')}")
    print(f"  New ZDEV bounds: {{'min': {zdev_min}, 'max': {zdev_max}}}")

    # Update the dictionary
    global_stats["z500_zonal_dev"] = {"min": zdev_min, "max": zdev_max}

    # Save back
    torch.save(global_stats, stats_path)
    print(f"💾 Updated stats saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess")
    parser.add_argument("--stats_path", type=str, default="/scratch/11353/afahad/geossub/geos_subc/ml_model/v5_global_stats.pt")
    args = parser.parse_args()

    update_zdev_stats(args.data_root, args.stats_path)
