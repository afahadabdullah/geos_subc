import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from dataset_hybrid import S2SHybridDataset

def diagnose_rossby_waves(data_root, output_dir="ml_diagnostics"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset (normalize=False to see raw values)
    dataset = S2SHybridDataset(data_root=data_root, start_year=2015, end_year=2020, normalize=False)
    
    # Pick a sample (e.g., middle of the dataset)
    idx = len(dataset) // 2
    sample = dataset[idx]
    
    # x_obs has shape (24, 181, 360). Z500 is channels 16-19.
    z500 = sample['x_obs'][16:20] # (4, 181, 360)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    
    for week in range(4):
        z_raw = z500[week]
        
        # Calculate Zonal Deviation
        zonal_mean = np.mean(z_raw, axis=1, keepdims=True) # Average over Longitude
        z_dev = z_raw - zonal_mean
        
        # Plot Raw Z500
        im1 = axes[week, 0].imshow(z_raw, cmap='viridis')
        axes[week, 0].set_title(f"Week {week+1} Raw Z500")
        fig.colorbar(im1, ax=axes[week, 0], fraction=0.046, pad=0.04)
        
        # Plot Zonal Deviation
        # Use RdBu_r to highlight Rossby Wave ridges/troughs
        im2 = axes[week, 1].imshow(z_dev, cmap='RdBu_r', vmin=-3000, vmax=3000)
        axes[week, 1].set_title(f"Week {week+1} Z500 Zonal Deviation (Rossby Waves)")
        fig.colorbar(im2, ax=axes[week, 1], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    diag_path = os.path.join(output_dir, "rossby_wave_diagnosis.png")
    plt.savefig(diag_path)
    plt.close()
    
    print(f"✅ Rossby wave diagnosis plot saved to {diag_path}")

if __name__ == "__main__":
    data_root = "/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/geos_subc/dataprocess"
    if not os.path.exists(data_root):
        # Fallback for TACC-like paths
        data_root = "/home1/11353/afahad/geos_subc/dataprocess"
    
    diagnose_rossby_waves(data_root)
