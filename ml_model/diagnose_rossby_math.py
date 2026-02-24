import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from dataset_hybrid import S2SHybridDataset

def diagnose_zonal_visuals(data_root, output_dir="ml_diagnostics"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset (normalize=False to see physical values)
    dataset = S2SHybridDataset(data_root=data_root, start_year=2015, end_year=2020, normalize=False)
    
    # Pick a sample
    idx = 0
    sample = dataset[idx]
    
    # x_obs has shape (24, 181, 360)
    # Re-calculate Absolute Z500 from the same logic for validation
    # (In V5.1, we swapped raw Z500 out, so we reconstruct it here for the plot)
    
    # Load raw Z500 directly from Zarr for this sample to show ABSOLUTE Z500
    meta = dataset.samples[idx]
    ds_zu = xr.open_zarr(meta["z500u250_path"], consolidated=False)
    z_var = next((c for c in ['z500', 'z', 'geopotential'] if c in ds_zu), None)
    z_abs = ds_zu[z_var].isel(S=meta['s_idx']).values # (4, 360, 181)
    # Match dataset_hybrid transpose
    if z_abs.ndim == 3 and z_abs.shape[1] == 360:
        z_abs = np.transpose(z_abs, (0, 2, 1)) # (4, 181, 360)
    ds_zu.close()

    # Get the Zonal Deviation we stored in the dataset
    # In V5.1, indices 16-19 are Zonal Deviation
    z_dev = sample['x_obs'][16:20].numpy() # (4, 181, 360)
    
    # Calculate Zonal Mean manually from z_abs
    z_mean = np.mean(z_abs, axis=2, keepdims=True) # (4, 181, 1)
    z_mean_2d = np.repeat(z_mean, 360, axis=2) # Broadcast for plotting (4, 181, 360)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    for week in range(4):
        # 1. Absolute Z500
        im1 = axes[week, 0].imshow(z_abs[week], cmap='magma')
        axes[week, 0].set_title(f"Week {week+1} Absolute Z500")
        fig.colorbar(im1, ax=axes[week, 0])
        
        # 2. Zonal Mean Profile (as 2D field)
        im2 = axes[week, 1].imshow(z_mean_2d[week], cmap='magma')
        axes[week, 1].set_title(f"Week {week+1} Zonal Mean $[Z]$")
        fig.colorbar(im2, ax=axes[week, 1])
        
        # 3. Zonal Deviation (Rossby Waves)
        # Use RdBu_r to highlight ridges/troughs
        im3 = axes[week, 2].imshow(z_dev[week], cmap='RdBu_r', vmin=-3000, vmax=3000)
        axes[week, 2].set_title(f"Week {week+1} Zonal Deviation $Z - [Z]$")
        fig.colorbar(im3, ax=axes[week, 2])
        
    plt.tight_layout()
    diag_path = os.path.join(output_dir, "zonal_dev_math_check.png")
    plt.savefig(diag_path)
    plt.close()
    
    print(f"✅ Rossby wave math check plot saved to {diag_path}")
    print(f"Stats for Week 1:")
    print(f"  Absolute Z500: Min {np.min(z_abs[0]):.2f} | Max {np.max(z_abs[0]):.2f}")
    print(f"  Zonal Mean:    Min {np.min(z_mean[0]):.2f} | Max {np.max(z_mean[0]):.2f}")
    print(f"  Zonal Dev:     Min {np.min(z_dev[0]):.2f} | Max {np.max(z_dev[0]):.2f}")

if __name__ == "__main__":
    data_root = "/scratch/11353/afahad/geossub/geos_subc/dataprocess"
    if not os.path.exists(data_root):
        data_root = "/Users/afahad/Library/CloudStorage/OneDrive-GeorgeMasonUniversity/MacMini/Projects/geos_subc/dataprocess"
    
    diagnose_zonal_visuals(data_root)
