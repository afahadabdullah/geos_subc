import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset import GeosSubCDataset
from ml_model.utils import denormalize

def diagnose():
    print("--- Data Pipeline Diagnostic ---")
    data_root = "dataprocess"
    
    # 1. Load Dataset
    print(f"Loading dataset from {data_root}...")
    try:
        ds = GeosSubCDataset(
            data_root=data_root, 
            start_year=2000, 
            end_year=2005, 
            mjo_file="mjo_processed.csv",
            preload=True,
            ocean_vars=True 
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Check Norm Stats
    print(f"\n[Normalization Stats]")
    print(f"  Log1p Min: {ds.norm_min:.4f}")
    print(f"  Log1p Max: {ds.norm_max:.4f}")
    
    range_val = ds.norm_max - ds.norm_min
    print(f"  Range: {range_val:.4f}")
    
    if range_val > 15:
        print("  WARNING: Range > 15 suggests extreme outliers. Data might be compressed.")
    
    # 3. Correlation Check
    print(f"\n[Alignment Check]")
    print("Computing correlation between Input (GEOS) and Target (GPCP) across 100 random samples...")
    
    indices = np.random.choice(len(ds), 100, replace=False)
    correlations = []
    
    for idx in tqdm(indices):
        sample = ds[idx]
        geos_norm = sample['input_forecast'].numpy()  # (4, Y, X)
        gpcp_norm = sample['target_truth'].numpy()    # (4, Y, X)
        
        # Flatten and compute correlation
        flat_geos = geos_norm.flatten()
        flat_gpcp = gpcp_norm.flatten()
        
        # Pearson Correlation
        if np.std(flat_geos) > 1e-6 and np.std(flat_gpcp) > 1e-6:
            corr = np.corrcoef(flat_geos, flat_gpcp)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)
            
    avg_corr = np.mean(correlations)
    print(f"\nAverage Pixel-wise Correlation: {avg_corr:.4f}")
    
    if avg_corr < 0.3:
        print("  CRITICAL WARNING: Correlation < 0.3. Data likely MISALIGNED or NOISE.")
    elif avg_corr < 0.5:
        print("  WARNING: Correlation low (0.3-0.5). Alignment might be off by weeks.")
    else:
        print("  OK: Correlation > 0.5. Alignment looks plausible.")

    # 4. Save Plots
    print(f"\n[Visual Inspection]")
    sample_idx = indices[0]
    sample = ds[sample_idx]
    
    geos_norm = sample['input_forecast']
    gpcp_norm = sample['target_truth']
    
    geos_raw = denormalize(geos_norm).numpy()
    gpcp_raw = denormalize(gpcp_norm).numpy()
    
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for w in range(4):
        # Input
        ax[0, w].imshow(geos_raw[w], cmap='YlGnBu', origin='upper')
        ax[0, w].set_title(f"Week {w+1} Input (GEOS)\nMean: {geos_raw[w].mean():.2f}")
        ax[0, w].axis('off')
        
        # Target
        ax[1, w].imshow(gpcp_raw[w], cmap='YlGnBu', origin='upper')
        ax[1, w].set_title(f"Week {w+1} Target (GPCP)\nMean: {gpcp_raw[w].mean():.2f}")
        ax[1, w].axis('off')
    
    plot_path = "ml_output/diagnostic_plot.png"
    os.makedirs("ml_output", exist_ok=True)
    plt.savefig(plot_path)
    print(f"Saved diagnostic sample plot to {plot_path}")

if __name__ == "__main__":
    diagnose()
