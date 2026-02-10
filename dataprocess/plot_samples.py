import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import random

def plot_random_sample(data_dir="dataprocess", output_file="sample_comparison.png"):
    # 1. Select a random year
    years = list(range(1999, 2017))
    year = random.choice(years)
    
    geos_path = f"{data_dir}/geos_subc_{year}.zarr"
    gpcp_path = f"{data_dir}/gpcp_weekly_{year}.zarr"
    
    if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
        print(f"Data for year {year} not found. Trying another...")
        # Simple retry logic or fail
        return

    print(f"Loading data for Year {year}...")
    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
    
    # 2. Select a random initialization date index
    if 'S' not in ds_geos.dims:
        print("Dimension 'S' matching failed.")
        return
        
    n_samples = ds_geos.sizes['S']
    idx = random.randint(0, n_samples - 1)
    init_date = ds_geos['S'].isel(S=idx).values
    
    print(f"Selected Init Date: {init_date} (Index {idx})")
    
    # 3. Extract Data for 4 Weeks
    # GEOS: (S, L, Y, X) -> select S=idx
    # GPCP: (S, L, Y, X) -> select S=idx
    # Variables: 'pr' from GEOS, 'precip' from GPCP
    
    geos_sample = ds_geos['pr'].isel(S=idx) # (4, Y, X)
    gpcp_sample = ds_gpcp['precip'].isel(S=idx) # (4, Y, X)
    
    # Check if GEOS has 'M' dimension (ensemble members) -> Mean it
    if 'M' in geos_sample.dims:
        geos_sample = geos_sample.mean(dim='M')
        
    # Load into memory (convert to numpy) to avoid dask/matplotlib issues
    print("Loading samples into memory...")
    geos_sample = geos_sample.compute()
    gpcp_sample = gpcp_sample.compute()
    
    # 4. Plot
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16), 
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    lead_weeks = [1, 2, 3, 4]
    
    # min/max for colorbar (using global min/max of this sample for comparison)
    vmin = min(geos_sample.min(), gpcp_sample.min())
    vmax = max(geos_sample.max(), gpcp_sample.max())
    
    for i in range(4):
        # GEOS
        ax_geos = axes[i, 0]
        im_geos = ax_geos.pcolormesh(geos_sample.coords['X'], geos_sample.coords['Y'], 
                                     geos_sample.isel(L=i), 
                                     transform=ccrs.PlateCarree(), 
                                     cmap='YlGnBu', vmin=vmin, vmax=vmax)
        ax_geos.coastlines()
        ax_geos.set_title(f"GEOS Fcst (Week {lead_weeks[i]})")
        
        # GPCP
        ax_gpcp = axes[i, 1]
        im_gpcp = ax_gpcp.pcolormesh(gpcp_sample.coords['longitude'], gpcp_sample.coords['latitude'], 
                                     gpcp_sample.isel(L=i), 
                                     transform=ccrs.PlateCarree(), 
                                     cmap='YlGnBu', vmin=vmin, vmax=vmax)
        ax_gpcp.coastlines()
        ax_gpcp.set_title(f"GPCP Obs (Week {lead_weeks[i]})")
        
    # Colorbar
    fig.colorbar(im_geos, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.05, label='Precipitation')
    
    plt.suptitle(f"Weekly Mean Precip Comparison\nInit: {init_date}", fontsize=16)
    
    out_dir = "visualization"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{output_file}"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_random_sample()
