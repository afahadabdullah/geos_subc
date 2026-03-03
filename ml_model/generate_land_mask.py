"""
Generate a static land-sea mask from SSS data (NaN over land) and save with diagnostic plot.
Land pixels get 1.5x weight, Ocean pixels get 1.0x weight in training loss.
"""
import os
import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def main():
    # Find an SSS zarr file
    data_root = os.environ.get("DATA_ROOT", "dataprocess")
    sss_path = None
    for year in range(2020, 1998, -1):
        p = os.path.join(data_root, f"sss_weekly_{year}.zarr")
        if os.path.exists(p):
            sss_path = p
            break
        p = os.path.join(data_root, "sss", f"{year}.zarr")
        if os.path.exists(p):
            sss_path = p
            break
    
    if sss_path is None:
        raise FileNotFoundError(f"No SSS zarr found in {data_root}")
    
    print(f"Loading SSS from: {sss_path}")
    ds = xr.open_zarr(sss_path, consolidated=False)
    
    # Find the SSS variable name
    var_name = next((c for c in ['sss', 'SSS', 'sos', 'SOS', 'sea_surface_salinity', 's_surface'] if c in ds), None)
    if var_name is None:
        raise ValueError(f"No SSS variable found in {sss_path}. Variables: {list(ds.data_vars)}")
    
    # Read first time step
    sss_raw = ds[var_name].isel(S=0).values  # [L, H, W] or [H, W]
    sss_raw = np.squeeze(sss_raw)
    if sss_raw.ndim == 3:
        sss_raw = sss_raw[0]  # Take first lead week [H, W]
    
    print(f"SSS shape: {sss_raw.shape}")
    print(f"SSS NaN count: {np.isnan(sss_raw).sum()} / {sss_raw.size}")
    
    # Land = where SSS is NaN (no ocean salinity data)
    land_mask = np.isnan(sss_raw)  # [H, W] boolean: True=Land, False=Ocean
    
    print(f"Land pixels: {land_mask.sum()} ({100*land_mask.sum()/land_mask.size:.1f}%)")
    print(f"Ocean pixels: {(~land_mask).sum()} ({100*(~land_mask).sum()/land_mask.size:.1f}%)")
    
    # Create weighting tensor: 1.5 for land, 1.0 for ocean
    # This gives approximately 60/40 relative emphasis (1.5 / (1.5+1.0) = 60%)
    land_ocean_weights = np.where(land_mask, 1.5, 1.0).astype(np.float32)
    
    # Save as .pt
    output_path = os.path.join(os.path.dirname(__file__), "land_sea_mask.pt")
    torch.save({
        'land_mask': torch.from_numpy(land_mask),           # [H, W] bool
        'land_ocean_weights': torch.from_numpy(land_ocean_weights),  # [H, W] float32
        'land_weight': 1.5,
        'ocean_weight': 1.0,
    }, output_path)
    print(f"\n✅ Saved land-sea mask to {output_path}")
    
    # --- Diagnostic Plot ---
    # Try to get lat/lon from a GEOS zarr for proper extent
    lats = np.linspace(-90, 90, land_mask.shape[0])
    lons = np.linspace(0, 360, land_mask.shape[1])
    
    for year in range(2020, 1998, -1):
        geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
        if os.path.exists(geos_path):
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                lats = ds_geos.Y.values
                lons = ds_geos.X.values
                ds_geos.close()
            except:
                pass
            break
    
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    proj = ccrs.PlateCarree()
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 7), subplot_kw={'projection': proj})
    
    # Panel 1: Raw SSS (showing NaN=white over land)
    im0 = axes[0].imshow(np.where(np.isnan(sss_raw), np.nan, sss_raw), 
                          cmap='viridis', origin='lower', extent=extent, 
                          transform=ccrs.PlateCarree())
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Raw SSS (White = NaN = Land)", fontsize=14)
    axes[0].add_feature(cfeature.COASTLINE, linewidth=0.8)
    axes[0].add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    
    # Panel 2: Binary Land Mask
    im1 = axes[1].imshow(land_mask.astype(float), cmap='RdYlGn_r', vmin=0, vmax=1,
                          origin='lower', extent=extent, transform=ccrs.PlateCarree())
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Land Mask (Red=Land, Green=Ocean)", fontsize=14)
    axes[1].add_feature(cfeature.COASTLINE, linewidth=0.8, color='blue')
    
    # Panel 3: Land/Ocean Loss Weights
    im2 = axes[2].imshow(land_ocean_weights, cmap='coolwarm', vmin=0.8, vmax=1.7,
                          origin='lower', extent=extent, transform=ccrs.PlateCarree())
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Loss Weights (Land=1.5, Ocean=1.0)", fontsize=14)
    axes[2].add_feature(cfeature.COASTLINE, linewidth=0.8)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "land_sea_mask_diagnostic.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"📊 Saved diagnostic plot to {plot_path}")

if __name__ == "__main__":
    main()
