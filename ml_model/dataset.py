import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
from arraylake import Client

class GeosSubCDataset(Dataset):
    def __init__(self, forecast_store_path, obs_group_name=None, mjo_data_path=None, transform=None):
        """
        Args:
            forecast_store_path (str): Path to local Zarr file (e.g., 'dataprocess/geos_subc_2000.zarr')
            obs_group_name (str): ArrayLake group name for observations (e.g., 'era5'). 
                                  If None, will use a placeholder or skip obs loading.
            mjo_data_path (str): Path to MJO indices file (e.g., CSV/NetCDF).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.forecast_store_path = forecast_store_path
        self.obs_group_name = obs_group_name
        self.mjo_data_path = mjo_data_path
        self.transform = transform
        
        # Load Forecast Data (Lazy)
        # We assume the file is local for now, but could be remote arraylake session
        print(f"Loading forecast data from {forecast_store_path}...")
        self.ds_forecast = xr.open_zarr(forecast_store_path, consolidated=False)
        
        # Connect to ArrayLake for Observations (if provided)
        if self.obs_group_name:
            print(f"Connecting to ArrayLake group: {obs_group_name}...")
            client = Client()
            repo = client.get_repo("umd/subc")
            self.obs_session = repo.readonly_session(branch="main")
            self.ds_obs = xr.open_zarr(self.obs_session.store, group=obs_group_name)
        else:
            self.ds_obs = None
            print("Warning: No observation group provided. Dataset will only yield forecasts.")

        # Indexing Strategy: Flatten (S, L) -> (Index)
        # S = Initialization Time, L = Lead Time
        # We need to map integer index -> (s_idx, l_idx) or just (s_idx) if we predict full sequence
        # For diffusion, usually we predict a fields at a specific target time.
        # User goal: "predict subseasonal... using initial obs mean"
        
        # Let's assume we treat every (Init Time S) as a sample, and we predict the sequence L?
        # Or predict specific leads?
        # For simplicity, let's index by Initialization Time 'S'.
        if 'S' in self.ds_forecast.dims:
            self.n_samples = self.ds_forecast.sizes['S']
            self.time_coords = self.ds_forecast['S'].values
        else:
            raise ValueError("Forecast dataset missing 'S' dimension.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 1. Get Forecast for Init Time S[idx]
        # Shape: (M, L, Y, X) -> we might average members M? 
        # User said "dynamical model forecast". Usually ensemble mean or all members.
        # Let's return Ensemble Mean for now to match dimensions with Obs.
        
        s_val = self.time_coords[idx]
        
        # Select data and convert to tensor
        # We load 'pr' and 'tas'
        f_pr = self.ds_forecast['pr'].isel(S=idx).mean(dim='M').values # (L, Y, X)
        f_tas = self.ds_forecast['tas'].isel(S=idx).mean(dim='M').values # (L, Y, X)
        
        # Stack variables: (C, L, Y, X) or (L, C, Y, X)
        # PyTorch Conv2D expects (C, H, W). Time (L) might be treated as batch or depth?
        # For Diffusers UNet2D, usually (Batch, Channels, Height, Width).
        # If we predict the whole sequence, we might stack L into Channels?
        # Or use a 3D UNet? User said "Diffusers model import directly".
        # Standard UNet2D is for images. 
        # We can treat Lead Times * Variables as Channels.
        # 2 Variables * 32 Lead Times = 64 Channels.
        
        x_forecast = np.stack([f_pr, f_tas], axis=0) # (2, L, Y, X)
        
        # 2. Get Initial Observation (State at T=0) (Conditioning)
        # This matches S[idx]
        if self.ds_obs:
            # Logic to find obs at s_val
            # obs_init = self.ds_obs.sel(time=s_val).values
            pass # Placeholder
            
        # 3. Get Truth Target (Obs at T = S + L) (Training Target)
        if self.ds_obs:
            # Logic to find obs for the whole lead time sequence
            pass # Placeholder
            
        # Return dict compatible with HuggingFace
        return {
            "pixel_values": torch.tensor(x_forecast, dtype=torch.float32), # Placeholder mapping
            # "conditioning": ...
        }

if __name__ == "__main__":
    # verification
    ds = GeosSubCDataset("dataprocess/geos_subc_2000.zarr")
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    print(f"Sample shape: {sample['pixel_values'].shape}")
