import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
import os

class GeosSubCDataset(Dataset):
    def __init__(self, data_root="dataprocess", start_year=1999, end_year=2016, mjo_file="mjo_processed.csv", transform=None):
        """
        Args:
            data_root (str): Root directory containing 'geos_subc_{year}.zarr' and 'gpcp_weekly_{year}.zarr'.
            start_year (int): Start year for data loading.
            end_year (int): End year for data loading.
            mjo_file (str): Filename of MJO CSV in data_root.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        self.years = list(range(start_year, end_year + 1))
        self.transform = transform
        
        # 1. Load MJO Data
        mjo_path = os.path.join(data_root, mjo_file)
        if os.path.exists(mjo_path):
            print(f"Loading MJO features from {mjo_path}...")
            self.df_mjo = pd.read_csv(mjo_path)
            self.df_mjo['S'] = pd.to_datetime(self.df_mjo['S'])
            self.df_mjo = self.df_mjo.set_index('S')
        else:
            print(f"Warning: MJO file not found at {mjo_path}. MJO features will be zeros.")
            self.df_mjo = None

        # 2. Index all available samples (Year, InitDate)
        # We need to scan files to build a global index map: idx -> (year, init_date, file_paths)
        # This might be slow if we open every Zarr.
        # Faster: Assume file existence and standard structure?
        # Safer: Open each year once to get 'S' coords.
        
        self.samples = []
        
        print(f"Indexing samples from {start_year} to {end_year}...")
        for year in self.years:
            geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
            gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
            
            if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
                # print(f"Skipping {year}: Missing GEOS or GPCP file.")
                continue
                
            # Open lazily to get S coords
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                if 'S' not in ds_geos.dims:
                    continue
                
                # Check GPCP S coords too? Or assume alignment (we processed them to align).
                # Let's trust alignment for speed, or check once.
                
                init_dates = pd.to_datetime(ds_geos['S'].values)
                
                for s_date in init_dates:
                    self.samples.append({
                        'year': year,
                        'S': s_date,
                        'geos_path': geos_path,
                        'gpcp_path': gpcp_path
                    })
            except Exception as e:
                print(f"Error indexing {year}: {e}")
                
        print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Metadata
        meta = self.samples[idx]
        year = meta['year']
        s_date = meta['S']
        
        # 2. Load Data (Lazy Open inside getitem might be slow? Xarray caches?)
        # Better: Keep dataset handles open? But too many files.
        # Xarray open_zarr is relatively cheap if metadata is consolidated, but here likely not.
        # Dask might warn.
        # Optimization: We could use a cache or open once in __init__ if logical.
        # For now, open on demand (might need optimization later).
        
        ds_geos = xr.open_zarr(meta['geos_path'], consolidated=False)
        ds_gpcp = xr.open_zarr(meta['gpcp_path'], consolidated=False)
        
        # 3. Select Time Slice
        # GEOS: (S, L, Y, X)
        # GPCP: (S, L, Y, X)
        
        # Select S
        # Xarray selection by value is reliable
        try:
            # We assume dimensions are (S, L, Y, X)
            # GEOS 'pr' (mm/day)
            # GPCP 'precip' (mm/day)
            
            # Forecast (Input)
            # We take mean over 'M' if exists (ensemble)
            f_data = ds_geos['pr'].sel(S=s_date)
            if 'M' in f_data.dims:
                f_data = f_data.mean(dim='M')
            
            # Ground Truth (Target)
            t_data = ds_gpcp['precip'].sel(S=s_date)
            
            # Convert to Numpy
            # Shape: (4, Y, X) -> (Lead, Lat, Lon)
            x_forecast = f_data.values.astype(np.float32)
            y_truth = t_data.values.astype(np.float32)
            
            # 4. MJO Features (Conditioning)
            if self.df_mjo is not None and s_date in self.df_mjo.index:
                mjo = self.df_mjo.loc[s_date]
                rmm_vals = np.array([mjo['RMM1_lagged'], mjo['RMM2_lagged']], dtype=np.float32)
            else:
                rmm_vals = np.zeros(2, dtype=np.float32)
            
            # 5. Missing Values / NaNs
            # Replace NaNs with 0 or mean?
            # Precipitation shouldn't have NaNs ideally if processed correctly.
            # But let's be safe.
            x_forecast = np.nan_to_num(x_forecast, nan=0.0)
            y_truth = np.nan_to_num(y_truth, nan=0.0)
            
            # Verify shapes match
            # Expected (4, Y, X)
            if x_forecast.shape != y_truth.shape:
                # Resize or align? 
                # They should match from our processing scripts.
                pass
            
            # Return Dictionary
            # standard diffusers image pipeline expects "pixel_values"
            # Here we have text/vector conditioning (MJO) + image input (Forecast) -> Target (Truth)
            # Actually, standard diffusion trains to predict NOISE added to TRUTH.
            # Condition is Forecast.
            
            return {
                "input_forecast": torch.tensor(x_forecast), # (4, 181, 360)
                "target_truth": torch.tensor(y_truth),      # (4, 181, 360)
                "mjo_conditioning": torch.tensor(rmm_vals), # (2,)
                "S": str(s_date)
            }
            
        except Exception as e:
            print(f"Error loading sample {idx} (Date {s_date}): {e}")
            # Return dummy or fail
            return self.__getitem__((idx + 1) % len(self))

if __name__ == "__main__":
    # verification
    ds = GeosSubCDataset(data_root="dataprocess", start_year=2000, end_year=2001)
    print(f"Sample 0: {ds[0]['S']}")
    print(f"Forecast S: {ds[0]['input_forecast'].shape}")
    print(f"Truth S: {ds[0]['target_truth'].shape}")
    print(f"MJO: {ds[0]['mjo_conditioning']}")
