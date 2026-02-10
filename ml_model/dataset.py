import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
import os

class GeosSubCDataset(Dataset):
    def __init__(self, data_root="dataprocess", start_year=1999, end_year=2016, mjo_file="mjo_processed.csv", transform=None, preload=True):
        """
        Args:
            data_root (str): Root directory containing 'geos_subc_{year}.zarr' and 'gpcp_weekly_{year}.zarr'.
            start_year (int): Start year for data loading.
            end_year (int): End year for data loading.
            mjo_file (str): Filename of MJO CSV in data_root.
            transform (callable, optional): Optional transform to be applied on a sample.
            preload (bool): If True, load all data into RAM (recommended for speed).
        """
        self.data_root = data_root
        self.years = list(range(start_year, end_year + 1))
        self.transform = transform
        self.preload = preload
        
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
        self.preloaded_geos = {} # (S, M) -> ndarray
        self.preloaded_gpcp = {} # S -> ndarray
        
        print(f"Indexing samples from {start_year} to {end_year}...")
        for year in self.years:
            geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
            gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
            
            if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
                continue
                
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
                
                if 'S' not in ds_geos.dims:
                    continue
                
                init_dates = pd.to_datetime(ds_geos['S'].values)
                
                # Check for M dimension
                members = [0]
                if 'M' in ds_geos.dims:
                    members = list(range(ds_geos.sizes['M']))
                
                if self.preload:
                    print(f"  Preloading year {year} into RAM...")
                    # Load the whole year's variables to avoid repeated sel
                    geos_vals = ds_geos['pr'].compute()
                    gpcp_vals = ds_gpcp['precip'].compute()
                
                for s_idx, s_date in enumerate(init_dates):
                    # Cache GPCP truth (same for all members M)
                    if self.preload:
                        self.preloaded_gpcp[s_date] = gpcp_vals.sel(S=s_date).values.astype(np.float32)
                    
                    for m_idx in members:
                        self.samples.append({
                            'year': year,
                            'S': s_date,
                            'M': m_idx,
                            'geos_path': geos_path,
                            'gpcp_path': gpcp_path
                        })
                        
                        if self.preload:
                            # Cache GEOS forecast for this specific member
                            f_val = geos_vals.sel(S=s_date)
                            if 'M' in f_val.dims:
                                f_val = f_val.sel(M=m_idx)
                            self.preloaded_geos[(s_date, m_idx)] = f_val.values.astype(np.float32)
                
                ds_geos.close()
                ds_gpcp.close()
                
            except Exception as e:
                print(f"Error indexing/preloading {year}: {e}")
                
        print(f"Found {len(self.samples)} samples (including ensemble members).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        s_date = meta['S']
        m_idx = meta['M']
        
        try:
            if self.preload:
                x_forecast = self.preloaded_geos[(s_date, m_idx)]
                y_truth = self.preloaded_gpcp[s_date]
            else:
                ds_geos = xr.open_zarr(meta['geos_path'], consolidated=False)
                ds_gpcp = xr.open_zarr(meta['gpcp_path'], consolidated=False)
                
                # Forecast (Input)
                # Select specific member M
                f_data = ds_geos['pr'].sel(S=s_date)
                if 'M' in f_data.dims:
                    f_data = f_data.sel(M=m_idx)
                
                # Ground Truth (Target) - GPCP only has one truth per S
                t_data = ds_gpcp['precip'].sel(S=s_date)
                
                x_forecast = f_data.values.astype(np.float32)
                y_truth = t_data.values.astype(np.float32)
                
                ds_geos.close()
                ds_gpcp.close()
            
            if self.df_mjo is not None and s_date in self.df_mjo.index:
                mjo = self.df_mjo.loc[s_date]
                rmm_vals = np.array([mjo['RMM1_lagged'], mjo['RMM2_lagged']], dtype=np.float32)
            else:
                rmm_vals = np.zeros(2, dtype=np.float32)
            
            x_forecast = np.nan_to_num(x_forecast, nan=0.0)
            y_truth = np.nan_to_num(y_truth, nan=0.0)
            
            return {
                "input_forecast": torch.tensor(x_forecast), 
                "target_truth": torch.tensor(y_truth),      
                "mjo_conditioning": torch.tensor(rmm_vals), 
                "S": str(s_date),
                "M": m_idx
            }
        except Exception as e:
            print(f"Error loading sample {idx} (Date {s_date}, Member {m_idx}): {e}")
            return self.__getitem__((idx + 1) % len(self))
            
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
