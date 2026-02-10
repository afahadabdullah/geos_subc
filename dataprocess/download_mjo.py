import pandas as pd
import requests
import numpy as np
import xarray as xr
import os
import glob
from tqdm import tqdm
from io import StringIO

def download_and_process_mjo(start_year=1999, end_year=2016, output_dir="dataprocess"):
    # 1. Download RMM Data
    url = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
    print(f"Downloading RMM data from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data_str = response.text
    except Exception as e:
        print(f"Failed to download MJO data: {e}")
        return

    # 2. Parse Data
    # The file has a header (lines usually start with comments or are just space separated)
    # Format: year month day RMM1 RMM2 phase amplitude source
    # Skip first 2 lines typically
    
    # We can use pd.read_csv with delim_whitespace
    # Inspect first few lines?
    
    # Let's try skipping header rows. Usually row 0 and 1 are header info.
    cols = ["year", "month", "day", "RMM1", "RMM2", "phase", "amplitude", "source"]
    
    try:
        df = pd.read_csv(StringIO(data_str), skiprows=2, delim_whitespace=True, names=cols, on_bad_lines='skip')
    except Exception as e:
        print(f"Error parsing MJO data: {e}")
        return

    # Create Date Column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.set_index('date').sort_index()
    
    print(f"Loaded RMM data from {df.index.min()} to {df.index.max()}")
    
    # 3. Process for GEOS Initialization Dates
    print(f"Processing lagged means for GEOS init dates ({start_year}-{end_year})...")
    
    mjo_features = []
    
    for year in range(start_year, end_year + 1):
        geos_path = f"{output_dir}/geos_subc_{year}.zarr"
        if not os.path.exists(geos_path):
            print(f"GEOS file {geos_path} not found. Skipping.")
            continue
            
        ds = xr.open_zarr(geos_path, consolidated=False)
        if 'S' not in ds.dims:
            continue
            
        init_dates = pd.to_datetime(ds['S'].values)
        
        for init_date in init_dates:
            # Defined Window: [Init - 28 days, Init - 1 day] (Past 4 weeks)
            end_date = init_date - pd.Timedelta(days=1)
            start_date = init_date - pd.Timedelta(days=28)
            
            # Slice RMM
            # 1e36 is missing value in BOM data usually? check numeric
            # Usually BOM missing is 999 or similar. Pandas might handle if standard.
            # But let's check
            
            mask = (df.index >= start_date) & (df.index <= end_date)
            subset = df.loc[mask]
            
            if len(subset) < 20: # Allow some missing days but not too many
                # print(f"Warning: insufficient MJO data for {init_date}")
                rmm1_mean = np.nan
                rmm2_mean = np.nan
            else:
                rmm1_mean = subset['RMM1'].mean()
                rmm2_mean = subset['RMM2'].mean()
            
            mjo_features.append({
                'S': init_date,
                'RMM1_lagged': rmm1_mean,
                'RMM2_lagged': rmm2_mean
            })
            
    # 4. Save to CSV
    if not mjo_features:
        print("No features extracted.")
        return
        
    df_out = pd.DataFrame(mjo_features)
    out_path = f"{output_dir}/mjo_processed.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved processed MJO features to {out_path}")
    
    # Preview
    print(df_out.head())

if __name__ == "__main__":
    download_and_process_mjo()
