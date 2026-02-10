import requests
import os
import time

def download_gpcp_data(start_year=1999, end_year=2016, output_dir="dataprocess/gpcp_data"):
    """
    Downloads GPCP Daily Precipitation data from NOAA GCS bucket.
    URL Pattern: https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/{year}/gpcp_v01r03_daily_d{year}*.nc
    Actually, the user link was: https://console.cloud.google.com/storage/browser/noaa-cdr-precip-gpcp-daily
    Public URL for GCS bucket files:
    https://storage.googleapis.com/noaa-cdr-precip-gpcp-daily/data/{year}/gpcp_v01r03_daily_d{year}{month}{day}_c*.nc
    
    We need to list files or just iterate dates. NetCDF files are usually daily.
    Let's check the filename format carefully.
    
    Format seems to be: gpcp_v01r03_daily_dYYYYMMDD_cYYYYMMDD.nc
    
    Since listing GCS bucket via HTTP without auth is tricky (need XML parsing), 
    and we know the date range, we can try to construct URLs.
    However, the `_cYYYYMMDD` suffix (creation date?) might vary.
    
    Strategy: 
    1. Try to use NCEI direct access which has standard directory listing?
       https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access/
       This is usually easier to scrape or guess.
       
    Let's try the NCEI URL first as it's the official archive and usually matches the GCS content.
    
    """
    base_url = "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-daily/access"
    
    os.makedirs(output_dir, exist_ok=True)
    
    from datetime import date, timedelta
    
    sdate = date(start_year, 1, 1)
    edate = date(end_year, 12, 31)
    
    delta = edate - sdate
    
    print(f"Downloading GPCP data from {sdate} to {edate}...")
    
    # We might need to parse index page to get exact filenames because of the suffix
    # Let's write a helper to get file list for a year
    
    for year in range(start_year, end_year + 1):
        year_url = f"{base_url}/{year}/"
        print(f"Checking {year_url}...")
        
        try:
            response = requests.get(year_url)
            if response.status_code != 200:
                print(f"Failed to access {year_url}")
                continue
                
            # Simple parsing of links ending in .nc
            content = response.text
            # Basic parsing
            import re
            # look for href="gpcp_v01r03_daily_d..."
            files = re.findall(r'href="(gpcp_v01r03_daily_d\d+_c\d+\.nc)"', content)
            
            # Remove duplicates
            files = sorted(list(set(files)))
            
            print(f"Found {len(files)} files for {year}")
            
            for fname in files:
                file_url = f"{year_url}{fname}"
                out_path = os.path.join(output_dir, str(year), fname)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                if os.path.exists(out_path):
                    # verify size? or just skip
                    print(f"Skipping {fname}, exists.")
                    continue
                
                print(f"Downloading {fname}...")
                r = requests.get(file_url, stream=True)
                if r.status_code == 200:
                    with open(out_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    print(f"Error downloading {fname}")
                
        except Exception as e:
            print(f"Error processing {year}: {e}")

if __name__ == "__main__":
    download_gpcp_data()
