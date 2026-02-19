"""
Calculate IVT Statistics (Mean, Std)
=====================================
Iterates through IVT weekly Zarr files to compute global mean and std
using Welford's online algorithm for numerical stability.

Output: ml_model/ivt_stats.json  ->  { "ivt_mean": float, "ivt_std": float }

Usage:
    python ml_model/calculate_stats_ivt.py
    python ml_model/calculate_stats_ivt.py --data_root dataprocess --start_year 1999 --end_year 2016
"""
import xarray as xr
import numpy as np
import os
import json
import argparse

DATA_ROOT = "dataprocess"
START_YEAR = 1999
END_YEAR = 2016
OUTPUT_FILE = "ml_model/ivt_stats.json"


def calculate_stats(data_root=DATA_ROOT, start_year=START_YEAR, end_year=END_YEAR,
                    output_file=OUTPUT_FILE):
    """
    Compute global mean and std of IVT (kg/m/s) across all training years
    using Welford's online algorithm for numerical stability.
    """
    count = 0
    mean = 0.0
    m2 = 0.0

    print(f"Calculating IVT Stats for {start_year}-{end_year}...")

    for year in range(start_year, end_year):
        ivt_path = os.path.join(data_root, f"ivt_weekly_{year}.zarr")

        if not os.path.exists(ivt_path):
            print(f"  Skipping {year}: {ivt_path} not found")
            continue

        try:
            ds = xr.open_zarr(ivt_path, consolidated=False)

            # Variable name check
            var_name = None
            for candidate in ['ivt', 'IVT', 'ivt_mag']:
                if candidate in ds:
                    var_name = candidate
                    break

            if var_name is None:
                print(f"  Skipping {year}: 'ivt' not found. Available: {list(ds.data_vars)}")
                ds.close()
                continue

            data = ds[var_name].values  # (S, L, H, W)
            ds.close()

            # Flatten and remove NaNs
            data_flat = data.flatten().astype(np.float64)
            data_flat = data_flat[~np.isnan(data_flat)]
            data_flat = data_flat[data_flat >= 0.0]  # IVT is non-negative

            if len(data_flat) == 0:
                print(f"  Skipping {year}: all NaN")
                continue

            # Welford's batch update
            n = len(data_flat)
            delta = data_flat - mean
            mean += delta.sum() / (count + n)
            m2 += (delta * (data_flat - mean)).sum()
            count += n

            print(f"  Processed {year}: {n:,} values. Running mean: {mean:.2f} kg/m/s")

        except Exception as e:
            print(f"  Error processing {year}: {e}")

    if count < 2:
        print("Insufficient data to compute stats.")
        return

    final_mean = float(mean)
    final_std = float(np.sqrt(m2 / (count - 1)))

    print(f"\nFinal IVT Stats:")
    print(f"  Mean: {final_mean:.4f} kg/m/s")
    print(f"  Std:  {final_std:.4f} kg/m/s")
    print(f"  N:    {count:,} values")

    stats = {
        'ivt_mean': final_mean,
        'ivt_std': final_std,
        'train_years': [start_year, end_year - 1],
        'n_values': count
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"  Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IVT mean/std from weekly Zarr files")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--start_year", type=int, default=START_YEAR)
    parser.add_argument("--end_year", type=int, default=END_YEAR)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    calculate_stats(
        data_root=args.data_root,
        start_year=args.start_year,
        end_year=args.end_year,
        output_file=args.output_file
    )
