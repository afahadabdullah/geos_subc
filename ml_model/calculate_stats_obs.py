"""
Calculate Observed Variable Statistics (Mean, Std)
===================================================
Computes global z-score stats (mean, std) for SST, SSS, and Soil Moisture
from their respective weekly Zarr files using Welford's online algorithm.

Output: ml_model/obs_stats.json
  {
    "sst_mean": float,  "sst_std": float,
    "sss_mean": float,  "sss_std": float,
    "sm_mean":  float,  "sm_std":  float,
    "ivt_mean": float,  "ivt_std": float   (if ivt_weekly files exist)
  }

Note: GPCP (prev) is intentionally excluded — it uses simple /10 scaling
      because it is the same physical quantity as the target.

Usage:
    python ml_model/calculate_stats_obs.py
    python ml_model/calculate_stats_obs.py --start_year 1999 --end_year 2016
"""
import xarray as xr
import numpy as np
import os
import json
import argparse

DATA_ROOT = "dataprocess"
START_YEAR = 1999
END_YEAR = 2016
OUTPUT_FILE = "ml_model/obs_stats.json"


class WelfordAccumulator:
    """Online mean/std using Welford's algorithm (numerically stable)."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, data: np.ndarray):
        """Update with a batch of values (1D float64 array, NaNs already removed)."""
        n = len(data)
        if n == 0:
            return
        data = data.astype(np.float64)
        delta = data - self.mean
        self.mean += delta.sum() / (self.count + n)
        self.m2 += (delta * (data - self.mean)).sum()
        self.count += n
        self.min_val = min(self.min_val, float(data.min()))
        self.max_val = max(self.max_val, float(data.max()))

    def result(self):
        if self.count < 2:
            return None, None, None, None
        return float(self.mean), float(np.sqrt(self.m2 / (self.count - 1))), self.min_val, self.max_val


def process_variable(name, zarr_pattern, var_candidates, data_root, start_year, end_year,
                     filter_fn=None):
    """
    Iterate over yearly Zarr files and accumulate stats for one variable.

    Args:
        name:           Human-readable name for logging.
        zarr_pattern:   f-string pattern, e.g. "sst_weekly_{year}.zarr"
        var_candidates: List of possible variable names inside the Zarr.
        filter_fn:      Optional lambda to filter values (e.g. remove negatives).
    """
    acc = WelfordAccumulator()
    print(f"\n  [{name}]")

    for year in range(start_year, end_year):
        path = os.path.join(data_root, zarr_pattern.format(year=year))
        if not os.path.exists(path):
            print(f"    Skipping {year}: {path} not found")
            continue

        try:
            ds = xr.open_zarr(path, consolidated=False)
            var_name = next((c for c in var_candidates if c in ds), None)
            if var_name is None:
                print(f"    Skipping {year}: none of {var_candidates} found. "
                      f"Available: {list(ds.data_vars)}")
                ds.close()
                continue

            data = ds[var_name].values.flatten().astype(np.float64)
            ds.close()

            # Remove NaNs
            data = data[~np.isnan(data)]
            if len(data) == 0:
                print(f"    Skipping {year}: all NaN")
                continue
                
            # Debug Stats range
            dmin, dmax = data.min(), data.max()
            # print(f"    {year} range: {dmin:.2f} to {dmax:.2f}")

            if filter_fn is not None:
                # Filter outliers / fill values
                mask = filter_fn(data)
                n_kept = mask.sum()
                if n_kept == 0:
                    print(f"    Skipping {year} (Range {dmin:.1f} to {dmax:.1f}): all values filtered out!")
                    continue
                data = data[mask]

            acc.update(data)
            print(f"    {year}: {len(data):,} values (Range {dmin:.1f}–{dmax:.1f}) val mean={data.mean():.2f}")

        except Exception as e:
            print(f"    Error {year}: {e}")

    mean, std, vmin, vmax = acc.result()
    if mean is None:
        print(f"    WARNING: No valid data found for {name}!")
    else:
        print(f"    Final → min={vmin:.4f} max={vmax:.4f} mean={mean:.4f}  std={std:.4f}  N={acc.count:,}")
    return mean, std, vmin, vmax


def calculate_stats(data_root=DATA_ROOT, start_year=START_YEAR, end_year=END_YEAR,
                    output_file=OUTPUT_FILE):

    print(f"Computing obs z-score stats for years {start_year}–{end_year - 1}...")

    stats = {}

    # SST (K or C) — check valid ranges for both
    # Kelvin: ~270-310 | Celsius: ~-2 to 35
    sst_mean, sst_std, sst_min, sst_max = process_variable(
        "SST", "sst_weekly_{year}.zarr", ["sst", "SST", "analysed_sst"],
        data_root, start_year, end_year,
        filter_fn=lambda x: ((x > 150) & (x < 350)) | ((x > -5) & (x < 50))
    )
    if sst_mean is not None:
        stats["sst_mean"] = sst_mean
        stats["sst_std"] = sst_std
        stats["sst_min"] = sst_min
        stats["sst_max"] = sst_max

    # SSS (psu)  — valid range ~20–40 psu
    sss_mean, sss_std, sss_min, sss_max = process_variable(
        "SSS", "sss_weekly_{year}.zarr", ["sss", "SSS", "so"],
        data_root, start_year, end_year,
        filter_fn=lambda x: (x > 0) & (x < 50)
    )
    if sss_mean is not None:
        stats["sss_mean"] = sss_mean
        stats["sss_std"] = sss_std
        stats["sss_min"] = sss_min
        stats["sss_max"] = sss_max

    # Soil Moisture (m³/m³)  — valid range 0–0.7
    sm_mean, sm_std, sm_min, sm_max = process_variable(
        "SM", "soilw_weekly_{year}.zarr", ["sm", "soil_moisture", "soilw", "swvl1", "var40"],
        data_root, start_year, end_year,
        filter_fn=lambda x: (x >= 0) & (x <= 1.0)
    )
    if sm_mean is not None:
        stats["sm_mean"] = sm_mean
        stats["sm_std"] = sm_std
        stats["sm_min"] = sm_min
        stats["sm_max"] = sm_max

    # IVT (kg/m/s)  — non-negative
    ivt_mean, ivt_std, ivt_min, ivt_max = process_variable(
        "IVT", "ivt_weekly_{year}.zarr", ["ivt", "IVT", "ivt_mag"],
        data_root, start_year, end_year,
        filter_fn=lambda x: x >= 0
    )
    if ivt_mean is not None:
        stats["ivt_mean"] = ivt_mean
        stats["ivt_std"] = ivt_std
        stats["ivt_min"] = ivt_min
        stats["ivt_max"] = ivt_max

    # Z500 (m²/s² geopotential at 500hPa)  — typical range ~45000–58000 m²/s²
    z500_mean, z500_std, z500_min, z500_max = process_variable(
        "Z500", "z500_u250_weekly_{year}.zarr", ["z500", "z", "geopotential"],
        data_root, start_year, end_year,
        filter_fn=lambda x: (x > 30000) & (x < 70000)
    )
    if z500_mean is not None:
        stats["z500_mean"] = z500_mean
        stats["z500_std"] = z500_std
        stats["z500_min"] = z500_min
        stats["z500_max"] = z500_max

    # U250 (m/s zonal wind at 250hPa)  — can be negative (easterly)
    u250_mean, u250_std, u250_min, u250_max = process_variable(
        "U250", "z500_u250_weekly_{year}.zarr", ["u250", "u", "u_component_of_wind"],
        data_root, start_year, end_year,
        filter_fn=lambda x: (x > -100) & (x < 150)
    )
    if u250_mean is not None:
        stats["u250_mean"] = u250_mean
        stats["u250_std"] = u250_std
        stats["u250_min"] = u250_min
        stats["u250_max"] = u250_max

    if not stats:
        print("\nNo stats computed — check that Zarr files exist in data_root.")
        return

    stats["train_years"] = [start_year, end_year - 1]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nSaved to {output_file}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in stats.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute obs z-score stats")
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--start_year",  type=int, default=START_YEAR)
    parser.add_argument("--end_year",    type=int, default=END_YEAR)
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    calculate_stats(args.data_root, args.start_year, args.end_year, args.output_file)
