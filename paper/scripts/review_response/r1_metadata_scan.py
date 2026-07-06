#!/usr/bin/env python3
"""Review response M2/M6: print forecast-archive metadata.

Reports, per year zarr: FIMr1p1 member count, generated member count,
number of start dates, init frequency, lead values, and grid shape.

Usage:
  python paper/scripts/review_response/r1_metadata_scan.py \
      [--forecast_dir dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast_dir",
                        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50")
    parser.add_argument("--years", default="2021,2022,2023")
    args = parser.parse_args()

    for year in [int(y) for y in args.years.split(",") if y.strip()]:
        path = Path(args.forecast_dir) / f"{year}.zarr"
        if not path.exists():
            print(f"{year}: MISSING {path}")
            continue
        ds = xr.open_zarr(path, consolidated=False, chunks=None)
        try:
            inits = pd.to_datetime(ds["init"].values)
            diffs = np.diff(inits.values).astype("timedelta64[D]").astype(int) if len(inits) > 1 else []
            print(f"\n=== {year} ({path}) ===")
            print(f"  dims                : {dict(ds.sizes)}")
            print(f"  n start dates       : {len(inits)}")
            print(f"  first / last init   : {inits.min().date()} / {inits.max().date()}")
            if len(diffs):
                vals, counts = np.unique(diffs, return_counts=True)
                print(f"  init spacing (days) : {dict(zip(vals.tolist(), counts.tolist()))}")
            print(f"  lead values         : {ds['lead'].values.tolist()}")
            for dim in ("geos_member", "ensemble"):
                if dim in ds.sizes:
                    print(f"  {dim:<19} : {ds.sizes[dim]}")
            for var in ds.data_vars:
                print(f"  var {var:<15} : dims={ds[var].dims}, dtype={ds[var].dtype}")
        finally:
            ds.close()


if __name__ == "__main__":
    main()
