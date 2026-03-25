"""
Inspect raw SSS NetCDF files and report likely source/product metadata.

This helper is meant to answer questions like:
- Where is the raw SSS data stored locally?
- What product/provider does it appear to come from?
- Which variable/coords are present in the raw files?

It mirrors the raw-file layout expected by ``process_sss.py``:
    dataprocess/SSS/copernicus_sss_data/<year>/**/*.nc

Usage:
    python dataprocess/check_sss_source.py
    python dataprocess/check_sss_source.py --year 2020
    python dataprocess/check_sss_source.py --year 2020 2021 --all-attrs
    python dataprocess/check_sss_source.py --file /path/to/file.nc
"""

import argparse
import glob
import os
from collections import defaultdict

import xarray as xr


DEFAULT_BASE_DIR = "dataprocess/SSS/copernicus_sss_data"
SOURCE_ATTR_KEYS = [
    "title",
    "id",
    "product_id",
    "cmems_product_id",
    "dataset_id",
    "project",
    "institution",
    "institution_name",
    "source",
    "summary",
    "platform",
    "references",
    "history",
    "license",
    "licence",
]
SSS_VAR_CANDIDATES = [
    "sss",
    "sos",
    "SSS",
    "SOS",
    "sea_surface_salinity",
    "ssd",
    "SSD",
    "s_surface",
]


def format_value(value, max_len=220):
    text = " ".join(str(value).split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def gather_files(base_dir, years=None):
    files = []
    if years:
        for year in years:
            pattern = os.path.join(base_dir, str(year), "**", "*.nc")
            files.extend(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(base_dir, "**", "*.nc")
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def detect_sss_var(ds):
    for name in SSS_VAR_CANDIDATES:
        if name in ds.data_vars:
            return name
    for name in ds.data_vars:
        if name not in ds.coords:
            return name
    return None


def detect_time_coord(ds):
    for name in ["time", "TIME", "t"]:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def detect_lat_lon(ds):
    lat_name = None
    lon_name = None
    for name in ["latitude", "lat", "y", "Y"]:
        if name in ds.coords or name in ds.dims:
            lat_name = name
            break
    for name in ["longitude", "lon", "x", "X"]:
        if name in ds.coords or name in ds.dims:
            lon_name = name
            break
    return lat_name, lon_name


def infer_source(attrs, path):
    blob = " ".join(format_value(v, max_len=1000) for v in attrs.values()).lower()
    blob = f"{path.lower()} {blob}"

    clues = []
    if "multiobs_glo_phy_s_surface_mynrt_015_013" in blob:
        clues.append("Copernicus product MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013")
    if "copernicus" in blob:
        clues.append("Copernicus")
    if "cmems" in blob:
        clues.append("CMEMS")
    if "marine.copernicus" in blob:
        clues.append("Copernicus Marine")

    if clues:
        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for clue in clues:
            if clue not in seen:
                deduped.append(clue)
                seen.add(clue)
        return ", ".join(deduped)
    return "No strong source clue found in attrs/path"


def inspect_file(path, show_all_attrs=False):
    print("\n" + "=" * 88)
    print(f"File: {path}")
    print(f"Size: {os.path.getsize(path) / (1024 ** 2):.2f} MB")

    ds = xr.open_dataset(path)
    try:
        print(f"Dims: {dict(ds.sizes)}")
        print(f"Coords: {list(ds.coords)}")
        print(f"Data vars: {list(ds.data_vars)}")

        sss_var = detect_sss_var(ds)
        lat_name, lon_name = detect_lat_lon(ds)
        time_name = detect_time_coord(ds)

        print(f"Detected SSS variable: {sss_var}")
        print(f"Detected coords: time={time_name}, lat={lat_name}, lon={lon_name}")

        if time_name is not None and ds.sizes.get(time_name, 0) > 0:
            time_values = ds[time_name].values
            print(f"Time coverage: {time_values[0]} -> {time_values[-1]} ({len(time_values)} steps)")

        source_attrs = {k: ds.attrs[k] for k in SOURCE_ATTR_KEYS if k in ds.attrs}
        print(f"Inferred source: {infer_source(ds.attrs, path)}")

        if source_attrs:
            print("Key global attrs:")
            for key, value in source_attrs.items():
                print(f"  {key}: {format_value(value)}")
        else:
            print("Key global attrs: none of the common source/product fields were present")

        if show_all_attrs:
            print("All global attrs:")
            for key in sorted(ds.attrs):
                print(f"  {key}: {format_value(ds.attrs[key], max_len=400)}")
    finally:
        ds.close()


def summarize_tree(files, base_dir):
    per_year = defaultdict(int)
    for path in files:
        rel = os.path.relpath(path, base_dir)
        year = rel.split(os.sep, 1)[0]
        per_year[year] += 1

    print(f"Base dir: {os.path.abspath(base_dir)}")
    print(f"Raw NetCDF files found: {len(files)}")
    if per_year:
        print("Files by year:")
        for year in sorted(per_year):
            print(f"  {year}: {per_year[year]}")

    print("Sample file paths:")
    for path in files[:3]:
        print(f"  {path}")
    if len(files) > 3:
        print(f"  ... ({len(files) - 3} more)")


def main():
    parser = argparse.ArgumentParser(description="Inspect raw SSS files and infer their source/product metadata.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help="Root directory containing raw SSS NetCDFs.")
    parser.add_argument("--year", type=int, nargs="+", default=None,
                        help="Restrict the search to one or more years.")
    parser.add_argument("--file", default=None,
                        help="Inspect one specific NetCDF file instead of searching the tree.")
    parser.add_argument("--all_attrs", action="store_true",
                        help="Print all global attrs from the sample file.")
    args = parser.parse_args()

    print("SSS source inspector")
    print(f"Expected raw layout from process_sss.py: {DEFAULT_BASE_DIR}/<year>/**/*.nc")

    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"File not found: {args.file}")
        inspect_file(args.file, show_all_attrs=args.all_attrs)
        return

    files = gather_files(args.base_dir, years=args.year)
    if not files:
        print(f"No raw SSS NetCDF files found under: {os.path.abspath(args.base_dir)}")
        print("This repo expects pre-downloaded Copernicus SSS files there before running process_sss.py.")
        return

    summarize_tree(files, args.base_dir)
    inspect_file(files[0], show_all_attrs=args.all_attrs)


if __name__ == "__main__":
    main()
