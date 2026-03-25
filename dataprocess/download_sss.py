"""
Download raw daily Copernicus Marine SSS NetCDF files for process_sss.py.

This script writes files into the layout expected by process_sss.py:
    dataprocess/SSS/copernicus_sss_data/<year>/**/*.nc

It uses the Copernicus Marine Toolbox Python API:
    - multiyear dataset for historical years
    - near-real-time dataset for newer years (2025+ by default)

Examples:
    python dataprocess/download_sss.py
    python dataprocess/download_sss.py --start_year 2025 --end_year 2025
    python dataprocess/download_sss.py --years 2024 2025
"""

import argparse
import os

try:
    import copernicusmarine
except ImportError as exc:
    raise SystemExit(
        "copernicusmarine is not installed. Install it first, for example:\n"
        "  pip install copernicusmarine"
    ) from exc


MY_DATASET_ID = "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
NRT_DATASET_ID = "cmems_obs-mob_glo_phy-sss_nrt_multi_P1D"
DEFAULT_DATASET_VERSION = "202311"
DEFAULT_OUTPUT_DIR = "dataprocess/SSS/copernicus_sss_data"
DEFAULT_START_YEAR = 1999
DEFAULT_END_YEAR = 2025
DEFAULT_NRT_START_YEAR = 2025


def choose_dataset(year, nrt_start_year, my_dataset_id, nrt_dataset_id):
    if year >= nrt_start_year:
        return nrt_dataset_id, "near-real-time"
    return my_dataset_id, "multiyear"


def download_year(
    year,
    output_dir,
    dataset_version,
    nrt_start_year,
    my_dataset_id,
    nrt_dataset_id,
):
    dataset_id, label = choose_dataset(
        year=year,
        nrt_start_year=nrt_start_year,
        my_dataset_id=my_dataset_id,
        nrt_dataset_id=nrt_dataset_id,
    )
    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    print(f"\n[Year {year}] Using {label} dataset: {dataset_id} (version {dataset_version})")
    print(f"[Year {year}] Output directory: {os.path.abspath(year_dir)}")

    copernicusmarine.get(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        filter=f"*/{year}/*",
        output_directory=year_dir,
    )

    print(f"[Year {year}] Download completed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine SSS daily files for process_sss.py"
    )
    parser.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Optional explicit year list. Overrides start/end year.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Root output directory for raw SSS files.")
    parser.add_argument("--dataset_version", default=DEFAULT_DATASET_VERSION,
                        help="Copernicus dataset version string, e.g. 202311.")
    parser.add_argument("--nrt_start_year", type=int, default=DEFAULT_NRT_START_YEAR,
                        help="Years >= this value use the near-real-time dataset.")
    parser.add_argument("--my_dataset_id", default=MY_DATASET_ID,
                        help="Multiyear Copernicus SSS dataset id.")
    parser.add_argument("--nrt_dataset_id", default=NRT_DATASET_ID,
                        help="Near-real-time Copernicus SSS dataset id.")
    return parser.parse_args()


def main():
    args = parse_args()
    years = args.years if args.years else list(range(args.start_year, args.end_year + 1))
    if not years:
        raise SystemExit("No years requested.")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting Copernicus SSS download")
    print(f"Years: {years}")
    print(f"Output root: {os.path.abspath(args.output_dir)}")
    print(
        "Dataset routing: "
        f"< {args.nrt_start_year} -> {args.my_dataset_id}, "
        f">= {args.nrt_start_year} -> {args.nrt_dataset_id}"
    )

    failures = []
    for year in years:
        try:
            download_year(
                year=year,
                output_dir=args.output_dir,
                dataset_version=args.dataset_version,
                nrt_start_year=args.nrt_start_year,
                my_dataset_id=args.my_dataset_id,
                nrt_dataset_id=args.nrt_dataset_id,
            )
        except Exception as exc:
            failures.append((year, str(exc)))
            print(f"[Year {year}] Download failed: {exc}")

    if failures:
        print("\nCompleted with failures:")
        for year, message in failures:
            print(f"  {year}: {message}")
        raise SystemExit(1)

    print("\nAll requested SSS downloads completed successfully.")


if __name__ == "__main__":
    main()
