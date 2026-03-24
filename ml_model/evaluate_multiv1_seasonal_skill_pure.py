#!/usr/bin/env python3
"""
Evaluate seasonal held-out skill using pure-noise multiv1 forecasts.

This is a thin wrapper around evaluate_multiv1_seasonal_skill.py that points
the ML forecast and climatology paths at the pure-noise outputs by default.
"""

import sys

from evaluate_multiv1_seasonal_skill import main as seasonal_main


DEFAULT_ARGS = [
    "--ml_dir",
    "dataprocess/gen_multiv1_pure_2020_2021",
    "--start_year",
    "2020",
    "--end_year",
    "2021",
    "--threshold_start_year",
    "1999",
    "--threshold_end_year",
    "2019",
    "--seasons",
    "DJF",
    "MAM",
    "JJA",
    "SON",
    "--variables",
    "tas",
    "pr",
    "--sample_chunk_size",
    "2",
    "--obs_clim_path",
    "dataprocess/clim_pure/obs_weekly_clim_1999_2021.zarr",
    "--output_dir",
    "ml_output_flowmulti/seasonal_skill_pure_2020_2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    seasonal_main()


if __name__ == "__main__":
    main()
