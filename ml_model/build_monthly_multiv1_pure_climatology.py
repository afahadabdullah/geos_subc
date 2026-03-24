#!/usr/bin/env python3
"""
Build monthly climatology for the pure-noise multiv1 workflow.

This combines:
- 1999-2019 pure-noise hindcasts
- 2020-2021 pure-noise forecasts

and writes monthly lead-specific climatology stores under dataprocess/clim_pure.
"""

import sys

from build_monthly_multiv1_climatology import main as build_main


DEFAULT_ARGS = [
    "--ml_hindcast_dir", "dataprocess/gen_multiv1_pure_1999_2019",
    "--ml_forecast_dir", "dataprocess/gen_multiv1_pure_2020_2021",
    "--output_dir", "dataprocess/clim_pure",
    "--start_year", "1999",
    "--end_year", "2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    build_main()


if __name__ == "__main__":
    main()
