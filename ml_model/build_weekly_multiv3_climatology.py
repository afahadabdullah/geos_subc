#!/usr/bin/env python3
"""
Build weekly climatology for the multi-v3 workflow.
"""

import sys

from build_weekly_multiv1_climatology import main as build_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--ml_hindcast_dir", "dataprocess/gen_multiv3_hindcast_1999_2019",
    "--ml_forecast_dir", "dataprocess/gen_multiv3",
    "--output_dir", "dataprocess/clim_v3",
    "--start_year", "1999",
    "--end_year", "2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    build_main()


if __name__ == "__main__":
    main()
