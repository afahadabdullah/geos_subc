#!/usr/bin/env python3
"""
Evaluate held-out seasonal skill for the multi-v3 workflow.
"""

import sys

from evaluate_multiv1_seasonal_skill import main as seasonal_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--ml_dir", "dataprocess/gen_multiv3",
    "--start_year", "2020",
    "--end_year", "2021",
    "--threshold_start_year", "1999",
    "--threshold_end_year", "2019",
    "--seasons", "DJF", "MAM", "JJA", "SON",
    "--variables", "tas", "pr",
    "--sample_chunk_size", "2",
    "--obs_clim_path", "dataprocess/clim_v3/obs_weekly_clim_1999_2021.zarr",
    "--output_dir", "ml_output_flowmulti_v3/seasonal_skill_2020_2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    seasonal_main()


if __name__ == "__main__":
    main()
