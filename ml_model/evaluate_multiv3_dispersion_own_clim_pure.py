#!/usr/bin/env python3
"""
Evaluate pure-noise multi-v3 dispersion in own-climatology anomaly space.
"""

import sys

from evaluate_multiv1_dispersion_own_clim import main as eval_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--ml_dir", "dataprocess/gen_multiv3_pure_2020_2021",
    "--start_year", "2021",
    "--end_year", "2021",
    "--init_months", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    "--variables", "tas", "pr",
    "--fair_member_count", "4",
    "--sample_chunk_size", "2",
    "--pit_bins", "10",
    "--pit_seed", "7",
    "--ml_clim_path", "dataprocess/clim_v3_pure/ml_weekly_ensmean_clim_1999_2021.zarr",
    "--geos_clim_path", "dataprocess/clim_v3_pure/geos_weekly_ensmean_clim_1999_2021.zarr",
    "--obs_clim_path", "dataprocess/clim_v3_pure/obs_weekly_clim_1999_2021.zarr",
    "--output_dir", "ml_output_flowmulti_v3/multiv3_dispersion_own_clim_pure_2021_2022",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    eval_main()


if __name__ == "__main__":
    main()
