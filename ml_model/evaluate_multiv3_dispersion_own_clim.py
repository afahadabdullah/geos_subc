#!/usr/bin/env python3
"""
Evaluate multi-v3 own-climatology anomaly dispersion for ML and GEOS.
"""

import sys

from evaluate_multiv1_dispersion_own_clim import main as eval_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--ml_dir", "dataprocess/gen_multiv3",
    "--start_year", "2020",
    "--end_year", "2021",
    "--init_months", "5", "6", "7", "8",
    "--variables", "tas", "pr",
    "--fair_member_count", "4",
    "--sample_chunk_size", "2",
    "--pit_bins", "10",
    "--pit_seed", "7",
    "--ml_clim_path", "dataprocess/clim_v3/ml_weekly_ensmean_clim_1999_2021.zarr",
    "--geos_clim_path", "dataprocess/clim_v3/geos_weekly_ensmean_clim_1999_2021.zarr",
    "--obs_clim_path", "dataprocess/clim_v3/obs_weekly_clim_1999_2021.zarr",
    "--output_dir", "ml_output_flowmulti_v3/multiv3_dispersion_own_clim_mjja_2020_2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    eval_main()


if __name__ == "__main__":
    main()
