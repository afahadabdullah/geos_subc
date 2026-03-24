#!/usr/bin/env python3
"""
Evaluate pure-noise multiv1 dispersion in own-climatology anomaly space.

Default assumption:
- evaluate 2021 init dates only
- this corresponds to anomaly verification spanning 2021-2022
- use pure-noise weekly climatology stores under dataprocess/clim_pure
"""

import sys

from evaluate_multiv1_dispersion_own_clim import main as eval_main


DEFAULT_ARGS = [
    "--ml_dir", "dataprocess/gen_multiv1_pure_2020_2021",
    "--start_year", "2021",
    "--end_year", "2021",
    "--init_months", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
    "--variables", "tas", "pr",
    "--fair_member_count", "4",
    "--sample_chunk_size", "2",
    "--pit_bins", "10",
    "--pit_seed", "7",
    "--ml_clim_path", "dataprocess/clim_pure/ml_weekly_ensmean_clim_1999_2021.zarr",
    "--geos_clim_path", "dataprocess/clim_pure/geos_weekly_ensmean_clim_1999_2021.zarr",
    "--obs_clim_path", "dataprocess/clim_pure/obs_weekly_clim_1999_2021.zarr",
    "--output_dir", "ml_output_flowmulti/multiv1_dispersion_own_clim_pure_2021_2022",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    eval_main()


if __name__ == "__main__":
    main()
