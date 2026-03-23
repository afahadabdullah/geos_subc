#!/usr/bin/env python3
"""
Generate ML hindcasts for 1999-2019 using the same stochastic multiv1 pipeline
used for the 2020-2021 forecasts.

This is intentionally a thin wrapper around generate_multiv1_ensembles.py so
the hindcast path stays aligned with the main ML generation logic:
EOF-LHS noise, rho blending, variance-head scaling, Euler solve, and Zarr
resume/finalize behavior all come from the shared generator.
"""

import sys

from generate_multiv1_ensembles import main as generate_main


DEFAULT_ARGS = [
    "--start_year", "1999",
    "--end_year", "2019",
    "--num_ensemble", "4",
    "--batch_size", "40",
    "--ensemble_chunk_size", "4",
    "--num_steps", "50",
    "--out_dir", "dataprocess/gen_multiv1_hindcast_1999_2019",
]


def main():
    # Put defaults first so any user-supplied CLI values later on can override
    # them normally via argparse.
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    generate_main()


if __name__ == "__main__":
    main()
