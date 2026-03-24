#!/usr/bin/env python3
"""
Generate 2020-2021 pure-noise multiv1 forecasts.

This reuses the shared multiv1 generator but forces:
- checkpoint = periodic_ckpt_epoch_215.pt
- pure Gaussian noise only
- no EOF-LHS perturbations
- no variance-head scaling
- 90 ensemble members
- 50 Euler steps
"""

import sys

from generate_multiv1_ensembles import main as generate_main


CHECKPOINT_PATH = "/home1/11353/afahad/geos_subc/ml_output_flowmulti/periodic_ckpt_epoch_215.pt"

DEFAULT_ARGS = [
    "--checkpoint", CHECKPOINT_PATH,
    "--start_year", "2020",
    "--end_year", "2021",
    "--num_ensemble", "90",
    "--batch_size", "4",
    "--ensemble_chunk_size", "30",
    "--num_steps", "50",
    "--pure_noise",
    "--out_dir", "dataprocess/gen_multiv1_pure_2020_2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    generate_main()


if __name__ == "__main__":
    main()
