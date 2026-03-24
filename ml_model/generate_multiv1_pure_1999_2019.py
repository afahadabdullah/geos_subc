#!/usr/bin/env python3
"""
Generate 1999-2019 pure-noise multiv1 hindcasts.

This reuses the shared multiv1 generator but forces:
- checkpoint = periodic_ckpt_epoch_215.pt
- pure Gaussian noise only
- no EOF-LHS perturbations
- no variance-head scaling
- 4 ensemble members
- 50 Euler steps
"""

import sys

from generate_multiv1_ensembles import main as generate_main


CHECKPOINT_PATH = "/home1/11353/afahad/geos_subc/ml_output_flowmulti/periodic_ckpt_epoch_215.pt"

DEFAULT_ARGS = [
    "--checkpoint", CHECKPOINT_PATH,
    "--start_year", "1999",
    "--end_year", "2019",
    "--num_ensemble", "4",
    "--batch_size", "40",
    "--ensemble_chunk_size", "4",
    "--num_steps", "50",
    "--pure_noise",
    "--out_dir", "dataprocess/gen_multiv1_pure_1999_2019",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    generate_main()


if __name__ == "__main__":
    main()
