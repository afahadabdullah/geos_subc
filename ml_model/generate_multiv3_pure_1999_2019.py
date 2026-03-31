#!/usr/bin/env python3
"""
Generate 1999-2019 pure-noise multi-v3 hindcasts.
"""

import sys

from generate_multiv3_ensembles import main as generate_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--start_year", "1999",
    "--end_year", "2019",
    "--num_ensemble", "4",
    "--batch_size", "40",
    "--ensemble_chunk_size", "4",
    "--num_steps", "50",
    "--pure_noise",
    "--out_dir", "dataprocess/gen_multiv3_pure_1999_2019",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    generate_main()


if __name__ == "__main__":
    main()
