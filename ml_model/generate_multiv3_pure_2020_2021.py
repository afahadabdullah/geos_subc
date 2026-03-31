#!/usr/bin/env python3
"""
Generate 2020-2021 pure-noise multi-v3 forecasts.
"""

import sys

from generate_multiv3_ensembles import main as generate_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--start_year", "2020",
    "--end_year", "2021",
    "--num_ensemble", "90",
    "--batch_size", "4",
    "--ensemble_chunk_size", "30",
    "--num_steps", "50",
    "--pure_noise",
    "--out_dir", "dataprocess/gen_multiv3_pure_2020_2021",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    generate_main()


if __name__ == "__main__":
    main()
