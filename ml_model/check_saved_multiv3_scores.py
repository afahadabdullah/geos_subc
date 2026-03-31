#!/usr/bin/env python3
"""
Quick saved-forecast verification check for the multi-v3 workflow.
"""

import sys

from check_saved_multiv1_scores import main as check_main


DEFAULT_ARGS = [
    "--config", "ml_model/config_flow_multiv3.yaml",
    "--ml_dir", "dataprocess/gen_multiv3",
    "--start_year", "2020",
    "--end_year", "2021",
    "--output_dir", "ml_output_flowmulti_v3/check_saved_scores_24months",
]


def main():
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    check_main()


if __name__ == "__main__":
    main()
