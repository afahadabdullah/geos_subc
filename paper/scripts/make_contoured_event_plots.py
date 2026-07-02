#!/usr/bin/env python3
"""Regenerate the contoured Figures 6 and 7 for the paper.

This script executes the event catalog and quantile forecast evaluators with targeted
arguments to only process the specific event regions and variables for lead week 4.
This avoids running the expensive evaluations for the entire event catalog, reducing
the total runtime to under 2 minutes.

Target events:
  - California Atmospheric Rivers: Region 'conus', Variable 'pr', Lead 4
  - UK July 2022 Heatwave: Region 'europe', Variable 't2m', Lead 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted script to generate contoured plots for Figs 6 & 7 at Lead 4."
    )
    parser.add_argument("--format", default="png", choices=("png", "pdf", "both"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)

    print(f"Working directory set to: {project_dir}")

    # 1. Regenerate Event Catalog spatial maps for target events at lead 4
    print("\n--- 🗺️ Generating Event Catalog Spatial Maps (Lead 4) ---")
    run_command([
        "python3", "ml_model/evaluate_event_catalog_flow_finalv1_global.py",
        "--regions", "conus,europe",
        "--variables", "pr,t2m",
        "--leads", "4",
        "--make_plots",
        "--overwrite"
    ])

    # 2. Regenerate Event Quantile spatial maps for target events at lead 4
    print("\n--- 📈 Generating Event Quantile Spatial Maps (Lead 4) ---")
    run_command([
        "python3", "ml_model/evaluate_event_quantile_forecast_flow_finalv1_global.py",
        "--regions", "conus,europe",
        "--variables", "pr,t2m",
        "--leads", "4",
        "--make_plots",
        "--overwrite"
    ])

    # 3. Run the paper figures script to compile the new figures
    print("\n--- 🎨 Rebuilding Paper Figures ---")
    run_command([
        "python3", "paper/scripts/make_paper_figures.py",
        "--format", args.format,
        "--dpi", str(args.dpi)
    ])

    print("\n✅ Done! Check your generated figures in paper/figures/")


if __name__ == "__main__":
    main()
