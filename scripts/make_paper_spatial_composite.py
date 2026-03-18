#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from paper_plot_common import apply_manuscript_style, ensure_dir, latest_dir, save_figure


PATTERNS = [
    ("corr_pr_wk1.png", "PR Correlation W1"),
    ("crps_pr_wk1.png", "PR CRPS W1"),
    ("rmse_pr_wk4.png", "PR RMSE W4"),
    ("corr_pr_wk4.png", "PR Correlation W4"),
    ("corr_t2m_wk1.png", "T2M Correlation W1"),
    ("crps_t2m_wk1.png", "T2M CRPS W1"),
    ("rmse_t2m_wk4.png", "T2M RMSE W4"),
    ("corr_t2m_wk4.png", "T2M Correlation W4"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compose spatial diagnostics into a paper-ready figure.")
    parser.add_argument("--base-dir", default=".", help="Directory to search for model output folders.")
    parser.add_argument("--plot-dir", default=None, help="Explicit directory containing metric map images.")
    parser.add_argument("--output-dir", default="paper/figures", help="Directory for the composite figure.")
    parser.add_argument("--demo", action="store_true", help="Create a demo layout without real map images.")
    return parser.parse_args()


def draw_placeholder(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.38, "Spatial diagnostic slot", ha="center", va="center", fontsize=9, color="#666666")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linestyle("--")


def build_composite(plot_dir, output_dir, demo=False):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for ax, (filename, title) in zip(axes, PATTERNS):
        if demo or plot_dir is None:
            draw_placeholder(ax, title)
            continue

        image_path = Path(plot_dir) / filename
        if image_path.exists():
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            draw_placeholder(ax, f"{title}\n(missing)")

    fig.suptitle("Spatial Skill Composite for Paper Figure Assembly", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "This figure assembles the native full-test metric maps into a compact paper-ready panel. "
        "Swap the selected inputs or edit PATTERNS in the script to emphasize different weeks or metrics.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    output_path = Path(output_dir) / "paper_analysis_spatial_composite.pdf"
    save_figure(fig, output_path)
    return output_path


def main():
    args = parse_args()
    apply_manuscript_style()
    output_dir = ensure_dir(args.output_dir)

    plot_dir = Path(args.plot_dir) if args.plot_dir else latest_dir(args.base_dir, "**/plots_full_test_multi")
    if args.demo:
        plot_dir = None
        print("Using demo placeholders for spatial composite.")
    elif plot_dir is not None:
        print(f"Using spatial maps from {plot_dir}")
    else:
        print("No spatial map directory found. Falling back to placeholder layout.")

    output_path = build_composite(plot_dir, output_dir, demo=args.demo or plot_dir is None)
    print(f"Wrote spatial composite: {output_path}")


if __name__ == "__main__":
    main()
