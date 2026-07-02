#!/usr/bin/env python3

import argparse
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from figure_specs import FIGURE_SPECS


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width))


def figure_size(rows, cols):
    width = max(10.0, cols * 3.4)
    height = max(6.0, rows * 2.4 + 1.9)
    return width, height


def draw_panel(ax, label, panel_text):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbfb")
    for spine in ax.spines.values():
        spine.set_visible(False)

    box = FancyBboxPatch(
        (0.05, 0.08),
        0.90,
        0.84,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4,
        edgecolor="#2b5d7d",
        facecolor="#eef5f9",
    )
    ax.add_patch(box)

    inner = Rectangle(
        (0.11, 0.22),
        0.78,
        0.50,
        linewidth=1.2,
        edgecolor="#7f8c8d",
        facecolor="white",
        linestyle="--",
    )
    ax.add_patch(inner)

    ax.text(0.10, 0.83, label, fontsize=10.5, fontweight="bold", ha="left", va="center")
    ax.text(0.50, 0.47, wrap(panel_text, 26), fontsize=9.2, ha="center", va="center", color="#183446")
    ax.text(
        0.50,
        0.15,
        "Final plot content goes here",
        fontsize=8.5,
        ha="center",
        va="center",
        color="#6b7c85",
        style="italic",
    )


def build_figure(spec, output_dir):
    rows, cols = spec["grid"]
    width, height = figure_size(rows, cols)
    fig = plt.figure(figsize=(width, height), constrained_layout=False)

    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[rows * 4.5, 1.6],
        hspace=0.12,
    )
    grid = outer[0].subgridspec(rows, cols, wspace=0.18, hspace=0.20)

    fig.suptitle(spec["title"], fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.5, 0.955, spec["subtitle"], ha="center", va="center", fontsize=10.5, color="#4f5b62")

    for idx in range(rows * cols):
        ax = fig.add_subplot(grid[idx // cols, idx % cols])
        panel_text = spec["panels"][idx] if idx < len(spec["panels"]) else f"Panel {idx + 1}"
        label = panel_text.split("\n", 1)[0]
        draw_panel(ax, label, panel_text)

    caption_ax = fig.add_subplot(outer[1])
    caption_ax.set_xlim(0, 1)
    caption_ax.set_ylim(0, 1)
    caption_ax.set_xticks([])
    caption_ax.set_yticks([])
    for spine in caption_ax.spines.values():
        spine.set_visible(False)
    caption_ax.set_facecolor("white")

    caption_box = FancyBboxPatch(
        (0.03, 0.10),
        0.94,
        0.78,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.1,
        edgecolor="#555555",
        facecolor="#f7f7f7",
    )
    caption_ax.add_patch(caption_box)

    caption_ax.text(
        0.05,
        0.77,
        "Reserved caption space",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
    )
    caption_ax.text(
        0.05,
        0.44,
        wrap(spec["caption"], 120),
        fontsize=9.2,
        ha="left",
        va="center",
        color="#333333",
    )

    output_path = output_dir / spec["filename"]
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate placeholder figures for the arXiv draft.")
    parser.add_argument(
        "--output-dir",
        default="paper/figures",
        help="Directory where the placeholder figure PDFs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing placeholder figures to {output_dir}")
    for spec in FIGURE_SPECS:
        path = build_figure(spec, output_dir)
        print(f"  - {path}")


if __name__ == "__main__":
    main()
