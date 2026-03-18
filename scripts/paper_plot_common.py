from pathlib import Path
import re

import matplotlib.pyplot as plt


PAPER_COLORS = {
    "GEOS": "#4C78A8",
    "Hybrid": "#E45756",
    "Structured": "#72B7B2",
    "Random": "#B279A2",
    "Variance": "#F58518",
    "Checkpoint": "#54A24B",
}


def apply_manuscript_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
        }
    )


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def latest_match(base_dir, pattern):
    base_dir = Path(base_dir)
    matches = [path for path in base_dir.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def latest_dir(base_dir, pattern):
    base_dir = Path(base_dir)
    matches = [path for path in base_dir.glob(pattern) if path.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def strategy_color(name):
    lname = name.lower()
    if "geos" in lname:
        return PAPER_COLORS["GEOS"]
    if "variance" in lname or "var" in lname:
        return PAPER_COLORS["Variance"]
    if "structured" in lname or "eof" in lname:
        return PAPER_COLORS["Structured"]
    if "random" in lname:
        return PAPER_COLORS["Random"]
    if "checkpoint" in lname or re.search(r"\be\d+\b", lname):
        return PAPER_COLORS["Checkpoint"]
    return PAPER_COLORS["Hybrid"]


def prettify_strategy(name):
    out = name
    out = out.replace("0. ", "")
    out = out.replace(" Baseline", "")
    out = out.replace("GEOS", "GEOS")
    out = out.replace("PR-T2M", "PR/T2M")
    return out.strip()
