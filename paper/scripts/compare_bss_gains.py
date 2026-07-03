#!/usr/bin/env python3
"""Compare raw vs calibrated BSS gains from matrix_summary_metrics.csv.

Prints lead-aggregated raw BSS gain (bss_diff) and calibrated BSS gain
(calibrated_bss_diff) side by side, per variable/lead/subset, so the better
metric can be chosen for the lead-skill table in main.tex.

Usage:
  python paper/scripts/compare_bss_gains.py --matrix-dir <dir> [--matrix-dir <dir2> ...]

Pass the all-grid and land-mask evaluation directories (one --matrix-dir each)
to get both masks in one report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEASGROUP = "valid_season_lead"
SUBSETS = ("all_data", "extreme_events")


def group_weights(df: pd.DataFrame) -> pd.Series | None:
    for name in ("weight_sum", "n_cases", "n_forecasts", "n_samples"):
        if name in df:
            return df[name]
    return None


def wavg(series: pd.Series, weights: pd.Series | None) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if weights is None:
        w = np.ones_like(values)
    else:
        w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(values) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values[ok], weights=w[ok]))


def report(matrix_dir: Path) -> None:
    csv_path = matrix_dir / "matrix_summary_metrics.csv"
    df = pd.read_csv(csv_path)
    print(f"\n=== {matrix_dir} ===")
    have_raw = "bss_diff" in df.columns
    have_cal = "calibrated_bss_diff" in df.columns
    if not (have_raw or have_cal):
        print("  no bss_diff / calibrated_bss_diff columns found")
        return
    if "group_type" in df.columns:
        df = df[df["group_type"].eq(SEASGROUP)]
    for subset in SUBSETS:
        sub = df[df["subset"].eq(subset)] if "subset" in df.columns else df
        if sub.empty:
            continue
        print(f"\n  subset={subset}")
        print(f"  {'var':<5} {'lead':<5} {'raw BSS gain':>13} {'cal BSS gain':>13} {'better':>8}")
        for (variable, lead), grp in sub.groupby(["variable", "lead"]):
            w = group_weights(grp)
            raw = wavg(grp["bss_diff"], w) if have_raw else float("nan")
            cal = wavg(grp["calibrated_bss_diff"], w) if have_cal else float("nan")
            if np.isfinite(raw) and np.isfinite(cal):
                better = "raw" if raw > cal else "cal"
            else:
                better = "-"
            print(f"  {variable:<5} {int(lead):<5} {raw:>13.4f} {cal:>13.4f} {better:>8}")
        # Aggregate over leads
        for variable, grp in sub.groupby("variable"):
            w = group_weights(grp)
            raw = wavg(grp["bss_diff"], w) if have_raw else float("nan")
            cal = wavg(grp["calibrated_bss_diff"], w) if have_cal else float("nan")
            print(f"  {variable:<5} {'mean':<5} {raw:>13.4f} {cal:>13.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", action="append", required=True,
                        help="Directory containing matrix_summary_metrics.csv (repeatable).")
    args = parser.parse_args()
    for item in args.matrix_dir:
        report(Path(item))


if __name__ == "__main__":
    main()
