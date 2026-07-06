#!/usr/bin/env python3
"""Review response M9: summarize the FULL event catalog, not just featured cases.

Aggregates event_selected_lead_metrics.csv per event: number of init/lead
pairs, mean event-mask CRPS skill, raw and calibrated BSS gains, and event
probability gains. Output doubles as the supplement table defusing the
case-selection concern.

Usage:
  python paper/scripts/review_response/r1_event_catalog_summary.py \
      [--event_dir ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_CANDIDATES = {
    "event_crps_skill": ["crps_on_obs_extreme_skill_pct", "crps_skill_pct"],
    "bss_gain_raw": ["bss_diff"],
    "bss_gain_cal": ["calibrated_bss_diff"],
    "event_prob_gain": ["event_probability_on_obs_extreme_diff",
                        "event_probability_top_tail_diff"],
}


def first_col(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_dir",
                        default="ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    path = event_dir / "event_selected_lead_metrics.csv"
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}\nColumns: {list(df.columns)}\n")

    rows = []
    for event_id, grp in df.groupby("event_id"):
        row = {
            "event_id": event_id,
            "event_name": str(grp["event_name"].iloc[0]) if "event_name" in grp else "",
            "variable": str(grp["variable"].iloc[0]) if "variable" in grp else "",
            "n_init_lead_pairs": int(len(grp.drop_duplicates(
                [c for c in ("init_time", "lead") if c in grp.columns]))) if
                {"init_time", "lead"} & set(grp.columns) else len(grp),
        }
        for out_name, candidates in METRIC_CANDIDATES.items():
            col = first_col(grp, candidates)
            if col is not None:
                vals = pd.to_numeric(grp[col], errors="coerce")
                row[out_name] = float(vals.mean())
                row[f"{out_name}_w4"] = float(
                    vals[grp["lead"].astype(int).eq(4)].mean()) if "lead" in grp else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["variable", "event_id"]).reset_index(drop=True)
    out_path = Path(args.out_csv) if args.out_csv else event_dir / "r1_event_catalog_summary.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(out.round(3).to_string(index=False))

    numeric = out.select_dtypes(float)
    print("\n--- catalog-wide summary ---")
    print(f"events: {len(out)}")
    for col in ("event_crps_skill", "bss_gain_raw", "bss_gain_cal"):
        if col in out:
            vals = out[col].dropna()
            if len(vals):
                print(f"{col}: mean {vals.mean():+.3f}, median {vals.median():+.3f}, "
                      f"improved {int((vals > 0).sum())}/{len(vals)} events")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
