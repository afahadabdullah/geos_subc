#!/usr/bin/env python3
"""Review response M3: bootstrap CI on the noise-prior ablation.

Reads one or more noise_comparison_global_*.csv files (per-row/batch scores
from ml_model/compare_noise_flow_finalv1_global.py) and bootstraps the
EOF-LHS minus Gaussian CRPS-skill difference over rows, paired by
batch/init where a pairing column exists.

Usage:
  python paper/scripts/review_response/r1_ablation_ci.py \
      ml_output_noise_compare_global_flow_finalv1/noise_comparison_global_*.csv
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

STRATEGY_COLS = ("strategy", "noise_mode", "noise", "sampler", "mode", "label", "config")
PAIR_COLS = ("batch", "batch_idx", "init_time", "init", "case_id", "sample")


def find_col(df: pd.DataFrame, candidates) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="noise comparison CSV path(s) or globs")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    paths = []
    for item in args.csvs:
        paths.extend(sorted(glob.glob(item)))
    if not paths:
        raise SystemExit("No CSV files matched.")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["__source"] = path
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(paths)} file(s). Columns: {list(df.columns)}")

    strat_col = find_col(df, STRATEGY_COLS)
    var_col = find_col(df, ("variable", "var", "target"))
    pair_col = find_col(df, PAIR_COLS)
    if strat_col is None or var_col is None:
        raise SystemExit("Could not identify strategy/variable columns; inspect the printout above.")

    if "crps_skill_pct" in df.columns:
        df["__skill"] = pd.to_numeric(df["crps_skill_pct"], errors="coerce")
    elif {"model_crps", "geos_crps"} <= set(df.columns):
        df["__skill"] = 100.0 * (1.0 - pd.to_numeric(df["model_crps"], errors="coerce")
                                 / pd.to_numeric(df["geos_crps"], errors="coerce"))
    else:
        raise SystemExit("No crps_skill_pct or model_crps/geos_crps columns found.")

    strategies = sorted(df[strat_col].astype(str).unique())
    gauss = next((s for s in strategies if "gauss" in s.lower()), strategies[0])
    others = [s for s in strategies if s != gauss]
    print(f"Strategies: {strategies}; Gaussian reference = {gauss}; pairing column = {pair_col}")

    for variable in sorted(df[var_col].astype(str).str.lower().unique()):
        vdf = df[df[var_col].astype(str).str.lower().eq(variable)]
        g_rows = vdf[vdf[strat_col].astype(str).eq(gauss)]
        for other in others:
            o_rows = vdf[vdf[strat_col].astype(str).eq(other)]
            if pair_col is not None:
                merged = pd.merge(
                    g_rows[[pair_col, "__skill"]].rename(columns={"__skill": "g"}),
                    o_rows[[pair_col, "__skill"]].rename(columns={"__skill": "o"}),
                    on=pair_col, how="inner",
                ).dropna()
                diffs = (merged["o"] - merged["g"]).to_numpy(dtype=float)
                if diffs.size < 4:
                    print(f"[{variable}] {other}: insufficient paired rows ({diffs.size})")
                    continue
                boots = np.array([
                    rng.choice(diffs, size=diffs.size, replace=True).mean()
                    for _ in range(args.n_boot)
                ])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                frac_pos = float((boots > 0).mean())
                print(f"[{variable}] {other} - {gauss} (paired, n={diffs.size}): "
                      f"mean {diffs.mean():+.2f} skill pts, 95% CI [{lo:+.2f}, {hi:+.2f}], "
                      f"P(diff>0) = {frac_pos:.3f}")
            else:
                g = g_rows["__skill"].dropna().to_numpy(dtype=float)
                o = o_rows["__skill"].dropna().to_numpy(dtype=float)
                if g.size < 4 or o.size < 4:
                    print(f"[{variable}] {other}: insufficient rows (g={g.size}, o={o.size})")
                    continue
                boots = np.array([
                    rng.choice(o, o.size, replace=True).mean()
                    - rng.choice(g, g.size, replace=True).mean()
                    for _ in range(args.n_boot)
                ])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                print(f"[{variable}] {other} - {gauss} (unpaired, n={o.size}/{g.size}): "
                      f"mean {o.mean() - g.mean():+.2f} skill pts, 95% CI [{lo:+.2f}, {hi:+.2f}], "
                      f"P(diff>0) = {float((boots > 0).mean()):.3f}")


if __name__ == "__main__":
    main()
