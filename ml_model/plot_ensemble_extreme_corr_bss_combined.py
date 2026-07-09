#!/usr/bin/env python3
"""Combine spatial correlation and BSS ensemble-size diagnostics.

This plot-only utility reads outputs already written by:

  * evaluate_ensemble_correlation_extremes_flow_finalv1_global.py
  * evaluate_ensemble_bss_extremes_flow_finalv1_global.py

The default figure keeps the visual story compact: FlowMatch gain over GEOS
for spatial correlation and q95-exceedance BSS as generated ensemble size
increases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CORR_DIR = (
    "ml_output_flow_finalv1_global_noisectx_t2mres/"
    "ensemble_corr_extreme_t2m30_pr30_regions_2021_2023_w1w4_memberboot50"
)
DEFAULT_BSS_DIR = (
    "ml_output_flow_finalv1_global_noisectx_t2mres/"
    "ensemble_bss_extreme_t2m30_pr30_regions_2021_2023_w1w4_memberboot50"
)
DEFAULT_OUT_DIR = (
    "ml_output_flow_finalv1_global_noisectx_t2mres/"
    "ensemble_extreme_spatial_corr_bss_combined"
)

VARIABLE_LABELS = {"pr": "PR", "t2m": "T2M"}
LEAD_COLORS = {1: "#7fb3d5", 2: "#4a7fb5", 3: "#2e5f96", 4: "#3b2f7d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corr_dir", default=DEFAULT_CORR_DIR)
    parser.add_argument("--bss_dir", default=DEFAULT_BSS_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--corr_target", choices=("ens_mean", "q95"), default="ens_mean")
    parser.add_argument("--variables", default="pr,t2m")
    parser.add_argument("--lead_values", default="1,2,3,4")
    parser.add_argument("--sample_sizes", default="", help="Optional comma-separated member counts to keep.")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show_ci", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--title", default="", help="Optional figure title.")
    return parser.parse_args()


def parse_csv_list(text: str, cast=str) -> list:
    values = []
    for item in str(text or "").split(","):
        token = item.strip()
        if token:
            values.append(cast(token))
    return values


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def filtered_summary(
    df: pd.DataFrame,
    variables: list[str],
    leads: list[int],
    sample_sizes: list[int],
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["variable"].astype(str).isin(variables)]
    out = out[out["lead"].astype(int).isin(leads)]
    if sample_sizes:
        out = out[out["member_count"].astype(int).isin(sample_sizes)]
    return out


def ci_for_line(
    summary_line: pd.DataFrame,
    boot_df: pd.DataFrame,
    variable: str,
    lead: int,
    metric: str,
    target: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not boot_df.empty:
        x_counts = summary_line["member_count"].astype(int).to_list()
        boot = boot_df[
            boot_df["variable"].astype(str).eq(variable)
            & boot_df["lead"].astype(int).eq(int(lead))
        ].copy()
        if target is not None and "target" in boot.columns:
            boot = boot[boot["target"].astype(str).eq(target)]
        if "member_count" in boot.columns:
            boot = boot[boot["member_count"].astype(int).isin(x_counts)]
        lower = f"{metric}_p025"
        upper = f"{metric}_p975"
        if not boot.empty and {lower, upper, "member_count"} <= set(boot.columns):
            boot = boot.sort_values("member_count")
            boot = boot.set_index(boot["member_count"].astype(int)).reindex(x_counts)
            return (
                boot[lower].to_numpy(dtype=float),
                boot[upper].to_numpy(dtype=float),
            )

    lower = f"{metric}_p05"
    upper = f"{metric}_p95"
    if {lower, upper} <= set(summary_line.columns):
        return (
            summary_line[lower].to_numpy(dtype=float),
            summary_line[upper].to_numpy(dtype=float),
        )
    return None, None


def robust_ylim(values: list[float], *, floor_low: float, floor_high: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor_low, floor_high
    lo = float(np.nanquantile(arr, 0.03))
    hi = float(np.nanquantile(arr, 0.97))
    pad = max(0.12 * (hi - lo), 0.05 * max(abs(lo), abs(hi), 1.0))
    lo = min(floor_low, lo - pad)
    hi = max(floor_high, hi + pad)
    if hi <= lo:
        hi = lo + max(1.0, abs(lo) * 0.1)
    return lo, hi


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style_axis(ax, *, zero_line: bool = True) -> None:
    ax.grid(True, axis="y", color="#dce3e8", lw=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5, colors="#2d3a45")
    if zero_line:
        ax.axhline(0.0, color="#77838f", lw=0.85, ls="--", zorder=0)


def plot_combined(
    corr_summary: pd.DataFrame,
    corr_boot: pd.DataFrame,
    bss_summary: pd.DataFrame,
    bss_boot: pd.DataFrame,
    args: argparse.Namespace,
) -> list[Path]:
    plt = import_matplotlib()
    variables = parse_csv_list(args.variables, str)
    leads = parse_csv_list(args.lead_values, int)
    sample_sizes = parse_csv_list(args.sample_sizes, int)

    corr = filtered_summary(corr_summary, variables, leads, sample_sizes)
    if "target" in corr.columns:
        corr = corr[corr["target"].astype(str).eq(args.corr_target)]
    bss = filtered_summary(bss_summary, variables, leads, sample_sizes)

    if corr.empty:
        raise ValueError(f"No correlation rows after filtering target={args.corr_target!r}.")
    if bss.empty:
        raise ValueError("No BSS rows after filtering.")

    variables = [var for var in variables if var in set(corr["variable"].astype(str)) | set(bss["variable"].astype(str))]
    if not variables:
        raise ValueError("No requested variables are present in the filtered summaries.")

    fig, axes = plt.subplots(
        2,
        len(variables),
        figsize=(3.75 * len(variables), 5.35),
        sharex="col",
        gridspec_kw={"hspace": 0.28, "wspace": 0.20},
    )
    axes = np.asarray(axes).reshape(2, len(variables))
    letters = "abcdefghijklmnopqrstuvwxyz"

    for vi, variable in enumerate(variables):
        label = VARIABLE_LABELS.get(variable, variable.upper())
        corr_ax = axes[0, vi]
        bss_ax = axes[1, vi]
        corr_values: list[float] = []
        bss_values: list[float] = []
        corr_ax.set_title(f"({letters[vi]}) {label} spatial correlation", loc="left", fontsize=10, fontweight="bold")
        bss_ax.set_title(
            f"({letters[len(variables) + vi]}) {label} q95 BSS",
            loc="left",
            fontsize=10,
            fontweight="bold",
        )

        for lead in leads:
            color = LEAD_COLORS.get(int(lead), "#355f8d")
            c_line = corr[
                corr["variable"].astype(str).eq(variable)
                & corr["lead"].astype(int).eq(int(lead))
            ].sort_values("member_count")
            if not c_line.empty:
                x = c_line["member_count"].to_numpy(dtype=float)
                y = c_line["corr_gain_pct_mean"].to_numpy(dtype=float)
                corr_values.extend(y[np.isfinite(y)].tolist())
                if args.show_ci:
                    lo, hi = ci_for_line(c_line, corr_boot, variable, lead, "corr_gain_pct", args.corr_target)
                    if lo is not None and hi is not None and len(lo) == len(x):
                        corr_ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
                        corr_values.extend(np.asarray(lo)[np.isfinite(lo)].tolist())
                        corr_values.extend(np.asarray(hi)[np.isfinite(hi)].tolist())
                corr_ax.plot(x, y, color=color, lw=1.85, marker="o", ms=3.5, label=f"W{lead}")

            b_line = bss[
                bss["variable"].astype(str).eq(variable)
                & bss["lead"].astype(int).eq(int(lead))
            ].sort_values("member_count")
            if not b_line.empty:
                x = b_line["member_count"].to_numpy(dtype=float)
                y = b_line["bss_gain_x100_mean"].to_numpy(dtype=float)
                bss_values.extend(y[np.isfinite(y)].tolist())
                if args.show_ci:
                    lo, hi = ci_for_line(b_line, bss_boot, variable, lead, "bss_gain_x100")
                    if lo is not None and hi is not None and len(lo) == len(x):
                        bss_ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
                        bss_values.extend(np.asarray(lo)[np.isfinite(lo)].tolist())
                        bss_values.extend(np.asarray(hi)[np.isfinite(hi)].tolist())
                bss_ax.plot(x, y, color=color, lw=1.85, marker="o", ms=3.5, label=f"W{lead}")

        style_axis(corr_ax)
        style_axis(bss_ax)
        if corr_ax.has_data():
            corr_ax.set_ylim(*robust_ylim(corr_values, floor_low=-2.0, floor_high=5.0))
        if bss_ax.has_data():
            bss_ax.set_ylim(*robust_ylim(bss_values, floor_low=-5.0, floor_high=5.0))
        bss_ax.set_xlabel("Generated members", fontsize=9.5)
        if vi == 0:
            corr_ax.set_ylabel("FlowMatch gain (%)", fontsize=9.5)
            bss_ax.set_ylabel("FlowMatch - GEOS (x100)", fontsize=9.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if args.title:
        legend_y = 0.950
        rect_top = 0.850
    else:
        legend_y = 0.985
        rect_top = 0.880

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, legend_y),
            fontsize=8.8,
            handlelength=1.8,
            columnspacing=1.2,
        )
    if args.title:
        fig.suptitle(str(args.title), fontsize=12.0, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=rect_top, hspace=0.30, wspace=0.22)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    stem = f"ensemble_extreme_spatial_corr_{args.corr_target}_bss_combined"
    if args.format in ("png", "both"):
        path = out_dir / f"{stem}.png"
        fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
        outputs.append(path)
    if args.format in ("pdf", "both"):
        path = out_dir / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    corr_dir = Path(args.corr_dir)
    bss_dir = Path(args.bss_dir)
    corr_summary = read_required_csv(corr_dir / "ensemble_correlation_summary.csv")
    bss_summary = read_required_csv(bss_dir / "ensemble_bss_summary.csv")
    corr_boot = read_optional_csv(corr_dir / "ensemble_correlation_case_bootstrap_ci.csv")
    bss_boot = read_optional_csv(bss_dir / "ensemble_bss_case_bootstrap_ci.csv")
    outputs = plot_combined(corr_summary, corr_boot, bss_summary, bss_boot, args)
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
