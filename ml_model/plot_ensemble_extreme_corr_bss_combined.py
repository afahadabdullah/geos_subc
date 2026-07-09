#!/usr/bin/env python3
"""Combine spatial correlation and BSS ensemble-size diagnostics.

This plot-only utility reads outputs already written by:

  * evaluate_ensemble_correlation_extremes_flow_finalv1_global.py
  * evaluate_ensemble_bss_extremes_flow_finalv1_global.py

The default figure keeps the visual story compact: FIMr1p1-FlowMatch gain over
FIMr1p1 for spatial correlation and q95-exceedance BSS as generated ensemble
size increases. Right-hand axes report the raw FIMr1p1 metric values.
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
MODEL_LABEL = "FIMr1p1-FlowMatch"
BASELINE_LABEL = "FIMr1p1"
CORR_SUM_COLUMNS = [
    "model_weight_sum",
    "model_x_sum",
    "model_y_sum",
    "model_x2_sum",
    "model_y2_sum",
    "model_xy_sum",
    "geos_weight_sum",
    "geos_x_sum",
    "geos_y_sum",
    "geos_x2_sum",
    "geos_y2_sum",
    "geos_xy_sum",
]
BSS_SUM_COLUMNS = [
    "model_bs_sum",
    "geos_bs_sum",
    "ref_bs_sum",
    "weight_sum",
    "obs_event_weight_sum",
    "model_prob_weight_sum",
    "geos_prob_weight_sum",
]


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
    parser.add_argument("--table_member_count", type=int, default=8)
    parser.add_argument("--table_bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--table_seed", type=int, default=202407)
    parser.add_argument("--print_all_table", action="store_true")
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


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


def corr_from_sums(row: dict[str, float] | pd.Series, prefix: str) -> float:
    w = float(row.get(f"{prefix}_weight_sum", np.nan))
    if not np.isfinite(w) or w <= 1e-12:
        return np.nan
    mx = float(row[f"{prefix}_x_sum"]) / w
    my = float(row[f"{prefix}_y_sum"]) / w
    cov = float(row[f"{prefix}_xy_sum"]) / w - mx * my
    vx = float(row[f"{prefix}_x2_sum"]) / w - mx * mx
    vy = float(row[f"{prefix}_y2_sum"]) / w - my * my
    denom = np.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom <= 1e-12:
        return np.nan
    value = cov / denom
    return float(value) if np.isfinite(value) else np.nan


def corr_metrics_from_sums(row: dict[str, float] | pd.Series) -> dict[str, float]:
    model_corr = corr_from_sums(row, "model")
    geos_corr = corr_from_sums(row, "geos")
    corr_diff = model_corr - geos_corr
    corr_gain_pct = (
        100.0 * corr_diff / abs(geos_corr)
        if np.isfinite(geos_corr) and abs(geos_corr) > 1e-12
        else np.nan
    )
    return {
        "model_corr": model_corr,
        "geos_corr": geos_corr,
        "corr_diff": corr_diff,
        "corr_gain_pct": corr_gain_pct,
    }


def bss_metrics_from_sums(row: dict[str, float] | pd.Series) -> dict[str, float]:
    ref = float(row.get("ref_bs_sum", np.nan))
    weight = float(row.get("weight_sum", np.nan))
    model_bs = float(row.get("model_bs_sum", np.nan))
    geos_bs = float(row.get("geos_bs_sum", np.nan))
    model_bss = 1.0 - model_bs / ref if np.isfinite(ref) and ref > 1e-12 else np.nan
    geos_bss = 1.0 - geos_bs / ref if np.isfinite(ref) and ref > 1e-12 else np.nan
    bss_gain = model_bss - geos_bss
    return {
        "model_bs": model_bs / weight if np.isfinite(weight) and weight > 0.0 else np.nan,
        "geos_bs": geos_bs / weight if np.isfinite(weight) and weight > 0.0 else np.nan,
        "ref_bs": ref / weight if np.isfinite(weight) and weight > 0.0 else np.nan,
        "model_bss": model_bss,
        "geos_bss": geos_bss,
        "bss_gain": bss_gain,
        "bss_gain_x100": 100.0 * bss_gain if np.isfinite(bss_gain) else np.nan,
    }


def sign_p_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"p_gt0": np.nan, "p_le0": np.nan, "p_two_sided": np.nan}
    # Plus-one correction prevents a printed zero p-value from finite bootstrap samples.
    p_gt0 = float((np.sum(arr > 0.0) + 1.0) / (arr.size + 1.0))
    p_le0 = float((np.sum(arr <= 0.0) + 1.0) / (arr.size + 1.0))
    return {
        "p_gt0": p_gt0,
        "p_le0": p_le0,
        "p_two_sided": min(1.0, 2.0 * min(p_gt0, p_le0)),
    }


def bootstrap_gain_pvalues(
    case_df: pd.DataFrame,
    *,
    group_cols: list[str],
    sum_cols: list[str],
    metric_func,
    sign_metric: str,
    repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if repeats <= 0 or case_df.empty or not {"case_id", *group_cols, *sum_cols} <= set(case_df.columns):
        return pd.DataFrame()
    rows = []
    for key, group in case_df.groupby(group_cols, dropna=False):
        cases = sorted(group["case_id"].astype(str).unique())
        if not cases:
            continue
        rows_by_case = {
            case: group[group["case_id"].astype(str).eq(case)].reset_index(drop=True)
            for case in cases
        }
        boot_values: list[float] = []
        for _ in range(int(repeats)):
            sampled_cases = rng.choice(cases, size=len(cases), replace=True)
            state = {col: 0.0 for col in sum_cols}
            for case in sampled_cases:
                case_rows = rows_by_case[str(case)]
                picked = case_rows.iloc[int(rng.integers(0, len(case_rows)))]
                for col in sum_cols:
                    state[col] += float(picked[col])
            value = metric_func(state).get(sign_metric, np.nan)
            if np.isfinite(value):
                boot_values.append(float(value))
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row["bootstrap_repeats_for_p"] = int(repeats)
        pvals = sign_p_values(boot_values)
        for name, value in pvals.items():
            row[f"{sign_metric}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def value_ci_table(
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    *,
    metric: str,
    group_cols: list[str],
) -> pd.DataFrame:
    cols = [*group_cols, f"{metric}_mean"]
    for optional in (f"{metric}_p05", f"{metric}_p95"):
        if optional in summary.columns:
            cols.append(optional)
    out = summary[[col for col in cols if col in summary.columns]].copy()
    if f"{metric}_mean" in out.columns:
        out = out.rename(columns={f"{metric}_mean": metric})
    low_col = f"{metric}_p05"
    high_col = f"{metric}_p95"
    out[f"{metric}_ci_low"] = out[low_col] if low_col in out.columns else np.nan
    out[f"{metric}_ci_high"] = out[high_col] if high_col in out.columns else np.nan

    boot_low = f"{metric}_p025"
    boot_high = f"{metric}_p975"
    if not boot.empty and {boot_low, boot_high, *group_cols} <= set(boot.columns):
        boot_sub = boot[[*group_cols, boot_low, boot_high]].copy()
        boot_sub = boot_sub.rename(columns={boot_low: f"{metric}_boot_low", boot_high: f"{metric}_boot_high"})
        out = out.merge(boot_sub, on=group_cols, how="left")
        out[f"{metric}_ci_low"] = out[f"{metric}_boot_low"].combine_first(out[f"{metric}_ci_low"])
        out[f"{metric}_ci_high"] = out[f"{metric}_boot_high"].combine_first(out[f"{metric}_ci_high"])
        out = out.drop(columns=[f"{metric}_boot_low", f"{metric}_boot_high"])
    drop_cols = [col for col in (low_col, high_col) if col in out.columns]
    return out.drop(columns=drop_cols)


def build_combined_table(
    corr_summary: pd.DataFrame,
    corr_boot: pd.DataFrame,
    corr_case: pd.DataFrame,
    bss_summary: pd.DataFrame,
    bss_boot: pd.DataFrame,
    bss_case: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    variables = parse_csv_list(args.variables, str)
    leads = parse_csv_list(args.lead_values, int)
    sample_sizes = parse_csv_list(args.sample_sizes, int)
    rng = np.random.default_rng(int(args.table_seed))

    corr = filtered_summary(corr_summary, variables, leads, sample_sizes)
    corr_boot_f = filtered_summary(corr_boot, variables, leads, sample_sizes) if not corr_boot.empty else corr_boot
    if "target" in corr.columns:
        corr = corr[corr["target"].astype(str).eq(args.corr_target)]
    if not corr_boot_f.empty and "target" in corr_boot_f.columns:
        corr_boot_f = corr_boot_f[corr_boot_f["target"].astype(str).eq(args.corr_target)]

    bss = filtered_summary(bss_summary, variables, leads, sample_sizes)
    bss_boot_f = filtered_summary(bss_boot, variables, leads, sample_sizes) if not bss_boot.empty else bss_boot

    corr_key = ["variable", "lead", "member_count"]
    corr_value_cols = [
        col for col in ("model_corr_mean", "geos_corr_mean", "corr_diff_mean")
        if col in corr.columns
    ]
    corr_table = corr[[*corr_key, *corr_value_cols]].copy()
    corr_gain = value_ci_table(corr, corr_boot_f, metric="corr_gain_pct", group_cols=["variable", "lead", "target", "member_count"])
    if "target" in corr_gain.columns:
        corr_gain = corr_gain.drop(columns=["target"])
    corr_table = corr_table.merge(corr_gain, on=corr_key, how="left")

    corr_p = pd.DataFrame()
    if not corr_case.empty:
        corr_case_f = filtered_summary(corr_case, variables, leads, sample_sizes)
        if "target" in corr_case_f.columns:
            corr_case_f = corr_case_f[corr_case_f["target"].astype(str).eq(args.corr_target)]
        corr_p = bootstrap_gain_pvalues(
            corr_case_f,
            group_cols=["variable", "lead", "member_count"],
            sum_cols=CORR_SUM_COLUMNS,
            metric_func=corr_metrics_from_sums,
            sign_metric="corr_diff",
            repeats=int(args.table_bootstrap_repeats),
            rng=rng,
        )
        if not corr_p.empty:
            corr_p = corr_p.rename(
                columns={
                    "bootstrap_repeats_for_p": "corr_p_bootstrap_repeats",
                    "corr_diff_p_gt0": "corr_gain_p_gt0",
                    "corr_diff_p_le0": "corr_gain_p_le0",
                    "corr_diff_p_two_sided": "corr_gain_p_two_sided",
                }
            )
            corr_table = corr_table.merge(corr_p, on=corr_key, how="left")

    bss_key = ["variable", "lead", "member_count"]
    bss_value_cols = [
        col for col in ("model_bss_mean", "geos_bss_mean", "bss_gain_mean")
        if col in bss.columns
    ]
    bss_table = bss[[*bss_key, *bss_value_cols]].copy()
    bss_gain = value_ci_table(bss, bss_boot_f, metric="bss_gain_x100", group_cols=bss_key)
    bss_table = bss_table.merge(bss_gain, on=bss_key, how="left")

    bss_p = pd.DataFrame()
    if not bss_case.empty:
        bss_case_f = filtered_summary(bss_case, variables, leads, sample_sizes)
        bss_p = bootstrap_gain_pvalues(
            bss_case_f,
            group_cols=bss_key,
            sum_cols=BSS_SUM_COLUMNS,
            metric_func=bss_metrics_from_sums,
            sign_metric="bss_gain",
            repeats=int(args.table_bootstrap_repeats),
            rng=rng,
        )
    elif not bss_boot_f.empty and {"bss_gain_p_gt0", "bss_gain_p_le0", *bss_key} <= set(bss_boot_f.columns):
        bss_p = bss_boot_f[[*bss_key, "bss_gain_p_gt0", "bss_gain_p_le0"]].copy()
        bss_p["bss_gain_p_two_sided"] = 2.0 * np.minimum(bss_p["bss_gain_p_gt0"], bss_p["bss_gain_p_le0"])
        bss_p["bss_gain_p_two_sided"] = bss_p["bss_gain_p_two_sided"].clip(upper=1.0)
    if not bss_p.empty:
        bss_p = bss_p.rename(columns={"bootstrap_repeats_for_p": "bss_p_bootstrap_repeats"})
        bss_table = bss_table.merge(bss_p, on=bss_key, how="left")

    table = corr_table.merge(bss_table, on=["variable", "lead", "member_count"], how="outer")
    for col in (
        "corr_gain_p_gt0",
        "corr_gain_p_le0",
        "corr_gain_p_two_sided",
        "bss_gain_p_gt0",
        "bss_gain_p_le0",
        "bss_gain_p_two_sided",
    ):
        if col not in table.columns:
            table[col] = np.nan
    table["variable"] = pd.Categorical(table["variable"], categories=variables, ordered=True)
    table = table.sort_values(["variable", "lead", "member_count"]).reset_index(drop=True)
    table["variable"] = table["variable"].astype(str)
    table = table.rename(
        columns={
            "model_corr_mean": "fimr1p1_flowmatch_corr",
            "geos_corr_mean": "fimr1p1_corr",
            "model_bss_mean": "fimr1p1_flowmatch_bss",
            "geos_bss_mean": "fimr1p1_bss",
        }
    )
    return table


def write_and_print_tables(table: pd.DataFrame, args: argparse.Namespace) -> list[Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ensemble_extreme_spatial_corr_{args.corr_target}_bss_combined"
    full_path = out_dir / f"{stem}_table.csv"
    write_csv(table, full_path)

    member = int(args.table_member_count)
    member_table = table[table["member_count"].astype(int).eq(member)].copy()
    member_path = out_dir / f"{stem}_member{member}_table.csv"
    write_csv(member_table, member_path)

    print(
        "\nP-value convention: p_le0 is the one-sided bootstrap probability that "
        f"{MODEL_LABEL} gain over {BASELINE_LABEL} <= 0; "
        "p_two_sided = 2*min(P(gain>0), P(gain<=0))."
    )
    display_cols = [
        "variable",
        "lead",
        "member_count",
        "fimr1p1_flowmatch_corr",
        "fimr1p1_corr",
        "corr_gain_pct",
        "corr_gain_pct_ci_low",
        "corr_gain_pct_ci_high",
        "corr_gain_p_le0",
        "corr_gain_p_two_sided",
        "fimr1p1_flowmatch_bss",
        "fimr1p1_bss",
        "bss_gain_x100",
        "bss_gain_x100_ci_low",
        "bss_gain_x100_ci_high",
        "bss_gain_p_le0",
        "bss_gain_p_two_sided",
    ]
    display_cols = [col for col in display_cols if col in table.columns]
    to_print = table if args.print_all_table else member_table
    label = "all member counts" if args.print_all_table else f"member count {member}"
    print(f"\nCombined correlation/BSS gain table ({label}):")
    if to_print.empty:
        print("  No rows matched the table filter.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 220):
            print(to_print[display_cols].to_string(index=False, float_format=lambda value: f"{value:8.3f}"))
    return [full_path, member_path]


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


def style_twin_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.0, colors="#6b7782")
    ax.yaxis.label.set_color("#6b7782")


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
    corr_raw_axes = []
    bss_raw_axes = []
    all_corr_raw_values: list[float] = []
    all_bss_raw_values: list[float] = []

    for vi, variable in enumerate(variables):
        label = VARIABLE_LABELS.get(variable, variable.upper())
        corr_ax = axes[0, vi]
        bss_ax = axes[1, vi]
        corr_raw_ax = corr_ax.twinx()
        bss_raw_ax = bss_ax.twinx()
        corr_raw_axes.append(corr_raw_ax)
        bss_raw_axes.append(bss_raw_ax)
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
                raw = c_line["geos_corr_mean"].to_numpy(dtype=float)
                all_corr_raw_values.extend(raw[np.isfinite(raw)].tolist())
                raw_val = float(raw[-1])
                x_max = float(np.max(x))
                x_min = float(np.min(x))
                x_segment_start = x_max - 0.07 * (x_max - x_min)
                corr_raw_ax.plot(
                    [x_segment_start, x_max],
                    [raw_val, raw_val],
                    color=color,
                    lw=1.5,
                    zorder=5,
                )
                corr_raw_ax.plot(
                    x_max,
                    raw_val,
                    marker="<",
                    color=color,
                    ms=4.5,
                    clip_on=False,
                    zorder=10,
                )

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
                raw = b_line["geos_bss_mean"].to_numpy(dtype=float)
                all_bss_raw_values.extend(raw[np.isfinite(raw)].tolist())
                raw_val = float(raw[-1])
                x_max = float(np.max(x))
                x_min = float(np.min(x))
                x_segment_start = x_max - 0.07 * (x_max - x_min)
                bss_raw_ax.plot(
                    [x_segment_start, x_max],
                    [raw_val, raw_val],
                    color=color,
                    lw=1.5,
                    zorder=5,
                )
                bss_raw_ax.plot(
                    x_max,
                    raw_val,
                    marker="<",
                    color=color,
                    ms=4.5,
                    clip_on=False,
                    zorder=10,
                )

        style_axis(corr_ax)
        corr_ax.grid(False)
        style_axis(bss_ax)
        bss_ax.grid(False)
        style_twin_axis(corr_raw_ax)
        style_twin_axis(bss_raw_ax)
        if corr_ax.has_data():
            corr_ax.set_ylim(*robust_ylim(corr_values, floor_low=-2.0, floor_high=5.0))
        if bss_ax.has_data():
            bss_ax.set_ylim(*robust_ylim(bss_values, floor_low=-5.0, floor_high=5.0))
        bss_ax.set_xlabel("Generated members", fontsize=9.5)
        if vi == 0:
            corr_ax.set_ylabel(f"{MODEL_LABEL} gain (%)", fontsize=9.5)
            bss_ax.set_ylabel(f"{MODEL_LABEL} - {BASELINE_LABEL} (x100)", fontsize=9.5)
        corr_raw_ax.set_ylabel(f"{BASELINE_LABEL} raw correlation", fontsize=8.5)
        bss_raw_ax.set_ylabel(f"{BASELINE_LABEL} raw BSS", fontsize=8.5)

    if all_corr_raw_values:
        corr_raw_ylim = robust_ylim(all_corr_raw_values, floor_low=0.0, floor_high=1.0)
        for ax in corr_raw_axes:
            ax.set_ylim(*corr_raw_ylim)
    if all_bss_raw_values:
        bss_raw_ylim = robust_ylim(all_bss_raw_values, floor_low=-0.2, floor_high=0.8)
        for ax in bss_raw_axes:
            ax.set_ylim(*bss_raw_ylim)

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
        fig.text(
            0.985,
            0.965 if not args.title else 0.925,
            f"dotted lines: {BASELINE_LABEL} raw values",
            ha="right",
            va="center",
            fontsize=8.1,
            color="#5f6b76",
        )
    if args.title:
        fig.suptitle(str(args.title), fontsize=12.0, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=rect_top, hspace=0.30, wspace=0.22)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path("paper/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    stem = "fig6_member_convergence_corr_bss"
    for folder in (out_dir, paper_dir):
        if args.format in ("png", "both"):
            path = folder / f"{stem}.png"
            fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight")
            outputs.append(path)
        if args.format in ("pdf", "both"):
            path = folder / f"{stem}.pdf"
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
    corr_case = read_optional_csv(corr_dir / "case_member_correlation_sums.csv")
    bss_case = read_optional_csv(bss_dir / "case_member_bss_sums.csv")
    outputs = plot_combined(corr_summary, corr_boot, bss_summary, bss_boot, args)
    table = build_combined_table(
        corr_summary,
        corr_boot,
        corr_case,
        bss_summary,
        bss_boot,
        bss_case,
        args,
    )
    outputs.extend(write_and_print_tables(table, args))
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
