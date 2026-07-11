#!/usr/bin/env python3
"""Build the supplement figures/tables promised in main.tex.

  S1a figS1a_member_convergence_allcase_allgrid : all-case CRPS/RMSE skill vs
      generated members over all grid cells, ALL leads (Sect. 5.3
      "supplement" reference). Input: ensemble_size_summary.csv from an
      all-case run of ml_model/evaluate_ensemble_tests_flow_finalv1_global.py
      (--s1-dir).
  S1b figS1b_member_convergence_allcase_land : same convergence diagnostic over
      land grid cells only (--s1-land-dir).
  S2  figS2_rank_histograms : rank histograms by variable and lead
      (Sect. 5.4 reference). Input: r1_rank_histogram.csv from
      paper/scripts/review_response/r1_fair_verification.py (--s2-csv).
  S3  suppS3_event_catalog_table.tex : LaTeX table of the full event catalog
      (Sect. 5.5 reference). Input: r1_event_catalog_summary.csv from
      r1_event_catalog_summary.py (--s3-csv).
  S4  figS4_event_pr_pakistan : Pakistan Aug-2022 shared-miss maps in the
      Fig. 7/8 layout (Sect. 5.5 reference). Requires the Pakistan event to be
      added to the event catalog and processed by the event evaluator +
      make_contoured_event_plots.py first (--event-dir, --pakistan-event-id).

Missing inputs render as labeled "pending" panels (S1/S2/S4) or are skipped
with a message (S3), so the script always runs end-to-end.

Usage:
  python paper/scripts/make_supplement_figures.py --format both
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_figures as mpf  # noqa: E402  (style + shared helpers)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

S1_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2023_monthly36_all_land_w1wk4_memberboot90_caseboot15/all",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_allcase",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2024_e90_s50_allcase",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2024_e90_s50",
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2023_e90_s50",
]
S1_LAND_DIR_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2023_monthly36_all_land_w1wk4_memberboot90_caseboot15/land",
]
S2_CSV_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/r1_fair_verification/r1_rank_histogram.csv",
]
S3_CSV_CANDIDATES = [
    "ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023/r1_event_catalog_summary.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="paper/figures/supplement")
    p.add_argument("--s1-dir", default=None,
                   help="Dir with all-case all-grid ensemble_size_summary.csv (all leads).")
    p.add_argument("--s1-land-dir", default=None,
                   help="Dir with all-case land-only ensemble_size_summary.csv (all leads).")
    p.add_argument("--s1-member-counts", default="4,6",
                   help="Comma-separated generated member counts to print/save S1 compact CSV reports.")
    p.add_argument("--s2-csv", default=None, help="r1_rank_histogram.csv path.")
    p.add_argument("--s3-csv", default=None, help="r1_event_catalog_summary.csv path.")
    p.add_argument("--event-dir",
                   default="ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023")
    p.add_argument("--pakistan-event-id", default=None,
                   help="Event id for the Pakistan case; auto-detected from *pakistan* NetCDF if omitted.")
    p.add_argument("--format", choices=("pdf", "png", "both"), default="both")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def first_existing(explicit: str | None, candidates: list[str]) -> Path:
    if explicit:
        return Path(explicit)
    for item in candidates:
        if Path(item).exists():
            return Path(item)
    return Path(candidates[0])


def parse_int_list(text: str) -> list[int]:
    return sorted({int(item.strip()) for item in str(text or "").split(",") if item.strip()})


# ---------------------------------------------------------------------------
# S1 — all-case, all-lead member convergence
# ---------------------------------------------------------------------------

def figure_s1(output_dir: Path, formats: list[str], dpi: int, s1_dir: Path,
              stem: str, context_label: str) -> list[Path]:
    df = mpf.read_csv_or_none(s1_dir / "ensemble_size_summary.csv")
    specs = [("crps_skill_pct", "CRPS skill (%)"), ("rmse_skill_pct", "RMSE skill (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True)
    letters = iter("abcd")
    lead_cmap = plt.get_cmap("viridis")

    for vi, variable in enumerate(("pr", "t2m")):
        for mi, (metric, label) in enumerate(specs):
            ax = axes[vi, mi]
            title = f"({next(letters)}) {mpf.VARIABLE_SHORT[variable]} {label}"
            if df is None or df.empty or f"{metric}_mean" not in df.columns:
                mpf.missing_panel(ax, title,
                                  "Missing all-case ensemble_size_summary.csv; run the "
                                  "ensemble tests without the extreme-event subset.")
                continue
            sub = df[df["variable"].astype(str).str.lower().eq(variable)]
            if sub.empty:
                mpf.missing_panel(ax, title, f"No rows for variable {variable}.")
                continue
            leads = sorted(sub["lead"].astype(int).unique())
            for li, lead in enumerate(leads):
                grp = sub[sub["lead"].astype(int).eq(lead)].sort_values("member_count")
                x = grp["member_count"].to_numpy(dtype=float)
                mean = grp[f"{metric}_mean"].to_numpy(dtype=float)
                color = lead_cmap(0.10 + 0.75 * li / max(len(leads) - 1, 1))
                lo_col, hi_col = f"{metric}_p05", f"{metric}_p95"
                if lo_col in grp and hi_col in grp:
                    ax.fill_between(x, grp[lo_col].to_numpy(dtype=float),
                                    grp[hi_col].to_numpy(dtype=float),
                                    color=color, alpha=0.14, lw=0)
                ax.plot(x, mean, color=color, lw=1.7, marker="o", ms=3.4,
                        label=f"Week {lead}")
            ax.axhline(0.0, color="#7a8794", lw=0.9, ls="--")
            mpf.style_axis(ax)
            mpf.panel_title(ax, title)
            if vi == 1:
                ax.set_xlabel("Generated members")
            if mi == 0:
                ax.set_ylabel(mpf.VARIABLE_LABELS[variable])
            if vi == 0 and mi == 1:
                ax.legend(loc="lower right", fontsize=7.5)
    fig.suptitle(context_label, fontsize=10.5, fontweight="bold",
                 color=mpf.TEXT_DARK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return mpf.save_figure(fig, output_dir, stem, formats, dpi)


def write_s1_member_reports(output_dir: Path, s1_dir: Path, stem: str,
                            context_label: str, member_counts: list[int]) -> list[Path]:
    """Print and save compact S1 convergence summaries for requested member counts."""
    if not member_counts:
        return []
    df = mpf.read_csv_or_none(s1_dir / "ensemble_size_summary.csv")
    if df is None or df.empty:
        print(f"S1 report skipped for {context_label}: missing {s1_dir / 'ensemble_size_summary.csv'}")
        return []

    preferred_cols = [
        "variable",
        "lead",
        "member_count",
        "n_member_repeats",
        "n_case_rows",
        "crps_skill_pct_mean",
        "crps_skill_pct_p05",
        "crps_skill_pct_p50",
        "crps_skill_pct_p95",
        "rmse_skill_pct_mean",
        "rmse_skill_pct_p05",
        "rmse_skill_pct_p50",
        "rmse_skill_pct_p95",
        "model_crps_mean",
        "geos_crps_mean",
        "model_rmse_mean",
        "geos_rmse_mean",
        "model_spread_rmse_ratio_mean",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for member_count in member_counts:
        subset = df[df["member_count"].astype(int).eq(int(member_count))].copy()
        path = output_dir / f"{stem}_member{int(member_count)}_report.csv"
        if subset.empty:
            print(
                f"\nS1 {context_label} ensemble size {int(member_count)} report: "
                "no rows found. Rerun the evaluator with --sample_sizes including "
                f"{int(member_count)}."
            )
            pd.DataFrame(columns=[col for col in preferred_cols if col in df.columns]).to_csv(path, index=False)
            print(f"Wrote empty {path}")
            written.append(path)
            continue

        report_cols = [col for col in preferred_cols if col in subset.columns]
        report = subset[report_cols].sort_values(["variable", "lead"]).reset_index(drop=True)
        report.to_csv(path, index=False)
        written.append(path)

        display_cols = [
            col
            for col in [
                "variable",
                "lead",
                "crps_skill_pct_mean",
                "crps_skill_pct_p05",
                "crps_skill_pct_p95",
                "rmse_skill_pct_mean",
                "rmse_skill_pct_p05",
                "rmse_skill_pct_p95",
                "model_spread_rmse_ratio_mean",
            ]
            if col in report.columns
        ]
        display = report[display_cols].copy()
        for col in display.columns:
            if col not in {"variable", "lead"}:
                display[col] = display[col].astype(float).round(3)
        print(f"\nS1 {context_label} ensemble size {int(member_count)} report:")
        print(display.to_string(index=False))
        print(f"Wrote {path}")

    return written


# ---------------------------------------------------------------------------
# S2 — rank histograms by variable and lead
# ---------------------------------------------------------------------------

def figure_s2(output_dir: Path, formats: list[str], dpi: int, s2_csv: Path) -> list[Path]:
    df = mpf.read_csv_or_none(s2_csv)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0), sharey="row")
    letters = iter("abcdefgh")

    for vi, variable in enumerate(("pr", "t2m")):
        for li, lead in enumerate(mpf.LEADS):
            ax = axes[vi, li]
            title = f"({next(letters)}) {mpf.VARIABLE_SHORT[variable]} week {lead}"
            if df is None or df.empty:
                mpf.missing_panel(ax, title,
                                  "Missing r1_rank_histogram.csv; run r1_fair_verification.py "
                                  "(rank component).")
                continue
            sub = df[df["variable"].astype(str).str.lower().eq(variable)
                     & df["lead"].astype(int).eq(lead)].sort_values("rank_bin")
            if sub.empty:
                mpf.missing_panel(ax, title, f"No rank rows for {variable} week {lead}.")
                continue
            freq = sub["frequency"].to_numpy(dtype=float)
            bins = sub["rank_bin"].to_numpy(dtype=int)
            n_bins = len(bins)
            color = mpf.C_PR if variable == "pr" else mpf.C_T2M
            ax.bar(bins, freq, width=0.92, color=color, edgecolor="none")
            ax.axhline(1.0 / max(n_bins, 1), color="#7a8794", lw=1.0, ls="--")
            mpf.style_axis(ax)
            mpf.panel_title(ax, title)
            ax.set_xlim(-1, n_bins)
            if vi == 1:
                ax.set_xlabel("Obs rank in ensemble")
            if li == 0:
                ax.set_ylabel("Frequency")
    fig.tight_layout()
    return mpf.save_figure(fig, output_dir, "figS2_rank_histograms", formats, dpi)


# ---------------------------------------------------------------------------
# S3 — event-catalog LaTeX table
# ---------------------------------------------------------------------------

def latex_escape(text: str) -> str:
    for src, dst in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        text = text.replace(src, dst)
    return text


def table_s3(output_dir: Path, s3_csv: Path) -> Path | None:
    df = mpf.read_csv_or_none(s3_csv)
    if df is None or df.empty:
        print(f"S3 skipped: {s3_csv} not found (run r1_event_catalog_summary.py first).")
        return None
    cols = [c for c in ("event_name", "variable", "n_init_lead_pairs",
                        "event_crps_skill", "bss_gain_raw", "bss_gain_cal") if c in df.columns]
    header = {"event_name": "Event", "variable": "Var.",
              "n_init_lead_pairs": "Init/lead pairs",
              "event_crps_skill": "Event CRPS skill (\\%)",
              "bss_gain_raw": "BSS gain", "bss_gain_cal": "Cal. BSS gain"}
    lines = [
        "% Auto-generated by make_supplement_figures.py — full event-catalog summary (S3)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Full event-catalog skill summary. Every catalog entry is listed; "
        "featured main-text cases are a subset of this table.}",
        "\\label{tab:supp-event-catalog}",
        "\\small",
        "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
        "\\toprule",
        " & ".join(header[c] for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.sort_values(["variable", "event_name"]).iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float) and not pd.isna(val):
                cells.append(f"{val:+.2f}" if "skill" in c or "gain" in c else f"{val:.0f}")
            else:
                cells.append(latex_escape(str(val)) if not pd.isna(val) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "suppS3_event_catalog_table.tex"
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# S4 — Pakistan shared-miss maps
# ---------------------------------------------------------------------------

def find_pakistan_event_id(event_dir: Path, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    hits = glob.glob(str(event_dir / "plots" / "spatial_maps" / "*pakistan*_lead4_spatial_data.nc"))
    if hits:
        name = Path(hits[0]).name
        return name.replace("_lead4_spatial_data.nc", "")
    return None


def figure_s4(output_dir: Path, formats: list[str], dpi: int,
              event_dir: Path, event_id: str | None) -> list[Path]:
    if event_id is None:
        print("S4: no Pakistan event NetCDF found. Add the event to the catalog, run the "
              "event evaluator and make_contoured_event_plots.py, then rerun this script "
              "(or pass --pakistan-event-id).")
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        mpf.missing_panel(ax, "Pakistan Aug-2022 shared miss",
                          "Event products pending; see script message.")
        return mpf.save_figure(fig, output_dir, "figS4_event_pr_pakistan", formats, dpi)
    return mpf.figure_event_case(output_dir, formats, dpi, event_dir,
                                 event_id, "figS4_event_pr_pakistan", is_t2m=False)


def main() -> None:
    mpf.set_style()
    args = parse_args()
    formats = mpf.output_formats(args.format)
    output_dir = Path(args.output_dir)
    s1_member_counts = parse_int_list(args.s1_member_counts)

    s1_dir = first_existing(args.s1_dir, S1_DIR_CANDIDATES)
    s1_land_dir = first_existing(args.s1_land_dir, S1_LAND_DIR_CANDIDATES)
    s2_csv = first_existing(args.s2_csv, S2_CSV_CANDIDATES)
    s3_csv = first_existing(args.s3_csv, S3_CSV_CANDIDATES)
    event_dir = Path(args.event_dir)
    pak_id = find_pakistan_event_id(event_dir, args.pakistan_event_id)

    print("Supplement input locations")
    print(f"  S1a all-grid dir: {s1_dir}")
    print(f"  S1b land dir    : {s1_land_dir}")
    print(f"  S2 rank csv     : {s2_csv}")
    print(f"  S3 catalog csv  : {s3_csv}")
    print(f"  S4 event id     : {pak_id}")

    written: list[Path] = []
    s1a_stem = "figS1a_member_convergence_allcase_allgrid"
    s1b_stem = "figS1b_member_convergence_allcase_land"
    written.extend(figure_s1(
        output_dir, formats, args.dpi, s1_dir,
        s1a_stem,
        "All-case ensemble convergence, all grid cells"))
    written.extend(write_s1_member_reports(
        output_dir, s1_dir, s1a_stem,
        "all-grid", s1_member_counts))
    written.extend(figure_s1(
        output_dir, formats, args.dpi, s1_land_dir,
        s1b_stem,
        "All-case ensemble convergence, land grid cells"))
    written.extend(write_s1_member_reports(
        output_dir, s1_land_dir, s1b_stem,
        "land-only", s1_member_counts))
    written.extend(figure_s2(output_dir, formats, args.dpi, s2_csv))
    s3 = table_s3(output_dir, s3_csv)
    if s3 is not None:
        written.append(s3)
    written.extend(figure_s4(output_dir, formats, args.dpi, event_dir, pak_id))

    print(f"\nWrote {len(written)} supplement files:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
