#!/usr/bin/env python3
"""Run and package the matrix evaluations needed by paper/main.tex.

This is a paper-level wrapper around the heavier evaluation scripts:

  - evaluate_matrix_suite_flow_finalv1_global.py
  - evaluate_regional_matrix_flow_finalv1_global.py

It runs the global matrix suite for the requested evaluation masks, runs the
regional post-processing from the land-focused spatial NetCDF, and exports
compact paper-facing CSV tables plus a manifest mapping outputs to manuscript
figures/tables.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

DEFAULT_FORECAST_DIR = "dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50"
DEFAULT_OUT_ROOT = "ml_output_flow_finalv1_global_noisectx_t2mres/paper_matrix_eval_global_2021_2023"
DEFAULT_LAND_MASK = "ml_model/land_ocean_mask_v6.pt"

PAPER_REQUIREMENTS = [
    {
        "paper_target": "Table headline-results",
        "source": "matrix_summary_metrics.csv",
        "export": "paper_headline_skill_<mask>.csv",
        "metrics": ["CRPS", "RMSE", "MAE", "correlation", "BSS", "calibrated BSS", "spread"],
    },
    {
        "paper_target": "Figure 3 lead-skill curves",
        "source": "matrix_summary_metrics.csv",
        "export": "paper_lead_skill_<mask>.csv",
        "metrics": ["CRPS skill", "RMSE skill", "spread", "calibrated BSS gain"],
    },
    {
        "paper_target": "Figure 4 season-by-lead matrices",
        "source": "matrix_summary_metrics.csv",
        "export": "paper_season_lead_skill_<mask>.csv",
        "metrics": ["CRPS skill", "RMSE skill", "calibrated BSS gain"],
    },
    {
        "paper_target": "Supplement month-by-lead matrices",
        "source": "matrix_summary_metrics.csv",
        "export": "paper_month_lead_skill_<mask>.csv",
        "metrics": ["CRPS", "RMSE", "bias", "correlation", "BSS", "calibrated BSS"],
    },
    {
        "paper_target": "Figure 5 spatial skill maps",
        "source": "matrix_spatial_metrics.nc",
        "export": "paper_spatial_inventory_<mask>.csv",
        "metrics": [
            "GEOS CRPS",
            "ML CRPS",
            "CRPS skill",
            "RMSE skill",
            "bias change",
            "correlation change",
            "spread",
            "BSS",
            "calibrated BSS",
        ],
    },
    {
        "paper_target": "Observed-extreme and probabilistic-event matrices",
        "source": "event_thresholds_and_frequencies.nc, bss_calibration_params.csv",
        "export": "paper_extreme_event_matrix_<mask>.csv",
        "metrics": ["event frequency", "raw BSS", "calibrated BSS", "event-subset CRPS"],
    },
    {
        "paper_target": "Table regional-results",
        "source": "regional_summary_metrics.csv, regional_overall_skill_table.csv",
        "export": "paper_regional_overall_skill.csv",
        "metrics": ["regional CRPS skill", "regional RMSE skill", "regional calibrated BSS gain"],
    },
]

PAPER_METRIC_COLUMNS = [
    "model_crps",
    "geos_crps",
    "model_rmse",
    "geos_rmse",
    "model_mae",
    "geos_mae",
    "model_bias",
    "geos_bias",
    "model_corr",
    "geos_corr",
    "model_bss",
    "geos_bss",
    "model_calibrated_bss",
    "geos_calibrated_bss",
    "model_spread",
    "geos_spread",
    "model_brier",
    "geos_brier",
    "model_brier_calibrated",
    "geos_brier_calibrated",
]

PAPER_WIDE_COLUMNS = [
    "eval_mask",
    "subset",
    "variable",
    "group_type",
    "group_value",
    "lead",
    "lead_label",
    "n_groups",
    "n_forecasts",
    "weight_sum",
    "geos_crps",
    "model_crps",
    "crps_skill_pct",
    "geos_rmse",
    "model_rmse",
    "rmse_skill_pct",
    "geos_mae",
    "model_mae",
    "mae_skill_pct",
    "geos_bias",
    "model_bias",
    "abs_bias_skill_pct",
    "geos_corr",
    "model_corr",
    "corr_diff",
    "geos_bss",
    "model_bss",
    "bss_diff",
    "geos_calibrated_bss",
    "model_calibrated_bss",
    "calibrated_bss_diff",
    "geos_spread",
    "model_spread",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-ready global/regional matrix evaluation bundle."
    )
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--skip-years", default="")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--main-tex", default="paper/main.tex")
    parser.add_argument("--variables", default="pr,t2m")
    parser.add_argument(
        "--eval-masks",
        default="all,land",
        help="Comma-separated matrix masks to run. Use all, land, ocean, or any subset.",
    )
    parser.add_argument("--land-mask-file", default=DEFAULT_LAND_MASK)
    parser.add_argument("--threshold-file", default="")
    parser.add_argument("--threshold-forecast-dir", default="")
    parser.add_argument("--threshold-start-year", default="")
    parser.add_argument("--threshold-end-year", default="")
    parser.add_argument("--threshold-skip-years", default="")
    parser.add_argument("--threshold-grouping", choices=("pooled", "monthly", "seasonal"), default="monthly")
    parser.add_argument("--extreme-quantile-pr", type=float, default=0.95)
    parser.add_argument("--extreme-quantile-t2m", type=float, default=0.95)
    parser.add_argument("--pr-min-threshold", type=float, default=5.0)
    parser.add_argument("--bss-calibration", choices=("logistic_cv", "base_rate", "none"), default="logistic_cv")
    parser.add_argument("--bss-calibration-grouping", choices=("lead_season", "lead", "global"), default="lead_season")
    parser.add_argument("--bss-calibration-bins", type=int, default=41)
    parser.add_argument("--bss-calibration-ridge", type=float, default=1.0)
    parser.add_argument("--bss-calibration-min-weight", type=float, default=100.0)
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument("--map-features", choices=("auto", "cartopy", "plain"), default="auto")
    parser.add_argument("--county-boundaries", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regional-source-mask", default="land")
    parser.add_argument("--regions", default="all")
    parser.add_argument("--regional-plot-metrics", default="crps_skill_pct,rmse_skill_pct,calibrated_bss_diff")
    parser.add_argument("--regional-plot-group-type", choices=("valid_season_lead", "valid_month_lead"), default="valid_season_lead")
    parser.add_argument("--regional-mask-source", choices=("auto", "natural_earth", "box"), default="auto")
    parser.add_argument("--regional-make-maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-regional", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip evaluation commands and rebuild paper CSVs/manifest from existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a manifest without running evaluators or exporting tables.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def mask_out_dir(out_root: Path, eval_mask: str) -> Path:
    return out_root / f"matrix_{eval_mask}"


def optional_arg(cmd: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    text = str(value)
    if text:
        cmd.extend([name, text])


def command_text(cmd: Iterable[object]) -> str:
    return shlex.join(str(item) for item in cmd)


def run_command(cmd: list[str], dry_run: bool) -> dict[str, object]:
    text = command_text(cmd)
    print(text)
    if dry_run:
        return {"command": text, "status": "dry_run"}
    subprocess.run(cmd, check=True)
    return {"command": text, "status": "completed"}


def build_matrix_command(args: argparse.Namespace, eval_mask: str, out_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(SCRIPT_DIR / "evaluate_matrix_suite_flow_finalv1_global.py"),
        "--forecast_dir",
        args.forecast_dir,
        "--start_year",
        str(args.start_year),
        "--end_year",
        str(args.end_year),
        "--skip_years",
        args.skip_years,
        "--out_dir",
        str(out_dir),
        "--variables",
        args.variables,
        "--extreme_quantile_pr",
        str(args.extreme_quantile_pr),
        "--extreme_quantile_t2m",
        str(args.extreme_quantile_t2m),
        "--pr_min_threshold",
        str(args.pr_min_threshold),
        "--threshold_grouping",
        args.threshold_grouping,
        "--eval_mask",
        eval_mask,
        "--bss_calibration",
        args.bss_calibration,
        "--bss_calibration_grouping",
        args.bss_calibration_grouping,
        "--bss_calibration_bins",
        str(args.bss_calibration_bins),
        "--bss_calibration_ridge",
        str(args.bss_calibration_ridge),
        "--bss_calibration_min_weight",
        str(args.bss_calibration_min_weight),
        "--map_features",
        args.map_features,
        "--county_boundaries",
        args.county_boundaries,
    ]
    optional_arg(cmd, "--threshold_file", args.threshold_file)
    optional_arg(cmd, "--threshold_forecast_dir", args.threshold_forecast_dir)
    optional_arg(cmd, "--threshold_start_year", args.threshold_start_year)
    optional_arg(cmd, "--threshold_end_year", args.threshold_end_year)
    optional_arg(cmd, "--threshold_skip_years", args.threshold_skip_years)
    optional_arg(cmd, "--max_runtime_minutes", args.max_runtime_minutes)
    if eval_mask in {"land", "ocean"}:
        optional_arg(cmd, "--land_mask_file", args.land_mask_file)
    if args.make_plots:
        cmd.append("--make_plots")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def build_regional_command(args: argparse.Namespace, source_dir: Path, out_root: Path, source_mask: str) -> list[str]:
    regional_out = out_root / f"regional_{source_mask}"
    cmd = [
        args.python,
        str(SCRIPT_DIR / "evaluate_regional_matrix_flow_finalv1_global.py"),
        "--matrix_spatial_file",
        str(source_dir / "matrix_spatial_metrics.nc"),
        "--metadata_file",
        str(source_dir / "matrix_eval_metadata.json"),
        "--out_dir",
        str(regional_out),
        "--land_mask_file",
        args.land_mask_file,
        "--regions",
        args.regions,
        "--variables",
        args.variables,
        "--subsets",
        "all_data,extreme_events",
        "--mask_source",
        args.regional_mask_source,
        "--plot_metrics",
        args.regional_plot_metrics,
        "--plot_group_type",
        args.regional_plot_group_type,
        "--map_features",
        args.map_features,
        "--county_boundaries",
        args.county_boundaries,
    ]
    if args.regional_make_maps:
        cmd.append("--make_maps")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def skill_pct(model_value: float, geos_value: float) -> float:
    if not np.isfinite(model_value) or not np.isfinite(geos_value) or abs(geos_value) <= 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - model_value / geos_value))


def weighted_average(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    if "weight_sum" in group:
        weights = pd.to_numeric(group["weight_sum"], errors="coerce").to_numpy(dtype=float)
    elif "effective_weight_sum" in group:
        weights = pd.to_numeric(group["effective_weight_sum"], errors="coerce").to_numpy(dtype=float)
    else:
        weights = np.ones_like(values, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def add_derived_metrics(row: dict[str, object]) -> None:
    row["crps_skill_pct"] = skill_pct(float(row.get("model_crps", np.nan)), float(row.get("geos_crps", np.nan)))
    row["rmse_skill_pct"] = skill_pct(float(row.get("model_rmse", np.nan)), float(row.get("geos_rmse", np.nan)))
    row["mae_skill_pct"] = skill_pct(float(row.get("model_mae", np.nan)), float(row.get("geos_mae", np.nan)))
    model_bias = float(row.get("model_bias", np.nan))
    geos_bias = float(row.get("geos_bias", np.nan))
    row["abs_bias_skill_pct"] = skill_pct(abs(model_bias), abs(geos_bias))
    row["corr_diff"] = float(row.get("model_corr", np.nan)) - float(row.get("geos_corr", np.nan))
    row["bss_diff"] = float(row.get("model_bss", np.nan)) - float(row.get("geos_bss", np.nan))
    row["calibrated_bss_diff"] = float(row.get("model_calibrated_bss", np.nan)) - float(
        row.get("geos_calibrated_bss", np.nan)
    )


def aggregate_summary(summary: pd.DataFrame, group_cols: list[str], eval_mask: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame()
    for keys, group in summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"eval_mask": eval_mask, "n_groups": int(len(group))}
        row.update({col: value for col, value in zip(group_cols, keys)})
        if "n_forecasts" in group:
            row["n_forecasts"] = int(pd.to_numeric(group["n_forecasts"], errors="coerce").fillna(0).sum())
        if "weight_sum" in group:
            row["weight_sum"] = float(pd.to_numeric(group["weight_sum"], errors="coerce").fillna(0.0).sum())
        for col in PAPER_METRIC_COLUMNS:
            if col in group:
                row[col] = weighted_average(group, col)
        add_derived_metrics(row)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def headline_long(headline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_specs = [
        ("CRPS", "geos_crps", "model_crps", "crps_skill_pct", "skill_pct"),
        ("RMSE", "geos_rmse", "model_rmse", "rmse_skill_pct", "skill_pct"),
        ("MAE", "geos_mae", "model_mae", "mae_skill_pct", "skill_pct"),
        ("Bias", "geos_bias", "model_bias", "abs_bias_skill_pct", "abs_bias_skill_pct"),
        ("Correlation", "geos_corr", "model_corr", "corr_diff", "ml_minus_geos"),
        ("BSS", "geos_bss", "model_bss", "bss_diff", "ml_minus_geos"),
        ("Calibrated BSS", "geos_calibrated_bss", "model_calibrated_bss", "calibrated_bss_diff", "ml_minus_geos"),
        ("Spread", "geos_spread", "model_spread", None, ""),
    ]
    for _, row in headline.iterrows():
        base = {
            "eval_mask": row.get("eval_mask"),
            "subset": row.get("subset"),
            "variable": row.get("variable"),
            "n_groups": row.get("n_groups"),
            "n_forecasts": row.get("n_forecasts"),
            "weight_sum": row.get("weight_sum"),
        }
        for label, geos_col, model_col, delta_col, delta_units in metric_specs:
            if geos_col not in row or model_col not in row:
                continue
            out = dict(base)
            out.update(
                {
                    "metric": label,
                    "geos": row.get(geos_col),
                    "ml": row.get(model_col),
                    "skill_or_gain": row.get(delta_col) if delta_col else np.nan,
                    "skill_or_gain_units": delta_units,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def write_csv(df: pd.DataFrame, path: Path) -> str | None:
    if df.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"Wrote {path}")
    return str(path)


def export_summary_tables(matrix_dir: Path, tables_dir: Path, eval_mask: str) -> list[str]:
    summary_path = matrix_dir / "matrix_summary_metrics.csv"
    if not summary_path.exists():
        print(f"Missing {summary_path}; skipping paper table export for mask={eval_mask}.")
        return []
    summary = pd.read_csv(summary_path)
    written: list[str] = []

    valid_season = summary[summary["group_type"].eq("valid_season_lead")].copy()
    headline = aggregate_summary(valid_season, ["subset", "variable"], eval_mask)
    path = write_csv(headline_long(headline), tables_dir / f"paper_headline_skill_{eval_mask}.csv")
    if path:
        written.append(path)

    lead = aggregate_summary(valid_season, ["subset", "variable", "lead"], eval_mask)
    lead["lead_label"] = lead["lead"].map(lambda value: f"week{int(value)}" if pd.notna(value) else "")
    path = write_csv(select_existing_columns(lead, PAPER_WIDE_COLUMNS), tables_dir / f"paper_lead_skill_{eval_mask}.csv")
    if path:
        written.append(path)

    season = valid_season.rename(columns={"group_value": "season"}).copy()
    season["eval_mask"] = eval_mask
    season_cols = [col.replace("group_value", "season") for col in PAPER_WIDE_COLUMNS]
    path = write_csv(select_existing_columns(season, season_cols), tables_dir / f"paper_season_lead_skill_{eval_mask}.csv")
    if path:
        written.append(path)

    valid_month = summary[summary["group_type"].eq("valid_month_lead")].rename(columns={"group_value": "month"}).copy()
    valid_month["eval_mask"] = eval_mask
    month_cols = [col.replace("group_value", "month") for col in PAPER_WIDE_COLUMNS]
    path = write_csv(select_existing_columns(valid_month, month_cols), tables_dir / f"paper_month_lead_skill_{eval_mask}.csv")
    if path:
        written.append(path)

    extreme = summary[summary["subset"].eq("extreme_events")].copy()
    extreme["eval_mask"] = eval_mask
    path = write_csv(
        select_existing_columns(extreme, PAPER_WIDE_COLUMNS),
        tables_dir / f"paper_extreme_event_matrix_{eval_mask}.csv",
    )
    if path:
        written.append(path)

    return written


def export_netcdf_inventory(nc_path: Path, out_path: Path, kind: str) -> str | None:
    if not nc_path.exists():
        return None
    try:
        import xarray as xr
    except Exception as exc:
        print(f"xarray unavailable; cannot inventory {nc_path}: {exc}")
        return None
    rows = []
    with xr.open_dataset(nc_path) as ds:
        for name, arr in ds.data_vars.items():
            values = arr.values
            finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else np.asarray([])
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "dims": ",".join(arr.dims),
                    "shape": "x".join(str(size) for size in arr.shape),
                    "finite_count": int(finite.size),
                    "finite_mean": float(np.nanmean(finite)) if finite.size else np.nan,
                    "finite_min": float(np.nanmin(finite)) if finite.size else np.nan,
                    "finite_max": float(np.nanmax(finite)) if finite.size else np.nan,
                    "attrs": json.dumps({key: str(value) for key, value in arr.attrs.items()}, sort_keys=True),
                }
            )
    return write_csv(pd.DataFrame(rows), out_path)


def export_regional_tables(regional_dir: Path, tables_dir: Path) -> list[str]:
    written: list[str] = []
    for source_name, target_name in [
        ("regional_overall_skill_table.csv", "paper_regional_overall_skill.csv"),
        ("regional_lead_season_skill_table.csv", "paper_regional_lead_season_skill.csv"),
        ("regional_summary_metrics.csv", "paper_regional_matrix_full.csv"),
    ]:
        source = regional_dir / source_name
        if not source.exists():
            continue
        df = pd.read_csv(source)
        path = write_csv(df, tables_dir / target_name)
        if path:
            written.append(path)
    return written


def extract_main_tex_mentions(main_tex: Path, out_path: Path) -> str | None:
    if not main_tex.exists():
        print(f"Missing {main_tex}; skipping main.tex mention extract.")
        return None
    keywords = [
        "CRPS",
        "RMSE",
        "BSS",
        "calibrated",
        "spread",
        "bias",
        "correlation",
        "season",
        "month",
        "lead",
        "spatial",
        "regional",
        "extreme",
        "event",
        "tail-risk",
    ]
    rows = []
    for line_no, line in enumerate(main_tex.read_text().splitlines(), start=1):
        if any(keyword.lower() in line.lower() for keyword in keywords):
            rows.append(f"{line_no}: {line.strip()}")
    if not rows:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"Wrote {out_path}")
    return str(out_path)


def write_manifest(
    out_root: Path,
    args: argparse.Namespace,
    commands: list[dict[str, object]],
    paper_tables: list[str],
    main_tex_mentions: str | None,
) -> Path:
    manifest = {
        "description": "Paper matrix evaluation bundle derived from paper/main.tex requirements.",
        "main_tex": str(Path(args.main_tex)),
        "main_tex_mentions": main_tex_mentions,
        "paper_requirements": PAPER_REQUIREMENTS,
        "forecast_dir": args.forecast_dir,
        "years": [args.start_year, args.end_year],
        "skip_years": args.skip_years,
        "variables": parse_list(args.variables),
        "eval_masks": parse_list(args.eval_masks),
        "land_mask_file": args.land_mask_file,
        "threshold_file": args.threshold_file or None,
        "threshold_grouping": args.threshold_grouping,
        "bss_calibration": args.bss_calibration,
        "out_root": str(out_root),
        "commands": commands,
        "paper_tables": paper_tables,
        "notes": [
            "Global matrix evaluator saves scalar CSVs, spatial NetCDF, event thresholds/frequencies, and BSS calibration params.",
            "Regional tables are derived from spatial metric maps and are area/sample-count weighted regional means.",
            "Neighborhood precipitation diagnostics and named-event quantile maps remain in the event catalog/quantile evaluators, not this matrix wrapper.",
        ],
    }
    path = out_root / "paper_matrix_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {path}")
    return path


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    tables_dir = out_root / "paper_tables"
    eval_masks = parse_list(args.eval_masks)
    if not eval_masks:
        raise ValueError("--eval-masks cannot be empty")
    bad_masks = sorted(set(eval_masks) - {"all", "land", "ocean"})
    if bad_masks:
        raise ValueError(f"Unknown eval masks {bad_masks}; expected all, land, or ocean.")
    if any(mask in {"land", "ocean"} for mask in eval_masks) and not args.land_mask_file:
        raise ValueError("--land-mask-file is required for land/ocean matrix evaluation.")
    if any(mask in {"land", "ocean"} for mask in eval_masks) and not Path(args.land_mask_file).exists():
        print(f"Warning: land mask not found locally: {args.land_mask_file}")

    out_root.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, object]] = []
    if not args.skip_matrix and not args.export_only:
        for eval_mask in eval_masks:
            out_dir = mask_out_dir(out_root, eval_mask)
            cmd = build_matrix_command(args, eval_mask, out_dir)
            commands.append(run_command(cmd, args.dry_run))

    regional_source_mask = args.regional_source_mask if args.regional_source_mask in eval_masks else eval_masks[0]
    regional_dir = out_root / f"regional_{regional_source_mask}"
    if not args.skip_regional and not args.export_only:
        source_dir = mask_out_dir(out_root, regional_source_mask)
        cmd = build_regional_command(args, source_dir, out_root, regional_source_mask)
        commands.append(run_command(cmd, args.dry_run))

    paper_tables: list[str] = []
    main_tex_mentions = extract_main_tex_mentions(Path(args.main_tex), tables_dir / "main_tex_matrix_mentions.txt")
    if not args.dry_run:
        for eval_mask in eval_masks:
            matrix_dir = mask_out_dir(out_root, eval_mask)
            paper_tables.extend(export_summary_tables(matrix_dir, tables_dir, eval_mask))
            spatial_path = export_netcdf_inventory(
                matrix_dir / "matrix_spatial_metrics.nc",
                tables_dir / f"paper_spatial_inventory_{eval_mask}.csv",
                "matrix_spatial_metrics",
            )
            if spatial_path:
                paper_tables.append(spatial_path)
            threshold_path = export_netcdf_inventory(
                matrix_dir / "event_thresholds_and_frequencies.nc",
                tables_dir / f"paper_threshold_inventory_{eval_mask}.csv",
                "event_thresholds_and_frequencies",
            )
            if threshold_path:
                paper_tables.append(threshold_path)
        paper_tables.extend(export_regional_tables(regional_dir, tables_dir))

    write_manifest(out_root, args, commands, paper_tables, main_tex_mentions)


if __name__ == "__main__":
    main()
