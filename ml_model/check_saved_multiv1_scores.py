#!/usr/bin/env python3
"""
Quick saved-forecast verification check for 24 monthly init dates across 2020-2021.

The script selects one common init date per calendar month (2020-01 through
2021-12), compares saved ML-120 and native GEOS ensembles against the already
processed observation targets, and reports area-weighted CRPS and RMSE by lead.
"""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml


PR_MAX_VALID_MM_DAY = 100.0

VAR_SPECS = {
    "pr": {
        "label": "PR",
        "unit": "mm/day",
        "obs_template": "gpcp_weekly_{year}.zarr",
        "var_candidates": ["pr", "precip", "PRECTOT", "flux_precip", "target", "total_precipitation"],
    },
    "tas": {
        "label": "T2M",
        "unit": "K",
        "obs_template": "t2m_weekly_{year}.zarr",
        "var_candidates": ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Check saved ML-120 vs GEOS scores on one init date per month for 2020-2021.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing geos_subc_<year>.zarr, t2m_weekly_<year>.zarr, and gpcp_weekly_<year>.zarr.",
    )
    parser.add_argument(
        "--ml_dir",
        type=str,
        default="dataprocess/gen_multiv1",
        help="Directory containing generated ML yearly zarr stores like 2020.zarr and 2021.zarr.",
    )
    parser.add_argument("--start_year", type=int, default=2020)
    parser.add_argument("--end_year", type=int, default=2021)
    parser.add_argument(
        "--geos_member_count",
        type=int,
        default=4,
        help="Maximum number of GEOS members to use for scoring.",
    )
    parser.add_argument(
        "--ml_member_count",
        type=int,
        default=None,
        help="Optional cap on ML members to use for scoring. Default uses all saved ML members.",
    )
    parser.add_argument(
        "--sample_chunk_size",
        type=int,
        default=2,
        help="Number of selected init dates to load per chunk.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_output_flowmulti/check_saved_scores_24months",
        help="Directory to save the summary CSV and text report.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def choose_name(items: Iterable[str], candidates: Sequence[str], label: str) -> str:
    item_set = set(items)
    for name in candidates:
        if name in item_set:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {sorted(item_set)}")


def choose_data_var(ds: xr.Dataset, candidates: Sequence[str], label: str) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {list(ds.data_vars)}")


def open_zarr_required(path: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_zarr(path, consolidated=False, chunks=None)


def sanitize_pr_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, np.nan)
    values = np.where(values < 0.0, 0.0, values)
    values = np.where(values > PR_MAX_VALID_MM_DAY, np.nan, values)
    return values


def infer_layout(ds: xr.Dataset, kind: str) -> Dict[str, str]:
    s_dim = choose_name(ds.dims, ["S", "time", "init_time"], f"{kind} init dimension")
    lead_dim = choose_name(ds.dims, ["L", "lead", "lead_time"], f"{kind} lead dimension")
    y_dim = choose_name(set(ds.dims) | set(ds.coords), ["Y", "latitude", "lat", "y"], f"{kind} latitude dimension")
    x_dim = choose_name(set(ds.dims) | set(ds.coords), ["X", "longitude", "lon", "x"], f"{kind} longitude dimension")
    member_dim = None
    for candidate in ["M", "member", "ensemble", "ensemble_member"]:
        if candidate in ds.dims:
            member_dim = candidate
            break
    return {
        "s_dim": s_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "member_dim": member_dim,
    }


def prepare_common_dates(*datasets_and_layouts) -> List[pd.Timestamp]:
    common = None
    for ds, layout in datasets_and_layouts:
        dates = set(pd.to_datetime(ds[layout["s_dim"]].values).normalize())
        common = dates if common is None else (common & dates)
    return [pd.Timestamp(item) for item in sorted(common)] if common else []


def exact_indices_for_dates(s_values: np.ndarray, dates: Sequence[pd.Timestamp]) -> List[int]:
    s_dates = pd.to_datetime(s_values).normalize()
    index_map = {pd.Timestamp(date): idx for idx, date in enumerate(s_dates)}
    missing = [date for date in dates if pd.Timestamp(date) not in index_map]
    if missing:
        missing_str = ", ".join(pd.Timestamp(date).strftime("%Y-%m-%d") for date in missing[:5])
        raise ValueError(f"Missing requested dates in dataset: {missing_str}")
    return [int(index_map[pd.Timestamp(date)]) for date in dates]


def extract_var_chunk(
    ds: xr.Dataset,
    layout: Dict[str, str],
    var_name: str,
    s_indices: Sequence[int],
    lead_idx: int,
    var_key: str,
    max_members: int = None,
) -> np.ndarray:
    da = ds[var_name].isel({layout["s_dim"]: list(s_indices), layout["lead_dim"]: int(lead_idx)})
    if layout["member_dim"] is not None and max_members is not None:
        da = da.isel({layout["member_dim"]: slice(0, int(max_members))})
    if layout["member_dim"] is None:
        da = da.transpose(layout["s_dim"], layout["y_dim"], layout["x_dim"])
    else:
        da = da.transpose(layout["s_dim"], layout["member_dim"], layout["y_dim"], layout["x_dim"])
    values = np.asarray(da.values, dtype=np.float64)
    if var_key == "pr":
        values = sanitize_pr_values(values)
    return values


def get_area_weights(lat_values: np.ndarray) -> np.ndarray:
    lat_values = np.asarray(lat_values, dtype=np.float64)
    return np.clip(np.cos(np.deg2rad(lat_values)), 0.0, None)


def weighted_mean(values: np.ndarray, lat_weights: np.ndarray, valid_mask: np.ndarray) -> float:
    weight_grid = np.broadcast_to(lat_weights[:, None], valid_mask.shape)
    weights = weight_grid[valid_mask]
    if weights.size == 0:
        return float("nan")
    return float(np.sum(values[valid_mask] * weights) / (np.sum(weights) + 1e-8))


def compute_area_weighted_rmse(forecast_mean: np.ndarray, obs: np.ndarray, lat_weights: np.ndarray) -> float:
    forecast_mean = np.asarray(forecast_mean, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(forecast_mean) & np.isfinite(obs)
    if not np.any(valid):
        return float("nan")
    err_sq = (forecast_mean - obs) ** 2
    mean_err_sq = weighted_mean(err_sq, lat_weights, valid)
    return float(np.sqrt(mean_err_sq)) if np.isfinite(mean_err_sq) else float("nan")


def compute_area_weighted_crps(ensemble: np.ndarray, obs: np.ndarray, lat_weights: np.ndarray) -> float:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(obs) & np.all(np.isfinite(ensemble), axis=0)
    if not np.any(valid):
        return float("nan")

    ens_valid = ensemble[:, valid]
    obs_valid = obs[valid]
    ens_sorted = np.sort(ens_valid, axis=0)
    e_count = ens_sorted.shape[0]
    mae_term = np.mean(np.abs(ens_sorted - obs_valid[None, :]), axis=0)
    coeff = (2.0 * np.arange(1, e_count + 1) - e_count - 1.0)[:, None]
    spread_term = np.sum(coeff * ens_sorted, axis=0) / float(e_count * e_count)
    crps_vals = mae_term - spread_term

    valid_mask = valid
    crps_map = np.full(obs.shape, np.nan, dtype=np.float64)
    crps_map[valid_mask] = crps_vals
    return weighted_mean(crps_map, lat_weights, valid_mask)


def select_one_date_per_month(common_dates: Sequence[pd.Timestamp], start_year: int, end_year: int) -> List[pd.Timestamp]:
    by_month: Dict[Tuple[int, int], List[pd.Timestamp]] = {}
    for date in common_dates:
        by_month.setdefault((int(date.year), int(date.month)), []).append(pd.Timestamp(date))

    selected = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            key = (year, month)
            if key not in by_month:
                raise ValueError(f"No common init date found for {year}-{month:02d}")
            choices = sorted(by_month[key])
            target_day = 15
            picked = min(choices, key=lambda item: (abs(int(item.day) - target_day), int(item.day)))
            selected.append(picked)
    return selected


def write_csv(path: str, fieldnames: Sequence[str], rows: List[Dict[str, object]]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    args.data_dir = args.data_dir or config["data_dir"]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 88)
    print("CHECK SAVED MULTIV1 SCORES")
    print(f"Years      : {args.start_year}-{args.end_year}")
    print(f"Data Dir   : {args.data_dir}")
    print(f"ML Dir     : {args.ml_dir}")
    print(f"Output Dir : {os.path.abspath(args.output_dir)}")
    print("=" * 88)

    selected_dates_by_year: Dict[int, List[pd.Timestamp]] = {}

    for year in range(args.start_year, args.end_year + 1):
        ml_path = os.path.join(args.ml_dir, f"{year}.zarr")
        geos_path = os.path.join(args.data_dir, f"geos_subc_{year}.zarr")
        pr_path = os.path.join(args.data_dir, f"gpcp_weekly_{year}.zarr")
        tas_path = os.path.join(args.data_dir, f"t2m_weekly_{year}.zarr")

        ml_ds = open_zarr_required(ml_path)
        geos_ds = open_zarr_required(geos_path)
        pr_ds = open_zarr_required(pr_path)
        tas_ds = open_zarr_required(tas_path)
        try:
            ml_layout = infer_layout(ml_ds, f"ML {year}")
            geos_layout = infer_layout(geos_ds, f"GEOS {year}")
            pr_layout = infer_layout(pr_ds, f"PR OBS {year}")
            tas_layout = infer_layout(tas_ds, f"T2M OBS {year}")
            common_dates = prepare_common_dates(
                (ml_ds, ml_layout),
                (geos_ds, geos_layout),
                (pr_ds, pr_layout),
                (tas_ds, tas_layout),
            )
            selected = select_one_date_per_month(
                [date for date in common_dates if int(date.year) == year],
                start_year=year,
                end_year=year,
            )
            selected_dates_by_year[year] = selected
        finally:
            ml_ds.close()
            geos_ds.close()
            pr_ds.close()
            tas_ds.close()

    all_selected_dates = [date for year in range(args.start_year, args.end_year + 1) for date in selected_dates_by_year[year]]
    print("Selected init dates:")
    for date in all_selected_dates:
        print(f"  {date.strftime('%Y-%m-%d')}")

    sample_rows: List[Dict[str, object]] = []

    for year in range(args.start_year, args.end_year + 1):
        selected_dates = selected_dates_by_year[year]
        ml_path = os.path.join(args.ml_dir, f"{year}.zarr")
        geos_path = os.path.join(args.data_dir, f"geos_subc_{year}.zarr")
        pr_path = os.path.join(args.data_dir, f"gpcp_weekly_{year}.zarr")
        tas_path = os.path.join(args.data_dir, f"t2m_weekly_{year}.zarr")

        print(f"\n[{year}] Opening datasets for {len(selected_dates)} selected init dates")
        ml_ds = open_zarr_required(ml_path)
        geos_ds = open_zarr_required(geos_path)
        pr_ds = open_zarr_required(pr_path)
        tas_ds = open_zarr_required(tas_path)
        try:
            ml_layout = infer_layout(ml_ds, f"ML {year}")
            geos_layout = infer_layout(geos_ds, f"GEOS {year}")
            pr_layout = infer_layout(pr_ds, f"PR OBS {year}")
            tas_layout = infer_layout(tas_ds, f"T2M OBS {year}")

            ml_pr_var = choose_data_var(ml_ds, VAR_SPECS["pr"]["var_candidates"], f"ML PR {year}")
            ml_tas_var = choose_data_var(ml_ds, VAR_SPECS["tas"]["var_candidates"], f"ML T2M {year}")
            geos_pr_var = choose_data_var(geos_ds, VAR_SPECS["pr"]["var_candidates"], f"GEOS PR {year}")
            geos_tas_var = choose_data_var(geos_ds, VAR_SPECS["tas"]["var_candidates"], f"GEOS T2M {year}")
            pr_obs_var = choose_data_var(pr_ds, VAR_SPECS["pr"]["var_candidates"], f"OBS PR {year}")
            tas_obs_var = choose_data_var(tas_ds, VAR_SPECS["tas"]["var_candidates"], f"OBS T2M {year}")

            ml_idx = exact_indices_for_dates(ml_ds[ml_layout["s_dim"]].values, selected_dates)
            geos_idx = exact_indices_for_dates(geos_ds[geos_layout["s_dim"]].values, selected_dates)
            pr_idx = exact_indices_for_dates(pr_ds[pr_layout["s_dim"]].values, selected_dates)
            tas_idx = exact_indices_for_dates(tas_ds[tas_layout["s_dim"]].values, selected_dates)

            lat_values = np.asarray(
                tas_ds[tas_layout["y_dim"]].values if tas_layout["y_dim"] in tas_ds.coords else np.arange(tas_ds.sizes[tas_layout["y_dim"]]),
                dtype=np.float64,
            )
            lat_weights = get_area_weights(lat_values)

            total_chunks = (len(selected_dates) + args.sample_chunk_size - 1) // args.sample_chunk_size
            for chunk_start in range(0, len(selected_dates), args.sample_chunk_size):
                chunk_end = min(len(selected_dates), chunk_start + args.sample_chunk_size)
                chunk_number = chunk_start // args.sample_chunk_size + 1
                lo = selected_dates[chunk_start].strftime("%Y-%m-%d")
                hi = selected_dates[chunk_end - 1].strftime("%Y-%m-%d")
                print(f"[{year}] Chunk {chunk_number}/{total_chunks}: {lo} .. {hi}")

                ml_chunk_idx = ml_idx[chunk_start:chunk_end]
                geos_chunk_idx = geos_idx[chunk_start:chunk_end]
                pr_chunk_idx = pr_idx[chunk_start:chunk_end]
                tas_chunk_idx = tas_idx[chunk_start:chunk_end]
                chunk_dates = selected_dates[chunk_start:chunk_end]

                for lead_idx in range(4):
                    ml_pr = extract_var_chunk(
                        ml_ds, ml_layout, ml_pr_var, ml_chunk_idx, lead_idx, "pr", max_members=args.ml_member_count
                    )
                    ml_tas = extract_var_chunk(
                        ml_ds, ml_layout, ml_tas_var, ml_chunk_idx, lead_idx, "tas", max_members=args.ml_member_count
                    )
                    geos_pr = extract_var_chunk(
                        geos_ds, geos_layout, geos_pr_var, geos_chunk_idx, lead_idx, "pr", max_members=args.geos_member_count
                    )
                    geos_tas = extract_var_chunk(
                        geos_ds, geos_layout, geos_tas_var, geos_chunk_idx, lead_idx, "tas", max_members=args.geos_member_count
                    )
                    obs_pr = extract_var_chunk(pr_ds, pr_layout, pr_obs_var, pr_chunk_idx, lead_idx, "pr")
                    obs_tas = extract_var_chunk(tas_ds, tas_layout, tas_obs_var, tas_chunk_idx, lead_idx, "tas")

                    for local_idx, init_date in enumerate(chunk_dates):
                        ml_pr_mean = np.nanmean(ml_pr[local_idx], axis=0)
                        ml_tas_mean = np.nanmean(ml_tas[local_idx], axis=0)
                        geos_pr_mean = np.nanmean(geos_pr[local_idx], axis=0)
                        geos_tas_mean = np.nanmean(geos_tas[local_idx], axis=0)

                        sample_rows.append(
                            {
                                "init_date": init_date.strftime("%Y-%m-%d"),
                                "year": int(init_date.year),
                                "month": int(init_date.month),
                                "lead_week": lead_idx + 1,
                                "variable": "pr",
                                "variable_label": "PR",
                                "ml_crps": compute_area_weighted_crps(ml_pr[local_idx], obs_pr[local_idx], lat_weights),
                                "ml_rmse": compute_area_weighted_rmse(ml_pr_mean, obs_pr[local_idx], lat_weights),
                                "geos_crps": compute_area_weighted_crps(geos_pr[local_idx], obs_pr[local_idx], lat_weights),
                                "geos_rmse": compute_area_weighted_rmse(geos_pr_mean, obs_pr[local_idx], lat_weights),
                            }
                        )
                        sample_rows.append(
                            {
                                "init_date": init_date.strftime("%Y-%m-%d"),
                                "year": int(init_date.year),
                                "month": int(init_date.month),
                                "lead_week": lead_idx + 1,
                                "variable": "tas",
                                "variable_label": "T2M",
                                "ml_crps": compute_area_weighted_crps(ml_tas[local_idx], obs_tas[local_idx], lat_weights),
                                "ml_rmse": compute_area_weighted_rmse(ml_tas_mean, obs_tas[local_idx], lat_weights),
                                "geos_crps": compute_area_weighted_crps(geos_tas[local_idx], obs_tas[local_idx], lat_weights),
                                "geos_rmse": compute_area_weighted_rmse(geos_tas_mean, obs_tas[local_idx], lat_weights),
                            }
                        )
        finally:
            ml_ds.close()
            geos_ds.close()
            pr_ds.close()
            tas_ds.close()

    summary_rows: List[Dict[str, object]] = []
    report_lines = [
        "Saved multiv1 monthly check (24 init dates across 2020-2021)",
        "Selection rule: one common init date per month, chosen nearest day 15",
        "Variables: PR and T2M",
        "",
        "Selected init dates:",
    ]
    report_lines.extend([f"  {date.strftime('%Y-%m-%d')}" for date in all_selected_dates])
    report_lines.append("")

    for var_key in ["pr", "tas"]:
        label = VAR_SPECS[var_key]["label"]
        unit = VAR_SPECS[var_key]["unit"]
        rows = [row for row in sample_rows if row["variable"] == var_key]
        report_lines.append(f"[{label}]")
        print(f"\n[{label}] By-lead averages across {len(all_selected_dates)} selected init dates")
        for lead_idx in range(1, 5):
            lead_rows = [row for row in rows if int(row["lead_week"]) == lead_idx]
            ml_crps = float(np.nanmean([row["ml_crps"] for row in lead_rows]))
            ml_rmse = float(np.nanmean([row["ml_rmse"] for row in lead_rows]))
            geos_crps = float(np.nanmean([row["geos_crps"] for row in lead_rows]))
            geos_rmse = float(np.nanmean([row["geos_rmse"] for row in lead_rows]))
            summary_rows.append(
                {
                    "variable": var_key,
                    "variable_label": label,
                    "lead_week": lead_idx,
                    "n_samples": len(lead_rows),
                    "ml_crps_mean": ml_crps,
                    "ml_rmse_mean": ml_rmse,
                    "geos_crps_mean": geos_crps,
                    "geos_rmse_mean": geos_rmse,
                }
            )
            print(
                f"  W{lead_idx}: "
                f"ML CRPS={ml_crps:.4f}, ML RMSE={ml_rmse:.4f} {unit} | "
                f"GEOS CRPS={geos_crps:.4f}, GEOS RMSE={geos_rmse:.4f} {unit}"
            )
            report_lines.append(
                f"  W{lead_idx}: ML CRPS={ml_crps:.4f}, ML RMSE={ml_rmse:.4f} {unit} | "
                f"GEOS CRPS={geos_crps:.4f}, GEOS RMSE={geos_rmse:.4f} {unit}"
            )

        all_ml_crps = float(np.nanmean([row["ml_crps"] for row in rows]))
        all_ml_rmse = float(np.nanmean([row["ml_rmse"] for row in rows]))
        all_geos_crps = float(np.nanmean([row["geos_crps"] for row in rows]))
        all_geos_rmse = float(np.nanmean([row["geos_rmse"] for row in rows]))
        print(
            f"  All-sample mean: "
            f"ML CRPS={all_ml_crps:.4f}, ML RMSE={all_ml_rmse:.4f} {unit} | "
            f"GEOS CRPS={all_geos_crps:.4f}, GEOS RMSE={all_geos_rmse:.4f} {unit}"
        )
        report_lines.append(
            f"  All-sample mean: ML CRPS={all_ml_crps:.4f}, ML RMSE={all_ml_rmse:.4f} {unit} | "
            f"GEOS CRPS={all_geos_crps:.4f}, GEOS RMSE={all_geos_rmse:.4f} {unit}"
        )
        report_lines.append("")

    sample_csv = os.path.join(args.output_dir, "sample_scores.csv")
    summary_csv = os.path.join(args.output_dir, "lead_summary.csv")
    report_txt = os.path.join(args.output_dir, "score_check_report.txt")
    write_csv(
        sample_csv,
        ["init_date", "year", "month", "lead_week", "variable", "variable_label", "ml_crps", "ml_rmse", "geos_crps", "geos_rmse"],
        sample_rows,
    )
    write_csv(
        summary_csv,
        ["variable", "variable_label", "lead_week", "n_samples", "ml_crps_mean", "ml_rmse_mean", "geos_crps_mean", "geos_rmse_mean"],
        summary_rows,
    )
    with open(report_txt, "w") as f:
        f.write("\n".join(report_lines).rstrip() + "\n")

    print(f"\n✅ Saved sample scores: {sample_csv}")
    print(f"✅ Saved lead summary: {summary_csv}")
    print(f"✅ Saved report: {report_txt}")


if __name__ == "__main__":
    main()
