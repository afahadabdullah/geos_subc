#!/usr/bin/env python3
"""
Build lead-specific weekly climatology from 1999-2021 for:
- ML multiv1 forecasts (ensemble mean per init date first)
- Raw GEOS forecasts (ensemble mean per init date first)
- Observations

The weekly grouping is by ISO init week, not valid week. That makes this a
forecast-system climatology suitable for lead-aware anomaly baselines while
staying aligned with how forecasts are initialized.
"""

import argparse
import os
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Build weekly lead-specific climatology for ML, GEOS, and observations.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing geos_subc_<year>.zarr, gpcp_weekly_<year>.zarr, and t2m_weekly_<year>.zarr.",
    )
    parser.add_argument(
        "--ml_hindcast_dir",
        type=str,
        default="dataprocess/gen_multiv1_hindcast_1999_2019",
        help="Directory containing 1999-2019 ML hindcast yearly Zarr stores.",
    )
    parser.add_argument(
        "--ml_forecast_dir",
        type=str,
        default="dataprocess/gen_multiv1",
        help="Directory containing 2020-2021 ML forecast yearly Zarr stores.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataprocess/clim",
        help="Directory to save climatology Zarr stores.",
    )
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2021)
    parser.add_argument("--overwrite", action="store_true")
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
    return xr.open_zarr(path, consolidated=False)


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
    pr_var = None
    tas_var = None
    try:
        pr_var = choose_data_var(ds, ["pr", "precip", "PRECTOT", "flux_precip", "target", "total_precipitation"], f"{kind} pr variable")
    except KeyError:
        pass
    try:
        tas_var = choose_data_var(ds, ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"], f"{kind} tas variable")
    except KeyError:
        pass
    return {
        "s_dim": s_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "member_dim": member_dim,
        "pr_var": pr_var,
        "tas_var": tas_var,
    }


def init_accumulator(lead_values: np.ndarray, y_values: np.ndarray, x_values: np.ndarray) -> Dict[str, np.ndarray]:
    lead_count = len(lead_values)
    y_count = len(y_values)
    x_count = len(x_values)
    return {
        "pr_sum": np.zeros((53, lead_count, y_count, x_count), dtype=np.float64),
        "pr_count": np.zeros((53, lead_count, y_count, x_count), dtype=np.float64),
        "tas_sum": np.zeros((53, lead_count, y_count, x_count), dtype=np.float64),
        "tas_count": np.zeros((53, lead_count, y_count, x_count), dtype=np.float64),
        "n_init_pr": np.zeros(53, dtype=np.int32),
        "n_init_tas": np.zeros(53, dtype=np.int32),
    }


def update_weekly_accumulator(acc: Dict[str, np.ndarray], values: np.ndarray, init_weeks: np.ndarray, var_key: str):
    # values shape [S, L, Y, X], init_weeks in [1, 53]
    for week in range(1, 54):
        mask = init_weeks == week
        if not np.any(mask):
            continue
        chunk = values[mask]
        valid = np.isfinite(chunk)
        acc[f"{var_key}_sum"][week - 1] += np.where(valid, chunk, 0.0).sum(axis=0)
        acc[f"{var_key}_count"][week - 1] += valid.sum(axis=0)
        acc[f"n_init_{var_key}"][week - 1] += int(np.count_nonzero(mask))


def finalize_mean(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    out = np.full_like(sum_arr, np.nan, dtype=np.float32)
    np.divide(sum_arr, count_arr, out=out, where=count_arr > 0.0)
    return out.astype(np.float32)


def resolve_ml_year_path(year: int, hindcast_dir: str, forecast_dir: str) -> str:
    candidates = [
        os.path.join(hindcast_dir, f"{year}.zarr"),
        os.path.join(forecast_dir, f"{year}.zarr"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find ML yearly store for {year}. Tried: {candidates}")


def extract_standard_coords(ds: xr.Dataset, layout: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lead_values = np.asarray(ds[layout["lead_dim"]].values if layout["lead_dim"] in ds.coords else np.arange(ds.sizes[layout["lead_dim"]]))
    y_values = np.asarray(ds[layout["y_dim"]].values if layout["y_dim"] in ds.coords else np.arange(ds.sizes[layout["y_dim"]]))
    x_values = np.asarray(ds[layout["x_dim"]].values if layout["x_dim"] in ds.coords else np.arange(ds.sizes[layout["x_dim"]]))
    return lead_values, y_values, x_values


def standardize_var(ds: xr.Dataset, layout: Dict[str, str], var_name: str, ensemble_mean_first: bool) -> Tuple[np.ndarray, np.ndarray]:
    da = ds[var_name]
    if ensemble_mean_first and layout["member_dim"] is not None:
        da = da.mean(dim=layout["member_dim"])
    da = da.transpose(layout["s_dim"], layout["lead_dim"], layout["y_dim"], layout["x_dim"])
    s_values = pd.to_datetime(da[layout["s_dim"]].values).normalize()
    init_weeks = np.asarray([int(item.isocalendar().week) for item in s_values], dtype=np.int32)
    return np.asarray(da.values, dtype=np.float64), init_weeks


def write_climatology_store(
    out_path: str,
    source_name: str,
    start_year: int,
    end_year: int,
    lead_values: np.ndarray,
    y_values: np.ndarray,
    x_values: np.ndarray,
    acc: Dict[str, np.ndarray],
    attrs: Dict[str, object],
    overwrite: bool,
):
    if os.path.exists(out_path):
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_path} (use --overwrite to replace)")
        if os.path.isdir(out_path):
            import shutil

            shutil.rmtree(out_path)
        else:
            os.remove(out_path)

    ds_out = xr.Dataset(
        data_vars={
            "pr": (("init_week", "L", "Y", "X"), finalize_mean(acc["pr_sum"], acc["pr_count"])),
            "tas": (("init_week", "L", "Y", "X"), finalize_mean(acc["tas_sum"], acc["tas_count"])),
            "n_init_pr": (("init_week",), acc["n_init_pr"].astype(np.int32)),
            "n_init_tas": (("init_week",), acc["n_init_tas"].astype(np.int32)),
        },
        coords={
            "init_week": np.arange(1, 54, dtype=np.int32),
            "L": lead_values,
            "Y": y_values,
            "X": x_values,
        },
        attrs={
            "climatology_kind": "init_week_and_lead",
            "description": "Weekly climatology averaged over init dates in each ISO init week; GEOS and ML use ensemble mean per init first.",
            "source_name": source_name,
            "start_year": int(start_year),
            "end_year": int(end_year),
            **attrs,
        },
    )
    ds_out["pr"].attrs.update({"long_name": "weekly_init_climatology_pr"})
    ds_out["tas"].attrs.update({"long_name": "weekly_init_climatology_tas"})
    ds_out["init_week"].attrs.update({"long_name": "iso_initialization_week"})
    ds_out.to_zarr(out_path, mode="w")
    ds_out.close()


def build_ml_climatology(args, data_dir: str):
    lead_values = None
    y_values = None
    x_values = None
    acc = None
    used_paths = []

    print("\n[ML] Building weekly climatology")
    for year in range(args.start_year, args.end_year + 1):
        path = resolve_ml_year_path(year, args.ml_hindcast_dir, args.ml_forecast_dir)
        print(f"[ML] {year}: {path}")
        ds = open_zarr_required(path)
        try:
            layout = infer_layout(ds, "ML")
            if layout["pr_var"] is None or layout["tas_var"] is None:
                raise ValueError(f"ML file for {year} is missing pr/tas variables: {path}")
            if acc is None:
                lead_values, y_values, x_values = extract_standard_coords(ds, layout)
                acc = init_accumulator(lead_values, y_values, x_values)
            pr_values, init_weeks = standardize_var(ds, layout, layout["pr_var"], ensemble_mean_first=True)
            tas_values, init_weeks_tas = standardize_var(ds, layout, layout["tas_var"], ensemble_mean_first=True)
            if not np.array_equal(init_weeks, init_weeks_tas):
                raise ValueError(f"ML init-week alignment mismatch in {path}")
            update_weekly_accumulator(acc, pr_values, init_weeks, "pr")
            update_weekly_accumulator(acc, tas_values, init_weeks, "tas")
            used_paths.append(path)
        finally:
            ds.close()

    out_path = os.path.join(args.output_dir, f"ml_weekly_ensmean_clim_{args.start_year}_{args.end_year}.zarr")
    write_climatology_store(
        out_path=out_path,
        source_name="ml_multiv1",
        start_year=args.start_year,
        end_year=args.end_year,
        lead_values=lead_values,
        y_values=y_values,
        x_values=x_values,
        acc=acc,
        attrs={
            "ml_hindcast_dir": os.path.abspath(args.ml_hindcast_dir),
            "ml_forecast_dir": os.path.abspath(args.ml_forecast_dir),
            "used_year_store_count": len(used_paths),
        },
        overwrite=args.overwrite,
    )
    print(f"[ML] Saved: {out_path}")


def build_geos_climatology(args, data_dir: str):
    lead_values = None
    y_values = None
    x_values = None
    acc = None

    print("\n[GEOS] Building weekly climatology")
    for year in range(args.start_year, args.end_year + 1):
        path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
        print(f"[GEOS] {year}: {path}")
        ds = open_zarr_required(path)
        try:
            layout = infer_layout(ds, "GEOS")
            if layout["pr_var"] is None or layout["tas_var"] is None:
                raise ValueError(f"GEOS file for {year} is missing pr/tas variables: {path}")
            if acc is None:
                lead_values, y_values, x_values = extract_standard_coords(ds, layout)
                acc = init_accumulator(lead_values, y_values, x_values)
            pr_values, init_weeks = standardize_var(ds, layout, layout["pr_var"], ensemble_mean_first=True)
            tas_values, init_weeks_tas = standardize_var(ds, layout, layout["tas_var"], ensemble_mean_first=True)
            if not np.array_equal(init_weeks, init_weeks_tas):
                raise ValueError(f"GEOS init-week alignment mismatch in {path}")
            update_weekly_accumulator(acc, pr_values, init_weeks, "pr")
            update_weekly_accumulator(acc, tas_values, init_weeks, "tas")
        finally:
            ds.close()

    out_path = os.path.join(args.output_dir, f"geos_weekly_ensmean_clim_{args.start_year}_{args.end_year}.zarr")
    write_climatology_store(
        out_path=out_path,
        source_name="geos",
        start_year=args.start_year,
        end_year=args.end_year,
        lead_values=lead_values,
        y_values=y_values,
        x_values=x_values,
        acc=acc,
        attrs={"data_dir": os.path.abspath(data_dir)},
        overwrite=args.overwrite,
    )
    print(f"[GEOS] Saved: {out_path}")


def build_obs_climatology(args, data_dir: str):
    lead_values = None
    y_values = None
    x_values = None
    acc = None

    print("\n[OBS] Building weekly climatology")
    for year in range(args.start_year, args.end_year + 1):
        pr_path = os.path.join(data_dir, f"gpcp_weekly_{year}.zarr")
        tas_path = os.path.join(data_dir, f"t2m_weekly_{year}.zarr")
        print(f"[OBS] {year}: PR={pr_path} | TAS={tas_path}")
        ds_pr = open_zarr_required(pr_path)
        ds_tas = open_zarr_required(tas_path)
        try:
            layout_pr = infer_layout(ds_pr, "OBS PR")
            layout_tas = infer_layout(ds_tas, "OBS TAS")
            if layout_pr["pr_var"] is None:
                raise ValueError(f"Obs precip file for {year} missing precip variable: {pr_path}")
            if layout_tas["tas_var"] is None:
                raise ValueError(f"Obs t2m file for {year} missing t2m variable: {tas_path}")

            s_pr = pd.to_datetime(ds_pr[layout_pr["s_dim"]].values).normalize()
            s_tas = pd.to_datetime(ds_tas[layout_tas["s_dim"]].values).normalize()
            if len(s_pr) != len(s_tas) or not np.array_equal(s_pr.values, s_tas.values):
                raise ValueError(f"Obs init-date alignment mismatch for {year}")

            if acc is None:
                lead_values, y_values, x_values = extract_standard_coords(ds_tas, layout_tas)
                acc = init_accumulator(lead_values, y_values, x_values)

            pr_values, init_weeks = standardize_var(ds_pr, layout_pr, layout_pr["pr_var"], ensemble_mean_first=False)
            tas_values, init_weeks_tas = standardize_var(ds_tas, layout_tas, layout_tas["tas_var"], ensemble_mean_first=False)
            if not np.array_equal(init_weeks, init_weeks_tas):
                raise ValueError(f"Obs init-week alignment mismatch in {year}")
            update_weekly_accumulator(acc, pr_values, init_weeks, "pr")
            update_weekly_accumulator(acc, tas_values, init_weeks, "tas")
        finally:
            ds_pr.close()
            ds_tas.close()

    out_path = os.path.join(args.output_dir, f"obs_weekly_clim_{args.start_year}_{args.end_year}.zarr")
    write_climatology_store(
        out_path=out_path,
        source_name="observations",
        start_year=args.start_year,
        end_year=args.end_year,
        lead_values=lead_values,
        y_values=y_values,
        x_values=x_values,
        acc=acc,
        attrs={"data_dir": os.path.abspath(data_dir)},
        overwrite=args.overwrite,
    )
    print(f"[OBS] Saved: {out_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or config["data_dir"]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("WEEKLY MULTIV1 CLIMATOLOGY")
    print(f"Years      : {args.start_year}-{args.end_year}")
    print(f"Data Dir   : {data_dir}")
    print(f"Output Dir : {os.path.abspath(args.output_dir)}")
    print("Basis      : ISO init-week + lead; ML/GEOS use ensemble mean per init first")
    print("=" * 80)

    build_ml_climatology(args, data_dir)
    build_geos_climatology(args, data_dir)
    build_obs_climatology(args, data_dir)

    print("\n✅ All weekly climatology stores written successfully.")


if __name__ == "__main__":
    main()
