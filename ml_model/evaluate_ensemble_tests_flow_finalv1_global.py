#!/usr/bin/env python3
"""Ensemble dispersion and member-count tests for generated global forecast Zarrs.

The script reads the yearly Zarr stores produced by
generate_forecast_zarr_flow_finalv1_global.py and evaluates the saved ensemble
members directly. It is intentionally downstream of model inference so member
count, dispersion, and bootstrap diagnostics can be iterated without generating
new forecasts.

Outputs:
  - dispersion_summary.csv: spread/error, variance/error, and CRPS diagnostics.
  - dispersion_rank_histogram.csv: weighted rank histogram counts.
  - ensemble_size_member_repeat_summary.csv: aggregate scores for each random
    member subset repeat.
  - ensemble_size_summary.csv: mean and member-resampling quantiles by ensemble
    size.
  - ensemble_size_bootstrap_ci.csv: case-bootstrap confidence intervals for
    ensemble-size skill.
  - case_member_metrics.csv: optional per-case weighted metric sums.
  - plots/*.png: ensemble-size, dispersion, and rank-histogram diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


VARIABLES = {
    "pr": {
        "model": "model_pr",
        "geos": "geos_pr",
        "obs": "obs_pr",
        "units": "mm/day",
    },
    "t2m": {
        "model": "model_t2m",
        "geos": "geos_t2m",
        "obs": "obs_t2m",
        "units": "K",
    },
}

SEASONS = {
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
    12: "DJF",
}

SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}

MONTH_ALIASES = {
    "djf": SEASON_MONTHS["DJF"],
    "mam": SEASON_MONTHS["MAM"],
    "jja": SEASON_MONTHS["JJA"],
    "summer": SEASON_MONTHS["JJA"],
    "son": SEASON_MONTHS["SON"],
}

SEASON_ALIASES = {
    "djf": "DJF",
    "mam": "MAM",
    "jja": "JJA",
    "summer": "JJA",
    "son": "SON",
}

REGIONS = {
    "global": {
        "name": "Global",
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lon_min": 0.0,
        "lon_max": 360.0,
    },
    "south_asia": {
        "name": "South Asia",
        "lat_min": 5.0,
        "lat_max": 35.0,
        "lon_min": 60.0,
        "lon_max": 100.0,
    },
    "uk": {
        "name": "United Kingdom",
        "lat_min": 49.0,
        "lat_max": 61.0,
        "lon_min": -11.0,
        "lon_max": 3.0,
    },
    "western_north_america": {
        "name": "Western North America",
        "lat_min": 32.0,
        "lat_max": 55.0,
        "lon_min": -125.0,
        "lon_max": -105.0,
    },
    "central_north_america": {
        "name": "Central North America",
        "lat_min": 30.0,
        "lat_max": 50.0,
        "lon_min": -105.0,
        "lon_max": -85.0,
    },
    "eastern_north_america": {
        "name": "Eastern North America",
        "lat_min": 30.0,
        "lat_max": 50.0,
        "lon_min": -85.0,
        "lon_max": -65.0,
    },
    "central_america": {
        "name": "Central America",
        "lat_min": 10.0,
        "lat_max": 25.0,
        "lon_min": -110.0,
        "lon_max": -75.0,
    },
    "western_europe": {
        "name": "Western Europe",
        "lat_min": 36.0,
        "lat_max": 60.0,
        "lon_min": -10.0,
        "lon_max": 15.0,
    },
    "eastern_europe": {
        "name": "Eastern Europe",
        "lat_min": 40.0,
        "lat_max": 60.0,
        "lon_min": 15.0,
        "lon_max": 35.0,
    },
    "mediterranean": {
        "name": "Mediterranean",
        "lat_min": 30.0,
        "lat_max": 45.0,
        "lon_min": -10.0,
        "lon_max": 40.0,
    },
    "middle_east": {
        "name": "Middle East",
        "lat_min": 20.0,
        "lat_max": 40.0,
        "lon_min": 35.0,
        "lon_max": 65.0,
    },
    "central_asia": {
        "name": "Central Asia",
        "lat_min": 35.0,
        "lat_max": 50.0,
        "lon_min": 60.0,
        "lon_max": 90.0,
    },
    "east_asia": {
        "name": "East Asia",
        "lat_min": 25.0,
        "lat_max": 45.0,
        "lon_min": 100.0,
        "lon_max": 145.0,
    },
    "southeast_asia": {
        "name": "Southeast Asia",
        "lat_min": -10.0,
        "lat_max": 25.0,
        "lon_min": 95.0,
        "lon_max": 125.0,
    },
    "australia": {
        "name": "Australia",
        "lat_min": -40.0,
        "lat_max": -15.0,
        "lon_min": 110.0,
        "lon_max": 155.0,
    },
    "southern_africa": {
        "name": "Southern Africa",
        "lat_min": -35.0,
        "lat_max": -15.0,
        "lon_min": 10.0,
        "lon_max": 40.0,
    },
    "south_america": {
        "name": "South America",
        "lat_min": -40.0,
        "lat_max": -15.0,
        "lon_min": -75.0,
        "lon_max": -45.0,
    },
}

REGION_SETS = {
    "heatwave": [
        "western_north_america",
        "central_north_america",
        "eastern_north_america",
        "central_america",
        "western_europe",
        "eastern_europe",
        "mediterranean",
        "middle_east",
        "central_asia",
        "south_asia",
        "east_asia",
        "southeast_asia",
        "australia",
        "southern_africa",
        "south_america",
    ],
}
REGION_SETS["global_extremes"] = REGION_SETS["heatwave"]
REGION_SETS["precip"] = REGION_SETS["heatwave"]

SUM_COLUMNS = [
    "model_weight_sum",
    "model_crps_sum",
    "model_sse_sum",
    "model_spread_sum",
    "model_variance_sum",
    "geos_weight_sum",
    "geos_crps_sum",
    "geos_sse_sum",
    "geos_spread_sum",
    "geos_variance_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble dispersion and member-count bootstrap tests on generated global Zarr forecasts."
    )
    parser.add_argument(
        "--forecast_dir",
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Directory containing YEAR.zarr stores.",
    )
    parser.add_argument("--start_year", type=int, default=2021)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--skip_years", default="", help="Comma-separated years to skip.")
    parser.add_argument(
        "--out_dir",
        default="ml_output_flow_finalv1_global_noisectx_t2mres/ensemble_tests_global_2021_2024_e90_s50",
    )
    parser.add_argument("--variables", default="pr,t2m", help="Comma-separated subset of pr,t2m.")
    parser.add_argument(
        "--sample_sizes",
        default="1,2,5,10,20,30,50,70,90",
        help="Comma-separated generated-ensemble sizes to test. Values above the available member count are skipped.",
    )
    parser.add_argument(
        "--member_bootstrap_repeats",
        type=int,
        default=50,
        help="Random member-subset repeats for each ensemble size; full-member evaluations are done once.",
    )
    parser.add_argument(
        "--case_bootstrap_repeats",
        type=int,
        default=500,
        help="Bootstrap repeats over initialization/lead cases for confidence intervals; <=0 disables.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--spatial_reduction",
        choices=("gridpoint", "regional_mean"),
        default="gridpoint",
        help=(
            "gridpoint verifies each grid cell then area-averages scores; regional_mean first "
            "area-averages each member/observation over --region, then verifies the regional mean."
        ),
    )
    parser.add_argument(
        "--region",
        choices=tuple(REGIONS),
        default="global",
        help="Named region used for regional filtering. In regional_mean mode, members are averaged over it first.",
    )
    parser.add_argument(
        "--region_bounds",
        default="",
        help="Optional custom region bounds as lat_min,lat_max,lon_min,lon_max.",
    )
    parser.add_argument("--eval_mask", choices=("all", "land", "ocean"), default="all")
    parser.add_argument(
        "--eval_masks",
        default="",
        help=(
            "Optional comma-separated evaluation masks to compute in one pass, e.g. all,land. "
            "When multiple masks are requested, outputs are written under --out_dir/MASK/."
        ),
    )
    parser.add_argument(
        "--land_mask_file",
        default=None,
        help="Optional .pt land mask with is_land or land_mask. Required for land/ocean masks.",
    )
    parser.add_argument("--lead_values", default="", help="Optional comma-separated lead values to keep.")
    parser.add_argument("--init_dates", default="", help="Optional comma-separated init date(s), YYYY-MM-DD.")
    parser.add_argument("--valid_dates", default="", help="Optional comma-separated valid date(s), YYYY-MM-DD.")
    parser.add_argument(
        "--extreme_event_count",
        type=int,
        default=0,
        help="If >0, scan requested years and select this many observed regional extreme events.",
    )
    parser.add_argument(
        "--extreme_event_variable",
        default="t2m",
        help="Observed variable(s) used to rank extreme events, e.g. t2m or t2m,pr.",
    )
    parser.add_argument(
        "--extreme_event_regions",
        default="heatwave",
        help="Comma-separated region names or a region set name such as heatwave.",
    )
    parser.add_argument(
        "--extreme_event_max_per_region",
        type=int,
        default=2,
        help="Maximum selected events per region; <=0 disables the cap.",
    )
    parser.add_argument(
        "--extreme_event_count_per_lead",
        action="store_true",
        help=(
            "Select --extreme_event_count cases separately for each lead week. "
            "Useful when comparing ensemble-size diagnostics across W1-W4."
        ),
    )
    parser.add_argument(
        "--init_months",
        default="",
        help=(
            "Optional initialization months to keep, e.g. 6,7,8. "
            "Season aliases DJF,MAM,JJA/ summer,SON are also accepted."
        ),
    )
    parser.add_argument(
        "--init_season_counts",
        default="",
        help=(
            "Optional balanced init selector as SEASON:COUNT pairs, e.g. JJA:5,DJF:5. "
            "Cannot be combined with --init_months, --max_inits_per_year, or --init_stride."
        ),
    )
    parser.add_argument(
        "--balanced_monthly_inits_per_year",
        type=int,
        default=0,
        help=(
            "If >0, randomly select this many init dates per year while covering all 12 calendar "
            "months across the requested years. For 2021-2023 with value 4, this selects 12 total "
            "dates: four per year and one date in each month Jan-Dec. Uses --seed."
        ),
    )
    parser.add_argument(
        "--one_init_per_month_per_year",
        action="store_true",
        help=(
            "Randomly select one available initialization in each calendar month for each requested year. "
            "For 2021-2023 this selects 36 total init dates. Uses --seed."
        ),
    )
    parser.add_argument(
        "--init_stride",
        type=int,
        default=1,
        help="Keep every Nth initialization after month filtering; use 4 for a one-quarter sample spread through the year.",
    )
    parser.add_argument(
        "--init_stride_offset",
        type=int,
        default=0,
        help="Zero-based offset for --init_stride. With --init_stride 4, offsets 0,1,2,3 give four interleaved samples.",
    )
    parser.add_argument("--max_inits_per_year", type=int, default=None)
    parser.add_argument(
        "--allow_missing_years",
        action="store_true",
        help="Skip missing YEAR.zarr stores instead of failing.",
    )
    parser.add_argument(
        "--skip_rank_histogram",
        action="store_true",
        help="Disable weighted rank histogram accumulation.",
    )
    parser.add_argument(
        "--rank_bins",
        type=int,
        default=0,
        help="If >0, coarsen ranks to this many bins; default writes exact ranks 0..M.",
    )
    parser.add_argument(
        "--write_case_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-case member-subset metric sums.",
    )
    parser.add_argument(
        "--make_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG diagnostic plots from the output CSVs.",
    )
    parser.add_argument(
        "--report_member_count",
        type=int,
        default=8,
        help=(
            "After ensemble_size_summary.csv is written, also print and save a compact report "
            "for this generated ensemble size. Use <=0 to disable."
        ),
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip evaluation and create plots from existing CSVs in --out_dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_int_set(text: str) -> set[int]:
    return {int(item.strip()) for item in str(text or "").split(",") if item.strip()}


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text or "").split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return sorted(dict.fromkeys(values))


def parse_eval_mask_list(text: str, default: str) -> list[str]:
    allowed = {"all", "land", "ocean"}
    raw = str(text or "").strip()
    masks = [item.strip().lower() for item in raw.split(",") if item.strip()] if raw else [default]
    if not masks:
        masks = [default]
    unknown = sorted(set(masks) - allowed)
    if unknown:
        raise ValueError(f"Unknown eval mask(s) {unknown}; valid options are {sorted(allowed)}.")
    return list(dict.fromkeys(masks))


def parse_date_filter(text: str) -> set[str]:
    dates = set()
    for item in str(text or "").split(","):
        token = item.strip()
        if not token:
            continue
        try:
            dates.add(pd.Timestamp(token).strftime("%Y-%m-%d"))
        except Exception as exc:
            raise ValueError(f"Invalid date {token!r}; expected YYYY-MM-DD.") from exc
    return dates


def parse_month_filter(text: str) -> set[int]:
    months: set[int] = set()
    for item in str(text or "").split(","):
        token = item.strip()
        if not token:
            continue
        alias = MONTH_ALIASES.get(token.lower())
        if alias is not None:
            months.update(alias)
            continue
        month = int(token)
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month {month}; expected values from 1 to 12.")
        months.add(month)
    return months


def parse_init_season_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in str(text or "").split(","):
        token = item.strip()
        if not token:
            continue
        if ":" in token:
            name, count_text = token.split(":", 1)
        elif "=" in token:
            name, count_text = token.split("=", 1)
        else:
            raise ValueError("--init_season_counts entries must look like JJA:5,DJF:5.")
        season = SEASON_ALIASES.get(name.strip().lower())
        if season is None:
            raise ValueError(f"Unknown init season {name!r}; valid options are {sorted(SEASON_MONTHS)}.")
        if season in counts:
            raise ValueError(f"Duplicate init season {season!r} in --init_season_counts.")
        count = int(count_text.strip())
        if count <= 0:
            raise ValueError("--init_season_counts values must be positive.")
        counts[season] = count
    return counts


def parse_region_list(text: str) -> list[str]:
    regions = []
    for item in str(text or "").split(","):
        token = item.strip()
        if not token:
            continue
        if token in REGION_SETS:
            regions.extend(REGION_SETS[token])
            continue
        if token not in REGIONS:
            raise ValueError(
                f"Unknown region {token!r}; valid regions are {sorted(REGIONS)} "
                f"or region sets {sorted(REGION_SETS)}."
            )
        regions.append(token)
    regions = list(dict.fromkeys(regions))
    if not regions:
        raise ValueError("Expected at least one region.")
    return regions


def parse_variables(text: str) -> list[str]:
    variables = [item.strip().lower() for item in str(text or "").split(",") if item.strip()]
    bad = [v for v in variables if v not in VARIABLES]
    if bad:
        raise ValueError(f"Unknown variables {bad}; valid options are {sorted(VARIABLES)}")
    if not variables:
        raise ValueError("--variables cannot be empty.")
    return variables


def parse_region_bounds(text: str, region: str) -> dict[str, float | str]:
    if str(text or "").strip():
        parts = [float(item.strip()) for item in str(text).split(",") if item.strip()]
        if len(parts) != 4:
            raise ValueError("--region_bounds must be lat_min,lat_max,lon_min,lon_max.")
        lat_min, lat_max, lon_min, lon_max = parts
        return {
            "name": "Custom region",
            "lat_min": min(lat_min, lat_max),
            "lat_max": max(lat_min, lat_max),
            "lon_min": lon_min,
            "lon_max": lon_max,
        }
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}; valid regions are {sorted(REGIONS)}")
    return dict(REGIONS[region])


def region_key(args: argparse.Namespace, region_bounds: dict[str, float | str]) -> str:
    return "custom" if str(args.region_bounds or "").strip() else str(args.region)


def store_path(forecast_dir: str, year: int) -> Path:
    base = Path(forecast_dir)
    candidates = [base / f"{year}.zarr", base / str(year)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def timestamp_now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def forecast_store_candidates(forecast_dir: str, year: int) -> list[Path]:
    base = Path(forecast_dir)
    return [base / f"{year}.zarr", base / str(year)]


def summarize_year_stores(directory: Path, years: list[int]) -> str:
    present = []
    for year in years:
        if any(candidate.exists() for candidate in forecast_store_candidates(str(directory), year)):
            present.append(year)
    if not present:
        return "no requested years"
    if present == years:
        return f"all requested years ({present[0]}-{present[-1]})"
    return "years " + ",".join(str(year) for year in present)


def nearby_forecast_dir_hints(forecast_dir: str, years: list[int], limit: int = 12) -> list[str]:
    requested = Path(forecast_dir)
    search_roots = []
    for root in (requested.parent, Path("dataprocess")):
        if root and root.exists() and root.is_dir() and root not in search_roots:
            search_roots.append(root)

    hints = []
    for root in search_roots:
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir() or candidate == requested:
                continue
            name = candidate.name.lower()
            if "gen_flow" not in name and "forecast" not in name:
                continue
            hints.append(f"  - {candidate} ({summarize_year_stores(candidate, years)})")
            if len(hints) >= limit:
                return hints
    return hints


def missing_forecast_message(forecast_dir: str, year: int, years: list[int]) -> str:
    candidates = forecast_store_candidates(forecast_dir, year)
    base = Path(forecast_dir)
    lines = [
        f"Missing forecast store for {year}.",
        f"Forecast directory: {base}",
        "Tried:",
    ]
    lines.extend(f"  - {candidate}" for candidate in candidates)
    if not base.exists():
        lines.append("The forecast directory itself does not exist.")
    hints = nearby_forecast_dir_hints(forecast_dir, years)
    if hints:
        lines.append("Nearby forecast directories:")
        lines.extend(hints)
    lines.append(
        "Use the directory that contains yearly stores named YYYY.zarr, or generate the missing years first."
    )
    return "\n".join(lines)


def validate_forecast_stores(forecast_dir: str, years: list[int], allow_missing_years: bool) -> list[int]:
    available = []
    for year in years:
        if store_path(forecast_dir, year).exists():
            available.append(year)
            continue
        message = missing_forecast_message(forecast_dir, year, years)
        if allow_missing_years:
            print(f"Skipping: {message}")
            continue
        raise FileNotFoundError(message)
    if not available:
        raise FileNotFoundError(
            "No requested forecast stores are available.\n"
            + missing_forecast_message(forecast_dir, years[0], years)
        )
    return available


def find_dim(dims: tuple[str, ...], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in dims:
            return candidate
    lower = {dim.lower(): dim for dim in dims}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find {label} dimension among {dims}")


def find_coord(ds: xr.Dataset, candidates: tuple[str, ...], size: int | None = None) -> tuple[str | None, np.ndarray]:
    for candidate in candidates:
        if candidate in ds.coords:
            return candidate, np.asarray(ds[candidate].values)
        if candidate in ds:
            arr = np.asarray(ds[candidate].values)
            if arr.ndim == 1:
                return candidate, arr
    if size is None:
        raise ValueError(f"Could not find coordinate among {candidates}")
    return None, np.arange(size)


def get_lat_lon(ds: xr.Dataset, sample_var: str) -> tuple[np.ndarray, np.ndarray]:
    da = ds[sample_var]
    lat_dim = find_dim(da.dims, ("lat", "latitude", "y"), "latitude")
    lon_dim = find_dim(da.dims, ("lon", "longitude", "x"), "longitude")
    _, lats = find_coord(ds, (lat_dim, "lat", "latitude"), size=da.sizes[lat_dim])
    _, lons = find_coord(ds, (lon_dim, "lon", "longitude"), size=da.sizes[lon_dim])
    return lats.astype(np.float64), lons.astype(np.float64)


def lead_index_values(ds: xr.Dataset, sample_var: str) -> list[tuple[int, int]]:
    da = ds[sample_var]
    lead_dim = find_dim(da.dims, ("lead", "lead_week", "week"), "lead")
    if lead_dim in ds.coords:
        values = np.asarray(ds[lead_dim].values)
    elif "lead" in ds:
        values = np.asarray(ds["lead"].values)
    else:
        values = np.arange(1, da.sizes[lead_dim] + 1)
    out = []
    for idx, value in enumerate(values):
        try:
            lead_value = int(value)
        except Exception:
            lead_value = idx + 1
        out.append((idx, lead_value))
    return out


def init_count(ds: xr.Dataset, sample_var: str) -> int:
    init_dim = find_dim(ds[sample_var].dims, ("init", "initialization", "time"), "init")
    return int(ds[sample_var].sizes[init_dim])


def timestamp_or_nat(value) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
        return ts if not pd.isna(ts) else pd.NaT
    except Exception:
        return pd.NaT


def date_key(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def dataset_time_value(ds: xr.Dataset, names: tuple[str, ...], init_idx: int, lead_idx: int | None = None) -> pd.Timestamp:
    for name in names:
        if name not in ds:
            continue
        arr = ds[name]
        indexers = {}
        for dim in arr.dims:
            lower = dim.lower()
            if "init" in lower or lower == "time":
                indexers[dim] = init_idx
            elif lead_idx is not None and "lead" in lower:
                indexers[dim] = lead_idx
        try:
            return timestamp_or_nat(arr.isel(indexers).values)
        except Exception:
            continue
    return pd.NaT


def case_times(ds: xr.Dataset, init_idx: int, lead_idx: int, lead_value: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    init_time = dataset_time_value(ds, ("init_time", "init", "time"), init_idx)
    valid_time = dataset_time_value(ds, ("valid_time", "target_time"), init_idx, lead_idx)
    if pd.isna(valid_time) and not pd.isna(init_time):
        valid_time = init_time + pd.to_timedelta(int(lead_value) * 7, unit="D")
    return init_time, valid_time


def valid_month_season(valid_time: pd.Timestamp) -> tuple[int | float, str]:
    if pd.isna(valid_time):
        return np.nan, ""
    month = int(valid_time.month)
    return month, SEASONS[month]


def filtered_init_indices(ds: xr.Dataset, sample_var: str, months: set[int]) -> list[int]:
    indices = []
    for init_idx in range(init_count(ds, sample_var)):
        if months:
            init_time = dataset_time_value(ds, ("init_time", "init", "time"), init_idx)
            if pd.isna(init_time) or int(init_time.month) not in months:
                continue
        indices.append(init_idx)
    return indices


def apply_init_stride(indices: list[int], stride: int, offset: int) -> list[int]:
    if stride <= 1:
        return list(indices)
    return [idx for position, idx in enumerate(indices) if position % stride == offset]


def filter_init_indices_by_dates(ds: xr.Dataset, indices: list[int], init_dates: set[str]) -> list[int]:
    if not init_dates:
        return list(indices)
    return [
        init_idx
        for init_idx in indices
        if date_key(dataset_time_value(ds, ("init_time", "init", "time"), init_idx)) in init_dates
    ]


def filtered_init_indices_by_season_counts(
    ds: xr.Dataset,
    sample_var: str,
    season_counts: dict[str, int],
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    available = {season: [] for season in season_counts}
    for init_idx in range(init_count(ds, sample_var)):
        init_time = dataset_time_value(ds, ("init_time", "init", "time"), init_idx)
        if pd.isna(init_time):
            continue
        init_month = int(init_time.month)
        for season in season_counts:
            if init_month in SEASON_MONTHS[season]:
                available[season].append(init_idx)
                break

    selected = []
    selected_counts = {}
    available_counts = {}
    for season, requested in season_counts.items():
        available_counts[season] = len(available[season])
        if len(available[season]) < requested:
            raise ValueError(
                f"Requested {requested} {season} init dates, but only found {len(available[season])}."
            )
        selected.extend(available[season][:requested])
        selected_counts[season] = requested
    return sorted(selected), selected_counts, available_counts


def assign_balanced_months_by_year(
    years: list[int],
    per_year: int,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    if per_year <= 0:
        return {}
    if not years:
        raise ValueError("--balanced_monthly_inits_per_year requires at least one selected year.")
    total_requested = len(years) * per_year
    if total_requested < 12:
        raise ValueError(
            "--balanced_monthly_inits_per_year selects too few dates to cover Jan-Dec once "
            f"({len(years)} years x {per_year} per year = {total_requested})."
        )

    months = list(range(1, 13))
    rng.shuffle(months)
    if total_requested > 12:
        extra_months = rng.choice(np.arange(1, 13), size=total_requested - 12, replace=True).tolist()
        months.extend(int(month) for month in extra_months)
        rng.shuffle(months)

    assignments = {}
    cursor = 0
    for year in years:
        year_months = sorted(int(month) for month in months[cursor : cursor + per_year])
        assignments[int(year)] = year_months
        cursor += per_year
    return assignments


def filtered_init_indices_by_months_random(
    ds: xr.Dataset,
    sample_var: str,
    months: list[int],
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, str], dict[str, int]]:
    available: dict[int, list[tuple[int, str]]] = {int(month): [] for month in months}
    for init_idx in range(init_count(ds, sample_var)):
        init_time = dataset_time_value(ds, ("init_time", "init", "time"), init_idx)
        if pd.isna(init_time):
            continue
        init_month = int(init_time.month)
        if init_month in available:
            available[init_month].append((init_idx, date_key(init_time)))

    selected_indices = []
    selected_dates = {}
    available_counts = {}
    for month in months:
        month = int(month)
        choices = available.get(month, [])
        available_counts[str(month)] = len(choices)
        if not choices:
            raise ValueError(f"Requested a balanced monthly init for month {month}, but none were found.")
        choice_idx = int(rng.integers(0, len(choices)))
        init_idx, init_date = choices[choice_idx]
        selected_indices.append(int(init_idx))
        selected_dates[str(month)] = init_date
    return sorted(selected_indices), selected_dates, available_counts


def load_forecast_array(ds: xr.Dataset, var_name: str, init_idx: int, lead_idx: int) -> np.ndarray:
    da = ds[var_name]
    init_dim = find_dim(da.dims, ("init", "initialization", "time"), "init")
    lead_dim = find_dim(da.dims, ("lead", "lead_week", "week"), "lead")
    member_dim = find_dim(da.dims, ("ensemble", "geos_member", "member"), "member")
    lat_dim = find_dim(da.dims, ("lat", "latitude", "y"), "latitude")
    lon_dim = find_dim(da.dims, ("lon", "longitude", "x"), "longitude")
    selected = da.isel({init_dim: init_idx, lead_dim: lead_idx}).transpose(member_dim, lat_dim, lon_dim)
    return np.asarray(selected.values, dtype=np.float32)


def load_obs_array(ds: xr.Dataset, var_name: str, init_idx: int, lead_idx: int) -> np.ndarray:
    da = ds[var_name]
    init_dim = find_dim(da.dims, ("init", "initialization", "time"), "init")
    lead_dim = find_dim(da.dims, ("lead", "lead_week", "week"), "lead")
    lat_dim = find_dim(da.dims, ("lat", "latitude", "y"), "latitude")
    lon_dim = find_dim(da.dims, ("lon", "longitude", "x"), "longitude")
    selected = da.isel({init_dim: init_idx, lead_dim: lead_idx}).transpose(lat_dim, lon_dim)
    return np.asarray(selected.values, dtype=np.float32)


def area_weights_from_lats(lats: np.ndarray, lon_count: int) -> np.ndarray:
    weights = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64)))
    weights = np.clip(weights, 0.0, None)
    return np.broadcast_to(weights[:, None], (weights.size, lon_count)).astype(np.float64, copy=False)


def longitude_mask(lons: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lons = np.asarray(lons, dtype=np.float64)
    if abs(float(lon_max) - float(lon_min)) >= 360.0:
        return np.ones(lons.shape, dtype=bool)
    if np.nanmax(lons) > 180.0:
        lon_min = lon_min % 360.0
        lon_max = lon_max % 360.0
        lons_norm = lons % 360.0
    else:
        lon_min = ((lon_min + 180.0) % 360.0) - 180.0
        lon_max = ((lon_max + 180.0) % 360.0) - 180.0
        lons_norm = ((lons + 180.0) % 360.0) - 180.0
    if lon_min <= lon_max:
        return (lons_norm >= lon_min) & (lons_norm <= lon_max)
    return (lons_norm >= lon_min) | (lons_norm <= lon_max)


def region_mask_from_bounds(lats: np.ndarray, lons: np.ndarray, region_bounds: dict[str, float | str]) -> np.ndarray:
    lat_min = float(region_bounds["lat_min"])
    lat_max = float(region_bounds["lat_max"])
    lon_min = float(region_bounds["lon_min"])
    lon_max = float(region_bounds["lon_max"])
    lat_mask = (np.asarray(lats, dtype=np.float64) >= lat_min) & (np.asarray(lats, dtype=np.float64) <= lat_max)
    lon_mask = longitude_mask(lons, lon_min, lon_max)
    return lat_mask[:, None] & lon_mask[None, :]


def load_eval_mask(args: argparse.Namespace, shape: tuple[int, int], eval_mask_name: str | None = None) -> np.ndarray:
    mask_name = eval_mask_name or args.eval_mask
    if mask_name == "all":
        return np.ones(shape, dtype=bool)
    if not args.land_mask_file:
        raise ValueError("--land_mask_file is required when evaluating land or ocean masks.")

    import torch

    cached = torch.load(args.land_mask_file, map_location="cpu", weights_only=True)
    if "is_land" in cached:
        land_mask = np.asarray(cached["is_land"], dtype=bool).squeeze()
    elif "land_mask" in cached:
        land_mask = np.asarray(cached["land_mask"], dtype=bool).squeeze()
    else:
        raise KeyError(f"{args.land_mask_file} is missing 'is_land' or 'land_mask'.")
    if land_mask.shape != shape:
        raise ValueError(f"Land mask shape {land_mask.shape} does not match forecast grid {shape}.")
    return land_mask if mask_name == "land" else ~land_mask


def weighted_spatial_mean(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    field64 = np.asarray(field, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    valid = np.isfinite(field64) & mask_bool
    weighted = np.where(valid, field64 * weights64, 0.0)
    denom = np.sum(np.where(valid, weights64, 0.0), axis=(-2, -1))
    numer = np.sum(weighted, axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom
    return np.where(denom > 0.0, out, np.nan)


def scalar_weighted_spatial_mean(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    value = weighted_spatial_mean(field, weights, mask)
    try:
        return float(np.asarray(value))
    except Exception:
        return np.nan


def regional_mean_metric_inputs(
    model: np.ndarray,
    geos: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_mean = weighted_spatial_mean(obs, weights, mask)
    model_mean = weighted_spatial_mean(model, weights, mask)
    geos_mean = weighted_spatial_mean(geos, weights, mask)
    metric_obs = np.asarray([[obs_mean]], dtype=np.float32)
    metric_model = np.asarray(model_mean[:, None, None], dtype=np.float32)
    metric_geos = np.asarray(geos_mean[:, None, None], dtype=np.float32)
    metric_weights = np.ones((1, 1), dtype=np.float64)
    metric_mask = np.ones((1, 1), dtype=bool)
    return metric_model, metric_geos, metric_obs, metric_weights, metric_mask


def crps_map(ensemble: np.ndarray, obs: np.ndarray) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    ens64 = ensemble.astype(np.float64, copy=False)
    obs64 = obs.astype(np.float64, copy=False)
    with np.errstate(invalid="ignore"):
        mae_term = np.nanmean(np.abs(ens64 - obs64[None, :, :]), axis=0)
    ens_sorted = np.sort(ens64, axis=0)
    member_count = ens_sorted.shape[0]
    coeff = ((2.0 * np.arange(1, member_count + 1, dtype=np.float64)) - member_count - 1.0)
    coeff /= float(member_count * member_count)
    spread_term = np.sum(coeff[:, None, None] * ens_sorted, axis=0)
    return mae_term - spread_term


def metric_fields(ensemble: np.ndarray, obs: np.ndarray) -> dict[str, np.ndarray]:
    ens64 = np.asarray(ensemble, dtype=np.float64)
    obs64 = np.asarray(obs, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(ens64, axis=0)
        spread = np.nanstd(ens64, axis=0)
    err = mean - obs64
    crps = crps_map(ens64, obs64)
    finite = np.isfinite(obs64) & np.isfinite(mean) & np.isfinite(spread) & np.isfinite(crps)
    return {
        "finite": finite,
        "sse": err * err,
        "crps": crps,
        "spread": spread,
        "variance": spread * spread,
    }


def reduce_fields(fields: dict[str, np.ndarray], weights: np.ndarray, finite: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.asarray(finite, dtype=bool)
    wm = np.where(finite, weights, 0.0)
    weight_sum = float(np.sum(wm))
    if weight_sum <= 0.0:
        return {
            f"{prefix}_weight_sum": 0.0,
            f"{prefix}_crps_sum": 0.0,
            f"{prefix}_sse_sum": 0.0,
            f"{prefix}_spread_sum": 0.0,
            f"{prefix}_variance_sum": 0.0,
        }
    return {
        f"{prefix}_weight_sum": weight_sum,
        f"{prefix}_crps_sum": float(np.sum(np.where(finite, fields["crps"], 0.0) * wm)),
        f"{prefix}_sse_sum": float(np.sum(np.where(finite, fields["sse"], 0.0) * wm)),
        f"{prefix}_spread_sum": float(np.sum(np.where(finite, fields["spread"], 0.0) * wm)),
        f"{prefix}_variance_sum": float(np.sum(np.where(finite, fields["variance"], 0.0) * wm)),
    }


def row_metrics_from_sums(row: dict[str, float] | pd.Series) -> dict[str, float]:
    model_w = float(row.get("model_weight_sum", 0.0))
    geos_w = float(row.get("geos_weight_sum", 0.0))
    out = {}
    if model_w > 0.0:
        out["model_crps"] = float(row["model_crps_sum"] / model_w)
        out["model_rmse"] = float(np.sqrt(row["model_sse_sum"] / model_w))
        out["model_spread"] = float(row["model_spread_sum"] / model_w)
        out["model_variance"] = float(row["model_variance_sum"] / model_w)
    else:
        out.update({k: np.nan for k in ("model_crps", "model_rmse", "model_spread", "model_variance")})
    if geos_w > 0.0:
        out["geos_crps"] = float(row["geos_crps_sum"] / geos_w)
        out["geos_rmse"] = float(np.sqrt(row["geos_sse_sum"] / geos_w))
        out["geos_spread"] = float(row["geos_spread_sum"] / geos_w)
        out["geos_variance"] = float(row["geos_variance_sum"] / geos_w)
    else:
        out.update({k: np.nan for k in ("geos_crps", "geos_rmse", "geos_spread", "geos_variance")})
    out["crps_skill_pct"] = (
        100.0 * (1.0 - out["model_crps"] / out["geos_crps"])
        if np.isfinite(out["model_crps"]) and np.isfinite(out["geos_crps"]) and out["geos_crps"] > 1e-12
        else np.nan
    )
    out["rmse_skill_pct"] = (
        100.0 * (1.0 - out["model_rmse"] / out["geos_rmse"])
        if np.isfinite(out["model_rmse"]) and np.isfinite(out["geos_rmse"]) and out["geos_rmse"] > 1e-12
        else np.nan
    )
    out["model_spread_rmse_ratio"] = (
        out["model_spread"] / out["model_rmse"]
        if np.isfinite(out["model_spread"]) and np.isfinite(out["model_rmse"]) and out["model_rmse"] > 1e-12
        else np.nan
    )
    out["geos_spread_rmse_ratio"] = (
        out["geos_spread"] / out["geos_rmse"]
        if np.isfinite(out["geos_spread"]) and np.isfinite(out["geos_rmse"]) and out["geos_rmse"] > 1e-12
        else np.nan
    )
    return out


def rank_counts(
    ensemble: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
    finite_mask: np.ndarray,
    rng: np.random.Generator,
    rank_bins: int,
) -> tuple[np.ndarray, int]:
    member_count = int(ensemble.shape[0])
    finite_members = np.all(np.isfinite(ensemble), axis=0)
    finite = finite_mask & finite_members & np.isfinite(obs)
    if not finite.any():
        bins = rank_bins if rank_bins > 0 else member_count + 1
        return np.zeros(bins, dtype=np.float64), bins

    less = np.sum(ensemble < obs[None, :, :], axis=0).astype(np.int32)
    equal = np.sum(ensemble == obs[None, :, :], axis=0).astype(np.int32)
    ranks = less
    tie_mask = finite & (equal > 0)
    if tie_mask.any():
        offsets = np.zeros_like(ranks)
        offsets[tie_mask] = rng.integers(0, equal[tie_mask] + 1)
        ranks = ranks + offsets

    flat_ranks = ranks[finite]
    flat_weights = weights[finite].astype(np.float64, copy=False)
    if rank_bins > 0:
        bin_idx = np.floor(flat_ranks * rank_bins / float(member_count + 1)).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, rank_bins - 1)
        return np.bincount(bin_idx, weights=flat_weights, minlength=rank_bins), rank_bins
    return np.bincount(flat_ranks, weights=flat_weights, minlength=member_count + 1), member_count + 1


def update_dispersion_state(
    state: dict[tuple[str, str, int], dict[str, float]],
    source: str,
    variable: str,
    lead: int,
    reduced: dict[str, float],
    prefix: str,
) -> None:
    key = (source, variable, int(lead))
    item = state.setdefault(
        key,
        {
            "n_cases": 0,
            "weight_sum": 0.0,
            "crps_sum": 0.0,
            "sse_sum": 0.0,
            "spread_sum": 0.0,
            "variance_sum": 0.0,
        },
    )
    if reduced[f"{prefix}_weight_sum"] <= 0.0:
        return
    item["n_cases"] += 1
    item["weight_sum"] += float(reduced[f"{prefix}_weight_sum"])
    item["crps_sum"] += float(reduced[f"{prefix}_crps_sum"])
    item["sse_sum"] += float(reduced[f"{prefix}_sse_sum"])
    item["spread_sum"] += float(reduced[f"{prefix}_spread_sum"])
    item["variance_sum"] += float(reduced[f"{prefix}_variance_sum"])


def update_rank_state(
    state: dict[tuple[str, str, int, int], np.ndarray],
    source: str,
    variable: str,
    lead: int,
    ensemble: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
    finite_mask: np.ndarray,
    rng: np.random.Generator,
    rank_bins: int,
) -> None:
    counts, bin_count = rank_counts(ensemble, obs, weights, finite_mask, rng, rank_bins)
    key = (source, variable, int(lead), int(bin_count))
    if key not in state:
        state[key] = np.zeros(bin_count, dtype=np.float64)
    state[key] += counts


def dispersion_summary_rows(
    dispersion_state: dict[tuple[str, str, int], dict[str, float]],
    rank_state: dict[tuple[str, str, int, int], np.ndarray],
) -> list[dict[str, float | int | str]]:
    rows = []
    for (source, variable, lead), state in sorted(dispersion_state.items()):
        w = float(state["weight_sum"])
        if w <= 0.0:
            continue
        mse = float(state["sse_sum"] / w)
        rmse = float(np.sqrt(mse))
        mean_spread = float(state["spread_sum"] / w)
        mean_variance = float(state["variance_sum"] / w)
        row = {
            "source": source,
            "variable": variable,
            "lead": lead,
            "n_cases": int(state["n_cases"]),
            "weight_sum": w,
            "crps": float(state["crps_sum"] / w),
            "rmse": rmse,
            "mean_spread": mean_spread,
            "mean_variance": mean_variance,
            "spread_rmse_ratio": mean_spread / rmse if rmse > 1e-12 else np.nan,
            "variance_mse_ratio": mean_variance / mse if mse > 1e-12 else np.nan,
        }
        rank_items = [
            counts for (src, var, ld, _), counts in rank_state.items()
            if src == source and var == variable and ld == lead
        ]
        if rank_items:
            counts = rank_items[0]
            total = float(np.sum(counts))
            expected = total / counts.size if counts.size else np.nan
            if expected > 0.0 and counts.size > 1:
                row["rank_chi2_per_bin"] = float(np.sum((counts - expected) ** 2 / expected) / (counts.size - 1))
                edge_mass = float(counts[0] + counts[-1])
                row["rank_edge_mass_ratio"] = edge_mass / (2.0 * expected)
            else:
                row["rank_chi2_per_bin"] = np.nan
                row["rank_edge_mass_ratio"] = np.nan
        rows.append(row)
    return rows


def rank_histogram_rows(rank_state: dict[tuple[str, str, int, int], np.ndarray]) -> list[dict[str, float | int | str]]:
    rows = []
    for (source, variable, lead, bin_count), counts in sorted(rank_state.items()):
        total = float(np.sum(counts))
        expected = total / bin_count if bin_count else np.nan
        for idx, count in enumerate(counts):
            rows.append(
                {
                    "source": source,
                    "variable": variable,
                    "lead": lead,
                    "rank_bin": idx,
                    "rank_bin_count": bin_count,
                    "weighted_count": float(count),
                    "relative_frequency": float(count / total) if total > 0.0 else np.nan,
                    "expected_uniform_count": expected,
                    "expected_relative_frequency": 1.0 / bin_count if bin_count else np.nan,
                }
            )
    return rows


def select_extreme_event_cases(
    args: argparse.Namespace,
    years: list[int],
    lead_filter: set[int] | None,
    init_months: set[int],
    init_date_filter: set[str],
    valid_date_filter: set[str],
    event_regions: list[str],
    event_variables: list[str],
) -> list[dict[str, object]]:
    event_count = int(args.extreme_event_count)
    if event_count <= 0:
        return []

    max_per_region = int(args.extreme_event_max_per_region)
    all_selected = []

    for variable in event_variables:
        obs_name = VARIABLES[variable]["obs"]
        candidates_by_region: dict[str, list[dict[str, object]]] = {region: [] for region in event_regions}
        region_cell_counts: dict[str, int] = {}

        for year in years:
            path = store_path(args.forecast_dir, year)
            ds = xr.open_zarr(path, consolidated=False, chunks=None)
            try:
                lats, lons = get_lat_lon(ds, VARIABLES[variable]["model"])
                weights = area_weights_from_lats(lats, len(lons))
                base_mask = load_eval_mask(args, (len(lats), len(lons)))
                lead_pairs = lead_index_values(ds, VARIABLES[variable]["model"])
                if lead_filter is not None:
                    lead_pairs = [(idx, lead) for idx, lead in lead_pairs if lead in lead_filter]
                init_indices = filter_init_indices_by_dates(
                    ds,
                    filtered_init_indices(ds, VARIABLES[variable]["model"], init_months),
                    init_date_filter,
                )

                region_masks = {}
                for region in event_regions:
                    bounds = REGIONS[region]
                    mask = base_mask & region_mask_from_bounds(lats, lons, bounds)
                    kept = int(np.sum(mask))
                    if kept <= 0:
                        raise ValueError(f"Extreme-event region {region!r} kept zero grid cells.")
                    region_masks[region] = mask
                    region_cell_counts[region] = kept

                for init_idx in init_indices:
                    for lead_idx, lead_value in lead_pairs:
                        init_time, valid_time = case_times(ds, init_idx, lead_idx, lead_value)
                        if valid_date_filter and date_key(valid_time) not in valid_date_filter:
                            continue
                        obs = load_obs_array(ds, obs_name, init_idx, lead_idx)
                        for region in event_regions:
                            score = scalar_weighted_spatial_mean(obs, weights, region_masks[region])
                            if not np.isfinite(score):
                                continue
                            candidates_by_region[region].append(
                                {
                                    "year": int(year),
                                    "init_idx": int(init_idx),
                                    "lead_idx": int(lead_idx),
                                    "lead": int(lead_value),
                                    "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
                                    "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
                                    "region": region,
                                    "region_name": str(REGIONS[region]["name"]),
                                    "event_score": float(score),
                                    "event_score_variable": variable,
                                    "region_cell_count": int(region_cell_counts[region]),
                                }
                            )
            finally:
                ds.close()

        for region in event_regions:
            candidates_by_region[region].sort(key=lambda item: float(item["event_score"]), reverse=True)

        if args.extreme_event_count_per_lead:
            selection_leads: list[int | None] = sorted(
                {
                    int(item["lead"])
                    for candidates in candidates_by_region.values()
                    for item in candidates
                }
            )
        else:
            selection_leads = [None]

        for selection_lead in selection_leads:
            candidates_for_lead = {}
            for region in event_regions:
                if selection_lead is None:
                    candidates_for_lead[region] = candidates_by_region[region]
                else:
                    candidates_for_lead[region] = [
                        item for item in candidates_by_region[region] if int(item["lead"]) == int(selection_lead)
                    ]

            selected = []
            selected_counts = {region: 0 for region in event_regions}
            rank_position = 0
            while len(selected) < event_count:
                progressed = False
                for region in event_regions:
                    if max_per_region > 0 and selected_counts[region] >= max_per_region:
                        continue
                    candidates = candidates_for_lead[region]
                    if rank_position >= len(candidates):
                        continue
                    selected.append(dict(candidates[rank_position]))
                    selected_counts[region] += 1
                    progressed = True
                    if len(selected) >= event_count:
                        break
                if not progressed:
                    break
                rank_position += 1

            if len(selected) < event_count:
                available = {region: len(candidates_for_lead[region]) for region in event_regions}
                lead_text = "" if selection_lead is None else f" for lead {selection_lead}"
                raise ValueError(
                    f"Requested {event_count} extreme {variable.upper()} events{lead_text}, "
                    f"but only selected {len(selected)}. Available candidates by region: {available}"
                )

            selected.sort(key=lambda item: float(item["event_score"]), reverse=True)
            for rank, item in enumerate(selected, start=1):
                item["event_rank"] = rank
                item["event_selection_lead"] = "" if selection_lead is None else int(selection_lead)
            lead_text = "" if selection_lead is None else f" for lead {selection_lead}"
            print(
                f"Selected {len(selected)} extreme {variable.upper()} events{lead_text} "
                f"across {len(event_regions)} regions using observed regional mean."
            )
            for item in selected[: min(10, len(selected))]:
                print(
                    f"  {variable.upper()} #{item['event_rank']:02d} {item['region']} "
                    f"score={float(item['event_score']):.3f} "
                    f"init={item['init_time'][:10]} valid={item['valid_time'][:10]} lead={item['lead']}"
                )
            all_selected.extend(selected)

    return all_selected


def aggregate_case_rows(case_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = case_df.groupby(group_cols, dropna=False)
    summed = grouped[SUM_COLUMNS].sum().reset_index()
    counts = grouped.size().rename("n_case_rows").reset_index()
    out = summed.merge(counts, on=group_cols, how="left")
    metrics = [row_metrics_from_sums(row) for _, row in out.iterrows()]
    return pd.concat([out, pd.DataFrame(metrics)], axis=1)


def summarize_member_repeats(repeat_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "model_crps",
        "geos_crps",
        "crps_skill_pct",
        "model_rmse",
        "geos_rmse",
        "rmse_skill_pct",
        "model_spread",
        "geos_spread",
        "model_spread_rmse_ratio",
    ]
    rows = []
    group_cols = ["variable", "lead", "member_count"]
    for key, group in repeat_summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["n_member_repeats"] = int(group["member_repeat"].nunique())
        row["n_case_rows"] = int(group["n_case_rows"].sum())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_p05"] = np.nan
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p95"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{metric}_p05"] = float(np.quantile(values, 0.05))
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            row[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def bootstrap_case_intervals(
    case_df: pd.DataFrame,
    repeats: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if repeats <= 0 or case_df.empty:
        return pd.DataFrame()
    metrics = ["model_crps", "crps_skill_pct", "model_rmse", "rmse_skill_pct", "model_spread_rmse_ratio"]
    rows = []
    group_cols = ["variable", "lead", "member_count"]
    for key, group in case_df.groupby(group_cols, dropna=False):
        repeat_values = np.asarray(sorted(group["member_repeat"].unique()), dtype=np.int64)
        case_values = np.asarray(sorted(group["case_id"].unique()))
        if repeat_values.size == 0 or case_values.size == 0:
            continue
        by_repeat = {
            int(rep): group[group["member_repeat"] == rep].set_index("case_id", drop=False)
            for rep in repeat_values
        }
        boot_values = {metric: [] for metric in metrics}
        for _ in range(repeats):
            rep = int(rng.choice(repeat_values))
            frame = by_repeat[rep]
            sample_ids = rng.choice(case_values, size=case_values.size, replace=True)
            sampled = frame.loc[sample_ids]
            sums = {column: float(sampled[column].sum()) for column in SUM_COLUMNS}
            metric_values = row_metrics_from_sums(sums)
            for metric in metrics:
                boot_values[metric].append(metric_values[metric])
        row = dict(zip(group_cols, key))
        row["n_cases"] = int(case_values.size)
        row["n_bootstrap_repeats"] = int(repeats)
        row["n_member_repeats"] = int(repeat_values.size)
        for metric in metrics:
            values = np.asarray(boot_values[metric], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_p025"] = np.nan
                row[f"{metric}_p50"] = np.nan
                row[f"{metric}_p975"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_p025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_p50"] = float(np.quantile(values, 0.50))
            row[f"{metric}_p975"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path}")


def write_member_count_report(summary: pd.DataFrame, out_dir: Path, member_count: int) -> str | None:
    if member_count <= 0:
        return None
    needed = {"variable", "lead", "member_count"}
    if summary.empty or not needed <= set(summary.columns):
        print(f"No member-count report written for ensemble size {member_count}: summary is empty.")
        return None

    subset = summary[summary["member_count"].astype(int).eq(int(member_count))].copy()
    if subset.empty:
        print(f"No member-count report rows found for ensemble size {member_count}.")
        return None

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
    report_cols = [col for col in preferred_cols if col in subset.columns]
    report = subset[report_cols].sort_values(["variable", "lead"]).reset_index(drop=True)
    path = out_dir / f"ensemble_size_member{int(member_count)}_report.csv"
    write_csv(report, path)

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
    print(f"\nEnsemble size {int(member_count)} report:")
    print(display.to_string(index=False))
    print()
    return str(path)


def write_evaluation_outputs(
    case_rows: list[dict[str, object]],
    dispersion_state: dict[tuple[str, str, int], dict[str, float]],
    rank_state: dict[tuple[str, str, int, int], np.ndarray],
    out_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
    metadata: dict[str, object],
    start_time: float,
    processed_cases: int,
    selected_init_counts_by_year: dict[str, int],
    selected_init_season_counts_by_year: dict[str, dict[str, int]],
    available_init_season_counts_by_year: dict[str, dict[str, int]],
    selected_balanced_monthly_dates_by_year: dict[str, dict[str, str]],
    available_balanced_monthly_counts_by_year: dict[str, dict[str, int]],
    plot_files: list[str] | None = None,
) -> dict[str, object]:
    if not case_rows:
        raise RuntimeError("No forecast cases were processed. Check --forecast_dir, years, variables, and filters.")

    out_dir.mkdir(parents=True, exist_ok=True)
    case_df = pd.DataFrame(case_rows)
    if args.write_case_metrics:
        write_csv(case_df, out_dir / "case_member_metrics.csv")

    repeat_summary = aggregate_case_rows(case_df, ["variable", "lead", "member_count", "member_repeat"])
    write_csv(repeat_summary, out_dir / "ensemble_size_member_repeat_summary.csv")

    ensemble_summary = summarize_member_repeats(repeat_summary)
    write_csv(ensemble_summary, out_dir / "ensemble_size_summary.csv")
    member_report_path = write_member_count_report(ensemble_summary, out_dir, int(args.report_member_count))

    bootstrap_ci = bootstrap_case_intervals(case_df, int(args.case_bootstrap_repeats), rng)
    if not bootstrap_ci.empty:
        write_csv(bootstrap_ci, out_dir / "ensemble_size_bootstrap_ci.csv")

    dispersion_df = pd.DataFrame(dispersion_summary_rows(dispersion_state, rank_state))
    write_csv(dispersion_df, out_dir / "dispersion_summary.csv")

    rank_df = pd.DataFrame(rank_histogram_rows(rank_state))
    if not rank_df.empty:
        write_csv(rank_df, out_dir / "dispersion_rank_histogram.csv")

    if plot_files is None:
        plot_files = make_diagnostic_plots(out_dir) if args.make_plots else []

    metadata.update(
        {
            "processed_init_lead_cases": processed_cases,
            "selected_init_counts_by_year": selected_init_counts_by_year,
            "selected_init_season_counts_by_year": selected_init_season_counts_by_year,
            "available_init_season_counts_by_year": available_init_season_counts_by_year,
            "selected_balanced_monthly_dates_by_year": selected_balanced_monthly_dates_by_year,
            "available_balanced_monthly_counts_by_year": available_balanced_monthly_counts_by_year,
            "case_member_rows": int(len(case_df)),
            "completed_at": timestamp_now_utc(),
            "elapsed_seconds": float(time.time() - start_time),
            "outputs": {
                "case_member_metrics": str(out_dir / "case_member_metrics.csv") if args.write_case_metrics else None,
                "ensemble_size_member_repeat_summary": str(out_dir / "ensemble_size_member_repeat_summary.csv"),
                "ensemble_size_summary": str(out_dir / "ensemble_size_summary.csv"),
                "ensemble_size_member_report": member_report_path,
                "ensemble_size_bootstrap_ci": str(out_dir / "ensemble_size_bootstrap_ci.csv")
                if not bootstrap_ci.empty
                else None,
                "dispersion_summary": str(out_dir / "dispersion_summary.csv"),
                "dispersion_rank_histogram": str(out_dir / "dispersion_rank_histogram.csv")
                if not rank_df.empty
                else None,
                "plots": plot_files,
            },
        }
    )
    with open(out_dir / "ensemble_test_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {out_dir / 'ensemble_test_metadata.json'}")
    return metadata


def _import_matplotlib():
    if not os.environ.get("MPLCONFIGDIR"):
        mpl_cache = Path(os.environ.get("TMPDIR") or "/tmp") / "geos_subc_matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def read_existing_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required CSV for plotting: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def variable_title(variable: str) -> str:
    return {"pr": "Precipitation", "t2m": "2 m temperature"}.get(str(variable), str(variable).upper())


def source_title(source: str) -> str:
    return {"model": "FlowMatch", "geos": "GEOS"}.get(str(source), str(source))


def available_variables(*frames: pd.DataFrame) -> list[str]:
    values = []
    for frame in frames:
        if not frame.empty and "variable" in frame:
            values.extend(str(v) for v in frame["variable"].dropna().unique())
    return sorted(dict.fromkeys(values))


def plot_ensemble_size_metric(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    zero_line: bool = False,
) -> Path | None:
    mean_col = f"{metric}_mean"
    if summary.empty or mean_col not in summary:
        return None

    plt = _import_matplotlib()
    variables = available_variables(summary)
    if not variables:
        return None
    fig, axes = plt.subplots(1, len(variables), figsize=(6.0 * len(variables), 4.0), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, max(1, int(summary["lead"].nunique()))))

    for ax, variable in zip(axes[0], variables):
        sub = summary[summary["variable"] == variable].copy()
        leads = sorted(sub["lead"].dropna().unique())
        for color, lead in zip(colors, leads):
            line = sub[sub["lead"] == lead].sort_values("member_count")
            x = line["member_count"].to_numpy(dtype=float)
            y = line[mean_col].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.7, label=f"Week {int(lead)}", color=color)

            ci = pd.DataFrame()
            if not bootstrap.empty and mean_col in bootstrap:
                ci = bootstrap[(bootstrap["variable"] == variable) & (bootstrap["lead"] == lead)].sort_values(
                    "member_count"
                )
                low_col = f"{metric}_p025"
                high_col = f"{metric}_p975"
            else:
                low_col = f"{metric}_p05"
                high_col = f"{metric}_p95"
                ci = line
            if low_col in ci and high_col in ci and not ci.empty:
                ax.fill_between(
                    ci["member_count"].to_numpy(dtype=float),
                    ci[low_col].to_numpy(dtype=float),
                    ci[high_col].to_numpy(dtype=float),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        if zero_line:
            ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
        ax.set_title(variable_title(variable))
        ax.set_xlabel("Generated ensemble members")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def plot_dispersion_summary(dispersion: pd.DataFrame, plot_dir: Path) -> list[Path]:
    if dispersion.empty:
        return []
    needed = {"source", "variable", "lead", "spread_rmse_ratio", "variance_mse_ratio"}
    if not needed.issubset(dispersion.columns):
        return []

    plt = _import_matplotlib()
    variables = available_variables(dispersion)
    outputs = []
    metrics = [
        ("spread_rmse_ratio", "Mean spread / RMSE", "dispersion_spread_rmse_ratio.png"),
        ("variance_mse_ratio", "Mean ensemble variance / MSE", "dispersion_variance_mse_ratio.png"),
    ]
    for metric, ylabel, filename in metrics:
        fig, axes = plt.subplots(1, len(variables), figsize=(5.6 * len(variables), 4.0), squeeze=False)
        for ax, variable in zip(axes[0], variables):
            sub = dispersion[dispersion["variable"] == variable]
            for source, group in sub.groupby("source"):
                group = group.sort_values("lead")
                ax.plot(
                    group["lead"].to_numpy(dtype=float),
                    group[metric].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.8,
                    label=source_title(source),
                )
            ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--")
            ax.set_title(variable_title(variable))
            ax.set_xlabel("Lead week")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sorted(sub["lead"].dropna().unique()))
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(ylabel, fontsize=13)
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Wrote {path}")
        outputs.append(path)
    return outputs


def plot_rank_histograms(rank_df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    if rank_df.empty:
        return []
    needed = {"source", "variable", "lead", "rank_bin", "rank_bin_count", "relative_frequency"}
    if not needed.issubset(rank_df.columns):
        return []

    plt = _import_matplotlib()
    outputs = []
    for (source, variable), group in rank_df.groupby(["source", "variable"]):
        leads = sorted(group["lead"].dropna().unique())
        if not leads:
            continue
        fig, axes = plt.subplots(1, len(leads), figsize=(3.4 * len(leads), 3.2), squeeze=False, sharey=True)
        for ax, lead in zip(axes[0], leads):
            sub = group[group["lead"] == lead].sort_values("rank_bin")
            x = sub["rank_bin"].to_numpy(dtype=float)
            y = sub["relative_frequency"].to_numpy(dtype=float)
            expected = sub["expected_relative_frequency"].to_numpy(dtype=float)
            ax.bar(x, y, width=0.85, color="#4C78A8", alpha=0.82)
            if expected.size:
                ax.axhline(float(np.nanmean(expected)), color="0.25", linestyle="--", linewidth=1.0)
            ax.set_title(f"Week {int(lead)}")
            ax.set_xlabel("Rank bin")
            ax.grid(True, axis="y", alpha=0.22)
        axes[0][0].set_ylabel("Relative frequency")
        fig.suptitle(f"{source_title(source)} rank histogram: {variable_title(variable)}", fontsize=12)
        fig.tight_layout()
        path = plot_dir / f"rank_histogram_{source}_{variable}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Wrote {path}")
        outputs.append(path)
    return outputs


def plot_diagnostics_dashboard(summary: pd.DataFrame, dispersion: pd.DataFrame, plot_dir: Path) -> Path | None:
    if summary.empty:
        return None
    plt = _import_matplotlib()
    variables = available_variables(summary, dispersion)
    if not variables:
        return None

    nrows = len(variables)
    columns = [
        ("crps_skill_pct_mean", "CRPS skill (%)", True),
        ("rmse_skill_pct_mean", "RMSE skill (%)", True),
        ("model_spread_rmse_ratio_mean", "FlowMatch spread / RMSE", False),
    ]
    fig, axes = plt.subplots(nrows, len(columns), figsize=(13.5, 3.6 * nrows), squeeze=False)
    lead_colors = plt.cm.viridis(np.linspace(0.12, 0.88, max(1, int(summary["lead"].nunique()))))

    for row_idx, variable in enumerate(variables):
        sub = summary[summary["variable"] == variable]
        leads = sorted(sub["lead"].dropna().unique())
        for col_idx, (metric, ylabel, zero_line) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            panel_label = f"({chr(ord('a') + row_idx * len(columns) + col_idx)})"
            metric_base = metric[:-5] if metric.endswith("_mean") else metric
            for color, lead in zip(lead_colors, leads):
                line = sub[sub["lead"] == lead].sort_values("member_count")
                low_col = f"{metric_base}_p05"
                high_col = f"{metric_base}_p95"
                if low_col in line and high_col in line:
                    ax.fill_between(
                        line["member_count"].to_numpy(dtype=float),
                        line[low_col].to_numpy(dtype=float),
                        line[high_col].to_numpy(dtype=float),
                        color=color,
                        alpha=0.14,
                        linewidth=0,
                    )
                ax.plot(
                    line["member_count"].to_numpy(dtype=float),
                    line[metric].to_numpy(dtype=float),
                    marker="o",
                    markersize=4.2,
                    linewidth=1.8,
                    label=f"W{int(lead)}",
                    color=color,
                )
            if zero_line:
                ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
            if "spread" in metric:
                ax.axhline(1.0, color="0.35", linewidth=0.9, linestyle="--")
            ax.set_title(f"{panel_label} {ylabel}", fontsize=11, fontweight="bold", loc="left", pad=8)
            ax.set_xlabel("Generated members")
            ax.set_ylabel(variable_title(variable) if col_idx == 0 else ylabel)
            ax.grid(True, alpha=0.22, linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            ax.legend(frameon=False, fontsize=7.5, ncol=2, handlelength=1.6)

    fig.suptitle("Extreme-event ensemble diagnostics", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    path = plot_dir / "ensemble_diagnostics_dashboard.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def make_diagnostic_plots(out_dir: Path) -> list[str]:
    summary = read_existing_csv(out_dir / "ensemble_size_summary.csv", required=True)
    bootstrap = read_existing_csv(out_dir / "ensemble_size_bootstrap_ci.csv")
    dispersion = read_existing_csv(out_dir / "dispersion_summary.csv")
    rank_df = read_existing_csv(out_dir / "dispersion_rank_histogram.csv")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    dashboard = plot_diagnostics_dashboard(summary, dispersion, plot_dir)
    if dashboard is not None:
        outputs.append(dashboard)
    for metric, ylabel, title, filename, zero_line in [
        (
            "crps_skill_pct",
            "CRPS skill vs GEOS (%)",
            "Skill as generated ensemble size increases",
            "ensemble_size_crps_skill.png",
            True,
        ),
        (
            "rmse_skill_pct",
            "RMSE skill vs GEOS (%)",
            "Ensemble-mean RMSE skill as generated ensemble size increases",
            "ensemble_size_rmse_skill.png",
            True,
        ),
        (
            "model_spread_rmse_ratio",
            "FlowMatch spread / RMSE",
            "Dispersion as generated ensemble size increases",
            "ensemble_size_spread_rmse_ratio.png",
            False,
        ),
    ]:
        path = plot_ensemble_size_metric(summary, bootstrap, metric, ylabel, title, plot_dir / filename, zero_line)
        if path is not None:
            outputs.append(path)
    outputs.extend(plot_dispersion_summary(dispersion, plot_dir))
    outputs.extend(plot_rank_histograms(rank_df, plot_dir))

    manifest = {
        "created_at": timestamp_now_utc(),
        "plot_count": len(outputs),
        "plots": [str(path) for path in outputs],
    }
    manifest_path = plot_dir / "plot_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")
    return [str(path) for path in outputs] + [str(manifest_path)]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.plot_only:
        summary = read_existing_csv(out_dir / "ensemble_size_summary.csv")
        report_path = write_member_count_report(summary, out_dir, int(args.report_member_count))
        if not args.make_plots:
            if report_path is None:
                print("--plot_only requested with --no-make_plots; nothing to do.")
            return
        make_diagnostic_plots(out_dir)
        return

    region_bounds = parse_region_bounds(args.region_bounds, args.region)
    active_region = region_key(args, region_bounds)
    variables = parse_variables(args.variables)
    sample_sizes = parse_int_list(args.sample_sizes)
    skip_years = parse_int_set(args.skip_years)
    eval_mask_names = parse_eval_mask_list(args.eval_masks, args.eval_mask)
    lead_filter = parse_int_set(args.lead_values) if args.lead_values else None
    init_date_filter = parse_date_filter(args.init_dates)
    valid_date_filter = parse_date_filter(args.valid_dates)
    init_months = parse_month_filter(args.init_months)
    init_season_counts = parse_init_season_counts(args.init_season_counts)
    balanced_monthly_inits_per_year = int(args.balanced_monthly_inits_per_year)
    one_init_per_month_per_year = bool(args.one_init_per_month_per_year)
    extreme_event_count = int(args.extreme_event_count)
    event_regions = parse_region_list(args.extreme_event_regions) if extreme_event_count > 0 else []
    event_variables = parse_variables(args.extreme_event_variable) if extreme_event_count > 0 else []
    if init_season_counts and (init_months or args.max_inits_per_year is not None):
        raise ValueError("--init_season_counts cannot be combined with --init_months or --max_inits_per_year.")
    if one_init_per_month_per_year and balanced_monthly_inits_per_year:
        raise ValueError("--one_init_per_month_per_year cannot be combined with --balanced_monthly_inits_per_year.")
    if one_init_per_month_per_year and (
        init_months
        or init_date_filter
        or init_season_counts
        or args.max_inits_per_year is not None
        or int(args.init_stride) != 1
        or extreme_event_count > 0
    ):
        raise ValueError(
            "--one_init_per_month_per_year cannot be combined with --init_months, --init_dates, "
            "--init_season_counts, --max_inits_per_year, --init_stride, or --extreme_event_count."
        )
    if balanced_monthly_inits_per_year < 0:
        raise ValueError("--balanced_monthly_inits_per_year must be >= 0.")
    if balanced_monthly_inits_per_year and (
        init_months
        or init_date_filter
        or init_season_counts
        or args.max_inits_per_year is not None
        or int(args.init_stride) != 1
        or extreme_event_count > 0
    ):
        raise ValueError(
            "--balanced_monthly_inits_per_year cannot be combined with --init_months, "
            "--init_dates, --init_season_counts, --max_inits_per_year, --init_stride, "
            "or --extreme_event_count."
        )
    if int(args.init_stride) < 1:
        raise ValueError("--init_stride must be >= 1.")
    if int(args.init_stride_offset) < 0 or int(args.init_stride_offset) >= int(args.init_stride):
        raise ValueError("--init_stride_offset must be in [0, --init_stride - 1].")
    if init_season_counts and int(args.init_stride) != 1:
        raise ValueError("--init_stride cannot be combined with --init_season_counts.")
    use_event_regions = extreme_event_count > 0
    use_static_region_mask = (
        not use_event_regions
        and (args.spatial_reduction == "regional_mean"
        or active_region != "global"
        or bool(str(args.region_bounds or "").strip()))
    )
    years = [year for year in range(args.start_year, args.end_year + 1) if year not in skip_years]
    years = validate_forecast_stores(args.forecast_dir, years, args.allow_missing_years)
    rng = np.random.default_rng(args.seed)
    balanced_months_by_year = assign_balanced_months_by_year(
        years,
        balanced_monthly_inits_per_year,
        rng,
    )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty. Use --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_event_cases = select_extreme_event_cases(
        args,
        years,
        lead_filter,
        init_months,
        init_date_filter,
        valid_date_filter,
        event_regions,
        event_variables,
    )
    selected_event_cases_by_year: dict[int, list[dict[str, object]]] = {}
    for event in selected_event_cases:
        selected_event_cases_by_year.setdefault(int(event["year"]), []).append(event)

    metadata = {
        "forecast_dir": os.path.abspath(args.forecast_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "skip_years": sorted(skip_years),
        "variables": variables,
        "sample_sizes_requested": sample_sizes,
        "member_bootstrap_repeats": args.member_bootstrap_repeats,
        "case_bootstrap_repeats": args.case_bootstrap_repeats,
        "seed": args.seed,
        "eval_mask": args.eval_mask,
        "eval_masks": eval_mask_names,
        "land_mask_file": os.path.abspath(args.land_mask_file) if args.land_mask_file else None,
        "rank_bins": args.rank_bins,
        "rank_histogram": not args.skip_rank_histogram,
        "spatial_reduction": args.spatial_reduction,
        "region": active_region if use_static_region_mask else None,
        "region_bounds": region_bounds if use_static_region_mask else None,
        "init_dates": sorted(init_date_filter),
        "valid_dates": sorted(valid_date_filter),
        "init_months": sorted(init_months),
        "init_season_counts_requested": init_season_counts,
        "balanced_monthly_inits_per_year": balanced_monthly_inits_per_year,
        "balanced_monthly_months_by_year": {
            str(year): months for year, months in sorted(balanced_months_by_year.items())
        },
        "one_init_per_month_per_year": one_init_per_month_per_year,
        "init_stride": int(args.init_stride),
        "init_stride_offset": int(args.init_stride_offset),
        "max_inits_per_year": args.max_inits_per_year,
        "extreme_event_count_requested": extreme_event_count,
        "extreme_event_count_per_lead": bool(args.extreme_event_count_per_lead),
        "extreme_event_variables": event_variables,
        "extreme_event_regions": event_regions,
        "extreme_event_max_per_region": args.extreme_event_max_per_region if use_event_regions else None,
        "selected_extreme_events": selected_event_cases,
        "started_at": timestamp_now_utc(),
    }

    eval_masks_by_name: dict[str, np.ndarray] = {}
    weights = None
    metric_masks_by_name: dict[str, np.ndarray] = {}
    metric_weights_by_name: dict[str, np.ndarray] = {}
    processed_cases = 0
    case_rows_by_mask: dict[str, list[dict[str, object]]] = {mask_name: [] for mask_name in eval_mask_names}
    dispersion_state_by_mask: dict[str, dict[tuple[str, str, int], dict[str, float]]] = {
        mask_name: {} for mask_name in eval_mask_names
    }
    rank_state_by_mask: dict[str, dict[tuple[str, str, int, int], np.ndarray]] = {
        mask_name: {} for mask_name in eval_mask_names
    }
    selected_init_counts_by_year: dict[str, int] = {}
    selected_init_season_counts_by_year: dict[str, dict[str, int]] = {}
    available_init_season_counts_by_year: dict[str, dict[str, int]] = {}
    selected_balanced_monthly_dates_by_year: dict[str, dict[str, str]] = {}
    available_balanced_monthly_counts_by_year: dict[str, dict[str, int]] = {}
    start_time = time.time()

    for year in years:
        path = store_path(args.forecast_dir, year)
        print(f"Opening {path}")
        ds = xr.open_zarr(path, consolidated=False, chunks=None)
        try:
            lats, lons = get_lat_lon(ds, VARIABLES[variables[0]]["model"])
            if not eval_masks_by_name:
                weights = area_weights_from_lats(lats, len(lons))
                for eval_mask_name in eval_mask_names:
                    eval_mask = load_eval_mask(args, (len(lats), len(lons)), eval_mask_name)
                    if use_static_region_mask:
                        region_mask = region_mask_from_bounds(lats, lons, region_bounds)
                        eval_mask = eval_mask & region_mask
                        kept = int(np.sum(eval_mask))
                        if kept <= 0:
                            raise ValueError(
                                f"Region mask for {region_bounds['name']} kept zero grid cells. "
                                f"Bounds={region_bounds}"
                            )
                        mask_label = "Regional mean" if args.spatial_reduction == "regional_mean" else "Region filter"
                        print(
                            f"{mask_label}: {region_bounds['name']} "
                            f"lat={region_bounds['lat_min']}..{region_bounds['lat_max']} "
                            f"lon={region_bounds['lon_min']}..{region_bounds['lon_max']}; "
                            f"kept {kept}/{eval_mask.size} grid cells after eval_mask={eval_mask_name}"
                        )
                    if args.spatial_reduction == "regional_mean":
                        metric_weights = np.ones((1, 1), dtype=np.float64)
                        metric_mask = np.ones((1, 1), dtype=bool)
                    else:
                        metric_weights = weights
                        metric_mask = eval_mask
                    kept = int(np.sum(eval_mask))
                    print(f"Evaluation mask: {eval_mask_name}; kept {kept}/{eval_mask.size} grid cells")
                    eval_masks_by_name[eval_mask_name] = eval_mask
                    metric_weights_by_name[eval_mask_name] = metric_weights
                    metric_masks_by_name[eval_mask_name] = metric_mask
            lead_pairs = lead_index_values(ds, VARIABLES[variables[0]]["model"])
            if lead_filter is not None:
                lead_pairs = [(idx, lead) for idx, lead in lead_pairs if lead in lead_filter]
            sample_var = VARIABLES[variables[0]]["model"]
            total_inits = init_count(ds, sample_var)
            case_specs: list[dict[str, object]] = []
            if use_event_regions:
                year_events = selected_event_cases_by_year.get(int(year), [])
                selected_init_counts_by_year[str(year)] = len({int(event["init_idx"]) for event in year_events})
                if year_events:
                    print(f"Extreme-event cases for {year}: {len(year_events)} region/init/lead cases")
                case_specs = [dict(event) for event in year_events]
            else:
                if one_init_per_month_per_year:
                    init_indices, selected_dates, available_counts = filtered_init_indices_by_months_random(
                        ds,
                        sample_var,
                        list(range(1, 13)),
                        rng,
                    )
                    selected_balanced_monthly_dates_by_year[str(year)] = selected_dates
                    available_balanced_monthly_counts_by_year[str(year)] = available_counts
                    selected_by_season = {}
                    available_by_season = {}
                elif balanced_months_by_year:
                    year_months = balanced_months_by_year[int(year)]
                    init_indices, selected_dates, available_counts = filtered_init_indices_by_months_random(
                        ds,
                        sample_var,
                        year_months,
                        rng,
                    )
                    selected_balanced_monthly_dates_by_year[str(year)] = selected_dates
                    available_balanced_monthly_counts_by_year[str(year)] = available_counts
                    selected_by_season = {}
                    available_by_season = {}
                elif init_season_counts:
                    init_indices, selected_by_season, available_by_season = filtered_init_indices_by_season_counts(
                        ds, sample_var, init_season_counts
                    )
                    selected_init_season_counts_by_year[str(year)] = selected_by_season
                    available_init_season_counts_by_year[str(year)] = available_by_season
                else:
                    init_indices = filtered_init_indices(ds, sample_var, init_months)
                    init_indices = filter_init_indices_by_dates(ds, init_indices, init_date_filter)
                    init_indices = apply_init_stride(
                        init_indices, int(args.init_stride), int(args.init_stride_offset)
                    )
                    if args.max_inits_per_year is not None:
                        init_indices = init_indices[: int(args.max_inits_per_year)]
                    selected_by_season = {}
                    available_by_season = {}
                selected_init_counts_by_year[str(year)] = len(init_indices)
                if one_init_per_month_per_year:
                    selected_text = ",".join(
                        f"{month:02d}={selected_balanced_monthly_dates_by_year[str(year)][str(month)]}"
                        for month in range(1, 13)
                    )
                    print(
                        f"One-init-per-month filter for {year}: {selected_text}; "
                        f"selected {len(init_indices)}/{total_inits} init dates"
                    )
                elif balanced_months_by_year:
                    selected_text = ",".join(
                        f"{int(month):02d}={selected_balanced_monthly_dates_by_year[str(year)][str(month)]}"
                        for month in balanced_months_by_year[int(year)]
                    )
                    print(
                        f"Balanced monthly init filter for {year}: {selected_text}; "
                        f"selected {len(init_indices)}/{total_inits} init dates"
                    )
                elif init_season_counts:
                    selected_text = ",".join(
                        f"{season}={selected_by_season[season]}" for season in init_season_counts
                    )
                    available_text = ",".join(
                        f"{season}={available_by_season[season]}" for season in init_season_counts
                    )
                    print(
                        f"Init season-count filter for {year}: selected {selected_text}; "
                        f"available {available_text}; total selected {len(init_indices)}/{total_inits}"
                    )
                elif (
                    init_months
                    or init_date_filter
                    or args.max_inits_per_year is not None
                    or int(args.init_stride) != 1
                ):
                    month_text = ",".join(str(month) for month in sorted(init_months)) if init_months else "all"
                    init_date_text = ",".join(sorted(init_date_filter)) if init_date_filter else "all"
                    print(
                        f"Init filter for {year}: months={month_text}; "
                        f"init_dates={init_date_text}; "
                        f"stride={int(args.init_stride)} offset={int(args.init_stride_offset)}; "
                        f"selected {len(init_indices)}/{total_inits} init dates"
                    )
                for init_idx in init_indices:
                    for lead_idx, lead_value in lead_pairs:
                        case_specs.append(
                            {
                                "year": int(year),
                                "init_idx": int(init_idx),
                                "lead_idx": int(lead_idx),
                                "lead": int(lead_value),
                                "region": active_region if use_static_region_mask else "",
                                "region_name": region_bounds["name"] if use_static_region_mask else "",
                                "event_rank": "",
                                "event_selection_lead": "",
                                "event_score": np.nan,
                                "event_score_variable": "",
                            }
                        )

            for case_spec in case_specs:
                init_idx = int(case_spec["init_idx"])
                lead_idx = int(case_spec["lead_idx"])
                lead_value = int(case_spec["lead"])
                init_time, valid_time = case_times(ds, init_idx, lead_idx, lead_value)
                if init_date_filter and date_key(init_time) not in init_date_filter:
                    continue
                if valid_date_filter and date_key(valid_time) not in valid_date_filter:
                    continue
                init_month, init_season = valid_month_season(init_time)
                valid_month, valid_season = valid_month_season(valid_time)
                case_region = str(case_spec.get("region", ""))
                case_region_name = str(case_spec.get("region_name", ""))
                case_event_variable = str(case_spec.get("event_score_variable", ""))
                case_variables = variables
                if use_event_regions and case_event_variable:
                    if case_event_variable not in variables:
                        continue
                    case_variables = [case_event_variable]
                case_suffix = f"_{case_region}" if case_region else ""
                if case_event_variable:
                    case_suffix = f"{case_suffix}_{case_event_variable}"
                case_id = f"{year}_{init_idx:04d}_lead{lead_value}{case_suffix}"
                for variable in case_variables:
                    spec = VARIABLES[variable]
                    obs = load_obs_array(ds, spec["obs"], init_idx, lead_idx)
                    model = load_forecast_array(ds, spec["model"], init_idx, lead_idx)
                    geos = load_forecast_array(ds, spec["geos"], init_idx, lead_idx)
                    model_members = int(model.shape[0])
                    geos_members = int(geos.shape[0])
                    usable_sizes = [size for size in sample_sizes if size <= model_members]
                    if not usable_sizes:
                        usable_sizes = [model_members]

                    for eval_mask_name in eval_mask_names:
                        case_eval_mask = eval_masks_by_name[eval_mask_name]
                        if use_event_regions:
                            event_region_bounds = REGIONS[case_region]
                            case_eval_mask = case_eval_mask & region_mask_from_bounds(lats, lons, event_region_bounds)
                            if int(np.sum(case_eval_mask)) <= 0:
                                raise ValueError(
                                    f"Event region {case_region!r} kept zero grid cells during evaluation."
                                )
                        if args.spatial_reduction == "regional_mean":
                            model_eval, geos_eval, obs_eval, case_weights, case_mask = regional_mean_metric_inputs(
                                model, geos, obs, weights, case_eval_mask
                            )
                        else:
                            model_eval, geos_eval, obs_eval = model, geos, obs
                            case_weights = metric_weights_by_name[eval_mask_name]
                            case_mask = case_eval_mask

                        geos_fields = metric_fields(geos_eval, obs_eval)
                        model_full_fields = metric_fields(model_eval, obs_eval)
                        common_full = case_mask & geos_fields["finite"] & model_full_fields["finite"]
                        geos_reduced = reduce_fields(geos_fields, case_weights, common_full, "geos")
                        model_full_reduced = reduce_fields(model_full_fields, case_weights, common_full, "model")
                        update_dispersion_state(
                            dispersion_state_by_mask[eval_mask_name],
                            "geos",
                            variable,
                            lead_value,
                            geos_reduced,
                            "geos",
                        )
                        update_dispersion_state(
                            dispersion_state_by_mask[eval_mask_name],
                            "model",
                            variable,
                            lead_value,
                            model_full_reduced,
                            "model",
                        )
                        if not args.skip_rank_histogram:
                            update_rank_state(
                                rank_state_by_mask[eval_mask_name],
                                "geos",
                                variable,
                                lead_value,
                                geos_eval,
                                obs_eval,
                                case_weights,
                                case_mask,
                                rng,
                                args.rank_bins,
                            )
                            update_rank_state(
                                rank_state_by_mask[eval_mask_name],
                                "model",
                                variable,
                                lead_value,
                                model_eval,
                                obs_eval,
                                case_weights,
                                case_mask,
                                rng,
                                args.rank_bins,
                            )

                        for size in usable_sizes:
                            repeats = 1 if size >= model_members else max(1, int(args.member_bootstrap_repeats))
                            for member_repeat in range(repeats):
                                if size >= model_members:
                                    member_idx = np.arange(model_members)
                                else:
                                    member_idx = rng.choice(model_members, size=size, replace=False)
                                sample = model_eval[member_idx, :, :]
                                sample_fields = metric_fields(sample, obs_eval)
                                common = case_mask & geos_fields["finite"] & sample_fields["finite"]
                                model_reduced = reduce_fields(sample_fields, case_weights, common, "model")
                                geos_for_sample = reduce_fields(geos_fields, case_weights, common, "geos")
                                row = {
                                    "case_id": case_id,
                                    "year": year,
                                    "init_index": init_idx,
                                    "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
                                    "init_month": init_month,
                                    "init_season": init_season,
                                    "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
                                    "valid_month": valid_month,
                                    "valid_season": valid_season,
                                    "lead": int(lead_value),
                                    "variable": variable,
                                    "member_count": int(size),
                                    "member_repeat": int(member_repeat),
                                    "model_members_available": model_members,
                                    "geos_members_available": geos_members,
                                    "spatial_reduction": args.spatial_reduction,
                                    "eval_mask": eval_mask_name,
                                    "region": case_region,
                                    "region_name": case_region_name,
                                    "event_rank": case_spec.get("event_rank", ""),
                                    "event_selection_lead": case_spec.get("event_selection_lead", ""),
                                    "event_score": case_spec.get("event_score", np.nan),
                                    "event_score_variable": case_spec.get("event_score_variable", ""),
                                }
                                row.update(model_reduced)
                                row.update(geos_for_sample)
                                row.update(row_metrics_from_sums(row))
                                case_rows_by_mask[eval_mask_name].append(row)
                processed_cases += 1
                if processed_cases % 20 == 0:
                    elapsed = (time.time() - start_time) / 60.0
                    print(f"Processed {processed_cases} init/lead cases in {elapsed:.1f} min")
        finally:
            ds.close()

    if not any(case_rows_by_mask.values()):
        raise RuntimeError("No forecast cases were processed. Check --forecast_dir, years, variables, and filters.")

    written_by_mask = {}
    for eval_mask_name in eval_mask_names:
        mask_rows = case_rows_by_mask[eval_mask_name]
        if not mask_rows:
            print(f"No rows for eval mask {eval_mask_name}; skipping output.")
            continue
        mask_out_dir = out_dir if len(eval_mask_names) == 1 else out_dir / eval_mask_name
        mask_metadata = dict(metadata)
        mask_metadata["eval_mask"] = eval_mask_name
        mask_metadata["out_dir"] = os.path.abspath(mask_out_dir)
        written_by_mask[eval_mask_name] = write_evaluation_outputs(
            mask_rows,
            dispersion_state_by_mask[eval_mask_name],
            rank_state_by_mask[eval_mask_name],
            mask_out_dir,
            args,
            rng,
            mask_metadata,
            start_time,
            processed_cases,
            selected_init_counts_by_year,
            selected_init_season_counts_by_year,
            available_init_season_counts_by_year,
            selected_balanced_monthly_dates_by_year,
            available_balanced_monthly_counts_by_year,
        )

    if len(eval_mask_names) > 1:
        manifest = {
            "created_at": timestamp_now_utc(),
            "forecast_dir": os.path.abspath(args.forecast_dir),
            "out_dir": os.path.abspath(out_dir),
            "eval_masks": eval_mask_names,
            "mask_output_dirs": {
                mask_name: str((out_dir / mask_name).resolve())
                for mask_name in eval_mask_names
                if mask_name in written_by_mask
            },
        }
        with open(out_dir / "ensemble_test_metadata.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {out_dir / 'ensemble_test_metadata.json'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from None
