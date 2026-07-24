#!/usr/bin/env python3
"""Compare raw FIM, quantile-mapped FIM, and FlowMatch on Fig. 5 extremes.

The defaults reproduce the extreme-case selection used by the original
2021--2023 Figure 5 member-convergence run:

* 30 observed precipitation cases and 30 observed T2M cases (60 total);
* lead weeks 3 and 4;
* the ``global_extremes`` set of 15 regional boxes;
* at most two selected cases per region and variable; and
* gridpoint scores area-averaged within each selected event region.

Systems:

* ``raw4``: the four native FIMr1p1/GEOS members;
* ``qm4``: the same four members after frozen 1999--2019 quantile mapping;
* ``flow8``: repeated random eight-member FlowMatch subsets; and
* ``flow4``: repeated random four-member subsets (equal-member control).

CRPS, ensemble-mean RMSE, and q95 quantile score are written per event.
Summaries include both the pooled regional score used by the Figure 5
evaluator and the mean over events of per-event skill. The latter is the
quantity described in the manuscript's extreme-event table caption.

The evaluator is login-node friendly: it opens one year at a time, loads one
selected event at a time, writes each completed event atomically, and resumes
by skipping existing event files. Event selection itself is checkpointed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_model import evaluate_ensemble_tests_flow_finalv1_global as ens  # noqa: E402


SYSTEM_MEMBERS = {"raw4": 4, "qm4": 4, "flow4": 4, "flow8": 8}
FIXED_SYSTEMS = {"raw4", "qm4"}
COMPARISONS = (
    ("qm4_vs_raw4", "qm4", "raw4"),
    ("flow8_vs_raw4", "flow8", "raw4"),
    ("flow8_vs_qm4", "flow8", "qm4"),
    ("flow4_vs_raw4", "flow4", "raw4"),
    ("flow4_vs_qm4", "flow4", "qm4"),
)
SUM_COLUMNS = (
    "weight_sum",
    "crps_sum",
    "sse_sum",
    "bias_sum",
    "spread_sum",
    "q95_score_sum",
    "q95_sse_sum",
)
SCORE_COLUMNS = ("crps", "rmse", "bias", "spread", "q95_score", "q95_rmse")
SKILL_METRICS = {
    "crps_skill_pct": "crps",
    "rmse_skill_pct": "rmse",
    "q95_skill_pct": "q95_score",
    "q95_rmse_skill_pct": "q95_rmse",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-dir",
        "--forecast_dir",
        dest="forecast_dir",
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Directory containing source YEAR.zarr stores.",
    )
    parser.add_argument(
        "--qm-dir",
        "--qm_dir",
        dest="qm_dir",
        required=True,
        help="Directory containing corrected YEAR.zarr stores with qm_pr/qm_t2m.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "qm_extreme_fig5_t2m30_pr30_2021_2023_wk3wk4"
        ),
    )
    parser.add_argument("--start-year", "--start_year", dest="start_year", type=int, default=2021)
    parser.add_argument("--end-year", "--end_year", dest="end_year", type=int, default=2023)
    parser.add_argument("--variables", default="pr,t2m")
    parser.add_argument("--lead-values", "--lead_values", dest="lead_values", default="3,4")
    parser.add_argument(
        "--extreme-event-count",
        "--extreme_event_count",
        dest="extreme_event_count",
        type=int,
        default=30,
        help="Cases selected per variable across the requested leads (30+30 = 60 by default).",
    )
    parser.add_argument(
        "--extreme-event-variable",
        "--extreme_event_variable",
        dest="extreme_event_variable",
        default="t2m,pr",
    )
    parser.add_argument(
        "--extreme-event-regions",
        "--extreme_event_regions",
        dest="extreme_event_regions",
        default="global_extremes",
    )
    parser.add_argument(
        "--extreme-event-max-per-region",
        "--extreme_event_max_per_region",
        dest="extreme_event_max_per_region",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--extreme-event-count-per-lead",
        "--extreme_event_count_per_lead",
        dest="extreme_event_count_per_lead",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Off reproduces the original 60-case W3-W4 Figure 5 selection.",
    )
    parser.add_argument("--eval-mask", "--eval_mask", dest="eval_mask",
                        choices=("all", "land", "ocean"), default="all")
    parser.add_argument("--land-mask-file", "--land_mask_file", dest="land_mask_file", default=None)
    parser.add_argument("--flow-repeats", "--flow_repeats", dest="flow_repeats", type=int, default=50)
    parser.add_argument("--case-bootstrap-repeats", "--case_bootstrap_repeats",
                        dest="case_bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this evaluator's existing output directory and start again.",
    )
    return parser.parse_args()


def timestamp_now_utc() -> str:
    return pd.Timestamp.now("UTC").isoformat()


def parse_variables(text: str) -> list[str]:
    return ens.parse_variables(text)


def config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "forecast_dir": str(Path(args.forecast_dir).resolve()),
        "qm_dir": str(Path(args.qm_dir).resolve()),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "variables": parse_variables(args.variables),
        "lead_values": sorted(ens.parse_int_set(args.lead_values)),
        "extreme_event_count": int(args.extreme_event_count),
        "extreme_event_variable": parse_variables(args.extreme_event_variable),
        "extreme_event_regions": ens.parse_region_list(args.extreme_event_regions),
        "extreme_event_max_per_region": int(args.extreme_event_max_per_region),
        "extreme_event_count_per_lead": bool(args.extreme_event_count_per_lead),
        "eval_mask": str(args.eval_mask),
        "land_mask_file": (
            str(Path(args.land_mask_file).resolve()) if args.land_mask_file else None
        ),
        "flow_repeats": int(args.flow_repeats),
        "seed": int(args.seed),
        "q95": 0.95,
        "systems": SYSTEM_MEMBERS,
    }


def config_digest(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    temporary.replace(path)


def prepare_output(args: argparse.Namespace, config: dict[str, object]) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir = out_dir / "case_metrics"
    case_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "run_config.json"
    requested = {"config_digest": config_digest(config), "config": config}
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("config_digest") != requested["config_digest"]:
            raise ValueError(
                f"{out_dir} contains a different run configuration. "
                "Use another --out-dir or explicitly pass --overwrite."
            )
    else:
        atomic_write_json(requested, config_path)
    return out_dir, case_dir


def selection_args(args: argparse.Namespace) -> argparse.Namespace:
    """Provide the attributes consumed by ens.select_extreme_event_cases."""
    return argparse.Namespace(
        forecast_dir=args.forecast_dir,
        extreme_event_count=int(args.extreme_event_count),
        extreme_event_max_per_region=int(args.extreme_event_max_per_region),
        extreme_event_count_per_lead=bool(args.extreme_event_count_per_lead),
        eval_mask=args.eval_mask,
        land_mask_file=args.land_mask_file,
    )


def select_or_load_events(
    args: argparse.Namespace,
    out_dir: Path,
    years: list[int],
) -> list[dict[str, object]]:
    path = out_dir / "selected_extreme_events.json"
    if path.exists():
        events = json.loads(path.read_text())
        print(f"Resuming with {len(events)} checkpointed extreme cases from {path}", flush=True)
        return events

    print("Selecting observed regional extremes (one observation field at a time) ...", flush=True)
    events = ens.select_extreme_event_cases(
        selection_args(args),
        years,
        set(ens.parse_int_set(args.lead_values)),
        set(),
        set(),
        set(),
        ens.parse_region_list(args.extreme_event_regions),
        parse_variables(args.extreme_event_variable),
    )
    atomic_write_json(events, path)
    print(f"Checkpointed {len(events)} selected cases to {path}", flush=True)
    return events


def coordinate_values(ds: xr.Dataset, name: str) -> np.ndarray:
    if name in ds.coords:
        return np.asarray(ds.coords[name].values)
    if name in ds:
        return np.asarray(ds[name].values)
    raise KeyError(f"Missing coordinate {name!r}.")


def validate_qm_archive(source: xr.Dataset, qm: xr.Dataset, year: int) -> None:
    source_init = pd.to_datetime(coordinate_values(source, "init")).normalize()
    qm_init = pd.to_datetime(coordinate_values(qm, "init")).normalize()
    if not np.array_equal(source_init.values, qm_init.values):
        raise ValueError(f"QM/source initialization coordinates differ for {year}.")
    if not np.array_equal(coordinate_values(source, "lead"), coordinate_values(qm, "lead")):
        raise ValueError(f"QM/source lead coordinates differ for {year}.")
    for coord in ("lat", "lon"):
        if not np.allclose(
            coordinate_values(source, coord),
            coordinate_values(qm, coord),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"QM/source {coord} coordinates differ for {year}.")


def q95_score_map(forecast: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quantile = np.nanquantile(np.asarray(forecast, dtype=np.float64), 0.95, axis=0)
    obs64 = np.asarray(obs, dtype=np.float64)
    error = quantile - obs64
    loss = np.where(obs64 >= quantile, 0.95 * (obs64 - quantile), 0.05 * error)
    return loss, error * error


def system_sums(
    forecast: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    fields = ens.metric_fields(forecast, obs)
    q95_score, q95_sse = q95_score_map(forecast, obs)
    finite = (
        np.asarray(mask, dtype=bool)
        & fields["finite"]
        & np.isfinite(q95_score)
        & np.isfinite(q95_sse)
    )
    weighted = np.where(finite, np.asarray(weights, dtype=np.float64), 0.0)
    weight_sum = float(np.sum(weighted))
    if weight_sum <= 0.0:
        return {column: 0.0 for column in SUM_COLUMNS}
    mean_error = np.nanmean(np.asarray(forecast, dtype=np.float64), axis=0) - obs
    return {
        "weight_sum": weight_sum,
        "crps_sum": float(np.sum(np.where(finite, fields["crps"], 0.0) * weighted)),
        "sse_sum": float(np.sum(np.where(finite, fields["sse"], 0.0) * weighted)),
        "bias_sum": float(np.sum(np.where(finite, mean_error, 0.0) * weighted)),
        "spread_sum": float(np.sum(np.where(finite, fields["spread"], 0.0) * weighted)),
        "q95_score_sum": float(np.sum(np.where(finite, q95_score, 0.0) * weighted)),
        "q95_sse_sum": float(np.sum(np.where(finite, q95_sse, 0.0) * weighted)),
    }


def scores_from_sums(sums: dict[str, float] | pd.Series) -> dict[str, float]:
    weight = float(sums.get("weight_sum", 0.0))
    if weight <= 0.0:
        return {column: np.nan for column in SCORE_COLUMNS}
    return {
        "crps": float(sums["crps_sum"] / weight),
        "rmse": float(math.sqrt(max(float(sums["sse_sum"]) / weight, 0.0))),
        "bias": float(sums["bias_sum"] / weight),
        "spread": float(sums["spread_sum"] / weight),
        "q95_score": float(sums["q95_score_sum"] / weight),
        "q95_rmse": float(math.sqrt(max(float(sums["q95_sse_sum"]) / weight, 0.0))),
    }


def event_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def event_case_id(event: dict[str, object]) -> str:
    return (
        f"{int(event['year'])}_{int(event['init_idx']):04d}_"
        f"lead{int(event['lead'])}_{event['region']}_{event['event_score_variable']}"
    )


def common_metadata(
    event: dict[str, object],
    case_id: str,
    init_time: pd.Timestamp,
    valid_time: pd.Timestamp,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "year": int(event["year"]),
        "init_index": int(event["init_idx"]),
        "init_time": "" if pd.isna(init_time) else init_time.isoformat(),
        "valid_time": "" if pd.isna(valid_time) else valid_time.isoformat(),
        "lead": int(event["lead"]),
        "variable": str(event["event_score_variable"]),
        "region": str(event["region"]),
        "region_name": str(event["region_name"]),
        "event_rank": int(event["event_rank"]),
        "event_score": float(event["event_score"]),
    }


def metric_row(
    metadata: dict[str, object],
    system: str,
    member_repeat: int,
    sums: dict[str, float],
) -> dict[str, object]:
    row = dict(metadata)
    row.update(
        {
            "system": system,
            "member_count": SYSTEM_MEMBERS[system],
            "member_repeat": int(member_repeat),
        }
    )
    row.update(sums)
    row.update(scores_from_sums(sums))
    return row


def evaluate_one_event(
    event: dict[str, object],
    source: xr.Dataset,
    qm_ds: xr.Dataset,
    args: argparse.Namespace,
    lats: np.ndarray,
    lons: np.ndarray,
    weights: np.ndarray,
    base_mask: np.ndarray,
) -> pd.DataFrame:
    variable = str(event["event_score_variable"])
    spec = ens.VARIABLES[variable]
    init_idx = int(event["init_idx"])
    lead_idx = int(event["lead_idx"])
    lead_value = int(event["lead"])
    init_time, valid_time = ens.case_times(source, init_idx, lead_idx, lead_value)
    region = str(event["region"])
    case_mask = base_mask & ens.region_mask_from_bounds(lats, lons, ens.REGIONS[region])
    if int(np.sum(case_mask)) <= 0:
        raise ValueError(f"Event region {region!r} kept zero cells.")

    # Only one selected case is resident. The 90-member field is the largest
    # object (~24 MiB on a 181x360 float32 grid).
    obs = ens.load_obs_array(source, spec["obs"], init_idx, lead_idx)
    flow = ens.load_forecast_array(source, spec["model"], init_idx, lead_idx)
    raw = ens.load_forecast_array(source, spec["geos"], init_idx, lead_idx)
    qm = ens.load_forecast_array(qm_ds, f"qm_{variable}", init_idx, lead_idx)
    if raw.shape[0] != 4 or qm.shape[0] != 4:
        raise ValueError(
            f"{event_case_id(event)} expected four raw/QM members, "
            f"found raw={raw.shape[0]}, qm={qm.shape[0]}."
        )
    if flow.shape[0] < 8:
        raise ValueError(f"{event_case_id(event)} has only {flow.shape[0]} FlowMatch members.")

    # Discard non-event cells before the repeated member calculations. This
    # keeps every temporary score array regional rather than global.
    event_weights = np.asarray(weights[case_mask], dtype=np.float64)[None, :]
    metric_mask = np.ones(event_weights.shape, dtype=bool)
    obs = np.asarray(obs[case_mask], dtype=np.float32)[None, :]
    flow = np.asarray(flow[:, case_mask], dtype=np.float32)[:, None, :]
    raw = np.asarray(raw[:, case_mask], dtype=np.float32)[:, None, :]
    qm = np.asarray(qm[:, case_mask], dtype=np.float32)[:, None, :]

    case_id = event_case_id(event)
    metadata = common_metadata(event, case_id, init_time, valid_time)
    rows = [
        metric_row(metadata, "raw4", 0, system_sums(raw, obs, event_weights, metric_mask)),
        metric_row(metadata, "qm4", 0, system_sums(qm, obs, event_weights, metric_mask)),
    ]
    rng = np.random.default_rng(event_seed(int(args.seed), case_id))
    repeats = max(1, int(args.flow_repeats))
    for repeat in range(repeats):
        index8 = rng.choice(flow.shape[0], size=8, replace=False)
        index4 = rng.choice(flow.shape[0], size=4, replace=False)
        rows.append(
            metric_row(
                metadata,
                "flow8",
                repeat,
                system_sums(flow[index8], obs, event_weights, metric_mask),
            )
        )
        rows.append(
            metric_row(
                metadata,
                "flow4",
                repeat,
                system_sums(flow[index4], obs, event_weights, metric_mask),
            )
        )
    return pd.DataFrame(rows)


def evaluate_events(
    events: list[dict[str, object]],
    args: argparse.Namespace,
    case_dir: Path,
) -> None:
    events_by_year: dict[int, list[dict[str, object]]] = {}
    for event in events:
        events_by_year.setdefault(int(event["year"]), []).append(event)

    completed = 0
    total = len(events)
    start = time.time()
    for year, year_events in sorted(events_by_year.items()):
        source_path = Path(args.forecast_dir) / f"{year}.zarr"
        qm_path = Path(args.qm_dir) / f"{year}.zarr"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source archive: {source_path}")
        if not qm_path.exists():
            raise FileNotFoundError(
                f"Missing QM archive: {qm_path}. Run qm_fim_baseline.py --stage evaluate first."
            )
        print(f"Opening {year}: {len(year_events)} selected cases", flush=True)
        source = xr.open_zarr(source_path, consolidated=False, chunks=None)
        qm_ds = xr.open_zarr(qm_path, consolidated=False, chunks=None)
        try:
            validate_qm_archive(source, qm_ds, year)
            sample_variable = str(year_events[0]["event_score_variable"])
            lats, lons = ens.get_lat_lon(source, ens.VARIABLES[sample_variable]["model"])
            weights = ens.area_weights_from_lats(lats, len(lons))
            base_mask = ens.load_eval_mask(args, (len(lats), len(lons)))
            for event in year_events:
                case_id = event_case_id(event)
                checkpoint = case_dir / f"{case_id}.csv"
                if checkpoint.exists():
                    completed += 1
                    print(f"  skip completed {case_id} ({completed}/{total})", flush=True)
                    continue
                frame = evaluate_one_event(
                    event, source, qm_ds, args, lats, lons, weights, base_mask
                )
                atomic_write_csv(frame, checkpoint)
                completed += 1
                elapsed = (time.time() - start) / 60.0
                print(
                    f"  saved {case_id} ({completed}/{total}, {elapsed:.1f} min)",
                    flush=True,
                )
        finally:
            source.close()
            qm_ds.close()


def read_case_metrics(case_dir: Path, expected_count: int) -> pd.DataFrame:
    paths = sorted(case_dir.glob("*.csv"))
    if len(paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} completed event files, found {len(paths)} in {case_dir}."
        )
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if frame["case_id"].nunique() != expected_count:
        raise RuntimeError("Completed event files do not contain the expected distinct case IDs.")
    return frame


def add_all_lead_rows(frame: pd.DataFrame) -> pd.DataFrame:
    all_leads = frame.copy()
    all_leads["lead"] = "all"
    original = frame.copy()
    original["lead"] = original["lead"].astype(str)
    return pd.concat([original, all_leads], ignore_index=True)


def sum_scores(group: pd.DataFrame) -> dict[str, float]:
    sums = {column: float(group[column].sum()) for column in SUM_COLUMNS}
    return scores_from_sums(sums)


def system_summary(case_df: pd.DataFrame) -> pd.DataFrame:
    work = add_all_lead_rows(case_df)
    repeat_rows: list[dict[str, object]] = []
    for keys, group in work.groupby(
        ["variable", "lead", "system", "member_count", "member_repeat"], sort=True
    ):
        variable, lead, system, member_count, member_repeat = keys
        row = {
            "variable": variable,
            "lead": lead,
            "system": system,
            "member_count": int(member_count),
            "member_repeat": int(member_repeat),
            "n_cases": int(group["case_id"].nunique()),
        }
        row.update(sum_scores(group))
        for metric in SCORE_COLUMNS:
            row[f"case_mean_{metric}"] = float(group[metric].mean())
        repeat_rows.append(row)
    repeats = pd.DataFrame(repeat_rows)

    rows: list[dict[str, object]] = []
    for keys, group in repeats.groupby(
        ["variable", "lead", "system", "member_count"], sort=True
    ):
        variable, lead, system, member_count = keys
        row = {
            "variable": variable,
            "lead": lead,
            "system": system,
            "member_count": int(member_count),
            "n_cases": int(group["n_cases"].max()),
            "n_member_repeats": int(group["member_repeat"].nunique()),
        }
        for metric in (*SCORE_COLUMNS, *(f"case_mean_{name}" for name in SCORE_COLUMNS)):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_p05"] = float(np.nanquantile(values, 0.05))
            row[f"{metric}_p95"] = float(np.nanquantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variable", "lead", "member_count"]).reset_index(drop=True)


def safe_skill(system_score: float, reference_score: float) -> float:
    if not np.isfinite(system_score) or not np.isfinite(reference_score) or reference_score <= 1e-12:
        return np.nan
    return 100.0 * (1.0 - system_score / reference_score)


def paired_repeat_rows(
    case_df: pd.DataFrame,
    comparison: str,
    system: str,
    reference: str,
) -> pd.DataFrame:
    system_rows = case_df[case_df["system"].eq(system)].copy()
    reference_rows = case_df[case_df["system"].eq(reference)].copy()
    if system in FIXED_SYSTEMS:
        system_rows["pair_repeat"] = 0
    else:
        system_rows["pair_repeat"] = system_rows["member_repeat"].astype(int)
    if reference in FIXED_SYSTEMS:
        repeat_values = sorted(system_rows["pair_repeat"].unique())
        reference_rows = pd.concat(
            [reference_rows.assign(pair_repeat=int(repeat)) for repeat in repeat_values],
            ignore_index=True,
        )
    else:
        reference_rows["pair_repeat"] = reference_rows["member_repeat"].astype(int)
    keys = ["case_id", "pair_repeat"]
    columns = [
        *keys,
        "variable",
        "lead",
        "region",
        "region_name",
        *SUM_COLUMNS,
        *SCORE_COLUMNS,
    ]
    merged = system_rows[columns].merge(
        reference_rows[[*keys, *SUM_COLUMNS, *SCORE_COLUMNS]],
        on=keys,
        suffixes=("_system", "_reference"),
        validate="one_to_one",
    )
    merged["comparison"] = comparison
    merged["system"] = system
    merged["reference"] = reference
    for skill_name, score_name in SKILL_METRICS.items():
        merged[skill_name] = [
            safe_skill(system_score, reference_score)
            for system_score, reference_score in zip(
                merged[f"{score_name}_system"],
                merged[f"{score_name}_reference"],
            )
        ]
    return merged


def comparison_case_rows(case_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            paired_repeat_rows(case_df, comparison, system, reference)
            for comparison, system, reference in COMPARISONS
        ],
        ignore_index=True,
    )


def comparison_repeat_metric(group: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for side in ("system", "reference"):
        sums = {
            column: float(group[f"{column}_{side}"].sum())
            for column in SUM_COLUMNS
        }
        for name, value in scores_from_sums(sums).items():
            out[f"{side}_{name}"] = value
    for skill_name, score_name in SKILL_METRICS.items():
        out[skill_name] = safe_skill(
            out[f"system_{score_name}"], out[f"reference_{score_name}"]
        )
        out[f"case_mean_{skill_name}"] = float(group[skill_name].mean())
    return out


def comparison_summary(comparison_cases: pd.DataFrame) -> pd.DataFrame:
    work = add_all_lead_rows(comparison_cases)
    repeat_rows: list[dict[str, object]] = []
    for keys, group in work.groupby(
        ["variable", "lead", "comparison", "system", "reference", "pair_repeat"],
        sort=True,
    ):
        variable, lead, comparison, system, reference, repeat = keys
        row = {
            "variable": variable,
            "lead": lead,
            "comparison": comparison,
            "system": system,
            "reference": reference,
            "pair_repeat": int(repeat),
            "n_cases": int(group["case_id"].nunique()),
        }
        row.update(comparison_repeat_metric(group))
        repeat_rows.append(row)
    repeats = pd.DataFrame(repeat_rows)

    metrics = [
        *(f"{side}_{score}" for side in ("system", "reference") for score in SCORE_COLUMNS),
        *SKILL_METRICS,
        *(f"case_mean_{skill}" for skill in SKILL_METRICS),
    ]
    rows: list[dict[str, object]] = []
    for keys, group in repeats.groupby(
        ["variable", "lead", "comparison", "system", "reference"], sort=True
    ):
        variable, lead, comparison, system, reference = keys
        row = {
            "variable": variable,
            "lead": lead,
            "comparison": comparison,
            "system": system,
            "reference": reference,
            "n_cases": int(group["n_cases"].max()),
            "n_member_repeats": int(group["pair_repeat"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_p05"] = float(np.nanquantile(values, 0.05))
            row[f"{metric}_p95"] = float(np.nanquantile(values, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["variable", "lead", "comparison"]
    ).reset_index(drop=True)


def regional_summary(comparison_cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in comparison_cases.groupby(
        ["variable", "lead", "region", "region_name", "comparison"], sort=True
    ):
        variable, lead, region, region_name, comparison = keys
        row = {
            "variable": variable,
            "lead": int(lead),
            "region": region,
            "region_name": region_name,
            "comparison": comparison,
            "n_cases": int(group["case_id"].nunique()),
        }
        for metric in SKILL_METRICS:
            by_repeat = group.groupby("pair_repeat")[metric].mean()
            row[f"case_mean_{metric}"] = float(by_repeat.mean())
            row[f"case_mean_{metric}_p05"] = float(by_repeat.quantile(0.05))
            row[f"case_mean_{metric}_p95"] = float(by_repeat.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def case_bootstrap(
    comparison_cases: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if repeats <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    work = add_all_lead_rows(comparison_cases)
    rows: list[dict[str, object]] = []
    grouping = ["variable", "lead", "comparison", "system", "reference"]
    for keys, group in work.groupby(grouping, sort=True):
        cases = sorted(group["case_id"].unique())
        repeat_values = sorted(group["pair_repeat"].unique())
        indexed = {
            int(repeat): group[group["pair_repeat"].eq(repeat)].set_index("case_id", drop=False)
            for repeat in repeat_values
        }
        boot = {
            metric: []
            for metric in (
                *SKILL_METRICS,
                *(f"case_mean_{name}" for name in SKILL_METRICS),
            )
        }
        for _ in range(repeats):
            repeat = int(rng.choice(repeat_values))
            sample_ids = rng.choice(cases, size=len(cases), replace=True)
            sample = indexed[repeat].loc[sample_ids]
            values = comparison_repeat_metric(sample)
            for metric in boot:
                boot[metric].append(values[metric])
        row = dict(zip(grouping, keys))
        row["n_cases"] = len(cases)
        row["n_bootstrap_repeats"] = int(repeats)
        for metric, values in boot.items():
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]
            row[f"{metric}_p025"] = float(np.quantile(array, 0.025)) if array.size else np.nan
            row[f"{metric}_p50"] = float(np.quantile(array, 0.50)) if array.size else np.nan
            row[f"{metric}_p975"] = float(np.quantile(array, 0.975)) if array.size else np.nan
            row[f"{metric}_p_gt0"] = float(np.mean(array > 0.0)) if array.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    out_dir: Path,
) -> Path:
    if not os.environ.get("MPLCONFIGDIR"):
        cache = Path(os.environ.get("TMPDIR") or "/tmp") / "geos_subc_matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected_comparisons = ["qm4_vs_raw4", "flow8_vs_raw4", "flow8_vs_qm4", "flow4_vs_qm4"]
    labels = ["QM4 / raw4", "Flow8 / raw4", "Flow8 / QM4", "Flow4 / QM4"]
    metric_specs = [
        ("case_mean_crps_skill_pct", "CRPS skill (%)"),
        ("case_mean_rmse_skill_pct", "RMSE skill (%)"),
        ("case_mean_q95_skill_pct", "q95-score skill (%)"),
    ]
    leads = [lead for lead in ("3", "4") if lead in set(summary["lead"].astype(str))]
    colors = {"3": "#4a7fb5", "4": "#3b2f7d"}
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharex=True)
    x = np.arange(len(selected_comparisons), dtype=float)
    width = 0.36
    for row_index, variable in enumerate(("pr", "t2m")):
        for column_index, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row_index, column_index]
            for lead_index, lead in enumerate(leads):
                values = []
                lower = []
                upper = []
                for comparison in selected_comparisons:
                    cell = summary[
                        summary["variable"].eq(variable)
                        & summary["lead"].astype(str).eq(lead)
                        & summary["comparison"].eq(comparison)
                    ]
                    value = float(cell[f"{metric}_mean"].iloc[0]) if not cell.empty else np.nan
                    values.append(value)
                    ci = bootstrap[
                        bootstrap["variable"].eq(variable)
                        & bootstrap["lead"].astype(str).eq(lead)
                        & bootstrap["comparison"].eq(comparison)
                    ]
                    if ci.empty:
                        lower.append(np.nan)
                        upper.append(np.nan)
                    else:
                        lo = float(ci[f"{metric}_p025"].iloc[0])
                        hi = float(ci[f"{metric}_p975"].iloc[0])
                        lower.append(max(0.0, value - lo))
                        upper.append(max(0.0, hi - value))
                position = x + (lead_index - (len(leads) - 1) / 2.0) * width
                errors = np.asarray([lower, upper], dtype=float)
                ax.bar(
                    position,
                    values,
                    width=width * 0.92,
                    color=colors.get(lead, "#4a7fb5"),
                    label=f"W{lead}",
                    yerr=errors,
                    capsize=2.5,
                    error_kw={"linewidth": 0.8},
                )
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.set_title(
                f"({chr(97 + row_index * 3 + column_index)}) "
                f"{'PR' if variable == 'pr' else 'T2M'} {ylabel}",
                loc="left",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=22, ha="right")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Observed regional extremes, 2021–2023 (30 PR + 30 T2M cases; mean per-event skill)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = out_dir / "qm_flow_extreme_comparison.png"
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def print_report(summary: pd.DataFrame) -> None:
    keep = summary[
        summary["comparison"].isin(
            ["qm4_vs_raw4", "flow8_vs_raw4", "flow8_vs_qm4", "flow4_vs_qm4"]
        )
    ].copy()
    columns = [
        "variable",
        "lead",
        "comparison",
        "n_cases",
        "case_mean_crps_skill_pct_mean",
        "case_mean_rmse_skill_pct_mean",
        "case_mean_q95_skill_pct_mean",
        "crps_skill_pct_mean",
        "rmse_skill_pct_mean",
        "q95_skill_pct_mean",
    ]
    print("\nExtreme-event comparison")
    print("  case_mean_* = mean of per-event regional skill (manuscript definition)")
    print("  final three columns = pooled regional skill (original Fig. 5 evaluator definition)")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(keep[columns].round(3).to_string(index=False))


def write_outputs(
    case_df: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
    config: dict[str, object],
    started_at: str,
) -> None:
    comparisons = comparison_case_rows(case_df)
    systems = system_summary(case_df)
    summary = comparison_summary(comparisons)
    regions = regional_summary(comparisons)
    bootstrap = case_bootstrap(
        comparisons, int(args.case_bootstrap_repeats), int(args.seed) + 991
    )

    atomic_write_csv(case_df, out_dir / "extreme_case_system_metrics.csv")
    atomic_write_csv(comparisons, out_dir / "extreme_case_comparisons.csv")
    atomic_write_csv(systems, out_dir / "extreme_system_summary.csv")
    atomic_write_csv(summary, out_dir / "extreme_comparison_summary.csv")
    atomic_write_csv(regions, out_dir / "extreme_regional_summary.csv")
    if not bootstrap.empty:
        atomic_write_csv(bootstrap, out_dir / "extreme_comparison_case_bootstrap_ci.csv")
    plot = make_plot(summary, bootstrap, out_dir) if args.make_plots else None
    metadata = {
        "config_digest": config_digest(config),
        "config": config,
        "started_at": started_at,
        "completed_at": timestamp_now_utc(),
        "processed_cases": int(case_df["case_id"].nunique()),
        "case_counts": (
            case_df.drop_duplicates("case_id").groupby("variable").size().astype(int).to_dict()
        ),
        "aggregation_note": (
            "case_mean_* is the mean across event-specific area-averaged skills; "
            "unprefixed skill is computed from scores pooled over all regional grid cells and cases."
        ),
        "outputs": {
            "case_system_metrics": str(out_dir / "extreme_case_system_metrics.csv"),
            "case_comparisons": str(out_dir / "extreme_case_comparisons.csv"),
            "system_summary": str(out_dir / "extreme_system_summary.csv"),
            "comparison_summary": str(out_dir / "extreme_comparison_summary.csv"),
            "regional_summary": str(out_dir / "extreme_regional_summary.csv"),
            "case_bootstrap_ci": (
                str(out_dir / "extreme_comparison_case_bootstrap_ci.csv")
                if not bootstrap.empty
                else None
            ),
            "plot": str(plot) if plot else None,
        },
    }
    atomic_write_json(metadata, out_dir / "metadata.json")
    print_report(summary)
    print(f"\nWrote completed metadata to {out_dir / 'metadata.json'}")


def main() -> None:
    args = parse_args()
    if int(args.start_year) > int(args.end_year):
        raise ValueError("--start-year must be <= --end-year.")
    if int(args.flow_repeats) < 1:
        raise ValueError("--flow-repeats must be >= 1.")
    if int(args.extreme_event_count) < 1:
        raise ValueError("--extreme-event-count must be >= 1.")
    config = config_from_args(args)
    out_dir, case_dir = prepare_output(args, config)
    started_at = timestamp_now_utc()
    years = list(range(int(args.start_year), int(args.end_year) + 1))
    ens.validate_forecast_stores(args.forecast_dir, years, False)
    events = select_or_load_events(args, out_dir, years)
    expected = int(args.extreme_event_count) * len(parse_variables(args.extreme_event_variable))
    if args.extreme_event_count_per_lead:
        expected *= len(ens.parse_int_set(args.lead_values))
    if len(events) != expected:
        raise RuntimeError(f"Expected {expected} selected cases, found {len(events)}.")
    evaluate_events(events, args, case_dir)
    case_df = read_case_metrics(case_dir, len(events))
    write_outputs(case_df, args, out_dir, config, started_at)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
