#!/usr/bin/env python3
"""Diagnose ECMWF event GRIB files before adding them to paper case figures.

The script is intentionally exploratory. It inventories every cfgrib group,
selects the likely event variable, extracts the event target-week samples by
valid date, writes a compact NetCDF product, and saves quick-look plots.

Default cases match the paper case studies:
  - UK July 2022 heatwave: T2M, target week 2022-07-14..2022-07-20
  - California Jan 2023 atmospheric-river event: PR, target week
    2023-01-05..2023-01-11

Example on Vista:
  python3 paper/scripts/diagnose_ecmwf_event_gribs.py \
    --uk-grib dataprocess/uk_heat.grib \
    --california-grib dataprocess/california_pr.grib \
    --output-dir dataprocess/ecmwf_event_grib_diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class EventCase:
    key: str
    label: str
    grib_default: str
    variable_kind: str
    target_start: str
    target_end: str
    plot_bbox: tuple[float, float, float, float]
    event_bbox: tuple[float, float, float, float]
    preferred_names: tuple[str, ...]
    preferred_param_ids: tuple[int, ...]


CASES: dict[str, EventCase] = {
    "uk_heat": EventCase(
        key="uk_heat",
        label="UK July 2022 heatwave",
        grib_default="dataprocess/uk_heat.grib",
        variable_kind="t2m",
        target_start="2022-07-14",
        target_end="2022-07-20",
        plot_bbox=(-11.0, 5.0, 48.0, 59.0),
        event_bbox=(-6.0, 2.0, 50.0, 56.0),
        preferred_names=("t2m", "2t", "167"),
        preferred_param_ids=(167,),
    ),
    "california_pr": EventCase(
        key="california_pr",
        label="California January 2023 atmospheric rivers",
        grib_default="dataprocess/california_pr.grib",
        variable_kind="pr",
        target_start="2023-01-05",
        target_end="2023-01-11",
        plot_bbox=(-128.0, -113.0, 31.0, 44.5),
        event_bbox=(-124.5, -117.0, 34.0, 41.5),
        preferred_names=("tp", "pr", "precip", "228"),
        preferred_param_ids=(228, 260048, 260015),
    ),
}


LAT_NAMES = ("latitude", "lat", "y")
LON_NAMES = ("longitude", "lon", "x")
MEMBER_DIMS = ("number", "realization", "member", "members", "ensemble", "perturbationNumber")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inventory and plot ECMWF event GRIB files for Fig. 7/8 comparison.",
    )
    parser.add_argument("--case", choices=("all", *CASES.keys()), default="all")
    parser.add_argument("--uk-grib", default=CASES["uk_heat"].grib_default)
    parser.add_argument("--california-grib", default=CASES["california_pr"].grib_default)
    parser.add_argument("--output-dir", default="dataprocess/ecmwf_event_grib_diagnostics")
    parser.add_argument(
        "--precip-mode",
        choices=("auto", "cumulative", "interval-accum", "rate", "raw"),
        default="cumulative",
        help=(
            "How to interpret precipitation values. Default assumes ECMWF total "
            "precipitation is accumulated by lead hour; auto tries to distinguish "
            "cumulative totals from interval accumulations."
        ),
    )
    parser.add_argument(
        "--cfgrib-indexpath",
        default="",
        help=(
            "cfgrib index path. Empty string disables .idx files, which is safer "
            "for scratch diagnostics but can be slower for large GRIBs."
        ),
    )
    parser.add_argument(
        "--grib-reader",
        choices=("auto", "cfgrib", "pygrib"),
        default="auto",
        help="GRIB reader backend. auto tries cfgrib first, then pygrib.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--max-members-panel", type=int, default=12)
    return parser.parse_args()


def log(lines: list[str], message: str = "") -> None:
    print(message)
    lines.append(message)


def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return obj.attrs.get(name, default)
    except Exception:
        return default


def _message_attr(message: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        try:
            return getattr(message, name)
        except Exception:
            pass
        try:
            return message[name]
        except Exception:
            pass
    return default


def _step_hours_from_message(message: Any) -> float:
    value = _message_attr(message, ("forecastTime", "endStep", "step"), None)
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    step_range = str(_message_attr(message, ("stepRange",), "") or "")
    if "-" in step_range:
        step_range = step_range.split("-")[-1]
    try:
        return float(step_range)
    except Exception:
        return float("nan")


def _message_time(message: Any) -> pd.Timestamp:
    value = _message_attr(message, ("analDate", "validityDate", "dataDate"), None)
    if value is not None:
        try:
            return pd.Timestamp(value)
        except Exception:
            pass
    data_date = _message_attr(message, ("dataDate",), None)
    data_time = int(_message_attr(message, ("dataTime",), 0) or 0)
    if data_date is not None:
        try:
            hour = data_time // 100
            minute = data_time % 100
            return pd.Timestamp(str(int(data_date))) + pd.Timedelta(hours=hour, minutes=minute)
        except Exception:
            pass
    return pd.NaT


def _message_valid_time(message: Any) -> pd.Timestamp:
    value = _message_attr(message, ("validDate",), None)
    if value is not None:
        try:
            return pd.Timestamp(value)
        except Exception:
            pass
    init_time = _message_time(message)
    step_hours = _step_hours_from_message(message)
    if pd.notna(init_time) and np.isfinite(step_hours):
        return init_time + pd.Timedelta(hours=float(step_hours))
    return pd.NaT


def _message_member(message: Any) -> int:
    for name in ("perturbationNumber", "number", "ensembleMember", "member"):
        value = _message_attr(message, (name,), None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    data_type = str(_message_attr(message, ("dataType",), "") or "").lower()
    if data_type in {"cf", "controlforecast", "control"}:
        return 0
    return 0


def _latlon_from_pygrib_message(message: Any) -> tuple[np.ndarray, np.ndarray]:
    lats2d, lons2d = message.latlons()
    lats2d = np.asarray(lats2d, dtype=np.float64)
    lons2d = np.asarray(lons2d, dtype=np.float64)
    if lats2d.ndim != 2 or lons2d.ndim != 2:
        raise ValueError("pygrib returned non-2D latitude/longitude arrays.")
    lat1d = lats2d[:, 0]
    lon1d = lons2d[0, :]
    return lat1d, lon1d


def _open_with_pygrib(path: Path) -> list[xr.Dataset]:
    try:
        import pygrib
    except Exception as exc:  # pragma: no cover - depends on Vista env
        raise RuntimeError("pygrib is not installed.") from exc

    groups: dict[tuple[Any, ...], list[Any]] = {}
    with pygrib.open(str(path)) as grbs:
        for message in grbs:
            key = (
                str(_message_attr(message, ("shortName",), "unknown")),
                int(_message_attr(message, ("paramId",), -1) or -1),
                str(_message_attr(message, ("typeOfLevel",), "unknown")),
                int(_message_attr(message, ("level",), 0) or 0),
                str(_message_attr(message, ("stepType",), "unknown")),
                str(_message_attr(message, ("units",), "")),
            )
            groups.setdefault(key, []).append(message)

    datasets: list[xr.Dataset] = []
    for group_idx, (key, messages) in enumerate(groups.items()):
        short_name, param_id, type_of_level, level, step_type, units = key
        lats, lons = _latlon_from_pygrib_message(messages[0])
        times = sorted({str(_message_time(message)) for message in messages})
        steps = sorted({_step_hours_from_message(message) for message in messages})
        members = sorted({_message_member(message) for message in messages})
        times_ts = pd.to_datetime(times)
        steps_td = pd.to_timedelta(steps, unit="h")

        time_lookup = {str(value): idx for idx, value in enumerate(times)}
        step_lookup = {float(value): idx for idx, value in enumerate(steps)}
        member_lookup = {int(value): idx for idx, value in enumerate(members)}

        data = np.full(
            (len(times), len(members), len(steps), len(lats), len(lons)),
            np.nan,
            dtype=np.float32,
        )
        valid_grid = np.empty((len(times), len(steps)), dtype="datetime64[ns]")
        valid_grid[:] = np.datetime64("NaT")

        for message in messages:
            time_key = str(_message_time(message))
            step_key = float(_step_hours_from_message(message))
            member_key = int(_message_member(message))
            ti = time_lookup[time_key]
            si = step_lookup[step_key]
            mi = member_lookup[member_key]
            values = np.asarray(message.values, dtype=np.float32)
            if values.shape != (len(lats), len(lons)):
                raise ValueError(
                    f"pygrib message shape {values.shape} does not match "
                    f"lat/lon shape {(len(lats), len(lons))}"
                )
            data[ti, mi, si, :, :] = values
            valid_grid[ti, si] = np.datetime64(_message_valid_time(message).to_datetime64())

        var_name = short_name if short_name and short_name != "unknown" else f"param{param_id}"
        ds = xr.Dataset(
            {
                var_name: (
                    ("time", "number", "step", "latitude", "longitude"),
                    data,
                    {
                        "GRIB_shortName": short_name,
                        "GRIB_paramId": param_id,
                        "GRIB_typeOfLevel": type_of_level,
                        "GRIB_level": level,
                        "GRIB_stepType": step_type,
                        "units": units,
                        "long_name": str(_message_attr(messages[0], ("name",), var_name)),
                        "reader": "pygrib",
                    },
                )
            },
            coords={
                "time": times_ts,
                "number": np.asarray(members, dtype=int),
                "step": steps_td,
                "latitude": lats,
                "longitude": lons,
                "valid_time": (("time", "step"), valid_grid),
            },
            attrs={"reader": "pygrib", "pygrib_group_index": group_idx},
        )
        datasets.append(ds)
    return datasets


def open_grib_datasets(path: Path, indexpath: str, reader: str = "auto") -> list[xr.Dataset]:
    errors: list[str] = []
    if reader in {"auto", "cfgrib"}:
        try:
            import cfgrib  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on Vista env
            errors.append(f"cfgrib unavailable: {exc}")
        else:
            backend_kwargs = {"indexpath": indexpath}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import cfgrib

                try:
                    return cfgrib.open_datasets(str(path), backend_kwargs=backend_kwargs)
                except Exception as exc:
                    errors.append(f"cfgrib failed to open file: {exc}")
                    if reader == "cfgrib":
                        raise

    if reader in {"auto", "pygrib"}:
        try:
            return _open_with_pygrib(path)
        except Exception as exc:
            errors.append(f"pygrib unavailable/failed: {exc}")
            if reader == "pygrib":
                raise

    details = "\n  - ".join(errors) if errors else "no reader attempted"
    raise RuntimeError(
        "Could not read GRIB file. Install one supported backend in geossub_env, "
        "then rerun the script.\n\n"
        "Recommended conda-forge install:\n"
        "  conda install -c conda-forge cfgrib eccodes\n\n"
        "Alternative:\n"
        "  conda install -c conda-forge pygrib\n\n"
        f"Reader errors:\n  - {details}"
    )

def coord_summary(values: np.ndarray, max_items: int = 5) -> str:
    arr = np.asarray(values)
    if arr.size == 0:
        return "empty"
    flat = arr.ravel()
    head = ", ".join(str(x) for x in flat[:max_items])
    if flat.size > max_items:
        head += ", ..."
    return f"shape={arr.shape}, first={head}"


def describe_dataset(ds: xr.Dataset, group_idx: int) -> list[str]:
    rows: list[str] = []
    rows.append(f"Group {group_idx}")
    rows.append(f"  dims: {dict(ds.sizes)}")
    coord_bits = []
    for name, coord in ds.coords.items():
        if name in ("latitude", "longitude", "lat", "lon", "time", "step", "valid_time", "number"):
            coord_bits.append(f"{name}: {coord_summary(coord.values)}")
    if coord_bits:
        rows.append("  coords:")
        rows.extend(f"    {bit}" for bit in coord_bits)
    rows.append("  data variables:")
    for var_name, da in ds.data_vars.items():
        attrs = da.attrs
        bits = {
            "shortName": attrs.get("GRIB_shortName"),
            "paramId": attrs.get("GRIB_paramId"),
            "typeOfLevel": attrs.get("GRIB_typeOfLevel"),
            "stepType": attrs.get("GRIB_stepType"),
            "units": attrs.get("units"),
            "long_name": attrs.get("long_name"),
        }
        bits_text = ", ".join(f"{k}={v}" for k, v in bits.items() if v is not None)
        rows.append(f"    {var_name}: dims={da.dims}, shape={da.shape}; {bits_text}")
    return rows


def find_coord_name(da: xr.DataArray, candidates: tuple[str, ...]) -> str | None:
    names = set(da.dims) | set(da.coords)
    for name in candidates:
        if name in names:
            return name
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def find_member_dim(da: xr.DataArray) -> str | None:
    for name in MEMBER_DIMS:
        if name in da.dims:
            return name
    for name in da.dims:
        lower = name.lower()
        if "member" in lower or "realization" in lower or lower == "number":
            return name
    return None


def variable_score(da: xr.DataArray, case: EventCase) -> int:
    attrs = da.attrs
    names = {
        str(da.name or "").lower(),
        str(attrs.get("GRIB_shortName", "")).lower(),
        str(attrs.get("shortName", "")).lower(),
        str(attrs.get("long_name", "")).lower(),
        str(attrs.get("standard_name", "")).lower(),
    }
    param = attrs.get("GRIB_paramId")
    try:
        param_int = int(param)
    except Exception:
        param_int = None

    score = 0
    if find_coord_name(da, LAT_NAMES) and find_coord_name(da, LON_NAMES):
        score += 5
    if param_int in case.preferred_param_ids:
        score += 30
    for preferred in case.preferred_names:
        preferred = preferred.lower()
        if any(preferred == name or preferred in name for name in names):
            score += 12
    if case.variable_kind == "t2m" and any("temperature" in name for name in names):
        score += 5
    if case.variable_kind == "pr" and any("precip" in name for name in names):
        score += 5
    if np.issubdtype(da.dtype, np.number):
        score += 2
    return score


def choose_variable(datasets: list[xr.Dataset], case: EventCase) -> tuple[int, str, xr.DataArray, list[tuple[int, str, int]]]:
    candidates: list[tuple[int, str, int]] = []
    for group_idx, ds in enumerate(datasets):
        for var_name, da in ds.data_vars.items():
            candidates.append((group_idx, var_name, variable_score(da, case)))
    if not candidates:
        raise ValueError("No data variables found in GRIB.")
    candidates = sorted(candidates, key=lambda item: item[2], reverse=True)
    group_idx, var_name, score = candidates[0]
    if score <= 0:
        raise ValueError("No plausible plottable variable found in GRIB.")
    return group_idx, var_name, datasets[group_idx][var_name], candidates


def timedelta_to_hours(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return arr.astype("timedelta64[s]").astype(np.float64) / 3600.0
    out = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").to_numpy(dtype=float)
    return out.reshape(arr.shape)


def as_sample_coord(stacked: xr.DataArray, name: str, sample_len: int) -> np.ndarray | None:
    if name not in stacked.coords:
        return None
    coord = stacked.coords[name]
    try:
        if "sample" in coord.dims:
            return np.asarray(coord.values)
        if coord.ndim == 0:
            return np.repeat(coord.values, sample_len)
        return np.asarray(coord.broadcast_like(stacked.isel({dim: 0 for dim in stacked.dims if dim != "sample"})).values)
    except Exception:
        if coord.ndim == 0:
            return np.repeat(coord.values, sample_len)
    return None


def build_member_sample_array(da: xr.DataArray) -> tuple[np.ndarray, dict[str, Any]]:
    lat_name = find_coord_name(da, LAT_NAMES)
    lon_name = find_coord_name(da, LON_NAMES)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not identify latitude/longitude coordinates for {da.name}.")

    member_dim = find_member_dim(da)
    work = da
    if member_dim is None:
        member_dim = "member"
        work = work.expand_dims({member_dim: [0]})

    sample_dims = [dim for dim in work.dims if dim not in {member_dim, lat_name, lon_name}]
    if sample_dims:
        stacked = work.stack(sample=sample_dims)
    else:
        stacked = work.expand_dims(sample=[0])

    sample_len = int(stacked.sizes["sample"])
    values = stacked.transpose(member_dim, "sample", lat_name, lon_name).load().values.astype(np.float64)

    lats = np.asarray(work[lat_name].values, dtype=float)
    lons = np.asarray(work[lon_name].values, dtype=float)
    members = np.asarray(work[member_dim].values)

    valid_values = as_sample_coord(stacked, "valid_time", sample_len)
    time_values = as_sample_coord(stacked, "time", sample_len)
    step_values = as_sample_coord(stacked, "step", sample_len)

    valid_times = None
    if valid_values is not None:
        valid_times = pd.to_datetime(valid_values.ravel(), errors="coerce")
    if valid_times is None or pd.isna(valid_times).all():
        if time_values is not None and step_values is not None:
            base = pd.to_datetime(time_values.ravel(), errors="coerce")
            step_td = pd.to_timedelta(step_values.ravel())
            valid_times = base + step_td
    if valid_times is None:
        valid_times = pd.to_datetime([pd.NaT] * sample_len)

    lead_hours = None
    if step_values is not None:
        lead_hours = timedelta_to_hours(step_values.ravel())
    elif time_values is not None and valid_times is not None:
        base = pd.to_datetime(time_values.ravel(), errors="coerce")
        lead_hours = (valid_times - base) / pd.Timedelta(hours=1)
        lead_hours = np.asarray(lead_hours, dtype=float)
    else:
        lead_hours = np.full(sample_len, np.nan)

    init_times = None
    if time_values is not None:
        init_times = pd.to_datetime(time_values.ravel(), errors="coerce")
    else:
        init_times = pd.to_datetime([pd.NaT] * sample_len)

    metadata = {
        "lat_name": lat_name,
        "lon_name": lon_name,
        "member_dim": member_dim,
        "sample_dims": sample_dims,
        "lats": lats,
        "lons": lons,
        "members": members,
        "valid_times": valid_times,
        "init_times": init_times,
        "lead_hours": np.asarray(lead_hours, dtype=float),
    }
    return values, metadata


def sort_samples(values: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    valid_times = pd.to_datetime(meta["valid_times"])
    lead_hours = np.asarray(meta["lead_hours"], dtype=float)
    fallback = np.where(np.isfinite(lead_hours), lead_hours, np.arange(values.shape[1], dtype=float))
    valid_ns = np.array([
        ts.value if not pd.isna(ts) else np.iinfo("int64").max
        for ts in valid_times
    ])
    order = np.lexsort((fallback, valid_ns))
    values = values[:, order, :, :]
    out = dict(meta)
    out["valid_times"] = valid_times[order]
    out["init_times"] = pd.to_datetime(meta["init_times"])[order]
    out["lead_hours"] = lead_hours[order]
    out["sample_order"] = order
    return values, out


def selected_sample_table(meta: dict[str, Any], selected: np.ndarray) -> pd.DataFrame:
    valid_times = pd.to_datetime(meta["valid_times"])
    init_times = pd.to_datetime(meta["init_times"])
    lead_hours = np.asarray(meta["lead_hours"], dtype=float)
    rows = []
    for idx in np.where(selected)[0]:
        rows.append(
            {
                "sample_index": int(idx),
                "init_time": None if pd.isna(init_times[idx]) else str(init_times[idx].date()),
                "valid_time": None if pd.isna(valid_times[idx]) else str(valid_times[idx]),
                "valid_date": None if pd.isna(valid_times[idx]) else str(valid_times[idx].date()),
                "lead_hour": None if not np.isfinite(lead_hours[idx]) else float(lead_hours[idx]),
            }
        )
    return pd.DataFrame(rows)


def target_mask(meta: dict[str, Any], case: EventCase) -> np.ndarray:
    valid_times = pd.to_datetime(meta["valid_times"])
    dates = valid_times.normalize()
    start = pd.Timestamp(case.target_start)
    end = pd.Timestamp(case.target_end)
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def infer_precip_mode(values: np.ndarray, attrs: dict[str, Any], selected: np.ndarray, user_mode: str) -> tuple[str, str]:
    if user_mode != "auto":
        return user_mode, f"forced by --precip-mode={user_mode}"

    step_type = str(attrs.get("GRIB_stepType", "")).lower()
    units = str(attrs.get("units", "")).lower()
    short_name = str(attrs.get("GRIB_shortName", "")).lower()

    if "kg m**-2 s**-1" in units or "kg m-2 s-1" in units or units in {"mm/s", "m/s"}:
        return "rate", f"auto: units look like a rate ({units})"

    if "accum" in step_type or short_name in {"tp", "cp", "lsp"}:
        # Use a domain/member mean time series to decide whether values are
        # cumulative since initialization. This is diagnostic, not a scientific
        # proof, but it catches the common ECMWF total-precipitation encoding.
        with np.errstate(invalid="ignore"):
            series = np.nanmean(values, axis=(0, 2, 3))
        diffs = np.diff(series)
        finite = np.isfinite(diffs)
        if finite.any():
            nondecreasing = float(np.mean(diffs[finite] >= -1e-8))
            if nondecreasing >= 0.85:
                return "cumulative", f"auto: {nondecreasing:.0%} of mean-step differences are nonnegative"
        return "interval-accum", "auto: accumulation stepType, but series is not clearly cumulative"

    if "m" in units or "mm" in units:
        selected_count = int(np.sum(selected))
        return "interval-accum", f"auto: precipitation-like units ({units}); selected samples={selected_count}"

    return "raw", "auto: no precipitation unit/step hint found"


def convert_temperature(values: np.ndarray, selected: np.ndarray, attrs: dict[str, Any]) -> tuple[np.ndarray, str, str]:
    subset = values[:, selected, :, :]
    with np.errstate(invalid="ignore"):
        week = np.nanmean(subset, axis=1)
    units = str(attrs.get("units", "") or "")
    finite = week[np.isfinite(week)]
    if "K" in units or (finite.size and np.nanmedian(finite) > 150.0):
        return week, "K", "Averaged selected valid times; kept T2M in Kelvin."
    if units.lower() in {"c", "degc", "degree celsius", "celsius"}:
        return week + 273.15, "K", "Averaged selected valid times; converted Celsius to Kelvin."
    return week, units or "unknown", "Averaged selected valid times without temperature unit conversion."


def unit_multiplier_to_mm(values: np.ndarray, units: str, mode: str) -> tuple[np.ndarray, str]:
    units_l = str(units or "").lower()
    out = values.copy()
    note = ""
    if mode == "rate":
        if "m/s" in units_l:
            out = out * 1000.0 * 86400.0
            note = "Converted m/s rate to mm/day."
        elif "kg" in units_l and "s" in units_l:
            out = out * 86400.0
            note = "Converted kg m-2 s-1 water-equivalent rate to mm/day."
        else:
            out = out * 86400.0
            note = "Assumed rate and multiplied by 86400 to get daily amount."
    else:
        if units_l.strip() in {"m", "metre", "meter"} or " m" == units_l.strip() or units_l == "m":
            out = out * 1000.0
            note = "Converted metres of water to mm."
        elif "m of water" in units_l:
            out = out * 1000.0
            note = "Converted metres of water equivalent to mm."
        elif "mm" in units_l:
            note = "Values already look like mm."
        else:
            note = f"Units are {units!r}; no confident accumulation unit conversion applied."
    return out, note


def convert_precipitation(
    values: np.ndarray,
    selected: np.ndarray,
    attrs: dict[str, Any],
    meta: dict[str, Any],
    user_mode: str,
    target_days: int,
) -> tuple[np.ndarray, str, str]:
    mode, mode_note = infer_precip_mode(values, attrs, selected, user_mode)
    units = str(attrs.get("units", "") or "")
    selected_idx = np.where(selected)[0]
    if selected_idx.size == 0:
        raise ValueError("No selected precipitation samples.")

    if mode == "cumulative":
        increments = []
        missing_previous = []
        lead = np.asarray(meta["lead_hours"], dtype=float)
        init_times = pd.to_datetime(meta["init_times"])
        for idx in selected_idx:
            previous_candidates = np.arange(idx)
            if len(previous_candidates):
                same_init = init_times[previous_candidates] == init_times[idx]
                if pd.isna(init_times[idx]):
                    same_init = np.ones_like(previous_candidates, dtype=bool)
                if np.isfinite(lead[idx]):
                    same_init &= np.isfinite(lead[previous_candidates]) & (lead[previous_candidates] < lead[idx])
                previous_candidates = previous_candidates[same_init]
            if len(previous_candidates) == 0:
                missing_previous.append(int(idx))
                continue
            if np.isfinite(lead[idx]) and np.any(np.isfinite(lead[previous_candidates])):
                prev_idx = int(previous_candidates[np.nanargmax(lead[previous_candidates])])
            else:
                prev_idx = int(previous_candidates[-1])
            inc = values[:, idx, :, :] - values[:, prev_idx, :, :]
            increments.append(inc)
        if not increments:
            raise ValueError("Cumulative precipitation selected the first sample only; no previous step for differencing.")
        inc_arr = np.stack(increments, axis=1)
        inc_arr = np.where(inc_arr < -1e-9, np.nan, inc_arr)
        inc_arr, unit_note = unit_multiplier_to_mm(inc_arr, units, mode)
        with np.errstate(invalid="ignore"):
            week = np.nansum(inc_arr, axis=1) / float(target_days)
        note = (
            f"{mode_note}. Treated as cumulative accumulation: differenced each selected step "
            f"from the previous lead hour for the same initialization, summed target increments, "
            f"divided by {target_days} days. {unit_note}"
        )
        if missing_previous:
            note += f" Missing previous step for selected indices {missing_previous}; skipped those increments."
        return week, "mm/day", note

    if mode == "interval-accum":
        subset = values[:, selected_idx, :, :]
        subset, unit_note = unit_multiplier_to_mm(subset, units, mode)
        lead = np.asarray(meta["lead_hours"], dtype=float)
        selected_lead = lead[selected_idx]
        finite_lead = selected_lead[np.isfinite(selected_lead)]
        if finite_lead.size >= 2:
            spacing = float(np.nanmedian(np.diff(np.sort(finite_lead))))
        else:
            spacing = math.nan
        if np.isfinite(spacing) and spacing < 23.5:
            with np.errstate(invalid="ignore"):
                week = np.nansum(subset, axis=1) / float(target_days)
            spacing_note = f"median selected spacing is {spacing:.1f} h, so summed interval accumulations over target week"
        else:
            with np.errstate(invalid="ignore"):
                week = np.nanmean(subset, axis=1)
            spacing_note = "selected samples look daily/coarser, so averaged interval accumulations"
        note = f"{mode_note}. {spacing_note}. {unit_note}"
        return week, "mm/day", note

    if mode == "rate":
        subset = values[:, selected_idx, :, :]
        subset, unit_note = unit_multiplier_to_mm(subset, units, mode)
        with np.errstate(invalid="ignore"):
            week = np.nanmean(subset, axis=1)
        return week, "mm/day", f"{mode_note}. Averaged selected rates. {unit_note}"

    subset = values[:, selected_idx, :, :]
    with np.errstate(invalid="ignore"):
        week = np.nanmean(subset, axis=1)
    return week, units or "raw", f"{mode_note}. Raw selected-sample average."


def wrap_and_sort_grid(
    fields: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    out = {key: np.asarray(value) for key, value in fields.items()}
    lat_arr = np.asarray(lats, dtype=float)
    lon_arr = np.asarray(lons, dtype=float)
    if np.nanmax(lon_arr) > 180.0:
        lon_arr = ((lon_arr + 180.0) % 360.0) - 180.0
    lon_order = np.argsort(lon_arr)
    lat_order = np.argsort(lat_arr)
    lon_arr = lon_arr[lon_order]
    lat_arr = lat_arr[lat_order]
    for key, value in out.items():
        out[key] = value[..., lat_order, :][..., lon_order]
    return out, lat_arr, lon_arr


def weighted_region_mean(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    if not lon_mask.any() or not lat_mask.any():
        return float("nan")
    sub = field[np.ix_(lat_mask, lon_mask)]
    weights = np.cos(np.deg2rad(lats[lat_mask]))[:, None]
    finite = np.isfinite(sub)
    if not finite.any():
        return float("nan")
    return float(np.nansum(np.where(finite, sub, 0.0) * weights) / np.nansum(np.where(finite, weights, 0.0)))


def member_region_values(member_fields: np.ndarray, lats: np.ndarray, lons: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray([weighted_region_mean(member_fields[i], lats, lons, bbox) for i in range(member_fields.shape[0])])


def robust_range(field: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(field)[np.isfinite(field)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.nanmean(finite))
        spread = max(abs(center) * 0.05, 1.0)
        return center - spread, center + spread
    return float(vmin), float(vmax)


def add_bbox(ax: plt.Axes, bbox: tuple[float, float, float, float], color: str = "black") -> None:
    lon_min, lon_max, lat_min, lat_max = bbox
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            linestyle="-",
        )
    )


def plot_map(ax: plt.Axes, lons: np.ndarray, lats: np.ndarray, field: np.ndarray,
             title: str, label: str, bbox: tuple[float, float, float, float],
             cmap: str = "viridis") -> None:
    vmin, vmax = robust_range(field)
    levels = np.linspace(vmin, vmax, 21)
    mesh = ax.contourf(lons, lats, np.clip(field, vmin, vmax), levels=levels, cmap=cmap)
    add_bbox(ax, bbox)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(bbox[0] - 2.0, bbox[1] + 2.0)
    ax.set_ylim(bbox[2] - 1.5, bbox[3] + 1.5)
    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08, shrink=0.88)
    cbar.set_label(label)


def plot_summary(
    case: EventCase,
    fields: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    member_region: np.ndarray,
    units: str,
    output_dir: Path,
    dpi: int,
    max_members_panel: int,
) -> list[Path]:
    output_paths: list[Path] = []
    mean = fields["ensemble_mean"]
    std = fields["ensemble_std"]
    q95 = fields["ensemble_q95"]
    q50 = fields["ensemble_q50"]
    bbox = case.plot_bbox

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    plot_map(axes[0, 0], lons, lats, mean, "ECMWF ensemble mean", units, bbox)
    plot_map(axes[0, 1], lons, lats, std, "ECMWF ensemble spread", units, bbox, cmap="magma")
    plot_map(axes[1, 0], lons, lats, q95, "ECMWF ensemble q95", units, bbox)
    ax = axes[1, 1]
    finite = member_region[np.isfinite(member_region)]
    if finite.size:
        ax.hist(finite, bins=min(20, max(5, finite.size // 3)), color="#1f5fa8", alpha=0.82)
        ax.axvline(np.nanmean(finite), color="black", linewidth=1.2, label="mean")
        ax.axvline(np.nanpercentile(finite, 95), color="#c07a2b", linewidth=1.2, label="q95")
        ax.legend(fontsize=8)
    ax.set_title("Event-box member means", fontsize=10, fontweight="bold")
    ax.set_xlabel(units)
    ax.set_ylabel("member count")
    fig.suptitle(f"{case.label}: ECMWF target-week diagnostics", fontsize=13, fontweight="bold")
    out = output_dir / f"{case.key}_ecmwf_week4_summary.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    output_paths.append(out)

    members = fields["member_weekly_mean"]
    n_panel = min(max_members_panel, members.shape[0])
    if n_panel > 1:
        ncols = 4
        nrows = int(math.ceil(n_panel / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False, constrained_layout=True)
        vmin, vmax = robust_range(members[:n_panel])
        levels = np.linspace(vmin, vmax, 19)
        for idx, ax in enumerate(axes.ravel()):
            if idx >= n_panel:
                ax.set_visible(False)
                continue
            mesh = ax.contourf(lons, lats, np.clip(members[idx], vmin, vmax), levels=levels, cmap="viridis")
            add_bbox(ax, case.event_bbox)
            ax.set_xlim(bbox[0] - 2.0, bbox[1] + 2.0)
            ax.set_ylim(bbox[2] - 1.5, bbox[3] + 1.5)
            ax.set_title(f"member {idx + 1}", fontsize=8)
            ax.tick_params(labelsize=7)
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.72, pad=0.04)
        cbar.set_label(units)
        fig.suptitle(f"{case.label}: first {n_panel} ECMWF members", fontsize=12, fontweight="bold")
        out = output_dir / f"{case.key}_ecmwf_first_members.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        output_paths.append(out)

    return output_paths


def write_processed_netcdf(
    case: EventCase,
    fields: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    units: str,
    selected_rows: pd.DataFrame,
    source_path: Path,
    output_dir: Path,
) -> Path:
    ds = xr.Dataset(
        {
            "member_weekly_mean": (("member", "lat", "lon"), fields["member_weekly_mean"]),
            "ensemble_mean": (("lat", "lon"), fields["ensemble_mean"]),
            "ensemble_std": (("lat", "lon"), fields["ensemble_std"]),
            "ensemble_q05": (("lat", "lon"), fields["ensemble_q05"]),
            "ensemble_q50": (("lat", "lon"), fields["ensemble_q50"]),
            "ensemble_q95": (("lat", "lon"), fields["ensemble_q95"]),
        },
        coords={
            "member": np.arange(fields["member_weekly_mean"].shape[0], dtype=int),
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "case": case.key,
            "label": case.label,
            "source_grib": str(source_path),
            "target_start": case.target_start,
            "target_end": case.target_end,
            "units": units,
            "selected_samples_json": selected_rows.to_json(orient="records"),
        },
    )
    for var in ds.data_vars:
        ds[var].attrs["units"] = units
    out = output_dir / f"{case.key}_ecmwf_week4_processed.nc"
    ds.to_netcdf(out)
    return out


def summarize_case(
    case: EventCase,
    grib_path: Path,
    output_root: Path,
    precip_mode: str,
    cfgrib_indexpath: str,
    grib_reader: str,
    dpi: int,
    max_members_panel: int,
) -> bool:
    case_dir = output_root / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []

    log(report_lines, "=" * 88)
    log(report_lines, f"Case: {case.label} ({case.key})")
    log(report_lines, f"GRIB: {grib_path}")
    log(report_lines, f"Target week: {case.target_start} through {case.target_end}")
    if not grib_path.exists():
        log(report_lines, f"ERROR: file does not exist: {grib_path}")
        (case_dir / "diagnostic_report.txt").write_text("\n".join(report_lines) + "\n")
        return False

    try:
        datasets = open_grib_datasets(grib_path, cfgrib_indexpath, reader=grib_reader)
    except Exception as exc:
        log(report_lines, "ERROR: could not read GRIB file.")
        for line in str(exc).splitlines():
            log(report_lines, f"  {line}" if line else "")
        report_path = case_dir / "diagnostic_report.txt"
        report_path.write_text("\n".join(report_lines) + "\n")
        print(f"Report written: {report_path}")
        return False

    reader_used = datasets[0].attrs.get("reader", "cfgrib") if datasets else grib_reader
    log(report_lines, f"GRIB groups opened: {len(datasets)} (reader={reader_used})")
    for group_idx, ds in enumerate(datasets):
        for row in describe_dataset(ds, group_idx):
            log(report_lines, row)

    group_idx, var_name, da, candidates = choose_variable(datasets, case)
    log(report_lines)
    log(report_lines, "Variable ranking:")
    for cand_group, cand_var, score in candidates[:12]:
        log(report_lines, f"  score={score:3d} group={cand_group} var={cand_var}")
    log(report_lines, f"Selected variable: group={group_idx}, var={var_name}")
    log(report_lines, f"Selected attrs: {json.dumps({k: str(v) for k, v in da.attrs.items() if k.startswith('GRIB_') or k in {'units', 'long_name', 'standard_name'}}, indent=2)}")

    values, meta = build_member_sample_array(da)
    values, meta = sort_samples(values, meta)
    selected = target_mask(meta, case)
    selected_rows = selected_sample_table(meta, selected)
    selected_csv = case_dir / f"{case.key}_selected_samples.csv"
    selected_rows.to_csv(selected_csv, index=False)

    log(report_lines)
    log(report_lines, f"Canonical array shape: member={values.shape[0]}, sample={values.shape[1]}, lat={values.shape[2]}, lon={values.shape[3]}")
    log(report_lines, f"Sample dims stacked: {meta['sample_dims']}")
    log(report_lines, f"Selected target-week samples: {int(selected.sum())}; CSV: {selected_csv}")
    if selected_rows.empty:
        log(report_lines, "ERROR: No GRIB valid dates fell inside the target week.")
        report_path = case_dir / "diagnostic_report.txt"
        report_path.write_text("\n".join(report_lines) + "\n")
        print(f"Report written: {report_path}")
        return False
    with pd.option_context("display.max_rows", 50, "display.width", 160):
        sample_text = selected_rows.to_string(index=False)
    log(report_lines, "Selected samples:")
    for line in sample_text.splitlines():
        log(report_lines, f"  {line}")

    attrs = dict(da.attrs)
    target_days = (pd.Timestamp(case.target_end) - pd.Timestamp(case.target_start)).days + 1
    if case.variable_kind == "t2m":
        member_week, units, conversion_note = convert_temperature(values, selected, attrs)
    else:
        member_week, units, conversion_note = convert_precipitation(
            values, selected, attrs, meta, precip_mode, target_days=target_days
        )
    log(report_lines)
    log(report_lines, f"Conversion: {conversion_note}")

    fields_raw = {
        "member_weekly_mean": member_week,
        "ensemble_mean": np.nanmean(member_week, axis=0),
        "ensemble_std": np.nanstd(member_week, axis=0),
        "ensemble_q05": np.nanpercentile(member_week, 5, axis=0),
        "ensemble_q50": np.nanpercentile(member_week, 50, axis=0),
        "ensemble_q95": np.nanpercentile(member_week, 95, axis=0),
    }
    fields, lats, lons = wrap_and_sort_grid(fields_raw, meta["lats"], meta["lons"])

    member_region = member_region_values(fields["member_weekly_mean"], lats, lons, case.event_bbox)
    ens_mean_region = weighted_region_mean(fields["ensemble_mean"], lats, lons, case.event_bbox)
    ens_q95_region = weighted_region_mean(fields["ensemble_q95"], lats, lons, case.event_bbox)
    ens_spread_region = weighted_region_mean(fields["ensemble_std"], lats, lons, case.event_bbox)
    log(report_lines)
    log(report_lines, "Event-box summary:")
    log(report_lines, f"  members: {fields['member_weekly_mean'].shape[0]}")
    log(report_lines, f"  ensemble mean area mean: {ens_mean_region:.4g} {units}")
    log(report_lines, f"  ensemble q95 area mean : {ens_q95_region:.4g} {units}")
    log(report_lines, f"  ensemble spread area mean: {ens_spread_region:.4g} {units}")
    if np.isfinite(member_region).any():
        log(report_lines, f"  member regional p05/p50/p95: {np.nanpercentile(member_region, [5, 50, 95])} {units}")

    nc_path = write_processed_netcdf(case, fields, lats, lons, units, selected_rows, grib_path, case_dir)
    plot_paths = plot_summary(case, fields, lats, lons, member_region, units, case_dir, dpi, max_members_panel)
    log(report_lines)
    log(report_lines, f"Processed NetCDF: {nc_path}")
    for path in plot_paths:
        log(report_lines, f"Plot: {path}")

    report_path = case_dir / "diagnostic_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Report written: {report_path}")

    for ds in datasets:
        ds.close()
    return True


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    requested = list(CASES) if args.case == "all" else [args.case]
    grib_paths = {
        "uk_heat": Path(args.uk_grib),
        "california_pr": Path(args.california_grib),
    }

    print(
        textwrap.dedent(
            f"""
            ECMWF GRIB diagnostics
              output_dir      : {output_root}
              case(s)         : {', '.join(requested)}
              grib reader     : {args.grib_reader}
              cfgrib indexpath: {args.cfgrib_indexpath!r}
            """
        ).strip()
    )
    failures: list[str] = []
    for key in requested:
        ok = summarize_case(
            CASES[key],
            grib_paths[key],
            output_root,
            precip_mode=args.precip_mode,
            cfgrib_indexpath=args.cfgrib_indexpath,
            grib_reader=args.grib_reader,
            dpi=args.dpi,
            max_members_panel=args.max_members_panel,
        )
        if not ok:
            failures.append(key)
    if failures:
        raise SystemExit(f"Failed case(s): {', '.join(failures)}. See diagnostic_report.txt in {output_root}.")


if __name__ == "__main__":
    main()
