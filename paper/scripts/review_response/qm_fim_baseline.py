#!/usr/bin/env python3
"""Leakage-free empirical quantile-mapping baseline for FIMr1p1/GEOS.

The default protocol is fixed for the manuscript:

* fit:        1999--2019 paired FIMr1p1 and observations;
* validation: 2020, diagnostic only (the mapping is not refit);
* evaluation: 2021--2023, untouched until the final evaluation.

Quantile mapping is fit independently by variable, lead week, verifying
calendar month, and grid point. FIMr1p1 ensemble members are pooled while
estimating the forecast CDF and are mapped member by member at application
time. Precipitation uses a square-root-space mixed distribution with an
observed-frequency dry threshold; T2M is mapped in physical temperature units.

The script never modifies the frozen flow-forecast archive. It writes:

  <out_dir>/qm_parameters/             fitted CDF pairs
  <out_dir>/corrected/<year>.zarr      qm_pr and qm_t2m members
  <out_dir>/qm*_per_init_metrics.csv   raw-vs-QM diagnostic scores
  <out_dir>/qm*_aggregate_metrics.csv  validation/evaluation summaries

Example:

  python paper/scripts/review_response/qm_fim_baseline.py \
      --data-root /scratch/11353/afahad/geossub/dataprocess \
      --forecast-dir dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50 \
      --out-dir ml_output_flow_finalv1_global_noisectx_t2mres/qm_fim_1999_2019
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr


VARIABLES = {
    "pr": {
        "forecast_aliases": ("pr", "precip", "PRECTOT", "flux_precip"),
        "observation_aliases": ("precip", "pr", "target", "total_precipitation"),
        "observation_paths": (
            "gpcp_weekly_{year}.zarr",
            "gpcp/{year}.zarr",
        ),
        "units": "mm/day",
    },
    "t2m": {
        "forecast_aliases": ("tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"),
        "observation_aliases": ("t2m", "tas", "T2M"),
        "observation_paths": ("t2m_weekly_{year}.zarr",),
        "units": "K",
    },
}

DIM_ALIASES = {
    "init": ("init", "S", "start", "initialization", "time"),
    "member": ("member", "M", "ensemble", "geos_member", "number"),
    "lead": ("lead", "L", "lead_time", "step"),
    "lat": ("lat", "latitude", "Y", "y"),
    "lon": ("lon", "longitude", "X", "x"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/scratch/11353/afahad/geossub/dataprocess")
    parser.add_argument(
        "--forecast-dir",
        default="dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50",
        help="Frozen generated-forecast archive containing evaluation raw FIM and observations.",
    )
    parser.add_argument(
        "--out-dir",
        default="ml_output_flow_finalv1_global_noisectx_t2mres/qm_fim_1999_2019",
    )
    parser.add_argument("--train-years", default="1999-2019")
    parser.add_argument("--validation-years", default="2020")
    parser.add_argument("--evaluation-years", default="2021-2023")
    parser.add_argument("--variables", default="pr,t2m")
    parser.add_argument(
        "--stage",
        choices=("all", "fit", "validate", "evaluate", "apply", "score"),
        default="fit",
        help=(
            "validate applies/scores 2020 only; evaluate applies/scores 2021-2023 "
            "only; apply/score operate on both splits."
        ),
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=51,
        help="Number of equally spaced empirical CDF knots, including 0 and 1.",
    )
    parser.add_argument(
        "--wet-threshold",
        type=float,
        default=0.1,
        help="Weekly-mean precipitation threshold in mm/day used for the dry mass.",
    )
    parser.add_argument(
        "--min-wet-samples",
        type=int,
        default=10,
        help="Minimum positive samples per grid point before positive-only PR CDFs are used.",
    )
    parser.add_argument(
        "--spatial-block-size",
        type=int,
        default=8192,
        help="Number of flattened grid points mapped at once.",
    )
    parser.add_argument(
        "--fit-lat-tile",
        type=int,
        default=30,
        help="Latitude rows loaded at once while fitting (memory-safety control).",
    )
    parser.add_argument(
        "--fit-lon-tile",
        type=int,
        default=60,
        help="Longitude columns loaded at once while fitting (memory-safety control).",
    )
    parser.add_argument(
        "--max-inits",
        type=int,
        default=None,
        help="Smoke-test limit per verifying month/year in fit, or per year otherwise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing parameter/corrected stores for the requested stage.",
    )
    return parser.parse_args()


def parse_years(value: str | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(value, str):
        return tuple(int(year) for year in value)
    years: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending year range: {token}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    if not years:
        raise ValueError("At least one year is required.")
    if len(set(years)) != len(years):
        raise ValueError(f"Duplicate years are not allowed: {value}")
    return tuple(years)


def validate_year_splits(
    train_years: Iterable[int],
    validation_years: Iterable[int],
    evaluation_years: Iterable[int],
) -> None:
    train = tuple(int(y) for y in train_years)
    validation = tuple(int(y) for y in validation_years)
    evaluation = tuple(int(y) for y in evaluation_years)
    if not train or not validation or not evaluation:
        raise ValueError("Training, validation, and evaluation splits must all be nonempty.")
    overlap = (
        (set(train) & set(validation))
        | (set(train) & set(evaluation))
        | (set(validation) & set(evaluation))
    )
    if overlap:
        raise ValueError(f"Year splits overlap: {sorted(overlap)}")
    if max(train) >= min(validation) or max(validation) >= min(evaluation):
        raise ValueError(
            "Expected chronological train < validation < evaluation splits, got "
            f"{min(train)}-{max(train)}, {min(validation)}-{max(validation)}, "
            f"{min(evaluation)}-{max(evaluation)}."
        )


def _find_name(names: Iterable[str], aliases: Iterable[str]) -> str | None:
    names = tuple(str(name) for name in names)
    lower_to_name = {name.lower(): name for name in names}
    for alias in aliases:
        if alias in names:
            return alias
        if alias.lower() in lower_to_name:
            return lower_to_name[alias.lower()]
    return None


def _select_variable(ds: xr.Dataset, aliases: Iterable[str], label: str) -> xr.DataArray:
    name = _find_name(ds.data_vars, aliases)
    if name is None:
        raise KeyError(
            f"Could not find {label}; tried {tuple(aliases)}, "
            f"available variables are {list(ds.data_vars)}."
        )
    return ds[name]


def _dimension_name(
    da: xr.DataArray,
    canonical: str,
    *,
    excluded: set[str] | None = None,
    required: bool = True,
) -> str | None:
    excluded = excluded or set()
    candidates = [dim for dim in da.dims if dim not in excluded]
    found = _find_name(candidates, DIM_ALIASES[canonical])
    if found is not None:
        return found
    if canonical == "lat":
        sized = [dim for dim in candidates if da.sizes[dim] in (180, 181)]
        if len(sized) == 1:
            return sized[0]
    if canonical == "lon":
        sized = [dim for dim in candidates if da.sizes[dim] in (360, 361)]
        if len(sized) == 1:
            return sized[0]
    if required:
        raise ValueError(f"Could not identify {canonical} dimension in {da.dims}.")
    return None


def standardize_forecast(da: xr.DataArray) -> xr.DataArray:
    init = _dimension_name(da, "init")
    lead = _dimension_name(da, "lead", excluded={init})
    member = _dimension_name(da, "member", excluded={init, lead}, required=False)
    excluded = {init, lead}
    if member is not None:
        excluded.add(member)
    lat = _dimension_name(da, "lat", excluded=excluded)
    lon = _dimension_name(da, "lon", excluded=excluded | {lat})
    rename = {init: "init", lead: "lead", lat: "lat", lon: "lon"}
    if member is not None:
        rename[member] = "member"
    da = da.rename(rename)
    da = da.isel(lead=slice(0, 4))
    if da.sizes["lead"] != 4:
        raise ValueError(
            f"Expected at least four forecast leads, found {da.sizes['lead']}."
        )
    if member is None:
        da = da.expand_dims(member=[1], axis=1)
    da = da.assign_coords(lead=np.arange(1, 5, dtype=np.int32))
    da = da.assign_coords(init=pd.to_datetime(da["init"].values).values)
    # Keep the archive's native dimension order while it is lazy. Transposing a
    # noncanonical Zarr array with chunks=None creates a vectorized xarray
    # indexer that can allocate a full-grid int64 index array on the next isel.
    # RawYear transposes only after selecting a small spatial tile.
    return da


def standardize_observation(da: xr.DataArray) -> xr.DataArray:
    init = _dimension_name(da, "init")
    lead = _dimension_name(da, "lead", excluded={init})
    lat = _dimension_name(da, "lat", excluded={init, lead})
    lon = _dimension_name(da, "lon", excluded={init, lead, lat})
    da = da.rename({init: "init", lead: "lead", lat: "lat", lon: "lon"})
    da = da.isel(lead=slice(0, 4))
    if da.sizes["lead"] != 4:
        raise ValueError(
            f"Expected at least four observation leads, found {da.sizes['lead']}."
        )
    da = da.assign_coords(lead=np.arange(1, 5, dtype=np.int32))
    da = da.assign_coords(init=pd.to_datetime(da["init"].values).values)
    # Preserve native lazy storage order; see standardize_forecast.
    return da


def geos_path(data_root: Path, year: int) -> Path:
    candidates = (
        data_root / f"geos_subc_{year}.zarr",
        data_root / "geos_s2s" / f"{year}.zarr",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"FIMr1p1/GEOS archive not found for {year}: {candidates}")


def observation_path(data_root: Path, year: int, variable: str) -> Path:
    candidates = tuple(
        data_root / pattern.format(year=year)
        for pattern in VARIABLES[variable]["observation_paths"]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{variable} observation archive not found for {year}: {candidates}"
    )


def canonical_grid(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    height, width = da.sizes["lat"], da.sizes["lon"]
    if (height, width) == (181, 360):
        return (
            np.linspace(-90.0, 90.0, height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
        )
    lat = np.asarray(da["lat"].values)
    lon = np.asarray(da["lon"].values)
    if lat.ndim != 1 or len(lat) != height:
        lat = np.arange(height, dtype=np.float32)
    if lon.ndim != 1 or len(lon) != width:
        lon = np.arange(width, dtype=np.float32)
    return lat.astype(np.float32), lon.astype(np.float32)


class RawYear:
    """Lazy handles for one year of paired raw FIMr1p1 and observations."""

    def __init__(self, data_root: Path, year: int, variables: Iterable[str]):
        self.year = int(year)
        self._datasets: list[xr.Dataset] = []
        geos_ds = xr.open_zarr(geos_path(data_root, year), consolidated=False, chunks=None)
        self._datasets.append(geos_ds)
        self.forecast: dict[str, xr.DataArray] = {}
        self.observation: dict[str, xr.DataArray] = {}
        self.obs_index: dict[str, np.ndarray] = {}
        for variable in variables:
            spec = VARIABLES[variable]
            self.forecast[variable] = standardize_forecast(
                _select_variable(
                    geos_ds,
                    spec["forecast_aliases"],
                    f"{variable} FIMr1p1 field",
                )
            )
            obs_ds = xr.open_zarr(
                observation_path(data_root, year, variable),
                consolidated=False,
                chunks=None,
            )
            self._datasets.append(obs_ds)
            self.observation[variable] = standardize_observation(
                _select_variable(
                    obs_ds,
                    spec["observation_aliases"],
                    f"{variable} observation",
                )
            )
            forecast_dates = pd.to_datetime(self.forecast[variable]["init"].values).normalize()
            obs_dates = pd.to_datetime(self.observation[variable]["init"].values).normalize()
            lookup = {date: index for index, date in enumerate(obs_dates)}
            missing = [date for date in forecast_dates if date not in lookup]
            if missing:
                raise ValueError(
                    f"{variable} observations for {year} are missing "
                    f"{len(missing)} FIM initialization dates; first={missing[0]}."
                )
            self.obs_index[variable] = np.asarray(
                [lookup[date] for date in forecast_dates], dtype=np.int64
            )
        first = self.forecast[next(iter(self.forecast))]
        self.init_dates = pd.to_datetime(first["init"].values).normalize()
        self.lats, self.lons = canonical_grid(first)

    def close(self) -> None:
        for ds in self._datasets:
            ds.close()

    def training_samples(
        self,
        variable: str,
        lead_index: int,
        verifying_month: int,
        max_inits: int | None = None,
        lat_slice: slice | None = None,
        lon_slice: slice | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(self.init_dates), dtype=np.int64)
        valid_dates = self.init_dates + pd.to_timedelta(7 * (lead_index + 1), unit="D")
        indices = indices[np.asarray(valid_dates.month == verifying_month)]
        if max_inits is not None:
            indices = indices[:max_inits]
        if indices.size == 0:
            shape = (0, len(self.lats), len(self.lons))
            return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
        spatial_indexers = {
            "lat": lat_slice or slice(None),
            "lon": lon_slice or slice(None),
        }
        forecast = self.forecast[variable].isel(
            init=indices,
            lead=lead_index,
            **spatial_indexers,
        ).transpose("init", "member", "lat", "lon").values
        obs_indices = self.obs_index[variable][indices]
        observation = self.observation[variable].isel(
            init=obs_indices,
            lead=lead_index,
            **spatial_indexers,
        ).transpose("init", "lat", "lon").values
        forecast = np.asarray(forecast, dtype=np.float32)
        observation = np.asarray(observation, dtype=np.float32)
        forecast = forecast.reshape(
            forecast.shape[0] * forecast.shape[1],
            forecast.shape[-2],
            forecast.shape[-1],
        )
        return forecast, observation

    def fields(
        self,
        variable: str,
        init_index: int,
        lead_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        forecast = (
            self.forecast[variable]
            .isel(init=init_index, lead=lead_index)
            .transpose("member", "lat", "lon")
            .values
        )
        obs_index = int(self.obs_index[variable][init_index])
        observation = (
            self.observation[variable]
            .isel(init=obs_index, lead=lead_index)
            .transpose("lat", "lon")
            .values
        )
        return (
            np.asarray(forecast, dtype=np.float32),
            np.asarray(observation, dtype=np.float32),
        )


def _nanquantile(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanquantile(values, quantiles, axis=0)
    return np.asarray(result, dtype=np.float32)


def column_quantile(
    values: np.ndarray,
    probabilities: np.ndarray,
    block_size: int = 8192,
) -> np.ndarray:
    """Column-wise quantiles with one probability per spatial column."""
    values = np.asarray(values, dtype=np.float32)
    shape = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size != flat.shape[1]:
        raise ValueError("Probability field does not match the values grid.")
    output = np.full(flat.shape[1], np.nan, dtype=np.float32)
    for start in range(0, flat.shape[1], block_size):
        stop = min(start + block_size, flat.shape[1])
        block = flat[:, start:stop]
        sorted_block = np.sort(block, axis=0)
        finite_count = np.sum(np.isfinite(sorted_block), axis=0)
        p = np.clip(probs[start:stop], 0.0, 1.0)
        position = p * np.maximum(finite_count - 1, 0)
        lower = np.floor(position).astype(np.int64)
        upper = np.ceil(position).astype(np.int64)
        columns = np.arange(stop - start)
        safe_lower = np.minimum(lower, np.maximum(finite_count - 1, 0))
        safe_upper = np.minimum(upper, np.maximum(finite_count - 1, 0))
        lo_value = sorted_block[safe_lower, columns]
        hi_value = sorted_block[safe_upper, columns]
        fraction = position - lower
        mapped = lo_value + fraction * (hi_value - lo_value)
        mapped[finite_count == 0] = np.nan
        output[start:stop] = mapped.astype(np.float32)
    return output.reshape(shape)


def fit_quantile_parameters(
    forecast: np.ndarray,
    observation: np.ndarray,
    variable: str,
    quantiles: np.ndarray,
    *,
    wet_threshold: float = 0.1,
    min_wet_samples: int = 10,
    block_size: int = 8192,
) -> dict[str, np.ndarray]:
    forecast = np.asarray(forecast, dtype=np.float32)
    observation = np.asarray(observation, dtype=np.float32)
    if forecast.ndim != 3 or observation.ndim != 3:
        raise ValueError("Expected forecast/observation samples with shape (sample, lat, lon).")
    if forecast.shape[1:] != observation.shape[1:]:
        raise ValueError(
            f"Forecast and observation grids differ: {forecast.shape} vs {observation.shape}."
        )
    if forecast.shape[0] < 2 or observation.shape[0] < 2:
        raise ValueError("At least two forecast and observation samples are required.")

    if variable == "t2m":
        return {
            "forecast_quantile": _nanquantile(forecast, quantiles),
            "observation_quantile": _nanquantile(observation, quantiles),
        }
    if variable != "pr":
        raise ValueError(f"Unsupported variable: {variable}")

    forecast = np.where(forecast >= 0.0, forecast, np.nan)
    observation = np.where(observation >= 0.0, observation, np.nan)
    obs_finite = np.sum(np.isfinite(observation), axis=0)
    obs_dry = np.sum(
        np.isfinite(observation) & (observation <= wet_threshold), axis=0
    )
    dry_probability = np.divide(
        obs_dry,
        obs_finite,
        out=np.zeros_like(obs_dry, dtype=np.float32),
        where=obs_finite > 0,
    )
    forecast_dry_threshold = column_quantile(
        forecast, dry_probability, block_size=block_size
    )

    forecast_sqrt = np.sqrt(forecast)
    observation_sqrt = np.sqrt(observation)
    forecast_positive = np.where(
        forecast > forecast_dry_threshold[None], forecast_sqrt, np.nan
    )
    observation_positive = np.where(
        observation > wet_threshold, observation_sqrt, np.nan
    )
    forecast_quantile = _nanquantile(forecast_positive, quantiles)
    observation_quantile = _nanquantile(observation_positive, quantiles)

    forecast_wet_count = np.sum(np.isfinite(forecast_positive), axis=0)
    observation_wet_count = np.sum(np.isfinite(observation_positive), axis=0)
    low_sample = (
        (forecast_wet_count < min_wet_samples)
        | (observation_wet_count < min_wet_samples)
    )
    if np.any(low_sample):
        fallback_forecast = _nanquantile(forecast_sqrt, quantiles)
        fallback_observation = _nanquantile(observation_sqrt, quantiles)
        forecast_quantile[:, low_sample] = fallback_forecast[:, low_sample]
        observation_quantile[:, low_sample] = fallback_observation[:, low_sample]

    always_dry = observation_wet_count == 0
    if np.any(always_dry):
        observation_quantile[:, always_dry] = 0.0

    return {
        "forecast_quantile": forecast_quantile,
        "observation_quantile": observation_quantile,
        "forecast_dry_threshold": forecast_dry_threshold.astype(np.float32),
        "observed_dry_probability": dry_probability.astype(np.float32),
        "forecast_wet_count": forecast_wet_count.astype(np.int32),
        "observation_wet_count": observation_wet_count.astype(np.int32),
    }


def _piecewise_map(
    values: np.ndarray,
    forecast_quantile: np.ndarray,
    observation_quantile: np.ndarray,
    *,
    block_size: int = 8192,
) -> np.ndarray:
    """Map values through gridpoint CDF knots with additive tail corrections."""
    values = np.asarray(values, dtype=np.float32)
    original_shape = values.shape
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError("Expected values with shape (member, lat, lon) or (lat, lon).")
    qf = np.asarray(forecast_quantile, dtype=np.float32)
    qo = np.asarray(observation_quantile, dtype=np.float32)
    if qf.shape != qo.shape or qf.shape[1:] != values.shape[1:]:
        raise ValueError(
            f"Quantile/value shapes are incompatible: {qf.shape}, {qo.shape}, {values.shape}."
        )

    member_count = values.shape[0]
    flat_values = values.reshape(member_count, -1)
    flat_qf = qf.reshape(qf.shape[0], -1)
    flat_qo = qo.reshape(qo.shape[0], -1)
    output = np.full_like(flat_values, np.nan, dtype=np.float32)

    for start in range(0, flat_values.shape[1], block_size):
        stop = min(start + block_size, flat_values.shape[1])
        x = flat_values[:, start:stop]
        fq = flat_qf[:, start:stop]
        oq = flat_qo[:, start:stop]
        finite_knots = np.all(np.isfinite(fq) & np.isfinite(oq), axis=0)

        # Number of internal knots not exceeding x gives the enclosing segment.
        segment = np.sum(x[:, None, :] >= fq[None, 1:, :], axis=1)
        segment = np.clip(segment, 0, fq.shape[0] - 2)
        columns = np.arange(stop - start)[None, :]
        f_lo = fq[segment, columns]
        f_hi = fq[segment + 1, columns]
        o_lo = oq[segment, columns]
        o_hi = oq[segment + 1, columns]
        width = f_hi - f_lo
        fraction = np.divide(
            x - f_lo,
            width,
            out=np.full_like(x, 0.5, dtype=np.float32),
            where=np.abs(width) > 1e-7,
        )
        mapped = o_lo + fraction * (o_hi - o_lo)

        # Outside the fitted range, retain the raw departure from the endpoint
        # rather than capping validation/evaluation extremes at the training max.
        below = x < fq[0][None]
        above = x > fq[-1][None]
        mapped = np.where(below, x + (oq[0] - fq[0])[None], mapped)
        mapped = np.where(above, x + (oq[-1] - fq[-1])[None], mapped)
        valid = np.isfinite(x) & finite_knots[None]
        output[:, start:stop] = np.where(valid, mapped, np.nan)

    mapped = output.reshape(values.shape)
    if len(original_shape) == 2:
        return mapped[0]
    return mapped


def apply_quantile_map(
    values: np.ndarray,
    parameters: dict[str, np.ndarray],
    variable: str,
    *,
    wet_threshold: float = 0.1,
    block_size: int = 8192,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if variable == "t2m":
        return _piecewise_map(
            values,
            parameters["forecast_quantile"],
            parameters["observation_quantile"],
            block_size=block_size,
        ).astype(np.float32)
    if variable != "pr":
        raise ValueError(f"Unsupported variable: {variable}")

    nonnegative = np.maximum(values, 0.0)
    mapped_sqrt = _piecewise_map(
        np.sqrt(nonnegative),
        parameters["forecast_quantile"],
        parameters["observation_quantile"],
        block_size=block_size,
    )
    mapped = np.square(np.maximum(mapped_sqrt, 0.0))
    dry_probability = parameters["observed_dry_probability"]
    dry_threshold = parameters["forecast_dry_threshold"]
    dry = (
        (dry_probability[None] > 0.0)
        & (nonnegative <= dry_threshold[None])
    )
    always_dry = dry_probability[None] >= (1.0 - 1e-7)
    mapped = np.where(dry | always_dry, 0.0, mapped)
    mapped = np.where(mapped < wet_threshold, 0.0, mapped)
    return mapped.astype(np.float32)


def parameter_store(parameter_root: Path, variable: str, lead: int, month: int) -> Path:
    return parameter_root / variable / f"lead{lead}_month{month:02d}.zarr"


def parameter_progress_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}.progress")


def parameter_complete_marker(path: Path) -> Path:
    return path.with_name(f"{path.name}.complete")


def tile_progress_marker(
    path: Path,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
) -> Path:
    return parameter_progress_dir(path) / (
        f"tile_lat{lat_start:04d}-{lat_stop:04d}_"
        f"lon{lon_start:04d}-{lon_stop:04d}.done"
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a small restart marker atomically on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)


def _parameter_names(variable: str) -> tuple[str, ...]:
    names = ("forecast_quantile", "observation_quantile")
    if variable == "pr":
        names += (
            "forecast_dry_threshold",
            "observed_dry_probability",
            "forecast_wet_count",
            "observation_wet_count",
        )
    return names


def _clear_parameter_state(path: Path) -> None:
    """Remove only the generated state for one variable/lead/month."""
    if path.exists():
        shutil.rmtree(path)
    progress = parameter_progress_dir(path)
    if progress.exists():
        shutil.rmtree(progress)
    complete = parameter_complete_marker(path)
    if complete.exists():
        complete.unlink()


def _store_layout_matches(
    path: Path,
    *,
    variable: str,
    quantile_count: int,
    height: int,
    width: int,
) -> bool:
    """Check Zarr metadata without loading global parameter arrays."""
    try:
        group = zarr.open_group(str(path), mode="r")
        expected_shapes = {
            "forecast_quantile": (quantile_count, height, width),
            "observation_quantile": (quantile_count, height, width),
        }
        if variable == "pr":
            expected_shapes.update(
                {
                    "forecast_dry_threshold": (height, width),
                    "observed_dry_probability": (height, width),
                    "forecast_wet_count": (height, width),
                    "observation_wet_count": (height, width),
                }
            )
        return all(
            name in group and tuple(group[name].shape) == shape
            for name, shape in expected_shapes.items()
        )
    except (KeyError, OSError, ValueError, zarr.errors.MetadataError):
        return False


def _legacy_store_is_complete(
    path: Path,
    *,
    variable: str,
    quantile_count: int,
    height: int,
    width: int,
) -> bool:
    """Recognize complete stores produced before tile checkpoints existed."""
    if not _store_layout_matches(
        path,
        variable=variable,
        quantile_count=quantile_count,
        height=height,
        width=width,
    ):
        return False
    try:
        group = zarr.open_group(str(path), mode="r")
        return all(
            group[name].nchunks_initialized == group[name].nchunks
            for name in _parameter_names(variable)
        )
    except (KeyError, OSError, ValueError, zarr.errors.MetadataError):
        return False


def initialize_parameter_store(
    path: Path,
    quantiles: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    variable: str,
    lead: int,
    month: int,
    train_years: tuple[int, ...],
    wet_threshold: float,
    lat_tile: int,
    lon_tile: int,
) -> None:
    """Create an empty, chunk-aligned parameter store without global arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(
        {
            "method": "empirical_quantile_mapping",
            "variable": variable,
            "lead_week": int(lead),
            "verifying_month": int(month),
            "training_years": f"{min(train_years)}-{max(train_years)}",
            "precipitation_transform": "sqrt_mixed_distribution"
            if variable == "pr"
            else "none",
            "wet_threshold_mm_day": float(wet_threshold),
            "tail_extrapolation": "additive_correction_in_mapping_space",
            "checkpoint_format_version": 1,
        }
    )

    coordinate_values = {
        "quantile": np.asarray(quantiles, dtype=np.float32),
        "lat": np.asarray(lats, dtype=np.float32),
        "lon": np.asarray(lons, dtype=np.float32),
    }
    for name, values in coordinate_values.items():
        array = group.create_dataset(
            name,
            data=values,
            chunks=(len(values),),
            overwrite=True,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = [name]

    quantile_shape = (len(quantiles), len(lats), len(lons))
    quantile_chunks = (
        len(quantiles),
        min(lat_tile, len(lats)),
        min(lon_tile, len(lons)),
    )
    for name in ("forecast_quantile", "observation_quantile"):
        array = group.create_dataset(
            name,
            shape=quantile_shape,
            chunks=quantile_chunks,
            dtype=np.float32,
            fill_value=np.nan,
            overwrite=True,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["quantile", "lat", "lon"]

    if variable == "pr":
        spatial_shape = (len(lats), len(lons))
        spatial_chunks = (
            min(lat_tile, len(lats)),
            min(lon_tile, len(lons)),
        )
        for name in ("forecast_dry_threshold", "observed_dry_probability"):
            array = group.create_dataset(
                name,
                shape=spatial_shape,
                chunks=spatial_chunks,
                dtype=np.float32,
                fill_value=np.nan,
                overwrite=True,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["lat", "lon"]
        for name in ("forecast_wet_count", "observation_wet_count"):
            array = group.create_dataset(
                name,
                shape=spatial_shape,
                chunks=spatial_chunks,
                dtype=np.int32,
                fill_value=0,
                overwrite=True,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["lat", "lon"]

    _atomic_write_json(
        parameter_progress_dir(path) / "INITIALIZED",
        {
            "variable": variable,
            "lead": int(lead),
            "month": int(month),
            "lat_tile": int(lat_tile),
            "lon_tile": int(lon_tile),
        },
    )


def write_parameter_tile(
    group: zarr.hierarchy.Group,
    parameters: dict[str, np.ndarray],
    *,
    lat_slice: slice,
    lon_slice: slice,
) -> None:
    """Persist all arrays for one spatial tile before its marker is created."""
    for name, values in parameters.items():
        if values.ndim == 3:
            group[name][:, lat_slice, lon_slice] = values
        elif values.ndim == 2:
            group[name][lat_slice, lon_slice] = values
        else:
            raise ValueError(f"Unexpected parameter rank for {name}: {values.ndim}")


def fit_all(
    data_root: Path,
    parameter_root: Path,
    train_years: tuple[int, ...],
    variables: tuple[str, ...],
    quantiles: np.ndarray,
    args: argparse.Namespace,
) -> None:
    parameter_root.mkdir(parents=True, exist_ok=True)
    manifest_path = parameter_root / "manifest.json"
    requested_manifest = {
        "method": "empirical_quantile_mapping",
        "train_years": list(train_years),
        "validation_years": list(parse_years(args.validation_years)),
        "evaluation_years": list(parse_years(args.evaluation_years)),
        "variables": list(variables),
        "n_quantiles": int(len(quantiles)),
        "quantiles": [float(q) for q in quantiles],
        "wet_threshold_mm_day": float(args.wet_threshold),
        "min_wet_samples": int(args.min_wet_samples),
        "fit_lat_tile": int(args.fit_lat_tile),
        "fit_lon_tile": int(args.fit_lon_tile),
        "max_inits_per_month": args.max_inits,
        "complete": False,
    }
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text())
        compare_keys = (
            "train_years",
            "variables",
            "n_quantiles",
            "wet_threshold_mm_day",
            "min_wet_samples",
            "fit_lat_tile",
            "fit_lon_tile",
            "max_inits_per_month",
        )
        mismatch = [key for key in compare_keys if existing.get(key) != requested_manifest[key]]
        if mismatch:
            raise ValueError(
                f"Existing QM manifest differs in {mismatch}; use a new --out-dir "
                "or pass --overwrite."
            )
    manifest_path.write_text(json.dumps(requested_manifest, indent=2) + "\n")

    for variable in variables:
        print(f"Fitting {variable} QM parameters from {train_years[0]}-{train_years[-1]} ...")
        archives = [RawYear(data_root, year, (variable,)) for year in train_years]
        try:
            lats, lons = archives[0].lats, archives[0].lons
            height, width = len(lats), len(lons)
            for lead_index in range(4):
                for month in range(1, 13):
                    lead = lead_index + 1
                    path = parameter_store(parameter_root, variable, lead, month)
                    progress_dir = parameter_progress_dir(path)
                    initialized_marker = progress_dir / "INITIALIZED"
                    complete_marker = parameter_complete_marker(path)

                    if args.overwrite:
                        _clear_parameter_state(path)

                    if complete_marker.exists() and not path.exists():
                        complete_marker.unlink()

                    if complete_marker.exists() and path.exists():
                        if _store_layout_matches(
                            path,
                            variable=variable,
                            quantile_count=len(quantiles),
                            height=height,
                            width=width,
                        ):
                            print(f"  Reusing completed {path}")
                            continue
                        print(f"  Rebuilding invalid completed store {path}")
                        _clear_parameter_state(path)

                    if (
                        path.exists()
                        and not complete_marker.exists()
                        and not initialized_marker.exists()
                    ):
                        if _legacy_store_is_complete(
                            path,
                            variable=variable,
                            quantile_count=len(quantiles),
                            height=height,
                            width=width,
                        ):
                            _atomic_write_json(
                                complete_marker,
                                {
                                    "variable": variable,
                                    "lead": lead,
                                    "month": month,
                                    "adopted_legacy_store": True,
                                },
                            )
                            print(f"  Adopted existing completed store {path}")
                            continue
                        print(f"  Removing incomplete uncheckpointed store {path}")
                        _clear_parameter_state(path)

                    if (
                        not path.exists()
                        or not initialized_marker.exists()
                        or not _store_layout_matches(
                            path,
                            variable=variable,
                            quantile_count=len(quantiles),
                            height=height,
                            width=width,
                        )
                    ):
                        if path.exists() or progress_dir.exists():
                            print(f"  Reinitializing interrupted store {path}")
                            _clear_parameter_state(path)
                        initialize_parameter_store(
                            path,
                            quantiles,
                            lats,
                            lons,
                            variable=variable,
                            lead=lead,
                            month=month,
                            train_years=train_years,
                            wet_threshold=args.wet_threshold,
                            lat_tile=args.fit_lat_tile,
                            lon_tile=args.fit_lon_tile,
                        )

                    tile_number = 0
                    tile_total = (
                        math.ceil(height / args.fit_lat_tile)
                        * math.ceil(width / args.fit_lon_tile)
                    )
                    completed_tiles = len(list(progress_dir.glob("tile_*.done")))
                    if completed_tiles:
                        print(
                            f"  Resuming {variable} lead={lead} month={month:02d}: "
                            f"{completed_tiles}/{tile_total} tiles already saved",
                            flush=True,
                        )
                    group = zarr.open_group(str(path), mode="a")
                    for lat_start in range(0, height, args.fit_lat_tile):
                        lat_stop = min(lat_start + args.fit_lat_tile, height)
                        for lon_start in range(0, width, args.fit_lon_tile):
                            lon_stop = min(lon_start + args.fit_lon_tile, width)
                            tile_number += 1
                            tile_marker = tile_progress_marker(
                                path,
                                lat_start,
                                lat_stop,
                                lon_start,
                                lon_stop,
                            )
                            if tile_marker.exists():
                                continue
                            lat_slice = slice(lat_start, lat_stop)
                            lon_slice = slice(lon_start, lon_stop)
                            forecast_parts, observation_parts = [], []
                            for archive in archives:
                                forecast, observation = archive.training_samples(
                                    variable,
                                    lead_index,
                                    month,
                                    max_inits=args.max_inits,
                                    lat_slice=lat_slice,
                                    lon_slice=lon_slice,
                                )
                                if forecast.shape[0]:
                                    forecast_parts.append(forecast)
                                    observation_parts.append(observation)
                            if not forecast_parts:
                                raise ValueError(
                                    f"No training samples for {variable}, "
                                    f"lead {lead}, month {month}."
                                )
                            forecast = np.concatenate(forecast_parts, axis=0)
                            observation = np.concatenate(observation_parts, axis=0)
                            tile_parameters = fit_quantile_parameters(
                                forecast,
                                observation,
                                variable,
                                quantiles,
                                wet_threshold=args.wet_threshold,
                                min_wet_samples=args.min_wet_samples,
                                block_size=args.spatial_block_size,
                            )
                            write_parameter_tile(
                                group,
                                tile_parameters,
                                lat_slice=lat_slice,
                                lon_slice=lon_slice,
                            )
                            _atomic_write_json(
                                tile_marker,
                                {
                                    "lat_start": lat_start,
                                    "lat_stop": lat_stop,
                                    "lon_start": lon_start,
                                    "lon_stop": lon_stop,
                                },
                            )
                            del forecast_parts, observation_parts
                            del forecast, observation, tile_parameters
                            print(
                                f"  {variable} lead={lead} month={month:02d} "
                                f"tile {tile_number}/{tile_total} "
                                f"lat={lat_start}:{lat_stop} lon={lon_start}:{lon_stop}",
                                flush=True,
                            )
                    del group
                    completed_tiles = len(list(progress_dir.glob("tile_*.done")))
                    if completed_tiles != tile_total:
                        raise RuntimeError(
                            f"Only {completed_tiles}/{tile_total} tiles were saved for "
                            f"{variable}, lead {lead}, month {month}."
                        )
                    _atomic_write_json(
                        complete_marker,
                        {
                            "variable": variable,
                            "lead": lead,
                            "month": month,
                            "tiles": tile_total,
                        },
                    )
                    print(f"  Completed {path}", flush=True)
        finally:
            for archive in archives:
                archive.close()

    requested_manifest["complete"] = True
    manifest_path.write_text(json.dumps(requested_manifest, indent=2) + "\n")
    print(f"Fitted QM parameters: {parameter_root}")


@lru_cache(maxsize=2)
def load_parameter_store(path_text: str) -> dict[str, np.ndarray]:
    ds = xr.open_zarr(path_text, consolidated=False, chunks=None)
    try:
        return {
            name: np.asarray(ds[name].values)
            for name in ds.data_vars
        }
    finally:
        ds.close()


def valid_times(init_dates: pd.DatetimeIndex) -> np.ndarray:
    return np.stack(
        [
            (init_dates + pd.to_timedelta(7 * lead, unit="D")).values
            for lead in range(1, 5)
        ],
        axis=1,
    )


def _write_one_init(
    path: Path,
    init_date: pd.Timestamp,
    corrected: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    first: bool,
    split: str,
    train_years: tuple[int, ...],
) -> None:
    member_count = next(iter(corrected.values())).shape[0]
    data_vars = {
        f"qm_{variable}": (
            ("init", "geos_member", "lead", "lat", "lon"),
            value[None].astype(np.float32),
        )
        for variable, value in corrected.items()
    }
    init_index = pd.DatetimeIndex([init_date])
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "init": init_index.values,
            "geos_member": np.arange(1, member_count + 1, dtype=np.int32),
            "lead": np.arange(1, 5, dtype=np.int32),
            "lat": lats.astype(np.float32),
            "lon": lons.astype(np.float32),
            "valid_time": (("init", "lead"), valid_times(init_index)),
        },
        attrs={
            "method": "empirical_quantile_mapping",
            "training_years": f"{min(train_years)}-{max(train_years)}",
            "split": split,
            "mapping_conditioning": "variable, lead week, verifying month, grid point",
        },
    )
    for variable in corrected:
        ds[f"qm_{variable}"].attrs.update(
            {
                "units": VARIABLES[variable]["units"],
                "long_name": f"quantile-mapped FIMr1p1 {variable}",
            }
        )
    if first:
        encoding = {
            name: {
                "dtype": "float32",
                "chunks": (1, member_count, 4, min(45, len(lats)), min(90, len(lons))),
            }
            for name in data_vars
        }
        ds.to_zarr(path, mode="w", encoding=encoding)
    else:
        ds.to_zarr(path, mode="a", append_dim="init")
    ds.close()


def _prepare_output_store(path: Path, overwrite: bool) -> tuple[Path, bool]:
    if path.exists():
        if not overwrite:
            print(f"Corrected archive exists, skipping: {path}")
            return path, False
        shutil.rmtree(path)
    tmp_path = Path(str(path) + ".tmp")
    if tmp_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Incomplete temporary store exists: {tmp_path}. "
                "Inspect it or rerun with --overwrite."
            )
        shutil.rmtree(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return tmp_path, True


def _map_fields(
    raw_by_variable: dict[str, list[np.ndarray]],
    init_date: pd.Timestamp,
    parameter_root: Path,
    variables: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    corrected: dict[str, np.ndarray] = {}
    for variable in variables:
        mapped_leads = []
        for lead_index, raw in enumerate(raw_by_variable[variable]):
            valid = init_date + pd.Timedelta(days=7 * (lead_index + 1))
            path = parameter_store(
                parameter_root, variable, lead_index + 1, valid.month
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing QM parameters: {path}")
            parameters = load_parameter_store(str(path))
            mapped_leads.append(
                apply_quantile_map(
                    raw,
                    parameters,
                    variable,
                    wet_threshold=args.wet_threshold,
                    block_size=args.spatial_block_size,
                )
            )
        corrected[variable] = np.stack(mapped_leads, axis=1)
    return corrected


def apply_raw_year(
    data_root: Path,
    year: int,
    output_path: Path,
    parameter_root: Path,
    train_years: tuple[int, ...],
    variables: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    tmp_path, should_write = _prepare_output_store(output_path, args.overwrite)
    if not should_write:
        return
    archive = RawYear(data_root, year, variables)
    try:
        n_init = len(archive.init_dates)
        if args.max_inits is not None:
            n_init = min(n_init, args.max_inits)
        for init_index in range(n_init):
            raw_by_variable = {
                variable: [
                    archive.fields(variable, init_index, lead_index)[0]
                    for lead_index in range(4)
                ]
                for variable in variables
            }
            corrected = _map_fields(
                raw_by_variable,
                archive.init_dates[init_index],
                parameter_root,
                variables,
                args,
            )
            _write_one_init(
                tmp_path,
                archive.init_dates[init_index],
                corrected,
                archive.lats,
                archive.lons,
                first=init_index == 0,
                split="validation",
                train_years=train_years,
            )
            print(f"  validation {year} init {init_index + 1}/{n_init}", flush=True)
    finally:
        archive.close()
    tmp_path.rename(output_path)


def apply_evaluation_year(
    forecast_dir: Path,
    year: int,
    output_path: Path,
    parameter_root: Path,
    train_years: tuple[int, ...],
    variables: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    tmp_path, should_write = _prepare_output_store(output_path, args.overwrite)
    if not should_write:
        return
    source_path = forecast_dir / f"{year}.zarr"
    if not source_path.exists():
        raise FileNotFoundError(f"Evaluation forecast archive not found: {source_path}")
    ds = xr.open_zarr(source_path, consolidated=False, chunks=None)
    try:
        init_dates = pd.to_datetime(ds["init"].values).normalize()
        n_init = len(init_dates)
        if args.max_inits is not None:
            n_init = min(n_init, args.max_inits)
        lats = np.asarray(ds["lat"].values, dtype=np.float32)
        lons = np.asarray(ds["lon"].values, dtype=np.float32)
        for init_index in range(n_init):
            raw_by_variable = {
                variable: [
                    np.asarray(
                        ds[f"geos_{variable}"].isel(
                            init=init_index, lead=lead_index
                        ).values,
                        dtype=np.float32,
                    )
                    for lead_index in range(4)
                ]
                for variable in variables
            }
            corrected = _map_fields(
                raw_by_variable,
                init_dates[init_index],
                parameter_root,
                variables,
                args,
            )
            _write_one_init(
                tmp_path,
                init_dates[init_index],
                corrected,
                lats,
                lons,
                first=init_index == 0,
                split="evaluation",
                train_years=train_years,
            )
            print(f"  evaluation {year} init {init_index + 1}/{n_init}", flush=True)
    finally:
        ds.close()
    tmp_path.rename(output_path)


def apply_all(
    data_root: Path,
    forecast_dir: Path,
    corrected_dir: Path,
    parameter_root: Path,
    train_years: tuple[int, ...],
    validation_years: tuple[int, ...],
    evaluation_years: tuple[int, ...],
    variables: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    manifest_path = parameter_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"QM parameter manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"QM parameter fit is incomplete: {manifest_path}")
    expected = {
        "train_years": list(train_years),
        "variables": list(variables),
        "wet_threshold_mm_day": float(args.wet_threshold),
    }
    mismatch = [
        key for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatch:
        raise ValueError(
            f"Application settings differ from the frozen QM fit in {mismatch}; "
            "use the manifest settings or create a new --out-dir."
        )
    corrected_dir.mkdir(parents=True, exist_ok=True)
    for year in validation_years:
        print(f"Applying frozen QM to validation year {year} ...")
        apply_raw_year(
            data_root,
            year,
            corrected_dir / f"{year}.zarr",
            parameter_root,
            train_years,
            variables,
            args,
        )
    for year in evaluation_years:
        print(f"Applying frozen QM to evaluation year {year} ...")
        apply_evaluation_year(
            forecast_dir,
            year,
            corrected_dir / f"{year}.zarr",
            parameter_root,
            train_years,
            variables,
            args,
        )
    load_parameter_store.cache_clear()


def crps_standard(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    member_count = ensemble.shape[0]
    term1 = np.nanmean(np.abs(ensemble - observation[None]), axis=0)
    sorted_ensemble = np.sort(ensemble, axis=0)
    coefficient = 2.0 * np.arange(1, member_count + 1) - member_count - 1.0
    gini = np.tensordot(coefficient, sorted_ensemble, axes=(0, 0))
    return term1 - gini / (member_count * member_count)


def area_weights(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    return np.clip(np.cos(np.deg2rad(lats)), 0.0, None)[:, None] * np.ones(
        (1, len(lons)), dtype=np.float64
    )


def weighted_terms(field: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(field) & (weights > 0.0)
    return (
        float(np.sum(field[valid] * weights[valid])),
        float(np.sum(weights[valid])),
    )


def _score_pair(
    raw: np.ndarray,
    qm: np.ndarray,
    observation: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    raw_mean = np.nanmean(raw.astype(np.float64), axis=0)
    qm_mean = np.nanmean(qm.astype(np.float64), axis=0)
    fields = {
        "raw_crps": crps_standard(raw, observation),
        "qm_crps": crps_standard(qm, observation),
        "raw_mse": (raw_mean - observation) ** 2,
        "qm_mse": (qm_mean - observation) ** 2,
        "raw_bias": raw_mean - observation,
        "qm_bias": qm_mean - observation,
        "raw_spread": np.nanstd(raw.astype(np.float64), axis=0),
        "qm_spread": np.nanstd(qm.astype(np.float64), axis=0),
    }
    row: dict[str, float] = {}
    for name, field in fields.items():
        total, weight = weighted_terms(field, weights)
        row[f"{name}_wsum"] = total
        row[f"{name}_w"] = weight
    return row


def score_all(
    data_root: Path,
    forecast_dir: Path,
    corrected_dir: Path,
    out_dir: Path,
    validation_years: tuple[int, ...],
    evaluation_years: tuple[int, ...],
    variables: tuple[str, ...],
    args: argparse.Namespace,
    *,
    output_stem: str = "qm",
) -> None:
    rows: list[dict[str, object]] = []
    for split, years in (
        ("validation", validation_years),
        ("evaluation", evaluation_years),
    ):
        for year in years:
            qm_path = corrected_dir / f"{year}.zarr"
            if not qm_path.exists():
                raise FileNotFoundError(f"Corrected archive not found: {qm_path}")
            qm_ds = xr.open_zarr(qm_path, consolidated=False, chunks=None)
            raw_archive = None
            source_ds = None
            try:
                if split == "validation":
                    raw_archive = RawYear(data_root, year, variables)
                    init_dates = raw_archive.init_dates
                    lats, lons = raw_archive.lats, raw_archive.lons
                else:
                    source_ds = xr.open_zarr(
                        forecast_dir / f"{year}.zarr",
                        consolidated=False,
                        chunks=None,
                    )
                    init_dates = pd.to_datetime(source_ds["init"].values).normalize()
                    lats = np.asarray(source_ds["lat"].values)
                    lons = np.asarray(source_ds["lon"].values)
                weights = area_weights(lats, lons)
                n_init = len(init_dates)
                if args.max_inits is not None:
                    n_init = min(n_init, args.max_inits)
                for init_index in range(n_init):
                    for lead_index in range(4):
                        for variable in variables:
                            if raw_archive is not None:
                                raw, observation = raw_archive.fields(
                                    variable, init_index, lead_index
                                )
                            else:
                                raw = source_ds[f"geos_{variable}"].isel(
                                    init=init_index, lead=lead_index
                                ).values
                                observation = source_ds[f"obs_{variable}"].isel(
                                    init=init_index, lead=lead_index
                                ).values
                            qm = qm_ds[f"qm_{variable}"].isel(
                                init=init_index, lead=lead_index
                            ).values
                            row: dict[str, object] = {
                                "split": split,
                                "year": int(year),
                                "variable": variable,
                                "lead": lead_index + 1,
                                "init_time": init_dates[init_index],
                                "n_members": int(qm.shape[0]),
                            }
                            row.update(
                                _score_pair(
                                    np.asarray(raw, dtype=np.float64),
                                    np.asarray(qm, dtype=np.float64),
                                    np.asarray(observation, dtype=np.float64),
                                    weights,
                                )
                            )
                            rows.append(row)
                    print(f"  score {split} {year} init {init_index + 1}/{n_init}", flush=True)
            finally:
                qm_ds.close()
                if raw_archive is not None:
                    raw_archive.close()
                if source_ds is not None:
                    source_ds.close()

    per_init = pd.DataFrame(rows)
    per_init.to_csv(out_dir / f"{output_stem}_per_init_metrics.csv", index=False)

    def aggregate(group: pd.DataFrame, name: str) -> float:
        return float(group[f"{name}_wsum"].sum() / max(group[f"{name}_w"].sum(), 1e-12))

    aggregate_rows = []
    grouping = ["split", "variable", "lead"]
    for keys, group in per_init.groupby(grouping, sort=True):
        split, variable, lead = keys
        row = {
            "split": split,
            "year": "pooled",
            "variable": variable,
            "lead": int(lead),
            "n_init_rows": len(group),
        }
        for name in (
            "raw_crps",
            "qm_crps",
            "raw_mse",
            "qm_mse",
            "raw_bias",
            "qm_bias",
            "raw_spread",
            "qm_spread",
        ):
            row[name] = aggregate(group, name)
        row["raw_rmse"] = math.sqrt(max(row["raw_mse"], 0.0))
        row["qm_rmse"] = math.sqrt(max(row["qm_mse"], 0.0))
        row["qm_crps_skill_pct"] = 100.0 * (
            1.0 - row["qm_crps"] / max(row["raw_crps"], 1e-12)
        )
        row["qm_rmse_skill_pct"] = 100.0 * (
            1.0 - row["qm_rmse"] / max(row["raw_rmse"], 1e-12)
        )
        row["raw_spread_rmse"] = row["raw_spread"] / max(row["raw_rmse"], 1e-12)
        row["qm_spread_rmse"] = row["qm_spread"] / max(row["qm_rmse"], 1e-12)
        aggregate_rows.append(row)

    # Add all-lead rows without averaging percentages.
    for keys, group in per_init.groupby(["split", "variable"], sort=True):
        split, variable = keys
        row = {
            "split": split,
            "year": "pooled",
            "variable": variable,
            "lead": "all",
            "n_init_rows": len(group),
        }
        for name in (
            "raw_crps",
            "qm_crps",
            "raw_mse",
            "qm_mse",
            "raw_bias",
            "qm_bias",
            "raw_spread",
            "qm_spread",
        ):
            row[name] = aggregate(group, name)
        row["raw_rmse"] = math.sqrt(max(row["raw_mse"], 0.0))
        row["qm_rmse"] = math.sqrt(max(row["qm_mse"], 0.0))
        row["qm_crps_skill_pct"] = 100.0 * (
            1.0 - row["qm_crps"] / max(row["raw_crps"], 1e-12)
        )
        row["qm_rmse_skill_pct"] = 100.0 * (
            1.0 - row["qm_rmse"] / max(row["raw_rmse"], 1e-12)
        )
        row["raw_spread_rmse"] = row["raw_spread"] / max(row["raw_rmse"], 1e-12)
        row["qm_spread_rmse"] = row["qm_spread"] / max(row["qm_rmse"], 1e-12)
        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(out_dir / f"{output_stem}_aggregate_metrics.csv", index=False)
    columns = [
        "split",
        "variable",
        "lead",
        "raw_crps",
        "qm_crps",
        "qm_crps_skill_pct",
        "raw_rmse",
        "qm_rmse",
        "qm_rmse_skill_pct",
        "raw_bias",
        "qm_bias",
        "raw_spread_rmse",
        "qm_spread_rmse",
    ]
    print("\nQM raw-FIM comparison")
    print(aggregate_df[columns].round(4).to_string(index=False))


def main() -> None:
    args = parse_args()
    train_years = parse_years(args.train_years)
    validation_years = parse_years(args.validation_years)
    evaluation_years = parse_years(args.evaluation_years)
    validate_year_splits(train_years, validation_years, evaluation_years)
    variables = tuple(v.strip() for v in args.variables.split(",") if v.strip())
    unknown = sorted(set(variables) - set(VARIABLES))
    if unknown:
        raise ValueError(f"Unknown variables: {unknown}")
    if args.n_quantiles < 3:
        raise ValueError("--n-quantiles must be at least 3.")
    if args.fit_lat_tile < 1 or args.fit_lon_tile < 1:
        raise ValueError("--fit-lat-tile and --fit-lon-tile must be positive.")
    quantiles = np.linspace(0.0, 1.0, args.n_quantiles, dtype=np.float64)

    data_root = Path(args.data_root)
    forecast_dir = Path(args.forecast_dir)
    out_dir = Path(args.out_dir)
    parameter_root = out_dir / "qm_parameters"
    corrected_dir = out_dir / "corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("FIMr1p1 empirical quantile-mapping protocol")
    print(f"  training:   {train_years[0]}-{train_years[-1]} (fit only)")
    print(
        f"  validation: {validation_years[0]}-{validation_years[-1]} "
        "(diagnostic only; no refit)"
    )
    print(f"  evaluation: {evaluation_years[0]}-{evaluation_years[-1]}")
    print(f"  variables:  {variables}")

    if args.stage in ("all", "fit"):
        fit_all(
            data_root,
            parameter_root,
            train_years,
            variables,
            quantiles,
            args,
        )
    if args.stage in ("all", "apply"):
        apply_all(
            data_root,
            forecast_dir,
            corrected_dir,
            parameter_root,
            train_years,
            validation_years,
            evaluation_years,
            variables,
            args,
        )
    if args.stage in ("all", "score"):
        score_all(
            data_root,
            forecast_dir,
            corrected_dir,
            out_dir,
            validation_years,
            evaluation_years,
            variables,
            args,
        )
    if args.stage == "validate":
        apply_all(
            data_root,
            forecast_dir,
            corrected_dir,
            parameter_root,
            train_years,
            validation_years,
            (),
            variables,
            args,
        )
        score_all(
            data_root,
            forecast_dir,
            corrected_dir,
            out_dir,
            validation_years,
            (),
            variables,
            args,
            output_stem="qm_validation",
        )
    if args.stage == "evaluate":
        apply_all(
            data_root,
            forecast_dir,
            corrected_dir,
            parameter_root,
            train_years,
            (),
            evaluation_years,
            variables,
            args,
        )
        score_all(
            data_root,
            forecast_dir,
            corrected_dir,
            out_dir,
            (),
            evaluation_years,
            variables,
            args,
            output_stem="qm_evaluation",
        )


if __name__ == "__main__":
    main()
