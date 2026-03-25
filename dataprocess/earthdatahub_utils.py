from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


EARTHDATAHUB_PAT_ENV = "EARTHDATAHUB_PAT"
EARTHDATAHUB_MACHINE = "data.earthdatahub.destine.eu"


@contextlib.contextmanager
def earthdatahub_netrc_from_env():
    """Create a temporary .netrc from the Earth Data Hub token env var."""
    token = os.environ.get(EARTHDATAHUB_PAT_ENV)
    if not token:
        raise RuntimeError(
            f"Missing required environment variable {EARTHDATAHUB_PAT_ENV}. "
            "Set it to your Earth Data Hub personal access token before running."
        )

    with tempfile.TemporaryDirectory(prefix="edh_netrc_") as tmpdir:
        netrc_path = os.path.join(tmpdir, ".netrc")
        with open(netrc_path, "w", encoding="utf-8") as f:
            f.write(f"machine {EARTHDATAHUB_MACHINE}\n")
            f.write(f"    password {token}\n")
        os.chmod(netrc_path, stat.S_IRUSR | stat.S_IWUSR)

        old_home = os.environ.get("HOME")
        old_netrc = os.environ.get("NETRC")
        os.environ["HOME"] = tmpdir
        os.environ["NETRC"] = netrc_path
        try:
            yield
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home

            if old_netrc is None:
                os.environ.pop("NETRC", None)
            else:
                os.environ["NETRC"] = old_netrc


def choose_coord_name(ds: xr.Dataset | xr.DataArray, candidates: Iterable[str], label: str) -> str:
    items = set(ds.coords) | set(ds.dims)
    for name in candidates:
        if name in items:
            return name
    raise KeyError(f"Could not find {label}. Tried {list(candidates)}. Available: {sorted(items)}")


def load_geos_layout(geos_path: str) -> Tuple[pd.DatetimeIndex, xr.DataArray, xr.DataArray]:
    if not os.path.exists(geos_path):
        raise FileNotFoundError(f"Missing GEOS reference file: {geos_path}")

    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    try:
        s_name = choose_coord_name(ds_geos, ["S", "time", "init_time"], "GEOS init dimension")
        lat_name = choose_coord_name(ds_geos, ["Y", "latitude", "lat", "y"], "GEOS latitude coordinate")
        lon_name = choose_coord_name(ds_geos, ["X", "longitude", "lon", "x"], "GEOS longitude coordinate")

        init_dates = pd.to_datetime(ds_geos[s_name].values)
        target_lat = ds_geos[lat_name].copy(deep=True)
        target_lon = ds_geos[lon_name].copy(deep=True)
    finally:
        ds_geos.close()

    return init_dates, target_lat, target_lon


def normalize_longitudes(da: xr.DataArray, lon_name: str, target_lon: xr.DataArray) -> xr.DataArray:
    lon = da[lon_name]
    target_min = float(np.nanmin(target_lon.values))
    target_max = float(np.nanmax(target_lon.values))
    source_min = float(np.nanmin(lon.values))
    source_max = float(np.nanmax(lon.values))

    if target_min >= 0.0 and source_min < 0.0:
        da = da.assign_coords({lon_name: np.mod(da[lon_name], 360.0)})
        da = da.sortby(lon_name)
    elif target_max <= 180.0 and source_max > 180.0:
        shifted = ((da[lon_name] + 180.0) % 360.0) - 180.0
        da = da.assign_coords({lon_name: shifted})
        da = da.sortby(lon_name)

    return da


def normalize_source_grid(da: xr.DataArray, target_lat: xr.DataArray, target_lon: xr.DataArray) -> xr.DataArray:
    lat_name = choose_coord_name(da, ["latitude", "lat", "Y", "y"], "source latitude coordinate")
    lon_name = choose_coord_name(da, ["longitude", "lon", "X", "x"], "source longitude coordinate")

    da = normalize_longitudes(da, lon_name, target_lon)
    da = da.sortby(lat_name)

    rename_dict = {}
    if lat_name != target_lat.name:
        rename_dict[lat_name] = target_lat.name
    if lon_name != target_lon.name:
        rename_dict[lon_name] = target_lon.name
    if rename_dict:
        da = da.rename(rename_dict)

    return da


def build_time_window(init_dates: pd.DatetimeIndex) -> Tuple[str, str]:
    min_init = pd.Timestamp(init_dates.min()).normalize()
    max_init = pd.Timestamp(init_dates.max()).normalize()
    start_date = (min_init - pd.Timedelta(days=28)).strftime("%Y-%m-%d")
    end_date = (max_init - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return start_date, end_date


def weekly_means_from_daily_dataset(
    ds_daily: xr.Dataset,
    init_dates: pd.DatetimeIndex,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
    desc: str,
) -> xr.Dataset:
    if "time" not in ds_daily.dims:
        raise KeyError(f"Expected a time dimension in daily dataset. Found dims: {dict(ds_daily.sizes)}")

    if ds_daily.sizes.get("time", 0) == 0:
        raise ValueError("Daily dataset is empty after time selection.")

    template = ds_daily.isel(time=0, drop=True)
    samples = []
    skipped = 0

    for init_date in tqdm(init_dates, desc=desc):
        weeks = []
        valid = True

        for w in range(4):
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)
            w_start = w_end - pd.Timedelta(days=6)
            chunk = ds_daily.sel(time=slice(w_start, w_end))

            if chunk.sizes.get("time", 0) < 7:
                valid = False
                break

            weeks.append(chunk.mean(dim="time"))

        if valid and len(weeks) == 4:
            sample = xr.concat(weeks, dim="L").assign_coords(L=np.arange(4))
        else:
            skipped += 1
            sample = xr.full_like(template.expand_dims(L=4), np.nan).assign_coords(L=np.arange(4))

        samples.append(sample)

    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} init dates were NaN-filled due to incomplete daily coverage")

    ds_out = xr.concat(samples, dim="S").assign_coords(S=init_dates)
    ds_out = ds_out.transpose("S", "L", target_lat.name, target_lon.name)
    return ds_out
