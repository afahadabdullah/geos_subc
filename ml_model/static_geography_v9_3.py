"""Build and validate the file-free/static geography inputs used by v9.3."""

from __future__ import annotations

import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr


STATIC_CHANNEL_NAMES = (
    "elevation_zscore",
    "land_mask",
    "latitude_normalized",
    "longitude_sin",
    "longitude_cos",
)


def _resolve_path(path):
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def _validate_grid(values, expected, name):
    values = np.asarray(values, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if values.shape != expected.shape or not np.allclose(values, expected, atol=1e-4):
        raise ValueError(
            f"Static geography {name} coordinates do not match the target grid: "
            f"artifact={values.shape}, target={expected.shape}."
        )


def _target_indices(lats, lons):
    lat_idx = np.rint(np.asarray(lats, dtype=np.float64) + 90.0).astype(np.int64)
    lon_idx = np.rint(np.mod(np.asarray(lons, dtype=np.float64), 360.0)).astype(np.int64)
    if np.any(lat_idx < 0) or np.any(lat_idx > 180):
        raise ValueError("Target latitudes are outside the supported 1-degree global grid.")
    return lat_idx, lon_idx


def _find_sss_path(data_dir):
    candidates = sorted(
        glob(os.path.join(data_dir, "sss_weekly_*.zarr"))
        + glob(os.path.join(data_dir, "sss", "*.zarr")),
        reverse=True,
    )
    return next((path for path in candidates if os.path.exists(path)), None)


def _load_land_mask(data_dir, lats, lons):
    sss_path = _find_sss_path(data_dir)
    if sss_path is None:
        warnings.warn(
            "No SSS Zarr was found for the v9.3 land mask. The static land channel "
            "will be zero until the artifact is rebuilt where SSS data are available.",
            RuntimeWarning,
        )
        return np.zeros((len(lats), len(lons)), dtype=np.float32), "zero_fallback"

    ds = xr.open_zarr(sss_path, consolidated=False)
    try:
        var_name = next(
            (
                name
                for name in ("sss", "SSS", "sos", "SOS", "sea_surface_salinity", "s_surface")
                if name in ds
            ),
            None,
        )
        if var_name is None:
            raise ValueError(f"No SSS variable found in {sss_path}; variables={list(ds.data_vars)}")
        da = ds[var_name]
        for dim in tuple(da.dims[:-2]):
            da = da.isel({dim: 0})
        global_sss = np.asarray(da.values).squeeze()
    finally:
        ds.close()

    if global_sss.shape[-2:] == (360, 181):
        global_sss = global_sss.T
    if global_sss.shape[-2:] != (181, 360):
        raise ValueError(
            f"Expected SSS on a 181x360 grid, got {global_sss.shape} from {sss_path}."
        )
    lat_idx, lon_idx = _target_indices(lats, lons)
    local_sss = np.take(np.take(global_sss, lat_idx, axis=-2), lon_idx, axis=-1)
    return np.isnan(local_sss).astype(np.float32), sss_path


def _coordinate_name(da, options):
    return next((name for name in options if name in da.coords), None)


def _find_elevation_path(config):
    data_dir = _resolve_path(config["data_dir"])
    configured = _resolve_path(config.get("elevation_file"))
    if configured:
        return configured if os.path.exists(configured) else None
    candidates = [
        os.path.join(data_dir, "era5_geopotential.nc"),
        os.path.join(data_dir, "era5_land_gepotential.nc"),
        os.path.join(data_dir, "orography.nc"),
        os.path.join(data_dir, "elevation.nc"),
    ]
    return next((path for path in candidates if path and os.path.exists(path)), None)


def _load_elevation(config, lats, lons):
    elevation_path = _find_elevation_path(config)
    if elevation_path is None:
        configured = _resolve_path(config.get("elevation_file"))
        if configured and bool(config.get("require_elevation_file", False)):
            raise FileNotFoundError(
                f"Configured v9.3 elevation file was not found: {configured}"
            )
        warnings.warn(
            "No elevation/orography file is available for v9.3. Using an explicit zero "
            "elevation channel. Set elevation_file and rebuild_static_geography=true when "
            "a real file becomes available.",
            RuntimeWarning,
        )
        zeros = np.zeros((len(lats), len(lons)), dtype=np.float32)
        return zeros, {
            "source": "zero_fallback",
            "available": False,
            "mean_m": 0.0,
            "std_m": 1.0,
        }, zeros.copy()

    ds = xr.open_dataset(elevation_path)
    try:
        requested_var = config.get("elevation_variable")
        if requested_var:
            if requested_var not in ds:
                raise ValueError(
                    f"elevation_variable={requested_var!r} is not in {elevation_path}; "
                    f"variables={list(ds.data_vars)}"
                )
            da = ds[requested_var]
        else:
            var_name = next(
                (
                    name
                    for name in (
                        "elevation",
                        "Elevation",
                        "elev",
                        "ELEV",
                        "orography",
                        "topography",
                        "z",
                        "geopotential",
                        "surface_geopotential",
                    )
                    if name in ds
                ),
                next(iter(ds.data_vars)),
            )
            da = ds[var_name]

        da = da.squeeze(drop=True)
        lat_name = _coordinate_name(da, ("latitude", "lat", "Y", "y"))
        lon_name = _coordinate_name(da, ("longitude", "lon", "X", "x"))
        if lat_name is None or lon_name is None:
            raise ValueError(
                f"Could not identify latitude/longitude coordinates in {elevation_path}."
            )
        for dim in tuple(da.dims):
            if dim not in {lat_name, lon_name}:
                da = da.isel({dim: 0})

        source_lons = np.mod(np.asarray(da[lon_name].values, dtype=np.float64), 360.0)
        lon_order = np.argsort(source_lons)
        sorted_lons = source_lons[lon_order]
        unique_lons, unique_positions = np.unique(sorted_lons, return_index=True)
        lon_order = lon_order[unique_positions]
        da = da.isel({lon_name: lon_order})
        da = da.assign_coords({lon_name: unique_lons})
        if float(da[lat_name][0]) > float(da[lat_name][-1]):
            da = da.isel({lat_name: slice(None, None, -1)})
        local = da.interp(
            {
                lat_name: xr.DataArray(np.asarray(lats), dims="target_lat"),
                lon_name: xr.DataArray(np.asarray(lons) % 360.0, dims="target_lon"),
            },
            method="linear",
        )
        elevation = np.asarray(local.values, dtype=np.float64)
    finally:
        ds.close()

    elevation = np.squeeze(elevation)
    if elevation.shape != (len(lats), len(lons)):
        raise ValueError(
            f"Interpolated elevation has shape {elevation.shape}; "
            f"expected {(len(lats), len(lons))}."
        )
    if not np.isfinite(elevation).all():
        # GLDAS may use missing values over water. Sea-level zero is the
        # physically meaningful fill for the local elevation input.
        elevation = np.nan_to_num(elevation, nan=0.0, posinf=0.0, neginf=0.0)

    # ERA5 geopotential is commonly stored in m2 s-2; convert it to metres.
    if np.nanpercentile(np.abs(elevation), 95) > 12000.0:
        elevation = elevation / 9.80665
    elevation = np.maximum(elevation, 0.0)
    mean_m = float(elevation.mean())
    std_m = float(elevation.std())
    if std_m < 1e-6:
        std_m = 1.0
    elevation_norm = np.clip((elevation - mean_m) / std_m, -5.0, 5.0).astype(np.float32)
    return elevation_norm, {
        "source": elevation_path,
        "available": True,
        "mean_m": mean_m,
        "std_m": std_m,
    }, elevation.astype(np.float32)


def _make_coordinate_channels(lats, lons):
    lat_grid, lon_grid = np.meshgrid(
        np.asarray(lats, dtype=np.float32),
        np.asarray(lons, dtype=np.float32),
        indexing="ij",
    )
    lat_norm = np.clip(lat_grid / 90.0, -1.0, 1.0)
    lon_rad = np.deg2rad(np.mod(lon_grid, 360.0))
    return (
        lat_norm.astype(np.float32),
        np.sin(lon_rad).astype(np.float32),
        np.cos(lon_rad).astype(np.float32),
    )


def _save_diagnostic(channels, lats, lons, path, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, len(STATIC_CHANNEL_NAMES), figsize=(25, 5), constrained_layout=True)
    extent = [float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1])]
    cmaps = ("terrain", "Greens", "coolwarm", "twilight", "twilight")
    for idx, (ax, name, cmap) in enumerate(zip(axes, STATIC_CHANNEL_NAMES, cmaps)):
        image = ax.imshow(
            channels[idx],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(name)
        ax.set_xlabel("longitude")
        if idx == 0:
            ax.set_ylabel("latitude")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"v9.3 static geography alignment | elevation={metadata['elevation']['source']}",
        fontsize=12,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_topography_diagnostic(channels, elevation_m, lats, lons, path, metadata):
    """Save a focused elevation/land alignment plot in physical and normalized units."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    elevation_meta = metadata.get("elevation", {})
    elevation_mean = float(elevation_meta.get("mean_m", 0.0))
    elevation_std = float(elevation_meta.get("std_m", 1.0))
    elevation_norm = np.asarray(channels[0], dtype=np.float32)
    elevation_m = np.maximum(np.asarray(elevation_m, dtype=np.float32), 0.0)
    land_mask = np.asarray(channels[1], dtype=np.float32)
    extent = [float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    panels = (
        (elevation_m, "terrain", "GLDAS elevation (m)"),
        (elevation_norm, "coolwarm", "Elevation normalized for model"),
        (land_mask, "Greens", "Land mask (1=land, 0=ocean)"),
    )
    for ax, (field, cmap, title) in zip(axes, panels):
        image = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "v9.3 South Asia static topography alignment\n"
        f"source={elevation_meta.get('source', 'unknown')} | "
        f"domain mean={elevation_mean:.1f} m, std={elevation_std:.1f} m",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_or_build_static_geography(config, lats, lons, output_dir=None):
    """Return a [5,H,W] float tensor and provenance metadata."""
    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    artifact_path = _resolve_path(
        config.get("static_geography_file", "ml_model/static_geography_multiv9_3_sa.pt")
    )
    rebuild = bool(config.get("rebuild_static_geography", False))

    artifact = None
    if artifact_path and os.path.exists(artifact_path) and not rebuild:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        _validate_grid(artifact["lats"], lats, "latitude")
        _validate_grid(artifact["lons"], lons, "longitude")
        channels = artifact["channels"].float()
        if tuple(channels.shape) != (len(STATIC_CHANNEL_NAMES), len(lats), len(lons)):
            raise ValueError(
                f"Static geography channels have shape {tuple(channels.shape)}, expected "
                f"{(len(STATIC_CHANNEL_NAMES), len(lats), len(lons))}."
            )
        metadata = artifact.get("metadata", {})
        elevation_m = artifact.get("elevation_m")
        if elevation_m is None:
            elevation_meta = metadata.get("elevation", {})
            elevation_m = (
                channels[0] * float(elevation_meta.get("std_m", 1.0))
                + float(elevation_meta.get("mean_m", 0.0))
            )
        elevation_m = torch.as_tensor(elevation_m, dtype=torch.float32)
        expected_elevation_path = _find_elevation_path(config)
        elevation_now_available = expected_elevation_path is not None
        land_now_available = _find_sss_path(_resolve_path(config["data_dir"])) is not None
        stale_elevation = (
            elevation_now_available
            and (
                not bool(metadata.get("elevation", {}).get("available", False))
                or _resolve_path(metadata.get("elevation", {}).get("source"))
                != expected_elevation_path
            )
        )
        stale_land = (
            land_now_available and metadata.get("land_source") == "zero_fallback"
        )
        rebuild = stale_elevation or stale_land

    if artifact is None or rebuild:
        land_mask, land_source = _load_land_mask(_resolve_path(config["data_dir"]), lats, lons)
        elevation, elevation_meta, elevation_m_array = _load_elevation(config, lats, lons)
        elevation_m = torch.from_numpy(elevation_m_array)
        lat_norm, lon_sin, lon_cos = _make_coordinate_channels(lats, lons)
        channel_array = np.stack(
            [elevation, land_mask, lat_norm, lon_sin, lon_cos],
            axis=0,
        ).astype(np.float32)
        channels = torch.from_numpy(channel_array)
        metadata = {
            "version": "v9.3",
            "channel_names": STATIC_CHANNEL_NAMES,
            "land_source": land_source,
            "elevation": elevation_meta,
            "grid_shape": (len(lats), len(lons)),
        }
        if artifact_path and bool(config.get("cache_static_geography", True)):
            artifact_dir = os.path.dirname(artifact_path)
            if artifact_dir:
                os.makedirs(artifact_dir, exist_ok=True)
            torch.save(
                {
                    "channels": channels,
                    "elevation_m": elevation_m,
                    "lats": torch.from_numpy(lats.copy()),
                    "lons": torch.from_numpy(lons.copy()),
                    "metadata": metadata,
                },
                artifact_path,
            )

    if output_dir and bool(config.get("plot_static_geography", True)):
        diagnostic_path = os.path.join(output_dir, "static_geography_v9_3_alignment.png")
        _save_diagnostic(channels.numpy(), lats, lons, diagnostic_path, metadata)
    if output_dir and bool(config.get("plot_topography", True)):
        topography_path = os.path.join(output_dir, "topography_v9_3_diagnostic.png")
        _save_topography_diagnostic(
            channels.numpy(),
            elevation_m.numpy(),
            lats,
            lons,
            topography_path,
            metadata,
        )

    return channels.contiguous(), metadata
