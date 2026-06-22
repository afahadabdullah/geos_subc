"""Build and validate the file-free/static geography inputs used by flow_finalv1_global."""

from __future__ import annotations

import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


STATIC_CHANNEL_NAMES = (
    "elevation_minmax",
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
    preferred = os.path.join(data_dir, "sss_weekly_2020.zarr")
    candidates = [preferred] + sorted(
        glob(os.path.join(data_dir, "sss_weekly_*.zarr"))
        + glob(os.path.join(data_dir, "sss", "*.zarr")),
        reverse=True,
    )
    return next((path for path in candidates if os.path.exists(path)), None)


def _load_cached_land_mask(mask_path, lats, lons):
    if not mask_path or not os.path.exists(mask_path):
        return None
    import torch

    cached = torch.load(mask_path, map_location="cpu", weights_only=True)
    if "is_land" not in cached:
        raise ValueError(
            f"Configured land-mask cache has no 'is_land' field: {mask_path}"
        )
    land_mask = np.asarray(cached["is_land"], dtype=np.float32).squeeze()
    expected_shape = (len(lats), len(lons))
    if land_mask.shape != expected_shape:
        raise ValueError(
            f"Configured land mask has shape {land_mask.shape}, expected {expected_shape}: "
            f"{mask_path}"
        )
    if not np.isin(land_mask, [0.0, 1.0]).all():
        raise ValueError(f"Configured land mask is not binary: {mask_path}")
    if land_mask.min() == land_mask.max():
        raise ValueError(f"Configured land mask is spatially uniform: {mask_path}")
    return land_mask


def _load_land_mask(data_dir, lats, lons, mask_cache_path=None):
    mask_cache_path = _resolve_path(mask_cache_path)
    cached_land_mask = _load_cached_land_mask(mask_cache_path, lats, lons)
    if cached_land_mask is not None:
        return cached_land_mask, mask_cache_path

    candidates = []
    preferred = _find_sss_path(data_dir)
    if preferred:
        candidates.append(preferred)
    candidates.extend(
        path
        for path in sorted(
            glob(os.path.join(data_dir, "sss_weekly_*.zarr"))
            + glob(os.path.join(data_dir, "sss", "*.zarr")),
            reverse=True,
        )
        if path not in candidates
    )
    if not candidates:
        warnings.warn(
            "No SSS Zarr was found for the flow_finalv1_global land mask. The static land channel "
            "will be zero until the artifact is rebuilt where SSS data are available.",
            RuntimeWarning,
        )
        return np.zeros((len(lats), len(lons)), dtype=np.float32), "zero_fallback"

    lat_idx, lon_idx = _target_indices(lats, lons)
    rejected = []
    for sss_path in candidates:
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
                rejected.append(f"{sss_path}: no SSS variable")
                continue
            da = ds[var_name]
            for dim in tuple(da.dims[:-2]):
                da = da.isel({dim: 0})
            global_sss = np.asarray(da.values).squeeze()
        finally:
            ds.close()

        if global_sss.shape[-2:] == (360, 181):
            global_sss = global_sss.T
        if global_sss.shape[-2:] != (181, 360):
            rejected.append(f"{sss_path}: shape={global_sss.shape}")
            continue
        local_sss = np.take(np.take(global_sss, lat_idx, axis=-2), lon_idx, axis=-1)
        land_mask = np.isnan(local_sss).astype(np.float32)
        if land_mask.min() != land_mask.max():
            return land_mask, sss_path
        rejected.append(
            f"{sss_path}: uniform mask value={float(land_mask.flat[0]):.0f}"
        )

    raise ValueError(
        "Could not derive a non-uniform global land mask from SSS. "
        f"Rejected sources: {'; '.join(rejected)}"
    )


def _is_axis_variable(variable, axis):
    standard_name = str(variable.attrs.get("standard_name", "")).strip().lower()
    units = str(variable.attrs.get("units", "")).strip().lower()
    if axis == "latitude":
        return standard_name == "latitude" or units in {
            "degrees_north",
            "degree_north",
            "degrees_n",
            "degree_n",
        }
    return standard_name == "longitude" or units in {
        "degrees_east",
        "degree_east",
        "degrees_e",
        "degree_e",
    }


def _find_spatial_axis(ds, da, axis):
    aliases = {
        "latitude": ("latitude", "lat", "Y", "y"),
        "longitude": ("longitude", "lon", "X", "x"),
    }[axis]

    # Prefer a coordinate already attached to the elevation field.
    for name in da.coords:
        variable = da.coords[name]
        if name in aliases or _is_axis_variable(variable, axis):
            if variable.ndim == 1:
                return variable.dims[0], np.asarray(variable.values, dtype=np.float64), name

    # GLDAS NetCDF files can store latitude/longitude as ordinary variables
    # rather than marking them as coordinates on GLDAS_elevation.
    for name in aliases:
        if name in ds.variables and ds[name].ndim == 1:
            variable = ds[name]
            return variable.dims[0], np.asarray(variable.values, dtype=np.float64), name
    for name, variable in ds.variables.items():
        if variable.ndim == 1 and _is_axis_variable(variable, axis):
            return variable.dims[0], np.asarray(variable.values, dtype=np.float64), name

    raise ValueError(
        f"Could not identify the {axis} axis for elevation variable {da.name!r}. "
        f"dims={da.dims}, coordinates={list(da.coords)}, variables={list(ds.variables)}"
    )


def _find_elevation_variable(ds, requested_var=None):
    if requested_var:
        if requested_var not in ds:
            raise ValueError(
                f"elevation_variable={requested_var!r} is not in the file; "
                f"variables={list(ds.data_vars)}"
            )
        return requested_var

    aliases = (
        "GLDAS_elevation",
        "gldas_elevation",
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
    for name in aliases:
        if name in ds.data_vars:
            return name
    for name, variable in ds.data_vars.items():
        if str(variable.attrs.get("standard_name", "")).strip().lower() == "elevation":
            return name
    raise ValueError(
        "Could not identify an elevation variable. "
        f"Data variables and standard_name values: "
        f"{[(name, variable.attrs.get('standard_name')) for name, variable in ds.data_vars.items()]}"
    )


def _bilinear_interpolate_regular(
    source,
    source_lats,
    source_lons,
    target_lats,
    target_lons,
):
    """Bilinear interpolation on monotonic 1-D lat/lon axes without SciPy."""
    source = np.asarray(source, dtype=np.float64)
    source_lats = np.asarray(source_lats, dtype=np.float64)
    source_lons = np.asarray(source_lons, dtype=np.float64)
    target_lats = np.asarray(target_lats, dtype=np.float64)
    target_lons = np.asarray(target_lons, dtype=np.float64)
    if source.shape != (source_lats.size, source_lons.size):
        raise ValueError(
            f"Elevation data shape {source.shape} does not match coordinate sizes "
            f"{source_lats.size}x{source_lons.size}."
        )
    if np.any(np.diff(source_lats) <= 0) or np.any(np.diff(source_lons) <= 0):
        raise ValueError("Source latitude and longitude must be strictly increasing.")
    if (
        target_lats.min() < source_lats.min()
        or target_lats.max() > source_lats.max()
        or target_lons.min() < source_lons.min()
        or target_lons.max() > source_lons.max()
    ):
        raise ValueError(
            "Target geography grid lies outside the source elevation grid: "
            f"target lat={target_lats.min()}..{target_lats.max()}, "
            f"lon={target_lons.min()}..{target_lons.max()}; "
            f"source lat={source_lats.min()}..{source_lats.max()}, "
            f"lon={source_lons.min()}..{source_lons.max()}."
        )

    lat_hi = np.searchsorted(source_lats, target_lats, side="right")
    lon_hi = np.searchsorted(source_lons, target_lons, side="right")
    lat_hi = np.clip(lat_hi, 1, source_lats.size - 1)
    lon_hi = np.clip(lon_hi, 1, source_lons.size - 1)
    lat_lo = lat_hi - 1
    lon_lo = lon_hi - 1
    lat_fraction = (
        (target_lats - source_lats[lat_lo])
        / (source_lats[lat_hi] - source_lats[lat_lo])
    )
    lon_fraction = (
        (target_lons - source_lons[lon_lo])
        / (source_lons[lon_hi] - source_lons[lon_lo])
    )

    lower_left = source[lat_lo[:, None], lon_lo[None, :]]
    lower_right = source[lat_lo[:, None], lon_hi[None, :]]
    upper_left = source[lat_hi[:, None], lon_lo[None, :]]
    upper_right = source[lat_hi[:, None], lon_hi[None, :]]
    lon_fraction = lon_fraction[None, :]
    lower = lower_left * (1.0 - lon_fraction) + lower_right * lon_fraction
    upper = upper_left * (1.0 - lon_fraction) + upper_right * lon_fraction
    lat_fraction = lat_fraction[:, None]
    return lower * (1.0 - lat_fraction) + upper * lat_fraction


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
                f"Configured flow_finalv1_global elevation file was not found: {configured}"
            )
        warnings.warn(
            "No elevation/orography file is available for flow_finalv1_global. Using an explicit zero "
            "elevation channel. Set elevation_file and rebuild_static_geography=true when "
            "a real file becomes available.",
            RuntimeWarning,
        )
        zeros = np.zeros((len(lats), len(lons)), dtype=np.float32)
        return zeros, {
            "source": "zero_fallback",
            "available": False,
            "normalization": "minmax",
            "min_m": 0.0,
            "max_m": 1.0,
        }, zeros.copy()

    ds = xr.open_dataset(elevation_path)
    try:
        requested_var = config.get("elevation_variable")
        var_name = _find_elevation_variable(ds, requested_var=requested_var)
        da = ds[var_name]

        da = da.squeeze(drop=True)
        lat_dim, source_lats, lat_name = _find_spatial_axis(ds, da, "latitude")
        lon_dim, source_lons, lon_name = _find_spatial_axis(ds, da, "longitude")
        for dim in tuple(da.dims):
            if dim not in {lat_dim, lon_dim}:
                da = da.isel({dim: 0})

        da = da.assign_coords({lat_dim: source_lats, lon_dim: source_lons})
        source_lons = np.mod(source_lons, 360.0)
        lon_order = np.argsort(source_lons)
        sorted_lons = source_lons[lon_order]
        unique_lons, unique_positions = np.unique(sorted_lons, return_index=True)
        lon_order = lon_order[unique_positions]
        da = da.isel({lon_dim: lon_order})
        da = da.assign_coords({lon_dim: unique_lons})
        if float(da[lat_dim][0]) > float(da[lat_dim][-1]):
            da = da.isel({lat_dim: slice(None, None, -1)})
        source_lats = np.asarray(da[lat_dim].values, dtype=np.float64)
        source_lons = np.asarray(da[lon_dim].values, dtype=np.float64)
        source_elevation = np.asarray(
            da.transpose(lat_dim, lon_dim).values,
            dtype=np.float64,
        )
        # GLDAS has missing values over water. Fill these with physical sea
        # level before interpolation so coastal 1-degree cells are averaged
        # consistently rather than becoming NaN.
        source_elevation = np.nan_to_num(
            source_elevation,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        source_lat_min = float(source_lats.min())
        source_lat_max = float(source_lats.max())
        target_lats = np.asarray(lats, dtype=np.float64)
        outside_source_latitudes = (
            (target_lats < source_lat_min) | (target_lats > source_lat_max)
        )

        # GLDAS longitudes are cell centers (typically 0.125..359.875).
        # Add wrapped edge columns so the global 0-degree model column is
        # interpolated continuously across the Greenwich seam.
        source_lons = np.concatenate(
            ([source_lons[-1] - 360.0], source_lons, [source_lons[0] + 360.0])
        )
        source_elevation = np.concatenate(
            (source_elevation[:, -1:], source_elevation, source_elevation[:, :1]),
            axis=1,
        )
        elevation = _bilinear_interpolate_regular(
            source_elevation,
            source_lats,
            source_lons,
            np.clip(target_lats, source_lat_min, source_lat_max),
            np.mod(np.asarray(lons, dtype=np.float64), 360.0),
        )
        # GLDAS does not cover the full Antarctic/polar cap. Those target rows
        # are represented as physical sea level rather than extrapolating the
        # nearest available terrain indefinitely toward the pole.
        elevation[outside_source_latitudes, :] = 0.0
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
    normalization = str(config.get("elevation_normalization", "minmax")).lower()
    if normalization != "minmax":
        raise ValueError(
            f"flow_finalv1_global elevation_normalization must be 'minmax', got {normalization!r}."
        )
    min_m = float(elevation.min())
    max_m = float(elevation.max())
    if max_m - min_m < 1e-6:
        raise ValueError(
            f"Elevation range is degenerate: min={min_m}, max={max_m}."
        )
    elevation_norm = (
        2.0 * (np.clip(elevation, min_m, max_m) - min_m) / (max_m - min_m) - 1.0
    ).astype(np.float32)
    return elevation_norm, {
        "source": elevation_path,
        "available": True,
        "normalization": normalization,
        "min_m": min_m,
        "max_m": max_m,
        "variable": var_name,
        "latitude_variable": lat_name,
        "longitude_variable": lon_name,
        "source_latitude_min": source_lat_min,
        "source_latitude_max": source_lat_max,
        "outside_source_latitude_rows_filled_sea_level": int(
            outside_source_latitudes.sum()
        ),
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
        f"flow_finalv1_global static geography alignment | elevation={metadata['elevation']['source']}",
        fontsize=12,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_topography_diagnostic(channels, elevation_m, lats, lons, path, metadata):
    """Save a focused elevation/land alignment plot in physical and normalized units."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    elevation_meta = metadata.get("elevation", {})
    elevation_min = float(elevation_meta.get("min_m", 0.0))
    elevation_max = float(elevation_meta.get("max_m", 1.0))
    elevation_norm = np.asarray(channels[0], dtype=np.float32)
    elevation_m = np.asarray(elevation_m, dtype=np.float32)
    land_mask = np.asarray(channels[1], dtype=np.float32)
    extent = [float(lons[0]), float(lons[-1]), float(lats[0]), float(lats[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
    panels = (
        (elevation_m, "terrain", "GLDAS elevation (m)"),
        (elevation_norm, "coolwarm", "Elevation min-max normalized [-1, 1]"),
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
        "flow_finalv1_global full-global static topography alignment\n"
        f"source={elevation_meta.get('source', 'unknown')} | "
        f"domain min={elevation_min:.1f} m, max={elevation_max:.1f} m",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_or_build_static_geography(config, lats, lons, output_dir=None):
    """Return a [5,H,W] float tensor and provenance metadata."""
    import torch

    lats = np.asarray(lats, dtype=np.float32)
    lons = np.asarray(lons, dtype=np.float32)
    artifact_path = _resolve_path(
        config.get("static_geography_file", "ml_model/static_geography_flow_finalv1_global.pt")
    )
    configured_land_mask_path = _resolve_path(config.get("static_land_mask_file"))
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
                0.5 * (channels[0] + 1.0)
                * (
                    float(elevation_meta.get("max_m", 1.0))
                    - float(elevation_meta.get("min_m", 0.0))
                )
                + float(elevation_meta.get("min_m", 0.0))
            )
        elevation_m = torch.as_tensor(elevation_m, dtype=torch.float32)
        expected_elevation_path = _find_elevation_path(config)
        expected_land_source = (
            configured_land_mask_path
            if configured_land_mask_path and os.path.exists(configured_land_mask_path)
            else None
        )
        elevation_now_available = expected_elevation_path is not None
        land_now_available = _find_sss_path(_resolve_path(config["data_dir"])) is not None
        stale_elevation = (
            elevation_now_available
            and (
                not bool(metadata.get("elevation", {}).get("available", False))
                or _resolve_path(metadata.get("elevation", {}).get("source"))
                != expected_elevation_path
                or metadata.get("elevation", {}).get("normalization") != "minmax"
            )
        )
        stale_land = (
            (
                expected_land_source is not None
                and _resolve_path(metadata.get("land_source")) != expected_land_source
            )
            or (
                land_now_available
                and metadata.get("land_source") == "zero_fallback"
            )
            or float(channels[1].min()) == float(channels[1].max())
        )
        rebuild = stale_elevation or stale_land

    if artifact is None or rebuild:
        land_mask, land_source = _load_land_mask(
            _resolve_path(config["data_dir"]),
            lats,
            lons,
            mask_cache_path=configured_land_mask_path,
        )
        elevation, elevation_meta, elevation_m_array = _load_elevation(config, lats, lons)
        elevation_m = torch.from_numpy(elevation_m_array)
        lat_norm, lon_sin, lon_cos = _make_coordinate_channels(lats, lons)
        channel_array = np.stack(
            [elevation, land_mask, lat_norm, lon_sin, lon_cos],
            axis=0,
        ).astype(np.float32)
        channels = torch.from_numpy(channel_array)
        metadata = {
            "version": "flow_finalv1_global",
            "channel_names": STATIC_CHANNEL_NAMES,
            "land_source": land_source,
            "land_pixels": int(land_mask.sum()),
            "ocean_pixels": int(land_mask.size - land_mask.sum()),
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
        diagnostic_path = os.path.join(output_dir, "static_geography_flow_finalv1_global_alignment.png")
        _save_diagnostic(channels.numpy(), lats, lons, diagnostic_path, metadata)
    if output_dir and bool(config.get("plot_topography", True)):
        topography_path = os.path.join(output_dir, "topography_flow_finalv1_global_diagnostic.png")
        _save_topography_diagnostic(
            channels.numpy(),
            elevation_m.numpy(),
            lats,
            lons,
            topography_path,
            metadata,
        )

    return channels.contiguous(), metadata
