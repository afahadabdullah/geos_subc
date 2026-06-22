#!/usr/bin/env python3
"""Preflight the GLDAS elevation file before launching expensive flow_finalv1_global training."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from static_geography_flow_finalv1_global import (
    _find_elevation_variable,
    _find_spatial_axis,
    _load_elevation,
)


DEFAULT_FILE = "/home1/11353/afahad/geos_subc/ml_model/GLDASp5_elevation_025d.nc4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect GLDAS topography schema and verify the flow_finalv1_global global grid."
    )
    parser.add_argument("--file", default=DEFAULT_FILE, help="GLDAS elevation NetCDF file")
    parser.add_argument("--variable", default=None, help="Optional explicit elevation variable")
    parser.add_argument(
        "--output",
        default="ml_output_flow_finalv1_global_noisectx_t2mres/"
        "topography_flow_finalv1_global_preflight.png",
        help="Diagnostic PNG path",
    )
    return parser.parse_args()


def finite_summary(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if not finite.any():
        return "no finite values"
    selected = values[finite]
    return (
        f"finite={finite.sum()}/{values.size}, missing={values.size - finite.sum()}, "
        f"min={selected.min():.3f}, mean={selected.mean():.3f}, "
        f"max={selected.max():.3f}"
    )


def main():
    args = parse_args()
    path = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Topography file not found: {path}")

    print("=" * 88)
    print("flow_finalv1_global GLDAS TOPOGRAPHY PREFLIGHT")
    print(f"File: {path}")
    print(f"Size: {os.path.getsize(path) / (1024 ** 2):.2f} MiB")

    ds = xr.open_dataset(path, decode_cf=True)
    try:
        print("\nDataset:")
        print(ds)
        print("\nVariables:")
        for name, variable in ds.variables.items():
            print(
                f"  {name}: dims={variable.dims}, shape={variable.shape}, "
                f"dtype={variable.dtype}, standard_name={variable.attrs.get('standard_name')!r}, "
                f"units={variable.attrs.get('units')!r}"
            )

        elevation_name = _find_elevation_variable(ds, requested_var=args.variable)
        elevation_da = ds[elevation_name].squeeze(drop=True)
        lat_dim, source_lats, lat_name = _find_spatial_axis(
            ds, elevation_da, "latitude"
        )
        lon_dim, source_lons, lon_name = _find_spatial_axis(
            ds, elevation_da, "longitude"
        )
        print("\nDetected schema:")
        print(f"  elevation variable : {elevation_name}")
        print(f"  elevation dims     : {elevation_da.dims}")
        print(f"  latitude           : variable={lat_name}, dim={lat_dim}, size={source_lats.size}")
        print(f"  longitude          : variable={lon_name}, dim={lon_dim}, size={source_lons.size}")
        print(
            f"  latitude range     : {source_lats.min():.3f} .. {source_lats.max():.3f} "
            f"({'ascending' if source_lats[-1] > source_lats[0] else 'descending'})"
        )
        print(
            f"  longitude range    : {source_lons.min():.3f} .. {source_lons.max():.3f}"
        )
        print(f"  raw elevation      : {finite_summary(elevation_da.values)}")
    finally:
        ds.close()

    target_lats = np.arange(-90.0, 91.0, 1.0, dtype=np.float32)
    target_lons = np.arange(0.0, 360.0, 1.0, dtype=np.float32)
    elevation_norm, metadata, elevation_m = _load_elevation(
        {
            "data_dir": os.path.dirname(path),
            "elevation_file": path,
            "elevation_variable": args.variable,
            "require_elevation_file": True,
        },
        target_lats,
        target_lons,
    )

    print("\nflow_finalv1_global target-grid result:")
    print(f"  expected shape     : (181, 360)")
    print(f"  elevation shape    : {elevation_m.shape}")
    print(f"  normalized shape   : {elevation_norm.shape}")
    print(f"  physical elevation : {finite_summary(elevation_m)}")
    print(f"  normalized         : {finite_summary(elevation_norm)}")
    print(f"  normalization      : {metadata['normalization']} -> [-1, 1]")
    print(f"  normalization min  : {metadata['min_m']:.3f} m")
    print(f"  normalization max  : {metadata['max_m']:.3f} m")
    if elevation_m.shape != (181, 360):
        raise RuntimeError(f"Wrong global shape: {elevation_m.shape}; expected (181, 360)")
    if not np.isfinite(elevation_m).all() or not np.isfinite(elevation_norm).all():
        raise RuntimeError("Topography contains NaN or infinite values after interpolation.")
    if float(np.max(elevation_m)) < 1000.0:
        raise RuntimeError(
            "Maximum elevation is below 1000 m; the global elevation alignment is invalid."
        )

    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extent = [0.0, 359.0, -90.0, 90.0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    physical = axes[0].imshow(
        elevation_m,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="terrain",
    )
    axes[0].set_title("GLDAS elevation on flow_finalv1_global grid (m)")
    normalized = axes[1].imshow(
        elevation_norm,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
    )
    axes[1].set_title("Min-max elevation used by model [-1, 1]")
    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    fig.colorbar(physical, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(normalized, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"flow_finalv1_global topography preflight | {os.path.basename(path)}\n"
        f"variable={metadata['variable']}, lat={metadata['latitude_variable']}, "
        f"lon={metadata['longitude_variable']}",
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPASS: topography is compatible with flow_finalv1_global")
    print(f"Diagnostic plot: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
