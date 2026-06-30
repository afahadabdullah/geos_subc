#!/usr/bin/env python3
"""
Regional post-processing for flow_finalv1_global matrix evaluation.

This script uses the saved matrix_spatial_metrics.nc from
evaluate_matrix_suite_flow_finalv1_global.py. It does not reread forecast
ensembles. Regional means are therefore area/sample-count weighted means of
the saved gridpoint metric maps. CRPS/MAE means are directly interpretable;
RMSE/BSS are regional means of gridpoint RMSE/BSS, not recomputed from raw
squared-error/Brier sums.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from evaluate_matrix_suite_flow_finalv1_global import (
    GROUP_TYPES,
    LEADS,
    MONTHS,
    SEASONS,
    SUBSETS,
    VARIABLES,
    add_map_overlays,
    configure_map_context,
    make_map_subplots,
)


REGIONS = [
    {
        "id": "conus",
        "label": "CONUS / USA",
        "bbox": (-125.0, -66.0, 24.0, 50.0),
        "description": "Contiguous United States bounding box.",
    },
    {
        "id": "bangladesh",
        "label": "Bangladesh",
        "countries": ["Bangladesh"],
        "bbox": (88.0, 93.0, 20.0, 27.0),
    },
    {
        "id": "india",
        "label": "India",
        "countries": ["India"],
        "bbox": (68.0, 98.0, 6.0, 37.5),
    },
    {
        "id": "pakistan",
        "label": "Pakistan",
        "countries": ["Pakistan"],
        "bbox": (60.0, 78.0, 23.0, 38.0),
    },
    {
        "id": "europe",
        "label": "Europe",
        "continent": "Europe",
        "bbox": (-25.0, 45.0, 34.0, 72.0),
    },
    {
        "id": "australia",
        "label": "Australia",
        "countries": ["Australia"],
        "bbox": (112.0, 154.0, -44.0, -10.0),
    },
    {
        "id": "africa",
        "label": "Africa",
        "continent": "Africa",
        "bbox": (-20.0, 55.0, -35.0, 38.0),
    },
    {
        "id": "south_america",
        "label": "South America",
        "continent": "South America",
        "bbox": (-82.0, -34.0, -56.0, 13.0),
    },
    {
        "id": "amazon_basin",
        "label": "Amazon Basin",
        "bbox": (-80.0, -45.0, -20.0, 10.0),
        "description": "Broad Amazon heat/drought/fire-prone land region.",
    },
    {
        "id": "sahel_west_africa",
        "label": "Sahel / West Africa",
        "bbox": (-18.0, 35.0, 8.0, 20.0),
    },
    {
        "id": "east_africa_horn",
        "label": "East Africa / Horn",
        "bbox": (25.0, 52.0, -12.0, 16.0),
    },
    {
        "id": "southern_africa",
        "label": "Southern Africa",
        "bbox": (10.0, 40.0, -35.0, -10.0),
    },
    {
        "id": "mediterranean_middle_east",
        "label": "Mediterranean / Middle East",
        "bbox": (-10.0, 60.0, 25.0, 45.0),
    },
    {
        "id": "southeast_asia",
        "label": "Southeast Asia",
        "bbox": (92.0, 125.0, -10.0, 25.0),
    },
    {
        "id": "east_asia",
        "label": "East Asia",
        "bbox": (105.0, 145.0, 20.0, 50.0),
    },
    {
        "id": "central_america_caribbean",
        "label": "Central America / Caribbean",
        "bbox": (-100.0, -60.0, 5.0, 25.0),
    },
]


DEFAULT_METRICS = [
    "model_rmse",
    "geos_rmse",
    "rmse_skill_pct",
    "model_mae",
    "geos_mae",
    "mae_skill_pct",
    "model_crps",
    "geos_crps",
    "crps_skill_pct",
    "model_corr",
    "geos_corr",
    "corr_diff",
    "model_bss",
    "geos_bss",
    "bss_diff",
    "model_calibrated_bss",
    "geos_calibrated_bss",
    "calibrated_bss_diff",
    "model_spread",
    "geos_spread",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Regional matrix evaluation from saved spatial metric NetCDF.")
    parser.add_argument(
        "--matrix_spatial_file",
        type=str,
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "matrix_eval_global_2021_2023_land_obsclim_chunked/matrix_spatial_metrics.nc"
        ),
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=(
            "ml_output_flow_finalv1_global_noisectx_t2mres/"
            "matrix_eval_global_2021_2023_land_obsclim_chunked/matrix_eval_metadata.json"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ml_output_flow_finalv1_global_noisectx_t2mres/regional_matrix_eval_global_2021_2023_land_obsclim",
    )
    parser.add_argument("--land_mask_file", type=str, default="ml_model/land_ocean_mask_v6.pt")
    parser.add_argument("--regions", type=str, default="all", help="Comma-separated region ids, or all.")
    parser.add_argument("--variables", type=str, default="pr,t2m")
    parser.add_argument("--subsets", type=str, default="all_data,extreme_events")
    parser.add_argument("--group_types", type=str, default="valid_season_lead,valid_month_lead")
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--mask_source", choices=("auto", "natural_earth", "box"), default="auto")
    parser.add_argument("--make_maps", action="store_true")
    parser.add_argument(
        "--plot_metrics",
        type=str,
        default="crps_skill_pct,rmse_skill_pct,calibrated_bss_diff",
    )
    parser.add_argument("--plot_group_type", choices=GROUP_TYPES, default="valid_season_lead")
    parser.add_argument("--map_features", choices=("auto", "cartopy", "plain"), default="auto")
    parser.add_argument("--county_boundaries", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_list(text, valid=None):
    items = [item.strip() for item in str(text or "").split(",") if item.strip()]
    if valid is not None:
        bad = [item for item in items if item not in valid]
        if bad:
            raise ValueError(f"Unknown values {bad}; expected subset of {sorted(valid)}")
    return items


def selected_regions(text):
    region_by_id = {region["id"]: region for region in REGIONS}
    if str(text).strip().lower() == "all":
        return REGIONS
    ids = parse_list(text, valid=region_by_id)
    return [region_by_id[region_id] for region_id in ids]


def lon_to_180(lons):
    lons = np.asarray(lons, dtype=np.float64)
    return ((lons + 180.0) % 360.0) - 180.0


def bbox_mask(lons, lats, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    lon180 = lon_to_180(lons)
    lon2d, lat2d = np.meshgrid(lon180, np.asarray(lats, dtype=np.float64))
    if lon_min <= lon_max:
        lon_ok = (lon2d >= lon_min) & (lon2d <= lon_max)
    else:
        lon_ok = (lon2d >= lon_min) | (lon2d <= lon_max)
    return lon_ok & (lat2d >= lat_min) & (lat2d <= lat_max)


def load_land_mask(path, shape):
    if not path:
        print("🌍 No land mask supplied; regional means use region mask only.")
        return np.ones(shape, dtype=bool), None
    if not os.path.exists(path):
        print(f"⚠️ Land mask not found ({path}); using all grid points. If matrix eval was land-only, sample_count still masks ocean.")
        return np.ones(shape, dtype=bool), None
    import torch

    cached = torch.load(path, map_location="cpu", weights_only=True)
    if "is_land" in cached:
        land = np.asarray(cached["is_land"], dtype=bool).squeeze()
    elif "land_mask" in cached:
        land = np.asarray(cached["land_mask"], dtype=bool).squeeze()
    else:
        raise ValueError(f"{path} is missing is_land or land_mask")
    if land.shape != shape:
        raise ValueError(f"Land mask shape {land.shape} does not match spatial metric grid {shape}")
    return land, os.path.abspath(path)


def load_country_records(mask_source):
    if mask_source == "box":
        return None
    try:
        import cartopy.io.shapereader as shapereader
        from cartopy.io import DownloadWarning
    except Exception as exc:
        if mask_source == "natural_earth":
            raise
        print(f"⚠️ Cartopy Natural Earth unavailable for exact regional masks ({exc}); falling back to boxes.")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DownloadWarning)
            path = shapereader.natural_earth(
                resolution="50m",
                category="cultural",
                name="admin_0_countries",
            )
        return list(shapereader.Reader(path).records())
    except Exception as exc:
        if mask_source == "natural_earth":
            raise
        print(f"⚠️ Natural Earth country polygons are not cached ({exc}); falling back to regional boxes.")
        return None


def record_matches_region(record, region):
    attrs = record.attributes
    names = {
        str(attrs.get("ADMIN", "")),
        str(attrs.get("NAME", "")),
        str(attrs.get("NAME_LONG", "")),
        str(attrs.get("SOVEREIGNT", "")),
    }
    if "countries" in region and names.intersection(set(region["countries"])):
        return True
    if "continent" in region and str(attrs.get("CONTINENT", "")) == str(region["continent"]):
        return True
    return False


def geometry_mask(records, region, lons, lats):
    if records is None or ("countries" not in region and "continent" not in region):
        return None
    geoms = [record.geometry for record in records if record_matches_region(record, region)]
    if not geoms:
        return None
    try:
        from shapely.ops import unary_union

        geom = unary_union(geoms)
    except Exception:
        geom = geoms[0] if len(geoms) == 1 else None
    if geom is None:
        return None

    lon180 = lon_to_180(lons)
    lon2d, lat2d = np.meshgrid(lon180, np.asarray(lats, dtype=np.float64))
    try:
        from shapely import contains_xy

        return contains_xy(geom, lon2d, lat2d)
    except Exception:
        pass
    try:
        from shapely import vectorized

        return vectorized.contains(geom, lon2d, lat2d)
    except Exception:
        pass
    return None


def build_region_masks(regions, lons, lats, land_mask, mask_source):
    records = load_country_records(mask_source)
    masks = {}
    metadata = []
    for region in regions:
        box = bbox_mask(lons, lats, region["bbox"])
        geom_mask = geometry_mask(records, region, lons, lats)
        source = "bbox"
        if geom_mask is not None:
            mask = geom_mask & box
            source = "natural_earth+bbox"
        else:
            mask = box
        mask = mask & land_mask
        masks[region["id"]] = mask.astype(bool)
        metadata.append(
            {
                "region": region["id"],
                "label": region["label"],
                "bbox": region["bbox"],
                "mask_source": source,
                "land_grid_points": int(mask.sum()),
            }
        )
        print(f"🌍 Region {region['id']}: {int(mask.sum())} land grid points ({source})")
    return masks, pd.DataFrame(metadata)


def area_weights(lats):
    weights = np.cos(np.deg2rad(np.asarray(lats, dtype=np.float64)))
    return np.clip(weights, 0.0, None)[:, None]


def weighted_mean(field, weights):
    field = np.asarray(field, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(field) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    return float(np.sum(field[finite] * weights[finite]) / np.sum(weights[finite]))


def skill_pct(model, geos):
    if not np.isfinite(model) or not np.isfinite(geos) or abs(geos) <= 1e-12:
        return np.nan
    return float(100.0 * (1.0 - model / geos))


def weighted_average_series(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return np.nan
    return float(np.average(values[finite], weights=weights[finite]))


def build_regional_summary(ds, regions, masks, variables, subsets, group_types, metrics):
    rows = []
    lats = ds["lat"].values
    base_area = area_weights(lats)
    group_values_by_type = {"valid_season_lead": SEASONS, "valid_month_lead": MONTHS}
    available_metrics = set(ds.data_vars)
    metrics = [metric for metric in metrics if metric in available_metrics]
    for region in regions:
        region_id = region["id"]
        region_mask = masks[region_id]
        region_area = base_area * region_mask.astype(np.float64)
        for variable in variables:
            for subset in subsets:
                for group_type in group_types:
                    for group_value in group_values_by_type[group_type]:
                        for lead in LEADS:
                            selector = {
                                "subset": subset,
                                "variable": variable,
                                "group_type": group_type,
                                "group_value": group_value,
                                "lead": int(lead),
                            }
                            sample = ds["sample_count"].sel(**selector).values.astype(np.float64, copy=False)
                            weights = region_area * np.where(np.isfinite(sample), sample, 0.0)
                            weight_sum = float(np.sum(weights))
                            row = {
                                "region": region_id,
                                "region_label": region["label"],
                                "variable": variable,
                                "subset": subset,
                                "group_type": group_type,
                                "group_value": group_value,
                                "lead": int(lead),
                                "lead_label": f"week{int(lead)}",
                                "land_grid_points": int(region_mask.sum()),
                                "valid_grid_points": int(np.sum(region_mask & np.isfinite(sample) & (sample > 0))),
                                "effective_weight_sum": weight_sum,
                                "sample_count_area_mean": weighted_mean(sample, region_area),
                            }
                            for metric in metrics:
                                if metric == "sample_count":
                                    continue
                                row[metric] = weighted_mean(ds[metric].sel(**selector).values, weights)

                            if "model_crps" in row and "geos_crps" in row:
                                row["crps_skill_pct"] = skill_pct(row["model_crps"], row["geos_crps"])
                            if "model_rmse" in row and "geos_rmse" in row:
                                row["rmse_skill_pct"] = skill_pct(row["model_rmse"], row["geos_rmse"])
                            if "model_mae" in row and "geos_mae" in row:
                                row["mae_skill_pct"] = skill_pct(row["model_mae"], row["geos_mae"])
                            if "model_corr" in row and "geos_corr" in row:
                                row["corr_diff"] = row["model_corr"] - row["geos_corr"]
                            if "model_bss" in row and "geos_bss" in row:
                                row["bss_diff"] = row["model_bss"] - row["geos_bss"]
                            if "model_calibrated_bss" in row and "geos_calibrated_bss" in row:
                                row["calibrated_bss_diff"] = row["model_calibrated_bss"] - row["geos_calibrated_bss"]
                            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_regional_summary(summary):
    rows = []
    for key, group in summary[summary["group_type"].eq("valid_season_lead")].groupby(["region", "region_label", "variable", "subset"]):
        region, label, variable, subset = key
        weights = group["effective_weight_sum"].astype(float)
        row = {
            "region": region,
            "region_label": label,
            "variable": variable,
            "subset": subset,
            "n_groups": int(len(group)),
            "effective_weight_sum": float(weights.sum()),
        }
        for col in [
            "model_crps",
            "geos_crps",
            "model_rmse",
            "geos_rmse",
            "model_mae",
            "geos_mae",
            "model_corr",
            "geos_corr",
            "model_calibrated_bss",
            "geos_calibrated_bss",
        ]:
            if col in group:
                row[col] = weighted_average_series(group[col], weights)
        row["crps_skill_pct"] = skill_pct(row.get("model_crps", np.nan), row.get("geos_crps", np.nan))
        row["rmse_skill_pct"] = skill_pct(row.get("model_rmse", np.nan), row.get("geos_rmse", np.nan))
        row["mae_skill_pct"] = skill_pct(row.get("model_mae", np.nan), row.get("geos_mae", np.nan))
        row["corr_diff"] = row.get("model_corr", np.nan) - row.get("geos_corr", np.nan)
        row["calibrated_bss_diff"] = row.get("model_calibrated_bss", np.nan) - row.get("geos_calibrated_bss", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["region", "variable", "subset"]).reset_index(drop=True)


def write_lead_season_table(summary, out_dir):
    table = summary[summary["group_type"].eq("valid_season_lead")].copy()
    table = table.rename(columns={"group_value": "season", "lead": "lead_week"})
    cols = [
        "region",
        "region_label",
        "variable",
        "subset",
        "season",
        "lead_week",
        "lead_label",
        "land_grid_points",
        "valid_grid_points",
        "effective_weight_sum",
        "geos_crps",
        "model_crps",
        "crps_skill_pct",
        "geos_rmse",
        "model_rmse",
        "rmse_skill_pct",
        "geos_calibrated_bss",
        "model_calibrated_bss",
        "calibrated_bss_diff",
    ]
    cols = [col for col in cols if col in table.columns]
    table = table[cols].sort_values(["region", "variable", "subset", "season", "lead_week"])
    path = os.path.join(out_dir, "regional_lead_season_skill_table.csv")
    table.to_csv(path, index=False, float_format="%.6f")
    print(f"✅ Wrote regional lead-season table: {path}")
    return table


def plot_scalar_heatmaps(table, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots", "regional_heatmaps")
    os.makedirs(plot_dir, exist_ok=True)
    metrics = ["crps_skill_pct", "rmse_skill_pct", "calibrated_bss_diff"]
    for (region, variable, subset), group in table.groupby(["region", "variable", "subset"]):
        for metric in metrics:
            if metric not in group:
                continue
            matrix = np.full((len(SEASONS), len(LEADS)), np.nan, dtype=np.float64)
            for i, season in enumerate(SEASONS):
                for j, lead in enumerate(LEADS):
                    match = group[(group["season"] == season) & (group["lead_week"] == lead)]
                    if not match.empty:
                        matrix[i, j] = float(match.iloc[0][metric])
            finite = matrix[np.isfinite(matrix)]
            vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0
            vmax = max(vmax, 1e-6)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            mesh = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(LEADS)))
            ax.set_xticklabels([f"week{lead}" for lead in LEADS])
            ax.set_yticks(np.arange(len(SEASONS)))
            ax.set_yticklabels(SEASONS)
            ax.set_title(f"{region} | {variable} | {subset} | {metric}")
            fig.colorbar(mesh, ax=ax, shrink=0.8)
            fig.tight_layout()
            out_path = os.path.join(plot_dir, f"{region}_{variable}_{subset}_{metric}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"✅ Wrote regional scalar heatmaps under: {plot_dir}")


def sorted_plot_grid(lons, lats, field, mask, bbox, pad=2.0):
    lons180 = lon_to_180(lons)
    order = np.argsort(lons180)
    lon_sorted = lons180[order]
    field_sorted = np.asarray(field)[:, order]
    mask_sorted = np.asarray(mask)[:, order]
    field_sorted = np.where(mask_sorted, field_sorted, np.nan)
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_keep = (lon_sorted >= lon_min - pad) & (lon_sorted <= lon_max + pad)
    lat_vals = np.asarray(lats, dtype=np.float64)
    lat_keep = (lat_vals >= lat_min - pad) & (lat_vals <= lat_max + pad)
    if not lon_keep.any() or not lat_keep.any():
        return lon_sorted, lat_vals, field_sorted
    return lon_sorted[lon_keep], lat_vals[lat_keep], field_sorted[np.ix_(lat_keep, lon_keep)]


def plot_region_spatial_grid(ds, region, mask, variable, subset, group_type, metric, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = SEASONS if group_type == "valid_season_lead" else MONTHS
    lats = ds["lat"].values
    lons = ds["lon"].values
    values = []
    prepared = {}
    for group_value in rows:
        for lead in LEADS:
            arr = ds[metric].sel(
                subset=subset,
                variable=variable,
                group_type=group_type,
                group_value=group_value,
                lead=lead,
            ).values
            plot_lons, plot_lats, plot_arr = sorted_plot_grid(lons, lats, arr, mask, region["bbox"])
            prepared[(group_value, lead)] = (plot_lons, plot_lats, plot_arr)
            if np.isfinite(plot_arr).any():
                values.append(plot_arr[np.isfinite(plot_arr)])
    combined = np.concatenate(values) if values else np.array([])
    center_zero = metric.endswith("_diff") or metric.endswith("_skill_pct") or "bias" in metric
    if combined.size:
        if center_zero:
            vmax = max(float(np.nanpercentile(np.abs(combined), 95)), 1e-6)
            vmin = -vmax
        else:
            vmin, vmax = np.nanpercentile(combined, [5, 95])
    else:
        vmin, vmax = (-1.0, 1.0) if center_zero else (0.0, 1.0)

    panel_width = 4.5
    panel_height = 2.4
    fig, axes = make_map_subplots(
        len(rows),
        len(LEADS),
        figsize=(panel_width * len(LEADS), max(panel_height * len(rows), 6)),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = "RdBu" if center_zero else "viridis"
    last_mesh = None
    for r_idx, group_value in enumerate(rows):
        for c_idx, lead in enumerate(LEADS):
            ax = axes[r_idx, c_idx]
            plot_lons, plot_lats, plot_arr = prepared[(group_value, lead)]
            mesh_kwargs = {}
            from evaluate_matrix_suite_flow_finalv1_global import MAP_CONTEXT

            if MAP_CONTEXT["enabled"]:
                mesh_kwargs["transform"] = MAP_CONTEXT["data_crs"]
            last_mesh = ax.pcolormesh(
                plot_lons,
                plot_lats,
                plot_arr,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
                **mesh_kwargs,
            )
            add_map_overlays(ax, plot_lons, plot_lats)
            ax.set_title(f"{group_value} week{lead}", fontsize=8)
            ax.set_xlabel("lon", fontsize=7)
            ax.set_ylabel("lat", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{region['label']} | {subset} | {variable} | {metric}", fontsize=13)
    if last_mesh is not None:
        fig.colorbar(last_mesh, ax=axes.ravel().tolist(), shrink=0.75)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_region_maps(ds, regions, masks, variables, subsets, plot_metrics, group_type, out_dir):
    plot_dir = os.path.join(out_dir, "plots", "regional_maps")
    os.makedirs(plot_dir, exist_ok=True)
    for region in regions:
        mask = masks[region["id"]]
        for variable in variables:
            for subset in subsets:
                for metric in plot_metrics:
                    if metric not in ds:
                        continue
                    out_path = os.path.join(plot_dir, f"{region['id']}_{variable}_{subset}_{group_type}_{metric}.png")
                    plot_region_spatial_grid(ds, region, mask, variable, subset, group_type, metric, out_path)
    print(f"✅ Wrote regional maps under: {plot_dir}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = xr.open_dataset(args.matrix_spatial_file)
    try:
        regions = selected_regions(args.regions)
        variables = parse_list(args.variables, valid=VARIABLES)
        subsets = parse_list(args.subsets, valid=SUBSETS)
        group_types = parse_list(args.group_types, valid=GROUP_TYPES)
        metrics = parse_list(args.metrics)
        plot_metrics = parse_list(args.plot_metrics)
        shape = (int(ds.sizes["lat"]), int(ds.sizes["lon"]))
        land_mask, land_source = load_land_mask(args.land_mask_file, shape)
        masks, region_meta = build_region_masks(regions, ds["lon"].values, ds["lat"].values, land_mask, args.mask_source)

        summary_path = os.path.join(args.out_dir, "regional_summary_metrics.csv")
        lead_table_path = os.path.join(args.out_dir, "regional_lead_season_skill_table.csv")
        overall_path = os.path.join(args.out_dir, "regional_overall_skill_table.csv")
        region_meta_path = os.path.join(args.out_dir, "regional_mask_metadata.csv")
        metadata_path = os.path.join(args.out_dir, "regional_eval_metadata.json")

        if os.path.exists(summary_path) and not args.overwrite:
            print(f"✅ Existing regional summary found: {summary_path}")
            summary = pd.read_csv(summary_path)
        else:
            summary = build_regional_summary(ds, regions, masks, variables, subsets, group_types, metrics)
            summary.to_csv(summary_path, index=False, float_format="%.6f")
            print(f"✅ Wrote regional summary: {summary_path}")

        lead_table = write_lead_season_table(summary, args.out_dir)
        overall = aggregate_regional_summary(summary)
        overall.to_csv(overall_path, index=False, float_format="%.6f")
        region_meta.to_csv(region_meta_path, index=False)
        print(f"✅ Wrote regional overall skill table: {overall_path}")
        print(f"✅ Wrote regional mask metadata: {region_meta_path}")

        metadata = {
            "matrix_spatial_file": os.path.abspath(args.matrix_spatial_file),
            "source_metadata_file": os.path.abspath(args.metadata_file) if args.metadata_file else None,
            "land_mask_file": land_source,
            "regions": [region["id"] for region in regions],
            "region_mean_note": (
                "Regional means are area/sample-count weighted means of saved gridpoint metric maps. "
                "RMSE/BSS are not recomputed from raw SSE/Brier sums."
            ),
            "variables": variables,
            "subsets": subsets,
            "group_types": group_types,
            "metrics": metrics,
            "plot_metrics": plot_metrics,
            "plot_group_type": args.plot_group_type,
        }
        if args.metadata_file and os.path.exists(args.metadata_file):
            with open(args.metadata_file) as f:
                metadata["source_matrix_metadata"] = json.load(f)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Wrote regional metadata: {metadata_path}")

        if args.make_maps:
            map_args = argparse.Namespace(map_features=args.map_features, county_boundaries=args.county_boundaries)
            configure_map_context(map_args)
            plot_scalar_heatmaps(lead_table, args.out_dir)
            make_region_maps(ds, regions, masks, variables, subsets, plot_metrics, args.plot_group_type, args.out_dir)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
