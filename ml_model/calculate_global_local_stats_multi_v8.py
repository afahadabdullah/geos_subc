#!/usr/bin/env python3
"""
Build v8 stats for the South Asia local/global multi-target flow model.

The v8 normalization file intentionally mixes two scopes:
- global predictor bounds for variables used by the full-global context encoder
- South Asia 55E-100E, 0N-40N local bounds for the T2M residual target

T2M v8 target is:
    ERA5_T2M - GEOS_ensemble_mean_T2M
"""

import argparse
import gc
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_flow_multi import S2SHybridDataset


DEFAULT_START_YEAR = 1999
DEFAULT_END_YEAR = 2020
DEFAULT_TARGET_DOMAIN_BOUNDS = {
    "label": "South Asia 55E-100E 0N-40N",
    "lat_min": 0.0,
    "lat_max": 40.0,
    "lon_min": 55.0,
    "lon_max": 100.0,
}


def update_bounds(bounds, key, tensor):
    valid = tensor[torch.isfinite(tensor)]
    if valid.numel() == 0:
        return
    b_min = float(valid.min().item())
    b_max = float(valid.max().item())
    if b_min < bounds[key]["min"]:
        bounds[key]["min"] = b_min
    if b_max > bounds[key]["max"]:
        bounds[key]["max"] = b_max


def scan_global_predictors(bounds, data_root, start_year, end_year, batch_size):
    print(f"Scanning global predictors and raw absolute fields ({start_year}-{end_year})...")
    dataset = S2SHybridDataset(
        data_root=data_root,
        start_year=start_year,
        end_year=end_year,
        normalize=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {data_root} for {start_year}-{end_year}.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Global stats")):
            x_obs = batch["x_obs"]
            update_bounds(bounds, "sst", x_obs[:, 0:4])
            update_bounds(bounds, "sss", x_obs[:, 4:8].clamp(min=25.0))
            update_bounds(bounds, "sm", x_obs[:, 8:12])
            update_bounds(bounds, "ivt", x_obs[:, 12:16])
            update_bounds(bounds, "z500_zonal_dev", x_obs[:, 16:20])
            update_bounds(bounds, "u250", x_obs[:, 20:24])
            update_bounds(bounds, "mjo", x_obs[:, 24:28])

            x_geos = batch["x_geos"]
            update_bounds(bounds, "geos_pr_raw", x_geos[:, :, 0])
            update_bounds(bounds, "geos_tas_raw", x_geos[:, :, 1])

            target_raw = batch["target_raw"]
            update_bounds(bounds, "target_t2m_raw", target_raw[:, 1])

            if i == 0:
                print("\n--- GLOBAL BATCH 0 DIAGNOSTICS ---")
                print(f"SST        : {x_obs[:, 0:4].min().item():.4f} .. {x_obs[:, 0:4].max().item():.4f}")
                print(f"SSS        : {x_obs[:, 4:8].min().item():.4f} .. {x_obs[:, 4:8].max().item():.4f}")
                print(f"SM         : {x_obs[:, 8:12].min().item():.4f} .. {x_obs[:, 8:12].max().item():.4f}")
                print(f"IVT        : {x_obs[:, 12:16].min().item():.4f} .. {x_obs[:, 12:16].max().item():.4f}")
                print(f"Z500_DEV   : {x_obs[:, 16:20].min().item():.4f} .. {x_obs[:, 16:20].max().item():.4f}")
                print(f"U250       : {x_obs[:, 20:24].min().item():.4f} .. {x_obs[:, 20:24].max().item():.4f}")
                print(f"MJO        : {x_obs[:, 24:28].min().item():.4f} .. {x_obs[:, 24:28].max().item():.4f}")
                print(f"GEOS PR    : {x_geos[:, :, 0].min().item():.4f} .. {x_geos[:, :, 0].max().item():.4f}")
                print(f"GEOS T2M   : {x_geos[:, :, 1].min().item():.4f} .. {x_geos[:, :, 1].max().item():.4f}")
                print(f"TARGET T2M : {target_raw[:, 1].min().item():.4f} .. {target_raw[:, 1].max().item():.4f}")
                print("----------------------------------\n")

            del x_obs, x_geos, target_raw, batch
            gc.collect()


def scan_sa_t2m_residual(bounds, data_root, start_year, end_year, batch_size, target_domain, target_domain_bounds):
    domain_label = target_domain_bounds.get("label", target_domain)
    print(f"Scanning {domain_label} T2M residual target bounds ({start_year}-{end_year})...")
    dataset = S2SHybridDataset(
        data_root=data_root,
        start_year=start_year,
        end_year=end_year,
        normalize=False,
        target_domain=target_domain,
        target_domain_bounds=target_domain_bounds,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No {target_domain} samples found in {data_root} for {start_year}-{end_year}.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="SA T2M residual stats")):
            target_t2m = batch["target_raw"][:, 1]
            geos_ens = batch["geos_ens_raw"]
            lead_idx = batch["lead_idx"].long()
            geos_t2m_all = geos_ens[:, :, 1]
            _, members, _, height, width = geos_t2m_all.shape
            gather_idx = lead_idx.view(-1, 1, 1, 1, 1).expand(-1, members, 1, height, width)
            geos_t2m_lead = geos_t2m_all.gather(2, gather_idx).squeeze(2).mean(dim=1)
            residual = target_t2m - geos_t2m_lead
            update_bounds(bounds, "target_t2m_residual_raw", residual)

            if i == 0:
                print("\n--- SA RESIDUAL BATCH 0 DIAGNOSTICS ---")
                print(f"T2M residual: {residual.min().item():.4f} .. {residual.max().item():.4f}")
                print("---------------------------------------\n")

            del target_t2m, geos_ens, lead_idx, geos_t2m_all, gather_idx, geos_t2m_lead, residual, batch
            gc.collect()


def finalize_bounds(bounds):
    # Robust physical limits retained from the v1 multi stats path.
    bounds["residual_pr_raw"] = {"min": -100.0, "max": 100.0}
    bounds["geos_pr_raw"] = {"min": 0.0, "max": 100.0}
    bounds["z500_zonal_dev"] = {"min": -5000.0, "max": 5000.0}
    bounds["mjo"] = {"min": -50.0, "max": 50.0}

    # Backward-compatible aliases used by existing dataset/train code.
    bounds["geos_raw"] = bounds["geos_pr_raw"]
    bounds["residual_raw"] = bounds["residual_pr_raw"]

    for key, value in bounds.items():
        if not isinstance(value, dict):
            continue
        vmin = float(value["min"])
        vmax = float(value["max"])
        if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin < vmax):
            raise RuntimeError(f"Invalid bounds for {key}: {vmin}..{vmax}")


def calculate_stats(data_root, out_path, start_year, end_year, batch_size, target_domain, target_domain_bounds):
    bounds = {
        "sst": {"min": float("inf"), "max": float("-inf")},
        "sss": {"min": float("inf"), "max": float("-inf")},
        "sm": {"min": float("inf"), "max": float("-inf")},
        "ivt": {"min": float("inf"), "max": float("-inf")},
        "u250": {"min": float("inf"), "max": float("-inf")},
        "z500_zonal_dev": {"min": float("inf"), "max": float("-inf")},
        "mjo": {"min": float("inf"), "max": float("-inf")},
        "geos_pr_raw": {"min": float("inf"), "max": float("-inf")},
        "geos_tas_raw": {"min": float("inf"), "max": float("-inf")},
        "target_t2m_raw": {"min": float("inf"), "max": float("-inf")},
        "target_t2m_residual_raw": {"min": float("inf"), "max": float("-inf")},
        "residual_pr_raw": {"min": float("inf"), "max": float("-inf")},
    }

    scan_global_predictors(bounds, data_root, start_year, end_year, batch_size)
    scan_sa_t2m_residual(bounds, data_root, start_year, end_year, batch_size, target_domain, target_domain_bounds)
    finalize_bounds(bounds)

    print("\n==================================")
    print(f"Calculated v8 Global/Local Bounds ({start_year}-{end_year})")
    print("==================================")
    for key, value in bounds.items():
        if isinstance(value, dict):
            print(f"{key.upper():<24} | Min: {value['min']:>12.4f} | Max: {value['max']:>12.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(bounds, out_path)
    print(f"\nSaved v8 stats to {out_path}!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DATA_DIR_OVERRIDE", "/scratch/11353/afahad/geossub/dataprocess"),
    )
    parser.add_argument("--out", type=str, default="ml_model/v8_sa_55e100e_0n40n_global_local_stats.pt")
    parser.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_domain", type=str, default="south_asia")
    parser.add_argument("--lat_min", type=float, default=DEFAULT_TARGET_DOMAIN_BOUNDS["lat_min"])
    parser.add_argument("--lat_max", type=float, default=DEFAULT_TARGET_DOMAIN_BOUNDS["lat_max"])
    parser.add_argument("--lon_min", type=float, default=DEFAULT_TARGET_DOMAIN_BOUNDS["lon_min"])
    parser.add_argument("--lon_max", type=float, default=DEFAULT_TARGET_DOMAIN_BOUNDS["lon_max"])
    parser.add_argument("--domain_label", type=str, default=DEFAULT_TARGET_DOMAIN_BOUNDS["label"])
    args = parser.parse_args()
    target_domain_bounds = {
        "label": args.domain_label,
        "lat_min": args.lat_min,
        "lat_max": args.lat_max,
        "lon_min": args.lon_min,
        "lon_max": args.lon_max,
    }

    calculate_stats(
        data_root=args.data_root,
        out_path=args.out,
        start_year=args.start_year,
        end_year=args.end_year,
        batch_size=args.batch_size,
        target_domain=args.target_domain,
        target_domain_bounds=target_domain_bounds,
    )


if __name__ == "__main__":
    main()
