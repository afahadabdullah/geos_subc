#!/usr/bin/env python3
"""
Audit the South Asia multi-v4 T2M data path.

This checks the raw GEOS/ERA5 T2M relationship and the dataset normalization
round trip before involving the trained model. It is meant to catch silent
unit, lead, orientation, missing-data, and scaling mistakes.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from dataset_flow_multi import S2SHybridDataset, get_target_domain_coords
from train_flow_multiv4 import compute_crps, compute_rmse, get_area_weights

EXPECTED_OUTPUT_DIR = "ml_output_flowmulti_v4_south_asia_global_context"
EXPECTED_LOCAL_OBS = ["sm", "mjo"]
EXPECTED_GLOBAL_CONTEXT = ["sst", "sss", "ivt", "z500_zonal_dev", "u250"]
T2M_MIN = 200.0
T2M_MAX = 320.0


def parse_args():
    parser = argparse.ArgumentParser(description="Audit SA multi-v4 T2M dataset/evaluation inputs.")
    parser.add_argument("--config", default="ml_model/config_flow_multiv4.yaml")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--full-year", action="store_true", help="Use every weekly init date.")
    parser.add_argument("--batch-limit", type=int, default=0, help="Max init batches; <=0 means all.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config):
    output_dir = config.get("output_dir")
    if output_dir != EXPECTED_OUTPUT_DIR:
        raise ValueError(f"Expected output_dir={EXPECTED_OUTPUT_DIR}, got {output_dir}")
    if config.get("target_domain") != "south_asia":
        raise ValueError(f"Expected target_domain=south_asia, got {config.get('target_domain')}")
    local_obs = list(config.get("local_obs_variables") or [])
    global_context = list(config.get("global_context_variables") or [])
    if local_obs != EXPECTED_LOCAL_OBS:
        raise ValueError(f"Expected local_obs_variables={EXPECTED_LOCAL_OBS}, got {local_obs}")
    if global_context != EXPECTED_GLOBAL_CONTEXT:
        raise ValueError(f"Expected global_context_variables={EXPECTED_GLOBAL_CONTEXT}, got {global_context}")


def weighted_mean(field, area_weights):
    weights = area_weights.expand_as(field)
    mask = torch.isfinite(field)
    if not mask.any():
        return float("nan")
    return ((torch.where(mask, field, torch.zeros_like(field)) * torch.where(mask, weights, torch.zeros_like(weights))).sum() /
            (torch.where(mask, weights, torch.zeros_like(weights)).sum() + 1e-8)).item()


def summarize(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = []
    for group_key in sorted(groups):
        values = groups[group_key]
        out.append({
            key: group_key,
            "n": len(values),
            "geos_rmse": float(np.mean([r["geos_rmse"] for r in values])),
            "geos_crps": float(np.mean([r["geos_crps"] for r in values])),
            "geos_bias": float(np.mean([r["geos_bias"] for r in values])),
            "target_mean": float(np.mean([r["target_mean"] for r in values])),
            "geos_mean": float(np.mean([r["geos_mean"] for r in values])),
            "target_norm_min": float(np.min([r["target_norm_min"] for r in values])),
            "target_norm_max": float(np.max([r["target_norm_max"] for r in values])),
            "target_recon_max_abs_err": float(np.max([r["target_recon_max_abs_err"] for r in values])),
            "geos_recon_max_abs_err": float(np.max([r["geos_recon_max_abs_err"] for r in values])),
        })
    return out


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    if "DATA_DIR_OVERRIDE" in os.environ:
        config["data_dir"] = os.environ["DATA_DIR_OVERRIDE"]
    validate_config(config)

    output_dir = args.output_dir or os.path.join(config["output_dir"], "t2m_audit")
    dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year,
        end_year=args.year,
        normalize=True,
        preload=False,
        stats_file=config.get("stats_file", "v1_multi_global_stats.pt"),
        subsample_monthly=not args.full_year,
        target_domain=config.get("target_domain"),
        target_domain_bounds=config.get("target_domain_bounds"),
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

    lats, _ = get_target_domain_coords(config.get("target_domain"), config.get("target_domain_bounds"))
    area_weights = get_area_weights(lats, torch.device("cpu"))

    rows = []
    batch_limit = None if args.batch_limit <= 0 else args.batch_limit
    for batch_idx, batch in enumerate(loader):
        if batch_limit is not None and batch_idx >= batch_limit:
            break
        if batch["y_target"].shape[0] != 4:
            raise ValueError(f"Expected one init per batch with 4 leads, got batch size {batch['y_target'].shape[0]}")

        init_date = f"{int(batch['year'][0]):04d}-{int(batch['month'][0]):02d}-{int(batch['day'][0]):02d}"
        target_full = batch["target_raw_full"][0, 1].float()  # [L,H,W]
        geos_ens = batch["geos_ens_raw"][0, :, 1].float()  # [M,L,H,W]
        geos_mean = geos_ens.mean(dim=0)

        target_norm = batch["y_target"][:, 1].float()  # [L,H,W]
        target_recon = ((target_norm.clamp(-1.0, 1.0) + 1.0) / 2.0) * (T2M_MAX - T2M_MIN) + T2M_MIN

        geos_norm = batch["x_geos"][0, 0, 1].float()  # [L,H,W]
        geos_recon = ((geos_norm.clamp(-1.0, 1.0) + 1.0) / 2.0) * (T2M_MAX - T2M_MIN) + T2M_MIN

        for lead in range(4):
            target = target_full[lead:lead + 1]
            geos = geos_mean[lead:lead + 1]
            geos_members = geos_ens[:, lead:lead + 1]
            rows.append({
                "init_date": init_date,
                "month": int(batch["month"][0]),
                "lead": lead + 1,
                "target_min": float(torch.nanmin(target).item()),
                "target_max": float(torch.nanmax(target).item()),
                "target_mean": weighted_mean(target, area_weights),
                "geos_min": float(torch.nanmin(geos).item()),
                "geos_max": float(torch.nanmax(geos).item()),
                "geos_mean": weighted_mean(geos, area_weights),
                "geos_bias": weighted_mean(geos - target, area_weights),
                "geos_rmse": compute_rmse(geos, target, area_weights),
                "geos_crps": compute_crps(geos_members.unsqueeze(1), target.unsqueeze(0), area_weights),
                "target_norm_min": float(torch.nanmin(target_norm[lead]).item()),
                "target_norm_max": float(torch.nanmax(target_norm[lead]).item()),
                "target_recon_max_abs_err": float(torch.nanmax(torch.abs(target_recon[lead] - target_full[lead])).item()),
                "geos_norm_min": float(torch.nanmin(geos_norm[lead]).item()),
                "geos_norm_max": float(torch.nanmax(geos_norm[lead]).item()),
                "geos_recon_max_abs_err": float(torch.nanmax(torch.abs(geos_recon[lead] - geos_mean[lead])).item()),
            })

    if not rows:
        raise RuntimeError("No T2M audit rows were produced.")

    row_fields = list(rows[0].keys())
    init_csv = os.path.join(output_dir, f"t2m_audit_{args.year}_{'full_year' if args.full_year else 'monthly'}_by_init.csv")
    write_csv(init_csv, rows, row_fields)

    lead_rows = summarize(rows, "lead")
    month_rows = summarize(rows, "month")
    summary_fields = list(lead_rows[0].keys())
    lead_csv = os.path.join(output_dir, f"t2m_audit_{args.year}_{'full_year' if args.full_year else 'monthly'}_by_lead.csv")
    month_csv = os.path.join(output_dir, f"t2m_audit_{args.year}_{'full_year' if args.full_year else 'monthly'}_by_month.csv")
    write_csv(lead_csv, lead_rows, summary_fields)
    write_csv(month_csv, month_rows, list(month_rows[0].keys()))

    print("\nT2M audit complete")
    print(f"  Config    : {args.config}")
    print(f"  Data dir  : {config['data_dir']}")
    print(f"  Sampling  : {'full weekly year' if args.full_year else 'monthly subset'}")
    print(f"  Init dates: {len(set(r['init_date'] for r in rows))}")
    print(f"  Rows      : {len(rows)}")
    print(f"  By init   : {init_csv}")
    print(f"  By lead   : {lead_csv}")
    print(f"  By month  : {month_csv}")
    print("\nLead summary:")
    for row in lead_rows:
        print(
            f"  Lead {row['lead']}: GEOS RMSE={row['geos_rmse']:.3f}, "
            f"CRPS={row['geos_crps']:.3f}, bias={row['geos_bias']:+.3f}, "
            f"target_norm=[{row['target_norm_min']:.3f},{row['target_norm_max']:.3f}], "
            f"target_recon_err={row['target_recon_max_abs_err']:.3e}, "
            f"geos_recon_err={row['geos_recon_max_abs_err']:.3e}"
        )


if __name__ == "__main__":
    main()
