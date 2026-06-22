#!/usr/bin/env python3
"""Fail-fast comparison of v9.3 and v9.5b targets plus GEOS statistics."""

import argparse
import os

import numpy as np
import torch
import yaml

from dataset_flow_multi import S2SHybridDataset as V93Dataset
from dataset_flow_multi_v9_5b import (
    GEOS_CONDITION_STATISTICS,
    S2SHybridDataset as V95bDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify v9.5b GEOS statistics without changing v9.3 targets."
    )
    parser.add_argument("--config", default="ml_model/config_flow_multiv9_5b.yaml")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def make_dataset(dataset_class, config, year):
    return dataset_class(
        data_root=config["data_dir"],
        start_year=year,
        end_year=year,
        normalize=True,
        preload=False,
        stats_file=config["stats_file"],
        target_domain=config.get("target_domain"),
        target_domain_bounds=config.get("target_domain_bounds"),
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
        t2m_target_mode=config.get("t2m_target_mode", "absolute"),
        t2m_residual_min=config.get("t2m_residual_min"),
        t2m_residual_max=config.get("t2m_residual_max"),
    )


def assert_close(name, actual, expected, atol=1e-6):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0, equal_nan=True):
        difference = torch.nan_to_num((actual - expected).abs(), nan=0.0).max().item()
        raise AssertionError(f"{name} mismatch; maximum absolute difference={difference:.8g}")


def main():
    args = parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    data_override = os.environ.get("DATA_DIR_OVERRIDE")
    if data_override:
        config["data_dir"] = data_override
    if not bool(config.get("use_geos_temporal_encoder", False)):
        raise AssertionError("v9.5b requires use_geos_temporal_encoder=True")

    year = int(args.year or config["train_start_year"])
    legacy = make_dataset(V93Dataset, config, year)
    current = make_dataset(V95bDataset, config, year)
    if len(legacy) != len(current):
        raise AssertionError(
            f"Dataset length changed: v9.3={len(legacy)}, v9.5b={len(current)}"
        )

    sample_index = int(args.sample_index)
    if sample_index < 0 or sample_index >= len(current):
        raise IndexError(f"sample-index {sample_index} outside 0..{len(current) - 1}")
    v93 = legacy[sample_index]
    v95b = current[sample_index]

    # Targets and all non-GEOS predictors must remain unchanged.
    for key in (
        "y_target",
        "target_raw",
        "target_raw_full",
        "x_obs",
        "x_global_context",
        "geos_ens_raw",
    ):
        assert_close(key, v95b[key], v93[key])

    x_geos = v95b["x_geos"]
    expected_shape = (
        len(GEOS_CONDITION_STATISTICS),
        2,
        4,
        x_geos.shape[-2],
        x_geos.shape[-1],
    )
    if tuple(x_geos.shape) != expected_shape:
        raise AssertionError(
            f"x_geos shape mismatch: actual={tuple(x_geos.shape)}, "
            f"expected={expected_shape}"
        )

    raw = v95b["geos_ens_raw"].float()
    member_count = int(raw.shape[0])
    member_count_by_lead = v95b["geos_member_count_raw"].float()
    bounds_path = config["stats_file"]
    if not os.path.isabs(bounds_path):
        bounds_path = os.path.join(os.path.dirname(__file__), bounds_path)
    bounds = torch.load(bounds_path, map_location="cpu", weights_only=True)
    g_min = float(bounds["geos_raw"]["min"])
    g_max = float(bounds["geos_raw"]["max"])

    def min_max_scale(value, vmin, vmax):
        return (
            2.0 * (torch.clamp(value, vmin, vmax) - vmin) / (vmax - vmin + 1e-6)
            - 1.0
        )

    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False)
    q10 = torch.quantile(raw, 0.10, dim=0)
    q90 = torch.quantile(raw, 0.90, dim=0)
    for statistic_index, statistic in ((0, mean), (2, q10), (3, q90)):
        expected = statistic.clone()
        expected[0] = min_max_scale(expected[0], g_min, g_max)
        expected[1] = min_max_scale(expected[1], 200.0, 320.0)
        assert_close(GEOS_CONDITION_STATISTICS[statistic_index], x_geos[statistic_index], expected)

    expected_std = std.clone()
    expected_std[0] = torch.clamp(
        torch.log1p(expected_std[0]) / np.log1p(max(g_max - g_min, 1.0)),
        0.0,
        1.0,
    )
    expected_std[1] = torch.clamp(
        torch.log1p(expected_std[1]) / np.log1p(120.0),
        0.0,
        1.0,
    )
    assert_close("std", x_geos[1], expected_std)

    count_reference = float(config.get("geos_member_count_reference", 4.0))
    expected_count = torch.clamp(
        torch.log1p(member_count_by_lead) / np.log1p(count_reference),
        min=0.0,
        max=1.0,
    )
    expected_count = expected_count.view(1, -1, 1, 1).expand_as(x_geos[4])
    assert_close(
        "member_count",
        x_geos[4],
        expected_count,
    )

    if torch.all(member_count_by_lead == 1):
        assert_close("deterministic std", x_geos[1], torch.zeros_like(x_geos[1]))
        assert_close("deterministic q10", x_geos[2], x_geos[0])
        assert_close("deterministic q90", x_geos[3], x_geos[0])

    print("PASS: v9.5b target/predictor equivalence and GEOS statistics")
    print(f"  config       : {args.config}")
    print(f"  year/index   : {year}/{sample_index}")
    print(f"  init/lead    : {v95b['year']}-{v95b['month']:02d}-{v95b['day']:02d} / {int(v95b['lead_idx']) + 1}")
    print(
        f"  members      : stored={member_count}, "
        f"available-by-lead={member_count_by_lead.tolist()}"
    )
    print(f"  x_geos shape : {tuple(x_geos.shape)}")
    print(f"  statistics   : {list(GEOS_CONDITION_STATISTICS)}")
    print(
        "  temporal     : enabled; all four x_geos leads are retained for "
        "the learned evolution encoder"
    )
    print("  targets      : exactly match the v9.3 dataset within atol=1e-6")


if __name__ == "__main__":
    main()
