#!/usr/bin/env python3
"""
Driver attribution for South Asia v9 forecasts using group occlusion.

The script reruns the same init dates with the same stochastic noise and zeros
one conditioning group at a time. Positive delta CRPS/RMSE means the forecast
got worse when that group was removed, so the group helped that target.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from compare_noise_multiv9_sa import (
    FlowMatchingModel,
    CustomFlowMatcher,
    decode_multi,
    load_noise_context,
    resolve_checkpoint,
    resolve_t2m_residual_bounds,
    summarize_metrics,
    validate_current_sa_v9_config,
)
from dataset_flow_multi import S2SHybridDataset, resolve_target_domain
from generate_forecast_zarr_multiv9_sa_testmode import generate_noise, get_autocast_context
from train_flow_multiv9 import euler_solve_chunked, get_area_weights, get_batch_global_context, get_target_domain_coords


DEFAULT_OUTPUT_DIR = (
    "ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/"
    "driver_attribution"
)


def parse_args():
    parser = argparse.ArgumentParser(description="South Asia v9 group-occlusion driver attribution.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv9.yaml")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--checkpoint", type=str, default="best_flow_ckpt.pt")
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--ode_batch_size", type=int, default=120)
    parser.add_argument("--batch_limit", type=int, default=12)
    parser.add_argument("--full-year", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--groups",
        type=str,
        default=(
            "geos_all,geos_pr,geos_t2m,local_sm,local_mjo,"
            "global_sst,global_sss,global_ivt,global_z500_zonal_dev,global_u250,"
            "all_global_context,all_local_obs"
        ),
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_groups(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def variable_slice(name, variables):
    if name not in variables:
        raise ValueError(f"{name!r} is not in configured variables {variables}")
    idx = variables.index(name)
    return slice(idx * 4, (idx + 1) * 4)


def apply_group_occlusion(batch, device, config, group):
    x_obs = batch["x_obs"].to(device).clone()
    x_geos = batch["x_geos"].to(device).clone()
    global_context = get_batch_global_context(batch, device)
    if global_context is not None:
        global_context = global_context.clone()

    local_vars = list(config.get("local_obs_variables") or [])
    global_vars = list(config.get("global_context_variables") or [])

    if bool(config.get("zero_geos_condition", False)) or group == "geos_all":
        x_geos.zero_()
    elif group == "geos_pr":
        x_geos[:, :, 0].zero_()
    elif group == "geos_t2m":
        x_geos[:, :, 1].zero_()

    if group == "all_local_obs":
        x_obs.zero_()
    elif group.startswith("local_"):
        var = group.replace("local_", "", 1)
        x_obs[:, variable_slice(var, local_vars)].zero_()

    if global_context is not None:
        if group == "all_global_context":
            global_context.zero_()
        elif group.startswith("global_"):
            var = group.replace("global_", "", 1)
            global_context[:, variable_slice(var, global_vars)].zero_()

    batch_size = x_obs.shape[0]
    height, width = x_obs.shape[-2:]
    x_geos_flat = x_geos.contiguous().view(batch_size, -1, height, width)
    months = batch["month"].to(device).float()
    sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    lead_idx = batch["lead_idx"].to(device).long()
    lead_val = (lead_idx.float() / 1.5) - 1.0
    lead_channel = lead_val.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    x_cond = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1)
    return x_cond, lead_idx, global_context


def expand_global_context(global_context, num_ensemble):
    if global_context is None:
        return None
    batch_size = global_context.shape[0]
    channels, height, width = global_context.shape[1:]
    return (
        global_context.unsqueeze(1)
        .expand(batch_size, num_ensemble, channels, height, width)
        .reshape(batch_size * num_ensemble, channels, height, width)
        .contiguous()
    )


def weighted_mean_torch(values, area_weights):
    weights = area_weights
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(0)
    weights = weights.expand_as(values)
    mask = torch.isfinite(values)
    num = torch.where(mask, values * weights, torch.zeros_like(values)).sum()
    den = torch.where(mask, weights, torch.zeros_like(weights)).sum()
    return float((num / (den + 1e-12)).detach().cpu())


@torch.no_grad()
def run_group(
    model,
    flow_matcher,
    batch,
    device,
    config,
    group,
    noise,
    num_ensemble,
    num_steps,
    ode_batch_size,
    area_weights,
    t2m_target_mode,
    t2m_residual_min,
    t2m_residual_max,
):
    x_cond, lead_idx, global_context = apply_group_occlusion(batch, device, config, group)
    batch_size, _, height, width = x_cond.shape
    num_inits = batch_size // 4
    x_cond_expanded = (
        x_cond.unsqueeze(1)
        .expand(batch_size, num_ensemble, -1, height, width)
        .reshape(batch_size * num_ensemble, -1, height, width)
        .contiguous()
    )
    lead_expanded = lead_idx.unsqueeze(1).expand(batch_size, num_ensemble).reshape(-1).long()
    global_expanded = expand_global_context(global_context, num_ensemble)

    beta_pr = float(config.get("validation_var_beta_pr", 0.45))
    beta_t2m = float(config.get("validation_var_beta_t2m", 0.03))
    coarse_kernel = config.get("validation_variance_coarse_kernel", 8)
    coarse_kernel = None if coarse_kernel in {None, "none", "None"} else int(coarse_kernel)
    mixed_precision = str(config.get("mixed_precision", "no")).lower()

    with get_autocast_context(device, mixed_precision):
        pred_norm = euler_solve_chunked(
            flow_matcher,
            model,
            noise,
            x_cond_expanded,
            num_steps=int(num_steps),
            lead_idx=lead_expanded,
            apply_flow_variance=True,
            variance_beta=(beta_pr, beta_t2m),
            variance_coarse_kernel=coarse_kernel,
            chunk_size=int(ode_batch_size),
            global_context=global_expanded,
        )

    pred_norm = pred_norm.view(batch_size, num_ensemble, 2, height, width).float()
    precip, t2m = decode_multi(
        pred_norm,
        batch,
        device,
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min,
        t2m_residual_max=t2m_residual_max,
    )
    ensemble_pr = precip.transpose(0, 1).reshape(num_ensemble, num_inits, 4, height, width)
    ensemble_t2m = t2m.transpose(0, 1).reshape(num_ensemble, num_inits, 4, height, width)
    target_raw = batch["target_raw_full"][0::4].to(device)
    metrics = summarize_metrics(ensemble_pr, ensemble_t2m, target_raw[:, 0], target_raw[:, 1], area_weights)
    return metrics, ensemble_pr.mean(dim=0), ensemble_t2m.mean(dim=0)


def rows_from_metrics(batch_index, group, metrics, baseline_metrics=None, pr_change=None, t2m_change=None):
    rows = []
    for lead_pos, row in enumerate(metrics):
        lead = "all" if lead_pos == 0 else lead_pos
        out = {
            "batch": batch_index,
            "group": group,
            "lead": lead,
            "pr_crps": row["pr_crps"],
            "pr_rmse": row["pr_rmse"],
            "t2m_crps": row["t2m_crps"],
            "t2m_rmse": row["t2m_rmse"],
        }
        if baseline_metrics is not None:
            base = baseline_metrics[lead_pos]
            out.update({
                "delta_pr_crps": row["pr_crps"] - base["pr_crps"],
                "delta_pr_rmse": row["pr_rmse"] - base["pr_rmse"],
                "delta_t2m_crps": row["t2m_crps"] - base["t2m_crps"],
                "delta_t2m_rmse": row["t2m_rmse"] - base["t2m_rmse"],
            })
        if lead_pos == 0 and pr_change is not None and t2m_change is not None:
            out["mean_abs_change_pr"] = pr_change
            out["mean_abs_change_t2m"] = t2m_change
        rows.append(out)
    return rows


def make_plots(summary, output_dir):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = summary[summary["lead"].astype(str) == "all"].copy()
    if total.empty:
        return []
    groups = list(total["group"])
    metrics = ["delta_pr_crps", "delta_t2m_crps", "mean_abs_change_pr", "mean_abs_change_t2m"]
    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, 0.35 * len(groups))))
    paths = []
    for ax, metric in zip(axes.ravel(), metrics):
        vals = total[metric].values
        colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]
        ax.barh(groups, vals, color=colors)
        ax.axvline(0.0, color="k", linewidth=1)
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plot_dir, "driver_attribution_total.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    if "DATA_DIR_OVERRIDE" in os.environ:
        config["data_dir"] = os.environ["DATA_DIR_OVERRIDE"]
    model_output_dir = args.model_output_dir or config["output_dir"]
    validate_current_sa_v9_config(config, model_output_dir)
    t2m_target_mode, t2m_residual_min, t2m_residual_max = resolve_t2m_residual_bounds(config)
    target_domain = config.get("target_domain")
    target_domain_bounds = config.get("target_domain_bounds")
    domain_info = resolve_target_domain(target_domain, target_domain_bounds)

    dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year,
        end_year=args.year,
        normalize=True,
        preload=False,
        stats_file=config.get("stats_file", "v8_sa_55e100e_0n40n_global_local_stats.pt"),
        subsample_monthly=not args.full_year,
        target_domain=target_domain,
        target_domain_bounds=target_domain_bounds,
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min,
        t2m_residual_max=t2m_residual_max,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_channels = int(dataset.obs_channel_count)
    cond_channels = obs_channels + 8 + 3
    model_in_channels = cond_channels + 2
    global_context_channels = int(dataset.global_context_channel_count)
    block_channels = tuple(int(v) for v in config.get("unet_block_out_channels", [128, 256, 512, 768]))
    model = FlowMatchingModel(
        in_channels=model_in_channels,
        out_channels=2,
        block_out_channels=block_channels,
        sample_size=(len(dataset.lats), len(dataset.lons)),
        global_context_channels=global_context_channels,
    ).to(device)
    ckpt_path = resolve_checkpoint(model_output_dir, args.checkpoint, None)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    flow_matcher = CustomFlowMatcher(device=device)
    lats, _ = get_target_domain_coords(target_domain, target_domain_bounds)
    area_weights = get_area_weights(lats, device)
    noise_context = load_noise_context(config, domain_info)
    groups = parse_groups(args.groups)

    rho_pr = float(config.get("validation_rho_pr", 0.25))
    rho_t2m = float(config.get("validation_rho_t2m", 0.08))

    print("\n" + "=" * 88)
    print("South Asia v9 driver attribution by group occlusion")
    print(f"  Config       : {args.config}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Checkpoint   : {ckpt_path} (epoch={checkpoint.get('epoch', 'unknown')})")
    print(f"  Year         : {args.year}")
    print(f"  Sampling     : {'full weekly year' if args.full_year else 'monthly'}")
    print(f"  Batch limit  : {args.batch_limit if args.batch_limit > 0 else 'none'}")
    print(f"  Ensembles    : {args.num_ensemble}")
    print(f"  ODE steps    : {args.num_steps}")
    print(f"  Groups       : {groups}")
    print("=" * 88)

    rows = []
    for b_idx, batch in enumerate(tqdm(loader, desc="SA driver attribution")):
        if args.batch_limit > 0 and b_idx >= args.batch_limit:
            break
        current_year = int(batch["year"][0].item())
        noise = generate_noise(
            batch=batch,
            num_ensemble=args.num_ensemble,
            device=device,
            year=current_year,
            use_eof_lhs_noise=True,
            noise_context=noise_context,
            rho_pr=rho_pr,
            rho_t2m=rho_t2m,
        )

        baseline_metrics, baseline_pr, baseline_t2m = run_group(
            model, flow_matcher, batch, device, config, "baseline", noise,
            args.num_ensemble, args.num_steps, args.ode_batch_size, area_weights,
            t2m_target_mode, t2m_residual_min, t2m_residual_max,
        )
        rows.extend(rows_from_metrics(b_idx, "baseline", baseline_metrics))

        for group in groups:
            metrics, group_pr, group_t2m = run_group(
                model, flow_matcher, batch, device, config, group, noise,
                args.num_ensemble, args.num_steps, args.ode_batch_size, area_weights,
                t2m_target_mode, t2m_residual_min, t2m_residual_max,
            )
            pr_change = weighted_mean_torch(torch.abs(group_pr - baseline_pr), area_weights)
            t2m_change = weighted_mean_torch(torch.abs(group_t2m - baseline_t2m), area_weights)
            rows.extend(
                rows_from_metrics(
                    b_idx, group, metrics, baseline_metrics=baseline_metrics,
                    pr_change=pr_change, t2m_change=t2m_change,
                )
            )

    by_batch = pd.DataFrame(rows)
    summary = (
        by_batch[by_batch["group"] != "baseline"]
        .groupby(["group", "lead"], as_index=False)
        .mean(numeric_only=True)
    )
    summary["lead_order"] = summary["lead"].map(lambda value: 0 if str(value) == "all" else int(value))
    summary = summary.sort_values(["lead_order", "delta_pr_crps", "delta_t2m_crps"], ascending=[True, False, False])
    paths = {
        "by_batch": os.path.join(args.output_dir, "driver_attribution_by_batch.csv"),
        "summary": os.path.join(args.output_dir, "driver_attribution_summary.csv"),
        "metadata": os.path.join(args.output_dir, "driver_attribution_metadata.json"),
    }
    by_batch.to_csv(paths["by_batch"], index=False)
    summary.to_csv(paths["summary"], index=False)
    plot_paths = make_plots(summary, args.output_dir)
    with open(paths["metadata"], "w") as f:
        json.dump(
            {
                "config": args.config,
                "model_output_dir": model_output_dir,
                "checkpoint": ckpt_path,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "year": args.year,
                "full_year": args.full_year,
                "batch_limit": args.batch_limit,
                "num_ensemble": args.num_ensemble,
                "num_steps": args.num_steps,
                "groups": groups,
                "plot_paths": plot_paths,
            },
            f,
            indent=2,
        )

    total = summary[summary["lead"].astype(str) == "all"].copy()
    if not total.empty:
        cols = [
            "group", "delta_pr_crps", "delta_pr_rmse", "delta_t2m_crps",
            "delta_t2m_rmse", "mean_abs_change_pr", "mean_abs_change_t2m",
        ]
        total = total.sort_values(["delta_pr_crps", "delta_t2m_crps"], ascending=False)
        float_cols = total[cols].select_dtypes(include=[np.floating]).columns
        total.loc[:, float_cols] = total[float_cols].round(4)
        print("\nDriver attribution summary (lead=all; positive delta means removing group hurts)")
        print(total[cols].to_string(index=False))

    print("\nDriver attribution complete")
    for label, path in paths.items():
        print(f"  {label:9s}: {path}")
    for path in plot_paths:
        print(f"  plot     : {path}")


if __name__ == "__main__":
    main()
