#!/usr/bin/env python3
"""
South Asia noise-strategy comparison for the current multi-target v4 flow model.

This is the SA-aware replacement for the older compare_noise_multi.py scripts:
it reads the training config, respects target_domain/local/global predictors,
loads the v4 local/global model class, and writes side-by-side GEOS / random / EOF-LHS
metrics to CSV.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))

import noise_utils
import noise_utils_multi
from dataset_flow_multi import S2SHybridDataset
from flow_matching_multi_v4 import CustomFlowMatcher, FlowMatchingModel
from train_flow_multiv4 import (
    compute_crps,
    compute_rmse,
    crop_eof_bases_to_domain,
    euler_solve_chunked,
    get_area_weights,
    get_batch_global_context,
    get_target_domain_coords,
    resolve_target_domain,
)

EXPECTED_OUTPUT_DIR = "ml_output_flowmulti_v4_south_asia_global_context"
EXPECTED_LOCAL_OBS = ["sm", "mjo"]
EXPECTED_GLOBAL_CONTEXT = ["sst", "sss", "ivt", "z500_zonal_dev", "u250"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SA multi-v4 local/global noise strategies.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv4.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--checkpoint", type=str, default="best_flow_ckpt.pt")
    parser.add_argument("--ckpt-rank", type=int, default=None)
    parser.add_argument("--num_ensemble", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--ode_batch_size", type=int, default=None)
    parser.add_argument("--batch_limit", type=int, default=12, help="Maximum loader batches to evaluate; <=0 means no limit.")
    parser.add_argument("--full-year", action="store_true", help="Use every weekly init date instead of one init per month.")
    parser.add_argument(
        "--setting",
        action="append",
        default=None,
        help=(
            "Repeatable EOF setting: label,rho_pr,rho_t2m,beta_pr,beta_t2m,coarse_kernel. "
            "Use none for coarse_kernel to disable coarse variance."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_current_sa_v4_config(config, output_dir):
    target_domain = config.get("target_domain")
    local_obs = list(config.get("local_obs_variables") or [])
    global_context = list(config.get("global_context_variables") or [])
    output_name = os.path.basename(os.path.normpath(output_dir))

    if target_domain != "south_asia":
        raise ValueError(f"Expected target_domain=south_asia, got {target_domain!r}")
    if output_name != EXPECTED_OUTPUT_DIR:
        raise ValueError(f"Expected output_dir ending in {EXPECTED_OUTPUT_DIR}, got {output_dir!r}")
    if local_obs != EXPECTED_LOCAL_OBS:
        raise ValueError(f"Expected local_obs_variables={EXPECTED_LOCAL_OBS}, got {local_obs}")
    if global_context != EXPECTED_GLOBAL_CONTEXT:
        raise ValueError(f"Expected global_context_variables={EXPECTED_GLOBAL_CONTEXT}, got {global_context}")


def resolve_checkpoint(output_dir, checkpoint, ckpt_rank):
    if ckpt_rank is not None:
        registry_path = os.path.join(output_dir, "model_registry.json")
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Missing registry for --ckpt-rank: {registry_path}")
        with open(registry_path, "r") as f:
            registry = json.load(f)
        if ckpt_rank < 1 or ckpt_rank > len(registry):
            raise ValueError(f"--ckpt-rank {ckpt_rank} out of range. Registry has {len(registry)} entries.")
        path = registry[ckpt_rank - 1]["path"]
        if os.path.isabs(path) or os.path.exists(path):
            return path
        return os.path.join(output_dir, path)

    return checkpoint if os.path.isabs(checkpoint) else os.path.join(output_dir, checkpoint)


def parse_setting(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Bad --setting {text!r}; expected label,rho_pr,rho_t2m,beta_pr,beta_t2m,coarse_kernel"
        )
    label, rho_pr, rho_t2m, beta_pr, beta_t2m, coarse = parts
    coarse_kernel = None if coarse.lower() in {"none", "no", "null", "0"} else int(coarse)
    return {
        "label": label,
        "rho_pr": float(rho_pr),
        "rho_t2m": float(rho_t2m),
        "beta_pr": float(beta_pr),
        "beta_t2m": float(beta_t2m),
        "coarse_kernel": coarse_kernel,
    }


def default_settings(config):
    return [{
        "label": "cfg_eof_var",
        "rho_pr": float(config.get("validation_rho_pr", 0.15)),
        "rho_t2m": float(config.get("validation_rho_t2m", config.get("validation_rho_pr", 0.15))),
        "beta_pr": float(config.get("validation_var_beta_pr", 0.3)),
        "beta_t2m": float(config.get("validation_var_beta_t2m", config.get("validation_var_beta_pr", 0.3))),
        "coarse_kernel": (
            None if config.get("validation_variance_coarse_kernel") in {None, "none"}
            else int(config.get("validation_variance_coarse_kernel"))
        ),
    }]


def resolve_aux_path(data_dir, filename):
    for base_dir in (
        os.path.join(data_dir, "eof"),
        data_dir,
        os.path.dirname(__file__),
    ):
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(data_dir, "eof", filename)


def maybe_load_eof(path, domain_info):
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return crop_eof_bases_to_domain(data.get("eof_bases"), domain_info)


def load_noise_context(config, domain_info):
    data_dir = config["data_dir"]
    context = {
        "pr_mjo": maybe_load_eof(resolve_aux_path(data_dir, "mjo_eof_bases.pt"), domain_info),
        "pr_nao": maybe_load_eof(resolve_aux_path(data_dir, "nao_eof_bases.pt"), domain_info),
        "pr_enso": maybe_load_eof(resolve_aux_path(data_dir, "enso_eof_bases.pt"), domain_info),
        "t2m_mjo": maybe_load_eof(resolve_aux_path(data_dir, "mjo_t2m_eof_bases.pt"), domain_info),
        "t2m_nao": maybe_load_eof(resolve_aux_path(data_dir, "nao_t2m_eof_bases.pt"), domain_info),
        "t2m_enso": maybe_load_eof(resolve_aux_path(data_dir, "enso_t2m_eof_bases.pt"), domain_info),
        "nao_lookup": None,
        "oni_lookup": None,
        "mjo_df": None,
    }

    nao_path = os.path.join(data_dir, "norm.daily.nao.index.b500101.current.ascii")
    oni_path = os.path.join(data_dir, "oni.ascii.txt")
    mjo_path = os.path.join(data_dir, "mjo_processed.csv")
    if os.path.exists(nao_path):
        context["nao_lookup"] = noise_utils.parse_nao_index(nao_path)
    if os.path.exists(oni_path):
        context["oni_lookup"] = noise_utils.parse_oni_index(oni_path)
    if os.path.exists(mjo_path):
        mjo_df = pd.read_csv(mjo_path, parse_dates=["S"])
        context["mjo_df"] = mjo_df.set_index(mjo_df["S"].dt.strftime("%Y-%m-%d"))

    return context


def expand_global_context(batch, device, num_ensemble):
    global_context = get_batch_global_context(batch, device)
    if global_context is None:
        return None
    batch_size = global_context.shape[0]
    channels, height, width = global_context.shape[1:]
    return (
        global_context.unsqueeze(1)
        .expand(batch_size, num_ensemble, channels, height, width)
        .reshape(batch_size * num_ensemble, channels, height, width)
    )


def build_condition(batch, device):
    x_obs = batch["x_obs"].to(device)
    x_geos = batch["x_geos"].to(device)
    batch_size = x_obs.shape[0]
    height, width = x_obs.shape[-2:]
    x_geos_flat = x_geos.contiguous().view(batch_size, -1, height, width)

    months = batch["month"].to(device).float()
    sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)

    lead_idx = batch["lead_idx"].to(device).long()
    lead_val = (lead_idx.float() / 1.5) - 1.0
    lead_channel = lead_val.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)

    return torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1), lead_idx


def decode_multi(pred_norm):
    pred_pr = torch.clamp(pred_norm[:, :, 0], min=-1.0, max=1.0)
    sqrt_pr = ((pred_pr + 1.0) / 2.0) * 7.071
    precip = torch.clamp(sqrt_pr ** 2, min=0.0)

    pred_t2m = torch.clamp(pred_norm[:, :, 1], min=-1.0, max=1.0)
    t2m = ((pred_t2m + 1.0) / 2.0) * (320.0 - 200.0) + 200.0
    return precip, t2m


def summarize_metrics(ensemble_pr, ensemble_t2m, target_pr, target_t2m, area_weights):
    total = {
        "pr_crps": compute_crps(ensemble_pr, target_pr, area_weights),
        "pr_rmse": compute_rmse(ensemble_pr.mean(dim=0), target_pr, area_weights),
        "t2m_crps": compute_crps(ensemble_t2m, target_t2m, area_weights),
        "t2m_rmse": compute_rmse(ensemble_t2m.mean(dim=0), target_t2m, area_weights),
    }
    out = [total]
    for lead in range(4):
        out.append({
            "pr_crps": compute_crps(ensemble_pr[:, :, lead:lead + 1], target_pr[:, lead:lead + 1], area_weights),
            "pr_rmse": compute_rmse(ensemble_pr.mean(dim=0)[:, lead:lead + 1], target_pr[:, lead:lead + 1], area_weights),
            "t2m_crps": compute_crps(ensemble_t2m[:, :, lead:lead + 1], target_t2m[:, lead:lead + 1], area_weights),
            "t2m_rmse": compute_rmse(ensemble_t2m.mean(dim=0)[:, lead:lead + 1], target_t2m[:, lead:lead + 1], area_weights),
        })
    return out


@torch.no_grad()
def run_model_strategy(
    name,
    model,
    flow_matcher,
    batch,
    device,
    num_ensemble,
    num_steps,
    area_weights,
    noise,
    use_variance,
    beta_pr=0.0,
    beta_t2m=0.0,
    coarse_kernel=None,
    ode_batch_size=None,
):
    target_raw = batch["target_raw_full"][0::4].to(device)
    target_pr = target_raw[:, 0]
    target_t2m = target_raw[:, 1]
    num_inits = target_pr.shape[0]

    x_cond, lead_idx = build_condition(batch, device)
    batch_size, _, height, width = x_cond.shape
    x_cond_expanded = (
        x_cond.unsqueeze(1)
        .expand(batch_size, num_ensemble, -1, height, width)
        .reshape(batch_size * num_ensemble, -1, height, width)
    )
    lead_expanded = lead_idx.unsqueeze(1).expand(batch_size, num_ensemble).reshape(-1).long()
    global_context_expanded = expand_global_context(batch, device, num_ensemble)

    pred_norm = euler_solve_chunked(
        flow_matcher,
        model,
        noise,
        x_cond_expanded,
        num_steps=num_steps,
        lead_idx=lead_expanded,
        apply_flow_variance=use_variance,
        variance_beta=(beta_pr, beta_t2m),
        variance_coarse_kernel=coarse_kernel,
        chunk_size=ode_batch_size,
        global_context=global_context_expanded,
    )
    pred_norm = pred_norm.view(batch_size, num_ensemble, 2, height, width)
    precip, t2m = decode_multi(pred_norm)
    ensemble_pr = precip.transpose(0, 1).view(num_ensemble, num_inits, 4, height, width)
    ensemble_t2m = t2m.transpose(0, 1).view(num_ensemble, num_inits, 4, height, width)

    return name, summarize_metrics(ensemble_pr, ensemble_t2m, target_pr, target_t2m, area_weights)


def geos_baseline(batch, device, area_weights):
    target_raw = batch["target_raw_full"][0::4].to(device)
    target_pr = target_raw[:, 0]
    target_t2m = target_raw[:, 1]
    geos_ens = batch["geos_ens_raw"][0::4].to(device)
    geos_pr = geos_ens[:, :, 0].transpose(0, 1)
    geos_t2m = geos_ens[:, :, 1].transpose(0, 1)
    return summarize_metrics(geos_pr, geos_t2m, target_pr, target_t2m, area_weights)


def format_total(metrics):
    total = metrics[0]
    return (
        f"PR CRPS={total['pr_crps']:.3f} RMSE={total['pr_rmse']:.3f} | "
        f"T2M CRPS={total['t2m_crps']:.3f} RMSE={total['t2m_rmse']:.3f}"
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    if "DATA_DIR_OVERRIDE" in os.environ:
        config["data_dir"] = os.environ["DATA_DIR_OVERRIDE"]

    output_dir = args.output_dir or config["output_dir"]
    validate_current_sa_v4_config(config, output_dir)
    stats_file = config.get("stats_file", "v1_multi_global_stats.pt")
    target_domain = config.get("target_domain")
    target_domain_bounds = config.get("target_domain_bounds")
    domain_info = resolve_target_domain(target_domain, target_domain_bounds)
    lats, _ = get_target_domain_coords(target_domain, target_domain_bounds)

    batch_limit = None if args.batch_limit is not None and args.batch_limit <= 0 else args.batch_limit
    dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year,
        end_year=args.year,
        normalize=True,
        preload=False,
        stats_file=stats_file,
        subsample_monthly=not args.full_year,
        target_domain=target_domain,
        target_domain_bounds=target_domain_bounds,
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

    obs_channels = int(dataset.obs_channel_count)
    cond_channels = obs_channels + 8 + 3
    model_in_channels = cond_channels + 2
    global_context_channels = int(dataset.global_context_channel_count)
    block_channels = tuple(int(v) for v in config.get("unet_block_out_channels", [128, 256, 512, 768]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMatchingModel(
        in_channels=model_in_channels,
        out_channels=2,
        block_out_channels=block_channels,
        sample_size=(len(dataset.lats), len(dataset.lons)),
        global_context_channels=global_context_channels,
    ).to(device)

    ckpt_path = resolve_checkpoint(output_dir, args.checkpoint, args.ckpt_rank)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    flow_matcher = CustomFlowMatcher(device=device)
    noise_context = load_noise_context(config, domain_info)
    area_weights = get_area_weights(lats, device)

    num_ensemble = int(args.num_ensemble or config.get("test_num_ensemble", config.get("validation_num_ensemble", 15)))
    num_steps = int(args.num_steps or config.get("test_num_steps", config.get("validation_num_steps", 10)))
    ode_batch_size = int(args.ode_batch_size or config.get("validation_ode_batch_size", num_ensemble * 4))
    settings = [parse_setting(s) for s in args.setting] if args.setting else default_settings(config)

    print("\n" + "=" * 88)
    print("South Asia noise comparison")
    print(f"  Config       : {args.config}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Checkpoint   : {ckpt_path} (epoch={checkpoint.get('epoch', 'unknown')})")
    print(f"  Data dir     : {config['data_dir']}")
    print(f"  Domain       : {target_domain or 'global'} ({len(dataset.lats)}x{len(dataset.lons)})")
    print(f"  Local vars   : {list(dataset.local_obs_variables)}")
    print(f"  Global ctx   : {list(dataset.global_context_variables)}")
    print(f"  Ensembles    : {num_ensemble}")
    print(f"  ODE steps    : {num_steps}")
    print(f"  ODE chunk    : {ode_batch_size}")
    print(f"  Sampling     : {'full weekly year' if args.full_year else 'monthly subset'}")
    print(f"  Batch limit  : {batch_limit if batch_limit is not None else 'none'}")
    print(f"  Settings     : {settings}")
    print("=" * 88 + "\n")

    results = {"0. GEOS": [], "1. Pure Random": []}
    for setting in settings:
        label = (
            f"EOF {setting['label']} "
            f"rhoPR{setting['rho_pr']:.2f}_rhoT{setting['rho_t2m']:.2f}_"
            f"bPR{setting['beta_pr']:.2f}_bT{setting['beta_t2m']:.2f}_"
            f"c{setting['coarse_kernel'] if setting['coarse_kernel'] is not None else 'none'}"
        )
        results[label] = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="SA noise compare")):
        if batch_limit is not None and batch_idx >= batch_limit:
            break

        height, width = batch["y_target"].shape[-2:]
        geos_metrics = geos_baseline(batch, device, area_weights)
        results["0. GEOS"].append(geos_metrics)

        pure_noise = torch.randn((batch["y_target"].shape[0] * num_ensemble, 2, height, width), device=device)
        _, pure_metrics = run_model_strategy(
            "1. Pure Random",
            model,
            flow_matcher,
            batch,
            device,
            num_ensemble,
            num_steps,
            area_weights,
            pure_noise,
            use_variance=False,
            ode_batch_size=ode_batch_size,
        )
        results["1. Pure Random"].append(pure_metrics)

        for setting in settings:
            label = (
                f"EOF {setting['label']} "
                f"rhoPR{setting['rho_pr']:.2f}_rhoT{setting['rho_t2m']:.2f}_"
                f"bPR{setting['beta_pr']:.2f}_bT{setting['beta_t2m']:.2f}_"
                f"c{setting['coarse_kernel'] if setting['coarse_kernel'] is not None else 'none'}"
            )
            eof_noise = noise_utils_multi.generate_dynamic_multimodal_noise_multi(
                batch=batch,
                E=num_ensemble,
                device=device,
                pr_mjo_bases=noise_context["pr_mjo"],
                pr_nao_bases=noise_context["pr_nao"],
                pr_enso_bases=noise_context["pr_enso"],
                t2m_mjo_bases=noise_context["t2m_mjo"],
                t2m_nao_bases=noise_context["t2m_nao"],
                t2m_enso_bases=noise_context["t2m_enso"],
                nao_lookup=noise_context["nao_lookup"],
                oni_lookup=noise_context["oni_lookup"],
                mjo_df=noise_context["mjo_df"],
                year=args.year,
                use_lhs=True,
                orthogonalize_lhs=True,
            )
            noise = noise_utils_multi.mix_noise_with_random_multi(
                eof_noise,
                setting["rho_pr"],
                setting["rho_t2m"],
            )
            _, metrics = run_model_strategy(
                label,
                model,
                flow_matcher,
                batch,
                device,
                num_ensemble,
                num_steps,
                area_weights,
                noise,
                use_variance=True,
                beta_pr=setting["beta_pr"],
                beta_t2m=setting["beta_t2m"],
                coarse_kernel=setting["coarse_kernel"],
                ode_batch_size=ode_batch_size,
            )
            results[label].append(metrics)

        print(f"\nBatch {batch_idx}:")
        for name, values in results.items():
            if len(values) == batch_idx + 1:
                print(f"  {name:<70} {format_total(values[-1])}")

    lead_labels = ["total", "week1", "week2", "week3", "week4"]
    rows = []
    for name, batches in results.items():
        for lead_idx, lead_label in enumerate(lead_labels):
            row = {"strategy": name, "lead": lead_label, "n_batches": len(batches)}
            for metric in ("pr_crps", "pr_rmse", "t2m_crps", "t2m_rmse"):
                row[metric] = float(np.mean([b[lead_idx][metric] for b in batches]))
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_label = os.path.splitext(os.path.basename(ckpt_path))[0]
    sample_tag = "full_year" if args.full_year else "monthly"
    csv_path = os.path.join(output_dir, f"noise_comparison_sa_{args.year}_{sample_tag}_{ckpt_label}.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\nFinal mean scores:")
    print(df[df["lead"] == "total"].to_string(index=False))
    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
