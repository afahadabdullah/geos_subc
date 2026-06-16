#!/usr/bin/env python3
"""
CONUS v9 checkpoint sweep under pure random noise.

This compares model checkpoints while keeping the ensemble-generation method
fixed to pure Gaussian noise. Noise is deterministic per batch so checkpoint
rankings are not affected by different random draws.

Default ordering is GEOS baseline, best_flow_ckpt.pt, then periodic/checkpoint
epochs 70, 80, ..., 140.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))

from compare_noise_multiv9_conus import (
    build_condition,
    decode_multi,
    expand_global_context,
    format_total,
    geos_baseline,
    load_config,
    resolve_t2m_residual_bounds,
    summarize_metrics,
    validate_current_sa_v9_config,
)
from dataset_flow_multi import S2SHybridDataset
from flow_matching_multi_v9 import CustomFlowMatcher, FlowMatchingModel
from train_flow_multiv9 import (
    euler_solve_chunked,
    get_area_weights,
    get_target_domain_coords,
    resolve_target_domain,
)


DEFAULT_EPOCHS = tuple(range(70, 141, 10))


def parse_epoch_list(text):
    if text is None:
        return list(DEFAULT_EPOCHS)
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def resolve_relative_checkpoint(output_dir, path):
    return path if os.path.isabs(path) else os.path.join(output_dir, path)


def resolve_epoch_checkpoint(output_dir, epoch):
    periodic = os.path.join(output_dir, f"periodic_ckpt_epoch_{epoch}.pt")
    if os.path.exists(periodic):
        return f"periodic_E{epoch}", periodic

    best_matches = sorted(glob.glob(os.path.join(output_dir, f"best_model_epoch_{epoch}_*.pt")))
    if best_matches:
        return f"best_model_E{epoch}", best_matches[0]

    return None


def resolve_checkpoints(
    output_dir,
    epochs,
    explicit_checkpoints,
    include_best=True,
    best_checkpoint="best_flow_ckpt.pt",
    allow_missing=True,
):
    resolved = []
    missing = []

    if include_best:
        best_path = resolve_relative_checkpoint(output_dir, best_checkpoint)
        if os.path.exists(best_path):
            resolved.append(("best_flow_ckpt", best_path))
        else:
            missing.append(best_checkpoint)

    if explicit_checkpoints:
        for item in explicit_checkpoints:
            path = resolve_relative_checkpoint(output_dir, item)
            if not os.path.exists(path):
                missing.append(path)
                continue
            resolved.append((os.path.splitext(os.path.basename(path))[0], path))
    else:
        for epoch in epochs:
            match = resolve_epoch_checkpoint(output_dir, epoch)
            if match is None:
                missing.append(f"epoch {epoch}")
                continue
            resolved.append(match)

    if missing:
        message = "Missing requested checkpoints: " + ", ".join(missing)
        if allow_missing:
            print(f"⚠️ {message}. Skipping missing entries.")
        else:
            raise FileNotFoundError(message)

    if not resolved:
        raise FileNotFoundError("No checkpoint files were found for the requested sweep.")
    return resolved


def format_checkpoint_name(source_label, checkpoint_epoch):
    epoch_text = "unknown" if checkpoint_epoch is None else str(checkpoint_epoch)
    if source_label == "best_flow_ckpt":
        return f"best_flow_ckpt.pt (epoch {epoch_text}) pure"
    if source_label.startswith("periodic_E"):
        requested_epoch = source_label.replace("periodic_E", "")
        suffix = "" if requested_epoch == epoch_text else f" ckpt_epoch={epoch_text}"
        return f"periodic_ckpt_epoch_{requested_epoch}.pt{suffix} pure"
    if source_label.startswith("best_model_E"):
        requested_epoch = source_label.replace("best_model_E", "")
        suffix = "" if requested_epoch == epoch_text else f" ckpt_epoch={epoch_text}"
        return f"best_model_epoch_{requested_epoch}.pt{suffix} pure"
    return f"{source_label} (epoch {epoch_text}) pure"


def deterministic_noise(shape, device, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


@torch.no_grad()
def run_pure_noise_checkpoint(
    model,
    flow_matcher,
    batch,
    device,
    num_ensemble,
    num_steps,
    ode_batch_size,
    area_weights,
    seed,
    t2m_target_mode,
    t2m_residual_min,
    t2m_residual_max,
    print_diag=False,
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
    noise = deterministic_noise((batch_size * num_ensemble, 2, height, width), device, seed)

    if print_diag:
        print(f"    📊 [Pure Noise] Shape: {list(noise.shape)}")
        for channel in range(noise.shape[1]):
            item = noise[:, channel]
            print(
                f"       Ch{channel}: Mean={item.mean():.4f}, Std={item.std():.4f}, "
                f"Min={item.min():.4f}, Max={item.max():.4f}"
            )

    pred_norm = euler_solve_chunked(
        flow_matcher,
        model,
        noise,
        x_cond_expanded,
        num_steps=num_steps,
        lead_idx=lead_expanded,
        apply_flow_variance=False,
        chunk_size=ode_batch_size,
        global_context=global_context_expanded,
    )
    pred_norm = pred_norm.view(batch_size, num_ensemble, 2, height, width)
    precip, t2m = decode_multi(
        pred_norm,
        batch,
        device,
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min,
        t2m_residual_max=t2m_residual_max,
    )

    ensemble_pr = precip.transpose(0, 1).view(num_ensemble, num_inits, 4, height, width)
    ensemble_t2m = t2m.transpose(0, 1).view(num_ensemble, num_inits, 4, height, width)
    return summarize_metrics(ensemble_pr, ensemble_t2m, target_pr, target_t2m, area_weights)


def mean_metrics(items, lead_index=0):
    return {
        "pr_crps": float(np.mean([entry[lead_index]["pr_crps"] for entry in items])),
        "pr_rmse": float(np.mean([entry[lead_index]["pr_rmse"] for entry in items])),
        "t2m_crps": float(np.mean([entry[lead_index]["t2m_crps"] for entry in items])),
        "t2m_rmse": float(np.mean([entry[lead_index]["t2m_rmse"] for entry in items])),
    }


def save_results(results, metadata, output_dir, year, sample_tag):
    lead_names = ["total", "week1", "week2", "week3", "week4"]
    rows = []
    detail_rows = []

    for name, batches in results.items():
        meta = metadata.get(name, {})
        for lead_idx, lead in enumerate(lead_names):
            metrics = mean_metrics(batches, lead_idx)
            row = {
                "checkpoint": name,
                "checkpoint_source": meta.get("source", ""),
                "checkpoint_epoch": meta.get("epoch", ""),
                "checkpoint_path": meta.get("path", ""),
                "lead": lead,
                "n_batches": len(batches),
                **metrics,
            }
            row["combined_crps"] = 0.5 * (row["pr_crps"] + row["t2m_crps"])
            row["combined_rmse"] = 0.5 * (row["pr_rmse"] + row["t2m_rmse"])
            rows.append(row)

        for batch_idx, batch_metrics in enumerate(batches):
            for lead_idx, lead in enumerate(lead_names):
                metrics = batch_metrics[lead_idx]
                detail_rows.append({
                    "checkpoint": name,
                    "checkpoint_source": meta.get("source", ""),
                    "checkpoint_epoch": meta.get("epoch", ""),
                    "checkpoint_path": meta.get("path", ""),
                    "batch": batch_idx,
                    "lead": lead,
                    **metrics,
                })

    summary = pd.DataFrame(rows)
    total_summary = summary[summary["lead"] == "total"].copy()
    total_summary = total_summary.sort_values("combined_crps").reset_index(drop=True)
    total_summary.insert(0, "rank", np.arange(1, len(total_summary) + 1))
    detail = pd.DataFrame(detail_rows)

    summary_path = os.path.join(output_dir, f"checkpoint_pure_noise_v9_conus_{year}_{sample_tag}_summary.csv")
    detail_path = os.path.join(output_dir, f"checkpoint_pure_noise_v9_conus_{year}_{sample_tag}_detail.csv")
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    detail.to_csv(detail_path, index=False, float_format="%.4f")
    return summary_path, detail_path, total_summary


def main():
    parser = argparse.ArgumentParser(description="Compare CONUS v9 checkpoints under pure random noise.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv9.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--epochs", type=str, default="70,80,90,100,110,120,130,140")
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--best_checkpoint", type=str, default="best_flow_ckpt.pt")
    parser.add_argument("--skip_best", action="store_true")
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--ode_batch_size", type=int, default=120)
    parser.add_argument("--batch_limit", type=int, default=12, help="<=0 means no limit.")
    parser.add_argument("--full-year", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any requested checkpoint is missing.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_diag", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    if "DATA_DIR_OVERRIDE" in os.environ:
        config["data_dir"] = os.environ["DATA_DIR_OVERRIDE"]

    output_dir = args.output_dir or config["output_dir"]
    validate_current_sa_v9_config(config, output_dir)
    target_domain = config.get("target_domain")
    target_domain_bounds = config.get("target_domain_bounds")
    domain_info = resolve_target_domain(target_domain, target_domain_bounds)
    lats, _ = get_target_domain_coords(target_domain, target_domain_bounds)
    t2m_target_mode, t2m_residual_min, t2m_residual_max = resolve_t2m_residual_bounds(config)

    dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year,
        end_year=args.year,
        normalize=True,
        preload=False,
        stats_file=config.get("stats_file", "v9_conus_125w66w_24n50n_global_local_stats.pt"),
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
    batch_limit = None if args.batch_limit is not None and args.batch_limit <= 0 else args.batch_limit
    sample_tag = "full_year" if args.full_year else "monthly"

    checkpoints = resolve_checkpoints(
        output_dir,
        parse_epoch_list(args.epochs),
        args.checkpoints,
        include_best=not args.skip_best,
        best_checkpoint=args.best_checkpoint,
        allow_missing=not args.strict,
    )

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
    flow_matcher = CustomFlowMatcher(device=device)
    area_weights = get_area_weights(lats, device)

    print("\n" + "=" * 96)
    print("CONUS v9 checkpoint sweep: pure random noise only")
    print(f"  Config       : {args.config}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Data dir     : {config['data_dir']}")
    print(f"  Domain       : {target_domain or 'global'} ({len(dataset.lats)}x{len(dataset.lons)})")
    print(f"  T2M target   : {t2m_target_mode} ({t2m_residual_min:.3f} .. {t2m_residual_max:.3f})")
    print(f"  Year         : {args.year}")
    print(f"  Sampling     : {sample_tag}")
    print(f"  Batch limit  : {batch_limit if batch_limit is not None else 'none'}")
    print(f"  Ensembles    : {args.num_ensemble}")
    print(f"  ODE steps    : {args.num_steps}")
    print("  Checkpoints  :")
    for label, path in checkpoints:
        print(f"    {label:<8} {path}")
    print("=" * 96 + "\n")

    results = {"0. GEOS": []}
    metadata = {
        "0. GEOS": {
            "source": "geos_baseline",
            "epoch": "",
            "path": "",
        }
    }
    cached_batches = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Loading batches")):
        if batch_limit is not None and batch_idx >= batch_limit:
            break
        cached_batches.append(batch)
        results["0. GEOS"].append(geos_baseline(batch, device, area_weights))

    for ckpt_idx, (source_label, ckpt_path) in enumerate(checkpoints):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        epoch = checkpoint.get("epoch", "unknown")
        name = format_checkpoint_name(source_label, epoch)
        results[name] = []
        metadata[name] = {
            "source": source_label,
            "epoch": epoch,
            "path": ckpt_path,
        }
        print(f"\n🧪 Evaluating {name}: {ckpt_path} (epoch={epoch})")

        for batch_idx, batch in enumerate(tqdm(cached_batches, desc=name)):
            seed = int(args.seed) + batch_idx
            metrics = run_pure_noise_checkpoint(
                model=model,
                flow_matcher=flow_matcher,
                batch=batch,
                device=device,
                num_ensemble=args.num_ensemble,
                num_steps=args.num_steps,
                ode_batch_size=args.ode_batch_size,
                area_weights=area_weights,
                seed=seed,
                t2m_target_mode=t2m_target_mode,
                t2m_residual_min=t2m_residual_min,
                t2m_residual_max=t2m_residual_max,
                print_diag=args.print_diag and ckpt_idx == 0 and batch_idx == 0,
            )
            results[name].append(metrics)

        total = mean_metrics(results[name], 0)
        print(f"  {name:<20} {format_total([total])}")

    summary_path, detail_path, ranking = save_results(results, metadata, output_dir, args.year, sample_tag)

    print("\nFinal ranking by combined CRPS:")
    print(ranking.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved detail CSV : {detail_path}")


if __name__ == "__main__":
    main()
