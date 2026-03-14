#!/usr/bin/env python3
"""
V4-style checkpoint sweep under pure random noise for the multi-target flow model.

This reuses the same evaluation setup as compare_noise_v4_multi.py, but instead of
comparing different noise strategies for one model, it compares different model
checkpoints using only pure Gaussian noise.
"""

import argparse
import glob
import os
import re
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

from dataset_flow_multi import S2SHybridDataset
from flow_matching_multi import CustomFlowMatcher, FlowMatchingModel
from train_flow_multiv1 import compute_crps, compute_rmse


def extract_epoch(path):
    match = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def evenly_select(items, k):
    if k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[0]]

    indices = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    selected = []
    used = set()
    for idx in indices:
        if idx in used:
            continue
        used.add(idx)
        selected.append(items[idx])

    for idx, item in enumerate(items):
        if len(selected) >= k:
            break
        if idx not in used:
            selected.append(item)

    selected.sort(key=lambda path: items.index(path))
    return selected


def resolve_checkpoints(output_dir, checkpoint_args, checkpoint_glob, max_checkpoints):
    if checkpoint_args:
        resolved = []
        for ckpt in checkpoint_args:
            ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(output_dir, ckpt)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            resolved.append(ckpt_path)
        return resolved

    candidates = sorted(
        glob.glob(os.path.join(output_dir, checkpoint_glob)),
        key=lambda path: (extract_epoch(path), path),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matched {checkpoint_glob!r} inside {output_dir}"
        )
    return evenly_select(candidates, max_checkpoints)


def checkpoint_label(index, ckpt_path):
    epoch = extract_epoch(ckpt_path)
    if epoch >= 0:
        return f"{index}. E{epoch}"
    return f"{index}. {os.path.basename(ckpt_path)}"


def compute_area_weights(height, device):
    lats = np.linspace(-90, 90, height)
    cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
    area_weights = torch.from_numpy(cos_weights).float().to(device)
    area_weights = area_weights / area_weights.sum()
    return area_weights.view(1, 1, height, 1)


@torch.no_grad()
def run_pure_random_strategy(
    model,
    flow_matcher,
    batch,
    device,
    num_ensemble,
    num_steps,
    use_var_head=False,
    print_diag=False,
):
    model.eval()

    vB = batch["y_target"].shape[0]
    height, width = batch["y_target"].shape[-2:]
    num_inits = vB // 4

    true_target_raw = batch["target_raw_full"][0::4].to(device)
    true_target_pr = true_target_raw[:, 0]
    true_target_t2m = true_target_raw[:, 1]

    fx_obs = batch["x_obs"].to(device)
    fx_geos = batch["x_geos"].to(device)
    fx_geos_cat = fx_geos.view(vB, -1, height, width)

    f_month = batch["month"].to(device).float()
    fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, height, width)
    fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, height, width)

    fl_idx = batch["lead_idx"].to(device).float()
    f_lead_val = (fl_idx / 1.5) - 1.0
    f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, height, width)

    fx_cond = torch.cat([fx_obs, fx_geos_cat, fsin_month, fcos_month, f_lead_channel], dim=1)
    fx_cond_expanded = (
        fx_cond.unsqueeze(1)
        .expand(vB, num_ensemble, -1, height, width)
        .reshape(vB * num_ensemble, -1, height, width)
        .clone()
    )
    lead_idx_expanded = (
        batch["lead_idx"].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
    )

    noise_expanded = torch.randn((vB * num_ensemble, 2, height, width), device=device)

    if print_diag:
        print(f"\n    📊 [Noise Diag] Shape: {list(noise_expanded.shape)}")
        for channel in range(noise_expanded.shape[1]):
            noise_slice = noise_expanded[:, channel]
            print(
                f"       Ch{channel}: Mean={noise_slice.mean():.4f}, Std={noise_slice.std():.4f}, "
                f"Min={noise_slice.min():.4f}, Max={noise_slice.max():.4f}"
            )

    p_x1_expanded = flow_matcher.euler_solve(
        model,
        noise_expanded,
        fx_cond_expanded,
        num_steps=num_steps,
        lead_idx=lead_idx_expanded,
        apply_flow_variance=use_var_head,
    )

    if print_diag:
        print(f"    📊 [ODE Output] Shape: {list(p_x1_expanded.shape)}")
        for channel in range(p_x1_expanded.shape[1]):
            pred_slice = p_x1_expanded[:, channel]
            print(
                f"       Ch{channel}: Mean={pred_slice.mean():.4f}, Std={pred_slice.std():.4f}, "
                f"Min={pred_slice.min():.4f}, Max={pred_slice.max():.4f}"
            )

    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, height, width)

    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    p_x1_pr = p_x1_batch[:, :, 0]
    week_sqrt = ((p_x1_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)

    t2m_min, t2m_max = 200.0, 320.0
    p_x1_t2m = p_x1_batch[:, :, 1]
    week_t2m = ((p_x1_t2m + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min

    ensemble_pr = week_precip.transpose(0, 1).reshape(num_ensemble, num_inits, 4, height, width)
    ensemble_t2m = week_t2m.transpose(0, 1).reshape(num_ensemble, num_inits, 4, height, width)

    area_weights = compute_area_weights(height, device)

    pr_crps = compute_crps(ensemble_pr, true_target_pr, area_weights)
    pr_rmse = compute_rmse(ensemble_pr.mean(dim=0), true_target_pr, area_weights)
    t2m_crps = compute_crps(ensemble_t2m, true_target_t2m, area_weights)
    t2m_rmse = compute_rmse(ensemble_t2m.mean(dim=0), true_target_t2m, area_weights)

    out = [{
        "pr_crps": pr_crps,
        "pr_rmse": pr_rmse,
        "t2m_crps": t2m_crps,
        "t2m_rmse": t2m_rmse,
    }]

    for lead in range(4):
        ens_pr_lead = ensemble_pr[:, :, lead:lead + 1, :, :]
        tgt_pr_lead = true_target_pr[:, lead:lead + 1, :, :]
        ens_t2m_lead = ensemble_t2m[:, :, lead:lead + 1, :, :]
        tgt_t2m_lead = true_target_t2m[:, lead:lead + 1, :, :]

        out.append({
            "pr_crps": compute_crps(ens_pr_lead, tgt_pr_lead, area_weights),
            "pr_rmse": compute_rmse(ens_pr_lead.mean(dim=0), tgt_pr_lead, area_weights),
            "t2m_crps": compute_crps(ens_t2m_lead, tgt_t2m_lead, area_weights),
            "t2m_rmse": compute_rmse(ens_t2m_lead.mean(dim=0), tgt_t2m_lead, area_weights),
        })

    return out


def compute_geos_baseline(batch, device):
    true_target_raw = batch["target_raw_full"][0::4].to(device)
    geos_ens_sample = batch["geos_ens_raw"][0::4].to(device)
    height = true_target_raw.shape[-2]
    area_weights = compute_area_weights(height, device)

    geos_pr_ens = geos_ens_sample[:, :, 0].transpose(0, 1)
    geos_t2m_ens = geos_ens_sample[:, :, 1].transpose(0, 1)
    tgt_pr = true_target_raw[:, 0]
    tgt_t2m = true_target_raw[:, 1]

    out = [{
        "pr_crps": compute_crps(geos_pr_ens, tgt_pr, area_weights),
        "pr_rmse": compute_rmse(geos_pr_ens.mean(dim=0), tgt_pr, area_weights),
        "t2m_crps": compute_crps(geos_t2m_ens, tgt_t2m, area_weights),
        "t2m_rmse": compute_rmse(geos_t2m_ens.mean(dim=0), tgt_t2m, area_weights),
    }]

    for lead in range(4):
        out.append({
            "pr_crps": compute_crps(geos_pr_ens[:, :, lead:lead + 1], tgt_pr[:, lead:lead + 1], area_weights),
            "pr_rmse": compute_rmse(geos_pr_ens.mean(dim=0)[:, lead:lead + 1], tgt_pr[:, lead:lead + 1], area_weights),
            "t2m_crps": compute_crps(geos_t2m_ens[:, :, lead:lead + 1], tgt_t2m[:, lead:lead + 1], area_weights),
            "t2m_rmse": compute_rmse(geos_t2m_ens.mean(dim=0)[:, lead:lead + 1], tgt_t2m[:, lead:lead + 1], area_weights),
        })

    return out, tgt_pr, tgt_t2m, geos_ens_sample


def mean_metric(results, name, lead_index):
    return {
        "pr_crps": np.mean([entry[lead_index]["pr_crps"] for entry in results[name]]),
        "pr_rmse": np.mean([entry[lead_index]["pr_rmse"] for entry in results[name]]),
        "t2m_crps": np.mean([entry[lead_index]["t2m_crps"] for entry in results[name]]),
        "t2m_rmse": np.mean([entry[lead_index]["t2m_rmse"] for entry in results[name]]),
    }


def print_table(results, names, months):
    blue = "\033[94m"
    orange = "\033[38;5;214m"
    bold = "\033[1m"
    reset = "\033[0m"
    total_width = max(140, 22 + len(names) * 37)

    print(f"\n{'─' * total_width}")
    header_parts = [f"  {'Sample':<8} {'Mon':>4} |"]
    for name in names:
        header_parts.append(f" {name + ' PR':>17} {name + ' T2M':>17}")
    print("".join(header_parts))
    print(f"{'─' * total_width}")

    def fmt_row(label, row_values):
        pr_crps_vals = [value["pr_crps"] for value in row_values]
        pr_rmse_vals = [value["pr_rmse"] for value in row_values]
        t2m_crps_vals = [value["t2m_crps"] for value in row_values]
        t2m_rmse_vals = [value["t2m_rmse"] for value in row_values]

        best_pr_crps = int(np.argmin(pr_crps_vals))
        best_pr_rmse = int(np.argmin(pr_rmse_vals))
        best_t2m_crps = int(np.argmin(t2m_crps_vals))
        best_t2m_rmse = int(np.argmin(t2m_rmse_vals))

        parts = []
        for idx, value in enumerate(row_values):
            pr_crps = f"{value['pr_crps']:>7.4f}"
            pr_rmse = f"({value['pr_rmse']:>7.4f})"
            t2m_crps = f"{value['t2m_crps']:>7.4f}"
            t2m_rmse = f"({value['t2m_rmse']:>7.4f})"

            if idx == best_pr_crps:
                pr_crps = f"{blue}{bold}{pr_crps}{reset}"
            if idx == best_pr_rmse:
                pr_rmse = f"{orange}{bold}{pr_rmse}{reset}"
            if idx == best_t2m_crps:
                t2m_crps = f"{blue}{bold}{t2m_crps}{reset}"
            if idx == best_t2m_rmse:
                t2m_rmse = f"{orange}{bold}{t2m_rmse}{reset}"

            parts.append(f"{pr_crps} {pr_rmse} {t2m_crps} {t2m_rmse}")

        print(f"  {label:<13} | {' | '.join(parts)}")

    num_batches = len(results[names[0]])
    for batch_index in range(num_batches):
        month = months[batch_index]
        fmt_row(f"Batch {batch_index:<2} {month:>4}", [results[name][batch_index][0] for name in names])
        for week in range(4):
            fmt_row(f"    W{week + 1}", [results[name][batch_index][week + 1] for name in names])

        n_done = batch_index + 1
        run_avg_total = []
        for name in names:
            run_avg_total.append({
                "pr_crps": np.mean([entry[0]["pr_crps"] for entry in results[name][:n_done]]),
                "pr_rmse": np.mean([entry[0]["pr_rmse"] for entry in results[name][:n_done]]),
                "t2m_crps": np.mean([entry[0]["t2m_crps"] for entry in results[name][:n_done]]),
                "t2m_rmse": np.mean([entry[0]["t2m_rmse"] for entry in results[name][:n_done]]),
            })
        fmt_row(f"  RunAvg({n_done})", run_avg_total)

        for week in range(4):
            run_avg_week = []
            for name in names:
                run_avg_week.append({
                    "pr_crps": np.mean([entry[week + 1]["pr_crps"] for entry in results[name][:n_done]]),
                    "pr_rmse": np.mean([entry[week + 1]["pr_rmse"] for entry in results[name][:n_done]]),
                    "t2m_crps": np.mean([entry[week + 1]["t2m_crps"] for entry in results[name][:n_done]]),
                    "t2m_rmse": np.mean([entry[week + 1]["t2m_rmse"] for entry in results[name][:n_done]]),
                })
            fmt_row(f"  AvgW{week + 1}({n_done})", run_avg_week)

        print(f"  {'─' * total_width}")


def save_csvs(results, names, months, output_dir, year):
    lead_suffixes = [" (Total)", " (W1)", " (W2)", " (W3)", " (W4)"]

    detailed_rows = []
    for batch_index in range(len(results[names[0]])):
        row = {"batch": batch_index, "month": months[batch_index]}
        for name in names:
            for lead_index, suffix in enumerate(lead_suffixes):
                metrics = results[name][batch_index][lead_index]
                row[f"{name}{suffix} PR_CRPS"] = metrics["pr_crps"]
                row[f"{name}{suffix} PR_RMSE"] = metrics["pr_rmse"]
                row[f"{name}{suffix} T2M_CRPS"] = metrics["t2m_crps"]
                row[f"{name}{suffix} T2M_RMSE"] = metrics["t2m_rmse"]
        detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    mean_row = {col: np.mean(detailed_df[col]) for col in detailed_df.columns if col != "batch"}
    mean_row["batch"] = "MEAN"
    detailed_df.loc[len(detailed_df)] = mean_row

    summary_rows = []
    for name in names:
        total = mean_metric(results, name, 0)
        summary = {
            "name": name,
            "combined_crps": 0.5 * (total["pr_crps"] + total["t2m_crps"]),
            "combined_rmse": 0.5 * (total["pr_rmse"] + total["t2m_rmse"]),
            "pr_crps": total["pr_crps"],
            "pr_rmse": total["pr_rmse"],
            "t2m_crps": total["t2m_crps"],
            "t2m_rmse": total["t2m_rmse"],
        }
        for week in range(4):
            weekly = mean_metric(results, name, week + 1)
            summary[f"w{week + 1}_pr_crps"] = weekly["pr_crps"]
            summary[f"w{week + 1}_pr_rmse"] = weekly["pr_rmse"]
            summary[f"w{week + 1}_t2m_crps"] = weekly["t2m_crps"]
            summary[f"w{week + 1}_t2m_rmse"] = weekly["t2m_rmse"]
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("combined_crps").reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

    detailed_path = os.path.join(output_dir, f"checkpoint_pure_noise_detailed_{year}.csv")
    summary_path = os.path.join(output_dir, f"checkpoint_pure_noise_summary_{year}.csv")
    detailed_df.to_csv(detailed_path, float_format="%.4f", index=False)
    summary_df.to_csv(summary_path, float_format="%.4f", index=False)

    return detailed_path, summary_path, summary_df


def main():
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints under pure random noise.")
    parser.add_argument("--output_dir", type=str, default="ml_output_flowmulti")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Checkpoint filenames or absolute paths.")
    parser.add_argument("--checkpoint_glob", type=str, default="periodic_ckpt_epoch_*.pt")
    parser.add_argument("--max_checkpoints", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=12)
    parser.add_argument("--use_var_head", action="store_true", help="Apply the flow variance head during ODE solve.")
    parser.add_argument("--print_diag", action="store_true", help="Print noise/output diagnostics for the first checkpoint and batch.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    stats_file = config.get("stats_file", "v1_multi_global_stats.pt")
    test_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year,
        end_year=args.year,
        normalize=True,
        preload=False,
        stats_file=stats_file,
        subsample_monthly=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    checkpoint_paths = resolve_checkpoints(
        args.output_dir, args.checkpoints, args.checkpoint_glob, args.max_checkpoints
    )
    checkpoint_names = [checkpoint_label(index + 1, path) for index, path in enumerate(checkpoint_paths)]

    print("\n" + "=" * 96)
    print("🚀 CHECKPOINT SWEEP (PURE RANDOM ONLY)")
    print(f"   Output Dir   : {os.path.abspath(args.output_dir)}")
    print(f"   Year         : {args.year}")
    print(f"   Ensemble     : {args.num_ensemble}")
    print(f"   ODE Steps    : {args.num_steps}")
    print(f"   Use Var Head : {args.use_var_head}")
    print("   Checkpoints  :")
    for name, path in zip(checkpoint_names, checkpoint_paths):
        print(f"     {name:<10} {os.path.abspath(path)}")
    print("=" * 96 + "\n")

    model = FlowMatchingModel(in_channels=41, out_channels=2).to(device)
    flow_matcher = CustomFlowMatcher(device=device)

    results = {"0. GEOS": []}
    months = []

    for batch_index, batch in enumerate(test_loader):
        if batch_index >= args.max_batches:
            break
        months.append(int(batch["month"][0].item()))
        geos_out, tgt_pr, tgt_t2m, geos_ens_sample = compute_geos_baseline(batch, device)
        results["0. GEOS"].append(geos_out)

        if batch_index == 0:
            month = int(batch["month"][0].item())
            print(f"🔍 [Diagnostic - Month {month}]")
            print(f"   Target PR   : Min={tgt_pr.min():.2f}, Max={tgt_pr.max():.2f}, Mean={tgt_pr.mean():.2f}")
            print(f"   Target T2M  : Min={tgt_t2m.min():.2f}, Max={tgt_t2m.max():.2f}, Mean={tgt_t2m.mean():.2f}")
            print(
                f"   GEOS PR     : Min={geos_ens_sample[:, :, 0].min():.2f}, "
                f"Max={geos_ens_sample[:, :, 0].max():.2f}, Mean={geos_ens_sample[:, :, 0].mean():.2f}"
            )
            print(
                f"   GEOS T2M    : Min={geos_ens_sample[:, :, 1].min():.2f}, "
                f"Max={geos_ens_sample[:, :, 1].max():.2f}, Mean={geos_ens_sample[:, :, 1].mean():.2f}"
            )
            print("   " + "─" * 50 + "\n")

    for label, ckpt_path in zip(checkpoint_names, checkpoint_paths):
        print(f"\n🧪 Evaluating {label} from {os.path.basename(ckpt_path)}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        checkpoint_results = []

        for batch_index, batch in enumerate(tqdm(test_loader, desc=label, ncols=100)):
            if batch_index >= args.max_batches:
                break
            checkpoint_results.append(
                run_pure_random_strategy(
                    model,
                    flow_matcher,
                    batch,
                    device,
                    args.num_ensemble,
                    args.num_steps,
                    use_var_head=args.use_var_head,
                    print_diag=args.print_diag and batch_index == 0 and label == checkpoint_names[0],
                )
            )
            torch.cuda.empty_cache()

        results[label] = checkpoint_results
        total = mean_metric(results, label, 0)
        print(
            f"   Mean Total -> PR CRPS {total['pr_crps']:.4f}, T2M CRPS {total['t2m_crps']:.4f}, "
            f"PR RMSE {total['pr_rmse']:.4f}, T2M RMSE {total['t2m_rmse']:.4f}"
        )
        del ckpt
        torch.cuda.empty_cache()

    all_names = ["0. GEOS"] + checkpoint_names
    print_table(results, all_names, months)

    detailed_path, summary_path, summary_df = save_csvs(results, all_names, months, args.output_dir, args.year)
    print("\n🏁 Final ranking by combined CRPS:")
    for _, row in summary_df.iterrows():
        print(
            f"   #{int(row['rank'])} {row['name']:<10} "
            f"Combined={row['combined_crps']:.4f}  "
            f"PR={row['pr_crps']:.4f}  T2M={row['t2m_crps']:.4f}"
        )

    print(f"\n💾 Detailed CSV saved to: {detailed_path}")
    print(f"💾 Summary CSV saved to : {summary_path}")


if __name__ == "__main__":
    main()
