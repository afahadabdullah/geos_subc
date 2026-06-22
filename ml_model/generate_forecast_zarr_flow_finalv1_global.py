#!/usr/bin/env python3
"""
Generate global flow_finalv1_global ensemble forecasts and save yearly Zarr stores.

The output is intended for reusable verification work. Each yearly store contains
model ensemble forecasts, raw GEOS ensemble forecasts, and verifying obs on the
same init/lead/lat/lon grid.
"""

import argparse
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import noise_utils_multi_finalv1_global
from compare_noise_flow_finalv1_global import (
    decode_multi,
    load_noise_context,
    resolve_checkpoint,
    resolve_t2m_residual_bounds,
    validate_global_config,
)
from dataset_flow_finalv1_global import S2SHybridDataset, resolve_target_domain
from flow_matching_finalv1_global import CustomFlowMatcher, FlowMatchingModel
from train_flow_finalv1_global import (
    euler_solve_chunked,
    geos_condition_channel_count,
    get_batch_global_context,
    select_geos_condition,
)


DEFAULT_OUT_DIR = "dataprocess/gen_flow_finalv1_global_junjul_e10clim_e100eval_s50"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate flow_finalv1_global forecasts to yearly global Zarr stores.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_finalv1_global.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best_flow_ckpt.pt")
    parser.add_argument("--start_year", type=int, default=2005)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--skip_years", type=str, default="2017", help="Comma-separated years to skip.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--months", type=str, default="6,7", help="Comma-separated init months to generate.")
    parser.add_argument("--clim_num_ensemble", type=int, default=10, help="Members/year before --eval_start_year.")
    parser.add_argument("--eval_num_ensemble", type=int, default=100, help="Members/year from --eval_start_year onward.")
    parser.add_argument("--eval_start_year", type=int, default=2021)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8, help="Must be a multiple of 4.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ensemble_chunk_size", type=int, default=30)
    parser.add_argument("--ode_batch_size", type=int, default=120)
    parser.add_argument("--member_chunk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--max_runtime_minutes",
        type=float,
        default=None,
        help="Stop cleanly after this many minutes, leaving .zarr.tmp resumable.",
    )
    parser.add_argument("--pure_noise", action="store_true", help="Disable EOF-LHS and variance-head scaling.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_months(text):
    months = tuple(sorted({int(item.strip()) for item in str(text).split(",") if item.strip()}))
    if not months:
        raise ValueError("--months cannot be empty")
    bad = [m for m in months if m < 1 or m > 12]
    if bad:
        raise ValueError(f"Invalid month(s): {bad}")
    return months


def parse_years(text):
    years = {int(item.strip()) for item in str(text or "").split(",") if item.strip()}
    return years


def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def batch_init_dates(batch):
    years = batch["year"][0::4].cpu().numpy().astype(int)
    months = batch["month"][0::4].cpu().numpy().astype(int)
    days = batch["day"][0::4].cpu().numpy().astype(int)
    return pd.to_datetime([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)])


def expected_init_dates(dataset):
    return pd.to_datetime([sample["date"] for sample in dataset.samples[0::4]]).normalize()


def inspect_tmp_store(path):
    ds = xr.open_zarr(path, consolidated=False, chunks=None)
    try:
        init_dates = pd.to_datetime(ds["init"].values).normalize()
        ensemble_size = int(ds.sizes.get("ensemble", 0))
        lead_size = int(ds.sizes.get("lead", 0))
        lat_size = int(ds.sizes.get("lat", 0))
        lon_size = int(ds.sizes.get("lon", 0))
    finally:
        ds.close()
    return {
        "init_dates": init_dates,
        "ensemble_size": ensemble_size,
        "lead_size": lead_size,
        "lat_size": lat_size,
        "lon_size": lon_size,
    }


def slice_batch_from_init(batch, skip_inits):
    if skip_inits <= 0:
        return batch
    skip_samples = int(skip_inits) * 4
    batch_size = int(batch["y_target"].shape[0])
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            sliced[key] = value[skip_samples:]
        else:
            sliced[key] = value
    return sliced


def verify_batch_dates(expected_dates, batch):
    actual_dates = batch_init_dates(batch).normalize()
    if len(expected_dates) != len(actual_dates) or not np.array_equal(expected_dates.values, actual_dates.values):
        raise ValueError(
            "Dataset/init-date ordering mismatch while writing Zarr. "
            f"Expected {list(expected_dates.strftime('%Y-%m-%d'))}, "
            f"got {list(actual_dates.strftime('%Y-%m-%d'))}."
        )


def deadline_reached(deadline):
    return deadline is not None and time.monotonic() >= deadline


def valid_times(init_dates):
    lead_days = pd.to_timedelta(np.array([7, 14, 21, 28]), unit="D")
    return init_dates.values[:, None] + lead_days.values[None, :]


def get_autocast_context(device, mixed_precision):
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_condition(batch, device, lead_selective_geos=True):
    x_obs = batch["x_obs"].to(device)
    x_geos = batch["x_geos"].to(device)
    batch_size = x_obs.shape[0]
    height, width = x_obs.shape[-2:]

    months = batch["month"].to(device).float()
    sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)

    lead_idx = batch["lead_idx"].to(device).long()
    x_geos_flat = select_geos_condition(x_geos, lead_idx, lead_selective_geos)
    lead_val = (lead_idx.float() / 1.5) - 1.0
    lead_channel = lead_val.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
    return torch.cat([x_obs, x_geos_flat, sin_month, cos_month, lead_channel], dim=1), lead_idx


def generate_noise(batch, num_ensemble, device, year, use_eof_lhs_noise, noise_context, rho_pr, rho_t2m):
    height, width = batch["y_target"].shape[-2:]
    if not use_eof_lhs_noise:
        return torch.randn((batch["y_target"].shape[0] * num_ensemble, 2, height, width), device=device)

    eof_noise = noise_utils_multi_finalv1_global.generate_dynamic_multimodal_noise_multi(
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
        year=year,
        use_lhs=True,
        orthogonalize_lhs=True,
    )
    return noise_utils_multi_finalv1_global.mix_noise_with_random_multi(eof_noise, rho_pr, rho_t2m)


@torch.no_grad()
def generate_batch_forecast(
    batch,
    model,
    flow_matcher,
    device,
    config,
    noise_context,
    num_ensemble,
    num_steps,
    ensemble_chunk_size,
    ode_batch_size,
    use_flow_variance,
    use_eof_lhs_noise,
    t2m_target_mode,
    t2m_residual_min,
    t2m_residual_max,
):
    x_cond, lead_idx = build_condition(
        batch,
        device,
        lead_selective_geos=bool(config.get("lead_selective_geos", True)),
    )
    global_context = get_batch_global_context(batch, device)
    batch_size, _, height, width = x_cond.shape
    num_inits = batch_size // 4

    rho_pr = float(config.get("validation_rho_pr", 0.25))
    rho_t2m = float(config.get("validation_rho_t2m", 0.08))
    beta_pr = float(config.get("validation_var_beta_pr", 0.45))
    beta_t2m = float(config.get("validation_var_beta_t2m", 0.03))
    coarse_kernel = config.get("validation_variance_coarse_kernel", 8)
    coarse_kernel = None if coarse_kernel in {None, "none", "None"} else int(coarse_kernel)
    mixed_precision = str(config.get("mixed_precision", "no")).lower()

    pr_chunks = []
    t2m_chunks = []
    ensemble_chunk_size = max(1, min(int(ensemble_chunk_size), int(num_ensemble)))

    for ens_start in range(0, num_ensemble, ensemble_chunk_size):
        ens_count = min(ensemble_chunk_size, num_ensemble - ens_start)
        x_cond_expanded = (
            x_cond.unsqueeze(1)
            .expand(batch_size, ens_count, -1, height, width)
            .reshape(batch_size * ens_count, -1, height, width)
            .contiguous()
        )
        lead_expanded = lead_idx.unsqueeze(1).expand(batch_size, ens_count).reshape(-1).long()

        if global_context is not None:
            gc, gh, gw = global_context.shape[1:]
            global_expanded = (
                global_context.unsqueeze(1)
                .expand(batch_size, ens_count, gc, gh, gw)
                .reshape(batch_size * ens_count, gc, gh, gw)
                .contiguous()
            )
        else:
            global_expanded = None

        current_year = int(batch["year"][0].item())
        noise = generate_noise(
            batch=batch,
            num_ensemble=ens_count,
            device=device,
            year=current_year,
            use_eof_lhs_noise=use_eof_lhs_noise,
            noise_context=noise_context,
            rho_pr=rho_pr,
            rho_t2m=rho_t2m,
        )

        with get_autocast_context(device, mixed_precision):
            pred_norm = euler_solve_chunked(
                flow_matcher,
                model,
                noise,
                x_cond_expanded,
                num_steps=int(num_steps),
                lead_idx=lead_expanded,
                apply_flow_variance=use_flow_variance,
                variance_beta=(beta_pr, beta_t2m),
                variance_coarse_kernel=coarse_kernel,
                chunk_size=int(ode_batch_size),
                global_context=global_expanded,
            )

        pred_norm = pred_norm.view(batch_size, ens_count, 2, height, width).float()
        precip, t2m = decode_multi(
            pred_norm,
            batch,
            device,
            t2m_target_mode=t2m_target_mode,
            t2m_residual_min=t2m_residual_min,
            t2m_residual_max=t2m_residual_max,
        )
        pr_chunks.append(
            precip.reshape(num_inits, 4, ens_count, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .cpu()
            .numpy()
        )
        t2m_chunks.append(
            t2m.reshape(num_inits, 4, ens_count, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .cpu()
            .numpy()
        )

    return np.concatenate(pr_chunks, axis=1), np.concatenate(t2m_chunks, axis=1)


def make_batch_dataset(batch, model_pr, model_t2m, lats, lons, attrs, member_chunk):
    init_dates = batch_init_dates(batch)
    target_raw = batch["target_raw_full"][0::4].float().numpy()
    obs_pr = target_raw[:, 0]
    obs_t2m = target_raw[:, 1]
    geos_raw = batch["geos_ens_raw"][0::4].float().numpy()
    geos_pr = geos_raw[:, :, 0]
    geos_t2m = geos_raw[:, :, 1]

    num_ensemble = model_pr.shape[1]
    geos_members = geos_pr.shape[1]
    lead_values = np.arange(1, 5, dtype=np.int32)
    encoding = {
        "model_pr": {"dtype": "float32", "chunks": (1, min(member_chunk, num_ensemble), 4, len(lats), len(lons))},
        "model_t2m": {"dtype": "float32", "chunks": (1, min(member_chunk, num_ensemble), 4, len(lats), len(lons))},
        "geos_pr": {"dtype": "float32", "chunks": (1, geos_members, 4, len(lats), len(lons))},
        "geos_t2m": {"dtype": "float32", "chunks": (1, geos_members, 4, len(lats), len(lons))},
        "obs_pr": {"dtype": "float32", "chunks": (1, 4, len(lats), len(lons))},
        "obs_t2m": {"dtype": "float32", "chunks": (1, 4, len(lats), len(lons))},
    }
    ds = xr.Dataset(
        data_vars={
            "model_pr": (("init", "ensemble", "lead", "lat", "lon"), model_pr.astype(np.float32, copy=False)),
            "model_t2m": (("init", "ensemble", "lead", "lat", "lon"), model_t2m.astype(np.float32, copy=False)),
            "geos_pr": (("init", "geos_member", "lead", "lat", "lon"), geos_pr.astype(np.float32, copy=False)),
            "geos_t2m": (("init", "geos_member", "lead", "lat", "lon"), geos_t2m.astype(np.float32, copy=False)),
            "obs_pr": (("init", "lead", "lat", "lon"), obs_pr.astype(np.float32, copy=False)),
            "obs_t2m": (("init", "lead", "lat", "lon"), obs_t2m.astype(np.float32, copy=False)),
        },
        coords={
            "init": init_dates.values,
            "ensemble": np.arange(1, num_ensemble + 1, dtype=np.int32),
            "geos_member": np.arange(1, geos_members + 1, dtype=np.int32),
            "lead": lead_values,
            "lat": np.asarray(lats, dtype=np.float32),
            "lon": np.asarray(lons, dtype=np.float32),
            "valid_time": (("init", "lead"), valid_times(init_dates)),
        },
        attrs=attrs,
    )
    ds["model_pr"].attrs.update({"units": "mm/day", "long_name": "model ensemble precipitation"})
    ds["geos_pr"].attrs.update({"units": "mm/day", "long_name": "GEOS ensemble precipitation"})
    ds["obs_pr"].attrs.update({"units": "mm/day", "long_name": "observed precipitation"})
    ds["model_t2m"].attrs.update({"units": "K", "long_name": "model ensemble 2m temperature"})
    ds["geos_t2m"].attrs.update({"units": "K", "long_name": "GEOS ensemble 2m temperature"})
    ds["obs_t2m"].attrs.update({"units": "K", "long_name": "observed 2m temperature"})
    return ds, encoding


def write_year(year, args, config, model, flow_matcher, device, checkpoint_path, checkpoint_meta, noise_context, deadline=None):
    months = parse_months(args.months)
    num_ensemble = int(args.eval_num_ensemble if year >= int(args.eval_start_year) else args.clim_num_ensemble)
    out_path = os.path.join(args.out_dir, f"{year}.zarr")
    tmp_path = out_path + ".tmp"
    t2m_target_mode, t2m_residual_min, t2m_residual_max = resolve_t2m_residual_bounds(config)
    dataset = S2SHybridDataset(
        data_root=args.data_dir,
        start_year=year,
        end_year=year,
        normalize=True,
        preload=False,
        stats_file=config.get("stats_file", "flow_finalv1_global_stats.pt"),
        subsample_monthly=False,
        target_domain=config.get("target_domain"),
        target_domain_bounds=config.get("target_domain_bounds"),
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min,
        t2m_residual_max=t2m_residual_max,
    )
    dataset.samples = [sample for sample in dataset.samples if int(sample["date"].month) in months]
    if len(dataset) == 0:
        raise RuntimeError(f"{year}: dataset indexed zero samples after filtering to months={months}")
    if len(dataset) % 4 != 0:
        raise RuntimeError(f"{year}: filtered sample count {len(dataset)} is not divisible by 4")
    if args.batch_size % 4 != 0:
        raise ValueError("--batch_size must be divisible by 4 so init dates stay grouped.")
    expected_dates = expected_init_dates(dataset)
    total_expected_inits = len(expected_dates)

    if os.path.exists(out_path):
        if not args.overwrite:
            print(f"✅ {year}: output exists at {out_path}. Skipping.")
            return True
        print(f"♻️ {year}: removing existing output {out_path}")
        remove_path(out_path)

    resume_offset = 0
    wrote_any = False
    if os.path.exists(tmp_path):
        if args.overwrite:
            print(f"♻️ {year}: removing existing temp output {tmp_path}")
            remove_path(tmp_path)
        else:
            tmp_info = inspect_tmp_store(tmp_path)
            if tmp_info["ensemble_size"] != num_ensemble:
                raise RuntimeError(
                    f"Cannot resume {year}: temp store has ensemble={tmp_info['ensemble_size']}, "
                    f"expected {num_ensemble}."
                )
            if tmp_info["lead_size"] != 4:
                raise RuntimeError(f"Cannot resume {year}: temp store has lead={tmp_info['lead_size']}, expected 4.")
            if tmp_info["lat_size"] != len(dataset.lats) or tmp_info["lon_size"] != len(dataset.lons):
                raise RuntimeError(
                    f"Cannot resume {year}: temp grid is {tmp_info['lat_size']}x{tmp_info['lon_size']}, "
                    f"expected {len(dataset.lats)}x{len(dataset.lons)}."
                )
            resume_offset = len(tmp_info["init_dates"])
            if resume_offset > total_expected_inits:
                raise RuntimeError(
                    f"Cannot resume {year}: temp store has {resume_offset} init dates, "
                    f"expected only {total_expected_inits}."
                )
            if not np.array_equal(tmp_info["init_dates"].values, expected_dates[:resume_offset].values):
                raise RuntimeError(
                    f"Cannot resume {year}: temp init dates do not match expected June/July prefix."
                )
            if resume_offset == total_expected_inits:
                print(f"✅ {year}: temp store already complete. Finalizing {out_path}.")
                os.rename(tmp_path, out_path)
                return True
            if resume_offset > 0:
                wrote_any = True
                print(f"♻️ {year}: resuming {tmp_path} from {resume_offset}/{total_expected_inits} init dates.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    use_eof_lhs_noise = not args.pure_noise
    use_flow_variance = not args.pure_noise
    attrs = {
        "generated_by": os.path.basename(__file__),
        "model_version": "flow_finalv1_global",
        "source_config": os.path.abspath(args.config),
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_meta.get("epoch", -1)),
        "checkpoint_is_variance_phase": bool(checkpoint_meta.get("is_variance_phase", False)),
        "init_months": ",".join(str(m) for m in months),
        "target_domain": str(config.get("target_domain")),
        "target_domain_bounds": str(config.get("target_domain_bounds")),
        "t2m_target_mode": str(t2m_target_mode),
        "num_ensemble": int(num_ensemble),
        "clim_num_ensemble": int(args.clim_num_ensemble),
        "eval_num_ensemble": int(args.eval_num_ensemble),
        "eval_start_year": int(args.eval_start_year),
        "num_steps": int(args.num_steps),
        "use_eof_lhs_noise": bool(use_eof_lhs_noise),
        "use_flow_variance": bool(use_flow_variance),
        "validation_rho_pr": float(config.get("validation_rho_pr", 0.25)),
        "validation_rho_t2m": float(config.get("validation_rho_t2m", 0.08)),
        "validation_var_beta_pr": float(config.get("validation_var_beta_pr", 0.45)),
        "validation_var_beta_t2m": float(config.get("validation_var_beta_t2m", 0.03)),
        "validation_variance_coarse_kernel": str(config.get("validation_variance_coarse_kernel", 8)),
    }

    total_inits = resume_offset
    seen_inits = 0
    pbar = tqdm(loader, desc=f"Generate flow_finalv1_global global {year}")
    for batch_idx, batch in enumerate(pbar):
        if deadline_reached(deadline):
            print(
                f"⏸️ {year}: soft runtime limit reached before batch {batch_idx}. "
                f"Leaving {tmp_path} for resume ({total_inits}/{total_expected_inits} init dates written)."
            )
            return False
        batch_size = batch["y_target"].shape[0]
        if batch_size % 4 != 0:
            raise ValueError(f"Batch {batch_idx} size {batch_size} is not divisible by 4.")
        num_batch_inits = batch_size // 4
        batch_start = seen_inits
        batch_end = batch_start + num_batch_inits
        seen_inits = batch_end
        if batch_end <= resume_offset:
            pbar.set_postfix(init_dates=total_inits)
            continue
        skip_inits = max(0, resume_offset - batch_start)
        write_batch = slice_batch_from_init(batch, skip_inits)
        write_start = batch_start + skip_inits
        write_end = batch_end
        verify_batch_dates(expected_dates[write_start:write_end], write_batch)
        model_pr, model_t2m = generate_batch_forecast(
            batch=write_batch,
            model=model,
            flow_matcher=flow_matcher,
            device=device,
            config=config,
            noise_context=noise_context,
            num_ensemble=num_ensemble,
            num_steps=args.num_steps,
            ensemble_chunk_size=args.ensemble_chunk_size,
            ode_batch_size=args.ode_batch_size,
            use_flow_variance=use_flow_variance,
            use_eof_lhs_noise=use_eof_lhs_noise,
            t2m_target_mode=t2m_target_mode,
            t2m_residual_min=t2m_residual_min,
            t2m_residual_max=t2m_residual_max,
        )
        ds_batch, encoding = make_batch_dataset(
            batch=write_batch,
            model_pr=model_pr,
            model_t2m=model_t2m,
            lats=dataset.lats,
            lons=dataset.lons,
            attrs=attrs,
            member_chunk=args.member_chunk,
        )
        if not wrote_any:
            ds_batch.to_zarr(tmp_path, mode="w", encoding=encoding)
            wrote_any = True
        else:
            ds_batch.to_zarr(tmp_path, mode="a", append_dim="init")
        batch_inits = model_pr.shape[0]
        total_inits += batch_inits
        ds_batch.close()
        pbar.set_postfix(init_dates=total_inits)

    if not wrote_any:
        raise RuntimeError(f"{year}: nothing was written")
    if total_inits != total_expected_inits:
        raise RuntimeError(
            f"{year}: wrote {total_inits} init dates, expected {total_expected_inits}. "
            f"Leaving temp store at {tmp_path} for inspection/resume."
        )
    os.rename(tmp_path, out_path)
    print(f"✅ {year}: saved {total_inits} init dates to {out_path}")
    return True


def main():
    args = parse_args()
    deadline = None
    if args.max_runtime_minutes is not None and args.max_runtime_minutes > 0:
        deadline = time.monotonic() + float(args.max_runtime_minutes) * 60.0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    if "DATA_DIR_OVERRIDE" in os.environ:
        config["data_dir"] = os.environ["DATA_DIR_OVERRIDE"]
    if args.data_dir is None:
        args.data_dir = config["data_dir"]
    else:
        config["data_dir"] = args.data_dir
    model_output_dir = args.model_output_dir or config["output_dir"]
    validate_global_config(config, model_output_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    domain_info = resolve_target_domain(config.get("target_domain"), config.get("target_domain_bounds"))
    t2m_target_mode, t2m_residual_min, t2m_residual_max = resolve_t2m_residual_bounds(config)
    probe_dataset = S2SHybridDataset(
        data_root=args.data_dir,
        start_year=args.start_year,
        end_year=args.start_year,
        normalize=True,
        preload=False,
        stats_file=config.get("stats_file", "flow_finalv1_global_stats.pt"),
        subsample_monthly=False,
        target_domain=config.get("target_domain"),
        target_domain_bounds=config.get("target_domain_bounds"),
        local_obs_variables=config.get("local_obs_variables"),
        global_context_variables=config.get("global_context_variables"),
        t2m_target_mode=t2m_target_mode,
        t2m_residual_min=t2m_residual_min,
        t2m_residual_max=t2m_residual_max,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_channels = int(probe_dataset.obs_channel_count)
    cond_channels = obs_channels + geos_condition_channel_count(bool(config.get("lead_selective_geos", True))) + 3
    model_in_channels = cond_channels + 2
    global_context_channels = int(probe_dataset.global_context_channel_count)
    block_channels = tuple(int(v) for v in config.get("unet_block_out_channels", [128, 256, 512, 768]))
    model = FlowMatchingModel(
        in_channels=model_in_channels,
        out_channels=2,
        block_out_channels=block_channels,
        sample_size=(len(probe_dataset.lats), len(probe_dataset.lons)),
        global_context_channels=global_context_channels,
        use_global_cross_attention=bool(config.get("use_global_cross_attention", True)),
        global_attention_heads=int(config.get("global_attention_heads", 4)),
        global_attention_layers=int(config.get("global_attention_layers", 1)),
    ).to(device)
    checkpoint_path = resolve_checkpoint(model_output_dir, args.checkpoint, None)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    flow_matcher = CustomFlowMatcher(device=device)
    noise_context = load_noise_context(config, domain_info)

    print("\n" + "=" * 88)
    print("Global flow_finalv1_global forecast Zarr generation")
    print(f"  Config       : {args.config}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Model output : {model_output_dir}")
    print(f"  Checkpoint   : {checkpoint_path} (epoch={checkpoint.get('epoch', 'unknown')})")
    print(f"  Years        : {args.start_year}-{args.end_year}")
    print(f"  Skip years   : {args.skip_years or 'none'}")
    print(f"  Init months  : {args.months}")
    print(f"  Out dir      : {args.out_dir}")
    print(f"  Domain       : {len(probe_dataset.lats)}x{len(probe_dataset.lons)}")
    print(f"  T2M target   : {t2m_target_mode} ({t2m_residual_min:.3f} .. {t2m_residual_max:.3f})")
    print(f"  Ensembles    : {args.clim_num_ensemble} before {args.eval_start_year}; {args.eval_num_ensemble} from {args.eval_start_year}")
    print(f"  ODE steps    : {args.num_steps}")
    print(f"  Ens chunk    : {args.ensemble_chunk_size}")
    print(f"  ODE batch    : {args.ode_batch_size}")
    print(f"  Soft runtime : {args.max_runtime_minutes if args.max_runtime_minutes else 'disabled'} minutes")
    print(f"  Noise mode   : {'pure Gaussian' if args.pure_noise else 'EOF-LHS + variance'}")
    print("=" * 88 + "\n")

    all_complete = True
    skip_years = parse_years(args.skip_years)
    for year in range(args.start_year, args.end_year + 1):
        if year in skip_years:
            print(f"⏭️ {year}: skipped by --skip_years.")
            continue
        if deadline_reached(deadline):
            print(f"⏸️ Soft runtime limit reached before starting {year}.")
            all_complete = False
            break
        torch.manual_seed(args.seed + year)
        np.random.seed(args.seed + year)
        completed = write_year(
            year,
            args,
            config,
            model,
            flow_matcher,
            device,
            checkpoint_path,
            checkpoint,
            noise_context,
            deadline=deadline,
        )
        if not completed:
            all_complete = False
            break
    if all_complete:
        print("✅ Requested flow_finalv1_global global Zarr generation range is complete.")
    else:
        print("⏸️ Requested flow_finalv1_global global Zarr generation range is incomplete and can be resumed.")


if __name__ == "__main__":
    main()
