#!/usr/bin/env python3
"""
Generate yearly multi-v1 ensemble forecasts and save them as Zarr stores.

Outputs are written under dataprocess/gen_multiv1/ by default and mirror the
GEOS-style grid/coord layout as closely as possible while adding an explicit
ensemble member dimension M.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from dataset_flow_multi import S2SHybridDataset
from flow_matching_multi import FlowMatchingModel, CustomFlowMatcher
import noise_utils
import noise_utils_multi
from train_flow_multiv1 import decode_multi_forecast_raw, euler_solve_chunked


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save multi-v1 ensemble forecasts to yearly Zarr stores.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Input data root. Defaults to config data_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Model output directory. Defaults to config output_dir.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path or filename. Defaults to best_flow_ckpt.pt.")
    parser.add_argument("--start_year", type=int, default=2020)
    parser.add_argument("--end_year", type=int, default=2021)
    parser.add_argument("--out_dir", type=str, default="dataprocess/gen_multiv1", help="Directory for yearly generated Zarr stores.")
    parser.add_argument("--num_ensemble", type=int, default=120)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4, help="Must be a multiple of 4 to preserve init/lead grouping.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ensemble_chunk_size", type=int, default=None, help="Generate this many ensemble members at a time.")
    parser.add_argument("--ode_batch_size", type=int, default=None, help="Max state batch passed through the ODE solver at once.")
    parser.add_argument("--member_chunk", type=int, default=10, help="Chunk size along ensemble member dimension in the output Zarr.")
    parser.add_argument(
        "--max_new_init_dates",
        type=int,
        default=None,
        help="Write at most this many new init dates across the run, then exit cleanly for scheduler-driven resume.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate years even if the final yearly Zarr already exists.")
    return parser.parse_args()


def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def choose_name(items, candidates, label):
    for name in candidates:
        if name in items:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}")


def choose_data_var(ds, candidates, label):
    for name in candidates:
        if name in ds.data_vars:
            return name
    if label == "tas":
        return "tas"
    raise KeyError(f"Could not find {label} variable. Available: {list(ds.data_vars)}")


def maybe_load_eof_bases(path):
    if not path or not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["eof_bases"] if isinstance(payload, dict) and "eof_bases" in payload else payload


def resolve_aux_path(data_dir, filename):
    search_roots = [
        os.path.join(data_dir, "eof"),
        data_dir,
        os.path.dirname(__file__),
    ]
    for root in search_roots:
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_noise_context(data_dir, accelerator):
    mjo_eof_bases = maybe_load_eof_bases(resolve_aux_path(data_dir, "mjo_eof_bases.pt"))
    nao_eof_bases = maybe_load_eof_bases(resolve_aux_path(data_dir, "nao_eof_bases.pt"))
    enso_eof_bases = maybe_load_eof_bases(resolve_aux_path(data_dir, "enso_eof_bases.pt"))

    t2m_mjo_eof_bases = maybe_load_eof_bases(resolve_aux_path(data_dir, "mjo_t2m_eof_bases.pt"))
    t2m_nao_eof_bases = maybe_load_eof_bases(resolve_aux_path(data_dir, "nao_t2m_eof_bases.pt"))
    t2m_enso_eof_bases = maybe_load_eof_bases(resolve_aux_path(data_dir, "enso_t2m_eof_bases.pt"))

    nao_lookup = noise_utils.parse_nao_index(os.path.join(data_dir, "norm.daily.nao.index.b500101.current.ascii"))
    oni_lookup = noise_utils.parse_oni_index(os.path.join(data_dir, "oni.ascii.txt"))

    mjo_df = None
    mjo_csv_path = os.path.join(data_dir, "mjo_processed.csv")
    if os.path.exists(mjo_csv_path):
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=["S"])
        mjo_df["date_str"] = mjo_df["S"].dt.strftime("%Y-%m-%d")
        mjo_df = mjo_df.set_index("date_str")

    accelerator.print("Loaded auxiliary generation context:")
    accelerator.print(f"  PR EOFs  : MJO={mjo_eof_bases is not None}, NAO={nao_eof_bases is not None}, ENSO={enso_eof_bases is not None}")
    accelerator.print(f"  T2M EOFs : MJO={t2m_mjo_eof_bases is not None}, NAO={t2m_nao_eof_bases is not None}, ENSO={t2m_enso_eof_bases is not None}")
    accelerator.print(f"  Indices  : NAO={nao_lookup is not None}, ONI={oni_lookup is not None}, MJO CSV={mjo_df is not None}")

    return {
        "pr_mjo_bases": mjo_eof_bases,
        "pr_nao_bases": nao_eof_bases,
        "pr_enso_bases": enso_eof_bases,
        "t2m_mjo_bases": t2m_mjo_eof_bases,
        "t2m_nao_bases": t2m_nao_eof_bases,
        "t2m_enso_bases": t2m_enso_eof_bases,
        "nao_lookup": nao_lookup,
        "oni_lookup": oni_lookup,
        "mjo_df": mjo_df,
    }


def resolve_checkpoint(model_output_dir, requested_checkpoint=None):
    if requested_checkpoint:
        if os.path.isabs(requested_checkpoint):
            checkpoint_path = requested_checkpoint
        else:
            candidate = os.path.join(model_output_dir, requested_checkpoint)
            checkpoint_path = candidate if os.path.exists(candidate) else os.path.abspath(requested_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Requested checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    best_ckpt = os.path.join(model_output_dir, "best_flow_ckpt.pt")
    if os.path.exists(best_ckpt):
        return best_ckpt

    registry_path = os.path.join(model_output_dir, "model_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
        if registry:
            best_entry = registry[0]
            path = best_entry["path"]
            checkpoint_path = path if os.path.isabs(path) else os.path.abspath(path)
            if os.path.exists(checkpoint_path):
                return checkpoint_path

    latest_ckpt = os.path.join(model_output_dir, "latest_flow_ckpt.pt")
    if os.path.exists(latest_ckpt):
        return latest_ckpt

    periodic_ckpts = sorted(glob.glob(os.path.join(model_output_dir, "periodic_ckpt_epoch_*.pt")))
    if periodic_ckpts:
        return periodic_ckpts[-1]

    raise FileNotFoundError(f"No usable checkpoint found in {model_output_dir}")


def build_condition_tensor(batch, device):
    vB = batch["y_target"].shape[0]
    H, W = batch["y_target"].shape[-2:]

    x_obs = batch["x_obs"].to(device)
    x_geos = batch["x_geos"].to(device)
    x_geos_cat = x_geos.view(vB, -1, H, W)

    month = batch["month"].to(device).float()
    sin_month = torch.sin(2 * np.pi * (month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    cos_month = torch.cos(2 * np.pi * (month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)

    lead_idx = batch["lead_idx"].to(device).float()
    lead_channel = ((lead_idx / 1.5) - 1.0).view(vB, 1, 1, 1).expand(vB, 1, H, W)

    return torch.cat([x_obs, x_geos_cat, sin_month, cos_month, lead_channel], dim=1)


def infer_reference_layout(data_dir, year, num_ensemble, member_chunk):
    ref_path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference GEOS file not found for {year}: {ref_path}")

    ds_ref = xr.open_zarr(ref_path, consolidated=False)
    try:
        s_dim = choose_name(ds_ref.dims, ["S", "time", "init_time"], "init dimension")
        lead_dim = choose_name(ds_ref.dims, ["L", "lead", "lead_time"], "lead dimension")
        y_dim = choose_name(set(ds_ref.dims) | set(ds_ref.coords), ["Y", "latitude", "lat", "y"], "latitude dimension")
        x_dim = choose_name(set(ds_ref.dims) | set(ds_ref.coords), ["X", "longitude", "lon", "x"], "longitude dimension")

        pr_name = choose_data_var(ds_ref, ["pr", "precip", "PRECTOT", "flux_precip"], "pr")
        tas_name = choose_data_var(ds_ref, ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"], "tas")

        s_values = np.array(ds_ref[s_dim].values)
        lead_values = np.array(ds_ref[lead_dim].values) if lead_dim in ds_ref.coords else np.arange(ds_ref.sizes[lead_dim])
        y_values = np.array(ds_ref[y_dim].values) if y_dim in ds_ref.coords else np.arange(ds_ref.sizes[y_dim])
        x_values = np.array(ds_ref[x_dim].values) if x_dim in ds_ref.coords else np.arange(ds_ref.sizes[x_dim])

        coord_attrs = {}
        for coord_name in [s_dim, lead_dim, y_dim, x_dim]:
            if coord_name in ds_ref.coords:
                coord_attrs[coord_name] = dict(ds_ref[coord_name].attrs)

        pr_attrs = dict(ds_ref[pr_name].attrs) if pr_name in ds_ref.data_vars else {}
        tas_attrs = dict(ds_ref[tas_name].attrs) if tas_name in ds_ref.data_vars else {}
    finally:
        ds_ref.close()

    return {
        "ref_path": ref_path,
        "s_dim": s_dim,
        "member_dim": "M",
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "pr_name": pr_name,
        "tas_name": tas_name,
        "s_values": s_values,
        "member_values": np.arange(num_ensemble, dtype=np.int32),
        "lead_values": lead_values,
        "y_values": y_values,
        "x_values": x_values,
        "coord_attrs": coord_attrs,
        "pr_attrs": pr_attrs,
        "tas_attrs": tas_attrs,
        "encoding": {
            pr_name: {
                "dtype": "float32",
                "chunks": (1, max(1, min(member_chunk, num_ensemble)), len(lead_values), len(y_values), len(x_values)),
            },
            tas_name: {
                "dtype": "float32",
                "chunks": (1, max(1, min(member_chunk, num_ensemble)), len(lead_values), len(y_values), len(x_values)),
            },
        },
    }


def get_autocast_context(device, accelerator):
    if device.type != "cuda":
        return nullcontext()
    if accelerator.mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if accelerator.mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def batch_init_dates(batch):
    years = batch["year"][0::4].cpu().numpy().astype(int)
    months = batch["month"][0::4].cpu().numpy().astype(int)
    days = batch["day"][0::4].cpu().numpy().astype(int)
    return pd.to_datetime([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)])


def verify_batch_dates(expected_s_values, batch):
    expected_dates = pd.to_datetime(expected_s_values).normalize()
    actual_dates = batch_init_dates(batch).normalize()
    if len(expected_dates) != len(actual_dates) or not np.array_equal(expected_dates.values, actual_dates.values):
        raise ValueError(
            "Dataset/init-date ordering mismatch while writing output. "
            f"Expected {list(expected_dates.strftime('%Y-%m-%d'))}, "
            f"got {list(actual_dates.strftime('%Y-%m-%d'))}."
        )


def make_output_batch_dataset(pr_batch, tas_batch, s_values, layout, dataset_attrs):
    ds_batch = xr.Dataset(
        data_vars={
            layout["pr_name"]: (
                (layout["s_dim"], layout["member_dim"], layout["lead_dim"], layout["y_dim"], layout["x_dim"]),
                pr_batch.astype(np.float32, copy=False),
            ),
            layout["tas_name"]: (
                (layout["s_dim"], layout["member_dim"], layout["lead_dim"], layout["y_dim"], layout["x_dim"]),
                tas_batch.astype(np.float32, copy=False),
            ),
        },
        coords={
            layout["s_dim"]: s_values,
            layout["member_dim"]: layout["member_values"],
            layout["lead_dim"]: layout["lead_values"],
            layout["y_dim"]: layout["y_values"],
            layout["x_dim"]: layout["x_values"],
        },
        attrs=dataset_attrs,
    )

    ds_batch[layout["pr_name"]].attrs.update(layout["pr_attrs"])
    ds_batch[layout["tas_name"]].attrs.update(layout["tas_attrs"])
    ds_batch[layout["member_dim"]].attrs.update({"long_name": "ensemble_member"})

    for coord_name, attrs in layout["coord_attrs"].items():
        if coord_name in ds_batch.coords:
            ds_batch[coord_name].attrs.update(attrs)

    return ds_batch


def inspect_tmp_store(tmp_path, layout):
    ds_tmp = xr.open_zarr(tmp_path, consolidated=False)
    try:
        s_dim = choose_name(ds_tmp.dims, [layout["s_dim"], "S", "time", "init_time"], "init dimension")
        member_dim = choose_name(ds_tmp.dims, [layout["member_dim"], "M"], "member dimension")
        lead_dim = choose_name(ds_tmp.dims, [layout["lead_dim"], "L", "lead", "lead_time"], "lead dimension")
        written_s_values = np.array(ds_tmp[s_dim].values)
        member_size = int(ds_tmp.sizes[member_dim])
        lead_size = int(ds_tmp.sizes[lead_dim])
    finally:
        ds_tmp.close()

    return {
        "written_s_values": written_s_values,
        "member_size": member_size,
        "lead_size": lead_size,
    }


def generate_noise(batch, num_ensemble, device, year, use_eof_lhs_noise, noise_context, rho_pr, rho_t2m):
    vB = batch["y_target"].shape[0]
    H, W = batch["x_obs"].shape[-2:]

    if not use_eof_lhs_noise:
        return torch.randn((vB * num_ensemble, 2, H, W), device=device)

    eof_noise = noise_utils_multi.generate_dynamic_multimodal_noise_multi(
        batch=batch,
        E=num_ensemble,
        device=device,
        pr_mjo_bases=noise_context["pr_mjo_bases"],
        pr_nao_bases=noise_context["pr_nao_bases"],
        pr_enso_bases=noise_context["pr_enso_bases"],
        t2m_mjo_bases=noise_context["t2m_mjo_bases"],
        t2m_nao_bases=noise_context["t2m_nao_bases"],
        t2m_enso_bases=noise_context["t2m_enso_bases"],
        nao_lookup=noise_context["nao_lookup"],
        oni_lookup=noise_context["oni_lookup"],
        mjo_df=noise_context["mjo_df"],
        year=year,
        use_lhs=True,
        orthogonalize_lhs=True,
    )
    return noise_utils_multi.mix_noise_with_random_multi(eof_noise, rho_pr, rho_t2m)


def write_year(
    year,
    args,
    config,
    accelerator,
    model,
    flow_matcher,
    checkpoint_path,
    checkpoint_meta,
    noise_context,
    max_new_init_dates=None,
):
    out_path = os.path.join(args.out_dir, f"{year}.zarr")
    tmp_path = out_path + ".tmp"

    if os.path.exists(out_path):
        if not args.overwrite:
            accelerator.print(f"✅ {year}: output already exists at {out_path}. Skipping.")
            return
        accelerator.print(f"♻️ {year}: removing existing output at {out_path}")
        remove_path(out_path)

    layout = infer_reference_layout(
        data_dir=args.data_dir,
        year=year,
        num_ensemble=args.num_ensemble,
        member_chunk=args.member_chunk,
    )
    total_expected_inits = len(layout["s_values"])

    resume_offset = 0
    wrote_any = False
    if os.path.exists(tmp_path):
        if args.overwrite:
            accelerator.print(f"🧹 {year}: removing existing temp store at {tmp_path}")
            remove_path(tmp_path)
        else:
            tmp_info = inspect_tmp_store(tmp_path, layout)
            written_s_values = tmp_info["written_s_values"]
            resume_offset = len(written_s_values)

            if tmp_info["member_size"] != int(args.num_ensemble):
                raise RuntimeError(
                    f"Cannot resume {year}: temp store has member dimension {tmp_info['member_size']}, "
                    f"but current run expects {args.num_ensemble}."
                )
            if tmp_info["lead_size"] != len(layout["lead_values"]):
                raise RuntimeError(
                    f"Cannot resume {year}: temp store has lead dimension {tmp_info['lead_size']}, "
                    f"but current run expects {len(layout['lead_values'])}."
                )
            if resume_offset > total_expected_inits:
                raise RuntimeError(
                    f"Cannot resume {year}: temp store already contains {resume_offset} init dates, "
                    f"but reference layout only expects {total_expected_inits}."
                )

            expected_prefix = pd.to_datetime(layout["s_values"][:resume_offset]).normalize()
            actual_prefix = pd.to_datetime(written_s_values).normalize()
            if not np.array_equal(expected_prefix.values, actual_prefix.values):
                raise RuntimeError(
                    f"Cannot resume {year}: temp store init dates do not match the expected prefix of the year."
                )

            if resume_offset == 0:
                accelerator.print(f"🧹 {year}: temp store exists but has no written init dates. Restarting from scratch.")
                remove_path(tmp_path)
            elif resume_offset == total_expected_inits:
                accelerator.print(
                    f"✅ {year}: temp store already has all {total_expected_inits} init dates. Finalizing {out_path}."
                )
                os.rename(tmp_path, out_path)
                return
            else:
                wrote_any = True
                accelerator.print(
                    f"♻️ {year}: resuming temp store at {tmp_path} "
                    f"from {resume_offset}/{total_expected_inits} init dates."
                )

    dataset = S2SHybridDataset(
        data_root=args.data_dir,
        start_year=year,
        end_year=year,
        normalize=True,
        preload=False,
        stats_file=config.get("stats_file", "v1_multi_global_stats.pt"),
        subsample_monthly=False,
    )
    if len(dataset) == 0:
        accelerator.print(f"⚠️ {year}: no samples were indexed. Skipping.")
        return

    indexed_inits = len(dataset) // 4
    if indexed_inits != total_expected_inits:
        accelerator.print(
            f"⚠️ {year}: dataset indexed {indexed_inits} inits, while reference GEOS has {total_expected_inits}. "
            "Proceeding with strict date checks."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(accelerator.device.type == "cuda"),
    )

    target_sqrt_min = 0.0
    target_sqrt_max = 7.071
    use_flow_variance = True
    use_eof_lhs_noise = True
    rho_pr = float(config.get("validation_rho_pr", 1.0))
    rho_t2m = float(config.get("validation_rho_t2m", rho_pr))
    beta_pr = float(config.get("validation_var_beta_pr", 1.0))
    beta_t2m = float(config.get("validation_var_beta_t2m", beta_pr))
    ode_batch_size = args.ode_batch_size if args.ode_batch_size is not None else int(config.get("validation_ode_batch_size", 120))
    ensemble_chunk_size = (
        args.ensemble_chunk_size
        if args.ensemble_chunk_size is not None
        else int(config.get("test_max_ensemble_per_chunk", 30))
    )
    ensemble_chunk_size = max(1, min(int(ensemble_chunk_size), int(args.num_ensemble)))

    dataset_attrs = {
        "generated_by": os.path.basename(__file__),
        "source_model_output_dir": os.path.abspath(args.output_dir),
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_is_variance_phase": bool(checkpoint_meta.get("is_variance_phase", False)),
        "source_reference_geos": os.path.abspath(layout["ref_path"]),
        "num_ensemble": int(args.num_ensemble),
        "num_steps": int(args.num_steps),
        "use_flow_variance": bool(use_flow_variance),
        "use_eof_lhs_noise": bool(use_eof_lhs_noise),
        "validation_rho_pr": float(rho_pr),
        "validation_rho_t2m": float(rho_t2m),
        "validation_var_beta_pr": float(beta_pr),
        "validation_var_beta_t2m": float(beta_t2m),
    }

    accelerator.print(
        f"🚀 {year}: generating {total_expected_inits} init dates with "
        f"{args.num_ensemble} members, {args.num_steps} steps, "
        f"use_flow_variance={use_flow_variance}, use_eof_lhs_noise={use_eof_lhs_noise}, "
        f"ensemble_chunk_size={ensemble_chunk_size}, resume_offset={resume_offset}, "
        f"max_new_init_dates={max_new_init_dates}"
    )

    offset = resume_offset
    seen_offset = 0
    new_written = 0
    stopped_due_to_limit = False
    for batch_idx, batch in enumerate(loader):
        vB = batch["y_target"].shape[0]
        if vB % 4 != 0:
            raise ValueError(
                f"Batch {batch_idx} for year {year} has size {vB}, which is not divisible by 4. "
                "Use a batch_size that is a multiple of 4."
            )

        num_inits = vB // 4
        batch_start = seen_offset
        batch_end = batch_start + num_inits
        seen_offset = batch_end

        if batch_end <= resume_offset:
            continue
        if max_new_init_dates is not None and (new_written + num_inits) > max_new_init_dates:
            stopped_due_to_limit = True
            accelerator.print(
                f"⏸️ {year}: reached per-run limit before batch {batch_idx}. "
                f"Written {new_written}/{max_new_init_dates} new init dates this run."
            )
            break
        if batch_start < resume_offset < batch_end:
            raise RuntimeError(
                f"Cannot safely resume {year}: temp store ends at init index {resume_offset}, "
                f"which falls inside batch {batch_idx} covering init indices [{batch_start}, {batch_end})."
            )

        expected_s_values = layout["s_values"][batch_start: batch_end]
        if len(expected_s_values) != num_inits:
            raise ValueError(
                f"Year {year} batch {batch_idx} would write beyond the expected S coordinate range "
                f"(offset={batch_start}, num_inits={num_inits}, total={len(layout['s_values'])})."
            )
        verify_batch_dates(expected_s_values, batch)

        H, W = batch["y_target"].shape[-2:]
        fx_cond = build_condition_tensor(batch, accelerator.device)
        pr_chunks = []
        tas_chunks = []

        for ens_start in range(0, args.num_ensemble, ensemble_chunk_size):
            ens_count = min(ensemble_chunk_size, args.num_ensemble - ens_start)

            fx_cond_expanded = (
                fx_cond.unsqueeze(1)
                .expand(vB, ens_count, -1, H, W)
                .reshape(vB * ens_count, -1, H, W)
                .contiguous()
            )
            lead_idx_expanded = (
                batch["lead_idx"]
                .to(accelerator.device)
                .unsqueeze(1)
                .expand(vB, ens_count)
                .reshape(-1)
                .long()
            )

            noise_expanded = generate_noise(
                batch=batch,
                num_ensemble=ens_count,
                device=accelerator.device,
                year=year,
                use_eof_lhs_noise=use_eof_lhs_noise,
                noise_context=noise_context,
                rho_pr=rho_pr,
                rho_t2m=rho_t2m,
            )

            with get_autocast_context(accelerator.device, accelerator):
                p_x1_expanded = euler_solve_chunked(
                    flow_matcher,
                    model,
                    noise_expanded,
                    fx_cond_expanded,
                    num_steps=int(args.num_steps),
                    lead_idx=lead_idx_expanded,
                    apply_flow_variance=use_flow_variance,
                    variance_beta=(beta_pr, beta_t2m),
                    chunk_size=ode_batch_size,
                )

            p_x1_batch = p_x1_expanded.view(vB, ens_count, 2, H, W).float()
            pred_raw = decode_multi_forecast_raw(
                p_x1_batch,
                target_sqrt_min=target_sqrt_min,
                target_sqrt_max=target_sqrt_max,
            )

            pr_chunk = (
                pred_raw[:, :, 0]
                .transpose(0, 1)
                .contiguous()
                .view(ens_count, num_inits, 4, H, W)
                .transpose(0, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
            tas_chunk = (
                pred_raw[:, :, 1]
                .transpose(0, 1)
                .contiguous()
                .view(ens_count, num_inits, 4, H, W)
                .transpose(0, 1)
                .contiguous()
                .cpu()
                .numpy()
            )

            pr_chunks.append(pr_chunk)
            tas_chunks.append(tas_chunk)

        pr_batch = np.concatenate(pr_chunks, axis=1)
        tas_batch = np.concatenate(tas_chunks, axis=1)

        ds_batch = make_output_batch_dataset(
            pr_batch=pr_batch,
            tas_batch=tas_batch,
            s_values=expected_s_values,
            layout=layout,
            dataset_attrs=dataset_attrs,
        )
        if not wrote_any:
            ds_batch.to_zarr(tmp_path, mode="w", encoding=layout["encoding"])
            wrote_any = True
        else:
            ds_batch.to_zarr(tmp_path, mode="a", append_dim=layout["s_dim"])
        ds_batch.close()

        offset = batch_end
        new_written += num_inits
        accelerator.print(f"  {year}: wrote {offset}/{total_expected_inits} init dates")

    if offset != total_expected_inits:
        if stopped_due_to_limit:
            accelerator.print(
                f"⏸️ {year}: leaving temp store at {tmp_path} after writing "
                f"{new_written} new init dates this run ({offset}/{total_expected_inits} total)."
            )
            return {
                "year": year,
                "completed": False,
                "new_written": new_written,
                "offset": offset,
                "total_expected_inits": total_expected_inits,
                "stopped_due_to_limit": True,
            }
        raise RuntimeError(
            f"Year {year} generation ended with {offset} written init dates, expected {total_expected_inits}. "
            f"Leaving temp store at {tmp_path} for inspection."
        )

    os.rename(tmp_path, out_path)
    accelerator.print(f"✅ {year}: saved {out_path}")
    return {
        "year": year,
        "completed": True,
        "new_written": new_written,
        "offset": offset,
        "total_expected_inits": total_expected_inits,
        "stopped_due_to_limit": False,
    }


def main():
    args = parse_args()
    if args.batch_size % 4 != 0:
        raise ValueError("--batch_size must be a multiple of 4 so lead weeks stay grouped by init.")
    if args.start_year > args.end_year:
        raise ValueError("--start_year must be <= --end_year")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    args.data_dir = args.data_dir or config["data_dir"]
    args.output_dir = args.output_dir or config.get("output_dir", "ml_output_flowmulti")
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    checkpoint_path = resolve_checkpoint(args.output_dir, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    accelerator.print("=" * 80)
    accelerator.print("MULTI-V1 ENSEMBLE GENERATION")
    accelerator.print(f"Config      : {os.path.abspath(args.config)}")
    accelerator.print(f"Data Dir    : {args.data_dir}")
    accelerator.print(f"Output Dir  : {os.path.abspath(args.out_dir)}")
    accelerator.print(f"Model Dir   : {os.path.abspath(args.output_dir)}")
    accelerator.print(f"Checkpoint  : {os.path.abspath(checkpoint_path)}")
    accelerator.print(f"Device      : {device}")
    accelerator.print(f"Precision   : {accelerator.mixed_precision}")
    accelerator.print(f"Run Init Cap: {args.max_new_init_dates}")
    accelerator.print("=" * 80)

    model = FlowMatchingModel(in_channels=41, out_channels=2).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    flow_matcher = CustomFlowMatcher(device=device)
    noise_context = load_noise_context(args.data_dir, accelerator)

    remaining_init_budget = args.max_new_init_dates
    total_new_written = 0
    stopped_due_to_limit = False
    for year in range(args.start_year, args.end_year + 1):
        if remaining_init_budget is not None and remaining_init_budget <= 0:
            stopped_due_to_limit = True
            break
        result = write_year(
            year=year,
            args=args,
            config=config,
            accelerator=accelerator,
            model=model,
            flow_matcher=flow_matcher,
            checkpoint_path=checkpoint_path,
            checkpoint_meta=checkpoint,
            noise_context=noise_context,
            max_new_init_dates=remaining_init_budget,
        )
        if result is None:
            continue
        total_new_written += int(result["new_written"])
        if remaining_init_budget is not None:
            remaining_init_budget -= int(result["new_written"])
        if not result["completed"]:
            stopped_due_to_limit = bool(result.get("stopped_due_to_limit", False))
            break

    accelerator.print(
        f"Run summary: wrote {total_new_written} new init dates; "
        f"stopped_due_to_limit={stopped_due_to_limit}"
    )


if __name__ == "__main__":
    main()
