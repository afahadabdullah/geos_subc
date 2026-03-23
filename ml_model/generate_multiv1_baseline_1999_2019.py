#!/usr/bin/env python3
"""
Generate yearly GEOS baseline ensembles for 1999-2019 and save them as Zarr.

This mirrors the output layout and resume/finalize behavior of
generate_multiv1_ensembles.py, but writes the raw GEOS precipitation/T2M
members directly instead of ML-generated samples.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset_flow_multi import S2SHybridDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GEOS baseline multi-v1 Zarr stores for 1999-2019.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Input data root. Defaults to config data_dir.")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2019)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataprocess/gen_multiv1_baseline_1999_2019",
        help="Directory for yearly generated Zarr stores.",
    )
    parser.add_argument("--num_ensemble", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=40, help="Must be a multiple of 4 to preserve init/lead grouping.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--member_chunk", type=int, default=3, help="Chunk size along ensemble member dimension in the output Zarr.")
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


def select_baseline_members(geos_ens_raw, num_ensemble):
    if geos_ens_raw.ndim != 6:
        raise ValueError(f"Expected geos_ens_raw to have 6 dims [B, M, C, L, H, W], got shape {tuple(geos_ens_raw.shape)}")

    init_level = geos_ens_raw[0::4].contiguous()
    source_members = int(init_level.shape[1])
    if source_members <= 0:
        raise ValueError("GEOS baseline batch has no source members to select from.")

    member_idx = torch.arange(num_ensemble, device=init_level.device) % source_members
    return init_level.index_select(1, member_idx), source_members


def write_year(year, args, data_dir, accelerator, max_new_init_dates=None):
    out_path = os.path.join(args.out_dir, f"{year}.zarr")
    tmp_path = out_path + ".tmp"

    if os.path.exists(out_path):
        if not args.overwrite:
            accelerator.print(f"✅ {year}: output already exists at {out_path}. Skipping.")
            return None
        accelerator.print(f"♻️ {year}: removing existing output at {out_path}")
        remove_path(out_path)

    layout = infer_reference_layout(
        data_dir=data_dir,
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
                return None
            else:
                wrote_any = True
                accelerator.print(
                    f"♻️ {year}: resuming temp store at {tmp_path} "
                    f"from {resume_offset}/{total_expected_inits} init dates."
                )

    dataset = S2SHybridDataset(
        data_root=data_dir,
        start_year=year,
        end_year=year,
        normalize=False,
        preload=False,
        subsample_monthly=False,
    )
    if len(dataset) == 0:
        accelerator.print(f"⚠️ {year}: no samples were indexed. Skipping.")
        return None

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

    dataset_attrs = {
        "generated_by": os.path.basename(__file__),
        "baseline_source": "raw_geos_members",
        "source_reference_geos": os.path.abspath(layout["ref_path"]),
        "num_ensemble": int(args.num_ensemble),
        "member_fill_mode": "repeat_existing_members_if_needed",
    }

    accelerator.print(
        f"🚀 {year}: generating GEOS baseline with {total_expected_inits} init dates, "
        f"{args.num_ensemble} members, resume_offset={resume_offset}, "
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

        baseline_batch, source_members = select_baseline_members(batch["geos_ens_raw"], args.num_ensemble)
        pr_batch = baseline_batch[:, :, 0].cpu().numpy()
        tas_batch = baseline_batch[:, :, 1].cpu().numpy()

        ds_batch = make_output_batch_dataset(
            pr_batch=pr_batch,
            tas_batch=tas_batch,
            s_values=expected_s_values,
            layout=layout,
            dataset_attrs={**dataset_attrs, "source_members_in_year_batch": int(source_members)},
        )
        if not wrote_any:
            ds_batch.to_zarr(tmp_path, mode="w", encoding=layout["encoding"])
            wrote_any = True
        else:
            ds_batch.to_zarr(tmp_path, mode="a", append_dim=layout["s_dim"])
        ds_batch.close()

        offset = batch_end
        new_written += num_inits
        accelerator.print(
            f"  {year}: wrote {offset}/{total_expected_inits} init dates "
            f"(source members={source_members}, output members={args.num_ensemble})"
        )

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
    if args.num_ensemble <= 0:
        raise ValueError("--num_ensemble must be positive.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator()

    accelerator.print("=" * 80)
    accelerator.print("MULTI-V1 GEOS BASELINE GENERATION")
    accelerator.print(f"Config      : {os.path.abspath(args.config)}")
    accelerator.print(f"Data Dir    : {data_dir}")
    accelerator.print(f"Output Dir  : {os.path.abspath(args.out_dir)}")
    accelerator.print(f"Device      : {accelerator.device}")
    accelerator.print(f"Precision   : {accelerator.mixed_precision}")
    accelerator.print(f"Run Init Cap: {args.max_new_init_dates}")
    accelerator.print("=" * 80)

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
            data_dir=data_dir,
            accelerator=accelerator,
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
