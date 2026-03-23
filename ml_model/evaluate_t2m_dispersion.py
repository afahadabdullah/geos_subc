#!/usr/bin/env python3
"""
Evaluate held-out 2020-2021 T2M ensemble dispersiveness for ML and GEOS.

This script compares:
- ML full ensemble (e.g. 120 members)
- ML downsampled to a fair member count (default 4)
- Raw GEOS ensemble

using only the held-out forecast years specified on the command line.

Outputs:
- lead-wise CSV summary
- uPIT histogram CSV
- text report
- summary metric PNG
- uPIT histogram PNG
"""

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate held-out T2M ensemble dispersion for ML and GEOS.")
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory containing geos_subc_<year>.zarr and t2m_weekly_<year>.zarr. Defaults to config data_dir.",
    )
    parser.add_argument(
        "--ml_dir",
        type=str,
        default="dataprocess/gen_multiv1",
        help="Directory containing generated ML yearly zarr stores like 2020.zarr and 2021.zarr.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ml_output_flowmulti/t2m_dispersion_eval",
        help="Directory to save summary CSVs, text report, and plots.",
    )
    parser.add_argument("--start_year", type=int, default=None, help="First held-out year. Defaults to config val_start_year.")
    parser.add_argument("--end_year", type=int, default=None, help="Last held-out year. Defaults to config val_end_year.")
    parser.add_argument(
        "--fair_member_count",
        type=int,
        default=4,
        help="Downsample ML to this many members for a fair comparison with GEOS.",
    )
    parser.add_argument(
        "--sample_chunk_size",
        type=int,
        default=2,
        help="Number of init dates to load per chunk. Lower this if memory is tight.",
    )
    parser.add_argument(
        "--pit_bins",
        type=int,
        default=10,
        help="Number of uPIT histogram bins.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def choose_name(items: Iterable[str], candidates: Sequence[str], label: str) -> str:
    item_set = set(items)
    for name in candidates:
        if name in item_set:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {sorted(item_set)}")


def choose_data_var(ds: xr.Dataset, candidates: Sequence[str], label: str) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {list(ds.data_vars)}")


def open_year_dataset(path: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_zarr(path, consolidated=False)


def infer_layout(ds: xr.Dataset, kind: str) -> Dict[str, str]:
    s_dim = choose_name(ds.dims, ["S", "time", "init_time"], f"{kind} init dimension")
    lead_dim = choose_name(ds.dims, ["L", "lead", "lead_time"], f"{kind} lead dimension")
    y_dim = choose_name(set(ds.dims) | set(ds.coords), ["Y", "latitude", "lat", "y"], f"{kind} latitude dimension")
    x_dim = choose_name(set(ds.dims) | set(ds.coords), ["X", "longitude", "lon", "x"], f"{kind} longitude dimension")
    member_dim = None
    for candidate in ["M", "member", "ensemble", "ensemble_member"]:
        if candidate in ds.dims:
            member_dim = candidate
            break
    tas_var = choose_data_var(ds, ["tas", "t2m", "T2M", "TAS", "tempt2m", "T2MS"], f"{kind} tas variable")
    return {
        "s_dim": s_dim,
        "lead_dim": lead_dim,
        "y_dim": y_dim,
        "x_dim": x_dim,
        "member_dim": member_dim,
        "tas_var": tas_var,
    }


def downsample_member_axis(data: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"fair member count must be positive, got {count}")
    n_members = int(data.shape[1])
    if n_members <= count:
        return np.asarray(data, dtype=np.float64)

    raw_idx = np.linspace(0, n_members - 1, count)
    idx = np.round(raw_idx).astype(int)
    idx = np.clip(idx, 0, n_members - 1)

    seen = set()
    unique_idx = []
    for item in idx.tolist():
        if item not in seen:
            unique_idx.append(item)
            seen.add(item)
    if len(unique_idx) < count:
        for item in range(n_members):
            if item not in seen:
                unique_idx.append(item)
                seen.add(item)
            if len(unique_idx) == count:
                break
    return np.asarray(data[:, unique_idx], dtype=np.float64)


def weighted_correlation(sum_w: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    if sum_w <= 0.0:
        return float("nan")
    mean_x = sum_x / sum_w
    mean_y = sum_y / sum_w
    var_x = max(0.0, (sum_x2 / sum_w) - mean_x * mean_x)
    var_y = max(0.0, (sum_y2 / sum_w) - mean_y * mean_y)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    cov_xy = (sum_xy / sum_w) - mean_x * mean_y
    return cov_xy / np.sqrt(var_x * var_y)


def classify_dispersion(coverage80: float, ssr: float) -> str:
    if np.isnan(coverage80) or np.isnan(ssr):
        return "unknown"
    if coverage80 < 0.76 or ssr < 0.90:
        return "underdispersive"
    if coverage80 > 0.84 or ssr > 1.10:
        return "overdispersive"
    return "near_calibrated"


@dataclass
class DispersionAccumulator:
    pit_bins: int
    sum_w: float = 0.0
    hit80: float = 0.0
    hit50: float = 0.0
    below_q10: float = 0.0
    above_q90: float = 0.0
    width80: float = 0.0
    width50: float = 0.0
    sum_spread: float = 0.0
    sum_sq_error: float = 0.0
    corr_sum_x: float = 0.0
    corr_sum_y: float = 0.0
    corr_sum_x2: float = 0.0
    corr_sum_y2: float = 0.0
    corr_sum_xy: float = 0.0
    sample_count: int = 0

    def __post_init__(self):
        self.pit_hist = np.zeros(self.pit_bins, dtype=np.float64)
        self.pit_edges = np.linspace(0.0, 1.0, self.pit_bins + 1)

    def update(self, forecast: np.ndarray, obs: np.ndarray, lat_weights: np.ndarray):
        # forecast: [B, M, Y, X], obs: [B, Y, X], lat_weights: [Y]
        forecast = np.asarray(forecast, dtype=np.float64)
        obs = np.asarray(obs, dtype=np.float64)
        lat_weights = np.asarray(lat_weights, dtype=np.float64)

        ens_mean = np.nanmean(forecast, axis=1)
        ens_std = np.nanstd(forecast, axis=1)
        q10 = np.nanpercentile(forecast, 10, axis=1)
        q25 = np.nanpercentile(forecast, 25, axis=1)
        q75 = np.nanpercentile(forecast, 75, axis=1)
        q90 = np.nanpercentile(forecast, 90, axis=1)

        weight_grid = np.broadcast_to(lat_weights[None, :, None], obs.shape)
        valid = (
            np.isfinite(obs)
            & np.isfinite(ens_mean)
            & np.isfinite(ens_std)
            & np.isfinite(q10)
            & np.isfinite(q25)
            & np.isfinite(q75)
            & np.isfinite(q90)
        )
        if not np.any(valid):
            return

        weights = weight_grid[valid]
        obs_valid = obs[valid]
        mean_valid = ens_mean[valid]
        std_valid = ens_std[valid]
        q10_valid = q10[valid]
        q25_valid = q25[valid]
        q75_valid = q75[valid]
        q90_valid = q90[valid]
        abs_error_valid = np.abs(mean_valid - obs_valid)
        sq_error_valid = (mean_valid - obs_valid) ** 2

        weight_sum = float(np.sum(weights))
        self.sum_w += weight_sum
        self.hit80 += float(np.sum(weights * ((obs_valid >= q10_valid) & (obs_valid <= q90_valid))))
        self.hit50 += float(np.sum(weights * ((obs_valid >= q25_valid) & (obs_valid <= q75_valid))))
        self.below_q10 += float(np.sum(weights * (obs_valid < q10_valid)))
        self.above_q90 += float(np.sum(weights * (obs_valid > q90_valid)))
        self.width80 += float(np.sum(weights * (q90_valid - q10_valid)))
        self.width50 += float(np.sum(weights * (q75_valid - q25_valid)))
        self.sum_spread += float(np.sum(weights * std_valid))
        self.sum_sq_error += float(np.sum(weights * sq_error_valid))
        self.corr_sum_x += float(np.sum(weights * std_valid))
        self.corr_sum_y += float(np.sum(weights * abs_error_valid))
        self.corr_sum_x2 += float(np.sum(weights * (std_valid ** 2)))
        self.corr_sum_y2 += float(np.sum(weights * (abs_error_valid ** 2)))
        self.corr_sum_xy += float(np.sum(weights * std_valid * abs_error_valid))
        self.sample_count += int(np.count_nonzero(valid))

        less_count = np.sum(forecast < obs[:, None, :, :], axis=1)
        equal_count = np.sum(forecast == obs[:, None, :, :], axis=1)
        pit = (less_count + 0.5 * equal_count) / float(forecast.shape[1])
        pit_valid = pit[valid]
        self.pit_hist += np.histogram(pit_valid, bins=self.pit_edges, weights=weights)[0]

    def finalize(self) -> Dict[str, float]:
        if self.sum_w <= 0.0:
            return {
                "coverage80": float("nan"),
                "coverage50": float("nan"),
                "below_q10": float("nan"),
                "above_q90": float("nan"),
                "width80": float("nan"),
                "width50": float("nan"),
                "mean_spread": float("nan"),
                "rmse_mean": float("nan"),
                "spread_skill_ratio": float("nan"),
                "spread_error_corr": float("nan"),
                "pit_l1_uniform": float("nan"),
                "sample_count": 0,
            }

        rmse_mean = float(np.sqrt(self.sum_sq_error / self.sum_w))
        mean_spread = float(self.sum_spread / self.sum_w)
        spread_skill_ratio = mean_spread / rmse_mean if rmse_mean > 0.0 else float("nan")
        spread_error_corr = weighted_correlation(
            self.sum_w,
            self.corr_sum_x,
            self.corr_sum_y,
            self.corr_sum_x2,
            self.corr_sum_y2,
            self.corr_sum_xy,
        )
        pit_density = self.pit_hist / np.sum(self.pit_hist) if np.sum(self.pit_hist) > 0.0 else np.full(self.pit_bins, np.nan)
        uniform_density = 1.0 / self.pit_bins
        pit_l1_uniform = float(np.nanmean(np.abs(pit_density - uniform_density)))
        return {
            "coverage80": float(self.hit80 / self.sum_w),
            "coverage50": float(self.hit50 / self.sum_w),
            "below_q10": float(self.below_q10 / self.sum_w),
            "above_q90": float(self.above_q90 / self.sum_w),
            "width80": float(self.width80 / self.sum_w),
            "width50": float(self.width50 / self.sum_w),
            "mean_spread": mean_spread,
            "rmse_mean": rmse_mean,
            "spread_skill_ratio": float(spread_skill_ratio),
            "spread_error_corr": float(spread_error_corr),
            "pit_l1_uniform": pit_l1_uniform,
            "sample_count": int(self.sample_count),
        }


def prepare_common_dates(ml_ds: xr.Dataset, ml_layout: Dict[str, str], geos_ds: xr.Dataset, geos_layout: Dict[str, str], obs_ds: xr.Dataset, obs_layout: Dict[str, str]) -> List[pd.Timestamp]:
    ml_dates = pd.to_datetime(ml_ds[ml_layout["s_dim"]].values).normalize()
    geos_dates = pd.to_datetime(geos_ds[geos_layout["s_dim"]].values).normalize()
    obs_dates = pd.to_datetime(obs_ds[obs_layout["s_dim"]].values).normalize()
    common = sorted(set(ml_dates) & set(geos_dates) & set(obs_dates))
    return [pd.Timestamp(item) for item in common]


def exact_indices_for_dates(s_values: np.ndarray, dates: Sequence[pd.Timestamp]) -> List[int]:
    s_dates = pd.to_datetime(s_values).normalize()
    index_map = {pd.Timestamp(date): idx for idx, date in enumerate(s_dates)}
    missing = [date for date in dates if pd.Timestamp(date) not in index_map]
    if missing:
        missing_str = ", ".join(pd.Timestamp(date).strftime("%Y-%m-%d") for date in missing[:5])
        raise ValueError(f"Missing requested dates in dataset: {missing_str}")
    return [int(index_map[pd.Timestamp(date)]) for date in dates]


def extract_tas_chunk(ds: xr.Dataset, layout: Dict[str, str], s_indices: Sequence[int], lead_idx: int) -> np.ndarray:
    da = ds[layout["tas_var"]].isel({layout["s_dim"]: list(s_indices), layout["lead_dim"]: int(lead_idx)})
    if layout["member_dim"] is None:
        da = da.transpose(layout["s_dim"], layout["y_dim"], layout["x_dim"])
    else:
        da = da.transpose(layout["s_dim"], layout["member_dim"], layout["y_dim"], layout["x_dim"])
    return np.asarray(da.values, dtype=np.float64)


def make_summary_figure(rows: List[Dict[str, object]], output_path: str):
    metrics = [
        ("coverage80", "Coverage q10-q90", 0.80),
        ("coverage50", "Coverage q25-q75", 0.50),
        ("width80", "Mean Width q10-q90 [K]", None),
        ("spread_skill_ratio", "Spread-Skill Ratio", 1.00),
        ("spread_error_corr", "Spread-Error Corr", None),
        ("pit_l1_uniform", "uPIT L1 From Uniform", 0.00),
    ]
    model_order = ["ML-120", "ML-4", "GEOS"]
    color_map = {"ML-120": "#08306b", "ML-4": "#2171b5", "GEOS": "#d94801"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title, target) in zip(axes, metrics):
        for model_name in model_order:
            model_rows = [row for row in rows if row["model"] == model_name]
            if not model_rows:
                continue
            lead_vals = [int(row["lead_week"]) for row in model_rows]
            metric_vals = [float(row[key]) for row in model_rows]
            ax.plot(lead_vals, metric_vals, marker="o", linewidth=2.0, color=color_map[model_name], label=model_name)
        if target is not None:
            ax.axhline(target, color="#7f7f7f", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Lead Week")
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Held-Out 2020-2021 T2M Dispersion Diagnostics", fontsize=15, y=1.04)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_pit_figure(hist_rows: List[Dict[str, object]], pit_bins: int, output_path: str):
    model_order = ["ML-120", "ML-4", "GEOS"]
    color_map = {"ML-120": "#08306b", "ML-4": "#2171b5", "GEOS": "#d94801"}
    bin_centers = np.linspace(0.5 / pit_bins, 1.0 - 0.5 / pit_bins, pit_bins)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True, sharex=True)
    for lead_idx, ax in enumerate(axes, start=1):
        ax.axhline(1.0 / pit_bins, color="#7f7f7f", linestyle="--", linewidth=1.0)
        for model_name in model_order:
            rows = [row for row in hist_rows if row["model"] == model_name and int(row["lead_week"]) == lead_idx]
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: int(item["bin_index"]))
            densities = [float(row["density"]) for row in rows]
            ax.plot(bin_centers, densities, marker="o", linewidth=1.8, color=color_map[model_name], label=model_name)
        ax.set_ylabel(f"W{lead_idx} density")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("uPIT")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Held-Out 2020-2021 T2M uPIT Histograms", fontsize=15, y=1.03)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str, fieldnames: Sequence[str], rows: List[Dict[str, object]]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or config["data_dir"]
    start_year = int(args.start_year if args.start_year is not None else config.get("val_start_year", 2020))
    end_year = int(args.end_year if args.end_year is not None else config.get("val_end_year", 2021))
    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "t2m_dispersion_summary.csv")
    pit_path = os.path.join(args.output_dir, "t2m_dispersion_pit.csv")
    report_path = os.path.join(args.output_dir, "t2m_dispersion_report.txt")
    plot_path = os.path.join(args.output_dir, "t2m_dispersion_summary.png")
    pit_plot_path = os.path.join(args.output_dir, "t2m_dispersion_pit.png")

    accumulators = {
        "ML-120": [DispersionAccumulator(args.pit_bins) for _ in range(4)],
        "ML-4": [DispersionAccumulator(args.pit_bins) for _ in range(4)],
        "GEOS": [DispersionAccumulator(args.pit_bins) for _ in range(4)],
    }
    year_common_counts = {}
    total_common_inits = 0

    for year in range(start_year, end_year + 1):
        ml_path = os.path.join(args.ml_dir, f"{year}.zarr")
        geos_path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
        obs_path = os.path.join(data_dir, f"t2m_weekly_{year}.zarr")

        ml_ds = open_year_dataset(ml_path)
        geos_ds = open_year_dataset(geos_path)
        obs_ds = open_year_dataset(obs_path)
        try:
            ml_layout = infer_layout(ml_ds, "ML")
            geos_layout = infer_layout(geos_ds, "GEOS")
            obs_layout = infer_layout(obs_ds, "OBS")

            common_dates = prepare_common_dates(ml_ds, ml_layout, geos_ds, geos_layout, obs_ds, obs_layout)
            if not common_dates:
                raise ValueError(f"No common init dates found for {year}.")
            year_common_counts[year] = len(common_dates)
            total_common_inits += len(common_dates)

            ml_indices = exact_indices_for_dates(ml_ds[ml_layout["s_dim"]].values, common_dates)
            geos_indices = exact_indices_for_dates(geos_ds[geos_layout["s_dim"]].values, common_dates)
            obs_indices = exact_indices_for_dates(obs_ds[obs_layout["s_dim"]].values, common_dates)

            lat_values = np.asarray(obs_ds[obs_layout["y_dim"]].values if obs_layout["y_dim"] in obs_ds.coords else np.arange(obs_ds.sizes[obs_layout["y_dim"]]), dtype=np.float64)
            lat_weights = np.clip(np.cos(np.deg2rad(lat_values)), 0.0, None)

            lead_count = min(
                int(ml_ds.sizes[ml_layout["lead_dim"]]),
                int(geos_ds.sizes[geos_layout["lead_dim"]]),
                int(obs_ds.sizes[obs_layout["lead_dim"]]),
                4,
            )

            for chunk_start in range(0, len(common_dates), args.sample_chunk_size):
                chunk_end = min(len(common_dates), chunk_start + args.sample_chunk_size)
                ml_chunk_idx = ml_indices[chunk_start:chunk_end]
                geos_chunk_idx = geos_indices[chunk_start:chunk_end]
                obs_chunk_idx = obs_indices[chunk_start:chunk_end]

                for lead_idx in range(lead_count):
                    ml_chunk = extract_tas_chunk(ml_ds, ml_layout, ml_chunk_idx, lead_idx)
                    geos_chunk = extract_tas_chunk(geos_ds, geos_layout, geos_chunk_idx, lead_idx)
                    obs_chunk = extract_tas_chunk(obs_ds, obs_layout, obs_chunk_idx, lead_idx)
                    if obs_chunk.ndim != 3:
                        raise ValueError(f"Expected obs chunk rank 3, got shape {obs_chunk.shape}")
                    if geos_chunk.ndim != 4:
                        raise ValueError(f"Expected GEOS chunk rank 4, got shape {geos_chunk.shape}")
                    if ml_chunk.ndim != 4:
                        raise ValueError(f"Expected ML chunk rank 4, got shape {ml_chunk.shape}")

                    ml_fair_chunk = downsample_member_axis(ml_chunk, args.fair_member_count)
                    accumulators["ML-120"][lead_idx].update(ml_chunk, obs_chunk, lat_weights)
                    accumulators["ML-4"][lead_idx].update(ml_fair_chunk, obs_chunk, lat_weights)
                    accumulators["GEOS"][lead_idx].update(geos_chunk, obs_chunk, lat_weights)
        finally:
            ml_ds.close()
            geos_ds.close()
            obs_ds.close()

    summary_rows: List[Dict[str, object]] = []
    pit_rows: List[Dict[str, object]] = []
    report_lines = [
        f"Held-out T2M dispersion evaluation for years {start_year}-{end_year}",
        f"Fair ML member count: {args.fair_member_count}",
        f"Total common init dates: {total_common_inits}",
        "Common init dates by year: " + ", ".join(f"{year}={count}" for year, count in sorted(year_common_counts.items())),
        "",
    ]

    for model_name, lead_accs in accumulators.items():
        report_lines.append(f"[{model_name}]")
        for lead_idx, acc in enumerate(lead_accs, start=1):
            metrics = acc.finalize()
            metrics["model"] = model_name
            metrics["lead_week"] = lead_idx
            metrics["dispersion_state"] = classify_dispersion(metrics["coverage80"], metrics["spread_skill_ratio"])
            summary_rows.append(metrics)

            pit_total = np.sum(acc.pit_hist)
            pit_density = acc.pit_hist / pit_total if pit_total > 0.0 else np.full(args.pit_bins, np.nan)
            for bin_idx in range(args.pit_bins):
                pit_rows.append(
                    {
                        "model": model_name,
                        "lead_week": lead_idx,
                        "bin_index": bin_idx,
                        "bin_left": acc.pit_edges[bin_idx],
                        "bin_right": acc.pit_edges[bin_idx + 1],
                        "density": float(pit_density[bin_idx]),
                    }
                )

            report_lines.append(
                "  "
                f"W{lead_idx}: coverage80={metrics['coverage80']:.3f}, coverage50={metrics['coverage50']:.3f}, "
                f"below_q10={metrics['below_q10']:.3f}, above_q90={metrics['above_q90']:.3f}, "
                f"width80={metrics['width80']:.3f} K, mean_spread={metrics['mean_spread']:.3f} K, "
                f"rmse_mean={metrics['rmse_mean']:.3f} K, ssr={metrics['spread_skill_ratio']:.3f}, "
                f"spread_err_corr={metrics['spread_error_corr']:.3f}, pit_l1={metrics['pit_l1_uniform']:.3f}, "
                f"state={metrics['dispersion_state']}"
            )
        report_lines.append("")

    summary_fields = [
        "model",
        "lead_week",
        "sample_count",
        "coverage80",
        "coverage50",
        "below_q10",
        "above_q90",
        "width80",
        "width50",
        "mean_spread",
        "rmse_mean",
        "spread_skill_ratio",
        "spread_error_corr",
        "pit_l1_uniform",
        "dispersion_state",
    ]
    pit_fields = ["model", "lead_week", "bin_index", "bin_left", "bin_right", "density"]
    write_csv(summary_path, summary_fields, summary_rows)
    write_csv(pit_path, pit_fields, pit_rows)
    make_summary_figure(summary_rows, plot_path)
    make_pit_figure(pit_rows, args.pit_bins, pit_plot_path)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines).rstrip() + "\n")

    print(f"✅ Saved dispersion summary CSV: {summary_path}")
    print(f"✅ Saved dispersion uPIT CSV: {pit_path}")
    print(f"✅ Saved dispersion report: {report_path}")
    print(f"✅ Saved dispersion summary plot: {plot_path}")
    print(f"✅ Saved dispersion uPIT plot: {pit_plot_path}")


if __name__ == "__main__":
    main()
