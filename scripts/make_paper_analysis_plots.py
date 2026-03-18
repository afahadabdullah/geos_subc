#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_common import (
    apply_manuscript_style,
    ensure_dir,
    latest_match,
    prettify_strategy,
    save_figure,
    strategy_color,
)


NOISE_COL_RE = re.compile(
    r"^(?P<strategy>.+?) \((?P<lead>Total|W1|W2|W3|W4)\) (?P<variable>PR|T2M)_(?P<metric>CRPS|RMSE)$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper analysis plots from native repo outputs.")
    parser.add_argument("--base-dir", default=".", help="Directory to search for outputs.")
    parser.add_argument("--output-dir", default="paper/figures", help="Directory for generated figures.")
    parser.add_argument("--noise-results", default=None, help="Path to noise_comparison_v4_multi_results_<year>.csv")
    parser.add_argument("--checkpoint-summary", default=None, help="Path to checkpoint_pure_noise_summary_<year>.csv")
    parser.add_argument("--training-log", default=None, help="Path to training_log_v5.csv")
    parser.add_argument("--model-registry", default=None, help="Path to model_registry.json")
    parser.add_argument("--test-summary", default=None, help="Path to test_summary_multi.json")
    parser.add_argument("--demo", action="store_true", help="Generate figures from demo data instead of discovered files.")
    return parser.parse_args()


def discover_inputs(base_dir):
    base_dir = Path(base_dir)
    return {
        "noise_results": latest_match(base_dir, "**/noise_comparison_v4_multi_results_*.csv"),
        "checkpoint_summary": latest_match(base_dir, "**/checkpoint_pure_noise_summary_*.csv"),
        "training_log": latest_match(base_dir, "**/training_log_v5.csv"),
        "model_registry": latest_match(base_dir, "**/model_registry.json"),
        "test_summary": latest_match(base_dir, "**/test_summary_multi.json"),
    }


def load_noise_results(csv_path):
    df = pd.read_csv(csv_path)
    if "batch" in df.columns:
        mean_rows = df[df["batch"].astype(str).str.upper() == "MEAN"]
        row = mean_rows.iloc[0] if not mean_rows.empty else df.mean(numeric_only=True)
    else:
        row = df.mean(numeric_only=True)

    rows = []
    source = row.to_dict() if isinstance(row, pd.Series) else row
    for column, value in source.items():
        match = NOISE_COL_RE.match(str(column))
        if not match:
            continue
        rows.append(
            {
                "strategy": match.group("strategy"),
                "lead": match.group("lead"),
                "variable": match.group("variable"),
                "metric": match.group("metric"),
                "value": float(value),
            }
        )

    parsed = pd.DataFrame(rows)
    if parsed.empty:
        raise ValueError(f"Could not parse any metric columns from {csv_path}")
    parsed["lead_index"] = parsed["lead"].map({"W1": 1, "W2": 2, "W3": 3, "W4": 4}).fillna(0).astype(int)
    return parsed


def load_checkpoint_summary(csv_path):
    df = pd.read_csv(csv_path)
    if "name" not in df.columns:
        raise ValueError(f"Checkpoint summary is missing 'name': {csv_path}")
    df = df.copy()
    df["display_name"] = df["name"].map(prettify_strategy)
    df["epoch_guess"] = df["name"].str.extract(r"E(\d+)").astype(float)
    return df


def load_training_log(csv_path):
    df = pd.read_csv(csv_path)
    rename_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=rename_map)
    return df


def load_registry(json_path):
    with open(json_path, "r") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return data


def load_test_summary(json_path):
    with open(json_path, "r") as handle:
        return json.load(handle)


def make_demo_noise_results():
    rows = []
    strategies = {
        "0. GEOS Baseline": {
            "PR": {"CRPS": [1.82, 1.94, 2.06, 2.18, 2.32], "RMSE": [8.6, 9.4, 10.3, 11.2, 12.0]},
            "T2M": {"CRPS": [1.12, 1.18, 1.25, 1.31, 1.38], "RMSE": [2.6, 2.9, 3.2, 3.5, 3.8]},
        },
        "1. Pure Random": {
            "PR": {"CRPS": [1.74, 1.88, 2.00, 2.12, 2.25], "RMSE": [8.3, 9.1, 10.0, 10.8, 11.7]},
            "T2M": {"CRPS": [1.08, 1.15, 1.21, 1.28, 1.34], "RMSE": [2.5, 2.8, 3.1, 3.4, 3.7]},
        },
        "3. EOF-LHS + Var": {
            "PR": {"CRPS": [1.63, 1.76, 1.88, 2.01, 2.14], "RMSE": [7.9, 8.7, 9.6, 10.4, 11.2]},
            "T2M": {"CRPS": [1.02, 1.08, 1.15, 1.21, 1.28], "RMSE": [2.3, 2.6, 2.9, 3.2, 3.4]},
        },
    }
    leads = ["Total", "W1", "W2", "W3", "W4"]
    for strategy, by_var in strategies.items():
        for variable, by_metric in by_var.items():
            for metric, values in by_metric.items():
                for lead, value in zip(leads, values):
                    rows.append(
                        {
                            "strategy": strategy,
                            "lead": lead,
                            "lead_index": 0 if lead == "Total" else int(lead[1:]),
                            "variable": variable,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)


def make_demo_checkpoint_summary():
    return pd.DataFrame(
        [
            {"rank": 1, "name": "1. E210", "display_name": "E210", "combined_crps": 1.425, "combined_rmse": 6.30, "pr_crps": 1.78, "t2m_crps": 1.07},
            {"rank": 2, "name": "2. E180", "display_name": "E180", "combined_crps": 1.448, "combined_rmse": 6.42, "pr_crps": 1.81, "t2m_crps": 1.09},
            {"rank": 3, "name": "3. E150", "display_name": "E150", "combined_crps": 1.470, "combined_rmse": 6.57, "pr_crps": 1.84, "t2m_crps": 1.10},
            {"rank": 4, "name": "0. GEOS", "display_name": "GEOS", "combined_crps": 1.520, "combined_rmse": 6.90, "pr_crps": 1.93, "t2m_crps": 1.11},
        ]
    )


def make_demo_training_log():
    epochs = np.arange(1, 61)
    train_loss = 0.18 * np.exp(-epochs / 20.0) + 0.04
    val_crps = 1.85 - 0.45 * (1.0 - np.exp(-(epochs - 5).clip(min=0) / 18.0))
    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_loss,
            "val_noise": np.zeros_like(epochs, dtype=float),
            "val_crps": val_crps,
        }
    )


def make_demo_registry():
    return [
        {"rank": 1, "epoch": 52, "val_loss": 1.41},
        {"rank": 2, "epoch": 47, "val_loss": 1.43},
        {"rank": 3, "epoch": 41, "val_loss": 1.45},
    ]


def make_demo_test_summary():
    return {
        "avg_crps_pr": 1.76,
        "avg_rmse_pr": 8.8,
        "avg_geos_crps_pr": 1.93,
        "avg_geos_rmse_pr": 9.6,
        "avg_crps_t2m": 1.06,
        "avg_rmse_t2m": 2.7,
        "avg_geos_crps_t2m": 1.13,
        "avg_geos_rmse_t2m": 3.0,
    }


def plot_overall_summary(noise_df, test_summary, output_dir):
    overall = noise_df[noise_df["lead"] == "Total"].copy()
    if overall.empty and test_summary is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    panels = [
        ("PR", "CRPS", "Precipitation CRPS"),
        ("PR", "RMSE", "Precipitation RMSE"),
        ("T2M", "CRPS", "2 m Temperature CRPS"),
        ("T2M", "RMSE", "2 m Temperature RMSE"),
    ]

    for ax, (variable, metric, title) in zip(axes.flatten(), panels):
        subset = overall[(overall["variable"] == variable) & (overall["metric"] == metric)].copy()
        if subset.empty and test_summary is not None:
            if variable == "PR":
                vals = [
                    ("GEOS", test_summary[f"avg_geos_{metric.lower()}_pr"]),
                    ("Hybrid", test_summary[f"avg_{metric.lower()}_pr"]),
                ]
            else:
                vals = [
                    ("GEOS", test_summary[f"avg_geos_{metric.lower()}_t2m"]),
                    ("Hybrid", test_summary[f"avg_{metric.lower()}_t2m"]),
                ]
            subset = pd.DataFrame(vals, columns=["strategy", "value"])
        subset = subset.sort_values("value")
        ax.bar(
            [prettify_strategy(name) for name in subset["strategy"]],
            subset["value"],
            color=[strategy_color(name) for name in subset["strategy"]],
        )
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Overall Probabilistic and Deterministic Skill", fontsize=14, fontweight="bold")
    output_path = output_dir / "paper_analysis_overall_skill.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_lead_skill(noise_df, output_dir):
    lead_df = noise_df[noise_df["lead"] != "Total"].copy()
    if lead_df.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    panels = [
        ("PR", "CRPS", "Precipitation CRPS by Lead"),
        ("PR", "RMSE", "Precipitation RMSE by Lead"),
        ("T2M", "CRPS", "2 m Temperature CRPS by Lead"),
        ("T2M", "RMSE", "2 m Temperature RMSE by Lead"),
    ]

    for ax, (variable, metric, title) in zip(axes.flatten(), panels):
        subset = lead_df[(lead_df["variable"] == variable) & (lead_df["metric"] == metric)].copy()
        for strategy in subset["strategy"].drop_duplicates():
            group = subset[subset["strategy"] == strategy].sort_values("lead_index")
            ax.plot(
                group["lead_index"],
                group["value"],
                marker="o",
                linewidth=2,
                label=prettify_strategy(strategy),
                color=strategy_color(strategy),
            )
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4], labels=["W1", "W2", "W3", "W4"])
        ax.set_ylabel(metric)

    axes[0, 0].legend(loc="best", frameon=False)
    fig.suptitle("Lead-Dependent Skill Degradation", fontsize=14, fontweight="bold")
    output_path = output_dir / "paper_analysis_lead_skill.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_checkpoint_sweep(summary_df, output_dir):
    if summary_df is None or summary_df.empty:
        return None

    plot_df = summary_df.copy().sort_values("combined_crps")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    axes[0].bar(
        plot_df["display_name"],
        plot_df["combined_crps"],
        color=[strategy_color(name) for name in plot_df["display_name"]],
    )
    axes[0].set_title("Checkpoint Ranking by Combined CRPS")
    axes[0].set_ylabel("Combined CRPS")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].scatter(plot_df["pr_crps"], plot_df["t2m_crps"], s=90, color="#E45756")
    for _, row in plot_df.iterrows():
        axes[1].annotate(row["display_name"], (row["pr_crps"], row["t2m_crps"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1].set_title("Checkpoint Trade-off: PR vs T2M CRPS")
    axes[1].set_xlabel("PR CRPS")
    axes[1].set_ylabel("T2M CRPS")

    fig.suptitle("Checkpoint Sweep Summary", fontsize=14, fontweight="bold")
    output_path = output_dir / "paper_analysis_checkpoint_sweep.pdf"
    save_figure(fig, output_path)
    return output_path


def plot_training_diagnostics(training_df, registry, output_dir):
    if training_df is None or training_df.empty:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)

    axes[0].plot(training_df["epoch"], training_df["train_loss"], color="#4C78A8", linewidth=2)
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss Trajectory")

    if "val_crps" in training_df.columns:
        axes[1].plot(training_df["epoch"], training_df["val_crps"], color="#E45756", linewidth=2)
        axes[1].set_ylabel("Validation CRPS")
        axes[1].set_title("Validation CRPS Trajectory")
        for item in registry or []:
            epoch = item.get("epoch")
            val_loss = item.get("val_loss")
            if epoch is not None and val_loss is not None:
                axes[1].scatter([epoch], [val_loss], color="#54A24B", s=45, zorder=5)

    axes[1].set_xlabel("Epoch")
    fig.suptitle("Training and Model-Selection Diagnostics", fontsize=14, fontweight="bold")
    output_path = output_dir / "paper_analysis_training_curves.pdf"
    save_figure(fig, output_path)
    return output_path


def main():
    args = parse_args()
    apply_manuscript_style()
    output_dir = ensure_dir(args.output_dir)

    discovered = discover_inputs(args.base_dir)
    paths = {
        "noise_results": Path(args.noise_results) if args.noise_results else discovered["noise_results"],
        "checkpoint_summary": Path(args.checkpoint_summary) if args.checkpoint_summary else discovered["checkpoint_summary"],
        "training_log": Path(args.training_log) if args.training_log else discovered["training_log"],
        "model_registry": Path(args.model_registry) if args.model_registry else discovered["model_registry"],
        "test_summary": Path(args.test_summary) if args.test_summary else discovered["test_summary"],
    }

    if args.demo:
        noise_df = make_demo_noise_results()
        checkpoint_df = make_demo_checkpoint_summary()
        training_df = make_demo_training_log()
        registry = make_demo_registry()
        test_summary = make_demo_test_summary()
        print("Using demo data for paper-analysis plots.")
    else:
        noise_df = load_noise_results(paths["noise_results"]) if paths["noise_results"] else None
        checkpoint_df = load_checkpoint_summary(paths["checkpoint_summary"]) if paths["checkpoint_summary"] else None
        training_df = load_training_log(paths["training_log"]) if paths["training_log"] else None
        registry = load_registry(paths["model_registry"]) if paths["model_registry"] else []
        test_summary = load_test_summary(paths["test_summary"]) if paths["test_summary"] else None
        if noise_df is None and checkpoint_df is None and training_df is None and test_summary is None:
            print("No native outputs found. Falling back to demo data.")
            noise_df = make_demo_noise_results()
            checkpoint_df = make_demo_checkpoint_summary()
            training_df = make_demo_training_log()
            registry = make_demo_registry()
            test_summary = make_demo_test_summary()

    written = []
    if noise_df is not None:
        path = plot_overall_summary(noise_df, test_summary, output_dir)
        if path:
            written.append(path)
        path = plot_lead_skill(noise_df, output_dir)
        if path:
            written.append(path)
    if checkpoint_df is not None:
        path = plot_checkpoint_sweep(checkpoint_df, output_dir)
        if path:
            written.append(path)
    if training_df is not None:
        path = plot_training_diagnostics(training_df, registry, output_dir)
        if path:
            written.append(path)

    if written:
        print("Wrote paper-analysis plots:")
        for path in written:
            print(f"  - {path}")
    else:
        print("No figures were generated.")


if __name__ == "__main__":
    main()
