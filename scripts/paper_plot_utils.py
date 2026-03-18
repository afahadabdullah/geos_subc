from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


LEAD_SUFFIXES = [" (Total)", " (W1)", " (W2)", " (W3)", " (W4)"]
METRIC_KEYS = ["PR_CRPS", "PR_RMSE", "T2M_CRPS", "T2M_RMSE"]


def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r") as handle:
        return json.load(handle)


def get_mean_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "batch" in df.columns:
        mean_rows = df[df["batch"].astype(str) == "MEAN"]
        if not mean_rows.empty:
            return mean_rows.iloc[0]
    return df.iloc[-1]


def extract_strategy_names(columns: Iterable[str]) -> List[str]:
    names = set()
    for column in columns:
        for suffix in LEAD_SUFFIXES:
            for metric in METRIC_KEYS:
                token = f"{suffix} {metric}"
                if column.endswith(token):
                    names.add(column[: -len(token)])
    return sorted(names)


def strategy_metric(mean_row: pd.Series, strategy: str, lead_suffix: str, metric: str) -> float:
    key = f"{strategy}{lead_suffix} {metric}"
    return float(mean_row[key])


def build_strategy_summary(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    mean_row = get_mean_row(df)
    if mean_row is None:
        return None

    rows = []
    for strategy in extract_strategy_names(df.columns):
        row = {"strategy": strategy}
        for metric in METRIC_KEYS:
            row[metric] = strategy_metric(mean_row, strategy, " (Total)", metric)
        row["COMBINED_CRPS"] = 0.5 * (row["PR_CRPS"] + row["T2M_CRPS"])
        row["COMBINED_RMSE"] = 0.5 * (row["PR_RMSE"] + row["T2M_RMSE"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_lead_summary(df: pd.DataFrame, strategy: str) -> Optional[pd.DataFrame]:
    mean_row = get_mean_row(df)
    if mean_row is None:
        return None

    rows = []
    for week_idx in range(1, 5):
        suffix = f" (W{week_idx})"
        rows.append(
            {
                "lead": week_idx,
                "PR_CRPS": strategy_metric(mean_row, strategy, suffix, "PR_CRPS"),
                "PR_RMSE": strategy_metric(mean_row, strategy, suffix, "PR_RMSE"),
                "T2M_CRPS": strategy_metric(mean_row, strategy, suffix, "T2M_CRPS"),
                "T2M_RMSE": strategy_metric(mean_row, strategy, suffix, "T2M_RMSE"),
            }
        )
    return pd.DataFrame(rows)


def build_test_summary_table(summary: Dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "GEOS",
                "PR_CRPS": summary["avg_geos_crps_pr"],
                "PR_RMSE": summary["avg_geos_rmse_pr"],
                "T2M_CRPS": summary["avg_geos_crps_t2m"],
                "T2M_RMSE": summary["avg_geos_rmse_t2m"],
            },
            {
                "model": "Hybrid FlowMatch-S2S",
                "PR_CRPS": summary["avg_crps_pr"],
                "PR_RMSE": summary["avg_rmse_pr"],
                "T2M_CRPS": summary["avg_crps_t2m"],
                "T2M_RMSE": summary["avg_rmse_t2m"],
            },
        ]
    )


def sanitize_label(label: str) -> str:
    label = label.replace("0. ", "")
    label = label.replace("1. ", "")
    label = label.replace("2. ", "")
    label = label.replace("3. ", "")
    return label.strip()
