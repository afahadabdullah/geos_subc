from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("qm_extreme_event_comparison.py")
SPEC = importlib.util.spec_from_file_location("qm_extreme_event_comparison", SCRIPT)
qm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qm)


class MetricTests(unittest.TestCase):
    def test_system_sums_and_scores(self):
        obs = np.asarray([[0.0, 1.0]], dtype=np.float32)
        forecast = np.asarray(
            [
                [[0.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 2.0]],
                [[0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        weights = np.ones_like(obs, dtype=np.float64)
        mask = np.ones_like(obs, dtype=bool)
        scores = qm.scores_from_sums(qm.system_sums(forecast, obs, weights, mask))
        self.assertAlmostEqual(scores["crps"], 0.0625)
        self.assertAlmostEqual(scores["rmse"], 0.0)
        self.assertAlmostEqual(scores["bias"], 0.0)
        self.assertTrue(np.isfinite(scores["q95_score"]))

    def test_safe_skill(self):
        self.assertAlmostEqual(qm.safe_skill(1.5, 2.0), 25.0)
        self.assertTrue(np.isnan(qm.safe_skill(1.0, 0.0)))


class SummaryTests(unittest.TestCase):
    @staticmethod
    def synthetic_case_frame() -> pd.DataFrame:
        rows = []
        for case_id, lead in (("case_a", 3), ("case_b", 4)):
            for system, repeat_count, score in (
                ("raw4", 1, 2.0),
                ("qm4", 1, 1.5),
                ("flow4", 2, 1.25),
                ("flow6", 2, 1.10),
                ("flow8", 2, 1.0),
                ("flow90", 1, 0.90),
            ):
                for repeat in range(repeat_count):
                    row = {
                        "case_id": case_id,
                        "variable": "pr",
                        "lead": lead,
                        "region": "uk",
                        "region_name": "United Kingdom",
                        "system": system,
                        "member_count": qm.SYSTEM_MEMBERS[system],
                        "member_repeat": repeat,
                        "weight_sum": 1.0,
                        "crps_sum": score,
                        "sse_sum": score * score,
                        "bias_sum": 0.0,
                        "spread_sum": 0.5,
                        "q95_score_sum": score,
                        "q95_sse_sum": score * score,
                        "forecast_mean_sum": 10.0 - score,
                        "obs_sum": 10.0,
                    }
                    row.update(qm.scores_from_sums(row))
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_flow8_qm4_summary_and_all_lead_row(self):
        comparisons = qm.comparison_case_rows(self.synthetic_case_frame())
        summary = qm.comparison_summary(comparisons)
        row = summary[
            summary["comparison"].eq("flow8_vs_qm4")
            & summary["lead"].astype(str).eq("all")
        ].iloc[0]
        self.assertEqual(int(row["n_cases"]), 2)
        self.assertEqual(int(row["n_member_repeats"]), 2)
        self.assertAlmostEqual(row["case_mean_crps_skill_pct_mean"], 100.0 / 3.0)
        self.assertAlmostEqual(row["crps_skill_pct_mean"], 100.0 / 3.0)

    def test_fixed_and_repeated_system_rows_align(self):
        comparisons = qm.comparison_case_rows(self.synthetic_case_frame())
        counts = comparisons.groupby("comparison").size().to_dict()
        self.assertEqual(counts["qm4_vs_raw4"], 2)
        self.assertEqual(counts["flow6_vs_raw4"], 4)
        self.assertEqual(counts["flow8_vs_raw4"], 4)
        self.assertEqual(counts["flow90_vs_raw4"], 2)
        self.assertEqual(counts["flow4_vs_qm4"], 4)


if __name__ == "__main__":
    unittest.main()
