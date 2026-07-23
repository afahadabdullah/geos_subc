#!/usr/bin/env python3
"""Synthetic unit tests for the FIMr1p1 quantile-mapping baseline."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from paper.scripts.review_response.qm_fim_baseline import (
    RawYear,
    apply_all,
    apply_quantile_map,
    fit_all,
    fit_quantile_parameters,
    parameter_complete_marker,
    parameter_progress_dir,
    parameter_store,
    parse_years,
    score_all,
    standardize_forecast,
    standardize_observation,
    tile_progress_marker,
    validate_year_splits,
)


class QuantileMappingTests(unittest.TestCase):
    def test_default_split_is_disjoint_and_chronological(self):
        train = parse_years("1999-2019")
        validation = parse_years("2020")
        evaluation = parse_years("2021-2023")
        validate_year_splits(train, validation, evaluation)
        self.assertEqual(train[0], 1999)
        self.assertEqual(train[-1], 2019)
        self.assertEqual(evaluation, (2021, 2022, 2023))

    def test_split_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_year_splits((1999, 2000), (2000,), (2001,))

    def test_t2m_affine_bias_is_removed_and_tail_is_not_capped(self):
        forecast = np.arange(10, dtype=np.float32)[:, None, None] + 270.0
        observation = forecast + 2.0
        quantiles = np.linspace(0.0, 1.0, 6)
        parameters = fit_quantile_parameters(
            forecast, observation, "t2m", quantiles
        )
        values = np.asarray([[[273.5]], [[281.0]]], dtype=np.float32)
        corrected = apply_quantile_map(values, parameters, "t2m")
        np.testing.assert_allclose(corrected, values + 2.0, atol=1e-5)

    def test_precipitation_mapping_preserves_dry_mass_and_nonnegativity(self):
        forecast = np.asarray(
            [0.0, 0.02, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0],
            dtype=np.float32,
        )[:, None, None]
        observation = np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 1.5, 3.0, 6.0, 10.0],
            dtype=np.float32,
        )[:, None, None]
        quantiles = np.linspace(0.0, 1.0, 6)
        parameters = fit_quantile_parameters(
            forecast,
            observation,
            "pr",
            quantiles,
            wet_threshold=0.1,
            min_wet_samples=3,
        )
        values = np.asarray([[[-1.0]], [[0.05]], [[1.0]], [[15.0]]], dtype=np.float32)
        corrected = apply_quantile_map(
            values, parameters, "pr", wet_threshold=0.1
        )
        self.assertEqual(float(corrected[0, 0, 0]), 0.0)
        self.assertEqual(float(corrected[1, 0, 0]), 0.0)
        self.assertGreater(float(corrected[2, 0, 0]), 0.0)
        self.assertGreater(float(corrected[3, 0, 0]), float(corrected[2, 0, 0]))
        self.assertTrue(np.all(corrected >= 0.0))

    def test_archive_dimensions_are_canonicalized_without_loading(self):
        forecast = xr.DataArray(
            np.zeros((2, 3, 4, 5, 6), dtype=np.float32),
            dims=("S", "M", "L", "Y", "X"),
            coords={"S": np.asarray(["2000-01-01", "2000-01-08"], dtype="datetime64[ns]")},
        )
        observation = xr.DataArray(
            np.zeros((2, 4, 5, 6), dtype=np.float32),
            dims=("S", "L", "Y", "X"),
            coords={"S": np.asarray(["2000-01-01", "2000-01-08"], dtype="datetime64[ns]")},
        )
        standardized_forecast = standardize_forecast(forecast)
        standardized_observation = standardize_observation(observation)
        self.assertEqual(
            standardized_forecast.dims,
            ("init", "member", "lead", "lat", "lon"),
        )
        self.assertEqual(
            standardized_observation.dims,
            ("init", "lead", "lat", "lon"),
        )
        np.testing.assert_array_equal(
            standardized_forecast["lead"].values, np.arange(1, 5)
        )

    def test_noncanonical_lazy_observation_is_tiled_before_transpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "t2m_weekly_2000.zarr"
            init = pd.date_range("2000-01-01", periods=5, freq="7D")
            values = np.arange(5 * 2 * 3 * 4, dtype=np.float32).reshape(
                5, 2, 3, 4
            )
            xr.Dataset(
                {
                    "t2m": (
                        ("S", "Y", "X", "L"),
                        values,
                    )
                },
                coords={"S": init.values},
            ).to_zarr(path, mode="w")

            ds = xr.open_zarr(path, consolidated=False, chunks=None)
            try:
                standardized = standardize_observation(ds["t2m"])
                self.assertEqual(
                    standardized.dims,
                    ("init", "lat", "lon", "lead"),
                )
                self.assertNotIn(
                    "Vectorized",
                    type(standardized.variable._data).__name__,
                )
                selected = (
                    standardized.isel(
                        init=np.asarray([1, 3]),
                        lead=2,
                        lat=slice(0, 1),
                        lon=slice(1, 3),
                    )
                    .transpose("init", "lat", "lon")
                    .values
                )
                expected = np.stack(
                    [values[index, 0:1, 1:3, 2] for index in (1, 3)]
                )
                np.testing.assert_array_equal(selected, expected)
            finally:
                ds.close()

            forecast = np.arange(
                5 * 2 * 4 * 2 * 3, dtype=np.float32
            ).reshape(5, 2, 4, 2, 3)
            xr.Dataset(
                {
                    "tas": (
                        ("S", "M", "L", "Y", "X"),
                        forecast,
                    )
                },
                coords={"S": init.values},
            ).to_zarr(root / "geos_subc_2000.zarr", mode="w")
            archive = RawYear(root, 2000, ("t2m",))
            try:
                forecast_tile, observation_tile = archive.training_samples(
                    "t2m",
                    lead_index=0,
                    verifying_month=1,
                    max_inits=2,
                    lat_slice=slice(0, 1),
                    lon_slice=slice(1, 3),
                )
                self.assertEqual(forecast_tile.shape, (4, 1, 2))
                self.assertEqual(observation_tile.shape, (2, 1, 2))
            finally:
                archive.close()


@unittest.skipUnless(
    os.environ.get("RUN_QM_INTEGRATION") == "1",
    "set RUN_QM_INTEGRATION=1 for the Zarr end-to-end smoke test",
)
class QuantileMappingIntegrationTest(unittest.TestCase):
    def test_tiny_fit_apply_score_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            forecast_dir = root / "forecasts"
            out_dir = root / "output"
            data_root.mkdir()
            forecast_dir.mkdir()

            for year in (1999, 2000, 2001):
                init = pd.date_range(f"{year}-01-01", periods=24, freq="15D")
                month = np.arange(len(init), dtype=np.float32)[:, None, None, None, None]
                member = np.arange(2, dtype=np.float32)[None, :, None, None, None]
                lead = np.arange(4, dtype=np.float32)[None, None, :, None, None]
                spatial = np.arange(6, dtype=np.float32).reshape(1, 1, 1, 2, 3)
                raw = 0.2 + month + member + lead + spatial
                obs = raw.mean(axis=1) * 0.8
                xr.Dataset(
                    {"pr": (("S", "M", "L", "Y", "X"), raw.astype(np.float32))},
                    coords={"S": init.values},
                ).to_zarr(data_root / f"geos_subc_{year}.zarr", mode="w")
                xr.Dataset(
                    {"precip": (("S", "L", "Y", "X"), obs.astype(np.float32))},
                    coords={"S": init.values},
                ).to_zarr(data_root / f"gpcp_weekly_{year}.zarr", mode="w")

            eval_init = pd.date_range("2002-01-01", periods=24, freq="15D")
            month = np.arange(len(eval_init), dtype=np.float32)[:, None, None, None, None]
            member = np.arange(4, dtype=np.float32)[None, :, None, None, None]
            lead = np.arange(4, dtype=np.float32)[None, None, :, None, None]
            spatial = np.arange(6, dtype=np.float32).reshape(1, 1, 1, 2, 3)
            eval_raw = 0.2 + month + member + lead + spatial
            eval_obs = eval_raw.mean(axis=1) * 0.8
            model = np.concatenate(
                [eval_raw * 0.8, eval_raw * 0.8 + 0.1], axis=1
            )
            xr.Dataset(
                {
                    "model_pr": (
                        ("init", "ensemble", "lead", "lat", "lon"),
                        model.astype(np.float32),
                    ),
                    "geos_pr": (
                        ("init", "geos_member", "lead", "lat", "lon"),
                        eval_raw.astype(np.float32),
                    ),
                    "obs_pr": (
                        ("init", "lead", "lat", "lon"),
                        eval_obs.astype(np.float32),
                    ),
                },
                coords={
                    "init": eval_init.values,
                    "ensemble": np.arange(1, 9),
                    "geos_member": np.arange(1, 5),
                    "lead": [1, 2, 3, 4],
                    "lat": [-1.0, 1.0],
                    "lon": [0.0, 1.0, 2.0],
                },
            ).to_zarr(forecast_dir / "2002.zarr", mode="w")

            args = SimpleNamespace(
                validation_years="2001",
                evaluation_years="2002",
                overwrite=False,
                wet_threshold=0.1,
                min_wet_samples=2,
                fit_lat_tile=1,
                fit_lon_tile=2,
                spatial_block_size=8,
                max_inits=None,
                subsample_repeats=2,
            )
            train_years = (1999, 2000)
            variables = ("pr",)
            parameter_root = out_dir / "qm_parameters"
            corrected_dir = out_dir / "corrected"
            fit_all(
                data_root,
                parameter_root,
                train_years,
                variables,
                np.linspace(0.0, 1.0, 5),
                args,
            )

            # Simulate an abrupt kill after three of four spatial tiles were
            # checkpointed. The rerun must repair only the unmarked tile.
            resume_store = parameter_store(parameter_root, "pr", 1, 1)
            missing_tile = tile_progress_marker(resume_store, 0, 1, 0, 2)
            missing_tile.unlink()
            parameter_complete_marker(resume_store).unlink()
            resume_group = zarr.open_group(str(resume_store), mode="a")
            resume_group["forecast_quantile"][:, 0:1, 0:2] = np.nan
            del resume_group

            untouched_store = parameter_store(parameter_root, "pr", 1, 2)
            untouched_completion = parameter_complete_marker(
                untouched_store
            ).read_text()
            fit_output = io.StringIO()
            with contextlib.redirect_stdout(fit_output):
                fit_all(
                    data_root,
                    parameter_root,
                    train_years,
                    variables,
                    np.linspace(0.0, 1.0, 5),
                    args,
                )
            self.assertIn("3/4 tiles already saved", fit_output.getvalue())
            self.assertTrue(missing_tile.exists())
            self.assertTrue(parameter_complete_marker(resume_store).exists())
            self.assertEqual(
                len(list(parameter_progress_dir(resume_store).glob("tile_*.done"))),
                4,
            )
            repaired_group = zarr.open_group(str(resume_store), mode="r")
            self.assertTrue(
                np.isfinite(
                    repaired_group["forecast_quantile"][:, 0:1, 0:2]
                ).all()
            )
            self.assertEqual(
                parameter_complete_marker(untouched_store).read_text(),
                untouched_completion,
            )

            # Stores written by the previous implementation have neither
            # checkpoint-format metadata nor progress/completion sidecars.
            # They should be adopted rather than discarded and refit.
            legacy_group = zarr.open_group(str(untouched_store), mode="a")
            del legacy_group.attrs["checkpoint_format_version"]
            del legacy_group
            shutil.rmtree(parameter_progress_dir(untouched_store))
            parameter_complete_marker(untouched_store).unlink()
            legacy_output = io.StringIO()
            with contextlib.redirect_stdout(legacy_output):
                fit_all(
                    data_root,
                    parameter_root,
                    train_years,
                    variables,
                    np.linspace(0.0, 1.0, 5),
                    args,
                )
            self.assertIn(
                "Adopted existing completed store", legacy_output.getvalue()
            )
            self.assertTrue(
                parameter_complete_marker(untouched_store).exists()
            )

            apply_all(
                data_root,
                forecast_dir,
                corrected_dir,
                parameter_root,
                train_years,
                (2001,),
                (2002,),
                variables,
                args,
            )
            score_all(
                data_root,
                forecast_dir,
                corrected_dir,
                out_dir,
                (2001,),
                (2002,),
                variables,
                args,
            )
            aggregate = pd.read_csv(out_dir / "qm_aggregate_metrics.csv")
            self.assertEqual(set(aggregate["split"]), {"validation", "evaluation"})
            self.assertTrue(np.isfinite(aggregate["qm_crps"]).all())
            self.assertTrue((aggregate["qm_crps"] >= 0.0).all())

            comparison_dir = out_dir / "flow_comparison"
            subprocess.run(
                [
                    sys.executable,
                    "paper/scripts/review_response/r1_fair_verification.py",
                    "--forecast_dir",
                    str(forecast_dir),
                    "--qm_dir",
                    str(corrected_dir),
                    "--years",
                    "2002",
                    "--variables",
                    "pr",
                    "--components",
                    "qm,boot",
                    "--model_members",
                    "8",
                    "--subsample_repeats",
                    "2",
                    "--n_boot",
                    "60",
                    "--out_dir",
                    str(comparison_dir),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[3],
                stdout=subprocess.DEVNULL,
            )
            comparison = pd.read_csv(
                comparison_dir / "r1_aggregate_metrics.csv"
            )
            self.assertTrue(np.isfinite(comparison["skill_model_vs_qm"]).all())
            self.assertTrue(
                np.isfinite(comparison["skill_model_k_vs_raw"]).all()
            )
            self.assertTrue(
                np.isfinite(comparison["skill_model_k_vs_qm"]).all()
            )
            self.assertTrue(
                (comparison["n_members_model_subsample"] == 8).all()
            )
            self.assertTrue((comparison["n_members_geos"] == 4).all())
            self.assertTrue((comparison["n_members_qm"] == 4).all())
            self.assertTrue(
                (comparison["n_members_model_qm_subsample"] == 4).all()
            )
            pooled = comparison[
                comparison["year"].eq("pooled")
                & comparison["lead"].eq("all")
            ]
            self.assertTrue(
                np.isfinite(pooled["skill_model_k_vs_qm_ci_lo"]).all()
            )


if __name__ == "__main__":
    unittest.main()
