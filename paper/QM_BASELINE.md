# FIMr1p1 Quantile-Mapping Baseline

`paper/scripts/review_response/qm_fim_baseline.py` implements the manuscript
protocol:

- parameter fitting: 1999–2019 only;
- validation: 2020 only, diagnostic and without refitting;
- evaluation: 2021–2023 only.

The method is empirical quantile mapping by target variable, lead week,
verifying month, and grid point. FIMr1p1 members are pooled to estimate the
forecast CDF and corrected member by member. Precipitation uses square-root
space, an explicit dry-probability mass, and a 0.1 mm/day wet threshold. T2M is
mapped in kelvin. Values outside the fitted range receive the endpoint
additive correction instead of being capped at a training-period extreme.

## Memory-safe execution

The fitting stage reads the global grid in spatial tiles. It never concatenates
global 1999–2019 samples in memory. The default 30×60 tile is conservative; for
a tightly limited login node, use 15×30 and a mapping block of 2048 points.
Application and scoring process one initialization and lead at a time. At most
two parameter stores are cached.

Each variable/lead/month parameter store is written independently. A stopped
fit can be restarted with the same command; completed stores are reused.
Do not add `--overwrite` when resuming.

Activate the project environment, then set paths appropriate for the cluster:

```bash
QM_DATA_ROOT=/scratch/11353/afahad/geossub/dataprocess
QM_FORECAST_DIR=/scratch/11353/afahad/geossub/geos_subc/dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50
QM_OUT_DIR=/scratch/11353/afahad/geossub/ml_output_flow_finalv1_global_noisectx_t2mres/qm_fim_1999_2019
```

Fit the frozen mapping:

```bash
python3 paper/scripts/review_response/qm_fim_baseline.py \
  --stage fit \
  --data-root "$QM_DATA_ROOT" \
  --forecast-dir "$QM_FORECAST_DIR" \
  --out-dir "$QM_OUT_DIR" \
  --train-years 1999-2019 \
  --validation-years 2020 \
  --evaluation-years 2021-2023 \
  --fit-lat-tile 15 \
  --fit-lon-tile 30 \
  --spatial-block-size 2048
```

Apply and score the unchanged mapping on 2020 validation only:

```bash
python3 paper/scripts/review_response/qm_fim_baseline.py \
  --stage validate \
  --data-root "$QM_DATA_ROOT" \
  --forecast-dir "$QM_FORECAST_DIR" \
  --out-dir "$QM_OUT_DIR" \
  --train-years 1999-2019 \
  --validation-years 2020 \
  --evaluation-years 2021-2023 \
  --fit-lat-tile 15 \
  --fit-lon-tile 30 \
  --spatial-block-size 2048
```

This writes `qm_validation_aggregate_metrics.csv`. Review that file before
opening the evaluation archive. The validation stage cannot read any
2021–2023 file.

Once the method specification is frozen, apply and score 2021–2023:

```bash
python3 paper/scripts/review_response/qm_fim_baseline.py \
  --stage evaluate \
  --data-root "$QM_DATA_ROOT" \
  --forecast-dir "$QM_FORECAST_DIR" \
  --out-dir "$QM_OUT_DIR" \
  --train-years 1999-2019 \
  --validation-years 2020 \
  --evaluation-years 2021-2023
```

This writes `qm_evaluation_aggregate_metrics.csv`. Neither stage refits the
1999–2019 mapping. The legacy `--stage apply` and `--stage score` options
operate on both splits and are mainly useful for reproducing an already frozen
run.

## Comparison with the flow ensemble

The corrected stores are under `$QM_OUT_DIR/corrected`. Add them to the
existing verification suite as follows:

```bash
python3 paper/scripts/review_response/r1_fair_verification.py \
  --forecast_dir "$QM_FORECAST_DIR" \
  --qm_dir "$QM_OUT_DIR/corrected" \
  --years 2021,2022,2023 \
  --components clim,acc,boot,qm \
  --model_members 4 \
  --out_dir "$QM_OUT_DIR/flow_comparison"
```

The important output columns are:

- `skill_qm_vs_raw`: four-member QM-FIM CRPS skill versus raw FIM;
- `skill_model_vs_qm`: CRPS skill of repeatedly sampled four-member flow
  ensembles versus the four QM-FIM members;
- `rmse_skill_model_vs_qm`: the corresponding ensemble-mean RMSE skill;
- `crpss_clim_qm`: QM-FIM CRPS skill versus climatology;
- `ext_skill_model_vs_qm`: four-versus-four CRPS skill on the observed-extreme
  subset;
- `acc_qm` and `spread_rmse_qm`: anomaly and dispersion diagnostics.

For a quick end-to-end smoke test, use a separate output directory together
with `--max-inits 2 --n-quantiles 11`. During fitting, `--max-inits` means at
most two initialization dates per verifying month and year, so all 12 monthly
maps are still exercised. Never reuse smoke-test parameters for manuscript
scores.

## Outputs and storage

With 51 quantile knots, the fitted parameter directory is expected to require
roughly 2.5–3 GB. Corrected forecasts contain only the four-member `qm_pr` and
`qm_t2m` fields; the frozen 90-member flow archive is not copied or modified.
