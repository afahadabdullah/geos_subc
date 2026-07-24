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

Each fitted spatial tile is written directly to its variable/lead/month Zarr
store. A small atomic marker is created only after every array in that tile is
on disk, and a separate completion marker is created after the final tile.
A stopped fit can therefore be restarted with the exact same command:
completed lead/month stores and completed tiles within the interrupted store
are skipped. Stores completed by the older, non-tile-checkpointed version are
recognized and adopted automatically. Do not add `--overwrite` when resuming;
that option intentionally deletes the saved checkpoints for each store.

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
  --model_members 8 \
  --out_dir "$QM_OUT_DIR/flow_comparison"
```

The important output columns are:

- `skill_qm_vs_raw`: four-member QM-FIM CRPS skill versus raw FIM;
- `skill_model_k_vs_raw`: repeatedly sampled eight-member flow CRPS skill
  versus the native four-member raw FIM;
- `skill_model_k_vs_qm`: repeatedly sampled eight-member flow CRPS skill
  versus the native four-member QM-FIM;
- `rmse_skill_model_k_vs_raw` and `rmse_skill_model_k_vs_qm`: corresponding
  ensemble-mean RMSE skills;
- `skill_model_vs_qm`: CRPS skill of repeatedly sampled four-member flow
  ensembles versus the four QM-FIM members, retained as the strict
  equal-ensemble-size control;
- `rmse_skill_model_vs_qm`: the corresponding ensemble-mean RMSE skill;
- `crpss_clim_qm`: QM-FIM CRPS skill versus climatology;
- `ext_skill_model_vs_qm`: four-versus-four CRPS skill on the observed-extreme
  subset;
- `acc_qm` and `spread_rmse_qm`: anomaly and dispersion diagnostics.

## Figure 5 extreme-event comparison

After corrected archives exist for 2021--2023, compare raw FIM-4, QM-FIM-4,
FlowMatch-6, and the full FlowMatch-90 ensemble on the original Figure 5
extreme subset. FlowMatch-4 and FlowMatch-8 controls are retained in the CSV
outputs:

```bash
python3 paper/scripts/review_response/qm_extreme_event_comparison.py \
  --forecast-dir "$QM_FORECAST_DIR" \
  --qm-dir "$QM_OUT_DIR/corrected" \
  --out-dir "$QM_OUT_DIR/extreme_fig5_comparison"
```

The defaults reproduce the 2021--2023 Figure 5 selection: 30 precipitation
plus 30 temperature cases selected across weeks 3--4 and 15 regional boxes,
with no more than two cases per region and variable. Scores are calculated at
grid points and cosine-latitude averaged within each event's regional box.

The job checkpoints `selected_extreme_events.json` and one CSV per completed
event under `case_metrics/`. Rerun the identical command after interruption;
completed events are skipped. Do not pass `--overwrite` when resuming.

Primary outputs are:

- `extreme_comparison_summary.csv`: pooled regional skill and mean per-event
  skill for QM4/raw4, flow8/raw4, flow8/QM4, flow4/raw4, and flow4/QM4;
- `extreme_comparison_case_bootstrap_ci.csv`: 95% event-resampling intervals;
- `extreme_regional_summary.csv`: comparison skill split by region;
- `extreme_case_system_metrics.csv`: absolute CRPS, RMSE, and q95 scores for
  every system and event; and
- `qm_flow_weekly_values_and_improvement.png`: one figure containing
  event-averaged observed/forecast regional values and weekly CRPS, RMSE, and
  q95-score improvements versus raw FIM-4 for QM-4, FlowMatch-6, and
  FlowMatch-90.

Columns prefixed with `case_mean_` average the event-specific regional skills
and match the extreme-event manuscript definition. Unprefixed skill columns
pool weighted score sums across regional grid cells and cases, matching the
original Figure 5 evaluator's aggregation.

For a quick end-to-end smoke test, use a separate output directory together
with `--max-inits 2 --n-quantiles 11`. During fitting, `--max-inits` means at
most two initialization dates per verifying month and year, so all 12 monthly
maps are still exercised. Never reuse smoke-test parameters for manuscript
scores.

## Outputs and storage

With 51 quantile knots, the fitted parameter directory is expected to require
roughly 2.5–3 GB. Corrected forecasts contain only the four-member `qm_pr` and
`qm_t2m` fields; the frozen 90-member flow archive is not copied or modified.
