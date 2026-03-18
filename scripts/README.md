# Paper Analysis Scripts

These scripts live at the repository root so the paper-analysis workflow is not tied to the ignored `paper/` directory.

## Included

- `paper_plot_common.py`: shared plotting style, file discovery, and small helpers.
- `make_paper_analysis_plots.py`: builds summary paper plots from existing evaluation outputs or demo data.
- `make_paper_spatial_composite.py`: assembles spatial map outputs into a paper-ready composite figure, with a demo fallback.

## Supported Native Inputs

The analysis script understands the output files already produced by the existing model/evaluation code:

- `noise_comparison_v4_multi_results_<year>.csv`
- `checkpoint_pure_noise_summary_<year>.csv`
- `training_log_v5.csv`
- `model_registry.json`
- `test_summary_multi.json`

## Quick Start

Generate demo figures:

```bash
python3 scripts/make_paper_analysis_plots.py --demo
python3 scripts/make_paper_spatial_composite.py --demo
```

Auto-discover existing outputs under the repo and write figures into `paper/figures`:

```bash
python3 scripts/make_paper_analysis_plots.py
python3 scripts/make_paper_spatial_composite.py
```

Point to explicit files:

```bash
python3 scripts/make_paper_analysis_plots.py \
  --noise-results /path/to/noise_comparison_v4_multi_results_2021.csv \
  --checkpoint-summary /path/to/checkpoint_pure_noise_summary_2021.csv \
  --training-log /path/to/training_log_v5.csv \
  --model-registry /path/to/model_registry.json \
  --test-summary /path/to/test_summary_multi.json
```
