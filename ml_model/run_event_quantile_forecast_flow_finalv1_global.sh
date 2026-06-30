#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

CONDA_DIR="${CONDA_DIR:-$ROOT_DIR/miniconda}"
ENV_NAME="${ENV_NAME:-geossub_env}"
if [[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

FORECAST_DIR="${EVENT_FORECAST_DIR:-dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50}"
THRESHOLD_FILE="${EVENT_THRESHOLD_FILE:-ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_land_obsclim_chunked/event_thresholds_and_frequencies.nc}"
LAND_MASK_FILE="${EVENT_LAND_MASK_FILE:-ml_model/land_ocean_mask_v6.pt}"
OUT_DIR="${EVENT_QUANTILE_OUT_DIR:-ml_output_flow_finalv1_global_noisectx_t2mres/event_quantile_eval_global_2021_2023}"
EVENT_CATALOG="${EVENT_CATALOG:-default}"
REGIONS="${EVENT_REGIONS:-all}"
VARIABLES="${EVENT_VARIABLES:-pr,t2m}"
LEADS="${EVENT_LEADS:-3,4}"
PROGRESSION_LEADS="${EVENT_PROGRESSION_LEADS:-1,2,3,4}"
REGIONAL_WEIGHTING="${EVENT_REGIONAL_WEIGHTING:-uniform}"
TAIL_FRACTION="${EVENT_TAIL_FRACTION:-0.10}"
SPATIAL_QUANTILE="${EVENT_SPATIAL_QUANTILE:-0.95}"
ENSEMBLE_QUANTILES="${EVENT_ENSEMBLE_QUANTILES:-0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99}"
LOSS_QUANTILES="${EVENT_LOSS_QUANTILES:-0.90,0.95,0.99}"
EVENT_AREA_FRACTION_THRESHOLD="${EVENT_AREA_FRACTION_THRESHOLD:-0.10}"
START_YEAR="${EVENT_START_YEAR:-2021}"
END_YEAR="${EVENT_END_YEAR:-2023}"
WINDOW_DAYS="${EVENT_TIMESERIES_WINDOW_DAYS:-42}"
TOLERANCE_DAYS="${EVENT_TOLERANCE_DAYS:-10}"
PR_QUANTILE="${EVENT_PR_QUANTILE:-0.95}"
T2M_QUANTILE="${EVENT_T2M_QUANTILE:-0.95}"
PR_MIN_THRESHOLD="${EVENT_PR_MIN_THRESHOLD:-5.0}"

plot_args=()
if [[ "${EVENT_MAKE_PLOTS:-1}" != "0" ]]; then
  plot_args+=(--make_plots)
fi
if [[ "${EVENT_OVERWRITE:-1}" != "0" ]]; then
  plot_args+=(--overwrite)
fi
if [[ "${EVENT_WRITE_MEMBER_VALUES:-0}" != "0" ]]; then
  plot_args+=(--write_member_values)
fi

echo "🧭 Event quantile/probabilistic forecast evaluation"
echo "   Forecast dir: $FORECAST_DIR"
echo "   Threshold file: $THRESHOLD_FILE"
echo "   Output dir: $OUT_DIR"
echo "   Regions: $REGIONS"
echo "   Variables: $VARIABLES"
echo "   Leads: $LEADS"
echo "   Fixed-init progression leads: $PROGRESSION_LEADS"
echo "   Regional weighting: $REGIONAL_WEIGHTING"
echo "   Tail fraction: $TAIL_FRACTION"
echo "   Spatial quantile: $SPATIAL_QUANTILE"
echo "   Ensemble quantiles: $ENSEMBLE_QUANTILES"
echo "   Loss quantiles: $LOSS_QUANTILES"
echo "   Event-area fraction threshold: $EVENT_AREA_FRACTION_THRESHOLD"

python ml_model/evaluate_event_quantile_forecast_flow_finalv1_global.py \
  --forecast_dir "$FORECAST_DIR" \
  --threshold_file "$THRESHOLD_FILE" \
  --land_mask_file "$LAND_MASK_FILE" \
  --out_dir "$OUT_DIR" \
  --event_catalog "$EVENT_CATALOG" \
  --regions "$REGIONS" \
  --variables "$VARIABLES" \
  --leads "$LEADS" \
  --progression_leads "$PROGRESSION_LEADS" \
  --regional_weighting "$REGIONAL_WEIGHTING" \
  --tail_fraction "$TAIL_FRACTION" \
  --spatial_quantile "$SPATIAL_QUANTILE" \
  --ensemble_quantiles "$ENSEMBLE_QUANTILES" \
  --loss_quantiles "$LOSS_QUANTILES" \
  --event_area_fraction_threshold "$EVENT_AREA_FRACTION_THRESHOLD" \
  --start_year "$START_YEAR" \
  --end_year "$END_YEAR" \
  --timeseries_window_days "$WINDOW_DAYS" \
  --event_tolerance_days "$TOLERANCE_DAYS" \
  --extreme_quantile_pr "$PR_QUANTILE" \
  --extreme_quantile_t2m "$T2M_QUANTILE" \
  --pr_min_threshold "$PR_MIN_THRESHOLD" \
  "${plot_args[@]}"
