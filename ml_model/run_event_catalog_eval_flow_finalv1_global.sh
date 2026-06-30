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
CALIBRATION_PARAMS="${EVENT_CALIBRATION_PARAMS:-ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_land_obsclim_chunked/bss_calibration_params.csv}"
LAND_MASK_FILE="${EVENT_LAND_MASK_FILE:-ml_model/land_ocean_mask_v6.pt}"
OUT_DIR="${EVENT_OUT_DIR:-ml_output_flow_finalv1_global_noisectx_t2mres/event_catalog_eval_global_2021_2023}"
EVENT_CATALOG="${EVENT_CATALOG:-default}"
REGIONS="${EVENT_REGIONS:-all}"
VARIABLES="${EVENT_VARIABLES:-pr,t2m}"
LEADS="${EVENT_LEADS:-3,4}"
START_YEAR="${EVENT_START_YEAR:-2021}"
END_YEAR="${EVENT_END_YEAR:-2023}"
WINDOW_DAYS="${EVENT_TIMESERIES_WINDOW_DAYS:-42}"
TOLERANCE_DAYS="${EVENT_TOLERANCE_DAYS:-10}"
MAP_FEATURES="${EVENT_MAP_FEATURES:-auto}"
COUNTY_BOUNDARIES="${EVENT_COUNTY_BOUNDARIES:-off}"
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

echo "🧭 Event catalog evaluation"
echo "   Forecast dir: $FORECAST_DIR"
echo "   Threshold file: $THRESHOLD_FILE"
echo "   Calibration params: $CALIBRATION_PARAMS"
echo "   Output dir: $OUT_DIR"
echo "   Regions: $REGIONS"
echo "   Variables: $VARIABLES"
echo "   Leads: $LEADS"

python ml_model/evaluate_event_catalog_flow_finalv1_global.py \
  --forecast_dir "$FORECAST_DIR" \
  --threshold_file "$THRESHOLD_FILE" \
  --calibration_params "$CALIBRATION_PARAMS" \
  --land_mask_file "$LAND_MASK_FILE" \
  --out_dir "$OUT_DIR" \
  --event_catalog "$EVENT_CATALOG" \
  --regions "$REGIONS" \
  --variables "$VARIABLES" \
  --leads "$LEADS" \
  --start_year "$START_YEAR" \
  --end_year "$END_YEAR" \
  --timeseries_window_days "$WINDOW_DAYS" \
  --event_tolerance_days "$TOLERANCE_DAYS" \
  --map_features "$MAP_FEATURES" \
  --county_boundaries "$COUNTY_BOUNDARIES" \
  --extreme_quantile_pr "$PR_QUANTILE" \
  --extreme_quantile_t2m "$T2M_QUANTILE" \
  --pr_min_threshold "$PR_MIN_THRESHOLD" \
  "${plot_args[@]}"
