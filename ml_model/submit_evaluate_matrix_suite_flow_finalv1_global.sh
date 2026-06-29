#!/bin/bash
#SBATCH -J GLOBAL_matrix_eval
#SBATCH -o ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_%j.log
#SBATCH -e ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p ml_output_flow_finalv1_global_noisectx_t2mres

FORECAST_DIR="${MATRIX_EVAL_FORECAST_DIR:-dataprocess/gen_flow_finalv1_global_fullyear_2021_2024_e90_s50}"
START_YEAR="${MATRIX_EVAL_START_YEAR:-2021}"
END_YEAR="${MATRIX_EVAL_END_YEAR:-2023}"
SKIP_YEARS="${MATRIX_EVAL_SKIP_YEARS:-}"
OUT_DIR="${MATRIX_EVAL_OUT_DIR:-ml_output_flow_finalv1_global_noisectx_t2mres/matrix_eval_global_2021_2023_e90_s50}"
VARIABLES="${MATRIX_EVAL_VARIABLES:-pr,t2m}"
MAX_RUNTIME_MINUTES="${MATRIX_EVAL_MAX_RUNTIME_MINUTES:-110}"
MAKE_PLOTS="${MATRIX_EVAL_MAKE_PLOTS:-1}"
OVERWRITE="${MATRIX_EVAL_OVERWRITE:-0}"
MAP_FEATURES="${MATRIX_EVAL_MAP_FEATURES:-auto}"
COUNTY_BOUNDARIES="${MATRIX_EVAL_COUNTY_BOUNDARIES:-off}"
EXTREME_Q_PR="${MATRIX_EVAL_EXTREME_Q_PR:-0.95}"
EXTREME_Q_T2M="${MATRIX_EVAL_EXTREME_Q_T2M:-0.95}"
PR_MIN_THRESHOLD="${MATRIX_EVAL_PR_MIN_THRESHOLD:-5.0}"

EXTRA_ARGS=()
if [ "$MAKE_PLOTS" = "1" ] || [ "$MAKE_PLOTS" = "true" ] || [ "$MAKE_PLOTS" = "TRUE" ]; then
    EXTRA_ARGS+=(--make_plots)
fi
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "TRUE" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "🚀 Matrix evaluation started at $(date) on $(hostname)"
echo "📌 Code: $(git rev-parse --short HEAD) branch $(git branch --show-current)"
echo "🎯 Forecast dir: $FORECAST_DIR"
echo "🎯 Years: $START_YEAR-$END_YEAR skip=$SKIP_YEARS"
echo "🎯 Variables: $VARIABLES"
echo "🎯 Output: $OUT_DIR"
echo "🎯 Extreme q: PR=$EXTREME_Q_PR T2M=$EXTREME_Q_T2M PR min=$PR_MIN_THRESHOLD"
echo "🎯 Spatial maps: map_features=$MAP_FEATURES county_boundaries=$COUNTY_BOUNDARIES"

python ml_model/evaluate_matrix_suite_flow_finalv1_global.py \
    --forecast_dir "$FORECAST_DIR" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --skip_years "$SKIP_YEARS" \
    --out_dir "$OUT_DIR" \
    --variables "$VARIABLES" \
    --extreme_quantile_pr "$EXTREME_Q_PR" \
    --extreme_quantile_t2m "$EXTREME_Q_T2M" \
    --pr_min_threshold "$PR_MIN_THRESHOLD" \
    --map_features "$MAP_FEATURES" \
    --county_boundaries "$COUNTY_BOUNDARIES" \
    --max_runtime_minutes "$MAX_RUNTIME_MINUTES" \
    "${EXTRA_ARGS[@]}"

echo "🏁 Matrix evaluation finished at $(date)"
