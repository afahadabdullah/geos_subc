#!/bin/bash
#SBATCH -J CONUS_raw
#SBATCH -o ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/raw_matrix_%j.log
#SBATCH -e ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/raw_matrix_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "📊 CONUS v9 June/July raw matrix evaluation started at $(date) on $(hostname)"

PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

echo "🔧 Using conda dir: $CONDA_DIR"
if [ ! -f "$CONDA_DIR/bin/activate" ]; then
    echo "❌ Missing $CONDA_DIR/bin/activate. Run: bash setup_env.sh"
    exit 1
fi
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "❌ Missing conda env at $CONDA_ENV_PATH. Run: bash setup_env.sh"
    exit 1
fi

source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
echo "✅ Conda environment active: ${CONDA_PREFIX:-unset}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export SYMPY_GROUND_TYPES=python

FORECAST_DIR="${CONUS_RAW_FORECAST_DIR:-dataprocess/gen_multiv9_conus_125w66w_24n50n_junjul_testmode_e100_s50}"
OUTPUT_DIR="${CONUS_RAW_OUTPUT_DIR:-ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/raw_matrix_junjul_testmode_2021_2024}"
START_YEAR="${CONUS_RAW_START_YEAR:-2021}"
END_YEAR="${CONUS_RAW_END_YEAR:-2024}"
MONTHS="${CONUS_RAW_MONTHS:-6,7}"
SKIP_YEARS="${CONUS_RAW_SKIP_YEARS:-}"
QUANTILES="${CONUS_RAW_QUANTILES:-0.90,0.95}"
DECISIONS="${CONUS_RAW_DECISIONS:-0.1,0.25,0.5}"

if [ ! -d "$FORECAST_DIR" ]; then
    echo "❌ Forecast directory not found: $FORECAST_DIR"
    exit 1
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Forecast dir: $FORECAST_DIR"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Years: $START_YEAR-$END_YEAR"
echo "🎯 Months: $MONTHS"
echo "🎯 Skip years: ${SKIP_YEARS:-none}"
echo "🎯 Extreme quantiles: $QUANTILES"
echo "🎯 Decision thresholds: $DECISIONS"

python ml_model/evaluate_junjul_raw_matrix_multiv9_conus.py \
    --forecast_dir "$FORECAST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --months "$MONTHS" \
    --skip_years "$SKIP_YEARS" \
    --extreme_quantiles "$QUANTILES" \
    --decision_thresholds "$DECISIONS"

echo "🏁 CONUS v9 June/July raw matrix evaluation finished at $(date)"
