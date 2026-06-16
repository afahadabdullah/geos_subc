#!/bin/bash
#SBATCH -J CONUS_amtx
#SBATCH -o ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/anomaly_matrix_%j.log
#SBATCH -e ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/anomaly_matrix_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "📊 CONUS v9 June/July anomaly matrix evaluation started at $(date) on $(hostname)"

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

ANOMALY_PATH="${CONUS_AMTX_ANOMALY_PATH:-dataprocess/clim_anom_multiv9_conus_125w66w_24n50n_junjul/v9_junjul_anomalies_2021_2023.zarr}"
CLIMATOLOGY_PATH="${CONUS_AMTX_CLIMATOLOGY_PATH:-dataprocess/clim_anom_multiv9_conus_125w66w_24n50n_junjul/v9_junjul_climatology_2005_2024.zarr}"
FORECAST_DIR="${CONUS_AMTX_FORECAST_DIR:-dataprocess/gen_multiv9_conus_125w66w_24n50n_junjul_e10clim_e100eval_s50}"
OUTPUT_DIR="${CONUS_AMTX_OUTPUT_DIR:-ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/anomaly_matrix_junjul}"
MONTHS="${CONUS_AMTX_MONTHS:-6,7}"
BASELINE_START="${CONUS_AMTX_BASELINE_START:-2005}"
BASELINE_END="${CONUS_AMTX_BASELINE_END:-2024}"
SKIP_YEARS="${CONUS_AMTX_SKIP_YEARS:-2017}"
QUANTILES="${CONUS_AMTX_QUANTILES:-0.90,0.95}"
DECISIONS="${CONUS_AMTX_DECISIONS:-0.1,0.25,0.5}"

if [ ! -d "$ANOMALY_PATH" ]; then
    echo "❌ Anomaly Zarr not found: $ANOMALY_PATH"
    exit 1
fi
if [ ! -d "$CLIMATOLOGY_PATH" ]; then
    echo "❌ Climatology Zarr not found: $CLIMATOLOGY_PATH"
    exit 1
fi
if [ ! -d "$FORECAST_DIR" ]; then
    echo "❌ Forecast directory not found: $FORECAST_DIR"
    exit 1
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Anomaly path: $ANOMALY_PATH"
echo "🎯 Climatology path: $CLIMATOLOGY_PATH"
echo "🎯 Forecast dir: $FORECAST_DIR"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Months: $MONTHS"
echo "🎯 Baseline years: $BASELINE_START-$BASELINE_END"
echo "🎯 Skip years: $SKIP_YEARS"
echo "🎯 Extreme quantiles: $QUANTILES"
echo "🎯 Decision thresholds: $DECISIONS"

python ml_model/evaluate_junjul_anomaly_matrix_multiv9_conus.py \
    --anomaly_path "$ANOMALY_PATH" \
    --climatology_path "$CLIMATOLOGY_PATH" \
    --forecast_dir "$FORECAST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --months "$MONTHS" \
    --baseline_start_year "$BASELINE_START" \
    --baseline_end_year "$BASELINE_END" \
    --skip_years "$SKIP_YEARS" \
    --extreme_quantiles "$QUANTILES" \
    --decision_thresholds "$DECISIONS"

echo "🏁 CONUS v9 June/July anomaly matrix evaluation finished at $(date)"
