#!/bin/bash
#SBATCH -J SA_v9_raw
#SBATCH -o ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/raw_matrix_%j.log
#SBATCH -e ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/raw_matrix_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "📊 South Asia v9 June/July raw matrix evaluation started at $(date) on $(hostname)"

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

FORECAST_DIR="${SA_RAW_FORECAST_DIR:-dataprocess/gen_multiv9_sa_55e100e_0n40n_junjul_testmode_e100_s50}"
OUTPUT_DIR="${SA_RAW_OUTPUT_DIR:-ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/raw_matrix_junjul_testmode_2021_2024_obsclim2001_2020}"
START_YEAR="${SA_RAW_START_YEAR:-2021}"
END_YEAR="${SA_RAW_END_YEAR:-2024}"
MONTHS="${SA_RAW_MONTHS:-6,7}"
SKIP_YEARS="${SA_RAW_SKIP_YEARS:-}"
CONFIG_PATH="${SA_RAW_CONFIG:-ml_model/config_flow_multiv9.yaml}"
OBS_CLIM_DATA_DIR="${SA_RAW_OBS_CLIM_DATA_DIR:-${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}}"
OBS_CLIM_START_YEAR="${SA_RAW_OBS_CLIM_START_YEAR:-2001}"
OBS_CLIM_END_YEAR="${SA_RAW_OBS_CLIM_END_YEAR:-2020}"
OBS_CLIM_SKIP_YEARS="${SA_RAW_OBS_CLIM_SKIP_YEARS:-}"
THRESHOLD_SOURCE="${SA_RAW_THRESHOLD_SOURCE:-obs_climatology}"
QUANTILES="${SA_RAW_QUANTILES:-0.90,0.95}"
DECISIONS="${SA_RAW_DECISIONS:-0.1,0.25,0.5}"

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
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Threshold source: $THRESHOLD_SOURCE"
echo "🎯 Self-threshold diagnostic: enabled (separate eval-period OBS/ML/GEOS quantiles)"
echo "🎯 Obs climatology data dir: $OBS_CLIM_DATA_DIR"
echo "🎯 Obs climatology years: $OBS_CLIM_START_YEAR-$OBS_CLIM_END_YEAR"
echo "🎯 Obs climatology skip years: ${OBS_CLIM_SKIP_YEARS:-none}"
echo "🎯 Extreme quantiles: $QUANTILES"
echo "🎯 Decision thresholds: $DECISIONS"

python ml_model/evaluate_junjul_raw_matrix_multiv9_sa.py \
    --forecast_dir "$FORECAST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --months "$MONTHS" \
    --skip_years "$SKIP_YEARS" \
    --config "$CONFIG_PATH" \
    --obs_clim_data_dir "$OBS_CLIM_DATA_DIR" \
    --obs_clim_start_year "$OBS_CLIM_START_YEAR" \
    --obs_clim_end_year "$OBS_CLIM_END_YEAR" \
    --obs_clim_skip_years "$OBS_CLIM_SKIP_YEARS" \
    --threshold_source "$THRESHOLD_SOURCE" \
    --extreme_quantiles "$QUANTILES" \
    --decision_thresholds "$DECISIONS"

echo "🏁 South Asia v9 June/July raw matrix evaluation finished at $(date)"
