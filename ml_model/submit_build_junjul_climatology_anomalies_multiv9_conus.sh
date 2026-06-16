#!/bin/bash
#SBATCH -J CONUS_clim
#SBATCH -o ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/clim_anom_%j.log
#SBATCH -e ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/clim_anom_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "📊 CONUS v9 June/July climatology/anomaly build started at $(date) on $(hostname)"

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

FORECAST_DIR="${CONUS_CLIM_FORECAST_DIR:-dataprocess/gen_multiv9_conus_125w66w_24n50n_junjul_e10clim_e100eval_s50}"
OUTPUT_DIR="${CONUS_CLIM_OUTPUT_DIR:-dataprocess/clim_anom_multiv9_conus_125w66w_24n50n_junjul}"
CLIM_START="${CONUS_CLIM_START:-2005}"
CLIM_END="${CONUS_CLIM_END:-2024}"
ANOM_START="${CONUS_ANOM_START:-2021}"
ANOM_END="${CONUS_ANOM_END:-2023}"
MONTHS="${CONUS_CLIM_MONTHS:-6,7}"
SKIP_YEARS="${CONUS_CLIM_SKIP_YEARS:-2017}"
OVERWRITE="${CONUS_CLIM_OVERWRITE:-0}"

if [ ! -d "$FORECAST_DIR" ]; then
    echo "❌ Forecast directory not found: $FORECAST_DIR"
    exit 1
fi

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "TRUE" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Forecast dir: $FORECAST_DIR"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Climatology years: $CLIM_START-$CLIM_END"
echo "🎯 Anomaly years: $ANOM_START-$ANOM_END"
echo "🎯 Months: $MONTHS"
echo "🎯 Skip years: $SKIP_YEARS"
echo "🎯 Overwrite: $OVERWRITE"

python ml_model/build_junjul_climatology_anomalies_multiv9_conus.py \
    --forecast_dir "$FORECAST_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --clim_start_year "$CLIM_START" \
    --clim_end_year "$CLIM_END" \
    --anom_start_year "$ANOM_START" \
    --anom_end_year "$ANOM_END" \
    --months "$MONTHS" \
    --skip_years "$SKIP_YEARS" \
    "${EXTRA_ARGS[@]}"

echo "🏁 CONUS v9 June/July climatology/anomaly build finished at $(date)"
