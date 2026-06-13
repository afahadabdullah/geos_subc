#!/bin/bash
#SBATCH -J SA_v9_tail
#SBATCH -o ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/extreme_tail_%j.log
#SBATCH -e ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/extreme_tail_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "📊 South Asia v9 extreme-tail analysis started at $(date) on $(hostname)"

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

FORECAST_DIR="${SA_TAIL_FORECAST_DIR:-dataprocess/gen_multiv9_sa_55e100e_0n40n_junjul_e10clim_e100eval_s50}"
OUTPUT_DIR="${SA_TAIL_OUTPUT_DIR:-ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/extreme_tail_junjul}"
BASE_START="${SA_TAIL_BASE_START:-2005}"
BASE_END="${SA_TAIL_BASE_END:-2020}"
EVAL_START="${SA_TAIL_EVAL_START:-2021}"
EVAL_END="${SA_TAIL_EVAL_END:-2024}"
QUANTILES="${SA_TAIL_QUANTILES:-0.90,0.95,0.99}"
DECISIONS="${SA_TAIL_DECISIONS:-0.1,0.25,0.5}"

if [ ! -d "$FORECAST_DIR" ]; then
    echo "❌ Forecast Zarr directory not found: $FORECAST_DIR"
    exit 1
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Forecast dir: $FORECAST_DIR"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Baseline years: $BASE_START-$BASE_END"
echo "🎯 Eval years: $EVAL_START-$EVAL_END"
echo "🎯 Quantiles: $QUANTILES"
echo "🎯 Decision thresholds: $DECISIONS"

python ml_model/analyze_extreme_tails_multiv9_sa.py \
    --forecast_dir "$FORECAST_DIR" \
    --baseline_start_year "$BASE_START" \
    --baseline_end_year "$BASE_END" \
    --eval_start_year "$EVAL_START" \
    --eval_end_year "$EVAL_END" \
    --quantiles "$QUANTILES" \
    --decision_thresholds "$DECISIONS" \
    --output_dir "$OUTPUT_DIR"

echo "🏁 South Asia v9 extreme-tail analysis finished at $(date)"
