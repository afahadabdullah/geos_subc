#!/bin/bash
#SBATCH -J SA_v9_xai
#SBATCH -o ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/driver_attribution_%j.log
#SBATCH -e ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/driver_attribution_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🔎 South Asia v9 driver attribution started at $(date) on $(hostname)"

PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

if [ ! -f "$CONDA_DIR/bin/activate" ]; then
    echo "❌ Missing $CONDA_DIR/bin/activate. Run: bash setup_env.sh"
    exit 1
fi
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "❌ Missing conda env at $CONDA_ENV_PATH. Run: bash setup_env.sh"
    exit 1
fi

source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export SYMPY_GROUND_TYPES=python
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/.matplotlib_cache}"

CONFIG_PATH="${SA_XAI_CONFIG:-ml_model/config_flow_multiv9.yaml}"
OUTPUT_DIR="${SA_XAI_OUTPUT_DIR:-ml_output_flowmulti_v9_sa_55e100e_0n40n_noisectx_t2mres/driver_attribution_raw_2021_2024}"
MODEL_OUTPUT_DIR="${SA_XAI_MODEL_OUTPUT_DIR:-}"
START_YEAR="${SA_XAI_START_YEAR:-2021}"
END_YEAR="${SA_XAI_END_YEAR:-2024}"
if [ -n "${SA_XAI_YEAR:-}" ]; then
    START_YEAR="$SA_XAI_YEAR"
    END_YEAR="$SA_XAI_YEAR"
fi
CHECKPOINT="${SA_XAI_CHECKPOINT:-best_flow_ckpt.pt}"
NUM_ENSEMBLE="${SA_XAI_ENSEMBLE:-30}"
NUM_STEPS="${SA_XAI_STEPS:-10}"
ODE_BATCH="${SA_XAI_ODE_BATCH:-120}"
BATCH_LIMIT="${SA_XAI_BATCH_LIMIT:-0}"
FULL_YEAR="${SA_XAI_FULL_YEAR:-0}"
XAI_GROUP_LIST="${SA_XAI_GROUPS:-geos_all,geos_pr,geos_t2m,local_sm,local_mjo,global_sst,global_sss,global_ivt,global_z500_zonal_dev,global_u250,all_global_context,all_local_obs}"

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

EXTRA_ARGS=()
if [ -n "$MODEL_OUTPUT_DIR" ]; then
    EXTRA_ARGS+=(--model_output_dir "$MODEL_OUTPUT_DIR")
fi
if [ "$FULL_YEAR" = "1" ] || [ "$FULL_YEAR" = "true" ] || [ "$FULL_YEAR" = "TRUE" ]; then
    EXTRA_ARGS+=(--full-year)
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Model output dir: ${MODEL_OUTPUT_DIR:-config output_dir}"
echo "🎯 Years: $START_YEAR-$END_YEAR"
echo "🎯 Target mode: raw observed PR/T2M, no anomaly transform"
echo "🎯 Checkpoint: $CHECKPOINT"
echo "🎯 Ensembles: $NUM_ENSEMBLE"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 Batch limit: $BATCH_LIMIT"
echo "🎯 Groups: $XAI_GROUP_LIST"

python ml_model/explain_driver_attribution_multiv9_sa.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --checkpoint "$CHECKPOINT" \
    --num_ensemble "$NUM_ENSEMBLE" \
    --num_steps "$NUM_STEPS" \
    --ode_batch_size "$ODE_BATCH" \
    --batch_limit "$BATCH_LIMIT" \
    --groups "$XAI_GROUP_LIST" \
    "${EXTRA_ARGS[@]}"

echo "🏁 South Asia v9 driver attribution finished at $(date)"
