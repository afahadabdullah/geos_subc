#!/bin/bash
#SBATCH -J SA_noise_cmp
#SBATCH -o ml_output_noise_compare_sa/compare_noise_%j.log
#SBATCH -e ml_output_noise_compare_sa/compare_noise_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 South Asia noise comparison started at $(date) on $(hostname)"

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
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

mkdir -p ml_output_noise_compare_sa

CONFIG_PATH="${SA_NOISE_CONFIG:-ml_model/config_flow_multiv4.yaml}"
YEAR="${SA_NOISE_YEAR:-2021}"
CHECKPOINT="${SA_NOISE_CHECKPOINT:-best_flow_ckpt.pt}"
NUM_ENSEMBLE="${SA_NOISE_ENSEMBLE:-30}"
NUM_STEPS="${SA_NOISE_STEPS:-10}"
BATCH_LIMIT="${SA_NOISE_BATCH_LIMIT:-12}"
ODE_BATCH_SIZE="${SA_NOISE_ODE_BATCH:-120}"
SETTING_ARGS=()
if [ -n "${SA_NOISE_SETTINGS:-}" ]; then
    IFS=';' read -ra SETTINGS <<< "$SA_NOISE_SETTINGS"
    for setting in "${SETTINGS[@]}"; do
        if [ -n "$setting" ]; then
            SETTING_ARGS+=(--setting "$setting")
        fi
    done
fi

CONFIG_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
if [ "$CONFIG_TARGET_DOMAIN" != "south_asia" ]; then
    echo "❌ Refusing to run non-SA config: target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Output dir: $CONFIG_OUTPUT_DIR"
echo "🎯 Year: $YEAR"
echo "🎯 Checkpoint: $CHECKPOINT"
echo "🎯 Ensemble members: $NUM_ENSEMBLE"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 Batch limit: $BATCH_LIMIT"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"
echo "🎯 Extra settings: ${SA_NOISE_SETTINGS:-config default}"

python ml_model/compare_noise_multiv4_sa.py \
    --config "$CONFIG_PATH" \
    --year "$YEAR" \
    --checkpoint "$CHECKPOINT" \
    --num_ensemble "$NUM_ENSEMBLE" \
    --num_steps "$NUM_STEPS" \
    --ode_batch_size "$ODE_BATCH_SIZE" \
    --batch_limit "$BATCH_LIMIT" \
    "${SETTING_ARGS[@]}"

echo "🏁 South Asia noise comparison finished at $(date)"
