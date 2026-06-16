#!/bin/bash
#SBATCH -J CONUS_ckpt
#SBATCH -o ml_output_checkpoint_compare_conus_v9/checkpoint_compare_%j.log
#SBATCH -e ml_output_checkpoint_compare_conus_v9/checkpoint_compare_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 CONUS v9 checkpoint comparison started at $(date) on $(hostname)"

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

mkdir -p ml_output_checkpoint_compare_conus_v9

CONFIG_PATH="${SA_CKPT_CONFIG:-ml_model/config_flow_multiv9.yaml}"
CONFIG_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")
CONFIG_T2M_MODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('t2m_target_mode'))")

YEAR="${SA_CKPT_YEAR:-2021}"
EPOCHS="${SA_CKPT_EPOCHS:-70,80,90,100,110,120,130,140}"
BEST_CHECKPOINT="${SA_CKPT_BEST_CHECKPOINT:-best_flow_ckpt.pt}"
NUM_ENSEMBLE="${SA_CKPT_ENSEMBLE:-30}"
NUM_STEPS="${SA_CKPT_STEPS:-10}"
ODE_BATCH_SIZE="${SA_CKPT_ODE_BATCH:-120}"
BATCH_LIMIT="${SA_CKPT_BATCH_LIMIT:-12}"
FULL_YEAR="${SA_CKPT_FULL_YEAR:-0}"
STRICT="${SA_CKPT_STRICT:-1}"
SEED="${SA_CKPT_SEED:-1234}"

if [ "$CONFIG_TARGET_DOMAIN" != "conus" ]; then
    echo "❌ Refusing to run non-CONUS config: target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi
if [ "$CONFIG_OUTPUT_DIR" != "ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres" ]; then
    echo "❌ Refusing unexpected output dir: output_dir=$CONFIG_OUTPUT_DIR"
    exit 1
fi
if [ "$CONFIG_LOCAL_VARS" != "sm,mjo" ]; then
    echo "❌ Refusing unexpected local predictors: $CONFIG_LOCAL_VARS"
    exit 1
fi
if [ "$CONFIG_GLOBAL_CONTEXT" != "sst,sss,ivt,z500_zonal_dev,u250" ]; then
    echo "❌ Refusing unexpected global context variables: $CONFIG_GLOBAL_CONTEXT"
    exit 1
fi
if [ "$CONFIG_T2M_MODE" != "geos_residual" ]; then
    echo "❌ Refusing non-residual T2M mode: t2m_target_mode=$CONFIG_T2M_MODE"
    exit 1
fi

SAMPLE_ARGS=()
if [ "$FULL_YEAR" = "1" ] || [ "$FULL_YEAR" = "true" ] || [ "$FULL_YEAR" = "TRUE" ]; then
    SAMPLE_ARGS+=(--full-year)
fi
STRICT_ARGS=()
if [ "$STRICT" = "1" ] || [ "$STRICT" = "true" ] || [ "$STRICT" = "TRUE" ]; then
    STRICT_ARGS+=(--strict)
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Output dir: $CONFIG_OUTPUT_DIR"
echo "🎯 Year: $YEAR"
echo "🎯 Best checkpoint: $BEST_CHECKPOINT"
echo "🎯 Epochs: $EPOCHS"
echo "🎯 Pure-noise ensemble members: $NUM_ENSEMBLE"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 ODE batch: $ODE_BATCH_SIZE"
echo "🎯 Batch limit: $BATCH_LIMIT"
echo "🎯 Strict checkpoint check: $STRICT"
echo "🎯 Sampling: $([ ${#SAMPLE_ARGS[@]} -gt 0 ] && echo full weekly year || echo monthly subset)"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"

python ml_model/compare_checkpoints_multiv9_sa.py \
    --config "$CONFIG_PATH" \
    --year "$YEAR" \
    --epochs "$EPOCHS" \
    --best_checkpoint "$BEST_CHECKPOINT" \
    --num_ensemble "$NUM_ENSEMBLE" \
    --num_steps "$NUM_STEPS" \
    --ode_batch_size "$ODE_BATCH_SIZE" \
    --batch_limit "$BATCH_LIMIT" \
    --seed "$SEED" \
    "${SAMPLE_ARGS[@]}" \
    "${STRICT_ARGS[@]}"

echo "🏁 CONUS v9 checkpoint comparison finished at $(date)"
