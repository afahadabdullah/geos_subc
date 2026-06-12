#!/bin/bash
#SBATCH -J SA_noise_v8
#SBATCH -o ml_output_noise_compare_sa_v8/compare_noise_%j.log
#SBATCH -e ml_output_noise_compare_sa_v8/compare_noise_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 South Asia v8 noise comparison started at $(date) on $(hostname)"

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

mkdir -p ml_output_noise_compare_sa_v8

CONFIG_PATH="${SA_NOISE_CONFIG:-ml_model/config_flow_multiv8.yaml}"
CONFIG_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")
CONFIG_T2M_MODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('t2m_target_mode'))")
CONFIG_TEST_ENSEMBLE=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH')).get('test_num_ensemble', yaml.safe_load(open('$CONFIG_PATH')).get('validation_num_ensemble', 15))))")
CONFIG_TEST_STEPS=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH')).get('test_num_steps', yaml.safe_load(open('$CONFIG_PATH')).get('validation_num_steps', 10))))")
YEAR="${SA_NOISE_YEAR:-2021}"
CHECKPOINT="${SA_NOISE_CHECKPOINT:-best_flow_ckpt.pt}"
NUM_ENSEMBLE="${SA_NOISE_ENSEMBLE:-$CONFIG_TEST_ENSEMBLE}"
NUM_STEPS="${SA_NOISE_STEPS:-$CONFIG_TEST_STEPS}"
BATCH_LIMIT="${SA_NOISE_BATCH_LIMIT:-12}"
ODE_BATCH_SIZE="${SA_NOISE_ODE_BATCH:-120}"
FULL_YEAR="${SA_NOISE_FULL_YEAR:-0}"
SETTING_ARGS=()
if [ -n "${SA_NOISE_SETTINGS:-}" ]; then
    IFS=';' read -ra SETTINGS <<< "$SA_NOISE_SETTINGS"
    for setting in "${SETTINGS[@]}"; do
        if [ -n "$setting" ]; then
            SETTING_ARGS+=(--setting "$setting")
        fi
    done
fi
SAMPLE_ARGS=()
if [ "$FULL_YEAR" = "1" ] || [ "$FULL_YEAR" = "true" ] || [ "$FULL_YEAR" = "TRUE" ]; then
    SAMPLE_ARGS+=(--full-year)
fi

if [ "$CONFIG_TARGET_DOMAIN" != "south_asia" ]; then
    echo "❌ Refusing to run non-SA config: target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi
if [ "$CONFIG_OUTPUT_DIR" != "ml_output_flowmulti_v8_sa_55e100e_0n40n_t2mres" ]; then
    echo "❌ Refusing to run old output dir: output_dir=$CONFIG_OUTPUT_DIR"
    exit 1
fi
if [ "$CONFIG_LOCAL_VARS" != "sm,mjo" ]; then
    echo "❌ Refusing to run unexpected local predictors: $CONFIG_LOCAL_VARS"
    exit 1
fi
if [ "$CONFIG_GLOBAL_CONTEXT" != "sst,sss,ivt,z500_zonal_dev,u250" ]; then
    echo "❌ Refusing to run unexpected global context variables: $CONFIG_GLOBAL_CONTEXT"
    exit 1
fi
if [ "$CONFIG_T2M_MODE" != "geos_residual" ]; then
    echo "❌ Refusing to run v8 compare without residual T2M mode: t2m_target_mode=$CONFIG_T2M_MODE"
    exit 1
fi

case "$CHECKPOINT" in
    /*) CKPT_PATH="$CHECKPOINT" ;;
    *) CKPT_PATH="$CONFIG_OUTPUT_DIR/$CHECKPOINT" ;;
esac
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Output dir: $CONFIG_OUTPUT_DIR"
echo "🎯 Local predictors: $CONFIG_LOCAL_VARS"
echo "🎯 Global context: $CONFIG_GLOBAL_CONTEXT"
echo "🎯 T2M target mode: $CONFIG_T2M_MODE"
echo "🎯 Year: $YEAR"
echo "🎯 Checkpoint: $CKPT_PATH"
echo "🎯 Ensemble members: $NUM_ENSEMBLE"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 Batch limit: $BATCH_LIMIT"
echo "🎯 Sampling: $([ ${#SAMPLE_ARGS[@]} -gt 0 ] && echo full weekly year || echo monthly subset)"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"
echo "🎯 Extra settings: ${SA_NOISE_SETTINGS:-config default}"

python ml_model/compare_noise_multiv8_sa.py \
    --config "$CONFIG_PATH" \
    --year "$YEAR" \
    --checkpoint "$CHECKPOINT" \
    --num_ensemble "$NUM_ENSEMBLE" \
    --num_steps "$NUM_STEPS" \
    --ode_batch_size "$ODE_BATCH_SIZE" \
    --batch_limit "$BATCH_LIMIT" \
    "${SAMPLE_ARGS[@]}" \
    "${SETTING_ARGS[@]}"

echo "🏁 South Asia v8 noise comparison finished at $(date)"
