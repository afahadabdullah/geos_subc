#!/bin/bash
#SBATCH -J SA_test_v7
#SBATCH -o ml_output_flowmulti_v7_south_asia_global_context_t2mres/test_%j.log
#SBATCH -e ml_output_flowmulti_v7_south_asia_global_context_t2mres/test_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🧪 South Asia Multi-v7 full test started at $(date) on $(hostname)"

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

CONFIG_PATH="${SA_TEST_CONFIG:-ml_model/config_flow_multiv7.yaml}"
TEST_YEAR="${SA_TEST_YEAR:-2022}"
CHECKPOINT="${SA_TEST_CHECKPOINT:-best_flow_ckpt.pt}"
NUM_ENSEMBLE="${SA_TEST_ENSEMBLE:-30}"
NUM_STEPS="${SA_TEST_STEPS:-10}"
MAX_ENSEMBLE_PER_CHUNK="${SA_TEST_MAX_ENSEMBLE_PER_CHUNK:-30}"
VALIDATION_ODE_BATCH="${SA_TEST_ODE_BATCH:-120}"
SAMPLE_PLOT_LIMIT="${SA_TEST_SAMPLE_PLOT_LIMIT:-0}"
EXPECTED_OUTPUT_DIR="ml_output_flowmulti_v7_south_asia_global_context_t2mres"

OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
MIXED_PRECISION=$(python -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")
CONFIG_T2M_MODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('t2m_target_mode'))")

if [ "$OUTPUT_DIR" != "$EXPECTED_OUTPUT_DIR" ] || [ "$CONFIG_TARGET_DOMAIN" != "south_asia" ]; then
    echo "❌ Refusing to test unexpected config: output_dir=$OUTPUT_DIR target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi
if [ "$CONFIG_LOCAL_VARS" != "sm,mjo" ]; then
    echo "❌ Refusing to test unexpected local predictors: $CONFIG_LOCAL_VARS"
    exit 1
fi
if [ "$CONFIG_GLOBAL_CONTEXT" != "sst,sss,ivt,z500_zonal_dev,u250" ]; then
    echo "❌ Refusing to test unexpected global context variables: $CONFIG_GLOBAL_CONTEXT"
    exit 1
fi
if [ "$CONFIG_T2M_MODE" != "geos_residual" ]; then
    echo "❌ Refusing to test v7 without residual T2M mode: t2m_target_mode=$CONFIG_T2M_MODE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

case "$CHECKPOINT" in
    /*)
        CKPT_PATH="$CHECKPOINT"
        CKPT_ARG="$CHECKPOINT"
        ;;
    */*)
        if [ -f "$CHECKPOINT" ]; then
            CKPT_PATH="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
            CKPT_ARG="$CKPT_PATH"
        else
            CKPT_PATH="$OUTPUT_DIR/$CHECKPOINT"
            CKPT_ARG="$CHECKPOINT"
        fi
        ;;
    *)
        CKPT_PATH="$OUTPUT_DIR/$CHECKPOINT"
        CKPT_ARG="$CHECKPOINT"
        ;;
esac
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    exit 1
fi

TEST_CONFIG="$OUTPUT_DIR/config_test_${TEST_YEAR}_${SLURM_JOB_ID:-manual}.yaml"
python - "$CONFIG_PATH" "$TEST_CONFIG" "$EXPECTED_OUTPUT_DIR" "$TEST_YEAR" "$NUM_ENSEMBLE" "$NUM_STEPS" "$MAX_ENSEMBLE_PER_CHUNK" "$VALIDATION_ODE_BATCH" "$SAMPLE_PLOT_LIMIT" <<'PY'
import sys
import yaml

src, dst, output_dir, year, ens, steps, ens_chunk, ode_batch, plot_limit = sys.argv[1:]
year = int(year)
with open(src, "r") as f:
    cfg = yaml.safe_load(f)

cfg["output_dir"] = output_dir
cfg["val_start_year"] = year
cfg["val_end_year"] = year
cfg["crps_val_start_year"] = year
cfg["crps_val_end_year"] = year
cfg["test_num_ensemble"] = int(ens)
cfg["test_num_steps"] = int(steps)
cfg["test_max_ensemble_per_chunk"] = int(ens_chunk)
cfg["validation_ode_batch_size"] = int(ode_batch)
cfg["test_sample_plot_limit"] = int(plot_limit)
cfg["force_variance_phase"] = True

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

TEST_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$TEST_CONFIG'))['output_dir'])")
if [ "$TEST_OUTPUT_DIR" != "$EXPECTED_OUTPUT_DIR" ]; then
    echo "❌ Refusing generated test config with unexpected output_dir=$TEST_OUTPUT_DIR"
    exit 1
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Base config: $CONFIG_PATH"
echo "🎯 Test config: $TEST_CONFIG"
echo "🎯 Output dir: $TEST_OUTPUT_DIR"
echo "🎯 T2M target mode: $CONFIG_T2M_MODE"
echo "🎯 Year: $TEST_YEAR"
echo "🎯 Checkpoint: $CKPT_PATH"
echo "🎯 Checkpoint argument: $CKPT_ARG"
echo "🎯 Ensemble members: $NUM_ENSEMBLE"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 Max ensemble/chunk: $MAX_ENSEMBLE_PER_CHUNK"
echo "🎯 ODE batch: $VALIDATION_ODE_BATCH"
echo "🎯 Sample plot limit: $SAMPLE_PLOT_LIMIT (0 means all init dates)"
echo "🎯 Noise mode: forced EOF-LHS + variance using config rho/beta/coarse settings"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"

accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" \
    ml_model/train_flow_multiv7.py \
    --config "$TEST_CONFIG" \
    --test \
    --ckpt "$CKPT_ARG"

echo "🏁 South Asia Multi-v7 full test finished at $(date)"
