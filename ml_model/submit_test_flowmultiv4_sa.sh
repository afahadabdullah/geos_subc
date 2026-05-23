#!/bin/bash
#SBATCH -J SA_test_v4
#SBATCH -o ml_output_flowmulti_v4_south_asia_global_context/test_%j.log
#SBATCH -e ml_output_flowmulti_v4_south_asia_global_context/test_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🧪 South Asia Multi-v4 full test started at $(date) on $(hostname)"

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

CONFIG_PATH="${SA_TEST_CONFIG:-ml_model/config_flow_multiv4.yaml}"
TEST_YEAR="${SA_TEST_YEAR:-2022}"
CHECKPOINT="${SA_TEST_CHECKPOINT:-best_flow_ckpt.pt}"
NUM_ENSEMBLE="${SA_TEST_ENSEMBLE:-30}"
NUM_STEPS="${SA_TEST_STEPS:-10}"
MAX_ENSEMBLE_PER_CHUNK="${SA_TEST_MAX_ENSEMBLE_PER_CHUNK:-30}"
VALIDATION_ODE_BATCH="${SA_TEST_ODE_BATCH:-120}"

OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
MIXED_PRECISION=$(python -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")

if [ "$OUTPUT_DIR" != "ml_output_flowmulti_v4_south_asia_global_context" ] || [ "$CONFIG_TARGET_DOMAIN" != "south_asia" ]; then
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

mkdir -p "$OUTPUT_DIR"

case "$CHECKPOINT" in
    /*) CKPT_PATH="$CHECKPOINT" ;;
    *) CKPT_PATH="$OUTPUT_DIR/$CHECKPOINT" ;;
esac
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    exit 1
fi

TEST_CONFIG="$OUTPUT_DIR/config_test_${TEST_YEAR}_${SLURM_JOB_ID:-manual}.yaml"
python - "$CONFIG_PATH" "$TEST_CONFIG" "$TEST_YEAR" "$NUM_ENSEMBLE" "$NUM_STEPS" "$MAX_ENSEMBLE_PER_CHUNK" "$VALIDATION_ODE_BATCH" <<'PY'
import sys
import yaml

src, dst, year, ens, steps, ens_chunk, ode_batch = sys.argv[1:]
year = int(year)
with open(src, "r") as f:
    cfg = yaml.safe_load(f)

cfg["val_start_year"] = year
cfg["val_end_year"] = year
cfg["crps_val_start_year"] = year
cfg["crps_val_end_year"] = year
cfg["test_num_ensemble"] = int(ens)
cfg["test_num_steps"] = int(steps)
cfg["test_max_ensemble_per_chunk"] = int(ens_chunk)
cfg["validation_ode_batch_size"] = int(ode_batch)

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Base config: $CONFIG_PATH"
echo "🎯 Test config: $TEST_CONFIG"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Year: $TEST_YEAR"
echo "🎯 Checkpoint: $CKPT_PATH"
echo "🎯 Ensemble members: $NUM_ENSEMBLE"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 Max ensemble/chunk: $MAX_ENSEMBLE_PER_CHUNK"
echo "🎯 ODE batch: $VALIDATION_ODE_BATCH"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"

accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" \
    ml_model/train_flow_multiv4.py \
    --config "$TEST_CONFIG" \
    --test \
    --ckpt "$CHECKPOINT"

echo "🏁 South Asia Multi-v4 full test finished at $(date)"
