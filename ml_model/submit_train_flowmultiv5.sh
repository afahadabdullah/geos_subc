#!/bin/bash
#SBATCH -J SA_flow_v5
#SBATCH -o ml_output_flowmulti_v5_south_asia_global_context/flow_%j.log
#SBATCH -e ml_output_flowmulti_v5_south_asia_global_context/flow_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 South Asia Multi-v5 local/global training job started at $(date) on $(hostname)"

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

if [ "${CLEANUP_OLD_PROCS:-0}" = "1" ]; then
    echo "🧹 Cleaning up previous SA flow v5 processes..."
    pkill -9 -u "$USER" -f "train_flow_multiv5.py" || true
    pkill -9 -u "$USER" -f "accelerate.*train_flow_multiv5.py" || true
    sleep 3
else
    echo "🧹 Skipping process cleanup. Set CLEANUP_OLD_PROCS=1 to enable it."
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"

CONFIG_PATH="ml_model/config_flow_multiv5.yaml"
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
CKPT_FILE="$OUTPUT_DIR/latest_flow_ckpt.pt"
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
STATS_FILENAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('stats_file', 'v1_multi_global_stats.pt'))")
STATS_PATH="ml_model/$STATS_FILENAME"

if [ "$OUTPUT_DIR" != "ml_output_flowmulti_v5_south_asia_global_context" ] || [ "$CONFIG_TARGET_DOMAIN" != "south_asia" ]; then
    echo "❌ Refusing to train unexpected config: output_dir=$OUTPUT_DIR target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi
if [ "$CONFIG_GLOBAL_CONTEXT" != "sst,sss,ivt,z500_zonal_dev,u250" ]; then
    echo "❌ Refusing to train unexpected global context variables: $CONFIG_GLOBAL_CONTEXT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

MAX_EPOCHS=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH'))['epochs']))" 2>/dev/null || echo "500")
MIXED_PRECISION=$(python -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))" 2>/dev/null || echo "no")
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Max epochs: $MAX_EPOCHS"
echo "🎯 Mixed precision: $MIXED_PRECISION"
echo "🎯 Output dir: $OUTPUT_DIR"
echo "🎯 Target domain: $CONFIG_TARGET_DOMAIN"
echo "🎯 Local predictors: $CONFIG_LOCAL_VARS"
echo "🎯 Global context: $CONFIG_GLOBAL_CONTEXT"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"
echo "🎯 Stats file: $STATS_PATH"

if [ -f "$STATS_PATH" ]; then
    if ! python - "$STATS_PATH" <<'PY'
import math
import sys
import torch

stats = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
bad = []
for key, value in stats.items():
    if isinstance(value, dict) and "min" in value and "max" in value:
        vmin, vmax = float(value["min"]), float(value["max"])
        if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin <= vmax):
            bad.append(f"{key}: {vmin}..{vmax}")
if bad:
    print("Invalid stats bounds: " + "; ".join(bad))
    sys.exit(1)
PY
    then
        echo "⚠️ Existing stats file is invalid. Removing $STATS_PATH before recomputing."
        rm -f "$STATS_PATH"
    fi
fi

if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing global statistics from $DATA_DIR_OVERRIDE..."
    python ml_model/calculate_global_stats_multi_v1.py --data_root "$DATA_DIR_OVERRIDE" --out "$STATS_PATH"
else
    echo "✅ Global stats found at $STATS_PATH. Skipping recalculation."
fi

if [ -f "$CKPT_FILE" ]; then
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(int(ckpt.get('epoch', -1)))" 2>/dev/null || echo "-1")
    if [ "$CURRENT_EPOCH" -ge "$MAX_EPOCHS" ]; then
        echo "✅ Checkpoint already at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Skipping training and resubmission."
        echo "🏁 Job finished at $(date)"
        exit 0
    fi
fi

echo "🔥 Launching Flow Matching Multi-Target v5 (SA target, local predictors, global SST/SSS context)..."
accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" ml_model/train_flow_multiv5.py --config "$CONFIG_PATH" \
    --epochs-per-run 20

echo "🔄 Checking if we need to resubmit..."
echo "📍 Current Host: $(hostname)"
echo "📍 Submit Host: $SLURM_SUBMIT_HOST"
echo "📍 Working Dir: $PWD"

if [ -f "$CKPT_FILE" ]; then
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(ckpt['epoch'])" 2>/dev/null || echo "-1")
    if [ "$CURRENT_EPOCH" -lt "$MAX_EPOCHS" ]; then
        echo "📍 Training at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Resubmitting job..."
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi
        echo "📡 Attempting SSH resubmission to $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" "cd $PWD && sbatch ml_model/submit_train_flowmultiv5.sh"
    else
        echo "✅ Final Epoch $CURRENT_EPOCH reached. Chaining complete."
    fi
else
    echo "⚠️ Checkpoint not found. Chaining stopped to prevent infinite loops."
fi

echo "🏁 South Asia Multi-v5 local/global training job finished at $(date)"
