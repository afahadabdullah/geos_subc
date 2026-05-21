#!/bin/bash
#SBATCH -J SA_flow_v4                # Job name
#SBATCH -o ml_output_flowmulti_v4_south_asia/flow_%j.log
#SBATCH -e ml_output_flowmulti_v4_south_asia/flow_%j.log
#SBATCH -p gh-dev                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 South Asia Multi-v4 training job started at $(date) on $(hostname)"

# Move to Scratch storage before activating the repo-local environment.
PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

# Environment Setup: use the repo-local conda created by setup_env.sh.
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

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export SYMPY_GROUND_TYPES=python
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

# Optional cleanup for stale processes from previous failed runs.
# Disabled by default because broad pkill patterns can stop unrelated jobs.
if [ "${CLEANUP_OLD_PROCS:-0}" = "1" ]; then
    echo "🧹 Cleaning up previous SA flow processes..."
    pkill -9 -u "$USER" -f "train_flow_multiv4.py" || true
    pkill -9 -u "$USER" -f "accelerate.*train_flow_multiv4.py" || true
    sleep 3
else
    echo "🧹 Skipping process cleanup. Set CLEANUP_OLD_PROCS=1 to enable it."
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"

# Ensure output directory exists
mkdir -p ml_output_flowmulti_v4_south_asia

CONFIG_PATH="ml_model/config_flow_multiv4.yaml"
CKPT_FILE="ml_output_flowmulti_v4_south_asia/latest_flow_ckpt.pt"
CONFIG_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
if [ "$CONFIG_OUTPUT_DIR" != "ml_output_flowmulti_v4_south_asia" ] || [ "$CONFIG_TARGET_DOMAIN" != "south_asia" ]; then
    echo "❌ Refusing to train unexpected config: output_dir=$CONFIG_OUTPUT_DIR target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi

MAX_EPOCHS=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH'))['epochs']))" 2>/dev/null || echo "500")
MIXED_PRECISION=$(python -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))" 2>/dev/null || echo "no")
echo "🎯 Max epochs from $CONFIG_PATH: $MAX_EPOCHS"
echo "🎯 Mixed precision from $CONFIG_PATH: $MIXED_PRECISION"
echo "🎯 Output dir from $CONFIG_PATH: $CONFIG_OUTPUT_DIR"
echo "🎯 Target domain from $CONFIG_PATH: $CONFIG_TARGET_DOMAIN"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"

# Run Global Stats calculation (only if stats file doesn't exist)
STATS_PATH="ml_model/v1_multi_global_stats.pt"
if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing global statistics..."
    python ml_model/calculate_global_stats_multi_v1.py
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

echo "🔥 Launching Flow Matching Multi-Target v4 Training (South Asia target domain, PR + T2M)..."
accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" ml_model/train_flow_multiv4.py --config "$CONFIG_PATH" \
    --epochs-per-run 20

# --- AUTOMATIC JOB CHAINING ---
echo "🔄 Checking if we need to resubmit..."
echo "📍 Current Host: $(hostname)"
echo "📍 Submit Host: $SLURM_SUBMIT_HOST"
echo "📍 Working Dir: $PWD"

if [ -f "$CKPT_FILE" ]; then
    # Safely get current epoch from checkpoint
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(ckpt['epoch'])" 2>/dev/null || echo "-1")
    
    if [ "$CURRENT_EPOCH" -lt "$MAX_EPOCHS" ]; then
        echo "📍 Training at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Resubmitting job..."
        # TACC Fix: Resubmit via SSH to the submission host (full domain)
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi

        echo "📡 Attempting SSH resubmission to $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" "cd $PWD && sbatch ml_model/submit_train_flowmultiv4.sh"
    else
        echo "✅ Final Epoch $CURRENT_EPOCH reached. Chaining complete."
    fi
else
    echo "⚠️ Checkpoint not found. Chaining stopped to prevent infinite loops."
fi

echo "🏁 South Asia Multi-v4 training job finished at $(date)"
