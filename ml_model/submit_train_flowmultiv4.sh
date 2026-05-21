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

# Environment Setup
echo "🔧 Loading ~/.bashrc..."
set +e
source ~/.bashrc
BASHRC_STATUS=$?
set -e
echo "🔧 ~/.bashrc exit status: $BASHRC_STATUS"
echo "🔧 Clearing any stale conda shell definitions..."
unset -f conda __conda_activate __conda_reactivate __conda_hashr 2>/dev/null || true
unalias conda 2>/dev/null || true
hash -r 2>/dev/null || true

if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
    echo "🔧 Sourcing CONDA_SH override: $CONDA_SH"
    source "$CONDA_SH"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "🔧 conda not found; searching for conda.sh..."
    for CONDA_SH in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/miniconda3/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/miniconda/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/anaconda3/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/miniforge3/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/mambaforge/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/geossub/geos_subc/miniconda/etc/profile.d/conda.sh" \
        "/home1/11353/afahad/geossub/geos_subc/miniconda3/etc/profile.d/conda.sh" \
        "/scratch/11353/afahad/geossub/miniconda/etc/profile.d/conda.sh" \
        "/scratch/11353/afahad/geossub/miniconda3/etc/profile.d/conda.sh" \
        "/scratch/11353/afahad/geossub/geos_subc/miniconda/etc/profile.d/conda.sh" \
        "/scratch/11353/afahad/geossub/geos_subc/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$CONDA_SH" ]; then
            echo "🔧 Sourcing $CONDA_SH"
            source "$CONDA_SH"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "❌ conda command is still unavailable. Check the conda install path on Vista."
    echo "   Tip: find /home1/11353/afahad /scratch/11353/afahad -maxdepth 7 -path '*/etc/profile.d/conda.sh' -print 2>/dev/null"
    echo "   Then submit with: CONDA_SH=/path/to/conda.sh sbatch ml_model/submit_train_flowmultiv4.sh"
    exit 1
fi
echo "🔧 Activating conda environment: geossub_env"
conda activate geossub_env
echo "✅ Conda environment active: ${CONDA_PREFIX:-unset}"

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# Cleanup zombie processes from previous failed runs (VRAM protection)
echo "🧹 Cleaning up previous zombie processes..."
pkill -9 -u $USER -f python
pkill -9 -u $USER -f accelerate
sleep 3

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
