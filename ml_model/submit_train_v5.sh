#!/bin/bash
#SBATCH -J flow_v5                   # Job name
#SBATCH -o ml_output_flow5/flow_%j.log
#SBATCH -j y                          # Merge stderr into stdout
#SBATCH -p gh-dev                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Flow v5 Training started at $(date) on $(hostname)"

# Environment Setup
source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# Cleanup zombie processes from previous failed runs (VRAM protection)
echo "🧹 Cleaning up previous zombie processes..."
pkill -9 -u $USER -f python
pkill -9 -u $USER -f accelerate
sleep 3

echo "🔄 Pulling latest from git..."
git pull

# Ensure output directory exists
mkdir -p ml_output_flow5

# Run Global Stats calculation (OVERRIDE: only if stats file doesn't exist)
STATS_PATH="ml_model/v5_global_stats.pt"
if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing global statistics..."
    python ml_model/calculate_global_stats_v5.py
else
    echo "✅ Global stats found at $STATS_PATH. Skipping recalculation."
fi

# Generate land-sea mask if it doesn't exist
MASK_PATH="ml_model/land_sea_mask.pt"
if [ ! -f "$MASK_PATH" ]; then
    echo "🗺️ Land-sea mask not found. Generating..."
    DATA_ROOT=/scratch/11353/afahad/geossub/geos_subc/dataprocess python ml_model/generate_land_mask.py
else
    echo "✅ Land-sea mask found at $MASK_PATH."
fi

echo "🔥 Launching Flow v5 Training (EOF-Native + Joint VH + Land/Ocean Weights)..."
accelerate launch --num_processes 1 --mixed_precision fp16 ml_model/train_flow_v5.py --config ml_model/config_flow_v5.yaml \
    --epochs-per-run 20

# --- AUTOMATIC JOB CHAINING ---
echo "🔄 Checking if we need to resubmit..."
CKPT_FILE="ml_output_flow5/latest_flow_ckpt.pt"
MAX_EPOCHS=1000

if [ -f "$CKPT_FILE" ]; then
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(ckpt['epoch'])" 2>/dev/null || echo "-1")
    
    if [ "$CURRENT_EPOCH" -lt "$((MAX_EPOCHS - 1))" ]; then
        echo "📍 Training at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Resubmitting job..."
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi
        echo "📡 Attempting SSH resubmission to $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" "cd $PWD && sbatch ml_model/submit_train_v5.sh"
    else
        echo "✅ Final Epoch $CURRENT_EPOCH reached. Chaining complete."
    fi
else
    echo "⚠️ Checkpoint not found. Chaining stopped to prevent infinite loops."
fi

echo "🏁 Job finished at $(date)"
