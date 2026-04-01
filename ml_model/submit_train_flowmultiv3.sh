#!/bin/bash
#SBATCH -J flow_multi_v3             # Job name
#SBATCH -o ml_output_flowmulti_v3/flow_%j.log
#SBATCH -e ml_output_flowmulti_v3/flow_%j.log
#SBATCH -p gh-dev                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Multi-v3 training job started at $(date) on $(hostname)"

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

echo "🔄 Pulling latest fixes from git..."
git pull --no-rebase origin flow_multi

# Ensure output directory exists
mkdir -p ml_output_flowmulti_v3

CONFIG_PATH="ml_model/config_flow_multiv3.yaml"
CKPT_FILE="ml_output_flowmulti_v3/latest_flow_ckpt.pt"
MAX_EPOCHS=$(python3 -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH'))['epochs']))" 2>/dev/null || echo "300")
MIXED_PRECISION=$(python3 -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))" 2>/dev/null || echo "no")
echo "🎯 Max epochs from $CONFIG_PATH: $MAX_EPOCHS"
echo "🎯 Mixed precision from $CONFIG_PATH: $MIXED_PRECISION"

# Run Global Stats calculation (only if stats file doesn't exist)
STATS_PATH="ml_model/v1_multi_global_stats.pt"
if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing global statistics..."
    python3 ml_model/calculate_global_stats_multi_v1.py
else
    echo "✅ Global stats found at $STATS_PATH. Skipping recalculation."
fi

if [ -f "$CKPT_FILE" ]; then
    CURRENT_EPOCH=$(python3 -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(int(ckpt.get('epoch', -1)))" 2>/dev/null || echo "-1")
    if [ "$CURRENT_EPOCH" -ge "$MAX_EPOCHS" ]; then
        echo "✅ Checkpoint already at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Skipping training and resubmission."
        echo "🏁 Job finished at $(date)"
        exit 0
    fi
fi

echo "🔥 Launching multi-v3 training..."
accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" ml_model/train_flow_multiv3.py --config "$CONFIG_PATH" \
    --epochs-per-run 15

# --- AUTOMATIC JOB CHAINING ---
echo "🔄 Checking if we need to resubmit..."
echo "📍 Current Host: $(hostname)"
echo "📍 Submit Host: $SLURM_SUBMIT_HOST"
echo "📍 Working Dir: $PWD"

if [ -f "$CKPT_FILE" ]; then
    # Safely get current epoch from checkpoint
    CURRENT_EPOCH=$(python3 -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(ckpt['epoch'])" 2>/dev/null || echo "-1")
    
    if [ "$CURRENT_EPOCH" -lt "$MAX_EPOCHS" ]; then
        echo "📍 Training at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Resubmitting job..."
        # TACC Fix: Resubmit via SSH to the submission host (full domain)
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi

        echo "📡 Attempting SSH resubmission to $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" "cd $PWD && sbatch ml_model/submit_train_flowmultiv3.sh"
    else
        echo "✅ Final Epoch $CURRENT_EPOCH reached. Chaining complete."
    fi
else
    echo "⚠️ Checkpoint not found. Chaining stopped to prevent infinite loops."
fi

echo "🏁 Multi-v3 training job finished at $(date)"
