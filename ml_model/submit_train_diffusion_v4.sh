#!/bin/bash
#SBATCH -J diff_v4                  # Job name
#SBATCH -o ml_output_diffusion_v4/diff_v4_%j.out
#SBATCH -e ml_output_diffusion_v4/diff_v4_%j.err
#SBATCH -p gh-dev                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Job started at $(date) on $(hostname)"

# Environment Setup
source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

echo "🔄 Pulling latest stability fixes from git..."
git pull

# Ensure output directory exists
mkdir -p ml_output_diffusion_v4

# Run Global Stats calculation (OVERRIDE: only if stats file doesn't exist)
STATS_PATH="ml_model/v4_global_stats.pt"
if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing global statistics..."
    python ml_model/calculate_global_stats_v4.py
else
    echo "✅ Global stats found at $STATS_PATH. Skipping recalculation."
fi

echo "🔥 Launching V4 Diffusion Training [15 Epochs this session]..."
accelerate launch --num_processes 1 --mixed_precision fp16 ml_model/train_diffusion_v4.py --config ml_model/config_diffusion_v4.yaml --epochs-per-run 15

# --- AUTOMATIC JOB CHAINING ---
echo "🔄 Checking if we need to resubmit..."
CKPT_FILE="ml_output_diffusion_v4/latest_diffusion_ckpt_v4.pt"
MAX_EPOCHS=1000

if [ -f "$CKPT_FILE" ]; then
    # Safely get current epoch from checkpoint
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(ckpt['epoch'])" 2>/dev/null || echo "-1")
    
    if [ "$CURRENT_EPOCH" -lt "$((MAX_EPOCHS - 1))" ]; then
        echo "📍 Training at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Resubmitting job..."
        # Resubmit this same script
        sbatch ml_model/submit_train_diffusion_v4.sh
    else
        echo "✅ Final Epoch $CURRENT_EPOCH reached. Chaining complete."
    fi
else
    echo "⚠️ Checkpoint not found. Chaining stopped to prevent infinite loops."
fi

echo "🏁 Job finished at $(date)"
