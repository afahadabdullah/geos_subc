#!/bin/bash
#SBATCH -J cmde_train                # Job name
#SBATCH -o cmde_train.o%j            # Stdout output file
#SBATCH -e cmde_train.e%j            # Stderr error file
#SBATCH -p gh-dev                    # Partition (GH200 dev queue)
#SBATCH -N 1                         # Total nodes
#SBATCH -n 1                         # Total MPI tasks
#SBATCH --gpus-per-node=1            # 1 GPU per node
#SBATCH -t 02:00:00                  # Run time (2 hours for dev; increase for gh queue)
#SBATCH -A EAR24012                  # Project/Allocation (update if different)

# ============================================================================
# CMDE Diffusion Model Training - TACC Vista Batch Job
# ============================================================================

echo "============================================"
echo "CMDE Training Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# --- Environment Setup ---
PROJ_DIR="/scratch/11353/afahad/geossub/geos_subc"
CONDA_DIR="/home1/11353/afahad/afahad/geossub/geos_subc/miniconda"
ENV_NAME="geossub_env"

# Load TACC modules
module load gcc cuda

# Activate conda
source "$CONDA_DIR/bin/activate" "$ENV_NAME"

# Critical environment variables
export SYMPY_GROUND_TYPES=python
export LD_LIBRARY_PATH="$CONDA_DIR/envs/$ENV_NAME/lib:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Diagnostics ---
echo ""
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null
echo ""

# --- Run Training ---
cd "$PROJ_DIR"

echo "Starting accelerate launch..."
accelerate launch ml_model/train.py

echo ""
echo "============================================"
echo "CMDE Training Job Finished: $(date)"
echo "============================================"
