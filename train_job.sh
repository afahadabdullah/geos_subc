#!/bin/bash
#SBATCH -J cmde_train          # Job name
#SBATCH -o ml_output_cmde/train_%j.log  # Output file (%j = job ID)
#SBATCH -e ml_output_cmde/train_%j.log  # Error file
#SBATCH -p gh-dev               # Partition (gh-dev for GH200 dev queue)
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 task
#SBATCH -t 02:00:00             # 2 hours (adjust as needed)
#SBATCH --mail-type=END,FAIL    # Email on completion or failure
#SBATCH --mail-user=afahad@gmu.edu

# ============================================================
# CMDE Training Job for TACC Vista (GH200)
# Submit with: sbatch train_job.sh
# Monitor with: squeue -u $USER
# Check output: tail -f ml_output_cmde/train_<JOBID>.out
# ============================================================

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOBID"

# Load required modules
module load gcc cuda

# Setup conda
source $WORK/geossub/geos_subc/miniconda/bin/activate geossub_env

# Environment variables
export SYMPY_GROUND_TYPES=python
export LD_LIBRARY_PATH="$WORK/geossub/geos_subc/miniconda/envs/geossub_env/lib:$LD_LIBRARY_PATH"

# Navigate to project
cd $SCRATCH/geossub/geos_subc

# Create output dir
mkdir -p ml_output_cmde/plots

# Print GPU info
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Launch training (auto-resumes from latest checkpoint)
accelerate launch ml_model/train.py

echo "Job finished at $(date)"
