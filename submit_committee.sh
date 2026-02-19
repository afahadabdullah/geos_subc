#!/bin/bash
#SBATCH -J train_committee
#SBATCH -o train_committee.o%j
#SBATCH -e train_committee.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A EAR24051

# --- Setup Environment ---
source /etc/profile.d/z00_lmod.sh
module load python/3.12

# Fix interactive vs batch conda activation
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geossub_env

export PYTHONUNBUFFERED=1

# TACC specific fixes
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Preload libstdc++ if needed (handled in python script too, but good to have)
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# PyTorch Memory Fixes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Training ---
PROJ_DIR="/work/10123/afahad/vista/geos_subc2" # Update if different on remote check user provided paths
# Wait, user is on Mac local? Or remote?
# User ssh'd to vista. So we are editing local and pushing.
# We don't know the exact remote path.
# I'll use a generic path or $PWD logic if possible.
# But SLURM starts in submit dir usually.

echo "Starting Committee Model Training..."
date

accelerator launch ml_model/train_committee.py

echo "Training Finished."
date
