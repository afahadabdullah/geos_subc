#!/bin/bash
#SBATCH -J diff_v4                  # Job name
#SBATCH -o ml_output_diffusion_v4/diff_v4_%j.out
#SBATCH -e ml_output_diffusion_v4/diff_v4_%j.err
#SBATCH -p gh                        # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 48:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Job started at $(date) on $(hostname)"

# Environment Setup
source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

echo "🔄 Pulling latest stability fixes from git..."
git pull

# Ensure output directory exists
mkdir -p ml_output_diffusion_v4

echo "📈 Global Statistics check..."
# Note: Stats are now forced robustly in the code, but we run the scan just in case
python ml_model/calculate_global_stats_v4.py

echo "🔥 Launching V4 Diffusion Training with Stability Patch..."
accelerate launch ml_model/train_diffusion_v4.py --config ml_model/config_diffusion_v4.yaml

echo "🏁 Job finished at $(date)"
