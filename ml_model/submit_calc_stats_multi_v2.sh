#!/bin/bash
#SBATCH -J calc_stats_multi_v2      # Job name
#SBATCH -o ml_output_flowmulti_v2/stats_multi_v2_%j.log
#SBATCH -e ml_output_flowmulti_v2/stats_multi_v2_%j.log
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

# Ensure log directory exists
mkdir -p ml_output_flowmulti_v2

echo "🔄 Pulling latest stability fixes from git..."
git pull

echo "🔥 Launching Global Statistics Calculation (Multi-Target v2)..."
python ml_model/calculate_global_stats_multi_v2.py

echo "🏁 Job finished at $(date)"
