#!/bin/bash
#SBATCH -J cmp_noise_multi             # Job name
#SBATCH -o ml_output_flowmulti/cmp_noise_v4_%j.log
#SBATCH -e ml_output_flowmulti/cmp_noise_v4_%j.log
#SBATCH -p gh-dev                      # Queue (partition) name
#SBATCH -N 1                           # Total # of nodes
#SBATCH -n 1                           # Total # of tasks
#SBATCH -t 02:00:00                    # Run time (hh:mm:ss)
#SBATCH -A ATM25008                    # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Starting multi-variate noise comparison (v4-multi)..."
date

source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# ─── Step 1: Run noise comparison ───
# Automaticaly detects best model in output_dir
python3 ml_model/compare_noise_v4_multi.py \
    --output_dir ml_output_flowmulti \
    --year 2022 \
    --num_ensemble 30 \
    --num_steps 10

echo "Multi-variate noise comparison finished!"
date
