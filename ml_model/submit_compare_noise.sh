#!/bin/bash
#SBATCH -J cmp_noise                  # Job name
#SBATCH -o ml_output_flow4/cmp_noise_%j.out
#SBATCH -e ml_output_flow4/cmp_noise_%j.err
#SBATCH -p gh-dev                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Starting noise comparison script..."
date

source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# Run the noise strategy comparison
# Arguments configured in the script explicitly but specifying year 2021 just to be safe
python3 ml_model/compare_noise.py --output_dir ml_output_flow4 --year 2021

echo "Testing finished!"
date
