#!/bin/bash
#SBATCH -J cmp_noise        # Job name
#SBATCH -o cmp_noise_%j.o   # Combined Output and Error file name
#SBATCH -p gh-dev           # partition
#SBATCH -N 1                # Total number of nodes
#SBATCH -n 1                # Number of tasks per node
#SBATCH -t 02:00:00         # Run time (d-hh:mm:ss)

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
