#!/bin/bash
#SBATCH -J test_flow        # Job name
#SBATCH -o test_flow_%j.o   # Combined Output and Error file name
#SBATCH -p gh-dev           # partition
#SBATCH -N 1                # Total number of nodes
#SBATCH -n 1                # Number of tasks per node
#SBATCH -t 02:00:00         # Run time (d-hh:mm:ss)

echo "Starting testing script..."
date

source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# The argument points to the best Flow Matcher checkpoint and the year to test
python3 ml_model/test_flow.py --config ml_model/config_flow.yaml --ckpt /home1/11353/afahad/geos_subc/ml_output_flow/best_model_epoch_208_crps_0.6159.pt --year 2015 --ensemble-size 20

echo "Testing finished!"
date
