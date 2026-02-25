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
conda activate pyenv

# The argument points to the latest Flow Matcher checkpoint and the year to test
# Adjust --ckpt to the best performing model if you prefer
python3 ml_model/test_flow.py --config ml_model/config_flow.yaml --ckpt ml_output_flow/latest_flow_ckpt.pt --year 2015 --ensemble-size 20

echo "Testing finished!"
date
