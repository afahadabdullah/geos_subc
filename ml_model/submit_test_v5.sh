#!/bin/bash
#SBATCH -J test_v5          # Job name
#SBATCH -o test_v5_%j.o     # Combined Output and Error file name
#SBATCH -p gh-dev           # partition
#SBATCH -N 1                # Total number of nodes
#SBATCH -n 1                # Number of tasks per node
#SBATCH -t 02:00:00         # Run time (d-hh:mm:ss)

echo "Starting Flow v5 testing script..."
date

source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

git pull

# Auto-discover best checkpoint
CKPT=$(ls -1 ml_output_flow5/best_model_epoch_*.pt 2>/dev/null | sort -t_ -k4 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "❌ No best_model checkpoint found in ml_output_flow5/. Exiting."
    exit 1
fi
echo "Using checkpoint: $CKPT"

for YEAR in 2021; do
    echo "--- Running Flow v5 Inference Validation for Year $YEAR ---"
    python3 ml_model/test_flow_v5.py --config ml_model/config_flow_v5.yaml \
        --ckpt $CKPT \
        --year $YEAR \
        --ensemble-size 30 \
        --steps 10
    echo "---------------------------------------------------"
done

echo "Testing finished!"
date
