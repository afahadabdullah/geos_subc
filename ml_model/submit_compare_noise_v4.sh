#!/bin/bash
#SBATCH -J cmp_noise_v4                # Job name
#SBATCH -o ml_output_flow4/cmp_noise_v4_%j.log
#SBATCH -e ml_output_flow4/cmp_noise_v4_%j.log
#SBATCH -p gh                          # Queue (partition) name
#SBATCH -N 1                           # Total # of nodes
#SBATCH -n 4                           # Total # of tasks
#SBATCH --gpus=1                       # Total # of GPUs
#SBATCH -t 02:00:00                    # Run time (hh:mm:ss)
#SBATCH -A ATM25008                    # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Starting multi-modal noise comparison (v4)..."
date

source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

# Move to Scratch storage
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# ─── Step 1: Compute NAO and ENSO EOF bases (if not already done) ───
if [ ! -f ml_model/nao_eof_bases.pt ]; then
    echo "Computing NAO EOF bases..."
    python3 dataprocess/compute_nao_eofs.py --data_dir /home1/11353/afahad/geos_subc/dataprocess
fi

if [ ! -f ml_model/enso_eof_bases.pt ]; then
    echo "Computing ENSO EOF bases..."
    python3 dataprocess/compute_enso_eofs.py --data_dir /home1/11353/afahad/geos_subc/dataprocess
fi

# ─── Step 2: Run noise comparison ───
python3 ml_model/compare_noise_v4.py \
    --output_dir ml_output_flow4 \
    --year 2022 \
    --num_ensemble 30 \
    --num_steps 10

echo "Noise comparison v4 finished!"
date
