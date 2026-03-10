#!/bin/bash
#SBATCH -J train_ssg                  # Job name
#SBATCH -o ml_output_ssg/train_ssg_%j.log
#SBATCH -e ml_output_ssg/train_ssg_%j.log
#SBATCH -p gh                          # Queue (partition) name
#SBATCH -N 1                           # Total # of nodes
#SBATCH -n 4                           # Total # of tasks
#SBATCH --gpus=1                       # Total # of GPUs
#SBATCH -t 04:00:00                    # Run time (hh:mm:ss)
#SBATCH -A ATM25008                    # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Starting Spatial Spread Generator (SSG) Network Training..."
date

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

# Create output dir if it doesn't exist
mkdir -p ml_output_ssg

# Run training with Accelerate (single GPU is allocated)
accelerate launch --num_processes 1 ml_model/train_ssg.py \
    --data_dir /scratch/11353/afahad/geossub/geos_subc/dataprocess/noise \
    --out_dir ml_output_ssg \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-3 \
    --resume

echo "SSG Training finished!"
date
