#!/bin/bash
#SBATCH -J gen_targets                # Job name
#SBATCH -o ml_output_flow4/gen_targets_%j.log
#SBATCH -e ml_output_flow4/gen_targets_%j.log
#SBATCH -p gh-dev                      # Queue (partition) name
#SBATCH -N 1                           # Total # of nodes
#SBATCH -n 1                           # Total # of tasks
#SBATCH -t 02:00:00                    # Run time (hh:mm:ss)
#SBATCH -A ATM25008                    # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Starting dataset generation for Spatial Spread Generator (SSG)..."
date

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

python3 ml_model/generate_noise_targets.py \
    --data_dir /scratch/11353/afahad/geossub/geos_subc/dataprocess \
    --out_dir /scratch/11353/afahad/geossub/geos_subc/dataprocess/noise \
    --checkpoint ml_output_flow4/BEST_model.pt \
    --start_year 2010 \
    --end_year 2020 \
    --batch_size 1 \
    --num_ensemble 15 \
    --num_steps 10

echo "Noise target generation finished!"
date
