#!/bin/bash
#SBATCH -J cmp_ckpt_multi
#SBATCH -o ml_output_flowmulti/cmp_ckpt_multi_%j.log
#SBATCH -e ml_output_flowmulti/cmp_ckpt_multi_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Starting checkpoint sweep under pure random noise..."
date

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

python3 ml_model/compare_noise_v4_ckpts_multi.py \
    --output_dir ml_output_flowmulti \
    --year 2021 \
    --num_ensemble 30 \
    --num_steps 10

echo "Checkpoint sweep finished!"
date
