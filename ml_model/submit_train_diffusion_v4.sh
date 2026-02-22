#!/bin/bash
#SBATCH -J diff_v4
#SBATCH -o ml_output_diffusion_v4/diff_v4_%j.out
#SBATCH -e ml_output_diffusion_v4/diff_v4_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A EAR24005

cd /home1/11353/afahad/geos_subc
source ~/.bashrc
conda activate geossub_env

echo "Pulling latest code fixes..."
git pull

echo "Rebuilding proper linear V4 Global Statistics..."
python ml_model/calculate_global_stats_v4.py

echo "Launching V4 Diffusion Training..."
accelerate launch ml_model/train_diffusion_v4.py
