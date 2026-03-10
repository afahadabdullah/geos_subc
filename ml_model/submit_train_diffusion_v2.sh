#!/bin/bash
#SBATCH -J train_diffusion_v2      # Job name
#SBATCH -o ml_output_diffusion_v2/job.%j.log
#SBATCH -e ml_output_diffusion_v2/job.%j.log
#SBATCH -p gh-dev                  # Queue (partition) name
#SBATCH -N 1                       # Total # of nodes
#SBATCH -n 1                       # Total # of tasks
#SBATCH -t 02:00:00                # Run time (hh:mm:ss)
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "Job started at $(date)"
echo "Running on node: $(hostname)"

source ~/.bashrc
conda activate geossub_env

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python: $(which python)"
python --version

# Ensure we are in the project root directory
cd /scratch/11353/afahad/geossub/geos_subc || exit 1

mkdir -p ml_output_diffusion_v2

# Run diffusion v2 script
python ml_model/train_diffusion_v2.py --config ml_model/config_diffusion.yaml

echo "Job finished at $(date)"
