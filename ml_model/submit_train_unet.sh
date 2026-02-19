#!/bin/bash
#SBATCH -J train_unet_v3_dev      # Job name
#SBATCH -o ml_output_unet/job.%j.out    # Name of stdout output file
#SBATCH -e ml_output_unet/job.%j.err    # Name of stderr error file
#SBATCH -p gh-dev                 # Queue (partition) name 'gh-dev' (Developer queue)
#SBATCH -N 1                      # Total # of nodes 
#SBATCH -n 1                      # Total # of tasks
#SBATCH -t 02:00:00               # Run time (hh:mm:ss) - 2 hours max
#SBATCH --mail-type=all           # Send email at begin and end of job
#SBATCH --mail-user=a.fahad@nasa.gov  # Email for notifications
# #SBATCH -A <YOUR_ALLOCATION>    # Allocation name (uncomment if needed)

# -----------------------------------------------------------------
# Setup Environment
# -----------------------------------------------------------------
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

# Load necessary modules (if specific modules are needed on Vista)
# module load python3

# Activate Conda Environment
# Assuming ~/.bashrc initializes conda
source ~/.bashrc
conda activate geossub_env

# Verify Environment
echo "Python: $(which python)"
python --version

# -----------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------
# Ensure output directory exists (though python script does this too)
mkdir -p ml_output_unet

# Run the training script
# Note: Assumes sbatch was called from the project root directory
# Config is set to 500 epochs in ml_model/config_unet.yaml
python ml_model/train_UNET_updated.py --config ml_model/config_unet.yaml

echo "Job finished at $(date)"
