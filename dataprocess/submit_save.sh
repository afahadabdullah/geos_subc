#!/bin/bash
#SBATCH -J geos_subc_save           # Job name
#SBATCH -o geos_subc_save.o%j       # Name of stdout output file
#SBATCH -e geos_subc_save.e%j       # Name of stderr error file
#SBATCH -p gg                       # Partition (Grace-Grace high memory nodes)
#SBATCH -N 1                        # Total # of nodes 
#SBATCH -n 1                        # Total # of mpi tasks
#SBATCH -t 06:00:00                 # Run time (hh:mm:ss)
#SBATCH -A 11353                    # Project/Allocation name

# Load the environment
# Adjust this path if your miniconda is in a different location
CONDA_DIR="/home1/11353/afahad/afahad/geossub/miniconda"
source "$CONDA_DIR/bin/activate" geossub_env

# Run the data saving script
# We run from the project root
cd /home1/11353/afahad/afahad/geossub
python dataprocess/load_data.py
