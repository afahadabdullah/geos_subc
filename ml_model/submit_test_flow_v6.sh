#!/bin/bash
#SBATCH -J test_v6
#SBATCH -o ml_output_flow6/test_v6_%j.log
#SBATCH -e ml_output_flow6/test_v6_%j.log
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

module load python
source /home1/11353/afahad/geos_subc/setup_env.sh

cd /home1/11353/afahad/geos_subc

# Run V6 evaluation for 2015 with high ensemble size and 50 steps
python ml_model/test_flow_v6.py \
    --year 2015 \
    --ensemble-size 15 \
    --steps 50 \
    --config ml_model/config_flow.yaml
