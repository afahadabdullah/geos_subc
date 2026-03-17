#!/bin/bash
#SBATCH -J test_flowmulti
#SBATCH -o ml_output_flowmulti/test_flowmulti_%j.log
#SBATCH -e ml_output_flowmulti/test_flowmulti_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🧪 Multi-target test job started at $(date) on $(hostname)"

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

echo "🔄 Pulling latest code..."
git pull

mkdir -p ml_output_flowmulti

CONFIG_PATH="ml_model/config_flow_multiv1.yaml"
CKPT_FILE="best_flow_ckpt.pt"

echo "🚀 Running flow multi test mode with checkpoint ${CKPT_FILE}"
accelerate launch --num_processes 1 --mixed_precision fp16 \
    ml_model/train_flow_multiv1.py \
    --config "$CONFIG_PATH" \
    --test \
    --ckpt "$CKPT_FILE"

echo "🏁 Test job finished at $(date)"
