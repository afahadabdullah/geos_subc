#!/bin/bash
#SBATCH -J gen_multiv1
#SBATCH -o ml_output_flowmulti/gen_multiv1_%j.log
#SBATCH -e ml_output_flowmulti/gen_multiv1_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Multi-v1 ensemble generation job started at $(date) on $(hostname)"

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

echo "🔄 Pulling latest code..."
git pull --no-rebase origin flow_multi

mkdir -p ml_output_flowmulti
mkdir -p dataprocess/gen_multiv1

CONFIG_PATH="ml_model/config_flow_multiv1.yaml"
STATS_PATH="ml_model/v1_multi_global_stats.pt"

if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing global statistics..."
    python3 ml_model/calculate_global_stats_multi_v1.py
else
    echo "✅ Global stats found at $STATS_PATH"
fi

echo "🎯 Generating 120-member multi-v1 ensembles for 2020-2021 using the best checkpoint"
accelerate launch --num_processes 1 --mixed_precision fp16 \
    ml_model/generate_multiv1_ensembles.py \
    --config "$CONFIG_PATH" \
    --start_year 2020 \
    --end_year 2021 \
    --num_ensemble 120 \
    --ensemble_chunk_size 30 \
    --num_steps 50 \
    --out_dir dataprocess/gen_multiv1

echo "🏁 Multi-v1 ensemble generation job finished at $(date)"
