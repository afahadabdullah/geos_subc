#!/bin/bash
#SBATCH -J seasonal_skill_pure
#SBATCH -o ml_output_flowmulti/seasonal_skill_pure_%j.log
#SBATCH -e ml_output_flowmulti/seasonal_skill_pure_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Pure-noise seasonal skill evaluation started at $(date) on $(hostname)"

source ~/.bashrc
conda activate geossub_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /scratch/11353/afahad/geossub/geos_subc || exit 1

echo "🔄 Pulling latest code..."
git pull --no-rebase origin flow_multi

mkdir -p ml_output_flowmulti

echo "🎯 Evaluating pure-noise seasonal skill for T2M and PR over 2020-2021"
python3 ml_model/evaluate_multiv1_seasonal_skill_pure.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess
status=$?

if [ "$status" -ne 0 ]; then
    echo "❌ Pure-noise seasonal skill evaluation failed with exit code $status."
    exit "$status"
fi

echo "✅ Outputs written under ml_output_flowmulti/seasonal_skill_pure_2020_2021"
echo "🏁 Pure-noise seasonal skill evaluation finished at $(date)"
