#!/bin/bash
#SBATCH -J seasonal_skill
#SBATCH -o ml_output_flowmulti/seasonal_skill_%j.log
#SBATCH -e ml_output_flowmulti/seasonal_skill_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Seasonal skill evaluation job started at $(date) on $(hostname)"

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

OUTPUT_DIR="ml_output_flowmulti/seasonal_skill_2020_2021"

echo "🎯 Evaluating seasonal held-out skill for T2M and PR over 2020-2021"
python3 ml_model/evaluate_multiv1_seasonal_skill.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess \
    --ml_dir dataprocess/gen_multiv1 \
    --start_year 2020 \
    --end_year 2021 \
    --threshold_start_year 1999 \
    --threshold_end_year 2019 \
    --seasons DJF MAM JJA SON \
    --variables tas pr \
    --sample_chunk_size 2 \
    --obs_clim_path dataprocess/clim/obs_weekly_clim_1999_2021.zarr \
    --output_dir "$OUTPUT_DIR"
status=$?

if [ "$status" -ne 0 ]; then
    echo "❌ Seasonal skill evaluation failed with exit code $status."
    exit "$status"
fi

echo "✅ Outputs written under $OUTPUT_DIR"
echo "🏁 Seasonal skill evaluation finished at $(date)"
