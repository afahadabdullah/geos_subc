#!/bin/bash
#SBATCH -J build_pure_clim
#SBATCH -o ml_output_flowmulti/build_multiv1_pure_climatology_%j.log
#SBATCH -e ml_output_flowmulti/build_multiv1_pure_climatology_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Pure-noise climatology build started at $(date) on $(hostname)"

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
mkdir -p dataprocess/clim_pure

echo "📅 Building pure weekly climatology (1999-2021)"
python3 ml_model/build_weekly_multiv1_pure_climatology.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess \
    --overwrite
status=$?
if [ "$status" -ne 0 ]; then
    echo "❌ Weekly pure climatology build failed with exit code $status."
    exit "$status"
fi

echo "📅 Building pure monthly climatology (1999-2021)"
python3 ml_model/build_monthly_multiv1_pure_climatology.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess \
    --overwrite
status=$?
if [ "$status" -ne 0 ]; then
    echo "❌ Monthly pure climatology build failed with exit code $status."
    exit "$status"
fi

echo "✅ Pure weekly and monthly climatology stores are ready in dataprocess/clim_pure"
echo "🏁 Pure-noise climatology build finished at $(date)"
