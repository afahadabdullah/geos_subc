#!/bin/bash
#SBATCH -J smoke_pure_t2m
#SBATCH -o ml_output_flowmulti/smoke_t2m_extremes_pure_%j.log
#SBATCH -e ml_output_flowmulti/smoke_t2m_extremes_pure_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Pure-noise T2M smoke test started at $(date) on $(hostname)"

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

python3 ml_model/smoke_test_t2m_extremes_pure.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess
status=$?

if [ "$status" -ne 0 ]; then
    echo "❌ Pure-noise T2M smoke test failed with exit code $status."
    exit "$status"
fi

echo "✅ Pure-noise smoke-test outputs written under ml_output_flowmulti/smoke_t2m_extremes_pure_2021_2022"
echo "🏁 Pure-noise T2M smoke test finished at $(date)"
