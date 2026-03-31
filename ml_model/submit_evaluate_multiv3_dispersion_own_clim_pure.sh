#!/bin/bash
#SBATCH -J disp_pure_own
#SBATCH -o ml_output_flowmulti/disp_ownclim_pure_%j.log
#SBATCH -e ml_output_flowmulti/disp_ownclim_pure_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Pure-noise own-climatology dispersion job started at $(date) on $(hostname)"

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

OUTPUT_DIR="ml_output_flowmulti/multiv1_dispersion_own_clim_pure_2020_june"

echo "🎯 Evaluating pure-noise own-climatology anomaly dispersion for June 2020 init dates"
python3 ml_model/evaluate_multiv1_dispersion_own_clim_pure.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess \
    --start_year 2020 \
    --end_year 2020 \
    --init_months 6 \
    --output_dir "$OUTPUT_DIR"
status=$?

if [ "$status" -ne 0 ]; then
    echo "❌ Pure-noise own-climatology dispersion evaluation failed with exit code $status."
    exit "$status"
fi

echo "✅ Outputs written under $OUTPUT_DIR"
echo "🏁 Pure-noise own-climatology dispersion job finished at $(date)"
