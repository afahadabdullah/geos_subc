#!/bin/bash
#SBATCH -J disp_ownclim
#SBATCH -o ml_output_flowmulti/disp_ownclim_%j.log
#SBATCH -e ml_output_flowmulti/disp_ownclim_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

echo "🚀 Own-climatology dispersion job started at $(date) on $(hostname)"

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

OUTPUT_DIR="ml_output_flowmulti/multiv1_dispersion_own_clim_mjja_2020_2021"

echo "🎯 Evaluating MJJA 2020-2021 own-climatology anomaly dispersion for T2M and PR"
python3 ml_model/evaluate_multiv1_dispersion_own_clim.py \
    --data_dir /home1/11353/afahad/geos_subc/dataprocess \
    --ml_dir dataprocess/gen_multiv1 \
    --start_year 2020 \
    --end_year 2021 \
    --init_months 5 6 7 8 \
    --variables tas pr \
    --fair_member_count 4 \
    --sample_chunk_size 2 \
    --pit_bins 10 \
    --pit_seed 7 \
    --ml_clim_path dataprocess/clim/ml_weekly_ensmean_clim_1999_2021.zarr \
    --geos_clim_path dataprocess/clim/geos_weekly_ensmean_clim_1999_2021.zarr \
    --obs_clim_path dataprocess/clim/obs_weekly_clim_1999_2021.zarr \
    --output_dir "$OUTPUT_DIR"
status=$?

if [ "$status" -ne 0 ]; then
    echo "❌ Own-climatology dispersion evaluation failed with exit code $status."
    exit "$status"
fi

echo "✅ Outputs written under $OUTPUT_DIR"
echo "🏁 Own-climatology dispersion job finished at $(date)"
