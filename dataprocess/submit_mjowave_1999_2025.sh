#!/bin/bash
#SBATCH -J mjowave_9925
#SBATCH -o dataprocess/logs/mjowave_1999_2025_%j.log
#SBATCH -e dataprocess/logs/mjowave_1999_2025_%j.log
#SBATCH -p gg
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -A ATM25008

set -eo pipefail

CONDA_DIR="${CONDA_DIR:-/home1/11353/afahad/afahad/geossub/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
REPO_ROOT="${REPO_ROOT:-/scratch/11353/afahad/geossub/geos_subc}"

echo "Job started at $(date) on $(hostname)"

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV_NAME"
elif [ -f "$CONDA_DIR/bin/activate" ]; then
    source "$CONDA_DIR/bin/activate" "$CONDA_ENV_NAME"
else
    echo "Conda environment setup failed. 'conda' was not found and fallback activate script is missing at $CONDA_DIR/bin/activate"
    exit 1
fi

set -u

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT" || {
    echo "Repo root not found or not accessible: $REPO_ROOT"
    exit 1
}

mkdir -p dataprocess/logs

echo "[$(date)] Starting MJO wave extraction for 1999-2025"
python3 dataprocess/extract_mjo_wave.py \
    --start_year 1999 \
    --end_year 2025 \
    --olr_glob "dataprocess/olr/OLR-Daily_v02r00*.nc" \
    --output_dir dataprocess \
    --overwrite

echo "[$(date)] Starting MJO wave weekly processing for 1999-2025"
python3 dataprocess/process_mjowave.py \
    --start_year 1999 \
    --end_year 2025 \
    --overwrite

echo "[$(date)] MJO wave pipeline complete"
