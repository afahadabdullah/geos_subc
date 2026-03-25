#!/bin/bash
#SBATCH -J ivt_arco_2325
#SBATCH -o dataprocess/logs/ivt_arco_2023_2025_%j.log
#SBATCH -e dataprocess/logs/ivt_arco_2023_2025_%j.log
#SBATCH -p gg
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -A ATM25008

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_DIR="${CONDA_DIR:-/home1/11353/afahad/afahad/geossub/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"

echo "Job started at $(date) on $(hostname)"

source ~/.bashrc || true

if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV_NAME"
elif [ -f "$CONDA_DIR/bin/activate" ]; then
    source "$CONDA_DIR/bin/activate" "$CONDA_ENV_NAME"
else
    echo "Conda environment setup failed. 'conda' was not found and fallback activate script is missing at $CONDA_DIR/bin/activate"
    exit 1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
mkdir -p dataprocess/logs dataprocess/era5_ivt

echo "[$(date)] Starting IVT extract for 2023-2025"
python3 dataprocess/extract_era5_ivt.py \
    --start_year 2023 \
    --end_year 2025 \
    --output_dir dataprocess/era5_ivt \
    --overwrite

echo "[$(date)] Starting IVT weekly processing for 2023-2025"
python3 dataprocess/process_ivt.py \
    --years 2023 2024 2025 \
    --daily_dir dataprocess/era5_ivt \
    --output_dir dataprocess \
    --overwrite

echo "[$(date)] IVT pipeline complete"
