#!/bin/bash
#SBATCH -J z500u250_2325
#SBATCH -o dataprocess/logs/z500_u250_arco_2023_2025_%j.log
#SBATCH -e dataprocess/logs/z500_u250_arco_2023_2025_%j.log
#SBATCH -p gg
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -A 11353

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_DIR="${CONDA_DIR:-/home1/11353/afahad/afahad/geossub/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"

if [ -f "$CONDA_DIR/bin/activate" ]; then
    source "$CONDA_DIR/bin/activate" "$CONDA_ENV_NAME"
else
    echo "Conda activate script not found at $CONDA_DIR/bin/activate"
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p dataprocess/logs dataprocess/era5_z500_u250

echo "[$(date)] Starting Z500/U250 extract for 2023-2025"
python3 dataprocess/extract_era5.py \
    --start_year 2023 \
    --end_year 2025 \
    --output_dir dataprocess/era5_z500_u250 \
    --overwrite

echo "[$(date)] Starting Z500/U250 weekly processing for 2023-2025"
python3 dataprocess/process_z500_u250.py \
    --years 2023 2024 2025 \
    --daily_dir dataprocess/era5_z500_u250 \
    --output_dir dataprocess \
    --overwrite

echo "[$(date)] Z500/U250 pipeline complete"
