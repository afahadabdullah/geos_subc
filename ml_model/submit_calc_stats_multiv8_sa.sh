#!/bin/bash
#SBATCH -J SA_stats_v8
#SBATCH -o ml_output_flowmulti_v8_sa_55e100e_0n40n_t2mres/stats_%j.log
#SBATCH -e ml_output_flowmulti_v8_sa_55e100e_0n40n_t2mres/stats_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "South Asia v8 stats job started at $(date) on $(hostname)"

PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

echo "Using conda dir: $CONDA_DIR"
if [ ! -f "$CONDA_DIR/bin/activate" ]; then
    echo "Missing $CONDA_DIR/bin/activate. Run: bash setup_env.sh"
    exit 1
fi
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Missing conda env at $CONDA_ENV_PATH. Run: bash setup_env.sh"
    exit 1
fi

source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
echo "Conda environment active: ${CONDA_PREFIX:-unset}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export SYMPY_GROUND_TYPES=python
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

CONFIG_PATH="${SA_STATS_CONFIG:-ml_model/config_flow_multiv8.yaml}"
OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
STATS_FILENAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('stats_file'))")
TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
DOMAIN_LABEL=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('label'))")
LAT_MIN=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lat_min'))")
LAT_MAX=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lat_max'))")
LON_MIN=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lon_min'))")
LON_MAX=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lon_max'))")
TRAIN_START_YEAR=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH')).get('train_start_year', 1999)))")
TRAIN_END_YEAR=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH')).get('train_end_year', 2020)))")

EXPECTED_OUTPUT_DIR="ml_output_flowmulti_v8_sa_55e100e_0n40n_t2mres"
EXPECTED_STATS_FILE="v8_sa_55e100e_0n40n_global_local_stats.pt"

if [ "$OUTPUT_DIR" != "$EXPECTED_OUTPUT_DIR" ]; then
    echo "Refusing unexpected output_dir=$OUTPUT_DIR"
    exit 1
fi
if [ "$STATS_FILENAME" != "$EXPECTED_STATS_FILE" ]; then
    echo "Refusing unexpected stats_file=$STATS_FILENAME"
    exit 1
fi
if [ "$TARGET_DOMAIN" != "south_asia" ]; then
    echo "Refusing unexpected target_domain=$TARGET_DOMAIN"
    exit 1
fi
if [ "$LAT_MIN" != "0.0" ] || [ "$LAT_MAX" != "40.0" ] || [ "$LON_MIN" != "55.0" ] || [ "$LON_MAX" != "100.0" ]; then
    echo "Refusing unexpected target bounds: lat=$LAT_MIN..$LAT_MAX lon=$LON_MIN..$LON_MAX"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
STATS_PATH="ml_model/$STATS_FILENAME"

echo "Code: $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Stats path: $STATS_PATH"
echo "Data dir: $DATA_DIR_OVERRIDE"
echo "Years: $TRAIN_START_YEAR-$TRAIN_END_YEAR"
echo "Target bounds: $DOMAIN_LABEL lat=$LAT_MIN..$LAT_MAX lon=$LON_MIN..$LON_MAX"

if [ -f "$STATS_PATH" ] && [ "${FORCE_RECREATE_STATS:-0}" != "1" ]; then
    echo "Stats file already exists. Set FORCE_RECREATE_STATS=1 to rebuild:"
    ls -lh "$STATS_PATH"
    echo "Job finished at $(date)"
    exit 0
fi

if [ -f "$STATS_PATH" ] && [ "${FORCE_RECREATE_STATS:-0}" = "1" ]; then
    BACKUP_PATH="${STATS_PATH}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up existing stats to $BACKUP_PATH"
    mv "$STATS_PATH" "$BACKUP_PATH"
fi

python ml_model/calculate_global_local_stats_multi_v8.py \
    --data_root "$DATA_DIR_OVERRIDE" \
    --out "$STATS_PATH" \
    --start_year "$TRAIN_START_YEAR" \
    --end_year "$TRAIN_END_YEAR" \
    --target_domain "$TARGET_DOMAIN" \
    --domain_label "$DOMAIN_LABEL" \
    --lat_min "$LAT_MIN" \
    --lat_max "$LAT_MAX" \
    --lon_min "$LON_MIN" \
    --lon_max "$LON_MAX"

echo "Verifying stats file..."
python - "$STATS_PATH" <<'PY'
import math
import sys
import torch

stats = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
required = [
    "sst",
    "sss",
    "sm",
    "ivt",
    "u250",
    "z500_zonal_dev",
    "mjo",
    "geos_pr_raw",
    "geos_tas_raw",
    "target_t2m_raw",
    "target_t2m_residual_raw",
]
bad = []
for key in required:
    value = stats.get(key)
    if not isinstance(value, dict):
        bad.append(f"{key}: missing")
        continue
    vmin = float(value.get("min", float("nan")))
    vmax = float(value.get("max", float("nan")))
    if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin < vmax):
        bad.append(f"{key}: {vmin}..{vmax}")
if bad:
    raise SystemExit("Invalid stats: " + "; ".join(bad))
print("Verified keys:", ", ".join(required))
print("T2M residual bounds:", stats["target_t2m_residual_raw"])
PY

ls -lh "$STATS_PATH"
echo "South Asia v8 stats job finished at $(date)"
