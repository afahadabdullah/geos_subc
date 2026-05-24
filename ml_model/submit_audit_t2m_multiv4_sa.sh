#!/bin/bash
#SBATCH -J SA_t2m_audit
#SBATCH -o ml_output_flowmulti_v4_south_asia_global_context/t2m_audit_%j.log
#SBATCH -e ml_output_flowmulti_v4_south_asia_global_context/t2m_audit_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

if [ ! -f "$CONDA_DIR/bin/activate" ]; then
    echo "Missing $CONDA_DIR/bin/activate. Run: bash setup_env.sh"
    exit 1
fi
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Missing conda env at $CONDA_ENV_PATH. Run: bash setup_env.sh"
    exit 1
fi

source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
export PYTHONUNBUFFERED=1
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

mkdir -p ml_output_flowmulti_v4_south_asia_global_context/t2m_audit

CONFIG_PATH="${SA_T2M_AUDIT_CONFIG:-ml_model/config_flow_multiv4.yaml}"
YEAR="${SA_T2M_AUDIT_YEAR:-2022}"
BATCH_LIMIT="${SA_T2M_AUDIT_BATCH_LIMIT:-0}"
FULL_YEAR="${SA_T2M_AUDIT_FULL_YEAR:-1}"

SAMPLE_ARGS=()
if [ "$FULL_YEAR" = "1" ] || [ "$FULL_YEAR" = "true" ] || [ "$FULL_YEAR" = "TRUE" ]; then
    SAMPLE_ARGS+=(--full-year)
fi

echo "T2M audit started at $(date) on $(hostname)"
echo "Config: $CONFIG_PATH"
echo "Year: $YEAR"
echo "Batch limit: $BATCH_LIMIT"
echo "Sampling: $([ ${#SAMPLE_ARGS[@]} -gt 0 ] && echo full weekly year || echo monthly subset)"
echo "Data dir: $DATA_DIR_OVERRIDE"

python ml_model/audit_t2m_multiv4_sa.py \
    --config "$CONFIG_PATH" \
    --year "$YEAR" \
    --batch-limit "$BATCH_LIMIT" \
    "${SAMPLE_ARGS[@]}"

echo "T2M audit finished at $(date)"
