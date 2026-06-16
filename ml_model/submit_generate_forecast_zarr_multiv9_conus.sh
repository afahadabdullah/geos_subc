#!/bin/bash
#SBATCH -J CONUS_zarr
#SBATCH -o ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/generate_zarr_%j.log
#SBATCH -e ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres/generate_zarr_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 CONUS v9 forecast Zarr generation started at $(date) on $(hostname)"

PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-geossub_env}"
CONDA_ENV_PATH="$CONDA_DIR/envs/$CONDA_ENV_NAME"

echo "🔧 Using conda dir: $CONDA_DIR"
if [ ! -f "$CONDA_DIR/bin/activate" ]; then
    echo "❌ Missing $CONDA_DIR/bin/activate. Run: bash setup_env.sh"
    exit 1
fi
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "❌ Missing conda env at $CONDA_ENV_PATH. Run: bash setup_env.sh"
    exit 1
fi

source "$CONDA_DIR/bin/activate" "$CONDA_ENV_PATH"
echo "✅ Conda environment active: ${CONDA_PREFIX:-unset}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export SYMPY_GROUND_TYPES=python
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

CONFIG_PATH="${CONUS_GEN_CONFIG:-ml_model/config_flow_multiv9.yaml}"
START_YEAR="${CONUS_GEN_START_YEAR:-2005}"
END_YEAR="${CONUS_GEN_END_YEAR:-2024}"
SKIP_YEARS="${CONUS_GEN_SKIP_YEARS:-2017}"
MONTHS="${CONUS_GEN_MONTHS:-6,7}"
CLIM_ENSEMBLE="${CONUS_GEN_CLIM_ENSEMBLE:-10}"
EVAL_ENSEMBLE="${CONUS_GEN_EVAL_ENSEMBLE:-100}"
EVAL_START_YEAR="${CONUS_GEN_EVAL_START_YEAR:-2021}"
NUM_STEPS="${CONUS_GEN_STEPS:-50}"
CHECKPOINT="${CONUS_GEN_CHECKPOINT:-best_flow_ckpt.pt}"
OUT_DIR="${CONUS_GEN_OUT_DIR:-dataprocess/gen_multiv9_conus_125w66w_24n50n_junjul_e10clim_e100eval_s50}"
BATCH_SIZE="${CONUS_GEN_BATCH_SIZE:-8}"
NUM_WORKERS="${CONUS_GEN_NUM_WORKERS:-1}"
ENSEMBLE_CHUNK="${CONUS_GEN_ENSEMBLE_CHUNK:-30}"
ODE_BATCH_SIZE="${CONUS_GEN_ODE_BATCH:-120}"
MEMBER_CHUNK="${CONUS_GEN_MEMBER_CHUNK:-10}"
SEED="${CONUS_GEN_SEED:-1234}"
OVERWRITE="${CONUS_GEN_OVERWRITE:-0}"
MAX_RUNTIME_MINUTES="${CONUS_GEN_MAX_RUNTIME_MINUTES:-100}"
AUTO_RESUBMIT="${CONUS_GEN_AUTO_RESUBMIT:-1}"

CONFIG_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")
CONFIG_T2M_MODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('t2m_target_mode'))")
CONFIG_DOMAIN_LABEL=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('label'))")
CONFIG_LAT_MIN=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lat_min'))")
CONFIG_LAT_MAX=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lat_max'))")
CONFIG_LON_MIN=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lon_min'))")
CONFIG_LON_MAX=$(python -c "import yaml; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); print(b.get('lon_max'))")
CONFIG_DOMAIN_SHAPE=$(python -c "import yaml, numpy as np; b=(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds') or {}); lats=np.linspace(-90.0,90.0,181); lons=np.arange(360); lat=((lats>=float(b['lat_min']))&(lats<=float(b['lat_max']))).sum(); lo=float(b['lon_min'])%360; hi=float(b['lon_max'])%360; lon=((lons>=lo)&(lons<=hi)).sum() if lo<=hi else ((lons>=lo)|(lons<=hi)).sum(); print(f'{lat}x{lon}')")

if [ "$CONFIG_OUTPUT_DIR" != "ml_output_flowmulti_v9_conus_125w66w_24n50n_noisectx_t2mres" ]; then
    echo "❌ Refusing unexpected output_dir=$CONFIG_OUTPUT_DIR"
    exit 1
fi
if [ "$CONFIG_TARGET_DOMAIN" != "conus" ]; then
    echo "❌ Refusing non-CONUS config: target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi
if [ "$CONFIG_LOCAL_VARS" != "sm,mjo" ]; then
    echo "❌ Refusing unexpected local predictors: $CONFIG_LOCAL_VARS"
    exit 1
fi
if [ "$CONFIG_GLOBAL_CONTEXT" != "sst,sss,ivt,z500_zonal_dev,u250" ]; then
    echo "❌ Refusing unexpected global context variables: $CONFIG_GLOBAL_CONTEXT"
    exit 1
fi
if [ "$CONFIG_T2M_MODE" != "geos_residual" ]; then
    echo "❌ Refusing v9 generation without residual T2M mode: t2m_target_mode=$CONFIG_T2M_MODE"
    exit 1
fi

case "$CHECKPOINT" in
    /*) CKPT_PATH="$CHECKPOINT" ;;
    *) CKPT_PATH="$CONFIG_OUTPUT_DIR/$CHECKPOINT" ;;
esac
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    exit 1
fi

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "TRUE" ]; then
    EXTRA_ARGS+=(--overwrite)
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Config: $CONFIG_PATH"
echo "🎯 Model output dir: $CONFIG_OUTPUT_DIR"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"
echo "🎯 Checkpoint: $CKPT_PATH"
echo "🎯 Selected CONUS domain: $CONFIG_DOMAIN_LABEL lat=$CONFIG_LAT_MIN..$CONFIG_LAT_MAX lon=$CONFIG_LON_MIN..$CONFIG_LON_MAX grid=$CONFIG_DOMAIN_SHAPE"
echo "🎯 Years: $START_YEAR-$END_YEAR"
echo "🎯 Skip years: $SKIP_YEARS"
echo "🎯 Init months: $MONTHS"
echo "🎯 Ensembles: $CLIM_ENSEMBLE before $EVAL_START_YEAR; $EVAL_ENSEMBLE from $EVAL_START_YEAR"
echo "🎯 ODE steps: $NUM_STEPS"
echo "🎯 Output Zarr dir: $OUT_DIR"
echo "🎯 Ensemble chunk: $ENSEMBLE_CHUNK"
echo "🎯 ODE batch: $ODE_BATCH_SIZE"
echo "🎯 Soft runtime minutes: $MAX_RUNTIME_MINUTES"
echo "🎯 Auto resubmit: $AUTO_RESUBMIT"

python ml_model/generate_forecast_zarr_multiv9_conus.py \
    --config "$CONFIG_PATH" \
    --data_dir "$DATA_DIR_OVERRIDE" \
    --model_output_dir "$CONFIG_OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --skip_years "$SKIP_YEARS" \
    --months "$MONTHS" \
    --clim_num_ensemble "$CLIM_ENSEMBLE" \
    --eval_num_ensemble "$EVAL_ENSEMBLE" \
    --eval_start_year "$EVAL_START_YEAR" \
    --num_steps "$NUM_STEPS" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --ensemble_chunk_size "$ENSEMBLE_CHUNK" \
    --ode_batch_size "$ODE_BATCH_SIZE" \
    --member_chunk "$MEMBER_CHUNK" \
    --seed "$SEED" \
    --max_runtime_minutes "$MAX_RUNTIME_MINUTES" \
    "${EXTRA_ARGS[@]}"

MISSING_YEARS=$(python - "$OUT_DIR" "$START_YEAR" "$END_YEAR" "$SKIP_YEARS" <<'PY'
import os
import sys

out_dir, start_year, end_year, skip_text = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
skip_years = {int(item.strip()) for item in skip_text.split(",") if item.strip()}
missing = []
for year in range(start_year, end_year + 1):
    if year in skip_years:
        continue
    if not os.path.isdir(os.path.join(out_dir, f"{year}.zarr")):
        missing.append(str(year))
print(",".join(missing))
PY
)

if [ -n "$MISSING_YEARS" ]; then
    echo "⏸️ Missing final yearly Zarr stores: $MISSING_YEARS"
    if [ "$AUTO_RESUBMIT" = "1" ] || [ "$AUTO_RESUBMIT" = "true" ] || [ "$AUTO_RESUBMIT" = "TRUE" ]; then
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi

        echo "📡 Resubmitting generation job via $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" \
            "cd $PWD && DATA_DIR_OVERRIDE='$DATA_DIR_OVERRIDE' CONUS_GEN_CONFIG='$CONFIG_PATH' CONUS_GEN_START_YEAR='$START_YEAR' CONUS_GEN_END_YEAR='$END_YEAR' CONUS_GEN_SKIP_YEARS='$SKIP_YEARS' CONUS_GEN_MONTHS='$MONTHS' CONUS_GEN_CLIM_ENSEMBLE='$CLIM_ENSEMBLE' CONUS_GEN_EVAL_ENSEMBLE='$EVAL_ENSEMBLE' CONUS_GEN_EVAL_START_YEAR='$EVAL_START_YEAR' CONUS_GEN_STEPS='$NUM_STEPS' CONUS_GEN_CHECKPOINT='$CHECKPOINT' CONUS_GEN_OUT_DIR='$OUT_DIR' CONUS_GEN_BATCH_SIZE='$BATCH_SIZE' CONUS_GEN_NUM_WORKERS='$NUM_WORKERS' CONUS_GEN_ENSEMBLE_CHUNK='$ENSEMBLE_CHUNK' CONUS_GEN_ODE_BATCH='$ODE_BATCH_SIZE' CONUS_GEN_MEMBER_CHUNK='$MEMBER_CHUNK' CONUS_GEN_SEED='$SEED' CONUS_GEN_OVERWRITE='0' CONUS_GEN_MAX_RUNTIME_MINUTES='$MAX_RUNTIME_MINUTES' CONUS_GEN_AUTO_RESUBMIT='$AUTO_RESUBMIT' sbatch ml_model/submit_generate_forecast_zarr_multiv9_conus.sh"
    else
        echo "ℹ️ Auto-resubmit disabled. Rerun this script to continue."
    fi
else
    echo "✅ All requested yearly Zarr stores are complete."
fi

echo "🏁 CONUS v9 forecast Zarr generation finished at $(date)"
