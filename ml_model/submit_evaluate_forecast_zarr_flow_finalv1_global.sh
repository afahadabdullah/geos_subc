#!/bin/bash
#SBATCH -J GLOBAL_eval_finalv1
#SBATCH -o ml_output_flow_finalv1_global_noisectx_t2mres/evaluate_zarr_%j.log
#SBATCH -e ml_output_flow_finalv1_global_noisectx_t2mres/evaluate_zarr_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 Global flow_finalv1 forecast Zarr evaluation started at $(date) on $(hostname)"

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

FORECAST_DIR="${GLOBAL_EVAL_FORECAST_DIR:-dataprocess/gen_flow_finalv1_global_junjul_2021_2024_e90_s50}"
START_YEAR="${GLOBAL_EVAL_START_YEAR:-2021}"
END_YEAR="${GLOBAL_EVAL_END_YEAR:-2024}"
SKIP_YEARS="${GLOBAL_EVAL_SKIP_YEARS:-}"
OUT_DIR="${GLOBAL_EVAL_OUT_DIR:-ml_output_flow_finalv1_global_noisectx_t2mres/zarr_eval_global_2021_2024_e90_s50}"
VARIABLES="${GLOBAL_EVAL_VARIABLES:-pr,t2m}"
MAX_RUNTIME_MINUTES="${GLOBAL_EVAL_MAX_RUNTIME_MINUTES:-105}"
AUTO_RESUBMIT="${GLOBAL_EVAL_AUTO_RESUBMIT:-1}"
OVERWRITE="${GLOBAL_EVAL_OVERWRITE:-0}"
MAKE_PLOTS="${GLOBAL_EVAL_MAKE_PLOTS:-1}"

if [ ! -d "$FORECAST_DIR" ]; then
    echo "❌ Forecast dir not found: $FORECAST_DIR"
    exit 1
fi

EXTRA_ARGS=()
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ] || [ "$OVERWRITE" = "TRUE" ]; then
    EXTRA_ARGS+=(--overwrite)
fi
if [ "$MAKE_PLOTS" = "1" ] || [ "$MAKE_PLOTS" = "true" ] || [ "$MAKE_PLOTS" = "TRUE" ]; then
    EXTRA_ARGS+=(--make_plots)
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"
echo "🎯 Forecast dir: $FORECAST_DIR"
echo "🎯 Years: $START_YEAR-$END_YEAR"
echo "🎯 Skip years: $SKIP_YEARS"
echo "🎯 Variables: $VARIABLES"
echo "🎯 Output dir: $OUT_DIR"
echo "🎯 Soft runtime minutes: $MAX_RUNTIME_MINUTES"
echo "🎯 Auto resubmit: $AUTO_RESUBMIT"
echo "🎯 Overwrite yearly metrics: $OVERWRITE"
echo "🎯 Make plots: $MAKE_PLOTS"

python ml_model/evaluate_forecast_zarr_flow_finalv1_global.py \
    --forecast_dir "$FORECAST_DIR" \
    --start_year "$START_YEAR" \
    --end_year "$END_YEAR" \
    --skip_years "$SKIP_YEARS" \
    --out_dir "$OUT_DIR" \
    --variables "$VARIABLES" \
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
    detail_path = os.path.join(out_dir, "yearly_metrics", f"{year}_per_init_lead_metrics.csv")
    state_path = os.path.join(out_dir, "yearly_metrics", f"{year}_direct_metric_state.csv")
    if not (os.path.isfile(detail_path) and os.path.isfile(state_path)):
        missing.append(str(year))
print(",".join(missing))
PY
)

SUMMARY_PATH="$OUT_DIR/summary_metrics.csv"
SPATIAL_MAP_PATH="$OUT_DIR/spatial_metric_maps.nc"
ORIENTATION_REPORT_PATH="$OUT_DIR/plots/orientation/orientation_report.json"
SPATIAL_MISSING=0
if [ "$MAKE_PLOTS" = "1" ] || [ "$MAKE_PLOTS" = "true" ] || [ "$MAKE_PLOTS" = "TRUE" ]; then
    if [ ! -f "$SPATIAL_MAP_PATH" ] || [ ! -f "$ORIENTATION_REPORT_PATH" ]; then
        SPATIAL_MISSING=1
    fi
fi

if [ -n "$MISSING_YEARS" ] || [ ! -f "$SUMMARY_PATH" ] || [ "$SPATIAL_MISSING" = "1" ]; then
    if [ -n "$MISSING_YEARS" ]; then
        echo "⏸️ Missing yearly metric files: $MISSING_YEARS"
    elif [ "$SPATIAL_MISSING" = "1" ]; then
        echo "⏸️ Spatial/orientation diagnostics not complete yet: $SPATIAL_MAP_PATH / $ORIENTATION_REPORT_PATH"
    else
        echo "⏸️ Summary not found yet: $SUMMARY_PATH"
    fi

    if [ "$AUTO_RESUBMIT" = "1" ] || [ "$AUTO_RESUBMIT" = "true" ] || [ "$AUTO_RESUBMIT" = "TRUE" ]; then
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi

        echo "📡 Resubmitting evaluation job via $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" \
            "cd $PWD && GLOBAL_EVAL_FORECAST_DIR='$FORECAST_DIR' GLOBAL_EVAL_START_YEAR='$START_YEAR' GLOBAL_EVAL_END_YEAR='$END_YEAR' GLOBAL_EVAL_SKIP_YEARS='$SKIP_YEARS' GLOBAL_EVAL_OUT_DIR='$OUT_DIR' GLOBAL_EVAL_VARIABLES='$VARIABLES' GLOBAL_EVAL_MAX_RUNTIME_MINUTES='$MAX_RUNTIME_MINUTES' GLOBAL_EVAL_AUTO_RESUBMIT='$AUTO_RESUBMIT' GLOBAL_EVAL_OVERWRITE='0' GLOBAL_EVAL_MAKE_PLOTS='$MAKE_PLOTS' sbatch ml_model/submit_evaluate_forecast_zarr_flow_finalv1_global.sh"
    else
        echo "ℹ️ Auto-resubmit disabled. Rerun this script to continue."
    fi
else
    echo "✅ Evaluation complete: $SUMMARY_PATH"
fi

echo "🏁 Global flow_finalv1 forecast Zarr evaluation finished at $(date)"
