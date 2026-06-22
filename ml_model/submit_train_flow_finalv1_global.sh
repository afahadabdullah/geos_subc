#!/bin/bash
#SBATCH -J GLOBAL_finalv1
#SBATCH -o ml_output_flow_finalv1_global_noisectx_t2mres/flow_%j.log
#SBATCH -e ml_output_flow_finalv1_global_noisectx_t2mres/flow_%j.log
#SBATCH -p gh-dev                    # Queue (partition) name
#SBATCH -N 1                         # Total # of nodes
#SBATCH -n 1                         # Total # of tasks
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH -A ATM25008                  # Project account
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -eo pipefail

echo "🚀 Full-global flow_finalv1_global residual-T2M training job started at $(date) on $(hostname)"

# Move to Scratch storage before activating the repo-local environment.
PROJECT_DIR="/scratch/11353/afahad/geossub/geos_subc"
cd "$PROJECT_DIR" || exit 1

# Environment Setup: use the repo-local conda created by setup_env.sh.
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

# Fix for "CXXABI_1.3.15 not found" Matplotlib error on TACC
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONUNBUFFERED=1
export SYMPY_GROUND_TYPES=python
export DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-/scratch/11353/afahad/geossub/dataprocess}"

# Optional cleanup for stale processes from previous failed runs.
# Disabled by default because broad pkill patterns can stop unrelated jobs.
if [ "${CLEANUP_OLD_PROCS:-0}" = "1" ]; then
    echo "🧹 Cleaning up previous global flow processes..."
    pkill -9 -u "$USER" -f "train_flow_finalv1_global.py" || true
    pkill -9 -u "$USER" -f "accelerate.*train_flow_finalv1_global.py" || true
    sleep 3
else
    echo "🧹 Skipping process cleanup. Set CLEANUP_OLD_PROCS=1 to enable it."
fi

echo "📌 Using checked-out code at $(git rev-parse --short HEAD) on branch $(git branch --show-current)"

CONFIG_PATH="ml_model/config_flow_finalv1_global.yaml"
CONFIG_OUTPUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH'))['output_dir'])")
OUTPUT_DIR="$CONFIG_OUTPUT_DIR"
CKPT_FILE="$OUTPUT_DIR/latest_flow_ckpt.pt"
EARLY_STOP_FILE="$OUTPUT_DIR/EARLY_STOPPED"
CONFIG_TARGET_DOMAIN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain'))")
CONFIG_GLOBAL_CONTEXT=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('global_context_variables', [])))")
CONFIG_LOCAL_VARS=$(python -c "import yaml; print(','.join(yaml.safe_load(open('$CONFIG_PATH')).get('local_obs_variables', [])))")
STATS_FILENAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('stats_file', 'flow_finalv1_global_stats.pt'))")
CONFIG_T2M_MODE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_PATH')).get('t2m_target_mode'))")
CONFIG_HAS_BOUNDS=$(python -c "import yaml; print(bool(yaml.safe_load(open('$CONFIG_PATH')).get('target_domain_bounds')))")
CONFIG_RHO_PR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('validation_rho_pr'))")
CONFIG_RHO_T2M=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('validation_rho_t2m'))")
CONFIG_BETA_PR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('validation_var_beta_pr'))")
CONFIG_BETA_T2M=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('validation_var_beta_t2m'))")
CONFIG_COARSE=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('validation_variance_coarse_kernel'))")
CONFIG_ELEVATION_FILE=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('elevation_file') or '')")
CONFIG_GEOS_STATS=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(','.join(c.get('geos_condition_statistics', [])))")
CONFIG_EXTERNAL_INIT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); print(c.get('finetune_init_checkpoint') or c.get('initial_checkpoint') or '')")
if [ "$CONFIG_OUTPUT_DIR" != "ml_output_flow_finalv1_global_noisectx_t2mres" ] || [ "$CONFIG_TARGET_DOMAIN" != "global" ]; then
    echo "❌ Refusing to train unexpected config: output_dir=$CONFIG_OUTPUT_DIR target_domain=$CONFIG_TARGET_DOMAIN"
    exit 1
fi
if [ "$CONFIG_HAS_BOUNDS" != "False" ]; then
    echo "❌ Full-global release must not define target_domain_bounds."
    exit 1
fi
if [ "$CONFIG_T2M_MODE" != "geos_residual" ]; then
    echo "❌ Refusing to train flow_finalv1_global without residual T2M mode: t2m_target_mode=$CONFIG_T2M_MODE"
    exit 1
fi
if [ "$CONFIG_LOCAL_VARS" != "sm,mjo" ]; then
    echo "❌ Refusing to train unexpected local predictors: $CONFIG_LOCAL_VARS"
    exit 1
fi
if [ "$CONFIG_GLOBAL_CONTEXT" != "sst,sss,ivt,z500_zonal_dev,u250" ]; then
    echo "❌ Refusing to train unexpected global context variables: $CONFIG_GLOBAL_CONTEXT"
    exit 1
fi
if [ "$CONFIG_GEOS_STATS" != "mean,std,q10,q90,member_count" ]; then
    echo "❌ Refusing unexpected flow_finalv1_global GEOS statistics: $CONFIG_GEOS_STATS"
    exit 1
fi
if [ -n "$CONFIG_EXTERNAL_INIT" ]; then
    echo "❌ flow_finalv1_global must train from scratch; external initialization is configured: $CONFIG_EXTERNAL_INIT"
    exit 1
fi
if [ -z "$CONFIG_ELEVATION_FILE" ] || [ ! -f "$CONFIG_ELEVATION_FILE" ]; then
    echo "❌ Required flow_finalv1_global elevation file is missing: $CONFIG_ELEVATION_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

MAX_EPOCHS=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH'))['epochs']))" 2>/dev/null || echo "500")
MIXED_PRECISION=$(python -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))" 2>/dev/null || echo "no")
echo "🎯 Max epochs from $CONFIG_PATH: $MAX_EPOCHS"
echo "🎯 Mixed precision from $CONFIG_PATH: $MIXED_PRECISION"
echo "🎯 Output dir from $CONFIG_PATH: $OUTPUT_DIR"
echo "🎯 Target domain from $CONFIG_PATH: $CONFIG_TARGET_DOMAIN"
echo "🎯 Target grid: full global 181x360 (90S-90N, 0E-359E)"
echo "🎯 T2M target mode: $CONFIG_T2M_MODE"
echo "🎯 Local predictors: $CONFIG_LOCAL_VARS"
echo "🎯 Global context: $CONFIG_GLOBAL_CONTEXT"
echo "🎯 GEOS conditioning statistics: $CONFIG_GEOS_STATS"
echo "🎯 Initialization: scratch (or resume this flow_finalv1_global output directory only)"
echo "🎯 Static elevation: $CONFIG_ELEVATION_FILE"
echo "🎯 Variance inference/fine-tune rho: PR=$CONFIG_RHO_PR T2M=$CONFIG_RHO_T2M"
echo "🎯 Variance inference/fine-tune beta: PR=$CONFIG_BETA_PR T2M=$CONFIG_BETA_T2M coarse=$CONFIG_COARSE"
echo "🎯 Data dir override: $DATA_DIR_OVERRIDE"
echo "🎯 Stats file: ml_model/$STATS_FILENAME"

# Fail-fast topography preflight before dataset preload or model construction.
echo "🗺️ Running GLDAS topography preflight..."
python ml_model/inspect_topography_flow_finalv1_global.py \
    --file "$CONFIG_ELEVATION_FILE" \
    --variable "GLDAS_elevation" \
    --output "$OUTPUT_DIR/topography_flow_finalv1_global_preflight.png"

# Run Global Stats calculation (only if stats file doesn't exist)
STATS_PATH="ml_model/$STATS_FILENAME"
if [ -f "$STATS_PATH" ]; then
    if ! python - "$STATS_PATH" <<'PY'
import math
import sys
import torch

stats = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
bad = []
for key, value in stats.items():
    if isinstance(value, dict) and "min" in value and "max" in value:
        vmin, vmax = float(value["min"]), float(value["max"])
        if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin <= vmax):
            bad.append(f"{key}: {vmin}..{vmax}")
if "target_t2m_residual_raw" not in stats:
    bad.append("missing target_t2m_residual_raw")
if bad:
    print("Invalid stats bounds: " + "; ".join(bad))
    sys.exit(1)
PY
    then
        echo "⚠️ Existing stats file is invalid. Removing $STATS_PATH before recomputing."
        rm -f "$STATS_PATH"
    fi
fi

if [ ! -f "$STATS_PATH" ]; then
    echo "📊 Stats file not found. Computing flow_finalv1_global statistics from $DATA_DIR_OVERRIDE..."
    python ml_model/calculate_global_stats_flow_finalv1_global.py \
        --data_root "$DATA_DIR_OVERRIDE" \
        --out "$STATS_PATH" \
        --target_domain "$CONFIG_TARGET_DOMAIN"
else
    echo "✅ flow_finalv1_global stats found at $STATS_PATH. Skipping recalculation."
fi

# Fail before the expensive multi-year preload if the copied target path or
# release target path and ensemble-statistic layout diverge from the reference.
echo "🧪 Verifying flow_finalv1_global targets and GEOS ensemble-statistic conditioning..."
python ml_model/inspect_geos_conditioning_flow_finalv1_global.py \
    --config "$CONFIG_PATH" \
    --year "$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH'))['train_start_year']))")"

if [ -f "$EARLY_STOP_FILE" ]; then
    echo "✅ Early-stop marker found. Not training or resubmitting."
    cat "$EARLY_STOP_FILE"
    echo "🏁 Job finished at $(date)"
    exit 0
fi

if [ -f "$CKPT_FILE" ]; then
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(int(ckpt.get('epoch', -1)))" 2>/dev/null || echo "-1")
    if [ "$CURRENT_EPOCH" -ge "$MAX_EPOCHS" ]; then
        echo "✅ Checkpoint already at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Skipping training and resubmission."
        echo "🏁 Job finished at $(date)"
        exit 0
    fi
else
    CURRENT_EPOCH=-1
fi

# Run 30 epochs per allocation.
# EPOCHS_PER_RUN remains available as an explicit one-off override.
EPOCHS_PER_RUN="${EPOCHS_PER_RUN:-30}"
echo "🎯 Epochs in this allocation: $EPOCHS_PER_RUN (resume checkpoint epoch: $CURRENT_EPOCH)"

echo "🔥 Launching flow_finalv1_global training (global target, target-grid/global predictors, residual T2M)..."
accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" ml_model/train_flow_finalv1_global.py --config "$CONFIG_PATH" \
    --epochs-per-run "$EPOCHS_PER_RUN"

# --- AUTOMATIC JOB CHAINING ---
if [ -f "$EARLY_STOP_FILE" ]; then
    echo "✅ Early-stop marker found. Not resubmitting."
    cat "$EARLY_STOP_FILE"
    echo "🏁 Job finished at $(date)"
    exit 0
fi

echo "🔄 Checking if we need to resubmit..."
echo "📍 Current Host: $(hostname)"
echo "📍 Submit Host: $SLURM_SUBMIT_HOST"
echo "📍 Working Dir: $PWD"

if [ -f "$CKPT_FILE" ]; then
    # Safely get current epoch from checkpoint
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(ckpt['epoch'])" 2>/dev/null || echo "-1")

    if [ "$CURRENT_EPOCH" -lt "$MAX_EPOCHS" ]; then
        echo "📍 Training at Epoch $CURRENT_EPOCH / $MAX_EPOCHS. Resubmitting job..."
        # TACC Fix: Resubmit via SSH to the submission host (full domain)
        SUBMIT_TARGET="${SLURM_SUBMIT_HOST}.vista.tacc.utexas.edu"
        if [[ "$SLURM_SUBMIT_HOST" == *"vista"* ]]; then
            SUBMIT_TARGET="$SLURM_SUBMIT_HOST"
        fi

        echo "📡 Attempting SSH resubmission to $SUBMIT_TARGET..."
        ssh -o StrictHostKeyChecking=no "$SUBMIT_TARGET" \
            "cd $PWD && sbatch ml_model/submit_train_flow_finalv1_global.sh"
    else
        echo "✅ Final Epoch $CURRENT_EPOCH reached. Chaining complete."
    fi
else
    echo "⚠️ Checkpoint not found. Chaining stopped to prevent infinite loops."
fi

echo "🏁 Full-global flow_finalv1_global residual-T2M training job finished at $(date)"
