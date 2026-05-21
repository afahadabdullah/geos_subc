#!/usr/bin/env bash
#SBATCH -J flow_multi_v4_resume
#SBATCH -o ml_output_flowmulti_v4/resume_%j.log
#SBATCH -e ml_output_flowmulti_v4/resume_%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ATM25008
#SBATCH --mail-type=all
#SBATCH --mail-user=a.fahad@nasa.gov

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/11353/afahad/geossub/geos_subc_clean}"
CONDA_DIR="${CONDA_DIR:-$PROJECT_DIR/miniconda}"
ENV_NAME="${ENV_NAME:-geossub_env}"
CONFIG_PATH="${CONFIG_PATH:-ml_model/config_flow_multiv4.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-ml_output_flowmulti_v4}"
EPOCHS_PER_RUN="${EPOCHS_PER_RUN:-1}"
HOME_STATS="${HOME_STATS:-/home1/11353/afahad/geos_subc/ml_model/v1_multi_global_stats.pt}"

echo "Resume job started: $(date) on $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Conda:   $CONDA_DIR"
echo "Epochs this run: $EPOCHS_PER_RUN"

cd "$PROJECT_DIR" || exit 1

mkdir -p "$OUTPUT_DIR"

archive_on_exit() {
    local status=$?
    echo "Job exit status: $status"
    if [ -f scripts/archive_scratch_to_work.sh ]; then
        echo "Archiving scratch outputs to WORK..."
        bash scripts/archive_scratch_to_work.sh || true
    else
        echo "Archive script not found; skipping archive."
    fi
    echo "Resume job finished: $(date)"
    exit "$status"
}
trap archive_on_exit EXIT

if [ ! -f "$CONDA_DIR/bin/activate" ]; then
    echo "ERROR: Conda activate script missing: $CONDA_DIR/bin/activate" >&2
    exit 1
fi

source "$CONDA_DIR/bin/activate" "$ENV_NAME"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "CONDA_PREFIX=$CONDA_PREFIX"
which python
which accelerate
python -c "import sys, torch; print(sys.executable); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Missing config: $CONFIG_PATH" >&2
    exit 1
fi

STATS_PATH="ml_model/v1_multi_global_stats.pt"
if [ ! -e "$STATS_PATH" ]; then
    if [ -f "$HOME_STATS" ]; then
        echo "Linking stats file from $HOME_STATS"
        ln -s "$HOME_STATS" "$STATS_PATH"
    else
        echo "ERROR: Missing stats file: $STATS_PATH and $HOME_STATS" >&2
        exit 1
    fi
fi

python -c "import torch; d=torch.load('$STATS_PATH', weights_only=True); print('stats keys', sorted(d.keys()))"

CKPT_FILE="$OUTPUT_DIR/latest_flow_ckpt.pt"
MAX_EPOCHS=$(python -c "import yaml; print(int(yaml.safe_load(open('$CONFIG_PATH'))['epochs']))")
MIXED_PRECISION=$(python -c "import yaml; print(str(yaml.safe_load(open('$CONFIG_PATH')).get('mixed_precision', 'no')))")

echo "Max epochs: $MAX_EPOCHS"
echo "Mixed precision: $MIXED_PRECISION"

if [ -f "$CKPT_FILE" ]; then
    CURRENT_EPOCH=$(python -c "import torch; ckpt=torch.load('$CKPT_FILE', map_location='cpu', weights_only=True); print(int(ckpt.get('epoch', -1)))")
    echo "Resuming from checkpoint: $CKPT_FILE"
    echo "Current checkpoint epoch: $CURRENT_EPOCH / $MAX_EPOCHS"
else
    echo "No latest checkpoint found at $CKPT_FILE; training will start from scratch."
fi

accelerate launch --num_processes 1 --mixed_precision "$MIXED_PRECISION" \
    ml_model/train_flow_multiv4.py \
    --config "$CONFIG_PATH" \
    --epochs-per-run "$EPOCHS_PER_RUN"
