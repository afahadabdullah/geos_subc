#!/usr/bin/env bash
set -Eeuo pipefail

# Archive important scratch/nobackup artifacts to TACC $WORK.
#
# This is intentionally a copy/sync script, not a "touch files to reset purge"
# script. TACC scratch is temporary; keep durable outputs in $WORK or Ranch.
#
# Typical use on Vista:
#   cd /scratch/11353/afahad/geossub/geos_subc_clean
#   bash scripts/archive_scratch_to_work.sh
#
# Optional overrides:
#   SRC_ROOT=/scratch/11353/afahad/geossub/geos_subc_clean \
#   DEST_ROOT=$WORK/geos_subc \
#   bash scripts/archive_scratch_to_work.sh

SRC_ROOT="${SRC_ROOT:-$(pwd)}"
DEST_ROOT="${DEST_ROOT:-${WORK:?WORK is not set}/geos_subc}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required but was not found." >&2
    exit 1
fi

mkdir -p "$DEST_ROOT"/{logs,ml_model,git_snapshots}

echo "Archive started: $(date)"
echo "Source:      $SRC_ROOT"
echo "Destination: $DEST_ROOT"

sync_dir() {
    local rel_path="$1"
    if [ -d "$SRC_ROOT/$rel_path" ]; then
        mkdir -p "$DEST_ROOT/$rel_path"
        echo "Syncing directory: $rel_path"
        rsync -a --human-readable --stats "$SRC_ROOT/$rel_path/" "$DEST_ROOT/$rel_path/"
    else
        echo "Skipping missing directory: $rel_path"
    fi
}

sync_file() {
    local rel_path="$1"
    if [ -f "$SRC_ROOT/$rel_path" ]; then
        mkdir -p "$DEST_ROOT/$(dirname "$rel_path")"
        echo "Syncing file: $rel_path"
        rsync -a --human-readable "$SRC_ROOT/$rel_path" "$DEST_ROOT/$rel_path"
    else
        echo "Skipping missing file: $rel_path"
    fi
}

# Model outputs/checkpoints/logs. Add more ml_output dirs here if needed.
for output_dir in "$SRC_ROOT"/ml_output*; do
    [ -d "$output_dir" ] || continue
    sync_dir "$(basename "$output_dir")"
done

# Small but critical reproducibility files.
sync_file environment.yml
sync_file setup_env.sh
sync_file ml_model/v1_multi_global_stats.pt
sync_file ml_model/config_flow_multiv1.yaml
sync_file ml_model/config_flow_multiv3.yaml
sync_file ml_model/config_flow_multiv4.yaml
sync_file ml_model/submit_train_flowmultiv3.sh
sync_file ml_model/submit_train_flowmultiv4.sh
sync_file ml_model/train_flow_multiv3.py
sync_file ml_model/train_flow_multiv4.py
sync_file ml_model/flow_matching_multi_v3.py
sync_file ml_model/flow_matching_multi_v4.py
sync_file ml_model/dataset_flow_multi.py
sync_file ml_model/noise_utils.py
sync_file ml_model/noise_utils_multi.py

# Preserve lightweight git state when this is a real repository.
if git -C "$SRC_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Writing git status snapshot."
    git -C "$SRC_ROOT" status --short > "$DEST_ROOT/git_snapshots/status_${RUN_TAG}.txt" || true
    git -C "$SRC_ROOT" log --oneline --decorate -20 > "$DEST_ROOT/git_snapshots/log_${RUN_TAG}.txt" || true
    git -C "$SRC_ROOT" rev-parse HEAD > "$DEST_ROOT/git_snapshots/head_${RUN_TAG}.txt" || true

    echo "Creating git bundle snapshot."
    git -C "$SRC_ROOT" bundle create "$DEST_ROOT/git_snapshots/geos_subc_${RUN_TAG}.bundle" --all || true
fi

echo "Archive complete: $(date)"
echo "Durable copy is under: $DEST_ROOT"
