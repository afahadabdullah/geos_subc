#!/bin/bash
# cleanup_gpu.sh - Surgical removal of zombie training processes

echo "Scanning for stale python/accelerate processes..."
PIDS=$(pgrep -u $USER -f "train.py")

if [ -z "$PIDS" ]; then
    echo "No stale training processes found for user $USER."
else
    echo "Found PIDs: $PIDS"
    echo "Killing processes..."
    kill -9 $PIDS
    echo "Done."
fi

echo "GPU Status after cleanup:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
