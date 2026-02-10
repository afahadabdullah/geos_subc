#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
CONDA_DIR="$PROJECT_DIR/miniconda"
ENV_NAME="geossub_env"

echo "Setting up Miniconda environment in $PROJECT_DIR"
echo "Architecture: aarch64 (for TACC Vista / ARM64)"

# 1. Download and Install Miniconda if not present
if [ ! -d "$CONDA_DIR" ]; then
    echo "Downloading Miniconda (aarch64)..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    echo "Installing Miniconda to $CONDA_DIR..."
    bash Miniconda3-latest-Linux-aarch64.sh -b -p "$CONDA_DIR"
    rm Miniconda3-latest-Linux-aarch64.sh
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

# 2. Initialize Conda for this script session
if [ -f "$CONDA_DIR/bin/activate" ]; then
    source "$CONDA_DIR/bin/activate"
else
    echo "Error: Miniconda activation script not found at $CONDA_DIR/bin/activate"
    exit 1
fi

# 3. Configure Mamba solver
echo "Configuring Mamba solver..."
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# 4. Create/Update environment (excluding PyTorch)
echo "Updating environment core from environment.yml..."
conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune

# 5. Install CUDA-enabled PyTorch for ARM64 (TACC Vista Specific)
echo "Installing/Verifying CUDA-enabled PyTorch for ARM64..."
source "$CONDA_DIR/bin/activate" "$ENV_NAME"

# Uninstall existing (potentially CPU-only) torch to ensure clean install
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install from PyTorch HTML index (Using cu121 which is stable for Grace Hopper)
echo "Downloading and installing ARM64 CUDA wheels..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Environment setup complete."
echo "To activate, run: source $CONDA_DIR/bin/activate $ENV_NAME"
# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torch version: {torch.__version__}')"
