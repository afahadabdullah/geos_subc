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
echo "This will download ~2GB of data, please wait..."
source "$CONDA_DIR/bin/activate" "$ENV_NAME"

# Try to load TACC modules (only works on compute nodes)
module load gcc cuda 2>/dev/null || echo "Note: 'module load' skipped (likely on login node)"

# Aggressively remove gmpy2 and other conflicting libraries
echo "Cleaning up conflicting dependencies..."
conda remove --force -y gmpy2 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio sympy gmpy2 2>/dev/null || true

# Physically delete gmpy2 folders if they persist (Sympy bug workaround)
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site packages at: $SITE_PACKAGES"
rm -rf "$SITE_PACKAGES/gmpy2"* "$SITE_PACKAGES/sympy"*

# Install from PyTorch HTML index (Using cu124 which is much newer for Grace Hopper)
echo "Downloading and installing ARM64 CUDA wheels (cu124)..."
pip install --progress-bar on torch torchvision torchaudio sympy --index-url https://download.pytorch.org/whl/cu124

# Force Sympy to ignore gmpy2 even if it sneaks back in
export SYMPY_GROUND_TYPES=python

echo "Environment setup complete."
echo "To activate, run: source $CONDA_DIR/bin/activate $ENV_NAME && export SYMPY_GROUND_TYPES=python"
# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torch version: {torch.__version__}')"
