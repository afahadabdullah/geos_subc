#!/bin/bash

# Configuration
PROJECT_DIR="/home1/11353/afahad/afahad/geossub"
CONDA_DIR="$PROJECT_DIR/miniconda"
ENV_NAME="geossub_env"

echo "Setting up Miniconda environment in $PROJECT_DIR"

# 1. Download and Install Miniconda if not present
if [ ! -d "$CONDA_DIR" ]; then
    echo "Downloading Miniconda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    echo "Installing Miniconda to $CONDA_DIR..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
    rm Miniconda3-latest-Linux-x86_64.sh
else
    echo "Miniconda already installed at $CONDA_DIR"
fi

# 2. Initialize Conda for this script session
source "$CONDA_DIR/bin/activate"

# 3. Configure Mamba solver
echo "Configuring Mamba solver..."
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# 4. Create environment from environment.yml
echo "Creating/Updating environment $ENV_NAME from environment.yml..."
if [ -f "$PROJECT_DIR/environment.yml" ]; then
    conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
else
    echo "Error: environment.yml not found in $PROJECT_DIR"
    exit 1
fi

echo "Environment setup complete."
echo "To activate, run: source $CONDA_DIR/bin/activate $ENV_NAME"
