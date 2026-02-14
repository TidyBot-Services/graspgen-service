#!/bin/bash
# GraspGen Service Setup Script
# Clones Contact-GraspNet PyTorch and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Cloning Contact-GraspNet PyTorch ==="
if [ ! -d "contact_graspnet_pytorch" ]; then
    git clone https://github.com/elchun/contact_graspnet_pytorch.git cgn_repo
    # Copy the package into our service directory
    cp -r cgn_repo/contact_graspnet_pytorch ./contact_graspnet_pytorch
    cp -r cgn_repo/checkpoints ./contact_graspnet_pytorch/checkpoints
    rm -rf cgn_repo
else
    echo "contact_graspnet_pytorch already exists, skipping clone"
fi

echo "=== Installing Contact-GraspNet package ==="
cd "$SCRIPT_DIR"
# Install pointnet2 ops if needed
if [ -d "contact_graspnet_pytorch/pointnet2" ]; then
    cd contact_graspnet_pytorch/pointnet2
    pip install -e . 2>/dev/null || echo "pointnet2 ops install skipped (may need manual compilation)"
    cd "$SCRIPT_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "Start the service with: python main.py"
echo "Or: uvicorn main:app --host 0.0.0.0 --port 8001"
