#!/bin/bash
# GraspGen Service Setup Script
# Clones Contact-GraspNet PyTorch and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing Python dependencies ==="
pip install fastapi uvicorn "numpy<2" opencv-python Pillow scipy requests trimesh pyyaml pyrender

echo "=== Cloning Contact-GraspNet PyTorch ==="
if [ ! -d "contact_graspnet_pytorch" ]; then
    git clone https://github.com/elchun/contact_graspnet_pytorch.git cgn_repo
    cp -r cgn_repo/contact_graspnet_pytorch ./contact_graspnet_pytorch
    cp -r cgn_repo/checkpoints ./contact_graspnet_pytorch/checkpoints
    rm -rf cgn_repo
    # Patch torch.load for PyTorch >= 2.6
    sed -i 's/torch.load(filename)/torch.load(filename, weights_only=False)/' contact_graspnet_pytorch/checkpoints.py
else
    echo "contact_graspnet_pytorch already exists, skipping clone"
fi

echo "=== Cloning Pointnet2 PyTorch ==="
if [ ! -d "Pointnet_Pointnet2_pytorch" ]; then
    git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
else
    echo "Pointnet_Pointnet2_pytorch already exists, skipping clone"
fi

echo ""
echo "=== Setup complete ==="
echo "Start the service with: python main.py"
echo "Or: uvicorn main:app --host 0.0.0.0 --port 8002"
