#!/usr/bin/env bash
# Setup script for RunPod GPU pods.
#
# Usage:
#   1. Create a RunPod pod with a PyTorch template (e.g., runpod/pytorch:2.1.0-py3.10-cuda12.1.0)
#   2. Attach a network volume (mounted at /workspace by default)
#   3. SSH in and run:  bash setup_runpod.sh [model_name]
#
# The network volume persists model weights across pod restarts.

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B}"
REPO_URL="https://github.com/ndouglas/adaptive-recursive-inference-experiment.git"
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/ari"
HF_CACHE="${WORKSPACE}/huggingface"

echo "=== RunPod Setup ==="
echo "Model: ${MODEL}"
echo "Project: ${PROJECT_DIR}"
echo "HF cache: ${HF_CACHE}"
echo ""

# --- Clone or update repo ---
if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "Updating existing repo..."
    cd "${PROJECT_DIR}"
    git pull --ff-only
else
    echo "Cloning repo..."
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    cd "${PROJECT_DIR}"
fi

# --- Install dependencies ---
echo ""
echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch transformers accelerate scipy numpy tqdm safetensors

# --- Set HF cache directory (persistent across pod restarts) ---
export HF_HOME="${HF_CACHE}"

# --- Pre-download model weights ---
echo ""
echo "Pre-downloading model weights to network volume..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['HF_HOME'] = '${HF_CACHE}'
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('${MODEL}')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('${MODEL}')
print('Done — weights cached at ${HF_CACHE}')
"

# --- Verify environment ---
echo ""
echo "Running environment verification..."
cd "${PROJECT_DIR}"
HF_HOME="${HF_CACHE}" python3 scripts/verify_environment.py

echo ""
echo "=== Setup complete ==="
echo "To run a sweep:"
echo "  cd ${PROJECT_DIR}"
echo "  HF_HOME=${HF_CACHE} python3 scripts/run_sweep.py --model ${MODEL} --results ${WORKSPACE}/sweep_results_7b.json"
