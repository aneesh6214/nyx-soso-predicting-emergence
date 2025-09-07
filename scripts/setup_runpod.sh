#!/usr/bin/env bash
set -euo pipefail

echo "== GPU =="
nvidia-smi || true
echo "== Disk =="
df -h /workspace || true

cd /workspace

# --- repo (use your fork) ---
REPO_DIR=/workspace/nyx-soso-predicting-emergence
FORK_HTTPS="https://github.com/oskarherlitz/nyx-soso-predicting-emergence.git"
BRANCH=feat/setup

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning your fork…"
  git clone "$FORK_HTTPS" nyx-soso-predicting-emergence
fi
cd "$REPO_DIR"
git remote set-url origin "$FORK_HTTPS" || true
git fetch origin
git checkout "$BRANCH" || git checkout -t "origin/$BRANCH"
git pull --ff-only origin "$BRANCH"

# --- python env ---
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -V
pip install -U pip wheel

# --- deps (stable, HF backend only) ---
pip install --no-cache-dir \
  "transformers==4.45.1" "peft==0.13.2" "accelerate==0.34.2" "huggingface_hub==0.24.6" \
  "lm-eval==0.4.2" "datasets==2.20.0" sentencepiece numpy scipy pandas scikit-learn \
  networkx matplotlib einops pyyaml

# --- standard dirs ---
mkdir -p data/evals /workspace/checkpoints

# --- set HF caches into /workspace (optional but helpful) ---
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

echo "== Quick CUDA check =="
python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY

echo "Setup complete."

# To Use:   
# chmod +x scripts/setup_runpod.sh
# bash scripts/setup_runpod.sh