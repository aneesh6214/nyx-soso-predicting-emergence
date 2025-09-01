#!/usr/bin/env bash
set -euo pipefail
cd /workspace/nyx-soso-predicting-emergence
zip -r evals_results.zip data/evals -x "*/__pycache__/*" "*.ipynb_checkpoints/*"
echo "Created evals_results.zip"
