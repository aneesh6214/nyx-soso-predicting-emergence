#!/usr/bin/env bash
set -euo pipefail

cd /workspace/nyx-soso-predicting-emergence

echo "== Cached checkpoints =="
ls -1 /workspace/checkpoints | grep eleutherai_pythia-6.9b-deduped || true

echo
echo "== Evaluated steps (have results.json) =="
for d in data/evals/pythia6.9b_step*/; do
  [ -d "$d" ] || continue
  if [ -f "${d}results.json" ]; then
    echo "$(basename "$d" | sed 's/pythia6.9b_step//')"
  fi
done | sort -n
