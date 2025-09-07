#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
TASKS="gsm8k,arc_challenge,hellaswag,winogrande"
BATCH=8
# Choose the steps you want; example set below
STEPS=("0" "1000" "4000" "20000" "40000" "60000" "80000" "100000" "120000" "140000" "143000")

# ---- paths & env ----
cd /workspace/nyx-soso-predicting-emergence
source .venv/bin/activate

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" data/evals /workspace/checkpoints

# ---- polite dataset prefetch (safe to skip if already cached) ----
python - <<'PY'
import time
from datasets import load_dataset, DownloadConfig
def grab(ds, *args, **kw):
    for i in range(6):
        try:
            print(f"[prefetch] {ds} {args} {kw}")
            load_dataset(ds, *args, download_config=DownloadConfig(max_retries=6), **kw)
            print(f"[prefetch] OK: {ds}")
            return
        except Exception as e:
            wait = min(60, 2**i)
            print(f"[prefetch] Retry in {wait}s due to: {e}")
            time.sleep(wait)
grab("openai/gsm8k", "main")
grab("ai2_arc", "ARC-Challenge")
grab("hellaswag")
grab("winogrande", "winogrande_xl")
PY

# ---- eval loop ----
for S in "${STEPS[@]}"; do
  CKPT_DIR="/workspace/checkpoints/eleutherai_pythia-6.9b-deduped-step${S}"
  RUN_DIR="data/evals/pythia6.9b_step${S}"
  RES_FILE="${RUN_DIR}/results.json"

  if [ -f "$RES_FILE" ]; then
    echo "== Skipping step ${S} (already evaluated) =="
    continue
  fi

  if [ ! -d "$CKPT_DIR" ]; then
    echo "== Fetching checkpoint step ${S} =="
    python scripts/01_fetch_checkpoint.py \
      --model eleutherai/pythia-6.9b-deduped \
      --rev "step${S}" \
      --outdir /workspace/checkpoints || { echo "Fetch failed for step ${S}"; continue; }
  else
    echo "== Using cached checkpoint ${CKPT_DIR} =="
  fi

  echo "== Evaluating step ${S} =="
  python -m lm_eval \
    --model hf \
    --model_args "pretrained=${CKPT_DIR},dtype=bfloat16" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH}" \
    --output_path "${RUN_DIR}" || { echo "Eval failed for step ${S}"; continue; }

  # Append into clean CSV (rebuilds from all JSONs)
  python scripts/aggregate_clean.py || true
done

echo "Done. See: data/evals/clean_results.csv"

# To Use:   
# chmod +x scripts/eval_loop.sh
# optional: edit the STEPS list at top
# bash scripts/eval_loop.sh