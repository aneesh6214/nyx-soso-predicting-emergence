#!/usr/bin/env python3
"""
Evaluate a local HF checkpoint with lm-eval + vLLM and save JSON to data/evals/.

Example:
  python scripts/02_run_eval.py \
    --ckpt /workspace/checkpoints/eleutherai_pythia-6.9b-deduped-step080000 \
    --tasks gsm8k,bbh_arc_challenge,arc_challenge,hellaswag,winogrande,mmlu \
    --batch_size 32 --tp 1 --dtype bf16 --outdir data/evals
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional

DEFAULT_TASKS = [
    # choose a few high-signal tasks by default
    "gsm8k",          # grade-school math word problems (often "thresholdy")
    "arc_challenge",  # hard multiple-choice science/commonsense
    "hellaswag",      # commonsense completion (easy to run, tracks scaling)
    "winogrande",     # coreference/reasoning
    "mmlu"            # broad knowledge; use 5-shot by lm-eval default
    # You can add: "bbh_*" (subset), "math", "humaneval", "mbpp" later
]

def extract_model_name(ckpt_path: str) -> str:
    """Extract a clean model name from checkpoint path for output naming."""
    path = Path(ckpt_path)
    # Handle paths like: .../eleutherai_pythia-6.9b-deduped-step080000
    if path.name.startswith("eleutherai_pythia"):
        return path.name
    # Fallback to just the directory name
    return path.name

def validate_checkpoint(ckpt_path: str) -> bool:
    """Validate that the checkpoint directory exists and contains model files."""
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        print(f"[ERROR] Checkpoint directory does not exist: {ckpt_path}")
        return False
    
    # Check for common model files
    model_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    if not any((ckpt_dir / f).exists() for f in model_files):
        print(f"[WARN] Checkpoint directory may not contain model files: {ckpt_path}")
        print(f"[WARN] Expected files like: {model_files}")
        return False
    
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Local path to HF snapshot dir (from 01_fetch_checkpoint.py)")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="Comma-separated lm-eval task list. Use `python -m lm_eval --tasks list` to see all.")
    ap.add_argument("--batch_size", type=int, default=32, help="lm-eval batch size (reduce if OOM)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size for vLLM")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","float32","auto"], help="dtype for vLLM")
    ap.add_argument("--outdir", default="data/evals", help="Directory to write results JSON")
    ap.add_argument("--max_samples", type=int, help="Limit number of samples per task (useful for quick testing)")
    args = ap.parse_args()

    # Validate checkpoint
    if not validate_checkpoint(args.ckpt):
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract model name for output file
    model_name = extract_model_name(args.ckpt)
    print(f"[INFO] Evaluating model: {model_name}")
    print(f"[INFO] Tasks: {args.tasks}")
    print(f"[INFO] Batch size: {args.batch_size}, Tensor parallel: {args.tp}, Dtype: {args.dtype}")

    # vLLM model args string
    model_args = ",".join([
        f"pretrained={args.ckpt}",
        f"tensor_parallel_size={args.tp}",
        f"dtype={args.dtype}",
        "trust_remote_code=True",
        "gpu_memory_utilization=0.9",  # Optimize for Pythia models
    ])

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--batch_size", str(args.batch_size),
        "--output_path", str(outdir)
    ]

    # Add max_samples if specified
    if args.max_samples:
        cmd.extend(["--limit", str(args.max_samples)])
        print(f"[INFO] Limiting to {args.max_samples} samples per task")

    print(f"\n[RUN] Command: {' '.join(cmd)}")
    print(f"[RUN] Starting evaluation...")
    
    try:
        subprocess.check_call(cmd)
        print(f"\n[SUCCESS] Evaluation completed!")
        print(f"[SUCCESS] Results written to: {outdir}")
        
        # List output files
        output_files = list(outdir.glob("*.json"))
        if output_files:
            print(f"[INFO] Output files:")
            for f in output_files:
                print(f"  - {f.name}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed with exit code {e.returncode}")
        print(f"[ERROR] Check if the model loads correctly and you have enough GPU memory")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Evaluation stopped by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
