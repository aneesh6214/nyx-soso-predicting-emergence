# scripts/00_smoke_eval.py
import json, os, subprocess, sys
from pathlib import Path
import time
import glob

OUT_DIR = Path("data/evals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Run lm-eval on a tiny CPU model
cmd = [
    sys.executable, "-m", "lm_eval",
    "--model", "hf",
    "--model_args", "pretrained=sshleifer/tiny-gpt2,dtype=float32,trust_remote_code=False",
    "--tasks", "hellaswag",
    "--num_fewshot", "0",
    "--limit", "50",          # tiny slice; sanity check only
    "--batch_size", "1",
    "--device", "cpu",
    "--output_path", str(OUT_DIR)   # <-- directory, let lm-eval name the file
]
print("Running:", " ".join(cmd))
start = time.time()
subprocess.check_call(cmd)
print(f"lm-eval finished in {time.time()-start:.1f}s")

# Find the newest JSON that lm-eval just wrote
candidates = sorted(glob.glob(str(OUT_DIR / "*.json")), key=os.path.getmtime)
if not candidates:
    raise SystemExit(f"No JSON found in {OUT_DIR}. (lm-eval may have changed its output path)")

latest = candidates[-1]
print("Reading:", latest)
with open(latest) as f:
    res = json.load(f)

metrics = res.get("results", {}).get("hellaswag", {})
acc = metrics.get("acc") or metrics.get("acc_norm") or (next(iter(metrics.values())) if metrics else None)
print("Hellaswag (tiny) metric:", acc)
print("Saved JSON:", latest)
