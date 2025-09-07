#!/usr/bin/env python3
import os, re, csv, json, glob
from pathlib import Path
from typing import Dict, Tuple, Optional

RESULTS_GLOB = "data/evals/**/results.json"
OUT_CSV = "data/evals/clean_results.csv"

def extract_step(ckpt_str: str, run_basename: str) -> Optional[int]:
    for s in (ckpt_str or "", run_basename or ""):
        m = re.search(r"step(\d+)", s)
        if m: return int(m.group(1))
    return None

def normalize_keys(metrics: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if not isinstance(v, (int, float)): continue
        base = k.split(",", 1)[0].strip()
        out.setdefault(base, v)
    return out

def pick_accuracy(task: str, raw: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    strict = raw.get("exact_match,strict-match")
    flex   = raw.get("exact_match,flexible-extract")
    m = normalize_keys(raw)

    if task == "gsm8k":
        if isinstance(strict, (int, float)): return "exact_match_strict", strict
        if isinstance(flex, (int, float)):   return "exact_match_flexible", flex
        if "exact_match" in m:               return "exact_match", m["exact_match"]
        return None, None

    if task in {"arc_challenge", "hellaswag"}:
        if "acc_norm" in m: return "acc_norm", m["acc_norm"]
        if "acc" in m:      return "acc", m["acc"]
        return None, None

    if task == "winogrande":
        return ("acc", m.get("acc")) if "acc" in m else (None, None)

    for k in ("acc_norm", "acc", "exact_match", "multiple_choice_grade"):
        if k in m: return k, m[k]
    return None, None

def main():
    paths = sorted(glob.glob(RESULTS_GLOB, recursive=True), key=os.path.getmtime)
    if not paths:
        raise SystemExit("No results.json found under data/evals/")

    latest = {}  # (step, task) -> (mtime, run_dir, ckpt_str, mdict)

    for p in paths:
        try:
            R = json.load(open(p))
        except Exception as e:
            print(f"[WARN] Could not parse {p}: {e}")
            continue

        run_dir = Path(p).parent.name
        ckpt_str = (R.get("config") or {}).get("model_args", "") or ""
        step = extract_step(ckpt_str, run_dir)
        if step is None: continue

        for task, mdict in (R.get("results") or {}).items():
            if not isinstance(mdict, dict): continue
            mtime = os.path.getmtime(p)
            key = (step, task)
            if key not in latest or mtime > latest[key][0]:
                latest[key] = (mtime, run_dir, ckpt_str, mdict)

    rows = []
    for (step, task), (_, run_dir, ckpt_str, mdict) in sorted(latest.items()):
        metric_used, acc = pick_accuracy(task, mdict)
        rows.append({
            "step": step,
            "task": task,
            "accuracy": acc,
            "metric_used": metric_used,
            "run_dir": run_dir,
        })

    rows.sort(key=lambda r: (r["task"], r["step"]))
    os.makedirs(Path(OUT_CSV).parent, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "task", "accuracy", "metric_used", "run_dir"])
        w.writeheader(); w.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows.")
    for r in rows[:20]:
        print(f"step={r['step']:>6}  task={r['task']:<13}  acc={r['accuracy']!s:<10} ({r['metric_used']})  run={r['run_dir']}")

if __name__ == "__main__":
    main()
