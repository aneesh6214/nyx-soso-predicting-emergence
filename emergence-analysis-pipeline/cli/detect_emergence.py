import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.emergence_detection import label_emergence_steps


def read_metrics_csv(path: Path) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    test_acc: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(float(row["step"])))
                test_acc.append(float(row["test_acc"]))
            except Exception:
                # Skip malformed rows
                continue
    return steps, test_acc


def main():
    parser = argparse.ArgumentParser(description="Detect emergence from metrics.csv")
    parser.add_argument(
        "--metrics",
        type=str,
        default=str(Path("runs/grokking_final/metrics.csv")),
        help="Path to metrics.csv with columns: step,train_acc,test_acc,train_loss,test_loss",
    )
    parser.add_argument("--jump_threshold", type=float, default=0.20)
    parser.add_argument("--stability_tol", type=float, default=0.02)
    parser.add_argument("--stability_horizon", type=int, default=3)
    parser.add_argument("--require_full_horizon", action="store_true")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")

    steps, acc = read_metrics_csv(metrics_path)
    if not steps:
        raise SystemExit("No rows found in metrics file or missing columns.")

    detected = label_emergence_steps(
        steps,
        acc,
        jump_threshold=args.jump_threshold,
        stability_tol=args.stability_tol,
        stability_horizon=args.stability_horizon,
        require_full_horizon=args.require_full_horizon,
    )

    if detected:
        # Report the first detected emergence step
        print(detected[0])
    else:
        print("-1")


if __name__ == "__main__":
    main()


