#!/usr/bin/env python3
"""
Aggregate evolution metrics across multiple SAE seeds and plot mean ± 95% CI.
Also saves the stacked long-format CSV for downstream analyses (e.g., lead-lag).
"""

import argparse
from pathlib import Path
from typing import Dict, List
import json
import pandas as pd
import os
import sys

# Ensure local imports work when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracking import EmergenceTracker
from core.emergence_detection import detect_emergence as _detect_emergence


def load_seed_metrics(seed_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    seed_to_df: Dict[str, pd.DataFrame] = {}
    for seed_dir in seed_dirs:
        evo_csv = seed_dir / 'evolution_analysis' / 'evolution_metrics.csv'
        if evo_csv.exists():
            df = pd.read_csv(evo_csv)
            # Ensure step present and sorted
            if 'step' in df.columns:
                df = df.sort_values('step')
            seed_to_df[seed_dir.name] = df
        else:
            print(f"[WARN] Missing metrics: {evo_csv}")
    return seed_to_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate across SAE seeds")
    parser.add_argument("--seeds_root", required=True, help="Root directory containing per-seed outputs (e.g., outputs/grokking)")
    parser.add_argument("--seed_glob", default="seed*", help="Glob under seeds_root to find seed dirs (default: seed*)")
    parser.add_argument("--output_name", default="aggregate", help="Subdir name to write aggregate outputs")
    parser.add_argument("--metrics", nargs="*", help="Optional list of metric columns to aggregate")
    args = parser.parse_args()

    seeds_root = Path(args.seeds_root)
    seed_dirs = sorted([p for p in seeds_root.glob(args.seed_glob) if p.is_dir()])
    if not seed_dirs:
        print(f"No seed directories found under {seeds_root} with pattern {args.seed_glob}")
        return

    print(f"Found {len(seed_dirs)} seed runs: {[p.name for p in seed_dirs]}")
    seed_to_df = load_seed_metrics(seed_dirs)
    if not seed_to_df:
        print("No evolution_metrics.csv found; aborting")
        return

    # Compute per-seed emergence steps from local test_acc if available
    seed_emergence_steps = []
    for seed, df in seed_to_df.items():
        if 'step' in df.columns and 'test_acc' in df.columns:
            steps = df['step'].tolist()
            acc = df['test_acc'].fillna(0.0).tolist()
            _, idxs = _detect_emergence(steps, acc, jump_threshold=0.20, stability_tol=0.02, stability_horizon=3, require_full_horizon=True)
            if idxs:
                # Use first detected point
                i = idxs[0]
                if 0 <= i < len(steps):
                    seed_emergence_steps.append(steps[i])

    agg = EmergenceTracker.aggregate_seed_metrics(seed_to_df, value_cols=args.metrics)
    out_dir = seeds_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated data
    agg['mean'].to_csv(out_dir / 'aggregate_mean.csv', index=False)
    agg['std'].to_csv(out_dir / 'aggregate_std.csv', index=False)
    agg['ci95'].to_csv(out_dir / 'aggregate_ci95.csv', index=False)
    agg['long'].to_csv(out_dir / 'aggregate_long.csv', index=False)

    # Plot default set (up to 6 metrics)
    EmergenceTracker.plot_aggregate(
        mean_df=agg['mean'],
        ci95_df=agg['ci95'],
        metrics=args.metrics,
        save_path=out_dir / 'aggregate_evolution.png',
        spaghetti_df=agg['long'],
        emergence_steps=seed_emergence_steps
    )

    # Save manifest
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump({
            'seeds': [p.name for p in seed_dirs],
            'metrics': args.metrics if args.metrics else [c for c in agg['mean'].columns if c != 'step']
        }, f, indent=2)

    print(f"Aggregate saved to: {out_dir}")


if __name__ == "__main__":
    main()


