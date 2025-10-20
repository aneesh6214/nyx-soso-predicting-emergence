#!/usr/bin/env python3
"""
Run the emergence analysis for multiple SAE seeds, organizing outputs per seed,
then aggregate across seeds and create an aggregate figure with mean±95% CI.

This script does not run any servers; it just orchestrates analysis scripts.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='*', default=[0,1,2,3,4,5,6,7], help='SAE seeds to run')
    parser.add_argument('--checkpoints_dir', default='runs/grokking_final/checkpoints', help='Path to checkpoints dir')
    parser.add_argument('--output_root', default='outputs/grokking', help='Root directory for outputs')
    parser.add_argument('--experiment', default='grokking', help='Experiment type')
    parser.add_argument('--sae_features', type=int, default=256)
    parser.add_argument('--sae_sparsity', type=float, default=0.01)
    parser.add_argument('--sae_epochs', type=int, default=100)
    parser.add_argument('--edge_threshold', type=float, default=0.05)
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--layer', default=None)
    args = parser.parse_args()

    seeds_root = Path(args.output_root)
    seeds_root.mkdir(parents=True, exist_ok=True)

    # 1) Per-seed runs
    for seed in args.seeds:
        seed_dir = seeds_root / f'seed{seed:03d}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            'emergence-analysis-pipeline/cli/run_analysis.py',
            '--experiment', args.experiment,
            '--checkpoint_dir', args.checkpoints_dir,
            '--sae_features', str(args.sae_features),
            '--sae_sparsity', str(args.sae_sparsity),
            '--sae_epochs', str(args.sae_epochs),
            '--edge_threshold', str(args.edge_threshold),
            '--n_clusters', str(args.n_clusters),
            '--track_evolution', '--predict_emergence',
            '--output', str(seed_dir), '--no_experiment_dir',
            '--sae_seed', str(seed)
        ]
        if args.layer:
            cmd += ['--layer', args.layer]
        print('Running seed:', seed, '\n ', ' '.join(cmd))
        subprocess.run(cmd, check=False)

    # 2) Aggregate across seeds
    agg_cmd = [
        sys.executable,
        'emergence-analysis-pipeline/cli/aggregate_seeds.py',
        '--seeds_root', str(seeds_root),
        '--seed_glob', 'seed*',
        '--output_name', 'aggregate'
    ]
    print('Aggregating across seeds:', '\n ', ' '.join(agg_cmd))
    subprocess.run(agg_cmd, check=False)


if __name__ == '__main__':
    main()


