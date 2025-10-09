#!/usr/bin/env python3
"""
Final test run with all optimizations for clean grokking.
This script uses all the lessons learned to get reliable grokking.
"""

import subprocess
import sys

print("""
╔══════════════════════════════════════════════════════════════╗
║             FINAL GROKKING TEST - OPTIMIZED                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                                ║
║  Settings optimized for clean grokking:                       ║
║  • Full-batch training (no shuffle)                           ║
║  • High weight decay (0.1)                                    ║
║  • Auto LR decay after grokking                               ║
║  • Early stopping 10k steps after grokking                    ║
║  • Save checkpoints & activations for analysis                ║
║                                                                ║
║  Expected timeline:                                           ║
║  • Steps 0-5k: Training → 100%                                ║
║  • Steps 5k-40k: Test stays at ~1% (memorization)             ║
║  • Steps 40k-60k: GROKKING! Test jumps to ~99%                ║
║  • Stops automatically 10k steps after grokking               ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝
""")

cmd = [
    sys.executable, "-m", "grokking.grok_clean",
    
    # Task parameters
    "--p", "97",
    "--train_fraction", "0.2",
    
    # Model parameters  
    "--d_model", "128",
    "--n_heads", "4",
    "--n_layers", "2",
    "--d_ff", "256",
    
    # Optimization (proven settings)
    "--lr", "1e-3",
    "--weight_decay", "0.1",  # High WD for grokking
    "--batch_size", "1881",   # Full batch on 20% of p=97
    "--no_shuffle",           # Deterministic gradients
    
    # Training control
    "--max_steps", "120000",
    "--eval_every", "1000",
    "--auto_lr_decay",        # Reduce LR after grokking for stability
    "--stop_after_grok", "10000",  # Stop shortly after grokking
    
    # Logging and analysis
    "--output_dir", "runs/grokking_final",
    "--seed", "42",
    "--wandb",
    "--save_activations",     # Save pre/post grok activations
]

print("Command:", " ".join(cmd))
print("-" * 60)
print("\nOutputs will be saved to runs/grokking_final/:")
print("  • checkpoint_pre_grok.pt  - Model before generalization")
print("  • checkpoint_post_grok.pt - Model after generalization")
print("  • activations/            - Hidden states for SAE analysis")
print("  • metrics.csv             - Full training curves")
print("-" * 60)

subprocess.run(cmd)
