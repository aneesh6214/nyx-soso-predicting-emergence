#!/usr/bin/env python3
"""
Run the grokking experiment with proven parameters.
This follows the exact recipe that reliably shows grokking.
"""

import subprocess
import sys

print("""
╔══════════════════════════════════════════════════════════════╗
║           GROKKING EXPERIMENT - PROVEN RECIPE                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                                ║
║  Task: Modular addition (a + b) mod 97                        ║
║  Split: 20% train, 80% test (small training set is KEY!)      ║
║  Model: 2-layer Transformer, 128d                             ║
║  Optimizer: AdamW with weight_decay=0.1 (CRITICAL!)           ║
║                                                                ║
║  Expected behavior:                                           ║
║  • Steps 0-5k: Training accuracy → 100%                       ║
║  • Steps 5k-50k: Test accuracy stays at ~1% (random)          ║
║  • Steps 50k-100k: GROKKING! Test accuracy jumps to ~99%      ║
║                                                                ║
║  Watch for the sudden jump in test accuracy!                  ║
║                                                                ║
╚══════════════════════════════════════════════════════════════╝
""")

cmd = [
    sys.executable, "-m", "grokking.grok_clean",
    
    # Task parameters (proven to work)
    "--p", "97",
    "--train_fraction", "0.2",  # Small training set is key!
    
    # Model parameters (small model)
    "--d_model", "128",
    "--n_heads", "4", 
    "--n_layers", "2",
    "--d_ff", "256",
    
    # Optimization (critical parameters)
    "--lr", "1e-3",
    "--weight_decay", "0.1",  # High weight decay is THE key to grokking!
    "--batch_size", "512",
    
    # Training
    "--max_steps", "200000",
    "--eval_every", "500",
    
    # Output
    "--output_dir", "runs/grokking",
    "--seed", "42",
    "--wandb",  # Enable wandb logging
]

print("Command:", " ".join(cmd))
print("-" * 60)

subprocess.run(cmd)
