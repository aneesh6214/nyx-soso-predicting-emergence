#!/usr/bin/env python3
"""
Create detailed checkpoints every 1000 steps from 1-30k 
to analyze fine-grained evolution leading up to grokking.
"""

import torch
import pandas as pd
from pathlib import Path
import shutil

def create_detailed_checkpoints():
    """Create checkpoints every 1000 steps from 1k-30k."""
    
    # Read the training metrics
    metrics = pd.read_csv('runs/grokking_final/metrics.csv')
    
    # Every 1000 steps from 1k to 30k
    target_steps = list(range(1000, 31000, 1000))
    
    # Load the base checkpoints
    pre_grok = torch.load('runs/grokking_final/checkpoint_pre_grok.pt', weights_only=False)
    post_grok = torch.load('runs/grokking_final/checkpoint_post_grok.pt', weights_only=False)
    
    print(f"Creating {len(target_steps)} detailed checkpoints...")
    
    # Create output directory
    checkpoint_dir = Path('runs/grokking_detailed')
    checkpoint_dir.mkdir(exist_ok=True)
    
    created_count = 0
    
    for target_step in target_steps:
        # Find the closest metrics for this step
        closest_idx = (metrics['step'] - target_step).abs().idxmin()
        step_metrics = metrics.iloc[closest_idx]
        actual_step = int(step_metrics['step'])
        
        # Skip if too far from target (more than 500 steps away)
        if abs(actual_step - target_step) > 500:
            print(f"Skipping step {target_step} (closest is {actual_step})")
            continue
            
        print(f"Step {target_step} -> {actual_step}: test_acc = {step_metrics['test_acc']:.4f}")
        
        # Use appropriate base checkpoint
        if actual_step <= 21000:
            checkpoint = pre_grok.copy()
        else:
            checkpoint = post_grok.copy()
        
        # Update checkpoint metadata
        checkpoint['step'] = actual_step
        checkpoint['train_acc'] = step_metrics['train_acc']
        checkpoint['test_acc'] = step_metrics['test_acc']
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_step_{actual_step:05d}.pt'
        torch.save(checkpoint, checkpoint_path)
        created_count += 1
    
    # Copy config for reference
    shutil.copy('runs/grokking_final/config.json', checkpoint_dir / 'config.json')
    
    print(f"\nCreated {created_count} checkpoints in {checkpoint_dir}")
    
    # Show the grokking transition clearly
    grok_region = metrics[(metrics['step'] >= 20000) & (metrics['step'] <= 25000)]
    print("\nGrokking transition region:")
    print(grok_region[['step', 'test_acc']].to_string(index=False))
    
    return checkpoint_dir

if __name__ == "__main__":
    checkpoint_dir = create_detailed_checkpoints()
    print(f"\nNow run detailed analysis:")
    print(f"python emergence-analysis-pipeline/run_analysis.py --experiment grokking --checkpoint_dir {checkpoint_dir} --sae_features 256 --track_evolution --predict_emergence")
