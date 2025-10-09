#!/usr/bin/env python3
"""
Clean implementation of grokking on modular addition.
Following the exact recipe that reliably produces grokking.

Recipe:
- Task: (a + b) mod p with p=97
- Split: 20% train, 80% test (small train set is key)
- Model: Tiny 2-layer Transformer
- Optimizer: AdamW with weight_decay=0.1 (critical!)
- Expected: Train → 100% quickly, test stays ~1% for long, then jumps to 99%
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# DATASET
# ============================================================================

class ModularAdditionDataset(Dataset):
    """Dataset for (a + b) mod p task."""
    
    def __init__(self, pairs: np.ndarray, targets: np.ndarray):
        self.pairs = torch.from_numpy(pairs).long()
        self.targets = torch.from_numpy(targets).long()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx], self.targets[idx]


def create_dataset(p: int = 97, train_fraction: float = 0.2, seed: int = 42):
    """Create train/test split for modular addition."""
    
    # Generate all possible (a, b) pairs and their targets c = (a + b) mod p
    all_pairs = []
    all_targets = []
    
    for a in range(p):
        for b in range(p):
            all_pairs.append([a, b])
            all_targets.append((a + b) % p)
    
    all_pairs = np.array(all_pairs)
    all_targets = np.array(all_targets)
    
    # Deterministic train/test split
    np.random.seed(seed)
    n_total = len(all_pairs)
    n_train = int(n_total * train_fraction)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_dataset = ModularAdditionDataset(all_pairs[train_indices], all_targets[train_indices])
    test_dataset = ModularAdditionDataset(all_pairs[test_indices], all_targets[test_indices])
    
    print(f"📊 Dataset: p={p}, total={n_total} pairs")
    print(f"   Train: {len(train_dataset)} ({train_fraction*100:.0f}%)")
    print(f"   Test: {len(test_dataset)} ({(1-train_fraction)*100:.0f}%)")
    
    return train_dataset, test_dataset


# ============================================================================
# MODEL: Option B - Tiny Transformer
# ============================================================================

class TinyTransformer(nn.Module):
    """Minimal Transformer for modular addition."""
    
    def __init__(self, vocab_size: int = 97, d_model: int = 128, n_heads: int = 4, 
                 n_layers: int = 2, d_ff: int = 256):
        super().__init__()
        
        # Token embedding (for a, b values)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (learned)
        self.pos_embedding = nn.Embedding(3, d_model)  # positions 0, 1, 2
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,  # No dropout for grokking
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        # x shape: (batch, 2) containing [a, b]
        batch_size = x.size(0)
        
        # Add a prediction token (can be any value, we'll use 0)
        x_with_pred = torch.cat([x, torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)], dim=1)
        
        # Embed tokens
        embeddings = self.embedding(x_with_pred)  # (batch, 3, d_model)
        
        # Add positional encodings
        positions = torch.arange(3, device=x.device)
        pos_embeddings = self.pos_embedding(positions)  # (3, d_model)
        embeddings = embeddings + pos_embeddings
        
        # Pass through transformer
        output = self.transformer(embeddings)  # (batch, 3, d_model)
        
        # Get prediction from the last position
        prediction = self.output(output[:, -1, :])  # (batch, vocab_size)
        
        return prediction


# ============================================================================
# TRAINING
# ============================================================================

def evaluate(model, dataloader, device):
    """Evaluate model accuracy and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for pairs, targets in dataloader:
            pairs = pairs.to(device)
            targets = targets.to(device)
            
            outputs = model(pairs)
            loss = F.cross_entropy(outputs, targets)
            
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def save_activations(model, dataloader, device, output_dir, phase_name):
    """Save model activations for sparse feature analysis."""
    model.eval()
    
    # Create activations directory
    activations_dir = output_dir / 'activations'
    activations_dir.mkdir(exist_ok=True)
    
    all_hidden_states = []
    all_outputs = []
    all_targets = []
    all_inputs = []
    
    print(f"💾 Saving activations for phase: {phase_name}")
    
    with torch.no_grad():
        for pairs, targets in dataloader:
            pairs = pairs.to(device)
            targets = targets.to(device)
            
            # Hook to capture hidden states
            hidden_states = []
            
            def hook_fn(module, input, output):
                hidden_states.append(output.detach().cpu())
            
            # Register hook on transformer output (before final projection)
            hook = model.transformer.register_forward_hook(hook_fn)
            
            # Forward pass
            outputs = model(pairs)
            
            # Remove hook
            hook.remove()
            
            # Store activations
            if hidden_states:
                all_hidden_states.append(hidden_states[0])
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(pairs.cpu())
            
            # Only save first batch for memory efficiency
            break
    
    # Save activations
    torch.save({
        'hidden_states': torch.cat(all_hidden_states, dim=0) if all_hidden_states else None,
        'outputs': torch.cat(all_outputs, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'inputs': torch.cat(all_inputs, dim=0),
    }, activations_dir / f'activations_{phase_name}.pt')
    
    print(f"   Saved activations to activations/activations_{phase_name}.pt")


def train_grokking(args):
    """Main training loop."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create dataset
    train_dataset, test_dataset = create_dataset(
        p=args.p,
        train_fraction=args.train_fraction,
        seed=args.seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.no_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = TinyTransformer(
        vocab_size=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🤖 Model: {num_params:,} parameters")
    
    # Optimizer with critical weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # 0.1 is the grokking switch!
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize wandb if requested
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project="grokking-modp",
            name=f"p{args.p}_wd{args.weight_decay}_train{args.train_fraction}_seed{args.seed}",
            config=config
        )
        print("🔗 Initialized wandb logging")
    elif args.wandb and not WANDB_AVAILABLE:
        print("⚠️  wandb requested but not available. Install with: pip install wandb")
    
    # CSV logging
    csv_path = output_dir / 'metrics.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['step', 'train_acc', 'test_acc', 'train_loss', 'test_loss'])
    csv_writer.writeheader()
    
    # Training loop
    step = 0
    best_test_acc = 0.0
    grokked = False
    grok_step = None
    pre_grok_saved = False
    
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Weight decay: {args.weight_decay} (critical for grokking!)")
    print(f"Train fraction: {args.train_fraction} (small is key!)")
    print("="*60 + "\n")
    
    # Initial evaluation
    train_acc, train_loss = evaluate(model, train_loader, device)
    test_acc, test_loss = evaluate(model, test_loader, device)
    print(f"Step {step:6d} | Train: {train_acc:6.2%} | Test: {test_acc:6.2%}")
    
    metrics_dict = {
        'step': step, 'train_acc': train_acc, 'test_acc': test_acc,
        'train_loss': train_loss, 'test_loss': test_loss
    }
    csv_writer.writerow(metrics_dict)
    
    # Log to wandb
    if wandb_run:
        wandb_run.log(metrics_dict)
    
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    while step < args.max_steps:
        model.train()
        
        for pairs, targets in train_loader:
            pairs = pairs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(pairs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional LR decay
            if args.lr_decay_step and step == args.lr_decay_step:
                for pg in optimizer.param_groups:
                    pg['lr'] *= args.lr_decay_factor
                if wandb_run:
                    wandb_run.log({'lr': optimizer.param_groups[0]['lr'], 'step': step})
            
            step += 1
            pbar.update(1)
            
            # Evaluation
            if step % args.eval_every == 0:
                train_acc, train_loss = evaluate(model, train_loader, device)
                test_acc, test_loss = evaluate(model, test_loader, device)
                
                pbar.set_postfix({
                    'train': f'{train_acc:.2%}',
                    'test': f'{test_acc:.2%}'
                })
                
                # Log to CSV and wandb
                metrics_dict = {
                    'step': step, 'train_acc': train_acc, 'test_acc': test_acc,
                    'train_loss': train_loss, 'test_loss': test_loss
                }
                csv_writer.writerow(metrics_dict)
                csv_file.flush()
                
                # Log to wandb
                if wandb_run:
                    wandb_run.log(metrics_dict)
                
                # Save regular checkpoints with activations every N steps
                if args.save_activations and step % args.activation_save_every == 0:
                    # Save checkpoint
                    ckpt_dir = output_dir / 'checkpoints'
                    ckpt_dir.mkdir(exist_ok=True)
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                    }, ckpt_dir / f'checkpoint_step_{step:05d}.pt')
                    
                    # Save activations for this step
                    save_activations(model, test_loader, device, output_dir, f'step_{step:05d}')
                    print(f"💾 Saved checkpoint and activations at step {step}")
                
                # Save pre-grok checkpoint if we're about to grok
                if not pre_grok_saved and test_acc < 0.05 and train_acc > 0.99:
                    # This is likely the last step before grokking
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                    }, output_dir / 'checkpoint_pre_grok.pt')
                    pre_grok_saved = True
                    print(f"💾 Saved pre-grok checkpoint at step {step}")
                
                # Check for grokking
                if not grokked and test_acc > 0.80:  # Lower threshold to catch grokking earlier
                    grokked = True
                    grok_step = step
                    print(f"\n🎯 GROKKING at step {step}! Test accuracy: {test_acc:.2%}")
                    
                    # Save post-grok checkpoint
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                    }, output_dir / 'checkpoint_post_grok.pt')
                    print(f"💾 Saved post-grok checkpoint at step {step}")
                    
                    # Optionally reduce LR for stability
                    if args.auto_lr_decay:
                        for pg in optimizer.param_groups:
                            pg['lr'] *= 0.1
                        print(f"📉 Reduced learning rate to {optimizer.param_groups[0]['lr']:.1e} for stability")
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    # Save best model
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                    }, output_dir / 'best_model.pt')
                
                # Print progress
                if step % (args.eval_every * 10) == 0:
                    print(f"Step {step:6d} | Train: {train_acc:6.2%} | Test: {test_acc:6.2%}")
                
                # Early stopping after grokking
                if grokked and (step - grok_step) >= args.stop_after_grok:
                    print(f"\n✋ Stopping {args.stop_after_grok} steps after grokking")
                    break
            
            if step >= args.max_steps:
                break
    
    pbar.close()
    csv_file.close()
    
    # Final evaluation
    train_acc, train_loss = evaluate(model, train_loader, device)
    test_acc, test_loss = evaluate(model, test_loader, device)
    
    # Final wandb log
    if wandb_run:
        wandb_run.log({
            'final_train_acc': train_acc,
            'final_test_acc': test_acc,
            'best_test_acc': best_test_acc,
            'grokked': grokked
        })
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"Final Train Accuracy: {train_acc:.2%}")
    print(f"Final Test Accuracy: {test_acc:.2%}")
    print(f"Best Test Accuracy: {best_test_acc:.2%}")
    
    if grokked:
        print("✅ Model successfully grokked!")
    else:
        print("⚠️  No grokking observed. Try:")
        print("   - Increasing weight_decay (try 0.2 or 0.3)")
        print("   - Decreasing train_fraction (try 0.15 or 0.1)")
        print("   - Training for more steps")
    print("="*60)
    
    # Finish wandb run
    if wandb_run:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description='Grokking on modular addition')
    
    # Task parameters
    parser.add_argument('--p', type=int, default=97, help='Prime modulus')
    parser.add_argument('--train_fraction', type=float, default=0.2, help='Fraction of data for training')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=256, help='Feed-forward dimension')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay (critical!)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=200000, help='Maximum training steps')
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling of training data (full-batch)')
    parser.add_argument('--lr_decay_step', type=int, default=0, help='Step at which to decay LR (0 = no decay)')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Multiply LR by this at decay step')
    
    # Logging parameters
    parser.add_argument('--eval_every', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--output_dir', type=str, default='runs/grokking', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--auto_lr_decay', action='store_true', help='Automatically decay LR after grokking')
    parser.add_argument('--stop_after_grok', type=int, default=10000, help='Stop training N steps after grokking')
    parser.add_argument('--save_activations', action='store_true', help='Save activations before/after grokking for analysis')
    parser.add_argument('--activation_save_every', type=int, default=1000, help='Save activations every N steps')
    
    args = parser.parse_args()
    
    train_grokking(args)


if __name__ == '__main__':
    main()
