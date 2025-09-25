#!/usr/bin/env python3
"""
Train Sparse Autoencoder (SAE) on neural activations for mechanistic interpretability.

Usage:
  python scripts/train_sae.py --activations data/activations/activations.pt --features 8192 --sparsity 0.1
  python scripts/train_sae.py --activations data/activations/activations.pt --resume_from data/sae/checkpoint_epoch_800.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import wandb


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 regularization."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_penalty: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_penalty = sparsity_penalty
        
        # Encoder: input -> hidden
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # Decoder: hidden -> input
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparsity regularization."""
        # Encode
        hidden = self.encoder(x)
        
        # Apply ReLU for sparsity
        hidden_sparse = torch.relu(hidden)
        
        # Decode
        reconstructed = self.decoder(hidden_sparse)
        
        return reconstructed, hidden_sparse
    
    def get_sparsity_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss."""
        return torch.mean(torch.abs(hidden))
    
    def get_reconstruction_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute MSE reconstruction loss."""
        return torch.mean((original - reconstructed) ** 2)


class SAETrainer:
    """Trainer for Sparse Autoencoder."""
    
    def __init__(
        self,
        model: SparseAutoencoder,
        learning_rate: float = 1e-3,
        sparsity_penalty: float = 0.01,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.sparsity_penalty = sparsity_penalty
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.history = {
            "reconstruction_loss": [],
            "sparsity_loss": [],
            "total_loss": [],
            "sparsity_rate": [],
            "learning_rate": []
        }
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_recon_loss = 0.0
        total_sparsity_loss = 0.0
        total_sparsity_rate = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training epoch"):
            # TensorDataset returns tuples, extract the tensor
            batch = batch[0].to(self.device)
            
            # Forward pass
            reconstructed, hidden = self.model(batch)
            
            # Compute losses
            recon_loss = self.model.get_reconstruction_loss(batch, reconstructed)
            sparsity_loss = self.model.get_sparsity_loss(hidden)
            total_loss = recon_loss + self.sparsity_penalty * sparsity_loss
            
            # Compute sparsity rate
            sparsity_rate = torch.mean((hidden > 0).float())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_sparsity_rate += sparsity_rate.item()
            num_batches += 1
        
        # Average metrics
        avg_recon_loss = total_recon_loss / num_batches
        avg_sparsity_loss = total_sparsity_loss / num_batches
        avg_sparsity_rate = total_sparsity_rate / num_batches
        avg_total_loss = avg_recon_loss + self.sparsity_penalty * avg_sparsity_loss
        
        return {
            "reconstruction_loss": avg_recon_loss,
            "sparsity_loss": avg_sparsity_loss,
            "total_loss": avg_total_loss,
            "sparsity_rate": avg_sparsity_rate,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        save_dir: Path,
        log_to_wandb: bool = False,
        start_epoch: int = 0
    ):
        """Train the SAE."""
        print(f"Training SAE for {num_epochs} epochs starting from epoch {start_epoch}...")
        
        if log_to_wandb:
            wandb.init(
                project="sae-pythia",
                config={
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "sparsity_penalty": self.sparsity_penalty,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "num_epochs": num_epochs,
                    "start_epoch": start_epoch
                }
            )
        
        best_loss = float('inf')
        
        for epoch in range(start_epoch, num_epochs):
            # Train epoch
            metrics = self.train_epoch(dataloader)
            
            # Update learning rate
            self.scheduler.step(metrics["total_loss"])
            
            # Log metrics
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Log to wandb
            if log_to_wandb:
                wandb.log(metrics, step=epoch)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Recon Loss: {metrics['reconstruction_loss']:.6f}, "
                      f"Sparsity Loss: {metrics['sparsity_loss']:.6f}, "
                      f"Total Loss: {metrics['total_loss']:.6f}, "
                      f"Sparsity Rate: {metrics['sparsity_rate']:.3f}")
            
            # Save best model
            if metrics["total_loss"] < best_loss:
                best_loss = metrics["total_loss"]
                self.save_model(save_dir / "best_model.pt")
            
            # Save checkpoint
            if epoch % 50 == 0:
                self.save_model(save_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Save final model and history
        self.save_model(save_dir / "final_model.pt")
        self.save_history(save_dir / "training_history.json")
        self.plot_training_curves(save_dir / "training_curves.png")
        
        if log_to_wandb:
            wandb.finish()
        
        print("✅ SAE training complete!")
    
    def save_model(self, path: Path):
        """Save model state."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "sparsity_penalty": self.sparsity_penalty,
            "input_dim": self.model.input_dim,
            "hidden_dim": self.model.hidden_dim
        }, path)
    
    def load_model(self, path: Path):
        """Load model state from checkpoint."""
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print("✅ Checkpoint loaded successfully!")
        return checkpoint
    
    def save_history(self, path: Path):
        """Save training history."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, path: Path):
        """Load training history."""
        if path.exists():
            with open(path, "r") as f:
                self.history = json.load(f)
            print(f"✅ Training history loaded from: {path}")
        else:
            print(f"⚠️  No training history found at: {path}")
    
    def plot_training_curves(self, path: Path):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reconstruction loss
        axes[0, 0].plot(self.history["reconstruction_loss"])
        axes[0, 0].set_title("Reconstruction Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("MSE Loss")
        axes[0, 0].grid(True)
        
        # Sparsity loss
        axes[0, 1].plot(self.history["sparsity_loss"])
        axes[0, 1].set_title("Sparsity Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("L1 Loss")
        axes[0, 1].grid(True)
        
        # Total loss
        axes[1, 0].plot(self.history["total_loss"])
        axes[1, 0].set_title("Total Loss")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Total Loss")
        axes[1, 0].grid(True)
        
        # Sparsity rate
        axes[1, 1].plot(self.history["sparsity_rate"])
        axes[1, 1].set_title("Sparsity Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Fraction Active")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


def load_activations(activations_path: Path) -> torch.Tensor:
    """Load activations from file."""
    print(f"Loading activations from: {activations_path}")
    activations = torch.load(activations_path)
    
    # Reshape to (num_samples, hidden_dim)
    if len(activations.shape) > 2:
        activations = activations.view(activations.shape[0], -1)
    
    print(f"Activation shape: {activations.shape}")
    return activations


def create_dataloader(activations: torch.Tensor, batch_size: int = 1024) -> torch.utils.data.DataLoader:
    """Create dataloader for activations."""
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    parser.add_argument("--activations", required=True, help="Path to activations.pt file")
    parser.add_argument("--features", type=int, default=8192, help="Number of SAE features (default: 8192)")
    parser.add_argument("--sparsity", type=float, default=0.01, help="Sparsity penalty (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--output", default="data/sae", help="Output directory (default: data/sae)")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--resume_from", help="Path to checkpoint file to resume from")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load activations
    activations = load_activations(Path(args.activations))
    input_dim = activations.shape[1]
    
    # Create SAE model
    model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.features,
        sparsity_penalty=args.sparsity
    )
    
    # Create trainer
    trainer = SAETrainer(
        model=model,
        learning_rate=args.lr,
        sparsity_penalty=args.sparsity
    )
    
    # Handle resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return
        
        # Load checkpoint
        checkpoint = trainer.load_model(checkpoint_path)
        
        # Extract epoch number from checkpoint filename
        if "checkpoint_epoch_" in checkpoint_path.name:
            start_epoch = int(checkpoint_path.name.split("_")[-1].split(".")[0]) + 1
            print(f"Resuming from epoch {start_epoch}")
        
        # Load training history if it exists
        history_path = output_dir / "training_history.json"
        trainer.load_history(history_path)
    
    # Create dataloader
    dataloader = create_dataloader(activations, args.batch_size)
    
    # Train
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.epochs,
        save_dir=output_dir,
        log_to_wandb=args.wandb,
        start_epoch=start_epoch
    )
    
    print(f"✅ SAE training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
