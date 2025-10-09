"""
Sparse Autoencoder implementation for feature extraction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
from pathlib import Path
import json


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with L1 regularization."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        sparsity_penalty: float = 0.01,
        tied_weights: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_penalty = sparsity_penalty
        self.tied_weights = tied_weights
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Decoder
        if tied_weights:
            # Use transposed encoder weights
            self.decoder = None
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        hidden = self.encoder(x) + self.encoder_bias
        # Apply ReLU for sparsity
        return torch.relu(hidden)
    
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        if self.tied_weights:
            # Use transposed encoder weights
            reconstructed = torch.matmul(hidden, self.encoder.weight) + self.decoder_bias
        else:
            reconstructed = self.decoder(hidden) + self.decoder_bias
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparsity regularization."""
        hidden = self.encode(x)
        reconstructed = self.decode(hidden)
        return reconstructed, hidden
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for input."""
        with torch.no_grad():
            return self.encode(x)
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstructed: torch.Tensor, 
        hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction and sparsity losses."""
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, x)
        
        # Sparsity loss (L1 on activations)
        sparsity_loss = torch.mean(torch.abs(hidden))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_penalty * sparsity_loss
        
        # Compute sparsity metrics
        sparsity_rate = torch.mean((hidden > 0).float())
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'sparsity_rate': sparsity_rate
        }


class SAETrainer:
    """Trainer for Sparse Autoencoder."""
    
    def __init__(
        self,
        model: SparseAutoencoder,
        learning_rate: float = 1e-3,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Forward pass
        reconstructed, hidden = self.model(batch)
        
        # Compute losses
        losses = self.model.compute_loss(batch, reconstructed, hidden)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Convert to Python floats for logging
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        activations: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
        seed: int = 42
    ) -> Dict[str, list]:
        """Train SAE on activations."""
        import random
        import numpy as np
        from torch.utils.data import DataLoader, TensorDataset
        
        # Set seeds for reproducible SAE training
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Create dataloader
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {
            'total_loss': [],
            'recon_loss': [],
            'sparsity_loss': [],
            'sparsity_rate': []
        }
        
        for epoch in range(num_epochs):
            epoch_losses = {k: 0.0 for k in history.keys()}
            num_batches = 0
            
            for batch in dataloader:
                batch = batch[0].to(self.device)
                losses = self.train_step(batch)
                
                for k, v in losses.items():
                    if k in epoch_losses:
                        epoch_losses[k] += v
                num_batches += 1
            
            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= num_batches
                history[k].append(epoch_losses[k])
            
            # Update learning rate
            self.scheduler.step(epoch_losses['total_loss'])
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={epoch_losses['total_loss']:.4f}, "
                      f"Sparsity={epoch_losses['sparsity_rate']:.3f}")
        
        return history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'sparsity_penalty': self.model.sparsity_penalty,
            'tied_weights': self.model.tied_weights
        }
        torch.save(checkpoint, path)
        
        # Save config as JSON for easy inspection
        config = {
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'sparsity_penalty': self.model.sparsity_penalty,
            'tied_weights': self.model.tied_weights
        }
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cuda"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = SparseAutoencoder(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            sparsity_penalty=checkpoint['sparsity_penalty'],
            tied_weights=checkpoint.get('tied_weights', False)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        trainer = cls(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return trainer
