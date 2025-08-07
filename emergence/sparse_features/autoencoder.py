"""
Sparse autoencoder model for feature extraction (placeholder).

This module will eventually provide a PyTorch nn.Module implementing a sparse
undercomplete or overcomplete autoencoder, along with training helpers to use
it on activation tensors.
"""

from __future__ import annotations

import torch
from torch import nn


class SparseAutoencoder(nn.Module):  # noqa: D101 – skeleton
    def __init__(self, input_dim: int, hidden_dim: int, *, l1_lambda: float = 1e-3):  # noqa: E501
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.l1_lambda = l1_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass (placeholder)."""
        raise NotImplementedError

    def loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute reconstruction + sparsity loss (placeholder)."""
        raise NotImplementedError
