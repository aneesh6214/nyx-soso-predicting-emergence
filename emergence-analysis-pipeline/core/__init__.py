"""Core modules for emergence analysis pipeline."""

from .sae import SparseAutoencoder, SAETrainer
from .coactivation import CoActivationAnalyzer
from .tracking import EmergenceTracker

__all__ = [
    'SparseAutoencoder',
    'SAETrainer', 
    'CoActivationAnalyzer',
    'EmergenceTracker'
]
