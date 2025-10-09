"""
Base adapter interface for different experiment types.
All experiment-specific adapters must inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch


class BaseAdapter(ABC):
    """Abstract base class for experiment adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adapter with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def load_checkpoints(self, checkpoint_paths: List[Path]) -> Dict[str, Any]:
        """
        Load model checkpoints from experiment.
        
        Returns:
            Dictionary mapping checkpoint names to loaded models/states
        """
        pass
    
    @abstractmethod
    def extract_activations(
        self, 
        checkpoint: Any, 
        layer_name: Optional[str] = None,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Extract activations from a specific layer.
        
        Args:
            checkpoint: Loaded checkpoint/model
            layer_name: Name of layer to extract from (None = default layer)
            num_samples: Number of samples to extract
            
        Returns:
            Tensor of shape (num_samples, feature_dim)
        """
        pass
    
    @abstractmethod
    def get_checkpoint_info(self, checkpoint: Any) -> Dict[str, Any]:
        """
        Extract metadata from checkpoint (step, accuracy, etc).
        
        Returns:
            Dictionary with checkpoint information
        """
        pass
    
    @abstractmethod
    def get_emergence_metric(self, checkpoint: Any) -> float:
        """
        Get emergence metric for this checkpoint.
        For grokking: validation accuracy
        For language models: perplexity or downstream task performance
        
        Returns:
            Float metric indicating emergence level
        """
        pass
    
    def get_layer_names(self) -> List[str]:
        """
        Get list of available layers for activation extraction.
        Override if experiment has specific layers of interest.
        """
        return ["default"]
    
    def prepare_inputs(self, checkpoint: Any) -> torch.Tensor:
        """
        Prepare input data for activation extraction.
        Override for experiment-specific input generation.
        """
        # Default: return random inputs
        # Subclasses should override with meaningful inputs
        return torch.randn(100, 512, device=self.device)
