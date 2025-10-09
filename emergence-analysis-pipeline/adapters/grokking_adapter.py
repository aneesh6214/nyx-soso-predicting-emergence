"""
Adapter for grokking experiments on modular arithmetic.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from .base_adapter import BaseAdapter


class GrokkingAdapter(BaseAdapter):
    """Adapter for grokking experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.p = config.get('p', 97)  # Prime modulus
        self.vocab_size = self.p
        
    def load_checkpoints(self, checkpoint_paths: List[Path]) -> Dict[str, Any]:
        """Load grokking model checkpoints."""
        checkpoints = {}
        
        for path in checkpoint_paths:
            if not path.exists():
                print(f"Warning: Checkpoint not found: {path}")
                continue
                
            checkpoint = torch.load(path, map_location=self.device)
            name = path.stem  # e.g., "checkpoint_pre_grok" or "checkpoint_post_grok"
            
            # Reconstruct model if needed
            if 'model_state_dict' in checkpoint:
                model = self._create_model(checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                checkpoint['model'] = model
            
            checkpoints[name] = checkpoint
            
        return checkpoints
    
    def _create_model(self, checkpoint: Dict) -> nn.Module:
        """Recreate model architecture from checkpoint."""
        # Define a simple transformer model matching the grokking experiment
        class TinyTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
                super().__init__()
                self.d_model = d_model
                
                # Token embedding (for a, b values) - matching grok_clean.py
                self.embedding = nn.Embedding(vocab_size, d_model)
                
                # Positional encoding (learned)
                self.pos_embedding = nn.Embedding(3, d_model)  # positions 0, 1, 2
                
                # Transformer blocks - matching exact structure
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                
                # Output head - matching name
                self.output = nn.Linear(d_model, vocab_size)
            
            def forward(self, x):
                # x shape: (batch, seq_len=2)
                batch_size, seq_len = x.shape
                
                # Token embeddings
                x_embed = self.embedding(x)
                
                # Add positional embeddings
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                pos_embed = self.pos_embedding(positions)
                x = x_embed + pos_embed
                
                # Pass through transformer
                x = self.transformer(x)
                
                # Output projection
                return self.output(x)
        
        # Extract model config from checkpoint or use defaults
        model = TinyTransformer(
            vocab_size=self.vocab_size,
            d_model=checkpoint.get('d_model', 128),
            n_heads=checkpoint.get('n_heads', 4),
            n_layers=checkpoint.get('n_layers', 2),
            d_ff=checkpoint.get('d_ff', 256)
        )
        return model.to(self.device)
    
    def extract_activations(
        self, 
        checkpoint: Any, 
        layer_name: Optional[str] = None,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """Extract activations using pre-saved activations from the grokking run."""
        
        # Determine which saved activations to use based on checkpoint step
        checkpoint_step = int(checkpoint.get('step', 0))
        
        # Try to find activation file for this specific step
        activations_file = Path(f'runs/grokking_final/activations/activations_step_{checkpoint_step:05d}.pt')
        
        # Fallback to pre/post grok if step-specific file doesn't exist
        if not activations_file.exists():
            if checkpoint_step <= 21000:
                activations_file = Path('runs/grokking_final/activations/activations_pre_grok.pt')
            else:
                activations_file = Path('runs/grokking_final/activations/activations_post_grok.pt')
        
        if not activations_file.exists():
            raise ValueError(f"Saved activations not found: {activations_file}")
        
        print(f"   Using saved activations: {activations_file.name}")
        
        # Load the saved activations
        saved_data = torch.load(activations_file, map_location='cpu')
        
        # Extract hidden states - shape: (batch, seq_len, d_model)
        if 'hidden_states' in saved_data:
            hidden_states = saved_data['hidden_states']
            
            # Analyze based on layer_name parameter
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, d_model = hidden_states.shape
                
                # Determine which position to analyze based on layer_name
                if layer_name == "position_0":
                    activations = hidden_states[:, 0, :]  # Token 'a' processing
                    print(f"   Analyzing position 0 (token a)")
                elif layer_name == "position_1": 
                    activations = hidden_states[:, 1, :]  # Token 'b' processing
                    print(f"   Analyzing position 1 (token b)")
                elif layer_name == "position_2":
                    activations = hidden_states[:, 2, :]  # Prediction step
                    print(f"   Analyzing position 2 (prediction)")
                else:
                    # Default: average over all positions
                    activations = hidden_states.mean(dim=1)
                    print(f"   Analyzing averaged activations across all positions")
            else:
                activations = hidden_states
            
            # Subsample if requested
            if num_samples < activations.shape[0]:
                # Use deterministic sampling for consistency
                indices = torch.linspace(0, activations.shape[0]-1, num_samples).long()
                activations = activations[indices]
            
            return activations.float()
        else:
            raise ValueError("No hidden_states found in saved activations")
    
    def _generate_inputs(self, num_samples: int) -> torch.Tensor:
        """Generate random modular addition problems."""
        # Generate random (a, b) pairs
        a_values = torch.randint(0, self.p, (num_samples,))
        b_values = torch.randint(0, self.p, (num_samples,))
        
        # Stack into input format expected by model
        inputs = torch.stack([a_values, b_values], dim=1).to(self.device)
        return inputs
    
    def get_checkpoint_info(self, checkpoint: Any) -> Dict[str, Any]:
        """Extract training step and accuracies from checkpoint."""
        return {
            'step': checkpoint.get('step', 0),
            'train_acc': checkpoint.get('train_acc', 0.0),
            'test_acc': checkpoint.get('test_acc', 0.0),
            'val_acc': checkpoint.get('test_acc', 0.0),  # Alias
        }
    
    def get_emergence_metric(self, checkpoint: Any) -> float:
        """For grokking, emergence metric = test accuracy."""
        info = self.get_checkpoint_info(checkpoint)
        return info.get('test_acc', 0.0)
    
    def get_layer_names(self) -> List[str]:
        """Available positions for activation extraction."""
        return ["position_0", "position_1", "position_2", "default"]
