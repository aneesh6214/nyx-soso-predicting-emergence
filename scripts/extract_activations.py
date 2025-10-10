#!/usr/bin/env python3
"""
Extract activations from Pythia 410M for mechanistic interpretability analysis.

Usage:
  python scripts/extract_activations.py --model pythia-410m-deduped --layer 6 --tokens 1000000 --output data/activations
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Optional

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("ERROR: transformer-lens not installed. Run: pip install transformer-lens")
    exit(1)


def load_model(model_name: str) -> HookedTransformer:
    """Load Pythia model with transformer-lens."""
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32  # Use float32 for better precision
    )
    print(f"Model loaded. Device: {model.cfg.device}")
    return model


def generate_text_samples(num_tokens: int, batch_size: int = 1000) -> List[str]:
    """Generate diverse text samples for activation extraction."""
    # Use a mix of text sources for diverse activations
    samples = []
    
    # Common English text patterns
    base_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is a subset of artificial intelligence. ",
        "The weather today is sunny with a chance of rain. ",
        "Mathematics is the language of the universe. ",
        "Technology continues to evolve at a rapid pace. ",
        "Research shows that neural networks can learn complex patterns. ",
        "The human brain contains billions of neurons. ",
        "Data science combines statistics, programming, and domain knowledge. ",
    ]
    
    # Generate enough text to reach target token count
    current_tokens = 0
    while current_tokens < num_tokens:
        for base_text in base_texts:
            if current_tokens >= num_tokens:
                break
            samples.append(base_text)
            current_tokens += len(base_text.split())
    
    return samples[:num_tokens]


def extract_activations(
    model: HookedTransformer,
    texts: List[str],
    layer: int,
    batch_size: int = 32
) -> Dict[str, torch.Tensor]:
    """Extract activations from specified layer."""
    print(f"Extracting activations from layer {layer}...")
    
    all_activations = []
    all_tokens = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize and get activations
        tokens = model.to_tokens(batch_texts)
        _, cache = model.run_with_cache(tokens)
        
        # Extract activations from specified layer
        layer_key = f"blocks.{layer}.mlp.hook_post"
        if layer_key in cache:
            activations = cache[layer_key]
            all_activations.append(activations.detach().cpu())
            all_tokens.append(tokens.detach().cpu())
        else:
            print(f"Warning: Layer key {layer_key} not found in cache")
            print(f"Available keys: {list(cache.keys())}")
    
    # Concatenate all batches
    if all_activations:
        all_activations = torch.cat(all_activations, dim=0)
        all_tokens = torch.cat(all_tokens, dim=0)
        
        return {
            "activations": all_activations,
            "tokens": all_tokens,
            "layer": layer,
            "model_name": model.cfg.model_name,
            "num_samples": len(texts),
            "activation_shape": all_activations.shape
        }
    else:
        raise ValueError("No activations extracted")


def save_activations(data: Dict[str, torch.Tensor], output_path: Path):
    """Save activations and metadata."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save activations tensor
    torch.save(data["activations"], output_path / "activations.pt")
    torch.save(data["tokens"], output_path / "tokens.pt")
    
    # Save metadata
    metadata = {
        "layer": data["layer"],
        "model_name": data["model_name"],
        "num_samples": data["num_samples"],
        "activation_shape": list(data["activation_shape"]),
        "dtype": str(data["activations"].dtype),
        "device": str(data["activations"].device)
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved activations to: {output_path}")
    print(f"Activation shape: {data['activation_shape']}")
    print(f"Total tokens: {data['num_samples']}")


def main():
    parser = argparse.ArgumentParser(description="Extract activations from Pythia model")
    parser.add_argument("--model", default="pythia-410m-deduped", 
                       help="Model name (default: pythia-410m-deduped)")
    parser.add_argument("--layer", type=int, default=6, 
                       help="Layer to extract activations from (default: 6)")
    parser.add_argument("--tokens", type=int, default=100000, 
                       help="Number of tokens to process (default: 100000)")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for processing (default: 32)")
    parser.add_argument("--output", default="data/activations", 
                       help="Output directory (default: data/activations)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.model)
    
    # Generate text samples
    print(f"Generating {args.tokens} tokens of text...")
    texts = generate_text_samples(args.tokens)
    
    # Extract activations
    activations_data = extract_activations(
        model, texts, args.layer, args.batch_size
    )
    
    # Save results
    save_activations(activations_data, output_path)
    
    print("✅ Activation extraction complete!")


if __name__ == "__main__":
    main()
