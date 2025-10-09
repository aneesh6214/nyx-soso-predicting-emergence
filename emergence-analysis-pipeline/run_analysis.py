#!/usr/bin/env python3
"""
Main entry point for emergence analysis pipeline.
Supports multiple experiment types through modular adapters.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sae import SparseAutoencoder, SAETrainer
from core.coactivation import CoActivationAnalyzer
from core.tracking import EmergenceTracker
from adapters.grokking_adapter import GrokkingAdapter


def load_adapter(experiment_type: str, config: Dict) -> 'BaseAdapter':
    """Load appropriate adapter for experiment type."""
    if experiment_type == "grokking":
        return GrokkingAdapter(config)
    elif experiment_type == "pythia":
        # Import only if needed
        from adapters.pythia_adapter import PythiaAdapter
        return PythiaAdapter(config)
    elif experiment_type == "custom":
        from adapters.generic_adapter import GenericAdapter
        return GenericAdapter(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def analyze_checkpoint(
    checkpoint: Any,
    checkpoint_name: str,
    adapter: 'BaseAdapter',
    sae_config: Dict,
    graph_config: Dict,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze a single checkpoint: extract activations, train SAE, build graph.
    
    Returns:
        Dictionary of metrics and results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint: {checkpoint_name}")
    print(f"{'='*60}")
    
    # Create checkpoint output directory
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract checkpoint info
    info = adapter.get_checkpoint_info(checkpoint)
    emergence_metric = adapter.get_emergence_metric(checkpoint)
    
    print(f"Step: {info.get('step', 'unknown')}")
    print(f"Emergence metric: {emergence_metric:.4f}")
    
    # Extract activations
    print("\n1. Extracting activations...")
    activations = adapter.extract_activations(
        checkpoint,
        layer_name=sae_config.get('layer_name'),
        num_samples=sae_config.get('num_samples', 1000)
    )
    print(f"   Extracted activations shape: {activations.shape}")
    
    # Train SAE
    print("\n2. Training Sparse Autoencoder...")
    sae_model = SparseAutoencoder(
        input_dim=activations.shape[1],
        hidden_dim=sae_config.get('hidden_dim', 512),
        sparsity_penalty=sae_config.get('sparsity_penalty', 0.01),
        tied_weights=sae_config.get('tied_weights', False)
    )
    
    trainer = SAETrainer(
        model=sae_model,
        learning_rate=sae_config.get('learning_rate', 1e-3)
    )
    
    history = trainer.train(
        activations,
        num_epochs=sae_config.get('num_epochs', 100),
        batch_size=sae_config.get('batch_size', 256),
        verbose=sae_config.get('verbose', False),
        seed=42  # Fixed seed for reproducible SAE training
    )
    
    # Save SAE model
    trainer.save_checkpoint(checkpoint_dir / 'sae_model.pt')
    print(f"   SAE trained. Final sparsity: {history['sparsity_rate'][-1]:.3f}")
    
    # Build co-activation graph
    print("\n3. Building co-activation graph...")
    analyzer = CoActivationAnalyzer(sae_model)
    
    # Compute co-activation matrix
    coact_matrix = analyzer.compute_coactivation_matrix(
        activations,
        threshold=graph_config.get('activation_threshold', 0.01),
        normalize=graph_config.get('normalize', True)
    )
    
    # Build graph
    graph = analyzer.build_coactivation_graph(
        edge_threshold=graph_config.get('edge_threshold', 0.1),
        min_degree=graph_config.get('min_degree', 1)
    )
    
    # Cluster features
    if graph.number_of_nodes() > 0:
        clusters = analyzer.cluster_features(
            n_clusters=min(graph_config.get('n_clusters', 10), graph.number_of_nodes()),
            method=graph_config.get('clustering_method', 'spectral')
        )
        print(f"   Found {len(clusters)} feature clusters")
    
    # Compute graph metrics
    metrics = analyzer.compute_graph_metrics()
    
    # Save visualizations
    print("\n4. Generating visualizations...")
    analyzer.visualize_coactivation_matrix(
        save_path=checkpoint_dir / 'coactivation_matrix.png'
    )
    analyzer.visualize_graph(
        save_path=checkpoint_dir / 'graph.png'
    )
    
    # Save results
    analyzer.save_results(checkpoint_dir)
    
    # Return metrics for tracking
    return {
        **info,
        **metrics,
        'test_acc': emergence_metric,
        'sae_sparsity': history['sparsity_rate'][-1]
    }


def main():
    parser = argparse.ArgumentParser(description="Emergence Analysis Pipeline")
    
    # Experiment selection
    parser.add_argument(
        "--experiment", 
        choices=["grokking", "pythia", "custom"],
        required=True,
        help="Type of experiment to analyze"
    )
    
    # Checkpoint specification
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory containing checkpoints (for grokking)"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="List of checkpoint files to analyze"
    )
    
    # Model specification (for Pythia)
    parser.add_argument("--model", help="Model name (for pythia)")
    parser.add_argument("--layer", help="Layer/position to analyze (e.g., position_0, position_1, position_2)")
    
    # SAE configuration
    parser.add_argument("--sae_features", type=int, default=512, 
                       help="Number of SAE features")
    parser.add_argument("--sae_sparsity", type=float, default=0.01,
                       help="SAE sparsity penalty")
    parser.add_argument("--sae_epochs", type=int, default=100,
                       help="SAE training epochs")
    
    # Graph configuration
    parser.add_argument("--edge_threshold", type=float, default=0.1,
                       help="Minimum co-activation for edges")
    parser.add_argument("--n_clusters", type=int, default=10,
                       help="Number of feature clusters")
    
    # Analysis options
    parser.add_argument("--track_evolution", action="store_true",
                       help="Track evolution across checkpoints")
    parser.add_argument("--predict_emergence", action="store_true",
                       help="Predict emergence points")
    
    # Output
    parser.add_argument("--output", default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = {
        'experiment': args.experiment,
        'model': args.model,
        'layer': args.layer,
        'p': 97 if args.experiment == "grokking" else None
    }
    
    sae_config = {
        'hidden_dim': args.sae_features,
        'sparsity_penalty': args.sae_sparsity,
        'num_epochs': args.sae_epochs,
        'layer_name': args.layer,  # Use layer argument directly
        'num_samples': 1000,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'tied_weights': False,
        'verbose': False
    }
    
    graph_config = {
        'activation_threshold': 0.01,
        'edge_threshold': args.edge_threshold,
        'min_degree': 1,
        'n_clusters': args.n_clusters,
        'clustering_method': 'spectral',
        'normalize': True
    }
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'config': config,
            'sae_config': sae_config,
            'graph_config': graph_config,
            'args': vars(args)
        }, f, indent=2)
    
    # Load adapter
    adapter = load_adapter(args.experiment, config)
    
    # Find checkpoints to analyze
    checkpoint_paths = []
    
    if args.checkpoints:
        # Explicit checkpoint list
        checkpoint_paths = [Path(p) for p in args.checkpoints]
    elif args.checkpoint_dir:
        # Directory of checkpoints
        ckpt_dir = Path(args.checkpoint_dir)
        if args.experiment == "grokking":
            # Look for pre/post grok checkpoints
            for pattern in ['checkpoint_pre_grok.pt', 'checkpoint_post_grok.pt', 'checkpoint_*.pt']:
                checkpoint_paths.extend(sorted(ckpt_dir.glob(pattern)))
        else:
            # General checkpoint pattern
            checkpoint_paths = sorted(ckpt_dir.glob('*.pt'))
    else:
        print("ERROR: Must specify either --checkpoint_dir or --checkpoints")
        return
    
    if not checkpoint_paths:
        print(f"ERROR: No checkpoints found")
        return
    
    print(f"Found {len(checkpoint_paths)} checkpoints to analyze")
    
    # Load checkpoints
    checkpoints = adapter.load_checkpoints(checkpoint_paths)
    
    # Create tracker for evolution analysis
    tracker = EmergenceTracker()
    
    # Analyze each checkpoint
    all_metrics = []
    for checkpoint_name, checkpoint in checkpoints.items():
        metrics = analyze_checkpoint(
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            adapter=adapter,
            sae_config=sae_config,
            graph_config=graph_config,
            output_dir=output_dir
        )
        
        all_metrics.append(metrics)
        
        # Add to tracker
        tracker.add_checkpoint(
            checkpoint_name=checkpoint_name,
            metrics=metrics,
            test_acc=metrics['test_acc'],
            step=metrics.get('step')
        )
    
    # Generate evolution analysis if requested
    if args.track_evolution and len(checkpoints) > 1:
        print(f"\n{'='*60}")
        print("Analyzing evolution across checkpoints...")
        print(f"{'='*60}")
        
        evolution_dir = output_dir / 'evolution_analysis'
        tracker.generate_report(evolution_dir)
        
        # Detect emergence points
        if args.predict_emergence:
            emergence_points = tracker.detect_emergence()
            if emergence_points:
                print(f"\n[FOUND] Detected emergence at checkpoints: {emergence_points}")
                for idx in emergence_points:
                    precursors = tracker.find_precursors(idx)
                    print(f"\nPrecursors for emergence at checkpoint {idx}:")
                    for key, value in precursors.items():
                        print(f"  {key}: {value:.4f}")
            else:
                print("\n[WARNING] No clear emergence points detected")
    
    print(f"\n[SUCCESS] Analysis complete! Results saved to: {output_dir}")
    print("\nView the report: open " + str(output_dir / 'evolution_analysis' / 'report.html'))


if __name__ == "__main__":
    main()
