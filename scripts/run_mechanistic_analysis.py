#!/usr/bin/env python3
"""
Master script to run complete mechanistic interpretability analysis pipeline.

Usage:
  python scripts/run_mechanistic_analysis.py --model pythia-410m-deduped --tokens 1000000
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete mechanistic analysis pipeline")
    parser.add_argument("--model", default="pythia-410m-deduped", help="Model name")
    parser.add_argument("--layer", type=int, default=6, help="Layer to analyze")
    parser.add_argument("--tokens", type=int, default=1000000, help="Number of tokens to process")
    parser.add_argument("--sae_features", type=int, default=8192, help="Number of SAE features")
    parser.add_argument("--sparsity", type=float, default=0.01, help="SAE sparsity penalty")
    parser.add_argument("--epochs", type=int, default=1000, help="SAE training epochs")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of feature clusters")
    parser.add_argument("--output_base", default="data", help="Base output directory")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip activation extraction")
    parser.add_argument("--skip_sae", action="store_true", help="Skip SAE training")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip co-activation analysis")
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path(args.output_base)
    activations_dir = base_dir / "activations"
    sae_dir = base_dir / "sae"
    analysis_dir = base_dir / "coactivation_analysis"
    
    print("🔬 Starting Mechanistic Interpretability Analysis Pipeline")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Tokens: {args.tokens:,}")
    print(f"SAE Features: {args.sae_features}")
    print(f"Output Base: {base_dir}")
    
    # Step 1: Extract activations
    if not args.skip_extraction:
        success = run_command([
            sys.executable, "scripts/extract_activations.py",
            "--model", args.model,
            "--layer", str(args.layer),
            "--tokens", str(args.tokens),
            "--output", str(activations_dir)
        ], "Extracting activations")
        
        if not success:
            print("❌ Activation extraction failed. Stopping pipeline.")
            return
    
    # Step 2: Train SAE
    if not args.skip_sae:
        activations_file = activations_dir / "activations.pt"
        if not activations_file.exists():
            print(f"❌ Activations file not found: {activations_file}")
            return
        
        success = run_command([
            sys.executable, "scripts/train_sae.py",
            "--activations", str(activations_file),
            "--features", str(args.sae_features),
            "--sparsity", str(args.sparsity),
            "--epochs", str(args.epochs),
            "--output", str(sae_dir)
        ], "Training Sparse Autoencoder")
        
        if not success:
            print("❌ SAE training failed. Stopping pipeline.")
            return
    
    # Step 3: Analyze co-activations
    if not args.skip_analysis:
        sae_model_file = sae_dir / "best_model.pt"
        activations_file = activations_dir / "activations.pt"
        
        if not sae_model_file.exists():
            print(f"❌ SAE model file not found: {sae_model_file}")
            return
        
        if not activations_file.exists():
            print(f"❌ Activations file not found: {activations_file}")
            return
        
        success = run_command([
            sys.executable, "scripts/coactivation_graphs.py",
            "--sae_model", str(sae_model_file),
            "--activations", str(activations_file),
            "--n_clusters", str(args.n_clusters),
            "--output", str(analysis_dir)
        ], "Analyzing co-activation patterns")
        
        if not success:
            print("❌ Co-activation analysis failed.")
            return
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Results saved to: {base_dir}")
    print(f"   ├── Activations: {activations_dir}")
    print(f"   ├── SAE Model: {sae_dir}")
    print(f"   └── Analysis: {analysis_dir}")
    print(f"\n📊 Key files:")
    print(f"   ├── {activations_dir}/activations.pt")
    print(f"   ├── {sae_dir}/best_model.pt")
    print(f"   ├── {sae_dir}/training_curves.png")
    print(f"   ├── {analysis_dir}/coactivation_matrix.png")
    print(f"   ├── {analysis_dir}/coactivation_graph.png")
    print(f"   └── {analysis_dir}/cluster_analysis.png")
    print(f"\n🔬 Next steps:")
    print(f"   1. Examine training curves in {sae_dir}/training_curves.png")
    print(f"   2. Analyze co-activation patterns in {analysis_dir}/")
    print(f"   3. Identify interpretable feature clusters")
    print(f"   4. Design ablation experiments")


if __name__ == "__main__":
    main()
