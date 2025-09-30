#!/usr/bin/env python3
"""
Analyze co-activation patterns in SAE features to identify modular circuits.

Usage:
  python scripts/coactivation_graphs.py --sae_model data/sae/best_model.pt --activations data/activations/activations.pt
"""

import argparse
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

# Import SparseAutoencoder from train_sae.py
from train_sae import SparseAutoencoder


class CoActivationAnalyzer:
    """Analyze co-activation patterns in SAE features."""
    
    def __init__(self, sae_model_path: Path, device: str = "cuda"):
        self.device = device
        self.sae_model = self._load_sae_model(sae_model_path)
        self.coactivation_matrix = None
        self.graph = None
        self.feature_clusters = None
    
    def _load_sae_model(self, model_path: Path):
        """Load trained SAE model."""
        print(f"Loading SAE model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model architecture
        model = SparseAutoencoder(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            sparsity_penalty=checkpoint["sparsity_penalty"]
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        print(f"SAE model loaded: {checkpoint['input_dim']} -> {checkpoint['hidden_dim']}")
        return model
    
    def compute_coactivation_matrix(
        self,
        activations: torch.Tensor,
        threshold: float = 0.01,
        batch_size: int = 1000,
        output_dir: Path = None
    ) -> np.ndarray:
        """Compute co-activation matrix between SAE features."""
        # Check if matrix already exists
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = output_dir / "coactivation_matrix.npy"
            
            if matrix_path.exists():
                print(f"Loading existing co-activation matrix from: {matrix_path}")
                coactivation_matrix = np.load(matrix_path)
                self.coactivation_matrix = coactivation_matrix
                print("✅ Co-activation matrix loaded successfully!")
                return coactivation_matrix
        
        print("Computing co-activation matrix...")
        
        with torch.no_grad():
            # Get SAE features for all activations
            all_features = []
            
            for i in range(0, len(activations), batch_size):
                batch = activations[i:i + batch_size].to(self.device)
                # Flatten to match SAE input dim used during training
                if batch.dim() > 2:
                    batch = batch.view(batch.size(0), -1)
                # Defensive check: ensure dims match SAE input
                expected_dim = getattr(self.sae_model, "input_dim", None)
                if expected_dim is not None and batch.size(1) != expected_dim:
                    raise ValueError(
                        f"Activation dim mismatch: got {batch.size(1)}, expected {expected_dim}. "
                        f"Ensure train_sae.py flattened activations the same way before training."
                    )
                _, features = self.sae_model(batch)
                all_features.append(features.cpu())
            
            # Concatenate all features
            features = torch.cat(all_features, dim=0)
            print(f"SAE features shape: {features.shape}")
            
            # Debug: Check feature activation statistics
            print(f"Features min: {features.min().item():.6f}")
            print(f"Features max: {features.max().item():.6f}")
            print(f"Features mean: {features.mean().item():.6f}")
            print(f"Non-zero features: {torch.count_nonzero(features).item()}")
            print(f"Features > 0: {torch.count_nonzero(features > 0).item()}")
            print(f"Features > 0.1: {torch.count_nonzero(features > 0.1).item()}")
            
            # Compute co-activation matrix
            # P(feature_i | feature_j) = P(feature_i AND feature_j) / P(feature_j)
            num_features = features.shape[1]
            if threshold is not None and hasattr(self, 'max_features') and self.max_features is not None:
                num_features = min(num_features, self.max_features)
                features = features[:, :num_features]
                print(f"Using top {num_features} features for co-activation analysis")
            
            coactivation_matrix = np.zeros((num_features, num_features))
            
            # Compute pairwise co-activations
            for i in tqdm(range(num_features), desc="Computing co-activations"):
                for j in range(num_features):
                    if i != j:
                        # Features are active when > 0
                        feature_i_active = (features[:, i] > 0).float()
                        feature_j_active = (features[:, j] > 0).float()
                        
                        # P(feature_i | feature_j)
                        joint_active = feature_i_active * feature_j_active
                        p_joint = torch.mean(joint_active)
                        p_j = torch.mean(feature_j_active)
                        
                        if p_j > 0:
                            coactivation_matrix[i, j] = (p_joint / p_j).item()
            
            self.coactivation_matrix = coactivation_matrix
            
            # Save the computed matrix
            if output_dir is not None:
                np.save(matrix_path, coactivation_matrix)
                print(f"✅ Co-activation matrix saved to: {matrix_path}")
            
            return coactivation_matrix
    
    def build_coactivation_graph(
        self,
        threshold: float = 0.1,
        min_edges: int = 5
    ) -> nx.Graph:
        """Build graph from co-activation matrix."""
        if self.coactivation_matrix is None:
            raise ValueError("Must compute co-activation matrix first")
        
        print(f"Building co-activation graph with threshold {threshold}...")
        
        # Debug: Check matrix statistics
        matrix = self.coactivation_matrix
        print(f"Matrix shape: {matrix.shape}")
        print(f"Matrix min: {matrix.min():.6f}")
        print(f"Matrix max: {matrix.max():.6f}")
        print(f"Matrix mean: {matrix.mean():.6f}")
        print(f"Non-zero values: {np.count_nonzero(matrix)}")
        print(f"Values > 0.001: {np.count_nonzero(matrix > 0.001)}")
        print(f"Values > 0.01: {np.count_nonzero(matrix > 0.01)}")
        print(f"Values > 0.1: {np.count_nonzero(matrix > 0.1)}")
        
        G = nx.Graph()
        
        # Add nodes (SAE features)
        num_features = self.coactivation_matrix.shape[0]
        G.add_nodes_from(range(num_features))
        
        # Add edges based on co-activation threshold
        edges_added = 0
        for i in range(num_features):
            for j in range(i + 1, num_features):
                coactivation = self.coactivation_matrix[i, j]
                if coactivation > threshold:
                    G.add_edge(i, j, weight=coactivation)
                    edges_added += 1
        
        print(f"Graph built: {G.number_of_nodes()} nodes, {edges_added} edges")
        
        # Filter out isolated nodes
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        print(f"Removed {len(isolated_nodes)} isolated nodes")
        
        self.graph = G
        return G
    
    def cluster_features(
        self,
        method: str = "spectral",
        n_clusters: int = 10,
        min_cluster_size: int = 5
    ) -> Dict[int, List[int]]:
        """Cluster features based on co-activation patterns."""
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        num_nodes = self.graph.number_of_nodes()
        if num_nodes == 0:
            raise ValueError("Co-activation graph has 0 nodes after filtering; try lowering the threshold.")
        if num_nodes < 2:
            print("[WARN] Graph has <2 nodes; skipping clustering.")
            self.feature_clusters = {0: list(self.graph.nodes())}
            return self.feature_clusters
        
        print(f"Clustering features using {method}... (nodes={num_nodes})")
        
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph).toarray()
        
        # If requested clusters exceed nodes, reduce automatically
        if n_clusters > num_nodes:
            print(f"[WARN] n_clusters={n_clusters} > nodes={num_nodes}. Using n_clusters={num_nodes}.")
            n_clusters = num_nodes
        if method == "spectral" and n_clusters < 2:
            print("[WARN] spectral clustering needs at least 2 clusters; falling back to DBSCAN.")
            method = "dbscan"
        
        if method == "spectral":
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            cluster_labels = clustering.fit_predict(adj_matrix)
        
        elif method == "dbscan":
            clustering = DBSCAN(
                eps=0.1,
                min_samples=min_cluster_size,
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(adj_matrix)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Group features by cluster
        clusters = {}
        for feature_idx, cluster_id in enumerate(cluster_labels):
            clusters.setdefault(int(cluster_id), []).append(feature_idx)
        
        # Filter out small clusters (keep noise cluster -1 separate)
        filtered_clusters = {
            cid: feats
            for cid, feats in clusters.items()
            if (cid == -1) or (len(feats) >= min_cluster_size)
        }
        
        print(f"Found {len(filtered_clusters)} clusters (including noise if any)")
        for cluster_id, features in filtered_clusters.items():
            print(f"  Cluster {cluster_id}: {len(features)} features")
        
        self.feature_clusters = filtered_clusters
        return filtered_clusters
    
    def analyze_cluster_properties(
        self,
        activations: torch.Tensor,
        cluster_id: int
    ) -> Dict[str, float]:
        """Analyze properties of a feature cluster."""
        if self.feature_clusters is None:
            raise ValueError("Must cluster features first")
        
        cluster_features = self.feature_clusters[cluster_id]
        
        with torch.no_grad():
            # Get SAE features
            _, features = self.sae_model(activations.to(self.device))
            features = features.cpu()
            
            # Analyze cluster
            cluster_activations = features[:, cluster_features]
            
            properties = {
                "num_features": len(cluster_features),
                "avg_activation": torch.mean(cluster_activations).item(),
                "activation_rate": torch.mean((cluster_activations > 0).float()).item(),
                "max_activation": torch.max(cluster_activations).item(),
                "std_activation": torch.std(cluster_activations).item()
            }
            
            return properties
    
    def visualize_coactivation_matrix(self, output_path: Path, max_features: int = 100):
        """Visualize co-activation matrix."""
        if self.coactivation_matrix is None:
            raise ValueError("Must compute co-activation matrix first")
        
        # Subsample for visualization
        if self.coactivation_matrix.shape[0] > max_features:
            indices = np.random.choice(
                self.coactivation_matrix.shape[0],
                max_features,
                replace=False
            )
            matrix_subset = self.coactivation_matrix[np.ix_(indices, indices)]
        else:
            matrix_subset = self.coactivation_matrix
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix_subset,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Co-activation Rate'}
        )
        plt.title("SAE Feature Co-activation Matrix")
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Index")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_graph(self, output_path: Path, layout: str = "spring"):
        """Visualize co-activation graph."""
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        print(f"Visualizing graph with {self.graph.number_of_nodes()} nodes...")
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.random_layout(self.graph)
        
        # Create plot
        plt.figure(figsize=(15, 12))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=20,
            node_color='lightblue',
            alpha=0.7
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.5,
            edge_color='gray',
            width=0.5
        )
        
        plt.title("SAE Feature Co-activation Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_clusters(self, output_path: Path):
        """Visualize feature clusters."""
        if self.feature_clusters is None:
            raise ValueError("Must cluster features first")
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Cluster sizes
        cluster_sizes = [len(features) for features in self.feature_clusters.values()]
        axes[0, 0].bar(range(len(cluster_sizes)), cluster_sizes)
        axes[0, 0].set_title("Cluster Sizes")
        axes[0, 0].set_xlabel("Cluster ID")
        axes[0, 0].set_ylabel("Number of Features")
        
        # Plot 2: Co-activation distribution
        if self.coactivation_matrix is not None:
            coactivations = self.coactivation_matrix[
                self.coactivation_matrix > 0
            ].flatten()
            axes[0, 1].hist(coactivations, bins=50, alpha=0.7)
            axes[0, 1].set_title("Co-activation Distribution")
            axes[0, 1].set_xlabel("Co-activation Rate")
            axes[0, 1].set_ylabel("Frequency")
        
        # Plot 3: Graph properties
        if self.graph is not None:
            degrees = [d for n, d in self.graph.degree()]
            axes[1, 0].hist(degrees, bins=20, alpha=0.7)
            axes[1, 0].set_title("Node Degree Distribution")
            axes[1, 0].set_xlabel("Degree")
            axes[1, 0].set_ylabel("Frequency")
        
        # Plot 4: Cluster connectivity
        if self.graph is not None:
            # Compute average clustering coefficient
            clustering_coeffs = nx.clustering(self.graph)
            avg_clustering = np.mean(list(clustering_coeffs.values()))
            axes[1, 1].text(0.5, 0.5, f"Avg Clustering\nCoefficient: {avg_clustering:.3f}",
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title("Graph Properties")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: Path):
        """Save analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save co-activation matrix
        if self.coactivation_matrix is not None:
            np.save(output_dir / "coactivation_matrix.npy", self.coactivation_matrix)
        
        # Save graph
        if self.graph is not None:
            nx.write_gml(self.graph, output_dir / "coactivation_graph.gml")
        
        # Save clusters
        if self.feature_clusters is not None:
            with open(output_dir / "feature_clusters.json", "w") as f:
                json.dump(self.feature_clusters, f, indent=2)
        
        # Save summary
        summary = {
            "num_features": self.coactivation_matrix.shape[0] if self.coactivation_matrix is not None else 0,
            "num_edges": self.graph.number_of_edges() if self.graph is not None else 0,
            "num_clusters": len(self.feature_clusters) if self.feature_clusters is not None else 0,
            "graph_density": nx.density(self.graph) if self.graph is not None else 0
        }
        
        with open(output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE feature co-activations")
    parser.add_argument("--sae_model", required=True, help="Path to trained SAE model")
    parser.add_argument("--activations", required=True, help="Path to activations.pt file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Co-activation threshold (default: 0.1)")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters (default: 10)")
    parser.add_argument("--max_features", type=int, default=None, help="Maximum number of features to analyze (default: all)")
    parser.add_argument("--output", default="data/coactivation_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load activations
    print(f"Loading activations from: {args.activations}")
    activations = torch.load(args.activations)
    
    # Create analyzer
    analyzer = CoActivationAnalyzer(Path(args.sae_model))
    
    # Set max features if specified
    if args.max_features is not None:
        analyzer.max_features = args.max_features
    
    # Compute co-activation matrix
    analyzer.compute_coactivation_matrix(activations, output_dir=output_dir)
    
    # Build graph
    analyzer.build_coactivation_graph(threshold=args.threshold)
    
    # Cluster features
    analyzer.cluster_features(n_clusters=args.n_clusters)
    
    # Visualize results
    analyzer.visualize_coactivation_matrix(output_dir / "coactivation_matrix.png")
    analyzer.visualize_graph(output_dir / "coactivation_graph.png")
    analyzer.visualize_clusters(output_dir / "cluster_analysis.png")
    
    # Save results
    analyzer.save_results(output_dir)
    
    print("✅ Co-activation analysis complete!")


if __name__ == "__main__":
    main()
