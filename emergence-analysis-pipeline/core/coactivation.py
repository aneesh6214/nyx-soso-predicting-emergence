"""
Co-activation graph analysis for SAE features.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns


class CoActivationAnalyzer:
    """Analyze co-activation patterns in SAE features."""
    
    def __init__(
        self,
        sae_model: 'SparseAutoencoder',
        device: str = "cuda"
    ):
        self.sae_model = sae_model.to(device)
        self.device = device
        self.coactivation_matrix = None
        self.graph = None
        self.feature_clusters = None
    
    def compute_coactivation_matrix(
        self,
        activations: torch.Tensor,
        threshold: float = 0.01,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute co-activation matrix from raw activations.
        
        Args:
            activations: Raw activations (before SAE)
            threshold: Minimum activation threshold to consider "active"
            normalize: Whether to normalize co-activations
        
        Returns:
            Co-activation matrix of shape (n_features, n_features)
        """
        print("Computing SAE features from activations...")
        
        # Get SAE features
        with torch.no_grad():
            features = self.sae_model.get_feature_activations(activations.to(self.device))
        
        # Threshold to get binary activation matrix
        active = (features > threshold).float()
        
        # Compute co-activation counts
        n_samples = active.shape[0]
        coactivation = torch.matmul(active.T, active) / n_samples
        
        if normalize:
            # Normalize by individual activation rates
            activation_rates = torch.mean(active, dim=0, keepdim=True)
            normalization = torch.sqrt(activation_rates.T @ activation_rates)
            normalization[normalization == 0] = 1.0  # Avoid division by zero
            coactivation = coactivation / normalization
        
        self.coactivation_matrix = coactivation.cpu().numpy()
        return self.coactivation_matrix
    
    def build_coactivation_graph(
        self,
        edge_threshold: float = 0.1,
        min_degree: int = 1
    ) -> nx.Graph:
        """
        Build NetworkX graph from co-activation matrix.
        
        Args:
            edge_threshold: Minimum co-activation to create edge
            min_degree: Minimum node degree to keep in graph
        
        Returns:
            NetworkX graph
        """
        if self.coactivation_matrix is None:
            raise ValueError("Must compute co-activation matrix first")
        
        G = nx.Graph()
        n_features = self.coactivation_matrix.shape[0]
        
        # Add all nodes
        G.add_nodes_from(range(n_features))
        
        # Add edges based on threshold
        for i in range(n_features):
            for j in range(i + 1, n_features):
                weight = self.coactivation_matrix[i, j]
                if weight > edge_threshold:
                    G.add_edge(i, j, weight=float(weight))
        
        # Remove low-degree nodes if specified
        if min_degree > 0:
            low_degree_nodes = [n for n, d in G.degree() if d < min_degree]
            G.remove_nodes_from(low_degree_nodes)
            print(f"Removed {len(low_degree_nodes)} nodes with degree < {min_degree}")
        
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        self.graph = G
        return G
    
    def cluster_features(
        self,
        n_clusters: int = 10,
        method: str = "spectral"
    ) -> Dict[int, List[int]]:
        """
        Cluster features based on co-activation patterns.
        
        Returns:
            Dictionary mapping cluster_id to list of feature indices
        """
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        # Get adjacency matrix from graph
        adjacency = nx.adjacency_matrix(self.graph).toarray()
        
        if method == "spectral":
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            labels = clustering.fit_predict(adjacency)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Group features by cluster
        clusters = {}
        for node, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)
        
        self.feature_clusters = clusters
        return clusters
    
    def compute_graph_metrics(self) -> Dict[str, float]:
        """
        Compute various graph metrics for emergence detection.
        
        Returns:
            Dictionary of metrics
        """
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        G = self.graph
        
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            'max_degree': max([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
            'num_components': nx.number_connected_components(G),
            'largest_component_size': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() > 0 else 0
        }
        
        # Clustering coefficient
        if G.number_of_nodes() > 0:
            metrics['avg_clustering'] = nx.average_clustering(G)
        else:
            metrics['avg_clustering'] = 0
        
        # Modularity if clusters exist
        if self.feature_clusters:
            # Convert clusters to partition format for modularity
            # Only include nodes that are still in the graph
            valid_communities = []
            for cluster_id, nodes in self.feature_clusters.items():
                valid_nodes = [n for n in nodes if n in G.nodes()]
                if valid_nodes:
                    valid_communities.append(set(valid_nodes))
            
            if valid_communities and len(valid_communities) > 1:
                try:
                    from networkx.algorithms.community import modularity
                    metrics['modularity'] = modularity(G, valid_communities)
                except:
                    # If modularity fails, set to 0
                    metrics['modularity'] = 0
            else:
                metrics['modularity'] = 0
        else:
            metrics['modularity'] = 0
        
        return metrics
    
    def visualize_coactivation_matrix(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Visualize co-activation matrix as heatmap."""
        if self.coactivation_matrix is None:
            raise ValueError("Must compute co-activation matrix first")
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            self.coactivation_matrix,
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Co-activation strength'}
        )
        plt.title('Feature Co-activation Matrix')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_graph(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10),
        layout: str = "spring"
    ):
        """Visualize co-activation graph."""
        if self.graph is None:
            raise ValueError("Must build graph first")
        
        plt.figure(figsize=figsize)
        
        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=1/np.sqrt(self.graph.number_of_nodes()))
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Draw nodes colored by cluster if available
        if self.feature_clusters:
            node_colors = []
            for node in self.graph.nodes():
                for cluster_id, nodes in self.feature_clusters.items():
                    if node in nodes:
                        node_colors.append(cluster_id)
                        break
                else:
                    node_colors.append(-1)  # Unclustered
        else:
            node_colors = 'lightblue'
        
        # Draw graph
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=50,
            cmap='tab20',
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.2,
            width=0.5
        )
        
        plt.title(f'Co-activation Graph ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_results(self, output_dir: Path):
        """Save analysis results to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save co-activation matrix
        if self.coactivation_matrix is not None:
            np.save(output_dir / 'coactivation_matrix.npy', self.coactivation_matrix)
        
        # Save graph
        if self.graph is not None:
            nx.write_gexf(self.graph, output_dir / 'coactivation_graph.gexf')
            
            # Also save as edge list for easy loading
            nx.write_edgelist(
                self.graph,
                output_dir / 'coactivation_graph.edges',
                data=['weight']
            )
        
        # Save clusters
        if self.feature_clusters is not None:
            # Convert numpy int types to regular Python ints for JSON serialization
            clusters_json = {}
            for cluster_id, nodes in self.feature_clusters.items():
                # Convert both key and values to regular Python types
                cluster_key = int(cluster_id) if hasattr(cluster_id, 'item') else cluster_id
                cluster_nodes = [int(n) if hasattr(n, 'item') else n for n in nodes]
                clusters_json[cluster_key] = cluster_nodes
            
            with open(output_dir / 'clusters.json', 'w') as f:
                json.dump(clusters_json, f, indent=2)
        
        # Save metrics
        metrics = self.compute_graph_metrics()
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
