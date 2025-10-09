"""
Track evolution of co-activation graphs across training checkpoints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


class EmergenceTracker:
    """Track graph evolution and detect emergence signatures."""
    
    def __init__(self):
        self.checkpoint_metrics = []
        self.emergence_points = []
    
    def add_checkpoint(
        self,
        checkpoint_name: str,
        metrics: Dict[str, float],
        test_acc: float,
        step: Optional[int] = None
    ):
        """Add metrics from a checkpoint."""
        record = {
            'checkpoint': checkpoint_name,
            'step': step if step is not None else len(self.checkpoint_metrics),
            'test_acc': test_acc,
            **metrics
        }
        self.checkpoint_metrics.append(record)
    
    def detect_emergence(
        self,
        threshold: float = 0.8,
        min_jump: float = 0.5,
        window: int = 3
    ) -> List[int]:
        """
        Detect emergence points based on metric jumps.
        
        Args:
            threshold: Minimum emergence_metric to consider "emerged"
            min_jump: Minimum increase in emergence_metric to flag
            window: Window size for computing jumps
        
        Returns:
            List of checkpoint indices where emergence detected
        """
        if len(self.checkpoint_metrics) < window:
            return []
        
        emergence_points = []
        
        for i in range(window, len(self.checkpoint_metrics)):
            current = self.checkpoint_metrics[i]['test_acc']
            prev_avg = np.mean([
                self.checkpoint_metrics[j]['test_acc'] 
                for j in range(i-window, i)
            ])
            
            # Check for sudden jump
            if current > threshold and (current - prev_avg) > min_jump:
                emergence_points.append(i)
        
        self.emergence_points = emergence_points
        return emergence_points
    
    def find_precursors(
        self,
        emergence_idx: int,
        lookback: int = 5
    ) -> Dict[str, Any]:
        """
        Find graph changes that precede emergence.
        
        Args:
            emergence_idx: Index of emergence checkpoint
            lookback: How many checkpoints to look back
        
        Returns:
            Dictionary of precursor signals
        """
        if emergence_idx <= 0:
            return {}
        
        start_idx = max(0, emergence_idx - lookback)
        
        # Get metrics before and at emergence
        pre_metrics = self.checkpoint_metrics[start_idx:emergence_idx]
        emergence_metrics = self.checkpoint_metrics[emergence_idx]
        
        precursors = {}
        
        # Look for increasing trends in key metrics
        for metric in ['density', 'avg_clustering', 'modularity', 'largest_component_size']:
            if metric in emergence_metrics:
                values = [m.get(metric, 0) for m in pre_metrics]
                if values:
                    # Compute trend
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    
                    # Compute relative change
                    if values[0] != 0:
                        rel_change = (emergence_metrics[metric] - values[0]) / values[0]
                    else:
                        rel_change = float('inf') if emergence_metrics[metric] > 0 else 0
                    
                    precursors[f'{metric}_trend'] = trend
                    precursors[f'{metric}_change'] = rel_change
        
        return precursors
    
    def compute_evolution_metrics(self) -> pd.DataFrame:
        """
        Compute metrics showing graph evolution.
        
        Returns:
            DataFrame with evolution metrics
        """
        df = pd.DataFrame(self.checkpoint_metrics)
        
        if len(df) > 1:
            # Compute deltas
            for col in ['density', 'avg_clustering', 'num_edges', 'largest_component_size']:
                if col in df.columns:
                    df[f'{col}_delta'] = df[col].diff()
                    df[f'{col}_rel_change'] = df[col].pct_change()
            
            # Compute rolling averages
            window = min(3, len(df) - 1)
            for col in ['test_acc', 'density', 'avg_clustering']:
                if col in df.columns:
                    df[f'{col}_rolling_avg'] = df[col].rolling(window=window, min_periods=1).mean()
        
        return df
    
    def plot_evolution(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Plot evolution of metrics across checkpoints.
        
        Args:
            metrics_to_plot: List of metrics to plot (None = default set)
            save_path: Path to save figure
        """
        if not self.checkpoint_metrics:
            print("No checkpoints to plot")
            return
        
        df = self.compute_evolution_metrics()
        
        if metrics_to_plot is None:
            # Default metrics to plot
            available_metrics = []
            for m in ['test_acc', 'density', 'avg_clustering', 
                     'num_edges', 'largest_component_size', 'modularity']:
                if m in df.columns:
                    available_metrics.append(m)
            metrics_to_plot = available_metrics[:6]  # Limit to 6 subplots
        
        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            print("No metrics to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2, 
            figsize=figsize,
            sharex=True
        )
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Plot metric
            ax.plot(df['step'], df[metric], marker='o', label=metric)
            
            # Mark emergence points
            for em_idx in self.emergence_points:
                if em_idx < len(df):
                    ax.axvline(
                        df.iloc[em_idx]['step'],
                        color='red',
                        linestyle='--',
                        alpha=0.5,
                        label='Emergence' if idx == 0 else None
                    )
            
            ax.set_ylabel(metric)
            ax.set_title(f'Evolution of {metric}')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend()
        
        # Set common x-label
        for ax in axes[-2:]:
            ax.set_xlabel('Training Step')
        
        plt.suptitle('Graph Evolution Across Training')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, output_dir: Path):
        """Generate emergence analysis report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw metrics
        df = self.compute_evolution_metrics()
        df.to_csv(output_dir / 'evolution_metrics.csv', index=False)
        
        # Detect emergence if not already done
        if not self.emergence_points:
            self.detect_emergence()
        
        # Generate summary
        summary = {
            'num_checkpoints': len(self.checkpoint_metrics),
            'emergence_points': self.emergence_points,
            'emergence_checkpoints': [
                self.checkpoint_metrics[i]['checkpoint']
                for i in self.emergence_points
            ] if self.emergence_points else [],
        }
        
        # Add precursor analysis
        if self.emergence_points:
            summary['precursors'] = []
            for em_idx in self.emergence_points:
                precursors = self.find_precursors(em_idx)
                summary['precursors'].append({
                    'checkpoint_idx': em_idx,
                    'checkpoint_name': self.checkpoint_metrics[em_idx]['checkpoint'],
                    **precursors
                })
        
        # Save summary
        with open(output_dir / 'emergence_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate plots
        self.plot_evolution(save_path=output_dir / 'evolution_plot.png')
        
        # Create HTML report
        self._create_html_report(output_dir, summary, df)
        
        print(f"Report generated: {output_dir}")
    
    def _create_html_report(self, output_dir: Path, summary: Dict, df: pd.DataFrame):
        """Create HTML report with findings."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Emergence Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
                .emergence {{ background: #ffe0e0; padding: 10px; margin: 10px 0; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Emergence Analysis Report</h1>
            
            <h2>Summary</h2>
            <div class="metric">
                <strong>Total Checkpoints Analyzed:</strong> {summary['num_checkpoints']}
            </div>
            
            <h2>Emergence Points Detected</h2>
        """
        
        if summary['emergence_checkpoints']:
            html += "<div class='emergence'>"
            html += f"<strong>Found {len(summary['emergence_checkpoints'])} emergence points:</strong><br>"
            for cp in summary['emergence_checkpoints']:
                html += f"• {cp}<br>"
            html += "</div>"
        else:
            html += "<div class='metric'>No clear emergence points detected</div>"
        
        # Add precursor analysis
        if 'precursors' in summary and summary['precursors']:
            html += "<h2>Precursor Signals</h2>"
            for prec in summary['precursors']:
                html += f"<div class='metric'>"
                html += f"<strong>Before {prec['checkpoint_name']}:</strong><br>"
                for key, value in prec.items():
                    if key not in ['checkpoint_idx', 'checkpoint_name']:
                        html += f"• {key}: {value:.4f}<br>"
                html += "</div>"
        
        # Add evolution plot
        html += """
            <h2>Metric Evolution</h2>
            <img src="evolution_plot.png" alt="Evolution Plot">
            
            <h2>Detailed Metrics</h2>
        """
        
        # Add metrics table
        html += df.head(20).to_html()
        
        html += """
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w') as f:
            f.write(html)
