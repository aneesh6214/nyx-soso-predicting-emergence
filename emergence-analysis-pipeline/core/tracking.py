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

from .emergence_detection import detect_emergence as _detect_emergence

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
        jump_threshold: float = 0.20,
        stability_tol: float = 0.02,
        stability_horizon: int = 3,
        require_full_horizon: bool = True,
    ) -> List[int]:
        """
        Detect emergence points using the formal detector based on a big accuracy
        jump followed by stability over the next H evaluations.
        
        Returns:
            List of checkpoint indices where emergence is detected.
        """
        if len(self.checkpoint_metrics) < 2:
            return []

        steps = [m['step'] for m in self.checkpoint_metrics]
        acc = [m.get('test_acc', 0.0) for m in self.checkpoint_metrics]

        _, idx = _detect_emergence(
            steps,
            acc,
            jump_threshold=jump_threshold,
            stability_tol=stability_tol,
            stability_horizon=stability_horizon,
            require_full_horizon=require_full_horizon,
        )

        self.emergence_points = idx
        return idx
    
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
    
    @staticmethod
    def compute_lead_lag(
        seed_metric_dfs: Dict[str, pd.DataFrame],
        metrics: List[str],
        step_col: str = 'step',
        acc_col: str = 'test_acc',
        lags: List[int] = [1, 2, 3, 4, 5]
    ) -> pd.DataFrame:
        """
        Compute across-seed lead–lag correlations between delta(metric) and delta(acc).
        Returns a long DataFrame with columns: metric, lag, r_mean, ci_low, ci_high, n_seeds.
        Uses Fisher z-transform to average correlations and compute 95% CI across seeds.
        """
        # Collect per-seed correlation arrays for each metric
        metric_to_seed_r: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
        for _seed, df in seed_metric_dfs.items():
            if acc_col not in df.columns or step_col not in df.columns:
                continue
            df_sorted = df.sort_values(step_col).reset_index(drop=True)
            # First differences over full timeline (np.diff avoids leading NaN)
            delta_acc = np.diff(df_sorted[acc_col].to_numpy(dtype=float))
            for metric in metrics:
                if metric not in df_sorted.columns:
                    continue
                delta_m = np.diff(df_sorted[metric].to_numpy(dtype=float))
                r_vals: List[float] = []
                for k in lags:
                    if len(delta_m) <= k or len(delta_acc) <= k:
                        r_vals.append(np.nan)
                        continue
                    a = delta_m[:-k]
                    b = delta_acc[k:]
                    # Check finite and non-constant
                    if (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and
                        np.std(a) > 0 and np.std(b) > 0):
                        # Pearson r via numpy
                        r = np.corrcoef(a, b)[0, 1]
                        r_vals.append(float(r))
                    else:
                        r_vals.append(np.nan)
                metric_to_seed_r[metric].append(np.array(r_vals, dtype=float))
        
        # Aggregate with Fisher z-transform and compute 95% CI across seeds
        rows: List[Dict[str, float]] = []
        for metric in metrics:
            seed_arrays = [arr for arr in metric_to_seed_r.get(metric, []) if arr.size > 0]
            if not seed_arrays:
                continue
            R = np.vstack(seed_arrays)  # shape: (n_seeds, n_lags)
            r_mean_list: List[float] = []
            ci_low_list: List[float] = []
            ci_high_list: List[float] = []
            n_used_list: List[int] = []
            for j in range(R.shape[1]):
                rj = R[:, j]
                mask = np.isfinite(rj)
                rj = rj[mask]
                n = int(mask.sum())
                if n == 0:
                    r_mean_list.append(np.nan)
                    ci_low_list.append(np.nan)
                    ci_high_list.append(np.nan)
                    n_used_list.append(0)
                    continue
                # Clip r to avoid infinite atanh
                rj = np.clip(rj, -0.999999, 0.999999)
                z = np.arctanh(rj)
                z_mean = float(np.mean(z))
                r_bar = float(np.tanh(z_mean))
                if n > 1:
                    z_se = float(np.std(z, ddof=1)) / np.sqrt(n)
                    z_lo = z_mean - 1.96 * z_se
                    z_hi = z_mean + 1.96 * z_se
                    lo = float(np.tanh(z_lo))
                    hi = float(np.tanh(z_hi))
                else:
                    lo = np.nan
                    hi = np.nan
                r_mean_list.append(r_bar)
                ci_low_list.append(lo)
                ci_high_list.append(hi)
                n_used_list.append(n)
            for lag, rm, lo, hi, n in zip(lags, r_mean_list, ci_low_list, ci_high_list, n_used_list):
                rows.append({'metric': metric, 'lag': lag, 'r_mean': rm, 'ci_low': lo, 'ci_high': hi, 'n_seeds': n})
        return pd.DataFrame(rows)
    
    @staticmethod
    def plot_lead_lag_heatmap(
        df: pd.DataFrame,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot heatmap of mean correlations (rows: metric, cols: lag). Values centered at 0 with annotations.
        """
        if df.empty:
            return
        pivot = df.pivot(index='metric', columns='lag', values='r_mean')
        plt.figure(figsize=figsize)
        ax = sns.heatmap(pivot, vmin=-1.0, vmax=1.0, center=0.0, cmap='RdBu_r', annot=True, fmt='.2f')
        ax.set_title('Lead–Lag correlation: Δmetric vs Δaccuracy (mean across seeds)')
        ax.set_xlabel('Lag (eval steps)')
        ax.set_ylabel('Metric')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def aggregate_seed_metrics(seed_metric_dfs: Dict[str, pd.DataFrame],
                               value_cols: Optional[List[str]] = None,
                               step_col: str = 'step') -> Dict[str, pd.DataFrame]:
        """
        Aggregate multiple seed DataFrames by step, computing mean and std and 95% CI.
        Returns dict with 'mean', 'std', 'ci95', and 'long' (stacked) DataFrames.
        """
        # Align on step via outer join, suffix by seed
        # Determine default value columns if not provided
        if value_cols is None:
            # Union of numeric columns across seeds, excluding identifiers
            numeric_cols = set()
            for df in seed_metric_dfs.values():
                for col in df.columns:
                    if col != step_col and pd.api.types.is_numeric_dtype(df[col]):
                        numeric_cols.add(col)
            value_cols = sorted(numeric_cols)

        # Build long-format table: [seed, step, metric, value]
        long_rows = []
        for seed, df in seed_metric_dfs.items():
            for col in value_cols:
                if col in df.columns:
                    tmp = df[[step_col, col]].copy()
                    tmp['metric'] = col
                    tmp['seed'] = seed
                    tmp.rename(columns={col: 'value'}, inplace=True)
                    long_rows.append(tmp)
        if not long_rows:
            return {'mean': pd.DataFrame(), 'std': pd.DataFrame(), 'ci95': pd.DataFrame(), 'long': pd.DataFrame()}

        long_df = pd.concat(long_rows, ignore_index=True)

        # Group by step and metric
        grouped = long_df.groupby([step_col, 'metric'])['value']
        mean_df = grouped.mean().unstack('metric').reset_index()
        std_df = grouped.std(ddof=1).unstack('metric').reset_index()
        count_df = grouped.count().unstack('metric').reset_index()

        # 95% CI using normal approximation: 1.96 * std/sqrt(n)
        ci95_df = std_df.copy()
        for col in value_cols:
            if col in std_df.columns and col in count_df.columns:
                n = count_df[col].replace(0, np.nan)
                ci95_df[col] = 1.96 * std_df[col] / np.sqrt(n)

        return {'mean': mean_df, 'std': std_df, 'ci95': ci95_df, 'long': long_df}

    @staticmethod
    def plot_aggregate(mean_df: pd.DataFrame,
                       ci95_df: pd.DataFrame,
                       metrics: Optional[List[str]] = None,
                       step_col: str = 'step',
                       save_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (14, 10),
                       spaghetti_df: Optional[pd.DataFrame] = None,
                       emergence_step: Optional[int] = None,
                       emergence_steps: Optional[List[int]] = None):
        """
        Plot mean with 95% CI ribbon for the given metrics.
        Optionally draw a dashed vertical line at the emergence step.
        """
        if mean_df.empty:
            return
        if metrics is None:
            # Default to the same metrics as single-seed evolution plots, but use sparsity instead of modularity
            default_order = [
                'test_acc',
                'density',
                'avg_clustering',
                'num_edges',
                'largest_component_size',
                'sae_sparsity'
            ]
            metrics = [m for m in default_order if m in mean_df.columns]
            if not metrics:
                metrics = [c for c in mean_df.columns if c != step_col]

        # Create subplots up to 6
        n_metrics = min(len(metrics), 6)
        fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=figsize, sharex=True)
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metrics[:n_metrics]):
            ax = axes[idx]
            x = mean_df[step_col]
            y = mean_df[metric]
            ci = ci95_df[metric] if metric in ci95_df.columns else None
            # Spaghetti lines per seed (faint)
            if spaghetti_df is not None:
                dfm = spaghetti_df[spaghetti_df['metric'] == metric]
                for seed, sdf in dfm.groupby('seed'):
                    ax.plot(sdf[step_col], sdf['value'], color='C0', alpha=0.15, linewidth=1)
            ax.plot(x, y, color='C0', label=f'{metric} (mean)', linewidth=2)
            if ci is not None:
                ax.fill_between(x, y - ci, y + ci, color='C0', alpha=0.2, label='95% CI' if idx == 0 else None)
            # Draw emergence line (single aggregated step)
            line_x = None
            if emergence_step is not None:
                line_x = emergence_step
            elif emergence_steps:
                # Use median across seeds
                median_step = int(np.median(emergence_steps))
                # Snap to nearest available step in x
                line_x = min(x, key=lambda v: abs(v - median_step)) if len(x) > 0 else median_step
            if line_x is not None:
                ax.axvline(line_x, color='red', linestyle='--', alpha=0.6, label='Emergence' if idx == 0 else None)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend()

        for ax in axes[-2:]:
            ax.set_xlabel('Training Step')
        plt.suptitle('Across-seed Mean ± 95% CI')
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
