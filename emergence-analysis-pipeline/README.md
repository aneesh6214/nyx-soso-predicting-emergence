# Emergence Analysis Pipeline

A modular framework for analyzing neural network emergence using Sparse Autoencoders (SAE) and co-activation graphs.

## Overview

This pipeline implements the methodology described in the informal proposal:
1. **Extract activations** from model checkpoints during training
2. **Train Sparse Autoencoders** to find interpretable features
3. **Build co-activation graphs** showing feature relationships
4. **Track graph evolution** to detect structural signatures of emergence
5. **Predict emergence** before it manifests in performance metrics

## Architecture

```
emergence-analysis-pipeline/
├── core/               # Core analysis modules
│   ├── sae.py         # Sparse Autoencoder
│   ├── coactivation.py # Co-activation graphs
│   ├── tracking.py    # Evolution tracking
│   └── visualization.py # Plotting utilities
├── adapters/          # Experiment-specific adapters
│   ├── base_adapter.py     # Abstract interface
│   ├── grokking_adapter.py # For grokking experiments
│   ├── pythia_adapter.py   # For Pythia LMs
│   └── generic_adapter.py  # For custom models
└── run_analysis.py    # Main entry point
```

## Quick Start

### For Grokking Experiments

```bash
# Analyze pre/post grokking checkpoints
python run_analysis.py \
    --experiment grokking \
    --checkpoint_dir runs/grokking_final \
    --sae_features 512 \
    --track_evolution \
    --predict_emergence
```

### For Pythia Models

```bash
# Analyze Pythia model layers
python run_analysis.py \
    --experiment pythia \
    --model pythia-410m-deduped \
    --layer 6 \
    --sae_features 8192
```

### For Custom Models

```bash
# Analyze any PyTorch checkpoints
python run_analysis.py \
    --experiment custom \
    --checkpoints model_step_*.pt \
    --track_evolution
```

## Key Features

- **Modular Design**: Easy to add new experiment types via adapters
- **SAE Training**: Automatic sparse feature extraction with L1 regularization
- **Graph Analysis**: NetworkX-based co-activation graph construction
- **Emergence Detection**: Automatic identification of phase transitions
- **Rich Visualizations**: Interactive Plotly graphs and matplotlib plots
- **Comprehensive Reports**: HTML reports with metrics and findings

## Outputs

```
outputs/[experiment_name]/
├── checkpoint_name/           # Per-checkpoint results
│   ├── sae_model.pt          # Trained SAE
│   ├── coactivation_matrix.npy # Co-activation data
│   ├── coactivation_graph.gexf # Graph structure
│   └── metrics.json          # Graph metrics
├── evolution_analysis/        # Cross-checkpoint analysis
│   ├── evolution_metrics.csv # Metric evolution
│   ├── emergence_summary.json # Detected emergence points
│   ├── evolution_plot.png    # Visualization
│   └── report.html           # Full HTML report
└── config.json               # Configuration used

```

## Testing

Run the test script to verify installation:

```bash
python emergence-analysis-pipeline/test_pipeline.py
```

## Adding New Experiments

1. Create new adapter in `adapters/` inheriting from `BaseAdapter`
2. Implement required methods:
   - `load_checkpoints()` - Load model checkpoints
   - `extract_activations()` - Get activations from layers
   - `get_checkpoint_info()` - Extract metadata
   - `get_emergence_metric()` - Define emergence metric

3. Register in run_analysis.py's `load_adapter()` function

## Requirements

See `requirements.txt` for full list. Core dependencies:
- PyTorch >= 2.0
- NetworkX
- scikit-learn
- matplotlib, seaborn, plotly
- numpy, pandas
