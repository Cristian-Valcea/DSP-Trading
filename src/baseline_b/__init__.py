"""
Baseline B: Supervised Intraday Baseline (Premarket + 1st Hour → 10:31→14:00)

This module implements a Ridge regression baseline to test whether existing
features (premarket + first hour RTH) contain tradable edge net of costs.

Components:
- data_loader: Load RTH parquet and premarket JSON data
- feature_builder: Compute 45-dim features (30 base + 6 premarket + 9 symbol)
- dataset: Generate (X, y) pairs with missing data tracking
- train: Ridge regression training with standardization
- backtest: Cost-aware portfolio simulation
- evaluate: Metrics and kill gates
"""

__version__ = "0.1.0"
