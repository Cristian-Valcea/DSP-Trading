"""Sleeves module for DSP-100K portfolio."""

from .sleeve_a import SleeveA, SleeveANotImplementedError
from .sleeve_b import SleeveB, TrendSignalGenerator, VolatilityTargeter
from .sleeve_c import SleeveC, PutSpreadManager

__all__ = [
    # Sleeve A (NOT IMPLEMENTED - placeholder)
    "SleeveA",
    "SleeveANotImplementedError",
    # Sleeve B (Cross-Asset Trend)
    "SleeveB",
    "TrendSignalGenerator",
    "VolatilityTargeter",
    # Sleeve C (SPY Put Spread Hedge)
    "SleeveC",
    "PutSpreadManager",
]
