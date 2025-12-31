"""Sleeves module for DSP-100K portfolio."""

from .sleeve_a import SleeveA
from .sleeve_b import SleeveB, TrendSignalGenerator, VolatilityTargeter
from .sleeve_dm import SleeveDM
from .sleeve_c import SleeveC, PutSpreadManager
from .sleeve_im import SleeveIM

__all__ = [
    # Sleeve A (Equity Momentum + SPY hedge)
    "SleeveA",
    # Sleeve B (Cross-Asset Trend)
    "SleeveB",
    "TrendSignalGenerator",
    "VolatilityTargeter",
    # Sleeve DM (ETF Dual Momentum)
    "SleeveDM",
    # Sleeve C (SPY Put Spread Hedge)
    "SleeveC",
    "PutSpreadManager",
    # Sleeve IM (Intraday ML Long/Short)
    "SleeveIM",
]
