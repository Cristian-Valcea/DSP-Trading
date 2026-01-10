#!/usr/bin/env python3
"""
Regime Features for Baseline B

Adds market-level regime context to improve signal quality:
1. SPY trend: 5-day vs 20-day EMA (bullish/bearish market)
2. VIX proxy: 20-day realized vol of SPY (high/low volatility regime)
3. SPY momentum: 5-day log return

These are computed from SPY RTH data available at feature cutoff (10:30 ET).
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import load_rth_data, get_rth_day_data, SPLIT_DATES


class RegimeFeatureBuilder:
    """
    Computes market regime features from SPY data.

    Features (3-dim):
    - spy_trend: sign(EMA5 - EMA20) at 10:30 ET (+1 bullish, -1 bearish, 0 neutral)
    - spy_vol_regime: 20-day realized vol percentile (0-1 scale)
    - spy_momentum: 5-day log return
    """

    def __init__(self, lookback_days: int = 25):
        """
        Initialize regime builder.

        Args:
            lookback_days: Days of history needed for regime calculation
        """
        self.lookback_days = lookback_days
        self._spy_cache = {}  # Cache SPY data by split
        self._daily_closes = {}  # Cache daily closes by split

    def _load_spy_data(self, split: str) -> pd.DataFrame:
        """Load and cache SPY data for a split."""
        if split not in self._spy_cache:
            self._spy_cache[split] = load_rth_data("SPY", split)
        return self._spy_cache[split]

    def _get_daily_closes(self, split: str) -> pd.DataFrame:
        """Get daily closing prices for SPY."""
        if split not in self._daily_closes:
            spy_df = self._load_spy_data(split)
            # Get last bar of each day (close at 16:00 ET)
            # Group by date and get the last row of each group
            spy_df = spy_df.copy()
            spy_df["trade_date"] = spy_df["timestamp"].dt.date
            daily = spy_df.groupby("trade_date").last().reset_index()
            # Rename trade_date to date for consistency
            daily = daily.rename(columns={"trade_date": "date"})
            daily["date"] = pd.to_datetime(daily["date"])
            self._daily_closes[split] = daily
        return self._daily_closes[split]

    def _compute_ema(self, prices: np.ndarray, span: int) -> float:
        """Compute EMA of price series, return last value."""
        if len(prices) < span:
            return np.nan
        alpha = 2 / (span + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def compute_regime_features(
        self,
        d: date,
        split: str,
    ) -> tuple[np.ndarray, dict]:
        """
        Compute 3-dim regime features for a given date.

        Uses data STRICTLY before the current day to avoid lookahead.

        Args:
            d: Trading date
            split: Data split for loading

        Returns:
            Tuple of (3-dim feature array, diagnostics dict)
        """
        features = np.zeros(3, dtype=np.float32)
        diagnostics = {"date": str(d)}

        try:
            daily = self._get_daily_closes(split)
        except FileNotFoundError:
            diagnostics["error"] = "SPY data not found"
            return features, diagnostics

        # Get closes BEFORE current date (no lookahead)
        mask = daily["date"].dt.date < d
        prior_data = daily[mask].tail(self.lookback_days)

        if len(prior_data) < 20:
            diagnostics["error"] = f"Insufficient history: {len(prior_data)} days"
            return features, diagnostics

        closes = prior_data["close"].values

        # Feature 1: SPY trend (EMA5 vs EMA20)
        ema5 = self._compute_ema(closes, 5)
        ema20 = self._compute_ema(closes, 20)

        if not np.isnan(ema5) and not np.isnan(ema20):
            # Normalize by price level
            trend_signal = (ema5 - ema20) / ema20
            features[0] = np.clip(trend_signal * 100, -1, 1)  # Scale and clip
            diagnostics["spy_trend_raw"] = trend_signal

        # Feature 2: Volatility regime (20-day realized vol)
        if len(closes) >= 20:
            log_returns = np.diff(np.log(closes[-21:]))
            realized_vol = np.std(log_returns) * np.sqrt(252)
            # Normalize to 0-1 scale (assume vol range 10%-50%)
            vol_percentile = np.clip((realized_vol - 0.10) / 0.40, 0, 1)
            features[1] = vol_percentile
            diagnostics["spy_vol_ann"] = realized_vol

        # Feature 3: SPY momentum (5-day return)
        if len(closes) >= 6:
            momentum = np.log(closes[-1] / closes[-6])
            features[2] = np.clip(momentum * 10, -1, 1)  # Scale and clip
            diagnostics["spy_momentum_raw"] = momentum

        diagnostics["n_days_used"] = len(prior_data)
        return features, diagnostics


# Feature names for documentation
REGIME_FEATURE_NAMES = [
    "spy_trend",      # EMA5/EMA20 trend signal
    "spy_vol_regime", # Volatility percentile
    "spy_momentum",   # 5-day momentum
]
NUM_REGIME_FEATURES = len(REGIME_FEATURE_NAMES)


if __name__ == "__main__":
    from datetime import date

    print("=== Regime Feature Validation ===")
    print(f"Features: {REGIME_FEATURE_NAMES}")
    print()

    builder = RegimeFeatureBuilder()

    # Test on a few dates
    test_dates = [
        date(2024, 1, 3),   # Early VAL
        date(2024, 3, 15),  # Mid VAL
        date(2024, 6, 15),  # Late VAL
    ]

    for d in test_dates:
        features, diag = builder.compute_regime_features(d, "val")
        print(f"{d}:")
        print(f"  spy_trend: {features[0]:.4f}")
        print(f"  spy_vol_regime: {features[1]:.4f}")
        print(f"  spy_momentum: {features[2]:.4f}")
        if "error" in diag:
            print(f"  ERROR: {diag['error']}")
        print()

    print("=== Validation Complete ===")
