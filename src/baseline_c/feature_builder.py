"""
Feature Builder for Baseline C

Computes 45-dimensional feature vector per (day, symbol, rebalance_time):
- 30 base features from StateBuilder at variable minute_idx
- 6 premarket segment features (3 returns + 3 bar counts)
- 9 symbol one-hot encoding

Key difference from Baseline B:
- Supports arbitrary minute_idx (not just 60 for 10:30)
- Same feature count (45), but computed at different times of day
"""

import sys
from datetime import date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add DQN src to path for StateBuilder import
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dsp100k" / "src"))

from dqn.state_builder import StateBuilder

from .data_loader import (
    SYMBOLS,
    FEATURE_MINUTE_IDX,
    get_bar_by_time,
    get_price_at_or_before,
    get_rth_day_data,
    load_premarket_json,
    load_rth_data,
    parse_premarket_bars,
)

# Symbol to index mapping for one-hot encoding
SYMBOL_TO_IDX = {s: i for i, s in enumerate(SYMBOLS)}

# Premarket segment boundaries (ET)
PREMARKET_SEGMENTS = [
    ("0400_0600", time(4, 0), time(6, 0)),
    ("0600_0800", time(6, 0), time(8, 0)),
    ("0800_0915", time(8, 0), time(9, 15)),
]

# Feature names for documentation
FEATURE_NAMES_BASE = StateBuilder.FEATURE_NAMES  # 30 features
FEATURE_NAMES_PREMARKET = [
    "r_pre_0400_0600", "r_pre_0600_0800", "r_pre_0800_0915",
    "pre_bar_count_0400_0600", "pre_bar_count_0600_0800", "pre_bar_count_0800_0915",
]
FEATURE_NAMES_SYMBOL = [f"symbol_{s}" for s in SYMBOLS]
FEATURE_NAMES = FEATURE_NAMES_BASE + FEATURE_NAMES_PREMARKET + FEATURE_NAMES_SYMBOL
NUM_FEATURES = len(FEATURE_NAMES)  # 30 + 6 + 9 = 45


class FeatureBuilder:
    """
    Builds 45-dimensional feature vectors for Baseline C.

    For each (day, symbol, rebalance_time), computes:
    1. Base features (30-dim) from StateBuilder at appropriate minute_idx
    2. Premarket segment features (6-dim) - same for all rebalance times on a day
    3. Symbol one-hot (9-dim)
    """

    def __init__(self, symbols: list[str] = None):
        """
        Initialize feature builder.

        Args:
            symbols: List of symbols to process (defaults to SYMBOLS)
        """
        self.symbols = symbols or SYMBOLS
        self.state_builder = StateBuilder(symbols=self.symbols)

    def compute_premarket_features(
        self,
        symbol: str,
        d: date
    ) -> tuple[np.ndarray, dict]:
        """
        Compute 6 premarket features for a symbol/day.

        Note: Premarket features are the same regardless of rebalance time
        (computed from 04:00-09:15 which is before any rebalance).

        Returns:
            Tuple of (features array [6], diagnostics dict)
        """
        features = np.zeros(6, dtype=np.float32)
        diagnostics = {"symbol": symbol, "date": str(d)}

        # Load premarket data
        pm_data = load_premarket_json(symbol, d)
        if pm_data is None:
            diagnostics["premarket_available"] = False
            return features, diagnostics

        pm_df = parse_premarket_bars(pm_data)
        diagnostics["premarket_available"] = True
        diagnostics["total_bars"] = len(pm_df)

        # Get boundary prices
        boundary_times = [time(4, 0), time(6, 0), time(8, 0), time(9, 15)]
        prices = {}
        for t in boundary_times:
            prices[t] = get_price_at_or_before(pm_df, t)

        # Compute segment returns and bar counts
        for i, (seg_name, seg_start, seg_end) in enumerate(PREMARKET_SEGMENTS):
            # Get bar count in segment
            if not pm_df.empty:
                bar_times = pm_df["timestamp"].dt.time
                mask = (bar_times >= seg_start) & (bar_times < seg_end)
                bar_count = mask.sum()
            else:
                bar_count = 0

            features[3 + i] = bar_count  # bar_count features at indices 3,4,5
            diagnostics[f"bar_count_{seg_name}"] = bar_count

            # Get segment return
            p_start = prices[seg_start]
            p_end = prices[seg_end]

            if p_start is not None and p_end is not None and p_start > 0:
                log_return = np.log(p_end / p_start)
            elif bar_count == 0:
                # No bars in segment → return 0 (spec §4.2)
                log_return = 0.0
            else:
                # Have bars but missing boundary price → edge case
                log_return = 0.0

            features[i] = log_return  # return features at indices 0,1,2
            diagnostics[f"r_{seg_name}"] = log_return

        return features, diagnostics

    def compute_base_features(
        self,
        rth_day_data: dict[str, pd.DataFrame],
        symbol: str,
        minute_idx: int,
        prior_close: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Compute 30 base features using StateBuilder at specified minute_idx.

        Args:
            rth_day_data: Dict mapping symbol -> DataFrame for one day
            symbol: Target symbol
            minute_idx: Minute index (09:30 = 0, so 10:30 = 60, 11:30 = 120, etc.)
            prior_close: Prior session close (for overnight gap feature)

        Returns:
            30-dim feature array or None if data insufficient
        """
        if symbol not in rth_day_data:
            return None

        day_df = rth_day_data[symbol]
        # Need at least minute_idx+1 bars (0 to minute_idx inclusive)
        if len(day_df) < minute_idx + 1:
            return None

        # Prepare premarket summary for StateBuilder (coarse version)
        premarket_data = {}

        # Prepare prior close map
        prior_close_map = {symbol: prior_close} if prior_close else {}

        # Reset state builder with day's data
        self.state_builder.reset(
            data_dict=rth_day_data,
            premarket_data=premarket_data,
            prior_close_by_symbol=prior_close_map,
        )

        # Get features at specified minute_idx
        features_all = self.state_builder.get_features(minute_idx=minute_idx)

        # Extract features for target symbol
        symbol_idx = self.state_builder.symbol_to_idx.get(symbol)
        if symbol_idx is None:
            return None

        return features_all[symbol_idx, :]

    def compute_symbol_onehot(self, symbol: str) -> np.ndarray:
        """
        Compute 9-dim symbol one-hot encoding.

        Args:
            symbol: Symbol name

        Returns:
            9-dim one-hot array
        """
        onehot = np.zeros(len(SYMBOLS), dtype=np.float32)
        if symbol in SYMBOL_TO_IDX:
            onehot[SYMBOL_TO_IDX[symbol]] = 1.0
        return onehot

    def build_features(
        self,
        rth_day_data: dict[str, pd.DataFrame],
        symbol: str,
        d: date,
        rebalance_time: time,
        prior_close: Optional[float] = None
    ) -> tuple[Optional[np.ndarray], dict]:
        """
        Build full 45-dim feature vector for a (day, symbol, rebalance_time) tuple.

        Args:
            rth_day_data: Dict mapping symbol -> DataFrame for one day
            symbol: Target symbol
            d: Trading date
            rebalance_time: One of the 4 rebalance times (10:31, 11:31, 12:31, 14:00)
            prior_close: Prior session close

        Returns:
            Tuple of (45-dim feature array or None, diagnostics dict)
        """
        diagnostics = {"symbol": symbol, "date": str(d), "rebalance_time": str(rebalance_time)}

        # Get minute_idx for this rebalance time
        minute_idx = FEATURE_MINUTE_IDX.get(rebalance_time)
        if minute_idx is None:
            diagnostics["skip_reason"] = f"unknown_rebalance_time_{rebalance_time}"
            return None, diagnostics

        diagnostics["minute_idx"] = minute_idx

        # 1. Base features (30-dim)
        base_features = self.compute_base_features(rth_day_data, symbol, minute_idx, prior_close)
        if base_features is None:
            diagnostics["skip_reason"] = "insufficient_rth_data"
            return None, diagnostics

        # 2. Premarket features (6-dim) - same for all rebalance times
        premarket_features, pm_diag = self.compute_premarket_features(symbol, d)
        diagnostics.update(pm_diag)

        # 3. Symbol one-hot (9-dim)
        symbol_onehot = self.compute_symbol_onehot(symbol)

        # Concatenate all features
        full_features = np.concatenate([
            base_features,       # 30 dims
            premarket_features,  # 6 dims
            symbol_onehot,       # 9 dims
        ])

        assert len(full_features) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(full_features)}"

        return full_features, diagnostics


def get_prior_close(rth_df: pd.DataFrame, current_date: date) -> Optional[float]:
    """
    Get the prior session's closing price for a symbol.

    Args:
        rth_df: Full RTH data for symbol
        current_date: Current trading date

    Returns:
        Prior close price or None
    """
    # Find the last bar before current_date
    prior_mask = rth_df["timestamp"].dt.date < current_date
    prior_bars = rth_df[prior_mask]

    if prior_bars.empty:
        return None

    return float(prior_bars.iloc[-1]["close"])


if __name__ == "__main__":
    from datetime import date

    print("=== Baseline C Feature Builder Validation ===")
    print(f"Total features: {NUM_FEATURES}")
    print(f"Base features: {len(FEATURE_NAMES_BASE)}")
    print(f"Premarket features: {len(FEATURE_NAMES_PREMARKET)}")
    print(f"Symbol features: {len(FEATURE_NAMES_SYMBOL)}")
    print()

    # Test feature building at different rebalance times
    builder = FeatureBuilder()

    test_date = date(2024, 1, 3)  # VAL split
    symbol = "AAPL"

    # Load RTH data
    rth_df = load_rth_data(symbol, "val")
    day_df = get_rth_day_data(rth_df, test_date)

    print(f"Testing {symbol} on {test_date}...")
    print(f"RTH bars for day: {len(day_df)}")

    # Get prior close
    prior_close = get_prior_close(rth_df, test_date)
    print(f"Prior close: {prior_close}")
    print()

    # Load all symbols for cross-asset features
    rth_day_data = {symbol: day_df}
    for s in SYMBOLS:
        if s != symbol:
            try:
                s_df = load_rth_data(s, "val")
                s_day = get_rth_day_data(s_df, test_date)
                if not s_day.empty:
                    rth_day_data[s] = s_day
            except FileNotFoundError:
                pass

    # Test all rebalance times
    for rebal_time in [time(10, 31), time(11, 31), time(12, 31), time(14, 0)]:
        features, diag = builder.build_features(
            rth_day_data, symbol, test_date, rebal_time, prior_close
        )

        if features is not None:
            print(f"Rebalance {rebal_time}: shape={features.shape}, "
                  f"min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
        else:
            print(f"Rebalance {rebal_time}: SKIPPED - {diag.get('skip_reason', 'unknown')}")

    print("\n=== Validation Complete ===")
