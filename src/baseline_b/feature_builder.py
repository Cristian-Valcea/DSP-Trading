"""
Feature Builder for Baseline B

Computes feature vector per (day, symbol):
- 30 base features from StateBuilder at minute_idx=60 (10:30 ET)
- 6 premarket segment features (3 returns + 3 bar counts)
- 9 symbol one-hot encoding

Premarket segments (ET):
- Segment A: 04:00-06:00
- Segment B: 06:00-08:00
- Segment C: 08:00-09:15
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
    Builds feature vectors for Baseline B.

    For each (day, symbol), computes:
    1. Base features (30-dim) from StateBuilder at 10:30 ET
    2. Premarket segment features (6-dim)
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
        prior_close: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Compute 30 base features using StateBuilder at minute_idx=60 (10:30 ET).

        Args:
            rth_day_data: Dict mapping symbol -> DataFrame for one day
            symbol: Target symbol
            prior_close: Prior session close (for overnight gap feature)

        Returns:
            30-dim feature array or None if data insufficient
        """
        if symbol not in rth_day_data:
            return None

        day_df = rth_day_data[symbol]
        if len(day_df) < 61:  # Need at least 61 bars (0-60 inclusive)
            return None

        # Prepare premarket summary for StateBuilder (coarse version)
        # Note: We'll add detailed premarket separately
        premarket_data = {}

        # Prepare prior close map
        prior_close_map = {symbol: prior_close} if prior_close else {}

        # Reset state builder with day's data
        self.state_builder.reset(
            data_dict=rth_day_data,
            premarket_data=premarket_data,
            prior_close_by_symbol=prior_close_map,
        )

        # Get features at minute_idx=60 (10:30 ET)
        features_all = self.state_builder.get_features(minute_idx=60)

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
        prior_close: Optional[float] = None
    ) -> tuple[Optional[np.ndarray], dict]:
        """
        Build full feature vector for a (day, symbol) pair.

        Args:
            rth_day_data: Dict mapping symbol -> DataFrame for one day
            symbol: Target symbol
            d: Trading date
            prior_close: Prior session close

        Returns:
            Tuple of (45-dim feature array or None, diagnostics dict)
        """
        diagnostics = {"symbol": symbol, "date": str(d)}

        # 1. Base features (30-dim)
        base_features = self.compute_base_features(rth_day_data, symbol, prior_close)
        if base_features is None:
            diagnostics["skip_reason"] = "insufficient_rth_data"
            return None, diagnostics

        # 2. Premarket features (6-dim)
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

    print("=== Feature Builder Validation ===")
    print(f"Total features: {NUM_FEATURES}")
    print(f"Base features: {len(FEATURE_NAMES_BASE)}")
    print(f"Premarket features: {len(FEATURE_NAMES_PREMARKET)}")
    print(f"Symbol features: {len(FEATURE_NAMES_SYMBOL)}")
    print()

    # Test feature building
    builder = FeatureBuilder()

    # Load test data
    test_date = date(2021, 12, 21)  # First day in train split
    symbol = "AAPL"

    # Load RTH data
    rth_df = load_rth_data(symbol, "train")
    day_df = get_rth_day_data(rth_df, test_date)

    print(f"Testing {symbol} on {test_date}...")
    print(f"RTH bars for day: {len(day_df)}")

    # Get prior close
    prior_close = get_prior_close(rth_df, test_date)
    print(f"Prior close: {prior_close}")

    # Build features
    rth_day_data = {symbol: day_df}

    # Need to load all symbols for cross-asset features
    for s in SYMBOLS:
        if s != symbol:
            try:
                s_df = load_rth_data(s, "train")
                s_day = get_rth_day_data(s_df, test_date)
                if not s_day.empty:
                    rth_day_data[s] = s_day
            except FileNotFoundError:
                pass

    features, diag = builder.build_features(rth_day_data, symbol, test_date, prior_close)

    if features is not None:
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
        print()
        print("Sample features:")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES[:10], features[:10])):
            print(f"  [{i:2d}] {name}: {val:.6f}")
        print("  ...")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES[30:36], features[30:36]), start=30):
            print(f"  [{i:2d}] {name}: {val:.6f}")
        print("  ...")
        for i, (name, val) in enumerate(zip(FEATURE_NAMES[36:], features[36:]), start=36):
            print(f"  [{i:2d}] {name}: {val:.1f}")
    else:
        print(f"Feature building failed: {diag.get('skip_reason', 'unknown')}")

    print("\nDiagnostics:", diag)
    print("\n=== Validation Complete ===")
