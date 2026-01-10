"""
Data Loader for Baseline C

Reuses Baseline B data loading infrastructure.
Adds support for getting bars at arbitrary times (not just 10:31/14:00).
"""

# Re-export everything from baseline_b.data_loader
from baseline_b.data_loader import (
    PROJECT_ROOT,
    DSP100K_ROOT,
    RTH_DATA_DIR,
    PREMARKET_CACHE_2021_2022,
    PREMARKET_CACHE_2023_PLUS,
    SPLIT_DATES,
    SYMBOLS,
    META_RENAME_DATE,
    load_rth_data,
    load_premarket_json,
    parse_premarket_bars,
    get_premarket_segment_bars,
    get_price_at_or_before,
    get_trading_days,
    get_rth_day_data,
    get_bar_by_time,
)

from datetime import time
from typing import Optional
import pandas as pd


# Baseline C rebalance times
REBALANCE_TIMES = [
    time(10, 31),
    time(11, 31),
    time(12, 31),
    time(14, 0),
]

# Feature minute indices (09:30 = minute_idx 0)
# Features use data through the prior minute close
FEATURE_MINUTE_IDX = {
    time(10, 31): 60,   # 10:30 close -> trade at 10:31 open
    time(11, 31): 120,  # 11:30 close -> trade at 11:31 open
    time(12, 31): 180,  # 12:30 close -> trade at 12:31 open
    time(14, 0): 269,   # 13:59 close -> trade at 14:00 open
}

# Interval definitions (start_time, end_time, interval_name)
INTERVALS = [
    (time(10, 31), time(11, 31), "10:31->11:31"),
    (time(11, 31), time(12, 31), "11:31->12:31"),
    (time(12, 31), time(14, 0), "12:31->14:00"),
    (time(14, 0), time(10, 31), "14:00->next10:31"),  # Overnight
]


def get_bar_open_price(day_df: pd.DataFrame, target_time: time) -> Optional[float]:
    """
    Get the open price at a specific time.

    Args:
        day_df: Single day RTH data
        target_time: Target time (e.g., time(10, 31))

    Returns:
        Open price or None if bar not found
    """
    bar = get_bar_by_time(day_df, target_time)
    if bar is None:
        return None
    return float(bar["open"])


def is_early_close_day(day_df: pd.DataFrame) -> bool:
    """
    Check if a day is an early close (no 14:00 bar).

    Args:
        day_df: Single day RTH data

    Returns:
        True if the day has no 14:00 bar
    """
    bar_14 = get_bar_by_time(day_df, time(14, 0))
    return bar_14 is None


if __name__ == "__main__":
    print("=== Baseline C Data Loader Validation ===")
    print(f"Rebalance times: {REBALANCE_TIMES}")
    print(f"Feature minute indices: {FEATURE_MINUTE_IDX}")
    print(f"Intervals: {[i[2] for i in INTERVALS]}")
    print()

    # Test loading
    from datetime import date
    trading_days = get_trading_days("val")
    print(f"VAL trading days: {len(trading_days)}")

    # Check for early close days
    spy_df = load_rth_data("SPY", "val")
    early_close_count = 0
    for d in trading_days[:10]:
        day_df = get_rth_day_data(spy_df, d)
        if is_early_close_day(day_df):
            early_close_count += 1
            print(f"  Early close: {d}")

    print(f"\n=== Validation Complete ===")
