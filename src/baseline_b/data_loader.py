"""
Data Loader for Baseline B

Loads RTH minute bars from parquet and premarket data from JSON caches.

Data Sources:
- RTH: ../data/dqn_{split}/{symbol}_{split}.parquet (timestamps in ET)
- Premarket (2021-2022): dsp100k/data/dqn_premarket_cache/{symbol}/{date}.json
- Premarket (2023+): dsp100k/data/sleeve_im/minute_bars/{symbol}/{date}.json

FB→META Handling:
- Use canonical symbol META everywhere
- For pre-2022-06-09, fallback to FB premarket cache if META not found
"""

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # wsl-home
DSP100K_ROOT = PROJECT_ROOT / "dsp100k"

# Data paths
RTH_DATA_DIR = PROJECT_ROOT / "data"  # ../data/dqn_{split}/
PREMARKET_CACHE_2021_2022 = DSP100K_ROOT / "data" / "dqn_premarket_cache"
PREMARKET_CACHE_2023_PLUS = DSP100K_ROOT / "data" / "sleeve_im" / "minute_bars"

# Split definitions (reuse DQN splits)
SPLIT_DATES = {
    "train": (date(2021, 12, 20), date(2023, 12, 29)),      # 510 days
    "val": (date(2024, 1, 2), date(2024, 6, 28)),           # 124 days
    "dev_test": (date(2024, 7, 1), date(2024, 12, 31)),     # 128 days
    "holdout": (date(2025, 1, 2), date(2025, 12, 19)),      # 243 days
}

# Universe
SYMBOLS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "SPY", "TSLA"]

# FB→META rename date
META_RENAME_DATE = date(2022, 6, 9)


def load_rth_data(symbol: str, split: str) -> pd.DataFrame:
    """
    Load RTH minute bar data for a symbol and split.

    Args:
        symbol: Stock symbol (use META, not FB)
        split: One of train, val, dev_test, holdout

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume, symbol]
        timestamps are in ET (timezone-naive datetime64[ns])
    """
    if split not in SPLIT_DATES:
        raise ValueError(f"Unknown split: {split}. Must be one of {list(SPLIT_DATES.keys())}")

    # File naming convention: lowercase symbol
    filename = f"{symbol.lower()}_{split}.parquet"
    filepath = RTH_DATA_DIR / f"dqn_{split}" / filename

    if not filepath.exists():
        raise FileNotFoundError(f"RTH data not found: {filepath}")

    df = pd.read_parquet(filepath)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def _get_premarket_fallback_symbols(symbol: str, d: date) -> list[str]:
    """
    Get list of symbols to try for premarket data, handling FB→META.

    For META pre-2022-06-09, try FB as fallback.
    """
    if symbol == "META" and d < META_RENAME_DATE:
        return ["META", "FB"]
    return [symbol]


def load_premarket_json(symbol: str, d: date) -> Optional[dict]:
    """
    Load premarket JSON for a symbol and date.

    Tries multiple caches and handles FB→META fallback.

    Args:
        symbol: Canonical symbol (META, not FB)
        d: Trading date

    Returns:
        Parsed JSON dict with keys: symbol, date, bars
        Or None if not found
    """
    date_str = d.strftime("%Y-%m-%d")

    # Determine which cache based on year
    if d.year <= 2022:
        cache_dir = PREMARKET_CACHE_2021_2022
    else:
        cache_dir = PREMARKET_CACHE_2023_PLUS

    # Try symbols (with FB fallback for META pre-rename)
    symbols_to_try = _get_premarket_fallback_symbols(symbol, d)

    for try_symbol in symbols_to_try:
        filepath = cache_dir / try_symbol / f"{date_str}.json"
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
            return data

    return None


def parse_premarket_bars(data: dict) -> pd.DataFrame:
    """
    Parse premarket JSON into DataFrame.

    Handles both formats:
    - 2021-2022 cache: timestamps with timezone offset (e.g., "2021-12-20T04:00:00-05:00")
    - 2023+ cache: timezone-naive timestamps (treat as ET)

    Args:
        data: Parsed JSON dict with "bars" key

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
        timestamps as datetime (ET)
    """
    bars = data.get("bars", [])
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(bars)

    # Parse timestamps - handle both formats
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # If timezone-aware, convert to ET and make naive
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    # Keep only OHLCV columns
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def get_premarket_segment_bars(
    premarket_df: pd.DataFrame,
    segment_start: time,
    segment_end: time
) -> pd.DataFrame:
    """
    Filter premarket bars to a specific time segment.

    Args:
        premarket_df: DataFrame with timestamp column
        segment_start: Start time (inclusive)
        segment_end: End time (exclusive, or up to and including last bar before)

    Returns:
        Filtered DataFrame
    """
    if premarket_df.empty:
        return premarket_df

    bar_times = premarket_df["timestamp"].dt.time
    mask = (bar_times >= segment_start) & (bar_times < segment_end)
    return premarket_df[mask].copy()


def get_price_at_or_before(premarket_df: pd.DataFrame, target_time: time) -> Optional[float]:
    """
    Get the close price of the last bar at or before target_time.

    Implements the "carry-forward last print" rule from spec §4.2.

    Args:
        premarket_df: DataFrame with timestamp and close columns
        target_time: Target time boundary

    Returns:
        Close price or None if no bars before target_time
    """
    if premarket_df.empty:
        return None

    bar_times = premarket_df["timestamp"].dt.time
    valid_bars = premarket_df[bar_times <= target_time]

    if valid_bars.empty:
        return None

    # Get the last bar at or before target_time
    last_bar = valid_bars.iloc[-1]
    return float(last_bar["close"])


def get_trading_days(split: str) -> list[date]:
    """
    Get list of trading days for a split by examining the RTH data.

    Args:
        split: One of train, val, dev_test, holdout

    Returns:
        Sorted list of unique trading dates
    """
    # Load SPY data as reference for trading days
    spy_df = load_rth_data("SPY", split)

    # Extract unique dates
    dates = spy_df["timestamp"].dt.date.unique()
    return sorted(dates)


def get_rth_day_data(rth_df: pd.DataFrame, d: date) -> pd.DataFrame:
    """
    Filter RTH data to a single trading day.

    Args:
        rth_df: Full RTH DataFrame for symbol/split
        d: Trading date

    Returns:
        DataFrame for that day only
    """
    mask = rth_df["timestamp"].dt.date == d
    return rth_df[mask].copy().reset_index(drop=True)


def get_bar_by_time(day_df: pd.DataFrame, target_time: time) -> Optional[pd.Series]:
    """
    Get a specific bar by its time.

    Args:
        day_df: Single day RTH data
        target_time: Target time (e.g., time(10, 31))

    Returns:
        Bar as Series or None if not found
    """
    if day_df.empty:
        return None

    bar_times = day_df["timestamp"].dt.time
    matching = day_df[bar_times == target_time]

    if matching.empty:
        return None

    return matching.iloc[0]


if __name__ == "__main__":
    # Quick validation
    print("=== Data Loader Validation ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DSP100K_ROOT: {DSP100K_ROOT}")
    print(f"RTH_DATA_DIR: {RTH_DATA_DIR}")
    print()

    # Test RTH loading
    print("Testing RTH data loading...")
    spy_train = load_rth_data("SPY", "train")
    print(f"SPY train shape: {spy_train.shape}")
    print(f"Date range: {spy_train['timestamp'].min()} to {spy_train['timestamp'].max()}")
    print()

    # Test premarket loading
    print("Testing premarket data loading...")
    test_date = date(2021, 12, 20)
    pm_data = load_premarket_json("AAPL", test_date)
    if pm_data:
        pm_df = parse_premarket_bars(pm_data)
        print(f"AAPL premarket {test_date}: {len(pm_df)} bars")
        print(f"Time range: {pm_df['timestamp'].iloc[0]} to {pm_df['timestamp'].iloc[-1]}")
    else:
        print(f"No premarket data for AAPL on {test_date}")
    print()

    # Test 2023+ premarket
    test_date_2023 = date(2023, 1, 3)
    pm_data_2023 = load_premarket_json("AAPL", test_date_2023)
    if pm_data_2023:
        pm_df_2023 = parse_premarket_bars(pm_data_2023)
        print(f"AAPL premarket {test_date_2023}: {len(pm_df_2023)} bars")
    print()

    # Test FB→META fallback
    print("Testing FB→META fallback...")
    fb_date = date(2022, 1, 3)  # Before rename
    meta_pm = load_premarket_json("META", fb_date)
    if meta_pm:
        print(f"META premarket on {fb_date} (pre-rename): found via {meta_pm.get('symbol', 'unknown')}")

    print("\n=== Validation Complete ===")
