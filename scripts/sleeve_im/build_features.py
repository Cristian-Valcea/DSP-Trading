#!/usr/bin/env python3
"""
Phase 3: Build feature+label dataset for Sleeve IM.

This script:
1. Loads JSON cache from data/sleeve_im/minute_bars/
2. Converts to DailyMinuteBars (with carry-forward on 04:00-10:30 window)
3. Runs quality checks and filters out no-trade days
4. Computes features from the feature window
5. Loads labels from RTH parquet files (11:30 entry → 15:59 exit)
6. Outputs a single dataset CSV/parquet for training

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source venv/bin/activate
    python scripts/sleeve_im/build_features.py

Output:
    data/sleeve_im/feature_dataset.parquet  (or .csv)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure we can import from src/
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from dsp.data.data_quality import (
    MAX_STALENESS_SECONDS,
    MAX_SYNTHETIC_PCT,
    MIN_PREMARKET_VOLUME,
    assess_daily_quality,
)
from dsp.data.minute_bar import (
    DailyMinuteBars,
    MinuteBar,
    RawMinuteBar,
    build_minute_bars,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Sleeve IM spec: locked feature window
FEATURE_WINDOW_START = time(4, 0)   # 04:00 ET
FEATURE_WINDOW_END = time(10, 30)   # 10:30 ET

# Label price source (from RTH parquet)
ENTRY_TIME = time(11, 30)  # 11:30 bar open
EXIT_TIME = time(15, 59)   # 15:59 bar close (or early close for half-days)

# Universe
SYMBOLS = ["QQQ", "TSLA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY"]

# Date range (2023-2024)
DATE_START = date(2023, 1, 1)
DATE_END = date(2024, 12, 31)

# Paths
PROJECT_ROOT = Path(__file__).parents[2]
JSON_CACHE_DIR = PROJECT_ROOT / "data" / "sleeve_im" / "minute_bars"
RTH_DATA_DIR = Path("/Users/Shared/wsl-export/wsl-home/data/stage1_raw")
OUTPUT_DIR = PROJECT_ROOT / "data" / "sleeve_im"

# Known NYSE half-days (close at 13:00 ET)
HALF_DAYS = {
    "2023-07-03", "2023-11-24",
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
}


# =============================================================================
# Data Loading
# =============================================================================


def load_json_cache(symbol: str, trading_date: date) -> Optional[Dict[str, Any]]:
    """Load JSON cache file for a symbol-day."""
    path = JSON_CACHE_DIR / symbol / f"{trading_date}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def json_to_daily_bars(data: Dict[str, Any], symbol: str, trading_date: date) -> DailyMinuteBars:
    """
    Convert JSON cache to DailyMinuteBars with carry-forward on 04:00-10:30 window.

    The JSON cache contains raw Polygon bars (sparse, real bars only).
    This function applies carry-forward logic to create a complete minute grid.
    """
    # Parse raw bars from JSON
    raw_bars = []
    for bar_data in data.get("bars", []):
        ts = datetime.fromisoformat(bar_data["timestamp"])
        # Filter to feature window (04:00-10:30 ET)
        if ts.time() < FEATURE_WINDOW_START or ts.time() > FEATURE_WINDOW_END:
            continue
        try:
            raw_bars.append(RawMinuteBar(
                timestamp=ts,
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data.get("volume", 0),
                trade_count=bar_data.get("trade_count", 0),
                vwap=bar_data.get("vwap"),
            ))
        except ValueError as e:
            logger.warning(f"Invalid bar data for {symbol} {trading_date}: {e}")
            continue

    # Get prior close for carry-forward initialization
    # For now, use the first real bar's open as prior_close approximation
    prior_close = raw_bars[0].open if raw_bars else 0.0

    # Build complete minute bar series with carry-forward
    return build_minute_bars(
        symbol=symbol,
        trading_date=trading_date,
        raw_bars=raw_bars,
        prior_session_last_price=prior_close,
        feature_window_start=FEATURE_WINDOW_START,
        feature_window_end=FEATURE_WINDOW_END,
    )


def load_rth_parquet(symbol: str) -> Optional[pd.DataFrame]:
    """Load RTH parquet file for a symbol."""
    path = RTH_DATA_DIR / f"{symbol.lower()}_1min.parquet"
    if not path.exists():
        logger.warning(f"RTH parquet not found: {path}")
        return None
    df = pd.read_parquet(path)
    # Ensure timestamp column is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_label_prices(
    rth_df: pd.DataFrame,
    trading_date: date,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get entry (11:30 open) and exit (15:59 close) prices from RTH data.

    Returns (entry_price, exit_price) or (None, None) if data missing.
    """
    day_data = rth_df[rth_df["timestamp"].dt.date == trading_date]
    if day_data.empty:
        return None, None

    date_str = trading_date.strftime("%Y-%m-%d")
    is_half_day = date_str in HALF_DAYS

    # Entry: 11:30 bar open
    entry_bar = day_data[day_data["timestamp"].dt.time == ENTRY_TIME]
    entry_price = entry_bar["open"].iloc[0] if len(entry_bar) > 0 else None

    # Exit: 15:59 bar close (normal day) or ~13:00 (half-day)
    exit_bar = day_data[day_data["timestamp"].dt.time == EXIT_TIME]
    if len(exit_bar) > 0:
        exit_price = exit_bar["close"].iloc[0]
    elif is_half_day:
        # Half-day: accept last bar if in 12:55-13:05 window
        last_bar = day_data.iloc[-1]
        last_time = last_bar["timestamp"].time()
        if time(12, 55) <= last_time <= time(13, 5):
            exit_price = last_bar["close"]
        else:
            exit_price = None  # Unexpected half-day close time
    else:
        exit_price = None  # Normal day missing 15:59 → skip

    return entry_price, exit_price


# =============================================================================
# Feature Computation
# =============================================================================


@dataclass
class FeatureRow:
    """Features for a single symbol-day."""

    # Identifiers
    symbol: str
    date: date

    # Quality metrics
    synthetic_pct: float
    max_staleness_min: float
    premarket_volume: int
    is_tradable: bool

    # Return features
    premarket_return: Optional[float]  # 04:00 → 09:30
    first_hour_return: Optional[float]  # 09:30 → 10:30
    feature_window_return: Optional[float]  # 04:00 → 10:30

    # Volume features
    total_volume: int
    premarket_vol_ratio: Optional[float]  # premarket / total

    # Volatility features
    premarket_volatility: Optional[float]  # std of returns
    max_bar_range_pct: float

    # Staleness features
    synthetic_bar_count: int
    stale_bar_count: int  # bars with staleness > 30 min
    avg_staleness_sec: float

    # Labels (from RTH data)
    entry_price: Optional[float]
    exit_price: Optional[float]
    label_return: Optional[float]  # exit / entry - 1
    label_binary: Optional[int]  # 1 if return > 0, else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame."""
        return {
            "symbol": self.symbol,
            "date": self.date,
            "synthetic_pct": self.synthetic_pct,
            "max_staleness_min": self.max_staleness_min,
            "premarket_volume": self.premarket_volume,
            "is_tradable": self.is_tradable,
            "premarket_return": self.premarket_return,
            "first_hour_return": self.first_hour_return,
            "feature_window_return": self.feature_window_return,
            "total_volume": self.total_volume,
            "premarket_vol_ratio": self.premarket_vol_ratio,
            "premarket_volatility": self.premarket_volatility,
            "max_bar_range_pct": self.max_bar_range_pct,
            "synthetic_bar_count": self.synthetic_bar_count,
            "stale_bar_count": self.stale_bar_count,
            "avg_staleness_sec": self.avg_staleness_sec,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "label_return": self.label_return,
            "label_binary": self.label_binary,
        }


def compute_features(
    daily_bars: DailyMinuteBars,
    entry_price: Optional[float],
    exit_price: Optional[float],
) -> FeatureRow:
    """
    Compute features from DailyMinuteBars.

    This is a simplified feature set for the baseline model.
    Full 54-feature set will be added in Phase 4.
    """
    # Quality assessment
    quality_report = assess_daily_quality(daily_bars)

    # Return features
    premarket_return = daily_bars.get_return(FEATURE_WINDOW_START, time(9, 30))
    first_hour_return = daily_bars.get_return(time(9, 30), FEATURE_WINDOW_END)
    feature_window_return = daily_bars.get_return(FEATURE_WINDOW_START, FEATURE_WINDOW_END)

    # Volume features
    total_vol = daily_bars.total_volume
    premarket_vol = daily_bars.premarket_volume
    premarket_vol_ratio = premarket_vol / total_vol if total_vol > 0 else None

    # Volatility features
    real_bars = [b for b in daily_bars.bars if not b.is_synthetic]
    if len(real_bars) >= 2:
        closes = [b.close for b in real_bars]
        returns = np.diff(closes) / closes[:-1]
        premarket_volatility = float(np.std(returns)) if len(returns) > 0 else None
    else:
        premarket_volatility = None

    max_bar_range_pct = max((b.bar_range_pct for b in real_bars), default=0.0)

    # Staleness features
    staleness_values = [
        b.seconds_since_last_trade
        for b in daily_bars.bars
        if b.seconds_since_last_trade != float("inf")
    ]
    avg_staleness = np.mean(staleness_values) if staleness_values else 0.0
    stale_bar_count = sum(1 for s in staleness_values if s > 1800)  # > 30 min

    # Label computation
    if entry_price is not None and exit_price is not None:
        label_return = exit_price / entry_price - 1
        label_binary = 1 if label_return > 0 else 0
    else:
        label_return = None
        label_binary = None

    return FeatureRow(
        symbol=daily_bars.symbol,
        date=daily_bars.date,
        synthetic_pct=quality_report.synthetic_pct,
        max_staleness_min=quality_report.max_staleness_seconds / 60.0,
        premarket_volume=quality_report.premarket_volume,
        is_tradable=quality_report.is_tradable,
        premarket_return=premarket_return,
        first_hour_return=first_hour_return,
        feature_window_return=feature_window_return,
        total_volume=total_vol,
        premarket_vol_ratio=premarket_vol_ratio,
        premarket_volatility=premarket_volatility,
        max_bar_range_pct=max_bar_range_pct,
        synthetic_bar_count=daily_bars.synthetic_bars,
        stale_bar_count=stale_bar_count,
        avg_staleness_sec=avg_staleness,
        entry_price=entry_price,
        exit_price=exit_price,
        label_return=label_return,
        label_binary=label_binary,
    )


# =============================================================================
# Main Pipeline
# =============================================================================


def get_trading_days(start: date, end: date) -> List[date]:
    """Get list of trading days (weekdays, excluding holidays)."""
    # Simple approach: use files that exist in JSON cache
    # This inherits the holiday list from the backfill script
    sample_dir = JSON_CACHE_DIR / "AAPL"
    if not sample_dir.exists():
        logger.error(f"JSON cache directory not found: {sample_dir}")
        return []

    days = []
    for f in sorted(sample_dir.glob("*.json")):
        d = date.fromisoformat(f.stem)
        if start <= d <= end:
            days.append(d)
    return days


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Phase 3: Build Feature+Label Dataset")
    logger.info("=" * 60)

    # Load RTH data for all symbols
    logger.info("Loading RTH parquet files...")
    rth_data: Dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        df = load_rth_parquet(symbol)
        if df is not None:
            rth_data[symbol] = df
            logger.info(f"  {symbol}: {len(df):,} bars")
        else:
            logger.warning(f"  {symbol}: RTH data not found!")

    # Get trading days from JSON cache
    trading_days = get_trading_days(DATE_START, DATE_END)
    logger.info(f"Trading days: {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})")

    # Process each symbol-day
    feature_rows: List[Dict[str, Any]] = []

    stats = {
        "total": 0,
        "loaded": 0,
        "tradable": 0,
        "has_label": 0,
        "quality_fail": 0,
        "missing_json": 0,
        "missing_label": 0,
    }

    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}...")
        rth_df = rth_data.get(symbol)

        for trading_date in trading_days:
            stats["total"] += 1

            # Load JSON cache
            json_data = load_json_cache(symbol, trading_date)
            if json_data is None:
                stats["missing_json"] += 1
                continue
            stats["loaded"] += 1

            # Convert to DailyMinuteBars (with carry-forward on 04:00-10:30)
            daily_bars = json_to_daily_bars(json_data, symbol, trading_date)

            # Get labels from RTH data
            if rth_df is not None:
                entry_price, exit_price = get_label_prices(rth_df, trading_date)
            else:
                entry_price, exit_price = None, None

            # Compute features
            features = compute_features(daily_bars, entry_price, exit_price)

            # Track stats
            if features.is_tradable:
                stats["tradable"] += 1
            else:
                stats["quality_fail"] += 1

            if features.label_binary is not None:
                stats["has_label"] += 1
            else:
                stats["missing_label"] += 1

            feature_rows.append(features.to_dict())

    # Create DataFrame
    logger.info("Creating feature dataset...")
    df = pd.DataFrame(feature_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Summary stats
    logger.info("=" * 60)
    logger.info("Dataset Summary")
    logger.info("=" * 60)
    logger.info(f"Total symbol-days attempted: {stats['total']:,}")
    logger.info(f"JSON cache loaded: {stats['loaded']:,}")
    logger.info(f"Quality PASS (tradable): {stats['tradable']:,}")
    logger.info(f"Quality FAIL (no-trade): {stats['quality_fail']:,}")
    logger.info(f"Has label: {stats['has_label']:,}")
    logger.info(f"Missing label: {stats['missing_label']:,}")
    logger.info(f"Missing JSON: {stats['missing_json']:,}")

    # Quality fail breakdown
    quality_fails = df[~df["is_tradable"]]
    if len(quality_fails) > 0:
        logger.info("\nQuality fail breakdown:")
        high_synth = (quality_fails["synthetic_pct"] > MAX_SYNTHETIC_PCT).sum()
        high_stale = (quality_fails["max_staleness_min"] > MAX_STALENESS_SECONDS / 60).sum()
        low_vol = (quality_fails["premarket_volume"] < MIN_PREMARKET_VOLUME).sum()
        logger.info(f"  High synthetic (>{MAX_SYNTHETIC_PCT:.0%}): {high_synth:,}")
        logger.info(f"  High staleness (>{MAX_STALENESS_SECONDS/60:.0f}m): {high_stale:,}")
        logger.info(f"  Low volume (<{MIN_PREMARKET_VOLUME:,}): {low_vol:,}")

    # Label distribution (for tradable days only)
    tradable = df[df["is_tradable"] & df["label_binary"].notna()]
    if len(tradable) > 0:
        label_1 = (tradable["label_binary"] == 1).sum()
        label_0 = (tradable["label_binary"] == 0).sum()
        logger.info(f"\nLabel distribution (tradable days with labels):")
        logger.info(f"  Positive (1): {label_1:,} ({label_1/len(tradable):.1%})")
        logger.info(f"  Negative (0): {label_0:,} ({label_0/len(tradable):.1%})")

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_parquet = OUTPUT_DIR / "feature_dataset.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"\nSaved: {output_parquet}")

    output_csv = OUTPUT_DIR / "feature_dataset.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved: {output_csv}")

    # Print sample
    logger.info("\nSample rows:")
    print(df.head(10).to_string())

    return df


if __name__ == "__main__":
    main()
