#!/usr/bin/env python3
"""
Backfill Premarket Data for DQN Training (Gate 2.7c)

Fetches premarket minute bars from Polygon.io for 2021-2022 training data.
Stores in dsp100k/data/dqn_premarket_cache/{SYMBOL}/{YYYY-MM-DD}.json

Usage:
    python scripts/dqn/backfill_premarket.py --start 2021-12-20 --end 2022-12-31
    python scripts/dqn/backfill_premarket.py --dry-run  # Preview without fetching
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


# DQN Universe
SYMBOLS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "SPY", "TSLA"]

# Polygon API configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io"

# Premarket session hours (ET)
PREMARKET_START = "04:00"  # 4:00 AM ET
PREMARKET_END = "09:30"    # 9:30 AM ET (RTH open)


def get_trading_calendar(start_date: str, end_date: str) -> list[str]:
    """Get NYSE trading dates between start and end (inclusive)."""
    # Use pandas market calendar if available, otherwise manual check
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        return [d.strftime("%Y-%m-%d") for d in schedule.index]
    except ImportError:
        # Fallback: return all business days (not perfect but close)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.date_range(start=start, end=end, freq="B")
        return [d.strftime("%Y-%m-%d") for d in dates]


def fetch_premarket_bars(symbol: str, date: str) -> list[dict]:
    """
    Fetch premarket minute bars from Polygon.io.

    Args:
        symbol: Ticker symbol (e.g., "AAPL")
        date: Date string (YYYY-MM-DD)

    Returns:
        List of bar dicts with timestamp, open, high, low, close, volume
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    # Convert date to timestamps for premarket window
    # Premarket: 04:00-09:30 ET
    # API expects Unix milliseconds
    date_obj = pd.Timestamp(date, tz="America/New_York")
    start_ts = date_obj.replace(hour=4, minute=0, second=0)
    end_ts = date_obj.replace(hour=9, minute=30, second=0)

    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute"
        f"/{start_ms}/{end_ms}?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK" or "results" not in data:
            return []

        bars = []
        for r in data["results"]:
            ts_ms = r["t"]
            ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_convert("America/New_York")

            bars.append({
                "timestamp": ts.isoformat(),
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"],
            })

        return bars

    except Exception as e:
        print(f"  ⚠️  Error fetching {symbol} {date}: {e}")
        return []


def save_premarket_cache(symbol: str, date: str, bars: list[dict], output_dir: Path):
    """Save premarket bars to cache file."""
    symbol_dir = output_dir / symbol.upper()
    symbol_dir.mkdir(parents=True, exist_ok=True)

    cache_file = symbol_dir / f"{date}.json"
    payload = {
        "symbol": symbol,
        "date": date,
        "session": "premarket",
        "bars": bars,
    }

    with open(cache_file, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Backfill premarket data for DQN training")
    parser.add_argument("--start", default="2021-12-20", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2022-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Symbols to backfill")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without fetching")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: dsp100k/data/dqn_premarket_cache/
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir.parents[1] / "data" / "dqn_premarket_cache"

    print(f"📦 Premarket Backfill (Gate 2.7c)")
    print(f"   Date range: {args.start} to {args.end}")
    print(f"   Symbols: {args.symbols}")
    print(f"   Output: {output_dir}")
    print()

    if not POLYGON_API_KEY and not args.dry_run:
        print("❌ POLYGON_API_KEY environment variable not set")
        print("   Set it with: export POLYGON_API_KEY=your_api_key")
        return 1

    # Get trading dates
    trading_dates = get_trading_calendar(args.start, args.end)
    print(f"📅 Found {len(trading_dates)} trading dates")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - would fetch:")
        total_calls = len(args.symbols) * len(trading_dates)
        print(f"   {total_calls} API calls ({len(args.symbols)} symbols × {len(trading_dates)} dates)")
        print(f"   Estimated time: {total_calls * args.delay / 60:.1f} minutes")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track progress
    total = len(args.symbols) * len(trading_dates)
    completed = 0
    errors = 0

    for symbol in args.symbols:
        print(f"📈 {symbol}:")

        for date in trading_dates:
            # Check if already cached
            cache_file = output_dir / symbol.upper() / f"{date}.json"
            if cache_file.exists():
                completed += 1
                continue

            # Fetch bars
            bars = fetch_premarket_bars(symbol, date)
            completed += 1

            if bars:
                save_premarket_cache(symbol, date, bars, output_dir)
                print(f"   ✅ {date}: {len(bars)} bars")
            else:
                errors += 1
                print(f"   ⚠️  {date}: no bars")

            # Rate limiting
            time.sleep(args.delay)

        print()

    print(f"✅ Complete: {completed}/{total} dates processed")
    if errors > 0:
        print(f"⚠️  Errors: {errors} dates with no data")

    return 0


if __name__ == "__main__":
    exit(main())
