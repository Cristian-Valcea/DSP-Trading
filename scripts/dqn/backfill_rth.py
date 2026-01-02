#!/usr/bin/env python3
"""
Gate 0: Backfill missing RTH (Regular Trading Hours) bars from Polygon.

This script identifies and backfills missing RTH data in stage1_raw.
It uses the same Polygon API that was used to create the original data.

Usage:
    # Preview what needs backfilling (dry-run)
    python scripts/dqn/backfill_rth.py --symbols META --dry-run

    # Actually backfill
    python scripts/dqn/backfill_rth.py --symbols META

    # Backfill all DQN universe symbols
    python scripts/dqn/backfill_rth.py --symbols all
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from time import sleep

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage1_raw"
QUALITY_REPORT = PROJECT_ROOT / "dsp100k" / "data" / "data_quality_report.json"

# DQN Universe (9 symbols)
DQN_UNIVERSE = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "SPY", "TSLA"]

# US Market Holidays (major ones, 2021-2025)
US_HOLIDAYS = {
    # 2021
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-04-02", "2021-05-31",
    "2021-07-05", "2021-09-06", "2021-11-25", "2021-12-24",
    # 2022
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20",
    "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-26",
    # 2023
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
    "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
}

# Polygon API
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
POLYGON_BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

RATE_LIMIT_CALLS_PER_MIN = 5  # Polygon Stocks Starter tier
RATE_LIMIT_DELAY_SECONDS = 60.0 / RATE_LIMIT_CALLS_PER_MIN + 0.5  # 12.5s + buffer

# Ticker changes (historical ticker → current ticker, effective date)
# FB was renamed to META on June 9, 2022
TICKER_CHANGES = {
    "META": ("FB", "2022-06-09"),  # Use FB before June 9, 2022
}


def get_polygon_bars(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """
    Fetch minute bars for a symbol/date from Polygon.

    Handles historical ticker changes (e.g., FB → META).
    Returns None if no data available.
    """
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    # Handle ticker changes (e.g., FB → META on 2022-06-09)
    api_ticker = symbol
    if symbol in TICKER_CHANGES:
        old_ticker, change_date = TICKER_CHANGES[symbol]
        if date < change_date:
            api_ticker = old_ticker

    url = f"{POLYGON_BASE_URL}/{api_ticker}/range/1/minute/{date}/{date}"
    params = {
        "apiKey": POLYGON_API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    # Internal retry on 429 (rate limit) to avoid silently skipping missing days.
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 429:
            break
        if attempt < 2:
            print("⏳ 429 rate limit, sleeping 60s...", end=" ")
            sleep(60)

    if response.status_code == 429:
        print(f"⚠️ still rate-limited (429)")
        return None

    if response.status_code != 200:
        print(f"   ⚠️  API error for {symbol} on {date}: {response.status_code} {response.text[:200]}")
        return None

    data = response.json()

    if data.get("resultsCount", 0) == 0:
        return None

    results = data.get("results", [])
    if not results:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Rename columns to match stage1_raw format
    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })

    # Convert timestamp (milliseconds) to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None)

    # Filter to RTH only (09:30 - 15:59)
    df = df[
        (df["timestamp"].dt.hour > 9) |
        ((df["timestamp"].dt.hour == 9) & (df["timestamp"].dt.minute >= 30))
    ]
    df = df[
        (df["timestamp"].dt.hour < 16)
    ]

    # Add symbol column
    df["symbol"] = symbol

    # Select and order columns to match stage1_raw format
    df = df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]

    return df


def identify_missing_days(symbol: str) -> list[str]:
    """Identify missing trading days for a symbol by analyzing the parquet file directly."""
    filepath = DATA_DIR / f"{symbol.lower()}_1min.parquet"

    if not filepath.exists():
        print(f"⚠️  Data file not found for {symbol}")
        return []

    # Load existing data
    df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)

    # Get actual trading days in data
    actual_days = set(df["date"].unique())

    # Get data range
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    # Generate expected trading days (weekdays minus holidays)
    all_days = pd.date_range(start=min_date, end=max_date, freq="B")
    expected_days = {
        d.strftime("%Y-%m-%d")
        for d in all_days
        if d.strftime("%Y-%m-%d") not in US_HOLIDAYS
    }

    # Find missing days
    missing_days = sorted(expected_days - actual_days)

    return missing_days


def backfill_symbol(symbol: str, dry_run: bool = True) -> dict:
    """Backfill missing data for a symbol."""
    print(f"\n📊 Processing {symbol}...")

    missing_days = identify_missing_days(symbol)

    if not missing_days:
        print(f"   ✅ No missing days to backfill")
        return {"symbol": symbol, "status": "no_action", "days_backfilled": 0}

    print(f"   Found {len(missing_days)} missing days")

    if dry_run:
        print(f"   🔍 DRY RUN - would backfill: {missing_days[:5]}{'...' if len(missing_days) > 5 else ''}")
        return {"symbol": symbol, "status": "dry_run", "missing_days": missing_days}

    # Load existing data
    filepath = DATA_DIR / f"{symbol.lower()}_1min.parquet"
    if filepath.exists():
        existing_df = pd.read_parquet(filepath)
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
    else:
        existing_df = pd.DataFrame()

    # Fetch and append missing days
    new_bars = []
    for date in missing_days:
        print(f"   📥 Fetching {date}...", end=" ")

        try:
            df = get_polygon_bars(symbol, date)
            if df is not None and len(df) > 0:
                new_bars.append(df)
                print(f"✅ {len(df)} bars")
            else:
                print("⚠️  No data (market closed? or API issue)")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Rate limiting (Starter tier default: 5 requests/minute)
        sleep(RATE_LIMIT_DELAY_SECONDS)

    if not new_bars:
        print(f"   ⚠️  No new data fetched")
        return {"symbol": symbol, "status": "no_data", "days_backfilled": 0}

    # Combine with existing data
    new_df = pd.concat(new_bars, ignore_index=True)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove duplicates and sort
    combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="first")
    combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

    # Save
    combined_df.to_parquet(filepath, index=False)
    print(f"   💾 Saved {len(combined_df)} total bars to {filepath.name}")

    return {
        "symbol": symbol,
        "status": "success",
        "days_backfilled": len(new_bars),
        "new_bars": sum(len(df) for df in new_bars),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing RTH bars from Polygon"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        help="Comma-separated symbols or 'all' for DQN universe",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be backfilled without making changes",
    )

    args = parser.parse_args()

    # Check API key
    if not args.dry_run and not POLYGON_API_KEY:
        print("❌ Error: POLYGON_API_KEY environment variable not set")
        print("   Set it with: export POLYGON_API_KEY=your_api_key")
        sys.exit(1)

    # Parse symbols
    if args.symbols.lower() == "all":
        symbols = DQN_UNIVERSE
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print("=" * 60)
    print("Gate 0: RTH Data Backfill")
    print("=" * 60)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")

    results = []
    for symbol in symbols:
        result = backfill_symbol(symbol, dry_run=args.dry_run)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success = sum(1 for r in results if r["status"] == "success")
    no_action = sum(1 for r in results if r["status"] in ("no_action", "no_data"))
    dry_run_count = sum(1 for r in results if r["status"] == "dry_run")

    if args.dry_run:
        print(f"📋 Dry run completed for {len(symbols)} symbols")
        print(f"   Would backfill: {len(symbols) - no_action}")
    else:
        print(f"✅ Backfilled: {success}")
        print(f"⏭️  Skipped: {no_action}")


if __name__ == "__main__":
    main()
