#!/usr/bin/env python3
"""
Gate 0: Verify stock split adjustments in stage1_raw data.

This script checks that major stock splits are properly adjusted in the historical data.
A properly adjusted dataset should show NO discontinuity around split dates.

Known splits in our universe (2021-2024):
- GOOGL: 20:1 split on July 15, 2022 (pre-split ~$2,200 → post-split ~$110)
- AMZN: 20:1 split on June 6, 2022 (pre-split ~$2,400 → post-split ~$120)
- TSLA: 3:1 split on August 25, 2022 (pre-split ~$900 → post-split ~$300)

Usage:
    python scripts/dqn/verify_splits.py --symbol GOOGL --date 2022-07-15
    python scripts/dqn/verify_splits.py --all  # Check all known splits
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage1_raw"

# Known splits with expected price ranges (split-adjusted)
KNOWN_SPLITS = {
    "GOOGL": {
        "date": "2022-07-15",
        "ratio": 20,
        "expected_range": (100, 140),  # Post-split adjusted price range
    },
    "AMZN": {
        "date": "2022-06-06",
        "ratio": 20,
        "expected_range": (100, 150),
    },
    "TSLA": {
        "date": "2022-08-25",
        "ratio": 3,
        "expected_range": (250, 350),
    },
}


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load minute data for a symbol from stage1_raw."""
    filepath = DATA_DIR / f"{symbol.lower()}_1min.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def check_split_adjustment(
    symbol: str,
    split_date: str,
    ratio: int,
    expected_range: tuple[float, float],
    window_days: int = 5,
) -> dict:
    """
    Check if a stock split is properly adjusted in the data.

    Returns a dict with:
    - passed: bool
    - details: dict with diagnostic info
    """
    df = load_symbol_data(symbol)
    split_dt = pd.Timestamp(split_date)

    # Get data around split date
    before_start = split_dt - timedelta(days=window_days)
    after_end = split_dt + timedelta(days=window_days)

    before_df = df[(df["timestamp"] >= before_start) & (df["timestamp"] < split_dt)]
    after_df = df[(df["timestamp"] >= split_dt) & (df["timestamp"] <= after_end)]

    if before_df.empty or after_df.empty:
        return {
            "passed": False,
            "details": {
                "error": f"Missing data around split date {split_date}",
                "before_rows": len(before_df),
                "after_rows": len(after_df),
            }
        }

    # Get price statistics
    before_close = before_df["close"].iloc[-1]  # Last close before split
    after_open = after_df["close"].iloc[0]  # First close after split

    before_mean = before_df["close"].mean()
    after_mean = after_df["close"].mean()

    # Check for discontinuity
    # If NOT adjusted: after_open ≈ before_close / ratio
    # If adjusted: after_open ≈ before_close (no discontinuity)
    price_change_pct = abs(after_open - before_close) / before_close * 100

    # In an unadjusted dataset, we'd see a ~95% drop for a 20:1 split
    # In an adjusted dataset, the change should be normal market movement (<10%)
    is_adjusted = price_change_pct < 15  # Allow 15% for normal market movement

    # Also check that prices are in expected range
    all_prices = df["close"]
    prices_in_range = (
        (all_prices >= expected_range[0] - 20).all() and
        (all_prices <= expected_range[1] * 10).all()  # Allow for price appreciation
    )

    # Check for any obvious 20x or 3x discontinuities
    df_sorted = df.sort_values("timestamp")
    df_sorted["close_pct_change"] = df_sorted["close"].pct_change()
    max_change = df_sorted["close_pct_change"].abs().max()

    # A 20:1 split would show as -95% or +1900% change
    has_extreme_discontinuity = max_change > 0.5  # 50% intraday change is suspicious

    passed = is_adjusted and not has_extreme_discontinuity

    return {
        "passed": passed,
        "details": {
            "symbol": symbol,
            "split_date": split_date,
            "split_ratio": f"{ratio}:1",
            "before_close": round(before_close, 2),
            "after_open": round(after_open, 2),
            "price_change_pct": round(price_change_pct, 2),
            "is_adjusted": is_adjusted,
            "before_mean": round(before_mean, 2),
            "after_mean": round(after_mean, 2),
            "max_pct_change": round(max_change * 100, 2),
            "has_extreme_discontinuity": has_extreme_discontinuity,
            "data_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            "total_rows": len(df),
        }
    }


def verify_single_split(symbol: str, date: str) -> bool:
    """Verify a single split (used when --symbol and --date provided)."""
    if symbol.upper() not in KNOWN_SPLITS:
        print(f"⚠️  Warning: {symbol} not in known splits list, using provided date")
        # Use default parameters
        result = check_split_adjustment(
            symbol=symbol.upper(),
            split_date=date,
            ratio=20,  # Assume 20:1 as common
            expected_range=(50, 500),
        )
    else:
        split_info = KNOWN_SPLITS[symbol.upper()]
        result = check_split_adjustment(
            symbol=symbol.upper(),
            split_date=split_info["date"],
            ratio=split_info["ratio"],
            expected_range=split_info["expected_range"],
        )

    if result["passed"]:
        print(f"✅ PASS: {symbol} split adjustment verified")
    else:
        print(f"❌ FAIL: {symbol} split adjustment FAILED")

    print(f"   Details:")
    for key, value in result["details"].items():
        print(f"     {key}: {value}")

    return result["passed"]


def verify_all_splits() -> bool:
    """Verify all known splits."""
    print("=" * 60)
    print("Gate 0: Stock Split Adjustment Verification")
    print("=" * 60)

    all_passed = True
    results = []

    for symbol, split_info in KNOWN_SPLITS.items():
        print(f"\n📊 Checking {symbol} ({split_info['ratio']}:1 split on {split_info['date']})...")

        try:
            result = check_split_adjustment(
                symbol=symbol,
                split_date=split_info["date"],
                ratio=split_info["ratio"],
                expected_range=split_info["expected_range"],
            )
            results.append((symbol, result))

            if result["passed"]:
                print(f"   ✅ PASS: Data is properly split-adjusted")
                print(f"      Price around split: ${result['details']['before_close']:.2f} → ${result['details']['after_open']:.2f}")
                print(f"      Change: {result['details']['price_change_pct']:.1f}% (expected <15% for adjusted data)")
            else:
                print(f"   ❌ FAIL: Data may NOT be split-adjusted!")
                print(f"      Price around split: ${result['details']['before_close']:.2f} → ${result['details']['after_open']:.2f}")
                print(f"      Change: {result['details']['price_change_pct']:.1f}%")
                if result["details"].get("has_extreme_discontinuity"):
                    print(f"      ⚠️  Found extreme discontinuity: {result['details']['max_pct_change']:.1f}%")
                all_passed = False

        except FileNotFoundError as e:
            print(f"   ⚠️  SKIP: Data file not found for {symbol}")
            results.append((symbol, {"passed": None, "details": {"error": str(e)}}))
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append((symbol, {"passed": False, "details": {"error": str(e)}}))
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, r in results if r["passed"] is True)
    failed_count = sum(1 for _, r in results if r["passed"] is False)
    skipped_count = sum(1 for _, r in results if r["passed"] is None)

    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⚠️  Skipped: {skipped_count}")

    if all_passed and failed_count == 0:
        print("\n🎉 Gate 0 (Split Verification): PASSED")
    else:
        print("\n💀 Gate 0 (Split Verification): FAILED")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify stock split adjustments in stage1_raw data"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Symbol to check (e.g., GOOGL)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Split date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all known splits",
    )

    args = parser.parse_args()

    if args.all:
        success = verify_all_splits()
    elif args.symbol:
        if not args.date and args.symbol.upper() not in KNOWN_SPLITS:
            print(f"Error: --date required for unknown symbol {args.symbol}")
            sys.exit(1)
        date = args.date or KNOWN_SPLITS[args.symbol.upper()]["date"]
        success = verify_single_split(args.symbol, date)
    else:
        print("Usage: verify_splits.py --symbol GOOGL [--date 2022-07-15]")
        print("       verify_splits.py --all")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
