#!/usr/bin/env python3
"""
Gate 0: Audit missing minute bars in stage1_raw data.

This script checks for:
1. Missing trading days
2. Missing bars within trading days
3. Data quality issues (NaN values, zero volumes)

RTH (Regular Trading Hours): 09:30 - 15:59 ET = 390 bars per day

Usage:
    python scripts/dqn/audit_missing_bars.py --symbols AAPL,GOOGL,MSFT
    python scripts/dqn/audit_missing_bars.py --symbols all --output data/data_quality_report.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "stage1_raw"

# DQN Universe (9 symbols)
DQN_UNIVERSE = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "SPY", "TSLA"]

# RTH expectations
RTH_START_HOUR = 9
RTH_START_MINUTE = 30
RTH_END_HOUR = 15
RTH_END_MINUTE = 59
EXPECTED_BARS_PER_DAY = 390  # 09:30 to 15:59 inclusive

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

# Early close days (close at 13:00 ET = 210 bars)
EARLY_CLOSE_DAYS = {
    "2021-11-26", "2022-11-25", "2023-11-24", "2024-11-29",  # Day after Thanksgiving
    "2021-12-24", "2023-12-24", "2024-12-24",  # Christmas Eve (when not weekend)
}
EXPECTED_BARS_EARLY_CLOSE = 210  # 09:30 to 12:59


def load_symbol_data(symbol: str) -> pd.DataFrame:
    """Load minute data for a symbol from stage1_raw."""
    filepath = DATA_DIR / f"{symbol.lower()}_1min.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    return df


def get_expected_trading_days(start_date: str, end_date: str) -> set[str]:
    """Get set of expected trading days (weekdays minus holidays)."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Generate all weekdays
    all_days = pd.date_range(start=start, end=end, freq="B")  # Business days

    # Remove holidays
    trading_days = {d.strftime("%Y-%m-%d") for d in all_days if d.strftime("%Y-%m-%d") not in US_HOLIDAYS}

    return trading_days


def audit_symbol(symbol: str) -> dict:
    """
    Audit a single symbol's data quality.

    Returns:
        dict with audit results
    """
    df = load_symbol_data(symbol)

    # Data range
    min_date = df["timestamp"].min()
    max_date = df["timestamp"].max()

    # Group by date
    daily_counts = df.groupby("date").size().to_dict()

    # Get expected trading days
    expected_days = get_expected_trading_days(
        min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d")
    )

    # Analyze each day
    actual_days = set(daily_counts.keys())
    missing_days = expected_days - actual_days
    extra_days = actual_days - expected_days  # Weekend/holiday data (not a problem)

    # Check bar counts per day
    incomplete_days = []
    for date_str, count in daily_counts.items():
        expected = EXPECTED_BARS_EARLY_CLOSE if date_str in EARLY_CLOSE_DAYS else EXPECTED_BARS_PER_DAY

        # Allow some tolerance (5 bars = ~1%)
        if count < expected - 5:
            incomplete_days.append({
                "date": date_str,
                "expected": expected,
                "actual": count,
                "missing": expected - count,
                "pct_complete": round(count / expected * 100, 1),
            })

    # Check for data quality issues
    nan_rows = df[df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)]
    zero_volume_rows = df[df["volume"] == 0]

    # Time continuity check (gaps within days)
    df_sorted = df.sort_values("timestamp")
    df_sorted["time_diff"] = df_sorted["timestamp"].diff()

    # Find gaps > 1 minute (excluding overnight/weekend gaps)
    intraday_gaps = []
    for idx, row in df_sorted.iterrows():
        if pd.notna(row["time_diff"]):
            diff_minutes = row["time_diff"].total_seconds() / 60
            # Intraday gap: > 1 minute but < 1 day
            if 1.5 < diff_minutes < 60 * 16:  # Between 1.5 min and 16 hours
                prev_ts = row["timestamp"] - row["time_diff"]
                # Check if it's within same trading day
                if prev_ts.date() == row["timestamp"].date():
                    intraday_gaps.append({
                        "from": str(prev_ts),
                        "to": str(row["timestamp"]),
                        "gap_minutes": round(diff_minutes, 1),
                    })

    # Calculate coverage metrics
    total_expected_days = len(expected_days)
    total_actual_days = len(actual_days & expected_days)
    day_coverage = total_actual_days / total_expected_days * 100 if total_expected_days > 0 else 0

    # Bar-level coverage
    total_expected_bars = sum(
        EXPECTED_BARS_EARLY_CLOSE if d in EARLY_CLOSE_DAYS else EXPECTED_BARS_PER_DAY
        for d in expected_days
    )
    total_actual_bars = len(df)
    bar_coverage = total_actual_bars / total_expected_bars * 100 if total_expected_bars > 0 else 0

    return {
        "symbol": symbol,
        "data_range": {
            "start": str(min_date),
            "end": str(max_date),
            "total_rows": len(df),
        },
        "coverage": {
            "expected_trading_days": total_expected_days,
            "actual_trading_days": total_actual_days,
            "day_coverage_pct": round(day_coverage, 2),
            "expected_bars": total_expected_bars,
            "actual_bars": total_actual_bars,
            "bar_coverage_pct": round(bar_coverage, 2),
        },
        "issues": {
            "missing_days_count": len(missing_days),
            "missing_days_sample": sorted(list(missing_days))[:10],  # First 10
            "incomplete_days_count": len(incomplete_days),
            "incomplete_days_sample": incomplete_days[:10],  # First 10
            "nan_rows": len(nan_rows),
            "zero_volume_rows": len(zero_volume_rows),
            "intraday_gaps_count": len(intraday_gaps),
            "intraday_gaps_sample": intraday_gaps[:10],  # First 10
        },
        "passed": (
            len(missing_days) < total_expected_days * 0.01 and  # <1% missing days
            len(nan_rows) == 0 and  # No NaN values
            bar_coverage > 98  # >98% bar coverage
        ),
    }


def audit_all_symbols(symbols: list[str], output_path: Optional[str] = None) -> dict:
    """Audit multiple symbols and generate report."""
    print("=" * 70)
    print("Gate 0: Missing Bars Audit")
    print("=" * 70)

    results = {}
    all_passed = True

    for symbol in symbols:
        print(f"\n📊 Auditing {symbol}...")

        try:
            result = audit_symbol(symbol)
            results[symbol] = result

            # Print summary
            cov = result["coverage"]
            issues = result["issues"]

            if result["passed"]:
                print(f"   ✅ PASS")
            else:
                print(f"   ❌ FAIL")
                all_passed = False

            print(f"      Date range: {result['data_range']['start'][:10]} to {result['data_range']['end'][:10]}")
            print(f"      Day coverage: {cov['actual_trading_days']}/{cov['expected_trading_days']} ({cov['day_coverage_pct']}%)")
            print(f"      Bar coverage: {cov['actual_bars']:,}/{cov['expected_bars']:,} ({cov['bar_coverage_pct']}%)")

            if issues["missing_days_count"] > 0:
                print(f"      ⚠️  Missing days: {issues['missing_days_count']}")
            if issues["incomplete_days_count"] > 0:
                print(f"      ⚠️  Incomplete days: {issues['incomplete_days_count']}")
            if issues["nan_rows"] > 0:
                print(f"      ⚠️  NaN rows: {issues['nan_rows']}")
            if issues["intraday_gaps_count"] > 0:
                print(f"      ⚠️  Intraday gaps: {issues['intraday_gaps_count']}")

        except FileNotFoundError as e:
            print(f"   ⚠️  SKIP: {e}")
            results[symbol] = {"error": str(e), "passed": None}
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results[symbol] = {"error": str(e), "passed": False}
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for r in results.values() if r.get("passed") is True)
    failed_count = sum(1 for r in results.values() if r.get("passed") is False)
    skipped_count = sum(1 for r in results.values() if r.get("passed") is None)

    print(f"✅ Passed: {passed_count}/{len(symbols)}")
    print(f"❌ Failed: {failed_count}/{len(symbols)}")
    print(f"⚠️  Skipped: {skipped_count}/{len(symbols)}")

    # Save to file if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {output_path}")

    if all_passed and failed_count == 0:
        print("\n🎉 Gate 0 (Missing Bars Audit): PASSED")
    else:
        print("\n💀 Gate 0 (Missing Bars Audit): FAILED")

    return {
        "results": results,
        "summary": {
            "passed": passed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "all_passed": all_passed and failed_count == 0,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit missing minute bars in stage1_raw data"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        help="Comma-separated symbols or 'all' for DQN universe",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path for detailed report",
    )

    args = parser.parse_args()

    # Parse symbols
    if args.symbols.lower() == "all":
        symbols = DQN_UNIVERSE
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Run audit
    report = audit_all_symbols(symbols, args.output)

    sys.exit(0 if report["summary"]["all_passed"] else 1)


if __name__ == "__main__":
    main()
