#!/usr/bin/env python3
"""
Backfill historical minute bar data for Sleeve IM.

This script fetches and caches 1-minute aggregates from Polygon.io
for the specified symbols and date range.

Usage:
    # Backfill default symbols (SPY, QQQ, IWM) for 2020-2024
    python scripts/sleeve_im/backfill_data.py

    # Backfill specific symbols
    python scripts/sleeve_im/backfill_data.py --symbols AAPL,MSFT,GOOGL

    # Backfill specific date range
    python scripts/sleeve_im/backfill_data.py --start 2023-01-01 --end 2023-12-31

    # Dry run (show what would be fetched)
    python scripts/sleeve_im/backfill_data.py --dry-run

    # Force refresh (ignore cache)
    python scripts/sleeve_im/backfill_data.py --force

Requirements:
    - POLYGON_API_KEY environment variable set
    - Polygon.io Starter tier ($29/mo) or higher
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dsp.data.polygon_fetcher import PolygonConfig, PolygonFetcher
from dsp.data.data_quality import DataQualityMonitor, get_quality_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default symbols for Sleeve IM universe
DEFAULT_SYMBOLS = [
    # Core ETFs
    "SPY",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000
    # Sector ETFs
    "XLF",  # Financials
    "XLK",  # Technology
    "XLE",  # Energy
    "XLV",  # Healthcare
    "XLI",  # Industrials
    # Large caps for single-stock testing
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
]

# Default date range (2020-2024)
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date(2024, 12, 31)

# Rate limiting (Polygon Starter tier)
REQUESTS_PER_MINUTE = 5
BATCH_SIZE = 5  # Symbols per batch


# =============================================================================
# Trading Calendar (Simple)
# =============================================================================


def is_trading_day(d: date) -> bool:
    """
    Check if date is a trading day (weekday).

    Note: This is a simplified check. For production, use a proper
    trading calendar that accounts for holidays.
    """
    return d.weekday() < 5  # Monday = 0, Friday = 4


def get_trading_days(start: date, end: date) -> List[date]:
    """
    Get list of trading days in date range.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)

    Returns:
        List of trading dates
    """
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


# Major US market holidays (simplified list)
US_HOLIDAYS = {
    # 2020
    date(2020, 1, 1),   # New Year's Day
    date(2020, 1, 20),  # MLK Day
    date(2020, 2, 17),  # Presidents Day
    date(2020, 4, 10),  # Good Friday
    date(2020, 5, 25),  # Memorial Day
    date(2020, 7, 3),   # Independence Day (observed)
    date(2020, 9, 7),   # Labor Day
    date(2020, 11, 26), # Thanksgiving
    date(2020, 12, 25), # Christmas

    # 2021
    date(2021, 1, 1),
    date(2021, 1, 18),
    date(2021, 2, 15),
    date(2021, 4, 2),
    date(2021, 5, 31),
    date(2021, 7, 5),
    date(2021, 9, 6),
    date(2021, 11, 25),
    date(2021, 12, 24),

    # 2022
    date(2022, 1, 17),
    date(2022, 2, 21),
    date(2022, 4, 15),
    date(2022, 5, 30),
    date(2022, 6, 20),  # Juneteenth
    date(2022, 7, 4),
    date(2022, 9, 5),
    date(2022, 11, 24),
    date(2022, 12, 26),

    # 2023
    date(2023, 1, 2),
    date(2023, 1, 16),
    date(2023, 2, 20),
    date(2023, 4, 7),
    date(2023, 5, 29),
    date(2023, 6, 19),
    date(2023, 7, 4),
    date(2023, 9, 4),
    date(2023, 11, 23),
    date(2023, 12, 25),

    # 2024
    date(2024, 1, 1),
    date(2024, 1, 15),
    date(2024, 2, 19),
    date(2024, 3, 29),
    date(2024, 5, 27),
    date(2024, 6, 19),
    date(2024, 7, 4),
    date(2024, 9, 2),
    date(2024, 11, 28),
    date(2024, 12, 25),
}


def get_trading_days_excluding_holidays(start: date, end: date) -> List[date]:
    """Get trading days excluding known holidays."""
    days = get_trading_days(start, end)
    return [d for d in days if d not in US_HOLIDAYS]


# =============================================================================
# Backfill Logic
# =============================================================================


async def backfill_symbol(
    fetcher: PolygonFetcher,
    symbol: str,
    trading_days: List[date],
    quality_monitor: DataQualityMonitor,
    force_refresh: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Backfill data for a single symbol.

    Args:
        fetcher: PolygonFetcher instance
        symbol: Stock symbol
        trading_days: List of trading dates to fetch
        quality_monitor: DataQualityMonitor instance
        force_refresh: Force fetch even if cached
        dry_run: If True, don't actually fetch

    Returns:
        Dict with statistics
    """
    stats = {
        "symbol": symbol,
        "total_days": len(trading_days),
        "fetched": 0,
        "cached": 0,
        "failed": 0,
        "quality_pass": 0,
        "quality_fail": 0,
    }

    for trading_date in trading_days:
        # Check cache
        if not force_refresh and fetcher._is_cached(symbol, trading_date):
            stats["cached"] += 1
            continue

        if dry_run:
            logger.info("[DRY RUN] Would fetch %s %s", symbol, trading_date)
            stats["fetched"] += 1
            continue

        try:
            # Fetch data
            daily_bars = await fetcher.get_minute_bars(
                symbol=symbol,
                trading_date=trading_date,
                force_refresh=force_refresh,
            )
            stats["fetched"] += 1

            # Check quality
            is_tradable, report = quality_monitor.check(daily_bars)
            if is_tradable:
                stats["quality_pass"] += 1
            else:
                stats["quality_fail"] += 1

        except Exception as e:
            logger.error("Failed to fetch %s %s: %s", symbol, trading_date, e)
            stats["failed"] += 1

    return stats


async def backfill_all(
    symbols: List[str],
    start_date: date,
    end_date: date,
    force_refresh: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Backfill data for all symbols.

    Args:
        symbols: List of stock symbols
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        force_refresh: Force fetch even if cached
        dry_run: If True, don't actually fetch

    Returns:
        Dict with aggregate statistics
    """
    # Get trading days
    trading_days = get_trading_days_excluding_holidays(start_date, end_date)
    logger.info(
        "Backfilling %d symbols for %d trading days (%s to %s)",
        len(symbols),
        len(trading_days),
        start_date,
        end_date,
    )

    # Initialize components
    config = PolygonConfig.from_env()
    quality_monitor = DataQualityMonitor()

    # Aggregate stats
    total_stats = {
        "symbols": len(symbols),
        "trading_days": len(trading_days),
        "total_requests": len(symbols) * len(trading_days),
        "fetched": 0,
        "cached": 0,
        "failed": 0,
        "quality_pass": 0,
        "quality_fail": 0,
    }

    async with PolygonFetcher(config) as fetcher:
        # Process symbols in batches to respect rate limits
        for i, symbol in enumerate(symbols):
            logger.info(
                "Processing symbol %d/%d: %s",
                i + 1,
                len(symbols),
                symbol,
            )

            stats = await backfill_symbol(
                fetcher=fetcher,
                symbol=symbol,
                trading_days=trading_days,
                quality_monitor=quality_monitor,
                force_refresh=force_refresh,
                dry_run=dry_run,
            )

            # Aggregate stats
            total_stats["fetched"] += stats["fetched"]
            total_stats["cached"] += stats["cached"]
            total_stats["failed"] += stats["failed"]
            total_stats["quality_pass"] += stats["quality_pass"]
            total_stats["quality_fail"] += stats["quality_fail"]

            # Log symbol summary
            logger.info(
                "  %s: %d fetched, %d cached, %d failed, %d/%d quality pass",
                symbol,
                stats["fetched"],
                stats["cached"],
                stats["failed"],
                stats["quality_pass"],
                stats["quality_pass"] + stats["quality_fail"],
            )

    return total_stats


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill historical minute bar data for Sleeve IM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Backfill default symbols for 2020-2024
    python scripts/sleeve_im/backfill_data.py

    # Backfill specific symbols
    python scripts/sleeve_im/backfill_data.py --symbols AAPL,MSFT,GOOGL

    # Backfill 2023 only
    python scripts/sleeve_im/backfill_data.py --start 2023-01-01 --end 2023-12-31

    # Dry run (show what would be fetched)
    python scripts/sleeve_im/backfill_data.py --dry-run

    # Force refresh (ignore cache)
    python scripts/sleeve_im/backfill_data.py --force
        """,
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (default: SPY,QQQ,IWM,...)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=str(DEFAULT_START_DATE),
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=str(DEFAULT_END_DATE),
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh (ignore cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without actually fetching",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = DEFAULT_SYMBOLS

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Check API key
    if not os.getenv("POLYGON_API_KEY"):
        logger.error("POLYGON_API_KEY environment variable not set")
        logger.error("Get an API key at https://polygon.io/")
        sys.exit(1)

    # Run backfill
    logger.info("=" * 60)
    logger.info("Sleeve IM Data Backfill")
    logger.info("=" * 60)
    logger.info("Symbols: %s", ", ".join(symbols))
    logger.info("Date range: %s to %s", start_date, end_date)
    logger.info("Force refresh: %s", args.force)
    logger.info("Dry run: %s", args.dry_run)
    logger.info("=" * 60)

    stats = asyncio.run(
        backfill_all(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            force_refresh=args.force,
            dry_run=args.dry_run,
        )
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("Backfill Complete")
    logger.info("=" * 60)
    logger.info("Symbols processed: %d", stats["symbols"])
    logger.info("Trading days: %d", stats["trading_days"])
    logger.info("Total requests: %d", stats["total_requests"])
    logger.info("Fetched: %d", stats["fetched"])
    logger.info("Cached: %d", stats["cached"])
    logger.info("Failed: %d", stats["failed"])
    logger.info(
        "Quality: %d pass, %d fail",
        stats["quality_pass"],
        stats["quality_fail"],
    )
    logger.info("=" * 60)

    # Exit with error if any failures
    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
