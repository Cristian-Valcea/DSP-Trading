"""
Fetch TLT and IEF bond ETF daily data from Polygon.io.

This script fetches daily aggregates for TLT (20+ Year Treasury Bond ETF) and
IEF (7-10 Year Treasury Bond ETF) from Polygon.io and saves them as parquet files
for use in TSMOM backtesting.

Requirements:
- Polygon.io API key (any tier - daily bars are in all tiers)
- Date range: 2021-01-05 to 2026-01-05
- Output format: Parquet with adjusted close for total return

Usage:
    python scripts/fetch_bond_etf_data.py --symbols TLT IEF
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import aiohttp
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Polygon API configuration
POLYGON_BASE_URL = "https://api.polygon.io"
DEFAULT_TIMEOUT = 30  # seconds


async def fetch_daily_aggregates(
    symbol: str,
    from_date: date,
    to_date: date,
    api_key: str,
    session: aiohttp.ClientSession,
) -> pd.DataFrame:
    """
    Fetch daily aggregates for a symbol from Polygon.io.

    Args:
        symbol: ETF symbol (TLT, IEF)
        from_date: Start date (inclusive)
        to_date: End date (inclusive)
        api_key: Polygon API key
        session: aiohttp ClientSession

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    Raises:
        Exception: If API call fails
    """
    logger.info(f"Fetching {symbol} from {from_date} to {to_date}...")

    # Build API endpoint for daily aggregates
    # Format: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    endpoint = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol.upper()}/range/"
        f"1/day/{from_date.isoformat()}/{to_date.isoformat()}"
    )

    params = {
        "adjusted": "true",  # Use adjusted prices for total return
        "sort": "asc",
        "limit": 50000,  # Max limit (should be enough for 5 years)
        "apiKey": api_key,
    }

    try:
        async with session.get(endpoint, params=params) as response:
            if response.status == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                return await fetch_daily_aggregates(
                    symbol, from_date, to_date, api_key, session
                )

            if response.status != 200:
                text = await response.text()
                raise Exception(f"API error {response.status}: {text}")

            data = await response.json()

    except asyncio.TimeoutError:
        raise Exception(f"Request timeout for {symbol}")

    # Parse results
    results = data.get("results", [])
    if not results:
        logger.warning(f"No data returned for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    bars = []
    for item in results:
        # Polygon fields:
        # t: timestamp (ms), o: open, h: high, l: low, c: close, v: volume
        # n: number of trades, vw: volume weighted avg price
        ts_ms = item.get("t", 0)
        ts = datetime.fromtimestamp(ts_ms / 1000)

        bars.append({
            "timestamp": ts,
            "open": item.get("o", 0),
            "high": item.get("h", 0),
            "low": item.get("l", 0),
            "close": item.get("c", 0),
            "volume": int(item.get("v", 0)),
            "trade_count": int(item.get("n", 0)),
            "vwap": item.get("vw"),
        })

    df = pd.DataFrame(bars)
    logger.info(f"Fetched {len(df)} bars for {symbol}")

    return df


async def main(symbols: list[str], output_dir: Path, api_key: str):
    """
    Main fetch routine.

    Args:
        symbols: List of ETF symbols to fetch (TLT, IEF)
        output_dir: Output directory for parquet files
        api_key: Polygon API key
    """
    # TSMOM date range per spec: 2021-01-05 to 2026-01-05
    from_date = date(2021, 1, 5)
    to_date = date(2026, 1, 5)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                df = await fetch_daily_aggregates(
                    symbol, from_date, to_date, api_key, session
                )

                if df.empty:
                    logger.error(f"No data fetched for {symbol}")
                    continue

                # Save to parquet
                output_file = output_dir / f"{symbol}_1d_{from_date.isoformat()}_{to_date.isoformat()}.parquet"
                df.to_parquet(output_file, index=False, engine="pyarrow")
                logger.info(f"Saved {len(df)} bars to {output_file}")

                # Print summary
                print(f"\n{symbol} Summary:")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"  Total bars: {len(df)}")
                print(f"  Expected trading days (~5 years): ~1260")
                print(f"  Coverage: {len(df)/1260*100:.1f}%")
                print(f"  First close: ${df['close'].iloc[0]:.2f}")
                print(f"  Last close: ${df['close'].iloc[-1]:.2f}")
                print(f"  Output: {output_file}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch bond ETF daily data from Polygon.io"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["TLT", "IEF"],
        help="ETF symbols to fetch (default: TLT IEF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tsmom"),
        help="Output directory for parquet files (default: data/tsmom)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Polygon API key (default: from POLYGON_API_KEY env var)",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: Polygon API key not found.")
        print("Set POLYGON_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Run fetch
    asyncio.run(main(args.symbols, args.output_dir, api_key))

    print("\n✅ Bond ETF data fetch complete!")
    print(f"Output location: {args.output_dir}")
    print("\nNext steps:")
    print("1. Validate data completeness (check bar counts)")
    print("2. Implement TSMOM backtester (src/dsp/backtest/tsmom_futures.py)")
    print("3. Run baseline + stress backtests")
