"""
Polygon.io Futures Data Fetcher for Sleeve ORB

Fetches 1-minute OHLCV data for micro futures contracts (MES, MNQ) from
Polygon.io (now Massive.com). Handles continuous series construction with
proper roll logic.

Polygon Futures Ticker Format:
- Base symbol + Month code + Year digit
- Examples: MESH5 (MES March 2025), MNQZ4 (MNQ December 2024)

Month Codes:
- F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun
- N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec

Micro E-mini Futures Roll Schedule:
- Quarterly contracts: H (Mar), M (Jun), U (Sep), Z (Dec)
- Roll 5 days before last trading day
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Timezones
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")  # CME uses Central Time

# API endpoints
POLYGON_BASE_URL = "https://api.polygon.io"

# Futures month codes
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"
}
MONTH_FROM_CODE = {v: k for k, v in MONTH_CODES.items()}

# Quarterly expiration months for micro E-minis
QUARTERLY_MONTHS = [3, 6, 9, 12]  # H, M, U, Z

# Roll schedule
DAYS_BEFORE_EXPIRY_ROLL = 5


@dataclass
class ContractInfo:
    """Futures contract information."""
    base_symbol: str
    month: int
    year: int

    @property
    def ticker(self) -> str:
        """Get Polygon ticker (e.g., MESH5)."""
        month_code = MONTH_CODES[self.month]
        year_digit = self.year % 10
        return f"{self.base_symbol}{month_code}{year_digit}"

    @property
    def expiry_month_start(self) -> date:
        """First day of expiration month."""
        return date(self.year, self.month, 1)

    def __str__(self) -> str:
        return self.ticker


def get_front_contract(base_symbol: str, as_of_date: date) -> ContractInfo:
    """
    Get the front-month contract as of a given date.

    For MES/MNQ, contracts are quarterly (H, M, U, Z).
    Roll 5 days before third Friday of expiration month.
    """
    # Find next expiration month
    year = as_of_date.year
    month = as_of_date.month

    # Find next quarterly month
    for m in QUARTERLY_MONTHS:
        if m >= month:
            exp_month = m
            exp_year = year
            break
    else:
        # Next year
        exp_month = QUARTERLY_MONTHS[0]  # March
        exp_year = year + 1

    # Check if we should roll to next contract
    # Third Friday is the last trading day for E-minis
    # Roll 5 days before
    third_friday = _third_friday(exp_year, exp_month)
    roll_date = third_friday - timedelta(days=DAYS_BEFORE_EXPIRY_ROLL)

    if as_of_date >= roll_date:
        # Roll to next contract
        idx = QUARTERLY_MONTHS.index(exp_month)
        if idx == len(QUARTERLY_MONTHS) - 1:
            exp_month = QUARTERLY_MONTHS[0]
            exp_year += 1
        else:
            exp_month = QUARTERLY_MONTHS[idx + 1]

    return ContractInfo(base_symbol=base_symbol, month=exp_month, year=exp_year)


def _third_friday(year: int, month: int) -> date:
    """Calculate third Friday of a month."""
    first_day = date(year, month, 1)
    # Find first Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    # Third Friday
    return first_friday + timedelta(weeks=2)


def get_contract_schedule(
    base_symbol: str,
    start_date: date,
    end_date: date,
) -> List[Tuple[ContractInfo, date, date]]:
    """
    Generate contract schedule with roll dates.

    Returns list of (contract, start_date, end_date) tuples.
    """
    schedule = []
    current_date = start_date

    while current_date <= end_date:
        contract = get_front_contract(base_symbol, current_date)

        # Find when this contract rolls
        third_friday = _third_friday(contract.year, contract.month)
        roll_date = third_friday - timedelta(days=DAYS_BEFORE_EXPIRY_ROLL)

        # Contract end date is roll date - 1 or end_date
        contract_end = min(roll_date - timedelta(days=1), end_date)

        if contract_end >= current_date:
            schedule.append((contract, current_date, contract_end))

        # Move to next day after roll
        current_date = roll_date

    return schedule


class FuturesFetcher:
    """
    Async client for fetching futures minute bars from Polygon.io.

    Features:
    - Per-contract data fetching with proper roll handling
    - Local caching in parquet format
    - Rate limiting for API tier
    - Continuous series construction with additive back-adjustment
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/orb",
        rate_limit: int = 5,
    ):
        """
        Initialize futures fetcher.

        Args:
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
            cache_dir: Directory for caching data
            rate_limit: Requests per minute
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not set")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit = rate_limit
        self._last_request_time = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "FuturesFetcher":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _rate_limit_wait(self):
        """Simple rate limiting."""
        now = asyncio.get_event_loop().time()
        min_interval = 60.0 / self.rate_limit
        wait_time = min_interval - (now - self._last_request_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited API request."""
        await self._ensure_session()
        await self._rate_limit_wait()

        url = f"{POLYGON_BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        async with self._session.get(url, params=params) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                return await self._api_request(endpoint, params)

            if response.status != 200:
                text = await response.text()
                raise Exception(f"API error {response.status}: {text}")

            return await response.json()

    async def fetch_contract_bars(
        self,
        contract: ContractInfo,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute bars for a specific contract.

        Args:
            contract: Contract to fetch
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ticker = contract.ticker

        # Polygon aggregates endpoint
        # Uses CT (Central Time) for futures
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"

        all_bars = []
        next_url = None

        while True:
            if next_url:
                # Paginated request
                async with self._session.get(next_url) as response:
                    if response.status != 200:
                        break
                    data = await response.json()
            else:
                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                }
                data = await self._api_request(endpoint, params)

            results = data.get("results", [])
            if not results:
                break

            for bar in results:
                # t: timestamp in ms UTC
                ts_ms = bar.get("t", 0)
                ts = datetime.utcfromtimestamp(ts_ms / 1000)

                all_bars.append({
                    "timestamp": ts,
                    "open": bar.get("o", 0),
                    "high": bar.get("h", 0),
                    "low": bar.get("l", 0),
                    "close": bar.get("c", 0),
                    "volume": int(bar.get("v", 0)),
                })

            # Check for pagination
            next_url = data.get("next_url")
            if next_url:
                next_url = f"{next_url}&apiKey={self.api_key}"
            else:
                break

        if not all_bars:
            logger.warning(f"No bars found for {ticker} {start_date} to {end_date}")
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        logger.info(f"Fetched {len(df)} bars for {ticker}")
        return df

    async def build_continuous_series(
        self,
        base_symbol: str,
        start_date: date,
        end_date: date,
        adjustment: str = "additive",
    ) -> pd.DataFrame:
        """
        Build continuous futures series with roll handling.

        Args:
            base_symbol: Base symbol (MES, MNQ)
            start_date: Start date
            end_date: End date
            adjustment: "additive" or "none"

        Returns:
            DataFrame with continuous price series
        """
        schedule = get_contract_schedule(base_symbol, start_date, end_date)

        all_dfs = []
        cumulative_adjustment = 0.0
        prev_close = None

        for contract, c_start, c_end in schedule:
            logger.info(f"Fetching {contract} from {c_start} to {c_end}")

            df = await self.fetch_contract_bars(contract, c_start, c_end)
            if df.empty:
                continue

            # Apply back-adjustment
            if adjustment == "additive" and prev_close is not None:
                # Calculate roll adjustment
                first_close = df.iloc[0]["close"]
                gap = first_close - prev_close
                cumulative_adjustment += gap

            if cumulative_adjustment != 0:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] - cumulative_adjustment

            df["contract"] = contract.ticker
            all_dfs.append(df)

            if not df.empty:
                prev_close = df.iloc[-1]["close"]

        if not all_dfs:
            return pd.DataFrame()

        continuous = pd.concat(all_dfs).sort_index()

        # Remove duplicate timestamps (can happen at roll)
        continuous = continuous[~continuous.index.duplicated(keep="last")]

        return continuous

    async def fetch_and_save(
        self,
        base_symbol: str,
        start_date: date,
        end_date: date,
        output_file: Optional[str] = None,
    ) -> Path:
        """
        Fetch data and save to parquet file.

        Args:
            base_symbol: Base symbol (MES, MNQ)
            start_date: Start date
            end_date: End date
            output_file: Output file path (auto-generated if None)

        Returns:
            Path to saved file
        """
        df = await self.build_continuous_series(base_symbol, start_date, end_date)

        if df.empty:
            raise ValueError(f"No data fetched for {base_symbol}")

        if output_file is None:
            output_file = self.cache_dir / f"{base_symbol}_1min_{start_date}_{end_date}.parquet"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file)

        logger.info(f"Saved {len(df)} bars to {output_file}")
        return output_file


async def fetch_orb_data(
    symbols: List[str] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2025-03-31",
    output_dir: str = "data/orb",
) -> Dict[str, Path]:
    """
    Fetch all data needed for ORB backtest.

    Args:
        symbols: List of base symbols (default: MES, MNQ)
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory

    Returns:
        Dict mapping symbol to output file path
    """
    if symbols is None:
        symbols = ["MES", "MNQ"]

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    results = {}

    async with FuturesFetcher(cache_dir=output_dir) as fetcher:
        for symbol in symbols:
            try:
                path = await fetcher.fetch_and_save(symbol, start, end)
                results[symbol] = path
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

    return results


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch futures data from Polygon.io")
    parser.add_argument(
        "--symbols",
        type=str,
        default="MES,MNQ",
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-03-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/orb",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",")]

    results = asyncio.run(fetch_orb_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
    ))

    print("\n=== Fetch Complete ===")
    for symbol, path in results.items():
        print(f"{symbol}: {path}")
