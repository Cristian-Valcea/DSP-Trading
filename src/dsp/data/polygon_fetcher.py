"""
Polygon.io data fetcher for Sleeve IM.

This module provides access to 1-minute aggregates from Polygon.io,
with caching, rate limiting, and error handling.

Polygon Tiers (for reference):
- Starter ($29/mo): 15-min delay, 5 req/min, aggregates only
- Developer ($79/mo): 15-min delay, unlimited req, trades + aggregates
- Advanced ($199/mo): Real-time, unlimited req, full access

For Sleeve IM, Starter tier is sufficient since:
- Features are computed at 10:30 ET for entry at 11:30 ET (1 hour buffer)
- 15-minute delay is fine for historical/feature data
- IBKR provides real-time quotes for execution
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# US Eastern timezone for all Polygon timestamps
ET = ZoneInfo("America/New_York")

from .minute_bar import (
    DailyMinuteBars,
    MinuteBar,
    RawMinuteBar,
    build_minute_bars,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# API endpoint
POLYGON_BASE_URL = "https://api.polygon.io"

# Rate limiting (Starter tier: 5 req/min)
DEFAULT_RATE_LIMIT = 5  # Requests per minute
DEFAULT_RATE_WINDOW = 60  # Window in seconds

# Timeouts
DEFAULT_TIMEOUT = 30  # seconds

# Cache settings
DEFAULT_CACHE_DIR = "data/sleeve_im/minute_bars"
CACHE_VERSION = "v1"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PolygonConfig:
    """Configuration for Polygon API client."""

    api_key: str
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window: int = DEFAULT_RATE_WINDOW
    timeout: int = DEFAULT_TIMEOUT
    cache_dir: str = DEFAULT_CACHE_DIR
    cache_enabled: bool = True

    @classmethod
    def from_env(cls, api_key_env: str = "POLYGON_API_KEY") -> "PolygonConfig":
        """
        Create config from environment variables.

        Args:
            api_key_env: Name of environment variable containing API key

        Returns:
            PolygonConfig instance

        Raises:
            ValueError: If API key is not set
        """
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Polygon API key not found. Set {api_key_env} environment variable."
            )
        return cls(api_key=api_key)


@dataclass
class RateLimiter:
    """Simple rate limiter for API calls."""

    max_requests: int
    window_seconds: int
    _timestamps: List[float] = None

    def __post_init__(self):
        self._timestamps = []

    async def acquire(self):
        """
        Wait until a request can be made.

        Blocks if rate limit would be exceeded.
        """
        now = asyncio.get_event_loop().time()

        # Remove old timestamps outside window
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        # If at limit, wait until oldest expires
        if len(self._timestamps) >= self.max_requests:
            wait_time = self._timestamps[0] - cutoff
            if wait_time > 0:
                logger.debug("Rate limit: waiting %.2fs", wait_time)
                await asyncio.sleep(wait_time)
            # Recursively try again after waiting
            await self.acquire()
            return

        # Record this request
        self._timestamps.append(now)


# =============================================================================
# Polygon Fetcher
# =============================================================================


class PolygonFetcher:
    """
    Async client for fetching minute bars from Polygon.io.

    Features:
    - 1-minute aggregate bars for feature window [01:30, 10:30] ET
    - Local caching in parquet/JSON format
    - Rate limiting for Starter tier
    - Automatic carry-forward for sparse premarket data
    """

    def __init__(self, config: Optional[PolygonConfig] = None):
        """
        Initialize Polygon fetcher.

        Args:
            config: Polygon configuration (defaults to from_env())
        """
        self.config = config or PolygonConfig.from_env()
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit,
            window_seconds=self.config.rate_window,
        )

        # Ensure cache directory exists
        if self.config.cache_enabled:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> "PolygonFetcher":
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _cache_path(self, symbol: str, trading_date: date) -> Path:
        """Get cache file path for a symbol-date."""
        return (
            Path(self.config.cache_dir)
            / symbol.upper()
            / f"{trading_date.isoformat()}.json"
        )

    def _is_cached(self, symbol: str, trading_date: date) -> bool:
        """Check if data is cached for symbol-date."""
        if not self.config.cache_enabled:
            return False
        return self._cache_path(symbol, trading_date).exists()

    def _load_from_cache(
        self, symbol: str, trading_date: date
    ) -> Optional[List[RawMinuteBar]]:
        """Load cached raw bars."""
        if not self.config.cache_enabled:
            return None

        cache_path = self._cache_path(symbol, trading_date)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            # Parse cached bars
            bars = []
            for item in data.get("bars", []):
                bars.append(
                    RawMinuteBar(
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        open=item["open"],
                        high=item["high"],
                        low=item["low"],
                        close=item["close"],
                        volume=item["volume"],
                        trade_count=item["trade_count"],
                        vwap=item.get("vwap"),
                    )
                )
            return bars

        except Exception as e:
            logger.warning("Failed to load cache for %s %s: %s", symbol, trading_date, e)
            return None

    def _save_to_cache(
        self, symbol: str, trading_date: date, bars: List[RawMinuteBar]
    ):
        """Save raw bars to cache."""
        if not self.config.cache_enabled:
            return

        cache_path = self._cache_path(symbol, trading_date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "symbol": symbol,
                "date": trading_date.isoformat(),
                "cache_version": CACHE_VERSION,
                "bars": [
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "trade_count": bar.trade_count,
                        "vwap": bar.vwap,
                    }
                    for bar in bars
                ],
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

        except Exception as e:
            logger.warning("Failed to save cache for %s %s: %s", symbol, trading_date, e)

    # =========================================================================
    # API Calls
    # =========================================================================

    async def _api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make an API request with rate limiting.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dict

        Raises:
            Exception: If API call fails
        """
        await self._ensure_session()
        await self._rate_limiter.acquire()

        url = f"{POLYGON_BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self.config.api_key

        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning("Rate limited, waiting %ds", retry_after)
                    await asyncio.sleep(retry_after)
                    return await self._api_request(endpoint, params)

                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error {response.status}: {text}")

                return await response.json()

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout for {endpoint}")

    async def _fetch_aggregates(
        self,
        symbol: str,
        trading_date: date,
        start_time: time = time(1, 30),
        end_time: time = time(10, 30),
    ) -> List[RawMinuteBar]:
        """
        Fetch 1-minute aggregates from Polygon API.

        Args:
            symbol: Stock symbol
            trading_date: Trading date
            start_time: Start time in ET (default 01:30 ET)
            end_time: End time in ET (default 10:30 ET)

        Returns:
            List of raw minute bars
        """
        # Build datetime range in ET timezone (Polygon uses ET for US equities)
        start_dt = datetime.combine(trading_date, start_time, tzinfo=ET)
        end_dt = datetime.combine(trading_date, end_time, tzinfo=ET)

        # Convert to UTC timestamps (milliseconds for Polygon API)
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        # API endpoint for aggregates
        # /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
        endpoint = (
            f"/v2/aggs/ticker/{symbol.upper()}/range/1/minute/{start_ts}/{end_ts}"
        )

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,  # Max limit
        }

        response = await self._api_request(endpoint, params)

        # Parse results
        bars = []
        results = response.get("results", [])

        for item in results:
            # Polygon fields:
            # t: timestamp (ms), o: open, h: high, l: low, c: close
            # v: volume, n: number of trades, vw: volume weighted avg price
            ts_ms = item.get("t", 0)
            # Convert UTC timestamp to ET, then make naive for consistency
            ts_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            ts = ts_utc.astimezone(ET).replace(tzinfo=None)

            bars.append(
                RawMinuteBar(
                    timestamp=ts,
                    open=item.get("o", 0),
                    high=item.get("h", 0),
                    low=item.get("l", 0),
                    close=item.get("c", 0),
                    volume=int(item.get("v", 0)),
                    trade_count=int(item.get("n", 0)),
                    vwap=item.get("vw"),
                )
            )

        return bars

    async def _fetch_prior_close(
        self, symbol: str, trading_date: date
    ) -> Optional[float]:
        """
        Fetch prior session close for overnight gap calculation.

        Args:
            symbol: Stock symbol
            trading_date: Trading date

        Returns:
            Prior session close price, or None if not available
        """
        # Get prior trading day (simple approximation - skip weekends)
        prior_date = trading_date - timedelta(days=1)
        if prior_date.weekday() == 6:  # Sunday
            prior_date -= timedelta(days=2)
        elif prior_date.weekday() == 5:  # Saturday
            prior_date -= timedelta(days=1)

        # Fetch daily bar for prior date
        endpoint = f"/v2/aggs/ticker/{symbol.upper()}/range/1/day/{prior_date.isoformat()}/{prior_date.isoformat()}"

        try:
            response = await self._api_request(endpoint)
            results = response.get("results", [])
            if results:
                return results[0].get("c")  # Close price
        except Exception as e:
            logger.warning("Failed to fetch prior close for %s: %s", symbol, e)

        return None

    # =========================================================================
    # Public API
    # =========================================================================

    async def get_minute_bars(
        self,
        symbol: str,
        trading_date: date,
        start_time: time = time(1, 30),
        end_time: time = time(10, 30),
        force_refresh: bool = False,
    ) -> DailyMinuteBars:
        """
        Get complete minute bar series for a symbol-date.

        This is the main entry point for Sleeve IM data retrieval.
        Returns processed bars with carry-forward logic applied.

        Args:
            symbol: Stock symbol
            trading_date: Trading date
            start_time: Feature window start (default 01:30 ET)
            end_time: Feature window end (default 10:30 ET)
            force_refresh: Force fetch from API (ignore cache)

        Returns:
            DailyMinuteBars with complete time grid and metadata
        """
        symbol = symbol.upper()

        # Check cache first
        if not force_refresh and self._is_cached(symbol, trading_date):
            raw_bars = self._load_from_cache(symbol, trading_date)
            if raw_bars is not None:
                logger.debug("Loaded %s %s from cache", symbol, trading_date)
            else:
                raw_bars = await self._fetch_aggregates(
                    symbol, trading_date, start_time, end_time
                )
                self._save_to_cache(symbol, trading_date, raw_bars)
        else:
            raw_bars = await self._fetch_aggregates(
                symbol, trading_date, start_time, end_time
            )
            self._save_to_cache(symbol, trading_date, raw_bars)

        # Get prior close for carry-forward
        prior_close = await self._fetch_prior_close(symbol, trading_date)
        if prior_close is None:
            # Fallback: use first available price
            if raw_bars:
                prior_close = raw_bars[0].open
            else:
                logger.warning(
                    "No prior close and no bars for %s %s", symbol, trading_date
                )
                prior_close = 100.0  # Placeholder

        # Build complete minute bar series with carry-forward
        daily_bars = build_minute_bars(
            symbol=symbol,
            trading_date=trading_date,
            raw_bars=raw_bars,
            prior_session_last_price=prior_close,
            feature_window_start=start_time,
            feature_window_end=end_time,
        )

        logger.info(
            "Fetched %s %s: %d bars (%d real, %.1f%% synthetic)",
            symbol,
            trading_date,
            daily_bars.total_bars,
            daily_bars.real_bars,
            daily_bars.synthetic_pct * 100,
        )

        return daily_bars

    async def get_multiple_symbols(
        self,
        symbols: List[str],
        trading_date: date,
        start_time: time = time(1, 30),
        end_time: time = time(10, 30),
        force_refresh: bool = False,
    ) -> Dict[str, DailyMinuteBars]:
        """
        Get minute bars for multiple symbols in parallel.

        Args:
            symbols: List of stock symbols
            trading_date: Trading date
            start_time: Feature window start (default 01:30 ET)
            end_time: Feature window end (default 10:30 ET)
            force_refresh: Force fetch from API (ignore cache)

        Returns:
            Dict mapping symbol to DailyMinuteBars
        """
        tasks = [
            self.get_minute_bars(s, trading_date, start_time, end_time, force_refresh)
            for s in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        bars_by_symbol = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error("Failed to fetch %s: %s", symbol, result)
            else:
                bars_by_symbol[symbol.upper()] = result

        return bars_by_symbol

    async def is_api_healthy(self) -> bool:
        """
        Check if Polygon API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Simple market status call
            endpoint = "/v1/marketstatus/now"
            await self._api_request(endpoint)
            return True
        except Exception as e:
            logger.error("Polygon API health check failed: %s", e)
            return False


# =============================================================================
# Convenience Functions
# =============================================================================


async def fetch_minute_bars(
    symbol: str,
    trading_date: date,
    api_key: Optional[str] = None,
) -> DailyMinuteBars:
    """
    Convenience function to fetch minute bars for a single symbol-date.

    Args:
        symbol: Stock symbol
        trading_date: Trading date
        api_key: Polygon API key (defaults to POLYGON_API_KEY env var)

    Returns:
        DailyMinuteBars with complete time grid and metadata
    """
    if api_key:
        config = PolygonConfig(api_key=api_key)
    else:
        config = PolygonConfig.from_env()

    async with PolygonFetcher(config) as fetcher:
        return await fetcher.get_minute_bars(symbol, trading_date)
