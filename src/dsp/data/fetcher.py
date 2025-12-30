"""
Data fetcher for DSP-100K.

Fetches historical and real-time market data from IBKR,
with caching and validation.
"""

import asyncio
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..ibkr import IBKRClient
from ..utils.time import MarketCalendar
from .cache import DataCache
from .validation import DataValidator, DataQualityReport, ValidationSeverity

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Unified data fetcher for DSP-100K.

    Features:
    - Fetches historical bars from IBKR
    - Automatic caching with staleness detection
    - Data validation before use
    - Batch fetching for universes
    - Split/dividend adjusted data for signals
    """

    # Bar size constants
    BAR_SIZE_1D = "1 day"
    BAR_SIZE_1H = "1 hour"
    BAR_SIZE_5M = "5 mins"

    # What-to-show constants
    ADJUSTED_LAST = "ADJUSTED_LAST"  # For signals (split/dividend adjusted)
    TRADES = "TRADES"                # For execution (raw OHLCV)

    def __init__(
        self,
        ibkr_client: IBKRClient,
        cache: Optional[DataCache] = None,
        validator: Optional[DataValidator] = None,
        calendar: Optional[MarketCalendar] = None,
    ):
        """
        Initialize the data fetcher.

        Args:
            ibkr_client: IBKR client for API calls
            cache: Data cache (created if not provided)
            validator: Data validator (created if not provided)
            calendar: Market calendar (created if not provided)
        """
        self.ibkr = ibkr_client
        self.cache = cache or DataCache()
        self.validator = validator or DataValidator()
        self.calendar = calendar or MarketCalendar()

    async def get_daily_bars(
        self,
        symbol: str,
        lookback_days: int = 365,
        end_date: Optional[date] = None,
        adjusted: bool = True,
        use_cache: bool = True,
        validate: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get daily bar data for a symbol.

        Args:
            symbol: Symbol to fetch
            lookback_days: Number of calendar days of history
            end_date: End date (default: latest complete session)
            adjusted: Use split/dividend adjusted prices
            use_cache: Try to use cached data
            validate: Validate data before returning

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Index is datetime
        """
        if end_date is None:
            end_date = self.calendar.get_latest_complete_session()

        start_date = end_date - timedelta(days=lookback_days)

        # Try cache first
        if use_cache:
            cached = self.cache.get_daily_bars(symbol, start_date, end_date)
            if cached is not None:
                logger.debug(f"Cache hit for {symbol} daily bars")
                return cached

        # Fetch from IBKR
        what_to_show = self.ADJUSTED_LAST if adjusted else self.TRADES
        duration = f"{lookback_days} D"

        try:
            df = await self.ibkr.get_historical_data(
                symbol=symbol,
                duration=duration,
                bar_size=self.BAR_SIZE_1D,
                what_to_show=what_to_show,
            )

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Standardize column names
            df = self._standardize_columns(df)

            # Validate if requested
            if validate:
                issues = self.validator.validate_daily_bars(symbol, df, end_date)
                critical = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
                if critical:
                    logger.error(f"Critical validation issues for {symbol}: {critical}")
                    return None

                errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
                if errors:
                    logger.warning(f"Validation errors for {symbol}: {len(errors)} issues")

            # Cache the result
            if use_cache:
                self.cache.put_daily_bars(symbol, start_date, end_date, df, "IBKR")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch daily bars for {symbol}: {e}")
            return None

    async def get_universe_bars(
        self,
        symbols: List[str],
        lookback_days: int = 365,
        end_date: Optional[date] = None,
        adjusted: bool = True,
        use_cache: bool = True,
        validate: bool = True,
        max_concurrent: int = 5,
    ) -> Tuple[Dict[str, pd.DataFrame], DataQualityReport]:
        """
        Get daily bars for a universe of symbols.

        Args:
            symbols: List of symbols to fetch
            lookback_days: Number of calendar days of history
            end_date: End date
            adjusted: Use adjusted prices
            use_cache: Try to use cached data
            validate: Validate data
            max_concurrent: Maximum concurrent fetches

        Returns:
            Tuple of (data dict, quality report)
        """
        if end_date is None:
            end_date = self.calendar.get_latest_complete_session()

        data: Dict[str, pd.DataFrame] = {}

        # Fetch in batches to respect rate limits
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
            async with semaphore:
                df = await self.get_daily_bars(
                    symbol=symbol,
                    lookback_days=lookback_days,
                    end_date=end_date,
                    adjusted=adjusted,
                    use_cache=use_cache,
                    validate=False,  # We'll validate the whole universe
                )
                return symbol, df

        # Fetch all symbols concurrently
        tasks = [fetch_one(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Fetch exception: {result}")
                continue

            symbol, df = result
            if df is not None:
                data[symbol] = df
            else:
                logger.warning(f"No data for {symbol}")

        # Validate universe
        report = self.validator.validate_universe(
            data=data,
            as_of_date=end_date,
            min_history_days=lookback_days // 2,  # At least half the requested history
        )

        if not report.passed:
            logger.warning(f"Universe validation failed:\n{report.summary()}")

        return data, report

    async def get_returns(
        self,
        symbol: str,
        lookback_days: int = 365,
        end_date: Optional[date] = None,
    ) -> Optional[pd.Series]:
        """
        Get daily returns for a symbol.

        Args:
            symbol: Symbol to fetch
            lookback_days: Number of calendar days
            end_date: End date

        Returns:
            Series of daily returns (index=dates)
        """
        df = await self.get_daily_bars(
            symbol=symbol,
            lookback_days=lookback_days + 5,  # Extra days for return calculation
            end_date=end_date,
            adjusted=True,
        )

        if df is None:
            return None

        returns = df["close"].pct_change().dropna()

        # Trim to requested period
        if end_date:
            start_date = end_date - timedelta(days=lookback_days)
            returns = returns[returns.index >= pd.Timestamp(start_date)]

        return returns

    async def get_returns_matrix(
        self,
        symbols: List[str],
        lookback_days: int = 365,
        end_date: Optional[date] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get returns matrix for a universe.

        Args:
            symbols: List of symbols
            lookback_days: Number of calendar days
            end_date: End date

        Returns:
            DataFrame with returns (index=dates, columns=symbols)
        """
        data, report = await self.get_universe_bars(
            symbols=symbols,
            lookback_days=lookback_days + 5,
            end_date=end_date,
        )

        if not data:
            return None

        # Build returns matrix
        returns_dict = {}
        for symbol, df in data.items():
            returns_dict[symbol] = df["close"].pct_change()

        returns = pd.DataFrame(returns_dict).dropna(how="all")

        # Validate returns matrix
        issues = self.validator.validate_returns_matrix(returns)
        if issues:
            logger.warning(f"Returns matrix validation: {len(issues)} issues")

        return returns

    async def get_volatility(
        self,
        symbol: str,
        lookback_days: int = 21,
        end_date: Optional[date] = None,
        annualize: bool = True,
    ) -> Optional[float]:
        """
        Calculate realized volatility for a symbol.

        Args:
            symbol: Symbol
            lookback_days: Lookback period in calendar days
            end_date: End date
            annualize: Annualize the volatility

        Returns:
            Volatility as decimal (0.20 = 20%)
        """
        returns = await self.get_returns(
            symbol=symbol,
            lookback_days=lookback_days + 10,
            end_date=end_date,
        )

        if returns is None or len(returns) < 5:
            return None

        # Use last lookback_days trading days
        returns = returns.tail(lookback_days)

        vol = returns.std()
        if annualize:
            vol *= (252 ** 0.5)  # Annualize

        return float(vol)

    async def get_trailing_return(
        self,
        symbol: str,
        months: int,
        end_date: Optional[date] = None,
        skip_month: bool = False,
    ) -> Optional[float]:
        """
        Calculate trailing return over a period.

        Args:
            symbol: Symbol
            months: Number of months
            end_date: End date
            skip_month: If True, skip the most recent month (momentum convention)

        Returns:
            Total return as decimal (0.10 = 10%)
        """
        lookback_days = months * 30 + 35  # Extra buffer

        if end_date is None:
            end_date = self.calendar.get_latest_complete_session()

        df = await self.get_daily_bars(
            symbol=symbol,
            lookback_days=lookback_days,
            end_date=end_date,
        )

        if df is None or df.empty:
            return None

        # Calculate period bounds
        if skip_month:
            # Skip most recent 21 trading days
            df_valid = df.iloc[:-21] if len(df) > 21 else df
        else:
            df_valid = df

        # Get approximately the right number of trading days
        trading_days = int(months * 21)  # ~21 trading days per month
        if len(df_valid) < trading_days:
            logger.warning(f"Insufficient history for {months}M return: {len(df_valid)} days")
            return None

        df_period = df_valid.tail(trading_days)

        # Calculate total return
        start_price = df_period["close"].iloc[0]
        end_price = df_period["close"].iloc[-1]

        if start_price <= 0:
            return None

        return float((end_price / start_price) - 1)

    async def get_trend_signals(
        self,
        symbol: str,
        end_date: Optional[date] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate trend signals for a symbol.

        Uses the Sleeve B convention:
        - 1M return (25% weight)
        - 3M return (50% weight)
        - 12M return with 1M skip (25% weight)

        Args:
            symbol: Symbol
            end_date: End date

        Returns:
            Dict with ret_1m, ret_3m, ret_12m_skip, composite_signal
        """
        # Fetch all required returns
        ret_1m = await self.get_trailing_return(symbol, months=1, end_date=end_date)
        ret_3m = await self.get_trailing_return(symbol, months=3, end_date=end_date)
        ret_12m = await self.get_trailing_return(
            symbol, months=12, end_date=end_date, skip_month=True
        )

        if ret_1m is None or ret_3m is None or ret_12m is None:
            return None

        # Calculate composite signal (weighted average)
        composite = 0.25 * ret_1m + 0.50 * ret_3m + 0.25 * ret_12m

        return {
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_12m_skip": ret_12m,
            "composite_signal": composite,
        }

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame column names.

        IBKR returns columns like 'Open', 'High', etc.
        We want lowercase: 'open', 'high', etc.
        """
        # Map common variations
        column_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Date": "date",
            "Adj Close": "adj_close",
            "Average": "vwap",
            "BarCount": "bar_count",
        }

        df = df.rename(columns=column_map)

        # Also lowercase any remaining columns
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

        return df

    async def prefetch_universe(
        self,
        symbols: List[str],
        lookback_days: int = 400,
    ) -> int:
        """
        Pre-fetch and cache data for a universe.

        Useful for warming the cache before market open.

        Args:
            symbols: List of symbols to prefetch
            lookback_days: History to fetch

        Returns:
            Number of symbols successfully cached
        """
        data, report = await self.get_universe_bars(
            symbols=symbols,
            lookback_days=lookback_days,
            use_cache=False,  # Force fresh fetch
        )

        logger.info(f"Prefetched {len(data)}/{len(symbols)} symbols")
        logger.info(report.summary())

        return len(data)


class UniverseManager:
    """
    Manages ETF universes for Sleeve B.

    Tracks universe membership, handles adds/removes,
    and enforces non-equity ETF constraints.
    """

    # Equity ETFs that are NOT allowed in Sleeve B
    # NOTE: REITs (VNQ, VNQI) are classified as equity instruments
    EQUITY_ETFS = {
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        "VOO", "IVV", "VT", "VXUS", "VEA",
        "VWO", "EFA", "EEM", "IEMG", "VGK",
        "VPL", "IJH", "IJR", "MDY", "IWB",
        "VNQ", "VNQI",  # REITs are equity instruments
    }

    # Default Sleeve B universe (non-equity ETFs only)
    DEFAULT_UNIVERSE = [
        # Bonds
        "TLT",   # 20+ Year Treasury
        "IEF",   # 7-10 Year Treasury
        "LQD",   # Investment Grade Corporate
        "HYG",   # High Yield Corporate
        "EMB",   # Emerging Markets Bonds
        "TIP",   # TIPS

        # Commodities
        "GLD",   # Gold
        "SLV",   # Silver
        "USO",   # Oil
        "UNG",   # Natural Gas
        "DBA",   # Agriculture
        "DBB",   # Base Metals

        # NOTE: VNQ/VNQI EXCLUDED - REITs are equity instruments

        # Currencies
        "UUP",   # US Dollar
        "FXE",   # Euro
        "FXY",   # Yen
        "FXB",   # British Pound

        # Volatility
        "VIXY",  # VIX Short-Term
    ]

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize universe manager.

        Args:
            symbols: Custom universe (default: DEFAULT_UNIVERSE)
        """
        self._symbols = set(symbols) if symbols else set(self.DEFAULT_UNIVERSE)
        self._validate_universe()

    def _validate_universe(self) -> None:
        """Validate that universe contains no equity ETFs."""
        equity_in_universe = self._symbols & self.EQUITY_ETFS
        if equity_in_universe:
            raise ValueError(
                f"Equity ETFs not allowed in Sleeve B universe: {equity_in_universe}"
            )

    @property
    def symbols(self) -> List[str]:
        """Get current universe as sorted list."""
        return sorted(self._symbols)

    def add(self, symbol: str) -> bool:
        """
        Add a symbol to the universe.

        Args:
            symbol: Symbol to add

        Returns:
            True if added, False if blocked (equity ETF)
        """
        if symbol in self.EQUITY_ETFS:
            logger.warning(f"Cannot add equity ETF {symbol} to Sleeve B universe")
            return False

        self._symbols.add(symbol)
        return True

    def remove(self, symbol: str) -> bool:
        """
        Remove a symbol from the universe.

        Args:
            symbol: Symbol to remove

        Returns:
            True if removed, False if not in universe
        """
        if symbol not in self._symbols:
            return False

        self._symbols.remove(symbol)
        return True

    def is_valid_etf(self, symbol: str) -> bool:
        """Check if an ETF is valid for Sleeve B (not equity)."""
        return symbol not in self.EQUITY_ETFS

    def filter_equity_etfs(self, symbols: List[str]) -> List[str]:
        """
        Filter out equity ETFs from a list.

        Args:
            symbols: List to filter

        Returns:
            List with equity ETFs removed
        """
        filtered = [s for s in symbols if s not in self.EQUITY_ETFS]
        removed = set(symbols) - set(filtered)
        if removed:
            logger.info(f"Filtered out equity ETFs: {removed}")
        return filtered
