"""
Minute bar data structures for Sleeve IM.

This module defines the MinuteBar dataclass used for intraday data processing,
with support for synthetic bar tracking (carry-forward logic for sparse premarket data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawMinuteBar:
    """
    Raw vendor 1-minute aggregate bar (as received from Polygon).

    This represents the unprocessed bar data before carry-forward logic is applied.
    """

    timestamp: datetime  # Minute start in ET (e.g., 09:30:00 for 09:30-09:31 bar)
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int  # Polygon "n" field - number of trades in bar
    vwap: Optional[float] = None  # Volume-weighted average price (if available)

    def __post_init__(self):
        """Validate bar data."""
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(f"Invalid price in bar: {self}")
        if self.high < self.low:
            raise ValueError(f"High < Low in bar: {self}")
        if self.volume < 0:
            raise ValueError(f"Negative volume in bar: {self}")


@dataclass
class MinuteBar:
    """
    Processed minute bar with carry-forward metadata.

    This is the primary data structure used by Sleeve IM for feature computation.
    It includes metadata about data quality (synthetic bars, staleness).
    """

    # Core OHLCV data
    timestamp: datetime  # Minute start in ET
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int

    # Data quality metadata (critical for model to understand data quality)
    is_synthetic: bool  # True if no prints in this minute (carry-forward applied)
    seconds_since_last_trade: float  # Staleness indicator
    last_trade_timestamp: Optional[datetime]  # When the last real trade occurred

    # Optional additional fields
    vwap: Optional[float] = None  # Volume-weighted average price

    @property
    def mid(self) -> float:
        """Return mid price (average of high and low)."""
        return (self.high + self.low) / 2.0

    @property
    def typical_price(self) -> float:
        """Return typical price (average of high, low, close)."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def dollar_volume(self) -> float:
        """Return approximate dollar volume."""
        if self.vwap is not None:
            return self.vwap * self.volume
        return self.typical_price * self.volume

    @property
    def bar_range(self) -> float:
        """Return bar range (high - low)."""
        return self.high - self.low

    @property
    def bar_range_pct(self) -> float:
        """Return bar range as percentage of close."""
        if self.close <= 0:
            return 0.0
        return (self.high - self.low) / self.close

    def is_gap_up(self, prior_close: float, threshold_pct: float = 0.01) -> bool:
        """Check if bar opened with a gap up."""
        if prior_close <= 0:
            return False
        return (self.open / prior_close - 1) > threshold_pct

    def is_gap_down(self, prior_close: float, threshold_pct: float = 0.01) -> bool:
        """Check if bar opened with a gap down."""
        if prior_close <= 0:
            return False
        return (self.open / prior_close - 1) < -threshold_pct


@dataclass
class DailyMinuteBars:
    """
    Complete minute bar series for one symbol-day.

    Contains the full time grid from feature_window_start to feature_window_end,
    with all bars (real and synthetic) and data quality summary.
    """

    symbol: str
    date: date
    bars: List[MinuteBar]

    # Data quality summary
    total_bars: int = 0
    real_bars: int = 0
    synthetic_bars: int = 0
    synthetic_pct: float = 0.0

    # Price summary
    prior_close: float = 0.0  # Prior session close (used for overnight gap)
    first_real_price: float = 0.0  # First real price in feature window
    last_price: float = 0.0  # Last price (close of last bar)

    # Volume summary
    total_volume: int = 0
    premarket_volume: int = 0  # Volume before 09:30 ET
    rth_volume: int = 0  # Volume during RTH (09:30-16:00 ET)

    def __post_init__(self):
        """Compute summary statistics."""
        if self.bars:
            self._compute_summary()

    def _compute_summary(self):
        """Compute summary statistics from bars."""
        self.total_bars = len(self.bars)
        self.real_bars = sum(1 for b in self.bars if not b.is_synthetic)
        self.synthetic_bars = self.total_bars - self.real_bars
        self.synthetic_pct = (
            self.synthetic_bars / self.total_bars if self.total_bars > 0 else 0.0
        )

        # Price summary
        if self.bars:
            self.last_price = self.bars[-1].close
            for bar in self.bars:
                if not bar.is_synthetic:
                    self.first_real_price = bar.open
                    break

        # Volume summary
        rth_start = time(9, 30)
        self.total_volume = sum(b.volume for b in self.bars if not b.is_synthetic)
        self.premarket_volume = sum(
            b.volume
            for b in self.bars
            if not b.is_synthetic and b.timestamp.time() < rth_start
        )
        self.rth_volume = sum(
            b.volume
            for b in self.bars
            if not b.is_synthetic and b.timestamp.time() >= rth_start
        )

    def get_bar_at(self, at_time: time) -> Optional[MinuteBar]:
        """Get the bar at or just before a specific time."""
        target_dt = datetime.combine(self.date, at_time)
        for bar in reversed(self.bars):
            if bar.timestamp <= target_dt:
                return bar
        return None

    def get_bars_in_range(
        self, start_time: time, end_time: time
    ) -> List[MinuteBar]:
        """Get all bars within a time range (inclusive)."""
        start_dt = datetime.combine(self.date, start_time)
        end_dt = datetime.combine(self.date, end_time)
        return [b for b in self.bars if start_dt <= b.timestamp <= end_dt]

    def get_return(self, start_time: time, end_time: time) -> Optional[float]:
        """
        Compute return between two times.

        Returns None if either price is not available.
        """
        start_bar = self.get_bar_at(start_time)
        end_bar = self.get_bar_at(end_time)

        if start_bar is None or end_bar is None:
            return None
        if start_bar.close <= 0:
            return None

        return end_bar.close / start_bar.close - 1

    def is_valid_for_trading(
        self,
        max_synthetic_pct: float = 0.70,
        min_premarket_volume: int = 10000,
    ) -> bool:
        """
        Check if this day's data is valid for trading.

        Args:
            max_synthetic_pct: Maximum allowed synthetic bar percentage
            min_premarket_volume: Minimum premarket volume required

        Returns:
            True if data quality is sufficient for trading
        """
        if self.synthetic_pct > max_synthetic_pct:
            logger.warning(
                "%s %s: Synthetic bar %% too high: %.1f%% > %.1f%%",
                self.symbol,
                self.date,
                self.synthetic_pct * 100,
                max_synthetic_pct * 100,
            )
            return False

        if self.premarket_volume < min_premarket_volume:
            logger.warning(
                "%s %s: Premarket volume too low: %d < %d",
                self.symbol,
                self.date,
                self.premarket_volume,
                min_premarket_volume,
            )
            return False

        return True


# =============================================================================
# Time Grid Utilities
# =============================================================================


def build_time_grid(
    trading_date: date,
    start_time: time = time(1, 30),
    end_time: time = time(10, 30),
) -> List[datetime]:
    """
    Build a complete 1-minute time grid for the feature window.

    Args:
        trading_date: The trading date
        start_time: Feature window start (default 01:30 ET)
        end_time: Feature window end (default 10:30 ET)

    Returns:
        List of datetime objects, one per minute
    """
    grid = []
    current = datetime.combine(trading_date, start_time)
    end = datetime.combine(trading_date, end_time)

    while current <= end:
        grid.append(current)
        current += timedelta(minutes=1)

    return grid


# =============================================================================
# Outlier Detection
# =============================================================================

# Thresholds for outlier detection
OUTLIER_PCT_THRESHOLD = 0.15  # 15% jump from carried price
OUTLIER_MIN_VOLUME = 1000  # Minimum shares for "real" minute bar
OUTLIER_MIN_TRADES = 3  # Minimum prints for "real" minute bar
EXTREME_MOVE_THRESHOLD = 0.30  # 30% move is always suspicious


def is_valid_bar(bar: RawMinuteBar, carried_price: float) -> bool:
    """
    Validate a raw bar against outlier detection rules.

    Returns True if bar is valid, False if should be discarded/flagged.

    Args:
        bar: Raw minute bar to validate
        carried_price: Last known valid price (for comparison)

    Returns:
        True if bar passes validation, False otherwise
    """
    if carried_price <= 0:
        return True  # Can't validate without reference

    pct_change = abs(bar.close / carried_price - 1)

    # Large jump on tiny volume/prints = likely bad print(s)
    if pct_change > OUTLIER_PCT_THRESHOLD:
        if bar.volume < OUTLIER_MIN_VOLUME or bar.trade_count < OUTLIER_MIN_TRADES:
            logger.debug(
                "Outlier detected: %.2f%% move on %d shares, %d trades",
                pct_change * 100,
                bar.volume,
                bar.trade_count,
            )
            return False

    # Extreme jump even on larger size (possible but flag)
    if pct_change > EXTREME_MOVE_THRESHOLD:
        logger.warning(
            "Extreme move detected: %.2f%% (threshold: %.2f%%)",
            pct_change * 100,
            EXTREME_MOVE_THRESHOLD * 100,
        )
        return False

    return True


# =============================================================================
# Carry-Forward Logic
# =============================================================================


def build_minute_bars(
    symbol: str,
    trading_date: date,
    raw_bars: List[RawMinuteBar],
    prior_session_last_price: float,
    feature_window_start: time = time(1, 30),
    feature_window_end: time = time(10, 30),
) -> DailyMinuteBars:
    """
    Build complete minute bar series with carry-forward logic.

    This function:
    1. Creates a complete time grid from start to end
    2. Fills in real bars where available
    3. Carries forward last known price for gaps (synthetic bars)
    4. Tracks staleness metadata

    Args:
        symbol: Stock symbol
        trading_date: Trading date
        raw_bars: List of raw bars from vendor (may have gaps)
        prior_session_last_price: Last price from prior session (for overnight gap)
        feature_window_start: Start of feature window (default 01:30 ET)
        feature_window_end: End of feature window (default 10:30 ET)

    Returns:
        DailyMinuteBars with complete time grid and metadata
    """
    # Build time grid
    grid = build_time_grid(trading_date, feature_window_start, feature_window_end)

    # Index raw bars by minute start
    by_ts: Dict[datetime, RawMinuteBar] = {b.timestamp: b for b in raw_bars}

    # Initialize carried price with prior session close
    carried_price = prior_session_last_price
    last_trade_ts: Optional[datetime] = None

    bars: List[MinuteBar] = []

    for minute_start in grid:
        raw = by_ts.get(minute_start)
        has_prints = (
            raw is not None and raw.trade_count > 0 and raw.volume > 0
        )

        if has_prints and is_valid_bar(raw, carried_price):
            # Real bar from vendor aggregates
            bar = MinuteBar(
                timestamp=minute_start,
                open=raw.open,
                high=raw.high,
                low=raw.low,
                close=raw.close,
                volume=raw.volume,
                trade_count=raw.trade_count,
                is_synthetic=False,
                seconds_since_last_trade=0.0,
                last_trade_timestamp=minute_start,  # Coarse (minute bucket)
                vwap=raw.vwap,
            )
            carried_price = bar.close
            last_trade_ts = minute_start
        else:
            # Synthetic bar (carry forward)
            seconds_since = (
                (minute_start - last_trade_ts).total_seconds()
                if last_trade_ts
                else float("inf")
            )
            bar = MinuteBar(
                timestamp=minute_start,
                open=carried_price,
                high=carried_price,
                low=carried_price,
                close=carried_price,
                volume=0,
                trade_count=0,
                is_synthetic=True,
                seconds_since_last_trade=seconds_since,
                last_trade_timestamp=last_trade_ts,
                vwap=None,
            )

        bars.append(bar)

    return DailyMinuteBars(
        symbol=symbol,
        date=trading_date,
        bars=bars,
        prior_close=prior_session_last_price,
    )
