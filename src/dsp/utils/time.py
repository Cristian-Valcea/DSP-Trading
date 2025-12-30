"""
Market calendar and time utilities for DSP-100K.

Uses pandas_market_calendars for accurate NYSE trading calendar.
"""

from datetime import date, datetime, time, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import pandas_market_calendars as mcal
    HAS_MARKET_CALENDAR = True
except ImportError:
    HAS_MARKET_CALENDAR = False


# US Eastern timezone
NY_TZ = ZoneInfo("America/New_York")


class MarketCalendar:
    """
    NYSE market calendar for trading day validation.

    Uses pandas_market_calendars for accurate holiday handling.
    """

    def __init__(self):
        if not HAS_MARKET_CALENDAR:
            raise ImportError(
                "pandas_market_calendars is required. "
                "Install with: pip install pandas_market_calendars"
            )
        self._calendar = mcal.get_calendar("NYSE")
        self._schedule_cache: Optional[pd.DataFrame] = None
        self._cache_range: Optional[Tuple[date, date]] = None

    def _ensure_schedule(self, start: date, end: date) -> pd.DataFrame:
        """Ensure schedule is cached for the given range."""
        if (
            self._schedule_cache is not None
            and self._cache_range is not None
            and start >= self._cache_range[0]
            and end <= self._cache_range[1]
        ):
            return self._schedule_cache

        # Extend cache range for efficiency
        cache_start = start - timedelta(days=30)
        cache_end = end + timedelta(days=30)

        self._schedule_cache = self._calendar.schedule(
            start_date=cache_start, end_date=cache_end
        )
        self._cache_range = (cache_start, cache_end)

        return self._schedule_cache

    def is_trading_day(self, dt: date) -> bool:
        """Check if a date is a trading day."""
        schedule = self._ensure_schedule(dt, dt)
        sessions = schedule.index.date
        return dt in sessions

    def get_trading_days(self, start: date, end: date) -> List[date]:
        """Get list of trading days in the range [start, end]."""
        schedule = self._ensure_schedule(start, end)
        sessions = schedule.index
        return [
            s.date()
            for s in sessions
            if start <= s.date() <= end
        ]

    def get_previous_trading_day(self, dt: date) -> date:
        """Get the previous trading day before dt."""
        schedule = self._ensure_schedule(dt - timedelta(days=10), dt)
        sessions = [s.date() for s in schedule.index if s.date() < dt]
        if not sessions:
            raise ValueError(f"No trading days found before {dt}")
        return sessions[-1]

    def get_next_trading_day(self, dt: date) -> date:
        """Get the next trading day after dt."""
        schedule = self._ensure_schedule(dt, dt + timedelta(days=10))
        sessions = [s.date() for s in schedule.index if s.date() > dt]
        if not sessions:
            raise ValueError(f"No trading days found after {dt}")
        return sessions[0]

    def get_latest_complete_session(self) -> date:
        """
        Get the latest complete trading session.

        Before market close, returns the previous session.
        After market close, returns today (if trading day).
        """
        now_ny = datetime.now(NY_TZ)
        today = now_ny.date()

        schedule = self._ensure_schedule(today - timedelta(days=10), today)

        # Check if today is a trading day
        today_sessions = schedule[schedule.index.date == today]

        if len(today_sessions) == 0:
            # Today is not a trading day, return most recent
            return self.get_previous_trading_day(today)

        # Check if market has closed (after 4 PM ET)
        market_close = time(16, 0)  # 4:00 PM ET

        if now_ny.time() >= market_close:
            return today
        else:
            return self.get_previous_trading_day(today)

    def get_session_times(self, dt: date) -> Tuple[datetime, datetime]:
        """Get market open and close times for a trading day."""
        schedule = self._ensure_schedule(dt, dt)
        day_schedule = schedule[schedule.index.date == dt]

        if len(day_schedule) == 0:
            raise ValueError(f"{dt} is not a trading day")

        row = day_schedule.iloc[0]
        return row["market_open"].to_pydatetime(), row["market_close"].to_pydatetime()

    def count_trading_days(self, start: date, end: date) -> int:
        """Count the number of trading days in the range [start, end]."""
        return len(self.get_trading_days(start, end))


def get_ny_time() -> datetime:
    """Get current time in New York timezone."""
    return datetime.now(NY_TZ)


def is_market_open() -> bool:
    """
    Check if the market is currently open.

    Returns True if:
    - Today is a trading day
    - Current time is between 9:30 AM and 4:00 PM ET
    """
    now = get_ny_time()
    today = now.date()

    try:
        calendar = MarketCalendar()
        if not calendar.is_trading_day(today):
            return False

        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= now.time() <= market_close
    except ImportError:
        # Fallback without calendar - assume weekdays are trading days
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= now.time() <= market_close


def is_in_execution_window(
    window_start: str = "09:35",
    window_end: str = "10:15",
) -> bool:
    """
    Check if current time is in the execution window.

    Args:
        window_start: Start time in HH:MM format (ET)
        window_end: End time in HH:MM format (ET)

    Returns:
        True if in execution window and market is open
    """
    if not is_market_open():
        return False

    now = get_ny_time()

    start_h, start_m = map(int, window_start.split(":"))
    end_h, end_m = map(int, window_end.split(":"))

    start = time(start_h, start_m)
    end = time(end_h, end_m)

    return start <= now.time() <= end


def parse_time(time_str: str) -> time:
    """Parse a time string in HH:MM format."""
    h, m = map(int, time_str.split(":"))
    return time(h, m)


def minutes_since_open() -> int:
    """Get minutes since market open (9:30 ET)."""
    now = get_ny_time()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    if now < market_open:
        return 0

    delta = now - market_open
    return int(delta.total_seconds() / 60)


def minutes_until_close() -> int:
    """Get minutes until market close (4:00 ET)."""
    now = get_ny_time()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    if now > market_close:
        return 0

    delta = market_close - now
    return int(delta.total_seconds() / 60)
