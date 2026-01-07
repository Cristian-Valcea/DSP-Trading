"""
ORB Futures Backtester - Opening Range Breakout Strategy Kill-Test

Implements the strategy defined in SLEEVE_ORB_MINIMAL_SPEC.md (v1.6):
- 30-minute Opening Range (09:30-09:59 ET)
- OCO entry at OR_high+buffer / OR_low-buffer
- Stop = max(1.0*OR_Width, 0.20*ATR_d)
- Target = 2.0 * stop
- Flatten at 15:55 ET

Kill-Test Purpose
-----------------
Validate that ORB has positive edge after transaction costs before
promoting to production. Uses pessimistic fill assumptions:
- Same-bar stop/target: assume STOP hit first
- Gap-through: fill at bar.open (worse than stop price)
- 1-tick slippage per side (baseline), 2-tick stress test
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from math import floor, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS (FROZEN per spec v1.6)
# ============================================================================

ET = pytz.timezone("America/New_York")

# Contract specifications
CONTRACTS = {
    "MES": {"point_value": 5.0, "tick_size": 0.25, "tick_value": 1.25, "commission_rt": 1.24},
    "MNQ": {"point_value": 2.0, "tick_size": 0.25, "tick_value": 0.50, "commission_rt": 1.24},
}

# Opening Range
OR_START = time(9, 30)
OR_END = time(10, 0)  # Exclusive - first entry bar is 10:00

# Entry window
ENTRY_START = time(10, 0)
ENTRY_END = time(14, 0)

# EOD flatten
EOD_FLATTEN = time(15, 55)
FOMC_FLATTEN = time(13, 55)

# Risk parameters
RISK_PER_TRADE_BPS = 20  # 0.20% of NAV
STOP_MULTIPLIER = 1.0
STOP_ATR_FLOOR = 0.20
TARGET_MULTIPLIER = 2.0
BUFFER_TICKS = 2

# Filter parameters
COMPRESSION_THRESHOLD = 0.20
EXHAUSTION_THRESHOLD = 2.0

# Indicator lookbacks
ATR_PERIOD = 14
OR_WIDTH_AVG_PERIOD = 20

# Walk-forward folds (per spec)
WALK_FORWARD_FOLDS = [
    {"fold_id": 1, "train_start": "2022-01-03", "train_end": "2022-06-30",
     "test_start": "2022-07-01", "test_end": "2022-09-30"},
    {"fold_id": 2, "train_start": "2022-07-01", "train_end": "2022-12-30",
     "test_start": "2023-01-03", "test_end": "2023-03-31"},
    {"fold_id": 3, "train_start": "2023-01-03", "train_end": "2023-06-30",
     "test_start": "2023-07-03", "test_end": "2023-09-29"},
    {"fold_id": 4, "train_start": "2023-07-03", "train_end": "2023-12-29",
     "test_start": "2024-01-02", "test_end": "2024-03-28"},
    {"fold_id": 5, "train_start": "2024-01-02", "train_end": "2024-06-28",
     "test_start": "2024-07-01", "test_end": "2024-09-30"},
    {"fold_id": 6, "train_start": "2024-07-01", "train_end": "2024-12-31",
     "test_start": "2025-01-02", "test_end": "2025-03-31"},
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ContractSpec:
    symbol: str
    point_value: float
    tick_size: float
    tick_value: float
    commission_rt: float


@dataclass
class ORBTrade:
    """Single ORB trade record."""
    date: date
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: time
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: time
    exit_price: float
    exit_reason: str  # "STOP", "TARGET", "EOD", "FOMC"
    contracts: int
    gross_pnl: float
    slippage_cost: float
    commission: float
    net_pnl: float
    or_high: float
    or_low: float
    or_width: float
    atr_d: float
    stop_dist: float


@dataclass
class DailyState:
    """Daily state for ORB strategy."""
    date: date
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_width: Optional[float] = None
    atr_d: Optional[float] = None
    or_width_avg: Optional[float] = None
    skip_reason: Optional[str] = None
    is_fomc_day: bool = False


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""
    fold_id: int
    train_period: str
    test_period: str
    n_trades: int
    n_wins: int
    n_losses: int
    gross_pnl: float
    net_pnl: float
    total_commission: float
    total_slippage: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_per_contract: float
    max_dd: float
    trades: List[ORBTrade] = field(default_factory=list)
    daily_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    passes_kill_tests: bool = False


@dataclass
class BacktestResult:
    """Complete backtest result."""
    folds: List[FoldResult]
    summary: Dict[str, Any]
    all_trades: List[ORBTrade]
    config: Dict[str, Any]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_date(s: str) -> date:
    """Parse YYYY-MM-DD string to date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_skip_dates(path: Path) -> Dict[date, Tuple[str, str]]:
    """Load skip dates from CSV."""
    skip_dates: Dict[date, Tuple[str, str]] = {}
    if not path.exists():
        logger.warning(f"Skip dates file not found: {path}")
        return skip_dates

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    dt = parse_date(parts[0])
                    event_type = parts[1]
                    action = parts[2]
                    skip_dates[dt] = (event_type, action)
                except ValueError:
                    continue
    return skip_dates


def get_contract_spec(symbol: str) -> ContractSpec:
    """Get contract specification."""
    if symbol not in CONTRACTS:
        raise ValueError(f"Unknown contract: {symbol}")
    c = CONTRACTS[symbol]
    return ContractSpec(
        symbol=symbol,
        point_value=c["point_value"],
        tick_size=c["tick_size"],
        tick_value=c["tick_value"],
        commission_rt=c["commission_rt"],
    )


def true_range_rth(high: float, low: float, prev_close: float) -> float:
    """Calculate True Range using RTH data only."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def max_drawdown(equity: pd.Series) -> float:
    """Calculate max drawdown from equity series."""
    if equity.empty or len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * sqrt(252))


# ============================================================================
# ORB BACKTESTER
# ============================================================================

class ORBBacktester:
    """
    Opening Range Breakout backtester for MES/MNQ micro futures.

    Key features:
    - 30-min OR construction (09:30-09:59)
    - OCO entry at OR±buffer
    - Dynamic stop sizing (max of OR_width, 0.20*ATR)
    - 2R profit target
    - Pessimistic same-bar resolution (stop first)
    - Gap-through fills at bar.open
    """

    def __init__(
        self,
        data_dir: Path,
        skip_dates_path: Optional[Path] = None,
        slippage_ticks: int = 1,
        use_profit_target: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.slippage_ticks = slippage_ticks
        self.use_profit_target = use_profit_target

        # Load skip dates
        skip_path = skip_dates_path or self.data_dir / "skip_dates.csv"
        self.skip_dates = load_skip_dates(skip_path)

        # Data cache
        self._minute_data: Dict[str, pd.DataFrame] = {}
        self._daily_data: Dict[str, pd.DataFrame] = {}

    def load_minute_data(self, symbol: str) -> pd.DataFrame:
        """Load 1-minute OHLC data for symbol."""
        if symbol in self._minute_data:
            return self._minute_data[symbol]

        # Try various file patterns
        patterns = [
            f"{symbol}_1min_*.parquet",
            f"{symbol}_1m.parquet",
            f"{symbol}.parquet",
        ]

        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                df = pd.read_parquet(files[0])
                break
        else:
            raise FileNotFoundError(f"No data file found for {symbol} in {self.data_dir}")

        # Ensure datetime index in ET
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                raise ValueError(f"Cannot determine datetime column for {symbol}")

        # Convert to ET if needed
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        elif str(df.index.tz) != str(ET):
            df.index = df.index.tz_convert(ET)

        # Ensure OHLCV columns exist
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                # Try uppercase
                if col.upper() in df.columns:
                    df = df.rename(columns={col.upper(): col})
                else:
                    raise ValueError(f"Missing column {col} for {symbol}")

        # Sort by time
        df = df.sort_index()

        self._minute_data[symbol] = df
        return df

    def build_daily_data(self, symbol: str) -> pd.DataFrame:
        """Build daily OHLC from minute data (RTH only)."""
        if symbol in self._daily_data:
            return self._daily_data[symbol]

        minute_df = self.load_minute_data(symbol)

        # Filter to RTH only (09:30-16:00)
        rth_mask = (
            (minute_df.index.time >= time(9, 30)) &
            (minute_df.index.time < time(16, 0))
        )
        rth_df = minute_df[rth_mask].copy()

        # Aggregate to daily
        daily = rth_df.resample("D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna()

        self._daily_data[symbol] = daily
        return daily

    def compute_atr(self, symbol: str, as_of_date: date) -> Optional[float]:
        """Compute ATR_d (14-period, RTH-only) as of prior day's close."""
        daily = self.build_daily_data(symbol)

        # Get data up to but not including as_of_date
        prior = daily[daily.index.date < as_of_date]
        if len(prior) < ATR_PERIOD + 1:
            return None

        # Calculate True Range
        tr_values = []
        for i in range(1, len(prior)):
            h = prior.iloc[i]["high"]
            l = prior.iloc[i]["low"]
            prev_c = prior.iloc[i-1]["close"]
            tr_values.append(true_range_rth(h, l, prev_c))

        if len(tr_values) < ATR_PERIOD:
            return None

        return float(pd.Series(tr_values).tail(ATR_PERIOD).mean())

    def compute_or_width_avg(
        self, symbol: str, as_of_date: date, or_widths: Dict[date, float]
    ) -> Optional[float]:
        """Compute 20-day average OR width as of prior day."""
        # Get OR widths for days before as_of_date
        prior_widths = [
            w for d, w in or_widths.items()
            if d < as_of_date and w is not None
        ]
        if len(prior_widths) < OR_WIDTH_AVG_PERIOD:
            return None

        return float(pd.Series(prior_widths).tail(OR_WIDTH_AVG_PERIOD).mean())

    def get_or_bars(self, symbol: str, dt: date) -> pd.DataFrame:
        """Get the 30 one-minute bars for Opening Range (09:30-09:59)."""
        minute_df = self.load_minute_data(symbol)

        # Filter to date and OR time window
        day_mask = minute_df.index.date == dt
        or_mask = (
            (minute_df.index.time >= OR_START) &
            (minute_df.index.time < OR_END)
        )

        return minute_df[day_mask & or_mask]

    def get_trading_bars(self, symbol: str, dt: date) -> pd.DataFrame:
        """Get all trading bars for the day (after OR, for simulation)."""
        minute_df = self.load_minute_data(symbol)

        day_mask = minute_df.index.date == dt
        trading_mask = (
            (minute_df.index.time >= ENTRY_START) &
            (minute_df.index.time < time(16, 0))
        )

        return minute_df[day_mask & trading_mask]

    def should_skip_day(self, dt: date) -> Tuple[bool, Optional[str]]:
        """Check if day should be skipped entirely."""
        if dt in self.skip_dates:
            event_type, action = self.skip_dates[dt]
            if action == "SKIP_FULL_DAY":
                return True, f"{event_type}_SKIP"
        return False, None

    def is_fomc_day(self, dt: date) -> bool:
        """Check if this is an FOMC day (early flatten)."""
        if dt in self.skip_dates:
            event_type, action = self.skip_dates[dt]
            if event_type == "FOMC" and action == "EARLY_FLATTEN":
                return True
        return False

    def apply_filters(
        self,
        or_width: float,
        or_width_avg: Optional[float],
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply regime filters.

        Returns (should_skip, reason).
        """
        if or_width_avg is None:
            return True, "WARM_UP_INCOMPLETE"

        # Compression filter
        if or_width < COMPRESSION_THRESHOLD * or_width_avg:
            return True, "COMPRESSION"

        # Exhaustion filter
        if or_width > EXHAUSTION_THRESHOLD * or_width_avg:
            return True, "EXHAUSTION"

        return False, None

    def compute_stop_distance(self, or_width: float, atr_d: float) -> float:
        """Compute stop distance as max(OR_width, 0.20*ATR)."""
        return max(STOP_MULTIPLIER * or_width, STOP_ATR_FLOOR * atr_d)

    def compute_position_size(
        self,
        nav: float,
        stop_dist: float,
        contract: ContractSpec,
    ) -> int:
        """
        Compute position size based on risk budget.

        Returns 0 if cannot size properly (do not force min(1)).
        """
        risk_dollars = nav * (RISK_PER_TRADE_BPS / 10000.0)
        contract_risk = stop_dist * contract.point_value

        if contract_risk <= 0:
            return 0

        qty = floor(risk_dollars / contract_risk)
        return max(0, qty)  # Don't force 1 - return 0 if can't size

    def simulate_entry(
        self,
        bars: pd.DataFrame,
        or_high: float,
        or_low: float,
        contract: ContractSpec,
        is_fomc: bool,
    ) -> Optional[Tuple[str, time, float, float]]:
        """
        Simulate entry using OCO stop-market orders.

        Returns (direction, entry_time, trigger_price, fill_price) or None.
        """
        buffer = BUFFER_TICKS * contract.tick_size
        long_trigger = or_high + buffer
        short_trigger = or_low - buffer

        # Determine flatten cutoff
        flatten_time = FOMC_FLATTEN if is_fomc else ENTRY_END

        for ts, bar in bars.iterrows():
            bar_time = ts.time()

            # Past entry window?
            if bar_time >= flatten_time:
                break

            # Check for OCO trigger
            # Long: bar.high >= long_trigger
            # Short: bar.low <= short_trigger

            long_triggered = bar["high"] >= long_trigger
            short_triggered = bar["low"] <= short_trigger

            if long_triggered and short_triggered:
                # Both triggered same bar - use open to determine which first
                # If open >= long_trigger, assume long triggered at open
                # If open <= short_trigger, assume short triggered at open
                # Otherwise, can't know - assume worst case (short for conservative)
                if bar["open"] >= long_trigger:
                    direction = "LONG"
                    trigger_price = long_trigger
                elif bar["open"] <= short_trigger:
                    direction = "SHORT"
                    trigger_price = short_trigger
                else:
                    # Ambiguous - skip this trade (or could assume worst)
                    continue
            elif long_triggered:
                direction = "LONG"
                trigger_price = long_trigger
            elif short_triggered:
                direction = "SHORT"
                trigger_price = short_trigger
            else:
                continue

            # Apply slippage to get fill price
            slip = self.slippage_ticks * contract.tick_size
            if direction == "LONG":
                fill_price = trigger_price + slip  # Pay more
            else:
                fill_price = trigger_price - slip  # Receive less

            return direction, bar_time, trigger_price, fill_price

        return None

    def simulate_exit(
        self,
        bars: pd.DataFrame,
        entry_time: time,
        entry_price: float,
        stop_dist: float,
        direction: str,
        contract: ContractSpec,
        is_fomc: bool,
    ) -> Tuple[time, float, str]:
        """
        Simulate exit (stop, target, or EOD flatten).

        Uses pessimistic resolution: if stop AND target hit same bar, assume STOP.
        Gap-through stops fill at bar.open.

        Returns (exit_time, exit_price, exit_reason).
        """
        if direction == "LONG":
            stop_price = entry_price - stop_dist
            target_price = entry_price + TARGET_MULTIPLIER * stop_dist
        else:
            stop_price = entry_price + stop_dist
            target_price = entry_price - TARGET_MULTIPLIER * stop_dist

        # Determine flatten time
        flatten_time = FOMC_FLATTEN if is_fomc else EOD_FLATTEN

        slip = self.slippage_ticks * contract.tick_size

        for ts, bar in bars.iterrows():
            bar_time = ts.time()

            # Skip bars before entry
            if bar_time <= entry_time:
                continue

            # EOD flatten check
            if bar_time >= flatten_time:
                exit_price = bar["close"]
                # Apply slippage on flatten
                if direction == "LONG":
                    exit_price -= slip
                else:
                    exit_price += slip
                reason = "FOMC" if is_fomc else "EOD"
                return bar_time, exit_price, reason

            if direction == "LONG":
                stop_hit = bar["low"] <= stop_price
                target_hit = self.use_profit_target and (bar["high"] >= target_price)

                if stop_hit and target_hit:
                    # Pessimistic: assume stop hit first
                    if bar["open"] <= stop_price:
                        # Gapped through stop
                        exit_price = bar["open"] - slip
                    else:
                        exit_price = stop_price - slip
                    return bar_time, exit_price, "STOP"

                if stop_hit:
                    if bar["open"] <= stop_price:
                        exit_price = bar["open"] - slip
                    else:
                        exit_price = stop_price - slip
                    return bar_time, exit_price, "STOP"

                if target_hit:
                    # No price improvement on target
                    exit_price = target_price - slip
                    return bar_time, exit_price, "TARGET"

            else:  # SHORT
                stop_hit = bar["high"] >= stop_price
                target_hit = self.use_profit_target and (bar["low"] <= target_price)

                if stop_hit and target_hit:
                    # Pessimistic: assume stop hit first
                    if bar["open"] >= stop_price:
                        exit_price = bar["open"] + slip
                    else:
                        exit_price = stop_price + slip
                    return bar_time, exit_price, "STOP"

                if stop_hit:
                    if bar["open"] >= stop_price:
                        exit_price = bar["open"] + slip
                    else:
                        exit_price = stop_price + slip
                    return bar_time, exit_price, "STOP"

                if target_hit:
                    exit_price = target_price + slip
                    return bar_time, exit_price, "TARGET"

        # Should not reach here if bars include EOD
        last_bar = bars.iloc[-1]
        exit_price = last_bar["close"]
        if direction == "LONG":
            exit_price -= slip
        else:
            exit_price += slip
        return bars.index[-1].time(), exit_price, "EOD"

    def run_single_day(
        self,
        dt: date,
        symbol: str,
        nav: float,
        or_width_history: Dict[date, float],
    ) -> Tuple[Optional[ORBTrade], DailyState]:
        """
        Run ORB strategy for a single day/symbol.

        Returns (trade or None, daily_state).
        """
        contract = get_contract_spec(symbol)
        state = DailyState(date=dt)

        # Check skip dates
        should_skip, skip_reason = self.should_skip_day(dt)
        if should_skip:
            state.skip_reason = skip_reason
            return None, state

        state.is_fomc_day = self.is_fomc_day(dt)

        # Get OR bars
        or_bars = self.get_or_bars(symbol, dt)
        if len(or_bars) < 20:  # Need reasonable OR
            state.skip_reason = "INCOMPLETE_OR"
            return None, state

        # Compute OR
        state.or_high = float(or_bars["high"].max())
        state.or_low = float(or_bars["low"].min())
        state.or_width = state.or_high - state.or_low

        # Store for history
        or_width_history[dt] = state.or_width

        # Compute ATR
        state.atr_d = self.compute_atr(symbol, dt)
        if state.atr_d is None:
            state.skip_reason = "ATR_WARMUP"
            return None, state

        # Compute OR width average
        state.or_width_avg = self.compute_or_width_avg(symbol, dt, or_width_history)
        if state.or_width_avg is None:
            state.skip_reason = "OR_WIDTH_WARMUP"
            return None, state

        # Apply filters
        should_filter, filter_reason = self.apply_filters(
            state.or_width, state.or_width_avg
        )
        if should_filter:
            state.skip_reason = filter_reason
            return None, state

        # Compute stop distance
        stop_dist = self.compute_stop_distance(state.or_width, state.atr_d)

        # Compute position size
        qty = self.compute_position_size(nav, stop_dist, contract)
        if qty < 1:
            state.skip_reason = "SIZE_TOO_SMALL"
            return None, state

        # Get trading bars
        trading_bars = self.get_trading_bars(symbol, dt)
        if trading_bars.empty:
            state.skip_reason = "NO_TRADING_BARS"
            return None, state

        # Simulate entry
        entry_result = self.simulate_entry(
            trading_bars, state.or_high, state.or_low, contract, state.is_fomc_day
        )
        if entry_result is None:
            state.skip_reason = "NO_BREAKOUT"
            return None, state

        direction, entry_time, trigger_price, entry_price = entry_result

        # Compute stop/target from fill price
        if direction == "LONG":
            stop_price = entry_price - stop_dist
            target_price = entry_price + TARGET_MULTIPLIER * stop_dist
        else:
            stop_price = entry_price + stop_dist
            target_price = entry_price - TARGET_MULTIPLIER * stop_dist

        # Simulate exit
        exit_time, exit_price, exit_reason = self.simulate_exit(
            trading_bars, entry_time, entry_price, stop_dist,
            direction, contract, state.is_fomc_day
        )

        # Calculate P&L
        if direction == "LONG":
            gross_pnl = (exit_price - entry_price) * contract.point_value * qty
        else:
            gross_pnl = (entry_price - exit_price) * contract.point_value * qty

        # Entry slippage is already in entry_price, exit slippage in exit_price
        # Compute slippage cost as difference from theoretical
        theoretical_entry = trigger_price
        theoretical_exit = stop_price if exit_reason == "STOP" else (
            target_price if exit_reason == "TARGET" else exit_price
        )

        if direction == "LONG":
            slippage_cost = (
                (entry_price - theoretical_entry) +
                (theoretical_exit - exit_price) if exit_reason != "EOD" else 0
            ) * contract.point_value * qty
        else:
            slippage_cost = (
                (theoretical_entry - entry_price) +
                (exit_price - theoretical_exit) if exit_reason != "EOD" else 0
            ) * contract.point_value * qty

        commission = contract.commission_rt * qty
        net_pnl = gross_pnl - commission

        trade = ORBTrade(
            date=dt,
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            contracts=qty,
            gross_pnl=gross_pnl,
            slippage_cost=slippage_cost,
            commission=commission,
            net_pnl=net_pnl,
            or_high=state.or_high,
            or_low=state.or_low,
            or_width=state.or_width,
            atr_d=state.atr_d,
            stop_dist=stop_dist,
        )

        return trade, state

    def run_period(
        self,
        start: date,
        end: date,
        symbols: List[str],
        initial_nav: float,
    ) -> Tuple[List[ORBTrade], pd.Series, Dict[str, Dict[date, float]]]:
        """
        Run backtest for a date range.

        Returns (trades, daily_equity, or_width_histories).
        """
        trades: List[ORBTrade] = []
        equity_points: List[Tuple[date, float]] = []

        # Track OR width history per symbol
        or_width_histories: Dict[str, Dict[date, float]] = {s: {} for s in symbols}

        # Current NAV
        nav = initial_nav

        # Get trading days
        all_dates = set()
        for sym in symbols:
            try:
                daily = self.build_daily_data(sym)
                sym_dates = set(d for d in daily.index.date if start <= d <= end)
                all_dates.update(sym_dates)
            except Exception as e:
                logger.warning(f"Could not load data for {sym}: {e}")

        trading_days = sorted(all_dates)

        for dt in trading_days:
            day_pnl = 0.0

            for sym in symbols:
                try:
                    trade, state = self.run_single_day(
                        dt, sym, nav, or_width_histories[sym]
                    )
                    if trade is not None:
                        trades.append(trade)
                        day_pnl += trade.net_pnl
                except Exception as e:
                    logger.warning(f"Error on {dt} {sym}: {e}")

            nav += day_pnl
            equity_points.append((dt, nav))

        # Build equity series
        if equity_points:
            equity = pd.Series(
                data=[v for _, v in equity_points],
                index=pd.DatetimeIndex([d for d, _ in equity_points]),
                name="equity",
            )
        else:
            equity = pd.Series(dtype=float, name="equity")

        return trades, equity, or_width_histories

    def evaluate_fold(
        self,
        fold: Dict[str, Any],
        symbols: List[str],
        initial_nav: float,
    ) -> FoldResult:
        """
        Evaluate a single walk-forward fold.

        The train period is used for indicator warm-up, test period for evaluation.
        """
        fold_id = fold["fold_id"]
        train_start = parse_date(fold["train_start"])
        train_end = parse_date(fold["train_end"])
        test_start = parse_date(fold["test_start"])
        test_end = parse_date(fold["test_end"])

        logger.info(f"Fold {fold_id}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")

        # Run train period for warm-up (build OR width histories, etc.)
        _, _, or_histories = self.run_period(train_start, train_end, symbols, initial_nav)

        # Run test period
        trades, equity, _ = self.run_period(test_start, test_end, symbols, initial_nav)

        # Compute metrics
        n_trades = len(trades)
        if n_trades == 0:
            return FoldResult(
                fold_id=fold_id,
                train_period=f"{train_start} to {train_end}",
                test_period=f"{test_start} to {test_end}",
                n_trades=0,
                n_wins=0,
                n_losses=0,
                gross_pnl=0.0,
                net_pnl=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_trade_per_contract=0.0,
                max_dd=0.0,
                trades=[],
                daily_equity=equity,
                passes_kill_tests=False,
            )

        n_wins = sum(1 for t in trades if t.net_pnl > 0)
        n_losses = n_trades - n_wins
        gross_pnl = sum(t.gross_pnl for t in trades)
        net_pnl = sum(t.net_pnl for t in trades)
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        total_contracts = sum(t.contracts for t in trades)

        win_rate = n_wins / n_trades if n_trades > 0 else 0.0
        avg_trade_per_contract = net_pnl / total_contracts if total_contracts > 0 else 0.0

        # Compute Sharpe on daily returns (include 0 on no-trade days)
        if not equity.empty and len(equity) > 1:
            daily_rets = equity.pct_change().fillna(0)
            sr = sharpe_ratio(daily_rets)
            mdd = max_drawdown(equity)
        else:
            sr = 0.0
            mdd = 0.0

        # Check if fold passes kill tests
        passes = (
            net_pnl > 0 and
            mdd >= -0.20 and
            n_trades >= 20
        )

        return FoldResult(
            fold_id=fold_id,
            train_period=f"{train_start} to {train_end}",
            test_period=f"{test_start} to {test_end}",
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_commission=total_commission,
            total_slippage=total_slippage,
            sharpe_ratio=sr,
            win_rate=win_rate,
            avg_trade_per_contract=avg_trade_per_contract,
            max_dd=mdd,
            trades=trades,
            daily_equity=equity,
            passes_kill_tests=passes,
        )

    def run_walk_forward(
        self,
        symbols: List[str] = None,
        initial_nav: float = 100_000.0,
        folds: List[Dict] = None,
    ) -> BacktestResult:
        """
        Run 6-fold walk-forward validation.
        """
        if symbols is None:
            symbols = list(CONTRACTS.keys())
        if folds is None:
            folds = WALK_FORWARD_FOLDS

        fold_results: List[FoldResult] = []
        all_trades: List[ORBTrade] = []

        for fold_def in folds:
            result = self.evaluate_fold(fold_def, symbols, initial_nav)
            fold_results.append(result)
            all_trades.extend(result.trades)

        # Compute aggregate OOS metrics
        n_pass = sum(1 for f in fold_results if f.passes_kill_tests)
        total_trades = sum(f.n_trades for f in fold_results)
        total_net_pnl = sum(f.net_pnl for f in fold_results)
        total_contracts = sum(sum(t.contracts for t in f.trades) for f in fold_results)

        if total_contracts > 0:
            agg_avg_trade = total_net_pnl / total_contracts
        else:
            agg_avg_trade = 0.0

        # Per-symbol profitability check
        mes_pnl = sum(t.net_pnl for t in all_trades if t.symbol == "MES")
        mnq_pnl = sum(t.net_pnl for t in all_trades if t.symbol == "MNQ")
        both_profitable = mes_pnl > 0 and mnq_pnl > 0

        # Aggregate Sharpe (combine all fold returns)
        if fold_results:
            all_sharpes = [f.sharpe_ratio for f in fold_results]
            mean_sharpe = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0
        else:
            mean_sharpe = 0.0

        # Check kill-test gates
        passes_all = (
            total_net_pnl > 0 and
            mean_sharpe >= 0.5 and
            n_pass >= 4 and
            both_profitable
        )

        summary = {
            "n_folds": len(fold_results),
            "n_pass": n_pass,
            "total_trades": total_trades,
            "total_net_pnl": total_net_pnl,
            "total_commission": sum(f.total_commission for f in fold_results),
            "total_slippage": sum(f.total_slippage for f in fold_results),
            "mean_sharpe": mean_sharpe,
            "mean_win_rate": sum(f.win_rate for f in fold_results) / len(fold_results) if fold_results else 0.0,
            "avg_trade_per_contract": agg_avg_trade,
            "mes_net_pnl": mes_pnl,
            "mnq_net_pnl": mnq_pnl,
            "both_symbols_profitable": both_profitable,
            "passes_kill_tests": passes_all,
            "slippage_assumption_ticks": self.slippage_ticks,
        }

        config = {
            "symbols": symbols,
            "initial_nav": initial_nav,
            "slippage_ticks_per_side": self.slippage_ticks,
            "or_minutes": 30,
            "stop_multiplier": STOP_MULTIPLIER,
            "stop_atr_floor": STOP_ATR_FLOOR,
            "target_multiplier": TARGET_MULTIPLIER,
            "use_profit_target": bool(self.use_profit_target),
            "buffer_ticks": BUFFER_TICKS,
            "risk_per_trade_bps": RISK_PER_TRADE_BPS,
        }

        return BacktestResult(
            folds=fold_results,
            summary=summary,
            all_trades=all_trades,
            config=config,
        )


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def format_fold_for_json(fold: FoldResult) -> Dict[str, Any]:
    """Format fold result for JSON output (matching Sleeve IM schema)."""
    return {
        "fold_id": fold.fold_id,
        "train_period": fold.train_period,
        "test_period": fold.test_period,
        "n_trades": fold.n_trades,
        "n_wins": fold.n_wins,
        "n_losses": fold.n_losses,
        "gross_pnl": round(fold.gross_pnl, 2),
        "net_pnl": round(fold.net_pnl, 2),
        "net_return": round(fold.net_pnl / 100_000.0, 4) if fold.net_pnl else 0.0,
        "sharpe_ratio": round(fold.sharpe_ratio, 4),
        "win_rate": round(fold.win_rate, 4),
        "avg_trade_per_contract": round(fold.avg_trade_per_contract, 2),
        "max_dd": round(fold.max_dd, 4),
        "total_commission": round(fold.total_commission, 2),
        "total_slippage": round(fold.total_slippage, 2),
        # Coerce to plain bool (avoid numpy/pandas scalar types that json can't serialize).
        "passes_kill_tests": bool(fold.passes_kill_tests),
    }


def save_results(result: BacktestResult, output_path: Path) -> None:
    """Save results to JSON file (Sleeve IM compatible format)."""
    output = {
        "folds": [format_fold_for_json(f) for f in result.folds],
        "summary": {
            "n_folds": result.summary["n_folds"],
            "n_pass": result.summary["n_pass"],
            "total_trades": result.summary["total_trades"],
            "total_net_pnl": round(result.summary["total_net_pnl"], 2),
            "mean_sharpe": round(result.summary["mean_sharpe"], 4),
            "mean_win_rate": round(result.summary["mean_win_rate"], 4),
            "avg_trade_per_contract": round(result.summary["avg_trade_per_contract"], 2),
            "mes_net_pnl": round(result.summary["mes_net_pnl"], 2),
            "mnq_net_pnl": round(result.summary["mnq_net_pnl"], 2),
            # Coerce to plain bool (avoid numpy/pandas scalar types that json can't serialize).
            "both_symbols_profitable": bool(result.summary["both_symbols_profitable"]),
            "passes_kill_tests": bool(result.summary["passes_kill_tests"]),
            "slippage_assumption_ticks": result.summary["slippage_assumption_ticks"],
        },
        "config": result.config,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def print_report(result: BacktestResult) -> None:
    """Print human-readable report."""
    print("\n" + "=" * 70)
    print("ORB FUTURES WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nConfig: {result.config['symbols']}, slippage={result.config['slippage_ticks_per_side']} ticks/side")
    print(f"Initial NAV: ${result.config['initial_nav']:,.0f}")

    print("\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Test Period':<25} {'Trades':<8} {'Net P&L':<12} {'Sharpe':<8} {'Win%':<8} {'MaxDD':<8} {'Pass'}")
    print("-" * 90)

    for f in result.folds:
        print(
            f"{f.fold_id:<6} "
            f"{f.test_period:<25} "
            f"{f.n_trades:<8} "
            f"${f.net_pnl:>10,.2f} "
            f"{f.sharpe_ratio:>7.2f} "
            f"{f.win_rate*100:>6.1f}% "
            f"{f.max_dd*100:>6.1f}% "
            f"{'✅' if f.passes_kill_tests else '❌'}"
        )

    print("\n--- Aggregate OOS Results ---")
    s = result.summary
    print(f"Total Trades: {s['total_trades']}")
    print(f"Total Net P&L: ${s['total_net_pnl']:,.2f}")
    print(f"Mean Sharpe: {s['mean_sharpe']:.2f}")
    print(f"Mean Win Rate: {s['mean_win_rate']*100:.1f}%")
    print(f"Avg Trade/Contract: ${s['avg_trade_per_contract']:.2f}")
    print(f"MES Net P&L: ${s['mes_net_pnl']:,.2f}")
    print(f"MNQ Net P&L: ${s['mnq_net_pnl']:,.2f}")
    print(f"Folds Passing: {s['n_pass']}/6")

    print("\n--- Kill-Test Verdict ---")
    if s["total_net_pnl"] <= 0:
        print("❌ FAIL: Net PnL <= 0")
    elif s["mean_sharpe"] < 0:
        print("❌ FAIL: Sharpe < 0")
    elif s["mean_sharpe"] < 0.5:
        print(f"🟡 MARGINAL: Sharpe {s['mean_sharpe']:.2f} < 0.5 threshold")
    elif s["n_pass"] < 4:
        print(f"❌ FAIL: Only {s['n_pass']}/6 folds pass (need ≥4)")
    elif not s["both_symbols_profitable"]:
        print("❌ FAIL: Not both MES and MNQ profitable")
    else:
        print("✅ PASS: Strategy meets all kill-test criteria!")

    overall = "🟢 PROMOTE" if s["passes_kill_tests"] else "🔴 KILL"
    print(f"\n{overall}: {'Ready for paper trading' if s['passes_kill_tests'] else 'Do not trade'}")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="ORB Futures Backtester")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/orb",
        help="Directory containing minute bar data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/orb/walk_forward_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--slippage",
        type=int,
        default=1,
        help="Slippage in ticks per side (1=baseline, 2=stress)",
    )
    parser.add_argument(
        "--initial-nav",
        type=float,
        default=100_000.0,
        help="Initial NAV for backtest",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="MES,MNQ",
        help="Comma-separated symbols to test",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging",
    )
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Disable profit target; use stop + mandatory flatten only (research variant).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease ensure MES/MNQ 1-minute data files exist in the data directory.")
        print("Expected files: MES_1min_*.parquet, MNQ_1min_*.parquet")
        return 1

    symbols = [s.strip() for s in args.symbols.split(",")]

    print(f"Running ORB walk-forward backtest...")
    print(f"Data dir: {data_dir}")
    print(f"Symbols: {symbols}")
    print(f"Slippage: {args.slippage} ticks/side")

    bt = ORBBacktester(
        data_dir=data_dir,
        slippage_ticks=args.slippage,
        use_profit_target=not args.no_target,
    )

    try:
        result = bt.run_walk_forward(
            symbols=symbols,
            initial_nav=args.initial_nav,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure minute bar data files exist for the specified symbols.")
        return 1

    # Save and print results
    save_results(result, Path(args.output))
    print_report(result)

    return 0 if result.summary["passes_kill_tests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
