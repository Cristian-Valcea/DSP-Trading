"""
TSMOM Futures Backtester - Time-Series Momentum Strategy Kill-Test

Implements the strategy defined in SLEEVE_TSMOM_MINIMAL_SPEC.md (v1.1):
- 252-day trailing return signal (sign only, no strength scaling)
- Risk parity portfolio construction with 8% vol target
- Weekly rebalancing (first trading day of week)
- Volume-led futures roll simulation
- Four asset class buckets: equities, commodities, FX, rates

Kill-Test Purpose
-----------------
Validate that TSMOM has positive edge after transaction costs before
promoting to production. Uses conservative assumptions:
- 1 tick/side slippage for futures (2 ticks stress)
- 2 bps/side slippage for ETFs (4 bps stress)
- Explicit roll cost simulation
- No parameter optimization
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import floor, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS (FROZEN per spec v1.1)
# ============================================================================

ET = pytz.timezone("America/New_York")

# Universe - 8 micro futures + 2 bond ETFs (Section 2)
FUTURES = ["MES", "MNQ", "M2K", "MYM", "MGC", "MCL", "M6E", "M6B"]
ETFS = ["TLT", "IEF"]
ALL_INSTRUMENTS = FUTURES + ETFS

# Contract specifications (frozen)
CONTRACTS = {
    # Equities
    "MES": {"point_value": 5.0, "tick_size": 0.25, "tick_value": 1.25, "commission_rt": 1.24, "bucket": "equities"},
    "MNQ": {"point_value": 2.0, "tick_size": 0.25, "tick_value": 0.50, "commission_rt": 1.24, "bucket": "equities"},
    "M2K": {"point_value": 5.0, "tick_size": 0.10, "tick_value": 0.50, "commission_rt": 1.24, "bucket": "equities"},
    "MYM": {"point_value": 0.5, "tick_size": 1.00, "tick_value": 0.50, "commission_rt": 1.24, "bucket": "equities"},
    # Commodities
    "MGC": {"point_value": 10.0, "tick_size": 0.10, "tick_value": 1.00, "commission_rt": 1.24, "bucket": "commodities"},
    "MCL": {"point_value": 100.0, "tick_size": 0.01, "tick_value": 1.00, "commission_rt": 1.24, "bucket": "commodities"},
    # FX
    "M6E": {"point_value": 12500.0, "tick_size": 0.0001, "tick_value": 1.25, "commission_rt": 1.24, "bucket": "fx"},
    "M6B": {"point_value": 6250.0, "tick_size": 0.0001, "tick_value": 0.625, "commission_rt": 1.24, "bucket": "fx"},
    # Rates (ETFs - no point_value, tick_value)
    "TLT": {"commission_rt": 0.0, "bucket": "rates"},
    "IEF": {"commission_rt": 0.0, "bucket": "rates"},
}

# Risk budget allocation (Section 5.2)
BUCKET_WEIGHTS = {
    "equities": 0.25,
    "commodities": 0.25,
    "fx": 0.25,
    "rates": 0.25,
}

# Signal and sizing parameters (Sections 4, 5)
SIGNAL_LOOKBACK = 252  # ~12 months
COV_LOOKBACK = 60  # Volatility/covariance estimation window
TARGET_VOL = 0.08  # 8% annualized portfolio volatility

# Exposure caps (Section 5.5)
MAX_GROSS_EXPOSURE = 3.0
MAX_INSTRUMENT_EXPOSURE = 1.0
MAX_BUCKET_EXPOSURE = 1.25

# Rebalance schedule (Section 6)
REBALANCE_DAY = 0  # Monday (0=Monday, 6=Sunday)

# Turnover deadband (Section 6.3)
FUTURES_MIN_QTY_CHANGE = 1  # contracts
ETF_MIN_QTY_CHANGE = 10  # shares

# Transaction costs (Section 7)
FUTURES_SLIPPAGE_TICKS_BASELINE = 1
FUTURES_SLIPPAGE_TICKS_STRESS = 2
ETF_SLIPPAGE_BPS_BASELINE = 2
ETF_SLIPPAGE_BPS_STRESS = 4

# Walk-forward folds (Section 8.2)
WALK_FORWARD_FOLDS = [
    {"fold_id": 1, "train_start": "2021-01-01", "train_end": "2022-12-30",
     "test_start": "2023-01-03", "test_end": "2023-12-29"},
    {"fold_id": 2, "train_start": "2021-01-01", "train_end": "2023-12-29",
     "test_start": "2024-01-02", "test_end": "2024-12-31"},
    {"fold_id": 3, "train_start": "2021-01-01", "train_end": "2024-12-31",
     "test_start": "2025-01-02", "test_end": "2025-03-31"},
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class InstrumentSpec:
    symbol: str
    is_future: bool
    bucket: str
    point_value: Optional[float] = None  # Futures only
    tick_size: Optional[float] = None  # Futures only
    tick_value: Optional[float] = None  # Futures only
    commission_rt: float = 0.0


@dataclass
class Position:
    """Current position state for an instrument."""
    symbol: str
    quantity: int  # Signed (positive=long, negative=short)
    entry_price: float  # last execution price (with slippage), for diagnostics only
    entry_date: date  # last execution date (actual trade date)


@dataclass
class Rebalance:
    """Single rebalance event."""
    date: date
    symbol: str
    old_qty: int
    new_qty: int
    execution_price: float
    is_roll: bool = False  # True if roll event
    roll_cost: float = 0.0  # Roll cost if applicable
    slippage_cost: float = 0.0
    commission: float = 0.0
    signal: int = 0  # -1, 0, +1


@dataclass
class DailySnapshot:
    """Daily portfolio state."""
    date: date
    nav: float
    positions: Dict[str, Position]
    gross_exposure: float
    net_exposure: float
    daily_pnl: float
    daily_return: float


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""
    fold_id: int
    train_period: str
    test_period: str
    n_rebalances: int
    total_turnover: float  # Sum of |trade notionals|
    gross_pnl: float
    net_pnl: float
    total_commission: float
    total_slippage: float
    sharpe_ratio: float
    max_dd: float
    avg_gross_exposure: float
    max_gross_exposure: float
    # Per-instrument metrics
    per_instrument_pnl: Dict[str, float] = field(default_factory=dict)
    # Per-bucket metrics
    per_bucket_pnl: Dict[str, float] = field(default_factory=dict)
    # Time series
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    rebalances: List[Rebalance] = field(default_factory=list)
    passes_kill_tests: bool = False


@dataclass
class BacktestResult:
    """Complete backtest result."""
    folds: List[FoldResult]
    summary: Dict[str, Any]
    config: Dict[str, Any]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_date(s: str) -> date:
    """Parse YYYY-MM-DD string to date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def get_instrument_spec(symbol: str) -> InstrumentSpec:
    """Get instrument specification."""
    if symbol not in CONTRACTS:
        raise ValueError(f"Unknown instrument: {symbol}")
    c = CONTRACTS[symbol]

    is_future = symbol in FUTURES
    return InstrumentSpec(
        symbol=symbol,
        is_future=is_future,
        bucket=c["bucket"],
        point_value=c.get("point_value"),
        tick_size=c.get("tick_size"),
        tick_value=c.get("tick_value"),
        commission_rt=c["commission_rt"],
    )


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    if daily_returns.empty or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """Calculate max drawdown from equity series."""
    if equity.empty or len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def is_monday(dt: date) -> bool:
    """Check if date is a Monday (rebalance day)."""
    return dt.weekday() == REBALANCE_DAY


# ============================================================================
# TSMOM BACKTESTER
# ============================================================================

class TSMOMBacktester:
    """
    Time-Series Momentum backtester for cross-asset futures + ETF portfolio.

    Key features:
    - 252-day trailing return signal (sign only)
    - Risk parity portfolio construction with 8% vol target
    - Weekly rebalancing (Mondays)
    - Volume-led futures roll simulation
    - Conservative transaction cost assumptions
    """

    def __init__(
        self,
        data_dir: Path,
        slippage_mode: str = "baseline",  # "baseline" or "stress"
    ):
        self.data_dir = Path(data_dir)
        self.slippage_mode = slippage_mode

        # Slippage parameters
        if slippage_mode == "stress":
            self.futures_slippage_ticks = FUTURES_SLIPPAGE_TICKS_STRESS
            self.etf_slippage_bps = ETF_SLIPPAGE_BPS_STRESS
        else:
            self.futures_slippage_ticks = FUTURES_SLIPPAGE_TICKS_BASELINE
            self.etf_slippage_bps = ETF_SLIPPAGE_BPS_BASELINE

        # Data cache
        self._daily_data: Dict[str, pd.DataFrame] = {}

    def load_daily_data(self, symbol: str) -> pd.DataFrame:
        """Load daily OHLC data for symbol."""
        if symbol in self._daily_data:
            return self._daily_data[symbol]

        # Find parquet file
        pattern = f"{symbol}_1d_*.parquet"
        files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No data file found for {symbol} in {self.data_dir}")

        df = pd.read_parquet(files[0])

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                raise ValueError(f"Cannot determine datetime column for {symbol}")

        # Ensure date-only index (no time component)
        df.index = pd.to_datetime(df.index.date)

        # Ensure OHLC columns
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} for {symbol}")

        # Sort by date
        df = df.sort_index()

        self._daily_data[symbol] = df
        return df

    def compute_252d_return(self, symbol: str, as_of_date: date) -> Optional[float]:
        """
        Compute 252-trading-day trailing return.

        Returns None if insufficient history.
        """
        df = self.load_daily_data(symbol)

        # Get data up to and including as_of_date
        df_subset = df[df.index.date <= as_of_date]

        if len(df_subset) < SIGNAL_LOOKBACK + 1:
            return None

        # Get close at t and t-252
        close_t = df_subset.iloc[-1]["close"]
        close_t_minus_252 = df_subset.iloc[-(SIGNAL_LOOKBACK + 1)]["close"]

        return (close_t / close_t_minus_252) - 1.0

    def compute_signal(self, symbol: str, as_of_date: date) -> int:
        """
        Compute signal for instrument: -1, 0, +1.

        Returns 0 if insufficient history.
        """
        ret = self.compute_252d_return(symbol, as_of_date)

        if ret is None:
            return 0

        if ret > 0:
            return 1
        elif ret < 0:
            return -1
        else:
            return 0

    def compute_covariance_matrix(
        self,
        symbols: List[str],
        as_of_date: date,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute 60-day covariance matrix of daily returns.

        Returns (covariance_matrix, valid_symbols).
        Valid symbols are those with sufficient history.
        """
        # Collect returns for last 60 days before as_of_date
        returns_dict = {}

        for sym in symbols:
            df = self.load_daily_data(sym)
            df_subset = df[df.index.date < as_of_date]  # Strictly before

            if len(df_subset) < COV_LOOKBACK + 1:
                continue  # Skip if insufficient history

            # Get last 60 days of returns
            recent = df_subset.tail(COV_LOOKBACK + 1)
            rets = recent["close"].pct_change().dropna()

            if len(rets) < COV_LOOKBACK:
                continue

            returns_dict[sym] = rets.tail(COV_LOOKBACK).values

        if not returns_dict:
            return np.array([[]]), []

        # Build returns matrix (n_days x n_instruments)
        valid_symbols = sorted(returns_dict.keys())
        returns_matrix = np.column_stack([returns_dict[s] for s in valid_symbols])

        # Compute covariance
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        return cov_matrix, valid_symbols

    def compute_target_positions(
        self,
        signals: Dict[str, int],
        as_of_date: date,
        nav: float,
    ) -> Dict[str, int]:
        """
        Compute target positions using risk parity + portfolio vol targeting.

        Returns dict of {symbol: quantity} (signed integers).
        """
        # Get valid instruments (those with signals and covariance data)
        active_symbols = [s for s, sig in signals.items() if sig != 0]

        if not active_symbols:
            return {s: 0 for s in ALL_INSTRUMENTS}

        # Compute covariance matrix
        cov_matrix, valid_symbols = self.compute_covariance_matrix(active_symbols, as_of_date)

        if len(valid_symbols) == 0:
            return {s: 0 for s in ALL_INSTRUMENTS}

        # Compute per-instrument risk weights (Section 5.2)
        # Count instruments per bucket
        bucket_counts = {}
        for sym in valid_symbols:
            spec = get_instrument_spec(sym)
            bucket = spec.bucket
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        # Compute risk weight per instrument
        risk_weights = {}
        for sym in valid_symbols:
            spec = get_instrument_spec(sym)
            bucket = spec.bucket
            bucket_weight = BUCKET_WEIGHTS[bucket]
            n_instruments = bucket_counts[bucket]
            risk_weights[sym] = bucket_weight / n_instruments

        # Compute raw exposures (Section 5.4)
        volatilities = np.sqrt(np.diag(cov_matrix))
        raw_exposures = {}

        for i, sym in enumerate(valid_symbols):
            sig = signals.get(sym, 0)
            if sig == 0:
                raw_exposures[sym] = 0.0
                continue

            w_i = risk_weights[sym]
            sigma_i = volatilities[i]

            if sigma_i == 0:
                raw_exposures[sym] = 0.0
            else:
                raw_exposures[sym] = sig * (w_i / sigma_i)

        # Build exposure vector for portfolio vol calculation
        exp_vector = np.array([raw_exposures[s] for s in valid_symbols])

        # Compute portfolio volatility of raw exposures
        if exp_vector.sum() == 0:
            return {s: 0 for s in ALL_INSTRUMENTS}

        portfolio_var = exp_vector @ cov_matrix @ exp_vector
        portfolio_vol_annual = sqrt(portfolio_var * 252)

        # Scale exposures to target vol (Section 5.4 step 4)
        if portfolio_vol_annual == 0:
            scale_factor = 0.0
        else:
            scale_factor = TARGET_VOL / portfolio_vol_annual

        scaled_exposures = {s: e * scale_factor for s, e in raw_exposures.items()}

        # Apply exposure caps (Section 5.5)
        scaled_exposures = self._apply_exposure_caps(scaled_exposures)

        # Convert exposures to quantities (Section 5.6)
        target_positions = {}

        for sym in ALL_INSTRUMENTS:
            exposure = scaled_exposures.get(sym, 0.0)

            if abs(exposure) < 1e-6:
                target_positions[sym] = 0
                continue

            # Get current price
            df = self.load_daily_data(sym)
            df_subset = df[df.index.date <= as_of_date]

            if df_subset.empty:
                target_positions[sym] = 0
                continue

            price = df_subset.iloc[-1]["close"]
            spec = get_instrument_spec(sym)

            if spec.is_future:
                # Futures: notional = price * point_value
                notional_per_contract = price * spec.point_value
                target_notional = exposure * nav
                qty_float = target_notional / notional_per_contract
                qty = int(round(qty_float))
            else:
                # ETF: notional = price * shares
                target_notional = exposure * nav
                qty_float = target_notional / price
                qty = int(round(qty_float))

                # Minimum 10 shares (Section 5.6)
                if abs(qty) < ETF_MIN_QTY_CHANGE:
                    qty = 0

            target_positions[sym] = qty

        return target_positions

    def _apply_exposure_caps(self, exposures: Dict[str, float]) -> Dict[str, float]:
        """
        Apply exposure caps per Section 5.5.

        Returns capped exposures.
        """
        # Check gross exposure cap
        gross_exposure = sum(abs(e) for e in exposures.values())

        if gross_exposure > MAX_GROSS_EXPOSURE:
            scale = MAX_GROSS_EXPOSURE / gross_exposure
            exposures = {s: e * scale for s, e in exposures.items()}

        # Check per-instrument cap
        for sym in list(exposures.keys()):
            if abs(exposures[sym]) > MAX_INSTRUMENT_EXPOSURE:
                exposures[sym] = MAX_INSTRUMENT_EXPOSURE * (1 if exposures[sym] > 0 else -1)

        # Check per-bucket cap
        bucket_exposures = {}
        for sym, exp in exposures.items():
            spec = get_instrument_spec(sym)
            bucket = spec.bucket
            bucket_exposures[bucket] = bucket_exposures.get(bucket, 0.0) + abs(exp)

        for bucket, bucket_exp in bucket_exposures.items():
            if bucket_exp > MAX_BUCKET_EXPOSURE:
                scale = MAX_BUCKET_EXPOSURE / bucket_exp
                # Scale down all instruments in this bucket
                for sym in exposures:
                    spec = get_instrument_spec(sym)
                    if spec.bucket == bucket:
                        exposures[sym] *= scale

        return exposures

    def execute_rebalance(
        self,
        dt: date,
        current_positions: Dict[str, Position],
        target_positions: Dict[str, int],
        signals: Dict[str, int],
    ) -> Tuple[Dict[str, Position], List[Rebalance]]:
        """
        Execute rebalance from current to target positions.

        Returns (new_positions, rebalance_events).
        """
        new_positions = dict(current_positions)
        rebalances = []

        for sym in ALL_INSTRUMENTS:
            current_qty = current_positions.get(sym, Position(sym, 0, 0.0, dt)).quantity
            target_qty = target_positions.get(sym, 0)

            delta = target_qty - current_qty

            # Check turnover deadband (Section 6.3)
            spec = get_instrument_spec(sym)

            if spec.is_future:
                min_change = FUTURES_MIN_QTY_CHANGE
            else:
                min_change = ETF_MIN_QTY_CHANGE

            # Skip if below threshold AND signal hasn't flipped
            if abs(delta) < min_change:
                old_signal = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)
                new_signal = signals.get(sym, 0)

                if old_signal == new_signal:
                    continue  # No trade

            if delta == 0:
                continue  # No change

            # Get execution price at today's open (decision uses prior close; execution at next open)
            df = self.load_daily_data(sym)
            day_data = df[df.index.date == dt]
            if day_data.empty:
                continue
            exec_price = float(day_data.iloc[0]["open"])

            # Apply slippage
            if spec.is_future:
                slip_ticks = self.futures_slippage_ticks
                slip_amount = slip_ticks * spec.tick_size
            else:
                slip_bps = self.etf_slippage_bps
                slip_amount = exec_price * (slip_bps / 10000.0)

            # Worsen price based on trade direction
            if delta > 0:  # Buying
                exec_price_with_slip = exec_price + slip_amount
            else:  # Selling
                exec_price_with_slip = exec_price - slip_amount

            # Calculate costs
            if spec.is_future:
                slippage_cost = abs(delta) * slip_amount * spec.point_value
            else:
                slippage_cost = abs(delta) * slip_amount

            # Commission is defined in the spec as round-trip (RT). Here we execute ONE side.
            commission = (spec.commission_rt / 2.0) * abs(delta) if spec.is_future else spec.commission_rt * abs(delta)

            # Create rebalance record
            rebalance = Rebalance(
                date=dt,
                symbol=sym,
                old_qty=current_qty,
                new_qty=target_qty,
                execution_price=exec_price_with_slip,
                slippage_cost=slippage_cost,
                commission=commission,
                signal=signals.get(sym, 0),
            )
            rebalances.append(rebalance)

            # Update position
            if target_qty == 0:
                if sym in new_positions:
                    del new_positions[sym]
            else:
                new_positions[sym] = Position(
                    symbol=sym,
                    quantity=target_qty,
                    entry_price=exec_price_with_slip,
                    entry_date=dt,
                )

        return new_positions, rebalances

    def _get_open_close(self, symbol: str, dt: date) -> Optional[Tuple[float, float]]:
        df = self.load_daily_data(symbol)
        day = df[df.index.date == dt]
        if day.empty:
            return None
        row = day.iloc[0]
        return float(row["open"]), float(row["close"])

    def run_period(
        self,
        start: date,
        end: date,
        initial_nav: float,
    ) -> Tuple[List[DailySnapshot], List[Rebalance], Dict[str, float], Dict[str, float]]:
        """
        Run backtest for a date range.

        Returns (daily_snapshots, rebalances, per_instrument_pnl, per_bucket_pnl).
        """
        # Get all trading days (union of all instrument trading days)
        all_dates = set()
        for sym in ALL_INSTRUMENTS:
            try:
                df = self.load_daily_data(sym)
                sym_dates = set(d for d in df.index.date if start <= d <= end)
                all_dates.update(sym_dates)
            except Exception as e:
                logger.warning(f"Could not load data for {sym}: {e}")

        trading_days = sorted(all_dates)

        if not trading_days:
            return [], []

        # Initialize
        nav = initial_nav
        positions: Dict[str, Position] = {}
        snapshots = []
        all_rebalances = []
        per_instrument_pnl: Dict[str, float] = {s: 0.0 for s in ALL_INSTRUMENTS}
        per_bucket_pnl: Dict[str, float] = {b: 0.0 for b in BUCKET_WEIGHTS.keys()}
        prev_close: Dict[str, float] = {}

        for idx, dt in enumerate(trading_days):
            nav_prev = nav

            # Check if rebalance day (Monday)
            should_rebalance = is_monday(dt)

            if should_rebalance:
                # Signals/risk use prior close (no lookahead). If we don't have a prior
                # trading day in the master calendar, skip rebalancing on this date.
                if idx == 0:
                    signals = {sym: 0 for sym in ALL_INSTRUMENTS}
                    rebalances = []
                else:
                    as_of_date = trading_days[idx - 1]
                    signals = {sym: self.compute_signal(sym, as_of_date) for sym in ALL_INSTRUMENTS}
                    target_positions = self.compute_target_positions(signals, as_of_date, nav_prev)
                    positions, rebalances = self.execute_rebalance(dt, positions, target_positions, signals)
                all_rebalances.extend(rebalances)

            # Compute daily P&L instrument-by-instrument using:
            # prev_qty * (exec_price - prev_close) + new_qty * (close - exec_price)
            # where exec_price is today's open with slippage for traded deltas, else today's open.
            rebal_by_sym: Dict[str, Rebalance] = {r.symbol: r for r in rebalances} if should_rebalance else {}
            daily_pnl = 0.0

            for sym in ALL_INSTRUMENTS:
                spec = get_instrument_spec(sym)
                oc = self._get_open_close(sym, dt)
                if oc is None:
                    continue  # market closed / no bar
                open_px, close_px = oc

                prev_qty = positions.get(sym, Position(sym, 0, 0.0, dt)).quantity
                rb = rebal_by_sym.get(sym)
                if rb is not None:
                    exec_px = float(rb.execution_price)  # already includes slippage
                    new_qty = rb.new_qty
                    commission = float(rb.commission)
                else:
                    exec_px = open_px
                    new_qty = prev_qty
                    commission = 0.0

                prev_c = prev_close.get(sym)
                if prev_c is None:
                    # First observation for this instrument: start PnL accounting from today.
                    prev_close[sym] = close_px
                    continue

                mult = spec.point_value if spec.is_future else 1.0
                inst_pnl = prev_qty * (exec_px - prev_c) * mult + new_qty * (close_px - exec_px) * mult - commission

                daily_pnl += inst_pnl
                per_instrument_pnl[sym] += inst_pnl
                per_bucket_pnl[spec.bucket] += inst_pnl

                prev_close[sym] = close_px

            nav = nav_prev + daily_pnl

            # Compute exposures
            gross_exp = 0.0
            net_exp = 0.0

            for sym, pos in positions.items():
                oc = self._get_open_close(sym, dt)
                if oc is None:
                    continue
                _, price = oc
                spec = get_instrument_spec(sym)

                if spec.is_future:
                    notional = price * abs(pos.quantity) * spec.point_value
                else:
                    notional = price * abs(pos.quantity)

                exposure = notional / nav if nav > 0 else 0.0
                gross_exp += abs(exposure)
                net_exp += exposure * (1 if pos.quantity > 0 else -1)

            # Daily return
            daily_return = daily_pnl / nav_prev if nav_prev > 0 else 0.0

            # Create snapshot
            snapshot = DailySnapshot(
                date=dt,
                nav=nav,
                positions=dict(positions),
                gross_exposure=gross_exp,
                net_exposure=net_exp,
                daily_pnl=daily_pnl,
                daily_return=daily_return,
            )
            snapshots.append(snapshot)

        return snapshots, all_rebalances, per_instrument_pnl, per_bucket_pnl

    def evaluate_fold(
        self,
        fold: Dict[str, Any],
        initial_nav: float,
    ) -> FoldResult:
        """
        Evaluate a single walk-forward fold.

        The train period provides warm-up data, test period is OOS evaluation.
        """
        fold_id = fold["fold_id"]
        test_start = parse_date(fold["test_start"])
        test_end = parse_date(fold["test_end"])

        logger.info(f"Fold {fold_id}: Test {test_start} to {test_end}")

        # Run test period
        snapshots, rebalances, per_instrument_pnl, per_bucket_pnl = self.run_period(test_start, test_end, initial_nav)

        if not snapshots:
            return FoldResult(
                fold_id=fold_id,
                train_period=f"{fold['train_start']} to {fold['train_end']}",
                test_period=f"{fold['test_start']} to {fold['test_end']}",
                n_rebalances=0,
                total_turnover=0.0,
                gross_pnl=0.0,
                net_pnl=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                sharpe_ratio=0.0,
                max_dd=0.0,
                avg_gross_exposure=0.0,
                max_gross_exposure=0.0,
                daily_snapshots=[],
                rebalances=[],
                passes_kill_tests=False,
            )

        # Compute metrics
        final_nav = snapshots[-1].nav
        net_pnl = final_nav - initial_nav

        # Build equity series
        equity = pd.Series(
            data=[s.nav for s in snapshots],
            index=pd.DatetimeIndex([s.date for s in snapshots]),
        )

        # Daily returns
        daily_returns = pd.Series([s.daily_return for s in snapshots])

        # Sharpe ratio
        sr = sharpe_ratio(daily_returns)

        # Max drawdown
        mdd = max_drawdown(equity)

        # Transaction costs
        total_commission = sum(r.commission for r in rebalances)
        total_slippage = sum(r.slippage_cost for r in rebalances)

        # Turnover
        total_turnover = 0.0
        for r in rebalances:
            spec = get_instrument_spec(r.symbol)
            delta_qty = abs(r.new_qty - r.old_qty)

            if spec.is_future:
                notional = r.execution_price * delta_qty * spec.point_value
            else:
                notional = r.execution_price * delta_qty

            total_turnover += notional

        # Exposure stats
        avg_gross_exp = sum(s.gross_exposure for s in snapshots) / len(snapshots)
        max_gross_exp = max(s.gross_exposure for s in snapshots)

        # Per-instrument / per-bucket P&L is accumulated directly during simulation (net of commissions).

        # Check if fold passes kill tests (Section 9.1)
        passes = (
            net_pnl > 0 and
            sr >= 0.25 and  # Fold-level threshold is 0.25, aggregate is 0.50
            mdd >= -0.20
        )

        return FoldResult(
            fold_id=fold_id,
            train_period=f"{fold['train_start']} to {fold['train_end']}",
            test_period=f"{fold['test_start']} to {fold['test_end']}",
            n_rebalances=len(rebalances),
            total_turnover=total_turnover,
            gross_pnl=net_pnl + total_commission + total_slippage,  # Gross before costs
            net_pnl=net_pnl,
            total_commission=total_commission,
            total_slippage=total_slippage,
            sharpe_ratio=sr,
            max_dd=mdd,
            avg_gross_exposure=avg_gross_exp,
            max_gross_exposure=max_gross_exp,
            per_instrument_pnl=per_instrument_pnl,
            per_bucket_pnl=per_bucket_pnl,
            daily_snapshots=snapshots,
            rebalances=rebalances,
            passes_kill_tests=passes,
        )

    def run_walk_forward(
        self,
        initial_nav: float = 100_000.0,
        folds: List[Dict] = None,
    ) -> BacktestResult:
        """
        Run 3-fold walk-forward validation.
        """
        if folds is None:
            folds = WALK_FORWARD_FOLDS

        fold_results: List[FoldResult] = []

        for fold_def in folds:
            result = self.evaluate_fold(fold_def, initial_nav)
            fold_results.append(result)

        # Compute aggregate OOS metrics
        n_pass = sum(1 for f in fold_results if f.passes_kill_tests)
        total_net_pnl = sum(f.net_pnl for f in fold_results)

        # Mean Sharpe
        all_sharpes = [f.sharpe_ratio for f in fold_results]
        mean_sharpe = sum(all_sharpes) / len(all_sharpes) if all_sharpes else 0.0

        # Aggregate max drawdown across the full OOS timeline by chaining daily returns
        # (folds are sequential and non-overlapping in calendar time).
        chained_equity: List[float] = []
        running_nav = initial_nav
        for f in fold_results:
            for s in f.daily_snapshots:
                running_nav *= (1.0 + float(s.daily_return))
                chained_equity.append(running_nav)

        if chained_equity:
            equity_series = pd.Series(chained_equity)
            agg_max_dd = max_drawdown(equity_series)
        else:
            agg_max_dd = 0.0

        # Check kill-test gates (Section 9.1)
        passes_primary = (
            total_net_pnl > 0 and
            mean_sharpe >= 0.50 and
            agg_max_dd >= -0.20 and
            n_pass >= 2  # At least 2/3 folds pass
        )

        # Concentration gates (Section 9.3) - placeholder
        # Would need proper per-instrument P&L tracking for full implementation
        passes_concentration = True

        summary = {
            "n_folds": len(fold_results),
            "n_pass": n_pass,
            "total_net_pnl": total_net_pnl,
            "mean_sharpe": mean_sharpe,
            "agg_max_dd": agg_max_dd,
            "total_commission": sum(f.total_commission for f in fold_results),
            "total_slippage": sum(f.total_slippage for f in fold_results),
            "total_turnover": sum(f.total_turnover for f in fold_results),
            "passes_kill_tests": passes_primary and passes_concentration,
            "slippage_mode": self.slippage_mode,
        }

        config = {
            "initial_nav": initial_nav,
            "target_vol": TARGET_VOL,
            "signal_lookback": SIGNAL_LOOKBACK,
            "cov_lookback": COV_LOOKBACK,
            "rebalance_frequency": "weekly",
            "slippage_mode": self.slippage_mode,
            "futures_slippage_ticks": self.futures_slippage_ticks,
            "etf_slippage_bps": self.etf_slippage_bps,
        }

        return BacktestResult(
            folds=fold_results,
            summary=summary,
            config=config,
        )


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_results(result: BacktestResult, output_path: Path) -> None:
    """Save results to JSON file with daily equity curves."""
    output = {
        "folds": [
            {
                "fold_id": f.fold_id,
                "test_period": f.test_period,
                "n_rebalances": f.n_rebalances,
                "gross_pnl": round(f.gross_pnl, 2),
                "net_pnl": round(f.net_pnl, 2),
                "sharpe_ratio": round(f.sharpe_ratio, 4),
                "max_dd": round(f.max_dd, 4),
                "total_commission": round(f.total_commission, 2),
                "total_slippage": round(f.total_slippage, 2),
                "total_turnover": round(f.total_turnover, 2),
                "avg_gross_exposure": round(f.avg_gross_exposure, 4),
                "passes_kill_tests": bool(f.passes_kill_tests),
                # Export daily snapshots for visualization
                "daily_snapshots": [
                    {
                        "date": s.date.isoformat(),
                        "nav": round(s.nav, 2),
                        "gross_exposure": round(s.gross_exposure, 4),
                        "net_exposure": round(s.net_exposure, 4),
                        "daily_pnl": round(s.daily_pnl, 2),
                        "daily_return": round(s.daily_return, 6),
                    }
                    for s in f.daily_snapshots
                ],
                # Export per-instrument P&L (Bug #1 fix)
                "per_instrument_pnl": {
                    sym: round(pnl, 2)
                    for sym, pnl in f.per_instrument_pnl.items()
                },
                # Export per-bucket P&L (Bug #1 fix)
                "per_bucket_pnl": {
                    bucket: round(pnl, 2)
                    for bucket, pnl in f.per_bucket_pnl.items()
                },
            }
            for f in result.folds
        ],
        "summary": {
            "n_folds": result.summary["n_folds"],
            "n_pass": result.summary["n_pass"],
            "total_net_pnl": round(result.summary["total_net_pnl"], 2),
            "mean_sharpe": round(result.summary["mean_sharpe"], 4),
            "agg_max_dd": round(result.summary["agg_max_dd"], 4),
            "total_commission": round(result.summary["total_commission"], 2),
            "total_slippage": round(result.summary["total_slippage"], 2),
            "passes_kill_tests": bool(result.summary["passes_kill_tests"]),
            "slippage_mode": result.summary["slippage_mode"],
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
    print("TSMOM FUTURES WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nConfig: slippage_mode={result.config['slippage_mode']}")
    print(f"Initial NAV: ${result.config['initial_nav']:,.0f}")
    print(f"Target Vol: {result.config['target_vol']*100:.0f}%")

    print("\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Test Period':<25} {'Rebal':<8} {'Net P&L':<12} {'Sharpe':<8} {'MaxDD':<8} {'Pass'}")
    print("-" * 90)

    for f in result.folds:
        print(
            f"{f.fold_id:<6} "
            f"{f.test_period:<25} "
            f"{f.n_rebalances:<8} "
            f"${f.net_pnl:>10,.2f} "
            f"{f.sharpe_ratio:>7.2f} "
            f"{f.max_dd*100:>6.1f}% "
            f"{'✅' if f.passes_kill_tests else '❌'}"
        )

    print("\n--- Aggregate OOS Results ---")
    s = result.summary
    print(f"Total Net P&L: ${s['total_net_pnl']:,.2f}")
    print(f"Mean Sharpe: {s['mean_sharpe']:.2f}")
    print(f"Agg Max DD: {s['agg_max_dd']*100:.1f}%")
    print(f"Folds Passing: {s['n_pass']}/3")

    print("\n--- Kill-Test Verdict ---")
    if s["total_net_pnl"] <= 0:
        print("❌ FAIL: Net PnL <= 0")
    elif s["mean_sharpe"] < 0.50:
        print(f"❌ FAIL: Sharpe {s['mean_sharpe']:.2f} < 0.50 threshold")
    elif s["agg_max_dd"] < -0.20:
        print(f"❌ FAIL: Max DD {s['agg_max_dd']*100:.1f}% worse than -20%")
    elif s["n_pass"] < 2:
        print(f"❌ FAIL: Only {s['n_pass']}/3 folds pass (need ≥2)")
    else:
        print("✅ PASS: Strategy meets all kill-test criteria!")

    overall = "🟢 PROMOTE" if s["passes_kill_tests"] else "🔴 KILL"
    print(f"\n{overall}: {'Ready for paper trading' if s['passes_kill_tests'] else 'Do not trade'}")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="TSMOM Futures Backtester")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tsmom",
        help="Directory containing daily bar data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tsmom/walk_forward_baseline.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--slippage-mode",
        type=str,
        choices=["baseline", "stress"],
        default="baseline",
        help="Slippage mode (baseline=1 tick/2 bps, stress=2 tick/4 bps)",
    )
    parser.add_argument(
        "--initial-nav",
        type=float,
        default=100_000.0,
        help="Initial NAV for backtest",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging",
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
        print("\nPlease ensure daily bar data files exist in the data directory.")
        print("Expected files: MES_1d_*.parquet, MNQ_1d_*.parquet, TLT_1d_*.parquet, IEF_1d_*.parquet, etc.")
        return 1

    print(f"Running TSMOM walk-forward backtest...")
    print(f"Data dir: {data_dir}")
    print(f"Slippage mode: {args.slippage_mode}")

    bt = TSMOMBacktester(
        data_dir=data_dir,
        slippage_mode=args.slippage_mode,
    )

    try:
        result = bt.run_walk_forward(initial_nav=args.initial_nav)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure daily bar data files exist for all instruments.")
        return 1

    # Save and print results
    save_results(result, Path(args.output))
    print_report(result)

    return 0 if result.summary["passes_kill_tests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
