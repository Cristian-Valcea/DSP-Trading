"""
Backtest for Volatility Risk Premium (VRP) Strategy.

Strategy
--------
Harvest the Variance Risk Premium (VRP) by shorting VIX futures in contango:
1. Universe: VIX Futures (VX) front-month
2. Signal: Adjusted contango (VX F1 - VIX spot - interest carry) > threshold
3. Entry filters: Contango min, VIX below 50d MA, VVIX below 90th percentile
4. Exit: Stop-loss (-15%), VIX level triggers (30/40), VVIX spike
5. Monthly rebalance (3rd trading day), 5-day cool-down after stop

Data Requirements
-----------------
- VIX Futures (VX F1): Daily settlement prices from CBOE
- VIX Spot Index: Daily close
- VVIX Index: Daily close (for vol-of-vol filter)
- Fed Funds Rate: Daily (for adjusted contango calculation)

Kill-Test Criteria (from SPEC_VRP.md)
-------------------------------------
Primary Gates (ALL must pass):
- G1: Sharpe Ratio >= 0.50
- G2: Net P&L > $0
- G3: Max Drawdown >= -30%

Stress Gates (ALL must pass):
- S1: Sharpe (2x costs) >= 0.30
- S2: Net P&L (2x costs) > $0

Fold Validation: 2/3 OOS folds must pass
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Strategy Constants (from SPEC_VRP.md)
# ============================================================================
# STRATEGY KILL-TEST STATUS: FAILED (Jan 8, 2026)
#
# All versions tested failed kill criteria (Sharpe < 0.50):
#   V1: Sharpe ~0.01 (entry filters too strict)
#   V2: Sharpe 0.03 (relaxed filters didn't help)
#   V3: Sharpe -0.67 (daily entry caused excessive trading)
#   V4: Sharpe 0.01 (minimal filters, best result)
#   V5: Sharpe -0.18 (no filters, worst result)
#
# Root cause: VIX futures don't reliably decay to spot
# - Monthly roll returns average -2.58%
# - Only 49.7% of months have positive short-VX returns
# - VIX spikes cause losses that dwarf contango gains
#
# CONCLUSION: VRP strategy is NOT VIABLE. Focus on TSMOM instead.

# V4 configuration preserved below (best performing but still fails)
# Signal thresholds
MIN_CONTANGO = 0.0  # No contango requirement
VIX_HIGH_REGIME = 28.0  # Only block above 28
VVIX_LOOKBACK = 126  # 6 months for percentile calculation
VVIX_HARD_LIMIT = 100.0  # Very loose - only block extreme VVIX
VVIX_PERCENTILE = 98  # Very loose percentile filter

# Exit triggers
STOP_LOSS_PCT = -0.30  # Wide -30% stop
VIX_REDUCE_LEVEL = 40.0  # Reduce at VIX 40
VIX_FLATTEN_LEVEL = 55.0  # Flatten at VIX 55 (crisis)

# Position sizing
MAX_NAV_PCT = 0.25  # 25% of NAV
MAX_MARGIN_PCT = 0.40  # 40% margin utilization
CONTRACT_MULTIPLIER = 1000  # $1,000 per VIX point
MARGIN_PER_CONTRACT = 5000  # Approximate margin requirement

# Timing
REBAL_DAY_OF_MONTH = 3  # Monthly entry (3rd trading day)
ROLL_DAYS_BEFORE_EXPIRY = 3  # Roll 3 days before expiry
COOL_DOWN_DAYS = 0  # No cool-down


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def _perf_metrics(equity: pd.Series) -> Dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 3:
        return {
            "return": 0.0,
            "cagr": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
        }

    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    vol = float(rets.std() * sqrt(252)) if rets.std() and rets.std() > 0 else 0.0
    sharpe = float(_safe_div(rets.mean(), rets.std()) * sqrt(252)) if rets.std() and rets.std() > 0 else 0.0

    max_dd = _max_drawdown(equity)
    calmar = float(_safe_div(cagr, abs(max_dd))) if max_dd < 0 else 0.0

    return {
        "return": total_return,
        "cagr": float(cagr),
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(max_dd),
        "calmar": calmar,
    }


@dataclass(frozen=True)
class Trade:
    dt: date
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    intent: str  # ENTRY, EXIT, STOP_LOSS, REDUCE, ROLL

    @property
    def notional(self) -> float:
        return float(self.quantity) * float(self.price) * CONTRACT_MULTIPLIER


@dataclass
class Position:
    """Active VRP position state."""
    entry_date: date
    entry_price: float
    contracts: int
    stop_loss_price: float

    @property
    def notional(self) -> float:
        return self.contracts * self.entry_price * CONTRACT_MULTIPLIER


@dataclass
class BacktestConfig:
    """Configuration for VRP backtest."""
    start_date: str = "2014-01-01"
    end_date: str = "2025-12-31"
    initial_nav: float = 100_000.0

    # Transaction costs
    futures_commission: float = 2.50  # Per contract per side
    futures_slippage_ticks: int = 1  # 1 tick = 0.05 points = $50

    # Stress test multiplier
    cost_multiplier: float = 1.0  # 2.0 for stress test


@dataclass
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    monthly_signals: pd.DataFrame
    position_days: int
    flat_days: int


class VRPBacktester:
    """
    VRP strategy backtester using real VIX futures data.

    Data sources:
    - VX F1: Front-month VIX futures settlement (CBOE)
    - VIX spot: VIX index close (Yahoo)
    - VVIX: Vol-of-Vol index (Yahoo)
    - Fed Funds: Risk-free rate (FRED)
    """

    def __init__(
        self,
        vx_f1: pd.DataFrame,
        vix_spot: pd.DataFrame,
        vvix: pd.DataFrame,
        fed_funds: pd.DataFrame,
    ):
        """
        Initialize with pre-loaded data.

        Args:
            vx_f1: DataFrame with 'vx_f1', 'f1_expiration' columns, date index
            vix_spot: DataFrame with 'vix_spot' column, date index
            vvix: DataFrame with 'vvix' column, date index
            fed_funds: DataFrame with 'fed_funds' column, date index
        """
        self.vx_f1 = vx_f1.copy()
        self.vix_spot = vix_spot.copy()
        self.vvix = vvix.copy()
        self.fed_funds = fed_funds.copy()

        # Merge all data into single dataframe
        self.data = self._build_merged_data()

    def _build_merged_data(self) -> pd.DataFrame:
        """Merge all data sources into single aligned dataframe."""
        # Start with VX F1 (our primary series)
        df = self.vx_f1[['vx_f1', 'f1_expiration']].copy()

        # Join VIX spot
        df = df.join(self.vix_spot[['vix_spot']], how='left')

        # Join VVIX
        df = df.join(self.vvix[['vvix']], how='left')

        # Join Fed Funds (forward-fill for weekends)
        ff = self.fed_funds[['fed_funds']].copy()
        ff = ff.reindex(df.index, method='ffill')
        df['fed_funds'] = ff['fed_funds']

        # Calculate derived features
        # Days to expiry
        df['days_to_expiry'] = (
            pd.to_datetime(df['f1_expiration']) - df.index
        ).dt.days

        # Raw contango
        df['raw_contango'] = df['vx_f1'] - df['vix_spot']

        # Adjusted contango (remove interest component)
        # Interest component = VIX * (rate / 100) * (days / 365)
        df['interest_component'] = (
            df['vix_spot'] * (df['fed_funds'] / 100) * (df['days_to_expiry'] / 365)
        )
        df['adjusted_contango'] = df['raw_contango'] - df['interest_component']

        # VIX 50-day MA
        df['vix_50d_ma'] = df['vix_spot'].rolling(50, min_periods=30).mean()

        # VVIX rolling percentile (126-day lookback)
        df['vvix_90th'] = df['vvix'].rolling(VVIX_LOOKBACK, min_periods=60).quantile(0.90)

        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['vx_f1', 'vix_spot', 'adjusted_contango', 'vix_50d_ma'])

        return df

    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Run the VRP backtest.

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        start = pd.Timestamp(_parse_date(config.start_date))
        end = pd.Timestamp(_parse_date(config.end_date))

        # Filter data to date range
        data = self.data[(self.data.index >= start) & (self.data.index <= end)].copy()

        if len(data) < 60:
            raise ValueError("Not enough data for backtest (need at least 60 days)")

        # State
        nav = config.initial_nav
        cash = nav  # Start fully in cash
        position: Optional[Position] = None
        equity_points: List[Tuple[pd.Timestamp, float]] = []
        trades: List[Trade] = []
        monthly_signals: List[Dict] = []

        # Cool-down state
        cool_down_until: Optional[pd.Timestamp] = None

        # Track days
        position_days = 0
        flat_days = 0

        # Count trading days of month
        last_month: Optional[Tuple[int, int]] = None
        trading_day_of_month = 0

        for dt, row in data.iterrows():
            # Track trading day of month
            if last_month is None or (dt.year, dt.month) != last_month:
                trading_day_of_month = 1
                last_month = (dt.year, dt.month)
            else:
                trading_day_of_month += 1

            # Get market data
            vx_price = row['vx_f1']
            vix_spot = row['vix_spot']
            vvix = row['vvix'] if pd.notna(row['vvix']) else 90.0  # Default if missing
            vvix_90th = row['vvix_90th'] if pd.notna(row['vvix_90th']) else 110.0
            adjusted_contango = row['adjusted_contango']
            vix_50d_ma = row['vix_50d_ma']
            days_to_expiry = row['days_to_expiry']

            # Check cool-down expiry
            cool_down_active = cool_down_until is not None and dt <= cool_down_until

            # Daily P&L if in position
            daily_pnl = 0.0

            if position is not None:
                # Calculate unrealized P&L (short position: profit when price falls)
                current_value = position.contracts * vx_price * CONTRACT_MULTIPLIER
                entry_value = position.contracts * position.entry_price * CONTRACT_MULTIPLIER
                unrealized_pnl = entry_value - current_value  # Short: entry - current
                pnl_pct = unrealized_pnl / entry_value

                # === EXIT CHECKS ===
                exit_reason = None
                reduction_pct = 0.0

                # Stop-loss check
                if pnl_pct <= STOP_LOSS_PCT:
                    exit_reason = "STOP_LOSS"
                    reduction_pct = 1.0
                    # Trigger cool-down
                    cool_down_until = dt + pd.Timedelta(days=COOL_DOWN_DAYS * 2)  # ~5 trading days

                # VIX level triggers
                elif vix_spot >= VIX_FLATTEN_LEVEL:
                    exit_reason = "VIX_FLATTEN"
                    reduction_pct = 1.0

                elif vix_spot >= VIX_REDUCE_LEVEL:
                    exit_reason = "VIX_REDUCE"
                    reduction_pct = 0.5

                # VVIX spike
                elif vvix > vvix_90th:
                    exit_reason = "VVIX_SPIKE"
                    reduction_pct = 0.5

                # Roll check (5 days before expiry)
                elif days_to_expiry <= ROLL_DAYS_BEFORE_EXPIRY:
                    exit_reason = "ROLL"
                    reduction_pct = 1.0  # Close and re-enter next month

                if exit_reason and reduction_pct > 0:
                    exit_contracts = int(position.contracts * reduction_pct)
                    if exit_contracts > 0:
                        # Calculate exit cost
                        slippage = config.futures_slippage_ticks * 0.05 * config.cost_multiplier
                        exit_price = vx_price + slippage  # Buying back short
                        commission = config.futures_commission * exit_contracts * config.cost_multiplier

                        # Realize P&L for short position:
                        # P&L = (entry_price - exit_price) × contracts × multiplier
                        # Positive when price falls (we sold high, bought back low)
                        # Negative when price rises (we sold low, bought back high)
                        realized_pnl = (position.entry_price - exit_price) * exit_contracts * CONTRACT_MULTIPLIER - commission

                        # Release the margin that was held + add P&L
                        margin_released = exit_contracts * MARGIN_PER_CONTRACT
                        cash += margin_released + realized_pnl

                        trades.append(Trade(
                            dt=dt.date(),
                            symbol="VX_F1",
                            side="BUY",  # Close short
                            quantity=exit_contracts,
                            price=exit_price,
                            commission=commission,
                            intent=exit_reason,
                        ))

                        if reduction_pct >= 1.0:
                            position = None
                        else:
                            position = Position(
                                entry_date=position.entry_date,
                                entry_price=position.entry_price,
                                contracts=position.contracts - exit_contracts,
                                stop_loss_price=position.stop_loss_price,
                            )

                position_days += 1
            else:
                flat_days += 1

            # === ENTRY CHECK (daily if REBAL_DAY_OF_MONTH=0, else monthly) ===
            can_check_entry = (
                REBAL_DAY_OF_MONTH == 0 or  # Daily entry mode
                trading_day_of_month == REBAL_DAY_OF_MONTH  # Monthly mode
            )
            if position is None and can_check_entry:
                can_enter = True
                block_reason = None

                # V3: Simplified filters - VVIX < 90 is the primary edge

                # Filter 1: VVIX below hard limit (primary filter - best edge found)
                if vvix >= VVIX_HARD_LIMIT:
                    can_enter = False
                    block_reason = f"VVIX {vvix:.1f} >= {VVIX_HARD_LIMIT} (vol-of-vol too high)"

                # Filter 2: VIX not in high regime (safety)
                elif vix_spot > VIX_HIGH_REGIME:
                    can_enter = False
                    block_reason = f"VIX {vix_spot:.1f} > {VIX_HIGH_REGIME} (high regime)"

                # Filter 3: Cool-down not active
                elif cool_down_active:
                    can_enter = False
                    block_reason = "Cool-down active"

                # Filter 4: Minimum contango (very loose)
                elif adjusted_contango < MIN_CONTANGO:
                    can_enter = False
                    block_reason = f"Contango {adjusted_contango:.2f} < {MIN_CONTANGO}"

                monthly_signals.append({
                    "date": dt.date(),
                    "vx_f1": vx_price,
                    "vix_spot": vix_spot,
                    "adjusted_contango": adjusted_contango,
                    "vix_50d_ma": vix_50d_ma,
                    "vvix": vvix,
                    "vvix_90th": vvix_90th,
                    "can_enter": can_enter,
                    "block_reason": block_reason,
                })

                if can_enter:
                    # Calculate position size
                    current_nav = cash  # All cash when flat
                    max_notional = current_nav * MAX_NAV_PCT
                    max_margin = current_nav * MAX_MARGIN_PCT

                    notional_per_contract = vx_price * CONTRACT_MULTIPLIER
                    contracts_by_nav = int(max_notional / notional_per_contract)
                    contracts_by_margin = int(max_margin / MARGIN_PER_CONTRACT)

                    num_contracts = min(contracts_by_nav, contracts_by_margin)

                    # Skip entry if position would be 0 (NAV too small)
                    if num_contracts <= 0:
                        can_enter = False
                        block_reason = f"NAV too small for 1 contract (need ${notional_per_contract:.0f} but max is ${max_notional:.0f})"
                        monthly_signals[-1]["can_enter"] = False
                        monthly_signals[-1]["block_reason"] = block_reason
                        continue

                    # Calculate entry cost
                    slippage = config.futures_slippage_ticks * 0.05 * config.cost_multiplier
                    entry_price = vx_price - slippage  # Selling short
                    commission = config.futures_commission * num_contracts * config.cost_multiplier

                    # Set aside margin
                    margin_required = num_contracts * MARGIN_PER_CONTRACT
                    cash -= margin_required + commission

                    # Calculate stop-loss price (15% loss on short = 15% price increase)
                    stop_loss_price = entry_price * (1 + abs(STOP_LOSS_PCT))

                    position = Position(
                        entry_date=dt.date(),
                        entry_price=entry_price,
                        contracts=num_contracts,
                        stop_loss_price=stop_loss_price,
                    )

                    trades.append(Trade(
                        dt=dt.date(),
                        symbol="VX_F1",
                        side="SELL",  # Short
                        quantity=num_contracts,
                        price=entry_price,
                        commission=commission,
                        intent="ENTRY",
                    ))

            # Calculate end-of-day NAV
            if position is not None:
                # Cash + margin + unrealized P&L
                margin_held = position.contracts * MARGIN_PER_CONTRACT
                current_value = position.contracts * vx_price * CONTRACT_MULTIPLIER
                entry_value = position.contracts * position.entry_price * CONTRACT_MULTIPLIER
                unrealized_pnl = entry_value - current_value
                nav = cash + margin_held + unrealized_pnl
            else:
                nav = cash

            equity_points.append((dt, nav))

        # Build results
        equity = pd.Series(
            data=[v for _, v in equity_points],
            index=pd.DatetimeIndex([d for d, _ in equity_points]),
            name="equity",
        )
        daily_returns = equity.pct_change().dropna()

        trades_df = pd.DataFrame([
            {
                "date": t.dt,
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "notional": t.notional,
                "commission": t.commission,
                "intent": t.intent,
            }
            for t in trades
        ])

        signals_df = pd.DataFrame(monthly_signals)

        metrics = _perf_metrics(equity)
        metrics["position_days"] = position_days
        metrics["flat_days"] = flat_days
        metrics["position_pct"] = position_days / (position_days + flat_days) if (position_days + flat_days) > 0 else 0.0
        metrics["num_trades"] = len(trades)

        return BacktestResult(
            equity=equity,
            daily_returns=daily_returns,
            trades=trades_df,
            metrics=metrics,
            monthly_signals=signals_df,
            position_days=position_days,
            flat_days=flat_days,
        )


def load_vrp_data(data_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all VRP data from parquet files."""
    data_dir = Path(data_dir)
    vx_f1 = pd.read_parquet(data_dir / "futures" / "VX_F1_CBOE.parquet")
    vix_spot = pd.read_parquet(data_dir / "indices" / "VIX_spot.parquet")
    vvix = pd.read_parquet(data_dir / "indices" / "VVIX.parquet")
    fed_funds = pd.read_parquet(data_dir / "rates" / "fed_funds.parquet")

    return vx_f1, vix_spot, vvix, fed_funds


def run_kill_test(
    backtester: VRPBacktester,
    initial_nav: float = 100_000.0,
) -> Dict:
    """
    Run complete kill-test with expanding window OOS validation.

    Folds:
        - Fold 1: Test 2023 (2014-2022 history)
        - Fold 2: Test 2024 (2014-2023 history)
        - Fold 3: Test 2025 (2014-2024 history, partial year)
    """
    folds = [
        {"name": "Fold 1 (OOS: 2023)", "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"name": "Fold 2 (OOS: 2024)", "test_start": "2024-01-01", "test_end": "2024-12-31"},
        {"name": "Fold 3 (OOS: 2025)", "test_start": "2025-01-01", "test_end": "2025-12-31"},
    ]

    results = {
        "folds": [],
        "overall_pass": False,
        "baseline_full": None,
        "stress_full": None,
    }

    # Run full-period baseline first
    baseline_config = BacktestConfig(
        start_date="2014-01-01",
        end_date="2025-12-31",
        initial_nav=initial_nav,
        cost_multiplier=1.0,
    )
    baseline_full = backtester.run(baseline_config)
    results["baseline_full"] = baseline_full.metrics

    # Run full-period stress test
    stress_config = BacktestConfig(
        start_date="2014-01-01",
        end_date="2025-12-31",
        initial_nav=initial_nav,
        cost_multiplier=2.0,
    )
    stress_full = backtester.run(stress_config)
    results["stress_full"] = stress_full.metrics

    # Run each fold
    for fold in folds:
        try:
            # Baseline
            baseline_cfg = BacktestConfig(
                start_date=fold["test_start"],
                end_date=fold["test_end"],
                initial_nav=initial_nav,
                cost_multiplier=1.0,
            )
            baseline = backtester.run(baseline_cfg)

            # Stress (2x costs)
            stress_cfg = BacktestConfig(
                start_date=fold["test_start"],
                end_date=fold["test_end"],
                initial_nav=initial_nav,
                cost_multiplier=2.0,
            )
            stress = backtester.run(stress_cfg)

            # Check gates
            baseline_pass = (
                baseline.metrics["sharpe"] >= 0.50 and
                baseline.metrics["return"] > 0 and  # Net P&L > 0
                baseline.metrics["max_dd"] >= -0.30
            )

            stress_pass = (
                stress.metrics["sharpe"] >= 0.30 and
                stress.metrics["return"] > 0
            )

            fold_pass = baseline_pass and stress_pass

            results["folds"].append({
                "name": fold["name"],
                "baseline": baseline.metrics,
                "stress": stress.metrics,
                "baseline_pass": baseline_pass,
                "stress_pass": stress_pass,
                "fold_pass": fold_pass,
            })
        except Exception as e:
            logger.warning(f"Fold {fold['name']} failed: {e}")
            results["folds"].append({
                "name": fold["name"],
                "error": str(e),
                "fold_pass": False,
            })

    # Overall: 2/3 folds must pass
    passed_folds = sum(1 for f in results["folds"] if f.get("fold_pass", False))
    results["overall_pass"] = passed_folds >= 2
    results["passed_folds"] = passed_folds
    results["total_folds"] = len(folds)

    return results


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _format_float(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def _print_results(result: BacktestResult, title: str = "VRP BACKTEST RESULTS") -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\n--- Performance Metrics ---")
    print(f"Total Return: {_format_pct(result.metrics['return'])}")
    print(f"CAGR: {_format_pct(result.metrics['cagr'])}")
    print(f"Volatility: {_format_pct(result.metrics['vol'])}")
    print(f"Sharpe Ratio: {_format_float(result.metrics['sharpe'])}")
    print(f"Max Drawdown: {_format_pct(result.metrics['max_dd'])}")
    print(f"Calmar Ratio: {_format_float(result.metrics['calmar'])}")

    print("\n--- Position Statistics ---")
    print(f"Position Days: {result.position_days}")
    print(f"Flat Days: {result.flat_days}")
    print(f"Time in Position: {_format_pct(result.metrics.get('position_pct', 0))}")
    print(f"Number of Trades: {result.metrics.get('num_trades', 0)}")

    if len(result.trades) > 0:
        print("\n--- Trade Summary ---")
        trades_by_intent = result.trades.groupby('intent').agg({
            'quantity': 'sum',
            'notional': 'sum',
            'commission': 'sum',
        })
        print(trades_by_intent.to_string())


def _print_kill_verdict(results: Dict) -> None:
    """Print kill-test verdict."""
    print("\n" + "=" * 60)
    print("VRP KILL-TEST RESULTS")
    print("=" * 60)

    # Full-period results
    print("\n--- Full Period (2014-2025) ---")
    baseline = results["baseline_full"]
    stress = results["stress_full"]
    print(f"Baseline Sharpe: {_format_float(baseline['sharpe'])} (target >= 0.50)")
    print(f"Baseline Return: {_format_pct(baseline['return'])} (target > 0)")
    print(f"Baseline Max DD: {_format_pct(baseline['max_dd'])} (target >= -30%)")
    print(f"Stress Sharpe: {_format_float(stress['sharpe'])} (target >= 0.30)")
    print(f"Stress Return: {_format_pct(stress['return'])} (target > 0)")

    # Fold results
    print("\n--- OOS Fold Results ---")
    for fold in results["folds"]:
        if "error" in fold:
            print(f"\n{fold['name']}: ERROR - {fold['error']}")
            continue

        status = "PASS" if fold["fold_pass"] else "FAIL"
        print(f"\n{fold['name']}: {status}")
        print(f"  Baseline: Sharpe={_format_float(fold['baseline']['sharpe'])}, "
              f"Return={_format_pct(fold['baseline']['return'])}, "
              f"MaxDD={_format_pct(fold['baseline']['max_dd'])}")
        print(f"  Stress:   Sharpe={_format_float(fold['stress']['sharpe'])}, "
              f"Return={_format_pct(fold['stress']['return'])}")
        print(f"  Gates: Baseline={fold['baseline_pass']}, Stress={fold['stress_pass']}")

    # Overall verdict
    print("\n" + "-" * 60)
    passed = results["passed_folds"]
    total = results["total_folds"]
    print(f"Folds Passed: {passed}/{total}")

    if results["overall_pass"]:
        print("\n KILL-TEST VERDICT: PASS")
        print("Strategy passes minimum requirements. Proceed to paper trading.")
    else:
        print("\n KILL-TEST VERDICT: FAIL")
        print("Strategy fails kill criteria. DO NOT TRADE.")


def main() -> int:
    """Main entry point for VRP backtest."""
    parser = argparse.ArgumentParser(description="Backtest VRP Strategy")
    parser.add_argument(
        "--data-dir",
        default="/Users/Shared/wsl-export/wsl-home/dsp100k/data/vrp",
        help="Path to VRP data directory",
    )
    parser.add_argument("--start", default="2014-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--initial-nav", type=float, default=100_000.0, help="Initial NAV")
    parser.add_argument("--cost-multiplier", type=float, default=1.0, help="Cost multiplier (2.0 for stress)")
    parser.add_argument("--kill-test", action="store_true", help="Run full kill-test validation")
    parser.add_argument("--output", default=None, help="Write equity to parquet")
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load data
    data_dir = Path(args.data_dir)
    print(f"Loading VRP data from {data_dir}...")
    vx_f1, vix_spot, vvix, fed_funds = load_vrp_data(data_dir)

    print(f"VX F1: {len(vx_f1)} rows ({vx_f1.index.min().date()} to {vx_f1.index.max().date()})")
    print(f"VIX Spot: {len(vix_spot)} rows")
    print(f"VVIX: {len(vvix)} rows")
    print(f"Fed Funds: {len(fed_funds)} rows")

    # Create backtester
    backtester = VRPBacktester(vx_f1, vix_spot, vvix, fed_funds)

    if args.kill_test:
        # Run full kill-test
        results = run_kill_test(backtester, initial_nav=args.initial_nav)
        _print_kill_verdict(results)
        return 0 if results["overall_pass"] else 1

    # Run single backtest
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_nav=args.initial_nav,
        cost_multiplier=args.cost_multiplier,
    )

    result = backtester.run(config)
    _print_results(result)

    # Print kill-test gates for single run
    print("\n--- Kill-Test Gates ---")
    sharpe = result.metrics["sharpe"]
    net_pnl = result.metrics["return"]
    max_dd = result.metrics["max_dd"]

    g1 = sharpe >= 0.50
    g2 = net_pnl > 0
    g3 = max_dd >= -0.30

    print(f"G1 Sharpe >= 0.50: {_format_float(sharpe)} {'PASS' if g1 else 'FAIL'}")
    print(f"G2 Net P&L > 0: {_format_pct(net_pnl)} {'PASS' if g2 else 'FAIL'}")
    print(f"G3 Max DD >= -30%: {_format_pct(max_dd)} {'PASS' if g3 else 'FAIL'}")

    if g1 and g2 and g3:
        print("\n BASELINE GATES: ALL PASS")
    else:
        print("\n BASELINE GATES: FAIL")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.equity.to_frame().to_parquet(out_path)
        print(f"\nWrote equity to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
