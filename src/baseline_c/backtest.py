"""
Backtest Module for Baseline C

Multi-interval portfolio simulation with position carry-over.

Per spec §7-8:
- Edge scoring: edge = max(|ŷ| - 2c, 0) where c = 10 bps
- Weight: w = sign(ŷ) * edge / sum(edges) * G
- Gross exposure: G = 10% of NAV
- Cost: c = 10 bps one-way, charged at every rebalance
- Position carry-over: positions can be held overnight
- Split boundaries: start flat, force flatten at end

Trading schedule:
- 4 rebalance times: 10:31, 11:31, 12:31, 14:00
- After 14:00, positions held until next day 10:31
"""

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data_loader import (
    SYMBOLS,
    REBALANCE_TIMES,
    FEATURE_MINUTE_IDX,
    get_bar_by_time,
    get_bar_open_price,
    get_rth_day_data,
    get_trading_days,
    is_early_close_day,
    load_rth_data,
)
from .dataset import INTERVAL_DEFS, DatasetGenerator
from .feature_builder import FeatureBuilder, get_prior_close
from .train import load_model


# Cost parameters (per spec §8)
COST_BPS = 10  # 10 bps one-way
COST_DECIMAL = COST_BPS / 10000  # 0.001
GROSS_EXPOSURE = 0.10  # 10% of NAV


@dataclass
class IntervalResult:
    """Single interval result within a day."""
    date: date
    interval_idx: int
    interval_name: str
    rebalance_time: time
    gross_return: float
    cost: float
    net_return: float
    turnover: float
    weights_before: dict
    weights_after: dict


@dataclass
class DailyResult:
    """Single day backtest result aggregated across intervals."""
    date: date
    intervals: list[IntervalResult]
    daily_gross_return: float
    daily_cost: float
    daily_net_return: float
    daily_turnover: float
    n_active_intervals: int  # intervals with non-zero exposure


@dataclass
class BacktestResult:
    """Complete backtest result."""
    split: str
    daily_results: list[DailyResult]

    # Aggregate metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_cost: float = 0.0
    avg_turnover: float = 0.0
    win_rate: float = 0.0
    trade_days: int = 0
    active_days: int = 0  # days with gross_exposure > 0 for at least one interval

    # Per-symbol metrics
    symbol_pnl: dict = field(default_factory=dict)

    def compute_metrics(self, risk_free_rate: float = 0.0):
        """Compute aggregate metrics from daily results."""
        if not self.daily_results:
            return

        net_returns = np.array([d.daily_net_return for d in self.daily_results])
        gross_returns = np.array([d.daily_gross_return for d in self.daily_results])
        costs = np.array([d.daily_cost for d in self.daily_results])
        turnovers = np.array([d.daily_turnover for d in self.daily_results])

        self.trade_days = len(self.daily_results)
        self.total_return = float(np.sum(net_returns))
        self.total_cost = float(np.sum(costs))
        self.avg_turnover = float(np.mean(turnovers))

        # Count active days
        self.active_days = sum(1 for d in self.daily_results if d.n_active_intervals > 0)

        # Annualize (252 trading days)
        if self.trade_days > 0:
            daily_mean = np.mean(net_returns)
            daily_std = np.std(net_returns)

            self.annualized_return = daily_mean * 252
            self.volatility = daily_std * np.sqrt(252)

            if self.volatility > 0:
                self.sharpe_ratio = (self.annualized_return - risk_free_rate) / self.volatility

            # Win rate
            self.win_rate = float(np.mean(net_returns > 0))

            # Max drawdown
            cumulative = np.cumsum(net_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            self.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "split": self.split,
            "trade_days": self.trade_days,
            "active_days": self.active_days,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_cost": self.total_cost,
            "avg_turnover": self.avg_turnover,
            "win_rate": self.win_rate,
            "symbol_pnl": self.symbol_pnl,
        }


class Backtester:
    """
    Multi-interval portfolio backtester for Baseline C.

    Key features:
    - 4 Ridge models (one per interval) make predictions
    - Position carry-over: weights persist between intervals and overnight
    - Costs charged at every rebalance where weights change
    - Start flat at split beginning, force flatten at split end
    """

    def __init__(
        self,
        models: list[Ridge],
        scalers: list[StandardScaler],
        cost_bps: float = COST_BPS,
        gross_exposure: float = GROSS_EXPOSURE,
        symbols: list[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize backtester.

        Args:
            models: List of 4 trained Ridge models (one per interval)
            scalers: List of 4 fitted StandardScalers
            cost_bps: One-way cost in basis points
            gross_exposure: Target gross exposure (fraction of NAV)
            symbols: Symbols to trade (defaults to all 9)
            verbose: Show progress
        """
        self.models = models
        self.scalers = scalers
        self.cost_decimal = cost_bps / 10000
        self.gross_exposure = gross_exposure
        self.symbols = symbols or SYMBOLS
        self.verbose = verbose
        self.feature_builder = FeatureBuilder(symbols=self.symbols)

    def _compute_edge(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute cost-aware edge.

        edge = max(|ŷ| - 2c, 0)
        """
        threshold = 2 * self.cost_decimal  # 20 bps round-trip
        return np.maximum(np.abs(y_pred) - threshold, 0)

    def _compute_target_weights(
        self,
        y_pred: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute target weights from predictions.

        w = sign(ŷ) * edge / sum(edges) * G

        Args:
            y_pred: Dict mapping symbol -> prediction

        Returns:
            Dict mapping symbol -> target weight
        """
        # Compute edges
        edges = {s: max(abs(y_pred[s]) - 2 * self.cost_decimal, 0) for s in y_pred}
        total_edge = sum(edges.values())

        if total_edge == 0:
            return {s: 0.0 for s in self.symbols}

        # Compute target weights
        weights = {}
        for s in self.symbols:
            if s in y_pred:
                sign_y = np.sign(y_pred[s])
                weights[s] = sign_y * edges[s] / total_edge * self.gross_exposure
            else:
                weights[s] = 0.0

        return weights

    def _compute_turnover(
        self,
        w_prev: dict[str, float],
        w_target: dict[str, float],
    ) -> float:
        """Compute turnover (sum of |weight changes|)."""
        turnover = 0.0
        for s in self.symbols:
            turnover += abs(w_target.get(s, 0.0) - w_prev.get(s, 0.0))
        return turnover

    def run(self, split: str) -> BacktestResult:
        """
        Run backtest on a split.

        Args:
            split: One of train, val, dev_test, holdout

        Returns:
            BacktestResult with daily P&L and metrics
        """
        if self.verbose:
            print("=" * 60)
            print(f"Running Baseline C Backtest on {split}")
            print("=" * 60)
            print(f"Cost: {self.cost_decimal * 10000:.1f} bps one-way")
            print(f"Gross exposure: {self.gross_exposure * 100:.1f}%")
            print()

        # Get trading days
        trading_days = get_trading_days(split)

        if self.verbose:
            print(f"Trading days: {len(trading_days)}")

        # Load all RTH data upfront
        rth_data = {}
        for symbol in self.symbols:
            rth_data[symbol] = load_rth_data(symbol, split)

        # Initialize weights (start flat per spec)
        current_weights = {s: 0.0 for s in self.symbols}

        # Track results
        daily_results = []
        symbol_pnl = {s: 0.0 for s in self.symbols}

        for day_idx, d in enumerate(trading_days):
            # Get RTH data for this day for all symbols
            rth_day_data = {}
            for symbol in self.symbols:
                day_df = get_rth_day_data(rth_data[symbol], d)
                if not day_df.empty:
                    rth_day_data[symbol] = day_df

            # Skip early close days entirely
            sample_day_df = list(rth_day_data.values())[0] if rth_day_data else None
            if sample_day_df is None or is_early_close_day(sample_day_df):
                continue

            day_intervals = []
            day_gross = 0.0
            day_cost = 0.0
            day_turnover = 0.0
            n_active = 0

            # Process each rebalance time
            for interval_idx in range(4):
                interval_name, start_time, end_time, is_overnight = INTERVAL_DEFS[interval_idx]

                # Get predictions for this interval
                predictions = {}
                for symbol in self.symbols:
                    if symbol not in rth_day_data:
                        continue

                    # Get prior close for features
                    prior_close = get_prior_close(rth_data[symbol], d)

                    # Build features at the rebalance time
                    features, diag = self.feature_builder.build_features(
                        rth_day_data, symbol, d, start_time, prior_close
                    )

                    if features is None:
                        continue

                    # Scale and predict
                    X = features.reshape(1, -1)
                    X_scaled = self.scalers[interval_idx].transform(X)
                    y_pred = self.models[interval_idx].predict(X_scaled)[0]
                    predictions[symbol] = y_pred

                # Compute target weights
                target_weights = self._compute_target_weights(predictions)

                # Compute turnover and cost
                turnover = self._compute_turnover(current_weights, target_weights)
                cost = turnover * self.cost_decimal

                # Compute return for this interval
                # For intraday intervals: use same day prices
                # For overnight interval: need next day prices
                interval_return = 0.0

                if is_overnight:
                    # Overnight interval: return is from 14:00 today to 10:31 next day
                    # For now, we compute this when we get next day's data
                    # Store current weights and we'll compute return at next day's 10:31
                    pass
                else:
                    # Intraday interval: compute return immediately
                    for symbol in self.symbols:
                        if symbol not in rth_day_data:
                            continue

                        day_df = rth_day_data[symbol]
                        p_start = get_bar_open_price(day_df, start_time)
                        p_end = get_bar_open_price(day_df, end_time)

                        if p_start is None or p_end is None or p_start <= 0:
                            continue

                        symbol_return = np.log(p_end / p_start)
                        weighted_return = target_weights[symbol] * symbol_return
                        interval_return += weighted_return
                        symbol_pnl[symbol] += weighted_return

                # Track if this interval had exposure
                gross_exp = sum(abs(w) for w in target_weights.values())
                if gross_exp > 0:
                    n_active += 1

                # Store interval result
                interval_result = IntervalResult(
                    date=d,
                    interval_idx=interval_idx,
                    interval_name=interval_name,
                    rebalance_time=start_time,
                    gross_return=interval_return,
                    cost=cost,
                    net_return=interval_return - cost,
                    turnover=turnover,
                    weights_before=current_weights.copy(),
                    weights_after=target_weights.copy(),
                )
                day_intervals.append(interval_result)

                # Update running totals
                day_gross += interval_return
                day_cost += cost
                day_turnover += turnover

                # Update current weights for next interval
                current_weights = target_weights.copy()

            # Handle overnight return from previous day
            # This is computed as part of the 14:00 interval but realized at next day 10:31
            # For simplicity, we'll include it in the day's P&L when positions are held overnight
            # The actual overnight return will be computed when we have next day's 10:31 price

            # Force flatten on last day of split
            if day_idx == len(trading_days) - 1:
                # Flatten all positions
                flatten_turnover = sum(abs(w) for w in current_weights.values())
                flatten_cost = flatten_turnover * self.cost_decimal
                day_cost += flatten_cost
                day_turnover += flatten_turnover
                current_weights = {s: 0.0 for s in self.symbols}

            # Store daily result
            daily_result = DailyResult(
                date=d,
                intervals=day_intervals,
                daily_gross_return=day_gross,
                daily_cost=day_cost,
                daily_net_return=day_gross - day_cost,
                daily_turnover=day_turnover,
                n_active_intervals=n_active,
            )
            daily_results.append(daily_result)

        # Create result and compute metrics
        result = BacktestResult(
            split=split,
            daily_results=daily_results,
            symbol_pnl=symbol_pnl,
        )
        result.compute_metrics()

        if self.verbose:
            self._print_results(result)

        return result

    def _print_results(self, result: BacktestResult):
        """Print formatted backtest results."""
        print()
        print("=" * 60)
        print("Backtest Results")
        print("=" * 60)
        print(f"Trading days: {result.trade_days}")
        print(f"Active days: {result.active_days}")
        print(f"Total return: {result.total_return * 100:.4f}%")
        print(f"Annualized return: {result.annualized_return * 100:.2f}%")
        print(f"Volatility: {result.volatility * 100:.2f}%")
        print(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
        print(f"Max drawdown: {result.max_drawdown * 100:.2f}%")
        print(f"Total cost: {result.total_cost * 100:.4f}%")
        print(f"Avg turnover: {result.avg_turnover * 100:.2f}%")
        print(f"Win rate: {result.win_rate * 100:.1f}%")
        print()
        print("P&L by symbol:")
        for sym, pnl in sorted(result.symbol_pnl.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sym}: {pnl * 100:+.4f}%")


def load_and_backtest(
    model_dir: Path,
    split: str = "val",
    verbose: bool = True,
) -> BacktestResult:
    """
    Load a saved model and run backtest.

    Args:
        model_dir: Path to saved model directory
        split: Split to backtest
        verbose: Show progress

    Returns:
        BacktestResult
    """
    models, scalers, config = load_model(model_dir)

    backtester = Backtester(
        models=models,
        scalers=scalers,
        verbose=verbose,
    )

    return backtester.run(split)


if __name__ == "__main__":
    from pathlib import Path

    print("=== Baseline C Backtester Validation ===")

    # Find the latest model
    checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints" / "baseline_c"

    if not checkpoints_dir.exists():
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        print("Please run train.py first to create a model.")
        exit(1)

    model_dirs = sorted(checkpoints_dir.glob("ridge_c_v0_*"))
    if not model_dirs:
        print("No model checkpoints found. Please run train.py first.")
        exit(1)

    model_dir = model_dirs[-1]  # Most recent
    print(f"Loading model from: {model_dir}")

    # Load model
    models, scalers, config = load_model(model_dir)
    print(f"Loaded {len([m for m in models if m is not None])} models")
    print()

    # Run backtest on val split
    backtester = Backtester(
        models=models,
        scalers=scalers,
        verbose=True,
    )

    result = backtester.run("val")

    print()
    print("=== Validation Complete ===")
    print(f"Result dict keys: {list(result.to_dict().keys())}")
