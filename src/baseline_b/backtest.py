"""
Backtest Module for Baseline B

Cost-aware portfolio simulation with conviction-sized gross exposure.

v1 (conviction-sized gross):
- Long-only: score_s = max(ŷ_s − T_long, 0) where T_long = m_long * 2c
- Raw weights: w_raw_s = k * score_s (k is tunable scale factor)
- Per-name cap: w_s = min(w_cap, w_raw_s)
- Dynamic gross: if sum(w_s) > G, scale down; otherwise allow gross < G

Previous (v0):
- Weight: w(d,s) = score / sum(|score|) * G (always normalized to full G)

Per spec §8-9:
- Gross exposure cap: G = 10% of NAV
- Cost: c = 10 bps one-way (20 bps round-trip)
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data_loader import SYMBOLS
from .dataset import DatasetGenerator, DatasetStats
from .train import load_model


# Cost parameters (per spec §8)
COST_BPS = 10  # 10 bps one-way
COST_DECIMAL = COST_BPS / 10000  # 0.001
GROSS_EXPOSURE = 0.10  # 10% of NAV


@dataclass
class DailyResult:
    """Single day backtest result."""
    date: date
    gross_return: float  # Sum of weighted returns before costs
    cost: float  # Transaction cost (turnover * cost rate)
    net_return: float  # gross_return - cost
    turnover: float  # Sum of |weight changes|
    n_positions: int  # Number of positions taken
    long_exposure: float  # Sum of positive weights
    short_exposure: float  # Sum of negative weights (as positive)
    gross_exposure: float  # long + short


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

    # Activity metrics (v1)
    total_days: int = 0  # Total days in dataset (including no-trade days)
    active_days: int = 0  # Days with at least one position
    activity_pct: float = 0.0  # active_days / total_days
    avg_gross_exposure: float = 0.0  # Average gross exposure on active days

    # Per-symbol metrics
    symbol_pnl: dict = field(default_factory=dict)

    def compute_metrics(self, risk_free_rate: float = 0.0, total_days: int = None):
        """Compute aggregate metrics from daily results.

        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation
            total_days: Total days in dataset (for activity calculation)
        """
        if not self.daily_results:
            return

        net_returns = np.array([d.net_return for d in self.daily_results])
        gross_returns = np.array([d.gross_return for d in self.daily_results])
        costs = np.array([d.cost for d in self.daily_results])
        turnovers = np.array([d.turnover for d in self.daily_results])
        gross_exposures = np.array([d.gross_exposure for d in self.daily_results])

        self.trade_days = len(self.daily_results)
        self.total_return = float(np.sum(net_returns))
        self.total_cost = float(np.sum(costs))
        self.avg_turnover = float(np.mean(turnovers))

        # Activity metrics (v1)
        self.active_days = sum(1 for d in self.daily_results if d.n_positions > 0)
        if total_days is not None and total_days > 0:
            self.total_days = total_days
            self.activity_pct = self.active_days / total_days
        else:
            self.total_days = self.trade_days
            self.activity_pct = self.active_days / self.trade_days if self.trade_days > 0 else 0.0

        # Average gross exposure on active days
        active_exposures = [d.gross_exposure for d in self.daily_results if d.n_positions > 0]
        self.avg_gross_exposure = float(np.mean(active_exposures)) if active_exposures else 0.0

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
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_cost": self.total_cost,
            "avg_turnover": self.avg_turnover,
            "win_rate": self.win_rate,
            # Activity metrics (v1)
            "total_days": self.total_days,
            "active_days": self.active_days,
            "activity_pct": self.activity_pct,
            "avg_gross_exposure": self.avg_gross_exposure,
            # Per-symbol
            "symbol_pnl": self.symbol_pnl,
        }


class Backtester:
    """
    Cost-aware portfolio backtester for Baseline B.

    v1 (conviction-sized gross):
    1. Compute score_s = max(ŷ_s − T_long, 0) for long-only
    2. Raw weights: w_raw_s = k * score_s
    3. Cap: w_s = min(w_cap, w_raw_s)
    4. If sum(w_s) > G: scale down to G; otherwise allow gross < G

    v0 (proportional weighting):
    1. Compute signed, thresholded score for each (day, symbol)
    2. Compute weights: w = score / sum(|score|) * G (always full G)
    """

    def __init__(
        self,
        model: Ridge,
        scaler: StandardScaler,
        cost_bps: float = COST_BPS,
        gross_exposure: float = GROSS_EXPOSURE,
        direction: str = "long_only",  # v1 default: long_only
        threshold_mult_long: float = 1.0,
        threshold_mult_short: float = 1.0,
        # v1 conviction-sized gross parameters
        weight_scale_k: float = 1.0,  # k: scale factor for raw weights
        weight_cap: float = 0.03,  # w_cap: per-name cap (3% default)
        sizing_mode: str = "v1",  # "v0" for proportional, "v1" for conviction-sized
        symbols: list[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize backtester.

        Args:
            model: Trained Ridge model
            scaler: Fitted StandardScaler
            cost_bps: One-way cost in basis points
            gross_exposure: Max gross exposure cap (fraction of NAV)
            direction: One of {long_short, long_only, net_long_bias}
            threshold_mult_long: Multiplier on the base trade threshold for longs
            threshold_mult_short: Multiplier on the base trade threshold for shorts
            weight_scale_k: v1 - Scale factor for raw weights (w_raw = k * score)
            weight_cap: v1 - Per-name weight cap (e.g., 0.03 = 3%)
            sizing_mode: "v0" for proportional (always full G), "v1" for conviction-sized
            symbols: Symbols to trade (defaults to all 9)
            verbose: Show progress
        """
        self.model = model
        self.scaler = scaler
        self.cost_decimal = cost_bps / 10000
        self.gross_exposure = gross_exposure
        self.direction = direction
        self.threshold_mult_long = threshold_mult_long
        self.threshold_mult_short = threshold_mult_short
        self.weight_scale_k = weight_scale_k
        self.weight_cap = weight_cap
        self.sizing_mode = sizing_mode
        self.symbols = symbols or SYMBOLS
        self.verbose = verbose

    def _compute_scores(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute signed, cost-aware scores used for proportional weights.

        Base threshold is round-trip cost: 2c.
        - long threshold:  (2c) * threshold_mult_long
        - short threshold: (2c) * threshold_mult_short

        direction modes:
        - long_short: symmetric long/short (default baseline behavior)
        - long_only: only take longs; shorts are disabled
        - net_long_bias: longs use long threshold; shorts require the stricter short threshold

        Returns:
            Signed scores (N,) where 0 means "do not trade this symbol today".
        """
        base_threshold = 2 * self.cost_decimal
        long_th = base_threshold * self.threshold_mult_long
        short_th = base_threshold * self.threshold_mult_short

        scores = np.zeros_like(y_pred)

        # Long side
        long_mask = y_pred > long_th
        scores[long_mask] = y_pred[long_mask] - long_th

        if self.direction == "long_only":
            return scores

        # Short side
        if self.direction in {"long_short", "net_long_bias"}:
            short_mask = y_pred < -short_th
            scores[short_mask] = y_pred[short_mask] + short_th  # negative score
            return scores

        raise ValueError(
            f"Unknown direction mode: {self.direction}. Expected one of: long_short, long_only, net_long_bias"
        )

    def _compute_weights(
        self,
        y_pred: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute portfolio weights.

        v1 (conviction-sized gross):
            1. Raw weights: w_raw_s = k * score_s
            2. Cap: w_s = min(w_cap, w_raw_s)
            3. If sum(w_s) > G: scale down; otherwise allow gross < G

        v0 (proportional):
            w(d,s) = score / sum(|score|) * G (always normalized to full G)

        If sum(|score|) == 0, return zero weights.

        Args:
            y_pred: Predictions (N,)
            scores: Signed scores (N,)

        Returns:
            Weights (N,)
        """
        if np.sum(np.abs(scores)) == 0:
            return np.zeros_like(y_pred)

        if self.sizing_mode == "v0":
            # Original proportional weighting - always normalizes to full G
            total_abs_score = np.sum(np.abs(scores))
            weights = scores / total_abs_score * self.gross_exposure
            return weights

        # v1: Conviction-sized gross
        # Step 1: Raw weights = k * score
        weights_raw = self.weight_scale_k * scores

        # Step 2: Cap per-name weight at w_cap
        # For longs (positive scores), cap at +w_cap
        # For shorts (negative scores), cap at -w_cap
        weights = np.clip(weights_raw, -self.weight_cap, self.weight_cap)

        # Step 3: If sum(|w|) > G, scale down proportionally
        total_gross = np.sum(np.abs(weights))
        if total_gross > self.gross_exposure:
            # Scale down all weights to hit exactly G
            scale_factor = self.gross_exposure / total_gross
            weights = weights * scale_factor

        # Otherwise leave as-is (gross can be < G on low-conviction days)
        return weights

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
            print(f"Running Backtest on {split} (sizing_mode={self.sizing_mode})")
            print("=" * 60)
            print(f"Cost: {self.cost_decimal * 10000:.1f} bps one-way")
            print(f"Gross exposure cap: {self.gross_exposure * 100:.1f}%")
            print(f"Direction: {self.direction}")
            print(f"Threshold mult (long): {self.threshold_mult_long}")
            if self.sizing_mode == "v1":
                print(f"Weight scale k: {self.weight_scale_k}")
                print(f"Weight cap: {self.weight_cap * 100:.1f}%")
            print()

        # Generate dataset
        generator = DatasetGenerator(symbols=self.symbols, verbose=self.verbose)
        X, y_true, metadata, stats = generator.generate(split)

        if len(X) == 0:
            if self.verbose:
                print("No samples to backtest!")
            return BacktestResult(split=split, daily_results=[])

        # Scale features and predict
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        # Add predictions to metadata
        metadata = metadata.copy()
        metadata["y_pred"] = y_pred
        metadata["y_true"] = y_true

        # Compute scores used for weights
        scores = self._compute_scores(y_pred)
        metadata["score"] = scores

        # Group by date for daily processing
        daily_results = []
        symbol_pnl = {s: 0.0 for s in self.symbols}

        for d, day_group in metadata.groupby("date"):
            day_pred = day_group["y_pred"].values
            day_true = day_group["y_true"].values
            day_scores = day_group["score"].values
            day_symbols = day_group["symbol"].values

            # Compute weights for this day
            day_weights = self._compute_weights(day_pred, day_scores)

            # Build weight dict for this day
            curr_weights = {s: 0.0 for s in self.symbols}
            for i, sym in enumerate(day_symbols):
                curr_weights[sym] = day_weights[i]

            # Compute turnover for INTRADAY strategy with mandatory flatten at 14:00
            # Each day: start at 0 → build position at 10:31 → flatten at 14:00 → end at 0
            # Entry turnover = sum(|weights|) at 10:31
            # Exit turnover = sum(|weights|) at 14:00 (same as entry since we flatten)
            # Total turnover = 2 * sum(|weights|) = round-trip
            entry_turnover = sum(abs(w) for w in curr_weights.values())
            exit_turnover = entry_turnover  # Flatten to zero = same magnitude
            turnover = entry_turnover + exit_turnover  # Full round-trip

            # Compute gross return (weighted sum of actual returns)
            gross_return = 0.0
            for i, sym in enumerate(day_symbols):
                weighted_return = day_weights[i] * day_true[i]
                gross_return += weighted_return
                symbol_pnl[sym] += weighted_return

            # Compute cost (one-way cost on each leg of turnover)
            # turnover already includes both entry + exit, so cost = turnover * cost_decimal
            cost = turnover * self.cost_decimal

            # Net return
            net_return = gross_return - cost

            # Compute exposures
            long_exposure = sum(w for w in curr_weights.values() if w > 0)
            short_exposure = sum(-w for w in curr_weights.values() if w < 0)
            gross_exp = long_exposure + short_exposure
            n_positions = sum(1 for w in curr_weights.values() if w != 0)

            daily_results.append(DailyResult(
                date=d,
                gross_return=gross_return,
                cost=cost,
                net_return=net_return,
                turnover=turnover,
                n_positions=n_positions,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                gross_exposure=gross_exp,
            ))

        # Get total unique days in dataset for activity calculation
        total_days = metadata["date"].nunique()

        # Create result and compute metrics
        result = BacktestResult(
            split=split,
            daily_results=daily_results,
            symbol_pnl=symbol_pnl,
        )
        result.compute_metrics(total_days=total_days)

        if self.verbose:
            print()
            print("=" * 60)
            print("Backtest Results")
            print("=" * 60)
            print(f"Trading days: {result.trade_days}")
            print(f"Total return: {result.total_return * 100:.4f}%")
            print(f"Annualized return: {result.annualized_return * 100:.2f}%")
            print(f"Volatility: {result.volatility * 100:.2f}%")
            print(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
            print(f"Max drawdown: {result.max_drawdown * 100:.2f}%")
            print(f"Total cost: {result.total_cost * 100:.4f}%")
            print(f"Avg turnover: {result.avg_turnover * 100:.2f}%")
            print(f"Win rate: {result.win_rate * 100:.1f}%")
            print()
            # Activity metrics (v1)
            print("Activity metrics:")
            print(f"  Total days: {result.total_days}")
            print(f"  Active days: {result.active_days}")
            print(f"  Activity %: {result.activity_pct * 100:.1f}%")
            print(f"  Avg gross exposure (active): {result.avg_gross_exposure * 100:.2f}%")
            print()
            print("P&L by symbol:")
            for sym, pnl in sorted(result.symbol_pnl.items(), key=lambda x: x[1], reverse=True):
                print(f"  {sym}: {pnl * 100:+.4f}%")

        return result

    def run_walk_forward(
        self,
        splits: list[str] = None,
    ) -> dict[str, BacktestResult]:
        """
        Run backtest on multiple splits sequentially.

        Args:
            splits: List of splits to backtest (defaults to val, dev_test)

        Returns:
            Dict mapping split -> BacktestResult
        """
        if splits is None:
            splits = ["val", "dev_test"]

        results = {}
        for split in splits:
            results[split] = self.run(split)

        return results


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
    model, scaler, config = load_model(model_dir)

    backtester = Backtester(
        model=model,
        scaler=scaler,
        verbose=verbose,
    )

    return backtester.run(split)


if __name__ == "__main__":
    from pathlib import Path

    print("=== Backtester Validation ===")

    # Find the latest model
    checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints" / "baseline_b"

    if not checkpoints_dir.exists():
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        print("Please run train.py first to create a model.")
        exit(1)

    model_dirs = sorted(checkpoints_dir.glob("ridge_v0_*"))
    if not model_dirs:
        print("No model checkpoints found. Please run train.py first.")
        exit(1)

    model_dir = model_dirs[-1]  # Most recent
    print(f"Loading model from: {model_dir}")

    # Load model
    model, scaler, config = load_model(model_dir)
    print(f"Model loaded: alpha={config['config']['alpha']}")
    print()

    # Run backtest on val split
    backtester = Backtester(
        model=model,
        scaler=scaler,
        verbose=True,
    )

    result = backtester.run("val")

    print()
    print("=== Validation Complete ===")
    print(f"Result dict keys: {list(result.to_dict().keys())}")
