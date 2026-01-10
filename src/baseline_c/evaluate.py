"""
Evaluate Module for Baseline C

Kill gates and comprehensive metrics per spec §9.

Kill Gates (all must pass):
1. net_CAGR > 0 (positive after-cost returns)
2. MaxDD <= 15%
3. active_days >= 30 (days with gross_exposure > 0 for at least one interval)

Additional Metrics:
- Sharpe ratio (annualized)
- Calmar ratio (CAGR / MaxDD)
- Win rate
- Avg turnover
- Cost drag analysis
- Per-interval breakdown
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .backtest import BacktestResult, Backtester, load_and_backtest
from .train import load_model


@dataclass
class KillGateResult:
    """Single kill gate evaluation."""
    name: str
    threshold: str
    actual: float
    passed: bool
    message: str


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    split: str
    backtest: BacktestResult
    kill_gates: list[KillGateResult]
    all_gates_passed: bool
    timestamp: str

    # Extended metrics
    cagr: float = 0.0
    calmar_ratio: float = 0.0
    gross_sharpe: float = 0.0  # Before costs
    net_sharpe: float = 0.0  # After costs
    cost_drag: float = 0.0  # Cost as % of gross return

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "split": self.split,
            "timestamp": self.timestamp,
            "all_gates_passed": self.all_gates_passed,
            "kill_gates": [
                {
                    "name": g.name,
                    "threshold": g.threshold,
                    "actual": g.actual,
                    "passed": g.passed,
                    "message": g.message,
                }
                for g in self.kill_gates
            ],
            "metrics": {
                "cagr": self.cagr,
                "annualized_return": self.backtest.annualized_return,
                "volatility": self.backtest.volatility,
                "sharpe_ratio": self.backtest.sharpe_ratio,
                "gross_sharpe": self.gross_sharpe,
                "net_sharpe": self.net_sharpe,
                "max_drawdown": self.backtest.max_drawdown,
                "calmar_ratio": self.calmar_ratio,
                "total_return": self.backtest.total_return,
                "total_cost": self.backtest.total_cost,
                "cost_drag": self.cost_drag,
                "avg_turnover": self.backtest.avg_turnover,
                "win_rate": self.backtest.win_rate,
                "trade_days": self.backtest.trade_days,
                "active_days": self.backtest.active_days,
            },
            "symbol_pnl": self.backtest.symbol_pnl,
        }


class Evaluator:
    """
    Evaluates Baseline C strategy against kill gates.

    Kill Gates (per spec §9):
    1. net_CAGR > 0
    2. MaxDD <= 15%
    3. active_days >= 30 (days with at least one interval having gross_exposure > 0)
    """

    # Kill gate thresholds
    MIN_CAGR = 0.0  # Must be positive
    MAX_DRAWDOWN = 0.15  # 15%
    MIN_ACTIVE_DAYS = 30

    def __init__(self, verbose: bool = True):
        """Initialize evaluator."""
        self.verbose = verbose

    def _compute_cagr(self, total_return: float, trade_days: int) -> float:
        """
        Compute CAGR from total return and trade days.

        CAGR = (1 + total_return)^(252/trade_days) - 1

        Args:
            total_return: Total return (decimal, e.g., 0.05 for 5%)
            trade_days: Number of trading days

        Returns:
            Annualized CAGR
        """
        if trade_days == 0:
            return 0.0

        # Handle negative total returns
        if total_return <= -1:
            return -1.0  # Complete loss

        years = trade_days / 252
        if years == 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def _compute_gross_sharpe(self, backtest: BacktestResult) -> float:
        """Compute Sharpe ratio before costs."""
        if not backtest.daily_results:
            return 0.0

        gross_returns = np.array([d.daily_gross_return for d in backtest.daily_results])
        daily_mean = np.mean(gross_returns)
        daily_std = np.std(gross_returns)

        if daily_std == 0:
            return 0.0

        return (daily_mean * 252) / (daily_std * np.sqrt(252))

    def evaluate(self, backtest: BacktestResult) -> EvaluationResult:
        """
        Evaluate backtest against kill gates.

        Args:
            backtest: BacktestResult from Backtester.run()

        Returns:
            EvaluationResult with gate status and metrics
        """
        if self.verbose:
            print("=" * 60)
            print(f"Evaluating {backtest.split} Results")
            print("=" * 60)

        # Compute extended metrics
        cagr = self._compute_cagr(backtest.total_return, backtest.trade_days)
        calmar = abs(cagr / backtest.max_drawdown) if backtest.max_drawdown > 0 else 0.0
        gross_sharpe = self._compute_gross_sharpe(backtest)

        # Cost drag: what % of gross return was eaten by costs
        gross_return = backtest.total_return + backtest.total_cost
        cost_drag = backtest.total_cost / abs(gross_return) if gross_return != 0 else 0.0

        # Evaluate kill gates
        kill_gates = []

        # Gate 1: net_CAGR > 0
        gate1_passed = cagr > self.MIN_CAGR
        kill_gates.append(KillGateResult(
            name="net_CAGR",
            threshold="> 0",
            actual=cagr,
            passed=gate1_passed,
            message=f"CAGR = {cagr * 100:.2f}% {'✅ PASS' if gate1_passed else '❌ FAIL'}",
        ))

        # Gate 2: MaxDD <= 15%
        gate2_passed = backtest.max_drawdown <= self.MAX_DRAWDOWN
        kill_gates.append(KillGateResult(
            name="max_drawdown",
            threshold="<= 15%",
            actual=backtest.max_drawdown,
            passed=gate2_passed,
            message=f"MaxDD = {backtest.max_drawdown * 100:.2f}% {'✅ PASS' if gate2_passed else '❌ FAIL'}",
        ))

        # Gate 3: active_days >= 30 (Baseline C specific)
        gate3_passed = backtest.active_days >= self.MIN_ACTIVE_DAYS
        kill_gates.append(KillGateResult(
            name="active_days",
            threshold=">= 30",
            actual=float(backtest.active_days),
            passed=gate3_passed,
            message=f"Active days = {backtest.active_days} {'✅ PASS' if gate3_passed else '❌ FAIL'}",
        ))

        all_passed = all(g.passed for g in kill_gates)

        result = EvaluationResult(
            split=backtest.split,
            backtest=backtest,
            kill_gates=kill_gates,
            all_gates_passed=all_passed,
            timestamp=datetime.now().isoformat(),
            cagr=cagr,
            calmar_ratio=calmar,
            gross_sharpe=gross_sharpe,
            net_sharpe=backtest.sharpe_ratio,
            cost_drag=cost_drag,
        )

        if self.verbose:
            print()
            print("Kill Gates:")
            for gate in kill_gates:
                print(f"  {gate.message}")
            print()
            print(f"Overall: {'✅ ALL GATES PASSED' if all_passed else '❌ STRATEGY KILLED'}")
            print()
            print("Extended Metrics:")
            print(f"  CAGR: {cagr * 100:.2f}%")
            print(f"  Gross Sharpe: {gross_sharpe:.3f}")
            print(f"  Net Sharpe: {backtest.sharpe_ratio:.3f}")
            print(f"  Calmar Ratio: {calmar:.3f}")
            print(f"  Cost Drag: {cost_drag * 100:.1f}% of gross return")

        return result

    def evaluate_multiple(
        self,
        backtests: dict[str, BacktestResult],
    ) -> dict[str, EvaluationResult]:
        """
        Evaluate multiple backtests.

        Args:
            backtests: Dict mapping split -> BacktestResult

        Returns:
            Dict mapping split -> EvaluationResult
        """
        results = {}
        for split, backtest in backtests.items():
            results[split] = self.evaluate(backtest)
        return results


def save_evaluation(
    result: EvaluationResult,
    output_dir: Path,
    filename: str = None,
) -> Path:
    """
    Save evaluation result to JSON.

    Args:
        result: EvaluationResult to save
        output_dir: Output directory
        filename: Optional filename (defaults to eval_{split}_{timestamp}.json)

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{result.split}_{ts}.json"

    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"Evaluation saved to: {filepath}")
    return filepath


def load_and_evaluate(
    model_dir: Path,
    split: str = "val",
    verbose: bool = True,
) -> EvaluationResult:
    """
    Load model, run backtest, and evaluate.

    Args:
        model_dir: Path to saved model directory
        split: Split to evaluate
        verbose: Show progress

    Returns:
        EvaluationResult
    """
    # Load model
    models, scalers, config = load_model(model_dir)

    # Run backtest
    backtester = Backtester(
        models=models,
        scalers=scalers,
        verbose=verbose,
    )
    backtest_result = backtester.run(split)

    # Evaluate
    evaluator = Evaluator(verbose=verbose)
    return evaluator.evaluate(backtest_result)


def run_full_evaluation(
    model_dir: Path,
    output_dir: Path = None,
    splits: list[str] = None,
    verbose: bool = True,
) -> dict[str, EvaluationResult]:
    """
    Run full evaluation on multiple splits.

    Args:
        model_dir: Path to saved model directory
        output_dir: Optional directory to save results
        splits: Splits to evaluate (defaults to val, dev_test)
        verbose: Show progress

    Returns:
        Dict mapping split -> EvaluationResult
    """
    if splits is None:
        splits = ["val", "dev_test"]

    # Load model
    models, scalers, config = load_model(model_dir)

    if verbose:
        print("=" * 60)
        print("Full Evaluation Run")
        print("=" * 60)
        print(f"Model: {model_dir}")
        print(f"Splits: {splits}")
        print()

    # Run backtests
    backtester = Backtester(models=models, scalers=scalers, verbose=verbose)
    backtests = {}
    for split in splits:
        backtests[split] = backtester.run(split)

    # Evaluate
    evaluator = Evaluator(verbose=verbose)
    results = evaluator.evaluate_multiple(backtests)

    # Save results
    if output_dir:
        for split, result in results.items():
            save_evaluation(result, output_dir)

    # Summary
    if verbose:
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for split, result in results.items():
            status = "✅ PASS" if result.all_gates_passed else "❌ KILL"
            print(f"{split}: {status} | CAGR={result.cagr*100:.2f}% | Sharpe={result.net_sharpe:.3f} | MaxDD={result.backtest.max_drawdown*100:.2f}% | Active={result.backtest.active_days}")

    return results


if __name__ == "__main__":
    from pathlib import Path

    print("=== Baseline C Evaluator Validation ===")

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
    print()

    # Run evaluation on val split
    result = load_and_evaluate(model_dir, split="val", verbose=True)

    print()
    print("=== Validation Complete ===")
    print(f"Result keys: {list(result.to_dict().keys())}")
