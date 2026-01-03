"""
Hierarchical Success Criteria for DQN Model Selection

Implements the NN_OBJECTIVE.md framework:
1. Primary objective: maximize CAGR proxy
2. Hard gates: reject models violating risk/activity constraints
3. Tie-breaker: use Sharpe only for close candidates

This prevents the "always FLAT" trivial optimum from being selected.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class GateResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    value: float
    threshold: float
    direction: str  # "min" or "max"
    message: str


@dataclass
class HierarchicalMetrics:
    """
    All metrics needed for hierarchical model selection.

    Aligned with NN_OBJECTIVE.md definitions:
    - cagr_proxy: exp(252 * mean(daily_log_returns)) - 1
    - max_drawdown_pct: max peak-to-trough decline in equity curve
    - annual_turnover: 252 * mean(daily_turnover)
    - activity_pct: % of steps with gross exposure > 0
    - sharpe_ratio: annualized Sharpe (tie-breaker only)
    """
    # Primary metric
    cagr_proxy: float = 0.0

    # Hard gate metrics
    max_drawdown_pct: float = 0.0
    annual_turnover: float = 0.0
    activity_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Auxiliary metrics (for diagnostics)
    mean_daily_return: float = 0.0
    std_daily_return: float = 0.0
    win_rate: float = 0.0
    num_episodes: int = 0
    total_steps: int = 0

    # Action distribution
    action_distribution: Dict[int, float] = field(default_factory=dict)


@dataclass
class SuccessCriteria:
    """
    Configurable success criteria for model selection.

    Default values from NN_OBJECTIVE.md:
    - max_drawdown_pct: <= 15%
    - annual_turnover: <= 250
    - activity_pct: >= 5%
    - sharpe_ratio: >= 0 (sanity gate)
    """
    # Hard gates (all must pass)
    max_drawdown_pct: float = 15.0  # Maximum allowed drawdown
    max_annual_turnover: float = 250.0  # Maximum annual turnover
    min_activity_pct: float = 5.0  # Minimum activity percentage
    min_sharpe: float = 0.0  # Minimum Sharpe (sanity gate)

    # Tie-breaker tolerance
    cagr_tolerance: float = 0.01  # Use Sharpe if CAGR within this band

    # W_max for scaling rewards to returns (from env config)
    w_max: float = 0.0167  # Default: 0.10 / (2 * 3) = 0.0167


def compute_hierarchical_metrics(
    episode_pnls: np.ndarray,
    episode_turnovers: np.ndarray,
    step_gross_exposures: np.ndarray,
    action_counts: Dict[int, int],
    w_max: float = 0.0167,
) -> HierarchicalMetrics:
    """
    Compute all metrics needed for hierarchical model selection.

    Args:
        episode_pnls: Array of daily PnL values (in "position units")
        episode_turnovers: Array of daily turnover values (sum of |position changes|)
        step_gross_exposures: Array of gross exposure at each step (for activity)
        action_counts: Dict mapping action -> count
        w_max: Weight multiplier to convert position units to returns

    Returns:
        HierarchicalMetrics with all computed values
    """
    num_episodes = len(episode_pnls)
    total_steps = len(step_gross_exposures)

    # Convert PnL to log-returns (NN_OBJECTIVE.md §Reward as implemented)
    # r_d = w_max * daily_pnl_d
    daily_returns = episode_pnls * w_max

    # CAGR proxy (NN_OBJECTIVE.md §Profit / CAGR proxy)
    # CAGR_proxy = exp(252 * mean(r_d)) - 1
    mean_daily_return = np.mean(daily_returns) if num_episodes > 0 else 0.0
    cagr_proxy = np.exp(252 * mean_daily_return) - 1

    # Sharpe ratio (annualized)
    std_daily_return = np.std(daily_returns) if num_episodes > 1 else 1e-10
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 1e-10 else 0.0

    # Max drawdown (NN_OBJECTIVE.md §Max drawdown)
    # Build equity curve: E_0 = 1, E_{d+1} = E_d * exp(r_d)
    if num_episodes > 0:
        cumulative_returns = np.cumsum(daily_returns)
        equity_curve = np.exp(cumulative_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_drawdown_pct = np.max(drawdowns) * 100
    else:
        max_drawdown_pct = 0.0

    # Annual turnover (NN_OBJECTIVE.md §Turnover)
    # turnover_d already in position units, multiply by w_max for portfolio-weight units
    daily_turnover = episode_turnovers * w_max
    annual_turnover = 252 * np.mean(daily_turnover) if num_episodes > 0 else 0.0

    # Activity percentage (NN_OBJECTIVE.md §Activity)
    # activity_pct = 100 * mean_t[ gross_t > 0 ]
    if total_steps > 0:
        activity_pct = 100 * np.mean(step_gross_exposures > 0)
    else:
        activity_pct = 0.0

    # Win rate
    win_rate = np.mean(episode_pnls > 0) if num_episodes > 0 else 0.0

    # Action distribution
    total_actions = sum(action_counts.values())
    action_distribution = {
        k: v / max(total_actions, 1) for k, v in action_counts.items()
    }

    return HierarchicalMetrics(
        cagr_proxy=float(cagr_proxy),
        max_drawdown_pct=float(max_drawdown_pct),
        annual_turnover=float(annual_turnover),
        activity_pct=float(activity_pct),
        sharpe_ratio=float(sharpe_ratio),
        mean_daily_return=float(mean_daily_return),
        std_daily_return=float(std_daily_return),
        win_rate=float(win_rate),
        num_episodes=num_episodes,
        total_steps=total_steps,
        action_distribution=action_distribution,
    )


def check_hard_gates(
    metrics: HierarchicalMetrics,
    criteria: Optional[SuccessCriteria] = None,
) -> List[GateResult]:
    """
    Check all hard gates for a model.

    Args:
        metrics: Computed metrics for the model
        criteria: Success criteria (uses defaults if None)

    Returns:
        List of GateResult for each gate
    """
    if criteria is None:
        criteria = SuccessCriteria()

    results = []

    # Gate 1: Max drawdown <= threshold
    passed = metrics.max_drawdown_pct <= criteria.max_drawdown_pct
    results.append(GateResult(
        name="max_drawdown_pct",
        passed=passed,
        value=metrics.max_drawdown_pct,
        threshold=criteria.max_drawdown_pct,
        direction="max",
        message=f"{'✅' if passed else '❌'} Max DD: {metrics.max_drawdown_pct:.2f}% {'<=' if passed else '>'} {criteria.max_drawdown_pct}%",
    ))

    # Gate 2: Annual turnover <= threshold
    passed = metrics.annual_turnover <= criteria.max_annual_turnover
    results.append(GateResult(
        name="annual_turnover",
        passed=passed,
        value=metrics.annual_turnover,
        threshold=criteria.max_annual_turnover,
        direction="max",
        message=f"{'✅' if passed else '❌'} Annual Turnover: {metrics.annual_turnover:.1f} {'<=' if passed else '>'} {criteria.max_annual_turnover}",
    ))

    # Gate 3: Activity >= threshold (prevents trivial FLAT optimum)
    passed = metrics.activity_pct >= criteria.min_activity_pct
    results.append(GateResult(
        name="activity_pct",
        passed=passed,
        value=metrics.activity_pct,
        threshold=criteria.min_activity_pct,
        direction="min",
        message=f"{'✅' if passed else '❌'} Activity: {metrics.activity_pct:.1f}% {'>=' if passed else '<'} {criteria.min_activity_pct}%",
    ))

    # Gate 4: Sharpe >= 0 (sanity gate)
    passed = metrics.sharpe_ratio >= criteria.min_sharpe
    results.append(GateResult(
        name="sharpe_ratio",
        passed=passed,
        value=metrics.sharpe_ratio,
        threshold=criteria.min_sharpe,
        direction="min",
        message=f"{'✅' if passed else '❌'} Sharpe: {metrics.sharpe_ratio:.4f} {'>=' if passed else '<'} {criteria.min_sharpe}",
    ))

    return results


def select_best_model(
    candidates: List[Dict[str, Any]],
    criteria: Optional[SuccessCriteria] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select best model using hierarchical criteria.

    1. Filter out models failing any hard gate
    2. Among survivors, select by highest CAGR proxy
    3. If two models are within tolerance, use Sharpe as tie-breaker

    Args:
        candidates: List of dicts with "name", "metrics" (HierarchicalMetrics), "gate_results"
        criteria: Success criteria (uses defaults if None)

    Returns:
        Best candidate dict, or None if all fail gates
    """
    if criteria is None:
        criteria = SuccessCriteria()

    # Filter to models passing all gates
    survivors = []
    for candidate in candidates:
        gates = candidate.get("gate_results", [])
        if all(g.passed for g in gates):
            survivors.append(candidate)

    if not survivors:
        return None

    # Sort by CAGR proxy (descending)
    survivors.sort(key=lambda x: x["metrics"].cagr_proxy, reverse=True)

    # Check if top candidates are within tolerance (tie-break with Sharpe)
    best = survivors[0]
    for candidate in survivors[1:]:
        cagr_diff = abs(best["metrics"].cagr_proxy - candidate["metrics"].cagr_proxy)
        if cagr_diff <= criteria.cagr_tolerance:
            # Within tolerance - use Sharpe as tie-breaker
            if candidate["metrics"].sharpe_ratio > best["metrics"].sharpe_ratio:
                best = candidate

    return best


def format_gate_report(
    metrics: HierarchicalMetrics,
    gate_results: List[GateResult],
    model_name: str = "Model",
) -> str:
    """
    Format a human-readable gate report.

    Args:
        metrics: Model metrics
        gate_results: Results of gate checks
        model_name: Name for display

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        f"Hierarchical Evaluation: {model_name}",
        "=" * 60,
        "",
        "📊 Primary Metric:",
        f"   CAGR Proxy: {metrics.cagr_proxy:.4f} ({metrics.cagr_proxy * 100:.2f}%)",
        "",
        "🚧 Hard Gates:",
    ]

    for gate in gate_results:
        lines.append(f"   {gate.message}")

    all_passed = all(g.passed for g in gate_results)
    lines.extend([
        "",
        "📈 Auxiliary Metrics:",
        f"   Mean Daily Return: {metrics.mean_daily_return:.6f}",
        f"   Std Daily Return: {metrics.std_daily_return:.6f}",
        f"   Win Rate: {metrics.win_rate:.2%}",
        f"   Episodes: {metrics.num_episodes}",
        f"   Total Steps: {metrics.total_steps:,}",
        "",
        "🎯 Action Distribution:",
    ])

    for action, pct in sorted(metrics.action_distribution.items()):
        action_names = {0: "FLAT", 1: "LONG_50", 2: "LONG_100", 3: "SHORT_50", 4: "SHORT_100"}
        name = action_names.get(action, f"Action_{action}")
        lines.append(f"   {name}: {pct:.2%}")

    lines.extend([
        "",
        "=" * 60,
        f"{'✅ ALL GATES PASSED' if all_passed else '❌ GATES FAILED'}",
        "=" * 60,
    ])

    return "\n".join(lines)


# Configuration file support (NN_OBJECTIVE.md §Implementation Configuration)
def load_criteria_from_yaml(path: str) -> SuccessCriteria:
    """Load success criteria from YAML config file."""
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f)

    gates = config.get("hard_gates", {})

    return SuccessCriteria(
        max_drawdown_pct=gates.get("max_drawdown_pct", {}).get("threshold", 15.0),
        max_annual_turnover=gates.get("annual_turnover", {}).get("threshold", 250.0),
        min_activity_pct=gates.get("activity_pct", {}).get("threshold", 5.0),
        min_sharpe=gates.get("sharpe_ratio", {}).get("threshold", 0.0),
        cagr_tolerance=config.get("cagr_tolerance", 0.01),
        w_max=config.get("w_max", 0.0167),
    )


if __name__ == "__main__":
    # Test with example data
    print("Testing success_criteria module...")

    # Simulate a FLAT policy (should fail activity gate)
    flat_pnls = np.zeros(50)  # No PnL if always flat
    flat_turnovers = np.zeros(50)  # No turnover
    flat_exposures = np.zeros(1000)  # No exposure
    flat_actions = {0: 9000, 1: 0, 2: 0, 3: 0, 4: 0}  # 100% FLAT

    flat_metrics = compute_hierarchical_metrics(
        flat_pnls, flat_turnovers, flat_exposures, flat_actions
    )
    flat_gates = check_hard_gates(flat_metrics)
    print("\n" + format_gate_report(flat_metrics, flat_gates, "Always-FLAT Policy"))

    # Simulate an active policy
    np.random.seed(42)
    active_pnls = np.random.randn(50) * 0.01  # Some variance
    active_turnovers = np.abs(np.random.randn(50)) * 0.5
    active_exposures = np.abs(np.random.randn(1000)) * 0.05 + 0.02
    active_actions = {0: 2000, 1: 2000, 2: 1000, 3: 2000, 4: 2000}

    active_metrics = compute_hierarchical_metrics(
        active_pnls, active_turnovers, active_exposures, active_actions
    )
    active_gates = check_hard_gates(active_metrics)
    print("\n" + format_gate_report(active_metrics, active_gates, "Active Policy"))
