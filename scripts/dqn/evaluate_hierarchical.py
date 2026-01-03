#!/usr/bin/env python
"""
Hierarchical DQN Evaluation Script

Evaluates models using the NN_OBJECTIVE.md framework:
1. Compute all metrics (CAGR, drawdown, turnover, activity, Sharpe)
2. Apply hard gates (reject FLAT-collapsed models)
3. Select by CAGR with Sharpe tie-breaker

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source ../venv/bin/activate

    # Evaluate single checkpoint
    PYTHONPATH=.. python scripts/dqn/evaluate_hierarchical.py \
        --checkpoint checkpoints/best_model.pt \
        --data-dir ../data/dqn_val \
        --episodes 50

    # Compare multiple checkpoints
    PYTHONPATH=.. python scripts/dqn/evaluate_hierarchical.py \
        --checkpoint checkpoints/sweep_sm0.0/best_model.pt \
        --checkpoint checkpoints/sweep_sm0.0005/best_model.pt \
        --data-dir ../data/dqn_val \
        --episodes 50

    # With custom criteria
    PYTHONPATH=.. python scripts/dqn/evaluate_hierarchical.py \
        --checkpoint checkpoints/best_model.pt \
        --criteria config/success_criteria.yaml
"""

import argparse
import sys
from pathlib import Path
import json
from typing import List, Dict, Any
import random

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hierarchical DQN evaluation (NN_OBJECTIVE.md)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="Path to model checkpoint(s) - can specify multiple",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/dqn_val",
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default=None,
        help="Path to success_criteria.yaml (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode details",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=1,
        help="Decision interval in minutes (must match training, e.g., 15 for DI=15)",
    )

    return parser.parse_args()


def evaluate_model_hierarchical(
    env,
    agent,
    num_episodes: int,
    w_max: float = 0.0167,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model and collect all metrics for hierarchical selection.

    Returns dict with:
    - metrics: HierarchicalMetrics
    - raw_data: episode-level arrays for further analysis
    """
    from dsp100k.src.dqn.success_criteria import compute_hierarchical_metrics

    episode_pnls = []
    episode_turnovers = []
    all_step_exposures = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        ep_turnover = 0.0
        prev_positions = np.zeros(env.num_symbols)

        while not done:
            # Get positions for switch margin
            current_positions = env.positions if hasattr(env, "positions") else None

            actions, q_values, entropy = agent.select_action(
                rolling_window=obs["rolling_window"],
                portfolio_state=obs["portfolio_state"],
                apply_constraint=True,
                explore=False,
                count_env_step=False,
                current_positions=current_positions,
            )

            # Track action distribution
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Track gross exposure (for activity calculation)
            positions = env.positions if hasattr(env, "positions") else np.zeros(env.num_symbols)
            gross_exposure = np.sum(np.abs(positions))
            all_step_exposures.append(gross_exposure)

            # Track turnover
            turnover = np.sum(np.abs(positions - prev_positions))
            ep_turnover += turnover
            prev_positions = positions.copy()

        # Episode-level metrics
        episode_pnls.append(info.get("daily_pnl", 0.0))
        episode_turnovers.append(ep_turnover)

        if verbose:
            print(f"  Episode {ep+1}: PnL={info.get('daily_pnl', 0):.6f}, Turnover={ep_turnover:.2f}")

    # Convert to arrays
    episode_pnls = np.array(episode_pnls)
    episode_turnovers = np.array(episode_turnovers)
    step_exposures = np.array(all_step_exposures)

    # Compute hierarchical metrics
    metrics = compute_hierarchical_metrics(
        episode_pnls=episode_pnls,
        episode_turnovers=episode_turnovers,
        step_gross_exposures=step_exposures,
        action_counts=action_counts,
        w_max=w_max,
    )

    return {
        "metrics": metrics,
        "raw_data": {
            "episode_pnls": episode_pnls.tolist(),
            "episode_turnovers": episode_turnovers.tolist(),
            "action_counts": action_counts,
        },
    }


def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 70)
    print("Hierarchical DQN Evaluation (NN_OBJECTIVE.md Framework)")
    print("=" * 70)

    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    checkpoints = [Path(cp).resolve() for cp in args.checkpoint]

    print(f"\nData: {data_dir}")
    print(f"Checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp}")

    # Verify paths
    if not data_dir.exists():
        print(f"\n❌ ERROR: Data directory not found: {data_dir}")
        return 1

    for cp in checkpoints:
        if not cp.exists():
            print(f"\n❌ ERROR: Checkpoint not found: {cp}")
            return 1

    # Load success criteria
    from dsp100k.src.dqn.success_criteria import (
        SuccessCriteria,
        check_hard_gates,
        format_gate_report,
        select_best_model,
        load_criteria_from_yaml,
    )

    if args.criteria:
        criteria = load_criteria_from_yaml(args.criteria)
        print(f"\nLoaded criteria from: {args.criteria}")
    else:
        criteria = SuccessCriteria()
        print("\nUsing default criteria")

    print(f"  Max Drawdown: <= {criteria.max_drawdown_pct}%")
    print(f"  Max Annual Turnover: <= {criteria.max_annual_turnover}")
    print(f"  Min Activity: >= {criteria.min_activity_pct}%")
    print(f"  Min Sharpe: >= {criteria.min_sharpe}")

    # Import modules
    print("\nLoading modules...")
    try:
        from dsp100k.src.dqn.env import DQNTradingEnv
        from dsp100k.src.dqn.agent import DQNAgent
    except ImportError as e:
        print(f"\n❌ ERROR: Failed to import modules: {e}")
        return 1

    # Create environment
    print("\nCreating environment...")
    env = DQNTradingEnv(
        data_dir=str(data_dir),
        apply_constraint=False,
        decision_interval=args.decision_interval,
    )
    print(f"  Dates: {len(env.available_dates)}")
    print(f"  Decision Interval: {args.decision_interval} minutes")
    print(f"  Symbols: {env.symbols}")

    # Evaluate each checkpoint
    candidates = []

    for cp in checkpoints:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {cp.name}")
        print("=" * 70)

        # Load agent
        agent = DQNAgent(
            num_symbols=env.num_symbols,
            device=args.device,
        )
        agent.load(str(cp))

        # Run evaluation
        print(f"\nRunning {args.episodes} episodes...")
        result = evaluate_model_hierarchical(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            w_max=criteria.w_max,
            verbose=args.verbose,
        )

        # Check gates
        gate_results = check_hard_gates(result["metrics"], criteria)

        # Print report
        print("\n" + format_gate_report(result["metrics"], gate_results, cp.name))

        # Store candidate
        candidates.append({
            "name": str(cp),
            "metrics": result["metrics"],
            "gate_results": gate_results,
            "raw_data": result["raw_data"],
        })

    # Select best model
    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    best = select_best_model(candidates, criteria)

    if best is None:
        print("\n❌ NO MODELS PASSED ALL GATES")
        print("\nAll candidates failed at least one hard gate.")
        print("The 'always FLAT' trivial optimum has been correctly rejected.")
        print("\nNext steps:")
        print("  1. Train longer (more episodes)")
        print("  2. Adjust hyperparameters (learning rate, exploration)")
        print("  3. Consider reward shaping to encourage exploration")
    else:
        print(f"\n✅ BEST MODEL: {best['name']}")
        print(f"   CAGR Proxy: {best['metrics'].cagr_proxy:.4f}")
        print(f"   Sharpe: {best['metrics'].sharpe_ratio:.4f}")
        print(f"   Activity: {best['metrics'].activity_pct:.1f}%")
        print(f"   Max DD: {best['metrics'].max_drawdown_pct:.2f}%")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Checkpoint':<30} {'CAGR':>10} {'Sharpe':>10} {'Activity':>10} {'Passed':>10}")
    print("-" * 70)
    for c in candidates:
        m = c["metrics"]
        passed = "✅" if all(g.passed for g in c["gate_results"]) else "❌"
        name = Path(c["name"]).parent.name + "/" + Path(c["name"]).name
        if len(name) > 28:
            name = "..." + name[-25:]
        print(f"{name:<30} {m.cagr_proxy:>10.4f} {m.sharpe_ratio:>10.4f} {m.activity_pct:>9.1f}% {passed:>10}")
    print("-" * 70)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "data_dir": str(data_dir),
            "episodes": args.episodes,
            "criteria": {
                "max_drawdown_pct": criteria.max_drawdown_pct,
                "max_annual_turnover": criteria.max_annual_turnover,
                "min_activity_pct": criteria.min_activity_pct,
                "min_sharpe": criteria.min_sharpe,
            },
            "candidates": [
                {
                    "name": c["name"],
                    "cagr_proxy": c["metrics"].cagr_proxy,
                    "sharpe_ratio": c["metrics"].sharpe_ratio,
                    "activity_pct": c["metrics"].activity_pct,
                    "max_drawdown_pct": c["metrics"].max_drawdown_pct,
                    "annual_turnover": c["metrics"].annual_turnover,
                    "gates_passed": all(g.passed for g in c["gate_results"]),
                    "action_distribution": c["metrics"].action_distribution,
                }
                for c in candidates
            ],
            "best_model": best["name"] if best else None,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Return exit code
    if best is None:
        return 1  # No model passed gates
    return 0


if __name__ == "__main__":
    sys.exit(main())
