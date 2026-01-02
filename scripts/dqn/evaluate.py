#!/usr/bin/env python
"""
DQN Evaluation Script for Gate 2

Evaluate a trained DQN agent on validation or test data.

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source ../venv/bin/activate

    # Evaluate on validation set
    PYTHONPATH=.. python scripts/dqn/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --data-dir ../data/dqn_val \
        --episodes 100

    # Evaluate on dev_test set (for debugging)
    PYTHONPATH=.. python scripts/dqn/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --data-dir ../data/dqn_dev_test \
        --episodes 50

    # Compare with baselines
    PYTHONPATH=.. python scripts/dqn/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --data-dir ../data/dqn_val \
        --compare-baselines
"""

import argparse
import sys
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Optional
import random

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


@dataclass
class EvalResults:
    """Evaluation results."""

    num_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    mean_trades: float = 0.0
    mean_entropy: float = 0.0
    long_ratio: float = 0.0
    short_ratio: float = 0.0
    flat_ratio: float = 0.0
    max_drawdown: float = 0.0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DQN agent")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
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
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Compare with baseline policies",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--conviction-threshold",
        type=float,
        default=0.0,
        help="Only take non-FLAT actions when Q(action)-Q(FLAT) >= threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level",
    )

    return parser.parse_args()


def evaluate_agent(
    env,
    agent,
    num_episodes: int,
    verbose: int = 1,
) -> EvalResults:
    """
    Evaluate agent on environment.

    Args:
        env: DQN trading environment
        agent: DQN agent
        num_episodes: Number of episodes to evaluate
        verbose: Verbosity level

    Returns:
        Evaluation results
    """
    rewards = []
    pnls = []
    trades_per_ep = []
    entropies = []
    long_counts = []
    short_counts = []
    flat_counts = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        ep_reward = 0.0
        ep_trades = 0
        ep_entropy = 0.0
        ep_long = 0
        ep_short = 0
        ep_flat = 0
        steps = 0

        prev_actions = None

        while not done:
            # Select action (greedy, no exploration)
            # Gate 2.7b: Pass current positions for switch margin (churn control)
            current_positions = env.positions if hasattr(env, "positions") else None
            actions, q_values, entropy = agent.select_action(
                rolling_window=obs["rolling_window"],
                portfolio_state=obs["portfolio_state"],
                apply_constraint=True,
                explore=False,
                count_env_step=False,
                current_positions=current_positions,  # Gate 2.7b: for switch margin
            )

            ep_entropy += entropy

            # Count action types
            for a in actions:
                if a == 0:
                    ep_flat += 1
                elif a in [1, 2]:
                    ep_long += 1
                elif a in [3, 4]:
                    ep_short += 1

            # Count trades
            if prev_actions is not None:
                ep_trades += np.sum(actions != prev_actions)
            prev_actions = actions.copy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)
        pnls.append(info.get("daily_pnl", 0.0))
        trades_per_ep.append(ep_trades)
        entropies.append(ep_entropy / max(steps, 1))
        long_counts.append(ep_long)
        short_counts.append(ep_short)
        flat_counts.append(ep_flat)

        if verbose >= 2:
            print(f"  Episode {ep+1}: reward={ep_reward:.6f}, pnl={info.get('daily_pnl', 0):.6f}, trades={ep_trades}")

    # Compute metrics
    rewards = np.array(rewards)
    pnls = np.array(pnls)

    # Sharpe (annualized)
    sharpe = (rewards.mean() / rewards.std()) * np.sqrt(252) if rewards.std() > 0 else 0.0

    # Win rate
    win_rate = np.mean(pnls > 0)

    # Action distribution
    total_actions = sum(long_counts) + sum(short_counts) + sum(flat_counts)
    long_ratio = sum(long_counts) / max(total_actions, 1)
    short_ratio = sum(short_counts) / max(total_actions, 1)
    flat_ratio = sum(flat_counts) / max(total_actions, 1)

    # Max drawdown (simple approximation)
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    return EvalResults(
        num_episodes=num_episodes,
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        mean_pnl=float(pnls.mean()),
        std_pnl=float(pnls.std()),
        sharpe=float(sharpe),
        win_rate=float(win_rate),
        mean_trades=float(np.mean(trades_per_ep)),
        mean_entropy=float(np.mean(entropies)),
        long_ratio=float(long_ratio),
        short_ratio=float(short_ratio),
        flat_ratio=float(flat_ratio),
        max_drawdown=float(max_drawdown),
    )


def evaluate_baseline(
    env,
    policy_fn,
    num_episodes: int,
    verbose: int = 1,
) -> EvalResults:
    """
    Evaluate a baseline policy.

    Args:
        env: DQN trading environment
        policy_fn: Function that takes observation and returns action
        num_episodes: Number of episodes
        verbose: Verbosity level

    Returns:
        Evaluation results
    """
    rewards = []
    pnls = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            actions = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)
        pnls.append(info.get("daily_pnl", 0.0))

    rewards = np.array(rewards)
    pnls = np.array(pnls)

    sharpe = (rewards.mean() / rewards.std()) * np.sqrt(252) if rewards.std() > 0 else 0.0

    return EvalResults(
        num_episodes=num_episodes,
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        mean_pnl=float(pnls.mean()),
        std_pnl=float(pnls.std()),
        sharpe=float(sharpe),
        win_rate=float(np.mean(pnls > 0)),
    )


def main():
    """Main evaluation function."""
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("DQN Evaluation - Gate 2")
    print("=" * 60)

    # Resolve paths
    checkpoint_path = Path(args.checkpoint).resolve()
    data_dir = Path(args.data_dir).resolve()

    print(f"\nPaths:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data: {data_dir}")

    # Verify paths
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    if not data_dir.exists():
        print(f"\n❌ ERROR: Data directory not found: {data_dir}")
        return 1

    # Import modules
    print("\nLoading modules...")
    try:
        from dsp100k.src.dqn.env import DQNTradingEnv
        from dsp100k.src.dqn.agent import DQNAgent
        from dsp100k.src.dqn.baselines import FlatPolicy, RandomPolicy, MomentumPolicy
    except ImportError as e:
        print(f"\n❌ ERROR: Failed to import modules: {e}")
        return 1

    # Create environment
    print("\nCreating environment...")
    env = DQNTradingEnv(
        data_dir=str(data_dir),
        apply_constraint=False,  # Agent handles top-K
    )
    print(f"  Dates: {len(env.available_dates)}")
    print(f"  Symbols: {env.symbols}")

    # Create agent and load checkpoint
    print("\nLoading agent...")
    agent = DQNAgent(
        num_symbols=env.num_symbols,
        device=args.device,
        conviction_threshold=args.conviction_threshold,
    )
    agent.load(str(checkpoint_path))
    # Allow evaluating different churn-governor settings without retraining.
    agent.conviction_threshold = args.conviction_threshold
    print(f"  Device: {agent.device}")
    print(f"  Train steps: {agent.steps:,}")
    print(f"  Env steps: {getattr(agent, 'env_steps', 0):,}")
    print(f"  Conviction threshold: {agent.conviction_threshold}")

    # Evaluate agent
    print(f"\nEvaluating agent ({args.episodes} episodes)...")
    results = evaluate_agent(env, agent, args.episodes, args.verbose)

    print("\n" + "=" * 60)
    print("DQN Agent Results")
    print("=" * 60)
    print(f"  Episodes: {results.num_episodes}")
    print(f"  Mean Reward: {results.mean_reward:.6f} (±{results.std_reward:.6f})")
    print(f"  Mean P&L: {results.mean_pnl:.6f} (±{results.std_pnl:.6f})")
    print(f"  Sharpe Ratio: {results.sharpe:.4f}")
    print(f"  Win Rate: {results.win_rate:.2%}")
    print(f"  Mean Trades/Episode: {results.mean_trades:.1f}")
    print(f"  Mean Entropy: {results.mean_entropy:.4f}")
    print(f"  Max Drawdown: {results.max_drawdown:.6f}")
    print(f"  Action Distribution:")
    print(f"    Long: {results.long_ratio:.2%}")
    print(f"    Short: {results.short_ratio:.2%}")
    print(f"    Flat: {results.flat_ratio:.2%}")

    # Compare with baselines if requested
    if args.compare_baselines:
        print("\n" + "=" * 60)
        print("Baseline Comparison")
        print("=" * 60)

        # Always-FLAT baseline
        print("\nEvaluating FLAT baseline...")
        flat_policy = FlatPolicy()
        flat_results = evaluate_baseline(
            env, lambda obs: flat_policy.act(obs), args.episodes, args.verbose
        )
        print(f"  FLAT: Sharpe = {flat_results.sharpe:.4f}, Win Rate = {flat_results.win_rate:.2%}")

        # Random baseline
        print("\nEvaluating RANDOM baseline...")
        random_policy = RandomPolicy()
        random_results = evaluate_baseline(
            env, lambda obs: random_policy.act(obs), args.episodes, args.verbose
        )
        print(f"  RANDOM: Sharpe = {random_results.sharpe:.4f}, Win Rate = {random_results.win_rate:.2%}")

        # Momentum baseline
        print("\nEvaluating MOMENTUM baseline...")
        momentum_policy = MomentumPolicy()
        momentum_results = evaluate_baseline(
            env, lambda obs: momentum_policy.act(obs), args.episodes, args.verbose
        )
        print(f"  MOMENTUM: Sharpe = {momentum_results.sharpe:.4f}, Win Rate = {momentum_results.win_rate:.2%}")

        # Summary table
        print("\n" + "-" * 60)
        print(f"{'Policy':<15} {'Sharpe':>10} {'Win Rate':>10} {'Mean P&L':>12}")
        print("-" * 60)
        print(f"{'DQN Agent':<15} {results.sharpe:>10.4f} {results.win_rate:>10.2%} {results.mean_pnl:>12.6f}")
        print(f"{'FLAT':<15} {flat_results.sharpe:>10.4f} {flat_results.win_rate:>10.2%} {flat_results.mean_pnl:>12.6f}")
        print(f"{'RANDOM':<15} {random_results.sharpe:>10.4f} {random_results.win_rate:>10.2%} {random_results.mean_pnl:>12.6f}")
        print(f"{'MOMENTUM':<15} {momentum_results.sharpe:>10.4f} {momentum_results.win_rate:>10.2%} {momentum_results.mean_pnl:>12.6f}")
        print("-" * 60)

    # Kill test evaluation
    print("\n" + "=" * 60)
    print("Kill Test Results")
    print("=" * 60)

    kill_tests = []

    # Test 1: Sharpe > 0
    passed = results.sharpe > 0
    kill_tests.append(("Val Sharpe > 0", passed, results.sharpe))
    if passed:
        print(f"  ✅ PASS: Sharpe = {results.sharpe:.4f} > 0")
    else:
        print(f"  ❌ FAIL: Sharpe = {results.sharpe:.4f} <= 0")

    # Test 2: Return > 0
    passed = results.mean_pnl > 0
    kill_tests.append(("Val Return > 0", passed, results.mean_pnl))
    if passed:
        print(f"  ✅ PASS: Mean P&L = {results.mean_pnl:.6f} > 0")
    else:
        print(f"  ❌ FAIL: Mean P&L = {results.mean_pnl:.6f} <= 0")

    # Test 3: Action entropy > 0.5 (agent uses multiple actions)
    passed = results.mean_entropy > 0.5
    kill_tests.append(("Action Entropy > 0.5", passed, results.mean_entropy))
    if passed:
        print(f"  ✅ PASS: Entropy = {results.mean_entropy:.4f} > 0.5")
    else:
        print(f"  ❌ FAIL: Entropy = {results.mean_entropy:.4f} <= 0.5 (possible collapse)")

    # Stretch goal
    if results.sharpe > 0.5:
        print(f"\n  🎯 STRETCH GOAL: Sharpe = {results.sharpe:.4f} > 0.5")

    # Overall result
    all_passed = all(t[1] for t in kill_tests)
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ GATE 2 KILL TEST: PASS")
    else:
        print("❌ GATE 2 KILL TEST: FAIL")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "checkpoint": str(checkpoint_path),
            "data_dir": str(data_dir),
            "agent_results": asdict(results),
            "kill_tests": [
                {"name": name, "passed": passed, "value": value}
                for name, passed, value in kill_tests
            ],
            "overall_passed": all_passed,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
