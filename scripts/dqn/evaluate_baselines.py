#!/usr/bin/env python3
"""
Gate 1: Evaluate Baseline Policies for Environment Validation

This script evaluates baseline policies on the DQN trading environment to:
1. Verify no look-ahead bias (FLAT policy Sharpe ≈ 0)
2. Verify transaction costs work (Random policy Sharpe < 0)
3. Establish baseline performance metrics

Kill Tests (MUST PASS):
- always_flat: abs(Sharpe) < 0.1
- random: Sharpe < 0 AND total_return < 0

Usage:
    python scripts/dqn/evaluate_baselines.py --data-dir ../data/dqn_dev_test --output results/baseline_evaluation.json
    python scripts/dqn/evaluate_baselines.py --data-dir ../data/dqn_dev_test --episodes 50 --verbose
    python scripts/dqn/evaluate_baselines.py --kill-tests-only  # Quick validation
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dsp100k.src.dqn.env import DQNTradingEnv
from dsp100k.src.dqn.baselines import (
    get_baseline_policies,
    get_kill_test_policies,
    BasePolicy,
)


def evaluate_policy(
    env: DQNTradingEnv,
    policy: BasePolicy,
    num_episodes: int = 20,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a policy over multiple episodes.

    Args:
        env: DQN trading environment
        policy: Policy to evaluate
        num_episodes: Number of episodes to run
        verbose: Print per-episode details

    Returns:
        Dictionary with evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    episode_trades = []
    all_rewards = []

    for ep in range(num_episodes):
        policy.reset()
        obs, info = env.reset()

        episode_return = 0.0
        episode_length = 0
        episode_trades_count = 0
        prev_positions = np.zeros(env.num_symbols)

        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1
            all_rewards.append(reward)

            # Count position changes as trades
            if "action" in info:
                curr_positions = info.get("positions", np.zeros(env.num_symbols))
                trades = np.sum(np.abs(curr_positions - prev_positions) > 0)
                episode_trades_count += trades
                prev_positions = curr_positions.copy()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_trades.append(episode_trades_count)

        if verbose:
            print(f"  Episode {ep+1:3d}: return={episode_return:+.6f}, length={episode_length}, trades={episode_trades_count}")

    # Compute statistics
    returns = np.array(episode_returns)
    rewards = np.array(all_rewards)

    # Sharpe ratio (daily returns assumption for intraday)
    # Using standard formula: mean / std * sqrt(252)
    # But for per-minute rewards, we scale differently
    mean_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 1e-10

    # Episode Sharpe (treating each episode as a "day")
    sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 1e-10 else 0.0

    # Per-step statistics
    mean_reward = np.mean(rewards) if len(rewards) > 0 else 0.0
    std_reward = np.std(rewards) if len(rewards) > 1 else 1e-10

    # Max drawdown (episode-level)
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    results = {
        "policy": policy.name,
        "num_episodes": num_episodes,
        "total_return": float(np.sum(returns)),
        "mean_episode_return": float(mean_return),
        "std_episode_return": float(std_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_trades_per_episode": float(np.mean(episode_trades)),
        "total_trades": int(np.sum(episode_trades)),
        "mean_reward_per_step": float(mean_reward),
        "std_reward_per_step": float(std_reward),
        "episode_returns": [float(r) for r in returns.tolist()],
    }

    return results


def run_kill_tests(results: dict[str, dict]) -> dict[str, Any]:
    """
    Run kill tests on evaluation results.

    Kill Tests:
    1. always_flat: abs(Sharpe) < 0.1 → No look-ahead bias
    2. random: Sharpe < 0 → Transaction costs working

    Args:
        results: Dictionary of policy results

    Returns:
        Kill test results with PASS/FAIL status
    """
    kill_tests = {}

    # Test 1: FLAT policy Sharpe ≈ 0
    if "always_flat" in results:
        flat_sharpe = results["always_flat"]["sharpe_ratio"]
        flat_return = results["always_flat"]["total_return"]
        flat_pass = abs(flat_sharpe) < 0.1

        kill_tests["flat_sharpe_zero"] = {
            "test": "FLAT policy Sharpe ≈ 0 (no look-ahead bias)",
            "condition": "abs(sharpe) < 0.1",
            "value": float(flat_sharpe),
            "total_return": float(flat_return),
            "status": "PASS" if flat_pass else "KILL",
            "message": (
                "No look-ahead bias detected" if flat_pass
                else f"CRITICAL: Look-ahead bias likely! Sharpe={flat_sharpe:.4f}"
            ),
        }

    # Test 2: Random policy Sharpe < 0 and return < 0
    if "random" in results:
        random_sharpe = results["random"]["sharpe_ratio"]
        random_return = results["random"]["total_return"]
        random_pass = random_sharpe < 0 and random_return < 0

        kill_tests["random_negative"] = {
            "test": "Random policy Sharpe < 0 (transaction costs work)",
            "condition": "sharpe < 0 AND total_return < 0",
            "sharpe": float(random_sharpe),
            "total_return": float(random_return),
            "status": "PASS" if random_pass else "KILL",
            "message": (
                "Transaction costs correctly dominate random trading" if random_pass
                else f"CRITICAL: Transaction costs not working! Sharpe={random_sharpe:.4f}, Return={random_return:.6f}"
            ),
        }

    # Overall status
    all_pass = all(
        test["status"] == "PASS"
        for test in kill_tests.values()
    )

    return {
        "tests": kill_tests,
        "overall_status": "PASS" if all_pass else "KILL",
        "timestamp": datetime.now().isoformat(),
    }


def print_results(results: dict[str, dict], kill_test_results: dict) -> None:
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 70)

    for policy_name, metrics in results.items():
        print(f"\n📊 {policy_name.upper()}")
        print(f"   Sharpe Ratio:     {metrics['sharpe_ratio']:+.4f}")
        print(f"   Total Return:     {metrics['total_return']:+.6f}")
        print(f"   Max Drawdown:     {metrics['max_drawdown']:.6f}")
        print(f"   Trades/Episode:   {metrics['mean_trades_per_episode']:.1f}")
        print(f"   Mean Ep Length:   {metrics['mean_episode_length']:.1f}")

    print("\n" + "=" * 70)
    print("KILL TEST RESULTS")
    print("=" * 70)

    for test_name, test_result in kill_test_results["tests"].items():
        status_emoji = "✅" if test_result["status"] == "PASS" else "❌"
        print(f"\n{status_emoji} {test_result['test']}")
        print(f"   Condition: {test_result['condition']}")
        print(f"   Status: {test_result['status']}")
        print(f"   {test_result['message']}")

    print("\n" + "=" * 70)
    overall_emoji = "✅" if kill_test_results["overall_status"] == "PASS" else "❌"
    print(f"{overall_emoji} OVERALL: {kill_test_results['overall_status']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline policies for DQN environment validation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/dqn_dev_test",
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_evaluation.json",
        help="Output file for results JSON",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per policy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--kill-tests-only",
        action="store_true",
        help="Only run kill test policies (flat, random)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode details",
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_path = (script_dir / args.output).resolve()

    # Validate data directory
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Gate 1: Baseline Policy Evaluation")
    print("=" * 70)
    print(f"\n📁 Data: {data_dir}")
    print(f"📄 Output: {output_path}")
    print(f"🔄 Episodes: {args.episodes}")
    print(f"🎲 Seed: {args.seed}")

    # Create environment
    print("\n🏗️ Creating environment...")
    env = DQNTradingEnv(
        data_dir=str(data_dir),
        window_size=60,
        target_gross=0.10,
        k_per_side=3,
        turnover_cost=0.0010,
        start_minute=61,  # 10:31 ET
        end_minute=270,   # 14:00 ET
        apply_constraint=True,
    )
    print(f"   Symbols: {env.symbols}")
    print(f"   Available dates: {len(env.available_dates)}")
    print(f"   Trading minutes: {env.num_trading_minutes}")

    # Get policies
    if args.kill_tests_only:
        policies = get_kill_test_policies(num_symbols=env.num_symbols, seed=args.seed)
        print(f"\n🎯 Running KILL TESTS ONLY: {list(policies.keys())}")
    else:
        policies = get_baseline_policies(num_symbols=env.num_symbols, seed=args.seed)
        print(f"\n🎯 Policies: {list(policies.keys())}")

    # Evaluate each policy
    results = {}
    for policy_name, policy in policies.items():
        print(f"\n📈 Evaluating {policy_name}...")
        results[policy_name] = evaluate_policy(
            env=env,
            policy=policy,
            num_episodes=args.episodes,
            verbose=args.verbose,
        )

    # Run kill tests
    kill_test_results = run_kill_tests(results)

    # Print results
    print_results(results, kill_test_results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "data_dir": str(data_dir),
            "episodes": args.episodes,
            "seed": args.seed,
            "symbols": env.symbols,
            "num_dates": len(env.available_dates),
            "trading_minutes": env.num_trading_minutes,
        },
        "results": results,
        "kill_tests": kill_test_results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n📄 Results saved to: {output_path}")

    # Exit with error code if kill tests failed
    if kill_test_results["overall_status"] == "KILL":
        print("\n❌ GATE 1 FAILED - Kill tests did not pass!")
        sys.exit(1)
    else:
        print("\n✅ GATE 1 PASSED - All kill tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
