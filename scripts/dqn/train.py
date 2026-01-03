#!/usr/bin/env python
"""
DQN Training Script for Gate 2

Train a Double DQN agent on the validated trading environment.

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source ../venv/bin/activate

    # Basic training
    PYTHONPATH=.. python scripts/dqn/train.py

    # With custom parameters
    PYTHONPATH=.. python scripts/dqn/train.py \
        --train-dir ../data/dqn_train \
        --val-dir ../data/dqn_val \
        --checkpoint-dir checkpoints \
        --episodes 50000 \
        --eval-freq 1000

    # Resume from checkpoint
    PYTHONPATH=.. python scripts/dqn/train.py --resume checkpoints/checkpoint_10000.pt
"""

import argparse
import sys
from pathlib import Path
import random

# Add parent directories to path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN agent for intraday trading")

    # Data paths
    parser.add_argument(
        "--train-dir",
        type=str,
        default="../data/dqn_train",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="../data/dqn_val",
        help="Directory containing validation data",
    )

    # Output paths
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs",
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=50000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Warmup episodes (fill replay buffer)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Evaluation frequency (episodes)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of episodes per evaluation",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5000,
        help="Checkpoint frequency (episodes)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (eval cycles without improvement). Set high (e.g., 50) to disable early stopping.",
    )

    # Agent parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=500000,
        help="Replay buffer size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for exploration",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final epsilon for exploration",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=int,
        default=100000,
        help="Epsilon decay steps",
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=1000,
        help="Target network update frequency (steps)",
    )

    # Environment parameters
    parser.add_argument(
        "--k-per-side",
        type=int,
        default=3,
        help="Max positions per side (long/short)",
    )
    parser.add_argument(
        "--turnover-cost",
        type=float,
        default=0.001,
        help="Transaction cost (10 bps)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Rolling window size",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=1,
        help="Minutes between decisions (1=every bar, 10/15/20 to reduce turnover)",
    )

    # Policy / action filtering
    parser.add_argument(
        "--conviction-threshold",
        type=float,
        default=0.0,
        help="Only take non-FLAT actions when Q(action)-Q(FLAT) >= threshold (greedy actions only)",
    )
    parser.add_argument(
        "--switch-margin",
        type=float,
        default=0.0,
        help="Gate 2.7b: Only change position if Q(new)-Q(current) >= margin (churn control)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Misc
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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    print("=" * 60)
    print("DQN Training - Gate 2A")
    print("=" * 60)

    # Resolve paths
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    log_dir = Path(args.log_dir).resolve()

    print(f"\nPaths:")
    print(f"  Train data: {train_dir}")
    print(f"  Val data: {val_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_dir}")

    # Verify data directories exist
    if not train_dir.exists():
        print(f"\n❌ ERROR: Training directory not found: {train_dir}")
        print("  Please run: python scripts/dqn/create_splits.py --source ../data/stage1_raw --output ../data")
        return 1

    if not val_dir.exists():
        print(f"\n❌ ERROR: Validation directory not found: {val_dir}")
        return 1

    # Import DQN modules
    print("\nLoading modules...")
    try:
        from dsp100k.src.dqn.env import DQNTradingEnv
        from dsp100k.src.dqn.agent import DQNAgent
        from dsp100k.src.dqn.trainer import DQNTrainer, TrainingConfig
    except ImportError as e:
        print(f"\n❌ ERROR: Failed to import DQN modules: {e}")
        print("  Make sure PYTHONPATH includes the project root")
        return 1

    # Create environments
    print("\nCreating environments...")

    # Training environment (with apply_constraint=False - agent handles it)
    train_env = DQNTradingEnv(
        data_dir=str(train_dir),
        window_size=args.window_size,
        k_per_side=args.k_per_side,
        turnover_cost=args.turnover_cost,
        decision_interval=args.decision_interval,
        apply_constraint=False,  # Agent handles top-K constraint
    )

    # Validation environment
    val_env = DQNTradingEnv(
        data_dir=str(val_dir),
        window_size=args.window_size,
        k_per_side=args.k_per_side,
        turnover_cost=args.turnover_cost,
        decision_interval=args.decision_interval,
        apply_constraint=False,  # Agent handles top-K constraint
    )

    print(f"  Train dates: {len(train_env.available_dates)}")
    print(f"  Val dates: {len(val_env.available_dates)}")
    print(f"  Symbols: {train_env.symbols}")
    print(f"  Trading window: {train_env.start_minute} - {train_env.end_minute} (RTH minutes)")
    print(f"  Decision interval: {args.decision_interval} min ({train_env.num_decision_points} decisions/day)")

    # Create agent
    print("\nCreating DQN agent...")
    agent = DQNAgent(
        num_symbols=train_env.num_symbols,
        window_size=args.window_size,
        num_features=30,  # Feature count from Gate 1
        portfolio_size=21,  # Portfolio state size from Gate 1
        num_actions=5,
        k_per_side=args.k_per_side,
        conviction_threshold=args.conviction_threshold,
        switch_margin=args.switch_margin,  # Gate 2.7b: churn control
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay,
        target_update_freq=args.target_update,
        device=args.device,
    )

    print(f"  Device: {agent.device}")
    print(f"  Model parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # Create TensorBoard writer
    writer = None
    if HAS_TENSORBOARD:
        tensorboard_dir = log_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        print(f"  TensorBoard: {tensorboard_dir}")
    else:
        print("  TensorBoard: Not available (install with: pip install tensorboard)")

    # Create trainer
    config = TrainingConfig(
        num_episodes=args.episodes,
        warmup_episodes=args.warmup,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        patience=args.patience,
        verbose=args.verbose,
    )

    trainer = DQNTrainer(
        train_env=train_env,
        val_env=val_env,
        agent=agent,
        config=config,
        tensorboard_writer=writer,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training parameters summary
    print("\nTraining parameters:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Eval frequency: {args.eval_freq}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Epsilon: {args.epsilon_start} → {args.epsilon_end} over {args.epsilon_decay:,} steps")
    print(f"  Target update: every {args.target_update:,} steps")
    print(f"  Conviction threshold: {args.conviction_threshold}")
    print(f"  Switch margin: {args.switch_margin}")
    print(f"  Decision interval: {args.decision_interval} min")

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    try:
        results = trainer.train()

        # Print results
        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)
        print(f"  Episodes: {results['episodes']}")
        print(f"  Total steps: {results['total_steps']:,}")
        print(f"  Time: {results['elapsed_time']/3600:.2f} hours")
        print(f"  Best val Sharpe: {results['best_val_sharpe']:.4f}")
        print(f"\nFinal metrics:")
        for key, value in results['final_metrics'].items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")

        # Kill test check
        print("\n" + "=" * 60)
        print("Kill Test Results")
        print("=" * 60)
        val_sharpe = results['final_metrics']['val_sharpe']
        if val_sharpe > 0:
            print(f"  ✅ PASS: Val Sharpe = {val_sharpe:.4f} > 0")
        else:
            print(f"  ❌ FAIL: Val Sharpe = {val_sharpe:.4f} <= 0")

        if val_sharpe > 0.5:
            print(f"  🎯 STRETCH GOAL: Val Sharpe = {val_sharpe:.4f} > 0.5")

        # Save final results
        results_path = log_dir / "training_results.json"
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer._save_checkpoint("interrupted_model.pt")
        print("Checkpoint saved: interrupted_model.pt")
        return 1

    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    sys.exit(main())
