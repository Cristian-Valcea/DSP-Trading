"""
DQN Training Loop for Multi-Symbol Trading

Features:
- Episode-based training with random date sampling
- Periodic validation on held-out dates
- Checkpoint saving with best model tracking
- TensorBoard logging
- Early stopping on plateau

Usage:
    trainer = DQNTrainer(train_env, val_env, agent, config)
    trainer.train(num_episodes=50000)
"""

import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Training parameters
    num_episodes: int = 50000
    warmup_episodes: int = 1000
    eval_freq: int = 1000
    eval_episodes: int = 50
    checkpoint_freq: int = 5000

    # Early stopping
    patience: int = 10  # Eval cycles without improvement
    min_delta: float = 0.01  # Minimum Sharpe improvement

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Logging
    log_freq: int = 100  # Episodes
    verbose: int = 1  # 0=silent, 1=progress, 2=debug


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""

    total_reward: float = 0.0
    num_steps: int = 0
    num_trades: int = 0
    final_pnl: float = 0.0
    action_entropy: float = 0.0
    long_count: int = 0
    short_count: int = 0
    flat_count: int = 0


@dataclass
class EvalMetrics:
    """Evaluation metrics over multiple episodes."""

    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_pnl: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    mean_entropy: float = 0.0
    action_distribution: Dict[int, float] = field(default_factory=dict)


class DQNTrainer:
    """
    Training loop for DQN trading agent.

    Handles:
    - Episode collection with exploration
    - Agent training
    - Periodic evaluation
    - Checkpoint management
    - Logging
    """

    def __init__(
        self,
        train_env,
        val_env,
        agent,
        config: Optional[TrainingConfig] = None,
        tensorboard_writer=None,
    ):
        """
        Initialize trainer.

        Args:
            train_env: Training environment (DQNTradingEnv)
            val_env: Validation environment (DQNTradingEnv)
            agent: DQN agent
            config: Training configuration
            tensorboard_writer: Optional TensorBoard SummaryWriter
        """
        self.train_env = train_env
        self.val_env = val_env
        self.agent = agent
        self.config = config or TrainingConfig()
        self.writer = tensorboard_writer

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_val_sharpe = -np.inf
        self.patience_counter = 0
        self.training_history = []

        # Recent metrics for logging
        self.recent_rewards = deque(maxlen=100)
        self.recent_pnls = deque(maxlen=100)
        self.recent_entropies = deque(maxlen=100)

        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        num_episodes: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            num_episodes: Override config num_episodes
            callback: Optional callback called after each episode

        Returns:
            Training results dict
        """
        num_episodes = num_episodes or self.config.num_episodes
        start_time = time.time()

        if self.config.verbose >= 1:
            print(f"Starting DQN training for {num_episodes} episodes")
            print(f"  Warmup: {self.config.warmup_episodes} episodes")
            print(f"  Eval frequency: {self.config.eval_freq} episodes")
            print(f"  Checkpoint frequency: {self.config.checkpoint_freq} episodes")

        for ep in range(num_episodes):
            self.episode = ep

            # Run training episode
            stats = self._run_episode(training=True)
            self.recent_rewards.append(stats.total_reward)
            self.recent_pnls.append(stats.final_pnl)
            self.recent_entropies.append(stats.action_entropy)

            # NOTE: Training now happens per env step inside _run_episode() (Gate 2.7a fix)
            # Old code trained once per episode, causing severe under-training.

            # Logging
            if ep % self.config.log_freq == 0 and ep > 0:
                self._log_progress(ep, num_episodes, start_time)

            # Evaluation
            if ep % self.config.eval_freq == 0 and ep >= self.config.warmup_episodes:
                eval_metrics = self._evaluate()
                should_stop = self._handle_eval_results(eval_metrics, ep)

                if should_stop:
                    if self.config.verbose >= 1:
                        print(f"\nEarly stopping at episode {ep}")
                    break

            # Checkpoint
            if ep % self.config.checkpoint_freq == 0 and ep > 0:
                self._save_checkpoint(f"checkpoint_{ep}.pt")

            # Callback
            if callback:
                callback(ep, stats)

        # Final evaluation
        final_metrics = self._evaluate()
        self._save_checkpoint("final_model.pt")
        self._save_training_history()

        elapsed = time.time() - start_time
        results = {
            "episodes": self.episode + 1,
            "total_steps": self.total_steps,
            "elapsed_time": elapsed,
            "best_val_sharpe": self.best_val_sharpe,
            "final_metrics": {
                "val_sharpe": final_metrics.sharpe,
                "val_mean_reward": final_metrics.mean_reward,
                "val_win_rate": final_metrics.win_rate,
            },
        }

        if self.config.verbose >= 1:
            print(f"\nTraining complete!")
            print(f"  Episodes: {results['episodes']}")
            print(f"  Total steps: {results['total_steps']}")
            print(f"  Time: {elapsed/3600:.1f} hours")
            print(f"  Best val Sharpe: {self.best_val_sharpe:.4f}")

        return results

    def _run_episode(self, training: bool = True) -> EpisodeStats:
        """
        Run a single episode.

        Args:
            training: Whether to store transitions and apply exploration

        Returns:
            Episode statistics
        """
        env = self.train_env if training else self.val_env
        stats = EpisodeStats()

        obs, info = env.reset()
        done = False

        prev_actions = None

        while not done:
            # Select action
            # Gate 2.7b: Pass current positions for switch margin (churn control)
            current_positions = env.positions if hasattr(env, "positions") else None
            actions, q_values, entropy = self.agent.select_action(
                rolling_window=obs["rolling_window"],
                portfolio_state=obs["portfolio_state"],
                apply_constraint=True,
                explore=training,           # greedy eval when training=False
                count_env_step=training,    # don't decay epsilon on eval episodes
                current_positions=current_positions,  # Gate 2.7b: for switch margin
            )

            stats.action_entropy += entropy

            # Count action types
            for a in actions:
                if a == 0:
                    stats.flat_count += 1
                elif a in [1, 2]:
                    stats.long_count += 1
                elif a in [3, 4]:
                    stats.short_count += 1

            # Count trades (action changes)
            if prev_actions is not None:
                stats.num_trades += np.sum(actions != prev_actions)
            prev_actions = actions.copy()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Store transition (if training)
            if training:
                self.agent.store_transition(
                    rolling_window=obs["rolling_window"],
                    portfolio_state=obs["portfolio_state"],
                    actions=actions,
                    reward=reward,
                    next_rolling_window=next_obs["rolling_window"],
                    next_portfolio_state=next_obs["portfolio_state"],
                    done=done,
                )

                # Train step after each env step (Gate 2.7a fix)
                # Standard DQN: 1 gradient update per env step
                if self.episode >= self.config.warmup_episodes:
                    loss = self.agent.train_step()
                    if loss is not None and self.writer:
                        self.writer.add_scalar("train/loss", loss, self.total_steps)

            stats.total_reward += reward
            stats.num_steps += 1
            self.total_steps += 1

            obs = next_obs

        stats.final_pnl = info.get("daily_pnl", 0.0)
        stats.action_entropy /= max(stats.num_steps, 1)

        return stats

    def _evaluate(self) -> EvalMetrics:
        """
        Evaluate agent on validation set.

        Returns:
            Evaluation metrics
        """
        rewards = []
        pnls = []
        entropies = []
        action_counts = {i: 0 for i in range(5)}

        # Run evaluation episodes
        for _ in range(self.config.eval_episodes):
            stats = self._run_episode(training=False)
            rewards.append(stats.total_reward)
            pnls.append(stats.final_pnl)
            entropies.append(stats.action_entropy)

            # Count actions
            total = stats.long_count + stats.short_count + stats.flat_count
            if total > 0:
                # Approximate action distribution from episode stats
                action_counts[0] += stats.flat_count
                action_counts[1] += stats.long_count // 2
                action_counts[2] += stats.long_count - stats.long_count // 2
                action_counts[3] += stats.short_count // 2
                action_counts[4] += stats.short_count - stats.short_count // 2

        rewards = np.array(rewards)
        pnls = np.array(pnls)

        # Compute Sharpe ratio (annualized assuming 252 trading days)
        if rewards.std() > 0:
            sharpe = (rewards.mean() / rewards.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Win rate
        win_rate = np.mean(pnls > 0)

        # Action distribution
        total_actions = sum(action_counts.values())
        action_dist = {k: v / max(total_actions, 1) for k, v in action_counts.items()}

        metrics = EvalMetrics(
            mean_reward=rewards.mean(),
            std_reward=rewards.std(),
            mean_pnl=pnls.mean(),
            sharpe=sharpe,
            win_rate=win_rate,
            mean_entropy=np.mean(entropies),
            action_distribution=action_dist,
        )

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("eval/mean_reward", metrics.mean_reward, self.episode)
            self.writer.add_scalar("eval/sharpe", metrics.sharpe, self.episode)
            self.writer.add_scalar("eval/win_rate", metrics.win_rate, self.episode)
            self.writer.add_scalar("eval/mean_entropy", metrics.mean_entropy, self.episode)

        return metrics

    def _handle_eval_results(self, metrics: EvalMetrics, episode: int) -> bool:
        """
        Handle evaluation results.

        Args:
            metrics: Evaluation metrics
            episode: Current episode

        Returns:
            True if training should stop (early stopping)
        """
        # Log results
        self.training_history.append({
            "episode": episode,
            "val_sharpe": metrics.sharpe,
            "val_mean_reward": metrics.mean_reward,
            "val_win_rate": metrics.win_rate,
            "val_mean_entropy": metrics.mean_entropy,
        })

        if self.config.verbose >= 1:
            print(f"\n📊 Eval @ Episode {episode}:")
            print(f"   Sharpe: {metrics.sharpe:.4f}")
            print(f"   Mean Reward: {metrics.mean_reward:.6f}")
            print(f"   Win Rate: {metrics.win_rate:.2%}")
            print(f"   Entropy: {metrics.mean_entropy:.4f}")
            print(f"   Action Dist: {metrics.action_distribution}")

        # Check for improvement
        if metrics.sharpe > self.best_val_sharpe + self.config.min_delta:
            self.best_val_sharpe = metrics.sharpe
            self.patience_counter = 0
            self._save_checkpoint("best_model.pt")

            if self.config.verbose >= 1:
                print(f"   ✅ New best model! Sharpe: {metrics.sharpe:.4f}")
        else:
            self.patience_counter += 1

            if self.config.verbose >= 1:
                print(f"   Patience: {self.patience_counter}/{self.config.patience}")

        # Check early stopping on strong signal or plateau
        if metrics.sharpe > 1.0:
            if self.config.verbose >= 1:
                print(f"   🎯 Strong signal found (Sharpe > 1.0)")
            return True

        if self.patience_counter >= self.config.patience:
            return True

        return False

    def _log_progress(self, episode: int, total: int, start_time: float):
        """Log training progress."""
        if len(self.recent_rewards) == 0:
            return

        elapsed = time.time() - start_time
        eps_per_sec = episode / elapsed if elapsed > 0 else 0

        mean_reward = np.mean(self.recent_rewards)
        mean_pnl = np.mean(self.recent_pnls)
        mean_entropy = np.mean(self.recent_entropies)

        metrics = self.agent.get_metrics()

        if self.config.verbose >= 1:
            print(
                f"Episode {episode}/{total} | "
                f"Reward: {mean_reward:.4f} | "
                f"PnL: {mean_pnl:.6f} | "
                f"Entropy: {mean_entropy:.2f} | "
                f"ε: {metrics.get('epsilon', 0):.3f} | "
                f"Buffer: {metrics.get('buffer_size', 0)} | "
                f"Speed: {eps_per_sec:.1f} ep/s"
            )

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar("train/mean_reward", mean_reward, episode)
            self.writer.add_scalar("train/mean_pnl", mean_pnl, episode)
            self.writer.add_scalar("train/entropy", mean_entropy, episode)
            self.writer.add_scalar("train/epsilon", metrics.get("epsilon", 0), episode)
            self.writer.add_scalar("train/buffer_size", metrics.get("buffer_size", 0), episode)

            if "loss_mean" in metrics:
                self.writer.add_scalar("train/loss_mean", metrics["loss_mean"], episode)
            if "q_mean" in metrics:
                self.writer.add_scalar("train/q_mean", metrics["q_mean"], episode)

    def _save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.checkpoint_dir / filename
        self.agent.save(str(path))

        # Also save training state
        state_path = self.checkpoint_dir / f"{filename}.state.json"
        state = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "best_val_sharpe": self.best_val_sharpe,
            "patience_counter": self.patience_counter,
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        if self.config.verbose >= 2:
            print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        path = self.checkpoint_dir / filename
        self.agent.load(str(path))

        # Load training state
        state_path = self.checkpoint_dir / f"{filename}.state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            self.episode = state.get("episode", 0)
            self.total_steps = state.get("total_steps", 0)
            self.best_val_sharpe = state.get("best_val_sharpe", -np.inf)
            self.patience_counter = state.get("patience_counter", 0)

        if self.config.verbose >= 1:
            print(f"Loaded checkpoint: {path}")

    def _save_training_history(self):
        """Save training history."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)


def create_trainer(
    train_env,
    val_env,
    agent,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    **kwargs,
) -> DQNTrainer:
    """
    Factory function to create DQN trainer.

    Args:
        train_env: Training environment
        val_env: Validation environment
        agent: DQN agent
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        **kwargs: Additional config parameters

    Returns:
        DQNTrainer instance
    """
    config = TrainingConfig(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        **kwargs,
    )

    return DQNTrainer(train_env, val_env, agent, config)


# Test code
if __name__ == "__main__":
    print("DQN Trainer module ready")
    print("Use create_trainer() to instantiate")
