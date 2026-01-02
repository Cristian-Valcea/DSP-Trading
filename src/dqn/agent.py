"""
DQN Agent with Top-K Portfolio Constraint

The agent applies top-K constraint using actual Q-value conviction,
not synthetic values. This ensures the constraint respects model confidence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from dsp100k.src.dqn.model import DuelingDQN, create_model
from dsp100k.src.dqn.replay_buffer import (
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
    create_replay_buffer,
)


class DQNAgent:
    """
    Double DQN Agent with Top-K Portfolio Constraint.

    Features:
    - Double DQN (separate policy and target networks)
    - Dueling architecture (value + advantage streams)
    - Top-K constraint using Q-value conviction
    - Epsilon-greedy exploration with decay
    - Prioritized experience replay
    """

    def __init__(
        self,
        num_symbols: int = 9,
        window_size: int = 60,
        num_features: int = 30,
        portfolio_size: int = 21,
        num_actions: int = 5,
        hidden_size: int = 128,
        k_per_side: int = 3,
        conviction_threshold: float = 0.0,
        switch_margin: float = 0.0,  # Gate 2.7b: Only change position if Q(new) - Q(current) > margin
        # Training params
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        # Exploration params
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        # Replay buffer params
        buffer_size: int = 500_000,
        batch_size: int = 256,
        prioritized_replay: bool = True,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        # Other
        target_update_freq: int = 1000,
        gradient_clip: float = 10.0,
        device: str = "auto",
    ):
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.num_symbols = num_symbols
        self.num_actions = num_actions
        self.k_per_side = k_per_side
        self.conviction_threshold = float(conviction_threshold)
        self.switch_margin = float(switch_margin)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip

        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        # NOTE:
        # - env_steps drives epsilon (exploration) decay.
        # - train_steps counts gradient updates (used for target updates, logging).
        self.env_steps = 0
        self.steps = 0  # train_steps (kept for backward compatibility)

        # Create networks
        self.policy_net = create_model(
            num_symbols=num_symbols,
            window_size=window_size,
            num_features=num_features,
            portfolio_size=portfolio_size,
            num_actions=num_actions,
            hidden_size=hidden_size,
            device=self.device,
        )

        self.target_net = create_model(
            num_symbols=num_symbols,
            window_size=window_size,
            num_features=num_features,
            portfolio_size=portfolio_size,
            num_actions=num_actions,
            hidden_size=hidden_size,
            device=self.device,
        )

        # Copy weights to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Loss function
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # Replay buffer
        self.replay_buffer = create_replay_buffer(
            capacity=buffer_size,
            prioritized=prioritized_replay,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )

        # Metrics tracking
        self.train_losses = []
        self.q_values_history = []

    @property
    def epsilon(self) -> float:
        """Current epsilon value with linear decay over environment interaction steps."""
        progress = min(self.env_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def apply_topk_constraint(
        self,
        q_values: np.ndarray,
        greedy_actions: np.ndarray,
    ) -> np.ndarray:
        """
        Apply top-K constraint using Q-value conviction.

        Args:
            q_values: (num_symbols, num_actions) Q-values
            greedy_actions: (num_symbols,) greedy action per symbol

        Returns:
            constrained_actions: (num_symbols,) actions after top-K constraint
        """
        actions = greedy_actions.copy()

        # Compute conviction = Q(greedy) - Q(flat)
        # Higher conviction = more confident the action is better than flat
        conviction = np.zeros(self.num_symbols)
        for i in range(self.num_symbols):
            conviction[i] = q_values[i, greedy_actions[i]] - q_values[i, 0]

        # Identify longs (actions 1-2) and shorts (actions 3-4)
        long_mask = (actions == 1) | (actions == 2)
        short_mask = (actions == 3) | (actions == 4)

        # Keep top-K longs by conviction
        long_indices = np.where(long_mask)[0]
        if len(long_indices) > self.k_per_side:
            long_convictions = conviction[long_indices]
            sorted_idx = np.argsort(long_convictions)
            # Keep only top-K (highest conviction)
            drop_indices = long_indices[sorted_idx[:-self.k_per_side]]
            actions[drop_indices] = 0  # Force FLAT

        # Keep top-K shorts by conviction
        short_indices = np.where(short_mask)[0]
        if len(short_indices) > self.k_per_side:
            short_convictions = conviction[short_indices]
            sorted_idx = np.argsort(short_convictions)
            # Keep only top-K (highest conviction)
            drop_indices = short_indices[sorted_idx[:-self.k_per_side]]
            actions[drop_indices] = 0  # Force FLAT

        return actions

    def select_action(
        self,
        rolling_window: np.ndarray,
        portfolio_state: np.ndarray,
        apply_constraint: bool = True,
        explore: bool = True,
        count_env_step: bool = True,
        current_positions: Optional[np.ndarray] = None,  # Gate 2.7b: For switch margin
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Select actions using epsilon-greedy policy with top-K constraint.

        Args:
            rolling_window: (window, num_symbols, features)
            portfolio_state: (portfolio_size,)
            apply_constraint: Whether to apply top-K constraint
            explore: Whether to apply epsilon-greedy exploration
            count_env_step: Whether to increment env_steps (for epsilon decay)
            current_positions: (num_symbols,) current position per symbol (-1, 0, 1)
                              Used for switch margin churn control (Gate 2.7b)

        Returns:
            actions: (num_symbols,) selected actions
            q_values: (num_symbols, num_actions) Q-values
            entropy: Action entropy (for logging)
        """
        # Convert to tensors
        rw = torch.FloatTensor(rolling_window).unsqueeze(0).to(self.device)
        ps = torch.FloatTensor(portfolio_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(rw, ps)[0].cpu().numpy()

        # Exploration / exploitation
        took_random_action = False
        if explore and np.random.random() < self.epsilon:
            actions = np.random.randint(0, self.num_actions, size=self.num_symbols)
            took_random_action = True
        else:
            actions = np.argmax(q_values, axis=1)

        # Apply top-K constraint
        if apply_constraint:
            actions = self.apply_topk_constraint(q_values, actions)

        # Optional churn governor:
        # Only take non-FLAT positions when the model's advantage over FLAT
        # exceeds a minimum conviction threshold.
        #
        # Applied only when not taking a purely random exploratory action.
        if (not took_random_action) and (self.conviction_threshold > 0.0):
            flat_q = q_values[:, 0]
            chosen_q = q_values[np.arange(self.num_symbols), actions]
            conviction = chosen_q - flat_q
            low_conviction = (actions != 0) & (conviction < self.conviction_threshold)
            actions[low_conviction] = 0

        # Gate 2.7b: Switch margin (hysteresis)
        # Only change position if Q(new) - Q(current) > switch_margin
        # This prevents flip-flopping on tiny Q-value differences.
        if (not took_random_action) and (self.switch_margin > 0.0) and (current_positions is not None):
            for i in range(self.num_symbols):
                current_pos = current_positions[i]
                new_action = actions[i]

                # Map current position to its "hold" action
                # Position: -1 (short) -> action 3 or 4 (SHORT)
                # Position: 0 (flat) -> action 0 (FLAT)
                # Position: +1 (long) -> action 1 or 2 (LONG)
                if current_pos == 0:
                    current_action = 0  # FLAT
                elif current_pos > 0:
                    # Long position - use action 1 (SMALL_LONG) as representative
                    current_action = 1
                else:
                    # Short position - use action 3 (SMALL_SHORT) as representative
                    current_action = 3

                # If new action differs from current position direction, check margin
                if new_action != current_action:
                    q_current = q_values[i, current_action]
                    q_new = q_values[i, new_action]
                    if q_new - q_current < self.switch_margin:
                        # Not enough improvement - keep current position
                        actions[i] = current_action

        # Compute action entropy (for monitoring)
        action_counts = np.bincount(actions, minlength=self.num_actions)
        action_probs = action_counts / self.num_symbols
        action_probs = action_probs[action_probs > 0]
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        if count_env_step:
            self.env_steps += 1

        return actions.astype(np.int32), q_values, entropy

    def store_transition(
        self,
        rolling_window: np.ndarray,
        portfolio_state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_rolling_window: np.ndarray,
        next_portfolio_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.add(
            rolling_window=rolling_window,
            portfolio_state=portfolio_state,
            actions=actions,
            reward=reward,
            next_rolling_window=next_rolling_window,
            next_portfolio_state=next_portfolio_state,
            done=done,
        )

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            loss: Training loss, or None if not enough samples
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        self.steps += 1  # train_steps

        # Sample batch
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        rolling_window = torch.FloatTensor(batch["rolling_window"]).to(self.device)
        portfolio_state = torch.FloatTensor(batch["portfolio_state"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_rolling_window = torch.FloatTensor(batch["next_rolling_window"]).to(self.device)
        next_portfolio_state = torch.FloatTensor(batch["next_portfolio_state"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current per-symbol Q-values
        current_q = self.policy_net(rolling_window, portfolio_state)  # (batch, num_symbols, num_actions)
        # Gather Q-values for taken actions: (batch, num_symbols)
        current_q_actions = current_q.gather(2, actions.unsqueeze(2)).squeeze(2)
        # Portfolio Q is the sum of per-symbol Qs for the joint action.
        current_q_sum = current_q_actions.sum(dim=1)  # (batch,)

        # Double DQN target
        with torch.no_grad():
            # Use policy net to select actions (per symbol), then apply top-K constraint
            next_q_policy = self.policy_net(next_rolling_window, next_portfolio_state)  # (batch, num_symbols, num_actions)
            next_q_policy_np = next_q_policy.cpu().numpy()
            greedy_next_actions = np.argmax(next_q_policy_np, axis=2)  # (batch, num_symbols)

            constrained_next_actions = np.zeros_like(greedy_next_actions, dtype=np.int64)
            for b in range(greedy_next_actions.shape[0]):
                constrained_next_actions[b] = self.apply_topk_constraint(
                    next_q_policy_np[b], greedy_next_actions[b]
                )

            next_actions = torch.from_numpy(constrained_next_actions).to(self.device)

            # Use target net to evaluate actions
            next_q_target = self.target_net(next_rolling_window, next_portfolio_state)
            next_q_values = next_q_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)

            # Portfolio target Q is the sum of per-symbol target Qs.
            next_q_sum = next_q_values.sum(dim=1)  # (batch,)

            # TD target for portfolio Q(s, a_vec)
            target_q = rewards + (1 - dones) * self.gamma * next_q_sum  # (batch,)

        # Compute loss with importance sampling weights
        td_errors = self.loss_fn(current_q_sum, target_q)  # (batch,)
        loss = (td_errors * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Update priorities
        if indices is not None:
            with torch.no_grad():
                td_error_numpy = td_errors.cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_error_numpy)

        # Soft update target network every step (Gate 2.7a fix)
        # With 1 gradient update per env step, soft updates should happen frequently.
        # tau=0.005 means each update moves target 0.5% toward policy.
        self.soft_update_target()

        # Track metrics
        self.train_losses.append(loss.item())
        self.q_values_history.append(current_q.mean().item())

        return loss.item()

    def soft_update_target(self):
        """Soft update target network weights."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def hard_update_target(self):
        """Hard update target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        """Save agent state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,  # train_steps (backward compatible)
            "env_steps": self.env_steps,
            "epsilon": self.epsilon,
            "conviction_threshold": self.conviction_threshold,
            "switch_margin": self.switch_margin,
            "train_losses": self.train_losses[-1000:],  # Keep last 1000
            "q_values_history": self.q_values_history[-1000:],
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Backward compatibility:
        # - old checkpoints stored only "steps" (train_steps).
        # - new checkpoints store "env_steps" separately.
        self.steps = checkpoint.get("steps", 0)
        self.env_steps = checkpoint.get("env_steps", 0)
        self.conviction_threshold = checkpoint.get(
            "conviction_threshold", self.conviction_threshold
        )
        self.switch_margin = checkpoint.get("switch_margin", self.switch_margin)
        self.train_losses = checkpoint.get("train_losses", [])
        self.q_values_history = checkpoint.get("q_values_history", [])

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        metrics = {
            "train_steps": self.steps,
            "env_steps": self.env_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
        }

        if self.train_losses:
            metrics["loss_mean"] = np.mean(self.train_losses[-100:])
            metrics["loss_std"] = np.std(self.train_losses[-100:])

        if self.q_values_history:
            metrics["q_mean"] = np.mean(self.q_values_history[-100:])

        if hasattr(self.replay_buffer, "beta"):
            metrics["per_beta"] = self.replay_buffer.beta

        return metrics


def create_agent(
    num_symbols: int = 9,
    k_per_side: int = 3,
    device: str = "auto",
    **kwargs,
) -> DQNAgent:
    """
    Factory function to create DQN agent.

    Args:
        num_symbols: Number of symbols to trade
        k_per_side: Max positions per side
        device: Device to use
        **kwargs: Additional agent parameters

    Returns:
        DQNAgent instance
    """
    return DQNAgent(
        num_symbols=num_symbols,
        k_per_side=k_per_side,
        device=device,
        **kwargs,
    )


# Test code
if __name__ == "__main__":
    print("Testing DQNAgent...")

    agent = create_agent(device="cpu")
    print(f"Agent created with {agent.steps} steps")
    print(f"Initial epsilon: {agent.epsilon:.4f}")

    # Test action selection
    rolling_window = np.random.randn(60, 9, 30).astype(np.float32)
    portfolio_state = np.random.randn(21).astype(np.float32)

    actions, q_values, entropy = agent.select_action(rolling_window, portfolio_state)
    print(f"Actions: {actions}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Entropy: {entropy:.4f}")

    # Test constraint
    print(f"\nTop-K constraint test:")
    print(f"  Actions before: {actions}")
    long_count = np.sum((actions == 1) | (actions == 2))
    short_count = np.sum((actions == 3) | (actions == 4))
    print(f"  Long positions: {long_count} (max: {agent.k_per_side})")
    print(f"  Short positions: {short_count} (max: {agent.k_per_side})")

    # Test training (with fake data)
    for i in range(100):
        agent.store_transition(
            rolling_window=np.random.randn(60, 9, 30).astype(np.float32),
            portfolio_state=np.random.randn(21).astype(np.float32),
            actions=np.random.randint(0, 5, size=9),
            reward=np.random.randn(),
            next_rolling_window=np.random.randn(60, 9, 30).astype(np.float32),
            next_portfolio_state=np.random.randn(21).astype(np.float32),
            done=i == 99,
        )

    # Train step (needs 256 samples min)
    for i in range(200):
        agent.store_transition(
            rolling_window=np.random.randn(60, 9, 30).astype(np.float32),
            portfolio_state=np.random.randn(21).astype(np.float32),
            actions=np.random.randint(0, 5, size=9),
            reward=np.random.randn(),
            next_rolling_window=np.random.randn(60, 9, 30).astype(np.float32),
            next_portfolio_state=np.random.randn(21).astype(np.float32),
            done=i == 199,
        )

    loss = agent.train_step()
    print(f"\nTraining loss: {loss:.6f}")
    print(f"Metrics: {agent.get_metrics()}")

    print("✅ Agent test passed!")
