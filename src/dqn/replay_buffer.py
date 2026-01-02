"""
Prioritized Experience Replay Buffer for DQN Training

Features:
- Sum-tree for O(log n) sampling
- Priority updates based on TD-error
- Importance sampling weights for bias correction
- Beta annealing from 0.4 → 1.0
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """Single experience transition."""
    rolling_window: np.ndarray      # (window, num_symbols, features)
    portfolio_state: np.ndarray     # (portfolio_size,)
    actions: np.ndarray             # (num_symbols,)
    reward: float
    next_rolling_window: np.ndarray
    next_portfolio_state: np.ndarray
    done: bool


class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.

    Supports O(log n) operations for:
    - Update priority
    - Sample by cumulative priority
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf node for given cumulative value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """Add data with given priority."""
        tree_idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at given tree index."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, value: float) -> Tuple[int, float, Any]:
        """
        Get data for cumulative value.

        Returns:
            tree_idx: Tree index for updates
            priority: Current priority
            data: Stored data
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1

        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    def min_priority(self) -> float:
        """Return minimum priority in tree."""
        if self.n_entries == 0:
            return 0.0

        # Leaf nodes start at capacity - 1
        leaf_priorities = self.tree[self.capacity - 1:self.capacity - 1 + self.n_entries]
        return np.min(leaf_priorities[leaf_priorities > 0]) if np.any(leaf_priorities > 0) else 1e-6


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with importance sampling.

    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta_start: Initial importance sampling exponent
        beta_frames: Number of frames to anneal beta to 1.0
    """

    def __init__(
        self,
        capacity: int = 500_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Current beta value (annealed from beta_start to 1.0)."""
        fraction = min(self.frame / self.beta_frames, 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def __len__(self) -> int:
        return self.tree.n_entries

    def add(
        self,
        rolling_window: np.ndarray,
        portfolio_state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_rolling_window: np.ndarray,
        next_portfolio_state: np.ndarray,
        done: bool,
    ):
        """Add transition with max priority (will be updated after first use)."""
        transition = Transition(
            rolling_window=rolling_window.copy(),
            portfolio_state=portfolio_state.copy(),
            actions=actions.copy(),
            reward=reward,
            next_rolling_window=next_rolling_window.copy(),
            next_portfolio_state=next_portfolio_state.copy(),
            done=done,
        )

        # New transitions get max priority to ensure they're sampled
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch with priority-weighted sampling.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            batch: Dict with batched transitions
            indices: Tree indices for priority updates
            weights: Importance sampling weights
        """
        self.frame += 1

        # Sample indices based on priority
        indices = []
        priorities = []
        transitions = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)

            tree_idx, priority, transition = self.tree.get(value)
            indices.append(tree_idx)
            priorities.append(priority)
            transitions.append(transition)

        indices = np.array(indices)
        priorities = np.array(priorities)

        # Compute importance sampling weights
        min_priority = self.tree.min_priority()
        max_weight = (min_priority * len(self) / priorities.min()) ** (-self.beta)

        weights = (len(self) * priorities / self.tree.total()) ** (-self.beta)
        weights = weights / max_weight  # Normalize

        # Build batch
        batch = {
            "rolling_window": np.stack([t.rolling_window for t in transitions]),
            "portfolio_state": np.stack([t.portfolio_state for t in transitions]),
            "actions": np.stack([t.actions for t in transitions]),
            "rewards": np.array([t.reward for t in transitions], dtype=np.float32),
            "next_rolling_window": np.stack([t.next_rolling_window for t in transitions]),
            "next_portfolio_state": np.stack([t.next_portfolio_state for t in transitions]),
            "dones": np.array([t.done for t in transitions], dtype=np.float32),
        }

        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.

        Args:
            indices: Tree indices from sample()
            td_errors: Absolute TD-errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self) >= min_size


class UniformReplayBuffer:
    """
    Simple uniform replay buffer (no prioritization).

    Useful as baseline comparison.
    """

    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        rolling_window: np.ndarray,
        portfolio_state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_rolling_window: np.ndarray,
        next_portfolio_state: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        transition = Transition(
            rolling_window=rolling_window.copy(),
            portfolio_state=portfolio_state.copy(),
            actions=actions.copy(),
            reward=reward,
            next_rolling_window=next_rolling_window.copy(),
            next_portfolio_state=next_portfolio_state.copy(),
            done=done,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], None, np.ndarray]:
        """
        Sample batch uniformly.

        Returns:
            batch: Dict with batched transitions
            indices: None (not used for uniform)
            weights: All ones (no importance sampling)
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]

        batch = {
            "rolling_window": np.stack([t.rolling_window for t in transitions]),
            "portfolio_state": np.stack([t.portfolio_state for t in transitions]),
            "actions": np.stack([t.actions for t in transitions]),
            "rewards": np.array([t.reward for t in transitions], dtype=np.float32),
            "next_rolling_window": np.stack([t.next_rolling_window for t in transitions]),
            "next_portfolio_state": np.stack([t.next_portfolio_state for t in transitions]),
            "dones": np.array([t.done for t in transitions], dtype=np.float32),
        }

        weights = np.ones(batch_size, dtype=np.float32)

        return batch, None, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """No-op for uniform buffer."""
        pass

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self) >= min_size


def create_replay_buffer(
    capacity: int = 500_000,
    prioritized: bool = True,
    alpha: float = 0.6,
    beta_start: float = 0.4,
    beta_frames: int = 100_000,
) -> PrioritizedReplayBuffer | UniformReplayBuffer:
    """
    Factory function to create replay buffer.

    Args:
        capacity: Maximum buffer size
        prioritized: Whether to use prioritized replay
        alpha: Priority exponent (only for PER)
        beta_start: Initial IS exponent (only for PER)
        beta_frames: Frames to anneal beta (only for PER)

    Returns:
        Replay buffer instance
    """
    if prioritized:
        return PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
        )
    else:
        return UniformReplayBuffer(capacity=capacity)


# Test code
if __name__ == "__main__":
    print("Testing PrioritizedReplayBuffer...")

    buffer = create_replay_buffer(capacity=1000, prioritized=True)

    # Add some transitions
    for i in range(100):
        buffer.add(
            rolling_window=np.random.randn(60, 9, 30).astype(np.float32),
            portfolio_state=np.random.randn(21).astype(np.float32),
            actions=np.random.randint(0, 5, size=9),
            reward=np.random.randn(),
            next_rolling_window=np.random.randn(60, 9, 30).astype(np.float32),
            next_portfolio_state=np.random.randn(21).astype(np.float32),
            done=i == 99,
        )

    print(f"Buffer size: {len(buffer)}")

    # Sample batch
    batch, indices, weights = buffer.sample(batch_size=32)
    print(f"Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.4f}, {weights.max():.4f}]")

    # Update priorities
    td_errors = np.random.uniform(0, 1, size=32)
    buffer.update_priorities(indices, td_errors)

    print(f"Beta: {buffer.beta:.4f}")

    print("✅ Replay buffer test passed!")
