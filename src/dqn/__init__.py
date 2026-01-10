"""
DQN Intraday Trading Module

Components:
- state_builder: Feature computation for 30-dim rolling window
- reward: Reward function with transaction costs
- constraints: Top-K portfolio constraint layer
- env: DQNTradingEnv gymnasium environment
- baselines: Baseline policies for kill test validation

Gate 2 (Training Pipeline):
- model: Dueling Double DQN network architecture
- replay_buffer: Prioritized Experience Replay
- agent: DQN agent with top-K constraint
- trainer: Training loop with evaluation
"""

from .state_builder import StateBuilder
from .reward import compute_reward, compute_portfolio_reward
from .constraints import apply_topk_constraint
from .env import DQNTradingEnv, make_env
from .baselines import (
    BasePolicy,
    FlatPolicy,
    RandomPolicy,
    MomentumPolicy,
    MeanReversionPolicy,
    BuyAndHoldPolicy,
    RSIMomentumPolicy,
    get_baseline_policies,
    get_kill_test_policies,
)

# Gate 2: Training components
from .model import DuelingDQN, create_model
from .replay_buffer import (
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
    create_replay_buffer,
)
from .agent import DQNAgent, create_agent
from .trainer import DQNTrainer, TrainingConfig, create_trainer

__all__ = [
    # State builder
    "StateBuilder",
    # Reward
    "compute_reward",
    "compute_portfolio_reward",
    # Constraints
    "apply_topk_constraint",
    # Environment
    "DQNTradingEnv",
    "make_env",
    # Baseline policies
    "BasePolicy",
    "FlatPolicy",
    "RandomPolicy",
    "MomentumPolicy",
    "MeanReversionPolicy",
    "BuyAndHoldPolicy",
    "RSIMomentumPolicy",
    "get_baseline_policies",
    "get_kill_test_policies",
    # Gate 2: Model
    "DuelingDQN",
    "create_model",
    # Gate 2: Replay Buffer
    "PrioritizedReplayBuffer",
    "UniformReplayBuffer",
    "create_replay_buffer",
    # Gate 2: Agent
    "DQNAgent",
    "create_agent",
    # Gate 2: Trainer
    "DQNTrainer",
    "TrainingConfig",
    "create_trainer",
]
