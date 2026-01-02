"""
DQN Intraday Trading Module

Components:
- state_builder: Feature computation for 30-dim rolling window
- reward: Reward function with transaction costs
- constraints: Top-K portfolio constraint layer
- env: DQNTradingEnv gymnasium environment
- baselines: Baseline policies for kill test validation
"""

from dsp100k.src.dqn.state_builder import StateBuilder
from dsp100k.src.dqn.reward import compute_reward, compute_portfolio_reward
from dsp100k.src.dqn.constraints import apply_topk_constraint
from dsp100k.src.dqn.env import DQNTradingEnv, make_env
from dsp100k.src.dqn.baselines import (
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
]
