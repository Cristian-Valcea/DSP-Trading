"""
Reward Function for DQN Intraday Trading

Implements dense per-step rewards based on:
- Portfolio P&L from position × log_return
- Turnover costs (10 bps default one-way)

Key design:
- Uses PREVIOUS position to compute reward (no look-ahead)
- Transaction costs applied on absolute position change
- No terminal bonus/penalty (handled by environment)
"""

import numpy as np


def compute_reward(
    position_prev: float,
    position_curr: float,
    log_return: float,
    turnover_cost: float = 0.0010,
) -> float:
    """
    Compute reward for a single symbol.

    The reward is the P&L from holding the previous position, minus
    transaction costs for any position change.

    Args:
        position_prev: Previous position {-1, -0.5, 0, +0.5, +1}
        position_curr: Current position after action
        log_return: log(price_t / price_{t-1})
        turnover_cost: One-way transaction cost (default 10 bps = 0.001)

    Returns:
        reward: P&L - transaction costs
    """
    # P&L from holding previous position
    # (position_prev * log_return) approximates (position * (price_t/price_{t-1} - 1))
    pnl = position_prev * log_return

    # Transaction cost for position change (one-way cost on turnover)
    turnover = abs(position_curr - position_prev)
    cost = turnover * turnover_cost

    return pnl - cost


def compute_portfolio_reward(
    positions_prev: np.ndarray,
    positions_curr: np.ndarray,
    log_returns: np.ndarray,
    weights: np.ndarray | None = None,
    turnover_cost: float = 0.0010,
) -> float:
    """
    Compute total portfolio reward across all symbols.

    Args:
        positions_prev: (num_symbols,) previous positions
        positions_curr: (num_symbols,) current positions
        log_returns: (num_symbols,) log returns for each symbol
        weights: (num_symbols,) optional symbol weights (default: equal weight)
        turnover_cost: One-way transaction cost

    Returns:
        total_reward: Sum of symbol rewards (optionally weighted)
    """
    num_symbols = len(positions_prev)

    if weights is None:
        weights = np.ones(num_symbols)

    total_reward = 0.0
    for i in range(num_symbols):
        symbol_reward = compute_reward(
            positions_prev[i],
            positions_curr[i],
            log_returns[i],
            turnover_cost,
        )
        total_reward += weights[i] * symbol_reward

    return total_reward


def action_to_position(action: int) -> float:
    """
    Convert discrete action to position value.

    Args:
        action: Integer action (0-4)

    Returns:
        position: Float position value

    Action mapping:
        0: FLAT      → 0.0
        1: LONG_50   → +0.5
        2: LONG_100  → +1.0
        3: SHORT_50  → -0.5
        4: SHORT_100 → -1.0
    """
    ACTION_TO_POSITION = {
        0: 0.0,     # FLAT
        1: 0.5,     # LONG_50
        2: 1.0,     # LONG_100
        3: -0.5,    # SHORT_50
        4: -1.0,    # SHORT_100
    }
    return ACTION_TO_POSITION.get(action, 0.0)


def actions_to_positions(actions: np.ndarray) -> np.ndarray:
    """
    Convert array of discrete actions to positions.

    Args:
        actions: (num_symbols,) integer actions

    Returns:
        positions: (num_symbols,) float positions
    """
    return np.array([action_to_position(a) for a in actions])
