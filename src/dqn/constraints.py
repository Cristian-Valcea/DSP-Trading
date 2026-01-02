"""
Portfolio Constraint Layer for DQN Intraday Trading

Implements the top-K constraint that limits positions to K longs and K shorts
based on conviction (Q-value advantage over FLAT).

This enforces gross exposure budgets without modifying the DQN action selection
logic. The constraint is applied AFTER Q-values are computed.
"""

import numpy as np


def apply_topk_constraint(
    q_values: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    Apply top-K constraint to Q-value based action selection.

    For each symbol, the best action is selected based on Q-values.
    Then, only the top-K most confident long positions and top-K most
    confident short positions are kept; all others are forced to FLAT.

    Args:
        q_values: (num_symbols, num_actions) Q-values
                  where actions are [FLAT, LONG_50, LONG_100, SHORT_50, SHORT_100]
        k: Number of positions to keep per side (default 3)

    Returns:
        actions: (num_symbols,) final action indices after constraint
    """
    num_symbols = q_values.shape[0]

    # Step 1: Select best action per symbol
    best_actions = q_values.argmax(axis=1)  # (num_symbols,)

    # Step 2: Compute conviction = Q(best) - Q(FLAT)
    # Conviction measures how much better the best action is vs doing nothing
    q_flat = q_values[:, 0]  # Q-value for action 0 (FLAT)
    q_best = q_values.max(axis=1)
    conviction = q_best - q_flat  # (num_symbols,)

    # Step 3: Identify long and short candidates
    # Long actions: 1 (LONG_50) or 2 (LONG_100)
    # Short actions: 3 (SHORT_50) or 4 (SHORT_100)
    is_long = (best_actions == 1) | (best_actions == 2)
    is_short = (best_actions == 3) | (best_actions == 4)

    long_indices = np.where(is_long)[0]
    short_indices = np.where(is_short)[0]

    # Step 4: Select top-K by conviction for each side
    top_k_longs = set()
    top_k_shorts = set()

    if len(long_indices) > 0:
        long_convictions = conviction[long_indices]
        # Get indices that would sort by conviction (descending)
        sorted_order = np.argsort(-long_convictions)
        top_k_long_indices = long_indices[sorted_order[:k]]
        top_k_longs = set(top_k_long_indices)

    if len(short_indices) > 0:
        short_convictions = conviction[short_indices]
        sorted_order = np.argsort(-short_convictions)
        top_k_short_indices = short_indices[sorted_order[:k]]
        top_k_shorts = set(top_k_short_indices)

    # Step 5: Build final actions (keep top-K, force others to FLAT)
    final_actions = np.zeros(num_symbols, dtype=np.int32)

    for i in range(num_symbols):
        if i in top_k_longs or i in top_k_shorts:
            final_actions[i] = best_actions[i]
        else:
            final_actions[i] = 0  # FLAT

    return final_actions


def compute_gross_exposure(
    positions: np.ndarray,
    w_max: float = 0.0167,
) -> float:
    """
    Compute gross exposure from positions.

    Args:
        positions: (num_symbols,) position values {-1, -0.5, 0, +0.5, +1}
        w_max: Maximum weight per symbol (default 1.67%)

    Returns:
        gross: Total gross exposure as fraction of portfolio
    """
    return np.sum(np.abs(positions)) * w_max


def compute_net_exposure(
    positions: np.ndarray,
    w_max: float = 0.0167,
) -> float:
    """
    Compute net exposure from positions.

    Args:
        positions: (num_symbols,) position values
        w_max: Maximum weight per symbol

    Returns:
        net: Net exposure (positive = net long, negative = net short)
    """
    return np.sum(positions) * w_max


def validate_portfolio_constraints(
    positions: np.ndarray,
    k: int = 3,
    w_max: float = 0.0167,
    target_gross: float = 0.10,
    tolerance: float = 0.05,
) -> dict:
    """
    Validate that portfolio satisfies all constraints.

    Args:
        positions: (num_symbols,) position values
        k: Max positions per side
        w_max: Max weight per symbol
        target_gross: Target gross exposure
        tolerance: Allowed deviation from target

    Returns:
        dict with validation results
    """
    gross = compute_gross_exposure(positions, w_max)
    net = compute_net_exposure(positions, w_max)

    num_longs = np.sum(positions > 0)
    num_shorts = np.sum(positions < 0)

    checks = {
        "gross_exposure": gross,
        "net_exposure": net,
        "num_longs": int(num_longs),
        "num_shorts": int(num_shorts),
        "max_gross_ok": gross <= target_gross * (1 + tolerance),
        "max_longs_ok": num_longs <= k,
        "max_shorts_ok": num_shorts <= k,
        "all_valid": True,
    }

    checks["all_valid"] = (
        checks["max_gross_ok"] and
        checks["max_longs_ok"] and
        checks["max_shorts_ok"]
    )

    return checks
