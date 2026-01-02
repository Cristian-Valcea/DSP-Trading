"""
Baseline Policies for DQN Trading Environment Validation

These baseline policies are used to validate the trading environment:
1. FlatPolicy - Always FLAT (detect look-ahead bias, Sharpe ≈ 0)
2. RandomPolicy - Random actions (verify transaction costs, Sharpe < 0)
3. MomentumPolicy - Follow short-term momentum (informational)
4. MeanReversionPolicy - Fade short-term moves (informational)

Kill Test Criteria:
- FLAT policy must have Sharpe ≈ 0 (±0.1) - otherwise look-ahead bias exists
- Random policy must have Sharpe < 0 - otherwise transaction costs not working
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Abstract base class for baseline policies."""

    @abstractmethod
    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Select actions based on observation.

        Args:
            obs: Dictionary with 'rolling_window' and 'portfolio_state'

        Returns:
            actions: (num_symbols,) array of action indices (0-4)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset policy state for new episode."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging/reporting."""
        pass


class FlatPolicy(BasePolicy):
    """
    Always-FLAT Policy (Critical Sanity Check)

    Returns action 0 (FLAT) for all symbols at all times.

    Expected Result:
        - Sharpe ratio ≈ 0.0 (within ±0.1)
        - Total return ≈ 0.0
        - Max drawdown = 0.0

    Kill Condition:
        If abs(Sharpe) > 0.1, there is likely look-ahead bias in the data
        or reward calculation.
    """

    def __init__(self, num_symbols: int = 9):
        """
        Initialize FlatPolicy.

        Args:
            num_symbols: Number of symbols to trade (default: 9)
        """
        self.num_symbols = num_symbols

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Return FLAT (action 0) for all symbols."""
        return np.zeros(self.num_symbols, dtype=np.int32)

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "always_flat"


class RandomPolicy(BasePolicy):
    """
    Random Policy (Transaction Cost Validation)

    Selects actions uniformly at random from {0, 1, 2, 3, 4}.

    Expected Result:
        - Sharpe ratio < 0 (negative due to transaction costs)
        - Total return < 0 (costs dominate random entry/exit)
        - High turnover

    Kill Condition:
        If Sharpe >= 0 or return >= 0, transaction costs are not being
        applied correctly.
    """

    def __init__(self, num_symbols: int = 9, seed: int | None = None):
        """
        Initialize RandomPolicy.

        Args:
            num_symbols: Number of symbols to trade (default: 9)
            seed: Random seed for reproducibility
        """
        self.num_symbols = num_symbols
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Return random actions for all symbols."""
        return self.rng.integers(0, 5, size=self.num_symbols).astype(np.int32)

    def reset(self) -> None:
        """Reset RNG to initial seed."""
        if self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

    @property
    def name(self) -> str:
        return "random"


class MomentumPolicy(BasePolicy):
    """
    Momentum Policy (Informational Baseline)

    Goes LONG if recent return > 0, SHORT if recent return < 0, else FLAT.
    Uses 5-minute return from the rolling window.

    Expected Result:
        - Sharpe in range [-0.5, +0.5] typically
        - Behavior depends on market regime

    This is NOT a kill test - just an informational baseline.
    """

    def __init__(
        self,
        num_symbols: int = 9,
        return_feature_idx: int = 1,  # log_return_5m
        threshold: float = 0.0,
        use_full_size: bool = True,
    ):
        """
        Initialize MomentumPolicy.

        Args:
            num_symbols: Number of symbols to trade (default: 9)
            return_feature_idx: Index of return feature in rolling window (default: 1 = 5m return)
            threshold: Return threshold for signal (default: 0.0)
            use_full_size: If True, use LONG_100/SHORT_100; else use LONG_50/SHORT_50
        """
        self.num_symbols = num_symbols
        self.return_feature_idx = return_feature_idx
        self.threshold = threshold
        self.use_full_size = use_full_size

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Select actions based on momentum signal.

        LONG if 5m return > threshold, SHORT if < -threshold, else FLAT.
        """
        rolling_window = obs["rolling_window"]
        # Get most recent bar (last row in window)
        # Shape: (window_size, num_symbols, num_features)
        latest = rolling_window[-1]  # (num_symbols, num_features)

        # Get return feature for each symbol
        returns = latest[:, self.return_feature_idx]  # (num_symbols,)

        # Map returns to actions
        actions = np.zeros(self.num_symbols, dtype=np.int32)

        if self.use_full_size:
            long_action = 2   # LONG_100
            short_action = 4  # SHORT_100
        else:
            long_action = 1   # LONG_50
            short_action = 3  # SHORT_50

        for i in range(self.num_symbols):
            if returns[i] > self.threshold:
                actions[i] = long_action  # Go long on positive momentum
            elif returns[i] < -self.threshold:
                actions[i] = short_action  # Go short on negative momentum
            else:
                actions[i] = 0  # Stay flat

        return actions

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "momentum"


class MeanReversionPolicy(BasePolicy):
    """
    Mean Reversion Policy (Informational Baseline)

    Goes SHORT if recent return > 0, LONG if recent return < 0, else FLAT.
    Bets on reversal of short-term moves.

    Expected Result:
        - Sharpe in range [-0.5, +0.5] typically
        - Opposite behavior to MomentumPolicy

    This is NOT a kill test - just an informational baseline.
    """

    def __init__(
        self,
        num_symbols: int = 9,
        return_feature_idx: int = 1,  # log_return_5m
        threshold: float = 0.0,
        use_full_size: bool = True,
    ):
        """
        Initialize MeanReversionPolicy.

        Args:
            num_symbols: Number of symbols to trade (default: 9)
            return_feature_idx: Index of return feature in rolling window (default: 1 = 5m return)
            threshold: Return threshold for signal (default: 0.0)
            use_full_size: If True, use LONG_100/SHORT_100; else use LONG_50/SHORT_50
        """
        self.num_symbols = num_symbols
        self.return_feature_idx = return_feature_idx
        self.threshold = threshold
        self.use_full_size = use_full_size

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Select actions based on mean reversion signal.

        SHORT if 5m return > threshold (fade the move),
        LONG if < -threshold (fade the move), else FLAT.
        """
        rolling_window = obs["rolling_window"]
        # Get most recent bar
        latest = rolling_window[-1]  # (num_symbols, num_features)

        # Get return feature for each symbol
        returns = latest[:, self.return_feature_idx]  # (num_symbols,)

        # Map returns to actions (OPPOSITE of momentum)
        actions = np.zeros(self.num_symbols, dtype=np.int32)

        if self.use_full_size:
            long_action = 2   # LONG_100
            short_action = 4  # SHORT_100
        else:
            long_action = 1   # LONG_50
            short_action = 3  # SHORT_50

        for i in range(self.num_symbols):
            if returns[i] > self.threshold:
                actions[i] = short_action  # Short after positive move (fade)
            elif returns[i] < -self.threshold:
                actions[i] = long_action  # Long after negative move (fade)
            else:
                actions[i] = 0  # Stay flat

        return actions

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "mean_reversion"


class BuyAndHoldPolicy(BasePolicy):
    """
    Buy and Hold Policy (Informational Baseline)

    Goes LONG_100 on all symbols and holds throughout the episode.

    Expected Result:
        - Should track market performance minus transaction costs
        - Entry cost at start, exit cost at end
        - Useful benchmark for active trading strategies
    """

    def __init__(self, num_symbols: int = 9):
        """
        Initialize BuyAndHoldPolicy.

        Args:
            num_symbols: Number of symbols to trade (default: 9)
        """
        self.num_symbols = num_symbols

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Return LONG_100 (action 2) for all symbols."""
        return np.full(self.num_symbols, fill_value=2, dtype=np.int32)

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "buy_and_hold"


class RSIMomentumPolicy(BasePolicy):
    """
    RSI-Based Momentum Policy (Informational Baseline)

    Uses RSI feature (index 8) to determine overbought/oversold conditions:
    - RSI > 70: SHORT (overbought, expect reversal)
    - RSI < 30: LONG (oversold, expect reversal)
    - Otherwise: FLAT

    This is a contrarian RSI strategy, not pure momentum.
    """

    def __init__(
        self,
        num_symbols: int = 9,
        rsi_feature_idx: int = 8,  # rsi_14
        overbought: float = 70.0,
        oversold: float = 30.0,
        use_full_size: bool = True,
    ):
        """
        Initialize RSIMomentumPolicy.

        Args:
            num_symbols: Number of symbols to trade (default: 9)
            rsi_feature_idx: Index of RSI feature (default: 8)
            overbought: RSI threshold for overbought (default: 70)
            oversold: RSI threshold for oversold (default: 30)
            use_full_size: If True, use LONG_100/SHORT_100
        """
        self.num_symbols = num_symbols
        self.rsi_feature_idx = rsi_feature_idx
        self.overbought = overbought
        self.oversold = oversold
        self.use_full_size = use_full_size

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Select actions based on RSI levels."""
        rolling_window = obs["rolling_window"]
        latest = rolling_window[-1]  # (num_symbols, num_features)

        # Get RSI for each symbol
        rsi = latest[:, self.rsi_feature_idx]  # (num_symbols,)

        actions = np.zeros(self.num_symbols, dtype=np.int32)

        if self.use_full_size:
            long_action = 2   # LONG_100
            short_action = 4  # SHORT_100
        else:
            long_action = 1   # LONG_50
            short_action = 3  # SHORT_50

        for i in range(self.num_symbols):
            if rsi[i] > self.overbought:
                actions[i] = short_action  # Overbought → short
            elif rsi[i] < self.oversold:
                actions[i] = long_action  # Oversold → long
            else:
                actions[i] = 0  # Neutral → flat

        return actions

    def reset(self) -> None:
        """No state to reset."""
        pass

    @property
    def name(self) -> str:
        return "rsi_contrarian"


def get_baseline_policies(num_symbols: int = 9, seed: int = 42) -> dict[str, BasePolicy]:
    """
    Get all baseline policies for evaluation.

    Args:
        num_symbols: Number of symbols to trade
        seed: Random seed for RandomPolicy

    Returns:
        Dictionary mapping policy name to policy instance
    """
    return {
        "always_flat": FlatPolicy(num_symbols=num_symbols),
        "random": RandomPolicy(num_symbols=num_symbols, seed=seed),
        "momentum": MomentumPolicy(num_symbols=num_symbols),
        "mean_reversion": MeanReversionPolicy(num_symbols=num_symbols),
        "buy_and_hold": BuyAndHoldPolicy(num_symbols=num_symbols),
        "rsi_contrarian": RSIMomentumPolicy(num_symbols=num_symbols),
    }


def get_kill_test_policies(num_symbols: int = 9, seed: int = 42) -> dict[str, BasePolicy]:
    """
    Get only the policies required for kill tests.

    Kill Tests:
    1. always_flat: Must have |Sharpe| < 0.1 (no look-ahead bias)
    2. random: Must have Sharpe < 0 (transaction costs working)

    Args:
        num_symbols: Number of symbols to trade
        seed: Random seed for RandomPolicy

    Returns:
        Dictionary with flat and random policies
    """
    return {
        "always_flat": FlatPolicy(num_symbols=num_symbols),
        "random": RandomPolicy(num_symbols=num_symbols, seed=seed),
    }
