"""
DQN Trading Environment for Intraday Multi-Symbol Trading

A Gymnasium environment for training DQN agents on intraday trading.
Supports 9 symbols with 5 actions each, top-K constraint, and transaction costs.

Key Features:
- 30-dimensional rolling window features per symbol
- 21-dimensional portfolio state
- Dense rewards with transaction costs
- Trading window: 10:31 ET - 14:00 ET (forced flat at end)
- Top-K constraint (default K=3 positions per side)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import random
import json
from datetime import datetime, time

from dsp100k.src.dqn.state_builder import StateBuilder
from dsp100k.src.dqn.reward import (
    compute_portfolio_reward,
    actions_to_positions,
    action_to_position,
)
from dsp100k.src.dqn.constraints import apply_topk_constraint


# Default DQN Universe
DQN_UNIVERSE = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "SPY", "TSLA"]


class DQNTradingEnv(gym.Env):
    """
    Intraday DQN trading environment.

    Observation Space (Dict):
        - rolling_window: (window_size, num_symbols, 30) - last N minutes of features
        - portfolio_state: (21,) - positions (9) + entry_returns (9) + aggregates (3)

    Action Space:
        MultiDiscrete([5] * 9) - 5 actions per symbol
        After top-K constraint: only K longs and K shorts kept

    Reward:
        Sum of per-symbol (position × log_return - turnover_cost)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dir: str,
        symbols: list[str] | None = None,
        window_size: int = 60,
        target_gross: float = 0.10,
        k_per_side: int = 3,
        w_max: float | None = None,
        turnover_cost: float = 0.0010,
        start_minute: int = 61,    # 10:31 ET = minute 61 of RTH
        end_minute: int = 270,      # 14:00 ET = minute 270 of RTH
        apply_constraint: bool = False,  # Agent handles top-K constraint
        premarket_cache_dir: str | None = None,
        render_mode: str | None = None,
    ):
        """
        Initialize the DQN trading environment.

        Args:
            data_dir: Directory containing {symbol}_{split}.parquet files
            symbols: List of symbols to trade (default: DQN_UNIVERSE)
            window_size: Number of bars in rolling window (default: 60)
            target_gross: Target gross exposure (default: 10%)
            k_per_side: Max positions per side (default: 3)
            w_max: Max weight per symbol (default: target_gross / (2 * k_per_side))
            turnover_cost: Transaction cost per unit turnover (default: 10 bps)
            start_minute: First decision minute in RTH (default: 61 = 10:31 ET)
            end_minute: Last decision minute in RTH (default: 270 = 14:00 ET)
            apply_constraint: Whether to apply top-K constraint (default: False - agent handles it)
            render_mode: Rendering mode (default: None)
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.symbols = symbols if symbols else DQN_UNIVERSE
        self.num_symbols = len(self.symbols)
        self.window_size = window_size
        self.target_gross = target_gross
        self.k_per_side = k_per_side
        self.w_max = w_max if w_max else target_gross / (2 * k_per_side)
        self.turnover_cost = turnover_cost
        self.start_minute = start_minute
        self.end_minute = end_minute
        self.apply_constraint = apply_constraint
        self.render_mode = render_mode
        self.premarket_cache_dir = self._resolve_premarket_cache_dir(premarket_cache_dir)

        # State builder for feature computation
        self.state_builder = StateBuilder(symbols=self.symbols)

        # Define observation space
        self.observation_space = spaces.Dict({
            "rolling_window": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(window_size, self.num_symbols, 30),
                dtype=np.float32,
            ),
            "portfolio_state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(21,),  # 9 positions + 9 entry_returns + 3 aggregates
                dtype=np.float32,
            ),
        })

        # Define action space (5 actions per symbol)
        self.action_space = spaces.MultiDiscrete([5] * self.num_symbols)

        # Load data index (list of available dates)
        self.available_dates = self._load_available_dates()
        if not self.available_dates:
            raise ValueError(f"No valid dates found in {data_dir}")

        # Episode state (initialized in reset)
        self.current_date: str | None = None
        self.current_minute: int = 0
        self.positions: np.ndarray = np.zeros(self.num_symbols, dtype=np.float32)
        self.entry_prices: np.ndarray = np.zeros(self.num_symbols, dtype=np.float32)
        self.daily_pnl: float = 0.0
        self.day_data: dict[str, pd.DataFrame] = {}

    def _load_available_dates(self) -> list[str]:
        """Load list of available trading dates from data files."""
        dates = set()

        for symbol in self.symbols:
            # Try different file patterns
            for pattern in [f"{symbol.lower()}_*.parquet", f"{symbol}_*.parquet"]:
                for filepath in self.data_dir.glob(pattern):
                    df = pd.read_parquet(filepath)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    symbol_dates = df["timestamp"].dt.date.unique()
                    if not dates:
                        dates = set(str(d) for d in symbol_dates)
                    else:
                        dates &= set(str(d) for d in symbol_dates)
                    break  # Found file for this symbol

        return sorted(list(dates))

    def _load_day_data(self, date_str: str) -> dict[str, pd.DataFrame]:
        """Load data for all symbols for a specific date."""
        return self._load_day_data_with_prior_close(date_str)[0]

    def _load_day_data_with_prior_close(
        self,
        date_str: str,
        prior_date_str: str | None = None,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
        """
        Load data for all symbols for a specific date, plus prior session close.

        Args:
            date_str: Trading date (YYYY-MM-DD)
            prior_date_str: Prior trading date (YYYY-MM-DD) for overnight gap; if None, gap=0
        """
        data = {}
        prior_close_by_symbol: dict[str, float] = {}

        date_obj = pd.Timestamp(date_str).date()
        prior_date_obj = pd.Timestamp(prior_date_str).date() if prior_date_str else None

        for symbol in self.symbols:
            for pattern in [f"{symbol.lower()}_*.parquet", f"{symbol}_*.parquet"]:
                for filepath in self.data_dir.glob(pattern):
                    df = pd.read_parquet(filepath)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    day_df = df[df["timestamp"].dt.date == date_obj].copy()

                    if not day_df.empty:
                        # Sort by timestamp and reset index
                        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
                        data[symbol] = day_df

                    if prior_date_obj is not None:
                        prev_df = df[df["timestamp"].dt.date == prior_date_obj]
                        if not prev_df.empty:
                            prev_df = prev_df.sort_values("timestamp")
                            prior_close_by_symbol[symbol] = float(prev_df.iloc[-1]["close"])
                    break

        return data, prior_close_by_symbol

    def _resolve_premarket_cache_dirs(self, premarket_cache_dir: str | None) -> list[Path]:
        """
        Resolve premarket cache directories (may be multiple locations).

        Returns list of existing cache directories to search.
        Priority: explicit path > dqn_premarket_cache > sleeve_im (Gate 2.7c)
        """
        candidates: list[Path] = []
        project_root = Path(__file__).resolve().parents[2]  # .../dsp100k

        # Explicit path (absolute or relative)
        if premarket_cache_dir:
            candidates.append(Path(premarket_cache_dir))

        # Gate 2.7c: New DQN-specific premarket cache (for 2021-2022 backfill)
        candidates.append(project_root / "data" / "dqn_premarket_cache")
        candidates.append(Path.cwd() / "dsp100k" / "data" / "dqn_premarket_cache")
        candidates.append(Path.cwd() / "data" / "dqn_premarket_cache")

        # Sleeve IM cache (existing 2023+ data)
        candidates.append(project_root / "data" / "sleeve_im" / "minute_bars")
        candidates.append(Path.cwd() / "dsp100k" / "data" / "sleeve_im" / "minute_bars")
        candidates.append(Path.cwd() / "data" / "sleeve_im" / "minute_bars")

        return [p for p in candidates if p.exists() and p.is_dir()]

    def _resolve_premarket_cache_dir(self, premarket_cache_dir: str | None) -> Path | None:
        """Resolve premarket cache directory (backward compatibility)."""
        dirs = self._resolve_premarket_cache_dirs(premarket_cache_dir)
        return dirs[0] if dirs else None

    def _load_premarket_summary(
        self,
        symbol: str,
        date_str: str,
        day_df: pd.DataFrame,
    ) -> dict:
        """
        Build premarket summary features from premarket cache.

        Searches multiple cache directories (Gate 2.7c: dqn_premarket_cache + sleeve_im).
        Uses bars in [04:00, 09:30) ET.
        """
        # Search all cache directories for the file
        cache_dirs = self._resolve_premarket_cache_dirs(None)
        if not cache_dirs:
            return {}

        cache_path = None
        for cache_dir in cache_dirs:
            candidate = cache_dir / symbol.upper() / f"{date_str}.json"
            if candidate.exists():
                cache_path = candidate
                break

        if cache_path is None:
            return {}

        try:
            with open(cache_path, "r") as f:
                payload = json.load(f)
        except Exception:
            return {}

        bars = payload.get("bars", [])
        if not isinstance(bars, list) or len(bars) == 0:
            return {}

        pm_start = time(4, 0)
        pm_end = time(9, 30)

        pm_open = None
        pm_close = None
        pm_volume = 0

        for bar in bars:
            try:
                ts = datetime.fromisoformat(bar["timestamp"])
            except Exception:
                continue
            t = ts.time()
            if t < pm_start or t >= pm_end:
                continue

            if pm_open is None:
                pm_open = float(bar.get("open") or bar.get("close") or 0.0)
            pm_close = float(bar.get("close") or pm_close or 0.0)
            pm_volume += int(bar.get("volume") or 0)

        if pm_open is None or pm_close is None or pm_open <= 0:
            pm_return = 0.0
        else:
            pm_return = pm_close / pm_open - 1.0

        # Scale premarket volume against first-hour RTH volume (09:30-10:30)
        ts = pd.to_datetime(day_df["timestamp"])
        rth_mask = (ts.dt.time >= time(9, 30)) & (ts.dt.time < time(10, 31))
        rth_first_hour_vol = float(day_df.loc[rth_mask, "volume"].sum())
        denom = rth_first_hour_vol if rth_first_hour_vol > 0 else 1.0

        return {
            "return": float(pm_return),
            "volume_ratio": float(pm_volume / denom),
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset environment to start of a trading day.

        Args:
            seed: Random seed
            options: Optional dict with "date" key to specify date

        Returns:
            observation: Initial observation
            info: Episode info
        """
        super().reset(seed=seed)

        # Select date
        if options and "date" in options:
            self.current_date = options["date"]
        else:
            self.current_date = random.choice(self.available_dates)

        # Determine prior trading date (within this split) for overnight gap
        prior_date_str = None
        try:
            idx = self.available_dates.index(self.current_date)
            if idx > 0:
                prior_date_str = self.available_dates[idx - 1]
        except ValueError:
            prior_date_str = None

        # Load day's data
        self.day_data, prior_close_by_symbol = self._load_day_data_with_prior_close(
            self.current_date, prior_date_str=prior_date_str
        )

        if not self.day_data:
            # If no data, try another date
            self.available_dates.remove(self.current_date)
            return self.reset(seed=seed)

        # Build premarket summary features (if cache exists)
        premarket_data: dict[str, dict] = {}
        for symbol, df in self.day_data.items():
            summary = self._load_premarket_summary(symbol, self.current_date, df)
            if summary:
                premarket_data[symbol] = summary

        # Initialize state builder
        self.state_builder.reset(
            self.day_data,
            premarket_data=premarket_data,
            prior_close_by_symbol=prior_close_by_symbol,
        )

        # Reset episode state
        self.current_minute = self.start_minute
        self.positions = np.zeros(self.num_symbols, dtype=np.float32)
        self.entry_prices = np.zeros(self.num_symbols, dtype=np.float32)
        self.daily_pnl = 0.0

        obs = self._get_observation()
        info = {
            "date": self.current_date,
            "minute": self.current_minute,
            "positions": self.positions.copy(),
        }

        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute trading action and advance one minute.

        Args:
            action: (num_symbols,) array of actions (0-4)

        Returns:
            observation: Next state
            reward: Portfolio reward
            terminated: True if episode over (end of day)
            truncated: Always False
            info: Additional info
        """
        # Apply top-K constraint if enabled
        if self.apply_constraint:
            # For constraint, we need Q-values, but we only have actions
            # Create synthetic Q-values where selected action has highest value
            q_values = np.zeros((self.num_symbols, 5), dtype=np.float32)
            for i, a in enumerate(action):
                q_values[i, a] = 1.0  # Selected action gets highest value
            action = apply_topk_constraint(q_values, k=self.k_per_side)

        # Convert actions to positions
        new_positions = actions_to_positions(action)

        # Get current prices
        prices = self._get_current_prices()
        prev_prices = self._get_prices_at_minute(self.current_minute - 1)

        # Compute log returns
        log_returns = np.zeros(self.num_symbols, dtype=np.float32)
        for i in range(self.num_symbols):
            if prev_prices[i] > 0 and prices[i] > 0:
                log_returns[i] = np.log(prices[i] / prev_prices[i])

        # Compute reward using PREVIOUS positions (no look-ahead)
        reward = compute_portfolio_reward(
            positions_prev=self.positions,
            positions_curr=new_positions,
            log_returns=log_returns,
            turnover_cost=self.turnover_cost,
        )

        # Update entry prices for new positions
        for i in range(self.num_symbols):
            if self.positions[i] == 0 and new_positions[i] != 0:
                # Opening new position
                self.entry_prices[i] = prices[i]
            elif new_positions[i] == 0:
                # Closing position
                self.entry_prices[i] = 0

        # Update positions
        self.positions = new_positions.astype(np.float32)
        self.daily_pnl += reward

        # Advance time
        self.current_minute += 1

        # Check if episode is done
        terminated = self.current_minute >= self.end_minute

        # Force flatten at end of day
        if terminated:
            # Compute final flatten cost
            flatten_cost = np.sum(np.abs(self.positions)) * self.turnover_cost
            self.daily_pnl -= flatten_cost
            reward -= flatten_cost
            self.positions = np.zeros(self.num_symbols, dtype=np.float32)

        obs = self._get_observation()
        info = {
            "date": self.current_date,
            "minute": self.current_minute,
            "positions": self.positions.copy(),
            "daily_pnl": self.daily_pnl,
            "action": action.tolist(),
        }

        return obs, reward, terminated, False, info

    def _get_observation(self) -> dict:
        """Build current observation."""
        # Rolling window features
        rolling_window = self.state_builder.get_rolling_window(
            self.current_minute, self.window_size
        )

        # Portfolio state
        portfolio_state = np.zeros(21, dtype=np.float32)
        portfolio_state[:9] = self.positions

        # Entry returns
        prices = self._get_current_prices()
        for i in range(self.num_symbols):
            if self.positions[i] != 0 and self.entry_prices[i] > 0 and prices[i] > 0:
                portfolio_state[9 + i] = np.log(prices[i] / self.entry_prices[i])

        # Aggregates
        gross = np.sum(np.abs(self.positions)) * self.w_max
        net = np.sum(self.positions) * self.w_max
        portfolio_state[18] = gross / self.target_gross  # Normalized gross
        portfolio_state[19] = net
        portfolio_state[20] = self.daily_pnl

        return {
            "rolling_window": rolling_window,
            "portfolio_state": portfolio_state,
        }

    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all symbols."""
        return self._get_prices_at_minute(self.current_minute)

    def _get_prices_at_minute(self, minute_idx: int) -> np.ndarray:
        """Get prices at a specific minute for all symbols."""
        prices = np.zeros(self.num_symbols, dtype=np.float32)

        for i, symbol in enumerate(self.symbols):
            if symbol in self.day_data:
                df = self.day_data[symbol]
                if minute_idx >= 0 and minute_idx < len(df):
                    prices[i] = df.iloc[minute_idx]["close"]
                elif minute_idx < 0:
                    prices[i] = df.iloc[0]["close"]
                else:
                    prices[i] = df.iloc[-1]["close"]

        return prices

    def render(self):
        """Render current state."""
        if self.render_mode == "human":
            print(f"Date: {self.current_date}, Minute: {self.current_minute}")
            print(f"Positions: {self.positions}")
            print(f"Daily P&L: {self.daily_pnl:.6f}")

    def close(self):
        """Cleanup resources."""
        pass

    @property
    def num_trading_minutes(self) -> int:
        """Number of trading minutes per episode."""
        return self.end_minute - self.start_minute


def make_env(
    data_dir: str,
    split: str = "train",
    **kwargs,
) -> DQNTradingEnv:
    """
    Factory function to create DQN trading environment.

    Args:
        data_dir: Base data directory
        split: Data split to use ("train", "val", "dev_test", "holdout")
        **kwargs: Additional arguments to DQNTradingEnv

    Returns:
        DQNTradingEnv instance
    """
    split_dir = Path(data_dir) / f"dqn_{split}"
    return DQNTradingEnv(data_dir=str(split_dir), **kwargs)
