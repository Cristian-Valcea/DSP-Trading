"""
State Builder for DQN Intraday Trading

Computes 30-dimensional feature vectors per symbol per minute bar.
Features are designed for intraday momentum/mean-reversion detection.

Feature Categories:
- Returns (4): 1m, 5m, 15m, 30m log returns
- Volume (2): volume ratio, dollar volume
- Microstructure (2): bar range, close position in range
- Technical (4): VWAP deviation, RSI, EMA ratio, ATR
- Volatility (3): high-low range, realized vol 5m, realized vol 15m
- Time (4): time sin/cos, day-of-week sin/cos
- Extended Hours (3): overnight gap, premarket return/volume
- Cross-Asset (8): SPY/QQQ returns, relative performance
"""

from typing import Optional
import numpy as np
import pandas as pd


class StateBuilder:
    """
    Build 30-dimensional state vectors for DQN trading.

    The state builder maintains rolling statistics and computes all features
    for each symbol at each minute bar.

    Usage:
        builder = StateBuilder(symbols=["AAPL", "AMZN", ...])
        builder.reset(data_dict)  # data_dict[symbol] = DataFrame
        features = builder.get_features(minute_idx)  # (num_symbols, 30)
    """

    # Feature indices for reference
    FEATURE_NAMES = [
        "log_return_1m",      # 0
        "log_return_5m",      # 1
        "log_return_15m",     # 2
        "log_return_30m",     # 3
        "volume_ratio",       # 4
        "dollar_volume",      # 5
        "bar_range_bps",      # 6
        "vwap_deviation",     # 7
        "rsi_14",             # 8
        "ema_ratio_20_60",    # 9
        "atr_14",             # 10
        "high_low_range",     # 11
        "close_vs_high",      # 12
        "time_sin",           # 13
        "time_cos",           # 14
        "day_of_week_sin",    # 15
        "day_of_week_cos",    # 16
        "overnight_gap",      # 17
        "premarket_return",   # 18
        "premarket_vol_ratio",# 19
        "spy_return_1m",      # 20
        "spy_return_15m",     # 21
        "qqq_return_1m",      # 22
        "qqq_return_15m",     # 23
        "realized_vol_5m",    # 24
        "realized_vol_15m",   # 25
        "return_vs_spy_1m",   # 26
        "return_vs_spy_15m",  # 27
        "return_vs_qqq_1m",   # 28
        "return_vs_qqq_15m",  # 29
    ]

    NUM_FEATURES = 30

    def __init__(
        self,
        symbols: list[str],
        volume_lookback: int = 20,
        rsi_period: int = 14,
        atr_period: int = 14,
        ema_short: int = 20,
        ema_long: int = 60,
    ):
        """
        Initialize state builder.

        Args:
            symbols: List of symbols to track
            volume_lookback: Lookback for volume SMA
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
            ema_short: Short EMA period
            ema_long: Long EMA period
        """
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        self.volume_lookback = volume_lookback
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.ema_short = ema_short
        self.ema_long = ema_long

        # Data storage (populated on reset)
        self.data: dict[str, pd.DataFrame] = {}
        self.precomputed_features: dict[str, np.ndarray] = {}

    def reset(self, data_dict: dict[str, pd.DataFrame], premarket_data: Optional[dict] = None):
        """
        Reset state builder with new day's data.

        Args:
            data_dict: Dictionary mapping symbol -> DataFrame with columns:
                       [timestamp, open, high, low, close, volume]
            premarket_data: Optional dict with premarket return/volume per symbol
        """
        self.data = data_dict
        self.premarket_data = premarket_data or {}

        # Precompute all features for each symbol
        for symbol in self.symbols:
            if symbol in data_dict:
                self.precomputed_features[symbol] = self._precompute_symbol_features(
                    symbol, data_dict[symbol]
                )

    def _precompute_symbol_features(self, symbol: str, df: pd.DataFrame) -> np.ndarray:
        """
        Precompute all features for a symbol's day of data.

        Returns:
            (num_bars, 30) array of features
        """
        n = len(df)
        features = np.zeros((n, self.NUM_FEATURES), dtype=np.float32)

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values.astype(np.float64)
        open_ = df["open"].values

        # Handle edge cases
        close = np.where(close <= 0, 1e-8, close)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        # 1. Log returns (1m, 5m, 15m, 30m)
        log_returns = np.log(close / prev_close)
        log_returns[0] = 0
        features[:, 0] = log_returns

        for i, lookback in enumerate([5, 15, 30], start=1):
            cum_ret = np.zeros(n)
            for j in range(n):
                start_idx = max(0, j - lookback + 1)
                cum_ret[j] = np.sum(log_returns[start_idx : j + 1])
            features[:, i] = cum_ret

        # 2. Volume features
        vol_sma = self._rolling_mean(volume, self.volume_lookback)
        vol_sma = np.where(vol_sma <= 0, 1, vol_sma)
        features[:, 4] = volume / vol_sma  # volume_ratio

        dollar_vol = close * volume
        # Scale dollar volume to reasonable range (divide by 1M)
        features[:, 5] = dollar_vol / 1e6  # dollar_volume

        # 3. Microstructure
        bar_range = (high - low) / close * 10000  # bps
        features[:, 6] = bar_range  # bar_range_bps

        hl_range = high - low
        hl_range = np.where(hl_range <= 0, 1e-8, hl_range)
        features[:, 12] = (close - low) / hl_range  # close_vs_high (0 = at low, 1 = at high)

        # 4. Technical indicators
        # VWAP deviation
        cum_pv = np.cumsum(close * volume)
        cum_vol = np.cumsum(volume)
        cum_vol = np.where(cum_vol <= 0, 1, cum_vol)
        vwap = cum_pv / cum_vol
        features[:, 7] = (close - vwap) / vwap  # vwap_deviation

        # RSI
        features[:, 8] = self._compute_rsi(close, self.rsi_period)

        # EMA ratio
        ema_short = self._ema(close, self.ema_short)
        ema_long = self._ema(close, self.ema_long)
        ema_long = np.where(ema_long <= 0, 1e-8, ema_long)
        features[:, 9] = ema_short / ema_long  # ema_ratio_20_60

        # ATR
        features[:, 10] = self._compute_atr(high, low, close, self.atr_period) / close

        # High-low range
        features[:, 11] = (high - low) / close  # high_low_range

        # 5. Time features
        timestamps = pd.to_datetime(df["timestamp"])
        # Minute of day (0-389 for RTH)
        minutes = timestamps.dt.hour * 60 + timestamps.dt.minute - 9 * 60 - 30  # Offset from 9:30
        minutes = np.clip(minutes, 0, 389)
        features[:, 13] = np.sin(2 * np.pi * minutes / 390)  # time_sin
        features[:, 14] = np.cos(2 * np.pi * minutes / 390)  # time_cos

        dow = timestamps.dt.dayofweek.values
        features[:, 15] = np.sin(2 * np.pi * dow / 5)  # day_of_week_sin
        features[:, 16] = np.cos(2 * np.pi * dow / 5)  # day_of_week_cos

        # 6. Overnight gap
        first_open = open_[0]
        yesterday_close = prev_close[0]  # Should be passed in, using prev_close[0] as proxy
        if yesterday_close > 0:
            features[:, 17] = (first_open - yesterday_close) / yesterday_close
        else:
            features[:, 17] = 0

        # 7. Premarket features (if available)
        if symbol in self.premarket_data:
            pm = self.premarket_data[symbol]
            features[:, 18] = pm.get("return", 0)  # premarket_return
            features[:, 19] = pm.get("volume_ratio", 0)  # premarket_vol_ratio

        # 8. Realized volatility
        features[:, 24] = self._rolling_std(log_returns, 5)  # realized_vol_5m
        features[:, 25] = self._rolling_std(log_returns, 15)  # realized_vol_15m

        # Note: Cross-asset features (20-23, 26-29) are computed in get_features()
        # because they require SPY/QQQ data

        return features

    def get_features(self, minute_idx: int) -> np.ndarray:
        """
        Get features for all symbols at a specific minute.

        Args:
            minute_idx: Index into the day's data (0 = 9:30, 60 = 10:30, etc.)

        Returns:
            (num_symbols, 30) array of features
        """
        features = np.zeros((self.num_symbols, self.NUM_FEATURES), dtype=np.float32)

        # Get SPY/QQQ returns for cross-asset features
        spy_ret_1m = 0.0
        spy_ret_15m = 0.0
        qqq_ret_1m = 0.0
        qqq_ret_15m = 0.0

        if "SPY" in self.precomputed_features:
            spy_features = self.precomputed_features["SPY"]
            if minute_idx < len(spy_features):
                spy_ret_1m = spy_features[minute_idx, 0]
                spy_ret_15m = spy_features[minute_idx, 2]

        if "QQQ" in self.precomputed_features:
            qqq_features = self.precomputed_features["QQQ"]
            if minute_idx < len(qqq_features):
                qqq_ret_1m = qqq_features[minute_idx, 0]
                qqq_ret_15m = qqq_features[minute_idx, 2]

        for i, symbol in enumerate(self.symbols):
            if symbol in self.precomputed_features:
                symbol_features = self.precomputed_features[symbol]
                if minute_idx < len(symbol_features):
                    features[i, :] = symbol_features[minute_idx]

                    # Add cross-asset features
                    features[i, 20] = spy_ret_1m
                    features[i, 21] = spy_ret_15m
                    features[i, 22] = qqq_ret_1m
                    features[i, 23] = qqq_ret_15m

                    # Relative performance
                    sym_ret_1m = features[i, 0]
                    sym_ret_15m = features[i, 2]
                    features[i, 26] = sym_ret_1m - spy_ret_1m
                    features[i, 27] = sym_ret_15m - spy_ret_15m
                    features[i, 28] = sym_ret_1m - qqq_ret_1m
                    features[i, 29] = sym_ret_15m - qqq_ret_15m

        return features

    def get_rolling_window(self, minute_idx: int, window_size: int = 60) -> np.ndarray:
        """
        Get rolling window of features for all symbols.

        Args:
            minute_idx: Current minute index
            window_size: Number of bars in the window

        Returns:
            (window_size, num_symbols, 30) array
        """
        window = np.zeros((window_size, self.num_symbols, self.NUM_FEATURES), dtype=np.float32)

        start_idx = max(0, minute_idx - window_size + 1)
        for i, idx in enumerate(range(start_idx, minute_idx + 1)):
            offset = window_size - (minute_idx - start_idx + 1) + i
            if offset >= 0:
                window[offset] = self.get_features(idx)

        return window

    # Helper methods

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.zeros_like(arr, dtype=np.float64)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = np.mean(arr[start : i + 1])
        return result

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.zeros_like(arr, dtype=np.float64)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            if i - start + 1 >= 2:
                result[i] = np.std(arr[start : i + 1], ddof=1)
        return result

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        """Compute exponential moving average."""
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(arr, dtype=np.float64)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI (Relative Strength Index)."""
        n = len(close)
        rsi = np.full(n, 50.0)  # Default to neutral

        if n < period + 1:
            return rsi

        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Initial average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - 100 / (1 + rs)
        elif avg_gain > 0:
            rsi[period] = 100
        else:
            rsi[period] = 50

        # Subsequent values using smoothed average
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - 100 / (1 + rs)
            elif avg_gain > 0:
                rsi[i] = 100
            else:
                rsi[i] = 50

        # Normalize to 0-1 range
        return rsi / 100

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute Average True Range."""
        n = len(close)
        atr = np.zeros(n)

        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )

        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr
