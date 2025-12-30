"""
Sleeve B: Cross-Asset Trend Following ETFs.

Implements the trend-following strategy for non-equity ETFs:
- Multi-horizon trend signals (1m/3m/12m-1m)
- Volatility targeting (3.5% annualized)
- Position sizing with single-name cap
- Weekly rebalancing with 10% threshold

Per SPEC_DSP_100K.md Section 2.2.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher, UniverseManager
from ..utils.config import SleeveBConfig
from ..utils.logging import get_audit_logger

logger = logging.getLogger(__name__)


@dataclass
class TrendSignal:
    """Trend signal for a single ETF."""
    symbol: str
    ret_1m: float          # 1-month return
    ret_3m: float          # 3-month return
    ret_12m_skip: float    # 12-month return with 1-month skip
    composite: float       # Weighted average signal
    signal_sign: int       # +1, 0, or -1
    as_of_date: date


@dataclass
class PositionTarget:
    """Target position for a single ETF."""
    symbol: str
    signal_weight: float   # Raw signal-based weight
    vol_scaled_weight: float  # After volatility scaling
    final_weight: float    # After cap and normalization
    target_shares: int     # Target share count
    target_notional: float # Target dollar value
    current_shares: int    # Current position
    delta_shares: int      # Change needed


@dataclass
class SleeveBAdjustment:
    """Complete adjustment plan for Sleeve B."""
    as_of_date: date
    signals: List[TrendSignal]
    positions: List[PositionTarget]
    sleeve_nav: float
    sleeve_vol_target: float
    estimated_turnover: float
    rebalance_needed: bool


class TrendSignalGenerator:
    """
    Generates trend signals for Sleeve B universe.

    Signal = 0.25 * ret_1m + 0.50 * ret_3m + 0.25 * ret_12m_skip

    Where:
    - ret_1m: 1-month trailing return
    - ret_3m: 3-month trailing return
    - ret_12m_skip: 12-month return skipping the most recent month
    """

    WEIGHTS = {
        "1m": 0.25,
        "3m": 0.50,
        "12m_skip": 0.25,
    }

    def __init__(self, data_fetcher: DataFetcher):
        """
        Initialize signal generator.

        Args:
            data_fetcher: Data fetcher for historical prices
        """
        self.fetcher = data_fetcher

    async def compute_signals(
        self,
        symbols: List[str],
        as_of_date: Optional[date] = None,
    ) -> Dict[str, TrendSignal]:
        """
        Compute trend signals for a universe.

        Args:
            symbols: List of ETF symbols
            as_of_date: Signal calculation date

        Returns:
            Dict mapping symbols to TrendSignal objects
        """
        if as_of_date is None:
            as_of_date = self.fetcher.calendar.get_latest_complete_session()

        signals: Dict[str, TrendSignal] = {}

        for symbol in symbols:
            try:
                signal_data = await self.fetcher.get_trend_signals(
                    symbol=symbol,
                    end_date=as_of_date,
                )

                if signal_data is None:
                    logger.warning(f"Could not compute signal for {symbol}")
                    continue

                # Determine signal sign
                composite = signal_data["composite_signal"]
                signal_sign = 1 if composite > 0 else (-1 if composite < 0 else 0)

                signals[symbol] = TrendSignal(
                    symbol=symbol,
                    ret_1m=signal_data["ret_1m"],
                    ret_3m=signal_data["ret_3m"],
                    ret_12m_skip=signal_data["ret_12m_skip"],
                    composite=composite,
                    signal_sign=signal_sign,
                    as_of_date=as_of_date,
                )

            except Exception as e:
                logger.error(f"Error computing signal for {symbol}: {e}")

        return signals

    def rank_signals(
        self,
        signals: Dict[str, TrendSignal],
    ) -> List[TrendSignal]:
        """
        Rank signals by strength.

        Args:
            signals: Dict of signals

        Returns:
            List of signals sorted by absolute composite strength
        """
        return sorted(
            signals.values(),
            key=lambda s: abs(s.composite),
            reverse=True,
        )


class VolatilityTargeter:
    """
    Volatility targeting for Sleeve B.

    Scales positions to achieve target volatility of 3.5% annualized.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        target_vol: float = 0.035,
        vol_lookback_days: int = 63,  # ~3 months
    ):
        """
        Initialize volatility targeter.

        Args:
            data_fetcher: Data fetcher for historical volatility
            target_vol: Target annualized volatility (0.035 = 3.5%)
            vol_lookback_days: Lookback for volatility estimation
        """
        self.fetcher = data_fetcher
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback_days

    async def get_volatilities(
        self,
        symbols: List[str],
        as_of_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """
        Get realized volatility for each symbol.

        Args:
            symbols: List of symbols
            as_of_date: Reference date

        Returns:
            Dict mapping symbols to annualized volatility
        """
        vols: Dict[str, float] = {}

        for symbol in symbols:
            vol = await self.fetcher.get_volatility(
                symbol=symbol,
                lookback_days=self.vol_lookback,
                end_date=as_of_date,
            )

            if vol is not None and vol > 0:
                vols[symbol] = vol
            else:
                logger.warning(f"Could not get volatility for {symbol}")

        return vols

    def compute_vol_weights(
        self,
        signals: Dict[str, TrendSignal],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute volatility-adjusted weights.

        Weight_i = signal_sign_i * (target_vol / vol_i) / sum(1/vol_j)

        This gives each asset equal risk contribution (inverse-vol weighting)
        then scaled to hit the target portfolio volatility.

        Args:
            signals: Signal dict
            volatilities: Volatility dict

        Returns:
            Dict mapping symbols to weights (sum of abs = 1)
        """
        # Get symbols with both signal and volatility
        valid_symbols = set(signals.keys()) & set(volatilities.keys())

        if not valid_symbols:
            return {}

        # Compute inverse-volatility weights
        inv_vols = {s: 1.0 / volatilities[s] for s in valid_symbols}
        total_inv_vol = sum(inv_vols.values())

        # Raw weights (equal risk contribution, scaled by signal sign)
        raw_weights = {}
        for symbol in valid_symbols:
            signal_sign = signals[symbol].signal_sign
            inv_vol_weight = inv_vols[symbol] / total_inv_vol
            raw_weights[symbol] = signal_sign * inv_vol_weight

        # Scale to target vol
        # Portfolio vol ≈ target_vol when sum(|w_i| * vol_i) ≈ target_vol
        total_risk = sum(abs(w) * volatilities[s] for s, w in raw_weights.items())

        if total_risk > 0:
            scale_factor = self.target_vol / total_risk
            weights = {s: w * scale_factor for s, w in raw_weights.items()}
        else:
            weights = raw_weights

        return weights


class SleeveB:
    """
    Sleeve B: Cross-Asset Trend Following.

    Complete implementation of the Sleeve B strategy:
    - Universe of non-equity ETFs
    - Multi-horizon trend signals
    - Volatility targeting (3.5% annualized)
    - Single-name caps (15% per position)
    - Weekly rebalancing with 10% threshold
    """

    def __init__(
        self,
        config: SleeveBConfig,
        data_fetcher: DataFetcher,
    ):
        """
        Initialize Sleeve B.

        Args:
            config: Sleeve B configuration
            data_fetcher: Data fetcher for market data
        """
        self.config = config
        self.fetcher = data_fetcher
        self.universe = UniverseManager(config.universe)
        self.signal_gen = TrendSignalGenerator(data_fetcher)
        self.vol_targeter = VolatilityTargeter(
            data_fetcher,
            target_vol=config.vol_target,
        )
        self.audit = get_audit_logger()

        # Current positions (symbol -> shares)
        self._positions: Dict[str, int] = {}
        self._last_rebalance: Optional[date] = None

    @property
    def symbols(self) -> List[str]:
        """Get current universe symbols."""
        return self.universe.symbols

    def set_positions(self, positions: Dict[str, int]) -> None:
        """
        Update current positions.

        Args:
            positions: Dict mapping symbols to share counts
        """
        self._positions = positions.copy()

    async def generate_adjustment(
        self,
        sleeve_nav: float,
        prices: Dict[str, float],
        as_of_date: Optional[date] = None,
        force_rebalance: bool = False,
    ) -> SleeveBAdjustment:
        """
        Generate adjustment plan for Sleeve B.

        Args:
            sleeve_nav: Net Asset Value allocated to Sleeve B
            prices: Current prices for each symbol
            as_of_date: Reference date
            force_rebalance: Force rebalance even if below threshold

        Returns:
            Complete adjustment plan
        """
        if as_of_date is None:
            as_of_date = self.fetcher.calendar.get_latest_complete_session()

        # Check if rebalance is needed
        rebalance_needed = self._should_rebalance(as_of_date, force_rebalance)

        # Compute signals
        signals = await self.signal_gen.compute_signals(
            symbols=self.symbols,
            as_of_date=as_of_date,
        )

        # Get volatilities
        volatilities = await self.vol_targeter.get_volatilities(
            symbols=self.symbols,
            as_of_date=as_of_date,
        )

        # Compute volatility-targeted weights
        vol_weights = self.vol_targeter.compute_vol_weights(signals, volatilities)

        # Apply single-name caps and normalize
        final_weights = self._apply_caps(vol_weights)

        # Convert to position targets
        positions = self._weights_to_positions(
            weights=final_weights,
            sleeve_nav=sleeve_nav,
            prices=prices,
            signals=signals,
            vol_weights=vol_weights,
        )

        # Calculate estimated turnover
        turnover = self._estimate_turnover(positions, prices)

        # Check rebalance threshold
        if not force_rebalance and turnover < self.config.rebal_threshold:
            rebalance_needed = False

        adjustment = SleeveBAdjustment(
            as_of_date=as_of_date,
            signals=list(signals.values()),
            positions=positions,
            sleeve_nav=sleeve_nav,
            sleeve_vol_target=self.config.vol_target,
            estimated_turnover=turnover,
            rebalance_needed=rebalance_needed,
        )

        # Log to audit
        self._log_adjustment(adjustment)

        return adjustment

    def _should_rebalance(
        self,
        as_of_date: date,
        force: bool,
    ) -> bool:
        """Determine if rebalance is due."""
        if force:
            return True

        if self._last_rebalance is None:
            return True

        # Weekly rebalancing (every 5 trading days)
        days_since = (as_of_date - self._last_rebalance).days
        return days_since >= 5

    def _apply_caps(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply single-name caps with iterative renormalization.

        Per spec: 15% cap per position, redistribute excess iteratively.
        """
        cap = self.config.single_name_cap
        capped = weights.copy()

        # Iterative capping algorithm
        max_iterations = 10
        for _ in range(max_iterations):
            # Find positions exceeding cap
            excess_total = 0.0
            non_capped_weight = 0.0

            for symbol, weight in capped.items():
                if abs(weight) > cap:
                    excess = abs(weight) - cap
                    excess_total += excess
                    capped[symbol] = cap if weight > 0 else -cap
                else:
                    non_capped_weight += abs(weight)

            if excess_total == 0:
                break

            # Redistribute excess proportionally to non-capped positions
            if non_capped_weight > 0:
                redistribution_factor = 1 + excess_total / non_capped_weight
                for symbol in capped:
                    if abs(capped[symbol]) < cap:
                        capped[symbol] *= redistribution_factor

        # Final normalization to ensure weights sum to target
        total_abs = sum(abs(w) for w in capped.values())
        if total_abs > 0:
            # Keep the same total exposure
            target_exposure = sum(abs(w) for w in weights.values())
            scale = target_exposure / total_abs
            capped = {s: w * scale for s, w in capped.items()}

        return capped

    def _weights_to_positions(
        self,
        weights: Dict[str, float],
        sleeve_nav: float,
        prices: Dict[str, float],
        signals: Dict[str, TrendSignal],
        vol_weights: Dict[str, float],
    ) -> List[PositionTarget]:
        """Convert weights to share targets."""
        positions: List[PositionTarget] = []

        for symbol, final_weight in weights.items():
            price = prices.get(symbol, 0)
            if price <= 0:
                logger.warning(f"Invalid price for {symbol}: {price}")
                continue

            target_notional = sleeve_nav * final_weight
            target_shares = int(target_notional / price)

            current_shares = self._positions.get(symbol, 0)
            delta_shares = target_shares - current_shares

            signal_weight = signals.get(symbol, TrendSignal(
                symbol=symbol, ret_1m=0, ret_3m=0, ret_12m_skip=0,
                composite=0, signal_sign=0, as_of_date=date.today()
            )).composite

            positions.append(PositionTarget(
                symbol=symbol,
                signal_weight=signal_weight,
                vol_scaled_weight=vol_weights.get(symbol, 0),
                final_weight=final_weight,
                target_shares=target_shares,
                target_notional=target_notional,
                current_shares=current_shares,
                delta_shares=delta_shares,
            ))

        return positions

    def _estimate_turnover(
        self,
        positions: List[PositionTarget],
        prices: Dict[str, float],
    ) -> float:
        """
        Estimate turnover as fraction of sleeve NAV.

        Turnover = sum(|delta_notional|) / (2 * NAV)
        """
        total_delta = 0.0
        total_nav = 0.0

        for pos in positions:
            price = prices.get(pos.symbol, 0)
            if price > 0:
                total_delta += abs(pos.delta_shares) * price
                total_nav += abs(pos.target_shares) * price

        if total_nav == 0:
            return 0.0

        return total_delta / (2 * total_nav)

    def _log_adjustment(self, adjustment: SleeveBAdjustment) -> None:
        """Log adjustment to audit trail."""
        for signal in adjustment.signals:
            self.audit.log_signal(
                sleeve="B",
                symbol=signal.symbol,
                signal=signal.composite,
                score=signal.signal_sign,
                reason=f"1m:{signal.ret_1m:.2%} 3m:{signal.ret_3m:.2%} 12m:{signal.ret_12m_skip:.2%}",
            )

    def confirm_rebalance(self, as_of_date: date) -> None:
        """
        Confirm that rebalance was executed.

        Updates last rebalance date.
        """
        self._last_rebalance = as_of_date
        logger.info(f"Sleeve B rebalance confirmed for {as_of_date}")

    def get_target_orders(
        self,
        adjustment: SleeveBAdjustment,
        min_notional: float = 100.0,
    ) -> List[Dict]:
        """
        Convert adjustment to order list.

        Args:
            adjustment: Adjustment plan
            min_notional: Minimum notional to generate order

        Returns:
            List of order dicts ready for execution
        """
        orders = []

        for pos in adjustment.positions:
            if pos.delta_shares == 0:
                continue

            notional = abs(pos.delta_shares * (pos.target_notional / max(pos.target_shares, 1)))
            if notional < min_notional:
                continue

            side = "BUY" if pos.delta_shares > 0 else "SELL"

            orders.append({
                "symbol": pos.symbol,
                "side": side,
                "quantity": abs(pos.delta_shares),
                "sleeve": "B",
                "reason": f"trend_signal={pos.signal_weight:.4f}",
            })

        return orders
