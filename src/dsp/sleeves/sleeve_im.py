"""
Sleeve IM: Intraday ML Long/Short Equity.

Implements an intraday machine learning strategy that:
- Uses morning price/volume patterns [01:30-10:30 ET] to predict afternoon returns
- Enters at 11:30 ET with Top-K long + Top-K short positions
- Exits flat via MOC orders by 15:50 ET
- Maintains dollar neutrality (net exposure <= 10% of gross)

This sleeve is designed to be:
- Daily (single entry/exit per day)
- Always flat overnight (no carry risk)
- Complementary to Sleeve DM (low correlation target < 0.3)

See SPEC_INTRADAY_ML.md for complete specification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Dict, List, Optional, Tuple

from ..data.fetcher import DataFetcher
from ..utils.config import SleeveIMConfig
from ..utils.logging import get_audit_logger

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class MinuteBar:
    """
    A single minute bar with carry-forward metadata.

    See SPEC_INTRADAY_ML.md Section 4.2 for construction rules.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int
    is_synthetic: bool  # True if no trades in this minute (carry-forward)
    seconds_since_last_trade: float
    last_trade_timestamp: Optional[datetime]


@dataclass(frozen=True)
class SleeveIMSignal:
    """
    Trading signal for a single symbol.
    """

    symbol: str
    prediction: float  # P(up) in [0, 1]
    confidence: float  # Model confidence
    direction: str  # "LONG", "SHORT", or "NEUTRAL"
    as_of_date: date
    as_of_time: time  # When signal was generated (should be ~10:31 ET)


@dataclass(frozen=True)
class SleeveIMPositionTarget:
    """
    Target position for a single symbol.
    """

    symbol: str
    direction: str  # "LONG" or "SHORT"
    weight: float  # Normalized weight within side
    target_shares: int
    target_notional: float
    current_shares: int
    delta_shares: int
    price_used: float


@dataclass(frozen=True)
class SleeveIMAdjustment:
    """
    Complete daily adjustment for Sleeve IM.
    """

    execution_date: date
    signal_date: date  # Same as execution_date for intraday
    signal_time: time
    sleeve_nav: float
    target_gross_exposure: float
    signals: List[SleeveIMSignal]
    targets: List[SleeveIMPositionTarget]
    long_notional: float
    short_notional: float
    net_exposure: float
    gross_exposure: float
    estimated_turnover: float
    rebalance_needed: bool = True  # For compatibility with orchestrator


class RiskAction:
    """Risk actions from the risk governor."""

    NONE = "NONE"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    EXIT_TODAY_NO_NEW = "EXIT_TODAY_NO_NEW"
    FLATTEN_AND_HALT = "FLATTEN_AND_HALT"


@dataclass
class SleeveIMRiskState:
    """
    Risk state for Sleeve IM.
    """

    peak_nav: Optional[float] = None
    gross_scale: float = 1.0
    halted: bool = False
    daily_pnl: float = 0.0
    drawdown: float = 0.0


# ============================================================================
# Main Sleeve Class
# ============================================================================


class SleeveIM:
    """
    Sleeve IM (Intraday ML Long/Short) implementation.

    Lifecycle:
    1. 01:30-10:30 ET: Data collection (external - Polygon fetcher)
    2. 10:31 ET: Feature computation + model inference → generate_signals()
    3. 11:25 ET: Position sizing → generate_adjustment()
    4. 11:30 ET: Entry execution → get_target_orders()
    5. 15:45 ET: MOC exit submission → get_exit_orders()
    6. 16:00 ET: Verify flat → confirm_flat()
    """

    def __init__(self, config: SleeveIMConfig, data_fetcher: DataFetcher):
        """
        Initialize Sleeve IM.

        Args:
            config: Sleeve IM configuration
            data_fetcher: Data fetcher for historical data (used for price lookup)
        """
        self.config = config
        self.fetcher = data_fetcher
        self.audit = get_audit_logger()

        # Universe
        self.universe = [s.upper() for s in (config.universe or [])]

        # Positions (symbol -> shares, positive = long, negative = short)
        self._positions: Dict[str, int] = {s: 0 for s in self.universe}

        # Risk state
        self._risk_state = SleeveIMRiskState()

        # Polygon API key (loaded from env)
        self._polygon_key: Optional[str] = None
        if config.polygon_api_key_env:
            self._polygon_key = os.getenv(config.polygon_api_key_env)

        # Model and scaler (loaded lazily)
        self._model = None
        self._scaler = None

        logger.info(
            "Sleeve IM initialized: universe=%d symbols, exposure=%.1f%%",
            len(self.universe),
            config.target_gross_exposure * 100,
        )

    @property
    def symbols(self) -> List[str]:
        """Return list of symbols in the universe."""
        return list(self.universe)

    def set_positions(self, positions: Dict[str, int]) -> None:
        """
        Set current positions from external source (IBKR).
        """
        self._positions = {s: int(positions.get(s, 0)) for s in self.universe}

    def get_positions(self) -> Dict[str, int]:
        """
        Get current positions.
        """
        return dict(self._positions)

    # ========================================================================
    # Signal Generation (10:31 ET)
    # ========================================================================

    async def generate_signals(
        self,
        *,
        as_of_date: date,
        minute_bars: Optional[Dict[str, List[MinuteBar]]] = None,
    ) -> List[SleeveIMSignal]:
        """
        Generate trading signals for the day.

        Called at 10:31 ET after feature window closes.

        Args:
            as_of_date: Trading date
            minute_bars: Pre-fetched minute bars (optional, will fetch if None)

        Returns:
            List of signals for all universe symbols
        """
        if not self.config.enabled:
            logger.info("Sleeve IM: Disabled, returning empty signals")
            return []

        # TODO: Phase 2 - Implement Polygon data fetching
        if minute_bars is None:
            logger.warning("Sleeve IM: minute_bars not provided, skipping")
            return []

        # TODO: Phase 3 - Implement feature computation
        # TODO: Phase 4 - Implement model inference

        signals = []
        signal_time = time(10, 31)

        for symbol in self.universe:
            # Placeholder: random signal for skeleton
            # Real implementation will use ML model
            signals.append(
                SleeveIMSignal(
                    symbol=symbol,
                    prediction=0.5,  # Neutral
                    confidence=0.0,
                    direction="NEUTRAL",
                    as_of_date=as_of_date,
                    as_of_time=signal_time,
                )
            )

        logger.info(
            "Sleeve IM: Generated %d signals for %s",
            len(signals),
            as_of_date,
        )

        return signals

    # ========================================================================
    # Position Sizing (11:25 ET)
    # ========================================================================

    async def generate_adjustment(
        self,
        *,
        sleeve_nav: float,
        prices: Dict[str, float],
        signals: List[SleeveIMSignal],
        as_of_date: Optional[date] = None,
    ) -> SleeveIMAdjustment:
        """
        Convert signals to position targets.

        Called at 11:25 ET before entry.

        Args:
            sleeve_nav: NAV allocated to this sleeve
            prices: Current prices for all symbols
            signals: Signals from generate_signals()
            as_of_date: Trading date (optional)

        Returns:
            Complete adjustment with position targets
        """
        if as_of_date is None:
            as_of_date = date.today()

        signal_time = time(11, 25)

        # Apply risk scaling
        effective_nav = sleeve_nav * self._risk_state.gross_scale

        # Calculate target gross exposure
        target_gross = effective_nav * self.config.target_gross_exposure

        if target_gross <= 0 or self._risk_state.halted:
            # Shadow mode or halted - no positions
            return SleeveIMAdjustment(
                execution_date=as_of_date,
                signal_date=as_of_date,
                signal_time=signal_time,
                sleeve_nav=sleeve_nav,
                target_gross_exposure=self.config.target_gross_exposure,
                signals=signals,
                targets=[],
                long_notional=0.0,
                short_notional=0.0,
                net_exposure=0.0,
                gross_exposure=0.0,
                estimated_turnover=0.0,
            )

        # Select top-K longs and top-K shorts
        longs, shorts = self._select_trades(signals)

        # Compute position sizes
        targets = self._compute_positions(
            longs=longs,
            shorts=shorts,
            signals={s.symbol: (s.prediction, s.confidence) for s in signals},
            prices=prices,
            target_gross=target_gross,
            nav=sleeve_nav,
        )

        # Calculate exposure metrics
        long_notional = sum(
            t.target_notional for t in targets if t.target_shares > 0
        )
        short_notional = abs(
            sum(t.target_notional for t in targets if t.target_shares < 0)
        )
        gross_exposure = long_notional + short_notional
        net_exposure = long_notional - short_notional

        # Calculate estimated turnover
        total_delta_notional = sum(abs(t.delta_shares) * t.price_used for t in targets)
        estimated_turnover = (
            (total_delta_notional / (2.0 * sleeve_nav)) if sleeve_nav > 0 else 0.0
        )

        # Audit logging
        for signal in signals:
            if signal.direction != "NEUTRAL":
                self.audit.log_signal(
                    sleeve="IM",
                    symbol=signal.symbol,
                    signal=signal.prediction,
                    score=1 if signal.direction == "LONG" else -1,
                    reason=f"p={signal.prediction:.3f} conf={signal.confidence:.3f} dir={signal.direction}",
                )

        return SleeveIMAdjustment(
            execution_date=as_of_date,
            signal_date=as_of_date,
            signal_time=signal_time,
            sleeve_nav=sleeve_nav,
            target_gross_exposure=self.config.target_gross_exposure,
            signals=signals,
            targets=targets,
            long_notional=long_notional,
            short_notional=short_notional,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            estimated_turnover=estimated_turnover,
        )

    def _select_trades(
        self,
        signals: List[SleeveIMSignal],
    ) -> Tuple[List[str], List[str]]:
        """
        Select symbols to trade based on model predictions.

        Returns:
            longs: List of symbols to go long
            shorts: List of symbols to go short
        """
        edge = self.config.edge_threshold
        top_k = self.config.top_k

        # Separate into long and short candidates
        long_candidates = [
            (s.symbol, s.confidence)
            for s in signals
            if s.prediction >= 0.5 + edge
        ]
        short_candidates = [
            (s.symbol, s.confidence)
            for s in signals
            if s.prediction <= 0.5 - edge
        ]

        # Rank by confidence
        long_ranked = sorted(long_candidates, key=lambda x: x[1], reverse=True)
        short_ranked = sorted(short_candidates, key=lambda x: x[1], reverse=True)

        # Select Top-K per side
        longs = [s for s, _ in long_ranked[:top_k]]
        shorts = [s for s, _ in short_ranked[:top_k]]

        return longs, shorts

    def _compute_positions(
        self,
        longs: List[str],
        shorts: List[str],
        signals: Dict[str, Tuple[float, float]],
        prices: Dict[str, float],
        target_gross: float,
        nav: float,
    ) -> List[SleeveIMPositionTarget]:
        """
        Compute position sizes (shares) for each symbol.

        Sizing is confidence-weighted within target gross exposure.
        """
        if not longs and not shorts:
            return []

        # Get confidences
        long_conf = {s: signals[s][1] for s in longs if s in signals}
        short_conf = {s: signals[s][1] for s in shorts if s in signals}

        # Normalize weights
        total_conf = sum(long_conf.values()) + sum(short_conf.values())
        if total_conf == 0:
            return []

        targets = []
        max_single_name = nav * self.config.max_single_name_pct

        # Long positions
        for symbol in longs:
            if symbol not in prices or prices[symbol] <= 0:
                continue

            weight = long_conf.get(symbol, 0) / total_conf
            notional = target_gross * weight
            notional = min(notional, max_single_name)  # Single-name cap

            price = prices[symbol]
            target_shares = int(notional / price)
            current_shares = self._positions.get(symbol, 0)
            delta_shares = target_shares - current_shares

            targets.append(
                SleeveIMPositionTarget(
                    symbol=symbol,
                    direction="LONG",
                    weight=weight,
                    target_shares=target_shares,
                    target_notional=target_shares * price,
                    current_shares=current_shares,
                    delta_shares=delta_shares,
                    price_used=price,
                )
            )

        # Short positions
        for symbol in shorts:
            if symbol not in prices or prices[symbol] <= 0:
                continue

            weight = short_conf.get(symbol, 0) / total_conf
            notional = target_gross * weight
            notional = min(notional, max_single_name)

            price = prices[symbol]
            target_shares = -int(notional / price)  # Negative for short
            current_shares = self._positions.get(symbol, 0)
            delta_shares = target_shares - current_shares

            targets.append(
                SleeveIMPositionTarget(
                    symbol=symbol,
                    direction="SHORT",
                    weight=weight,
                    target_shares=target_shares,
                    target_notional=target_shares * price,
                    current_shares=current_shares,
                    delta_shares=delta_shares,
                    price_used=price,
                )
            )

        # Enforce dollar-neutral if required
        if self.config.dollar_neutral:
            targets = self._enforce_dollar_neutral(targets, prices, nav)

        return targets

    def _enforce_dollar_neutral(
        self,
        targets: List[SleeveIMPositionTarget],
        prices: Dict[str, float],
        nav: float,
    ) -> List[SleeveIMPositionTarget]:
        """
        Adjust positions to be approximately dollar-neutral.
        """
        long_notional = sum(
            t.target_shares * prices.get(t.symbol, t.price_used)
            for t in targets
            if t.target_shares > 0
        )
        short_notional = abs(
            sum(
                t.target_shares * prices.get(t.symbol, t.price_used)
                for t in targets
                if t.target_shares < 0
            )
        )

        net_exposure = long_notional - short_notional
        gross_exposure = long_notional + short_notional
        max_net = gross_exposure * self.config.max_net_exposure_pct_gross if gross_exposure > 0 else 0.0

        if abs(net_exposure) <= max_net:
            return targets  # Already within bounds

        # Scale down the larger side
        new_targets = []
        for t in targets:
            if net_exposure > 0 and t.target_shares > 0:
                # Long-heavy, scale down longs
                scale = (short_notional + max_net) / long_notional if long_notional > 0 else 1.0
                new_shares = int(t.target_shares * scale)
            elif net_exposure < 0 and t.target_shares < 0:
                # Short-heavy, scale down shorts
                scale = (long_notional + max_net) / short_notional if short_notional > 0 else 1.0
                new_shares = int(t.target_shares * scale)
            else:
                new_shares = t.target_shares

            price = prices.get(t.symbol, t.price_used)
            new_targets.append(
                SleeveIMPositionTarget(
                    symbol=t.symbol,
                    direction=t.direction,
                    weight=t.weight,
                    target_shares=new_shares,
                    target_notional=new_shares * price,
                    current_shares=t.current_shares,
                    delta_shares=new_shares - t.current_shares,
                    price_used=price,
                )
            )

        return new_targets

    # ========================================================================
    # Order Generation (11:30 ET)
    # ========================================================================

    def get_target_orders(
        self,
        adjustment: SleeveIMAdjustment,
        *,
        min_notional: float = 100.0,
    ) -> List[Dict]:
        """
        Convert adjustment to executable entry orders.

        Called at 11:30 ET for entry.
        """
        orders = []

        for t in adjustment.targets:
            if t.delta_shares == 0:
                continue

            # Notional filter to avoid noise trades
            notional = abs(t.delta_shares) * t.price_used
            if notional < min_notional:
                continue

            orders.append(
                {
                    "symbol": t.symbol,
                    "side": "BUY" if t.delta_shares > 0 else "SELL",
                    "quantity": abs(int(t.delta_shares)),
                    "sleeve": "IM",
                    "reason": f"intraday_ml dir={t.direction} signal_date={adjustment.signal_date}",
                    "order_type": "LMT",  # Marketable limit for entry
                    "slippage_cap_bps": self.config.entry_slippage_cap_bps,
                }
            )

        logger.info(
            "Sleeve IM: Generated %d entry orders for %s",
            len(orders),
            adjustment.execution_date,
        )

        return orders

    # ========================================================================
    # Exit Orders (15:45 ET)
    # ========================================================================

    def get_exit_orders(
        self,
        prices: Dict[str, float],
        *,
        min_notional: float = 100.0,
    ) -> List[Dict]:
        """
        Generate MOC exit orders to flatten all positions.

        Called at 15:45 ET for MOC submission (deadline 15:50 ET).
        """
        orders = []

        for symbol, qty in self._positions.items():
            if qty == 0:
                continue

            price = prices.get(symbol, 0.0)
            if price <= 0:
                logger.warning("Sleeve IM: No price for %s, skipping exit", symbol)
                continue

            notional = abs(qty) * price
            if notional < min_notional:
                continue

            orders.append(
                {
                    "symbol": symbol,
                    "side": "SELL" if qty > 0 else "BUY",
                    "quantity": abs(qty),
                    "sleeve": "IM",
                    "reason": "intraday_ml_moc_exit",
                    "order_type": "MOC",  # Market on Close
                }
            )

        logger.info("Sleeve IM: Generated %d MOC exit orders", len(orders))

        return orders

    # ========================================================================
    # Risk Management
    # ========================================================================

    def update_risk_state(
        self,
        current_nav: float,
        sleeve_nav: float,
        daily_pnl: float,
    ) -> str:
        """
        Update risk state and return required action.
        """
        # Update peak
        if self._risk_state.peak_nav is None or sleeve_nav > self._risk_state.peak_nav:
            self._risk_state.peak_nav = sleeve_nav

        # Calculate drawdown
        if self._risk_state.peak_nav and self._risk_state.peak_nav > 0:
            drawdown = (self._risk_state.peak_nav - sleeve_nav) / self._risk_state.peak_nav
        else:
            drawdown = 0.0

        self._risk_state.drawdown = drawdown
        self._risk_state.daily_pnl = daily_pnl

        # Check hard stop
        if drawdown >= self.config.drawdown_hard_stop:
            self._risk_state.halted = True
            logger.critical(
                "Sleeve IM: HALTED - Drawdown %.1f%% >= %.1f%%",
                drawdown * 100,
                self.config.drawdown_hard_stop * 100,
            )
            return RiskAction.FLATTEN_AND_HALT

        # Check warning level
        if drawdown >= self.config.drawdown_warning:
            self._risk_state.gross_scale = 0.5
            logger.warning(
                "Sleeve IM: Reducing exposure - Drawdown %.1f%% >= %.1f%%",
                drawdown * 100,
                self.config.drawdown_warning * 100,
            )
            return RiskAction.REDUCE_EXPOSURE

        # Check daily loss limit
        daily_loss_pct = -daily_pnl / sleeve_nav if sleeve_nav > 0 else 0
        if daily_loss_pct >= self.config.daily_loss_limit:
            logger.warning(
                "Sleeve IM: Daily loss limit - Loss %.2f%% >= %.2f%%",
                daily_loss_pct * 100,
                self.config.daily_loss_limit * 100,
            )
            return RiskAction.EXIT_TODAY_NO_NEW

        # Normal operation
        self._risk_state.gross_scale = 1.0
        return RiskAction.NONE

    def confirm_flat(self) -> bool:
        """
        Verify all positions are flat (called at 16:00 ET).
        """
        total_exposure = sum(abs(qty) for qty in self._positions.values())
        is_flat = total_exposure == 0

        if is_flat:
            logger.info("Sleeve IM: Confirmed FLAT at EOD")
        else:
            positions_str = ", ".join(
                f"{s}={q}" for s, q in self._positions.items() if q != 0
            )
            logger.error("Sleeve IM: NOT FLAT at EOD: %s", positions_str)

        return is_flat

    def reset_daily_state(self) -> None:
        """
        Reset daily state for new trading day.
        """
        self._risk_state.daily_pnl = 0.0
        logger.info("Sleeve IM: Daily state reset")
