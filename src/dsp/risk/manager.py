"""
Risk manager for DSP-100K.

Implements portfolio-level risk controls:
- Volatility monitoring and targeting (7% portfolio, 8% cap)
- Drawdown monitoring (6% warning, 10% hard stop)
- Position concentration limits
- Double-strike logic for deleveraging

Per SPEC_DSP_100K.md Section 4.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..ibkr import IBKRClient, AccountSummary, Position
from ..utils.config import RiskConfig
from ..utils.logging import get_audit_logger
from .margin import MarginMonitor, MarginStatus

logger = logging.getLogger(__name__)


# Default path for risk state persistence
DEFAULT_RISK_STATE_PATH = Path("data/risk_state.json")


class RiskLevel(str, Enum):
    """Risk level states."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of risk alerts."""
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_BREACH = "drawdown_breach"
    VOLATILITY_HIGH = "volatility_high"
    MARGIN_WARNING = "margin_warning"
    MARGIN_BREACH = "margin_breach"
    CONCENTRATION = "concentration"
    DOUBLE_STRIKE = "double_strike"


@dataclass
class RiskAlert:
    """A risk alert."""
    alert_type: AlertType
    level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    requires_action: bool = False
    recommended_action: Optional[str] = None


@dataclass
class VolatilityMetrics:
    """Portfolio volatility metrics."""
    sleeve_a_vol: float
    sleeve_b_vol: float
    sleeve_c_delta: float    # Sleeve C doesn't have vol, just delta
    portfolio_vol: float     # Combined portfolio volatility
    vol_target: float
    scale_factor: float      # 1.0 if at target, < 1.0 if over


@dataclass
class DrawdownMetrics:
    """Drawdown metrics."""
    current_nav: float
    peak_nav: float
    drawdown_pct: float
    drawdown_dollars: float
    days_since_peak: int
    warning_threshold: float
    hard_stop_threshold: float


@dataclass
class RiskStatus:
    """Complete risk status."""
    timestamp: datetime
    level: RiskLevel
    alerts: List[RiskAlert]
    volatility: VolatilityMetrics
    drawdown: DrawdownMetrics
    margin: MarginStatus
    double_strike_active: bool
    scale_factor: float


class RiskManager:
    """
    Portfolio risk manager.

    Monitors and controls:
    - Portfolio volatility (7% target, 8% cap)
    - Drawdown (6% warning, 10% hard stop)
    - Margin utilization (60% cap)
    - Position concentrations
    - Double-strike deleveraging
    """

    def __init__(
        self,
        config: RiskConfig,
        ibkr_client: IBKRClient,
        data_fetcher: DataFetcher,
        state_path: Optional[Path] = None,
    ):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration
            ibkr_client: IBKR client for account data
            data_fetcher: Data fetcher for volatility calculations
            state_path: Path for persisting risk state (default: data/risk_state.json)
        """
        self.config = config
        self.ibkr = ibkr_client
        self.fetcher = data_fetcher
        self.margin_monitor = MarginMonitor(ibkr_client, config.margin_cap)
        self.audit = get_audit_logger()
        self._state_path = state_path or DEFAULT_RISK_STATE_PATH

        # Peak tracking for drawdown
        self._peak_nav: float = 0.0
        self._peak_date: Optional[date] = None

        # Double-strike tracking
        self._strike_history: List[Tuple[date, float]] = []
        self._double_strike_active: bool = False

        # Current scale factor (1.0 = normal, < 1.0 = deleveraged)
        self._scale_factor: float = 1.0

        # Load persisted state on init
        self._load_state()

    async def get_status(
        self,
        as_of_date: Optional[date] = None,
    ) -> RiskStatus:
        """
        Get comprehensive risk status.

        Args:
            as_of_date: Reference date

        Returns:
            Complete risk status
        """
        if as_of_date is None:
            as_of_date = self.fetcher.calendar.get_latest_complete_session()

        alerts: List[RiskAlert] = []

        # Get account summary
        summary = await self.ibkr.get_account_summary()
        current_nav = summary.nlv

        # Update peak tracking
        self._update_peak(current_nav, as_of_date)

        # Get margin status
        margin = await self.margin_monitor.get_status()
        margin_alerts = self._check_margin(margin)
        alerts.extend(margin_alerts)

        # Calculate drawdown
        drawdown = self._calculate_drawdown(current_nav, as_of_date)
        drawdown_alerts = self._check_drawdown(drawdown)
        alerts.extend(drawdown_alerts)

        # Calculate volatility (simplified for now)
        volatility = await self._calculate_volatility(as_of_date)
        vol_alerts = self._check_volatility(volatility)
        alerts.extend(vol_alerts)

        # Check double-strike
        self._check_double_strike(drawdown, as_of_date)

        # Determine overall risk level
        level = self._determine_risk_level(alerts)

        # Calculate scale factor
        self._scale_factor = self._calculate_scale_factor(
            volatility, drawdown, margin
        )

        return RiskStatus(
            timestamp=datetime.now(),
            level=level,
            alerts=alerts,
            volatility=volatility,
            drawdown=drawdown,
            margin=margin,
            double_strike_active=self._double_strike_active,
            scale_factor=self._scale_factor,
        )

    def _update_peak(self, current_nav: float, as_of_date: date) -> None:
        """Update peak NAV tracking."""
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav
            self._peak_date = as_of_date
            self._save_state()  # Persist on new high

    def _calculate_drawdown(
        self,
        current_nav: float,
        as_of_date: date,
    ) -> DrawdownMetrics:
        """Calculate drawdown metrics."""
        if self._peak_nav <= 0:
            self._peak_nav = current_nav
            self._peak_date = as_of_date

        drawdown_dollars = self._peak_nav - current_nav
        drawdown_pct = drawdown_dollars / self._peak_nav if self._peak_nav > 0 else 0

        days_since_peak = 0
        if self._peak_date:
            days_since_peak = (as_of_date - self._peak_date).days

        return DrawdownMetrics(
            current_nav=current_nav,
            peak_nav=self._peak_nav,
            drawdown_pct=drawdown_pct,
            drawdown_dollars=drawdown_dollars,
            days_since_peak=days_since_peak,
            warning_threshold=self.config.dd_warning,
            hard_stop_threshold=self.config.dd_hard_stop,
        )

    def _check_drawdown(self, dd: DrawdownMetrics) -> List[RiskAlert]:
        """Check drawdown against thresholds."""
        alerts: List[RiskAlert] = []

        if dd.drawdown_pct >= self.config.dd_hard_stop:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN_BREACH,
                level=RiskLevel.CRITICAL,
                message=f"Drawdown {dd.drawdown_pct:.1%} exceeds hard stop {self.config.dd_hard_stop:.0%}",
                value=dd.drawdown_pct,
                threshold=self.config.dd_hard_stop,
                requires_action=True,
                recommended_action="HALT TRADING - Liquidate to cash",
            ))
            self.audit.log_risk_event("DRAWDOWN_BREACH", {
                "drawdown_pct": dd.drawdown_pct,
                "threshold": self.config.dd_hard_stop,
                "current_nav": dd.current_nav,
                "peak_nav": dd.peak_nav,
            })
        elif dd.drawdown_pct >= self.config.dd_warning:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN_WARNING,
                level=RiskLevel.HIGH,
                message=f"Drawdown {dd.drawdown_pct:.1%} exceeds warning {self.config.dd_warning:.0%}",
                value=dd.drawdown_pct,
                threshold=self.config.dd_warning,
                requires_action=False,
                recommended_action="Monitor closely, consider reducing exposure",
            ))

        return alerts

    async def _calculate_volatility(
        self,
        as_of_date: date,
    ) -> VolatilityMetrics:
        """
        Calculate portfolio volatility.

        Simplified implementation - uses proxy calculations.
        """
        # Get portfolio returns (proxy using SPY for now)
        # In production, would calculate from actual portfolio positions
        portfolio_vol = await self.fetcher.get_volatility(
            "SPY", lookback_days=21, end_date=as_of_date
        ) or 0.15

        # Sleeve volatilities (simplified)
        sleeve_a_vol = self.config.sleeve_a_vol_target
        sleeve_b_vol = self.config.sleeve_b_vol_target

        # Scale factor to hit target
        scale_factor = 1.0
        if portfolio_vol > self.config.vol_cap:
            scale_factor = self.config.vol_cap / portfolio_vol

        return VolatilityMetrics(
            sleeve_a_vol=sleeve_a_vol,
            sleeve_b_vol=sleeve_b_vol,
            sleeve_c_delta=0.0,  # Would calculate from options positions
            portfolio_vol=portfolio_vol,
            vol_target=self.config.vol_target,
            scale_factor=scale_factor,
        )

    def _check_volatility(self, vol: VolatilityMetrics) -> List[RiskAlert]:
        """Check volatility against targets."""
        alerts: List[RiskAlert] = []

        if vol.portfolio_vol > self.config.vol_cap:
            alerts.append(RiskAlert(
                alert_type=AlertType.VOLATILITY_HIGH,
                level=RiskLevel.HIGH,
                message=f"Portfolio vol {vol.portfolio_vol:.1%} exceeds cap {self.config.vol_cap:.0%}",
                value=vol.portfolio_vol,
                threshold=self.config.vol_cap,
                requires_action=True,
                recommended_action=f"Scale positions by {vol.scale_factor:.2f}",
            ))

        return alerts

    def _check_margin(self, margin: MarginStatus) -> List[RiskAlert]:
        """Check margin against thresholds."""
        alerts: List[RiskAlert] = []

        if margin.utilization_pct >= self.config.margin_cap:
            alerts.append(RiskAlert(
                alert_type=AlertType.MARGIN_BREACH,
                level=RiskLevel.CRITICAL,
                message=f"Margin {margin.utilization_pct:.1%} exceeds cap {self.config.margin_cap:.0%}",
                value=margin.utilization_pct,
                threshold=self.config.margin_cap,
                requires_action=True,
                recommended_action="Reduce positions to meet margin cap",
            ))
        elif margin.utilization_pct >= self.config.margin_cap * 0.9:
            alerts.append(RiskAlert(
                alert_type=AlertType.MARGIN_WARNING,
                level=RiskLevel.HIGH,
                message=f"Margin {margin.utilization_pct:.1%} approaching cap",
                value=margin.utilization_pct,
                threshold=self.config.margin_cap,
                requires_action=False,
            ))

        return alerts

    def _check_double_strike(
        self,
        drawdown: DrawdownMetrics,
        as_of_date: date,
    ) -> None:
        """
        Check and update double-strike logic.

        Per spec: If drawdown breaches warning twice in rolling 365 days,
        halve exposure for 30 days.
        """
        before_history = self._strike_history.copy()
        before_double_strike = self._double_strike_active

        # Clean old strikes (older than 365 days)
        cutoff = as_of_date - timedelta(days=365)
        self._strike_history = [
            (d, dd) for d, dd in self._strike_history
            if d >= cutoff
        ]
        if self._strike_history != before_history:
            self._save_state()

        # Check for new strike
        if drawdown.drawdown_pct >= self.config.dd_warning:
            # Only add if not already struck today
            if not any(d == as_of_date for d, _ in self._strike_history):
                self._strike_history.append((as_of_date, drawdown.drawdown_pct))

                self.audit.log_risk_event("DD_STRIKE", {
                    "date": str(as_of_date),
                    "drawdown_pct": drawdown.drawdown_pct,
                    "strike_count": len(self._strike_history),
                })
                self._save_state()

        # Check for double strike
        if len(self._strike_history) >= 2:
            if not self._double_strike_active:  # Only log on transition
                self._double_strike_active = True
                self.audit.log_risk_event("DOUBLE_STRIKE_ACTIVE", {
                    "strikes": [(str(d), dd) for d, dd in self._strike_history],
                })
                self._save_state()  # Persist state change
        else:
            self._double_strike_active = False
            if before_double_strike and not self._double_strike_active:
                self._save_state()

    def _determine_risk_level(self, alerts: List[RiskAlert]) -> RiskLevel:
        """Determine overall risk level from alerts."""
        if any(a.level == RiskLevel.CRITICAL for a in alerts):
            return RiskLevel.CRITICAL
        elif any(a.level == RiskLevel.HIGH for a in alerts):
            return RiskLevel.HIGH
        elif any(a.level == RiskLevel.ELEVATED for a in alerts):
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.NORMAL

    def _calculate_scale_factor(
        self,
        volatility: VolatilityMetrics,
        drawdown: DrawdownMetrics,
        margin: MarginStatus,
    ) -> float:
        """
        Calculate position scale factor.

        Scale factor is the product of:
        - Volatility adjustment (to stay within vol cap)
        - Margin adjustment (to stay within margin cap)
        - Double-strike adjustment (50% if active)
        """
        scale = 1.0

        # Volatility scaling
        if volatility.scale_factor < 1.0:
            scale *= volatility.scale_factor

        # Margin scaling (if approaching cap)
        if margin.utilization_pct > self.config.margin_cap * 0.9:
            margin_scale = self.config.margin_cap / margin.utilization_pct
            scale *= min(1.0, margin_scale)

        # Double-strike (50% reduction)
        if self._double_strike_active:
            scale *= 0.50

        # Hard stop (zero exposure)
        if drawdown.drawdown_pct >= self.config.dd_hard_stop:
            scale = 0.0

        return scale

    @property
    def scale_factor(self) -> float:
        """Get current scale factor."""
        return self._scale_factor

    @property
    def is_trading_halted(self) -> bool:
        """Check if trading should be halted."""
        return self._scale_factor <= 0

    def reset_peak(self, new_peak: float, peak_date: date) -> None:
        """
        Reset peak tracking (e.g., after capital injection).

        Args:
            new_peak: New peak NAV
            peak_date: Date of new peak
        """
        self._peak_nav = new_peak
        self._peak_date = peak_date
        logger.info(f"Reset peak NAV to ${new_peak:,.2f} on {peak_date}")
        self._save_state()

    def clear_double_strike(self) -> None:
        """
        Clear double-strike status (manual override).

        Should only be used with explicit approval.
        """
        self._strike_history = []
        self._double_strike_active = False
        self.audit.log_risk_event("DOUBLE_STRIKE_CLEARED", {
            "reason": "Manual override",
        })
        self._save_state()

    def _load_state(self) -> None:
        """Load persisted risk state from disk."""
        if not self._state_path.exists():
            logger.info(f"No persisted risk state found at {self._state_path}")
            return

        try:
            with open(self._state_path) as f:
                data = json.load(f)

            # Restore peak tracking
            self._peak_nav = data.get("peak_nav", 0.0)
            if data.get("peak_date"):
                self._peak_date = date.fromisoformat(data["peak_date"])

            # Restore strike history
            self._strike_history = [
                (date.fromisoformat(d), dd) for d, dd in data.get("strike_history", [])
            ]
            self._double_strike_active = data.get("double_strike_active", False)

            logger.info(
                f"Loaded risk state: peak_nav=${self._peak_nav:,.2f}, "
                f"peak_date={self._peak_date}, "
                f"strikes={len(self._strike_history)}, "
                f"double_strike={self._double_strike_active}"
            )

        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")

    def _save_state(self) -> None:
        """Persist risk state to disk."""
        try:
            # Ensure directory exists
            self._state_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "peak_nav": self._peak_nav,
                "peak_date": str(self._peak_date) if self._peak_date else None,
                "strike_history": [
                    (str(d), dd) for d, dd in self._strike_history
                ],
                "double_strike_active": self._double_strike_active,
                "last_saved": datetime.now().isoformat(),
            }

            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved risk state to {self._state_path}")

        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
