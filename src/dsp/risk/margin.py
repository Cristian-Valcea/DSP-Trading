"""
Margin monitoring for DSP-100K.

Tracks margin utilization and enforces the 60% hard cap.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..ibkr import IBKRClient, AccountSummary, MarginImpact
from ..ibkr.models import Order
from ..utils.logging import get_audit_logger

logger = logging.getLogger(__name__)


class MarginState(str, Enum):
    """Margin utilization states."""
    HEALTHY = "healthy"       # < 40%
    ELEVATED = "elevated"     # 40-50%
    WARNING = "warning"       # 50-55%
    CRITICAL = "critical"     # 55-60%
    BREACH = "breach"         # > 60%


@dataclass
class MarginStatus:
    """Current margin status."""
    nlv: float
    margin_used: float
    available_funds: float
    buying_power: float
    utilization_pct: float
    state: MarginState
    headroom: float          # Dollars available before 60% cap
    timestamp: datetime


@dataclass
class OrderMarginCheck:
    """Result of pre-trade margin check."""
    order: Order
    approved: bool
    reason: str
    current_utilization: float
    projected_utilization: float
    margin_impact: Optional[MarginImpact] = None


class MarginMonitor:
    """
    Monitors margin utilization and enforces caps.

    Per SPEC_DSP_100K.md Section 4.2:
    - 60% margin cap (hard limit)
    - Reject orders that would breach cap
    - Alert on elevated utilization
    """

    # Utilization thresholds
    HEALTHY_MAX = 0.40
    ELEVATED_MAX = 0.50
    WARNING_MAX = 0.55
    CRITICAL_MAX = 0.60
    HARD_CAP = 0.60

    def __init__(
        self,
        ibkr_client: IBKRClient,
        hard_cap: float = HARD_CAP,
    ):
        """
        Initialize margin monitor.

        Args:
            ibkr_client: IBKR client for account data
            hard_cap: Maximum allowed margin utilization
        """
        self.ibkr = ibkr_client
        self.hard_cap = hard_cap
        self.audit = get_audit_logger()

        # Cache last known status
        self._last_status: Optional[MarginStatus] = None
        self._status_age_threshold = 60  # Refresh if older than 60 seconds

    async def get_status(self, force_refresh: bool = False) -> MarginStatus:
        """
        Get current margin status.

        Args:
            force_refresh: Force refresh from IBKR

        Returns:
            Current margin status
        """
        # Check cache
        if not force_refresh and self._last_status:
            age = (datetime.now() - self._last_status.timestamp).total_seconds()
            if age < self._status_age_threshold:
                return self._last_status

        # Fetch from IBKR
        summary = await self.ibkr.get_account_summary()

        utilization = summary.margin_usage
        state = self._classify_utilization(utilization)
        headroom = max(0, (self.hard_cap - utilization) * summary.nlv)

        status = MarginStatus(
            nlv=summary.nlv,
            margin_used=summary.margin_used,
            available_funds=summary.available_funds,
            buying_power=summary.buying_power,
            utilization_pct=utilization,
            state=state,
            headroom=headroom,
            timestamp=datetime.now(),
        )

        self._last_status = status
        return status

    def _classify_utilization(self, utilization: float) -> MarginState:
        """Classify utilization into state."""
        if utilization <= self.HEALTHY_MAX:
            return MarginState.HEALTHY
        elif utilization <= self.ELEVATED_MAX:
            return MarginState.ELEVATED
        elif utilization <= self.WARNING_MAX:
            return MarginState.WARNING
        elif utilization <= self.CRITICAL_MAX:
            return MarginState.CRITICAL
        else:
            return MarginState.BREACH

    async def check_order(
        self,
        order: Order,
    ) -> OrderMarginCheck:
        """
        Check if an order would breach margin cap.

        Uses IBKR's what-if functionality to estimate margin impact.

        Args:
            order: Proposed order

        Returns:
            Margin check result with approval status
        """
        # Get current status
        current = await self.get_status()

        # If already in breach, reject new risk-adding orders
        if current.state == MarginState.BREACH:
            return OrderMarginCheck(
                order=order,
                approved=False,
                reason=f"Margin already breached ({current.utilization_pct:.1%})",
                current_utilization=current.utilization_pct,
                projected_utilization=current.utilization_pct,
            )

        # Get what-if margin impact
        try:
            impact = await self.ibkr.what_if_order(order)
        except Exception as e:
            logger.error(f"What-if margin check failed: {e}")
            # Conservative: reject if we can't check
            return OrderMarginCheck(
                order=order,
                approved=False,
                reason=f"Margin check failed: {e}",
                current_utilization=current.utilization_pct,
                projected_utilization=current.utilization_pct,
            )

        # Calculate projected utilization
        # current.margin_used comes from MaintMarginReq, so use maint_margin_change for consistency.
        projected_margin = current.margin_used + impact.maint_margin_change
        projected_utilization = projected_margin / current.nlv if current.nlv > 0 else 1.0

        # Check against cap
        approved = projected_utilization <= self.hard_cap

        if approved:
            reason = f"Projected margin {projected_utilization:.1%} within cap"
        else:
            reason = f"Would breach {self.hard_cap:.0%} cap: {projected_utilization:.1%}"
            self.audit.log_risk_event(
                "MARGIN_BLOCK",
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "current_util": current.utilization_pct,
                    "projected_util": projected_utilization,
                    "hard_cap": self.hard_cap,
                },
            )

        return OrderMarginCheck(
            order=order,
            approved=approved,
            reason=reason,
            current_utilization=current.utilization_pct,
            projected_utilization=projected_utilization,
            margin_impact=impact,
        )

    async def check_orders_batch(
        self,
        orders: List[Order],
    ) -> List[OrderMarginCheck]:
        """
        Check multiple orders for margin impact.

        Checks orders sequentially, considering cumulative impact.

        Args:
            orders: List of proposed orders

        Returns:
            List of margin check results
        """
        results: List[OrderMarginCheck] = []

        for order in orders:
            check = await self.check_order(order)
            results.append(check)

            # If rejected, subsequent orders might still be OK
            # but we should be conservative

        return results

    def get_max_order_size(
        self,
        price: float,
        current_headroom: Optional[float] = None,
    ) -> int:
        """
        Calculate maximum order size within margin headroom.

        Args:
            price: Security price
            current_headroom: Available margin headroom (uses cached if None)

        Returns:
            Maximum share quantity
        """
        if current_headroom is None:
            if self._last_status:
                current_headroom = self._last_status.headroom
            else:
                return 0

        if price <= 0 or current_headroom <= 0:
            return 0

        # Conservative: assume 50% margin requirement
        margin_requirement = 0.50
        max_notional = current_headroom / margin_requirement
        max_shares = int(max_notional / price)

        return max_shares
