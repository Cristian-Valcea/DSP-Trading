"""
Order executor for DSP-100K.

Handles order placement with proper price rounding,
execution window enforcement, and fill monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional

from ..ibkr import IBKRClient
from ..ibkr.models import Order, OrderStatusInfo, Fill, Quote
from ..risk.margin import MarginMonitor
from ..utils.logging import get_audit_logger
from ..utils.time import is_in_execution_window, get_ny_time

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """Report for a single order execution."""
    order: Order
    submitted: bool
    filled: bool
    fill_qty: int
    fill_price: float
    slippage_bps: float
    commission: float
    status: str
    error: Optional[str] = None


@dataclass
class BatchExecutionReport:
    """Report for a batch of executions."""
    sleeve: str
    total_orders: int
    orders_submitted: int
    orders_filled: int
    total_notional: float
    total_commission: float
    average_slippage_bps: float
    executions: List[ExecutionReport] = field(default_factory=list)


class OrderExecutor:
    """
    Executes orders with proper controls.

    Features:
    - Execution window enforcement (09:35-10:15 ET)
    - Marketable limit orders with tick rounding
    - Fill monitoring with timeout
    - Slippage calculation
    - Margin pre-checks
    """

    # Default execution window
    DEFAULT_WINDOW_START = "09:35"
    DEFAULT_WINDOW_END = "10:15"

    # Execution parameters
    FILL_TIMEOUT_SECONDS = 60
    LIMIT_BUFFER_BPS = 10       # 10 bps buffer for marketable limits
    MAX_SLIPPAGE_BPS = 50       # Alert if slippage > 50 bps

    def __init__(
        self,
        ibkr_client: IBKRClient,
        margin_monitor: MarginMonitor,
        window_start: str = DEFAULT_WINDOW_START,
        window_end: str = DEFAULT_WINDOW_END,
    ):
        """
        Initialize order executor.

        Args:
            ibkr_client: IBKR client for order placement
            margin_monitor: Margin monitor for pre-checks
            window_start: Execution window start (HH:MM ET)
            window_end: Execution window end (HH:MM ET)
        """
        self.ibkr = ibkr_client
        self.margin = margin_monitor
        self.window_start = window_start
        self.window_end = window_end
        self.audit = get_audit_logger()

    async def execute_orders(
        self,
        orders: List[Dict],
        check_margin: bool = True,
        force_execution: bool = False,
    ) -> BatchExecutionReport:
        """
        Execute a batch of orders.

        Args:
            orders: List of order dicts (symbol, side, quantity, sleeve, reason)
            check_margin: Check margin before each order
            force_execution: Execute even outside window

        Returns:
            Batch execution report
        """
        # Check execution window
        if not force_execution and not is_in_execution_window(
            self.window_start, self.window_end
        ):
            logger.warning("Outside execution window, skipping orders")
            return BatchExecutionReport(
                sleeve=orders[0].get("sleeve", "?") if orders else "?",
                total_orders=len(orders),
                orders_submitted=0,
                orders_filled=0,
                total_notional=0,
                total_commission=0,
                average_slippage_bps=0,
            )

        sleeve = orders[0].get("sleeve", "?") if orders else "?"
        executions: List[ExecutionReport] = []
        total_notional = 0
        total_commission = 0
        total_slippage = 0
        fill_count = 0

        for order_dict in orders:
            try:
                report = await self._execute_single(
                    order_dict,
                    check_margin=check_margin,
                )
                executions.append(report)

                if report.filled:
                    total_notional += report.fill_qty * report.fill_price
                    total_commission += report.commission
                    total_slippage += report.slippage_bps
                    fill_count += 1

            except Exception as e:
                logger.error(f"Execution error for {order_dict}: {e}")
                executions.append(ExecutionReport(
                    order=Order(
                        symbol=order_dict.get("symbol", "?"),
                        side=order_dict.get("side", "?"),
                        quantity=order_dict.get("quantity", 0),
                    ),
                    submitted=False,
                    filled=False,
                    fill_qty=0,
                    fill_price=0,
                    slippage_bps=0,
                    commission=0,
                    status="ERROR",
                    error=str(e),
                ))

        avg_slippage = total_slippage / fill_count if fill_count > 0 else 0

        return BatchExecutionReport(
            sleeve=sleeve,
            total_orders=len(orders),
            orders_submitted=sum(1 for e in executions if e.submitted),
            orders_filled=fill_count,
            total_notional=total_notional,
            total_commission=total_commission,
            average_slippage_bps=avg_slippage,
            executions=executions,
        )

    async def _execute_single(
        self,
        order_dict: Dict,
        check_margin: bool,
    ) -> ExecutionReport:
        """Execute a single order."""
        symbol = order_dict["symbol"]
        side = order_dict["side"]
        quantity = order_dict["quantity"]
        sleeve = order_dict.get("sleeve", "")
        reason = order_dict.get("reason", "")

        # Get current quote
        quote = await self.ibkr.get_quote(symbol)
        if quote is None:
            raise ValueError(f"Could not get quote for {symbol}")

        # Calculate limit price (marketable)
        limit_price = self._calculate_limit_price(quote, side)

        # Round to tick size
        min_tick = await self.ibkr.get_min_tick(symbol)
        from ..ibkr.client import round_to_tick
        limit_price = round_to_tick(limit_price, min_tick)

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="LMT",
            limit_price=limit_price,
            sleeve=sleeve,
            reason=reason,
        )

        # Margin check
        if check_margin:
            margin_check = await self.margin.check_order(order)
            if not margin_check.approved:
                self.audit.log_suppression(
                    sleeve=sleeve,
                    symbol=symbol,
                    reason="MARGIN_BLOCK",
                    details={"message": margin_check.reason},
                )
                return ExecutionReport(
                    order=order,
                    submitted=False,
                    filled=False,
                    fill_qty=0,
                    fill_price=0,
                    slippage_bps=0,
                    commission=0,
                    status="MARGIN_BLOCKED",
                    error=margin_check.reason,
                )

        # Log order intent
        self.audit.log_order(
            sleeve=sleeve,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="LMT",
            limit_price=limit_price,
            reason=reason,
        )

        # Submit order
        status = await self.ibkr.place_order(order)

        if status is None:
            return ExecutionReport(
                order=order,
                submitted=False,
                filled=False,
                fill_qty=0,
                fill_price=0,
                slippage_bps=0,
                commission=0,
                status="SUBMIT_FAILED",
                error="Order submission returned None",
            )

        order.order_id = status.order_id

        # Wait for fill
        fill_status = await self._wait_for_fill(
            status.order_id,
            timeout=self.FILL_TIMEOUT_SECONDS,
        )

        if fill_status is None or not fill_status.is_filled:
            # Order not filled - may need to cancel
            if fill_status and fill_status.is_active:
                await self.ibkr.cancel_order(status.order_id)

            return ExecutionReport(
                order=order,
                submitted=True,
                filled=False,
                fill_qty=fill_status.filled_quantity if fill_status else 0,
                fill_price=fill_status.avg_fill_price if fill_status else 0,
                slippage_bps=0,
                commission=0,
                status=fill_status.status if fill_status else "TIMEOUT",
            )

        # Calculate slippage
        slippage_bps = self._calculate_slippage(
            quote, side, fill_status.avg_fill_price
        )

        # Get commission from executions
        commission = await self._get_order_commission(status.order_id)

        # Log fill
        self.audit.log_fill(
            sleeve=sleeve,
            symbol=symbol,
            side=side,
            quantity=fill_status.filled_quantity,
            fill_price=fill_status.avg_fill_price,
            commission=commission,
            slippage_bps=slippage_bps,
        )

        # Alert on high slippage
        if slippage_bps > self.MAX_SLIPPAGE_BPS:
            logger.warning(
                f"High slippage on {symbol}: {slippage_bps:.1f} bps "
                f"(threshold: {self.MAX_SLIPPAGE_BPS})"
            )

        return ExecutionReport(
            order=order,
            submitted=True,
            filled=True,
            fill_qty=fill_status.filled_quantity,
            fill_price=fill_status.avg_fill_price,
            slippage_bps=slippage_bps,
            commission=commission,
            status="FILLED",
        )

    def _calculate_limit_price(
        self,
        quote: Quote,
        side: str,
    ) -> float:
        """
        Calculate marketable limit price.

        For buys: mid + buffer (pay up slightly)
        For sells: mid - buffer (accept slightly less)
        """
        mid = quote.mid
        buffer = mid * (self.LIMIT_BUFFER_BPS / 10000)

        if side == "BUY":
            # Pay up to ask + small buffer for fills
            return min(quote.ask + buffer, mid + 2 * buffer) if quote.ask > 0 else mid + buffer
        else:
            # Accept down to bid - small buffer
            return max(quote.bid - buffer, mid - 2 * buffer) if quote.bid > 0 else mid - buffer

    def _calculate_slippage(
        self,
        quote: Quote,
        side: str,
        fill_price: float,
    ) -> float:
        """
        Calculate slippage in basis points.

        Slippage = (fill_price - mid) / mid * 10000
        Positive = paid more (buys) or received less (sells) than mid
        """
        mid = quote.mid
        if mid <= 0:
            return 0

        if side == "BUY":
            # Positive slippage means paid more than mid
            slippage = (fill_price - mid) / mid * 10000
        else:
            # Positive slippage means received less than mid
            slippage = (mid - fill_price) / mid * 10000

        return slippage

    async def _wait_for_fill(
        self,
        order_id: int,
        timeout: int,
    ) -> Optional[OrderStatusInfo]:
        """Wait for order to fill or timeout."""
        start_time = datetime.now()

        while True:
            status = await self.ibkr.get_order_status(order_id)

            if status is None:
                return None

            if status.is_filled:
                return status

            if status.is_cancelled:
                return status

            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                return status

            # Wait before next check
            await asyncio.sleep(0.5)

    async def _get_order_commission(self, order_id: int) -> float:
        """Get total commission for an order."""
        executions = await self.ibkr.get_executions()

        total = 0
        for fill in executions:
            if fill.order_id == order_id:
                total += fill.commission

        return total
