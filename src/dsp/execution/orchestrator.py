"""
Daily orchestrator for DSP-100K.

The main execution engine that coordinates:
- Pre-market data refresh
- Signal generation for all sleeves
- Risk checks and position scaling
- Order generation and execution
- Post-trade reconciliation

Per SPEC_DSP_100K.md Section 5.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..data.cache import DataCache
from ..data.fetcher import DataFetcher
from ..ibkr import IBKRClient, AccountSummary, Position
from ..risk import RiskManager, RiskStatus, RiskAlert
from ..risk.margin import MarginMonitor
from ..sleeves import SleeveA, SleeveB, SleeveC, SleeveDM
from ..utils.config import DSPConfig
from ..utils.logging import get_audit_logger, setup_logging
from ..utils.time import MarketCalendar, get_ny_time, is_market_open
from .order_executor import OrderExecutor, BatchExecutionReport

logger = logging.getLogger(__name__)

SPY_SYMBOL = "SPY"


class OrchestratorState(str, Enum):
    """Orchestrator states."""
    IDLE = "idle"
    RUNNING = "running"
    HALTED = "halted"
    ERROR = "error"


@dataclass
class DailyPlan:
    """Daily execution plan."""
    as_of_date: date
    sleeve_a_orders: List[Dict]
    sleeve_b_orders: List[Dict]
    sleeve_dm_orders: List[Dict]
    sleeve_c_orders: List[Dict]
    risk_status: RiskStatus
    scale_factor: float
    estimated_turnover: float
    plan_time: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of daily execution."""
    as_of_date: date
    success: bool
    sleeve_a_report: Optional[BatchExecutionReport]
    sleeve_b_report: Optional[BatchExecutionReport]
    sleeve_dm_report: Optional[BatchExecutionReport]
    sleeve_c_report: Optional[BatchExecutionReport]
    risk_status: RiskStatus
    total_commission: float
    total_slippage_bps: float
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None


class DailyOrchestrator:
    """
    Daily execution orchestrator for DSP-100K.

    Coordinates all daily trading activities:
    1. Pre-market: Data refresh, risk check
    2. Signal generation: Sleeve B trends, Sleeve C hedges
    3. Position sizing: Apply risk scale factor
    4. Execution: Place orders in execution window
    5. Post-trade: Reconciliation, reporting
    """

    def __init__(self, config: DSPConfig):
        """
        Initialize orchestrator.

        Args:
            config: Complete DSP configuration
        """
        self.config = config
        self.state = OrchestratorState.IDLE
        self.audit = get_audit_logger()
        self.calendar = MarketCalendar()

        # Initialize components (will be fully initialized in connect())
        self._ibkr: Optional[IBKRClient] = None
        self._fetcher: Optional[DataFetcher] = None
        self._risk: Optional[RiskManager] = None
        self._margin: Optional[MarginMonitor] = None
        self._executor: Optional[OrderExecutor] = None
        self._sleeve_a: Optional[SleeveA] = None
        self._sleeve_b: Optional[SleeveB] = None
        self._sleeve_dm: Optional[SleeveDM] = None
        self._sleeve_c: Optional[SleeveC] = None
        self._cache: Optional[DataCache] = None

        # Tracking
        self._last_execution: Optional[ExecutionResult] = None
        self._positions: Dict[str, Position] = {}
        self._external_positions: Dict[str, Position] = {}
        self._external_positions_blocking: bool = False

    async def initialize(self) -> bool:
        """
        Initialize all components and connect to IBKR.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing DSP-100K orchestrator...")

        try:
            # Initialize IBKR client
            self._ibkr = IBKRClient(
                host=self.config.ibkr.host,
                port=self.config.ibkr.port,
                client_id=self.config.ibkr.client_id,
            )

            # Connect to IBKR
            connected = await self._ibkr.connect()
            if not connected:
                logger.error("Failed to connect to IBKR")
                return False

            # Initialize data components
            self._cache = DataCache()
            self._fetcher = DataFetcher(self._ibkr, self._cache)

            # Initialize risk components
            self._margin = MarginMonitor(self._ibkr, self.config.risk.margin_cap)
            self._risk = RiskManager(self.config.risk, self._ibkr, self._fetcher)

            # Initialize execution
            self._executor = OrderExecutor(
                self._ibkr,
                self._margin,
                window_start=self.config.execution.window_start,
                window_end=self.config.execution.window_end,
            )

            # Initialize sleeves
            if self.config.sleeve_b.enabled and self.config.sleeve_dm.enabled:
                raise ValueError(
                    "Configuration error: sleeve_b and sleeve_dm are both enabled. "
                    "These ETF sleeves can overlap and create unintended duplicate exposure. "
                    "Enable only one."
                )

            if self.config.sleeve_a.enabled:
                if not self.config.sleeve_a_universe:
                    raise ValueError(
                        "Sleeve A is enabled but no universe was loaded. "
                        "Create `config/universes/sleeve_a_universe.yaml` (see docs/spec) "
                        "or set sleeve_a.enabled=false."
                    )
                self._sleeve_a = SleeveA(self.config.sleeve_a, self._fetcher, self.config.sleeve_a_universe)
            else:
                self._sleeve_a = None

            self._sleeve_b = SleeveB(self.config.sleeve_b, self._fetcher)
            if self.config.sleeve_dm.enabled:
                self._sleeve_dm = SleeveDM(self.config.sleeve_dm, self._fetcher)
            else:
                self._sleeve_dm = None

            # Sleeve C (options) - AUTO-DISABLED until options execution implemented
            # The executor only supports STK/ETF; Sleeve C emits OPT orders that would
            # silently fail. Hard-disable it here to prevent false sense of hedging.
            if self.config.sleeve_c.enabled:
                logger.warning(
                    "⚠️  Sleeve C AUTO-DISABLED: Options execution not implemented. "
                    "Put spread orders would be silently ignored by executor. "
                    "Set sleeve_c.enabled=false in config to suppress this warning."
                )
            self._sleeve_c_enabled = False  # Override config until options work
            self._sleeve_c = SleeveC(self.config.sleeve_c, self._ibkr)

            # Load current positions
            await self._sync_positions()

            self.state = OrchestratorState.IDLE
            logger.info("✅ DSP-100K orchestrator initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = OrchestratorState.ERROR
            return False

    async def shutdown(self) -> None:
        """Shutdown orchestrator and disconnect from IBKR."""
        logger.info("Shutting down DSP-100K orchestrator...")

        if self._ibkr:
            await self._ibkr.disconnect()

        self.state = OrchestratorState.IDLE
        logger.info("Orchestrator shutdown complete")

    async def _sync_positions(self) -> None:
        """Sync current positions from IBKR."""
        self._positions = await self._ibkr.get_positions()

        # External positions safety check:
        # - Always block non-STK positions (FUT/OPT/etc).
        # - Also block STK positions that are not in the enabled sleeves' managed universes
        #   to prevent mixing strategies (contaminates margin/risk + invalidates paper tests).
        managed_symbols: set[str] = set()
        if self._sleeve_a is not None and self.config.sleeve_a.enabled:
            managed_symbols |= set(self._sleeve_a.universe_symbols) | {SPY_SYMBOL}
        if self.config.sleeve_b.enabled:
            managed_symbols |= set(self._sleeve_b.symbols)
        if self._sleeve_dm is not None and self.config.sleeve_dm.enabled:
            managed_symbols |= set(self._sleeve_dm.symbols)

        self._external_positions = {}
        for symbol, pos in self._positions.items():
            if pos.quantity == 0:
                continue
            sec_type = (pos.sec_type or "").upper()
            if sec_type != "STK":
                self._external_positions[symbol] = pos
                continue
            if managed_symbols and symbol not in managed_symbols:
                self._external_positions[symbol] = pos
        self._external_positions_blocking = (
            bool(self._external_positions) and not bool(self.config.general.allow_external_positions)
        )
        if self._external_positions:
            summary = ", ".join(
                f"{p.symbol}({(p.sec_type or '').upper()})={p.quantity}"
                for p in list(self._external_positions.values())[:10]
            )
            more = "" if len(self._external_positions) <= 10 else f" …(+{len(self._external_positions) - 10} more)"
            if self._external_positions_blocking:
                logger.warning(
                    "⚠️  External/unmanaged positions detected: %s%s. "
                    "Trading will be BLOCKED unless you flatten them or set "
                    "general.allow_external_positions=true (or DSP_ALLOW_EXTERNAL_POSITIONS=true).",
                    summary,
                    more,
                )
            else:
                logger.warning(
                    "⚠️  External/unmanaged positions detected: %s%s. "
                    "Proceeding because allow_external_positions=true.",
                    summary,
                    more,
                )

        # Update sleeves with current positions
        if self._sleeve_a is not None:
            sleeve_a_allowed = set(self._sleeve_a.universe_symbols) | {SPY_SYMBOL}
            sleeve_a_positions = {
                s: int(p.quantity) for s, p in self._positions.items()
                if s in sleeve_a_allowed
            }
            self._sleeve_a.set_positions(sleeve_a_positions)

        sleeve_b_positions = {
            s: int(p.quantity) for s, p in self._positions.items()
            if s in self._sleeve_b.symbols
        }
        self._sleeve_b.set_positions(sleeve_b_positions)

        if self._sleeve_dm is not None:
            sleeve_dm_positions = {
                s: int(p.quantity) for s, p in self._positions.items()
                if s in set(self._sleeve_dm.symbols)
            }
            self._sleeve_dm.set_positions(sleeve_dm_positions)

        logger.info(f"Synced {len(self._positions)} positions from IBKR")

    async def run_daily(
        self,
        force: bool = False,
    ) -> ExecutionResult:
        """
        Run complete daily execution cycle.

        Args:
            force: Force execution even outside normal hours

        Returns:
            Execution result with details
        """
        # Use the latest *complete* trading session as the signal date. Execution happens
        # during the current session's window (09:35–10:15 ET), which is the next trading day.
        today = self.calendar.get_latest_complete_session()
        start_time = datetime.now()

        # Check if market is open (or force)
        if not force and not is_market_open():
            logger.info("Market is closed, skipping execution")
            return ExecutionResult(
                as_of_date=today,
                success=False,
                sleeve_a_report=None,
                sleeve_b_report=None,
                sleeve_dm_report=None,
                sleeve_c_report=None,
                risk_status=await self._risk.get_status(),
                total_commission=0,
                total_slippage_bps=0,
                errors=["Market closed"],
            )

        self.state = OrchestratorState.RUNNING
        errors: List[str] = []

        try:
            # 1. Pre-market preparation
            logger.info("=" * 50)
            logger.info(f"Starting daily execution for {today}")
            logger.info("=" * 50)

            # Sync positions
            await self._sync_positions()

            if self._external_positions_blocking:
                msg = (
                    "External (non-STK) positions are present in the account; "
                    "DSP-100K v1 refuses to trade to avoid mixing strategies. "
                    "Flatten them first, or set general.allow_external_positions=true "
                    "(DSP_ALLOW_EXTERNAL_POSITIONS=true) to override."
                )
                logger.error("🛑 %s", msg)
                self.state = OrchestratorState.ERROR
                return ExecutionResult(
                    as_of_date=today,
                    success=False,
                    sleeve_a_report=None,
                    sleeve_b_report=None,
                    sleeve_dm_report=None,
                    sleeve_c_report=None,
                    risk_status=await self._risk.get_status(today),
                    total_commission=0,
                    total_slippage_bps=0,
                    errors=[msg],
                )

            # Get account summary
            summary = await self._ibkr.get_account_summary()
            logger.info(f"Account NLV: {summary.currency} {summary.nlv:,.2f}")

            # 2. Risk check
            risk_status = await self._risk.get_status(today)
            logger.info(f"Risk status: {risk_status.level.value}")

            if self._risk.is_trading_halted:
                logger.error("🛑 Trading halted due to risk breach")
                self.state = OrchestratorState.HALTED
                return ExecutionResult(
                    as_of_date=today,
                    success=False,
                    sleeve_a_report=None,
                    sleeve_b_report=None,
                    sleeve_dm_report=None,
                    sleeve_c_report=None,
                    risk_status=risk_status,
                    total_commission=0,
                    total_slippage_bps=0,
                    errors=["Trading halted - risk breach"],
                )

            # 3. Generate daily plan
            plan = await self._generate_plan(today, summary, risk_status)
            logger.info(
                "Generated plan: Sleeve A=%d orders, Sleeve B=%d orders, Sleeve DM=%d orders",
                len(plan.sleeve_a_orders),
                len(plan.sleeve_b_orders),
                len(plan.sleeve_dm_orders),
            )
            logger.info(
                "Scale factor (effective): %.2f (risk_mgr=%.2f, manual=%.2f)",
                plan.scale_factor,
                risk_status.scale_factor,
                float(self.config.general.risk_scale),
            )

            # 4. Execute Sleeve A
            sleeve_a_report = None
            if plan.sleeve_a_orders:
                logger.info("Executing Sleeve A orders...")
                sleeve_a_report = await self._executor.execute_orders(
                    plan.sleeve_a_orders,
                    force_execution=force,
                )
                logger.info(
                    "Sleeve A: %d/%d filled",
                    sleeve_a_report.orders_filled,
                    sleeve_a_report.total_orders,
                )

            # 5. Execute Sleeve B
            sleeve_b_report = None
            if plan.sleeve_b_orders:
                logger.info("Executing Sleeve B orders...")
                sleeve_b_report = await self._executor.execute_orders(
                    plan.sleeve_b_orders,
                    force_execution=force,
                )
                logger.info(
                    f"Sleeve B: {sleeve_b_report.orders_filled}/{sleeve_b_report.total_orders} filled"
                )

            # 5b. Execute Sleeve DM
            sleeve_dm_report = None
            if plan.sleeve_dm_orders:
                logger.info("Executing Sleeve DM orders...")
                sleeve_dm_report = await self._executor.execute_orders(
                    plan.sleeve_dm_orders,
                    force_execution=force,
                )
                logger.info(
                    "Sleeve DM: %d/%d filled",
                    sleeve_dm_report.orders_filled,
                    sleeve_dm_report.total_orders,
                )

            # 6. Execute Sleeve C (hedges)
            sleeve_c_report = None
            if plan.sleeve_c_orders:
                logger.info("Executing Sleeve C orders...")
                sleeve_c_report = await self._executor.execute_orders(
                    plan.sleeve_c_orders,
                    force_execution=force,
                )
                logger.info(
                    f"Sleeve C: {sleeve_c_report.orders_filled}/{sleeve_c_report.total_orders} filled"
                )

            # 7. Post-trade reconciliation
            await self._sync_positions()

            # Calculate totals
            total_commission = 0
            total_slippage = 0
            fill_count = 0

            if sleeve_a_report:
                total_commission += sleeve_a_report.total_commission
                if sleeve_a_report.orders_filled > 0:
                    total_slippage += sleeve_a_report.average_slippage_bps * sleeve_a_report.orders_filled
                    fill_count += sleeve_a_report.orders_filled

            if sleeve_b_report:
                total_commission += sleeve_b_report.total_commission
                if sleeve_b_report.orders_filled > 0:
                    total_slippage += sleeve_b_report.average_slippage_bps * sleeve_b_report.orders_filled
                    fill_count += sleeve_b_report.orders_filled

            if sleeve_dm_report:
                total_commission += sleeve_dm_report.total_commission
                if sleeve_dm_report.orders_filled > 0:
                    total_slippage += sleeve_dm_report.average_slippage_bps * sleeve_dm_report.orders_filled
                    fill_count += sleeve_dm_report.orders_filled

            if sleeve_c_report:
                total_commission += sleeve_c_report.total_commission
                if sleeve_c_report.orders_filled > 0:
                    total_slippage += sleeve_c_report.average_slippage_bps * sleeve_c_report.orders_filled
                    fill_count += sleeve_c_report.orders_filled

            avg_slippage = total_slippage / fill_count if fill_count > 0 else 0

            # Log completion
            self.audit.log(
                "DAILY_EXECUTION_COMPLETE",
                {
                    "date": str(today),
                    "sleeve_a_fills": sleeve_a_report.orders_filled if sleeve_a_report else 0,
                    "sleeve_b_fills": sleeve_b_report.orders_filled if sleeve_b_report else 0,
                    "sleeve_dm_fills": sleeve_dm_report.orders_filled if sleeve_dm_report else 0,
                    "sleeve_c_fills": sleeve_c_report.orders_filled if sleeve_c_report else 0,
                    "total_commission": total_commission,
                    "avg_slippage_bps": avg_slippage,
                    "scale_factor": plan.scale_factor,
                },
            )

            result = ExecutionResult(
                as_of_date=today,
                success=True,
                sleeve_a_report=sleeve_a_report,
                sleeve_b_report=sleeve_b_report,
                sleeve_dm_report=sleeve_dm_report,
                sleeve_c_report=sleeve_c_report,
                risk_status=risk_status,
                total_commission=total_commission,
                total_slippage_bps=avg_slippage,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now(),
            )

            self._last_execution = result
            self.state = OrchestratorState.IDLE

            logger.info("=" * 50)
            logger.info("Daily execution complete")
            logger.info("=" * 50)

            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            errors.append(str(e))
            self.state = OrchestratorState.ERROR

            return ExecutionResult(
                as_of_date=today,
                success=False,
                sleeve_a_report=None,
                sleeve_b_report=None,
                sleeve_dm_report=None,
                sleeve_c_report=None,
                risk_status=await self._risk.get_status(),
                total_commission=0,
                total_slippage_bps=0,
                errors=errors,
                start_time=start_time,
                end_time=datetime.now(),
            )

    async def _generate_plan(
        self,
        as_of_date: date,
        summary: AccountSummary,
        risk_status: RiskStatus,
    ) -> DailyPlan:
        """Generate daily execution plan."""

        # Calculate NAV allocations
        total_nav = summary.nlv
        sleeve_a_nav = total_nav * float(self.config.general.sleeve_a_nav_pct)
        sleeve_b_nav = total_nav * float(self.config.general.sleeve_b_nav_pct)
        sleeve_dm_nav = total_nav * float(self.config.general.sleeve_dm_nav_pct)

        if (
            sleeve_a_nav + sleeve_b_nav + sleeve_dm_nav
            > total_nav * (1.0 - float(self.config.general.cash_buffer) + 1e-6)
        ):
            logger.warning(
                "Allocations exceed cash buffer: sleeve_a_nav_pct=%.2f sleeve_b_nav_pct=%.2f sleeve_dm_nav_pct=%.2f cash_buffer=%.2f",
                float(self.config.general.sleeve_a_nav_pct),
                float(self.config.general.sleeve_b_nav_pct),
                float(self.config.general.sleeve_dm_nav_pct),
                float(self.config.general.cash_buffer),
            )

        # Apply risk scale factors:
        # - risk_status.scale_factor is computed by RiskManager (drawdown/vol/margin)
        # - config.general.risk_scale is an operator-controlled manual scale (e.g. 0.25 for small-size live)
        scale_factor = risk_status.scale_factor * float(self.config.general.risk_scale)

        sleeve_a_orders: List[Dict] = []
        if self._sleeve_a is not None and self.config.sleeve_a.enabled:
            # Sleeve A uses last close prices for sizing by default (signal_date = as_of_date).
            # The executor will fetch real-time quotes per order when submitting marketable limits.
            sleeve_a_adjustment = await self._sleeve_a.generate_adjustment(
                sleeve_nav=sleeve_a_nav * scale_factor,
                prices={},
                as_of_date=as_of_date,
            )
            sleeve_a_orders = self._sleeve_a.get_target_orders(sleeve_a_adjustment)

        # Generate Sleeve B adjustment
        sleeve_b_orders: List[Dict] = []
        estimated_turnover = 0.0
        if self.config.sleeve_b.enabled:
            prices = await self._get_current_prices(self._sleeve_b.symbols)
            adjustment = await self._sleeve_b.generate_adjustment(
                sleeve_nav=sleeve_b_nav * scale_factor,
                prices=prices,
                as_of_date=as_of_date,
            )
            estimated_turnover = adjustment.estimated_turnover

            # Convert to orders
            if adjustment.rebalance_needed:
                sleeve_b_orders = self._sleeve_b.get_target_orders(adjustment)

        # Generate Sleeve DM (ETF Dual Momentum) adjustment
        sleeve_dm_orders: List[Dict] = []
        if self._sleeve_dm is not None and self.config.sleeve_dm.enabled:
            dm_prices = await self._get_current_prices(self._sleeve_dm.symbols)
            dm_adjustment = await self._sleeve_dm.generate_adjustment(
                sleeve_nav=sleeve_dm_nav * scale_factor,
                prices=dm_prices,
                as_of_date=as_of_date,
            )
            estimated_turnover = max(float(estimated_turnover), float(dm_adjustment.estimated_turnover))
            if dm_adjustment.rebalance_needed:
                sleeve_dm_orders = self._sleeve_dm.get_target_orders(dm_adjustment)

        # Generate Sleeve C orders (hedges) - DISABLED until options execution works
        sleeve_c_orders = []
        if self._sleeve_c_enabled:
            hedge_status = await self._sleeve_c.get_status(total_nav, as_of_date)

            if hedge_status.needs_roll:
                roll_plan = await self._sleeve_c.plan_roll(total_nav, as_of_date)
                if roll_plan:
                    # Close existing spreads
                    # (simplified - would need actual close orders)
                    # Open new spreads
                    sleeve_c_orders = self._sleeve_c.get_orders(roll_plan.new_spread_target)

            elif hedge_status.needs_increase:
                new_hedge = await self._sleeve_c.plan_new_hedge(total_nav, as_of_date)
                if new_hedge:
                    sleeve_c_orders = self._sleeve_c.get_orders(new_hedge)
        else:
            logger.debug("Sleeve C disabled - skipping hedge orders")

        return DailyPlan(
            as_of_date=as_of_date,
            sleeve_a_orders=sleeve_a_orders,
            sleeve_b_orders=sleeve_b_orders,
            sleeve_dm_orders=sleeve_dm_orders,
            sleeve_c_orders=sleeve_c_orders,
            risk_status=risk_status,
            scale_factor=scale_factor,
            estimated_turnover=estimated_turnover,
        )

    async def _get_current_prices(
        self,
        symbols: List[str],
    ) -> Dict[str, float]:
        """Get current prices for symbols."""
        prices: Dict[str, float] = {}

        for symbol in symbols:
            try:
                quote = await self._ibkr.get_quote(symbol)
                if quote:
                    prices[symbol] = quote.mid
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")

        return prices

    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        return {
            "state": self.state.value,
            "last_execution": {
                "date": str(self._last_execution.as_of_date) if self._last_execution else None,
                "success": self._last_execution.success if self._last_execution else None,
            },
            "positions_count": len(self._positions),
            "ibkr_connected": self._ibkr.is_connected if self._ibkr else False,
        }


async def main():
    """Main entry point for daily execution."""
    import os
    from ..utils.config import load_config

    # Setup logging
    setup_logging(level="INFO", json_format=False)

    # Load configuration
    config_path = os.getenv("DSP_CONFIG_PATH", "config/dsp100k.yaml")
    config = load_config(config_path)

    # Create and run orchestrator
    orchestrator = DailyOrchestrator(config)

    try:
        # Initialize
        if not await orchestrator.initialize():
            logger.error("Failed to initialize orchestrator")
            return

        # Run daily execution
        result = await orchestrator.run_daily()

        if result.success:
            logger.info(f"✅ Daily execution successful")
            logger.info(f"   Commission: ${result.total_commission:.2f}")
            logger.info(f"   Avg Slippage: {result.total_slippage_bps:.1f} bps")
        else:
            logger.error(f"❌ Daily execution failed: {result.errors}")

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
