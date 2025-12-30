"""
Sleeve C: Tail Hedge via SPY Put-Spread.

Implements the protective put spread strategy:
- 25-delta / 10-delta put spreads on SPY
- 30-45 DTE targeting
- 1.25% annual premium budget (~0.104%/month)
- Roll 10 days before expiry

Per SPEC_DSP_100K.md Section 2.3.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..ibkr import IBKRClient
from ..ibkr.models import OptionContract, OptionChain, PutSpread
from ..utils.config import SleeveCConfig
from ..utils.logging import get_audit_logger
from ..utils.time import MarketCalendar

logger = logging.getLogger(__name__)


@dataclass
class SpreadTarget:
    """Target put spread specification."""
    underlying: str
    expiry: date
    long_strike: float      # Higher strike (25-delta)
    short_strike: float     # Lower strike (10-delta)
    contracts: int          # Number of contracts
    max_premium: float      # Maximum premium to pay
    estimated_premium: float  # Estimated cost


@dataclass
class HedgeStatus:
    """Current hedge status."""
    active_spreads: List[PutSpread]
    total_contracts: int
    total_delta: float
    total_premium_paid: float
    days_to_expiry: int
    needs_roll: bool
    needs_increase: bool
    coverage_ratio: float   # Notional protected / portfolio NAV


@dataclass
class RollPlan:
    """Plan for rolling expiring spreads."""
    spreads_to_close: List[PutSpread]
    new_spread_target: SpreadTarget
    estimated_roll_cost: float
    roll_reason: str


class PutSpreadManager:
    """
    Manages put spread positions for Sleeve C.

    Handles:
    - Strike selection (delta-based)
    - Expiry selection (30-45 DTE)
    - Position sizing (budget-based)
    - Rolling logic
    """

    # Target deltas
    LONG_PUT_DELTA = -0.25   # 25-delta put
    SHORT_PUT_DELTA = -0.10  # 10-delta put

    # DTE parameters
    TARGET_DTE_MIN = 30
    TARGET_DTE_MAX = 45
    ROLL_TRIGGER_DTE = 10   # Roll when DTE drops below this

    def __init__(
        self,
        ibkr_client: IBKRClient,
        calendar: Optional[MarketCalendar] = None,
    ):
        """
        Initialize put spread manager.

        Args:
            ibkr_client: IBKR client for options data
            calendar: Market calendar
        """
        self.ibkr = ibkr_client
        self.calendar = calendar or MarketCalendar()

    async def find_strikes_by_delta(
        self,
        underlying: str,
        target_long_delta: float,
        target_short_delta: float,
        min_dte: int = TARGET_DTE_MIN,
        max_dte: int = TARGET_DTE_MAX,
    ) -> Optional[SpreadTarget]:
        """
        Find optimal strikes for a put spread based on delta.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            target_long_delta: Target delta for long put (-0.25)
            target_short_delta: Target delta for short put (-0.10)
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry

        Returns:
            SpreadTarget with selected strikes, or None if not found
        """
        try:
            # Get option chain
            chain = await self.ibkr.get_option_chain(underlying)

            if chain is None:
                logger.error(f"Could not get option chain for {underlying}")
                return None

            # Filter expirations by DTE
            valid_expiries = chain.filter_by_dte(min_dte, max_dte)

            if not valid_expiries:
                logger.warning(f"No valid expiries found for {underlying} ({min_dte}-{max_dte} DTE)")
                return None

            # Use the nearest valid expiry
            target_expiry = valid_expiries[0]

            # Find strikes closest to target deltas
            long_strike = await self._find_strike_by_delta(
                underlying=underlying,
                expiry=target_expiry,
                target_delta=target_long_delta,
                chain=chain,
            )

            short_strike = await self._find_strike_by_delta(
                underlying=underlying,
                expiry=target_expiry,
                target_delta=target_short_delta,
                chain=chain,
            )

            if long_strike is None or short_strike is None:
                logger.error("Could not find suitable strikes")
                return None

            if long_strike <= short_strike:
                logger.error(f"Invalid spread: long_strike ({long_strike}) <= short_strike ({short_strike})")
                return None

            # Get quotes to estimate premium
            long_quote = await self.ibkr.get_option_quote(
                underlying, long_strike, target_expiry, "P"
            )
            short_quote = await self.ibkr.get_option_quote(
                underlying, short_strike, target_expiry, "P"
            )

            if long_quote is None or short_quote is None:
                estimated_premium = 0.0
            else:
                long_mid = long_quote.mid or 0.0
                short_mid = short_quote.mid or 0.0
                estimated_premium = max(0, long_mid - short_mid)

            return SpreadTarget(
                underlying=underlying,
                expiry=target_expiry,
                long_strike=long_strike,
                short_strike=short_strike,
                contracts=0,  # To be set by sizing logic
                max_premium=0.0,  # To be set by budget logic
                estimated_premium=estimated_premium,
            )

        except Exception as e:
            logger.error(f"Error finding strikes: {e}")
            return None

    async def _find_strike_by_delta(
        self,
        underlying: str,
        expiry: date,
        target_delta: float,
        chain: OptionChain,
    ) -> Optional[float]:
        """
        Find strike closest to target delta.

        Args:
            underlying: Underlying symbol
            expiry: Expiration date
            target_delta: Target delta (negative for puts)
            chain: Option chain

        Returns:
            Strike price or None
        """
        best_strike = None
        best_delta_diff = float('inf')

        for strike in chain.strikes:
            contract = chain.get_contract(strike, expiry, "P")
            if contract is None:
                # Fetch individual quote
                contract = await self.ibkr.get_option_quote(
                    underlying, strike, expiry, "P"
                )

            if contract is None or contract.delta is None:
                continue

            delta_diff = abs(contract.delta - target_delta)
            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_strike = strike

        return best_strike

    def calculate_contracts(
        self,
        spread_target: SpreadTarget,
        portfolio_nav: float,
        annual_budget_pct: float,
        months_remaining: int = 12,
    ) -> int:
        """
        Calculate number of contracts based on budget.

        Args:
            spread_target: Target spread specification
            portfolio_nav: Total portfolio NAV
            annual_budget_pct: Annual budget as percentage (0.0125 = 1.25%)
            months_remaining: Months remaining in budget year

        Returns:
            Number of contracts to purchase
        """
        if spread_target.estimated_premium <= 0:
            return 0

        # Monthly budget
        monthly_budget = portfolio_nav * annual_budget_pct / 12

        # Available budget (proportional to remaining months)
        available_budget = monthly_budget * min(months_remaining, 2)  # Max 2 months per trade

        # Premium per contract (in dollars, each contract = 100 shares)
        premium_per_contract = spread_target.estimated_premium * 100

        if premium_per_contract <= 0:
            return 0

        # Calculate contracts
        contracts = int(available_budget / premium_per_contract)

        return max(0, contracts)


class SleeveC:
    """
    Sleeve C: Tail Hedge Implementation.

    Complete implementation of the put spread hedge:
    - 25-delta / 10-delta put spreads on SPY
    - 30-45 DTE with 10-day roll trigger
    - 1.25% annual premium budget
    """

    def __init__(
        self,
        config: SleeveCConfig,
        ibkr_client: IBKRClient,
    ):
        """
        Initialize Sleeve C.

        Args:
            config: Sleeve C configuration
            ibkr_client: IBKR client for options trading
        """
        self.config = config
        self.ibkr = ibkr_client
        self.spread_manager = PutSpreadManager(ibkr_client)
        self.audit = get_audit_logger()
        self.calendar = MarketCalendar()

        # Current positions
        self._active_spreads: List[PutSpread] = []
        self._ytd_premium_spent: float = 0.0
        self._budget_year_start: Optional[date] = None

    def set_positions(self, spreads: List[PutSpread]) -> None:
        """
        Update current spread positions.

        Args:
            spreads: List of active put spreads
        """
        self._active_spreads = spreads.copy()

    def set_ytd_premium(self, amount: float, year_start: date) -> None:
        """
        Set year-to-date premium spent.

        Args:
            amount: Total premium spent this year
            year_start: Budget year start date
        """
        self._ytd_premium_spent = amount
        self._budget_year_start = year_start

    async def get_status(
        self,
        portfolio_nav: float,
        as_of_date: Optional[date] = None,
    ) -> HedgeStatus:
        """
        Get current hedge status.

        Args:
            portfolio_nav: Total portfolio NAV
            as_of_date: Reference date

        Returns:
            Complete hedge status
        """
        if as_of_date is None:
            as_of_date = self.calendar.get_latest_complete_session()

        total_contracts = 0
        total_delta = 0.0
        total_premium = 0.0
        min_dte = float('inf')

        for spread in self._active_spreads:
            total_contracts += 1  # Each spread is one "unit"
            if spread.current_delta is not None:
                total_delta += spread.current_delta
            total_premium += spread.entry_cost
            dte = spread.dte
            if dte < min_dte:
                min_dte = dte

        # If no spreads, min_dte stays as inf; convert to 0 for status but
        # needs_roll should be False when there are no spreads to roll.
        has_spreads = bool(self._active_spreads)
        min_dte = int(min_dte) if min_dte != float('inf') else 0

        # Check if roll is needed - only if we have spreads AND they're near expiry
        needs_roll = has_spreads and (min_dte <= self.config.roll_dte)

        # Calculate coverage ratio
        if portfolio_nav > 0 and total_contracts > 0:
            # Each SPY contract covers ~$50k at current prices
            spy_multiplier = 100  # 100 shares per contract
            avg_strike = sum(s.long_put.strike for s in self._active_spreads) / len(self._active_spreads) if self._active_spreads else 500
            coverage_ratio = (total_contracts * spy_multiplier * avg_strike) / portfolio_nav
        else:
            coverage_ratio = 0.0

        # Check if we need more hedges
        target_budget = portfolio_nav * self.config.annual_budget
        remaining_budget = target_budget - self._ytd_premium_spent
        needs_increase = remaining_budget > target_budget * 0.08  # More than 1 month budget remaining

        return HedgeStatus(
            active_spreads=self._active_spreads.copy(),
            total_contracts=total_contracts,
            total_delta=total_delta,
            total_premium_paid=total_premium,
            days_to_expiry=min_dte,
            needs_roll=needs_roll,
            needs_increase=needs_increase,
            coverage_ratio=coverage_ratio,
        )

    async def plan_roll(
        self,
        portfolio_nav: float,
        as_of_date: Optional[date] = None,
    ) -> Optional[RollPlan]:
        """
        Plan a roll of expiring spreads.

        Args:
            portfolio_nav: Total portfolio NAV
            as_of_date: Reference date

        Returns:
            Roll plan if roll is needed, None otherwise
        """
        if as_of_date is None:
            as_of_date = self.calendar.get_latest_complete_session()

        # Find spreads that need to be rolled
        spreads_to_roll = [
            s for s in self._active_spreads
            if s.dte <= self.config.roll_dte
        ]

        if not spreads_to_roll:
            return None

        # Find new spread target
        new_target = await self.spread_manager.find_strikes_by_delta(
            underlying=self.config.underlying,
            target_long_delta=self.config.long_delta_target,
            target_short_delta=self.config.short_delta_target,
            min_dte=self.config.target_dte_min,
            max_dte=self.config.target_dte_max,
        )

        if new_target is None:
            logger.error("Could not find new spread for roll")
            return None

        # Calculate contracts based on budget
        months_in_year = 12
        if self._budget_year_start:
            months_remaining = max(1, months_in_year - (as_of_date.month - self._budget_year_start.month))
        else:
            months_remaining = months_in_year

        contracts = self.spread_manager.calculate_contracts(
            spread_target=new_target,
            portfolio_nav=portfolio_nav,
            annual_budget_pct=self.config.annual_budget,
            months_remaining=months_remaining,
        )

        new_target.contracts = contracts
        new_target.max_premium = portfolio_nav * self.config.annual_budget / 12

        # Estimate roll cost
        close_credit = sum(s.current_value for s in spreads_to_roll)
        open_debit = new_target.estimated_premium * new_target.contracts * 100
        estimated_roll_cost = open_debit - close_credit

        return RollPlan(
            spreads_to_close=spreads_to_roll,
            new_spread_target=new_target,
            estimated_roll_cost=estimated_roll_cost,
            roll_reason=f"DTE <= {self.config.roll_dte}",
        )

    async def plan_new_hedge(
        self,
        portfolio_nav: float,
        as_of_date: Optional[date] = None,
    ) -> Optional[SpreadTarget]:
        """
        Plan a new hedge position.

        Args:
            portfolio_nav: Total portfolio NAV
            as_of_date: Reference date

        Returns:
            Spread target if hedge is needed, None otherwise
        """
        if as_of_date is None:
            as_of_date = self.calendar.get_latest_complete_session()

        # Check budget
        annual_budget = portfolio_nav * self.config.annual_budget
        remaining_budget = annual_budget - self._ytd_premium_spent

        if remaining_budget < annual_budget * 0.05:  # Less than 5% of budget remaining
            logger.info("Hedge budget exhausted for year")
            return None

        # Check if we have active hedges
        if self._active_spreads:
            min_dte = min(s.dte for s in self._active_spreads)
            if min_dte > self.config.roll_dte:
                logger.info(f"Active hedges have {min_dte} DTE, no new hedge needed")
                return None

        # Find spread target
        target = await self.spread_manager.find_strikes_by_delta(
            underlying=self.config.underlying,
            target_long_delta=self.config.long_delta_target,
            target_short_delta=self.config.short_delta_target,
            min_dte=self.config.target_dte_min,
            max_dte=self.config.target_dte_max,
        )

        if target is None:
            return None

        # Calculate contracts
        months_remaining = max(1, 12 - as_of_date.month + 1)
        contracts = self.spread_manager.calculate_contracts(
            spread_target=target,
            portfolio_nav=portfolio_nav,
            annual_budget_pct=self.config.annual_budget,
            months_remaining=months_remaining,
        )

        target.contracts = contracts
        target.max_premium = remaining_budget / 2  # Don't spend more than half remaining budget

        # Log to audit
        self.audit.log(
            "HEDGE_PLAN",
            {
                "sleeve": "C",
                "underlying": target.underlying,
                "expiry": str(target.expiry),
                "long_strike": target.long_strike,
                "short_strike": target.short_strike,
                "contracts": target.contracts,
                "estimated_premium": target.estimated_premium,
            },
        )

        return target

    def get_orders(
        self,
        target: SpreadTarget,
    ) -> List[Dict]:
        """
        Convert spread target to order list.

        Args:
            target: Spread target specification

        Returns:
            List of order dicts for execution
        """
        if target.contracts <= 0:
            return []

        orders = []

        # Long put (buy)
        orders.append({
            "symbol": target.underlying,
            "sec_type": "OPT",
            "strike": target.long_strike,
            "expiry": target.expiry,
            "right": "P",
            "side": "BUY",
            "quantity": target.contracts,
            "order_type": "LMT",
            "sleeve": "C",
            "reason": "long_leg_25delta",
        })

        # Short put (sell)
        orders.append({
            "symbol": target.underlying,
            "sec_type": "OPT",
            "strike": target.short_strike,
            "expiry": target.expiry,
            "right": "P",
            "side": "SELL",
            "quantity": target.contracts,
            "order_type": "LMT",
            "sleeve": "C",
            "reason": "short_leg_10delta",
        })

        return orders

    def record_trade(
        self,
        spread: PutSpread,
        premium_paid: float,
    ) -> None:
        """
        Record a completed spread trade.

        Args:
            spread: The traded spread
            premium_paid: Premium paid for the spread
        """
        self._active_spreads.append(spread)
        self._ytd_premium_spent += premium_paid

        self.audit.log(
            "HEDGE_EXECUTED",
            {
                "sleeve": "C",
                "long_strike": spread.long_put.strike,
                "short_strike": spread.short_put.strike,
                "expiry": str(spread.long_put.expiry),
                "premium_paid": premium_paid,
                "ytd_premium": self._ytd_premium_spent,
            },
        )

    def record_close(
        self,
        spread: PutSpread,
        close_value: float,
    ) -> None:
        """
        Record a closed spread.

        Args:
            spread: The closed spread
            close_value: Value received on close
        """
        self._active_spreads = [
            s for s in self._active_spreads
            if s != spread
        ]

        pnl = close_value - spread.entry_cost

        self.audit.log(
            "HEDGE_CLOSED",
            {
                "sleeve": "C",
                "long_strike": spread.long_put.strike,
                "short_strike": spread.short_put.strike,
                "expiry": str(spread.long_put.expiry),
                "close_value": close_value,
                "entry_cost": spread.entry_cost,
                "pnl": pnl,
            },
        )
