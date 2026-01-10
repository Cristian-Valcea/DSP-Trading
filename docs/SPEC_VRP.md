# SPEC_VRP.md — Volatility Risk Premium Strategy Technical Specification

**Version**: 1.0
**Date**: 2026-01-08
**Status**: Pre-Registered Specification (Frozen)
**Author**: Claude
**Audience**: Development Team

---

## 1. Overview

### 1.1 Strategy Summary
Harvest the Variance Risk Premium (VRP) by selling VIX futures in contango, with tail hedging via OTM VIX calls.

### 1.2 Key Characteristics
| Property | Value |
|----------|-------|
| **Asset Class** | Volatility (VIX Futures + VIX Options) |
| **Universe** | VX (CBOE VIX Futures), VIX Options |
| **Rebalance Frequency** | Monthly (3rd trading day) |
| **Signal Type** | Contango + Volatility Momentum + Risk Filters |
| **Risk Management** | Stop-loss + OTM Call Hedge + Cool-Down Rule |
| **Target Volatility** | N/A (position sizing via margin cap) |
| **Max Leverage** | 1.0x (notional ≤ sleeve NAV) |

### 1.3 Data Requirements
| Data Type | Source | Frequency | Historical Depth |
|-----------|--------|-----------|------------------|
| VIX Spot Index | CBOE / Polygon | Daily close | 2010-present |
| VIX Futures (VX) | Databento / CBOE | Daily OHLC + Open Interest | 2010-present |
| VIX1D Index | CBOE | Daily close | 2023-present |
| VVIX Index | CBOE | Daily close | 2010-present |
| VIX Options | CBOE / Databento | Daily (for hedge pricing) | 2015-present |
| Fed Funds Rate | FRED | Daily | 2010-present |

---

## 2. Signal Generation

### 2.1 Primary Signal: Adjusted Contango

```python
def calculate_adjusted_contango(
    vix_spot: float,
    front_month_futures: float,
    days_to_expiry: int,
    risk_free_rate: float
) -> float:
    """
    Calculate TRUE risk premium after removing interest rate component.

    Args:
        vix_spot: Current VIX spot index value
        front_month_futures: Front-month VIX futures price
        days_to_expiry: Days until front-month expiration
        risk_free_rate: Annualized risk-free rate (e.g., 0.045 for 4.5%)

    Returns:
        Adjusted contango in VIX points (TRUE risk premium)
    """
    raw_contango = front_month_futures - vix_spot
    interest_component = vix_spot * risk_free_rate * (days_to_expiry / 365)
    adjusted_contango = raw_contango - interest_component
    return adjusted_contango
```

### 2.2 Entry Filters (ALL Must Pass)

```python
def check_entry_filters(
    adjusted_contango: float,
    vix_spot: float,
    vix_50d_ma: float,
    vix1d: float,
    vvix: float,
    vvix_90th_pct: float,
    vix1d_vix_95th_pct: float,
    cool_down_active: bool
) -> tuple[bool, str]:
    """
    Check all entry filters for VRP strategy.

    Returns:
        (can_enter, reason) - True if all filters pass, else False with reason
    """
    # Filter 1: Minimum adjusted contango
    MIN_CONTANGO = 0.5  # points
    if adjusted_contango < MIN_CONTANGO:
        return False, f"Contango too thin: {adjusted_contango:.2f} < {MIN_CONTANGO}"

    # Filter 2: Volatility momentum (VIX below 50-day MA = vol falling)
    if vix_spot > vix_50d_ma:
        return False, f"Vol rising: VIX {vix_spot:.1f} > 50d MA {vix_50d_ma:.1f}"

    # Filter 3: VIX1D/VIX ratio (short-term fear check)
    vix1d_ratio = vix1d / vix_spot
    VIX1D_HARD_LIMIT = 1.2
    if vix1d_ratio > max(vix1d_vix_95th_pct, VIX1D_HARD_LIMIT):
        return False, f"Short-term fear elevated: VIX1D/VIX {vix1d_ratio:.2f}"

    # Filter 4: VVIX filter (correlation breakdown warning)
    if vvix > vvix_90th_pct:
        return False, f"VVIX elevated: {vvix:.1f} > 90th pct {vvix_90th_pct:.1f}"

    # Filter 5: Cool-down rule (no re-entry after recent stop)
    if cool_down_active:
        return False, "Cool-down active after recent stop-loss"

    # Filter 6: Absolute VIX level (high-risk regime)
    VIX_HIGH_REGIME = 25.0
    if vix_spot > VIX_HIGH_REGIME:
        return False, f"High-vol regime: VIX {vix_spot:.1f} > {VIX_HIGH_REGIME}"

    # ----------------------------------------------------------------
    # OPTIONAL: ML Overlay (Phase 2 Implementation)
    # See: SLEEVE_VRP_ML_ENHANCEMENT.md for full specification
    #
    # ML filter acts as additional entry blocker using ALSTM regime
    # classifier. Only activates AFTER baseline filters pass.
    # Requires: Baseline VRP to pass kill-test first.
    # ----------------------------------------------------------------

    return True, "All filters pass"
```

### 2.3 Exit Signals

```python
def check_exit_signals(
    vix_spot: float,
    position_pnl_pct: float,
    vvix: float,
    vvix_90th_pct: float
) -> tuple[bool, str, float]:
    """
    Check exit conditions during position holding.

    Returns:
        (should_exit, reason, reduction_pct)
        reduction_pct: 1.0 = full exit, 0.5 = reduce 50%, 0.0 = no action
    """
    # Circuit Breaker 1: Hard stop-loss
    STOP_LOSS_PCT = -0.15
    if position_pnl_pct <= STOP_LOSS_PCT:
        return True, f"Stop-loss triggered: {position_pnl_pct:.1%}", 1.0

    # Circuit Breaker 2: VIX level triggers
    VIX_REDUCE_LEVEL = 30.0
    VIX_FLATTEN_LEVEL = 40.0

    if vix_spot >= VIX_FLATTEN_LEVEL:
        return True, f"Emergency flatten: VIX {vix_spot:.1f} >= {VIX_FLATTEN_LEVEL}", 1.0

    if vix_spot >= VIX_REDUCE_LEVEL:
        return True, f"Reduce exposure: VIX {vix_spot:.1f} >= {VIX_REDUCE_LEVEL}", 0.5

    # Circuit Breaker 3: VVIX spike during position
    if vvix > vvix_90th_pct:
        return True, f"VVIX spike: {vvix:.1f} > 90th pct", 0.5

    return False, "No exit signal", 0.0
```

---

## 3. Portfolio Construction

### 3.1 Position Sizing

```python
@dataclass
class VRPPositionConfig:
    """Position sizing parameters for VRP strategy."""
    max_nav_pct: float = 0.10          # Max 10% of sleeve NAV in VIX exposure
    max_margin_pct: float = 0.30       # Max 30% margin utilization
    contract_multiplier: float = 1000  # $1,000 per VIX point
    margin_per_contract: float = 5000  # Approximate margin requirement

def calculate_position_size(
    sleeve_nav: float,
    front_month_price: float,
    config: VRPPositionConfig
) -> int:
    """
    Calculate number of VIX futures contracts to short.

    Returns:
        Number of contracts (integer, always >= 0)
    """
    # Method 1: NAV-based limit
    max_notional = sleeve_nav * config.max_nav_pct
    notional_per_contract = front_month_price * config.contract_multiplier
    contracts_by_nav = int(max_notional / notional_per_contract)

    # Method 2: Margin-based limit
    max_margin = sleeve_nav * config.max_margin_pct
    contracts_by_margin = int(max_margin / config.margin_per_contract)

    # Take the more conservative limit
    num_contracts = min(contracts_by_nav, contracts_by_margin)

    return max(0, num_contracts)
```

### 3.2 Tail Hedge Construction (Long Wings)

```python
@dataclass
class TailHedgeConfig:
    """Configuration for OTM VIX call hedge."""
    strike_offset: float = 25.0      # Strike = VIX spot + offset
    min_strike: float = 35.0         # Minimum absolute strike
    max_cost_pct: float = 0.15       # Max 15% of expected premium for hedge
    contract_ratio: float = 1.0      # 1 call per 1 short future

def construct_tail_hedge(
    vix_spot: float,
    num_futures_contracts: int,
    front_month_price: float,
    front_month_expiry: str,
    option_chain: pd.DataFrame,
    config: TailHedgeConfig
) -> dict:
    """
    Construct OTM VIX call hedge for gap risk protection.

    Args:
        vix_spot: Current VIX spot
        num_futures_contracts: Number of short futures
        front_month_price: Price of the short future (for budget calc)
        front_month_expiry: Expiration date (YYYY-MM-DD)
        option_chain: DataFrame with columns [strike, bid, ask, expiry]
        config: Hedge configuration

    Returns:
        Dict with hedge details or None if no suitable option
    """
    # Calculate target strike
    base_target_strike = max(
        vix_spot + config.strike_offset,
        config.min_strike
    )

    # Calculate max budget per contract (e.g., 15% of futures premium)
    # Futures premium approx = price * 1000 (assuming we capture full value, simplified)
    # Better: Budget based on expected monthly decay or raw premium
    max_hedge_cost = (front_month_price * 1000) * config.max_cost_pct

    # Filter for valid strikes
    calls = option_chain[
        (option_chain['expiry'] == front_month_expiry) &
        (option_chain['strike'] >= base_target_strike)
    ].sort_values('strike')

    if calls.empty:
        return None

    # Select strike that fits in budget
    selected = None
    for _, call in calls.iterrows():
        cost = call['ask'] * 1000 * config.contract_ratio
        if cost <= max_hedge_cost:
            selected = call
            break
    
    if selected is None:
        # If even the furthest OTM is too expensive, log warning or take cheapest
        selected = calls.iloc[-1] # Take highest strike (cheapest)

    # Calculate number of calls to buy
    num_calls = int(num_futures_contracts * config.contract_ratio)

    return {
        'instrument': f'VIX {selected["strike"]:.0f} Call',
        'expiry': front_month_expiry,
        'strike': selected['strike'],
        'quantity': num_calls,
        'side': 'BUY',
        'estimated_cost': selected['ask'] * num_calls * 1000,  # $1000 multiplier
        'max_protection_level': selected['strike']
    }
```

### 3.3 Complete Trade Structure

```python
def construct_vrp_trade(
    sleeve_nav: float,
    market_data: dict,
    option_chain: pd.DataFrame,
    position_config: VRPPositionConfig,
    hedge_config: TailHedgeConfig
) -> dict:
    """
    Construct complete VRP trade with futures short + call hedge.

    Returns:
        Trade structure with both legs
    """
    # Calculate futures position
    num_contracts = calculate_position_size(
        sleeve_nav,
        market_data['front_month_price'],
        position_config
    )

    if num_contracts == 0:
        return {'status': 'NO_TRADE', 'reason': 'Position size too small'}

    # Construct hedge
    hedge = construct_tail_hedge(
        market_data['vix_spot'],
        num_contracts,
        market_data['front_month_price'],
        market_data['front_month_expiry'],
        option_chain,
        hedge_config
    )

    # Build trade structure
    trade = {
        'status': 'READY',
        'legs': [
            {
                'instrument': f"VX {market_data['front_month_expiry']}",
                'side': 'SELL',
                'quantity': num_contracts,
                'order_type': 'LIMIT',
                'limit_price': market_data['front_month_bid'],
                'notional': num_contracts * market_data['front_month_price'] * 1000
            }
        ],
        'total_notional': num_contracts * market_data['front_month_price'] * 1000,
        'margin_required': num_contracts * position_config.margin_per_contract
    }

    if hedge:
        trade['legs'].append(hedge)
        trade['hedge_cost'] = hedge['estimated_cost']
    else:
        trade['hedge_cost'] = 0
        trade['warning'] = 'No suitable hedge option found'

    return trade
```

---

## 4. Execution

### 4.1 Execution Timing

```python
from datetime import datetime, timedelta
import exchange_calendars as xcals

def get_execution_date(year: int, month: int) -> datetime:
    """
    Get the 3rd trading day of the month for execution.

    We avoid the 1st trading day to reduce crowding effects.
    """
    nyse = xcals.get_calendar('NYSE')

    # Get all trading days in the month
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)

    trading_days = nyse.sessions_in_range(start, end)

    if len(trading_days) < 3:
        raise ValueError(f"Not enough trading days in {year}-{month}")

    # Return 3rd trading day (index 2)
    return trading_days[2].to_pydatetime()

EXECUTION_TIME = "09:35:00"  # 5 minutes after open for liquidity
```

### 4.2 Order Types

```python
@dataclass
class OrderConfig:
    """Order execution configuration."""
    futures_order_type: str = 'LIMIT'
    futures_limit_offset: float = 0.05    # Limit = mid + offset (aggressive)
    options_order_type: str = 'LIMIT'
    options_limit_offset: float = 0.10    # Wider for options
    timeout_seconds: int = 300             # 5 minute fill timeout
    slice_threshold: int = 10              # Split orders > 10 contracts

    # ETH (Extended Trading Hours) Configuration
    # VIX futures trade nearly 24/5, but liquidity varies dramatically
    allow_eth_execution: bool = True       # VIX futures trade 24/5
    eth_slippage_multiplier: float = 2.0   # Higher slippage in ETH
    eth_start_time: str = "18:00:00"       # ETH starts 6PM ET (previous day)
    eth_end_time: str = "09:30:00"         # ETH ends at RTH open
    eth_liquidity_threshold: float = 0.3   # Min liquidity ratio vs RTH (skip if lower)

def is_eth_session(current_time: datetime, config: OrderConfig) -> bool:
    """
    Check if current time is in Extended Trading Hours (ETH).

    VIX futures trade nearly 24/5 on CFE:
    - RTH: 9:30 AM - 4:15 PM ET (primary liquidity)
    - ETH: 4:30 PM - 9:30 AM ET next day (thin liquidity)
    """
    from datetime import time
    t = current_time.time()
    eth_start = time(18, 0)  # 6 PM ET
    eth_end = time(9, 30)    # 9:30 AM ET

    # ETH spans midnight
    if t >= eth_start or t < eth_end:
        return True
    return False

def check_eth_liquidity(
    market_data: dict,
    rth_volume_avg: float,
    config: OrderConfig
) -> tuple[bool, str]:
    """
    Check if ETH liquidity is sufficient for execution.

    Returns:
        (can_execute, reason)
    """
    current_volume = market_data.get('volume', 0)
    liquidity_ratio = current_volume / rth_volume_avg if rth_volume_avg > 0 else 0

    if liquidity_ratio < config.eth_liquidity_threshold:
        return False, f"ETH liquidity {liquidity_ratio:.1%} < {config.eth_liquidity_threshold:.0%} threshold"

    return True, "ETH liquidity acceptable"

def create_futures_order(
    symbol: str,
    side: str,
    quantity: int,
    market_data: dict,
    config: OrderConfig,
    current_time: datetime = None
) -> dict:
    """
    Create futures order with appropriate limit price.

    Adjusts slippage buffer for ETH sessions.
    """
    mid = (market_data['bid'] + market_data['ask']) / 2

    # Check if ETH session (wider spreads, less liquidity)
    if current_time and is_eth_session(current_time, config):
        slippage_multiplier = config.eth_slippage_multiplier
    else:
        slippage_multiplier = 1.0

    # Adjust limit offset for session
    adjusted_offset = config.futures_limit_offset * slippage_multiplier

    if side == 'SELL':
        limit_price = mid - adjusted_offset
    else:
        limit_price = mid + adjusted_offset

    # Round to tick size (0.05 for VX)
    limit_price = round(limit_price / 0.05) * 0.05

    return {
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'order_type': config.futures_order_type,
        'limit_price': limit_price,
        'tif': 'DAY',
        'timeout': config.timeout_seconds,
        'session': 'ETH' if current_time and is_eth_session(current_time, config) else 'RTH'
    }
```

### 4.3 Roll Procedure

```python
def calculate_roll_date(front_month_expiry: datetime) -> datetime:
    """
    Calculate roll date: 5 business days before expiration.

    VIX futures expire on Wednesday, so roll typically on prior Wednesday.
    """
    nyse = xcals.get_calendar('NYSE')

    # Find 5 business days before expiry
    sessions_before = nyse.sessions_in_range(
        front_month_expiry - timedelta(days=14),
        front_month_expiry - timedelta(days=1)
    )

    if len(sessions_before) < 5:
        raise ValueError("Cannot calculate roll date")

    return sessions_before[-5].to_pydatetime()

def execute_roll(
    current_position: dict,
    next_month_symbol: str,
    market_data: dict
) -> list[dict]:
    """
    Execute roll from front month to next month.

    Returns list of orders: [close_front, open_next, roll_hedge]
    """
    orders = []

    # Close front month (buy back short)
    orders.append({
        'symbol': current_position['futures_symbol'],
        'side': 'BUY',
        'quantity': current_position['futures_quantity'],
        'order_type': 'LIMIT',
        'intent': 'ROLL_CLOSE'
    })

    # Open next month (new short)
    orders.append({
        'symbol': next_month_symbol,
        'side': 'SELL',
        'quantity': current_position['futures_quantity'],
        'order_type': 'LIMIT',
        'intent': 'ROLL_OPEN'
    })

    # Close expiring hedge (let expire or sell if value)
    if current_position.get('hedge_symbol'):
        orders.append({
            'symbol': current_position['hedge_symbol'],
            'side': 'SELL',
            'quantity': current_position['hedge_quantity'],
            'order_type': 'LIMIT',
            'intent': 'ROLL_HEDGE_CLOSE'
        })

    # Open new hedge (buy calls for next month)
    # Handled separately via construct_tail_hedge()

    return orders
```

---

## 5. Risk Management

### 5.1 Stop-Loss Implementation

```python
@dataclass
class StopLossConfig:
    """Stop-loss order configuration."""
    trigger_pct: float = -0.15           # -15% drawdown trigger
    order_type: str = 'STOP_LIMIT'
    limit_offset: float = 2.0            # Limit = stop + offset (slippage buffer)

def place_stop_loss_order(
    position: dict,
    entry_price: float,
    config: StopLossConfig
) -> dict:
    """
    Place stop-loss order immediately after entry.

    Stop-loss lives in market 24/7 for automated protection.
    """
    # Calculate stop trigger price
    # For short position: stop triggers when price RISES
    stop_price = entry_price * (1 - config.trigger_pct)  # e.g., 16.2 * 1.15 = 18.63

    # Limit price with slippage buffer
    limit_price = stop_price + config.limit_offset

    return {
        'symbol': position['futures_symbol'],
        'side': 'BUY',  # Buy to close short
        'quantity': position['futures_quantity'],
        'order_type': config.order_type,
        'stop_price': round(stop_price, 2),
        'limit_price': round(limit_price, 2),
        'tif': 'GTC',  # Good-til-canceled
        'intent': 'STOP_LOSS'
    }
```

### 5.2 Cool-Down State Machine

```python
from enum import Enum
from datetime import datetime, timedelta

class CoolDownState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"

@dataclass
class CoolDownManager:
    """
    Manages cool-down period after stop-loss triggers.

    Rule: After stop-loss, no re-entry for 5 trading days OR rest of month.
    """
    state: CoolDownState = CoolDownState.INACTIVE
    triggered_date: datetime = None
    min_cool_down_days: int = 5

    def trigger_cool_down(self, trigger_date: datetime) -> None:
        """Activate cool-down after stop-loss."""
        self.state = CoolDownState.ACTIVE
        self.triggered_date = trigger_date

    def check_cool_down(self, current_date: datetime) -> tuple[bool, str]:
        """
        Check if cool-down is still active.

        Returns:
            (is_active, reason)
        """
        if self.state == CoolDownState.INACTIVE:
            return False, "Cool-down inactive"

        # Check 1: Same month as trigger
        if (current_date.year == self.triggered_date.year and
            current_date.month == self.triggered_date.month):
            return True, f"Same month as stop-loss ({self.triggered_date.date()})"

        # Check 2: Minimum days elapsed
        nyse = xcals.get_calendar('NYSE')
        sessions = nyse.sessions_in_range(self.triggered_date, current_date)

        if len(sessions) < self.min_cool_down_days:
            return True, f"Only {len(sessions)} days since stop (need {self.min_cool_down_days})"

        # Cool-down expired
        self.state = CoolDownState.INACTIVE
        self.triggered_date = None
        return False, "Cool-down expired"
```

### 5.3 Regime-Adaptive Thresholds

```python
import numpy as np
from collections import deque

class AdaptiveThresholds:
    """
    Calculate regime-adaptive thresholds using rolling percentiles.

    Uses 6-month (126 trading day) lookback window.
    """
    def __init__(self, lookback_days: int = 126):
        self.lookback = lookback_days
        self.vvix_history = deque(maxlen=lookback_days)
        self.vix1d_ratio_history = deque(maxlen=lookback_days)

    def update(self, vvix: float, vix1d_ratio: float) -> None:
        """Add daily observation."""
        self.vvix_history.append(vvix)
        self.vix1d_ratio_history.append(vix1d_ratio)

    def get_vvix_threshold(self, percentile: float = 90) -> float:
        """Get VVIX threshold at given percentile."""
        if len(self.vvix_history) < 20:  # Minimum history
            return 110.0  # Fallback to hard threshold
        return np.percentile(list(self.vvix_history), percentile)

    def get_vix1d_ratio_threshold(self, percentile: float = 95) -> float:
        """Get VIX1D/VIX ratio threshold at given percentile."""
        if len(self.vix1d_ratio_history) < 20:
            return 1.2  # Fallback to hard threshold
        return max(
            np.percentile(list(self.vix1d_ratio_history), percentile),
            1.2  # Never go below hard floor
        )
```

### 5.4 Dynamic Hedge Budgeting

```python
def calculate_dynamic_hedge_budget(
    base_budget_pct: float,
    vvix: float,
    vvix_50d_ma: float,
    term_slope: float
) -> float:
    """
    Dynamically adjust hedge budget based on market conditions.

    When volatility of volatility is elevated, spend more on tail protection.

    Args:
        base_budget_pct: Baseline hedge budget (e.g., 0.15 = 15%)
        vvix: Current VVIX level
        vvix_50d_ma: 50-day moving average of VVIX
        term_slope: F6 - F1 (negative = backwardation = danger)

    Returns:
        Adjusted hedge budget percentage
    """
    budget = base_budget_pct

    # Increase hedge budget when VVIX elevated
    if vvix > vvix_50d_ma * 1.1:  # VVIX 10% above average
        budget *= 1.25  # Spend 25% more on hedges

    if vvix > vvix_50d_ma * 1.2:  # VVIX 20% above average
        budget *= 1.50  # Spend 50% more on hedges

    # Increase when term structure inverted (backwardation)
    if term_slope < -1.0:  # Backwardation > 1 point
        budget *= 1.25

    # Cap at reasonable maximum
    return min(budget, 0.30)  # Never spend > 30% on hedges
```

### 5.5 Collateral Yield Tracking

```python
@dataclass
class CollateralYieldConfig:
    """Configuration for cash collateral yield accrual."""
    # VRP typically holds significant cash as margin buffer
    # This cash can earn risk-free yield (T-bills or money market)

    yield_instrument: str = "T-BILL"       # T-bills or money market
    yield_rate_source: str = "FRED:DTB3"   # 3-month T-bill rate
    reinvest_frequency: str = "daily"      # Accrue daily
    min_cash_balance_pct: float = 0.40     # Keep 40% in cash minimum

def calculate_collateral_yield(
    cash_balance: float,
    tbill_rate: float,
    days: int = 1
) -> float:
    """
    Calculate yield earned on cash collateral.

    Args:
        cash_balance: Cash not used as margin
        tbill_rate: Annualized T-bill rate (e.g., 0.045 = 4.5%)
        days: Number of days to accrue

    Returns:
        Yield earned in dollars
    """
    daily_rate = tbill_rate / 365
    return cash_balance * daily_rate * days
```

### 5.6 Margin Expansion Check

```python
def check_margin_expansion_risk(
    current_margin: float,
    sleeve_nav: float,
    vix_spot: float,
    position_notional: float
) -> tuple[bool, str]:
    """
    Check for broker margin expansion risk.

    During high volatility, brokers often increase margin requirements
    to 100% of notional or higher. This can force liquidation.

    Args:
        current_margin: Current margin requirement
        sleeve_nav: Total sleeve NAV
        vix_spot: Current VIX level
        position_notional: Total notional exposure

    Returns:
        (is_at_risk, reason)
    """
    margin_usage = current_margin / sleeve_nav

    # Normal regime margin check
    if margin_usage > 0.50:
        return True, f"Margin usage {margin_usage:.1%} > 50% - reduce position"

    # High VIX regime: assume margin could go to 100% notional
    if vix_spot > 25:
        # Simulate 100% margin scenario
        worst_case_margin = position_notional
        worst_case_usage = worst_case_margin / sleeve_nav
        if worst_case_usage > 0.70:
            return True, f"High VIX ({vix_spot:.1f}): potential margin expansion to {worst_case_usage:.1%}"

    # Extreme VIX: assume margin could exceed notional
    if vix_spot > 40:
        # Some brokers charge 150% margin in crisis
        extreme_margin = position_notional * 1.5
        extreme_usage = extreme_margin / sleeve_nav
        if extreme_usage > 0.80:
            return True, f"Extreme VIX ({vix_spot:.1f}): margin could expand to {extreme_usage:.1%}"

    return False, "Margin within acceptable bounds"
```

### 5.7 Daily Risk Monitoring

```python
@dataclass
class DailyRiskCheck:
    """Daily risk monitoring output."""
    timestamp: datetime
    position_pnl_pct: float
    vix_spot: float
    margin_usage_pct: float
    margin_expansion_risk: bool
    collateral_yield_accrued: float
    vvix: float
    vvix_threshold: float
    stop_loss_active: bool
    action_required: str
    details: str

def daily_risk_check(
    position: dict,
    sleeve_nav: float,
    market_data: dict,
    thresholds: AdaptiveThresholds,
    tbill_rate: float = 0.045
) -> DailyRiskCheck:
    """
    Perform daily risk check on VRP position.

    Called every trading day at market close.
    Includes margin expansion checks and collateral yield tracking.
    """
    # Calculate current P&L
    entry_price = position['entry_price']
    current_price = market_data['front_month_price']
    # Short position: profit when price falls
    pnl_pct = (entry_price - current_price) / entry_price

    # Calculate Margin Usage (approximate)
    margin_usage_pct = position['margin_required'] / sleeve_nav

    # Check margin expansion risk
    margin_at_risk, margin_reason = check_margin_expansion_risk(
        position['margin_required'],
        sleeve_nav,
        market_data['vix_spot'],
        position['notional']
    )

    # Calculate collateral yield (cash not used as margin)
    cash_balance = sleeve_nav - position['margin_required']
    collateral_yield = calculate_collateral_yield(cash_balance, tbill_rate, days=1)

    # Get adaptive thresholds
    vvix_threshold = thresholds.get_vvix_threshold(90)

    # Determine action
    action = "HOLD"
    details = []

    # Check VIX level
    if market_data['vix_spot'] >= 40:
        action = "EMERGENCY_FLATTEN"
        details.append(f"VIX {market_data['vix_spot']:.1f} >= 40")
    elif market_data['vix_spot'] >= 30:
        action = "REDUCE_50PCT"
        details.append(f"VIX {market_data['vix_spot']:.1f} >= 30")

    # Check Margin Expansion Risk (NEW)
    if margin_at_risk:
        if action == "HOLD":
            action = "REDUCE_50PCT"
        details.append(margin_reason)

    # Check VVIX
    if market_data['vvix'] > vvix_threshold:
        if action == "HOLD":
            action = "REDUCE_50PCT"
        details.append(f"VVIX {market_data['vvix']:.1f} > {vvix_threshold:.1f}")

    # Check P&L (stop-loss should auto-trigger, but verify)
    if pnl_pct <= -0.15:
        action = "STOP_LOSS_TRIGGERED"
        details.append(f"P&L {pnl_pct:.1%} <= -15%")

    return DailyRiskCheck(
        timestamp=datetime.now(),
        position_pnl_pct=pnl_pct,
        vix_spot=market_data['vix_spot'],
        margin_usage_pct=margin_usage_pct,
        margin_expansion_risk=margin_at_risk,
        collateral_yield_accrued=collateral_yield,
        vvix=market_data['vvix'],
        vvix_threshold=vvix_threshold,
        stop_loss_active=position.get('stop_loss_order_id') is not None,
        action_required=action,
        details="; ".join(details) if details else "All clear"
    )
```

---

## 6. Backtesting Framework

### 6.1 Backtest Configuration

```python
@dataclass
class BacktestConfig:
    """Configuration for VRP backtest."""
    # Date range
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"

    # Initial capital
    initial_nav: float = 100_000.0
    annual_risk_free_rate: float = 0.045  # 4.5% yield on cash collateral

    # Transaction costs
    futures_commission: float = 2.50      # Per contract per side
    futures_slippage_ticks: int = 1       # 1 tick = 0.05 points = $50
    options_commission: float = 1.50      # Per contract per side
    options_slippage_pct: float = 0.05    # 5% of premium

    # Stress test multipliers
    cost_multiplier: float = 1.0          # 2.0 for stress test

    # Position config
    position_config: VRPPositionConfig = None
    hedge_config: TailHedgeConfig = None

    def __post_init__(self):
        if self.position_config is None:
            self.position_config = VRPPositionConfig()
        if self.hedge_config is None:
            self.hedge_config = TailHedgeConfig()
```

### 6.2 Continuous Futures Construction

```python
def build_continuous_vix_futures(
    raw_data: pd.DataFrame,
    roll_days_before_expiry: int = 5
) -> pd.DataFrame:
    """
    Build continuous VIX futures series with proper roll handling.

    Args:
        raw_data: DataFrame with columns [date, contract, open, high, low, close, volume]
        roll_days_before_expiry: Days before expiry to roll

    Returns:
        DataFrame with continuous front-month series
    """
    # Get expiration dates for each contract
    expirations = get_vix_expiration_dates(raw_data['contract'].unique())

    continuous = []
    current_contract = None
    roll_adjustment = 0.0

    for date in sorted(raw_data['date'].unique()):
        day_data = raw_data[raw_data['date'] == date]

        # Determine which contract is front-month
        front_month = get_front_month_contract(date, expirations, roll_days_before_expiry)

        if front_month != current_contract:
            # Roll day - calculate adjustment
            if current_contract is not None:
                old_price = day_data[day_data['contract'] == current_contract]['close'].iloc[0]
                new_price = day_data[day_data['contract'] == front_month]['close'].iloc[0]
                roll_adjustment += old_price - new_price
            current_contract = front_month

        # Get front month data
        fm_data = day_data[day_data['contract'] == front_month].iloc[0]

        continuous.append({
            'date': date,
            'contract': front_month,
            'close': fm_data['close'],
            'close_adjusted': fm_data['close'] + roll_adjustment,
            'volume': fm_data['volume'],
            'days_to_expiry': (expirations[front_month] - date).days
        })

    return pd.DataFrame(continuous)
```

### 6.3 Backtest Engine

```python
class VRPBacktester:
    """
    Walk-forward backtester for VRP strategy.
    """
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cool_down = CoolDownManager()
        self.thresholds = AdaptiveThresholds()

    def run(
        self,
        vix_data: pd.DataFrame,
        futures_data: pd.DataFrame,
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run backtest and return daily returns.

        Returns:
            DataFrame with columns [date, nav, position, pnl, signal, action]
        """
        results = []
        nav = self.config.initial_nav
        position = None

        for date in self._get_trading_dates(futures_data):
            # Get market data for date
            market_data = self._get_market_data(date, vix_data, futures_data)

            # Update adaptive thresholds
            self.thresholds.update(
                market_data['vvix'],
                market_data['vix1d'] / market_data['vix_spot']
            )

            # Accrue interest on cash (NAV - Margin)
            # Simplified: Accrue on full NAV if fully collateralized by T-bills
            nav += nav * (self.config.annual_risk_free_rate / 252)

            # Daily P&L if in position
            daily_pnl = 0.0
            if position:
                daily_pnl = self._calculate_daily_pnl(position, market_data)

                # Check exit signals
                should_exit, reason, reduction = check_exit_signals(
                    # Note: Pass updated margin/NAV info here if needed
                    market_data['vix_spot'],
                    position['pnl_pct'],
                    market_data['vvix'],
                    self.thresholds.get_vvix_threshold(90)
                )

                if should_exit and reduction == 1.0:
                    # Full exit
                    nav += self._close_position(position, market_data)
                    if 'stop' in reason.lower():
                        self.cool_down.trigger_cool_down(date)
                    position = None

            # Check for entry (3rd trading day of month)
            if position is None and self._is_execution_day(date):
                # Check cool-down
                cool_down_active, _ = self.cool_down.check_cool_down(date)

                # Check entry filters
                can_enter, reason = check_entry_filters(
                    market_data['adjusted_contango'],
                    market_data['vix_spot'],
                    market_data['vix_50d_ma'],
                    market_data['vix1d'],
                    market_data['vvix'],
                    self.thresholds.get_vvix_threshold(90),
                    self.thresholds.get_vix1d_ratio_threshold(95),
                    cool_down_active
                )

                if can_enter:
                    position = self._open_position(nav, market_data, options_data)

            nav += daily_pnl
            results.append({
                'date': date,
                'nav': nav,
                'position': position is not None,
                'pnl': daily_pnl,
                'vix': market_data['vix_spot'],
                'contango': market_data['adjusted_contango']
            })

        return pd.DataFrame(results)
```

---

## 7. Kill-Test Criteria

### 7.1 Primary Gates (Must Pass ALL)

| Gate | Metric | Threshold | Formula |
|------|--------|-----------|---------|
| G1 | Sharpe Ratio | ≥ 0.50 | `mean(returns) / std(returns) * sqrt(252)` |
| G2 | Net P&L | > $0 | `final_nav - initial_nav` |
| G3 | Max Drawdown | ≥ -30% | `min(cumulative_returns) / peak` |

### 7.2 Stress Gates (Must Pass ALL)

| Gate | Metric | Threshold | Stress Factor |
|------|--------|-----------|---------------|
| S1 | Sharpe (2× costs) | ≥ 0.30 | `cost_multiplier = 2.0` |
| S2 | Net P&L (2× costs) | > $0 | `cost_multiplier = 2.0` |

### 7.3 Fold Validation

```python
def run_kill_test(
    backtester: VRPBacktester,
    data: dict
) -> dict:
    """
    Run complete kill-test with expanding window OOS validation.

    Folds:
        - Fold 1: Train 2018-2022, Test 2023
        - Fold 2: Train 2018-2023, Test 2024
        - Fold 3: Train 2018-2024, Test 2025 (partial)
    """
    folds = [
        {'train_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'},
        {'train_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31'},
        {'train_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31'},
    ]

    results = {
        'baseline': [],
        'stress': [],
        'fold_pass': [],
        'overall_pass': False
    }

    for i, fold in enumerate(folds):
        # Run baseline backtest on OOS period
        baseline = backtester.run(data, fold['test_start'], fold['test_end'])
        baseline_metrics = calculate_metrics(baseline)

        # Run stress backtest (2× costs)
        stress_config = BacktestConfig(cost_multiplier=2.0)
        stress_backtester = VRPBacktester(stress_config)
        stress = stress_backtester.run(data, fold['test_start'], fold['test_end'])
        stress_metrics = calculate_metrics(stress)

        # Check gates
        baseline_pass = (
            baseline_metrics['sharpe'] >= 0.50 and
            baseline_metrics['net_pnl'] > 0 and
            baseline_metrics['max_dd'] >= -0.30
        )

        stress_pass = (
            stress_metrics['sharpe'] >= 0.30 and
            stress_metrics['net_pnl'] > 0
        )

        fold_pass = baseline_pass and stress_pass

        results['baseline'].append(baseline_metrics)
        results['stress'].append(stress_metrics)
        results['fold_pass'].append(fold_pass)

    # Overall: 2/3 folds must pass
    results['overall_pass'] = sum(results['fold_pass']) >= 2

    return results
```

---

## 8. Data Schemas

### 8.1 Market Data Schema

```python
@dataclass
class VRPMarketData:
    """Daily market data for VRP strategy."""
    date: datetime

    # VIX indices
    vix_spot: float              # VIX spot index
    vix_50d_ma: float            # 50-day moving average of VIX
    vix1d: float                 # 1-day VIX index
    vvix: float                  # Volatility of VIX

    # Futures
    front_month_symbol: str      # e.g., "VXF26" (Feb 2026)
    front_month_price: float     # Settlement price
    front_month_bid: float
    front_month_ask: float
    front_month_expiry: datetime
    days_to_expiry: int

    back_month_symbol: str       # e.g., "VXG26" (Mar 2026)
    back_month_price: float

    # Derived
    raw_contango: float          # front - spot
    adjusted_contango: float     # raw - interest component

    # Rates
    risk_free_rate: float        # Fed funds or T-bill rate
```

### 8.2 Position Schema

```python
@dataclass
class VRPPosition:
    """Active VRP position state."""
    # Futures leg
    futures_symbol: str
    futures_quantity: int
    futures_entry_price: float
    futures_entry_date: datetime

    # Hedge leg (optional)
    hedge_symbol: str = None
    hedge_strike: float = None
    hedge_quantity: int = 0
    hedge_entry_price: float = 0.0

    # Risk orders
    stop_loss_order_id: str = None
    stop_loss_price: float = None

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def pnl_pct(self) -> float:
        """Current P&L as percentage of notional."""
        notional = self.futures_quantity * self.futures_entry_price * 1000
        return (self.realized_pnl + self.unrealized_pnl) / notional
```

### 8.3 Trade Log Schema

```python
@dataclass
class VRPTradeLog:
    """Trade execution log entry."""
    timestamp: datetime
    trade_id: str

    # Order details
    symbol: str
    side: str
    quantity: int
    order_type: str
    limit_price: float = None
    stop_price: float = None

    # Execution
    fill_price: float = None
    fill_quantity: int = 0
    commission: float = 0.0
    slippage: float = 0.0

    # Context
    intent: str  # ENTRY, EXIT, ROLL, STOP_LOSS, HEDGE
    signal_reason: str = None
```

---

## 9. File Structure

```
dsp100k/
├── config/
│   └── dsp100k_vrp.yaml           # VRP configuration
├── src/
│   └── dsp/
│       ├── sleeves/
│       │   └── sleeve_vrp.py      # Main VRP sleeve implementation
│       ├── signals/
│       │   └── vrp_signal.py      # Signal generation
│       ├── risk/
│       │   ├── vrp_stop_loss.py   # Stop-loss management
│       │   ├── cool_down.py       # Cool-down state machine
│       │   └── adaptive_thresholds.py
│       └── backtest/
│           └── vrp_backtest.py    # Backtester
├── data/
│   └── vrp/
│       ├── vix_futures/           # VIX futures data
│       ├── vix_options/           # VIX options data (for hedges)
│       └── indices/               # VIX, VIX1D, VVIX
└── tests/
    └── test_vrp/
        ├── test_signal.py
        ├── test_position_sizing.py
        ├── test_stop_loss.py
        └── test_backtest.py
```

---

## 10. Configuration File (YAML)

```yaml
# dsp100k_vrp.yaml
sleeve:
  name: vrp
  enabled: false  # Enable after kill-test passes

universe:
  futures:
    - symbol: VX
      exchange: CFE
      multiplier: 1000
      tick_size: 0.05
  options:
    - symbol: VIX
      exchange: CBOE
      multiplier: 100

signal:
  min_contango: 0.5
  vix_ma_period: 50
  vix1d_hard_limit: 1.2
  high_vol_regime_threshold: 25.0

position:
  max_nav_pct: 0.10
  max_margin_pct: 0.30
  margin_per_contract: 5000

hedge:
  strike_offset: 25
  min_strike: 35
  max_cost_pct: 0.15
  contract_ratio: 1.0

risk:
  stop_loss_pct: -0.15
  vix_reduce_level: 30
  vix_flatten_level: 40
  cool_down_days: 5

thresholds:
  lookback_days: 126
  vvix_percentile: 90
  vix1d_ratio_percentile: 95

execution:
  day_of_month: 3  # 3rd trading day
  time: "09:35:00"
  futures_limit_offset: 0.05
  options_limit_offset: 0.10
  roll_days_before_expiry: 5
  # ETH (Extended Trading Hours) settings
  allow_eth_execution: true
  eth_slippage_multiplier: 2.0
  eth_start_time: "18:00:00"
  eth_end_time: "09:30:00"
  eth_liquidity_threshold: 0.30

dynamic_hedge:
  base_budget_pct: 0.15
  vvix_threshold_10pct: 1.1   # VVIX > 110% of 50d MA
  vvix_threshold_20pct: 1.2   # VVIX > 120% of 50d MA
  backwardation_threshold: -1.0
  max_budget_pct: 0.30

collateral:
  yield_instrument: "T-BILL"
  yield_rate_source: "FRED:DTB3"
  reinvest_frequency: "daily"
  min_cash_balance_pct: 0.40

margin_expansion:
  normal_threshold_pct: 0.50
  high_vix_level: 25
  high_vix_max_usage: 0.70
  extreme_vix_level: 40
  extreme_vix_multiplier: 1.5
  extreme_vix_max_usage: 0.80

backtest:
  start_date: "2018-01-01"
  end_date: "2025-12-31"
  initial_nav: 100000
  futures_commission: 2.50
  futures_slippage_ticks: 1
  options_commission: 1.50
  options_slippage_pct: 0.05
```

---

## 11. Implementation Checklist

### Phase 1: Data & Infrastructure (Day 1)
- [ ] Acquire VIX futures historical data from Databento
- [ ] Acquire VIX, VIX1D, VVIX index data from CBOE
- [ ] Acquire VIX options data for hedge pricing
- [ ] Build continuous futures constructor
- [ ] Implement data validation checks

### Phase 2: Signal & Position Logic (Day 1-2)
- [ ] Implement `calculate_adjusted_contango()`
- [ ] Implement `check_entry_filters()`
- [ ] Implement `check_exit_signals()`
- [ ] Implement `calculate_position_size()`
- [ ] Implement `construct_tail_hedge()`
- [ ] Unit tests for all signal functions

### Phase 3: Risk Management (Day 2)
- [ ] Implement `StopLossConfig` and order placement
- [ ] Implement `CoolDownManager` state machine
- [ ] Implement `AdaptiveThresholds` calculator
- [ ] Implement `calculate_dynamic_hedge_budget()`
- [ ] Implement `calculate_collateral_yield()`
- [ ] Implement `check_margin_expansion_risk()`
- [ ] Implement `daily_risk_check()` with full integration
- [ ] Implement ETH session detection and liquidity checks
- [ ] Unit tests for risk components

### Phase 4: Backtesting (Day 2-3)
- [ ] Implement `VRPBacktester` class
- [ ] Implement transaction cost modeling
- [ ] Implement roll simulation
- [ ] Run baseline backtest
- [ ] Run stress backtest (2× costs)
- [ ] Generate kill-test report

### Phase 5: Integration (Day 3-4)
- [ ] Create `sleeve_vrp.py` sleeve class
- [ ] Integrate with DSP CLI
- [ ] Add to `dsp100k_vrp.yaml` config
- [ ] Dry-run test with --strict mode
- [ ] Documentation and handoff

---

## 12. Appendix: VIX Futures Expiration Calendar

VIX futures expire on the Wednesday 30 days before the 3rd Friday of the following month.

```python
def get_vix_expiration(year: int, month: int) -> datetime:
    """
    Calculate VIX futures expiration date.

    Expiration is Wednesday 30 days before 3rd Friday of NEXT month.
    """
    from dateutil.relativedelta import relativedelta, FR, WE

    # Get 3rd Friday of next month
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    third_friday = next_month + relativedelta(weekday=FR(3))

    # Go back 30 days
    target = third_friday - timedelta(days=30)

    # Find the Wednesday on or before target
    while target.weekday() != 2:  # Wednesday = 2
        target -= timedelta(days=1)

    return target
```

---

**Document Version**: 1.0
**Status**: Pre-Registered (Frozen)
**Next Step**: Data acquisition and Phase 1 implementation
