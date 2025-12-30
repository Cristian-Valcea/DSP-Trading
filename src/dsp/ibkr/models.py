"""
Data models for IBKR integration.

These models represent the core data structures used in communication
with Interactive Brokers TWS/Gateway.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING_SUBMIT = "PendingSubmit"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    INACTIVE = "Inactive"
    API_PENDING = "ApiPending"
    API_CANCELLED = "ApiCancelled"
    ERROR = "Error"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    GTD = "GTD"  # Good Till Date
    OPG = "OPG"  # Market on Open
    FOK = "FOK"  # Fill or Kill


@dataclass
class AccountSummary:
    """
    IBKR account summary.

    From reqAccountSummary: NLV, AvailableFunds, BuyingPower, etc.
    """
    nlv: float  # Net Liquidation Value
    available_funds: float
    buying_power: float
    margin_used: float
    cash: float
    currency: str = "USD"
    account_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def margin_usage(self) -> float:
        """Calculate margin usage as fraction of NLV."""
        if self.nlv <= 0:
            return 0.0
        return self.margin_used / self.nlv


@dataclass
class Position:
    """
    Current position in a security.

    From reqPositions or reqPositionsMulti.
    """
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    account: str = ""
    sleeve: str = ""  # "A", "B", "C", or "HEDGE"
    sec_type: str = "STK"  # STK, OPT, FUT, etc.
    currency: str = "USD"

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class Order:
    """
    Order to be submitted to IBKR.

    All orders are submitted via placeOrder.
    """
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    order_type: str = "LMT"  # "MKT", "LMT", etc.
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: str = "DAY"  # Time in force
    sleeve: str = ""  # For tracking which sleeve generated the order
    reason: str = ""  # Audit trail: why this order was placed
    outside_rth: bool = False  # Allow trading outside regular hours

    # Order identifiers (set after submission)
    order_id: Optional[int] = None
    client_order_id: Optional[str] = None

    def __post_init__(self):
        """Validate order parameters."""
        if self.order_type == "LMT" and self.limit_price is None:
            raise ValueError("Limit orders require a limit_price")
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {self.side}")


@dataclass
class OrderStatusInfo:
    """
    Status information for a submitted order.
    """
    order_id: int
    status: str
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    last_fill_price: float = 0.0
    last_fill_time: Optional[datetime] = None
    parent_id: int = 0
    client_id: int = 0
    why_held: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == "Filled"

    @property
    def is_active(self) -> bool:
        return self.status in ("Submitted", "PreSubmitted", "PendingSubmit")

    @property
    def is_cancelled(self) -> bool:
        return self.status in ("Cancelled", "ApiCancelled")


@dataclass
class Fill:
    """
    Execution fill from IBKR.

    From execDetails callback.
    """
    order_id: int
    exec_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    account: str = ""

    @property
    def notional(self) -> float:
        """Calculate notional value of fill."""
        return self.quantity * self.price


@dataclass
class MarginImpact:
    """
    Margin impact from a what-if order.

    From whatIfOrder to estimate margin before submission.
    """
    init_margin_change: float
    maint_margin_change: float
    equity_with_loan: float
    post_trade_margin: float  # As fraction of NLV
    commission: float
    max_buy_power: float = 0.0

    @property
    def would_breach_cap(self, cap: float = 0.60) -> bool:
        """Check if post-trade margin would exceed cap."""
        return self.post_trade_margin > cap


@dataclass
class Quote:
    """
    Market quote for a security.

    From reqMktData snapshot.
    """
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.mid > 0:
            return (self.spread / self.mid) * 10000
        return 0.0


@dataclass
class DailyBar:
    """
    Daily OHLCV bar.

    From reqHistoricalData.
    """
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float  # For signals (split/dividend adjusted)
    source: str = "IBKR"  # "IBKR" | "CACHE"
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3


@dataclass
class OptionContract:
    """
    Option contract details.
    """
    symbol: str  # Underlying symbol
    strike: float
    expiry: date
    right: str  # "P" for put, "C" for call
    multiplier: int = 100
    exchange: str = "SMART"
    currency: str = "USD"

    # Greeks (from model greeks)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_vol: Optional[float] = None

    # Pricing
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    underlying_price: Optional[float] = None

    @property
    def dte(self) -> int:
        """Days to expiration."""
        return (self.expiry - date.today()).days

    @property
    def mid(self) -> Optional[float]:
        """Mid price if available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def is_put(self) -> bool:
        return self.right == "P"

    @property
    def is_call(self) -> bool:
        return self.right == "C"


@dataclass
class OptionChain:
    """
    Option chain for an underlying.
    """
    underlying: str
    expirations: List[date]
    strikes: List[float]
    contracts: Dict[str, OptionContract] = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=datetime.now)

    def get_contract(self, strike: float, expiry: date, right: str) -> Optional[OptionContract]:
        """Get specific contract from chain."""
        key = f"{self.underlying}_{expiry}_{strike}_{right}"
        return self.contracts.get(key)

    def filter_by_dte(self, min_dte: int, max_dte: int) -> List[date]:
        """Filter expirations by DTE range."""
        today = date.today()
        return [
            exp for exp in self.expirations
            if min_dte <= (exp - today).days <= max_dte
        ]


@dataclass
class PutSpread:
    """
    Put debit spread for Sleeve C hedging.
    """
    long_put: OptionContract
    short_put: OptionContract
    entry_date: date
    entry_cost: float  # Premium paid
    current_value: float = 0.0

    @property
    def dte(self) -> int:
        """Days to expiration of long leg."""
        return self.long_put.dte

    @property
    def max_profit(self) -> float:
        """Maximum profit (at long strike - entry cost)."""
        return (self.long_put.strike - self.short_put.strike) * self.long_put.multiplier - self.entry_cost

    @property
    def max_loss(self) -> float:
        """Maximum loss (entry cost)."""
        return self.entry_cost

    @property
    def current_delta(self) -> Optional[float]:
        """Net delta of spread."""
        if self.long_put.delta is not None and self.short_put.delta is not None:
            return self.long_put.delta - self.short_put.delta
        return None

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L."""
        return self.current_value - self.entry_cost
