"""
IBKR client wrapper for DSP-100K.

Provides async interface to Interactive Brokers TWS/Gateway with:
- Connection management with auto-reconnection
- Rate limiting to avoid API throttling
- Timeout protection on all operations
- Health monitoring
"""

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Contract, Stock, Option, Order as IBOrder
from ib_insync import MarketOrder, LimitOrder, util

from .models import (
    AccountSummary,
    DailyBar,
    Fill,
    MarginImpact,
    OptionChain,
    OptionContract,
    Order,
    OrderStatusInfo,
    Position,
    Quote,
)

logger = logging.getLogger(__name__)

# US Eastern timezone for market hours
NY_TZ = ZoneInfo("America/New_York")


class RateLimiter:
    """
    Simple rate limiter to avoid IBKR API throttling.

    IBKR recommends no more than 50 messages per second.
    We use a more conservative limit.
    """

    def __init__(self, max_calls: int = 30, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to stay within rate limit."""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old calls outside the window
            self._calls = [t for t in self._calls if now - t < self.period]

            if len(self._calls) >= self.max_calls:
                # Wait until oldest call expires
                sleep_time = self.period - (now - self._calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)

            self._calls.append(now)


class ConnectionMonitor:
    """
    Monitor connection health and trigger reconnection.
    """

    def __init__(self, ib: IB, reconnect_callback):
        self.ib = ib
        self.reconnect_callback = reconnect_callback
        self.last_heartbeat = datetime.now()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start connection monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        """Stop connection monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()

    def heartbeat(self) -> None:
        """Record a successful API call."""
        self.last_heartbeat = datetime.now()

    async def _monitor_loop(self) -> None:
        """Monitor connection and reconnect if needed."""
        while self._monitoring:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check if connection is stale
                if not self.ib.isConnected():
                    logger.warning("Connection lost - attempting reconnect")
                    await self.reconnect_callback()
                elif (datetime.now() - self.last_heartbeat).total_seconds() > 60:
                    logger.warning("No heartbeat for 60s - checking connection")
                    # Trigger a lightweight API call to verify connection
                    try:
                        self.ib.reqCurrentTime()
                        self.heartbeat()
                    except Exception:
                        logger.warning("Heartbeat check failed - reconnecting")
                        await self.reconnect_callback()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")


class IBKRClient:
    """
    Async wrapper for IBKR TWS/Gateway API.

    Provides all data and trading operations needed for DSP-100K
    with built-in rate limiting, timeout protection, and reconnection.

    Usage:
        async with IBKRClient(host, port, client_id) as client:
            summary = await client.get_account_summary()
            positions = await client.get_positions()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize IBKR client.

        Args:
            host: TWS/Gateway host
            port: TWS/Gateway port (7497 for TWS, 4001 for Gateway)
            client_id: Unique client ID for this connection
            timeout: Default timeout for API calls (seconds)
            max_retries: Maximum reconnection attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._ib: Optional[IB] = None
        self._rate_limiter = RateLimiter(max_calls=30, period_seconds=1.0)
        self._monitor: Optional[ConnectionMonitor] = None
        self._connected = False

        # Cache for contract details (min tick, etc.)
        self._contract_cache: Dict[str, Contract] = {}
        self._min_tick_cache: Dict[str, float] = {}

    async def __aenter__(self) -> "IBKRClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Connect to TWS/Gateway with retry logic.

        Returns:
            True if connection successful

        Raises:
            ConnectionError if all retries exhausted
        """
        if self._connected and self._ib and self._ib.isConnected():
            return True

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to IBKR {self.host}:{self.port} (attempt {attempt + 1})")

                self._ib = IB()

                # Use ib_insync async API directly (thread-safe, no executor).
                await asyncio.wait_for(
                    self._ib.connectAsync(
                        self.host,
                        self.port,
                        clientId=self.client_id,
                        timeout=self.timeout,
                    ),
                    timeout=self.timeout * 2,
                )

                self._connected = True

                # Start connection monitoring
                self._monitor = ConnectionMonitor(self._ib, self._reconnect)
                self._monitor.start()

                logger.info("Connected to IBKR successfully")
                return True

            except asyncio.TimeoutError:
                logger.warning(f"Connection timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Connection failed: {e} (attempt {attempt + 1})")

            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Failed to connect to IBKR after {self.max_retries} attempts"
        )

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._monitor:
            self._monitor.stop()
            self._monitor = None

        if self._ib and self._ib.isConnected():
            try:
                self._ib.disconnect()
                logger.info("Disconnected from IBKR")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._connected = False
        self._ib = None

    async def _reconnect(self) -> None:
        """Internal reconnect handler."""
        logger.info("Attempting reconnection...")
        await self.disconnect()
        await asyncio.sleep(1.0)
        await self.connect()

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self.is_connected:
            raise ConnectionError("Not connected to IBKR")

    async def _api_call(self, func, *args, **kwargs):
        """
        Execute an API call with rate limiting and timeout.

        Args:
            func: The ib_insync function to call
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the API call
        """
        self._ensure_connected()
        await self._rate_limiter.acquire()

        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=self.timeout)

            if self._monitor:
                self._monitor.heartbeat()

            return result

        except asyncio.TimeoutError:
            logger.error(f"API call timeout: {func.__name__}")
            raise
        except Exception as e:
            logger.error(f"API call error: {func.__name__}: {e}")
            raise

    # =========================================================================
    # Account & Positions
    # =========================================================================

    async def get_account_summary(self) -> AccountSummary:
        """
        Get account summary (NLV, available funds, etc.).

        Returns:
            AccountSummary with current account state
        """
        self._ensure_connected()

        # Use async API; calling the sync wrapper from an async loop will crash.
        account_values = await self._api_call(self._ib.accountSummaryAsync)

        # Parse account values into summary
        wanted_tags = {
            "NetLiquidation",
            "AvailableFunds",
            "BuyingPower",
            "MaintMarginReq",
            "TotalCashValue",
        }
        values_by_currency: Dict[str, Dict[str, float]] = {}
        for av in account_values:
            if av.tag not in wanted_tags:
                continue
            try:
                val = float(av.value) if av.value else 0.0
            except (TypeError, ValueError):
                logger.debug("Skipping non-numeric account value: %s=%r", av.tag, av.value)
                continue

            currency = av.currency or "BASE"
            values_by_currency.setdefault(currency, {})[av.tag] = val

        # Choose the currency bucket that contains the largest NetLiquidation.
        best_currency = "BASE"
        best_nlv = -1.0
        for currency, vals in values_by_currency.items():
            nlv = vals.get("NetLiquidation", 0.0)
            if nlv > best_nlv:
                best_nlv = nlv
                best_currency = currency

        values = values_by_currency.get(best_currency, {})

        return AccountSummary(
            nlv=values.get("NetLiquidation", 0.0),
            available_funds=values.get("AvailableFunds", 0.0),
            buying_power=values.get("BuyingPower", 0.0),
            margin_used=values.get("MaintMarginReq", 0.0),
            cash=values.get("TotalCashValue", 0.0),
            currency=best_currency,
            account_id=account_values[0].account if account_values else "",
        )

    async def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping symbol to Position
        """
        self._ensure_connected()

        await self._api_call(self._ib.reqPositionsAsync)
        positions = self._ib.positions()

        result = {}
        for pos in positions:
            symbol = pos.contract.symbol
            result[symbol] = Position(
                symbol=symbol,
                quantity=int(pos.position),
                avg_cost=pos.avgCost,
                market_value=pos.position * pos.avgCost,  # Approximate
                unrealized_pnl=0.0,  # Updated below
                account=pos.account,
                sec_type=pos.contract.secType,
                currency=pos.contract.currency,
            )

        # Update market values and PnL
        portfolio = self._ib.portfolio()

        for item in portfolio:
            symbol = item.contract.symbol
            if symbol in result:
                result[symbol].market_value = item.marketValue
                result[symbol].unrealized_pnl = item.unrealizedPNL
                result[symbol].realized_pnl = item.realizedPNL

        return result

    # =========================================================================
    # Market Data
    # =========================================================================

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = "2 Y",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical price data.

        Args:
            symbol: Stock/ETF symbol
            duration: Duration string (e.g., "2 Y", "6 M", "30 D")
            bar_size: Bar size (e.g., "1 day", "1 hour")
            what_to_show: Data type ("ADJUSTED_LAST", "TRADES", "BID_ASK")
            use_rth: Use regular trading hours only

        Returns:
            DataFrame with OHLCV data
        """
        self._ensure_connected()

        contract = await self._get_contract(symbol)

        bars = await self._api_call(
            self._ib.reqHistoricalDataAsync,
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,  # Use string dates
        )

        if not bars:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        data = []
        for bar in bars:
            ts = pd.Timestamp(bar.date)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            ts = ts.normalize()
            data.append({
                "date": ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "adjusted_close": bar.close,  # For ADJUSTED_LAST, close IS adjusted
            })

        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        return df

    async def get_quote(self, symbol: str) -> Quote:
        """
        Get current market quote.

        Args:
            symbol: Stock/ETF symbol

        Returns:
            Quote with bid/ask/last prices
        """
        self._ensure_connected()

        contract = await self._get_contract(symbol)
        tickers = await self._api_call(self._ib.reqTickersAsync, contract)
        ticker = tickers[0] if tickers else None
        if ticker is None:
            raise ValueError(f"No ticker returned for {symbol}")

        return Quote(
            symbol=symbol,
            bid=ticker.bid if ticker.bid and ticker.bid > 0 else 0.0,
            ask=ticker.ask if ticker.ask and ticker.ask > 0 else 0.0,
            last=ticker.last if ticker.last and ticker.last > 0 else 0.0,
            bid_size=int(ticker.bidSize) if ticker.bidSize else 0,
            ask_size=int(ticker.askSize) if ticker.askSize else 0,
            volume=int(ticker.volume) if ticker.volume else 0,
        )

    async def get_min_tick(self, symbol: str) -> float:
        """
        Get minimum tick size for a symbol.

        Args:
            symbol: Stock/ETF/Option symbol

        Returns:
            Minimum tick size (e.g., 0.01 for most stocks)
        """
        if symbol in self._min_tick_cache:
            return self._min_tick_cache[symbol]

        self._ensure_connected()

        contract = await self._get_contract(symbol)

        details = await self._api_call(self._ib.reqContractDetailsAsync, contract)

        if details:
            min_tick = details[0].minTick
            self._min_tick_cache[symbol] = min_tick
            return min_tick

        # Default to 0.01 for stocks
        return 0.01

    async def _get_contract(self, symbol: str, sec_type: str = "STK") -> Contract:
        """
        Get or create a qualified contract.

        Args:
            symbol: Symbol
            sec_type: Security type (STK, OPT, etc.)

        Returns:
            Qualified Contract object
        """
        cache_key = f"{symbol}_{sec_type}"

        if cache_key in self._contract_cache:
            return self._contract_cache[cache_key]

        if sec_type == "STK":
            contract = Stock(symbol, "SMART", "USD")
        else:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = "SMART"
            contract.currency = "USD"

        # Qualify the contract
        qualified = await self._api_call(self._ib.qualifyContractsAsync, contract)

        if qualified:
            self._contract_cache[cache_key] = qualified[0]
            return qualified[0]

        return contract

    # =========================================================================
    # Options
    # =========================================================================

    async def get_option_chain(
        self,
        underlying: str,
        exchange: str = "SMART",
    ) -> OptionChain:
        """
        Get option chain for an underlying.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            exchange: Exchange to query

        Returns:
            OptionChain with expirations, strikes, and contracts
        """
        self._ensure_connected()

        # Get chain info
        chains = await self._api_call(
            self._ib.reqSecDefOptParamsAsync,
            underlying,
            "",
            "STK",
            0,  # No specific conId
        )

        if not chains:
            logger.warning(f"No option chain found for {underlying}")
            return OptionChain(underlying=underlying, expirations=[], strikes=[])

        # Use the first chain (usually SMART)
        chain = chains[0]

        # Parse expirations
        expirations = []
        for exp in chain.expirations:
            try:
                expiry_date = datetime.strptime(exp, "%Y%m%d").date()
                expirations.append(expiry_date)
            except ValueError:
                continue

        expirations.sort()

        # Parse strikes
        strikes = sorted(chain.strikes)

        return OptionChain(
            underlying=underlying,
            expirations=expirations,
            strikes=strikes,
        )

    async def get_option_quote(
        self,
        underlying: str,
        strike: float,
        expiry: date,
        right: str,  # "P" or "C"
    ) -> OptionContract:
        """
        Get quote and greeks for a specific option contract.

        Args:
            underlying: Underlying symbol
            strike: Strike price
            expiry: Expiration date
            right: "P" for put, "C" for call

        Returns:
            OptionContract with pricing and greeks
        """
        self._ensure_connected()

        # Create option contract
        contract = Option(
            underlying,
            expiry.strftime("%Y%m%d"),
            strike,
            right,
            "SMART",
        )

        # Qualify contract
        qualified = await self._api_call(self._ib.qualifyContractsAsync, contract)

        if not qualified:
            raise ValueError(f"Could not qualify option contract: {underlying} {strike} {expiry} {right}")

        contract = qualified[0]

        # Request market data with greeks
        tickers = await self._api_call(
            self._ib.reqTickersAsync,
            contract,
        )
        ticker = tickers[0] if tickers else None
        if ticker is None:
            raise ValueError(f"No ticker returned for option {underlying} {strike} {expiry} {right}")

        # Build OptionContract
        opt = OptionContract(
            symbol=underlying,
            strike=strike,
            expiry=expiry,
            right=right,
            bid=ticker.bid if ticker.bid and ticker.bid > 0 else None,
            ask=ticker.ask if ticker.ask and ticker.ask > 0 else None,
            last=ticker.last if ticker.last and ticker.last > 0 else None,
        )

        # Add greeks if available
        if getattr(ticker, "modelGreeks", None):
            opt.delta = ticker.modelGreeks.delta
            opt.gamma = ticker.modelGreeks.gamma
            opt.theta = ticker.modelGreeks.theta
            opt.vega = ticker.modelGreeks.vega
            opt.implied_vol = ticker.modelGreeks.impliedVol
            opt.underlying_price = ticker.modelGreeks.undPrice

        return opt

    # =========================================================================
    # Order Management
    # =========================================================================

    async def place_order(self, order: Order) -> OrderStatusInfo:
        """
        Place an order.

        Args:
            order: Order to place

        Returns:
            OrderStatusInfo with order status
        """
        self._ensure_connected()

        contract = await self._get_contract(order.symbol)

        # Create IBKR order
        if order.order_type == "MKT":
            ib_order = MarketOrder(
                action=order.side,
                totalQuantity=order.quantity,
                tif=order.tif,
                outsideRth=order.outside_rth,
            )
        elif order.order_type == "LMT":
            if order.limit_price is None:
                raise ValueError("Limit orders require a limit_price")
            ib_order = LimitOrder(
                action=order.side,
                totalQuantity=order.quantity,
                lmtPrice=order.limit_price,
                tif=order.tif,
                outsideRth=order.outside_rth,
            )
        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        # Place order
        trade = self._ib.placeOrder(contract, ib_order)

        # Wait briefly for status update
        await asyncio.sleep(0.5)

        # Return status
        return OrderStatusInfo(
            order_id=trade.order.orderId,
            status=trade.orderStatus.status,
            filled_quantity=int(trade.orderStatus.filled),
            remaining_quantity=int(trade.orderStatus.remaining),
            avg_fill_price=trade.orderStatus.avgFillPrice,
        )

    async def what_if_order(self, order: Order) -> MarginImpact:
        """
        Get margin impact of a potential order (what-if).

        Args:
            order: Order to check

        Returns:
            MarginImpact with estimated margin changes
        """
        self._ensure_connected()

        contract = await self._get_contract(order.symbol)

        # Create IBKR order
        if order.order_type == "MKT":
            ib_order = MarketOrder(
                action=order.side,
                totalQuantity=order.quantity,
            )
        else:
            ib_order = LimitOrder(
                action=order.side,
                totalQuantity=order.quantity,
                lmtPrice=order.limit_price or 0,
            )

        # Request what-if
        what_if = await self._api_call(self._ib.whatIfOrderAsync, contract, ib_order)

        # Get current NLV for margin ratio
        summary = await self.get_account_summary()

        post_margin = float(what_if.maintMarginChange) if what_if.maintMarginChange else 0.0
        post_margin_ratio = post_margin / summary.nlv if summary.nlv > 0 else 0.0

        return MarginImpact(
            init_margin_change=float(what_if.initMarginChange) if what_if.initMarginChange else 0.0,
            maint_margin_change=float(what_if.maintMarginChange) if what_if.maintMarginChange else 0.0,
            equity_with_loan=float(what_if.equityWithLoan) if what_if.equityWithLoan else 0.0,
            post_trade_margin=post_margin_ratio,
            commission=float(what_if.commission) if what_if.commission else 0.0,
        )

    async def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation submitted
        """
        self._ensure_connected()

        # Find the trade
        for trade in self._ib.trades():
            if trade.order.orderId == order_id:
                self._ib.cancelOrder(trade.order)
                return True

        logger.warning(f"Order {order_id} not found for cancellation")
        return False

    async def get_order_status(self, order_id: int) -> Optional[OrderStatusInfo]:
        """
        Get status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatusInfo or None if not found
        """
        self._ensure_connected()

        for trade in self._ib.trades():
            if trade.order.orderId == order_id:
                return OrderStatusInfo(
                    order_id=trade.order.orderId,
                    status=trade.orderStatus.status,
                    filled_quantity=int(trade.orderStatus.filled),
                    remaining_quantity=int(trade.orderStatus.remaining),
                    avg_fill_price=trade.orderStatus.avgFillPrice,
                    last_fill_price=trade.orderStatus.lastFillPrice,
                    parent_id=trade.order.parentId,
                    client_id=trade.order.clientId,
                )

        return None

    async def get_executions(self) -> List[Fill]:
        """
        Get today's executions.

        Returns:
            List of Fill objects
        """
        self._ensure_connected()

        fills = self._ib.fills()

        result = []
        for fill in fills:
            result.append(Fill(
                order_id=fill.execution.orderId,
                exec_id=fill.execution.execId,
                symbol=fill.contract.symbol,
                side=fill.execution.side,
                quantity=int(fill.execution.shares),
                price=fill.execution.price,
                commission=fill.commissionReport.commission if fill.commissionReport else 0.0,
                timestamp=fill.execution.time,
                exchange=fill.execution.exchange,
                account=fill.execution.acctNumber,
            ))

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    async def request_current_time(self) -> datetime:
        """
        Get current server time.

        Returns:
            Server timestamp
        """
        self._ensure_connected()

        dt = await self._api_call(self._ib.reqCurrentTimeAsync)

        return dt


def round_to_tick(price: float, min_tick: float) -> float:
    """
    Round price to the nearest tick.

    Args:
        price: Price to round
        min_tick: Minimum tick size

    Returns:
        Rounded price
    """
    if min_tick <= 0:
        return round(price, 2)

    return round(price / min_tick) * min_tick
