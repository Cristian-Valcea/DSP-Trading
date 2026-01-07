"""
IBKR Futures Data Fetcher for Sleeve ORB

Fetches 1-minute OHLCV data for micro futures contracts (MES, MNQ) from
Interactive Brokers TWS/Gateway. Handles continuous series construction
with proper roll logic.

IBKR Futures Contract Format:
- Symbol: MES, MNQ (base symbol)
- Exchange: CME
- SecType: FUT
- LastTradeDateOrContractMonth: YYYYMM (e.g., 202503 for March 2025)

Usage:
    python -m dsp.data.ibkr_futures_fetcher \
        --symbols MES,MNQ \
        --start 2022-01-01 \
        --end 2025-03-31 \
        --output-dir data/orb
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ib_insync import
try:
    from ib_insync import IB, Contract, Future, util
except ImportError:
    raise ImportError("ib_insync required: pip install ib_insync")

logger = logging.getLogger(__name__)

# Timezones
ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")  # CME uses Central Time

# Quarterly expiration months for E-mini/Micro futures
QUARTERLY_MONTHS = [3, 6, 9, 12]  # H, M, U, Z
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Roll schedule
DAYS_BEFORE_EXPIRY_ROLL = 5

# IBKR rate limiting
IBKR_PACING_DELAY = 2.0  # seconds between requests
MAX_BARS_PER_REQUEST = 50000  # IBKR limit


@dataclass
class ContractInfo:
    """Futures contract information."""
    base_symbol: str
    month: int
    year: int

    @property
    def expiry_yyyymm(self) -> str:
        """Get IBKR lastTradeDateOrContractMonth format (YYYYMM)."""
        return f"{self.year}{self.month:02d}"

    @property
    def ticker(self) -> str:
        """Human-readable ticker (e.g., MESH5)."""
        month_code = MONTH_CODES.get(self.month, "?")
        year_digit = self.year % 10
        return f"{self.base_symbol}{month_code}{year_digit}"

    def __str__(self) -> str:
        return self.ticker


def _third_friday(year: int, month: int) -> date:
    """Calculate third Friday of a month."""
    first_day = date(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)


def get_front_contract(base_symbol: str, as_of_date: date) -> ContractInfo:
    """
    Get the front-month contract as of a given date.

    For MES/MNQ, contracts are quarterly (H, M, U, Z).
    Roll 5 days before third Friday of expiration month.
    """
    year = as_of_date.year
    month = as_of_date.month

    # Find next quarterly month
    for m in QUARTERLY_MONTHS:
        if m >= month:
            exp_month = m
            exp_year = year
            break
    else:
        exp_month = QUARTERLY_MONTHS[0]  # March next year
        exp_year = year + 1

    # Check if we should roll to next contract
    third_friday = _third_friday(exp_year, exp_month)
    roll_date = third_friday - timedelta(days=DAYS_BEFORE_EXPIRY_ROLL)

    if as_of_date >= roll_date:
        idx = QUARTERLY_MONTHS.index(exp_month)
        if idx == len(QUARTERLY_MONTHS) - 1:
            exp_month = QUARTERLY_MONTHS[0]
            exp_year += 1
        else:
            exp_month = QUARTERLY_MONTHS[idx + 1]

    return ContractInfo(base_symbol=base_symbol, month=exp_month, year=exp_year)


def get_contract_schedule(
    base_symbol: str,
    start_date: date,
    end_date: date,
) -> List[Tuple[ContractInfo, date, date]]:
    """
    Generate contract schedule with roll dates.

    Returns list of (contract, start_date, end_date) tuples.
    """
    schedule = []
    current_date = start_date

    while current_date <= end_date:
        contract = get_front_contract(base_symbol, current_date)
        third_friday = _third_friday(contract.year, contract.month)
        roll_date = third_friday - timedelta(days=DAYS_BEFORE_EXPIRY_ROLL)
        contract_end = min(roll_date - timedelta(days=1), end_date)

        if contract_end >= current_date:
            schedule.append((contract, current_date, contract_end))

        current_date = roll_date

    return schedule


class IBKRFuturesFetcher:
    """
    Fetches futures minute bars from IBKR TWS/Gateway.

    Features:
    - Per-contract data fetching with proper roll handling
    - RTH-only data (9:30 AM - 4:00 PM ET)
    - Local caching in parquet format
    - Continuous series construction with additive back-adjustment
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 99,
        cache_dir: str = "data/orb",
    ):
        """
        Initialize IBKR futures fetcher.

        Args:
            host: TWS/Gateway host
            port: TWS/Gateway port (7497 for TWS paper, 4002 for Gateway paper)
            client_id: IBKR client ID
            cache_dir: Directory for caching data
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ib: Optional[IB] = None
        self._recent_errors: List[Tuple[int, int, str]] = []

    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        if self.ib is not None and self.ib.isConnected():
            return True

        self.ib = IB()
        try:
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=20,
            )
            # Capture API errors so we can diagnose "no bars" scenarios.
            self.ib.errorEvent += self._on_ib_error
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")
            try:
                accounts = self.ib.managedAccounts()
                if accounts:
                    logger.info("Managed accounts: %s", ", ".join(accounts))
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib is not None and self.ib.isConnected():
            try:
                self.ib.errorEvent -= self._on_ib_error
            except Exception:
                pass
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    def _on_ib_error(self, req_id: int, error_code: int, error_string: str, contract=None):
        """
        Collect recent IBKR API errors.

        Common relevant codes:
        - 200: No security definition has been found for the request
        - 354: Requested market data is not subscribed
        - 162: Historical market data service error / no data returned
        """
        # Keep a small rolling buffer for diagnosis.
        self._recent_errors.append((req_id, error_code, error_string))
        self._recent_errors = self._recent_errors[-50:]

    def _drain_recent_errors(self) -> List[Tuple[int, int, str]]:
        errs = list(self._recent_errors)
        self._recent_errors.clear()
        return errs

    def _make_contract(self, contract_info: ContractInfo) -> Future:
        """Create IBKR Future contract."""
        return Future(
            symbol=contract_info.base_symbol,
            lastTradeDateOrContractMonth=contract_info.expiry_yyyymm,
            exchange="CME",
            currency="USD",
        )

    def _get_available_contracts(self, base_symbol: str) -> Dict[str, Future]:
        """
        Get all available futures contracts for a symbol.

        Uses reqContractDetails without expiry to get all tradable contracts.
        Returns dict mapping expiry YYYYMM to qualified contract.
        """
        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("Not connected to IBKR")

        contracts_by_expiry = {}

        # Try CME first (standard), then GLOBEX (common routing)
        for exchange in ["CME", "GLOBEX"]:
            self._drain_recent_errors()
            # Query without expiry to get all available contracts
            query = Future(symbol=base_symbol, exchange=exchange, currency="USD")

            try:
                details_list = self.ib.reqContractDetails(query)
                if details_list:
                    for details in details_list:
                        contract = details.contract
                        expiry = contract.lastTradeDateOrContractMonth
                        if expiry and len(expiry) >= 6:
                            # Store the qualified contract
                            expiry_key = expiry[:6]  # YYYYMM
                            if expiry_key not in contracts_by_expiry:
                                contracts_by_expiry[expiry_key] = contract
                                logger.debug(f"Found contract: {contract.localSymbol} expiry={expiry}")

                    if contracts_by_expiry:
                        logger.info(f"Found {len(contracts_by_expiry)} available {base_symbol} contracts via {exchange}")
                        break
            except Exception as e:
                logger.debug(f"Error querying {base_symbol} on {exchange}: {e}")
                continue

        if not contracts_by_expiry:
            errs = self._drain_recent_errors()
            if errs:
                logger.warning("IBKR errors during contract query: %s", errs[-5:])

        return contracts_by_expiry

    def _qualify_with_fallback_exchanges(
        self, contract_info: ContractInfo, exchanges: List[str]
    ) -> Optional[Future]:
        """
        Qualify a futures contract, trying multiple exchanges.

        IBKR commonly routes CME equity index futures via "GLOBEX" even though the venue is CME.
        When the exchange string is mismatched, IB can return error 200 (no security definition).

        Note: This only works for non-expired contracts. For historical data of expired
        contracts, IBKR may not return contract details. Use _get_available_contracts
        to see what's actually available.
        """
        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("Not connected to IBKR")

        for exchange in exchanges:
            self._drain_recent_errors()
            contract = Future(
                symbol=contract_info.base_symbol,
                lastTradeDateOrContractMonth=contract_info.expiry_yyyymm,
                exchange=exchange,
                currency="USD",
            )
            try:
                qualified = self.ib.qualifyContracts(contract)
            except Exception as e:
                logger.debug(
                    "Exception qualifying %s on exchange=%s: %s", contract_info, exchange, e
                )
                continue

            if qualified:
                if exchange != "CME":
                    logger.info(
                        "Qualified %s using exchange=%s (fallback).", contract_info, exchange
                    )
                return qualified[0]

        errs = self._drain_recent_errors()
        if errs:
            logger.warning("IBKR errors during qualify: %s", errs[-5:])
        return None

    def fetch_contract_bars(
        self,
        contract_info: ContractInfo,
        start_date: date,
        end_date: date,
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch 1-minute bars for a specific contract.

        Args:
            contract_info: Contract to fetch
            start_date: Start date
            end_date: End date
            use_rth: If True, fetch RTH only (9:30-16:00 ET)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if not self.ib or not self.ib.isConnected():
            raise RuntimeError("Not connected to IBKR")

        # Qualify the contract
        try:
            # Try common exchange routes first. Many CME index futures qualify via "GLOBEX".
            contract = self._qualify_with_fallback_exchanges(
                contract_info, exchanges=["GLOBEX", "CME"]
            )
            if contract is None:
                logger.warning(f"Could not qualify contract {contract_info}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error qualifying {contract_info}: {e}")
            return pd.DataFrame()

        # IBKR historical data request
        # Need to chunk requests - max ~1 year of 1-min data per request
        all_bars = []
        current_end = datetime.combine(end_date, datetime.max.time())
        current_end = current_end.replace(tzinfo=ET)
        start_dt = datetime.combine(start_date, datetime.min.time())
        start_dt = start_dt.replace(tzinfo=ET)

        while current_end > start_dt:
            # Request up to 1 month at a time for 1-min bars
            duration = "1 M"

            try:
                logger.info(f"Fetching {contract_info} ending {current_end.date()}")

                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime=current_end,
                    durationStr=duration,
                    barSizeSetting="1 min",
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )

                if bars:
                    for bar in bars:
                        bar_dt = bar.date
                        if isinstance(bar_dt, str):
                            bar_dt = datetime.strptime(bar_dt, "%Y%m%d %H:%M:%S")
                        if bar_dt.tzinfo is None:
                            bar_dt = bar_dt.replace(tzinfo=ET)

                        all_bars.append({
                            "timestamp": bar_dt,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": int(bar.volume),
                        })

                    # Move window back
                    earliest = min(b["timestamp"] for b in all_bars[-len(bars):])
                    current_end = earliest - timedelta(minutes=1)
                else:
                    errs = self._drain_recent_errors()
                    if errs:
                        # Surface the most relevant recent errors.
                        logger.warning("IBKR returned 0 bars; recent errors: %s", errs[-5:])
                        codes = {c for _, c, _ in errs}
                        if 354 in codes:
                            logger.warning(
                                "Likely cause: missing CME futures market data subscription (error 354)."
                            )
                        if 200 in codes:
                            logger.warning(
                                "Likely cause: contract definition mismatch (error 200). "
                                "Check symbol/exchange/expiry format for %s.",
                                contract_info,
                            )
                        if 162 in codes:
                            logger.warning(
                                "Likely cause: IBKR HMDS returned no data (error 162). "
                                "May be permissions, pacing, or requesting outside available history."
                            )
                    # No more data
                    break

                # IBKR pacing
                time.sleep(IBKR_PACING_DELAY)

            except Exception as e:
                logger.error(f"Error fetching {contract_info}: {e}")
                break

        if not all_bars:
            logger.warning(f"No bars found for {contract_info}")
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Filter to requested date range
        start_ts = pd.Timestamp(start_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df.index >= start_ts) & (df.index < end_ts)]

        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]

        logger.info(f"Fetched {len(df)} bars for {contract_info}")
        return df

    def build_continuous_series(
        self,
        base_symbol: str,
        start_date: date,
        end_date: date,
        adjustment: str = "additive",
    ) -> pd.DataFrame:
        """
        Build continuous futures series with roll handling.

        Args:
            base_symbol: Base symbol (MES, MNQ)
            start_date: Start date
            end_date: End date
            adjustment: "additive" or "none"

        Returns:
            DataFrame with continuous price series
        """
        schedule = get_contract_schedule(base_symbol, start_date, end_date)

        all_dfs = []
        cumulative_adjustment = 0.0
        prev_close = None

        for contract, c_start, c_end in schedule:
            logger.info(f"Fetching {contract} from {c_start} to {c_end}")

            df = self.fetch_contract_bars(contract, c_start, c_end)
            if df.empty:
                continue

            # Apply back-adjustment
            if adjustment == "additive" and prev_close is not None:
                first_close = df.iloc[0]["close"]
                gap = first_close - prev_close
                cumulative_adjustment += gap

            if cumulative_adjustment != 0:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] - cumulative_adjustment

            df["contract"] = contract.ticker
            all_dfs.append(df)

            if not df.empty:
                prev_close = df.iloc[-1]["close"]

        if not all_dfs:
            return pd.DataFrame()

        continuous = pd.concat(all_dfs).sort_index()
        continuous = continuous[~continuous.index.duplicated(keep="last")]

        return continuous

    def fetch_and_save(
        self,
        base_symbol: str,
        start_date: date,
        end_date: date,
        output_file: Optional[str] = None,
    ) -> Path:
        """
        Fetch data and save to parquet file.

        Args:
            base_symbol: Base symbol (MES, MNQ)
            start_date: Start date
            end_date: End date
            output_file: Output file path (auto-generated if None)

        Returns:
            Path to saved file
        """
        df = self.build_continuous_series(base_symbol, start_date, end_date)

        if df.empty:
            raise ValueError(f"No data fetched for {base_symbol}")

        if output_file is None:
            output_file = self.cache_dir / f"{base_symbol}_1min_{start_date}_{end_date}.parquet"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file)

        logger.info(f"Saved {len(df)} bars to {output_file}")
        return output_file


def fetch_orb_data(
    symbols: List[str] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2025-03-31",
    output_dir: str = "data/orb",
    host: str = "127.0.0.1",
    port: int = 7497,
) -> Dict[str, Path]:
    """
    Fetch all data needed for ORB backtest via IBKR.

    Args:
        symbols: List of base symbols (default: MES, MNQ)
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory
        host: IBKR host
        port: IBKR port

    Returns:
        Dict mapping symbol to output file path
    """
    if symbols is None:
        symbols = ["MES", "MNQ"]

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    results = {}
    fetcher = IBKRFuturesFetcher(host=host, port=port, cache_dir=output_dir)

    try:
        if not fetcher.connect():
            raise RuntimeError("Could not connect to IBKR")

        for symbol in symbols:
            try:
                path = fetcher.fetch_and_save(symbol, start, end)
                results[symbol] = path
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

    finally:
        fetcher.disconnect()

    return results


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch futures data from IBKR")
    parser.add_argument(
        "--symbols",
        type=str,
        default="MES,MNQ",
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-03-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/orb",
        help="Output directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="IBKR host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7497,
        help="IBKR port (7497=TWS paper, 4002=Gateway paper)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    symbols = [s.strip() for s in args.symbols.split(",")]

    results = fetch_orb_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
    )

    print("\n=== Fetch Complete ===")
    for symbol, path in results.items():
        print(f"{symbol}: {path}")
