"""
Backtest for DSP-100K Sleeve A (Equity Momentum + SPY Hedge).

Purpose
-------
Provide a "negative filter" kill test for Sleeve A before paper trading.
If this backtest can't beat cash with survivorship bias HELPING it,
the strategy is definitely broken.

IMPORTANT: This backtest uses a fixed S&P 100 universe (survivorship bias).
A passing result proves NOTHING positive. A failing result proves the strategy
is broken and should not be traded.

Design:
- 12-1 momentum: 252-day return skipping most recent 21 days
- Monthly stock rebalance: first trading day of each month
- Weekly SPY hedge update: every 5 trading days
- Name cap: 4% per position
- Sector cap: 20% gross per sector
- Vol cap: 5% annualized portfolio volatility
- Beta target: ≤ 0.60 net (long beta minus SPY hedge)
- SPY hedge cap: 20% of sleeve NAV
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.cache import DataCache
from ..data.fetcher import DataFetcher
from ..ibkr.client import IBKRClient
from ..utils.config import DSPConfig, load_config
from ..utils.time import MarketCalendar

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())  # negative


def _perf_metrics(equity: pd.Series) -> Dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 3:
        return {
            "return": 0.0,
            "cagr": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
        }

    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    vol = float(rets.std() * sqrt(252)) if rets.std() and rets.std() > 0 else 0.0
    sharpe = float(_safe_div(rets.mean(), rets.std()) * sqrt(252)) if rets.std() and rets.std() > 0 else 0.0

    max_dd = _max_drawdown(equity)
    calmar = float(_safe_div(cagr, abs(max_dd))) if max_dd < 0 else 0.0

    return {
        "return": total_return,
        "cagr": float(cagr),
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(max_dd),
        "calmar": calmar,
    }


@dataclass(frozen=True)
class Trade:
    dt: date
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float

    @property
    def notional(self) -> float:
        return float(self.quantity) * float(self.price)


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    correlation_to_spy: Optional[float]


class SleeveABacktester:
    """
    Backtest Sleeve A: Long-only 12-1 momentum + SPY hedge.

    Key parameters from locked spec:
    - n_long: 20 names
    - max_weight_per_name: 4%
    - max_sector_gross: 20%
    - vol_target: 5%
    - beta_limit: 0.60
    - spy_hedge_cap: 20%
    """

    # Match SleeveA constants
    _MOM_SKIP_DAYS = 21
    _MOM_WINDOW_DAYS = 252
    _ADV_LOOKBACK_DAYS = 20
    _BETA_LOOKBACK_DAYS = 63
    _VOL_LOOKBACK_DAYS = 63

    def __init__(
        self,
        config: DSPConfig,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ):
        self.config = config
        self.calendar = MarketCalendar()

        self._host = host or config.ibkr.host
        self._port = port or config.ibkr.port
        self._client_id = client_id or (config.ibkr.client_id + 5001)

    async def run(
        self,
        *,
        start: date,
        end: date,
        allocation: float = 0.60,
        initial_nlv: Optional[float] = None,
        execution: str = "open",  # "open" or "close"
        commission_per_share: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> BacktestResult:
        """
        Backtest Sleeve A using IBKR historical bars (ADJUSTED_LAST).

        Execution model (no look-ahead):
        - Signals use closes up to t-1.
        - Stock rebalance monthly (first trading day of month).
        - SPY hedge update weekly (every 5 trading days).
        - Rebalance at t open (execution="open") or t close (execution="close").
        """
        if execution not in {"open", "close"}:
            raise ValueError("execution must be 'open' or 'close'")

        if initial_nlv is None:
            initial_nlv = float(self.config.general.nlv_target)

        commission_per_share = (
            float(commission_per_share)
            if commission_per_share is not None
            else float(self.config.transaction_costs.stock_commission_per_share)
        )
        slippage_bps = (
            float(slippage_bps)
            if slippage_bps is not None
            else float(self.config.transaction_costs.stock_slippage_bps)
        )

        # Universe from config
        universe = self.config.sleeve_a_universe
        if not universe:
            raise ValueError("Sleeve A universe is empty")

        symbols = [self._normalize_symbol(e["symbol"]) for e in universe]
        symbol_to_sector = {self._normalize_symbol(e["symbol"]): e["sector"] for e in universe}

        # Buffer needed for 12M momentum + skip-month + beta/vol windows.
        buffer_days = 650
        fetch_start = start - timedelta(days=buffer_days)

        ib = IBKRClient(host=self._host, port=self._port, client_id=self._client_id)
        await ib.connect()
        try:
            fetcher = DataFetcher(ib, DataCache())

            bars: Dict[str, pd.DataFrame] = {}
            logger.info("Fetching historical data for %d symbols + SPY...", len(symbols))
            for sym in symbols + ["SPY"]:
                df = await fetcher.get_daily_bars(
                    sym,
                    lookback_days=(end - fetch_start).days + 10,
                    end_date=end,
                    adjusted=True,
                    use_cache=True,
                    validate=True,
                )
                if df is None or df.empty:
                    logger.warning("Missing historical data for %s, skipping", sym)
                    continue
                bars[sym] = df

            if "SPY" not in bars:
                raise RuntimeError("Missing SPY historical data")

            # Filter symbols to those with data
            symbols = [s for s in symbols if s in bars]
            logger.info("Using %d symbols with complete data", len(symbols))

            dates = self.calendar.get_trading_days(start, end)
            if len(dates) < 10:
                raise ValueError("Backtest window too short")

            # State
            cash = float(initial_nlv)
            positions: Dict[str, int] = {s: 0 for s in symbols + ["SPY"]}
            equity_points: List[Tuple[date, float]] = []
            trades: List[Trade] = []
            last_stock_rebalance: Optional[date] = None
            last_hedge_update: Optional[date] = None

            # Current weights for hedge maintenance
            current_long_weights: Dict[str, float] = {}

            def _price(sym: str, dt: date, col: str) -> float:
                ts = pd.Timestamp(dt)
                if sym not in bars or ts not in bars[sym].index:
                    raise KeyError(f"{sym} missing {dt}")
                return float(bars[sym].loc[ts, col])

            def _equity(dt: date, col: str) -> float:
                total = cash
                for sym, qty in positions.items():
                    if qty == 0:
                        continue
                    try:
                        total += qty * _price(sym, dt, col)
                    except KeyError:
                        pass
                return float(total)

            def _is_first_trading_day_of_month(dt: date) -> bool:
                try:
                    prev_td = self.calendar.get_previous_trading_day(dt)
                except Exception:
                    return True
                return prev_td.month != dt.month

            def _should_rebalance_stocks(dt: date) -> bool:
                if last_stock_rebalance is None:
                    return True
                return _is_first_trading_day_of_month(dt)

            def _should_update_hedge(dt: date) -> bool:
                if last_hedge_update is None:
                    return True
                # Weekly: every 5 trading days
                return (dt - last_hedge_update).days >= 5

            def _momentum_12_1(sym: str, end_dt: date) -> Optional[float]:
                """12-month return skipping most recent month."""
                if sym not in bars:
                    return None
                closes = bars[sym]["close"].loc[:pd.Timestamp(end_dt)].dropna()
                if len(closes) < self._MOM_SKIP_DAYS + self._MOM_WINDOW_DAYS + 2:
                    return None
                closes = closes.iloc[:-self._MOM_SKIP_DAYS]  # skip most recent month
                window = closes.tail(self._MOM_WINDOW_DAYS)
                if len(window) < self._MOM_WINDOW_DAYS:
                    return None
                start_px = float(window.iloc[0])
                end_px = float(window.iloc[-1])
                if start_px <= 0:
                    return None
                return end_px / start_px - 1.0

            def _adv(sym: str, end_dt: date) -> Optional[float]:
                if sym not in bars or "volume" not in bars[sym].columns:
                    return None
                sub = bars[sym][["close", "volume"]].loc[:pd.Timestamp(end_dt)].dropna().tail(self._ADV_LOOKBACK_DAYS)
                if len(sub) < 5:
                    return None
                dv = (sub["close"].astype(float) * sub["volume"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
                if dv.empty:
                    return None
                return float(dv.mean())

            def _beta_to_spy(sym: str, end_dt: date) -> Optional[float]:
                if sym not in bars:
                    return None
                stock = bars[sym]["close"].loc[:pd.Timestamp(end_dt)].pct_change().dropna()
                spy = bars["SPY"]["close"].loc[:pd.Timestamp(end_dt)].pct_change().dropna()
                joined = pd.concat([stock.rename("s"), spy.rename("m")], axis=1).dropna().tail(self._BETA_LOOKBACK_DAYS)
                if len(joined) < 20:
                    return None
                var_m = float(joined["m"].var())
                if var_m <= 0:
                    return None
                cov = float(joined["s"].cov(joined["m"]))
                return float(cov / var_m)

            def _portfolio_vol(weights: Dict[str, float], end_dt: date) -> float:
                """Compute portfolio volatility from weights."""
                if not weights:
                    return 0.0
                returns = {}
                for sym in weights:
                    if sym not in bars:
                        continue
                    returns[sym] = bars[sym]["close"].loc[:pd.Timestamp(end_dt)].pct_change()
                if not returns:
                    return 0.0
                rets = pd.DataFrame(returns).dropna(how="all").tail(self._VOL_LOOKBACK_DAYS)
                if len(rets) < 20:
                    return 0.0
                w = pd.Series(weights).reindex(rets.columns).fillna(0.0)
                port = (rets.fillna(0.0) * w).sum(axis=1)
                vol = float(port.std() * sqrt(252)) if port.std() and port.std() > 0 else 0.0
                return vol

            def _select_longs(signal_dt: date) -> Tuple[List[Tuple[str, float, str]], Dict[str, float]]:
                """
                Select top n_long names by 12-1 momentum, apply caps.
                Returns: (selected list of (symbol, momentum, sector), weights dict)
                """
                candidates = []
                for sym in symbols:
                    mom = _momentum_12_1(sym, signal_dt)
                    adv = _adv(sym, signal_dt)
                    if mom is None or adv is None:
                        continue

                    # Price filter
                    try:
                        px = _price(sym, signal_dt, "close")
                    except KeyError:
                        continue
                    if px < float(self.config.sleeve_a.min_price):
                        continue

                    # ADV filter
                    if adv < float(self.config.sleeve_a.min_adv):
                        continue

                    sector = symbol_to_sector.get(sym, "Unknown")
                    candidates.append((sym, mom, sector))

                if not candidates:
                    return [], {}

                # Sort by momentum descending, take top n_long
                candidates.sort(key=lambda x: x[1], reverse=True)
                n_long = int(self.config.sleeve_a.n_long)
                selected = candidates[:n_long]

                # Initial equal weight at name cap
                name_cap = float(self.config.sleeve_a.max_weight_per_name)
                weights = {s[0]: name_cap for s in selected}

                # Apply sector cap
                sector_cap = float(self.config.sleeve_a.max_sector_gross)
                sector_to_syms: Dict[str, List[str]] = {}
                for sym, _, sector in selected:
                    sector_to_syms.setdefault(sector, []).append(sym)

                for sector, syms in sector_to_syms.items():
                    gross = sum(weights[s] for s in syms)
                    if gross > sector_cap:
                        factor = sector_cap / gross
                        for s in syms:
                            weights[s] *= factor

                # Apply vol cap
                vol = _portfolio_vol(weights, signal_dt)
                vol_cap = float(self.config.sleeve_a.vol_target)
                if vol > vol_cap > 0:
                    scale = vol_cap / vol
                    weights = {s: w * scale for s, w in weights.items()}

                # Check beta feasibility - if too high, drop lowest momentum names
                beta_limit = float(self.config.sleeve_a.beta_limit)
                spy_cap = float(self.config.sleeve_a.spy_hedge_cap)
                max_achievable_beta = beta_limit + spy_cap

                while len(selected) > 1:
                    beta_long = sum(
                        weights.get(s, 0.0) * (_beta_to_spy(s, signal_dt) or 1.0)
                        for s, _, _ in selected
                    )
                    if beta_long <= max_achievable_beta + 1e-6:
                        break
                    # Drop lowest momentum name
                    selected = selected[:-1]
                    # Recompute weights
                    weights = {s[0]: name_cap for s in selected}
                    for sector, syms in sector_to_syms.items():
                        syms_in_selected = [s for s in syms if s in weights]
                        if not syms_in_selected:
                            continue
                        gross = sum(weights[s] for s in syms_in_selected)
                        if gross > sector_cap:
                            factor = sector_cap / gross
                            for s in syms_in_selected:
                                weights[s] *= factor
                    vol = _portfolio_vol(weights, signal_dt)
                    if vol > vol_cap > 0:
                        scale = vol_cap / vol
                        weights = {s: w * scale for s, w in weights.items()}

                return selected, weights

            def _compute_hedge_weight(long_weights: Dict[str, float], signal_dt: date) -> float:
                """Compute SPY short weight to reduce beta to target."""
                if not long_weights:
                    return 0.0

                beta_long = sum(
                    w * (_beta_to_spy(s, signal_dt) or 1.0)
                    for s, w in long_weights.items()
                )
                beta_limit = float(self.config.sleeve_a.beta_limit)
                spy_cap = float(self.config.sleeve_a.spy_hedge_cap)
                required = max(0.0, beta_long - beta_limit)
                return min(spy_cap, required)

            def _execute_trades(
                dt: date,
                targets: Dict[str, int],
                exec_col: str,
            ) -> None:
                nonlocal cash
                for sym in set(targets.keys()) | set(positions.keys()):
                    if sym not in bars:
                        continue
                    target = targets.get(sym, 0)
                    current = positions.get(sym, 0)
                    delta = target - current
                    if delta == 0:
                        continue

                    try:
                        px = _price(sym, dt, exec_col)
                    except KeyError:
                        continue

                    side = "BUY" if delta > 0 else "SELL"
                    qty = abs(int(delta))

                    # Slippage always against us
                    slip = float(slippage_bps) / 10000.0
                    fill_px = px * (1 + slip) if side == "BUY" else px * (1 - slip)
                    comm = qty * float(commission_per_share)

                    if side == "BUY":
                        cash -= qty * fill_px + comm
                    else:
                        cash += qty * fill_px - comm

                    positions[sym] = target
                    trades.append(
                        Trade(
                            dt=dt,
                            symbol=sym,
                            side=side,
                            quantity=qty,
                            price=float(fill_px),
                            commission=float(comm),
                        )
                    )

            # Run backtest
            for i, dt in enumerate(dates):
                exec_col = "open" if execution == "open" else "close"

                # Determine signal date (previous trading day)
                if i == 0:
                    signal_dt = self.calendar.get_previous_trading_day(dt)
                else:
                    signal_dt = dates[i - 1]

                rebalance_stocks = _should_rebalance_stocks(dt)
                update_hedge = _should_update_hedge(dt) or rebalance_stocks

                equity_at_exec = _equity(dt, exec_col)
                sleeve_nav = equity_at_exec * float(allocation)

                targets: Dict[str, int] = {}

                if rebalance_stocks:
                    # Full stock selection
                    selected, long_weights = _select_longs(signal_dt)
                    current_long_weights = long_weights

                    # Stock targets
                    for sym, w in long_weights.items():
                        try:
                            px = _price(sym, dt, exec_col)
                        except KeyError:
                            continue
                        if px <= 0:
                            continue
                        targets[sym] = int((sleeve_nav * w) / px)

                    # Exit positions not in selection
                    for sym in positions:
                        if sym == "SPY":
                            continue
                        if positions[sym] != 0 and sym not in targets:
                            targets[sym] = 0

                    last_stock_rebalance = dt

                if update_hedge:
                    # Compute hedge
                    if not rebalance_stocks:
                        # Maintain existing weights
                        long_weights = current_long_weights

                    hedge_weight = _compute_hedge_weight(long_weights, signal_dt)

                    # SPY target (short)
                    try:
                        spy_px = _price("SPY", dt, exec_col)
                    except KeyError:
                        spy_px = 0.0

                    if spy_px > 0 and hedge_weight > 0:
                        targets["SPY"] = -int((sleeve_nav * hedge_weight) / spy_px)
                    else:
                        targets["SPY"] = 0

                    last_hedge_update = dt

                # Execute trades
                if targets:
                    _execute_trades(dt, targets, exec_col)

                # End-of-day equity marked at close
                equity_points.append((dt, _equity(dt, "close")))

            equity = pd.Series(
                data=[v for _, v in equity_points],
                index=pd.to_datetime([d for d, _ in equity_points]),
                name="equity",
            )
            daily_returns = equity.pct_change().dropna()

            # SPY correlation (diagnostic)
            spy_close = bars["SPY"]["close"].reindex(equity.index).dropna()
            spy_rets = spy_close.pct_change().dropna()
            joined = pd.concat([daily_returns.rename("dsp"), spy_rets.rename("spy")], axis=1).dropna()
            corr = float(joined["dsp"].corr(joined["spy"])) if len(joined) > 20 else None

            trades_df = pd.DataFrame(
                [
                    {
                        "date": t.dt,
                        "symbol": t.symbol,
                        "side": t.side,
                        "quantity": t.quantity,
                        "price": t.price,
                        "notional": t.notional,
                        "commission": t.commission,
                    }
                    for t in trades
                ]
            )

            metrics = _perf_metrics(equity)
            return BacktestResult(
                equity=equity,
                daily_returns=daily_returns,
                trades=trades_df,
                metrics=metrics,
                correlation_to_spy=corr,
            )

        finally:
            await ib.disconnect()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for IBKR (e.g., BRK.B -> BRK B)."""
        return str(symbol).upper().strip().replace(".", " ")


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.2f}%"


def _format_float(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def _window(equity: pd.Series, start: date, end: date) -> pd.Series:
    idx = equity.index
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return equity.loc[(idx >= s) & (idx <= e)]


def _print_kill_table(equity: pd.Series) -> None:
    """
    Print kill-test windows.

    Key windows for negative filter:
    - 2022-2024: Recent full cycle including drawdown year
    - 2023-2024: Most recent period
    """
    windows = [
        ("Full 5Y", date(equity.index[-1].year - 4, 1, 1), equity.index[-1].date()),
        ("2020-2021", date(2020, 1, 1), date(2021, 12, 31)),
        ("2022-2024", date(2022, 1, 1), date(2024, 12, 31)),
        ("2023-2024", date(2023, 1, 1), date(2024, 12, 31)),
        ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ]

    rows = []
    for name, s, e in windows:
        sub = _window(equity, s, e)
        if len(sub) < 10:
            continue
        m = _perf_metrics(sub)
        rows.append(
            {
                "window": name,
                "return": m["return"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
            }
        )

    if not rows:
        print("No windows have enough data to evaluate.")
        return

    df = pd.DataFrame(rows)
    df["return"] = df["return"].map(lambda x: f"{x*100:.2f}%")
    df["max_dd"] = df["max_dd"].map(lambda x: f"{x*100:.2f}%")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:.2f}")
    print(df.to_string(index=False))


def _print_kill_verdict(metrics: Dict[str, float]) -> str:
    """
    Print kill verdict based on metrics.

    Kill criteria from spec:
    - Sharpe < 0 in any major window: FAIL
    - Max DD > 15%: FAIL (kill_drawdown_threshold)
    """
    sharpe = metrics.get("sharpe", 0.0)
    max_dd = metrics.get("max_dd", 0.0)

    if sharpe < 0:
        return "❌ FAIL: Sharpe < 0 (can't even beat cash with survivorship bias helping)"
    if max_dd < -0.15:
        return f"❌ FAIL: Max DD {max_dd*100:.1f}% exceeds -15% threshold"
    if sharpe < 0.3:
        return f"⚠️ MARGINAL: Sharpe {sharpe:.2f} is weak (survivorship bias may be hiding problems)"
    return f"✅ PASS negative filter: Sharpe {sharpe:.2f}, Max DD {max_dd*100:.1f}%"


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="Backtest DSP-100K Sleeve A (Equity Momentum)")
    parser.add_argument("--config", default="config/dsp100k.yaml", help="Path to dsp100k.yaml")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--allocation", type=float, default=0.60, help="Fraction of NLV allocated to Sleeve A (default: 0.60)")
    parser.add_argument("--initial-nlv", type=float, default=None, help="Initial NLV (default: config.general.nlv_target)")
    parser.add_argument("--execution", choices=["open", "close"], default="open", help="Rebalance at open (default) or close")
    parser.add_argument("--output", default=None, help="Write equity/trades to a parquet file")
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config, strict=False)
    bt = SleeveABacktester(config)
    result = await bt.run(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        allocation=float(args.allocation),
        initial_nlv=args.initial_nlv,
        execution=args.execution,
    )

    print("\n" + "=" * 60)
    print("SLEEVE A BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Total return: {_format_pct(result.metrics['return'])}")
    print(f"CAGR: {_format_pct(result.metrics['cagr'])}")
    print(f"Volatility: {_format_pct(result.metrics['vol'])}")
    print(f"Sharpe: {_format_float(result.metrics['sharpe'])}")
    print(f"Max DD: {_format_pct(result.metrics['max_dd'])}")
    print(f"Calmar: {_format_float(result.metrics['calmar'])}")
    print(f"Corr to SPY: {_format_float(result.correlation_to_spy)}")

    print("\n" + "-" * 60)
    print("KILL-TEST WINDOWS (negative filter)")
    print("-" * 60)
    _print_kill_table(result.equity)

    print("\n" + "-" * 60)
    print("VERDICT")
    print("-" * 60)
    print(_print_kill_verdict(result.metrics))

    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    print("⚠️  This backtest uses a FIXED S&P 100 universe (survivorship bias).")
    print("    A PASS proves NOTHING positive about the strategy.")
    print("    A FAIL proves the strategy is broken and should NOT be traded.")

    if args.output:
        out_path = args.output
        out_dir = out_path.rsplit("/", 1)[0] if "/" in out_path else "."
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        payload = pd.DataFrame({"equity": result.equity})
        payload.to_parquet(out_path)
        print(f"\nWrote equity to {out_path}")

    return 0


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
