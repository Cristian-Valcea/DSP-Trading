"""
Backtest for DSP-100K Sleeve B (Cross-Asset Trend ETFs).

Purpose
-------
Provide a "V15-style" pre-trade kill test for the currently implemented
production Sleeve B logic (signals, vol targeting, caps, and weekly rebal).

This is a backtest for Sleeve B only. Sleeve A is not implemented and Sleeve C
options execution is currently disabled in production, so portfolio-level
results for the full DSP spec are not available yet.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


class SleeveBBacktester:
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
        self._client_id = client_id or (config.ibkr.client_id + 5000)

    async def run(
        self,
        *,
        start: date,
        end: date,
        allocation: float = 0.30,
        initial_nlv: Optional[float] = None,
        execution: str = "open",  # "open" or "close"
        commission_per_share: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> BacktestResult:
        """
        Backtest Sleeve B using IBKR historical bars (ADJUSTED_LAST).

        Execution model (no look-ahead):
        - Signals use closes up to t-1.
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
            else float(self.config.transaction_costs.etf_slippage_bps)
        )

        symbols = list(self.config.sleeve_b.universe)
        if not symbols:
            raise ValueError("Sleeve B universe is empty")

        # Buffer needed for 12M skip-month + 63d vol.
        buffer_days = 650
        fetch_start = start - timedelta(days=buffer_days)

        ib = IBKRClient(host=self._host, port=self._port, client_id=self._client_id)
        await ib.connect()
        try:
            fetcher = DataFetcher(ib, DataCache())

            bars: Dict[str, pd.DataFrame] = {}
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
                    raise RuntimeError(f"Missing historical data for {sym}")
                bars[sym] = df

            dates = self.calendar.get_trading_days(start, end)
            if len(dates) < 10:
                raise ValueError("Backtest window too short")

            # State
            cash = float(initial_nlv)
            positions: Dict[str, int] = {s: 0 for s in symbols}
            equity_points: List[Tuple[date, float]] = []
            trades: List[Trade] = []
            last_rebalance: Optional[date] = None

            def _price(sym: str, dt: date, col: str) -> float:
                ts = pd.Timestamp(dt)
                if ts not in bars[sym].index:
                    raise KeyError(f"{sym} missing {dt}")
                return float(bars[sym].loc[ts, col])

            def _equity(dt: date, col: str) -> float:
                total = cash
                for sym, qty in positions.items():
                    if qty == 0:
                        continue
                    total += qty * _price(sym, dt, col)
                return float(total)

            def _should_rebalance(dt: date) -> bool:
                if last_rebalance is None:
                    return True
                return (dt - last_rebalance).days >= 5

            def _trailing_return(sym: str, end_dt: date, *, months: int, skip_month: bool) -> Optional[float]:
                series = bars[sym]["close"].loc[: pd.Timestamp(end_dt)].dropna()
                if series.empty:
                    return None
                if skip_month:
                    if len(series) <= 21:
                        return None
                    series = series.iloc[:-21]
                trading_days = int(months * 21)
                if len(series) < trading_days:
                    return None
                window = series.tail(trading_days)
                start_px = float(window.iloc[0])
                end_px = float(window.iloc[-1])
                if start_px <= 0:
                    return None
                return end_px / start_px - 1.0

            def _vol(sym: str, end_dt: date, *, lookback: int = 63) -> Optional[float]:
                series = bars[sym]["close"].loc[: pd.Timestamp(end_dt)].dropna()
                if len(series) < lookback + 2:
                    return None
                rets = series.pct_change().dropna().tail(lookback)
                if len(rets) < 5:
                    return None
                v = float(rets.std() * sqrt(252))
                return v if v > 0 else None

            def _weights(signal_dt: date) -> Dict[str, float]:
                # Compute signals/vols for symbols that have enough history.
                ret_1m: Dict[str, float] = {}
                ret_3m: Dict[str, float] = {}
                ret_12m: Dict[str, float] = {}
                vols: Dict[str, float] = {}
                for sym in symbols:
                    r1 = _trailing_return(sym, signal_dt, months=1, skip_month=False)
                    r3 = _trailing_return(sym, signal_dt, months=3, skip_month=False)
                    r12 = _trailing_return(sym, signal_dt, months=12, skip_month=True)
                    v = _vol(sym, signal_dt, lookback=63)
                    if r1 is None or r3 is None or r12 is None or v is None:
                        continue
                    ret_1m[sym] = r1
                    ret_3m[sym] = r3
                    ret_12m[sym] = r12
                    vols[sym] = v

                if not vols:
                    return {}

                composite: Dict[str, float] = {}
                for sym in vols:
                    composite[sym] = 0.25 * ret_1m[sym] + 0.50 * ret_3m[sym] + 0.25 * ret_12m[sym]

                # Inverse-vol weights with sign of composite.
                inv_vol = {s: 1.0 / vols[s] for s in vols}
                inv_sum = sum(inv_vol.values())
                raw = {}
                for sym in inv_vol:
                    sign = 1 if composite[sym] > 0 else (-1 if composite[sym] < 0 else 0)
                    raw[sym] = sign * (inv_vol[sym] / inv_sum)

                total_risk = sum(abs(w) * vols[s] for s, w in raw.items())
                if total_risk <= 0:
                    return {}

                scaled = {s: w * (float(self.config.sleeve_b.vol_target) / total_risk) for s, w in raw.items()}

                # Apply single-name cap with iterative renormalization (matches SleeveB._apply_caps).
                cap = float(self.config.sleeve_b.single_name_cap)
                capped = dict(scaled)
                for _ in range(10):
                    excess_total = 0.0
                    non_capped_weight = 0.0
                    for sym, w in list(capped.items()):
                        if abs(w) > cap:
                            excess_total += abs(w) - cap
                            capped[sym] = cap if w > 0 else -cap
                        else:
                            non_capped_weight += abs(w)
                    if excess_total == 0:
                        break
                    if non_capped_weight > 0:
                        factor = 1 + excess_total / non_capped_weight
                        for sym in list(capped.keys()):
                            if abs(capped[sym]) < cap:
                                capped[sym] *= factor

                # Final normalization to keep same total exposure as pre-cap.
                total_abs_before = sum(abs(w) for w in scaled.values())
                total_abs_after = sum(abs(w) for w in capped.values())
                if total_abs_after > 0 and total_abs_before > 0:
                    scale = total_abs_before / total_abs_after
                    capped = {s: w * scale for s, w in capped.items()}

                return capped

            # Run
            for i, dt in enumerate(dates):
                # Mark equity at execution price basis (open/close) and then at close for reporting.
                exec_col = "open" if execution == "open" else "close"

                # Determine signal date (previous trading day)
                if i == 0:
                    signal_dt = self.calendar.get_previous_trading_day(dt)
                else:
                    signal_dt = dates[i - 1]

                # Rebalance at dt
                if _should_rebalance(dt):
                    weights = _weights(signal_dt)
                    equity_at_exec = _equity(dt, exec_col)
                    sleeve_nav = equity_at_exec * float(allocation)

                    # Convert to target shares.
                    targets: Dict[str, int] = {s: 0 for s in symbols}
                    for sym, w in weights.items():
                        px = _price(sym, dt, exec_col)
                        if px <= 0:
                            continue
                        targets[sym] = int((sleeve_nav * w) / px)

                    # Execute trades at exec_col.
                    for sym in symbols:
                        delta = targets[sym] - positions[sym]
                        if delta == 0:
                            continue

                        side = "BUY" if delta > 0 else "SELL"
                        qty = abs(int(delta))
                        px = _price(sym, dt, exec_col)

                        # Slippage always against us.
                        slip = float(slippage_bps) / 10000.0
                        fill_px = px * (1 + slip) if side == "BUY" else px * (1 - slip)
                        comm = qty * float(commission_per_share)

                        if side == "BUY":
                            cash -= qty * fill_px + comm
                        else:
                            cash += qty * fill_px - comm

                        positions[sym] += int(delta)
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

                    last_rebalance = dt

                # End-of-day equity marked at close.
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


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="Backtest DSP-100K Sleeve B (ETFs trend)")
    parser.add_argument("--config", default="config/dsp100k.yaml", help="Path to dsp100k.yaml")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--allocation", type=float, default=0.30, help="Fraction of NLV allocated to Sleeve B (default: 0.30)")
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
    bt = SleeveBBacktester(config)
    result = await bt.run(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        allocation=float(args.allocation),
        initial_nlv=args.initial_nlv,
        execution=args.execution,
    )

    print("\nSleeve B backtest summary")
    print(f"- Total return: {_format_pct(result.metrics['return'])}")
    print(f"- CAGR: {_format_pct(result.metrics['cagr'])}")
    print(f"- Vol: {_format_pct(result.metrics['vol'])}")
    print(f"- Sharpe: {_format_float(result.metrics['sharpe'])}")
    print(f"- Max DD: {_format_pct(result.metrics['max_dd'])}")
    print(f"- Corr to SPY: {_format_float(result.correlation_to_spy)}")

    print("\nKill-test windows (diagnostic)")
    _print_kill_table(result.equity)

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
