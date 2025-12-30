"""
Backtest for Asset-Class Dual Momentum (ETF-only, no survivorship bias).

Strategy
--------
Gary Antonacci-style "Dual Momentum":
1. Universe: SPY, EFA, EEM, IEF, TLT, TIP, GLD, PDBC, UUP (9 risky assets)
2. Cash: SHY (short-term treasuries)
3. Monthly rebalance (first trading day of month)
4. Rank by 12-1 momentum (12-month return, skip most recent month)
5. Hold top K assets where momentum > 0
6. If no asset has momentum > 0, go 100% cash (SHY)
7. Vol-target the portfolio to 8% annual volatility

Kill-Test Purpose
-----------------
If this can't beat SHY (cash) in a backtest with NO survivorship bias,
it's definitely broken. ETFs don't have the single-stock survivorship
problem that killed Sleeve A.
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

import pandas as pd

from ..data.cache import DataCache
from ..data.fetcher import DataFetcher
from ..ibkr.client import IBKRClient
from ..utils.config import DSPConfig, load_config
from ..utils.time import MarketCalendar

logger = logging.getLogger(__name__)

# ============================================================================
# Strategy Constants (FROZEN)
# ============================================================================

# Universe: 9 risky assets + 1 cash equivalent
RISKY_UNIVERSE = ["SPY", "EFA", "EEM", "IEF", "TLT", "TIP", "GLD", "PDBC", "UUP"]
CASH_ETF = "SHY"

# Momentum calculation
MOM_WINDOW_DAYS = 252  # 12 months
MOM_SKIP_DAYS = 21     # Skip most recent month (avoid mean reversion)

# Portfolio construction
TOP_K = 3              # Hold top K assets by momentum
VOL_TARGET = 0.08      # 8% annual volatility target
VOL_LOOKBACK = 63      # 3 months for vol estimate

# Rebalance timing
REBAL_FREQ = "monthly"  # First trading day of each month


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


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
    monthly_holdings: pd.DataFrame


class ETFDualMomentumBacktester:
    """
    Asset-class Dual Momentum backtester.

    No survivorship bias because ETFs don't delist/go bankrupt like stocks.
    """

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
        initial_nlv: float = 100_000.0,
        top_k: int = TOP_K,
        vol_target: float = VOL_TARGET,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5.0,
    ) -> BacktestResult:
        """
        Run the dual momentum backtest.

        Execution model:
        - Signals computed on last trading day of previous month
        - Rebalance at open on first trading day of month
        """
        all_symbols = RISKY_UNIVERSE + [CASH_ETF]

        # Buffer for 12-month momentum + vol lookback
        buffer_days = 400
        fetch_start = start - timedelta(days=buffer_days)

        ib = IBKRClient(host=self._host, port=self._port, client_id=self._client_id)
        await ib.connect()

        try:
            fetcher = DataFetcher(ib, DataCache())

            bars: Dict[str, pd.DataFrame] = {}
            for sym in all_symbols:
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
            if len(dates) < 20:
                raise ValueError("Backtest window too short")

            # State
            cash = float(initial_nlv)
            positions: Dict[str, int] = {s: 0 for s in all_symbols}
            equity_points: List[Tuple[date, float]] = []
            trades: List[Trade] = []
            monthly_holdings: List[Dict] = []
            last_rebal_month: Optional[Tuple[int, int]] = None

            def _price(sym: str, dt: date, col: str) -> float:
                ts = pd.Timestamp(dt)
                if ts not in bars[sym].index:
                    # Find nearest prior date
                    prior = bars[sym].index[bars[sym].index <= ts]
                    if len(prior) == 0:
                        raise KeyError(f"{sym} missing {dt}")
                    return float(bars[sym].loc[prior[-1], col])
                return float(bars[sym].loc[ts, col])

            def _equity(dt: date, col: str) -> float:
                total = cash
                for sym, qty in positions.items():
                    if qty == 0:
                        continue
                    total += qty * _price(sym, dt, col)
                return float(total)

            def _is_first_of_month(dt: date) -> bool:
                if last_rebal_month is None:
                    return True
                return (dt.year, dt.month) != last_rebal_month

            def _momentum_12_1(sym: str, signal_dt: date) -> Optional[float]:
                """12-1 momentum: 12-month return, skip most recent month."""
                series = bars[sym]["close"].loc[: pd.Timestamp(signal_dt)].dropna()
                if len(series) < MOM_WINDOW_DAYS + MOM_SKIP_DAYS:
                    return None

                # Skip most recent 21 days
                series = series.iloc[:-MOM_SKIP_DAYS]

                # 12-month window
                window = series.tail(MOM_WINDOW_DAYS)
                if len(window) < MOM_WINDOW_DAYS:
                    return None

                start_px = float(window.iloc[0])
                end_px = float(window.iloc[-1])
                if start_px <= 0:
                    return None
                return end_px / start_px - 1.0

            def _portfolio_vol(weights: Dict[str, float], signal_dt: date) -> float:
                """Estimate portfolio volatility from individual asset vols."""
                # Simple: sum of weight * vol (ignores correlations - conservative)
                total_vol = 0.0
                for sym, w in weights.items():
                    if w == 0:
                        continue
                    series = bars[sym]["close"].loc[: pd.Timestamp(signal_dt)].dropna()
                    if len(series) < VOL_LOOKBACK + 2:
                        continue
                    rets = series.pct_change().dropna().tail(VOL_LOOKBACK)
                    vol = float(rets.std() * sqrt(252))
                    total_vol += abs(w) * vol
                return total_vol

            def _compute_weights(signal_dt: date) -> Dict[str, float]:
                """Dual momentum: rank by 12-1, hold top K with positive momentum."""
                mom_scores: Dict[str, float] = {}
                for sym in RISKY_UNIVERSE:
                    m = _momentum_12_1(sym, signal_dt)
                    if m is not None:
                        mom_scores[sym] = m

                # Sort by momentum descending
                ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)

                # Take top K with positive momentum
                selected = []
                for sym, m in ranked[:top_k]:
                    if m > 0:
                        selected.append(sym)

                if not selected:
                    # All momentum negative -> go to cash
                    return {CASH_ETF: 1.0}

                # Equal weight among selected
                raw_weight = 1.0 / len(selected)
                weights = {s: raw_weight for s in selected}

                # Vol-target scaling
                port_vol = _portfolio_vol(weights, signal_dt)
                if port_vol > 0:
                    scale = vol_target / port_vol
                    scale = min(scale, 1.5)  # Don't lever more than 1.5x
                    weights = {s: w * scale for s, w in weights.items()}

                    # If scaled down, remainder to cash
                    total_weight = sum(weights.values())
                    if total_weight < 1.0:
                        weights[CASH_ETF] = 1.0 - total_weight

                return weights

            # Run simulation
            for i, dt in enumerate(dates):
                # Signal date is previous trading day
                if i == 0:
                    signal_dt = self.calendar.get_previous_trading_day(dt)
                else:
                    signal_dt = dates[i - 1]

                # Monthly rebalance on first trading day of month
                if _is_first_of_month(dt):
                    weights = _compute_weights(signal_dt)
                    equity_at_open = _equity(dt, "open")

                    # Record holdings for diagnostics
                    monthly_holdings.append({
                        "date": dt,
                        "equity": equity_at_open,
                        **{f"w_{s}": weights.get(s, 0.0) for s in all_symbols}
                    })

                    # Convert to target shares
                    targets: Dict[str, int] = {s: 0 for s in all_symbols}
                    for sym, w in weights.items():
                        if w == 0:
                            continue
                        px = _price(sym, dt, "open")
                        if px <= 0:
                            continue
                        targets[sym] = int((equity_at_open * w) / px)

                    # Execute trades
                    for sym in all_symbols:
                        delta = targets[sym] - positions[sym]
                        if delta == 0:
                            continue

                        side = "BUY" if delta > 0 else "SELL"
                        qty = abs(delta)
                        px = _price(sym, dt, "open")

                        # Slippage
                        slip = slippage_bps / 10000.0
                        fill_px = px * (1 + slip) if side == "BUY" else px * (1 - slip)
                        comm = qty * commission_per_share

                        if side == "BUY":
                            cash -= qty * fill_px + comm
                        else:
                            cash += qty * fill_px - comm

                        positions[sym] += delta
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

                    last_rebal_month = (dt.year, dt.month)

                # End-of-day equity
                equity_points.append((dt, _equity(dt, "close")))

            # Build results
            equity = pd.Series(
                data=[v for _, v in equity_points],
                index=pd.to_datetime([d for d, _ in equity_points]),
                name="equity",
            )
            daily_returns = equity.pct_change().dropna()

            # SPY correlation
            spy_close = bars["SPY"]["close"].reindex(equity.index).dropna()
            spy_rets = spy_close.pct_change().dropna()
            joined = pd.concat([daily_returns.rename("strat"), spy_rets.rename("spy")], axis=1).dropna()
            corr = float(joined["strat"].corr(joined["spy"])) if len(joined) > 20 else None

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

            holdings_df = pd.DataFrame(monthly_holdings)

            metrics = _perf_metrics(equity)
            return BacktestResult(
                equity=equity,
                daily_returns=daily_returns,
                trades=trades_df,
                metrics=metrics,
                correlation_to_spy=corr,
                monthly_holdings=holdings_df,
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


def _print_kill_table(equity: pd.Series) -> Dict[str, float]:
    """Print kill-test windows and return the 2022-2024 metrics."""
    windows = [
        ("2020-2024", date(2020, 1, 1), date(2024, 12, 31)),
        ("2022-2024", date(2022, 1, 1), date(2024, 12, 31)),
        ("2023-2024", date(2023, 1, 1), date(2024, 12, 31)),
        ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ]

    kill_metrics = {}
    rows = []
    for name, s, e in windows:
        sub = _window(equity, s, e)
        if len(sub) < 10:
            continue
        m = _perf_metrics(sub)
        rows.append({
            "window": name,
            "return": m["return"],
            "cagr": m["cagr"],
            "vol": m["vol"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
        })
        if name == "2022-2024":
            kill_metrics = m

    if not rows:
        print("No windows have enough data to evaluate.")
        return {}

    df = pd.DataFrame(rows)
    df["return"] = df["return"].map(lambda x: f"{x*100:.2f}%")
    df["cagr"] = df["cagr"].map(lambda x: f"{x*100:.2f}%")
    df["vol"] = df["vol"].map(lambda x: f"{x*100:.2f}%")
    df["max_dd"] = df["max_dd"].map(lambda x: f"{x*100:.2f}%")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:.2f}")
    print(df.to_string(index=False))

    return kill_metrics


def _print_kill_verdict(metrics: Dict[str, float]) -> str:
    """Determine if strategy passes the kill test."""
    sharpe = metrics.get("sharpe", 0.0)
    max_dd = metrics.get("max_dd", 0.0)
    vol = metrics.get("vol", 0.0)
    cagr = metrics.get("cagr", 0.0)

    verdicts = []

    # Primary kill criteria
    if sharpe < 0:
        verdicts.append(f"❌ FAIL: Sharpe {sharpe:.2f} < 0 (can't beat cash)")
    elif sharpe < 0.3:
        verdicts.append(f"⚠️  WEAK: Sharpe {sharpe:.2f} < 0.3 (marginally better than cash)")
    else:
        verdicts.append(f"✅ PASS: Sharpe {sharpe:.2f} >= 0.3")

    # Drawdown check
    if max_dd < -0.25:
        verdicts.append(f"❌ FAIL: Max DD {max_dd*100:.1f}% exceeds -25% threshold")
    elif max_dd < -0.15:
        verdicts.append(f"⚠️  WARN: Max DD {max_dd*100:.1f}% is significant")
    else:
        verdicts.append(f"✅ PASS: Max DD {max_dd*100:.1f}% acceptable")

    # Vol target check
    if vol < 0.04:
        verdicts.append(f"⚠️  WARN: Vol {vol*100:.1f}% very low (strategy too conservative?)")
    elif vol > 0.12:
        verdicts.append(f"⚠️  WARN: Vol {vol*100:.1f}% exceeds 12% target range")
    else:
        verdicts.append(f"✅ PASS: Vol {vol*100:.1f}% in target range")

    # Overall verdict
    if sharpe < 0 or max_dd < -0.25:
        overall = "\n🔴 KILL: Strategy fails negative filter - DO NOT TRADE"
    elif sharpe < 0.3 or max_dd < -0.15:
        overall = "\n🟡 MARGINAL: Strategy passes but edge is weak"
    else:
        overall = "\n🟢 TRADABLE: Strategy passes kill test - proceed to paper trading"

    return "\n".join(verdicts) + overall


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="Backtest ETF Dual Momentum")
    parser.add_argument("--config", default="config/dsp100k.yaml", help="Path to dsp100k.yaml")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--initial-nlv", type=float, default=100_000.0, help="Initial NLV")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of top assets to hold (default: {TOP_K})")
    parser.add_argument("--vol-target", type=float, default=VOL_TARGET, help=f"Annual vol target (default: {VOL_TARGET})")
    parser.add_argument("--output", default=None, help="Write equity/trades to parquet")
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config, strict=False)
    bt = ETFDualMomentumBacktester(config)
    result = await bt.run(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        initial_nlv=args.initial_nlv,
        top_k=args.top_k,
        vol_target=args.vol_target,
    )

    print("\n" + "="*60)
    print("ETF DUAL MOMENTUM BACKTEST RESULTS")
    print("="*60)
    print(f"\nStrategy: Top-{args.top_k} by 12-1 momentum, vol-target {args.vol_target*100:.0f}%")
    print(f"Universe: {', '.join(RISKY_UNIVERSE)} + {CASH_ETF} (cash)")

    print("\n--- Overall Metrics ---")
    print(f"Total return: {_format_pct(result.metrics['return'])}")
    print(f"CAGR: {_format_pct(result.metrics['cagr'])}")
    print(f"Volatility: {_format_pct(result.metrics['vol'])}")
    print(f"Sharpe: {_format_float(result.metrics['sharpe'])}")
    print(f"Max DD: {_format_pct(result.metrics['max_dd'])}")
    print(f"Calmar: {_format_float(result.metrics['calmar'])}")
    print(f"Corr to SPY: {_format_float(result.correlation_to_spy)}")

    print("\n--- Kill-Test Windows ---")
    kill_metrics = _print_kill_table(result.equity)

    print("\n--- Kill Test Verdict (2022-2024) ---")
    print(_print_kill_verdict(kill_metrics))

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
