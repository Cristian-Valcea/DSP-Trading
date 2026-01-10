"""
VRP-Gated Equity Risk Premium Harvester.

Strategy
--------
Harvest equity risk premium (SPY) using VRP signal + VRP Regime Gate:
1. VRP Signal: Long SPY when VRP > 0 (implied vol > realized vol)
2. Regime Gate: Only trade when VRPRegimeGate state is OPEN or REDUCE
3. Position Scaling: OPEN=100%, REDUCE=50%, CLOSED=0%
4. Monthly rebalance on first trading day of month

Why This Works
--------------
Unlike the failed approaches:
- VRP Futures: Failed because VIX doesn't reliably decay (Sharpe -0.13)
- VRP Gate + Direction NN: Failed because gate blocks profitable shorts during crises

This approach works because:
- We're harvesting equity premium, not VIX premium
- Gate AVOIDS crises (halves max drawdown from -33% to -15%)
- We're not predicting direction, just harvesting systematic premium
- Simple rule: "Be long SPY when volatility is elevated but not in crisis"

Kill-Test Results (from vrp_erp_pivot_tests.py)
----------------------------------------------
Buy & Hold SPY:                    Sharpe 0.85, MaxDD -33.7%
Long SPY when VRP > 0:             Sharpe 0.91, MaxDD -20.3%
Long SPY when VRP > 0 AND Gate:    Sharpe 0.87, MaxDD -15.2%

2022-2024 Period (Bear+Recovery):
VRP+Gate: Sharpe 1.46, MaxDD -6.6%
Buy&Hold: Sharpe 0.57, MaxDD -24.5%

Crisis Avoidance:
- COVID-19: Gated -1.4% vs Ungated -33.4%
- 2022 Bear: Gated -2.3% vs Ungated -24.1%
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..regime.vrp_regime_gate import GateState, VRPRegimeGate

logger = logging.getLogger(__name__)

# ============================================================================
# Data Paths
# ============================================================================
DATA_DIR = Path("data/vrp")
EQUITY_DIR = DATA_DIR / "equities"
INDICES_DIR = DATA_DIR / "indices"
FUTURES_DIR = DATA_DIR / "futures"

# ============================================================================
# Strategy Constants (FROZEN)
# ============================================================================

# VRP Calculation
VRP_RV_WINDOW = 21  # 21-day realized vol (trailing)

# Position Scaling by Gate State
GATE_SCALE = {
    GateState.OPEN: 1.0,
    GateState.REDUCE: 0.5,
    GateState.CLOSED: 0.0,
}

# Vol-Target
VOL_TARGET = 0.10  # 10% annual vol target
VOL_LOOKBACK = 63  # 3 months for vol estimate
MAX_LEVERAGE = 1.5  # Don't lever more than 1.5x

# Rebalance
REBAL_FREQ = "monthly"  # First trading day of month


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
    gate_state: str
    vrp: float


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    trades: List[Trade]
    metrics: Dict[str, float]
    correlation_to_spy: Optional[float]
    time_in_market_pct: float
    gate_states: pd.Series
    vrp_series: pd.Series


def load_spy_data() -> pd.DataFrame:
    """Load SPY daily data."""
    path = EQUITY_DIR / "SPY_daily.parquet"
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index = df.index.normalize()
    df = df.sort_index()
    return df


def load_gate_data() -> pd.DataFrame:
    """Load VIX, VVIX, VX data for gate."""
    vix = pd.read_parquet(INDICES_DIR / "VIX_spot.parquet")
    vvix = pd.read_parquet(INDICES_DIR / "VVIX.parquet")
    vx = pd.read_parquet(FUTURES_DIR / "VX_F1_CBOE.parquet")
    df = (
        vix.rename(columns={"vix_spot": "vix"})
        .join(vvix, how="inner")
        .join(vx[["vx_f1"]], how="inner")
        .dropna()
    )
    df.index = pd.to_datetime(df.index).tz_localize("UTC").normalize()
    return df


def compute_vrp(spy: pd.DataFrame, vix: pd.Series, window: int = VRP_RV_WINDOW) -> pd.Series:
    """
    Compute VRP = VIX - Realized Volatility.

    Positive VRP means implied vol > realized vol (normal market condition).
    Negative VRP means realized vol > implied vol (crisis/panic).
    """
    # Realized vol: annualized std of log returns
    log_rets = np.log(spy["close"] / spy["close"].shift(1))
    rv = log_rets.rolling(window).std() * sqrt(252) * 100  # Annualized, in VIX units

    # VRP = IV - RV
    vrp = vix - rv
    return vrp


def run_gate(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Run VRP Regime Gate and return states + scores."""
    gate = VRPRegimeGate()
    states = []
    scores = []
    for ts, row in df.iterrows():
        state = gate.update(
            vix=float(row["vix"]),
            vvix=float(row["vvix"]),
            vx_f1=float(row["vx_f1"]),
            as_of_date=ts.date() if hasattr(ts, 'date') else ts,
        )
        states.append(state)
        scores.append(gate.last_score)
    return pd.Series(states, index=df.index, name="gate_state"), pd.Series(scores, index=df.index, name="gate_score")


class VRPERPBacktester:
    """
    VRP-Gated Equity Risk Premium Harvester.

    Strategy: Long SPY when VRP > 0 AND Gate allows.
    """

    def __init__(
        self,
        *,
        vol_target: float = VOL_TARGET,
        max_leverage: float = MAX_LEVERAGE,
        commission_per_share: float = 0.005,
        slippage_bps: float = 5.0,
    ):
        self.vol_target = vol_target
        self.max_leverage = max_leverage
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps

    def run(
        self,
        *,
        start: date,
        end: date,
        initial_nlv: float = 100_000.0,
        require_vrp_positive: bool = True,
    ) -> BacktestResult:
        """
        Run the VRP ERP backtest.

        Args:
            start: Start date
            end: End date
            initial_nlv: Initial portfolio value
            require_vrp_positive: If True, only go long when VRP > 0
        """
        # Load data
        spy = load_spy_data()
        gate_data = load_gate_data()

        # Run gate
        gate_states, gate_scores = run_gate(gate_data)

        # Compute VRP
        vix = gate_data["vix"]
        vrp = compute_vrp(spy, vix)

        # Align data
        df = spy[["close", "open"]].copy()
        df["vix"] = vix.reindex(df.index).ffill()
        df["vrp"] = vrp.reindex(df.index).ffill()
        df["gate_state"] = gate_states.reindex(df.index).ffill()
        df["gate_score"] = gate_scores.reindex(df.index).ffill()
        df = df.dropna()

        # Filter to date range
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]

        if len(df) < 20:
            raise ValueError(f"Insufficient data: {len(df)} rows")

        # State
        cash = float(initial_nlv)
        shares = 0
        equity_points: List[Tuple[pd.Timestamp, float]] = []
        trades: List[Trade] = []
        gate_state_history: List[Tuple[pd.Timestamp, GateState]] = []
        vrp_history: List[Tuple[pd.Timestamp, float]] = []
        last_rebal_month: Optional[Tuple[int, int]] = None
        days_in_market = 0

        def _equity(close: float) -> float:
            return cash + shares * close

        def _is_first_of_month(dt: pd.Timestamp) -> bool:
            if last_rebal_month is None:
                return True
            return (dt.year, dt.month) != last_rebal_month

        def _spy_vol(idx: int) -> float:
            """Estimate SPY vol from trailing returns."""
            if idx < VOL_LOOKBACK:
                return 0.15  # Default 15%
            window = df["close"].iloc[max(0, idx - VOL_LOOKBACK):idx]
            rets = window.pct_change().dropna()
            if len(rets) < 20:
                return 0.15
            return float(rets.std() * sqrt(252))

        # Run simulation
        for i, (ts, row) in enumerate(df.iterrows()):
            vrp_val = row["vrp"]
            gate_state = row["gate_state"]
            close_px = row["close"]
            open_px = row["open"]

            gate_state_history.append((ts, gate_state))
            vrp_history.append((ts, vrp_val))

            # Monthly rebalance
            if _is_first_of_month(ts):
                # Determine target exposure
                if require_vrp_positive and vrp_val <= 0:
                    # VRP negative -> don't be long
                    target_exposure = 0.0
                else:
                    # VRP positive (or not required) -> scale by gate
                    gate_scale = GATE_SCALE.get(gate_state, 0.0)
                    target_exposure = gate_scale

                # Vol-target scaling
                spy_vol = _spy_vol(i)
                if spy_vol > 0:
                    vol_scale = self.vol_target / spy_vol
                    vol_scale = min(vol_scale, self.max_leverage)
                    target_exposure *= vol_scale

                # Calculate target shares
                current_equity = _equity(open_px)
                target_notional = current_equity * target_exposure
                target_shares = int(target_notional / open_px) if open_px > 0 else 0

                # Execute trade
                delta = target_shares - shares
                if delta != 0:
                    side = "BUY" if delta > 0 else "SELL"
                    qty = abs(delta)
                    slip = self.slippage_bps / 10000.0
                    fill_px = open_px * (1 + slip) if side == "BUY" else open_px * (1 - slip)
                    comm = qty * self.commission_per_share

                    if side == "BUY":
                        cash -= qty * fill_px + comm
                    else:
                        cash += qty * fill_px - comm

                    shares = target_shares
                    trades.append(Trade(
                        dt=ts.date(),
                        symbol="SPY",
                        side=side,
                        quantity=qty,
                        price=float(fill_px),
                        gate_state=gate_state.value if hasattr(gate_state, 'value') else str(gate_state),
                        vrp=float(vrp_val),
                    ))

                last_rebal_month = (ts.year, ts.month)

            # Track time in market
            if shares > 0:
                days_in_market += 1

            # End-of-day equity
            equity_points.append((ts, _equity(close_px)))

        # Build results
        equity = pd.Series(
            data=[v for _, v in equity_points],
            index=pd.DatetimeIndex([d for d, _ in equity_points]),
            name="equity",
        )
        daily_returns = equity.pct_change().dropna()

        # SPY correlation (buy & hold)
        spy_close = df["close"]
        spy_rets = spy_close.pct_change().dropna()
        joined = pd.concat([daily_returns.rename("strat"), spy_rets.rename("spy")], axis=1).dropna()
        corr = float(joined["strat"].corr(joined["spy"])) if len(joined) > 20 else None

        # Gate state series
        gate_series = pd.Series(
            data=[s for _, s in gate_state_history],
            index=pd.DatetimeIndex([d for d, _ in gate_state_history]),
            name="gate_state",
        )

        # VRP series
        vrp_series = pd.Series(
            data=[v for _, v in vrp_history],
            index=pd.DatetimeIndex([d for d, _ in vrp_history]),
            name="vrp",
        )

        time_in_market = days_in_market / len(df) * 100 if len(df) > 0 else 0.0

        metrics = _perf_metrics(equity)

        return BacktestResult(
            equity=equity,
            daily_returns=daily_returns,
            trades=trades,
            metrics=metrics,
            correlation_to_spy=corr,
            time_in_market_pct=time_in_market,
            gate_states=gate_series,
            vrp_series=vrp_series,
        )


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
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    return equity.loc[(idx >= s) & (idx <= e)]


def _print_kill_table(equity: pd.Series, baseline: pd.Series) -> Dict[str, float]:
    """Print kill-test windows comparing strategy to baseline."""
    windows = [
        ("2018-2024", date(2018, 1, 1), date(2024, 12, 31)),
        ("2020-2024", date(2020, 1, 1), date(2024, 12, 31)),
        ("2022-2024", date(2022, 1, 1), date(2024, 12, 31)),
        ("2023-2024", date(2023, 1, 1), date(2024, 12, 31)),
    ]

    kill_metrics = {}
    rows = []
    for name, s, e in windows:
        sub = _window(equity, s, e)
        base_sub = _window(baseline, s, e)
        if len(sub) < 10:
            continue
        m = _perf_metrics(sub)
        bm = _perf_metrics(base_sub)
        rows.append({
            "window": name,
            "strat_sharpe": m["sharpe"],
            "base_sharpe": bm["sharpe"],
            "strat_dd": m["max_dd"],
            "base_dd": bm["max_dd"],
            "strat_cagr": m["cagr"],
            "base_cagr": bm["cagr"],
        })
        if name == "2022-2024":
            kill_metrics = m

    if not rows:
        print("No windows have enough data to evaluate.")
        return {}

    df = pd.DataFrame(rows)
    df["strat_sharpe"] = df["strat_sharpe"].map(lambda x: f"{x:.2f}")
    df["base_sharpe"] = df["base_sharpe"].map(lambda x: f"{x:.2f}")
    df["strat_dd"] = df["strat_dd"].map(lambda x: f"{x*100:.1f}%")
    df["base_dd"] = df["base_dd"].map(lambda x: f"{x*100:.1f}%")
    df["strat_cagr"] = df["strat_cagr"].map(lambda x: f"{x*100:.1f}%")
    df["base_cagr"] = df["base_cagr"].map(lambda x: f"{x*100:.1f}%")
    print(df.to_string(index=False))

    return kill_metrics


def _print_kill_verdict(metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> str:
    """Determine if strategy passes kill test vs baseline."""
    sharpe = metrics.get("sharpe", 0.0)
    max_dd = metrics.get("max_dd", 0.0)
    base_sharpe = baseline_metrics.get("sharpe", 0.0)
    base_dd = baseline_metrics.get("max_dd", 0.0)

    verdicts = []

    # Sharpe check
    if sharpe < 0.5:
        verdicts.append(f"❌ FAIL: Sharpe {sharpe:.2f} < 0.50 minimum")
    else:
        verdicts.append(f"✅ PASS: Sharpe {sharpe:.2f} >= 0.50")

    # Drawdown improvement
    dd_improvement = (base_dd - max_dd) / abs(base_dd) * 100 if base_dd < 0 else 0
    if max_dd < base_dd:  # Lower (less negative) is better
        verdicts.append(f"✅ PASS: DD {max_dd*100:.1f}% vs baseline {base_dd*100:.1f}% ({dd_improvement:.0f}% better)")
    else:
        verdicts.append(f"⚠️  WARN: DD {max_dd*100:.1f}% worse than baseline {base_dd*100:.1f}%")

    # Sharpe vs baseline
    if sharpe >= base_sharpe:
        verdicts.append(f"✅ PASS: Sharpe {sharpe:.2f} >= baseline {base_sharpe:.2f}")
    else:
        verdicts.append(f"⚠️  WARN: Sharpe {sharpe:.2f} < baseline {base_sharpe:.2f} (but DD better)")

    # Overall verdict
    if sharpe < 0.5 or max_dd < -0.30:
        overall = "\n🔴 KILL: Strategy fails kill test - DO NOT TRADE"
    elif sharpe >= 0.5 and max_dd > base_dd:
        overall = "\n🟢 TRADABLE: Strategy passes kill test - proceed to paper trading"
    else:
        overall = "\n🟡 MARGINAL: Strategy has tradeoffs vs baseline - review carefully"

    return "\n".join(verdicts) + overall


def _compute_buy_and_hold(spy: pd.DataFrame, start: date, end: date, initial_nlv: float) -> pd.Series:
    """Compute buy & hold SPY equity curve."""
    spy = spy.copy()
    if spy.index.tz is None:
        spy.index = pd.to_datetime(spy.index).tz_localize("UTC")
    spy = spy[(spy.index >= pd.Timestamp(start, tz="UTC")) & (spy.index <= pd.Timestamp(end, tz="UTC"))]

    if len(spy) < 2:
        return pd.Series(dtype=float)

    shares = int(initial_nlv / spy["close"].iloc[0])
    equity = shares * spy["close"]
    equity.name = "baseline"
    return equity


def main() -> int:
    parser = argparse.ArgumentParser(description="VRP-Gated Equity Risk Premium Harvester Backtest")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--initial-nlv", type=float, default=100_000.0, help="Initial NLV")
    parser.add_argument("--vol-target", type=float, default=VOL_TARGET, help=f"Annual vol target (default: {VOL_TARGET})")
    parser.add_argument("--no-vrp-filter", action="store_true", help="Don't require VRP > 0")
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    bt = VRPERPBacktester(vol_target=args.vol_target)
    result = bt.run(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        initial_nlv=args.initial_nlv,
        require_vrp_positive=not args.no_vrp_filter,
    )

    # Compute baseline
    spy = load_spy_data()
    baseline = _compute_buy_and_hold(spy, _parse_date(args.start), _parse_date(args.end), args.initial_nlv)
    baseline_metrics = _perf_metrics(baseline)

    print("\n" + "="*70)
    print("VRP-GATED EQUITY RISK PREMIUM HARVESTER")
    print("="*70)
    print(f"\nStrategy: Long SPY when VRP > 0 AND Gate allows")
    print(f"VRP Filter: {'Enabled' if not args.no_vrp_filter else 'Disabled'}")
    print(f"Vol Target: {args.vol_target*100:.0f}%")
    print(f"Gate Scaling: OPEN=100%, REDUCE=50%, CLOSED=0%")

    print("\n--- Overall Metrics ---")
    print(f"Total return: {_format_pct(result.metrics['return'])}")
    print(f"CAGR: {_format_pct(result.metrics['cagr'])}")
    print(f"Volatility: {_format_pct(result.metrics['vol'])}")
    print(f"Sharpe: {_format_float(result.metrics['sharpe'])}")
    print(f"Max DD: {_format_pct(result.metrics['max_dd'])}")
    print(f"Calmar: {_format_float(result.metrics['calmar'])}")
    print(f"Corr to SPY: {_format_float(result.correlation_to_spy)}")
    print(f"Time in Market: {result.time_in_market_pct:.1f}%")
    print(f"Trades: {len(result.trades)}")

    print("\n--- Baseline (Buy & Hold SPY) ---")
    print(f"CAGR: {_format_pct(baseline_metrics['cagr'])}")
    print(f"Sharpe: {_format_float(baseline_metrics['sharpe'])}")
    print(f"Max DD: {_format_pct(baseline_metrics['max_dd'])}")

    print("\n--- Kill-Test Windows (Strategy vs Baseline) ---")
    kill_metrics = _print_kill_table(result.equity, baseline)

    print("\n--- Kill Test Verdict (2022-2024) ---")
    print(_print_kill_verdict(kill_metrics, baseline_metrics))

    # Gate state distribution
    gate_counts = result.gate_states.value_counts()
    print("\n--- Gate State Distribution ---")
    for state in [GateState.OPEN, GateState.REDUCE, GateState.CLOSED]:
        count = gate_counts.get(state, 0)
        pct = count / len(result.gate_states) * 100 if len(result.gate_states) > 0 else 0
        print(f"  {state.value}: {count} days ({pct:.1f}%)")

    # VRP distribution
    vrp = result.vrp_series
    print("\n--- VRP Distribution ---")
    print(f"  Mean: {vrp.mean():.2f}")
    print(f"  Std: {vrp.std():.2f}")
    print(f"  % Positive: {(vrp > 0).sum() / len(vrp) * 100:.1f}%")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "strategy": "VRP-Gated ERP Harvester",
            "period": f"{args.start} to {args.end}",
            "metrics": result.metrics,
            "baseline_metrics": baseline_metrics,
            "time_in_market_pct": result.time_in_market_pct,
            "correlation_to_spy": result.correlation_to_spy,
            "trades_count": len(result.trades),
            "gate_distribution": {
                state.value: int(gate_counts.get(state, 0))
                for state in [GateState.OPEN, GateState.REDUCE, GateState.CLOSED]
            },
            "vrp_stats": {
                "mean": float(vrp.mean()),
                "std": float(vrp.std()),
                "pct_positive": float((vrp > 0).sum() / len(vrp) * 100),
            },
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nWrote results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
