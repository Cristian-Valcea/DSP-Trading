"""
ETF Carry Sleeve Backtester (FX + Rates Proxies)

Implements the pre-registered strategy in:
  dsp100k/docs/SPEC_SLEEVE_CARRY.md

Key points:
- Weekly rebalance using carry differentials (foreign 3M rate - US 3M rate)
- FX carry basket: long top-2, short bottom-2 (FXE/FXY/FXB/FXA)
- Rates anchor: 25% IEF + 25% SHY (long-only)
- Optional VRP regime gate throttles FX basket only (OPEN/REDUCE/CLOSED)
- Costs: 5 bps slippage per side + $0.005/share commission
- Borrow costs ignored (v1.0), shorts modeled mechanically

Data inputs (local parquet files):
- ETF daily bars (Polygon): data/carry/equities/{SYMBOL}_daily.parquet
- Carry rates (FRED, daily forward-filled): data/carry/rates/carry_rates.parquet
- US 3M rate (FRED): data/vrp/rates/tbill_3m.parquet
- VRP gate inputs (optional):
  - data/vrp/indices/VIX_spot.parquet
  - data/vrp/indices/VVIX.parquet
  - data/vrp/futures/vx_f1.parquet
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..regime.vrp_regime_gate import GateState, VRPRegimeGate
from ..utils.time import MarketCalendar


FX_SYMBOLS = ["FXE", "FXY", "FXB", "FXA"]
RATES_SYMBOLS = ["SHY", "IEF", "TLT"]
ALL_SYMBOLS = sorted(set(FX_SYMBOLS + RATES_SYMBOLS + ["UUP"]))

# Mapping of FX ETF -> rate column in carry_rates.parquet
FX_RATE_COL = {
    "FXE": "eur_3m",
    "FXY": "jpy_3m",
    "FXB": "gbp_3m",
    "FXA": "aud_3m",
}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


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
        }

    rets = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    vol = float(rets.std() * sqrt(252)) if rets.std() and rets.std() > 0 else 0.0
    sharpe = float((rets.mean() / rets.std()) * sqrt(252)) if rets.std() and rets.std() > 0 else 0.0
    max_dd = _max_drawdown(equity)

    return {
        "return": total_return,
        "cagr": float(cagr),
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": float(max_dd),
    }


@dataclass(frozen=True)
class Trade:
    dt: date
    symbol: str
    quantity: int  # signed shares
    price: float
    commission: float


@dataclass(frozen=True)
class CarryConfig:
    # Allocations (fractions of sleeve NAV)
    fx_gross: float = 0.50  # FX carry basket gross allocation
    shy_alloc: float = 0.25
    ief_alloc: float = 0.25

    # FX basket construction
    fx_long_k: int = 2
    fx_short_k: int = 2

    # Costs
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005

    # Gate behavior (FX only)
    use_gate: bool = False

    # Gate scaling (pre-registered)
    fx_mult_open: float = 1.0
    fx_mult_reduce: float = 0.5
    fx_mult_closed: float = 0.0


class ETFCarrBacktester:
    def __init__(
        self,
        *,
        prices_dir: Path,
        carry_rates_path: Path,
        us_rate_path: Path,
        vix_path: Optional[Path] = None,
        vvix_path: Optional[Path] = None,
        vx_f1_path: Optional[Path] = None,
        config: Optional[CarryConfig] = None,
    ):
        self.prices_dir = prices_dir
        self.carry_rates_path = carry_rates_path
        self.us_rate_path = us_rate_path
        self.config = config or CarryConfig()
        self.calendar = MarketCalendar()

        self.vix_path = vix_path
        self.vvix_path = vvix_path
        self.vx_f1_path = vx_f1_path

        # Load data
        self.prices = self._load_prices()
        self.carry_rates = self._load_carry_rates()
        self.us_rate = self._load_us_rate()

        self.gate: Optional[VRPRegimeGate] = VRPRegimeGate() if self.config.use_gate else None
        self.vix = self._load_optional_series(self.vix_path, expected_col=None)
        self.vvix = self._load_optional_series(self.vvix_path, expected_col=None)
        self.vx_f1 = self._load_optional_series(self.vx_f1_path, expected_col="vx_f1")

        self._align_indices()

    def _load_prices(self) -> Dict[str, pd.DataFrame]:
        prices: Dict[str, pd.DataFrame] = {}

        missing: List[str] = []
        for sym in ALL_SYMBOLS:
            path = self.prices_dir / f"{sym}_daily.parquet"
            if not path.exists():
                # Allow re-use of VRP equities folder for some symbols (SPY/QQQ/TLT were stored there)
                fallback = Path("dsp100k/data/vrp/equities") / f"{sym}_daily.parquet"
                if fallback.exists():
                    path = fallback
                else:
                    missing.append(sym)
                    continue

            df = pd.read_parquet(path).copy()
            # Normalize index to date
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
                df.index = df.index.normalize()
            df = df.sort_index()
            required_cols = {"open", "close"}
            if not required_cols.issubset(set(df.columns)):
                raise ValueError(f"{path} missing required cols: {required_cols - set(df.columns)}")
            prices[sym] = df

        if missing:
            raise RuntimeError(
                "Missing ETF price files for: "
                + ", ".join(missing)
                + f". Fetch via Polygon into {self.prices_dir} using "
                + "`python -m dsp.data.equity_daily_fetcher --symbols ... --output-dir ...`"
            )

        return prices

    def _load_carry_rates(self) -> pd.DataFrame:
        df = pd.read_parquet(self.carry_rates_path).copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
            df.index = df.index.normalize()
        df = df.sort_index()
        return df

    def _load_us_rate(self) -> pd.Series:
        df = pd.read_parquet(self.us_rate_path).copy()
        if "tbill_3m" not in df.columns:
            raise ValueError(f"US rate file missing tbill_3m column: {self.us_rate_path}")
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
            df.index = df.index.normalize()
        s = df["tbill_3m"].sort_index().ffill()
        return s

    def _load_optional_series(self, path: Optional[Path], expected_col: Optional[str]) -> Optional[pd.Series]:
        if path is None:
            return None
        if not path.exists():
            return None
        df = pd.read_parquet(path).copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
            df.index = df.index.normalize()
        df = df.sort_index()
        if expected_col is None:
            # First column (VIX/VVIX typically stored as single-column parquet)
            if df.shape[1] < 1:
                return None
            return df.iloc[:, 0].astype(float).ffill()
        if expected_col not in df.columns:
            return None
        return df[expected_col].astype(float).ffill()

    def _align_indices(self) -> None:
        # Align carry rates and US rates (calendar days); prices are trading days.
        # We use forward-fill and then lookup by signal date.
        self.carry_rates = self.carry_rates.ffill()
        self.us_rate = self.us_rate.ffill()

    def _weekly_rebalance_days(self, start: date, end: date) -> List[date]:
        days = self.calendar.get_trading_days(start, end)
        rebals: List[date] = []
        prev_key: Optional[Tuple[int, int]] = None
        for dt in days:
            iso = dt.isocalendar()
            key = (iso.year, iso.week)
            if key != prev_key:
                rebals.append(dt)
                prev_key = key
        return rebals

    def _gate_multiplier(self, as_of: date) -> float:
        if not self.config.use_gate or self.gate is None:
            return self.config.fx_mult_open

        if self.vix is None or self.vvix is None or self.vx_f1 is None:
            # If gate enabled but data missing, fail closed for safety
            return self.config.fx_mult_closed

        ts = pd.Timestamp(as_of)
        if ts not in self.vix.index or ts not in self.vvix.index or ts not in self.vx_f1.index:
            return self.config.fx_mult_closed

        state = self.gate.update(
            vix=float(self.vix.loc[ts]),
            vvix=float(self.vvix.loc[ts]),
            vx_f1=float(self.vx_f1.loc[ts]),
            as_of_date=as_of,
        )
        if state == GateState.OPEN:
            return self.config.fx_mult_open
        if state == GateState.REDUCE:
            return self.config.fx_mult_reduce
        return self.config.fx_mult_closed

    def _carry_diffs(self, signal_date: date) -> Dict[str, float]:
        ts = pd.Timestamp(signal_date)
        if ts not in self.carry_rates.index:
            # Use last available prior day (carry_rates is daily so this is unlikely)
            ts = self.carry_rates.index[self.carry_rates.index <= ts].max()
        us = float(self.us_rate.loc[:ts].iloc[-1])
        diffs: Dict[str, float] = {}
        for sym in FX_SYMBOLS:
            col = FX_RATE_COL[sym]
            foreign = float(self.carry_rates.loc[ts, col])
            diffs[sym] = foreign - us
        return diffs

    def run(
        self,
        *,
        start: date,
        end: date,
        initial_nlv: float = 100_000.0,
    ) -> Dict:
        days = self.calendar.get_trading_days(start, end)
        if len(days) < 50:
            raise ValueError("Backtest window too short")

        rebals = set(self._weekly_rebalance_days(start, end))

        cash = float(initial_nlv)
        pos: Dict[str, int] = {s: 0 for s in ALL_SYMBOLS}
        trades: List[Trade] = []
        equity_points: List[Tuple[date, float]] = []

        slip = self.config.slippage_bps / 10_000.0

        # Map for quick previous trading day lookup
        prev_day: Dict[date, Optional[date]] = {}
        last: Optional[date] = None
        for d in days:
            prev_day[d] = last
            last = d

        def _price(sym: str, d: date, field: str) -> float:
            ts = pd.Timestamp(d)
            df = self.prices[sym]
            if ts not in df.index:
                ts = df.index[df.index <= ts].max()
            return float(df.loc[ts, field])

        for d in days:
            # Rebalance at open on rebalance day, using previous trading day's rates as signal
            if d in rebals:
                sig = prev_day.get(d)
                if sig is None:
                    # Need a prior day to avoid lookahead
                    pass
                else:
                    nav_prev_close = cash + sum(pos[s] * _price(s, sig, "close") for s in pos.keys())
                    fx_mult = self._gate_multiplier(sig)
                    diffs = self._carry_diffs(sig)

                    ranked = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)
                    longs = [s for s, _ in ranked[: self.config.fx_long_k]]
                    shorts = [s for s, _ in ranked[-self.config.fx_short_k :]]

                    targets: Dict[str, int] = {s: 0 for s in ALL_SYMBOLS}

                    # Rates anchor (long-only)
                    shy_target_usd = nav_prev_close * self.config.shy_alloc
                    ief_target_usd = nav_prev_close * self.config.ief_alloc
                    targets["SHY"] = int(shy_target_usd / _price("SHY", d, "open"))
                    targets["IEF"] = int(ief_target_usd / _price("IEF", d, "open"))

                    # FX basket (long/short)
                    fx_budget = nav_prev_close * self.config.fx_gross * fx_mult
                    per_leg = fx_budget / (self.config.fx_long_k + self.config.fx_short_k)

                    for s in longs:
                        targets[s] = int(per_leg / _price(s, d, "open"))
                    for s in shorts:
                        targets[s] = -int(per_leg / _price(s, d, "open"))

                    # Execute deltas at open with slippage + commissions
                    for s, tgt in targets.items():
                        delta = tgt - pos[s]
                        if delta == 0:
                            continue
                        px_open = _price(s, d, "open")
                        exec_px = px_open * (1.0 + slip) if delta > 0 else px_open * (1.0 - slip)
                        commission = abs(delta) * self.config.commission_per_share
                        cash -= float(delta) * float(exec_px)
                        cash -= float(commission)
                        pos[s] = tgt
                        trades.append(Trade(dt=d, symbol=s, quantity=delta, price=float(exec_px), commission=float(commission)))

            # End-of-day equity mark
            equity = cash + sum(pos[s] * _price(s, d, "close") for s in pos.keys())
            equity_points.append((d, float(equity)))

        equity = pd.Series({pd.Timestamp(d): v for d, v in equity_points}).sort_index()
        metrics = _perf_metrics(equity)

        trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
            columns=["dt", "symbol", "quantity", "price", "commission"]
        )

        return {
            "equity": equity,
            "trades": trades_df,
            "metrics": metrics,
        }


def _run_kill_test(
    backtester: ETFCarrBacktester,
    *,
    initial_nlv: float,
    folds: List[Tuple[date, date]],
    sharpe_threshold: float = 0.50,
    max_dd_threshold: float = -0.20,
) -> Dict:
    fold_results = []
    pass_count = 0

    for i, (fold_start, fold_end) in enumerate(folds, start=1):
        out = backtester.run(start=fold_start, end=fold_end, initial_nlv=initial_nlv)
        m = out["metrics"]
        net_pnl = float(out["equity"].iloc[-1] - out["equity"].iloc[0])
        passes = (net_pnl > 0) and (m["sharpe"] >= sharpe_threshold) and (m["max_dd"] >= max_dd_threshold)
        pass_count += int(passes)
        fold_results.append(
            {
                "fold_id": i,
                "test_period": f"{fold_start.isoformat()} to {fold_end.isoformat()}",
                "net_pnl": net_pnl,
                "sharpe": m["sharpe"],
                "cagr": m["cagr"],
                "max_dd": m["max_dd"],
                "passes": bool(passes),
            }
        )

    mean_sharpe = float(np.mean([f["sharpe"] for f in fold_results])) if fold_results else 0.0
    overall_pass = pass_count >= 2  # 2/3 folds

    return {
        "folds": fold_results,
        "summary": {
            "n_folds": len(fold_results),
            "n_pass": int(pass_count),
            "pass_rate": float(pass_count) / float(len(fold_results)) if fold_results else 0.0,
            "mean_sharpe": mean_sharpe,
            "passes_kill_test": bool(overall_pass),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF Carry backtest (SPEC_SLEEVE_CARRY v1.0).")
    p.add_argument("--prices-dir", default="dsp100k/data/carry/equities", help="Directory of *_daily.parquet files.")
    p.add_argument("--carry-rates", default="dsp100k/data/carry/rates/carry_rates.parquet")
    p.add_argument("--us-rate", default="dsp100k/data/vrp/rates/tbill_3m.parquet")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--initial-nlv", type=float, default=100_000.0)
    p.add_argument("--use-gate", action="store_true", help="Throttle FX basket with VRPRegimeGate.")
    p.add_argument("--output", default="dsp100k/data/carry/etf_carry_evaluation.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    cfg = CarryConfig(use_gate=bool(args.use_gate))
    bt = ETFCarrBacktester(
        prices_dir=Path(args.prices_dir),
        carry_rates_path=Path(args.carry_rates),
        us_rate_path=Path(args.us_rate),
        vix_path=Path("dsp100k/data/vrp/indices/VIX_spot.parquet"),
        vvix_path=Path("dsp100k/data/vrp/indices/VVIX.parquet"),
        vx_f1_path=Path("dsp100k/data/vrp/futures/vx_f1.parquet"),
        config=cfg,
    )

    # Default 3 yearly OOS folds (expanding-window philosophy; warmup handled by using prior dates for signal)
    folds = [
        (date(2022, 1, 1), date(2022, 12, 31)),
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
    ]

    out = _run_kill_test(bt, initial_nlv=args.initial_nlv, folds=folds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"Wrote evaluation: {args.output}")
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

