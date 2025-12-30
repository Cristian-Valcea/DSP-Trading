"""
Sleeve DM: Asset-Class Dual Momentum (ETF-only).

Implements a Gary Antonacci-style "Dual Momentum" ETF rotation:
- Universe: configurable risky ETFs + a cash ETF
- Signal: 12-1 momentum (12-month return skipping most recent month)
- Selection: top K assets with positive momentum; otherwise go to cash
- Rebalance: monthly (first NYSE trading day of the month)
- Vol targeting: conservative estimate (sum(|w| * vol_i)), with max leverage cap

This sleeve is designed to be:
- ETF-only (no survivorship bias like single stocks)
- Operationally simple (monthly trades)
- Reproducible vs the backtest module in `dsp.backtest.etf_dual_momentum`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from math import sqrt
from typing import Dict, List, Optional

import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.config import SleeveDMConfig
from ..utils.logging import get_audit_logger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DualMomentumSignal:
    symbol: str
    momentum_12_1: float
    selected: bool
    as_of_date: date  # signal date (previous close)


@dataclass(frozen=True)
class DualMomentumPositionTarget:
    symbol: str
    weight: float
    price_used: float
    target_shares: int
    target_notional: float
    current_shares: int
    delta_shares: int


@dataclass(frozen=True)
class SleeveDMAdjustment:
    execution_date: date
    signal_date: date
    sleeve_nav: float
    top_k: int
    vol_target: float
    max_leverage: float
    signals: List[DualMomentumSignal]
    targets: List[DualMomentumPositionTarget]
    estimated_turnover: float
    rebalance_needed: bool


class SleeveDM:
    """
    Sleeve DM (Dual Momentum) implementation.
    """

    def __init__(self, config: SleeveDMConfig, data_fetcher: DataFetcher):
        self.config = config
        self.fetcher = data_fetcher
        self.audit = get_audit_logger()

        self.risky_universe = [s.upper() for s in (config.risky_universe or [])]
        self.cash_symbol = str(config.cash_symbol).upper()
        self.symbols = list(dict.fromkeys(self.risky_universe + [self.cash_symbol]))

        # Current positions (symbol -> shares)
        self._positions: Dict[str, int] = {s: 0 for s in self.symbols}
        self._last_rebalance: Optional[date] = None

    def set_positions(self, positions: Dict[str, int]) -> None:
        self._positions = {s: int(positions.get(s, 0)) for s in self.symbols}

    def _is_first_trading_day_of_month(self, execution_date: date) -> bool:
        """
        True iff execution_date is the first NYSE trading day of the month.
        """
        try:
            prev_td = self.fetcher.calendar.get_previous_trading_day(execution_date)
        except Exception:
            return True
        return prev_td.month != execution_date.month

    async def generate_adjustment(
        self,
        *,
        sleeve_nav: float,
        prices: Dict[str, float],
        as_of_date: Optional[date] = None,
        force_rebalance: bool = False,
    ) -> SleeveDMAdjustment:
        """
        Generate the monthly rebalance adjustment.

        Notes on dates (matching the orchestrator/backtests):
        - `as_of_date` is the latest *complete* session (signal date, t-1 close).
        - Execution happens on the next trading day (t open) during the configured window.
        """
        if as_of_date is None:
            as_of_date = self.fetcher.calendar.get_latest_complete_session()

        if not self.risky_universe:
            raise ValueError("Sleeve DM risky_universe is empty")

        signal_date = as_of_date
        execution_date = self.fetcher.calendar.get_next_trading_day(signal_date)

        rebalance_needed = force_rebalance or self._is_first_trading_day_of_month(execution_date)

        # Fetch daily bars needed for momentum + vol.
        lookback_days = int(self.config.lookback_days)
        bars: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = await self.fetcher.get_daily_bars(
                sym,
                lookback_days=lookback_days,
                end_date=signal_date,
                adjusted=True,
                use_cache=True,
                validate=True,
            )
            if df is None or df.empty:
                raise RuntimeError(f"Missing historical data for {sym}")
            bars[sym] = df

        def _momentum_12_1(sym: str) -> Optional[float]:
            series = bars[sym]["close"].loc[: pd.Timestamp(signal_date)].dropna()
            need = int(self.config.momentum_window_days) + int(self.config.momentum_skip_days)
            if len(series) < need:
                return None
            series = series.iloc[: -int(self.config.momentum_skip_days)]
            window = series.tail(int(self.config.momentum_window_days))
            if len(window) < int(self.config.momentum_window_days):
                return None
            start_px = float(window.iloc[0])
            end_px = float(window.iloc[-1])
            if start_px <= 0:
                return None
            return end_px / start_px - 1.0

        def _asset_vol(sym: str) -> Optional[float]:
            series = bars[sym]["close"].loc[: pd.Timestamp(signal_date)].dropna()
            if len(series) < int(self.config.vol_lookback_days) + 2:
                return None
            rets = series.pct_change().dropna().tail(int(self.config.vol_lookback_days))
            if len(rets) < 5:
                return None
            v = float(rets.std() * sqrt(252))
            return v if v > 0 else None

        # Momentum scores for risky assets.
        mom_scores: Dict[str, float] = {}
        for sym in self.risky_universe:
            m = _momentum_12_1(sym)
            if m is None:
                logger.warning("Insufficient history for %s momentum on %s", sym, signal_date)
                continue
            mom_scores[sym] = float(m)

        ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [sym for sym, m in ranked[: int(self.config.top_k)] if m > float(self.config.min_momentum)]

        # Base weights (pre-vol-target).
        weights: Dict[str, float] = {s: 0.0 for s in self.symbols}
        if not selected:
            weights[self.cash_symbol] = 1.0
        else:
            w0 = 1.0 / len(selected)
            for sym in selected:
                weights[sym] = w0

            # Conservative portfolio vol estimate: sum(|w| * vol_i) over risky holdings.
            vols: Dict[str, float] = {}
            for sym in selected:
                v = _asset_vol(sym)
                if v is not None:
                    vols[sym] = v

            port_vol = sum(abs(weights[sym]) * vols.get(sym, 0.0) for sym in selected)
            if port_vol > 0:
                scale = float(self.config.vol_target) / port_vol
                scale = min(scale, float(self.config.max_leverage))
                for sym in selected:
                    weights[sym] *= scale

            total_risky = sum(weights[sym] for sym in selected)
            # Remainder goes to cash ETF (never negative).
            if total_risky < 1.0:
                weights[self.cash_symbol] = 1.0 - total_risky

        # Build signal objects for diagnostics/audit.
        signals: List[DualMomentumSignal] = []
        for sym in self.risky_universe:
            m = mom_scores.get(sym, float("nan"))
            signals.append(
                DualMomentumSignal(
                    symbol=sym,
                    momentum_12_1=float(m) if pd.notna(m) else float("nan"),
                    selected=sym in selected,
                    as_of_date=signal_date,
                )
            )

        # Convert weights to share targets using current prices (fallback: last close).
        targets: List[DualMomentumPositionTarget] = []
        for sym in self.symbols:
            w = float(weights.get(sym, 0.0))
            px = float(prices.get(sym, 0.0) or 0.0)
            if px <= 0:
                # Fallback to last signal close for sizing (still no look-ahead).
                try:
                    px = float(bars[sym]["close"].iloc[-1])
                except Exception:
                    px = 0.0
            if px <= 0:
                raise ValueError(f"Missing price for {sym}")

            target_notional = float(sleeve_nav) * w
            target_shares = int(target_notional / px) if px > 0 else 0
            current_shares = int(self._positions.get(sym, 0))
            delta_shares = int(target_shares - current_shares)
            targets.append(
                DualMomentumPositionTarget(
                    symbol=sym,
                    weight=w,
                    price_used=float(px),
                    target_shares=target_shares,
                    target_notional=target_notional,
                    current_shares=current_shares,
                    delta_shares=delta_shares,
                )
            )

        # Estimated turnover (only meaningful when rebalance_needed, but compute regardless).
        # Turnover = sum(|delta_notional|) / (2 * sleeve_nav)
        total_delta_notional = 0.0
        for t in targets:
            px = float(prices.get(t.symbol, 0.0) or 0.0)
            if px <= 0:
                try:
                    px = float(bars[t.symbol]["close"].iloc[-1])
                except Exception:
                    px = 0.0
            total_delta_notional += abs(t.delta_shares) * px
        estimated_turnover = (total_delta_notional / (2.0 * sleeve_nav)) if sleeve_nav > 0 else 0.0

        # Audit logging
        for sig in signals:
            if pd.isna(sig.momentum_12_1):
                continue
            self.audit.log_signal(
                sleeve="DM",
                symbol=sig.symbol,
                signal=sig.momentum_12_1,
                score=1 if sig.selected else 0,
                reason=f"12-1 momentum={sig.momentum_12_1:.2%} selected={sig.selected}",
            )

        return SleeveDMAdjustment(
            execution_date=execution_date,
            signal_date=signal_date,
            sleeve_nav=float(sleeve_nav),
            top_k=int(self.config.top_k),
            vol_target=float(self.config.vol_target),
            max_leverage=float(self.config.max_leverage),
            signals=signals,
            targets=targets,
            estimated_turnover=float(estimated_turnover),
            rebalance_needed=bool(rebalance_needed),
        )

    def get_target_orders(
        self,
        adjustment: SleeveDMAdjustment,
        *,
        min_notional: Optional[float] = None,
    ) -> List[Dict]:
        """
        Convert adjustment to executable order dicts.
        """
        min_notional = float(min_notional) if min_notional is not None else float(self.config.min_order_notional)

        orders: List[Dict] = []
        for t in adjustment.targets:
            if t.delta_shares == 0:
                continue

            # Notional filter to avoid noise trades.
            notional = abs(t.delta_shares) * float(t.price_used)
            if notional < min_notional:
                continue

            orders.append(
                {
                    "symbol": t.symbol,
                    "side": "BUY" if t.delta_shares > 0 else "SELL",
                    "quantity": abs(int(t.delta_shares)),
                    "sleeve": "DM",
                    "reason": f"dual_momentum w={t.weight:.3f} signal_date={adjustment.signal_date}",
                }
            )

        return orders

    def confirm_rebalance(self, execution_date: date) -> None:
        self._last_rebalance = execution_date
        logger.info("Sleeve DM rebalance confirmed for %s", execution_date)
