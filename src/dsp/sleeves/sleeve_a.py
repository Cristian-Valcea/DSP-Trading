"""
Sleeve A: Equity Momentum (Long-Only) + SPY Hedge.

v1.0 design constraints (explicitly acknowledged):
- Universe is a fixed, versioned YAML mapping of ~100 large caps to sectors.
- No survivorship fix: any backtest is a negative filter only.
- No single-name shorts (borrow/HTB realism requires data we don't have yet).

Sleeve A is intended to be a practical "return engine" for small accounts:
- Long-only 12-1 momentum on a fixed large-cap universe
- Strict single-name and sector caps
- SPY short hedge to reduce (not neutralize) beta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..utils.config import SleeveAConfig
from ..utils.logging import get_audit_logger

logger = logging.getLogger(__name__)

SPY_SYMBOL = "SPY"


def _normalize_ibkr_symbol(symbol: str) -> str:
    """
    Normalize a symbol for IBKR contracts/position keys.

    Example:
    - BRK.B (Wikipedia / some feeds) → BRK B (IBKR)
    """
    return str(symbol).upper().strip().replace(".", " ")


@dataclass(frozen=True)
class UniverseMember:
    symbol: str  # canonical display symbol (e.g., "BRK.B")
    ibkr_symbol: str  # trading symbol used with IBKR (e.g., "BRK B")
    sector: str


@dataclass(frozen=True)
class MomentumSignal:
    symbol: str
    sector: str
    momentum_12_1: float
    price: float
    adv: float
    beta_to_spy: Optional[float]
    as_of_date: date


@dataclass(frozen=True)
class PositionTarget:
    symbol: str
    sec_type: str  # "STK" or "ETF"
    sector: str
    target_weight: float  # fraction of sleeve NAV (positive for longs, negative for short hedge)
    target_shares: int
    current_shares: int
    delta_shares: int
    reason: str

    @property
    def is_trade(self) -> bool:
        return self.delta_shares != 0


@dataclass(frozen=True)
class SleeveAAdjustment:
    signal_date: date
    execution_date: date
    rebalance_stocks: bool
    sleeve_nav: float
    selected: List[MomentumSignal]
    targets: List[PositionTarget]
    sizing_prices: Dict[str, float]
    estimated_turnover: float
    beta_long: float
    beta_net: float
    hedge_weight: float


class SleeveA:
    """
    Sleeve A (v1.0): Long-only 12-1 momentum + SPY hedge.

    Interface is intentionally similar to Sleeve B:
    - `set_positions()` from IBKR sync
    - `generate_adjustment()` returns an object that can be converted to orders
    """

    # Conservative defaults: long book uses vol cap only (no leverage).
    _MOM_SKIP_DAYS = 21
    _MOM_WINDOW_DAYS = 252
    _ADV_LOOKBACK_DAYS = 20
    _BETA_LOOKBACK_DAYS = 63
    _VOL_LOOKBACK_DAYS = 63

    def __init__(
        self,
        config: SleeveAConfig,
        data_fetcher: DataFetcher,
        universe: List[Dict],
    ):
        self.config = config
        self.fetcher = data_fetcher
        self.calendar = data_fetcher.calendar
        self.audit = get_audit_logger()

        self._universe: List[UniverseMember] = self._parse_universe(universe)
        self._universe_by_ibkr_symbol: Dict[str, UniverseMember] = {
            m.ibkr_symbol: m for m in self._universe
        }

        # Current positions (IBKR symbols → shares). Includes SPY hedge if present.
        self._positions: Dict[str, int] = {}

    @property
    def universe_symbols(self) -> List[str]:
        return [m.ibkr_symbol for m in self._universe]

    def set_positions(self, positions: Dict[str, int]) -> None:
        self._positions = {str(k).upper().strip(): int(v) for k, v in positions.items()}

    async def generate_adjustment(
        self,
        *,
        sleeve_nav: float,
        prices: Dict[str, float],
        as_of_date: Optional[date] = None,
        force_rebalance: bool = False,
    ) -> SleeveAAdjustment:
        if as_of_date is None:
            as_of_date = self.calendar.get_latest_complete_session()

        # Orchestrator uses latest complete session (signal date). Execution is next trading day.
        execution_date = self._infer_execution_date(as_of_date)

        if not self.config.enabled:
            return SleeveAAdjustment(
                signal_date=as_of_date,
                execution_date=execution_date,
                rebalance_stocks=False,
                sleeve_nav=float(sleeve_nav),
                selected=[],
                targets=[],
                sizing_prices={},
                estimated_turnover=0.0,
                beta_long=0.0,
                beta_net=0.0,
                hedge_weight=0.0,
            )

        if sleeve_nav <= 0:
            raise ValueError("sleeve_nav must be positive")

        rebalance_stocks = bool(force_rebalance or self._is_stock_rebalance_day(execution_date))

        # Decide stock targets.
        sizing_prices: Dict[str, float] = {str(k).upper().strip(): float(v) for k, v in (prices or {}).items()}
        if rebalance_stocks:
            selected_signals, long_weights = await self._build_long_targets(as_of_date)
            for sig in selected_signals:
                sizing_prices.setdefault(sig.symbol, float(sig.price))
        else:
            selected_signals = []
            # For hedge maintenance we need a best-effort price for current longs.
            held = [s for s, sh in self._positions.items() if sh > 0 and s != SPY_SYMBOL]
            for sym in held:
                if float(sizing_prices.get(sym, 0.0) or 0.0) > 0:
                    continue
                df = await self.fetcher.get_daily_bars(
                    sym,
                    lookback_days=30,
                    end_date=as_of_date,
                    adjusted=True,
                    use_cache=True,
                    validate=True,
                )
                if df is not None and not df.empty:
                    sizing_prices[sym] = float(df["close"].iloc[-1])

            long_weights = self._current_long_weights(sizing_prices, sleeve_nav)

        # Ensure we always have a SPY sizing price for hedge share computation.
        if float(sizing_prices.get(SPY_SYMBOL, 0.0) or 0.0) <= 0:
            spy_df = await self.fetcher.get_daily_bars(
                SPY_SYMBOL,
                lookback_days=30,
                end_date=as_of_date,
                adjusted=True,
                use_cache=True,
                validate=True,
            )
            if spy_df is not None and not spy_df.empty:
                sizing_prices[SPY_SYMBOL] = float(spy_df["close"].iloc[-1])

        # Hedge decision (based on the long book weights; on non-rebalance days this uses current positions).
        beta_long, hedge_weight = await self._compute_hedge_weight(as_of_date, long_weights)
        beta_net = beta_long - hedge_weight

        # Build share targets.
        targets = self._weights_to_targets(
            long_weights=long_weights,
            selected=selected_signals,
            hedge_weight=hedge_weight,
            sleeve_nav=float(sleeve_nav),
            prices=sizing_prices,
            rebalance_stocks=rebalance_stocks,
        )

        turnover = self._estimate_turnover(targets, sizing_prices, sleeve_nav=sleeve_nav)

        # Log signals for audit (only when we actually rebalance stock selection).
        if rebalance_stocks:
            for sig in selected_signals:
                self.audit.log_signal(
                    sleeve="A",
                    symbol=sig.symbol,
                    signal=sig.momentum_12_1,
                    score=sig.momentum_12_1,
                    reason=f"sector={sig.sector} price={sig.price:.2f} adv={sig.adv:,.0f}",
                )

        return SleeveAAdjustment(
            signal_date=as_of_date,
            execution_date=execution_date,
            rebalance_stocks=rebalance_stocks,
            sleeve_nav=float(sleeve_nav),
            selected=selected_signals,
            targets=targets,
            sizing_prices=sizing_prices,
            estimated_turnover=float(turnover),
            beta_long=float(beta_long),
            beta_net=float(beta_net),
            hedge_weight=float(hedge_weight),
        )

    def get_target_orders(
        self,
        adjustment: SleeveAAdjustment,
        *,
        min_notional: float = 100.0,
        prices: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Convert targets to executor-ready order dicts.

        `prices` is optional and only used to suppress tiny notional trades.
        """
        orders: List[Dict] = []
        px = prices or adjustment.sizing_prices or {}

        for t in adjustment.targets:
            if not t.is_trade:
                continue

            # Suppress tiny trades.
            price = float(px.get(t.symbol, 0.0) or 0.0)
            if price > 0:
                notional = abs(t.delta_shares) * price
                if notional < float(min_notional):
                    continue
                if adjustment.sleeve_nav > 0:
                    notional_pct = float(notional) / float(adjustment.sleeve_nav)
                    band = (
                        float(self.config.hedge_trade_band)
                        if t.symbol == SPY_SYMBOL
                        else float(self.config.trade_band)
                    )
                    if notional_pct < band:
                        continue

            side = "BUY" if t.delta_shares > 0 else "SELL"
            orders.append(
                {
                    "symbol": t.symbol,
                    "sec_type": t.sec_type,
                    "side": side,
                    "quantity": abs(int(t.delta_shares)),
                    "sleeve": "A",
                    "reason": t.reason,
                }
            )

        return orders

    # ---------------------------------------------------------------------
    # Universe / schedule
    # ---------------------------------------------------------------------

    def _parse_universe(self, universe: List[Dict]) -> List[UniverseMember]:
        members: List[UniverseMember] = []
        for entry in universe or []:
            symbol = str(entry.get("symbol") or "").strip()
            sector = str(entry.get("sector") or "").strip()
            if not symbol or not sector:
                continue
            ibkr_symbol = str(entry.get("ibkr_symbol") or _normalize_ibkr_symbol(symbol)).strip()
            members.append(UniverseMember(symbol=symbol, ibkr_symbol=ibkr_symbol, sector=sector))

        if not members:
            raise ValueError(
                "Sleeve A universe is empty. Provide config/universes/sleeve_a_universe.yaml "
                "with sleeve_a_universe.symbols entries."
            )

        # Deduplicate by ibkr_symbol (keep first occurrence).
        seen = set()
        deduped = []
        for m in members:
            key = m.ibkr_symbol
            if key in seen:
                continue
            seen.add(key)
            deduped.append(m)
        return deduped

    def _infer_execution_date(self, signal_date: date) -> date:
        """
        Translate the signal date (latest complete session) to the intended execution date.

        In the normal workflow, we trade during the next session open (09:35-10:15 ET).
        """
        try:
            return self.calendar.get_next_trading_day(signal_date)
        except Exception:
            # Best-effort fallback: if signal_date is already a trading day and next isn't available,
            # treat execution_date as signal_date.
            return signal_date

    def _is_stock_rebalance_day(self, execution_date: date) -> bool:
        """
        Monthly rebalance on the first trading day of each month.

        Deterministic definition: a day is the first trading day of a month if the previous
        trading day's month differs.
        """
        try:
            prev_td = self.calendar.get_previous_trading_day(execution_date)
        except Exception:
            return True
        return prev_td.month != execution_date.month

    # ---------------------------------------------------------------------
    # Signal building
    # ---------------------------------------------------------------------

    async def _build_long_targets(self, signal_date: date) -> Tuple[List[MomentumSignal], Dict[str, float]]:
        """
        Build the long book:
        - Filter by price and ADV
        - Rank by 12-1 momentum
        - Apply name cap, sector cap, and vol cap
        - Ensure beta is feasible under SPY hedge cap by dropping the lowest-momentum names
        """
        members = self._universe
        symbols = [m.ibkr_symbol for m in members]

        # Need 12M + skip-month + beta/vol windows with cushion.
        lookback_days = 650

        bars_by_symbol, _ = await self.fetcher.get_universe_bars(
            symbols=symbols + [SPY_SYMBOL],
            lookback_days=lookback_days,
            end_date=signal_date,
            adjusted=True,
            use_cache=True,
            validate=True,
            max_concurrent=5,
        )

        spy_bars = bars_by_symbol.get(SPY_SYMBOL)
        if spy_bars is None or spy_bars.empty:
            raise RuntimeError("Missing SPY history for beta/vol computation")

        rows: List[MomentumSignal] = []
        for m in members:
            df = bars_by_symbol.get(m.ibkr_symbol)
            if df is None or df.empty:
                continue

            price = float(df["close"].iloc[-1])
            adv = self._compute_adv(df, lookback=self._ADV_LOOKBACK_DAYS)
            mom = self._compute_momentum_12_1(df)
            if mom is None or adv is None:
                continue

            if price < float(self.config.min_price):
                continue
            if adv < float(self.config.min_adv):
                continue

            beta = self._compute_beta_to_spy(df, spy_bars, lookback=self._BETA_LOOKBACK_DAYS)
            rows.append(
                MomentumSignal(
                    symbol=m.ibkr_symbol,
                    sector=m.sector,
                    momentum_12_1=float(mom),
                    price=float(price),
                    adv=float(adv),
                    beta_to_spy=beta,
                    as_of_date=signal_date,
                )
            )

        if not rows:
            logger.warning("Sleeve A: no candidates pass filters on %s", signal_date)
            return [], {}

        # Rank by momentum (descending).
        rows.sort(key=lambda r: r.momentum_12_1, reverse=True)

        target_n = int(self.config.n_long)
        selected = rows[: max(1, min(target_n, len(rows)))]

        # Enforce beta feasibility: beta_long must be <= beta_limit + spy_cap.
        # If not, reduce gross by dropping the lowest momentum names.
        beta_limit = float(self.config.beta_limit)
        spy_cap = float(self.config.spy_hedge_cap)
        while True:
            long_weights = self._initial_equal_weights(selected)
            long_weights = self._apply_sector_cap(long_weights)
            long_weights = self._apply_vol_cap(long_weights, bars_by_symbol, lookback=self._VOL_LOOKBACK_DAYS)

            beta_long = await self._portfolio_beta(signal_date, long_weights, spy_bars, bars_by_symbol)
            if beta_long <= beta_limit + spy_cap + 1e-6:
                break
            if len(selected) <= 1:
                logger.warning(
                    "Sleeve A: beta infeasible even with 1 name (beta_long=%.3f > %.3f). "
                    "Will proceed with reduced beta hedge only.",
                    beta_long,
                    beta_limit + spy_cap,
                )
                break
            selected = selected[:-1]

        # Recompute final weights for the returned selection.
        long_weights = self._initial_equal_weights(selected)
        long_weights = self._apply_sector_cap(long_weights)
        long_weights = self._apply_vol_cap(long_weights, bars_by_symbol, lookback=self._VOL_LOOKBACK_DAYS)

        return selected, long_weights

    def _initial_equal_weights(self, selected: List[MomentumSignal]) -> Dict[str, float]:
        if not selected:
            return {}
        w = float(self.config.max_weight_per_name)
        return {s.symbol: w for s in selected}

    def _compute_momentum_12_1(self, df: pd.DataFrame) -> Optional[float]:
        closes = df["close"].dropna()
        if len(closes) < self._MOM_SKIP_DAYS + self._MOM_WINDOW_DAYS + 2:
            return None
        closes = closes.iloc[: -self._MOM_SKIP_DAYS]  # skip most recent month
        window = closes.tail(self._MOM_WINDOW_DAYS)
        if len(window) < self._MOM_WINDOW_DAYS:
            return None
        start_px = float(window.iloc[0])
        end_px = float(window.iloc[-1])
        if start_px <= 0:
            return None
        return end_px / start_px - 1.0

    def _compute_adv(self, df: pd.DataFrame, *, lookback: int) -> Optional[float]:
        if "volume" not in df.columns:
            return None
        sub = df[["close", "volume"]].dropna().tail(int(lookback))
        if len(sub) < max(5, int(lookback * 0.6)):
            return None
        dv = (sub["close"].astype(float) * sub["volume"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
        if dv.empty:
            return None
        return float(dv.mean())

    def _compute_beta_to_spy(self, stock_bars: pd.DataFrame, spy_bars: pd.DataFrame, *, lookback: int) -> Optional[float]:
        stock = stock_bars["close"].pct_change().dropna()
        spy = spy_bars["close"].pct_change().dropna()
        joined = pd.concat([stock.rename("s"), spy.rename("m")], axis=1).dropna().tail(int(lookback))
        if len(joined) < max(20, int(lookback * 0.6)):
            return None
        var_m = float(joined["m"].var())
        if var_m <= 0:
            return None
        cov = float(joined["s"].cov(joined["m"]))
        return float(cov / var_m)

    async def _portfolio_beta(
        self,
        signal_date: date,
        weights: Dict[str, float],
        spy_bars: pd.DataFrame,
        bars_by_symbol: Dict[str, pd.DataFrame],
    ) -> float:
        if not weights:
            return 0.0

        # Use precomputed per-name beta where possible; if missing, estimate quickly.
        betas: Dict[str, float] = {}
        for sym in weights:
            df = bars_by_symbol.get(sym)
            if df is None or df.empty:
                continue
            b = self._compute_beta_to_spy(df, spy_bars, lookback=self._BETA_LOOKBACK_DAYS)
            if b is None or np.isnan(b):
                continue
            betas[sym] = float(b)

        # Fallback: assume beta=1 for any missing name to avoid under-hedging.
        beta_long = 0.0
        for sym, w in weights.items():
            beta_long += float(w) * float(betas.get(sym, 1.0))
        return float(beta_long)

    def _apply_sector_cap(self, weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return {}
        cap = float(self.config.max_sector_gross)
        sector_to_symbols: Dict[str, List[str]] = {}
        for sym in weights:
            sector = (self._universe_by_ibkr_symbol.get(sym).sector if sym in self._universe_by_ibkr_symbol else "Unknown")
            sector_to_symbols.setdefault(sector, []).append(sym)

        capped = dict(weights)
        for sector, syms in sector_to_symbols.items():
            gross = sum(abs(capped[s]) for s in syms)
            if gross <= cap or gross <= 0:
                continue
            factor = cap / gross
            for s in syms:
                capped[s] *= factor
        return capped

    def _apply_vol_cap(
        self,
        weights: Dict[str, float],
        bars_by_symbol: Dict[str, pd.DataFrame],
        *,
        lookback: int,
    ) -> Dict[str, float]:
        if not weights:
            return {}

        # Build returns matrix for weighted portfolio.
        returns = {}
        for sym in weights:
            df = bars_by_symbol.get(sym)
            if df is None or df.empty:
                continue
            returns[sym] = df["close"].pct_change()
        if not returns:
            return dict(weights)

        rets = pd.DataFrame(returns).dropna(how="all").tail(int(lookback))
        if len(rets) < max(20, int(lookback * 0.6)):
            return dict(weights)

        w = pd.Series(weights).reindex(rets.columns).fillna(0.0)
        port = (rets.fillna(0.0) * w).sum(axis=1)
        vol = float(port.std() * sqrt(252)) if port.std() and port.std() > 0 else 0.0
        cap = float(self.config.vol_target)
        if vol <= 0 or vol <= cap:
            return dict(weights)

        scale = cap / vol
        return {s: float(wt) * float(scale) for s, wt in weights.items()}

    # ---------------------------------------------------------------------
    # Hedge sizing
    # ---------------------------------------------------------------------

    async def _compute_hedge_weight(self, signal_date: date, long_weights: Dict[str, float]) -> Tuple[float, float]:
        """
        Compute long beta and the SPY short weight to reduce beta to <= beta_limit,
        subject to spy_hedge_cap.
        """
        if not long_weights:
            return 0.0, 0.0

        # Fetch SPY history only (per-name betas computed from cached bars inside selection path).
        spy_bars = await self.fetcher.get_daily_bars(
            SPY_SYMBOL,
            lookback_days=400,
            end_date=signal_date,
            adjusted=True,
            use_cache=True,
            validate=True,
        )
        if spy_bars is None or spy_bars.empty:
            raise RuntimeError("Missing SPY history for hedge sizing")

        # Estimate portfolio beta as sum(w_i * beta_i). Missing betas default to 1.0.
        # We only have weights here, so compute betas on-demand for the held names.
        beta_long = 0.0
        for sym, w in long_weights.items():
            df = await self.fetcher.get_daily_bars(
                sym,
                lookback_days=400,
                end_date=signal_date,
                adjusted=True,
                use_cache=True,
                validate=True,
            )
            if df is None or df.empty:
                beta = 1.0
            else:
                b = self._compute_beta_to_spy(df, spy_bars, lookback=self._BETA_LOOKBACK_DAYS)
                beta = float(b) if b is not None and not np.isnan(b) else 1.0
            beta_long += float(w) * float(beta)

        beta_limit = float(self.config.beta_limit)
        spy_cap = float(self.config.spy_hedge_cap)
        required = max(0.0, beta_long - beta_limit)
        hedge_weight = min(spy_cap, required)

        return float(beta_long), float(hedge_weight)

    def _current_long_weights(self, prices: Dict[str, float], sleeve_nav: float) -> Dict[str, float]:
        """
        Compute current long weights from positions (excluding SPY hedge).
        """
        weights: Dict[str, float] = {}
        if sleeve_nav <= 0:
            return weights
        for sym, shares in self._positions.items():
            if shares <= 0:
                continue
            if sym == SPY_SYMBOL:
                continue
            px = float(prices.get(sym, 0.0) or 0.0)
            if px <= 0:
                continue
            weights[sym] = (shares * px) / float(sleeve_nav)
        return weights

    # ---------------------------------------------------------------------
    # Orders / targets
    # ---------------------------------------------------------------------

    def _weights_to_targets(
        self,
        *,
        long_weights: Dict[str, float],
        selected: List[MomentumSignal],
        hedge_weight: float,
        sleeve_nav: float,
        prices: Dict[str, float],
        rebalance_stocks: bool,
    ) -> List[PositionTarget]:
        """
        Translate desired weights into share targets (including SPY hedge).
        """
        targets: Dict[str, int] = {}

        # Stock targets (long-only).
        if rebalance_stocks:
            for sym, w in long_weights.items():
                px = float(prices.get(sym, 0.0) or 0.0)
                if px <= 0:
                    continue
                targets[sym] = int((sleeve_nav * float(w)) / px)
        else:
            # No stock rebalance: keep current positions.
            for sym, sh in self._positions.items():
                if sym == SPY_SYMBOL:
                    continue
                if sh != 0:
                    targets[sym] = int(sh)

        # SPY hedge target (short).
        spy_px = float(prices.get(SPY_SYMBOL, 0.0) or 0.0)
        if spy_px > 0 and hedge_weight > 0:
            hedge_shares = int((sleeve_nav * float(hedge_weight)) / spy_px)
            targets[SPY_SYMBOL] = -hedge_shares
        else:
            # No hedge desired → flat SPY (unless position already exists; then flatten).
            if self._positions.get(SPY_SYMBOL, 0) != 0:
                targets[SPY_SYMBOL] = 0

        # Build PositionTarget list including exits.
        universe_syms = set(self.universe_symbols)
        selected_syms = {s.symbol for s in selected}

        all_syms = set(targets.keys()) | set(self._positions.keys())
        # Only manage:
        # - universe stocks + any current holdings among them
        # - SPY hedge
        filtered = set()
        for sym in all_syms:
            if sym == SPY_SYMBOL:
                filtered.add(sym)
            elif sym in universe_syms:
                filtered.add(sym)

        out: List[PositionTarget] = []
        for sym in sorted(filtered):
            target_shares = int(targets.get(sym, 0))
            current_shares = int(self._positions.get(sym, 0))
            delta = int(target_shares - current_shares)
            if sym == SPY_SYMBOL:
                sector = "HEDGE"
                sec_type = "ETF"
                reason = f"beta_hedge weight={hedge_weight:.3f}"
                weight = -float(hedge_weight) if target_shares < 0 else 0.0
            else:
                member = self._universe_by_ibkr_symbol.get(sym)
                sector = member.sector if member else "Unknown"
                sec_type = "STK"
                weight = float(long_weights.get(sym, 0.0))
                if rebalance_stocks:
                    mom = next((s.momentum_12_1 for s in selected if s.symbol == sym), None)
                    reason = f"12_1_mom={mom:.4f}" if mom is not None else "exit_not_selected"
                else:
                    reason = "hold (no monthly rebalance)"

            out.append(
                PositionTarget(
                    symbol=sym,
                    sec_type=sec_type,
                    sector=sector,
                    target_weight=float(weight),
                    target_shares=target_shares,
                    current_shares=current_shares,
                    delta_shares=delta,
                    reason=reason,
                )
            )

        return out

    def _estimate_turnover(self, targets: List[PositionTarget], prices: Dict[str, float], *, sleeve_nav: float) -> float:
        if sleeve_nav <= 0:
            return 0.0
        total_delta = 0.0
        for t in targets:
            if t.delta_shares == 0:
                continue
            px = float(prices.get(t.symbol, 0.0) or 0.0)
            if px <= 0:
                continue
            total_delta += abs(t.delta_shares) * px
        return float(total_delta) / (2.0 * float(sleeve_nav))


__all__ = [
    "SleeveA",
    "UniverseMember",
    "MomentumSignal",
    "PositionTarget",
    "SleeveAAdjustment",
]
