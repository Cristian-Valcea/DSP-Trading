#!/usr/bin/env python3
"""
DSP-100K Daily Digest (Markdown)

Writes a human-readable daily status snapshot to:
  dsp100k/data/daily_digest/digest_YYYY-MM-DD.md

Primary sources (local, if present):
- VRP-CS paper log: data/vrp/paper_trading/daily_log.csv
- VRP-ERP paper log: data/vrp/paper_trading/vrp_erp_log.csv
- Vol overlay state: data/vol_target_overlay_state.json (fallback: compute from data/vrp/equities/SPY_daily.parquet)
- Indices: data/vrp/indices/VIX_spot.parquet, data/vrp/indices/VVIX.parquet

Optional live enrichment (IBKR via ib_insync):
- Account summary + positions for DM holdings + SPY

Usage:
  cd dsp100k
  source ../venv/bin/activate
  python scripts/daily_digest.py
  python scripts/daily_digest.py --live
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "daily_digest"
VRP_CS_LOG = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "daily_log.csv"
VRP_ERP_LOG = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "vrp_erp_log.csv"
VOL_STATE = PROJECT_ROOT / "data" / "vol_target_overlay_state.json"
SPY_DAILY = PROJECT_ROOT / "data" / "vrp" / "equities" / "SPY_daily.parquet"
VIX_SPOT = PROJECT_ROOT / "data" / "vrp" / "indices" / "VIX_spot.parquet"
VVIX = PROJECT_ROOT / "data" / "vrp" / "indices" / "VVIX.parquet"


DM_SYMBOLS = ["EFA", "EEM", "GLD", "SHY"]  # paper DM holdings universe (minimal)


def _today() -> date:
    return datetime.now().date()


def _read_last_row_csv(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    return {str(k): "" if pd.isna(v) else str(v) for k, v in row.items()}


def _read_last_value_parquet(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df is None or df.empty:
        return None
    if isinstance(df, pd.Series):
        val = df.dropna().iloc[-1]
        return float(val)
    if isinstance(df, pd.DataFrame):
        s = df.iloc[:, 0].astype(float).dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    return None


def _compute_vol_mult_from_spy() -> Tuple[float, Optional[float]]:
    """
    Compute vol multiplier from local SPY daily parquet.
    Returns (multiplier, realized_vol) where realized_vol is annualized.
    """
    if not SPY_DAILY.exists():
        return 1.0, None
    df = pd.read_parquet(SPY_DAILY)
    if df is None or df.empty or "close" not in df.columns:
        return 1.0, None
    closes = df["close"].astype(float).dropna()
    if len(closes) < 30:
        return 1.0, None
    log_rets = np.log(closes / closes.shift(1)).dropna()
    lookback = log_rets.tail(21)
    if len(lookback) < 10:
        return 1.0, None
    realized_vol = float(lookback.std() * np.sqrt(252)) if lookback.std() and lookback.std() > 0 else 0.0
    if realized_vol <= 0:
        return 1.0, realized_vol
    raw = 0.10 / realized_vol
    mult = float(min(1.50, max(0.25, raw)))
    return mult, realized_vol


def _load_vol_state() -> Tuple[float, str, Optional[float]]:
    if VOL_STATE.exists():
        try:
            with open(VOL_STATE, "r", encoding="utf-8") as f:
                data = json.load(f)
            mult = float(data.get("last_multiplier", 1.0))
            return mult, "state_file", None
        except Exception:
            pass
    mult, realized = _compute_vol_mult_from_spy()
    return mult, "computed_spy", realized


@dataclass(frozen=True)
class IBKRSnapshot:
    nlv: Optional[float]
    cash: Optional[float]
    positions: Dict[str, Dict[str, float]]  # symbol -> {position, avg_cost, market_price, market_value}


def _ibkr_snapshot(host: str, port: int, client_id: int) -> IBKRSnapshot:
    try:
        from ib_insync import IB
    except Exception:
        return IBKRSnapshot(nlv=None, cash=None, positions={})

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    try:
        summary = ib.accountSummary()
        nlv = None
        cash = None
        for s in summary:
            if s.tag == "NetLiquidation":
                nlv = float(s.value)
            if s.tag in ("TotalCashValue", "TotalCashBalance"):
                cash = float(s.value)

        positions: Dict[str, Dict[str, float]] = {}
        for p in ib.positions():
            sym = getattr(p.contract, "symbol", "") or getattr(p.contract, "localSymbol", "")
            if not sym:
                continue
            # keep STK/ETF positions by symbol; keep VXM/VX by localSymbol prefix
            positions[sym] = {
                "position": float(p.position),
                "avg_cost": float(getattr(p, "avgCost", 0.0) or 0.0),
            }
        return IBKRSnapshot(nlv=nlv, cash=cash, positions=positions)
    finally:
        ib.disconnect()


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.2f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2%}"


def _digest_path(out_dir: Path, as_of: date) -> Path:
    return out_dir / f"digest_{as_of.isoformat()}.md"


def build_digest(
    *,
    as_of: date,
    live: bool,
    host: str,
    port: int,
    client_id: int,
) -> str:
    lines: List[str] = []
    lines.append(f"# DSP-100K Daily Digest — {as_of.isoformat()}")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    # Vol overlay
    vol_mult, vol_src, realized_vol = _load_vol_state()
    lines.append("## Vol-Target Overlay")
    if vol_src == "state_file":
        lines.append(f"- Multiplier: `{vol_mult:.2f}` (from `data/vol_target_overlay_state.json`)")
    else:
        rv = f"{realized_vol:.1%}" if realized_vol is not None else "n/a"
        lines.append(f"- Multiplier: `{vol_mult:.2f}` (computed from SPY, realized_vol={rv})")
    lines.append("")

    # Indices snapshot (last close in parquet)
    vix = _read_last_value_parquet(VIX_SPOT)
    vvix = _read_last_value_parquet(VVIX)
    lines.append("## Vol Indices (last close)")
    lines.append(f"- VIX: `{vix:.2f}`" if vix is not None else "- VIX: `n/a`")
    lines.append(f"- VVIX: `{vvix:.2f}`" if vvix is not None else "- VVIX: `n/a`")
    lines.append("")

    # VRP-CS status (from paper log)
    vrp_cs = _read_last_row_csv(VRP_CS_LOG)
    lines.append("## Sleeve VRP-CS (Calendar Spread)")
    if vrp_cs is None:
        lines.append(f"- No log found at `{VRP_CS_LOG}`")
    else:
        # The monitor script logs key fields; display what exists.
        entry = vrp_cs.get("entry_spread", "") or vrp_cs.get("entry_spread_points", "")
        cur = vrp_cs.get("current_spread", "") or vrp_cs.get("current_spread_points", "")
        pnl = vrp_cs.get("pnl", "") or vrp_cs.get("pnl_usd", "")
        roll_by = vrp_cs.get("roll_by", "")
        gate = vrp_cs.get("gate_state", "")
        lines.append(f"- Entry spread: `{entry}` | Current spread: `{cur}` | PnL: `{pnl}`")
        if roll_by:
            lines.append(f"- Roll-by: `{roll_by}`")
        if gate:
            lines.append(f"- Gate: `{gate}`")
    lines.append("")

    # VRP-ERP status (from paper log)
    vrp_erp = _read_last_row_csv(VRP_ERP_LOG)
    lines.append("## Sleeve VRP-ERP (VIX-gated SPY)")
    if vrp_erp is None:
        lines.append(f"- No log found at `{VRP_ERP_LOG}`")
    else:
        lines.append(
            "- "
            + " | ".join(
                [
                    f"VIX `{vrp_erp.get('vix', 'n/a')}`",
                    f"Regime `{vrp_erp.get('regime', 'n/a')}`",
                    f"Target `{vrp_erp.get('target_shares', 'n/a')}`",
                    f"Actual `{vrp_erp.get('actual_shares', 'n/a')}`",
                    f"Drift `{vrp_erp.get('drift_pct', 'n/a')}`",
                ]
            )
        )
    lines.append("")

    # Live account snapshot (optional)
    if live:
        snap = _ibkr_snapshot(host, port, client_id)
        lines.append("## IBKR Snapshot (live)")
        lines.append(f"- Net Liquidation: `{_fmt_money(snap.nlv)}`")
        lines.append(f"- Total Cash: `{_fmt_money(snap.cash)}`")
        if snap.positions:
            lines.append("")
            lines.append("### Positions (selected)")
            # DM holdings + SPY + any VX/VXM contracts if present
            want = set(DM_SYMBOLS + ["SPY"])
            for sym in sorted(snap.positions.keys()):
                if sym in want or sym.startswith("VX") or sym.startswith("VXM"):
                    p = snap.positions[sym]
                    lines.append(f"- `{sym}`: pos `{p.get('position', 0.0)}` avg_cost `{p.get('avg_cost', 0.0):.2f}`")
        else:
            lines.append("- Positions: `n/a`")
        lines.append("")

    # Alerts (lightweight heuristics)
    lines.append("## Alerts")
    alerts: List[str] = []
    if vrp_cs is not None:
        roll_by = vrp_cs.get("roll_by", "")
        if roll_by:
            try:
                rb = date.fromisoformat(roll_by)
                days = (rb - as_of).days
                if days <= 1:
                    alerts.append(f"VRP-CS roll-by is in `{days}` day(s): `{roll_by}`")
            except Exception:
                pass
    if vrp_erp is not None:
        try:
            drift_str = (vrp_erp.get("drift_pct") or "").replace("%", "")
            drift = float(drift_str) / 100.0 if drift_str else None
            if drift is not None and drift >= 0.05:
                alerts.append(f"VRP-ERP drift >= 5%: `{vrp_erp.get('drift_pct')}`")
        except Exception:
            pass
    if not alerts:
        lines.append("- None")
    else:
        for a in alerts:
            lines.append(f"- ⚠️ {a}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="DSP-100K daily digest (markdown).")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--no-write", action="store_true", help="Print only; do not write file")
    p.add_argument("--live", action="store_true", help="Enrich with IBKR snapshot via ib_insync")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=901)
    args = p.parse_args()

    as_of = _today() if not args.date else date.fromisoformat(args.date)
    out_dir = Path(args.out_dir)

    digest = build_digest(
        as_of=as_of,
        live=bool(args.live),
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )

    if args.no_write:
        print(digest)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    path = _digest_path(out_dir, as_of)
    path.write_text(digest, encoding="utf-8")
    print(f"Wrote digest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

