#!/usr/bin/env python3
"""
Sleeve C (Tail Hedge) daily monitor + trade suggestion.

This script is intentionally "ops-first":
- It does NOT place option orders (DSP executor is STK/ETF only).
- It produces a concrete, deterministic recommendation for the next SPY put spread
  (expiry, strikes, qty, max debit) using live IBKR option greeks/quotes.
- It detects existing SPY option legs (if any) to flag roll timing (DTE <= roll trigger).

Usage:
  source venv/bin/activate
  python scripts/sleeve_c_daily_monitor.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import date, datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsp.ibkr import IBKRClient
from dsp.sleeves.sleeve_c import PutSpreadManager
from dsp.utils.config import load_config
from dsp.utils.time import get_ny_time


def _parse_yyyymmdd(s: str) -> Optional[date]:
    if not s:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date()
    except ValueError:
        return None


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_csv_row(path: Path, row: Dict) -> None:
    _ensure_parent_dir(path)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _detect_put_spread_legs(
    positions_raw: List[Dict],
    underlying: str,
) -> List[Dict]:
    """
    Return a simplified list of SPY option legs (puts only), suitable for human inspection.
    We don't attempt perfect pairing into spreads because IBKR may report partial legs,
    and pairing logic can hide issues.
    """
    legs: List[Dict] = []
    for p in positions_raw:
        if p.get("secType") != "OPT":
            continue
        if p.get("symbol") != underlying:
            continue
        if p.get("right") != "P":
            continue
        qty = float(p.get("position") or 0.0)
        if abs(qty) < 1e-9:
            continue
        expiry = _parse_yyyymmdd(str(p.get("lastTradeDateOrContractMonth") or ""))
        legs.append({
            "localSymbol": p.get("localSymbol") or "",
            "expiry": str(expiry) if expiry else "",
            "strike": float(p.get("strike") or 0.0),
            "qty": qty,
            "avgCost": float(p.get("avgCost") or 0.0),
        })
    legs.sort(key=lambda x: (x["expiry"], x["strike"], x["qty"]))
    return legs


async def _run_live(args: argparse.Namespace) -> int:
    cfg = load_config(strict=False)
    sleeve_c = cfg.sleeve_c
    now_ny = get_ny_time()
    today = now_ny.date()

    ib = IBKRClient(host=cfg.ibkr.host, port=cfg.ibkr.port, client_id=cfg.ibkr.client_id)
    if not await ib.connect():
        print("ERROR: Could not connect to IBKR. Is TWS/Gateway running?")
        return 2

    try:
        acct = await ib.get_account_summary()
        nav = float(args.nav) if args.nav is not None else float(acct.nlv)

        # Find candidate spread (expiry + strikes + quote-based premium estimate).
        mgr = PutSpreadManager(ib)
        target = await mgr.find_strikes_by_delta(
            underlying=sleeve_c.underlying,
            target_long_delta=sleeve_c.long_delta_target,
            target_short_delta=sleeve_c.short_delta_target,
            min_dte=sleeve_c.target_dte_min,
            max_dte=sleeve_c.target_dte_max,
        )

        # Positions (raw, so options don't collide under symbol="SPY").
        positions_raw = await ib.get_positions_raw()
        put_legs = _detect_put_spread_legs(positions_raw, sleeve_c.underlying)

        # Budgeting: simple "monthly slice" rule; match SleeveC.calculate_contracts behavior.
        annual_budget_usd = nav * sleeve_c.annual_budget_pct
        monthly_budget_usd = annual_budget_usd / 12.0
        # Heuristic: at most 2 months of premium per trade (matches SleeveC logic).
        max_trade_budget_usd = monthly_budget_usd * 2.0

        # Estimate debit per spread (USD).
        debit_mid = None
        if target is not None and target.estimated_premium > 0:
            debit_mid = target.estimated_premium
        debit_mid_usd = float(debit_mid or 0.0) * 100.0

        contracts = 0
        if debit_mid_usd > 0:
            contracts = int(max_trade_budget_usd // debit_mid_usd)
        contracts = max(0, min(int(sleeve_c.max_spreads), contracts))

        # Recommended limit debit: small buffer above mid to improve fill probability.
        # IMPORTANT: This is guidance for a TWS combo order, not a claim of best execution.
        limit_debit = None
        if debit_mid and debit_mid > 0:
            limit_debit = round(debit_mid * 1.05, 2)

        # Roll / DTE info (best-effort; for each expiry, use earliest).
        min_dte = None
        if put_legs:
            expiries = [date.fromisoformat(l["expiry"]) for l in put_legs if l["expiry"]]
            if expiries:
                min_dte = min((e - today).days for e in expiries)

        needs_roll = (min_dte is not None) and (min_dte <= sleeve_c.roll_dte_trigger)

        print("SLEEVE C (TAIL HEDGE) — DAILY MONITOR")
        print(f"As of (NY): {now_ny.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Underlying: {sleeve_c.underlying}")
        print(f"NAV used: ${nav:,.2f}")
        print(f"Annual budget: {sleeve_c.annual_budget_pct*100:.2f}%  (${annual_budget_usd:,.2f}/year)")
        print(f"Monthly budget: ${monthly_budget_usd:,.2f} (cap per trade ≈ ${max_trade_budget_usd:,.2f})")
        print("")

        if put_legs:
            print("Current SPY put option legs (raw):")
            for leg in put_legs:
                print(f"  {leg['expiry']}  P {leg['strike']:.0f}  qty={leg['qty']:+.0f}  avgCost={leg['avgCost']:.2f}  {leg['localSymbol']}")
            if min_dte is not None:
                print(f"Min DTE across legs: {min_dte}  (roll trigger: <= {sleeve_c.roll_dte_trigger})  needs_roll={needs_roll}")
            print("")
        else:
            print("Current SPY put option legs: NONE detected (no hedge on).")
            print("")

        if target is None:
            print("ERROR: Could not construct a candidate spread from IBKR option data.")
            print("Most common causes: missing OPRA options market data (IBKR error 354), or running outside liquid hours.")
            return 3

        print("Suggested next spread (for manual TWS execution):")
        print(f"  Expiry: {target.expiry}  (DTE={(target.expiry - today).days})")
        print(f"  BUY  1x {sleeve_c.underlying} {target.expiry} P {target.long_strike:.0f}  (target delta {sleeve_c.long_delta_target})")
        print(f"  SELL 1x {sleeve_c.underlying} {target.expiry} P {target.short_strike:.0f} (target delta {sleeve_c.short_delta_target})")
        print(f"  Mid debit (est): ${target.estimated_premium:.2f}  (= ${debit_mid_usd:,.2f} per spread)")
        if limit_debit is not None:
            print(f"  Limit debit (suggested): <= ${limit_debit:.2f}")
        print(f"  Suggested qty (budget + cap): {contracts} spreads  (max_spreads={sleeve_c.max_spreads})")
        print("")
        print("Execution note: enter as a single COMBO/vertical spread order in TWS (atomic fill).")
        print("Verification: after fill, re-run this script to confirm legs + DTE + roll flags.")

        # Log row
        log_path = Path(args.log_path)
        row = {
            "date": str(today),
            "time_ny": now_ny.strftime("%H:%M:%S"),
            "nav": round(nav, 2),
            "annual_budget_pct": sleeve_c.annual_budget_pct,
            "underlying": sleeve_c.underlying,
            "has_legs": int(bool(put_legs)),
            "min_dte": min_dte if min_dte is not None else "",
            "needs_roll": int(needs_roll),
            "suggested_expiry": str(target.expiry),
            "suggested_long_strike": float(target.long_strike),
            "suggested_short_strike": float(target.short_strike),
            "estimated_debit": round(float(target.estimated_premium), 4),
            "suggested_limit_debit": round(float(limit_debit), 4) if limit_debit is not None else "",
            "suggested_spreads": int(contracts),
            "max_trade_budget_usd": round(max_trade_budget_usd, 2),
        }
        _append_csv_row(log_path, row)

        return 0

    finally:
        await ib.disconnect()


def main() -> int:
    p = argparse.ArgumentParser(description="Sleeve C daily monitor (manual execution assistant).")
    p.add_argument("--live", action="store_true", help="Connect to IBKR for live option chain + positions.")
    p.add_argument("--nav", type=float, default=None, help="Override NAV used for budgeting (default: IBKR NLV).")
    p.add_argument(
        "--log-path",
        type=str,
        default="data/sleeve_c/paper_trading/sleeve_c_log.csv",
        help="CSV log path.",
    )
    args = p.parse_args()

    if not args.live:
        print("ERROR: This script currently requires --live (IBKR option chain + greeks).")
        return 2

    return asyncio.run(_run_live(args))


if __name__ == "__main__":
    raise SystemExit(main())
