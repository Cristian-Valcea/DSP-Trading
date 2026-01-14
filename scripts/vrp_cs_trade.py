#!/usr/bin/env python3
"""
VRP-CS Trade Operations (Calendar Spread)

Enables Close / Open / Roll operations for the VXM calendar spread without using TWS UI.
Designed to be callable from the Control UI (non-interactive) via `--confirm YES`.

Default position config:
  data/vrp/paper_trading/position_config.json

Examples:
  # Preview close (no orders placed)
  python scripts/vrp_cs_trade.py close --dry-run

  # Execute close (requires TWS/Gateway running)
  python scripts/vrp_cs_trade.py close --live --confirm YES

  # Preview open
  python scripts/vrp_cs_trade.py open --front VXMG6 --back VXMH6 --dry-run

  # Execute roll (two-calendar-combos: close old spread, open new spread)
  python scripts/vrp_cs_trade.py roll --new-front VXMG6 --new-back VXMH6 --live --confirm YES
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsp.regime.vrp_regime_gate import VRPRegimeGate
from dsp.utils.config import load_config


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "position_config.json"
TRADES_LOG_PATH = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "trades.csv"

MONTH_CODE_TO_NUM = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
NUM_TO_MONTH_CODE = {v: k for k, v in MONTH_CODE_TO_NUM.items()}


def _round_to_tick(price: float, tick: float = 0.05) -> float:
    return round(price / tick) * tick


def _vxm_code_to_yyyymm(code: str) -> str:
    """
    Convert a VXM contract code like VXMG6 -> YYYYMM (e.g., 202602).

    Notes:
      - Assumes current decade.
      - If year digit is behind the current year digit by >5, assume next decade.
    """
    code = code.strip().upper()
    if not code.startswith("VXM"):
        raise ValueError(f"Expected VXM* code, got: {code}")
    if len(code) != 5:
        raise ValueError(f"Expected 5-char VXM code like VXMG6, got: {code}")

    month_code = code[3]
    year_digit = code[4]
    if month_code not in MONTH_CODE_TO_NUM:
        raise ValueError(f"Invalid month code in {code}: {month_code}")
    if not year_digit.isdigit():
        raise ValueError(f"Invalid year digit in {code}: {year_digit}")

    month = MONTH_CODE_TO_NUM[month_code]
    now = datetime.now()
    base_decade = (now.year // 10) * 10
    year = base_decade + int(year_digit)

    # If the inferred year is "too far" in the past, bump a decade.
    if year < now.year - 5:
        year += 10

    return f"{year:04d}{month:02d}"


def _next_vxm_code(code: str) -> str:
    code = code.strip().upper()
    if not code.startswith("VXM") or len(code) != 5:
        raise ValueError(f"Expected VXM* code like VXMG6, got: {code}")

    month_num = MONTH_CODE_TO_NUM[code[3]]
    year_digit = int(code[4])

    month_num += 1
    if month_num == 13:
        month_num = 1
        year_digit = (year_digit + 1) % 10

    return f"VXM{NUM_TO_MONTH_CODE[month_num]}{year_digit}"


def _load_position_config(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) if DEFAULT_CONFIG_PATH.exists() else {}


def _save_position_config(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_trade_log(row: Dict[str, Any]) -> None:
    TRADES_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRADES_LOG_PATH.exists()
    with open(TRADES_LOG_PATH, "a", encoding="utf-8") as f:
        if write_header:
            f.write("ts_iso,action,mode,details_json\n")
        f.write(
            ",".join(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    row.get("action", ""),
                    row.get("mode", ""),
                    json.dumps(row.get("details", {}), sort_keys=True),
                ]
            )
            + "\n"
        )


@dataclass(frozen=True)
class PretradeCheckResult:
    ok: bool
    gate_state: str
    vix: float
    vx1: float
    vx2: float
    spread: float
    contango_pct: float
    notes: str


async def _connect_ib(client_id_offset: int = 60):
    from ib_insync import IB

    cfg = load_config(strict=False)
    ib = IB()
    await ib.connectAsync(
        host=cfg.ibkr.host,
        port=cfg.ibkr.port,
        clientId=cfg.ibkr.client_id + client_id_offset,
        timeout=cfg.ibkr.timeout_s,
    )
    return ib


async def _qualify_vxm_future(ib, code: str):
    from ib_insync import Future

    yyyymm = _vxm_code_to_yyyymm(code)
    contract = Future(symbol="VXM", lastTradeDateOrContractMonth=yyyymm, exchange="CFE")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        raise ValueError(f"Could not qualify contract: {code} ({yyyymm})")
    return qualified[0]


def _mid(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last and last > 0:
        return last
    return None


async def _get_leg_quotes(ib, near, far) -> Tuple[Dict[str, float], Dict[str, float]]:
    [t_near, t_far] = await ib.reqTickersAsync(near, far)

    def _q(t) -> Dict[str, float]:
        bid = float(t.bid) if t.bid else 0.0
        ask = float(t.ask) if t.ask else 0.0
        last = float(t.last) if t.last else 0.0
        mid = _mid(bid, ask, last) or 0.0
        return {"bid": bid, "ask": ask, "last": last, "mid": mid}

    return _q(t_near), _q(t_far)


async def _pretrade_checks(
    ib,
    front_code: str,
    back_code: str,
    *,
    vix: Optional[float],
    vvix: Optional[float],
    min_spread: float = 0.50,
    max_vix: float = 30.0,
) -> PretradeCheckResult:
    front = await _qualify_vxm_future(ib, front_code)
    back = await _qualify_vxm_future(ib, back_code)
    q_front, q_back = await _get_leg_quotes(ib, front, back)

    vx1 = q_front["mid"]
    vx2 = q_back["mid"]
    spread = vx2 - vx1

    vix_used = float(vix) if vix is not None else (vx1 - 1.0)
    vvix_used = float(vvix) if vvix is not None else 85.0
    contango_pct = 0.0 if vix_used <= 0 else 100.0 * (vx1 - vix_used) / vix_used

    gate = VRPRegimeGate()
    state = gate.update(vix=vix_used, vvix=vvix_used, vx_f1=vx1)
    gate_state = getattr(state, "value", str(state))

    ok = True
    notes = []
    if gate_state != "OPEN":
        ok = False
        notes.append(f"gate={gate_state}")
    if vix_used >= max_vix:
        ok = False
        notes.append(f"vix>={max_vix}")
    if spread <= min_spread:
        ok = False
        notes.append(f"spread<={min_spread}")

    return PretradeCheckResult(
        ok=ok,
        gate_state=gate_state,
        vix=vix_used,
        vx1=vx1,
        vx2=vx2,
        spread=spread,
        contango_pct=contango_pct,
        notes=", ".join(notes) if notes else "OK",
    )


async def _check_expected_positions(ib, short_code: str, long_code: str, qty: int) -> Tuple[bool, str]:
    """
    Verify that we have the expected VXM legs:
      short_code: -qty
      long_code: +qty
    """
    short_c = await _qualify_vxm_future(ib, short_code)
    long_c = await _qualify_vxm_future(ib, long_code)

    positions = ib.positions()
    pos_by_conid = {p.contract.conId: float(p.position) for p in positions if getattr(p.contract, "secType", "") == "FUT"}

    short_pos = pos_by_conid.get(short_c.conId, 0.0)
    long_pos = pos_by_conid.get(long_c.conId, 0.0)

    expected_short = -float(qty)
    expected_long = float(qty)

    ok = (short_pos == expected_short) and (long_pos == expected_long)
    msg = f"{short_code} pos={short_pos} (exp {expected_short}); {long_code} pos={long_pos} (exp {expected_long})"
    return ok, msg


async def _place_calendar_spread(
    ib,
    *,
    near_code: str,
    far_code: str,
    action: str,
    quantity: int,
    limit_buffer: float,
    order_type: str,
    outside_rth: bool = False,
) -> Tuple[int, str]:
    from ib_insync import Bag, ComboLeg, LimitOrder, MarketOrder

    near = await _qualify_vxm_future(ib, near_code)
    far = await _qualify_vxm_future(ib, far_code)
    q_near, q_far = await _get_leg_quotes(ib, near, far)

    spread_mid = q_far["mid"] - q_near["mid"]
    if spread_mid == 0:
        raise ValueError("Could not compute spread midpoint (missing quotes)")

    action = action.upper()
    if action not in ("BUY", "SELL"):
        raise ValueError(f"Invalid action: {action}")

    # IMPORTANT (IBKR combo semantics):
    # - `ComboLeg.action` expresses the leg directions for a BUY of the combo.
    # - The IB order `action` (BUY/SELL) flips those legs automatically.
    # If we invert both legs AND set order.action=SELL, IBKR flips again and we
    # can accidentally OPEN when we intended to CLOSE.
    #
    # For a VXM calendar spread, define BUY-combo legs as:
    #   - SELL near (front month)
    #   - BUY  far  (back month)
    # That opens: short near, long far.
    # Then use order.action=SELL to close that spread.
    near_leg_action = "SELL"
    far_leg_action = "BUY"

    limit_price = _round_to_tick(spread_mid + limit_buffer) if action == "BUY" else _round_to_tick(spread_mid - limit_buffer)

    combo = Bag()
    combo.symbol = "VXM"
    combo.exchange = "CFE"
    combo.currency = "USD"
    combo.comboLegs = [
        ComboLeg(conId=near.conId, ratio=1, action=near_leg_action, exchange="CFE"),
        ComboLeg(conId=far.conId, ratio=1, action=far_leg_action, exchange="CFE"),
    ]

    order_type = order_type.upper()
    if order_type == "MKT":
        ib_order = MarketOrder(action=action, totalQuantity=quantity, outsideRth=outside_rth, tif='DAY')
    elif order_type == "LMT":
        ib_order = LimitOrder(action=action, totalQuantity=quantity, lmtPrice=limit_price, outsideRth=outside_rth, tif='DAY')
    else:
        raise ValueError(f"Unsupported order type: {order_type}")

    trade = ib.placeOrder(combo, ib_order)
    await asyncio.sleep(0.5)
    order_id = int(trade.order.orderId or 0)
    status = str(trade.orderStatus.status or "")

    return order_id, status


def _require_confirmation(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if not args.live:
        raise SystemExit("ERROR: Refusing to place orders without --live. Use --dry-run to preview.")
    if args.confirm != "YES":
        raise SystemExit("ERROR: Refusing to place live orders without --confirm YES (required for UI/non-interactive).")


async def cmd_close(args: argparse.Namespace) -> int:
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = _load_position_config(cfg_path)
    short_code = args.short or cfg.get("short_contract")
    long_code = args.long or cfg.get("long_contract")
    qty = int(args.quantity or cfg.get("position_size", 1))

    print("=" * 60)
    print("VRP-CS CLOSE SPREAD")
    print("=" * 60)
    print(f"Close: {short_code}/{long_code} qty={qty}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No orders will be placed ***\n")
        return 0

    _require_confirmation(args)

    ib = await _connect_ib(client_id_offset=args.client_id_offset)
    try:
        ok, msg = await _check_expected_positions(ib, short_code, long_code, qty)
        if not ok and not args.override_checks:
            raise SystemExit(f"ERROR: Position mismatch. {msg}\nUse --override-checks to force.")
        if not ok:
            print(f"WARNING: Position mismatch override. {msg}")

        order_id, status = await _place_calendar_spread(
            ib,
            near_code=short_code,
            far_code=long_code,
            action="SELL",
            quantity=qty,
            limit_buffer=args.limit_buffer,
            order_type=args.order_type,
        )

        print(f"Submitted close spread orderId={order_id} status={status}")
        _append_trade_log({"action": "close", "mode": "live", "details": {"order_id": order_id, "status": status}})
        return 0
    finally:
        ib.disconnect()


async def cmd_open(args: argparse.Namespace) -> int:
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = _load_position_config(cfg_path)
    front = args.front
    back = args.back
    qty = int(args.quantity or cfg.get("position_size", 1))

    if not front or not back:
        raise SystemExit("ERROR: open requires --front and --back")

    print("=" * 60)
    print("VRP-CS OPEN SPREAD")
    print("=" * 60)
    print(f"Open: short {front}, long {back} qty={qty}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No orders will be placed ***\n")
        return 0

    _require_confirmation(args)

    ib = await _connect_ib(client_id_offset=args.client_id_offset)
    try:
        checks = await _pretrade_checks(
            ib,
            front,
            back,
            vix=args.vix,
            vvix=args.vvix,
            min_spread=args.min_spread,
            max_vix=args.max_vix,
        )
        print(
            f"Pretrade: gate={checks.gate_state} vix={checks.vix:.2f} "
            f"spread={checks.spread:.2f} contango={checks.contango_pct:.1f}% ({checks.notes})"
        )
        if not checks.ok and not args.override_checks:
            raise SystemExit(f"ERROR: Pretrade checks failed: {checks.notes}\nUse --override-checks to force.")

        order_id, status = await _place_calendar_spread(
            ib,
            near_code=front,
            far_code=back,
            action="BUY",
            quantity=qty,
            limit_buffer=args.limit_buffer,
            order_type=args.order_type,
        )
        print(f"Submitted open spread orderId={order_id} status={status}")
        _append_trade_log({"action": "open", "mode": "live", "details": {"order_id": order_id, "status": status}})
        return 0
    finally:
        ib.disconnect()


async def cmd_roll(args: argparse.Namespace) -> int:
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = _load_position_config(cfg_path)
    short_code = cfg.get("short_contract")
    long_code = cfg.get("long_contract")
    qty = int(args.quantity or cfg.get("position_size", 1))

    new_front = args.new_front or long_code
    new_back = args.new_back or _next_vxm_code(new_front)

    print("=" * 60)
    print("VRP-CS ROLL SPREAD (two calendar combos)")
    print("=" * 60)
    print(f"Current: short {short_code}, long {long_code}, qty={qty}")
    print(f"New:     short {new_front}, long {new_back}, qty={qty}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No orders will be placed ***\n")
        print("Would submit:")
        print(f"  1) CLOSE old spread: SELL {qty}x {short_code}/{long_code}")
        print(f"  2) OPEN new spread:  BUY  {qty}x {new_front}/{new_back}")
        return 0

    _require_confirmation(args)

    ib = await _connect_ib(client_id_offset=args.client_id_offset)
    try:
        # Validate expected current position unless overridden
        ok, msg = await _check_expected_positions(ib, short_code, long_code, qty)
        if not ok and not args.override_checks:
            raise SystemExit(f"ERROR: Position mismatch. {msg}\nUse --override-checks to force.")
        if not ok:
            print(f"WARNING: Position mismatch override. {msg}")

        checks = await _pretrade_checks(
            ib,
            new_front,
            new_back,
            vix=args.vix,
            vvix=args.vvix,
            min_spread=args.min_spread,
            max_vix=args.max_vix,
        )
        print(
            f"Pretrade (new spread): gate={checks.gate_state} vix={checks.vix:.2f} "
            f"spread={checks.spread:.2f} contango={checks.contango_pct:.1f}% ({checks.notes})"
        )
        if not checks.ok and not args.override_checks:
            raise SystemExit(f"ERROR: Pretrade checks failed: {checks.notes}\nUse --override-checks to force.")

        close_id, close_status = await _place_calendar_spread(
            ib,
            near_code=short_code,
            far_code=long_code,
            action="SELL",
            quantity=qty,
            limit_buffer=args.limit_buffer,
            order_type=args.order_type,
        )
        print(f"Submitted close leg orderId={close_id} status={close_status}")

        open_id, open_status = await _place_calendar_spread(
            ib,
            near_code=new_front,
            far_code=new_back,
            action="BUY",
            quantity=qty,
            limit_buffer=args.limit_buffer,
            order_type=args.order_type,
        )
        print(f"Submitted open leg orderId={open_id} status={open_status}")

        _append_trade_log(
            {
                "action": "roll",
                "mode": "live",
                "details": {
                    "close_order_id": close_id,
                    "close_status": close_status,
                    "open_order_id": open_id,
                    "open_status": open_status,
                    "from": f"{short_code}/{long_code}",
                    "to": f"{new_front}/{new_back}",
                },
            }
        )
        return 0
    finally:
        ib.disconnect()


async def cmd_update_config(args: argparse.Namespace) -> int:
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = _load_position_config(cfg_path)

    short_code = args.short or cfg.get("short_contract")
    long_code = args.long or cfg.get("long_contract")
    short_price = float(args.short_price) if args.short_price is not None else float(cfg.get("short_entry_price", 0))
    long_price = float(args.long_price) if args.long_price is not None else float(cfg.get("long_entry_price", 0))
    entry_dt = args.entry_date or date.today().isoformat()
    entry_spread = float(args.entry_spread) if args.entry_spread is not None else (long_price - short_price)

    stop_loss = float(args.stop_loss) if args.stop_loss is not None else round(entry_spread * 1.25, 2)
    take_profit = float(args.take_profit) if args.take_profit is not None else round(entry_spread * 0.50, 2)

    cfg.update(
        {
            "short_contract": short_code,
            "long_contract": long_code,
            "short_entry_price": short_price,
            "long_entry_price": long_price,
            "entry_date": entry_dt,
            "entry_spread": entry_spread,
            "stop_loss_spread": stop_loss,
            "take_profit_spread": take_profit,
        }
    )

    # Optional roll/expiry fields
    if args.vx1_expiry:
        cfg["vx1_expiry"] = args.vx1_expiry
    if args.roll_by_date:
        cfg["roll_by_date"] = args.roll_by_date

    _save_position_config(cfg_path, cfg)
    print(f"Saved VRP-CS config: {cfg_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VRP-CS trade operations (VXM calendar spread).")
    p.add_argument("--config", type=Path, default=None, help=f"Config path (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--client-id-offset", type=int, default=60, help="IBKR client ID offset to avoid conflicts")

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_exec_flags(sp):
        sp.add_argument("--dry-run", action="store_true", help="Preview only; do not place orders")
        sp.add_argument("--live", action="store_true", help="Required to place real orders")
        sp.add_argument("--confirm", type=str, default="", help="Set to YES for non-interactive confirmation")
        sp.add_argument("--override-checks", action="store_true", help="Allow execution even if checks fail")
        sp.add_argument("--order-type", type=str, default="LMT", choices=["LMT", "MKT"], help="Order type")
        sp.add_argument("--limit-buffer", type=float, default=0.05, help="Limit buffer (spread points)")
        sp.add_argument("--quantity", type=int, default=None, help="Override config position_size")

    sp_close = sub.add_parser("close", help="Close the current spread")
    add_exec_flags(sp_close)
    sp_close.add_argument("--short", type=str, default=None, help="Override config short contract")
    sp_close.add_argument("--long", type=str, default=None, help="Override config long contract")

    sp_open = sub.add_parser("open", help="Open a new spread")
    add_exec_flags(sp_open)
    sp_open.add_argument("--front", type=str, required=True, help="Front month contract (short)")
    sp_open.add_argument("--back", type=str, required=True, help="Back month contract (long)")
    sp_open.add_argument("--vix", type=float, default=None, help="VIX spot (optional; else approximated)")
    sp_open.add_argument("--vvix", type=float, default=None, help="VVIX (optional; else 85.0)")
    sp_open.add_argument("--min-spread", type=float, default=0.50, help="Minimum VX2-VX1 spread required")
    sp_open.add_argument("--max-vix", type=float, default=30.0, help="Maximum allowed VIX for opening")

    sp_roll = sub.add_parser("roll", help="Roll the spread (close old, open new)")
    add_exec_flags(sp_roll)
    sp_roll.add_argument("--new-front", type=str, default=None, help="New front month (default: current long)")
    sp_roll.add_argument("--new-back", type=str, default=None, help="New back month (default: +1 month from new-front)")
    sp_roll.add_argument("--vix", type=float, default=None, help="VIX spot (optional; else approximated)")
    sp_roll.add_argument("--vvix", type=float, default=None, help="VVIX (optional; else 85.0)")
    sp_roll.add_argument("--min-spread", type=float, default=0.50, help="Minimum VX2-VX1 spread required")
    sp_roll.add_argument("--max-vix", type=float, default=30.0, help="Maximum allowed VIX for opening")

    sp_uc = sub.add_parser("update-config", help="Update saved position config (no trading)")
    sp_uc.add_argument("--short", type=str, required=False, default=None)
    sp_uc.add_argument("--long", type=str, required=False, default=None)
    sp_uc.add_argument("--short-price", type=float, required=False, default=None)
    sp_uc.add_argument("--long-price", type=float, required=False, default=None)
    sp_uc.add_argument("--entry-date", type=str, required=False, default=None)
    sp_uc.add_argument("--entry-spread", type=float, required=False, default=None)
    sp_uc.add_argument("--stop-loss", type=float, required=False, default=None)
    sp_uc.add_argument("--take-profit", type=float, required=False, default=None)
    sp_uc.add_argument("--vx1-expiry", type=str, required=False, default=None)
    sp_uc.add_argument("--roll-by-date", type=str, required=False, default=None)

    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "close":
        return asyncio.run(cmd_close(args))
    if args.cmd == "open":
        return asyncio.run(cmd_open(args))
    if args.cmd == "roll":
        return asyncio.run(cmd_roll(args))
    if args.cmd == "update-config":
        return asyncio.run(cmd_update_config(args))

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
