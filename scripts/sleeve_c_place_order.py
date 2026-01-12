#!/usr/bin/env python3
"""
Place a Sleeve C (Tail Hedge) put spread order.

This script places the put spread order recommended by sleeve_c_daily_monitor.py.

Usage:
  source venv/bin/activate
  python scripts/sleeve_c_place_order.py \
      --expiry 2026-02-13 \
      --long-strike 575 \
      --short-strike 545 \
      --quantity 5 \
      --limit 3.08 \
      --live

Add --dry-run to simulate without placing the order.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsp.ibkr import IBKRClient
from dsp.utils.config import load_config


async def place_spread(args: argparse.Namespace) -> int:
    """Place the put spread order."""
    cfg = load_config(strict=False)

    # Parse expiry date
    try:
        expiry = datetime.strptime(args.expiry, "%Y-%m-%d").date()
    except ValueError:
        print(f"ERROR: Invalid expiry date format: {args.expiry} (use YYYY-MM-DD)")
        return 1

    print("=" * 60)
    print("SLEEVE C PUT SPREAD ORDER")
    print("=" * 60)
    print(f"Underlying: {args.underlying}")
    print(f"Expiry: {expiry}")
    print(f"Long Strike (BUY): {args.long_strike}")
    print(f"Short Strike (SELL): {args.short_strike}")
    print(f"Quantity: {args.quantity} spreads")
    print(f"Limit Debit: ${args.limit:.2f} per spread")
    print(f"Total Max Cost: ${args.limit * args.quantity * 100:.2f}")
    print("=" * 60)

    if args.dry_run:
        print("\n*** DRY RUN MODE - No order will be placed ***\n")
        return 0

    if not args.live:
        print("\nERROR: Must specify --live to place a real order.")
        print("Use --dry-run to preview without placing.")
        return 2

    # Confirmation (supports non-interactive use from the Control UI)
    if args.confirm == "YES":
        pass
    else:
        if not sys.stdin.isatty():
            print("\nERROR: Refusing to place a live order without confirmation in a non-interactive session.")
            print("Use one of:")
            print("  - --dry-run")
            print("  - --confirm YES")
            return 2

        print("\n⚠️  WARNING: This will place a REAL order.")
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Order cancelled.")
            return 0

    ib = IBKRClient(host=cfg.ibkr.host, port=cfg.ibkr.port, client_id=cfg.ibkr.client_id)
    if not await ib.connect():
        print("ERROR: Could not connect to IBKR. Is TWS/Gateway running?")
        return 2

    try:
        print("\nPlacing order...")

        status = await ib.place_option_spread(
            underlying=args.underlying,
            expiry=expiry,
            long_strike=float(args.long_strike),
            short_strike=float(args.short_strike),
            right="P",  # Put spread
            quantity=args.quantity,
            limit_price=args.limit,
            action="BUY",  # Debit spread
        )

        print("\n" + "=" * 60)
        print("ORDER SUBMITTED")
        print("=" * 60)
        print(f"Order ID: {status.order_id}")
        print(f"Status: {status.status}")
        print(f"Filled: {status.filled_quantity}")
        print(f"Remaining: {status.remaining_quantity}")
        if status.avg_fill_price > 0:
            print(f"Avg Fill Price: ${status.avg_fill_price:.2f}")
        print("=" * 60)

        if status.status in ("Submitted", "PreSubmitted", "PendingSubmit"):
            print("\n✅ Order is working. Check TWS for fill status.")
            print("   The order will remain open until filled or cancelled.")
        elif status.status == "Filled":
            print("\n✅ Order FILLED!")
        else:
            print(f"\n⚠️  Order status: {status.status}")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR placing order: {e}")
        return 3

    finally:
        await ib.disconnect()


def main() -> int:
    p = argparse.ArgumentParser(description="Place Sleeve C put spread order.")
    p.add_argument("--underlying", type=str, default="SPY", help="Underlying symbol.")
    p.add_argument("--expiry", type=str, required=True, help="Expiry date (YYYY-MM-DD).")
    p.add_argument("--long-strike", type=float, required=True, help="Long put strike (higher).")
    p.add_argument("--short-strike", type=float, required=True, help="Short put strike (lower).")
    p.add_argument("--quantity", type=int, required=True, help="Number of spreads.")
    p.add_argument("--limit", type=float, required=True, help="Limit debit per spread.")
    p.add_argument("--live", action="store_true", help="Place real order (required).")
    p.add_argument("--dry-run", action="store_true", help="Preview without placing order.")
    p.add_argument(
        "--confirm",
        type=str,
        default="",
        help="Non-interactive confirmation. Set to YES to place the order without prompting (for UI automation).",
    )
    args = p.parse_args()

    if args.long_strike <= args.short_strike:
        print("ERROR: Long strike must be higher than short strike for a put spread.")
        return 1

    return asyncio.run(place_spread(args))


if __name__ == "__main__":
    raise SystemExit(main())
