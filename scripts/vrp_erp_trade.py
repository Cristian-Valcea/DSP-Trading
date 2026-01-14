#!/usr/bin/env python3
"""VRP-ERP SPY Trade Execution Script.

Executes SPY buy/sell orders to rebalance to target position.
Called from UI after monitor calculates the required trade.

Usage:
    # Sell 2 shares (market order)
    python scripts/vrp_erp_trade.py --action SELL --qty 2 --live --confirm YES

    # Buy 5 shares (market order)
    python scripts/vrp_erp_trade.py --action BUY --qty 5 --live --confirm YES

    # Dry run (no execution)
    python scripts/vrp_erp_trade.py --action SELL --qty 2 --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def execute_spy_order(action: str, qty: int, dry_run: bool = False) -> dict:
    """Execute SPY market order via IBKR.

    Args:
        action: "BUY" or "SELL"
        qty: Number of shares
        dry_run: If True, don't actually place order

    Returns:
        dict with status, order_id, fill info
    """
    action = action.upper()
    if action not in ("BUY", "SELL"):
        return {"status": "error", "message": f"Invalid action: {action}"}

    if qty <= 0:
        return {"status": "error", "message": f"Invalid quantity: {qty}"}

    print(f"\n{'='*60}")
    print(f"  VRP-ERP SPY TRADE EXECUTION")
    print(f"{'='*60}")
    print(f"  Action: {action} {qty} SPY")
    print(f"  Order Type: MARKET")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    if dry_run:
        print("🔍 DRY RUN - Order would be placed but not executed")
        return {
            "status": "dry_run",
            "action": action,
            "qty": qty,
            "message": f"Would {action} {qty} SPY at market"
        }

    # Execute via IBKR
    try:
        from ib_insync import IB, Stock, MarketOrder

        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=202, timeout=10)

        # Create SPY contract
        spy = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(spy)

        # Create market order
        order = MarketOrder(action, qty)

        print(f"📤 Submitting {action} {qty} SPY @ MARKET...")

        # Place order
        trade = ib.placeOrder(spy, order)

        # Wait for fill (up to 30 seconds)
        timeout = 30
        filled = False
        for _ in range(timeout * 2):
            ib.sleep(0.5)
            if trade.orderStatus.status == 'Filled':
                filled = True
                break
            elif trade.orderStatus.status in ('Cancelled', 'ApiCancelled', 'Inactive'):
                break

        status = trade.orderStatus.status
        fill_price = trade.orderStatus.avgFillPrice if filled else None
        filled_qty = int(trade.orderStatus.filled) if filled else 0

        ib.disconnect()

        if filled:
            print(f"\n✅ ORDER FILLED: {action} {filled_qty} SPY @ ${fill_price:.2f}")

            # Log the trade
            log_trade(action, filled_qty, fill_price)

            return {
                "status": "filled",
                "action": action,
                "qty": filled_qty,
                "fill_price": fill_price,
                "order_id": trade.order.orderId,
                "message": f"Filled {action} {filled_qty} SPY @ ${fill_price:.2f}"
            }
        else:
            print(f"\n⚠️ ORDER STATUS: {status}")
            return {
                "status": status.lower(),
                "action": action,
                "qty": qty,
                "message": f"Order {status}"
            }

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def log_trade(action: str, qty: int, fill_price: float):
    """Append trade to VRP-ERP log."""
    log_path = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "vrp_erp_trades.csv"

    # Create file with header if needed
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            f.write("date,time,action,qty,fill_price,value\n")

    now = datetime.now()
    value = qty * fill_price * (1 if action == "BUY" else -1)

    row = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        action,
        str(qty),
        f"{fill_price:.2f}",
        f"{value:.2f}"
    ]

    with open(log_path, 'a') as f:
        f.write(",".join(row) + "\n")

    print(f"📝 Trade logged to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="VRP-ERP SPY Trade Execution")
    parser.add_argument('--action', required=True, choices=['BUY', 'SELL', 'buy', 'sell'],
                        help="Trade action (BUY or SELL)")
    parser.add_argument('--qty', type=int, required=True,
                        help="Number of shares")
    parser.add_argument('--live', action='store_true',
                        help="Execute live order (requires --confirm YES)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Preview order without execution")
    parser.add_argument('--confirm', type=str,
                        help="Confirmation (must be 'YES' for live orders)")

    args = parser.parse_args()

    # Validate live execution requires confirmation
    if args.live and args.confirm != "YES":
        print("❌ Live execution requires --confirm YES")
        sys.exit(1)

    # Default to dry-run if neither specified
    dry_run = args.dry_run or not args.live

    result = execute_spy_order(args.action.upper(), args.qty, dry_run=dry_run)

    # Print result as JSON for UI parsing
    print(f"\n--- RESULT ---")
    print(json.dumps(result, indent=2))

    if result["status"] in ("filled", "dry_run"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
