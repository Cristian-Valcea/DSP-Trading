#!/usr/bin/env python3
"""
Panic Flatten Script - Close All Positions for a Specific Sleeve

USAGE:
    python scripts/panic_flatten_sleeve.py --sleeve dm          # Flatten Sleeve DM (ETFs)
    python scripts/panic_flatten_sleeve.py --sleeve vrp-erp     # Flatten VRP-ERP (SPY)
    python scripts/panic_flatten_sleeve.py --sleeve vrp-cs      # Flatten VRP-CS (VX futures)
    python scripts/panic_flatten_sleeve.py --sleeve c           # Flatten Sleeve C (SPY options)
    python scripts/panic_flatten_sleeve.py --sleeve all         # Flatten EVERYTHING

    Add --dry-run to preview without executing
    Add --force to skip confirmation prompt

SAFETY:
    - Requires explicit --confirm flag for live execution
    - Shows positions before closing
    - Logs all actions to panic_log.csv
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Sleeve definitions - which symbols belong to each sleeve
SLEEVE_SYMBOLS = {
    "dm": ["SPY", "EFA", "EEM", "IEF", "TLT", "TIP", "GLD", "PDBC", "UUP", "SHY"],
    "vrp-erp": ["SPY"],  # Note: overlaps with DM - careful!
    "vrp-cs": [],  # VXM futures - handled separately
    "c": [],  # SPY options - handled separately
}

# Security types per sleeve
SLEEVE_SEC_TYPES = {
    "dm": ["STK"],
    "vrp-erp": ["STK"],
    "vrp-cs": ["FUT"],
    "c": ["OPT"],
}


def get_positions_for_sleeve(ib, sleeve: str) -> list:
    """Get positions matching a specific sleeve."""
    positions = ib.positions()
    matching = []

    sec_types = SLEEVE_SEC_TYPES.get(sleeve, [])
    symbols = SLEEVE_SYMBOLS.get(sleeve, [])

    for pos in positions:
        contract = pos.contract

        # Match by security type and symbol
        if sleeve == "dm":
            # Sleeve DM: ETF stocks from the universe
            if contract.secType == "STK" and contract.symbol in symbols:
                matching.append(pos)
        elif sleeve == "vrp-erp":
            # VRP-ERP: Only SPY stock
            if contract.secType == "STK" and contract.symbol == "SPY":
                matching.append(pos)
        elif sleeve == "vrp-cs":
            # VRP-CS: VXM futures
            if contract.secType == "FUT" and contract.symbol in ["VXM", "VX"]:
                matching.append(pos)
        elif sleeve == "c":
            # Sleeve C: SPY options (puts)
            if contract.secType == "OPT" and contract.symbol == "SPY":
                matching.append(pos)
        elif sleeve == "all":
            # ALL: everything
            matching.append(pos)

    return matching


def format_position(pos) -> str:
    """Format position for display."""
    c = pos.contract
    qty = int(pos.position)
    avg = pos.avgCost

    if c.secType == "STK":
        return f"  {c.symbol}: {qty:+d} shares @ ${avg:.2f}"
    elif c.secType == "FUT":
        return f"  {c.localSymbol}: {qty:+d} contracts @ ${avg:.2f}"
    elif c.secType == "OPT":
        return f"  {c.localSymbol}: {qty:+d} contracts @ ${avg:.2f}"
    else:
        return f"  {c.symbol} ({c.secType}): {qty:+d} @ ${avg:.2f}"


async def flatten_position(ib, pos, dry_run: bool = True) -> dict:
    """Close a single position. Returns order result."""
    from ib_insync import MarketOrder

    contract = pos.contract
    qty = int(pos.position)

    if qty == 0:
        return {"status": "skip", "reason": "zero position"}

    # Determine action (opposite of current position)
    action = "SELL" if qty > 0 else "BUY"
    close_qty = abs(qty)

    result = {
        "symbol": contract.symbol,
        "localSymbol": contract.localSymbol,
        "secType": contract.secType,
        "action": action,
        "qty": close_qty,
        "status": "pending",
    }

    if dry_run:
        result["status"] = "dry-run"
        print(f"  [DRY-RUN] Would {action} {close_qty} {contract.localSymbol or contract.symbol}")
        return result

    # Execute order
    try:
        ib.qualifyContracts(contract)
        order = MarketOrder(action, close_qty)
        trade = ib.placeOrder(contract, order)

        # Wait for fill (max 60 seconds)
        timeout = 60
        start = datetime.now()
        while trade.orderStatus.status not in ["Filled", "Cancelled", "ApiCancelled"]:
            await asyncio.sleep(0.5)
            if (datetime.now() - start).seconds > timeout:
                result["status"] = "timeout"
                result["error"] = f"Order not filled within {timeout}s"
                return result

        if trade.orderStatus.status == "Filled":
            result["status"] = "filled"
            result["fill_price"] = trade.orderStatus.avgFillPrice
            print(f"  [FILLED] {action} {close_qty} {contract.localSymbol or contract.symbol} @ ${result['fill_price']:.2f}")
        else:
            result["status"] = "cancelled"
            result["error"] = trade.orderStatus.status

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [ERROR] {contract.symbol}: {e}")

    return result


def log_panic_action(sleeve: str, positions: list, results: list, dry_run: bool):
    """Log panic action to CSV."""
    log_path = PROJECT_ROOT / "data" / "panic_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()

    with open(log_path, "a") as f:
        if write_header:
            f.write("timestamp,sleeve,dry_run,positions_found,positions_closed,errors\n")

        errors = sum(1 for r in results if r.get("status") in ["error", "timeout"])
        closed = sum(1 for r in results if r.get("status") == "filled")

        f.write(f"{datetime.now().isoformat()},{sleeve},{dry_run},{len(positions)},{closed},{errors}\n")


async def main():
    parser = argparse.ArgumentParser(description="Panic Flatten - Close All Positions for a Sleeve")
    parser.add_argument("--sleeve", required=True, choices=["dm", "vrp-erp", "vrp-cs", "c", "all"],
                        help="Which sleeve to flatten")
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--confirm", action="store_true", help="Required for live execution")
    args = parser.parse_args()

    # Safety check
    if not args.dry_run and not args.confirm:
        print("ERROR: Live execution requires --confirm flag")
        print("Use --dry-run to preview, or add --confirm to execute")
        sys.exit(1)

    # Connect to IBKR
    from ib_insync import IB

    ib = IB()
    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PORT", "4002"))  # Default to paper
    client_id = 220  # Unique client ID for panic script

    print(f"\n{'='*60}")
    print(f"  PANIC FLATTEN - Sleeve: {args.sleeve.upper()}")
    print(f"  {'[DRY-RUN MODE]' if args.dry_run else '[LIVE EXECUTION]'}")
    print(f"{'='*60}")

    try:
        print(f"\nConnecting to IBKR {host}:{port}...")
        ib.connect(host, port, clientId=client_id, timeout=20)
        ib.reqMarketDataType(1)

        account = ib.managedAccounts()[0]
        is_paper = account.startswith("DU")
        print(f"Account: {account} ({'PAPER' if is_paper else 'LIVE'})")

        # Get positions for this sleeve
        positions = get_positions_for_sleeve(ib, args.sleeve)

        if not positions:
            print(f"\nNo positions found for sleeve '{args.sleeve}'")
            ib.disconnect()
            return

        # Display positions
        print(f"\nPositions to close ({len(positions)}):")
        for pos in positions:
            print(format_position(pos))

        # Confirmation
        if not args.dry_run and not args.force:
            print(f"\n⚠️  WARNING: This will close {len(positions)} position(s)!")
            confirm = input("Type 'FLATTEN' to confirm: ")
            if confirm != "FLATTEN":
                print("Aborted.")
                ib.disconnect()
                return

        # Execute
        print(f"\n{'Previewing' if args.dry_run else 'Executing'} flatten orders...")
        results = []
        for pos in positions:
            result = await flatten_position(ib, pos, dry_run=args.dry_run)
            results.append(result)

        # Log action
        log_panic_action(args.sleeve, positions, results, args.dry_run)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY:")
        if args.dry_run:
            print(f"  Would close {len(positions)} position(s)")
        else:
            filled = sum(1 for r in results if r.get("status") == "filled")
            errors = sum(1 for r in results if r.get("status") in ["error", "timeout"])
            print(f"  Closed: {filled}/{len(positions)}")
            if errors:
                print(f"  Errors: {errors}")
        print(f"{'='*60}\n")

        ib.disconnect()

    except Exception as e:
        print(f"\nERROR: {e}")
        try:
            ib.disconnect()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
