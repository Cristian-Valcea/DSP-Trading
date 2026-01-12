# SLEEVE_C_PAPER_TRADING_RUNBOOK.md — Sleeve C (Tail Hedge)

## Goal
Maintain a small, controlled SPY crash hedge using a 25Δ/10Δ SPY put spread, rolling before expiration.

## What You Run Daily
1) Connect TWS / IB Gateway  
2) Run:

`python scripts/sleeve_c_daily_monitor.py --live`

This prints:
- Existing SPY put legs (if any)
- Whether a roll is needed (DTE <= trigger)
- Suggested next spread: expiry, strikes, qty, and a max debit

## How You Execute (Manual in TWS)
Use a **single combo/vertical spread order** (atomic fill):
- Underlying: `SPY`
- Strategy: **Put Vertical**
- Expiry/strikes: exactly as the monitor script prints
- Action: **BUY** the spread (debit)
- Quantity: `suggested_spreads`
- Limit: `suggested_limit_debit` (or better)

## Roll Procedure
Roll when **Min DTE <= 10** (or earlier if liquidity deteriorates):
1) Close the expiring spread (sell the spread to close).
2) Open the new spread (buy the new spread to open).
3) Re-run `python scripts/sleeve_c_daily_monitor.py --live` to verify legs and that Min DTE reset to ~30–45.

## Verification Checklist (Post-Trade)
- Legs exist in IBKR positions (2 option legs per spread)
- Same expiry on both legs
- Long strike > short strike
- Quantities match (e.g., +3 long puts and -3 short puts)
- Log updated: `data/sleeve_c/paper_trading/sleeve_c_log.csv`
- Digest updated: `python scripts/daily_digest.py --live`

## Known Constraints
- DSP does **not** place options orders automatically yet.
- If IBKR does not return greeks/quotes, you likely need an options market data subscription (OPRA).

