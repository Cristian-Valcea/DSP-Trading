# VRP-CS Roll Runbook — ONE PAGE, STRESS-EXECUTABLE

**Current Roll**: VXMF6/VXMG6 → VXMG6/VXMH6
**Roll-By Date**: **Wed Jan 14, 2026** (N-5 trading days before Jan 21 expiry)
**Roll Destination Rule**: Always "+1 month" — short near, long far.

---

## 1. PRE-ROLL CHECKS (10 min before)

| Check | Condition | Action If Fails |
|-------|-----------|-----------------|
| **Gate State** | = OPEN | DO NOT ROLL. Close existing spread, wait for re-entry. |
| **Contango** | VX2 - VX1 > 0.50 | If backwardation: close spread, do NOT open new. |
| **VX1 Bid/Ask** | Width ≤ 0.15 | If wider: wait 30 min for liquidity, else escalate. |
| **VX2 Bid/Ask** | Width ≤ 0.15 | Same. |
| **New VX3 Bid/Ask** | Width ≤ 0.20 | (for new back month) Same. |
| **Position Match** | Confirm -1 VXMF6, +1 VXMG6 | If mismatch: reconcile before roll. |
| **Margin Headroom** | ≥ $3,000 free | If tight: reduce position or add funds. |

**Record in log**: date, time, gate_state, contango_pct, bid_ask_widths

---

## 2. ROLL EXECUTION (Combo Order Method)

### Option A: Two Combo Tickets (PREFERRED, 2 orders)

```
TWS → Trading → Combo/Strategy → Futures Calendar Spread

Step 1 — CLOSE old spread (VXMF6/VXMG6):
1. Select: VXM
2. Near Month: Jan 2026 (VXMF6) — BUY
3. Far Month: Feb 2026 (VXMG6) — SELL
4. Action: SELL 1 spread (buys F6, sells G6)
5. Order Type: LMT
6. Limit: Current spread ± 0.05 (be patient)
7. Submit → wait for fill

Step 2 — OPEN new spread (VXMG6/VXMH6):
1. Select: VXM
2. Near Month: Feb 2026 (VXMG6) — SELL
3. Far Month: Mar 2026 (VXMH6) — BUY
4. Action: BUY 1 spread (sells G6, buys H6)
5. Order Type: LMT
6. Limit: Current spread ± 0.05 (be patient)
7. Submit → wait for fill
```

This produces the correct target position: **-1 VXMG6, +1 VXMH6**.

### Option B: Explicit Legs (If combos fail)

**Step 1 — Close old spread**:
```
BUY 1 VXMF6 (close short)
SELL 1 VXMG6 (close long)
Order Type: LMT at current spread ± 0.05
```

**Step 2 — Open new spread** (only if gate OPEN, contango > 0.50):
```
SELL 1 VXMG6 (open new short)
BUY 1 VXMH6 (open new long)
Order Type: LMT at current spread ± 0.05
```

---

## 3. FAIL-CLOSED RULE

**If any of these conditions are true → DO NOTHING, ESCALATE:**

| Condition | Response |
|-----------|----------|
| Combo spread market > 0.30 wide | Wait or close legs individually at MKT. |
| Gate = CLOSED or REDUCE | Close existing spread, do NOT open new. |
| VIX > 30 | No new positions. Flatten existing. |
| Fill takes > 15 min | Cancel. Re-evaluate. |
| Orphan leg detected post-fill | IMMEDIATELY close orphan leg at MKT. |

**Escalation**: Call interactive-brokers.com/en/index.php?f=1560 if stuck.

---

## 4. POST-ROLL CHECKS (Immediately after fill)

| Check | Expected | If Wrong |
|-------|----------|----------|
| **Leg 1 (Short)** | -1 VXMG6 (Feb) | Orphan. Close at MKT. |
| **Leg 2 (Long)** | +1 VXMH6 (Mar) | Orphan. Close at MKT. |
| **Old Short (F6)** | 0 (closed) | Still open? Close. |
| **Old Long (G6)** | 0 (closed) | Still open? Close. |
| **Net Positions** | 2 contracts total | If ≠ 2, investigate. |

---

## 5. UPDATE LOGS

**Edit** `scripts/vrp_cs_daily_monitor.py` `PositionConfig`:
```python
short_contract: str = "VXMG6"      # <- was VXMF6
long_contract: str = "VXMH6"       # <- was VXMG6
short_entry_price: float = XX.XX   # <- new fill
long_entry_price: float = XX.XX    # <- new fill
entry_date: str = "2026-01-14"     # <- today
entry_spread: float = X.XX         # <- new spread
stop_loss_spread: float = X.XX * 1.25
take_profit_spread: float = X.XX * 0.50
vx1_expiry: str = "2026-02-19"     # <- Feb expiry
roll_by_date: str = "2026-02-12"   # <- N-5
```

**Run daily digest** to confirm:
```bash
python scripts/daily_digest.py
```

---

## 6. ROLL CALENDAR 2026

| Roll # | Close VX1 | Open Short | Open Long | Roll-By |
|--------|-----------|------------|-----------|---------|
| **1** | VXMF6 (Jan) | VXMG6 (Feb) | VXMH6 (Mar) | **Jan 14** |
| 2 | VXMG6 (Feb) | VXMH6 (Mar) | VXMJ6 (Apr) | Feb 12 |
| 3 | VXMH6 (Mar) | VXMJ6 (Apr) | VXMK6 (May) | Mar 11 |
| 4 | VXMJ6 (Apr) | VXMK6 (May) | VXMM6 (Jun) | Apr 8 |

---

## QUICK DECISION TREE

```
Is Gate OPEN?
 ├── NO → Close spread, do NOT roll. Wait.
 └── YES → Is contango > 0.50?
            ├── NO → Close spread, do NOT roll.
            └── YES → Is spread market < 0.30 wide?
                       ├── NO → Wait 30 min or escalate.
                       └── YES → EXECUTE ROLL.
```

---

*Last updated: 2026-01-10*
*Print this page. Tape to desk. Use under stress.*
