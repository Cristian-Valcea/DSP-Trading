# DSP-100K Operational Manual

**Version**: 1.0
**Date**: 2026-01-10
**Status**: Execute + Verify Mode

---

## Overview

This manual provides day-by-day operational instructions for the DSP-100K paper trading system. At this point, engineering work is complete — the remaining work is operational execution and verification.

**Active Sleeves**:
- **Sleeve DM**: ETF Dual Momentum (LIVE since Jan 5, 2026)
- **Sleeve VRP-ERP**: VIX-gated SPY exposure (go-live Jan 12, 2026)
- **Sleeve VRP-CS**: VIX Calendar Spread (first roll Jan 14, 2026)

---

## Mon Jan 12 — VRP-ERP Go-Live in Paper

### Objective
Start paper trading the VRP-ERP sleeve with live IBKR integration.

### Pre-Market (Before 09:30 ET)

1. **Ensure IBKR TWS/Gateway is running and logged in**
   ```bash
   # Verify connection (optional)
   cd /Users/Shared/wsl-export/wsl-home/dsp100k
   source ../venv/bin/activate
   python -c "from ib_insync import IB; ib=IB(); ib.connect('127.0.0.1', 7497, clientId=900, timeout=10); print('Connected:', ib.isConnected()); ib.disconnect()"
   ```

2. **Run VRP-ERP monitor with live quotes**
   ```bash
   cd /Users/Shared/wsl-export/wsl-home/dsp100k
   source ../venv/bin/activate
   python scripts/vrp_erp_daily_monitor.py --live --base 10000
   ```

   Expected output:
   - VIX level and regime classification
   - Target SPY shares based on regime
   - Any drift warnings if positions exist

3. **Place SPY order if needed**
   - If monitor shows `target_shares > 0` and `actual_shares = 0`:
     - Use TWS to BUY the indicated SPY shares
   - If monitor shows `target_shares = 0`:
     - No action needed (risk-off regime)

### Post-Market (After 16:00 ET)

4. **Run daily digest to verify integration**
   ```bash
   python scripts/daily_digest.py
   cat data/daily_digest/digest_2026-01-12.md
   ```

   Verify:
   - [ ] VRP-ERP section shows today's data
   - [ ] Regime and target shares are populated
   - [ ] No alerts (or expected alerts only)

### Success Criteria
- [ ] VRP-ERP monitor runs without errors
- [ ] SPY position matches target (if applicable)
- [ ] Daily digest captures VRP-ERP status

---

## Tue Jan 13 — VRP-CS Roll Rehearsal

### Objective
Perform a full dry-run of the roll procedure using the stress-executable runbook.

### Rehearsal Steps

1. **Print the roll runbook**
   ```bash
   cat docs/VRP_CS_ROLL_RUNBOOK.md
   # Or open in browser/editor and print
   ```

2. **Walk through Section 1: Pre-Roll Checks**

   Open TWS and verify each check:
   - [ ] Gate State = OPEN (from daily monitor)
   - [ ] Contango > 0.50 (VX2 - VX1)
   - [ ] VX1 Bid/Ask width ≤ 0.15
   - [ ] VX2 Bid/Ask width ≤ 0.15
   - [ ] New VX3 (VXMH6) Bid/Ask width ≤ 0.20
   - [ ] Position matches: -1 VXMF6, +1 VXMG6
   - [ ] Margin headroom ≥ $3,000

3. **Navigate to TWS Combo Order Screen (DO NOT SUBMIT)**
   ```
   TWS → Trading → Combo/Strategy → Futures Calendar Spread
   Select: VXM
   Near Month: Feb 2026 (VXMG6) — SELL
   Far Month: Mar 2026 (VXMH6) — BUY
   ```

   Verify you can see:
   - [ ] Spread quote (bid/ask)
   - [ ] Combined margin requirement
   - [ ] Order entry fields

4. **Document any issues**
   - Note any UI confusion or unexpected states
   - Identify questions for tomorrow

### Rehearsal Checklist
- [ ] All pre-roll checks understood
- [ ] TWS combo order screen navigated
- [ ] Runbook is physically printed and accessible
- [ ] Questions documented (if any)

---

## Wed Jan 14 — VRP-CS Roll Execution

### Objective
Execute the VRP-CS roll from F6/G6 to G6/H6.

### Timeline

| Time (ET) | Action |
|-----------|--------|
| 09:00 | Run pre-roll checks |
| 09:30 | Market open — wait for spreads to settle |
| 10:00 | Execute roll (if all checks pass) |
| 10:15 | Verify positions |
| 10:30 | Update logs and config |

### Execution Steps

1. **Pre-Roll Checks (09:00 ET)**

   Run the daily monitor:
   ```bash
   python scripts/vrp_cs_daily_monitor.py --live
   ```

   Verify per runbook Section 1:
   - [ ] Gate = OPEN
   - [ ] Contango > 0.50
   - [ ] All bid/ask widths ≤ 0.15 (VX1, VX2), ≤ 0.20 (VX3)
   - [ ] Positions: -1 VXMF6, +1 VXMG6
   - [ ] Margin headroom ≥ $3,000

2. **Execute Roll (10:00 ET)**

   **Option A: Single Combo Ticket (PREFERRED)**
   ```
   TWS → Trading → Combo/Strategy → Futures Calendar Spread
   1. Select: VXM
   2. Near Month: Feb 2026 (VXMG6) — SELL
   3. Far Month: Mar 2026 (VXMH6) — BUY
   4. Action: BUY 1 spread
   5. Order Type: LMT
   6. Limit: Current spread ± 0.05
   7. Submit → Wait for fill
   ```

   **Then close old F6 short:**
   ```
   BUY to cover 1 VXMF6 (front month)
   Order Type: LMT at current price ± 0.05
   ```

3. **Post-Roll Verification (10:15 ET)**

   Check positions in TWS:
   - [ ] VXMF6 = 0 (closed)
   - [ ] VXMG6 = -1 (new short)
   - [ ] VXMH6 = +1 (new long)
   - [ ] Net = 2 contracts total

   **If orphan leg detected → CLOSE AT MARKET IMMEDIATELY**

4. **Update Config (10:30 ET)**

   Edit `scripts/vrp_cs_daily_monitor.py` PositionConfig:
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

5. **Run Daily Digest to Confirm**
   ```bash
   python scripts/daily_digest.py
   cat data/daily_digest/digest_2026-01-14.md
   ```

### Fail-Closed Rules

**If any of these → DO NOT ROLL, ESCALATE:**
- Gate = CLOSED or REDUCE
- Contango < 0.50
- Spread market > 0.30 wide
- VIX > 30
- Fill takes > 15 minutes
- Orphan leg detected post-fill

### Roll Success Criteria
- [ ] Old spread closed (F6/G6 = 0)
- [ ] New spread open (G6/H6 = -1/+1)
- [ ] No orphan legs
- [ ] Config updated with new prices
- [ ] Daily digest reflects new position

---

## After Roll — Ongoing Operations

### Success Criteria (One-Time Definition)

**VRP-CS Success (per roll cycle)**:
- Roll executed without orphan legs
- No stop-loss triggered (spread < 2.04)
- Gate remained OPEN throughout
- P&L tracked in daily_log.csv

**VRP-ERP Success (daily)**:
- Regime correctly classified
- Position drift < 5%
- No missed signals

**System Success (daily)**:
- Daily digest generated
- No critical alerts
- Vol-target state < 3 days stale

### Daily Digest Schedule

Run once per trading day, after market close:

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate

# Standard digest (offline mode)
python scripts/daily_digest.py

# With live IBKR data (recommended)
python scripts/daily_digest.py --live

# View digest
cat data/daily_digest/digest_$(date +%Y-%m-%d).md
```

### Weekly Review Checklist

Every Friday afternoon:
- [ ] Review week's digests for alerts
- [ ] Check vol-target state freshness
- [ ] Verify all sleeve logs have entries
- [ ] Note any position drift > 5%
- [ ] Check upcoming roll dates (next 2 weeks)

---

## Quick Reference

### Key Commands

```bash
# Activate environment
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate

# VRP-ERP daily monitor
python scripts/vrp_erp_daily_monitor.py --live --base 10000

# VRP-CS daily monitor
python scripts/vrp_cs_daily_monitor.py --live

# Daily digest (offline)
python scripts/daily_digest.py

# Daily digest (with IBKR)
python scripts/daily_digest.py --live

# DM rebalance (first trading day of month)
DSP_RISK_SCALE=0.095 PYTHONPATH=src python -m dsp.cli --strict \
  -c config/dsp100k_etf_dual_momentum.yaml run
```

### Key Files

| File | Purpose |
|------|---------|
| `docs/VRP_CS_ROLL_RUNBOOK.md` | One-page roll execution checklist |
| `docs/VRP_CS_PAPER_TRADING_CHECKLIST.md` | Detailed VRP-CS documentation |
| `scripts/vrp_cs_daily_monitor.py` | VRP-CS position monitor |
| `scripts/vrp_erp_daily_monitor.py` | VRP-ERP regime monitor |
| `scripts/daily_digest.py` | Daily status digest generator |
| `data/vrp/paper_trading/daily_log.csv` | VRP-CS trade log |
| `data/vrp/paper_trading/vrp_erp_log.csv` | VRP-ERP trade log |
| `data/daily_digest/` | Daily digest archive |

### Roll Calendar 2026

| Roll # | Close VX1 | Open Short | Open Long | Roll-By |
|--------|-----------|------------|-----------|---------|
| **1** | VXMF6 (Jan) | VXMG6 (Feb) | VXMH6 (Mar) | **Jan 14** |
| 2 | VXMG6 (Feb) | VXMH6 (Mar) | VXMJ6 (Apr) | Feb 12 |
| 3 | VXMH6 (Mar) | VXMJ6 (Apr) | VXMK6 (May) | Mar 11 |
| 4 | VXMJ6 (Apr) | VXMK6 (May) | VXMM6 (Jun) | Apr 8 |

---

## Emergency Contacts

- **IBKR Support**: interactive-brokers.com/en/index.php?f=1560
- **TWS Issues**: Restart TWS, check API settings enabled

---

*Last updated: 2026-01-10*
*Print this manual. Keep it accessible during trading hours.*
