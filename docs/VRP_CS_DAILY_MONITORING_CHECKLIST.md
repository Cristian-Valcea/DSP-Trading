# VRP-CS Daily Monitoring Checklist

**Version**: 1.0
**Date**: 2026-01-09
**Strategy**: VRP Calendar Spread (Long VX2, Short VX1)
**Paper Trading Status**: LIVE since 2026-01-09

---

## Current Position Reference

| Contract | Position | Entry Price | Role |
|----------|----------|-------------|------|
| **VXMF6** | -1 | 16.24 | Short front month (Jan) |
| **VXMG6** | +1 | 17.87 | Long back month (Feb) |

| Metric | Value |
|--------|-------|
| Entry Spread | 1.63 |
| Stop-Loss | > 2.04 (25% wider) |
| Take-Profit | < 0.82 (50% narrower) |
| Roll By | 2026-01-14 |
| VX1 Expiry | 2026-01-21 |

---

## Daily Monitoring Script

Run this script daily to get a full status report:

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/vrp_cs_daily_monitor.py
```

**With live IBKR quotes:**
```bash
python scripts/vrp_cs_daily_monitor.py --live
```

The script will:
1. Calculate current spread and P&L
2. Check stop-loss/take-profit triggers
3. Check VRP Regime Gate status
4. Check roll timing
5. Append record to `data/vrp/paper_trading/daily_log.csv`

---

## Pre-Market Checklist (Before 9:30 ET)

### Step 1: Check VIX Spot Level
- [ ] Open CBOE website or TWS
- [ ] Record VIX spot value: _______
- [ ] **PASS** if VIX < 30
- [ ] **WARN** if VIX 30-40 (watch closely)
- [ ] **FAIL** if VIX > 40 (no new positions)

### Step 2: Check VX Term Structure
- [ ] Record VXM front month (F6): _______
- [ ] Record VXM back month (G6): _______
- [ ] Calculate spread: VX2 - VX1 = _______
- [ ] **PASS** if spread < 2.04 (stop-loss not hit)
- [ ] **FAIL** if spread >= 2.04 (exit required)

### Step 3: Calculate Contango
- [ ] Contango % = (VX1 - VIX) / VIX × 100 = _______%
- [ ] **PASS** if contango > 2%
- [ ] **WARN** if contango 0-2%
- [ ] **FAIL** if contango < 0% (backwardation - crisis signal)

### Step 4: Check Regime Gate
- [ ] Run: `python -c "from dsp.regime.vrp_regime_gate import VRPRegimeGate; g=VRPRegimeGate(); print(g.update(vix=___, vvix=___, vx_f1=___))"`
- [ ] Gate State: _______
- [ ] **PASS** if OPEN
- [ ] **WARN** if REDUCE
- [ ] **FAIL** if CLOSED (exit required)

### Step 5: Check Roll Timing
- [ ] Days until roll date (Jan 14): _______
- [ ] **PASS** if > 2 days
- [ ] **WARN** if 1-2 days (prepare roll)
- [ ] **ACTION** if 0 days (execute roll today)

---

## Intraday Monitoring (9:30 - 16:00 ET)

Check these items 2-3 times during the day (e.g., 10:00, 13:00, 15:30):

### Spread Check
- [ ] Current spread: _______
- [ ] Entry spread: 1.63
- [ ] Spread change: _______ (+ means widening, - means narrowing)

### P&L Check (approximate)
```
P&L = -(Spread Change) × $100 × Position Size
P&L = -(Current - 1.63) × $100 × 1
```
- [ ] Unrealized P&L: $_______

### Exit Trigger Check
- [ ] Spread > 2.04? (stop-loss) _______
- [ ] Spread < 0.82? (take-profit) _______
- [ ] Gate → CLOSED? _______

**If ANY trigger hit**: Execute exit immediately via TWS

---

## Post-Market Checklist (After 16:00 ET)

### Step 1: Record Daily Metrics
- [ ] Closing spread: _______
- [ ] Daily P&L: $_______
- [ ] Gate state at close: _______

### Step 2: Run Monitoring Script
```bash
python scripts/vrp_cs_daily_monitor.py
```
- [ ] Script completed successfully
- [ ] Log appended to daily_log.csv

### Step 3: Review Tomorrow's Action
- [ ] Roll due tomorrow? _______
- [ ] Any concerns for overnight? _______

---

## Exit Procedures

### Stop-Loss Exit (Spread > 2.04)

1. Open TWS
2. Navigate to VXM positions
3. **Close Short Leg First** (buy back VXMF6):
   - Symbol: VXMF6
   - Action: BUY
   - Quantity: 1
   - Order Type: MKT (for speed)
4. **Close Long Leg** (sell VXMG6):
   - Symbol: VXMG6
   - Action: SELL
   - Quantity: 1
   - Order Type: MKT

**Alternative - Combo Order:**
- Trading → Combo/Strategy → Close Position
- Select both legs
- SELL the spread

### Take-Profit Exit (Spread < 0.82)

Same procedure as stop-loss, but less urgency.
Can use LMT orders to capture better fills.

### Gate Closure Exit

Same procedure as stop-loss.
Exit immediately - gate closure signals potential crisis.

---

## Roll Procedure (When DTE ≤ 5 days)

### Pre-Roll Checklist
- [ ] Verify entry conditions for new spread:
  - VIX < 30? _______
  - Contango > 2%? _______
  - Gate OPEN? _______

### Roll Execution Steps

1. **Exit Current Spread:**
   ```
   BUY 1 VXMF6 (close short front month)
   SELL 1 VXMG6 (close long back month)
   ```

2. **Re-evaluate Conditions:**
   - If conditions NOT met → Stay flat
   - If conditions met → Proceed to step 3

3. **Enter New Spread:**
   ```
   SELL 1 VXMG6 (new front month - was back month)
   BUY 1 VXMH6 (new back month - Mar)
   ```

4. **Update Position Config:**
   - Edit scripts/vrp_cs_daily_monitor.py (PositionConfig class)
   - Or create config JSON file

### Post-Roll Verification
- [ ] Old positions closed (VXMF6 = 0, old VXMG6 closed)
- [ ] New positions established
- [ ] New entry spread recorded: _______
- [ ] New stop-loss calculated: entry × 1.25 = _______
- [ ] New take-profit calculated: entry × 0.50 = _______

---

## Jan 14, 2026 Roll Rehearsal

**Target Date**: January 14, 2026 (Tuesday)
**VX1 Expiry**: January 21, 2026 (Wednesday)

### Pre-Roll Day (Jan 13)
- [ ] Verify VXMG6 quotes available (Feb contract)
- [ ] Verify VXMH6 quotes available (Mar contract)
- [ ] Calculate expected new spread
- [ ] Review roll procedure steps
- [ ] Set reminder for roll execution

### Roll Day Workflow
1. **07:00 ET**: Pre-market prep
2. **09:35-10:00 ET**: Execute roll during stable period
3. **By 10:30 ET**: Verify new position established
4. **Post-market**: Update monitoring docs

---

## Quick Reference Commands

```bash
# Daily monitoring report
python scripts/vrp_cs_daily_monitor.py

# With live IBKR quotes
python scripts/vrp_cs_daily_monitor.py --live

# Check regime gate state
python -c "
from dsp.regime.vrp_regime_gate import VRPRegimeGate
g = VRPRegimeGate()
state = g.update(vix=15.0, vvix=85.0, vx_f1=16.5)
print(f'State: {state}, Score: {g.last_score:.3f}')
"

# View daily log
cat data/vrp/paper_trading/daily_log.csv | column -s, -t

# View last 5 entries
tail -5 data/vrp/paper_trading/daily_log.csv | column -s, -t
```

---

## Alerts & Notifications

### Critical Alerts (Exit Required)
- Spread > 2.04 (stop-loss)
- Spread < 0.82 (take-profit)
- Gate → CLOSED
- VIX > 40

### Warning Alerts (Monitor Closely)
- Spread > 1.90 (approaching stop-loss)
- Contango < 2%
- Gate → REDUCE
- Days to roll ≤ 2

### Calendar Reminders
- [ ] Jan 13: Pre-roll preparation
- [ ] Jan 14: ROLL EXECUTION DAY
- [ ] Feb 12: Pre-roll for Feb expiry (if position active)

---

## Metrics Tracking (4-Week Paper Trading)

| Criterion | Target | Week 1 | Week 2 | Week 3 | Week 4 |
|-----------|--------|--------|--------|--------|--------|
| Position tracking accuracy | 100% | | | | |
| Roll execution success | 100% | | | | |
| Gate signal reliability | > 95% | | | | |
| No unintended overnight positions | 100% | | | | |
| Slippage < 2 ticks avg | Yes | | | | |

---

*Checklist created: 2026-01-09*
*First roll: 2026-01-14*
*Strategy: VRP Calendar Spread - Paper Trading*
