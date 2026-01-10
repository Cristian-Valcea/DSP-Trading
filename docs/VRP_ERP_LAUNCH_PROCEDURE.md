# VRP-ERP Paper Trading Launch Procedure

**Version**: 1.0
**Date**: 2026-01-09
**Strategy**: VRP-ERP (VIX-Regime-Gated SPY Exposure Scaling)
**Type**: Defensive Overlay (NOT alpha generator)
**Status**: READY FOR LAUNCH

---

## Executive Summary

VRP-ERP is a simple defensive strategy that scales SPY exposure based on VIX regime:
- **CALM** (VIX < 15): 100% SPY
- **ELEVATED** (15-20): 75% SPY
- **HIGH** (20-30): 50% SPY
- **CRISIS** (> 30): 25% SPY

**Why run this?** Complements VRP-CS by providing portfolio-level crisis protection.
VRP-CS generates alpha; VRP-ERP protects the equity allocation.

---

## Pre-Launch Checklist

### Step 1: Verify IBKR Setup

```bash
# Check that SPY is tradeable
# In TWS: Symbols → SPY → Verify quotes display
```

- [ ] IBKR paper account logged in (DU8009825)
- [ ] SPY quotes displaying in TWS
- [ ] US Stock trading permissions enabled
- [ ] Sufficient cash for initial position

### Step 2: Determine Base Allocation

For DSP-100K portfolio with €1.05M NAV:

| Allocation Model | Base SPY $ | At CALM (100%) | At CRISIS (25%) |
|-----------------|------------|----------------|-----------------|
| Conservative (5%) | $52,500 | $52,500 | $13,125 |
| Moderate (10%) | $105,000 | $105,000 | $26,250 |
| Paper Test (fixed) | $10,000 | $10,000 | $2,500 |

**Recommended for paper trading**: Start with fixed $10,000 base allocation.

- [ ] Base allocation selected: $__________

### Step 3: Check Current VIX Regime

```bash
# Get current VIX level
# Source: CBOE website, TWS, or Yahoo Finance

# Today's VIX: _______ (as of ______ ET)
```

| VIX Level | Regime | SPY % | With $10k Base |
|-----------|--------|-------|----------------|
| < 15 | CALM | 100% | $10,000 |
| 15-20 | ELEVATED | 75% | $7,500 |
| 20-30 | HIGH | 50% | $5,000 |
| > 30 | CRISIS | 25% | $2,500 |

- [ ] Current VIX: _______
- [ ] Current Regime: _______
- [ ] Target SPY Position: $_______

### Step 4: Calculate Target Shares

```python
# SPY is ~$595 as of Jan 2026 (approximate)
spy_price = 595.00  # Update with current price
target_allocation = 10000  # Your base allocation
regime_multiplier = 1.00   # 1.00 for CALM, 0.75/0.50/0.25 for others

target_value = target_allocation * regime_multiplier
target_shares = int(target_value / spy_price)

print(f"Target: {target_shares} shares = ${target_shares * spy_price:.2f}")
```

- [ ] Current SPY Price: $_______
- [ ] Target Shares: _______

---

## Launch Execution (Day 1)

### Timing
- **Best Window**: 09:35 - 10:15 ET (opening volatility settled)
- **Alternative**: 15:30 - 15:45 ET (MOC-style entry)

### Step 1: Run Monitor Script FIRST (Source of Truth)

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/vrp_erp_daily_monitor.py --live --base 10000
```

The script will:
1. Fetch SPY price and position from IBKR (if `--live` works)
2. Prompt for VIX (manual entry)
3. Calculate target shares: `floor(target_dollars / SPY_last)`
4. Show regime and action needed
5. Save state to `data/vrp/paper_trading/vrp_erp_state.json`

**If `--live` fails**: Fall back to manual entry mode (script will prompt).

### Step 2: Order Entry

**Use Limit Order at MID Price** (SPY is liquid - avoid systematic overpay)

```
Symbol: SPY
Action: BUY
Quantity: [target_shares from Step 1]
Order Type: LMT
Limit Price: [MID = (bid + ask) / 2]
Time in Force: DAY
```

**Why MID, not bid+0.01?**
- SPY has tight spreads ($0.01-0.02)
- Placing at MID usually fills within seconds
- Avoid systematic overpay on entries
- If not filling, move to MID + $0.01 (still better than bid)

### Step 3: Execute and Confirm

1. [ ] Run monitor script (`--live --base 10000`) → record target shares
2. [ ] In TWS: Note SPY bid/ask spread
3. [ ] Calculate MID price: `(bid + ask) / 2`
4. [ ] Enter LMT order at MID
5. [ ] Submit order
6. [ ] Wait for fill (usually <30 seconds)
7. [ ] Record fill details below

### Step 4: Post-Fill Verification

**Re-run monitor to confirm drift ≈ 0:**
```bash
python scripts/vrp_erp_daily_monitor.py --live --base 10000
```

Expected output:
- `Drift: 0.0%` (or very close)
- State saved to `vrp_erp_state.json`
- Log appended to `vrp_erp_log.csv`

### Fill Record

| Field | Value |
|-------|-------|
| Date | |
| Time (ET) | |
| Shares Filled | |
| Fill Price | |
| Total Cost | |
| Commission | |
| Slippage (vs mid) | |
| Actual Exposure ($) | |

---

## Daily Monitoring

### Morning Check (Before 9:30 ET)

1. **Get VIX Level**
   - Source: CBOE, TWS, or run monitoring script
   - VIX: _______

2. **Determine Regime**
   | VIX Range | Regime |
   |-----------|--------|
   | < 15 | CALM |
   | 15-20 | ELEVATED |
   | 20-30 | HIGH |
   | > 30 | CRISIS |

3. **Calculate Target Position**
   ```
   Target $ = Base × Regime Multiplier
   Target Shares = Target $ / SPY Price
   ```

4. **Compare to Current**
   - Current Shares: _______
   - Target Shares: _______
   - Drift %: _______

5. **Action Decision**
   - [ ] No action needed (drift < 5%)
   - [ ] Rebalance needed (regime change or drift > 5%)

### Rebalancing Rules

**ONLY rebalance if:**
1. Regime changed AND held for 2+ consecutive days, OR
2. Position drift > 5% from target

**2-Day Confirmation Rule:**
```
Day 1: VIX crosses threshold → Note transition
Day 2: VIX still in new regime → OK to rebalance
Day 2: VIX back in old regime → Cancel transition
```

This prevents whipsaws at regime boundaries.

### Rebalancing Execution

**To Increase SPY (buy):**
```
Action: BUY
Quantity: [target_shares - current_shares]
Order Type: LMT at MID price
```

**To Decrease SPY (sell):**
```
Action: SELL
Quantity: [current_shares - target_shares]
Order Type: LMT at MID price
```

**Why MID price?** SPY is extremely liquid. MID fills quickly and avoids systematic slippage.

**Preferred Timing**: 09:35-10:15 ET or 15:30-15:45 ET

---

## Monitoring Script

Run daily for status report:

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/vrp_erp_daily_monitor.py
```

The script will:
1. Display current VIX and regime
2. Calculate target SPY position
3. Compare to actual (if connected to IBKR)
4. Flag any rebalancing needed
5. Log to data/vrp/paper_trading/vrp_erp_log.csv

---

## Record Keeping

### Daily Log Format

| Date | VIX | Regime | Target % | Actual % | Drift | Action | Notes |
|------|-----|--------|----------|----------|-------|--------|-------|
| 2026-01-10 | 14.8 | CALM | 100% | 100% | 0% | None | Initial |
| | | | | | | | |

### Weekly Summary

| Week | Avg VIX | Regime Days | Rebalances | SPY Return | Strategy Return |
|------|---------|-------------|------------|------------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |

---

## Integration with VRP-CS

Both strategies run in parallel:

| Check | VRP-CS | VRP-ERP |
|-------|--------|---------|
| Daily monitoring | Spread, gate, roll | VIX regime, drift |
| Action frequency | Daily | Weekly (unless regime change) |
| Instruments | VXM futures | SPY ETF |
| Risk type | VIX term structure | Equity exposure |
| Correlation | ~0.1 to SPY | ~0.5 to SPY |

**Shared Signal**: VRP Regime Gate
- Gate OPEN → VRP-ERP probably in CALM/ELEVATED
- Gate CLOSED → VRP-ERP probably in HIGH/CRISIS

---

## Paper Trading Success Criteria (4 Weeks)

| Criterion | Target | Week 1 | Week 2 | Week 3 | Week 4 |
|-----------|--------|--------|--------|--------|--------|
| Regime tracking accuracy | 100% | | | | |
| Rebalance execution | 100% | | | | |
| No missed signals | 100% | | | | |
| Drift < 5% maintained | 100% | | | | |
| No whipsaws (2-day rule) | 100% | | | | |
| SPY fill quality | < 2 ticks | | | | |

---

## Quick Reference Commands

```bash
# Daily monitoring
python scripts/vrp_erp_daily_monitor.py

# Check VIX regime only
python -c "
vix = 14.8  # Update with current VIX
if vix < 15: print('CALM - 100%')
elif vix < 20: print('ELEVATED - 75%')
elif vix < 30: print('HIGH - 50%')
else: print('CRISIS - 25%')
"

# Calculate target shares
python -c "
spy_price = 595.0  # Update with current price
base = 10000       # Your base allocation
vix = 14.8         # Current VIX

if vix < 15: mult = 1.00
elif vix < 20: mult = 0.75
elif vix < 30: mult = 0.50
else: mult = 0.25

target = int(base * mult / spy_price)
print(f'Target: {target} shares @ \${spy_price} = \${target * spy_price:.2f}')
"

# View log
cat data/vrp/paper_trading/vrp_erp_log.csv | column -s, -t
```

---

## Risk Reminders

### VRP-ERP is NOT:
- An alpha generator (it reduces returns in bull markets)
- A pure hedge (still ~50% correlated to SPY)
- A replacement for VRP-CS (they serve different purposes)

### VRP-ERP IS:
- A defensive overlay that reduces drawdowns
- Simple to operate (just SPY scaling)
- Complementary to VRP-CS

### Warning Signs:
- Whipsawing between regimes (>2 changes per week)
- Large drift accumulating (>10%)
- VIX stuck at threshold (e.g., 15.0-15.5 for days)

---

## Emergency Procedures

### Market Crash (VIX > 40)
1. Immediately reduce to 25% (CRISIS level)
2. Do NOT wait for 2-day confirmation
3. Document the override

### Flash Crash
1. Do NOT trade during extreme volatility
2. Wait for VIX to stabilize
3. Reassess regime after 1 hour

### System Failure
1. Check SPY position manually in TWS
2. Calculate target offline using quick reference
3. Execute manual rebalance if needed

---

*Launch Procedure created: 2026-01-09*
*Strategy: VRP-ERP Defensive Overlay*
*Run in parallel with VRP-CS*
