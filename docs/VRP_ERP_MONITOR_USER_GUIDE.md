# VRP-ERP Daily Monitor Script - User Guide

**Script**: `scripts/vrp_erp_daily_monitor.py`
**Purpose**: Calculate target SPY position based on VIX regime and track daily rebalancing signals
**Strategy**: VRP-ERP (Equity Risk Premium defensive overlay)

---

## Quick Start

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/vrp_erp_daily_monitor.py --live --base 10000
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--live` | Fetch SPY price and position from IBKR (still prompts for VIX) | Off (manual entry) |
| `--base AMOUNT` | Base allocation in dollars | $10,000 |
| `--no-log` | Skip appending to daily log CSV | Off (log enabled) |

### Examples

```bash
# Standard daily check (recommended)
python scripts/vrp_erp_daily_monitor.py --live --base 10000

# Manual entry mode (IBKR not available)
python scripts/vrp_erp_daily_monitor.py --base 10000

# Larger allocation
python scripts/vrp_erp_daily_monitor.py --live --base 50000

# Quick check without logging
python scripts/vrp_erp_daily_monitor.py --live --base 10000 --no-log
```

---

## Input Prompts Explained

The script will prompt you for market data. **Enter NUMBERS ONLY** - no symbols, no text, no order commands.

### Prompt 1: VIX Spot

```
VIX Spot: _
```

**What to enter**: The current VIX index value (a decimal number)

**Where to find VIX**:
- **TWS**: Search for "VIX" or "$VIX.X"
- **CBOE**: https://www.cboe.com/tradable_products/vix/
- **Yahoo Finance**: https://finance.yahoo.com/quote/%5EVIX/
- **Google**: Search "VIX index"

**Examples**:
| VIX Level | What to Type | Regime |
|-----------|--------------|--------|
| 12.50 | `12.50` | CALM |
| 14.80 | `14.80` | CALM |
| 16.50 | `16.50` | ELEVATED |
| 18.25 | `18.25` | ELEVATED |
| 22.00 | `22.00` | HIGH |
| 35.00 | `35.00` | CRISIS |

**Common Mistakes**:
- ❌ `VIX 16.5` - Don't include text
- ❌ `$16.50` - Don't include dollar sign
- ❌ `16,50` - Use period, not comma
- ✅ `16.50` - Correct!

---

### Prompt 2: SPY Price (Manual Mode Only)

```
SPY Price: _
```

**What to enter**: Current SPY ETF price (a decimal number)

**Where to find SPY price**:
- **TWS**: Search for "SPY" → look at Last/Mid price
- **Yahoo Finance**: https://finance.yahoo.com/quote/SPY
- **Google**: Search "SPY stock price"

**Examples**:
| SPY Price | What to Type |
|-----------|--------------|
| $594.24 | `594.24` |
| $595.00 | `595` or `595.00` |
| $598.50 | `598.50` |

**Common Mistakes**:
- ❌ `$594.24` - Don't include dollar sign
- ❌ `SPY 594.24` - Don't include ticker
- ❌ `594,24` - Use period, not comma
- ✅ `594.24` - Correct!

**Note**: If using `--live` mode, SPY price is fetched automatically from IBKR.

---

### Prompt 3: Current SPY Shares

```
Current SPY shares (0 if none): _
```

**What to enter**: Number of SPY shares you currently hold (whole number)

**Where to find**:
- **TWS**: Account → Portfolio → SPY position
- **If starting fresh (Day 1)**: Enter `0`

**Examples**:
| Position | What to Type |
|----------|--------------|
| No position | `0` |
| 12 shares | `12` |
| 16 shares | `16` |

**Common Mistakes**:
- ❌ `12 shares` - Don't include "shares"
- ❌ `SPY: 12` - Don't include ticker
- ❌ `-5` - Position should be positive (long SPY only)
- ✅ `12` - Correct!

**Note**: If using `--live` mode, position is fetched automatically from IBKR.

---

### Prompt 4: VIX (Live Mode Only)

When using `--live`, after fetching SPY data from IBKR:

```
SPY Price: $594.24
SPY Position: 12 shares
Enter current VIX: _
```

**What to enter**: Same as Prompt 1 - just the VIX number

The script fetches SPY data from IBKR but still needs manual VIX input because VIX isn't easily available through the standard IBKR API.

---

## Complete Example Session

### Scenario: Day 1 Launch (No Existing Position)

```
$ python scripts/vrp_erp_daily_monitor.py --live --base 10000

Fetching live quotes from IBKR...

SPY Price: $594.24
SPY Position: 0 shares
Enter current VIX: 16.5

============================================================
  VRP-ERP DAILY MONITORING REPORT
  2026-01-09 14:35:00
============================================================

--- VIX REGIME ---
VIX Spot:    16.50
Regime:      ELEVATED (75% exposure)

  Thresholds: CALM (<15), ELEVATED (15-20), HIGH (20-30), CRISIS (>30)

--- SPY POSITION ---
SPY Price:       $594.24
Current Shares:  0
Current Value:   $0.00

--- TARGET vs ACTUAL ---
Base Allocation: $10,000.00
Target Value:    $7,500.00 (75%)
Target Shares:   12
Actual Shares:   0
Drift:           100.0%

Delta:           BUY 12 shares

--- REBALANCING ---
SIGNAL: REBALANCE NEEDED
Reason: Drift 100.0% exceeds 5% threshold

  >>> BUY 12 SPY shares <<<

--- ACTION REQUIRED ---
  -> BUY 12 SPY shares
  -> Use LMT order at MID price (SPY is liquid - avoid systematic overpay)
  -> Best timing: 09:35-10:15 ET or 15:30-15:45 ET

============================================================

Log appended to: data/vrp/paper_trading/vrp_erp_log.csv
```

### Scenario: Daily Check (Position Exists, No Rebalance Needed)

```
$ python scripts/vrp_erp_daily_monitor.py --live --base 10000

Fetching live quotes from IBKR...

SPY Price: $595.50
SPY Position: 12 shares
Enter current VIX: 16.2

============================================================
  VRP-ERP DAILY MONITORING REPORT
  2026-01-10 09:00:00
============================================================

--- VIX REGIME ---
VIX Spot:    16.20
Regime:      ELEVATED (75% exposure)
Days in Regime: 2

  Thresholds: CALM (<15), ELEVATED (15-20), HIGH (20-30), CRISIS (>30)

--- SPY POSITION ---
SPY Price:       $595.50
Current Shares:  12
Current Value:   $7,146.00
Avg Cost:        $594.24
Unrealized P&L:  $+15.12

--- TARGET vs ACTUAL ---
Base Allocation: $10,000.00
Target Value:    $7,500.00 (75%)
Target Shares:   12
Actual Shares:   12
Drift:           0.0%

--- REBALANCING ---
Status: No rebalancing needed

--- ACTION REQUIRED ---
  None - Position is on target

============================================================

Log appended to: data/vrp/paper_trading/vrp_erp_log.csv
```

---

## Understanding the Output

### VIX Regime Section

| Regime | VIX Range | SPY Exposure | Meaning |
|--------|-----------|--------------|---------|
| **CALM** | < 15 | 100% | Low volatility, full equity exposure |
| **ELEVATED** | 15 - 20 | 75% | Slight risk-off signal |
| **HIGH** | 20 - 30 | 50% | Moderate risk-off |
| **CRISIS** | > 30 | 25% | Maximum defense |

### Target Calculation

```
Target Value = Base Allocation × Regime Multiplier
Target Shares = floor(Target Value / SPY Price)
```

**Example** (VIX = 16.5, Base = $10,000, SPY = $594.24):
```
Regime = ELEVATED (75%)
Target Value = $10,000 × 0.75 = $7,500
Target Shares = floor($7,500 / $594.24) = floor(12.62) = 12 shares
```

### Drift Calculation

```
Drift = |Actual Shares - Target Shares| / Target Shares
```

**Example** (Target = 12, Actual = 10):
```
Drift = |10 - 12| / 12 = 2/12 = 16.7%
```

### Rebalance Signals

The script triggers rebalancing when:
1. **Drift > 5%**: Position has drifted too far from target
2. **Regime change confirmed**: VIX regime changed AND held for 2+ days

**2-Day Confirmation Rule**: Prevents whipsaws at regime boundaries. If VIX crosses a threshold, the script waits 2 consecutive days before recommending rebalancing.

---

## Output Files

### State File: `data/vrp/paper_trading/vrp_erp_state.json`

Tracks the current regime state for the 2-day confirmation rule.

```json
{
  "regime": "ELEVATED",
  "vix": 16.50,
  "days_in_regime": 2,
  "last_transition_date": "2026-01-09"
}
```

| Field | Description |
|-------|-------------|
| `regime` | Current VIX regime |
| `vix` | Last recorded VIX value |
| `days_in_regime` | Consecutive days in this regime |
| `last_transition_date` | When regime last changed |

### Log File: `data/vrp/paper_trading/vrp_erp_log.csv`

Daily tracking log with all inputs and calculations.

| Column | Description | Example |
|--------|-------------|---------|
| `date` | Date of check | `2026-01-09` |
| `time` | Time of check | `14:35:00` |
| `vix` | VIX value entered | `16.50` |
| `spy_price` | SPY price | `594.24` |
| `regime` | VIX regime | `ELEVATED` |
| `target_shares` | Calculated target | `12` |
| `actual_shares` | Current position | `0` |
| `actual_exposure_usd` | Position value | `0.00` |
| `avg_cost` | Average cost (from IBKR) | `594.24` |
| `fill_price` | Fill price if traded | (manual entry) |
| `drift_pct` | Drift percentage | `100.00%` |
| `rebalance_signal` | Whether to rebalance | `True` |
| `action_taken` | What you did | (manual entry) |
| `notes` | Any notes | (manual entry) |

---

## Troubleshooting

### Error: "Invalid input. Using placeholder values."

**Cause**: You entered non-numeric text at a prompt.

**Solution**: Enter numbers only. Examples:
- VIX: `16.5` (not "VIX 16.5")
- SPY: `594.24` (not "$594.24")
- Shares: `12` (not "12 shares")

### Error: "IBKR connection failed"

**Cause**: IBKR TWS/Gateway not running or not accessible.

**Solution**:
1. Ensure TWS or IB Gateway is running
2. Check it's logged in
3. Check API is enabled (Edit → Global Configuration → API → Enable)
4. Check port is 7497 (paper trading)
5. Fall back to manual mode (run without `--live`)

### Output Shows Wrong Target Shares

**Cause**: Likely entered wrong SPY price.

**Check**: SPY should be ~$590-600 range (as of Jan 2026). If you entered $60 or $6000, the calculation will be wrong.

**Solution**: Re-run script with correct SPY price.

### Script Says "Regime changing (1/2 days)" - No Action

**Cause**: VIX crossed a regime threshold but 2-day confirmation hasn't completed.

**What This Means**:
- The strategy waits 2 days before acting on regime changes
- This prevents whipsaws from temporary VIX spikes
- Check again tomorrow - if regime persists, rebalancing will trigger

**Special Case - Day 1 Launch**: If this is your first day and you have 0 shares, you may want to establish an initial position anyway. In that case, manually calculate:
```
Target = floor(Base × Regime_Multiplier / SPY_Price)
```

---

## Day 1 Launch vs Daily Monitoring

### Day 1 Launch (No Previous State)

When no state file exists:
- Script treats it as fresh start
- High drift (100%) triggers rebalance signal immediately
- Establish your initial position

### Daily Monitoring (State Exists)

When state file exists:
- Script tracks days in regime
- 2-day confirmation rule applies for regime changes
- Only drift > 5% OR confirmed regime change triggers rebalancing

### Resetting for Fresh Start

If you need to reset (e.g., corrupted test data):

```bash
# Delete state file
rm data/vrp/paper_trading/vrp_erp_state.json

# Optionally clear log
echo "date,time,vix,spy_price,regime,target_shares,actual_shares,actual_exposure_usd,avg_cost,fill_price,drift_pct,rebalance_signal,action_taken,notes" > data/vrp/paper_trading/vrp_erp_log.csv
```

---

## Order Execution Guide

When the script says **"BUY X SPY shares"**:

### Step 1: Get MID Price in TWS

1. Open TWS
2. Search for SPY
3. Note the **Bid** and **Ask** prices
4. Calculate **MID** = (Bid + Ask) / 2

**Example**:
- Bid: $594.20
- Ask: $594.24
- MID: ($594.20 + $594.24) / 2 = **$594.22**

### Step 2: Place Limit Order

```
Symbol:      SPY
Action:      BUY (or SELL if reducing)
Quantity:    [number from script]
Order Type:  LMT
Limit Price: [MID price calculated]
Time Force:  DAY
```

### Step 3: Wait for Fill

SPY is very liquid. At MID price, expect fill within 30 seconds during market hours.

If not filling after 1 minute:
- Adjust to MID + $0.01 for BUY orders
- Adjust to MID - $0.01 for SELL orders

### Step 4: Record Fill

After fill, optionally update the CSV log with:
- `fill_price`: Actual fill price
- `action_taken`: "BUY 12" or "SELL 4"
- `notes`: Any relevant notes

---

## Quick Reference Card

### VIX Regime Thresholds

| VIX Level | Regime | SPY Allocation |
|-----------|--------|----------------|
| < 15 | CALM | 100% |
| 15 - 20 | ELEVATED | 75% |
| 20 - 30 | HIGH | 50% |
| > 30 | CRISIS | 25% |

### Target Shares Formula

```
Target Shares = floor(Base × Multiplier / SPY_Price)
```

### Daily Workflow

```bash
# Morning check (before 9:30 ET)
python scripts/vrp_erp_daily_monitor.py --live --base 10000

# If rebalance needed:
# 1. Calculate MID price in TWS
# 2. Place LMT order at MID
# 3. Wait for fill
# 4. Re-run script to verify drift ≈ 0%
```

---

*User Guide Version: 1.0*
*Created: 2026-01-09*
*Strategy: VRP-ERP Defensive Overlay*
