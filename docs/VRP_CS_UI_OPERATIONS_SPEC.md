# VRP-CS UI Operations Specification

**Version**: 1.1
**Date**: 2026-01-14
**Author**: Claude
**Status**: ✅ APPROVED - Ready for Implementation

---

## 1. Overview

### 1.1 Goal
Enable all VRP-CS (Calendar Spread) operations to be performed directly from the web UI without needing to use bash/terminal commands.

### 1.2 Current State

| Operation | Currently Available | Location |
|-----------|---------------------|----------|
| Monitor (live IBKR) | ✅ Yes | "Run Monitor (Live)" button |
| Monitor (manual values) | ✅ Yes | Manual input form + button |
| View position status | ✅ Yes | VRP-CS panel shows spread, P&L, roll date |
| **Close spread** | ✅ Yes | "Trade Ops" buttons |
| **Open new spread** | ✅ Yes | "Trade Ops" buttons |
| **Execute roll** | ✅ Yes | "Trade Ops" buttons |
| Update position config | ✅ Yes | "Edit Config" modal |

### 1.3 Desired State

All VRP-CS lifecycle operations accessible from UI:
1. **Monitor** - Already done ✅
2. **Roll Spread** - Combined close + open (priority #1)
3. **Close Spread** - Exit current position (priority #2)
4. **Update Config** - Save new position details (priority #3)
5. **Open Spread** - Enter new position (priority #4)

---

## 2. Approved Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Contract Selection** | Dropdown (next 6-12 months) + Custom override | Covers normal cases + flexibility for edge cases |
| **Roll Method** | 2-order runbook style (preferred) + 4-leg fallback | Matches VRP_CS_ROLL_RUNBOOK.md |
| **Order Type** | LIMIT with ±0.05 buffer (default) | Per runbook; MARKET only via emergency toggle |
| **Config Storage** | `data/vrp/paper_trading/position_config.json` | No editing Python files |
| **API Pattern** | Extend `/api/dsp/run/{action}` + dedicated `/api/dsp/vrp_cs/config` | Consistent with existing sleeve_c pattern |
| **Priority** | Roll → Close → Config Editor → Open | Roll is needed TODAY |

---

## 3. Roll Math Correction

### 3.1 Current Position
```
Short: -1 VXMF6 (Jan front month)
Long:  +1 VXMG6 (Feb back month)
```

### 3.2 Target Position After Roll
```
Short: -1 VXMG6 (Feb, becomes new front month)
Long:  +1 VXMH6 (Mar, new back month)
```

### 3.3 Net Trades Required

| Contract | Current | Target | Net Trade |
|----------|---------|--------|-----------|
| VXMF6 | -1 | 0 | **BUY 1** (close short) |
| VXMG6 | +1 | -1 | **SELL 2** (close long + open short) |
| VXMH6 | 0 | +1 | **BUY 1** (open long) |

**Total: 4 contracts traded across 3 symbols**

### 3.4 Execution Methods (per VRP_CS_ROLL_RUNBOOK.md)

**Option A (IMPLEMENTED): Two calendar spread combo orders (2 orders, 4 legs)**

1. **Close old spread**: `SELL 1` calendar spread `VXMF6/VXMG6`  
   → legs: `BUY 1 VXMF6` + `SELL 1 VXMG6`
2. **Open new spread**: `BUY 1` calendar spread `VXMG6/VXMH6`  
   → legs: `SELL 1 VXMG6` + `BUY 1 VXMH6`

This is the default implementation because it uses standard 1:1 calendar combos (simple limit pricing and fewer automation foot-guns).

**Option B (FUTURE): Ratio BAG (single combo + one leg)**

`BUY 1 VXMF6` + `BAG: SELL 2 VXMG6 + BUY 1 VXMH6`  
Not implemented yet (needs careful combo pricing/fill handling).

---

## 4. Operations to Implement

### 4.1 Roll Spread (PRIORITY #1)

**Purpose**: Execute complete roll from current to next month spread

**Pre-Roll Checks** (automated, from runbook):
| Check | Condition | Action If Fails |
|-------|-----------|-----------------|
| Gate State | = OPEN | Block unless override |
| VIX | < 30 | Block unless override |
| Spread sanity | VX2 - VX1 > 0.50 | Block unless override |
| Position Match | Confirm current position in IBKR | Block unless override |

Additional runbook checks (bid/ask width, time window, margin headroom) are planned but not yet enforced by the first implementation.

**UI Elements**:
```
┌─────────────────────────────────────────────────────────┐
│ 🔄 ROLL SPREAD                                          │
├─────────────────────────────────────────────────────────┤
│ Current Position (from IBKR):                           │
│   Short: VXMF6 @ 16.41 (-1)                            │
│   Long:  VXMG6 @ 18.18 (+1)                            │
│   Spread: 1.77                                          │
├─────────────────────────────────────────────────────────┤
│ Roll Destination:                                       │
│   New Short: [VXMG6 ▼] (auto-filled)                   │
│   New Long:  [VXMH6 ▼] (auto-filled)                   │
│   Est. Spread: ~1.85 (live quote)                      │
├─────────────────────────────────────────────────────────┤
│ Pre-Roll Checks:                                        │
│   ✅ Gate: OPEN                                         │
│   ✅ Contango: 1.77 > 0.50                             │
│   ✅ Bid/Ask F6: 0.05 ≤ 0.15                           │
│   ✅ Bid/Ask G6: 0.08 ≤ 0.15                           │
│   ✅ Bid/Ask H6: 0.10 ≤ 0.20                           │
│   ✅ Position Match: Confirmed                          │
│   ⚠️ Time: 04:30 ET (outside 09:35-15:45 window)       │
├─────────────────────────────────────────────────────────┤
│ Execution Method:                                       │
│   ○ 2-Order Combo (Preferred)                          │
│   ○ 4-Leg Explicit (Fallback)                          │
├─────────────────────────────────────────────────────────┤
│ Order Type:                                             │
│   ● LIMIT ±0.05 (Default)                              │
│   ○ MARKET (Emergency - extra confirmation)            │
├─────────────────────────────────────────────────────────┤
│ [Run Pre-Checks]  [Preview Orders]  [Execute Roll ▶]   │
└─────────────────────────────────────────────────────────┘
```

**Backend Logic**:
```
1. Fetch current IBKR positions
2. Validate position matches config
3. Run all pre-roll checks
4. If any critical check fails → block with explanation
5. If warning checks fail → show warning, require acknowledgment
6. Generate orders based on method:
   - 2-Order: [BUY 1 old_front], [BAG: SELL 2 middle + BUY 1 new_back]
   - 4-Leg: [BUY 1 old_front], [SELL 1 old_back], [SELL 1 new_front], [BUY 1 new_back]
7. If dry-run: show orders with prices
8. If execute: place orders, poll for fills, timeout at 15 min
9. Post-roll: verify positions, update config, log results
```

---

### 4.2 Close Spread (PRIORITY #2)

**Purpose**: Exit current position completely

**UI Elements**:
- Button: "❌ Close Spread" (warning color)
- Shows current position from IBKR
- Confirmation dialog with estimated P&L

**Orders Generated**:
```
BUY 1 [short_contract]  (close short)
SELL 1 [long_contract]  (close long)
```

**Or as combo**: SELL 1 spread (atomic close)

---

### 4.3 Config Editor (PRIORITY #3)

**Purpose**: Update position config after trades (or manual corrections)

**Config File**: `data/vrp/paper_trading/position_config.json`

**UI Elements**:
```
┌─────────────────────────────────────────────────────────┐
│ ⚙️ POSITION CONFIG EDITOR                               │
├─────────────────────────────────────────────────────────┤
│ Contracts:                                              │
│   Short Contract: [VXMG6    ]                          │
│   Long Contract:  [VXMH6    ]                          │
├─────────────────────────────────────────────────────────┤
│ Entry Prices:                                           │
│   Short Entry: [17.50  ]                               │
│   Long Entry:  [18.35  ]                               │
│   Entry Date:  [2026-01-14]                            │
├─────────────────────────────────────────────────────────┤
│ Calculated:                                             │
│   Entry Spread: 0.85 (auto-calculated)                 │
│   Stop-Loss:    1.06 (125% of entry)                   │
│   Take-Profit:  0.43 (50% of entry)                    │
├─────────────────────────────────────────────────────────┤
│ Roll Schedule:                                          │
│   VX1 Expiry:  [2026-02-19]                            │
│   Roll-By:     [2026-02-12] (auto: N-5 days)           │
├─────────────────────────────────────────────────────────┤
│ [Load from IBKR Fills]  [Calculate]  [Save Config]     │
└─────────────────────────────────────────────────────────┘
```

**"Load from IBKR Fills" button**: Fetches recent fill prices to auto-populate entry prices.

---

### 4.4 Open Spread (PRIORITY #4)

**Purpose**: Enter new position (when currently flat)

**Pre-entry Checks**: Same as roll (gate, contango, spreads, margin)

**UI Elements**: Similar to roll panel but simpler (no close leg)

---

## 5. Contract Selector

### 5.1 Dropdown Options (Auto-Generated)

Next 6-12 VXM contract months:
```
VXMF6 (Jan 2026)
VXMG6 (Feb 2026)
VXMH6 (Mar 2026)
VXMJ6 (Apr 2026)
VXMK6 (May 2026)
VXMM6 (Jun 2026)
VXMN6 (Jul 2026)
VXMQ6 (Aug 2026)
VXMU6 (Sep 2026)
VXMV6 (Oct 2026)
VXMX6 (Nov 2026)
VXMZ6 (Dec 2026)
[Custom...]
```

### 5.2 Auto-Fill Logic

When roll is initiated:
- New Short = Current Long (VXMG6)
- New Long = Current Long + 1 month (VXMH6)

---

## 6. Safety Features

### 6.1 Order Type Defaults

| Mode | Order Type | Buffer | Confirmation |
|------|------------|--------|--------------|
| **Normal** | LIMIT | ±0.05 | Single confirm |
| **Emergency** | MARKET | N/A | Double confirm + warning |

### 6.2 Emergency Mode Requirements

Before allowing MARKET orders:
1. Show current bid/ask width
2. Warn: "MARKET orders may fill at unfavorable prices"
3. Require typing "MARKET" to confirm
4. Log emergency order decision

### 6.3 Time Window Guardrails

| Time (ET) | Behavior |
|-----------|----------|
| 09:35 - 15:45 | Normal execution |
| Pre-09:35 | Warn: "Pre-market - liquidity may be thin" |
| Post-15:45 | Warn: "After hours - consider waiting" |
| Weekend | Block: "Markets closed" |

### 6.4 Fail-Closed Rules (from runbook)

| Condition | Response |
|-----------|----------|
| Spread market > 0.30 wide | Block, suggest waiting or escalate |
| Gate = CLOSED or REDUCE | Block new positions, allow close only |
| VIX > 30 | Block new positions, allow close only |
| Fill timeout > 15 min | Auto-cancel, show escalation options |
| Orphan leg detected | Alert with "Close orphan at MKT" button |

---

## 7. Backend Implementation

### 7.1 New Script: `scripts/vrp_cs_trade.py`

```bash
# Roll spread (2 calendar combo orders)
python scripts/vrp_cs_trade.py roll --new-front VXMG6 --new-back VXMH6 --dry-run
python scripts/vrp_cs_trade.py roll --new-front VXMG6 --new-back VXMH6 --live --confirm YES

# Close spread
python scripts/vrp_cs_trade.py close --dry-run
python scripts/vrp_cs_trade.py close --live --confirm YES

# Open spread
python scripts/vrp_cs_trade.py open \
  --front VXMG6 --back VXMH6 --dry-run
python scripts/vrp_cs_trade.py open \
  --front VXMG6 --back VXMH6 --live --confirm YES

# Update config (after manual fill entry)
python scripts/vrp_cs_trade.py update-config \
  --short VXMG6 --long VXMH6 \
  --short-price 17.50 --long-price 18.35 \
  --entry-date 2026-01-14
```

### 7.2 Config File Path

**Default**: `data/vrp/paper_trading/position_config.json`

Both `vrp_cs_daily_monitor.py` and `vrp_cs_trade.py` will:
1. Check for this file first
2. Fall back to hardcoded defaults if not found
3. Write updated config here after trades

### 7.3 API Endpoints

**Extend existing pattern** (like `sleeve_c_execute`):

| Action | Endpoint | Method |
|--------|----------|--------|
| Roll (preview) | `/api/dsp/run/vrp_cs_roll?mode=preview&new_front=X&new_back=Y` | POST |
| Roll (execute) | `/api/dsp/run/vrp_cs_roll?mode=execute&new_front=X&new_back=Y` | POST |
| Close (preview) | `/api/dsp/run/vrp_cs_close?mode=preview` | POST |
| Close (execute) | `/api/dsp/run/vrp_cs_close?mode=execute` | POST |
| Open (preview) | `/api/dsp/run/vrp_cs_open?mode=preview&front=X&back=Y` | POST |
| Open (execute) | `/api/dsp/run/vrp_cs_open?mode=execute&front=X&back=Y` | POST |
| Open | `/api/dsp/run/vrp_cs_open?front=X&back=Y&confirm=YES` | POST |
| Pre-checks | `/api/dsp/run/vrp_cs_prechecks` | POST |

**Dedicated config endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/dsp/vrp_cs/config` | GET | Read current config |
| `/api/dsp/vrp_cs/config` | POST | Update config (JSON body) |
| `/api/dsp/vrp_cs/contracts` | GET | Get available contract list |

---

## 8. Implementation Phases

### Phase 1: Backend Script `vrp_cs_trade.py` (3-4 hours)
1. Create script with roll, close, open, update-config commands
2. Implement IBKR order placement via ib_insync
3. Implement pre-roll checks
4. Implement config file read/write
5. Test via CLI with --dry-run

### Phase 2: Config File Migration (30 min)
1. Create initial `position_config.json` from current hardcoded values
2. Update `vrp_cs_daily_monitor.py` to read from JSON by default
3. Test monitor still works

### Phase 3: API Endpoints (1-2 hours)
1. Add action handlers to server.py
2. Add config GET/POST endpoints
3. Add contracts endpoint
4. Test via curl

### Phase 4: UI Components (2-3 hours)
1. Add Roll Panel to dsp100k.html
2. Add Close Spread button
3. Add Config Editor modal
4. Wire to API endpoints
5. Add pre-check display

### Phase 5: Testing & Polish (1-2 hours)
1. End-to-end dry-run testing
2. Paper trading test with real orders
3. Error handling and edge cases
4. Documentation update

**Total Estimated Time**: 8-12 hours

---

## 9. File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `scripts/vrp_cs_trade.py` | CREATE | Trade execution script |
| `scripts/vrp_cs_daily_monitor.py` | MODIFY | Read config from JSON |
| `src/control_ui/server.py` | MODIFY | Add API endpoints |
| `src/control_ui/templates/dsp100k.html` | MODIFY | Add UI components |
| `data/vrp/paper_trading/position_config.json` | CREATE | Position config |
| `data/vrp/paper_trading/trades.csv` | CREATE | Trade audit log |

---

## 10. Today's Roll Execution Plan

Given that the roll is due **today (Jan 14)** and implementation will take 8-12 hours:

**Recommendation**: Execute today's roll manually via TWS using the runbook, then implement UI for future rolls.

**Alternative**: If you want to wait for UI, I can prioritize and deliver:
- Phase 1 (script) in ~3-4 hours
- You could then run via CLI at ~09:35 ET

**Your call.**

---

*Document updated: 2026-01-14*
*Status: ✅ APPROVED - Decisions locked, ready for implementation*
