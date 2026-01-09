# VRP Calendar Spread - Paper Trading Launch Checklist

**Version**: 1.1
**Date**: 2026-01-09
**Strategy**: VRP Calendar Spread (Long VX2, Short VX1)
**Kill-Test Result**: Sharpe 1.21, Max DD -9.1% ✅ PASSED
**Paper Trading Status**: 🟢 **LIVE** - First trade executed 2026-01-09

---

## 0. Current Position Status

### Active Trade (as of 2026-01-09)

| Contract | Position | Entry Price | Current | Role |
|----------|----------|-------------|---------|------|
| **VXMF6** | -1 | 16.24 | 16.25 | Short front month |
| **VXMG6** | +1 | 17.87 | 17.83 | Long back month |

| Metric | Value |
|--------|-------|
| **Entry Spread** | 1.63 points |
| **Entry Date** | 2026-01-09 |
| **Entry Notional** | $163 |
| **Stop-Loss** | Spread > 2.04 (25% wider) |
| **Take-Profit** | Spread < 0.82 (50% narrower) |
| **Roll By** | 2026-01-14 (5 trading days before Jan 21 expiry) |

### Verified Setup

- [x] IBKR CFE trading permissions enabled
- [x] CFE market data subscription active
- [x] Paper account selected (DU8009825)
- [x] Sufficient margin available ($1,044,827)
- [x] VRP Regime Gate: OPEN (contango 10.19%)
- [x] First spread order filled via combo order

---

## 1. IBKR Account Prerequisites

### 1.1 Trading Permissions Required

| Permission | Exchange | Product | How to Enable |
|------------|----------|---------|---------------|
| **VIX Futures** | CFE (CBOE Futures Exchange) | VX | Account Management → Trading Permissions → Futures → United States → CFE |
| **VIX Micro Futures** | CFE | VXM | Same as above (included with CFE permission) |

**⚠️ IMPORTANT**: VX futures trade on **CFE** (CBOE Futures Exchange), NOT CME. Ensure CFE is enabled, not just CME.

### 1.2 Market Data Subscriptions Required

| Subscription | Monthly Cost | Purpose |
|--------------|--------------|---------|
| **CBOE Futures Exchange (CFE)** | ~$1/month | Real-time VX/VXM quotes |
| **CBOE One** (optional) | ~$1/month | VIX spot index (can use delayed) |

**To Subscribe**: Account Management → Market Data Subscriptions → CBOE Futures Exchange

### 1.3 Account Type Verification

```
Required: Margin account (VX futures require margin)
Minimum: $25,000 recommended for proper position sizing
Paper Account: Use paper trading account first (DU prefix)
```

---

## 2. Contract Specifications

### 2.1 VX Futures (Standard)

| Attribute | Value |
|-----------|-------|
| **Symbol** | VX |
| **Exchange** | CFE |
| **Multiplier** | $1,000 per VIX point |
| **Tick Size** | 0.05 ($50 per tick) |
| **Trading Hours** | 17:00 - 16:00 ET (next day), nearly 23 hours |
| **Settlement** | Cash-settled |
| **Expiration** | Wednesday 30 days before SPX options expiration |
| **Margin** | ~$10,000-15,000 per contract (varies with VIX level) |

### 2.2 VXM Futures (Micro) - RECOMMENDED FOR PAPER TRADING

| Attribute | Value |
|-----------|-------|
| **Symbol** | VXM |
| **Exchange** | CFE |
| **Multiplier** | $100 per VIX point (1/10th of VX) |
| **Tick Size** | 0.05 ($5 per tick) |
| **Margin** | ~$1,000-1,500 per contract |
| **Advantage** | Lower capital requirement, better position sizing granularity |

**Recommendation**: Start with VXM for paper trading to validate execution before scaling to VX.

---

## 3. Calendar Spread Construction

### 3.1 Position Structure

```
LONG:  1 VXM M2 (second month)
SHORT: 1 VXM M1 (front month)

Example (January 2026):
  LONG:  VXM Feb 2026 (VXM G6)
  SHORT: VXM Jan 2026 (VXM F6)
```

### 3.2 Month Codes

| Month | Code | Month | Code |
|-------|------|-------|------|
| January | F | July | N |
| February | G | August | Q |
| March | H | September | U |
| April | J | October | V |
| May | K | November | X |
| June | M | December | Z |

### 3.3 Contract Selection Logic

```python
# On any given day:
VX1 = First VX contract expiring > today
VX2 = Second VX contract expiring > today

# Example: If today is Jan 9, 2026
# VX1 = VXM F6 (Jan 2026, expires ~Jan 22)
# VX2 = VXM G6 (Feb 2026, expires ~Feb 19)
```

---

## 4. Entry Rules

### 4.1 Entry Conditions (ALL must be true)

| Condition | Check | Threshold |
|-----------|-------|-----------|
| **Contango** | VX2 - VX1 > 0 | > 2% |
| **VIX Level** | VIX spot | < 30 |
| **Gate State** | VRP Regime Gate | OPEN |
| **Days to Expiry** | VX1 DTE | > 5 days |

### 4.2 Position Sizing

```python
# Conservative paper trading sizing
notional_per_spread = 1000  # $1,000 per VIX point
max_spreads = 2             # Start with 2 max spreads
margin_buffer = 2.0         # Keep 2x margin requirement

# Example: VIX at 15, VXM micro contract
# Spread notional = 15 * $100 = $1,500
# Margin per spread = ~$1,500
# With 2x buffer = $3,000 allocated per spread
```

---

## 5. Exit Rules

### 5.1 Exit Conditions (ANY triggers exit)

| Condition | Trigger | Action |
|-----------|---------|--------|
| **Stop-Loss** | Spread widens 25% | Exit immediately |
| **Take-Profit** | Spread narrows 50% | Exit immediately |
| **Gate Closure** | Gate → CLOSED | Exit immediately |
| **Roll Date** | VX1 DTE ≤ 5 days | Roll to next month |

### 5.2 Roll Procedure

```
5 days before VX1 expiration:
1. Exit current spread (close VX1 short, close VX2 long)
2. Re-evaluate entry conditions
3. If conditions met: Open new spread (short new VX1, long new VX2)
4. If conditions NOT met: Stay flat until conditions return
```

---

## 6. Roll Calendar (2026)

| VX1 Expiry | Roll By | VX1 Symbol | VX2 Symbol |
|------------|---------|------------|------------|
| Jan 22, 2026 | Jan 17 | VXM F6 | VXM G6 |
| Feb 19, 2026 | Feb 14 | VXM G6 | VXM H6 |
| Mar 18, 2026 | Mar 13 | VXM H6 | VXM J6 |
| Apr 15, 2026 | Apr 10 | VXM J6 | VXM K6 |
| May 20, 2026 | May 15 | VXM K6 | VXM M6 |
| Jun 17, 2026 | Jun 12 | VXM M6 | VXM N6 |

**First Roll Event**: January 17, 2026 (8 days from now)

---

## 7. Daily Monitoring Checklist

### 7.1 Pre-Market (Before 9:30 ET)

- [ ] Check VIX spot level (target: < 30)
- [ ] Check VX1-VX2 contango (target: > 2%)
- [ ] Run VRP Regime Gate update
- [ ] Verify positions match target
- [ ] Check days to VX1 expiration

### 7.2 During Market

- [ ] Monitor spread P&L
- [ ] Check for stop-loss trigger (spread widened 25%)
- [ ] Check for take-profit trigger (spread narrowed 50%)
- [ ] Monitor gate state changes

### 7.3 Post-Market

- [ ] Record daily P&L
- [ ] Update position log
- [ ] Note any fills/slippage

---

## 8. Monitoring Metrics

### 8.1 Key Metrics to Track

| Metric | Target | Warning | Fail |
|--------|--------|---------|------|
| **Daily Sharpe** | > 0 | < 0 for 5 days | < 0 for 10 days |
| **Max Drawdown** | < 5% | 5-10% | > 10% |
| **Win Rate** | > 45% | 40-45% | < 40% |
| **Avg Slippage** | < 1 tick | 1-2 ticks | > 2 ticks |

### 8.2 Paper Trading Success Criteria (4 weeks)

| Criterion | Target | Status |
|-----------|--------|--------|
| Position tracking accuracy | 100% | ⏳ |
| Roll execution success | 100% | ⏳ |
| Gate signal reliability | > 95% | ⏳ |
| No unintended overnight positions | 100% | ⏳ |
| Slippage within bounds | < 2 ticks avg | ⏳ |

---

## 9. Implementation Files

| File | Purpose |
|------|---------|
| `src/dsp/data/vx_term_structure_builder.py` | Build VX1-VX4 continuous series |
| `src/dsp/backtest/vrp_calendar_spread.py` | Backtest engine (reference for live) |
| `src/dsp/regime/vrp_regime_gate.py` | Regime gate for crisis detection |
| `data/vrp/term_structure/vx_term_structure.parquet` | Historical term structure data |

---

## 10. Risk Warnings

### 10.1 Known Risks

1. **Term Structure Inversion**: Backwardation causes losses (gate helps detect)
2. **VIX Spike Events**: Both legs move, but spread can widen sharply
3. **Liquidity Risk**: VXM less liquid than VX, wider spreads
4. **Roll Risk**: Execution around roll dates requires care

### 10.2 Circuit Breakers

| Condition | Action |
|-----------|--------|
| Portfolio DD > 10% | Flatten all positions |
| VIX > 40 | No new positions |
| Gate CLOSED > 5 days | Review strategy |

---

## 11. Quick Reference - IBKR Order Entry

### 11.1 Opening a Calendar Spread

```
TWS Combo Order:
1. Trading → Combo/Strategy → Futures Calendar Spread
2. Select: VXM (or VX)
3. Near Month: Current front month (SELL)
4. Far Month: Next month (BUY)
5. Action: BUY spread (buys VX2, sells VX1)
6. Order Type: LMT (use limit orders)
7. Limit Price: Current spread ± 0.05
```

### 11.2 Closing a Calendar Spread

```
Same as opening but:
- Action: SELL spread (sells VX2, buys VX1)
- Or close individual legs with market orders if urgent
```

---

## 12. Go-Live Checklist

### Before First Trade

- [ ] IBKR CFE trading permissions enabled
- [ ] CFE market data subscription active
- [ ] Paper account selected (DU prefix)
- [ ] Sufficient margin available
- [ ] VRP Regime Gate code deployed
- [ ] Daily monitoring script ready
- [ ] Roll calendar in trading calendar

### First Trade Day

- [ ] Run gate update with live data
- [ ] Verify gate state = OPEN
- [ ] Verify contango > 2%
- [ ] Verify VIX < 30
- [ ] Place first spread order (1 VXM spread)
- [ ] Confirm fill and positions
- [ ] Log trade details

---

*Document generated: 2026-01-09*
*Strategy: VRP Calendar Spread - PASSED Kill-Test*
*Recommended: Start with VXM (micro) for paper trading*
