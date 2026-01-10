# VRP-ERP Defensive Overlay - Paper Trading Launch Checklist

**Version**: 1.0
**Date**: 2026-01-09
**Strategy**: VRP-ERP (Equity Risk Premium hedge using VIX regime signals)
**Kill-Test Result**: Sharpe 0.67, Max DD -12.3% ✅ PASSED

---

## 1. Strategy Overview

VRP-ERP is a **defensive overlay** that scales SPY exposure based on VIX regime signals. It's NOT an alpha generator - it's a tail-risk hedge that reduces equity exposure when volatility regime turns unfavorable.

### 1.1 How It Works

```
VIX Regime → SPY Exposure Scaling

CALM (VIX < 15):      100% SPY exposure
ELEVATED (15-20):      75% SPY exposure
HIGH (20-30):          50% SPY exposure
CRISIS (> 30):         25% SPY exposure
```

### 1.2 Why It Complements VRP-CS

| Strategy | Type | Correlation | Purpose |
|----------|------|-------------|---------|
| **VRP-CS** | Alpha | ~0.1 to SPY | Harvest volatility risk premium |
| **VRP-ERP** | Defensive | ~0.5 to SPY | Reduce equity exposure in stress |

Running both: VRP-CS generates alpha, VRP-ERP protects equity allocation.

---

## 2. IBKR Requirements (Simpler than VRP-CS)

### 2.1 Trading Permissions

| Permission | Product | How to Enable |
|------------|---------|---------------|
| **US Stocks** | SPY ETF | Account Management → Trading Permissions → Stocks → United States |

**Note**: No special permissions needed - SPY is a standard equity product.

### 2.2 Market Data

| Subscription | Cost | Required? |
|--------------|------|-----------|
| **US Securities Snapshot** | Free with activity | Yes |
| **CBOE One** (VIX index) | ~$1/month | Recommended (can use delayed) |

---

## 3. Position Sizing

### 3.1 Base Allocation

```python
# Example: $100,000 portfolio, 40% equity allocation
base_equity_allocation = 40000  # $40k to equity sleeve

# VRP-ERP scales this based on regime
regime_scaling = {
    "CALM": 1.00,      # Full $40k in SPY
    "ELEVATED": 0.75,  # $30k in SPY, $10k cash
    "HIGH": 0.50,      # $20k in SPY, $20k cash
    "CRISIS": 0.25,    # $10k in SPY, $30k cash
}
```

### 3.2 Paper Trading Size

```
Recommended starting size:
- Base SPY allocation: $10,000
- Minimum trade: ~$2,500 (25% of base)
- Maximum position: $10,000 (100% of base)
```

---

## 4. Regime Classification Rules

### 4.1 VIX Thresholds

| Regime | VIX Range | SPY Scaling | Action |
|--------|-----------|-------------|--------|
| **CALM** | < 15 | 100% | Full equity exposure |
| **ELEVATED** | 15 - 20 | 75% | Light reduction |
| **HIGH** | 20 - 30 | 50% | Moderate reduction |
| **CRISIS** | > 30 | 25% | Heavy reduction |

### 4.2 Transition Rules

```python
# Smoothing to avoid whipsaws
min_days_in_regime = 2  # Must stay 2 days before acting
vix_buffer = 0.5        # VIX must cross threshold ± 0.5 to switch

# Example: VIX at 15.3 (just above ELEVATED threshold)
# Wait 2 days at VIX > 15.5 before reducing from 100% → 75%
```

---

## 5. Daily Operations

### 5.1 Morning Checklist (Before 9:30 ET)

- [ ] Check VIX spot level
- [ ] Determine current regime
- [ ] Calculate target SPY position
- [ ] Compare to actual position
- [ ] Note any rebalancing needed

### 5.2 Rebalancing Logic

```python
# Only rebalance if:
# 1. Regime changed (and held for 2+ days)
# 2. Position drift > 5% of target

target_shares = int(target_allocation / spy_price)
current_shares = get_current_spy_position()
drift_pct = abs(current_shares - target_shares) / target_shares

if regime_changed and days_in_new_regime >= 2:
    rebalance()
elif drift_pct > 0.05:
    rebalance()
```

### 5.3 Order Execution

```
Order Type: Market-on-Close (MOC) preferred
Timing: Submit MOC orders by 3:45 PM ET
Alternative: VWAP between 3:30-4:00 PM ET
```

---

## 6. Monitoring Metrics

### 6.1 Daily Tracking

| Metric | Description |
|--------|-------------|
| VIX Level | Daily close |
| Regime | CALM/ELEVATED/HIGH/CRISIS |
| Target SPY % | Based on regime |
| Actual SPY % | Current position |
| Drift | Target vs Actual |
| SPY Return | Daily return |
| Strategy Return | After scaling |

### 6.2 Paper Trading Success Criteria (4 weeks)

| Criterion | Target | Status |
|-----------|--------|--------|
| Regime tracking accuracy | 100% | ⏳ |
| Rebalance execution | 100% | ⏳ |
| No missed signals | 100% | ⏳ |
| Drift < 5% maintained | 100% | ⏳ |
| Smooth transitions | No whipsaws | ⏳ |

---

## 7. Integration with VRP-CS

### 7.1 Portfolio Structure

```
Total Portfolio: $100,000

Sleeve Allocation:
├── Sleeve DM (ETF Momentum): 15% = $15,000  [LIVE]
├── VRP-CS (Calendar Spread): 10% = $10,000  [PAPER]
├── VRP-ERP (Defensive):      40% = $40,000  [PAPER]
└── Cash Buffer:              35% = $35,000

VRP-ERP operates on the equity portion:
- When CALM: Full $40k in SPY
- When CRISIS: Only $10k in SPY, rest to cash
```

### 7.2 Correlation Benefits

| Scenario | VRP-CS | VRP-ERP | Portfolio Effect |
|----------|--------|---------|------------------|
| Normal Market | Earns roll yield | Full SPY exposure | Both contribute |
| Vol Spike | Gate closes, flat | Reduces SPY | Both protect |
| Recovery | Gate reopens | Scales up SPY | Both participate |

---

## 8. Implementation Files

| File | Purpose |
|------|---------|
| `src/dsp/regime/vrp_regime_gate.py` | VIX regime classification |
| `src/dsp/sleeves/sleeve_erp.py` | ERP defensive overlay (to be created) |
| `data/vrp/indices/VIX_spot.parquet` | Historical VIX data |

---

## 9. Risk Warnings

### 9.1 Known Limitations

1. **Not Alpha**: VRP-ERP reduces returns in bull markets
2. **Lag**: 2-day regime confirmation causes entry/exit delay
3. **Whipsaws**: Frequent regime changes increase trading costs
4. **Correlation**: Still ~50% correlated to SPY (not a pure hedge)

### 9.2 When VRP-ERP Helps

- ✅ Prolonged volatility spikes (2020 COVID, 2022 rate hikes)
- ✅ Gradual market deterioration
- ✅ Elevated VIX regimes

### 9.3 When VRP-ERP Hurts

- ❌ Sharp V-shaped recoveries (reduce too late, scale up too late)
- ❌ Low-volatility bull markets (underperforms buy-and-hold)
- ❌ Frequent regime oscillations (trading costs)

---

## 10. Go-Live Checklist

### Before First Trade

- [ ] IBKR US stock permissions verified
- [ ] VIX data source configured
- [ ] Paper account selected (DU prefix)
- [ ] Base SPY allocation determined
- [ ] Regime classification code deployed
- [ ] Daily monitoring spreadsheet ready

### First Trade Day

- [ ] Check current VIX level
- [ ] Classify current regime
- [ ] Calculate target SPY position
- [ ] Place initial SPY order
- [ ] Confirm fill
- [ ] Log starting position

---

## 11. Comparison: VRP-CS vs VRP-ERP

| Aspect | VRP-CS | VRP-ERP |
|--------|--------|---------|
| **Type** | Alpha strategy | Defensive overlay |
| **Instrument** | VX futures | SPY ETF |
| **Complexity** | Higher (rolls, spreads) | Lower (simple scaling) |
| **Capital** | ~$10k per spread | Scales with equity |
| **IBKR Setup** | CFE permissions needed | Standard permissions |
| **Frequency** | Daily monitoring | Weekly rebalancing |
| **Sharpe** | 1.21 | 0.67 |
| **Purpose** | Generate returns | Reduce drawdowns |

---

## 12. Parallel Paper Trading Plan

### Week 1-2: Setup and First Trades

| Day | VRP-CS | VRP-ERP |
|-----|--------|---------|
| Day 1 | Verify CFE permissions | Place initial SPY position |
| Day 2 | Check VX quotes | Monitor VIX regime |
| Day 3 | First spread trade | Track drift |
| Day 4-5 | Monitor spread P&L | First rebalance (if needed) |

### Week 3-4: Validation

| Metric | VRP-CS Target | VRP-ERP Target |
|--------|---------------|----------------|
| Trades executed | 2-4 | 1-2 rebalances |
| Fills confirmed | 100% | 100% |
| Roll handled | 1 (Jan 17) | N/A |
| Drift controlled | N/A | < 5% |

### After 4 Weeks

- Review paper trading results
- Compare to backtest expectations
- Decision point: Scale to live or extend paper

---

*Document generated: 2026-01-09*
*Strategy: VRP-ERP Defensive Overlay - PASSED Kill-Test*
*Run in parallel with VRP-CS for comprehensive VRP coverage*
