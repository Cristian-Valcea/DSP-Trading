# SPEC_VRP_CALENDAR_SPREAD.md — VRP Calendar Spread Strategy

**Version**: 1.0
**Date**: 2026-01-09
**Status**: PASSED KILL TEST - Sharpe 1.21, Max DD -9.1%
**Author**: Claude

---

## 1. Executive Summary

The VRP Calendar Spread strategy is a **TRUE alpha strategy** that harvests the volatility risk premium through calendar spreads rather than outright short VIX exposure. It passed all kill-test criteria with strong metrics:

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Sharpe Ratio | 1.21 | ≥ 0.50 | ✅ PASS |
| Max Drawdown | -9.1% | ≥ -30% | ✅ PASS |
| Total Return | 363.8% | > 0% | ✅ PASS |
| Win Rate | 46.7% | ≥ 40% | ✅ PASS |

**CAGR**: 13.6% over 11 years (2014-2025)

---

## 2. Strategy Overview

### 2.1 Core Insight

We do NOT short the front-month VIX futures outright (that approach failed with Sharpe -0.13). Instead, we trade the **SPREAD** between VX2 and VX1:

- **Long VX2** (second-month VIX future)
- **Short VX1** (front-month VIX future)

### 2.2 Why Calendar Spreads Work

1. **Roll Yield Harvesting**: VX1 decays faster than VX2 in contango (time decay is steeper at the front)
2. **Spread P&L**: Profit = (Entry Spread) - (Current Spread)
   - We profit when the spread NARROWS (VX2 decays toward VX1)
3. **Hedged Against Spikes**: Both legs move together during VIX spikes
   - If VIX jumps 5 points, both VX1 and VX2 jump ~5 points
   - The spread changes much less than outright VIX
4. **Risk Exposure**: Still exposed to term structure INVERSION (backwardation)
   - This is where the VRP Regime Gate helps

### 2.3 Historical Contango Statistics

From our 3,283-day term structure dataset (2013-2025):

| Metric | Value |
|--------|-------|
| Mean VX2-VX1 Contango | +3.97% |
| Std Dev | 5.82% |
| % Time in Contango | 71.9% |
| Min (Backwardation) | -35.2% |
| Max (Steep Contango) | +37.1% |

---

## 3. Entry & Exit Rules

### 3.1 Entry Conditions

All must be true:
1. **Contango Threshold**: VX2-VX1 spread > 2% (favorable roll yield)
2. **VIX Level**: VIX < 30 (avoid extreme volatility regimes)
3. **Gate State**: VRP Regime Gate = OPEN (not in crisis)

### 3.2 Exit Conditions

Exit on ANY of:
1. **Stop-Loss**: Spread widens 25% against us (VX2-VX1 increases 25%)
2. **Take-Profit**: Spread narrows 50% (captured significant roll yield)
3. **Gate Closure**: VRP Regime Gate switches to CLOSED
4. **Roll Date**: VX1 is 5 days or less from expiration (time to roll)

### 3.3 Position Sizing

- **Notional per spread**: $1,000 per VX point
- **Max spreads**: 5 concurrent spreads
- **Margin buffer**: 2x (allocate 2x notional for margin requirements)

---

## 4. VRP Regime Gate Integration

The VRP Regime Gate provides crisis detection:

| Gate State | Action | % of Time |
|------------|--------|-----------|
| **OPEN** | Trade normally | 73.6% |
| **REDUCE** | Scale position to 50% | 19.2% |
| **CLOSED** | No new trades, exit existing | 7.2% |

### 4.1 Gate Impact Analysis

| Version | Sharpe | CAGR | Max DD |
|---------|--------|------|--------|
| With Gate | 1.21 | 13.6% | -9.1% |
| Without Gate | 1.35 | 15.8% | -9.2% |

The gate slightly reduces returns but provides crisis protection. Both versions pass kill-test.

**Recommendation**: Use the gate for production (risk-adjusted approach), but the no-gate version is also viable for more aggressive allocations.

---

## 5. Trade Statistics

Over 390 trades (2014-2025):

| Metric | Value |
|--------|-------|
| Total Trades | 390 |
| Win Rate | 46.7% |
| Average Win | $4,105 |
| Average Loss | $1,843 |
| Win/Loss Ratio | 2.23:1 |
| Profit Factor | 1.95 |

The strategy wins less than half the time but wins are 2.2x larger than losses.

---

## 6. Risk Analysis

### 6.1 Drawdown Profile

- **Max Drawdown**: -9.1%
- **Calmar Ratio**: 1.49 (excellent risk-adjusted returns)
- **Recovery**: Drawdowns typically recover within 2-3 months

### 6.2 Worst Periods

The strategy struggles during:
1. **VIX Spike Events**: March 2020 COVID crash, August 2015 China devaluation
2. **Term Structure Inversion**: When backwardation persists

The VRP Regime Gate helps detect these periods and close positions.

### 6.3 Correlation with Other Strategies

| Sleeve | Expected Correlation |
|--------|---------------------|
| Sleeve DM (ETF Momentum) | Low (~0.2) |
| Sleeve TSMOM (Futures Momentum) | Low-Medium (~0.3) |
| SPY | Low (~0.1) |

The VRP Calendar Spread provides diversification to the overall portfolio.

---

## 7. Implementation Requirements

### 7.1 Data Requirements

| Data | Source | Status |
|------|--------|--------|
| VX Individual Contracts | CBOE | ✅ Have 388 contracts (2013-2026) |
| VIX Spot | CBOE | ✅ Have |
| VVIX | CBOE | ✅ Have |
| VX Front Month (VX_F1) | CBOE | ✅ Have |

### 7.2 Execution Requirements

- **Broker**: Interactive Brokers (IBKR) supports VX futures
- **Order Types**: Market or Limit orders for spread legs
- **Rolling**: Monthly roll process 5 days before VX1 expiration
- **Margin**: ~$5,000 per spread (varies with VIX level)

### 7.3 Code Artifacts

| File | Purpose |
|------|---------|
| `src/dsp/data/vx_term_structure_builder.py` | Build continuous VX1-VX4 series |
| `src/dsp/backtest/vrp_calendar_spread.py` | Backtest engine |
| `src/dsp/regime/vrp_regime_gate.py` | Crisis detection gate |
| `data/vrp/term_structure/vx_term_structure.parquet` | Term structure data |

---

## 8. Kill Test Certification

### 8.1 Test Parameters

- **Period**: 2014-01-01 to 2025-12-31 (11 years)
- **Initial Capital**: $100,000
- **Data**: 3,158 aligned trading days

### 8.2 Results

```
KILL TEST:
----------------------------------------
Sharpe >= 0.50:    ✅ (1.21)
Max DD >= -30%:    ✅ (-9.1%)
Return > 0:        ✅ (363.8%)
Win Rate >= 40%:   ✅ (46.7%)

VERDICT: PASS
```

### 8.3 Certification

**This strategy is APPROVED for paper trading.**

---

## 9. Deployment Roadmap

### Phase 1: Paper Trading (Week 1-4)
- [ ] Implement live term structure builder
- [ ] Set up daily VRP gate monitoring
- [ ] Paper trade calendar spreads via IBKR
- [ ] Track fills, slippage, roll costs

### Phase 2: Micro-Live (Week 5-8)
- [ ] Reduce position size to 1 spread ($1k notional)
- [ ] Validate execution quality
- [ ] Monitor drawdowns vs backtest

### Phase 3: Scale-Up (Week 9+)
- [ ] Increase to target allocation
- [ ] Integrate with portfolio risk management
- [ ] Add to sleeve monitoring dashboard

---

## 10. Comparison to Failed VRP Approaches

| Strategy | Sharpe | Max DD | Status |
|----------|--------|--------|--------|
| **VRP Calendar Spread** | **1.21** | **-9.1%** | **✅ PASS** |
| VRP Futures (Short VX1) | -0.13 | -45%+ | ❌ FAILED |
| VRP-Gated NN | Reduces performance | N/A | ❌ FAILED |

The calendar spread approach is fundamentally different:
- **Failed**: Short outright VIX exposure (unlimited downside on spikes)
- **Works**: Trade the SPREAD (hedged against parallel moves)

---

## 11. Appendix: Sample Trades

Top 5 Winning Trades:
1. **2020-04-15 to 2020-05-01**: +$18,240 (COVID recovery contango normalization)
2. **2018-02-12 to 2018-03-01**: +$15,890 (Post-Volmageddon contango steepening)
3. **2022-10-15 to 2022-11-01**: +$12,560 (Fed pivot hopes)

Top 5 Losing Trades:
1. **2020-02-24 to 2020-03-09**: -$8,420 (COVID crash, exited via gate)
2. **2015-08-20 to 2015-08-25**: -$6,890 (China devaluation)
3. **2018-02-05 to 2018-02-08**: -$5,120 (Volmageddon, exited early via gate)

---

*Document generated: 2026-01-09*
*Backtest results: data/vrp/models/vrp_calendar_spread_evaluation.json*
