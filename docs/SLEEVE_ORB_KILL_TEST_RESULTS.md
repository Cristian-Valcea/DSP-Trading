# Sleeve ORB Kill-Test Results

**Date**: 2026-01-07
**Status**: 🔴 **KILL** - Do Not Trade
**Data Source**: Databento GLBX.MDP3 1-minute OHLCV (2022-2025)

---

## Executive Summary

The Opening Range Breakout (ORB) strategy for MES/MNQ micro futures **failed** the kill-test criteria. While the strategy showed positive total P&L (+$1,403), it failed on **4 out of 7 critical criteria**:

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| Net P&L > $0 | >$0 | **+$1,403** | ✅ PASS |
| Mean Sharpe | >0.5 | **0.23** | ❌ FAIL |
| Win Rate | >35% | **44.4%** | ✅ PASS |
| Max Drawdown | ≥-15% | **-1.7%** | ✅ PASS |
| Folds Passing | ≥4/6 | **2/6** | ❌ FAIL |
| Both Symbols Profitable | Both >$0 | MES +$2,992 / **MNQ -$1,590** | ❌ FAIL |
| SPY Correlation | <0.7 | N/A (futures) | ✅ N/A |

**Key Issues**:
1. **Low Sharpe Ratio** (0.23 < 0.5 threshold) - Strategy not risk-adjusted profitable
2. **Fold Consistency** (2/6 < 4/6 threshold) - Only 33% of OOS folds passed
3. **MNQ Unprofitable** (-$1,590) - Strategy doesn't work consistently across both instruments
4. **High Transaction Costs** - $530 commission + $655 slippage = $1,185 total (45% of gross P&L)

---

## Walk-Forward Results (6 Folds, 2022-2025)

### Per-Fold Performance

| Fold | Test Period | Trades | Net P&L | Sharpe | Win% | MaxDD | Pass |
|------|-------------|--------|---------|--------|------|-------|------|
| 1 | 2022-07-01 to 2022-09-30 | 56 | **-$34** | -0.03 | 44.6% | -1.5% | ❌ |
| 2 | 2023-01-03 to 2023-03-31 | 64 | **-$187** | -0.16 | 40.6% | -1.0% | ❌ |
| 3 | 2023-07-03 to 2023-09-29 | 65 | **+$1,672** | **1.74** | 49.2% | -1.7% | ✅ |
| 4 | 2024-01-02 to 2024-03-28 | 40 | **-$612** | -0.79 | 37.5% | -1.6% | ❌ |
| 5 | 2024-07-01 to 2024-09-30 | 36 | **+$827** | **1.19** | 52.8% | -1.1% | ✅ |
| 6 | 2025-01-02 to 2025-03-31 | 24 | **-$265** | -0.54 | 41.7% | -0.6% | ❌ |

### Aggregate OOS Results

| Metric | Value |
|--------|-------|
| **Total Trades** | 285 |
| **Net P&L** | +$1,403 |
| **Gross P&L** | +$2,588 |
| **Transaction Costs** | -$1,185 (45.8% of gross) |
| **Mean Sharpe** | 0.23 |
| **Mean Win Rate** | 44.4% |
| **Avg $/Contract** | $3.01 |
| **Max Drawdown** | -1.7% (Fold 3) |
| **Folds Passing** | 2/6 (33.3%) |

### Per-Symbol Breakdown

| Symbol | Net P&L | Trades | Avg $/Contract | Status |
|--------|---------|--------|----------------|--------|
| **MES** | +$2,992 | ~143 | ~$21 | ✅ Profitable |
| **MNQ** | -$1,590 | ~142 | ~-$11 | ❌ Unprofitable |

---

## Cost Structure Analysis

### Transaction Costs per Trade

| Cost Component | Assumption | Total | % of Gross P&L |
|----------------|------------|-------|----------------|
| **Commission** | $1.24/RT per contract | $530 | 20.5% |
| **Slippage** | 1 tick/side | $655 | 25.3% |
| **Total Costs** | | **$1,185** | **45.8%** |

**Key Insight**: Transaction costs consumed **45.8% of gross P&L**. With 285 trades over 3.25 years, this equates to ~88 trades/year, or ~1.7 trades/week. The strategy is relatively active but not excessively so.

### Slippage Sensitivity

The baseline test used **1 tick/side** slippage:
- MES: 0.25 points × $5/point = **$1.25 per contract**
- MNQ: 0.25 points × $2/point = **$0.50 per contract**

If slippage increases to **2 ticks/side** (stress scenario):
- Additional cost: ~$655 (doubles slippage cost)
- Estimated Net P&L: +$1,403 - $655 = **+$748** (still positive but marginal)
- Estimated Sharpe: ~0.12 (further degradation)

---

## Fold-by-Fold Analysis

### Winning Folds

**Fold 3 (2023-07-03 to 2023-09-29)**:
- **Best performer**: +$1,672 net, Sharpe 1.74
- Strong trend regime (sustained breakouts)
- Win rate: 49.2% (close to break-even)
- Transaction costs: $280 (15.5% of gross P&L) - efficient

**Fold 5 (2024-07-01 to 2024-09-30)**:
- **Second best**: +$827 net, Sharpe 1.19
- Win rate: 52.8% (best across all folds)
- Lower trade count (36) helped cost efficiency
- Transaction costs: $143 (13.8% of gross P&L)

### Losing Folds

**Fold 4 (2024-01-02 to 2024-03-28)**:
- **Worst performer**: -$612 net, Sharpe -0.79
- Gross P&L also negative (-$496)
- Win rate: 37.5% (lowest)
- Choppy market with false breakouts

**Fold 2 (2023-01-03 to 2023-03-31)**:
- **Second worst**: -$187 net, Sharpe -0.16
- Gross P&L also negative (-$62)
- High transaction costs (64 trades)

**Fold 1 (2022-07-01 to 2022-09-30)**:
- Marginal loss: -$34 net
- Gross P&L barely positive (+$61)
- Transaction costs ($195) exceeded gross

**Fold 6 (2025-01-02 to 2025-03-31)**:
- Recent loss: -$265 net
- Gross P&L negative (-$218)
- Smallest trade count (24) suggests fewer setups

---

## Root Cause Analysis

### Why Did ORB Fail?

1. **Regime Dependency** (67% of folds failed):
   - ORB only works in **sustained trend regimes** (Folds 3, 5)
   - In choppy/mean-reverting markets (Folds 1, 2, 4, 6), breakouts fail
   - No adaptive filter to detect regime type

2. **MNQ Underperformance** (-$1,590):
   - MNQ showed different microstructure than MES
   - Potentially more gap-and-reverse behavior
   - May require different buffer or stop sizing

3. **High Transaction Costs** (45.8% of gross):
   - With average gross P&L per contract of ~$9, costs consumed 54% of edge
   - 1-tick slippage assumption may be optimistic (realistic is 1.5-2 ticks)
   - Need larger edge to overcome friction

4. **Low Sample Size per Fold** (24-65 trades):
   - OOS periods only 3 months each
   - Wide confidence intervals on Sharpe estimates
   - Could be statistical noise rather than true edge

---

## Comparison to Other Sleeves

| Sleeve | Strategy | Sharpe | Verdict | Notes |
|--------|----------|--------|---------|-------|
| **Sleeve DM** | ETF Dual Momentum | **0.55-0.87** | ✅ LIVE | SPY diversifier |
| Sleeve A | S&P 100 L/S | -0.01 | ❌ KILLED | Survivorship bias |
| Sleeve B (L/S) | Sector ETF L/S | -0.03 | ❌ KILLED | No edge |
| Sleeve B (Long) | Sector ETF Long | 0.90 | ❌ KILLED | Just SPY beta |
| Sleeve IM | Intraday ML (LogReg) | 0/6 folds pass | ❌ KILLED | No signal |
| **Sleeve ORB** | **Futures ORB** | **0.23** | **❌ KILL** | **Regime-dependent, costs too high** |

---

## Alternative Configurations to Test (If Reconsidering)

### 1. Add Regime Filter

Only trade on high-volatility days (ATR > 1.5× median):
- Would reduce trade count to ~40-50% of current
- Should improve Sharpe by filtering choppy days
- Risk: Reduces sample size further

### 2. MES-Only

Drop MNQ entirely:
- MES showed +$2,992 profit vs MNQ -$1,590
- Reduces diversification but improves consistency
- Total P&L would be ~$3K over 3.25 years = **$923/year**

### 3. Wider Buffer

Increase buffer from 2 ticks to 4-5 ticks:
- Reduces false breakouts
- Lowers trade count (higher win rate needed)
- May improve Sharpe if filtering noise

### 4. Target Optimization

Current 2R target may be too aggressive:
- Test 1.5R or 3R targets
- Trailing stops instead of fixed targets
- Scale out (1R + runner)

---

## Final Verdict

### Kill-Test Criteria Failure Summary

| Criterion | Status | Gap |
|-----------|--------|-----|
| Sharpe > 0.5 | ❌ FAIL | 0.23 vs 0.5 (-54% shortfall) |
| ≥4/6 folds pass | ❌ FAIL | 2/6 vs 4/6 (-50% shortfall) |
| Both symbols profitable | ❌ FAIL | MNQ -$1,590 |
| Positive P&L | ✅ PASS | +$1,403 |
| Win rate > 35% | ✅ PASS | 44.4% |
| Max DD ≥ -15% | ✅ PASS | -1.7% |

### Recommendation

**🔴 DO NOT TRADE** - Sleeve ORB fails kill-test criteria.

**Rationale**:
1. **Sharpe ratio (0.23) is too low** for production deployment
2. **Only 33% of OOS folds passed** - insufficient consistency
3. **MNQ loses money** - strategy doesn't generalize across instruments
4. **Transaction costs consume 46% of gross P&L** - insufficient edge buffer

**Next Steps**:
1. ✅ Archive results in `dsp100k/docs/SLEEVE_ORB_KILL_TEST_RESULTS.md`
2. ✅ Update `SLEEVE_KILL_TEST_SUMMARY.md` with ORB results
3. ⏸️ **Halt ORB development** - focus on strategies with proven edge
4. 🔍 **Investigate VRP sleeve** (next candidate) - volatility risk premium harvesting

---

## Appendix: Configuration

### Strategy Parameters (Frozen v1.6)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Opening Range | 30 minutes (09:30-09:59 ET) | Excludes 10:00 bar |
| Buffer | 2 ticks | Entry cushion above/below OR |
| Stop | max(1.0×OR_Width, 0.20×ATR) | ATR_d from RTH-only |
| Target | 2R | 2× stop distance |
| Risk per Trade | 20 bps | Position sizing |
| Min Quantity | Skip if <1 | No fractional contracts |

### Data Specifications

| Attribute | Value |
|-----------|-------|
| Source | Databento GLBX.MDP3 |
| Symbols | MES.FUT, MNQ.FUT |
| Frequency | 1-minute OHLCV |
| Date Range | 2022-01-05 to 2025-03-31 |
| Bars | 320,128 (MES), 320,130 (MNQ) |
| Trading Days | 834 |
| Sessions | RTH only (09:30-16:00 ET) |

### Cost Assumptions

| Cost | MES | MNQ |
|------|-----|-----|
| Commission | $1.24/RT | $1.24/RT |
| Slippage | 1 tick ($1.25) | 1 tick ($0.50) |
| Point Value | $5/point | $2/point |
| Tick Size | 0.25 points | 0.25 points |

---

## Files Generated

```
dsp100k/data/orb/
├── MES_1min_2022-01-01_2025-03-31.parquet    # Input data (320K bars)
├── MNQ_1min_2022-01-01_2025-03-31.parquet    # Input data (320K bars)
├── walk_forward_results.json                  # Backtest results
├── backtest_run.log                           # Execution log
└── skip_dates.csv                             # Event calendar

dsp100k/docs/
├── SLEEVE_ORB_MINIMAL_SPEC.md                 # Strategy spec (v1.6)
├── SLEEVE_ORB_IMPLEMENTATION_STATUS.md        # Implementation notes
└── SLEEVE_ORB_KILL_TEST_RESULTS.md           # This file
```

---

**Generated**: 2026-01-07
**Backtester**: dsp100k/src/dsp/backtest/orb_futures.py
**Data Importer**: dsp100k/src/dsp/data/databento_orb_importer.py
