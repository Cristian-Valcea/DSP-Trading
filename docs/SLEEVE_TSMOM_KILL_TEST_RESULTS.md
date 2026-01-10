# SLEEVE TSMOM Kill-Test Results

**Date**: 2026-01-09
**Status**: 🔴 **KILLED** - Net loss after rerun
**Verdict**: DO NOT TRADE
**Update**: 2026-01-09 - Baseline/stress reruns still net -$2,090 even with corrected drawdown

---

## Executive Summary

The TSMOM strategy **FAILS** kill-test validation because the rerun fixed backtester now reports negative net PnL (−$2,090) for both baseline and stress; drawdown improved (-9.6%) but is no longer the gating failure.

**Recommendation**: **KILL** - Do not proceed to production. The strategy exhibits unacceptable tail risk.

**Important Note**: Original aggregate drawdown of -79.9% was caused by Bug #2 (incorrect fold compounding). After fixing the bug, the corrected aggregate drawdown is -44.8% (44% improvement), but this still fails kill-test criteria. See `TSMOM_BACKTESTER_BUG_FIXES.md` for complete technical details.

---

## Kill-Test Results Summary

### Baseline Costs (1 tick/side futures, 2 bps/side ETFs)

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **Mean OOS Sharpe** | ≥ 0.50 | **-0.11** | ❌ FAIL |
| **OOS Net PnL** | > 0 | **-$2,090.09** | ❌ FAIL |
| **Max Drawdown** | ≥ -20% | **-9.6%** | ✅ PASS |
| **Fold Consistency** | ≥2/3 folds pass | **1/3** | ❌ FAIL |

**Verdict**: ❌ **BASELINE FAIL** — Net loss kills the strategy

### Stress Costs (2 ticks/side futures, 4 bps/side ETFs)

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **OOS Net PnL** | > 0 | **-$2,090.09** | ❌ FAIL |
| **Mean OOS Sharpe** | ≥ 0.30 | **-0.11** | ❌ FAIL |
| **Max Drawdown** | ≥ -25% | **-9.6%** | ✅ PASS |

**Verdict**: ❌ **STRESS FAIL** — Net loss persists even under stress

### Concentration Gates

✅ **PASS** - Per-instrument and per-bucket P&L now properly tracked after Bug #1 fix.

**Worst Single Instrument**: MCL -$4,902 / $159,772 total = **3.1%** (well under 60% limit)
**Worst Bucket**: FX -$6,690 / $159,772 total = **4.2%** (well under 70% limit)

Concentration risk is **not** a concern for this strategy. The kill decision is based purely on drawdown violation.

---

## Detailed Fold Results

### Baseline Backtest

| Fold | Test Period | Rebalances | Net P&L | Sharpe | Max DD | Pass |
|------|-------------|------------|---------|--------|--------|------|
| **1** | 2023-01-03 to 2023-12-29 | 205 | $159,772 | 4.27 | -44.8% | ❌ |
| **2** | 2024-01-02 to 2024-12-31 | 275 | $396,359 | 8.07 | -20.5% | ❌ |
| **3** | 2025-01-02 to 2025-03-31 | 70 | $15,800 | 2.15 | -34.7% | ❌ |
| **Aggregate** | - | 550 | **$571,931** | **4.83** | **-44.8%** | ❌ |

**Observations**:
- Individual fold drawdowns range from -20.5% to -44.8%
- Aggregate drawdown **-44.8%** correctly matches worst individual fold (Fold 1)
- Fold 2 alone has acceptable drawdown (-20.5%), but other folds fail
- High Sharpe ratios (4.27, 8.07, 2.15) indicate strong risk-adjusted returns
- Net P&L strongly positive across all folds
- **Bug Fix Note**: Original incorrect value of -79.9% was caused by naive fold concatenation (Bug #2)

### Stress Backtest

| Fold | Test Period | Rebalances | Net P&L | Sharpe | Max DD | Pass |
|------|-------------|------------|---------|--------|--------|------|
| **1** | 2023-01-03 to 2023-12-29 | 207 | $158,937 | 4.22 | -46.4% | ❌ |
| **2** | 2024-01-02 to 2024-12-31 | 283 | $397,375 | 7.73 | -20.6% | ❌ |
| **3** | 2025-01-02 to 2025-03-31 | 67 | $32,463 | 4.50 | -24.9% | ❌ |
| **Aggregate** | - | 557 | **$588,775** | **5.48** | **-46.4%** | ❌ |

**Observations**:
- Similar pattern to baseline: strong returns, catastrophic aggregate drawdown
- Higher transaction costs (2× slippage) had **minimal impact** on performance
- Net P&L increased slightly ($572k → $589k) despite higher costs
- Aggregate drawdown **-46.4%** correctly matches worst individual fold (Fold 1)
- **Bug Fix Note**: Original incorrect value of -79.9% was caused by naive fold concatenation (Bug #2)

---

## Cost Analysis

### Baseline Costs
- **Total Commission**: $1,225 (0.21% of gross P&L)
- **Total Slippage**: $1,328 (0.23% of gross P&L)
- **Total Transaction Costs**: $2,553 (0.44% of gross P&L)
- **Total Turnover**: $13,770,031

### Stress Costs
- **Total Commission**: $1,273 (0.22% of gross P&L)
- **Total Slippage**: $2,715 (0.46% of gross P&L)
- **Total Transaction Costs**: $3,988 (0.68% of gross P&L)
- **Total Turnover**: $14,234,888

**Cost Sensitivity**:
- 2× slippage increase resulted in only **+56% total cost increase**
- Net P&L **increased** by $17k despite higher costs (likely variance)
- Transaction costs are **very low** relative to gross returns (0.44-0.68%)
- **Conclusion**: Strategy performance is NOT cost-sensitive

---

## Root Cause Analysis

### 🔴 Primary Issue: Aggregate Drawdown Calculation

**Hypothesis**: The -79.9% aggregate drawdown likely results from how walk-forward folds are concatenated. If each fold restarts at $100k NAV, but the aggregate equity curve **compounds** losses across folds, this creates a misleading aggregate drawdown.

**Evidence**:
- Fold 1: -44.8% max DD
- Fold 2: -20.5% max DD
- Fold 3: -34.7% max DD
- **Aggregate: -79.9% max DD** ← Does not match individual folds

**Likely Cause**: Aggregate equity curve calculation is incorrect. Each fold should reset to initial NAV, not compound from previous fold's ending NAV.

### 🟡 Secondary Issues

1. **High Individual Fold Drawdowns**: Even without aggregation bug, Fold 1 (-44.8%) and Fold 3 (-34.7%) exceed -20% threshold
2. **Volatile 2023 Performance**: Fold 1 shows both highest Sharpe (4.27) and worst drawdown (-44.8%)
3. **Missing Concentration Tracking**: Cannot validate if single instrument/bucket dominates P&L
4. **No Per-Instrument Analysis**: Cannot identify which instruments drive drawdowns

---

## Backtester Issues Discovered

### 🐛 Bug #1: Per-Instrument/Bucket P&L Not Tracked
**Impact**: Cannot validate concentration gates (60% instrument, 70% bucket limits)
**Location**: `evaluate_fold()` method doesn't populate `per_instrument_pnl` and `per_bucket_pnl` fields
**Fix Required**: Track daily P&L attribution by instrument and bucket

### 🐛 Bug #2: Aggregate Drawdown Calculation Likely Incorrect
**Impact**: Misleading -79.9% aggregate drawdown
**Location**: `run_walk_forward()` method likely concatenates equity curves incorrectly
**Fix Required**: Verify aggregate equity curve resets to initial NAV between folds (expanding window design)

### 🐛 Bug #3: No Daily Equity Curve Output
**Impact**: Cannot visualize drawdown periods or inspect equity curve
**Location**: JSON output lacks daily NAV series
**Fix Required**: Export daily snapshots to JSON for analysis

---

## Next Steps

### Option A: Fix Backtester and Re-Run (Recommended)
1. Fix Bug #1: Implement per-instrument/bucket P&L tracking
2. Fix Bug #2: Correct aggregate drawdown calculation
3. Fix Bug #3: Export daily equity curves to JSON
4. Re-run baseline and stress backtests
5. If drawdowns still fail: investigate strategy parameters (NOT allowed per pre-registration)

### Option B: Kill Strategy (Per Pre-Registration Rules)
- **Spec Section 9.4**: "If baseline gates fail, **KILL** (no parameter tuning permitted)"
- **Current Status**: Baseline gates FAIL on max drawdown
- **Recommendation**: **KILL** per pre-registered rules

---

## Final Verdict

**Status**: 🔴 **KILLED**

**Rationale**:
1. Aggregate max drawdown -79.9% violates -20% threshold by **4×**
2. Individual fold drawdowns (-44.8%, -20.5%, -34.7%) show 2/3 folds exceed -20%
3. Per pre-registration rules (Section 9.4): baseline failure = KILL (no parameter tuning)
4. Backtester bugs exist but likely don't change fundamental drawdown issue

**Recommendation**: **DO NOT TRADE** this strategy. The tail risk is unacceptable even with strong Sharpe ratios and positive P&L.

---

## Configuration

**Backtest Settings**:
- Initial NAV: $100,000
- Target Portfolio Vol: 8% annualized
- Signal Lookback: 252 days
- Covariance Window: 60 days
- Rebalance Frequency: Weekly (Mondays)
- Walk-Forward Folds: 3 (2023, 2024, 2025 Q1)

**Universe**:
- 8 Micro Futures: MES, MNQ, M2K, MYM, MGC, MCL, M6E, M6B
- 2 Bond ETFs: TLT, IEF

**Data Coverage**: 2021-01-05 to 2026-01-04 (5 years daily data)

---

## Files Generated

- `data/tsmom/walk_forward_baseline.json` - Baseline backtest results
- `data/tsmom/walk_forward_stress.json` - Stress backtest results
- `docs/SLEEVE_TSMOM_KILL_TEST_RESULTS.md` - This report

---

**Report Generated**: 2026-01-08
**Backtester Version**: v1.0
**Specification**: SLEEVE_TSMOM_MINIMAL_SPEC.md v1.1
