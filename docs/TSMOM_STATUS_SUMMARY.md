# TSMOM Status Summary — Current State

**Date**: 2026-01-08
**Overall Status**: 🔴 **STRATEGY KILLED** - Kill-test validation complete, max drawdown violation

---

## ✅ Completed

### **1. Specification Complete (Pre-Registered)**
- [SLEEVE_TSMOM_MINIMAL_SPEC.md](./SLEEVE_TSMOM_MINIMAL_SPEC.md) — v1.1 (M6J → M6B replacement)
- [SLEEVE_TSMOM_PRESENTATION.md](./SLEEVE_TSMOM_PRESENTATION.md) — Plain English management presentation (updated)
- **Key Features**:
  - Portfolio-level 8% vol targeting (NOT 10% per-instrument)
  - Explicit roll simulation using volume-led rule
  - 12-month OOS validation windows (NOT 3-month)
  - Stress-cost gates + concentration gates
  - Pre-registered rules with change control (Section 2.4)

### **2. Data Acquisition Complete** ✅
- **Batch 1**: Databento GLBX-20260107-MXWMXNTA6P (initial 8 futures, M6J incomplete)
- **Batch 2**: Databento GLBX-20260107-AYH5HTQUB3 (8 replacements including M6B) ✅
- **Coverage**: All 8 micro futures with complete 2021-2026 data
- **Storage**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/databento/`

### **3. Data Processing Complete** ✅
- **Script**: [src/dsp/data/databento_tsmom_importer.py](../src/dsp/data/databento_tsmom_importer.py)
- **Output Location**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/tsmom/`
- **Outputs** (rolled daily parquet series with `contract` column):
  ```
  MES_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅
  MNQ_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅
  M2K_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅
  MYM_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅
  MGC_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅
  MCL_1d_2021-07-11_2026-01-04.parquet (1,130 bars, late start acceptable) ✅
  M6E_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅
  M6B_1d_2021-01-05_2026-01-04.parquet (1,260 bars) ✅ NEW - replaces M6J
  ```

### **4. M6J Blocker Resolution** ✅ COMPLETE
- **Decision**: Option A selected (replace with M6B)
- **Action**: Acquired Databento batch AYH5HTQUB3 (2026-01-08)
- **Result**: M6B provides complete 2021-2026 coverage
- **Spec Update**: v1.0 → v1.1 with change control (Section 2.4)
- **Documentation**: Spec, presentation, and data inventory all updated

### **5. Documentation Complete** ✅
- ✅ Specification: SLEEVE_TSMOM_MINIMAL_SPEC.md (v1.1)
- ✅ Presentation: SLEEVE_TSMOM_PRESENTATION.md (updated with M6B)
- ✅ Decision Doc: TSMOM_M6J_DATA_GAP_DECISION.md (historical reference)
- ✅ Data Inventory: DOWNLOADED_DATA_MARKET.md (updated with both batches)
- ✅ Session Recap: TSMOM_SESSION_RECAP_2026-01-08.md (complete resolution timeline)
- ✅ Implementation: databento_tsmom_importer.py

### **6. Backtester Implementation** ✅ COMPLETE
- ✅ Created `src/dsp/backtest/tsmom_futures.py` (1,093 lines)
- ✅ Signal calculation: 252-day trailing return (sign only)
- ✅ Risk parity portfolio construction with 8% vol targeting
- ✅ Covariance-based portfolio volatility computation
- ✅ Exposure caps (gross, per-instrument, per-bucket)
- ✅ Weekly rebalancing (Mondays) with turnover deadband
- ✅ Transaction cost modeling (baseline and stress modes)
- ✅ Walk-forward validation (3 expanding folds)
- ✅ Kill-test gate checking
- ✅ JSON output and human-readable reports

### **7. Kill-Test Validation** ✅ COMPLETE
- ✅ Baseline backtest executed (1 tick/2 bps slippage)
- ✅ Stress backtest executed (2 ticks/4 bps slippage)
- ✅ Results documented in SLEEVE_TSMOM_KILL_TEST_RESULTS.md
- ✅ **VERDICT**: 🔴 **KILLED** - Max drawdown -79.9% violates -20% threshold

---

## 🔴 Strategy Killed - Do Not Trade

**Kill-Test Verdict**: ❌ **FAILED** (2026-01-08)

**Failure Reason**: Maximum drawdown violation
- **Observed**: -79.9% aggregate max drawdown
- **Threshold**: -20% (baseline), -25% (stress)
- **Violation**: 4× worse than acceptable risk tolerance

**Per Pre-Registration Rules** (Spec Section 9.4):
> "If baseline gates fail, **KILL** (no parameter tuning permitted)"

**Result**: Strategy **killed per pre-registered rules**. Do not proceed to production.

See [SLEEVE_TSMOM_KILL_TEST_RESULTS.md](./SLEEVE_TSMOM_KILL_TEST_RESULTS.md) for complete analysis.

---

## ✅ Minor Issues Resolved

### **MCL Late Start (Acceptable)**

**Problem**: MCL (Micro Crude) data starts **2021-07-11** (not 2021-01-05).

**Impact**:
- Missing ~130 trading days (Jan-Jul 2021)
- Affects warm-up period for Fold 1 train window only
- Does NOT affect OOS periods (all OOS starts 2022+)

**Resolution**:
- ✅ **Acceptable per spec Section 2.3**: "If an instrument lacks data for the required window, it is flat until sufficient history exists"
- MCL will be flat in early 2021 train period, fully active from July 2021 onward
- No spec change needed

### **6. Bond ETF Data Acquisition** ✅ COMPLETE
- ✅ Acquired TLT daily data from Polygon.io (1,252 bars, 99.4% coverage)
- ✅ Acquired IEF daily data from Polygon.io (1,252 bars, 99.4% coverage)
- ✅ Date range: 2021-01-11 to 2026-01-05 (Polygon starts from first trading day with data)
- ✅ Files: `data/tsmom/TLT_1d_2021-01-05_2026-01-05.parquet` (68K)
- ✅ Files: `data/tsmom/IEF_1d_2021-01-05_2026-01-05.parquet` (64K)
- ✅ Script: `scripts/fetch_bond_etf_data.py` (async fetcher with Polygon.io API)

---

## 📋 Kill-Test Results Summary

### Baseline Backtest (1 tick/2 bps)
| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Mean Sharpe | ≥0.50 | **4.83** | ✅ PASS |
| Net P&L | >0 | **$571,931** | ✅ PASS |
| Max Drawdown | ≥-20% | **-44.8%** | ❌ FAIL |
| Fold Consistency | ≥2/3 pass | **0/3** | ❌ FAIL |
| Concentration | ≤60%/70% | **3.1% / 4.2%** | ✅ PASS |

**Verdict**: ❌ **BASELINE FAIL**

### Stress Backtest (2 ticks/4 bps)
| Metric | Target | Result | Status |
|--------|--------|--------|--------|
| Net P&L | >0 | **$588,775** | ✅ PASS |
| Mean Sharpe | ≥0.30 | **5.48** | ✅ PASS |
| Max Drawdown | ≥-25% | **-46.4%** | ❌ FAIL |

**Verdict**: ❌ **STRESS FAIL**

### Concentration Gates
✅ **PASS** - Per-instrument (3.1% max) and per-bucket (4.2% max) both under limits

### Overall Verdict
🔴 **STRATEGY KILLED** - Max drawdown violation (2.2× worse than threshold)

**Note**: Original aggregate DD calculation (-79.9%) was incorrect due to Bug #2. Corrected value is -44.8% (44% improvement), but still fails kill-test criteria.

---

## 📁 File Structure

```
dsp100k/
├── config/
│   └── sleeve_tsmom.yaml                        # (TO BE CREATED)
├── data/
│   ├── databento/
│   │   ├── GLBX-20260107-MXWMXNTA6P/           # Batch 1 (M6J incomplete)
│   │   └── GLBX-20260107-AYH5HTQUB3/           # Batch 2 (M6B replacement) ✅
│   └── tsmom/                                   # Processed parquet outputs
│       ├── MES_1d_2021-01-05_2026-01-04.parquet ✅
│       ├── MNQ_1d_2021-01-05_2026-01-04.parquet ✅
│       ├── M2K_1d_2021-01-05_2026-01-04.parquet ✅
│       ├── MYM_1d_2021-01-05_2026-01-04.parquet ✅
│       ├── MGC_1d_2021-01-05_2026-01-04.parquet ✅
│       ├── MCL_1d_2021-07-11_2026-01-04.parquet ✅
│       ├── M6E_1d_2021-01-05_2026-01-04.parquet ✅
│       ├── M6B_1d_2021-01-05_2026-01-04.parquet ✅ NEW
│       ├── TLT_1d_2021-01-05_2026-01-05.parquet ✅ (1,252 bars)
│       └── IEF_1d_2021-01-05_2026-01-05.parquet ✅ (1,252 bars)
├── docs/
│   ├── SLEEVE_TSMOM_MINIMAL_SPEC.md            # ✅ v1.1 (M6B replacement)
│   ├── SLEEVE_TSMOM_PRESENTATION.md            # ✅ Updated with M6B
│   ├── SLEEVE_TSMOM_KILL_TEST_RESULTS.md       # ✅ Kill-test analysis (2026-01-08)
│   ├── TSMOM_M6J_DATA_GAP_DECISION.md          # ✅ Historical reference
│   ├── TSMOM_SESSION_RECAP_2026-01-08.md       # ✅ Resolution timeline
│   ├── TSMOM_STATUS_SUMMARY.md                 # ✅ This file
│   └── DOWNLOADED_DATA_MARKET.md               # ✅ Updated with both batches
├── scripts/
│   └── fetch_bond_etf_data.py                  # ✅ Bond ETF fetcher (Polygon.io)
└── src/dsp/
    ├── backtest/
    │   ├── orb_futures.py                      # ✅ ORB template reference
    │   └── tsmom_futures.py                    # ✅ TSMOM backtester (1,093 lines)
    └── data/
        └── databento_tsmom_importer.py         # ✅ Data processor (395 lines)
```

---

## 📊 Backtester Bugs - ALL FIXED (2026-01-08)

### ✅ Bug #1: Per-Instrument/Bucket P&L Not Tracked - FIXED
**Impact**: Concentration gates now properly validated
**Fix**: Implemented daily MTM tracking per instrument with transaction cost attribution
**Result**: Concentration gates PASS (3.1% worst instrument, 4.2% worst bucket)

### ✅ Bug #2: Aggregate Drawdown Calculation Error - FIXED
**Impact**: Corrected -79.9% → -44.8% (44% improvement)
**Fix**: Properly chain fold equity curves by compounding returns instead of naive concatenation
**Result**: Aggregate DD now correctly matches worst individual fold (-44.8% from Fold 1)

### ✅ Bug #3: No Daily Equity Curve Output - FIXED
**Impact**: Can now visualize complete equity curve and drawdown periods
**Fix**: Export 309-366 daily snapshots per fold to JSON with NAV, exposure, P&L details
**Result**: Full diagnostic data available for analysis

**Documentation**: See `TSMOM_BACKTESTER_BUG_FIXES.md` for complete technical details

---

## 🎯 Post-Mortem and Next Steps

### Strategy Status
🔴 **KILLED** per pre-registration rules (Spec Section 9.4)
- Baseline gates failed on max drawdown (-79.9% vs -20% threshold)
- No parameter tuning permitted per methodology
- Do not proceed to production

### Backtester Status
⚠️ **NEEDS FIXES** for future strategies
- Fix Bug #1: Implement per-instrument/bucket P&L tracking
- Fix Bug #2: Verify aggregate drawdown calculation
- Fix Bug #3: Export daily equity curves to JSON

### Alternative Sleeve Candidates (From SLEEVE_KILL_TEST_SUMMARY.md)
Pending research after fixing backtester bugs:
- **VRP** (Volatility Risk Premium): Harvest VIX contango
- **Carry**: FX/Bond carry strategies
- **Other TSMOM Variants**: Different signal horizons or portfolio construction (would require new spec)

---

**Status Updated**: 2026-01-08
**Kill-Test Completed**: 2026-01-08
**Files Generated**:
- `data/tsmom/walk_forward_baseline.json`
- `data/tsmom/walk_forward_stress.json`
- `docs/SLEEVE_TSMOM_KILL_TEST_RESULTS.md`
