# TSMOM Status Summary — Current State

**Date**: 2026-01-08
**Overall Status**: 🟢 **READY FOR BACKTEST** - All futures data complete, bond data acquisition next

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

---

## ✅ No Current Blockers

**M6J Data Gap**: ✅ **RESOLVED** (2026-01-08)
- Option A selected: M6B acquired as replacement
- Databento batch AYH5HTQUB3 delivered with complete 2021-2026 coverage
- Spec updated to v1.1 with formal change control
- See [TSMOM_SESSION_RECAP_2026-01-08.md](./TSMOM_SESSION_RECAP_2026-01-08.md) for complete resolution timeline

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

## ⏳ Pending Work

### **1. Implement TSMOM Backtester** ← **NEXT IMMEDIATE TASK**
- [ ] Create `src/dsp/backtest/tsmom_futures.py` (follow ORB template)
- [ ] Implement signal calculation (252d lookback per spec Section 4)
- [ ] Implement risk parity portfolio construction (spec Section 5)
- [ ] Implement volume-led roll simulation (spec Section 3.5)
- [ ] Implement walk-forward validation (3 expanding folds, spec Section 8)
- [ ] Implement kill criteria evaluation (spec Section 9)

### **2. Run Baseline Backtest**
- [ ] Execute baseline: 1 tick/side futures + 2 bps/side ETFs
- [ ] Execute stress: 2 ticks/side futures + 4 bps/side ETFs
- [ ] Generate JSON outputs per spec Section 10 (fold metrics, PnL breakdown)

### **3. Evaluate Kill Criteria**
- [ ] Check primary gates (Sharpe ≥0.5, PnL >0, DD ≥-20%, 2/3 folds pass)
- [ ] Check stress gates (PnL >0, Sharpe ≥0.3, DD ≥-25%)
- [ ] Check concentration gates (no >60% single instrument, no >70% single bucket)
- [ ] Document results in kill-test report

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
│   ├── TSMOM_M6J_DATA_GAP_DECISION.md          # ✅ Historical reference
│   ├── TSMOM_SESSION_RECAP_2026-01-08.md       # ✅ Resolution timeline
│   ├── TSMOM_STATUS_SUMMARY.md                 # ✅ This file
│   └── DOWNLOADED_DATA_MARKET.md               # ✅ Updated with both batches
├── scripts/
│   └── fetch_bond_etf_data.py                  # ✅ Bond ETF fetcher (Polygon.io)
└── src/dsp/
    ├── backtest/
    │   ├── orb_futures.py                      # ✅ ORB template reference
    │   └── tsmom_futures.py                    # (TO BE CREATED)
    └── data/
        └── databento_tsmom_importer.py         # ✅ Data processor (395 lines)
```

---

## 📊 Success Criteria Reminder

**Kill-Test Gates** (from spec Section 9):

**Primary Gates (Baseline Costs)**:
- Mean OOS Sharpe ≥ 0.50 ✅/❌
- OOS Net PnL > 0 ✅/❌
- Max Drawdown ≥ -20% ✅/❌
- Fold Consistency: ≥2/3 folds with Sharpe ≥0.25 AND PnL >0 ✅/❌

**Stress Gates (2× Slippage)**:
- OOS Net PnL > 0 ✅/❌
- Mean OOS Sharpe ≥ 0.30 ✅/❌
- Max Drawdown ≥ -25% ✅/❌

**Concentration Gates**:
- No single instrument >60% of absolute OOS PnL ✅/❌
- No single bucket >70% of absolute OOS PnL ✅/❌

**All gates must pass. If baseline fails: KILL (no parameter tuning).**

---

## 🎯 Next Immediate Actions

### **1. Implement TSMOM Backtester** ← **NEXT STEP**
- Create `src/dsp/backtest/tsmom_futures.py` following ORB template
- Reference: `src/dsp/backtest/orb_futures.py` for walk-forward framework
- Implement per spec Sections 3-10 (signal, portfolio, roll, validation, gates)
- Use data from `data/tsmom/` (8 futures + 2 ETFs, all ready)

### **2. Execute Kill-Test Validation**
- Run baseline + stress backtests
- Evaluate all gates (primary, stress, concentration)
- Document results in `SLEEVE_TSMOM_KILL_TEST_RESULTS.md`
- Verdict: PASS → promote, FAIL → kill (no parameter tuning)

---

**Status Updated**: 2026-01-08
**Last Commit**: (pending - status summary + session recap updates)
