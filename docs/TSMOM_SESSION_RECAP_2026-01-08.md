# TSMOM Session Recap — M6J Blocker Resolution

**Date**: 2026-01-08
**Session Type**: Data Acquisition & Specification Update
**Status**: ✅ **BLOCKER RESOLVED** — All 8 futures instruments ready for backtesting

---

## 🎯 Session Achievements

### **Critical Blocker Resolved: M6J Data Gap**

**Problem Statement**:
- M6J (Micro USD/JPY) data ended 2024-03-18, missing 20+ months needed for validation
- Blocked fold 2 OOS (2024 Q2-Q4) and fold 3 OOS (2025 Q1) completely
- Without resolution, TSMOM baseline validation could not proceed

**Resolution Selected**: **Option A (Replace with Alternative FX Micro)**

**Actions Taken**:
1. ✅ Acquired Databento batch GLBX-20260107-AYH5HTQUB3 (8 instruments)
2. ✅ Extracted and imported M6B (Micro GBP/USD) with complete 2021-2026 coverage
3. ✅ Updated specification from v1.0 to v1.1 with formal change control (Section 2.4)
4. ✅ Updated management presentation (SLEEVE_TSMOM_PRESENTATION.md)
5. ✅ Updated data inventory (DOWNLOADED_DATA_MARKET.md) with full provenance
6. ✅ Updated project status (CLAUDE.md) to reflect READY FOR BACKTEST state

---

## 📊 Data Inventory Status

### **Futures Data (8 instruments) - ✅ COMPLETE**

| Symbol | Name | Coverage | Status |
|--------|------|----------|--------|
| **MES** | Micro S&P 500 | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars |
| **MNQ** | Micro Nasdaq-100 | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars |
| **M2K** | Micro Russell 2000 | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars |
| **MYM** | Micro Dow | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars |
| **MGC** | Micro Gold | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars |
| **MCL** | Micro Crude | 2021-07-11 → 2026-01-04 | ✅ 1,130 bars (late start acceptable) |
| **M6E** | Micro EUR/USD | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars |
| **M6B** | Micro GBP/USD | 2021-01-05 → 2026-01-04 | ✅ 1,260 bars (NEW) |

**Data Source**: Databento GLBX.MDP3 ohlcv-1d schema
**Format**: Parquet with rolled daily series (contract column included)
**Storage**: `dsp100k/data/tsmom/`
**Importer**: `databento_tsmom_importer.py` (395 lines, complete)

### **Bond Data (2 ETFs) - ⏳ PENDING**

| Symbol | Name | Coverage Needed | Status |
|--------|------|-----------------|--------|
| **TLT** | iShares 20+ Year Treasury | 2021-01-05 → 2026-01-05 | ⏳ To be acquired |
| **IEF** | iShares 7-10 Year Treasury | 2021-01-05 → 2026-01-05 | ⏳ To be acquired |

**Data Source**: Polygon.io (daily bars)
**Format**: Parquet (adjusted close preferred for total return proxy)
**Next Step**: Fetch via Polygon API and store in `dsp100k/data/tsmom/`

---

## 📝 Specification Updates

### **Version History**

**v1.0 (2026-01-07)**:
- Initial pre-registered specification
- 10 instruments: 8 futures (including M6J) + 2 bond ETFs
- Frozen baseline with kill-test criteria

**v1.1 (2026-01-08)** ← **CURRENT**:
- M6J replaced with M6B (formal change control)
- Added Section 2.4 documenting replacement rationale
- All other parameters unchanged (pre-registration integrity maintained)
- Universe: 8 futures (M6E, M6B) + 2 bond ETFs

### **Change Control Documentation (Section 2.4)**

```markdown
### 2.4 Change note (v1.1)
`M6J` was replaced with `M6B` due to incomplete `M6J` daily coverage
in the available 2021–2026 Databento delivery. `M6B` provides complete
coverage and preserves the intent (FX diversifier with micro sizing).
```

**Why This Matters**:
- Maintains pre-registration discipline (no silent spec edits)
- Explicit documentation prevents "data snooping" accusations
- Version increment signals material change to reviewers
- Rationale preserved for audit trail

---

## 🔬 Technical Decision: Option A vs Alternatives

### **Option A: Replace M6J with M6B (SELECTED)**

**Rationale**:
- ✅ Maintains 2×2 FX exposure (EUR + GBP instead of EUR + JPY)
- ✅ Keeps 10-instrument universe intact (statistical robustness)
- ✅ Preserves validation window (2021-2026, all 3 folds)
- ✅ Minimal spec change (just symbol swap in Section 2.1)
- ✅ M6B provides complete coverage (verified in delivery)

**Alternatives Rejected**:

**Option B: Use M6E-only for FX (9 instruments)**
- ❌ Reduces FX bucket to single instrument (concentrated risk)
- ❌ Risk budget adjustment needed (25% → single instrument)
- ❌ Less robust diversification in FX exposure

**Option C: Shorten validation window to 2021-2024 Q1**
- ❌ Reduces OOS data from 39 months → 27 months
- ❌ Fold 3 only 3 months (statistically weak)
- ❌ Missing 2024 H2 + 2025 market conditions

**Option D: Exclude M6J from Fold 3 only**
- ❌ Violates pre-registration (inconsistent universe across folds)
- ❌ Creates hidden degree of freedom
- ❌ Fails scientific rigor test

---

## 📦 Databento Batch Details (AYH5HTQUB3)

### **8 Instruments Acquired**

**Micro FX (2 - primary use)**:
- M6B.FUT (Micro GBP/USD) — Full coverage 2021-2026 ✅
- M6A.FUT (Micro AUD/USD) — Full coverage 2021-2026 ✅

**Full-Size FX (2 - research backup)**:
- 6J.FUT (JPY futures) — Full-size contract
- 6C.FUT (CAD futures) — Full-size contract

**Rates (2 - potential future use)**:
- ZN.FUT (10Y Treasury Note) — Full-size
- SR3.FUT (3M SOFR) — Full-size

**Commodities (2 - potential future use)**:
- HG.FUT (Copper) — Macro growth proxy
- ZC.FUT (Corn) — Agri diversifier

**Storage**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/databento/GLBX-20260107-AYH5HTQUB3/`

**Outcome**: M6B and M6A both provide complete micro-FX coverage and can replace M6J. M6B selected for spec v1.1 (preserves 2×2 FX structure with EUR + GBP).

---

## 📂 Documentation Updates

### **Files Updated**

1. **SLEEVE_TSMOM_MINIMAL_SPEC.md** (v1.0 → v1.1)
   - Updated FX universe: M6J → M6B
   - Added Section 2.4 change note
   - Version incremented, date updated to 2026-01-08

2. **SLEEVE_TSMOM_PRESENTATION.md**
   - Updated FX list: M6E, M6B (replaced M6J at line 43)
   - Maintains consistency with technical spec

3. **DOWNLOADED_DATA_MARKET.md**
   - Added new section for Databento batch AYH5HTQUB3
   - Documented all 8 instruments with full provenance
   - Listed conversion outputs (M6B_1d_2021-01-05_2026-01-04.parquet)
   - Updated data inventory summary table

4. **CLAUDE.md** (project status)
   - Header status: BLOCKED → READY FOR BACKTEST
   - TSMOM section rewritten with complete status update
   - Component status table added (futures ✅, bonds ⏳)
   - M6J blocker resolution documented
   - Next steps enumerated (TLT/IEF → backtester → kill-test)

### **New Files Created**

5. **TSMOM_SESSION_RECAP_2026-01-08.md** (this document)
   - Comprehensive session summary
   - M6J blocker resolution timeline
   - Data inventory status
   - Specification change control documentation
   - Next steps with acceptance criteria

---

## ✅ Acceptance Criteria Met

### **M6J Blocker Resolution**

- [x] Alternative FX micro identified (M6B)
- [x] Complete 2021-2026 coverage verified (1,260 daily bars)
- [x] Data acquired and imported (parquet in `data/tsmom/`)
- [x] Specification updated with change control (v1.0 → v1.1)
- [x] Management presentation updated (consistency)
- [x] Data inventory updated (full provenance)
- [x] Project status updated (CLAUDE.md)

### **Pre-Registration Integrity Maintained**

- [x] Version incremented (v1.0 → v1.1)
- [x] Change documented explicitly (Section 2.4)
- [x] Rationale explained (M6J incomplete coverage)
- [x] All other parameters unchanged (signal, portfolio, validation)
- [x] No parameter tuning or result-driven changes

---

## 🎯 Next Steps (Priority Order)

### **1. Acquire TLT + IEF Bond Data** (Next Immediate Task)

**Data Requirements**:
- **TLT**: iShares 20+ Year Treasury Bond ETF (daily, 2021-01-05 → 2026-01-05)
- **IEF**: iShares 7-10 Year Treasury Bond ETF (daily, 2021-01-05 → 2026-01-05)

**Data Source**: Polygon.io REST API
**Format**: Parquet with daily bars (adjusted close preferred for total return proxy)
**Storage**:
- `dsp100k/data/tsmom/TLT_1d_2021-01-05_2026-01-05.parquet`
- `dsp100k/data/tsmom/IEF_1d_2021-01-05_2026-01-05.parquet`

**Implementation Notes**:
- Use existing data acquisition patterns from ORB/TSMOM importers
- Verify dividend adjustment (adjusted close includes dividends = total return proxy)
- If unadjusted only, document limitation in results section

### **2. Implement TSMOM Backtester**

**File**: `dsp100k/src/dsp/backtest/tsmom_futures.py`

**Components to Implement**:
1. Signal calculation (252-day lookback, sign of trailing return)
2. Risk parity portfolio construction (8% vol target, covariance estimation)
3. Volume-led roll simulation (5-day MA, 3-day trigger per spec Section 3.5)
4. Walk-forward validation framework (3 expanding-window folds)
5. Kill criteria evaluation (Sharpe/PnL/DD/concentration gates)

**Reference Implementation**: `dsp100k/src/dsp/backtest/orb_futures.py` (ORB kill-test baseline)

**Expected Output**: JSON results file following ORB precedent with fold-level metrics

### **3. Run Baseline Backtest**

**Execution**:
```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
PYTHONPATH=src python -m dsp.backtest.tsmom_futures \
  --data-dir data/tsmom \
  --slippage 1 \
  --output data/tsmom/baseline_results.json
```

**Transaction Costs** (baseline):
- Futures: 1 tick/side + $1.24 commission
- ETFs: 2 bps/side + $0 commission

### **4. Run Stress Backtest**

**Execution**:
```bash
PYTHONPATH=src python -m dsp.backtest.tsmom_futures \
  --data-dir data/tsmom \
  --slippage 2 \
  --output data/tsmom/stress_results.json
```

**Transaction Costs** (stress = 2× slippage):
- Futures: 2 ticks/side + $1.24 commission
- ETFs: 4 bps/side + $0 commission

### **5. Evaluate Kill Criteria**

**Primary Gates** (baseline costs):
- Mean OOS Sharpe ≥ 0.50 ✅/❌
- OOS Net PnL > 0 ✅/❌
- Max Drawdown ≥ -20% ✅/❌
- Fold Consistency: ≥2/3 folds with Sharpe ≥0.25 AND PnL >0 ✅/❌

**Stress Gates** (2× slippage):
- OOS Net PnL > 0 ✅/❌
- Mean OOS Sharpe ≥ 0.30 ✅/❌
- Max Drawdown ≥ -25% ✅/❌

**Concentration Gates**:
- No single instrument >60% of absolute OOS PnL ✅/❌
- No single bucket >70% of absolute OOS PnL ✅/❌

**Verdict**: ALL gates must pass. If baseline fails: **KILL** (no parameter tuning).

### **6. Document Results**

**File**: `dsp100k/docs/SLEEVE_TSMOM_KILL_TEST_RESULTS.md`

**Template**: Follow ORB kill-test format (`SLEEVE_ORB_KILL_TEST_RESULTS.md`)

**Required Sections**:
- Executive summary (pass/fail verdict)
- Fold-by-fold performance metrics
- Per-instrument PnL contributions
- Per-bucket PnL contributions
- Concentration analysis
- Baseline vs stress comparison
- Failure analysis (if applicable)
- Verdict with gate checklist

---

## 📊 Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| 2026-01-07 | TSMOM spec v1.0 complete | ✅ Complete |
| 2026-01-07 | M6J data gap discovered | ✅ Identified |
| 2026-01-07 | Decision doc created (4 options) | ✅ Complete |
| 2026-01-08 | Databento batch AYH5HTQUB3 acquired | ✅ Complete |
| 2026-01-08 | M6B data imported | ✅ Complete |
| 2026-01-08 | Spec updated to v1.1 | ✅ Complete |
| 2026-01-08 | Documentation updated | ✅ Complete |
| **TBD** | **TLT/IEF data acquisition** | ⏳ **Next** |
| **TBD** | **Backtester implementation** | ⏳ Pending |
| **TBD** | **Kill-test execution** | ⏳ Pending |
| **TBD** | **Results documentation** | ⏳ Pending |

---

## 🎓 Key Learnings

### **Pre-Registration Discipline**

**What Went Right**:
- Version increment (v1.0 → v1.1) signals material change
- Section 2.4 explicitly documents replacement rationale
- No silent spec edits (maintains audit trail)
- Management presentation kept in sync with technical spec
- Data inventory updated with full provenance chain

**Why This Matters**:
- Prevents "data snooping" accusations (spec changed before seeing results)
- Maintains scientific rigor (no result-driven parameter changes)
- Enables external replication (complete documentation)
- Protects against overfitting (no post-hoc rationalization)

### **Robust Decision Framework**

**Four Options Evaluated**:
1. Option A: Replace with M6B ← **SELECTED**
2. Option B: Use M6E-only (9 instruments) ← Rejected (concentration risk)
3. Option C: Shorten validation window ← Rejected (statistical weakness)
4. Option D: Partial universe ← Rejected (violates pre-registration)

**Decision Criteria**:
- ✅ Maintains validation integrity (full 2021-2026 window)
- ✅ Preserves 10-instrument breadth (statistical robustness)
- ✅ Minimizes spec changes (only symbol swap)
- ✅ Maintains FX diversification (2×2 structure)

### **Data Acquisition Strategy**

**Proactive Approach**:
- Acquired 8 instruments (not just M6B alone)
- Secured M6A as alternate FX replacement option
- Obtained full-size contracts for research flexibility
- Added rates (ZN, SR3) and commodities (HG, ZC) for future use

**Benefits**:
- Flexibility for future sleeve variations
- Backup options if M6B data issues discovered later
- Research opportunities for cross-asset TSMOM extensions
- Single data acquisition cost vs multiple batches

---

## 📁 Related Documentation

**Core TSMOM Documents**:
- `SLEEVE_TSMOM_MINIMAL_SPEC.md` (v1.1) — Pre-registered specification
- `SLEEVE_TSMOM_PRESENTATION.md` — Management presentation
- `TSMOM_M6J_DATA_GAP_DECISION.md` — Options analysis (historical)
- `TSMOM_STATUS_SUMMARY.md` — High-level status tracker
- `DOWNLOADED_DATA_MARKET.md` — Complete data inventory

**Implementation References**:
- `src/dsp/data/databento_tsmom_importer.py` — Data extraction script
- `src/dsp/backtest/orb_futures.py` — ORB kill-test baseline (template)
- `docs/SLEEVE_ORB_KILL_TEST_RESULTS.md` — Results format reference

**Kill-Test Context**:
- `docs/SLEEVE_KILL_TEST_SUMMARY.md` — Kill-test methodology across all sleeves
- `docs/SLEEVE_ORB_IMPLEMENTATION_STATUS.md` — ORB precedent

---

## ✅ Session Status: COMPLETE

**Achievements**:
- ✅ M6J blocker resolved with Option A (M6B replacement)
- ✅ All 8 futures instruments ready with complete 2021-2026 coverage
- ✅ Specification updated to v1.1 with formal change control
- ✅ Documentation synchronized across technical and management docs
- ✅ Data inventory updated with full provenance chain
- ✅ Project status updated to READY FOR BACKTEST

**Next Session**:
- ⏳ Acquire TLT + IEF bond data from Polygon.io
- ⏳ Implement TSMOM backtester following spec Sections 3-10
- ⏳ Execute baseline + stress kill-tests
- ⏳ Evaluate gates and document verdict

**TSMOM Project Status**: 🟢 **UNBLOCKED** — Ready to proceed with backtest implementation

---

**Maintained By**: Development Team
**Last Updated**: 2026-01-08
**Commit**: (to be added after commit)
