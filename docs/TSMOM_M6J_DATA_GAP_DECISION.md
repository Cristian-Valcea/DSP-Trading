# TSMOM M6J Data Gap — Decision Required

**Date**: 2026-01-07
**Status**: 🟡 **BLOCKED** — Awaiting decision on M6J coverage gap before proceeding with backtest

---

## Problem Statement

The Databento GLBX.MDP3 batch (job GLBX-20260107-MXWMXNTA6P) delivered daily data for all 8 requested micro futures, but **M6J (Micro USD/JPY) coverage ends on 2024-03-18** — more than 20 months before the target end date (2026-01-05).

**Impact**:
- TSMOM spec requires full 2021-2026 coverage for all instruments to maintain validation integrity
- M6J data stops in March 2024, meaning we have no data for:
  - Fold 2 OOS period: 2024-01-02 to 2024-12-31 (✅ partial: Jan-Mar only)
  - Fold 3 OOS period: 2025-01-02 to 2025-03-31 (❌ completely missing)

---

## Root Cause

**Databento batch limitations** — the delivered data only includes what was available in their system for M6J at the time of request. This is not a code bug or request error; it's a **vendor data availability** issue.

---

## Options for Resolution

### **Option A: Replace M6J with Another FX Micro (Recommended)**

**Action**: Replace `M6J` (USD/JPY) with a different FX micro that has full coverage.

**Candidates**:
- `M6A` (Micro AUD/USD) — check Databento availability
- `M6B` (Micro GBP/USD) — check Databento availability
- Alternative: Use `6E` (full-size EUR/USD) if no other micro FX available

**Pros**:
- ✅ Maintains 2×2 FX exposure (EUR + one other)
- ✅ Keeps validation window intact (2021-2026)
- ✅ Minimal spec change (just swap symbol in universe section)

**Cons**:
- ⚠️ Requires re-requesting data from Databento (additional cost)
- ⚠️ Need to verify new symbol has full coverage before committing

**Implementation**:
1. Check Databento coverage for `M6A` or `M6B` (2021-01-05 to 2026-01-05)
2. If full coverage exists, request new batch
3. Update spec Section 2.1 to replace `M6J` with chosen symbol
4. Re-run `databento_tsmom_importer.py` with new data
5. Document change in `DOWNLOADED_DATA_MARKET.md`

---

### **Option B: Use Single FX Instrument (M6E Only)**

**Action**: Drop `M6J` from universe entirely, use only `M6E` (Micro EUR/USD) for FX exposure.

**Universe becomes**:
- Equities: 4 instruments (MES, MNQ, M2K, MYM)
- Commodities: 2 instruments (MGC, MCL)
- **FX: 1 instrument (M6E only)**
- Rates: 2 instruments (TLT, IEF)
- **Total: 9 instruments** (down from 10)

**Pros**:
- ✅ No additional data cost
- ✅ M6E has full coverage (2021-2026)
- ✅ Can proceed with backtest immediately

**Cons**:
- ❌ Reduces FX bucket to single instrument (less robust)
- ❌ Risk budget weights need adjustment (Section 5.2):
  - Current: FX bucket gets 25% risk weight split between M6E and M6J (12.5% each)
  - New: FX bucket gets 25% risk weight to M6E only (concentrated)
- ⚠️ Less diversification in FX exposure (only EUR/USD, no JPY/USD)

**Implementation**:
1. Update spec Section 2.1 to list 9 instruments (remove M6J)
2. Update spec Section 5.2 risk budget: FX bucket = 25% to M6E only
3. Document decision in spec with rationale
4. Proceed with backtest using existing data

---

### **Option C: Shorten Validation Window to 2021-2024 Q1**

**Action**: Adjust validation folds to only use data where all instruments have coverage.

**New validation design**:
- Fold 1: Train 2021, Test 2022 (12 months OOS)
- Fold 2: Train 2021-2022, Test 2023 (12 months OOS)
- Fold 3: Train 2021-2023, Test 2024 Q1 (3 months OOS)

**Pros**:
- ✅ Uses all instruments as originally specified
- ✅ No additional data cost
- ✅ M6J available for full validation window

**Cons**:
- ❌ Reduces total OOS data to 27 months (vs 39 months in original design)
- ❌ Fold 3 only 3 months (less statistically robust)
- ❌ Missing recent market conditions (2024 H2, 2025)
- ⚠️ Weakens "fold consistency" gate (fewer folds, less conviction)

**Implementation**:
1. Update spec Section 8.2 with new fold definitions
2. Document rationale (M6J data limitation)
3. Proceed with backtest using existing data

---

### **Option D: Exclude M6J from Fold 3 Only (Partial Universe)**

**Action**: Use M6J for Folds 1-2, exclude it from Fold 3 (treat as "flat" in 2025).

**Pros**:
- ✅ Maximizes data usage (M6J contributes where available)
- ✅ No additional cost

**Cons**:
- ❌ Violates spec Section 2.3 "no instrument additions/removals mid-validation"
- ❌ Creates inconsistent universe across folds (complicates interpretation)
- ❌ Risk budget weights change between folds (not pre-registered)
- ⚠️ This is a "hidden degree of freedom" — fails spirit of pre-registration

**Recommendation**: ❌ **DO NOT USE** — violates spec integrity

---

## Recommendation

**Priority Order**:

### **1st Choice: Option A (Replace M6J with M6A or M6B)**
- Check Databento coverage for alternative FX micros
- If full coverage exists, request new batch
- Minimal spec impact (just symbol swap)
- Maintains validation integrity and 10-instrument breadth

### **2nd Choice: Option B (Single FX Instrument)**
- If Option A not viable (no alternative FX micro with full coverage)
- Proceed with 9 instruments (M6E only for FX)
- Update risk budget in spec
- Document as "FX bucket concentrated in EUR/USD only"

### **Avoid**: Options C and D
- Option C: Shortening window weakens statistical power
- Option D: Violates pre-registration spirit

---

## Next Steps

**Immediate Action Required**:
1. **Check Databento availability** for `M6A` (AUD/USD) and `M6B` (GBP/USD) covering 2021-01-05 to 2026-01-05
2. **If full coverage exists**: Request new batch, proceed with Option A
3. **If no full coverage**: Implement Option B (9-instrument universe)
4. **Update spec** with chosen resolution
5. **Proceed with backtest** once data/spec aligned

---

## MCL Data Gap (Secondary Issue)

**MCL (Micro Crude) starts 2021-07-11** (not 2021-01-05).

**Impact**:
- Missing ~130 trading days (Jan-Jul 2021)
- Affects warm-up period for Fold 1 train window only
- Does NOT affect OOS periods (all OOS starts 2022+)

**Resolution**:
- ✅ **Acceptable as-is** per spec Section 2.3:
  > "If an instrument lacks data for the required window, it is **flat** until sufficient history exists"
- MCL will be flat in early 2021 train period, fully active from July 2021 onward
- No spec change needed

---

## Status Summary

| Issue | Severity | Status | Action |
|-------|----------|--------|--------|
| **M6J ends 2024-03-18** | 🔴 **BLOCKER** | Pending decision | Choose Option A or B |
| **MCL starts 2021-07-11** | 🟡 Minor | Accepted | Flat until Jul 2021 (spec allows) |

---

**Decision Maker**: User/Management
**Technical Blocker Owner**: Development team (awaiting decision)
