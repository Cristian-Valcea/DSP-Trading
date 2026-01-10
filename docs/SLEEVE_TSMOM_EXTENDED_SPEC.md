# SLEEVE_TSMOM_EXTENDED_SPEC.md — Cross-Asset TSMOM (Extended Research Universe)
**Status**: RESEARCH SPEC — Not eligible for promotion as-is  
**Version**: 0.1 (draft, pre-registration intent)  
**Date**: 2026-01-08  
**Scope**: Build and evaluate TSMOM on a broader cross-asset futures set using the same research discipline, while keeping the production-candidate sleeve (micro-only v1.1) unchanged.

---

## 0. Purpose (Why this exists)
We now own daily futures data beyond the micro universe. This spec defines a **separate** “extended universe” research track to:
- Test whether breadth materially improves robustness (portfolio effect).
- Learn which assets contribute diversifying trend returns.
- Avoid confusing research conclusions with a production-ready sleeve (contract sizes differ materially).

**Non-goal:** This document is *not* an approval to trade full-size futures. It is a research specification and must not be promoted without a separate sizing/margin review.

---

## 1. Strategy (same core as micro sleeve)
**Strategy class:** Time-Series Momentum (TSMOM), cross-asset trend following  
**Signal:** sign of trailing 12-month return (`252` trading days)  
**Rebalance:** weekly (first trading day of week)  
**Execution model:** decision at prior close; execute at next open; conservative slippage

This keeps “edge source” constant so differences are attributable to universe breadth rather than parameter tuning.

---

## 2. Universe
### 2.1 Baseline micro universe (from v1.1, unchanged)
Micros:
- `MES`, `MNQ`, `M2K`, `MYM`, `MGC`, `MCL`, `M6E`, `M6B`
ETFs (rates):
- `TLT`, `IEF` (Polygon daily)

### 2.2 Extended futures add-ons (Databento daily)
These are **full-size** or non-micro contracts. They are included for research breadth:
- FX: `6J` (JPY), `6C` (CAD)
- Rates: `ZN` (10Y Treasury Note), `SR3` (3M SOFR)
- Commodities: `HG` (Copper), `ZC` (Corn)
- (Optional micro FX): `M6A` (AUD) as an alternate micro FX diversifier

### 2.3 Universe versions (to prevent hindsight edits)
We evaluate three fixed universes:
- **U1 (Micro-only):** v1.1 sleeve universe (promotion candidate)
- **U2 (Micro + micro-FX extra):** U1 + `M6A`
- **U3 (Extended):** U1 + `M6A, 6J, 6C, ZN, SR3, HG, ZC`

No substitutions within U1/U2/U3 after results are observed.

---

## 3. Data Requirements & Handling
### 3.1 Data sources (already acquired)
- Futures daily: Databento `GLBX.MDP3`, schema `ohlcv-1d`, CSV+zstd
- Rates ETFs daily: Polygon stocks starter (TLT/IEF)

### 3.2 Instrument identification
Databento downloads include `symbology.csv` and a `symbol` field in the OHLCV files.
We treat symbols with `-` as **spreads** and ignore them; we use only **outright** contracts.

### 3.3 Continuous futures series construction
We do **explicit roll simulation**, not back-adjustment:
- Use the deterministic, volume-led roll rule implemented in `src/dsp/data/databento_tsmom_importer.py`.
- Keep a `contract` column so PnL can be attributed to the correct traded contract.

---

## 4. Portfolio Construction & Risk Targeting (extended-safe)
### 4.1 Sleeve-level risk target
Target sleeve volatility remains:
- `target_vol_sleeve = 8%` annualized

### 4.2 Bucket weights (extended)
To avoid the extended set being dominated by one macro complex, we keep the same high-level intent:
- Equities: 25%
- Commodities: 25%
- FX: 25%
- Rates: 25%

Within each bucket, weights are equal across instruments **present in that universe version**.

### 4.3 Exposure caps (critical when including full-size futures)
Hard caps apply to **all universe versions**, but are especially important for U3:
- Gross exposure cap: `sum(|e_i|) <= 2.0`
- Per-instrument exposure cap: `|e_i| <= 0.50`
- Per-bucket gross exposure cap: `sum_{i in bucket} |e_i| <= 0.90`

Rationale: full-size contracts can introduce accidental “effective leverage” after contract rounding.

### 4.4 Contract rounding rule
Positions must be tradeable:
- Futures: integer contracts, rounded to nearest.
- If rounding produces 0 contracts, that instrument is 0 exposure.

**Guardrail:** If contract rounding causes realized gross exposure to exceed caps, reduce positions proportionally (priority: reduce the largest exposures first).

---

## 5. Transaction Cost Model (extended)
Costs differ across micro vs full-size contracts and across asset classes.

### 5.1 Futures commission
Use a **per-contract round-trip commission table** in code (single source of truth), with defaults that can be updated to match the broker’s schedule.
- For research backtests, record both:
  - `commission_usd_per_rt` per contract
  - `exchange_and_reg_fees_usd_per_rt` (if modeled)

### 5.2 Slippage model
Slippage is modeled in **ticks per side**, because tick size differs by contract.
- Baseline: `1 tick/side`
- Stress: `2 ticks/side`

### 5.3 ETFs
Same as v1.1:
- 2 bps/side baseline; 4 bps/side stress

---

## 6. Validation (extended universe)
The extended universe is evaluated with the same fold structure as v1.1 so comparisons are apples-to-apples.

Folds (expanding, non-overlapping OOS):
- Fold 1 OOS: 2023 calendar year
- Fold 2 OOS: 2024 calendar year
- Fold 3 OOS: 2025-01-02 to 2025-03-31

We report:
- Fold-level Sharpe, DD, turnover, costs
- Contribution by instrument and by bucket
- “Breadth benefit” comparison: U1 vs U2 vs U3

---

## 7. Decision Rules (research outcomes, not promotion gates)
### 7.1 What we are looking for
U3 is considered *useful* if it shows:
- Higher robustness than U1 (more stable fold-to-fold),
- Lower concentration (no single instrument dominates),
- Similar or improved stress-cost survivability.

### 7.2 What we will NOT do
- We do not re-optimize lookbacks/thresholds per universe.
- We do not select instruments based on backtest “winners”.

### 7.3 Promotion boundary
Only **U1** (micro-only sleeve) can be considered for promotion under the current DSP-100K sizing constraints. U3 can inform:
- which additional micros to buy later (if they exist),
- whether a “separate managed futures sleeve” with larger capital is justified.

---

## 8. Implementation Artifacts
- Importer: `dsp100k/src/dsp/data/databento_tsmom_importer.py`
- Rolled daily series: `dsp100k/data/tsmom/*.parquet`
- Spec (micro sleeve): `dsp100k/docs/SLEEVE_TSMOM_MINIMAL_SPEC.md`
- Presentation: `dsp100k/docs/SLEEVE_TSMOM_PRESENTATION.md`

