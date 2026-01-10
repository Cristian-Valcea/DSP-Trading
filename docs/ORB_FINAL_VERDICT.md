# ORB Final Verdict (MES/MNQ Micro Futures)

**Date**: 2026-01-07  
**Decision**: 🔴 **KILL** (do not promote)  
**Scope**: Sleeve ORB as specified in `docs/SLEEVE_ORB_MINIMAL_SPEC.md` (MES+MNQ, 30-min ORB, RTH-only, 1–2 ticks/side cost model)

---

## Executive Summary (Management)
We implemented and tested an Opening Range Breakout (ORB) sleeve on **micro index futures**:
- **MES** (Micro S&P 500 futures)
- **MNQ** (Micro Nasdaq-100 futures)

Using **real 1-minute futures data (Databento, 2022–2025)**, ORB does **not** meet our robustness requirements:
- The strategy is **regime-dependent** (works in some quarters, fails in others).
- Results do not generalize across instruments (**MES positive, MNQ negative**).
- Risk-adjusted performance is below our promotion threshold.

**Outcome:** ORB is **not a reliable diversifier** for the portfolio and should not be deployed.

---

## Evidence (Kill-Test Outputs)
All results below are out-of-sample (6-fold walk-forward, 2022–2025). Output files:
- `dsp100k/data/orb/walk_forward_results.json` (baseline, 1 tick/side)
- `dsp100k/data/orb/walk_forward_results_stress.json` (stress, 2 ticks/side)
- `dsp100k/data/orb/walk_forward_results_no_target.json` (variant: no profit target)

### Baseline (1 tick/side): FAIL
- Total trades: `285`
- Total net PnL: `+$1,402.77`
- Mean Sharpe: `0.2341` (threshold `≥ 0.5`)
- Folds passing: `2/6` (threshold `≥ 4/6`)
- Per-symbol net PnL: `MES +$2,992.43`, `MNQ -$1,589.66`

### Stress (2 ticks/side): FAIL
- Total net PnL: `+$737.27`
- Mean Sharpe: `0.0952`
- Folds passing: `2/6`
- Per-symbol net PnL: `MES +$2,406.43`, `MNQ -$1,669.16`

### Variant (“No Target”): FAIL
- Total net PnL: `-$90.95`
- Mean Sharpe: `-0.0574`
- Folds passing: `4/6` (but fails on net PnL and Sharpe)

### Diagnostic: MES-only is closer but still fails promotion gate
File: `dsp100k/data/orb/walk_forward_results_mes_only.json`
- Total net PnL: `+$2,992.43`
- Mean Sharpe: `0.8535`
- Folds passing: `3/6` (still below `4/6` requirement)

**Conclusion:** even with variants, the signal is not stable enough to promote.

---

## Root Cause (Why ORB Failed)
1. **Signal robustness (primary):** performance flips across folds; ORB only works in some trend regimes.
2. **Instrument stability (primary):** the edge does not transfer cleanly across MES vs MNQ (MNQ persistently negative).
3. **Costs (secondary):** costs reduce outcomes, but the strategy fails robustness gates even before considering “optimization to perfection”.

---

## The $16 Databento Purchase Was Not Wasted
This data unlocks multiple intraday/intraweek research directions beyond ORB:
- Cross-asset intraday signals and regime filters (requires more instruments than MES/MNQ)
- Overnight drift studies (close→open / open→close decompositions)
- Event-window rules (macro releases), risk-off detection, volatility regime controls

The important value is that we now have a repeatable pipeline from raw vendor data → normalized 1-minute bars → walk-forward evaluation.

---

## Recommended Next Sleeves (More Likely to Be Robust)
These are designed to be **multi-instrument** and **lower turnover per unit of signal**, which tends to generalize better:

1. **Intraweek Time-Series Trend (2–10 day holds)**
   - Use simple trend signals (e.g., 5–20 day returns / MA) across multiple futures.
   - Vol-targeted; rebalanced daily/weekly; designed as a diversifier.
   - Data needed: add more products to Databento (e.g., `M2K`, `MCL`, `MGC`, `M6E`, and optionally rates).

2. **Overnight Drift Sleeve (close→open)**
   - Hold exposure only overnight (close to next open), flat intraday.
   - Evaluates whether the return source is more stable than intraday breakouts.
   - Data needed: 1-minute bars are sufficient (derive close/open consistently).

3. **Macro Event Window Sleeve (scheduled releases only)**
   - Trade only around pre-registered events (CPI/FOMC/NFP) with fixed entry/exit windows and tight risk caps.
   - Low trade count, clear walk-forward, easy to kill fast if no edge.
   - Data needed: existing 1-minute bars + event calendar.

---

## Decision
ORB (MES+MNQ) is **KILLED** and should not proceed to production or paper trading.

If ORB is revisited in the future, it should only be as part of a broader, cross-asset framework with explicit regime detection and re-validated from scratch.

