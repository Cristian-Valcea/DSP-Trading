# SPEC_SLEEVE_CARRY.md — ETF Carry Sleeve (FX + Rates Proxies)

**Version**: 1.0  
**Date**: 2026-01-10  
**Status**: Pre-Registered (Frozen after approval)  
**Audience**: DSP-100K engineering + reviewers  

## 1. Objective
Add a new **carry** sleeve that is distinct from DM (momentum) and VRP-CS (volatility term structure) by harvesting **interest-rate differentials** using liquid ETFs as proxies.

This is an **ETF-first** implementation (no futures curve data required). If the ETF version passes kill tests, we can graduate to futures later.

## 2. Universe (ETF Proxies)
### 2.1 FX ETFs (USD base)
- `FXE` (EURUSD)
- `FXY` (JPYUSD)
- `FXB` (GBPUSD)
- `FXA` (AUDUSD)
- `UUP` (DXY / USD basket proxy)

### 2.2 Rates ETFs (USD)
- `SHY` (1–3y Treasuries; cash proxy)
- `IEF` (7–10y Treasuries)
- `TLT` (20+ year Treasuries)

Notes:
- The sleeve is designed to work with Polygon daily bars (already subscribed).
- FX ETFs embed local cash yield via their holdings; this makes them reasonable proxies for carry at weekly horizons.

## 3. Core Hypothesis (Testable)
At weekly horizons, currencies with **higher short rates vs USD** tend to outperform lower-rate currencies on average (positive carry), but suffer crash risk in risk-off regimes.

We therefore:
- run a **cross-sectional carry basket**
- apply a **crisis throttle** using the already-built VRP regime gate (optional but pre-registered here)

## 4. Data Requirements
### 4.1 Prices (required)
- Daily OHLCV for all ETFs above from Polygon (2010–present).

### 4.2 Short-rate proxies (required for “true carry score”)
We use **free FRED** short-rate series (OECD 3-month money market rates) to avoid paid data.

Required FRED series (exact tickers must be verified before implementation):
- US: 3M T-bill (`DTB3`) or equivalent (already in `dsp100k/data/vrp/rates/tbill_3m.parquet`)
- Euro Area: OECD 3M (`IR3TIB01EZM156N`)
- Japan: OECD 3M (`IR3TIB01JPM156N`)
- UK: OECD 3M (`IR3TIB01GBM156N`)
- Australia: OECD 3M (`IR3TIB01AUM156N`)

If any series is missing or too sparse, we fall back to the closest available policy rate series on FRED (must be explicitly documented and then frozen).

## 5. Signal Construction
### 5.1 Carry Differential (weekly)
For each currency `c`:

`carry_diff_c = r_c - r_US`

Where `r_c` is the foreign short rate, `r_US` is US short rate.

### 5.2 Cross-Sectional Portfolio
Each rebalance date (weekly, Monday close → trade Tuesday open, or next trading day):
- Rank currencies by `carry_diff_c`
- Long top **2**
- Short bottom **2**
- Equal-weight legs (risk parity is v1.1; v1.0 keeps it simple)

Implementation note:
- Shorts are implemented by shorting the corresponding FX ETFs (margin required).
- `UUP` is not in the long/short ranks; it is used only as a sanity-check proxy for “USD strength regime” (optional diagnostic).

### 5.3 Rates “carry anchor”
Allocate a fixed portion of the sleeve to rates carry:
- Baseline: 25% allocated to `IEF` (intermediate duration)
- Cash buffer: 25% allocated to `SHY`

Rationale:
- This dampens the pure FX crash profile and makes the sleeve more robust.

Total sleeve allocation (v1.0):
- 50% FX carry basket (L/S)
- 25% IEF (long-only)
- 25% SHY (long-only)

## 6. Risk Management
### 6.1 Vol-Targeting Overlay (portfolio-level)
The global overlay from `SPEC_VOL_TARGET_OVERLAY.md` applies on top of this sleeve’s notional (i.e., it can scale the sleeve exposure as part of the portfolio).

### 6.2 VRP Regime Gate (optional but pre-registered)
Use `VRPRegimeGate` states to throttle FX carry only:
- `OPEN`: 100% FX basket notional
- `REDUCE`: 50% FX basket notional
- `CLOSED`: 0% FX basket notional (rates anchor remains 50% in SHY/IEF)

This is intended to avoid classic carry crash behavior in volatility spikes.

### 6.3 Position limits
- Max gross FX notional: 50% of sleeve NAV
- Max single ETF absolute notional: 20% of sleeve NAV

## 7. Rebalance & Execution
- Rebalance frequency: weekly (once per week, fixed day)
- Orders: use limit-at-mid guidance for liquid ETFs; if no mid available, use close-based slippage model in backtest

## 8. Cost Model (Backtest)
- Slippage: 5 bps per side (entry and exit)
- Commission: $0.005/share
- Borrow cost on shorts: **ignored in v1.0 backtest** (must be added before live if shorts are used).  
  If borrow cannot be reliably modeled, run the sleeve long-only variant as a fallback:
  - Long top 2 carry currencies
  - Remaining FX allocation to SHY

## 9. Validation / Walk-Forward
Use the same philosophy as other sleeves:
- Train window is **warm-up only** (no parameter tuning)
- OOS evaluation windows: yearly folds (2018–2024) to match current DSP windows

## 10. Kill Criteria (v1.0)
PASS only if all hold:
- Net PnL > 0 (after costs)
- Annualized Sharpe ≥ 0.50
- Max drawdown ≥ -20%
- Fold pass rate ≥ 2/3 (for 3 yearly folds)
- “Gate value-add”: gated variant Sharpe ≥ ungated Sharpe (if gate enabled)

KILL if any fail.

## 11. Deliverables (Implementation)
Planned code artifacts:
- `dsp100k/src/dsp/backtest/etf_carry.py` (backtester)
- `dsp100k/src/dsp/sleeves/sleeve_carry.py` (production sleeve, only if PASS)
- `dsp100k/config/sleeve_carry.yaml` (frozen parameters)
- `dsp100k/data/carry/` (inputs + results)

---

**Approval to Freeze (v1.0):** ____________________  **Date:** __________

