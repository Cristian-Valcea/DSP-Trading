# SLEEVE_TSMOM_MINIMAL_SPEC.md — Cross-Asset Time-Series Momentum (TSMOM)
**Status**: DRAFT — Pending data + kill-test validation  
**Version**: 1.0 (pre-registered)  
**Date**: 2026-01-07  
**Authors**: User + Reviewer + Codex consolidation

---

## 0. Why This Spec Exists (and what it is NOT)
**Goal:** Define a *falsifiable*, *low-degree-of-freedom* baseline sleeve that tests whether cross-asset trend is robust enough to merit a production sleeve.

This spec is intentionally “boring”:
- No regime filters, no clever gates, no parameter search
- Portfolio construction + risk budgeting is the main design choice
- If the baseline fails, we **kill** (we do not tune until it works)

---

## 1. Strategy Identity
**Type:** Intraweek / multi-day trend following (time-series momentum)  
**Holding period:** Multi-day to multi-week (rebalanced weekly)  
**Objective:** A diversifying return stream vs equity beta and vs Sleeve DM

**Core hypothesis:** The sign of the trailing 12-month return contains predictive information about the sign of the next 1–4 weeks return across multiple asset classes, and the edge becomes tradable once we (a) diversify across instruments and (b) control turnover/costs.

---

## 2. Universe (Frozen)
We run a **10-instrument** universe spanning **4 economic buckets**. Futures are **micros** (size-compatible), rates exposure is via **ETFs** (micro rate futures are not reliably available/compatible with our sizing/data stack).

### 2.1 Micro Futures (Databento, GLBX.MDP3)
**Equity indices (3):**
- `MES` — Micro E-mini S&P 500
- `MNQ` — Micro E-mini Nasdaq-100
- `M2K` — Micro E-mini Russell 2000

**Commodities (2):**
- `MGC` — Micro Gold
- `MCL` — Micro Crude Oil

**FX (2):**
- `M6E` — Micro EUR/USD
- `M6J` — Micro JPY/USD

**Equity indices (1):**
- `MYM` — Micro E-mini Dow

> Note: This list is intentionally conservative (high liquidity, widely supported). If any symbol is unavailable in the chosen Databento dataset/schema, the sleeve is **blocked** until the universe is re-approved.

### 2.2 Rates (ETFs via Polygon)
- `TLT` — iShares 20+ Year Treasury Bond ETF
- `IEF` — iShares 7–10 Year Treasury Bond ETF

### 2.3 Universe invariants
- No instrument additions/removals mid-validation.
- No “swap MCL for something else” after seeing results.
- If an instrument lacks data for the required window, it is **flat** until sufficient history exists (no backfilled shortcuts).

---

## 3. Data & Return Conventions (Critical)
This sleeve is **daily** in logic, even if the raw vendor feed is intraday.

### 3.1 Futures data (Databento)
- Preferred schema: `ohlcv-1d` (daily bars) for the full validation window.
- If we only have `ohlcv-1m` for some products (e.g., existing MES/MNQ ORB purchase), we must aggregate to daily bars *before* running the sleeve.

**Daily bar definition (futures):**
- Use the vendor’s daily bar session definition when using `ohlcv-1d`.
- If aggregating from `ohlcv-1m`, the daily bar must be constructed using **exchange session boundaries** for that product (not “09:30–16:00 ET for everything”).

### 3.2 ETF data (Polygon)
- Frequency: daily bars.
- Use **adjusted close** (total return proxy) if available in our pipeline; otherwise use unadjusted close and accept that dividends are ignored (this is a known limitation and must be stated in results).

### 3.3 Timezone
All “trading day” alignment is done in `America/New_York`. Vendor timestamps in UTC must be converted consistently.

### 3.4 Futures roll handling (NO back-adjusted price series)
This is a major source of silent error. For this sleeve:
- We do **explicit roll simulation** using a pre-registered roll calendar (Section 3.5).
- Strategy PnL includes the economic effect of rolling (carry/roll yield), because that is part of a futures investor’s realized return.
- We do **not** use additive/ratio back-adjusted continuous prices to “hide” roll gaps.

### 3.5 Futures roll rule (pre-registered, product-agnostic)
We use a **volume-led roll** to avoid hidden degrees-of-freedom in “expiry calendar” implementations across different product families.

**Required data invariant:** For each futures root symbol (e.g., `MES`, `MCL`), the dataset must include **all listed contracts** that could become front/next during the window (not only the “front month”), otherwise the roll cannot be reproduced.

**Definitions (per root symbol):**
- “Eligible contracts” on day *t*: contracts that have a daily bar on *t* (not halted/absent).
- “Front” and “next”: the two nearest expiries by contract month ordering among eligible contracts.
- `V5(c, t)`: 5-day moving average of daily volume for contract *c* ending on day *t* (requires 5 days of history).

**Roll trigger:**
- We hold the current front contract until the next contract is clearly more liquid:
  - If `V5(next, t) > V5(front, t)` for **3 consecutive trading days**, we roll at the **close** of day *t* (the 3rd day).

**Execution:**
- On roll day close: close old contract, open new contract (same direction exposure).
- We do not roll more than once per week per root symbol (guardrail against noisy volume flips).

This rule is fully deterministic given daily volume and contract ordering.

### 3.6 Trading-day alignment & missing bars (pre-registered)
We compute returns on a **master sleeve calendar** defined as the **union** of all instrument trading days observed in the dataset over the window.

For any instrument *i* on day *t*:
- If *i* has a valid daily bar on *t*: compute the return normally.
- If *i* has **no** daily bar on *t*: treat the market as closed and set `r_i(t) = 0`.

**Data-quality guardrail:** A “missing bar” is only acceptable if it is consistent with market closure. If an instrument shows more than **5** missing bars inside an OOS fold that are not explainable by standard US market holidays, the dataset is considered incomplete and the backtest is **blocked** until fixed.

---

## 4. Signal (Zero-Cleverness Baseline)
### 4.1 Lookback
`lookback = 252` trading days (≈ 12 months).

### 4.2 Signal definition
For each instrument *i* on decision day *t*:
- Compute trailing return using daily closes:
  - `R_i(t) = close_i(t) / close_i(t - lookback) - 1`
- Signal:
  - `s_i(t) = sign(R_i(t))` in `{ -1, 0, +1 }`
  - If insufficient history: `s_i(t) = 0` (flat)

No strength scaling, no thresholds, no multi-horizon voting in v1.0.

---

## 5. Portfolio Construction (This is the real spec)
### 5.1 What we target (portfolio-level, not per-instrument)
We target a **single sleeve volatility**:
- `target_vol_sleeve = 8%` annualized

This is **not** “10% per instrument” (which would imply an absurd portfolio volatility).

### 5.2 Risk budget weights (pre-registered)
We allocate risk budget by bucket, then equally within bucket:
- Equities: 25% (MES, MNQ, M2K, MYM equally)
- Commodities: 25% (MGC, MCL equally)
- FX: 25% (M6E, M6J equally)
- Rates: 25% (TLT, IEF equally)

Let `w_i` be the per-instrument risk weight implied by the above.

### 5.3 Volatility and covariance estimates
All estimates are computed on **daily returns** and use only data strictly before the rebalance decision.
- Lookback for vol/cov: `60` trading days
- Return series includes **0-return** on non-trading days (as applicable) only if the market is truly closed for that instrument. We do not inject synthetic “0” for missing data.

### 5.4 Risk-parity style sizing with portfolio scaling
At each rebalance:
1. Compute recent covariance matrix `Σ` of instrument returns (60d).
2. Define “raw” exposures:
   - `e_raw_i = s_i * (w_i / σ_i)` where `σ_i = sqrt(Σ_ii)`
3. Compute portfolio volatility of raw exposures:
   - `vol_raw = sqrt(e_raw^T Σ e_raw) * sqrt(252)`
4. Scale all exposures to match the sleeve target:
   - `k = target_vol_sleeve / vol_raw` (if `vol_raw == 0`, set all to 0)
   - `e_i = k * e_raw_i`

Interpretation:
- `e_i` is target **notional exposure as a fraction of sleeve NAV** (i.e., leverage on that instrument).
- Portfolio targeting uses correlations (critical when equities are clustered).

### 5.5 Exposure caps (kill-switch against accidental leverage)
Hard caps (apply after scaling; if caps bind, rescale remaining proportionally):
- Gross exposure cap: `sum(|e_i|) <= 3.0`
- Per-instrument cap: `|e_i| <= 1.0`
- Per-bucket gross cap: `sum_{i in bucket} |e_i| <= 1.25`

Rationale: prevents “all risk piles into the lowest-vol instrument” during calm regimes and avoids leverage blowups from covariance estimation errors.

### 5.6 Converting exposures to tradeable quantities
Quantization rules (pre-registered):
- Futures: integer contracts, rounded to nearest; if rounding yields 0, position is 0.
- ETFs: integer shares, rounded to nearest; minimum trade size is 10 shares (otherwise skip).

Contract notionals use: `notional = price * contract_multiplier`.  
Multipliers/tick values must be sourced from a single authoritative table in code and validated once (fail fast if missing).

---

## 6. Rebalance & Execution Model
### 6.1 Schedule
Rebalance frequency: **weekly**.  
Rebalance day: first trading day of the week (normally Monday).

### 6.2 Decision vs execution timing (no lookahead)
- Signals and risk estimates are computed using data through **prior close**.
- Orders are executed at the **next open**.

### 6.3 Turnover deadband (to avoid pointless micro-churn)
We trade only if:
- Signal flips (`s_i` changes sign), OR
- Contract delta magnitude is ≥ 1 contract, OR
- ETF delta magnitude is ≥ 10 shares.

Otherwise hold.

---

## 7. Transaction Costs (Conservative, frozen)
We must report results under:
1) **Baseline** costs, and  
2) **Stress** costs (2× slippage).

### 7.1 Futures costs
- Commission: $1.24 per round-trip per contract (IBKR micros; update if broker schedule changes).
- Slippage: **1 tick per side** baseline; **2 ticks per side** stress.
- Apply slippage by worsening the execution price (entry and exit), not as an after-the-fact fee (this matters for path-dependent PnL).

### 7.2 ETF costs
- Commission: $0 (assumed), but record if non-zero.
- Slippage: 2 bps per side baseline; 4 bps per side stress.

---

## 8. Validation Design (appropriate for multi-day systems)
### 8.1 Required data window (minimum)
Because the signal uses 252d lookback and sizing uses 60d covariance:
- We need at least **~312 trading days** of history before the first OOS day.
- Therefore, if we want OOS to start on `2023-01-03`, we must have reliable data back to at least `2021-11-01` (conservatively: `2021-01-01`).

### 8.2 Walk-forward folds (expanding window, non-overlapping OOS)
We use OOS “years” to reduce noise:
- Fold 1: Train = 2021-01-01 → 2022-12-30, Test(OOS) = 2023-01-03 → 2023-12-29
- Fold 2: Train = 2021-01-01 → 2023-12-29, Test(OOS) = 2024-01-02 → 2024-12-31
- Fold 3: Train = 2021-01-01 → 2024-12-31, Test(OOS) = 2025-01-02 → 2025-03-31

Train is used only to provide sufficient lookback and initial covariance/vol estimates. There is **no** parameter fitting in v1.0.

---

## 9. Kill Criteria (aligned to a trend diversifier)
This sleeve is not an intraday “hit-rate” strategy. Win rate is not a promotion gate.

### 9.1 Primary gates (baseline costs)
All must pass:
- **Mean OOS Sharpe (daily returns, annualized √252, includes 0-return non-trading days):** `>= 0.50`
- **OOS Net PnL:** `> 0`
- **Max drawdown (on daily NAV):** `>= -20%`
- **Fold consistency:** at least **2/3** folds have `Sharpe >= 0.25` *and* `Net PnL > 0`

### 9.2 Stress gates (2× slippage)
All must pass:
- **OOS Net PnL:** `> 0`
- **Mean OOS Sharpe:** `>= 0.30`
- **Max drawdown:** `>= -25%`

Rationale: stress is a *fragility test*, not the primary performance target.

### 9.3 Concentration / “one-line-item fluke” gates
All must pass (baseline costs):
- No single instrument contributes more than **60%** of absolute OOS PnL:  
  `abs(pnl_i) / sum_j abs(pnl_j) <= 0.60`
- No single bucket contributes more than **70%** of absolute OOS PnL (same definition).

If the sleeve “works” only because one instrument had a lucky trend, we treat it as non-robust.

---

## 10. Outputs (pre-registered)
Backtest outputs must include:
- Fold-level metrics (PnL, Sharpe, DD, turnover, cost breakdown)
- Per-instrument and per-bucket PnL contributions
- Exposure statistics (gross exposure, max instrument exposure, max bucket exposure)
- Baseline and stress runs in separate JSON files

Schema should follow existing ORB/IM walk-forward JSON conventions to reduce tooling friction.

---

## 11. Freeze / Change Control
Once approved, **no changes** to:
- Universe
- Lookback windows (252 signal, 60 cov)
- Rebalance schedule (weekly)
- Portfolio risk target (8% sleeve vol) and caps
- Kill criteria

Allowed changes:
- Data parsing / timestamp bugs
- Contract spec table fixes (only if demonstrably wrong vs official specs)

If the baseline fails: **KILL**, do not “parameter search” inside this spec.
