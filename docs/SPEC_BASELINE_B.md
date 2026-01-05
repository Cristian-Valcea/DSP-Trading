# SPEC_BASELINE_B.md — Supervised Intraday Baseline “B” (Premarket + 1st Hour → 10:31→14:00)

## 0) Goal (one question)
Do our **existing** feature set (premarket + first hour RTH context) contain **tradable edge net of costs**, without RL?

This is a fast, clean “is there signal?” experiment.

## 1) Universe
Use the same 9 symbols:

`AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA`

## 2) Trading Schedule (ET)
Single decision per day:

- **Feature cutoff:** 10:30 (uses all data up to and including the 10:30 bar)
- **Entry:** 10:31 (use **10:31 bar open**)
- **Exit:** 14:00 (use **14:00 bar close**) — mandatory flatten

This baseline is **one entry + one exit per symbol per day** (no re-entry in v0).

## 3) Data Sources (current project conventions)
- **RTH minute bars (preferred for split reuse):** `../data/dqn_{split}/{symbol}_{split}.parquet`  
  Where `{split} ∈ {train, val, dev_test, holdout}`.
- **RTH upstream source (if needed):** `data/stage1_raw/{symbol}_1min.parquet`
- **Premarket minute bars (JSON cache):**
  - `data/dqn_premarket_cache/` (2021–2022 backfill)
  - `data/sleeve_im/minute_bars/` (2023–2024 backfill)

This spec assumes timestamps are in **ET** (after prior timezone fixes).

### 3.1 Splits (reuse existing DQN splits)
Use the already-defined DQN split date ranges (same symbols, same calendars):

- **TRAIN:** 2021-12-20 → 2023-12-29 (510 days)
- **VAL:** 2024-01-02 → 2024-06-28 (124 days)
- **DEV_TEST:** 2024-07-01 → 2024-12-31 (128 days)
- **HOLDOUT (do not touch until VAL+DEV_TEST pass):** 2025-01-02 → 2025-12-19 (243 days)

### 3.2 META ticker rename (FB → META)
For the same underlying company:
- Use **canonical symbol `META`** everywhere in this baseline.
- For dates before the rename (pre-2022-06-09), premarket files may contain `"symbol": "FB"` internally and/or exist under `data/dqn_premarket_cache/FB/`. This is normal.
- **Join by (canonical symbol, date)**:
  - When canonical symbol is `META`, load premarket from:
    1) `data/dqn_premarket_cache/META/{date}.json` if present, else
    2) `data/dqn_premarket_cache/FB/{date}.json` as fallback.
  - Treat both as **META** for feature/label generation.

## 4) How Premarket Data Is Used
We do **not** feed the full premarket time-series into the model. We compress it into a few summary signals per symbol/day.

### 4.1 Premarket segmented returns (your request)
Define three premarket return segments (ET):

- Segment A: **04:00–06:00**
- Segment B: **06:00–08:00**
- Segment C: **08:00–09:15**

For each segment, compute a **log return**:

- `r_pre_0400_0600 = log(P_0600 / P_0400)`
- `r_pre_0600_0800 = log(P_0800 / P_0600)`
- `r_pre_0800_0915 = log(P_0915 / P_0800)`

Optional derived totals:
- `r_pre_0400_0915 = r_pre_0400_0600 + r_pre_0600_0800 + r_pre_0800_0915`

### 4.2 “Carry-forward last print” rule (needed for thin premarket)
Premarket liquidity varies (GOOGL can be sparse). We make the segment prices well-defined using:

- Use the **last known close** at or before the segment boundary time (carry-forward).
- Also record **data availability** so “flat return” is not confused with “no prints”.

Minimum set of availability diagnostics per segment:
- `pre_bar_count_0400_0600`
- `pre_bar_count_0600_0800`
- `pre_bar_count_0800_0915`

If a segment has `pre_bar_count == 0`, the return is set to `0.0` **and** the bar_count feature communicates “missingness”.

### 4.3 Segment boundary price convention (open vs close)
Use **close prices** for the segment boundary prices (`P_0400`, `P_0600`, `P_0800`, `P_0915`), with carry-forward.

Rationale:
- Premarket is only used as an input feature (computed long before 10:31), so using close-at-boundary is stable and simple.
- Any missingness is represented explicitly via the bar-count features.

## 5) Feature Vector X(d, s) at 10:31
For each trading day `d` and symbol `s`, build one feature vector at the decision time.

### 5.1 Baseline v0 (reuse what we already trust)
Start from the existing 30-dim state at 10:30:

- `X_base(d,s) = StateBuilder.get_features(minute_idx=60)`

This already includes:
- overnight gap (#17)
- premarket return + premarket vol ratio (#18–19) (coarse summary)
- first-hour RTH dynamics and cross-asset context (#0–16, #20–29)

Reference implementation:
- `StateBuilder` lives at `dsp100k/src/dqn/state_builder.py` (see `FEATURE_NAMES` for the 30-dim schema).

### 5.2 Add the segmented premarket inputs
Append the 3 segmented returns + 3 bar-count diagnostics:

- `r_pre_0400_0600`
- `r_pre_0600_0800`
- `r_pre_0800_0915`
- `pre_bar_count_0400_0600`
- `pre_bar_count_0600_0800`
- `pre_bar_count_0800_0915`

### 5.3 Symbol conditioning (pooled model)
We will train one pooled model across all 9 symbols. Add a `symbol_id` one-hot (9 dims) to allow symbol offsets:

- `symbol_onehot(s) ∈ {0,1}^9`

### 5.4 Standardization
Standardize features using **train only** (mean/std from train split), then apply to VAL/DEV_TEST/HOLDOUT.

## 6) Label y(d, s)
Regression target per (day, symbol):

- `y(d,s) = log(P_exit / P_entry)`

Prices:
- `P_entry`: **10:31 bar open**
- `P_exit`: **14:00 bar close**

Skip the (day, symbol) if either bar is missing (halts/data gap/early close); no fallback in v0.

### 6.1 Index/time alignment (for implementers)
RTH bars start at **09:30 ET**. If you index from 09:30 as `minute_idx=0`:
- 10:30 corresponds to `minute_idx=60` (used for features)
- 10:31 corresponds to `minute_idx=61` (entry open)
- 14:00 corresponds to `minute_idx=270` (exit close)

Implementation guidance:
- Prefer selecting bars by **timestamp** (e.g., `timestamp.time() == 10:31`) for robustness.
- If a required timestamp is missing for a symbol/day, that sample is skipped and counted.

## 7) Model (v0)
Start simple and robust:
- Ridge regression on pooled data (with standardized inputs).

Optional v0.1:
- Lasso (feature sparsity check)
- LightGBM/XGBoost (only after Ridge baseline is understood)

Ridge alpha selection:
- v0 default: fixed alpha (e.g., `alpha=1.0`) to keep the first pass simple.
- v0.1: tune alpha on TRAIN only (walk-forward within TRAIN) or tune on VAL and confirm on DEV_TEST (preferred if keeping DEV_TEST as the final check).

## 8) Trading Rule (Portfolio Construction)
### 8.1 Constraints
- Max open positions: **9 total** (≤ 1 per symbol)
- Target gross exposure: **G = 10% of NAV**
- **Dollar-neutral is not required** (net exposure can be non-zero)
- Sizing is **proportional to predicted return** (your decision)

### 8.2 Costs (default)
- One-way all-in cost: `c = 10 bps = 0.001`
- Round-trip: `2c = 0.002`

### 8.3 Cost-aware scores (prevents trading pure noise)
For each symbol, get model prediction `ŷ(d,s)` (log-return forecast for 10:31→14:00).

Define a base no-trade threshold from costs:
- Base threshold: `T = 2c` (round-trip cost)

Add two multipliers to make the threshold stricter and/or asymmetric:
- Long threshold: `T_long = m_long * T`
- Short threshold: `T_short = m_short * T`

Trading direction mode (risk-control lever):
- `long_short` (default): allow both longs and shorts
- `long_only`: disable shorts entirely
- `net_long_bias`: allow shorts only if `ŷ` is **very** negative (use a larger `m_short`)

Signed score per symbol:
- If `ŷ(d,s) > T_long`: `score(d,s) = ŷ(d,s) - T_long`
- Else if shorts enabled and `ŷ(d,s) < -T_short`: `score(d,s) = ŷ(d,s) + T_short` (negative)
- Else: `score(d,s) = 0`

Eligibility:
- trade symbol `s` on day `d` iff `score(d,s) != 0` and the data-quality policy allows it.

### 8.4 Weighting proportional to score (gross-normalized)
Let `S = {s : eligible}` and `Z = sum_{s∈S} |score(d,s)|`.

- If `Z == 0`: go flat.
- Else allocate:
  - `w(d,s) = G * score(d,s) / Z`

Interpretation:
- `w > 0` ⇒ long
- `w < 0` ⇒ short
- `sum_s |w(d,s)| = G` by construction
- `sum_s w(d,s)` (net) floats; no neutrality constraint.

Optional concentration guard (recommended, but tuneable):
- per-name cap `|w(d,s)| <= w_cap` (default suggestion: `w_cap = 0.03`)
- if capping binds, redistribute remaining gross across uncapped names.

### 8.5 Execution (backtest semantics)
- Enter at 10:31 open, exit at 14:00 close.
- One entry and one exit per day for any traded symbol.

## 9) Backtest P&L Accounting (net of costs)
Daily portfolio log-return:

- `r_gross(d) = sum_s w(d,s) * log(P_exit / P_entry)`
- `r_cost(d)  = sum_s (2c) * |w(d,s)|`
- `r_net(d)   = r_gross(d) - r_cost(d)`

## 10) Evaluation (VAL first, then DEV_TEST)
Primary success metric:
- **Net CAGR** from `r_net(d)`

Report alongside:
- Max drawdown (equity curve from `exp(cumsum(r_net))`)
- Hit rate (percent of days with `r_net(d) > 0`)
- Turnover (will be low here; ~1 round trip per traded name per day)
- Activity (percent of days with at least one trade)

### 10.1 Proposed pass/kill gates (aligned with “profit-first”)
- `net_CAGR > 0` on **VAL** and **DEV_TEST**
- `MaxDD <= 15%`
- `trade_days >= 30` (avoid “lucky few trades”)

Sharpe can be reported, but is **not** the primary gate in this baseline.

## 11) Missing-data accounting (required)
Do not silently drop missing cases.

At minimum, log counts by split and by symbol:
- missing 10:31 entry bar
- missing 14:00 exit bar
- missing premarket segment bars (bar_count == 0)
- total samples produced vs skipped

These diagnostics are part of the acceptance criteria for the baseline run.

## 11) Next Extensions (only if v0 shows net_CAGR > 0)
If v0 passes, we can add controlled complexity:
- Multi-horizon labels (10:31→11:31 / 12:31 / 14:00) and select the best expected holding period.
- Add a net-exposure cap if needed (e.g., `|net| <= 5%`) to reduce market beta.

## 12) Output artifacts (suggested)
Keep outputs consistent with existing repo patterns:
- Model artifacts: `dsp100k/checkpoints/baseline_b/{run_id}/` (ridge + scaler + feature schema + run config)
- Results: `dsp100k/results/baseline_b/{run_id}/` (metrics JSON + skipped-sample report + equity curve)

`{run_id}` can be a timestamp or a short name like `ridge_v0_cost10bps_g10`.

## 13) Holdout policy (2025)
v0 should not automatically evaluate HOLDOUT.

It is fine to implement a holdout evaluation path **behind an explicit flag** (e.g., `--run-holdout`), but:
- HOLDOUT must not be used for hyperparameter tuning or model selection.
- HOLDOUT evaluation is only run after VAL and DEV_TEST pass the gates.
