# SPEC_BASELINE_C.md — Supervised Multi‑Time “Baseline C” (Re‑entry + Optional Overnight)

## 0) Goal
Baseline B (single decision 10:31→14:00) failed on both VAL and DEV_TEST.

Baseline C changes the **trading design** (multiple decision times + re‑entry + ability to hold overnight) while staying **supervised** (no RL), to answer:

> Do the same feature family become tradable if we allow smarter exits/holding periods (instead of forcing 14:00 flatten)?

## 1) Universe
Same 9 symbols:

`AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA`

## 2) Trading & Decision Schedule (ET)
### 2.1 Rebalance times
At each rebalance time we can change target weights per symbol (including exiting to 0, flipping long↔short, and re‑entering later).

Baseline C (v0) uses these **tradable rebalance times**:
- `T1 = 10:31`
- `T2 = 11:31`
- `T3 = 12:31`
- `T4 = 14:00` (**last tradable time of day**)

### 2.2 Re‑entry
Re‑entry is allowed:
- a symbol can be exited at one rebalance time and re‑entered at a later rebalance time (same day or later).
- max one position per symbol at any moment (so ≤ 9 open positions total).

### 2.3 Holding beyond 14:00 (no forced daily exit)
There is **no mandatory flatten at 14:00**.

If a position remains open after the 14:00 rebalance:
- it is held (unchanged) until the next rebalance time, which is **next trading day 10:31**.

This means you may carry positions **overnight** (and cannot exit until the next day’s 10:31 rebalance).

## 3) Data Sources & Splits
Reuse the existing DQN split files for RTH:
- `../data/dqn_train/{symbol}_train.parquet`
- `../data/dqn_val/{symbol}_val.parquet`
- `../data/dqn_dev_test/{symbol}_dev_test.parquet`
- Holdout later: `../data/dqn_holdout/{symbol}_holdout.parquet`

Premarket JSON caches (already in repo):
- `data/dqn_premarket_cache/` (2021–2022)
- `data/sleeve_im/minute_bars/` (2023–2024)

META rename handling (FB→META):
- canonical symbol is `META`
- for pre‑2022‑06‑09, load premarket from `META/{date}.json` or fallback `FB/{date}.json`.

## 4) Features (X) — computed at each rebalance time
For each (day `d`, symbol `s`, rebalance time `Tj`) we build one feature vector.

### 4.1 Premarket summary (same as Baseline B)
Inputs (per day/symbol):
- `r_pre_0400_0600`, `r_pre_0600_0800`, `r_pre_0800_0915` (log returns, carry-forward last print)
- `pre_bar_count_0400_0600`, `pre_bar_count_0600_0800`, `pre_bar_count_0800_0915`

### 4.2 RTH context up to the rebalance time
Reuse `StateBuilder` 30-dim features from `dsp100k/src/dqn/state_builder.py` at a time‑aligned index.

Convention:
- we compute features using data **through the prior minute close**,
- then trade at the **next minute open** (no look‑ahead).

Index mapping (09:30 is minute_idx=0):
- For 10:31 trades: feature minute_idx = 60 (10:30)
- For 11:31 trades: feature minute_idx = 120 (11:30)
- For 12:31 trades: feature minute_idx = 180 (12:30)
- For 14:00 trades: feature minute_idx = 269 (13:59)

Implementation note:
- `StateBuilder.get_features(minute_idx=...)` supports arbitrary minute indices (not just 60), as long as that index exists for the day.
- If a day is an early-close day (no 14:00 bar), the 14:00 rebalance is not tradable; v0 should **skip** those days entirely for simplicity (same behavior as Baseline B when 14:00 is missing).

### 4.3 Symbol conditioning
Append 9‑dim symbol one‑hot in fixed order:
`[AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA]`.

## 5) Labels (y) — interval returns between rebalances
This is a **multi‑horizon** supervised setup: we train a separate model per interval.

Execution price convention:
- Use **bar open** at each rebalance time as the execution price.

Intervals (v0):
- `y_10:31(d,s) = log(P_11:31_open / P_10:31_open)`
- `y_11:31(d,s) = log(P_12:31_open / P_11:31_open)`
- `y_12:31(d,s) = log(P_14:00_open / P_12:31_open)`
- `y_14:00(d,s) = log(P_nextday_10:31_open / P_14:00_open)`  (overnight + morning)

Overnight definition:
- `nextday` means the **next trading session** (not “tomorrow” on the calendar).  
  If `d` is a Friday or pre-holiday, this interval can span multiple calendar days; **include it as-is** (this matches the real risk of holding overnight/weekend).

Split boundary rule (required):
- Do not create labels that cross split boundaries (e.g., last VAL day 14:00 → next day in DEV_TEST is dropped for VAL).

Split evaluation boundary rule (required):
- Each split backtest starts **flat** at its first 10:31 rebalance (no carry-in positions from the prior split).
- At the end of a split, do **not** carry positions into the next split; treat the split’s last day as a forced flatten at 14:00 for evaluation purposes.

## 6) Models (v0)
Train 4 pooled Ridge regressors (one per interval), each on standardized features:
- Ridge(10:31→11:31)
- Ridge(11:31→12:31)
- Ridge(12:31→14:00)
- Ridge(14:00→next 10:31)

Notes:
- Standardization is fit on TRAIN only.
- Alpha can be fixed in v0 (simplicity), tuned later if we see promise.

## 7) Policy (how predictions become positions)
At each rebalance time `Tj`:
1. For each symbol `s`, predict next-interval return `ŷ_j(d,s)`.
2. Convert to a cost-aware score:
   - `edge = max(|ŷ| - 2c, 0)` where `c` is one‑way cost in return units (default `c=0.001` for 10 bps)
   - `score = sign(ŷ) * edge`
3. Convert scores to target weights under gross constraint `G=10%`:
   - if `Z = sum_s |score_s| == 0`: target all weights to 0
   - else `w_target(s) = G * score_s / Z`
4. Apply trades to move from previous weights `w_prev` to `w_target` (this naturally supports re‑entry and per-symbol exits).

No dollar‑neutral constraint; net exposure is allowed to float.

## 8) Costs & P&L accounting (per rebalance interval)
At rebalance time `Tj` (execution at bar open):
- turnover = `sum_s |w_target(s) - w_prev(s)|`
- cost = `c * turnover`

Cost timing clarification:
- Costs are charged **at every rebalance time** where weights change.
- Therefore, for the 14:00→next 10:31 interval:
  - any position change at **14:00** incurs cost at 14:00,
  - any position change at **next-day 10:31** incurs cost at 10:31 (as part of entering the 10:31→11:31 interval).

Then hold `w_target` through the interval and realize:
- gross = `sum_s w_target(s) * log(P_next / P_now)`
- net interval return = `gross - cost`

## 9) Evaluation & Gates
Evaluate on VAL, then DEV_TEST (holdout 2025 only if both pass):

Primary:
- net CAGR (profitability after costs)

Risk:
- Max drawdown ≤ 15%

Sanity:
- minimum trade activity, to avoid trivial “always flat”.

v0 gate definition:
- **Active days ≥ 30** per split, where an active day means `gross_exposure > 0` for at least one interval on that day.

## 10) Why this is the right next step (after Baseline B)
Baseline B only tested one horizon (10:31→14:00) with a forced exit.

Baseline C tests whether:
- the same information is useful for **shorter intraday intervals** and/or
- holding winners longer (including overnight) improves net performance
without changing the model class or adding RL complexity.
