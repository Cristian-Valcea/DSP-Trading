# SLEEVE_ORB_MINIMAL_SPEC.md — Opening Range Breakout (Futures)

**Status**: DRAFT — Pending kill-test validation
**Version**: 1.6
**Date**: 2026-01-06
**Authors**: User + Reviewer + Claude consolidation

---

## 1. Strategy Identity
**Type:** Intraday Momentum / Volatility Breakout
**Thesis:** The first 30 minutes of the US cash session establishes a valuation range. A breakout from this range, filtered for volatility regimes, indicates a high probability of trend continuation for the session.

**Key Differentiator from Failed Sleeve IM/DQN:**
- ORB is *reactive* (follow momentum), not *predictive* (guess direction)
- 1-2 trades/day vs 200+ trades/month
- ~$4/contract cost vs 20 bps/side ($200+ per $100K position)
- Simple rules vs complex ML features

## 2. Universe & Session

### A. Contract Specifications (MICROS ONLY)

| Contract | Underlying | Point Value | Tick Size | Tick Value | Margin (approx) |
|----------|------------|-------------|-----------|------------|-----------------|
| **MES** | S&P 500 Index | $5/pt | 0.25 pts | $1.25/tick | ~$1,500 |
| **MNQ** | Nasdaq-100 Index | $2/pt | 0.25 pts | $0.50/tick | ~$1,800 |

⚠️ **Micros only**: We trade MES/MNQ (micros), NOT ES/NQ (minis). Minis are 10× larger ($50/pt, $20/pt) and require different sizing.

### B. Cost Model (Tick-Based Derivation)

| Component | MES (ticks) | MES ($) | MNQ (ticks) | MNQ ($) |
|-----------|-------------|---------|-------------|---------|
| Commission (RT) | - | $1.24 | - | $1.24 |
| Slippage entry | 1 tick | $1.25 | 1 tick | $0.50 |
| Slippage exit | 1 tick | $1.25 | 1 tick | $0.50 |
| **Total RT cost** | **2 ticks + comm** | **$3.74** | **2 ticks + comm** | **$2.24** |

*Execution penalty modeled as 1 tick per side (spread+slippage) as baseline; kill-test also reruns at 2 ticks/side as stress test.*

### C. Session & Data

*   **Primary Contracts:** `MES` (Micro S&P 500), `MNQ` (Micro Nasdaq 100).
*   **Session:** RTH Only (09:30 – 16:00 ET). **No ETH/Globex trading.**
*   **Granularity:** 1-minute OHLC bars (RTH session only, not 24h).
*   **Data Source:** Polygon.io (`/v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}`)
*   **Timezone:** All strategy logic uses `America/New_York` (handles EST/EDT). Vendor timestamps (UTC) must be converted to ET before computing OR, entries, and exits.
*   **Polygon tickers (must be pre-registered before backfill):**
    *   This spec assumes 1-minute `MES`/`MNQ` data is available via the DSP-100K pipeline.
    *   Exact Polygon ticker strings for micro futures depend on whether we use a **vendor continuous series** vs **per-contract tickers**.
    *   Before coding/backfill, we must choose and document one of:
        1. Vendor continuous ticker (preferred if Polygon provides it) OR
        2. Per-contract tickers + our fixed roll schedule (N=5) to build a continuous series.
    *   Once decided, record the chosen ticker format(s) in `config/sleeve_orb.yaml` and in this section.

### D. Contract Choice Rationale

**We are trading Micros exclusively (MES, MNQ), not Minis (ES, NQ).**

**Why Micros?** Granular position sizing. A single ES contract with a 20-point stop represents $1,000 risk ($50/pt × 20 pts). If our risk budget per trade is $200 (20 bps of $100K), we cannot trade ES without massively over-risking. MES at $5/pt × 20 pts = $100 risk allows proper sizing.

| Contract | Point Value | 20-pt Stop Risk | Fits $200 Budget? |
|----------|-------------|-----------------|-------------------|
| ES (mini) | $50/pt | $1,000 | ❌ 5× over-risk |
| MES (micro) | $5/pt | $100 | ✅ Can size 2 contracts |
| NQ (mini) | $20/pt | $400 | ❌ 2× over-risk |
| MNQ (micro) | $2/pt | $40 | ✅ Can size 5 contracts |

**Bottom line:** Minis require $500K+ sleeve NAV to size properly at 20 bps risk. Micros work at $25K+.

### E. Data Vendor & Backtest Constraints

**Data Source:** Polygon.io (via existing DSP-100K data pipeline)

**Constraints:**
1. **1-minute OHLC bars only** — No tick data available. We cannot reconstruct intra-bar order of events.
2. **RTH data only** — We fetch RTH session (09:30–16:00 ET), not 24-hour Globex data. This is intentional: ORB is an RTH strategy.
3. **No bid/ask spread data** — Execution penalty modeled as 1 tick per side (baseline assumption for liquid micros during RTH). Also re-run kill-tests at 2 ticks/side as a stress test.

**Backtest Resolution (Pessimistic Same-Bar Rule):**

Because we only have 1-min OHLC (no tick data), when both stop AND target are touched in the same bar, we cannot know which was hit first. We assume **STOP was hit first** (maximum loss scenario).

```python
# Same-bar conflict: Stop AND Target both in range
if bar.low <= stop_price and bar.high >= target_price:
    # No tick data → assume worst case
    fill_price = stop_price  # Loss, not profit
    exit_reason = "STOP (pessimistic)"
```

This prevents overfitting to lucky bar sequences that might have been stops in reality.

**Why Polygon.io?**
- Already integrated in DSP-100K data pipeline
- Sufficient for strategy logic (OR construction, ATR calculation)
- Cost-effective ($29/mo) vs tick-level feeds ($500+/mo)
- If strategy passes kill-test with pessimistic assumptions, it's robust

### F. Contract Rollover & Continuous Series

```python
# Data reality check:
# - If we use a vendor-provided continuous futures ticker (e.g., Polygon continuous series),
#   then roll logic is vendor-defined; document the ticker and the vendor's roll/adjust rules.
# - If we build our own continuous series from individual contracts, we must pre-register the roll rule.
#
# Pre-registered roll rule for ORB (default if we build it ourselves): Fixed schedule
# - Roll N trading days before the last trading day (default N=5).
# - Rationale: deterministic, easy to reproduce, avoids "volume crossover" ambiguity with 1-min OHLC-only data.
#
# Back-adjustment method for ORB (pre-registered):
# - Use additive (difference) back-adjustment to remove roll gaps while preserving point/tick moves.
# - Do NOT use ratio adjustment for this sleeve because ORB uses absolute price levels (OR_high/OR_low) and
#   stop distances in index points.
#
# Live execution:
# - Trade the front-month micro contract and roll on the same fixed schedule (or earlier if liquidity shifts),
#   updating the contract symbol in config.

# ATR calculation: Use RTH bars only for ATR_d
# - Exclude overnight gaps from True Range calculation
# - TR = max(H-L, |H-prev_close_RTH|, |L-prev_close_RTH|)
```

⚠️ **Roll dates affect OR/ATR**: Document exact roll dates in backtest. Check for anomalies around quarterly expirations (Mar, Jun, Sep, Dec).

## 3. Core Logic (ORB-30)

### A. Construction
1.  **Opening Range (OR):** Use the 30 one-minute bars with bar-open timestamps `09:30` through `09:59` ET (inclusive). This is the half-open interval **[09:30, 10:00)** and explicitly **excludes** the `10:00` bar.
2.  **OR Width:** `OR_High - OR_Low`.
3.  **Daily ATR (RTH-only definition):**
    ```python
    # ATR_d = 14-period ATR on DAILY RTH bars
    # Use RTH close-to-close, NOT 24h session
    # This avoids overnight gap inflation of True Range

    def true_range_rth(day):
        # H, L from today's RTH session (09:30-16:00)
        # prev_close from YESTERDAY's RTH close (16:00)
        H = day.rth_high
        L = day.rth_low
        prev_C = previous_day.rth_close
        return max(H - L, abs(H - prev_C), abs(L - prev_C))

    ATR_d = rolling_mean(true_range_rth, window=14)
    ```
    ⚠️ **Computed on prior day's close** (no lookahead). On day T, ATR_d uses days [T-14, T-1].

### B. Filters (Regime Guards)
*Trade is SKIPPED if any condition is met:*
1.  **Compression (Chop Risk):** `OR_Width < 0.20 * Average_OR_Width_20d` (Range too tight, likely false breakout).
2.  **Exhaustion (Whiplash Risk):** `OR_Width > 2.0 * Average_OR_Width_20d` (Range too wide, move already happened).
3.  **News:** Date is in `SKIP_DATES` (FOMC, CPI, NFP).

**News Calendar Handling (Pre-Registered):**

| Event | Time (ET) | Action | Rationale |
|-------|-----------|--------|-----------|
| **FOMC** | 14:00 | Force flat by 13:55 if in position | Statement causes 1-2% moves |
| **CPI** | 08:30 | Skip entire day | Pre-market release, OR is polluted |
| **NFP** | 08:30 | Skip entire day | Pre-market release, OR is polluted |
| **Quad-witching** | All day | Skip entire day | Erratic vol from expiration flows |
| **Half-days** | 13:00 close | Skip entire day | Truncated session |

```python
# News handling logic
def should_trade(date, current_time=None):
    if date in SKIP_DATES_FULL:  # CPI, NFP, quad-witching, half-days
        return False, "SKIP_FULL_DAY"

    if date in FOMC_DATES:
        if current_time is None:
            return True, "FOMC_DAY"  # Can enter, but forced exit later
        if current_time >= time(13, 55):
            return False, "FOMC_FLATTEN"  # Force flat before 14:00

    return True, "NORMAL"

# In position management:
if in_position and is_fomc_day and current_time >= time(13, 55):
    flatten_position("Pre-FOMC mandatory exit")

# On FOMC days:
# - Entry is allowed only until 13:55 ET (we do not allow new entries after the flatten cutoff).
# - Cancel any unfilled entry orders at 13:55 ET as part of the flatten procedure.
```

⚠️ **Filter discipline**: These filters are **frozen** before backtest. No post-hoc tuning.

### C. Entry (OCO - One Cancels Other)
*   **Time Window:** Entry allowed between `10:00:00` and `14:00:00` ET.
*   **Buffer:** 2 ticks (0.50 pts for MES/MNQ).
*   **Triggers:**
    *   **Long:** Stop Market at `OR_High + Buffer`.
    *   **Short:** Stop Market at `OR_Low - Buffer`.
*   **Constraint:** Max 1 trade per symbol per day. No re-entries.
*   **Unfilled order handling:** If no breakout occurs by `14:00:00` ET, cancel any unfilled entry orders for that symbol (do not leave them working into the late session).
*   **First eligible bar:** Because OR excludes the `10:00` bar, the first bar eligible to trigger an entry is the `10:00` bar.

## 4. Risk & Sizing

### A. Stop Loss Distance
Dynamic hybrid stop to prevent tight stops on compression days:
```python
stop_dist = max(
    1.0 * OR_Width,       # Structural stop: at least the opening range width
    0.20 * ATR_d          # Volatility floor: 20% of ATR_d (RTH-only)
)
```

### B. Position Sizing
*   **Risk Target:** 20 bps (0.20%) of Sleeve NAV per trade.
*   **Scope:** Risk target is **per symbol** (MES and MNQ are sized independently). Worst-case daily risk is ~2× if both symbols trigger on the same day.
*   **NAV update frequency (pre-registered):** Use **start-of-day NAV** (prior RTH close) for sizing. Do not rescale intra-day after fills.
*   **Formula:**
    ```python
    risk_dollars = sleeve_nav * 0.0020
    contract_risk_dollars = stop_dist * point_value  # MES: $5/pt, MNQ: $2/pt
    qty = floor(risk_dollars / contract_risk_dollars)

    # CRITICAL: Do NOT force min(1) - that violates risk cap!
    if qty < 1:
        # Stop is too wide for our risk budget
        # Option A: Skip this trade (preserve risk discipline)
        # Option B: Use fractional sizing via micro-only (we're already on micros)
        skip_trade = True  # Preferred: no trade if can't size properly
    ```

⚠️ **Risk cap discipline**: Using `max(1, qty)` would violate the 20 bps rule on wide-stop days. If stop_dist × point_value > risk_dollars, we **skip the trade** rather than over-risk.

**Example (MES, wide stop day):**
- Sleeve NAV: $50,000 → risk_dollars = $100
- stop_dist: 25 pts (wide OR + ATR floor)
- contract_risk = 25 × $5 = $125 > $100
- qty = floor(100/125) = 0 → **SKIP TRADE** (not force to 1)

## 5. Execution & Management

### A. Exits
1.  **Initial Stop:** Entry fill price ± `stop_dist`.
2.  **Profit Target:** Entry fill price ± (`2.0 * stop_dist`).
    *   *Note:* 2.0R allows capturing trend days better than 1.5R.
3.  **Time Stop (EOD):** Hard flatten at `15:55:00` ET.

### B. Backtest Resolution (Pessimistic)

**Same-bar resolution:**
If `Low <= Stop` AND `High >= Target` in the *same 1-minute bar*:
*   **Assumption:** STOP was hit first.
*   **Result:** Max Loss.
*   *Rationale:* Intraday data lacks tick precision; assuming profit leads to overestimation.

**Gap fills (price opens beyond stop/target):**
```python
# If bar opens beyond our stop level, fill at OPEN (not stop price)
# This models gap-through scenarios realistically

def simulate_fill(entry_price, stop_price, target_price, bar, direction):
    if direction == "LONG":
        # Check stop first (pessimistic)
        if bar.open <= stop_price:
            # Gapped through stop - fill at open (worse than stop)
            return ("STOP", bar.open)
        elif bar.low <= stop_price:
            return ("STOP", stop_price)
        # Then check target
        elif bar.open >= target_price:
            # Gapped through target - fill at target (no price improvement)
            return ("TARGET", target_price)
        elif bar.high >= target_price:
            return ("TARGET", target_price)
    # ... symmetric for SHORT
```

⚠️ **Gap-through stops are filled at bar.open, not stop price.** This is more conservative and realistic.

**Fill price definition (pre-registered):**
- The entry trigger level is `OR_High + Buffer` (long) / `OR_Low - Buffer` (short).
- The backtest entry fill price applies the execution penalty (see Cost Model) and then stop/target levels are computed from the resulting **fill price** (not from the trigger level).

### C. Cost Model (Per Contract)
*   **Commission:** $1.24 round-trip (IBKR Tiered).
*   **Execution penalty (baseline):** 1 tick entry + 1 tick exit (MES: $2.50 RT, MNQ: $1.00 RT).
*   **Stress test:** Re-run kill-tests at 2 ticks/side.
*   **Total penalty (baseline):** MES: $3.74 RT/contract, MNQ: $2.24 RT/contract deducted from PnL.

**Backtest application (pre-registered):**
- Apply the execution penalty as **adverse price movement** on fills:
  - Entry fill: worse by `+1 tick` (long) / `-1 tick` (short) vs the trigger level.
  - Exit fill: worse by `-1 tick` (long exit) / `+1 tick` (short exit) vs the stop/target level.
- Deduct commission as a fixed cash cost ($1.24 RT/contract) per completed round-trip.
- Do not double-count by also subtracting a separate “slippage $” after applying adverse fills.

## 6. Kill Criteria (Validation Gates)

The strategy must pass these gates on **Out-of-Sample** data (Walk-Forward):
*   **Sharpe definition:** Daily return series on sleeve NAV (include 0-return on no-trade days), annualized with √252, net of all costs.

| Metric | Threshold | Rationale |
| :--- | :--- | :--- |
| **Net PnL (OOS)** | > 0 | Must not lose money after costs. |
| **Annualized Sharpe (daily returns)** | > 0.5 | Must justify risk capital. |
| **Win Rate** | > 35% | With 2:1 R:R, breakeven is 33%. |
| **Avg Trade** | > $10 | Net of all costs (per micro contract). |
| **Max Drawdown** | ≥ -15% | Tighter than EOD sleeves (intraday leverage). |
| **SPY Correlation** | < 0.7 | Must provide diversification vs Sleeve DM. |
| **Walk-Forward Pass Rate** | ≥ 4/6 folds | Must generalize across regimes. |
| **Multi-Contract Stability** | Pass | Profitable on *both* MES and MNQ individually. |

### A. Decision Hierarchy (pre-registered)
1. **Immediate kill** if any immediate KILL condition triggers in the OOS results.
2. If **Net PnL ≤ 0** (OOS aggregate) → **KILL**.
3. If **Annualized Sharpe ≤ 0** (OOS aggregate) → **KILL**.
4. If **0 < Annualized Sharpe < 0.5** → investigate a simpler variant (e.g., “No Target” / EOD-only exit) and/or re-run under higher cost assumptions (2 ticks/side); do not promote.
5. If **Annualized Sharpe ≥ 0.5** AND all other gates pass → **PROMOTE**.

### B. Walk-Forward “Pass” Definition
A fold is counted as a **pass** if, for that fold’s OOS test window:
- Combined sleeve (MES+MNQ) **Net PnL > 0** (after costs), AND
- Combined sleeve **Max Drawdown ≥ -20%**, AND
- Combined sleeve has at least **20 trades** total (to avoid declaring a “pass” on tiny sample size).

### C. Multi-Contract Stability Definition
To avoid “works only on one ticker” outcomes:
- Over the full OOS aggregate (all folds combined), **MES Net PnL > 0** AND **MNQ Net PnL > 0** (after costs).

### D. SPY Correlation Definition
`SPY Correlation` is computed as:
- Correlation of **daily sleeve returns** vs **daily SPY returns**, on **all RTH trading days** (including 0-return no-trade days),
- Using close-to-close returns aligned to the RTH close (16:00 ET).
Also report (not gate on) correlation on **trade days only** for additional context.

### E. Drawdown Definition
Drawdown is computed on the daily sleeve equity curve (NAV) using RTH close marks:
- `DD_t = (NAV_t - peak(NAV_0..NAV_t)) / peak(NAV_0..NAV_t)`
- Max drawdown is `min(DD_t)` over the window (negative number).
Intraday (bar-by-bar) drawdown is not used for the kill gates.

### F. “Avg Trade” Definition
`Avg Trade` is measured **per micro contract round-trip**:
- `AvgTrade = total_net_pnl_dollars / total_contract_round_trips`
Where `total_contract_round_trips` is the sum of filled entry contracts (each implies one round-trip when exited).

**Immediate KILL conditions:**
- Net PnL < 0 across both contracts (OOS aggregate)
- Win rate < 25% (worse than random with 2:1 R:R)
- Max DD < -20% in any fold
- Costs exceed gross edge (repeat of Sleeve IM failure mode)

## 7. Implementation Roadmap

1.  **Data:** Ensure 1-min OHLC for `MES` and `MNQ` (2022-2025) is in `data/orb/`.
2.  **Backtester:** Implement `src/dsp/backtest/orb_futures.py`.
3.  **Validation:** Run 6-fold walk-forward test (6-month train, 3-month test, no overlap).
    *   **Note:** ORB is rule-based with frozen parameters; the "train" window is used for indicator warm-up (ATR/OR-width stats), not model fitting.

### Walk-Forward Folds (Exact Dates, Non-Overlapping)

| Fold | Train Start | Train End | Test Start | Test End | Notes |
|------|-------------|-----------|------------|----------|-------|
| 1 | 2022-01-03 | 2022-06-30 | 2022-07-01 | 2022-09-30 | Initial |
| 2 | 2022-07-01 | 2022-12-30 | 2023-01-03 | 2023-03-31 | Roll Q3→Q4 |
| 3 | 2023-01-03 | 2023-06-30 | 2023-07-03 | 2023-09-29 | Mid-period |
| 4 | 2023-07-03 | 2023-12-29 | 2024-01-02 | 2024-03-28 | 2023 H2 |
| 5 | 2024-01-02 | 2024-06-28 | 2024-07-01 | 2024-09-30 | 2024 H1 |
| 6 | 2024-07-01 | 2024-12-31 | 2025-01-02 | 2025-03-31 | Final (holdout) |

⚠️ **No overlap between train/test.** All percentiles (OR_width_20d, ATR_d) computed on **prior data only** (no lookahead).

```python
# Feature calculation must use ONLY data available at decision time
# Example: Average_OR_Width_20d on day T uses days [T-20, T-1], NOT day T
or_width_avg = df.loc[:current_date - 1, 'or_width'].tail(20).mean()
```

**Indicator warm-up handling (pre-registered):**
- Do not shorten lookbacks (ATR=14, OR-width stats=20).
- If a required lookback is unavailable on a given day/symbol, **skip trading** for that day/symbol.
- In walk-forward runs, the train window provides sufficient history so the first test day should have indicators defined.
- In any “single run” backtest starting at the earliest available date, the first ~20 trading days will naturally be skipped until indicators are defined.

**Walk-forward results output (pre-registered):**
- Use the same top-level JSON shape as Sleeve IM (`folds` list + `summary`) for consistency.
- Reuse common keys where applicable (`fold_id`, `train_period`, `test_period`, `n_trades`, `sharpe_ratio`, `win_rate`, `passes_kill_tests`) and add ORB-specific fields (e.g., `net_pnl_dollars`, `net_return`, `max_dd`, `avg_trade_per_contract`, `trade_days`, `cost_assumption_ticks_per_side`).
- Reference schema example: `dsp100k/data/sleeve_im/walk_forward_results.json`.

4.  **Decision:**
    *   If **all kill criteria** pass: Promote to `src/dsp/sleeves/sleeve_orb.py`.
    *   If **Net PnL ≤ 0** (OOS aggregate) or **Annualized Sharpe ≤ 0**: Kill immediately.
    *   If **0 < Annualized Sharpe < 0.5**: Analyze "No Target" (EOD exit) variant and/or higher cost assumptions (2 ticks/side), then re-run kill-test.

---

## 8. Production Execution (If Passes Kill-Test)

### A. IBKR Order Types
```python
# Entry: OCO stop-market orders (placed at 10:00 ET)
# Note: Use unique OCA groups per symbol/day.
buffer = 2 * tick_size

long_entry = Order(
    action="BUY",
    orderType="STP",  # Stop-market
    auxPrice=or_high + buffer,
    totalQuantity=qty,
    tif="DAY",
    ocaGroup=f"ORB_ENTRY_{date}_{symbol}",
    ocaType=1,
)

short_entry = Order(
    action="SELL",
    orderType="STP",  # Stop-market
    auxPrice=or_low - buffer,
    totalQuantity=qty,
    tif="DAY",
    ocaGroup=f"ORB_ENTRY_{date}_{symbol}",
    ocaType=1,
)

# Exit: OCA profit-target limit + stop-loss stop (submit after entry fill).
exit_action = "SELL" if entry_side == "LONG" else "BUY"

take_profit = LimitOrder(
    action=exit_action,
    totalQuantity=qty,
    lmtPrice=target_price,
    tif="DAY",
    ocaGroup=f"ORB_EXIT_{date}_{symbol}",
    ocaType=1,
)

stop_loss = Order(
    action=exit_action,
    orderType="STP",  # Stop-market
    auxPrice=stop_price,
    totalQuantity=qty,
    tif="DAY",
    ocaGroup=f"ORB_EXIT_{date}_{symbol}",
    ocaType=1,
)

# Submit exits immediately after entry fill (or as attached children via parentId/transmit).

# Partial fills (policy):
# - Do not use AON (all-or-none) on micros.
# - If entry partially fills, immediately:
#   1) cancel the remaining unfilled entry quantity for that side, and
#   2) place exits sized to the filled quantity only (stop-loss + take-profit OCA group).
# - If additional fills occur before cancellation is confirmed, adjust exit quantities to match net position.
```

### B. EOD Flatten (15:55 ET)
```python
# Cancel any open orders
for o in open_orders:
    ib.cancelOrder(o)

# Flatten position at market
if position != 0:
    flatten_order = MarketOrder(
        action="SELL" if position > 0 else "BUY",
        totalQuantity=abs(position)
    )
    ib.placeOrder(contract, flatten_order)
```

### C. Monitoring Metrics
| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| Slippage vs expected | > 2 ticks | Log warning, investigate |
| Fill rate | < 90% | Check order types |
| Daily P&L | < -2% sleeve NAV | Manual review |
| Win rate (rolling 20) | < 25% | Consider pause |

---

## 9. Capital Requirements

| Scenario | Sleeve NAV | Contracts/Trade | Margin Required |
|----------|------------|-----------------|-----------------|
| **Minimum** | $25,000 | 1-2 | ~$4,000 |
| **Target** | $50,000 | 2-4 | ~$8,000 |
| **Full** | $100,000 | 4-8 | ~$15,000 |

**Margin buffer**: Keep 2× overnight margin as buffer for intraday moves.

---

## 10. File Structure

```
dsp100k/
├── src/dsp/backtest/
│   └── orb_futures.py              # Kill-test backtester
├── src/dsp/sleeves/
│   └── sleeve_orb.py               # Production sleeve (if passes)
├── config/
│   └── sleeve_orb.yaml             # Parameters (frozen after validation)
├── data/orb/
│   ├── MES_1min_2022_2025.parquet  # Historical data
│   ├── MNQ_1min_2022_2025.parquet
│   ├── skip_dates.csv              # FOMC/CPI/NFP calendar
│   └── walk_forward_results.json   # Validation results
└── docs/
    └── SLEEVE_ORB_MINIMAL_SPEC.md  # This document
```

---

## 11. Risk Acknowledgments

**Known risks specific to ORB:**
1. **False breakouts**: Range breaks, reverses → stopped out
2. **Gap days**: OR may capture entire day's range → no edge left
3. **Low vol regimes**: Compression filter may skip many days
4. **Correlation to equities**: MES/MNQ track SPY → limited diversification

**Mitigations:**
- Buffer reduces false breakout entries
- ATR floor prevents tight stops
- 2:1 R:R allows profit on trend days
- Diversification comes from uncorrelated return stream (intraday vs monthly)

---

## Appendix A: Parameter Sensitivity (To Test in Backtest)

| Parameter | Default | Range to Test | Notes |
|-----------|---------|---------------|-------|
| OR Length | 30 min | [30, 60] | 60 min = fewer trades, wider range |
| Buffer | 2 ticks | [1, 2, 3] | Tighter = more fills, more noise |
| Stop mult | 1.0×OR | [0.75, 1.0, 1.5] | Tighter = more stops, better R:R |
| Target mult | 2.0×stop | [1.5, 2.0, 3.0] | Higher = fewer wins, bigger wins |
| ATR floor | 0.20 | [0.15, 0.20, 0.25] | Higher = wider minimum stop |

⚠️ **Discipline**: Pick ONE set of parameters before backtest. No optimization on test data.
