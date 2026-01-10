# PRESENTATION_BASELINE_B.md — Baseline “B” (Premarket + 1st Hour → 10:31→14:00)

## Executive Summary (non-technical)
Baseline “B” is a **simple, supervised machine‑learning** experiment designed to answer one question:

> **Do our current premarket + first-hour signals contain tradable edge after realistic trading costs?**

It intentionally avoids complex reinforcement learning. If Baseline B cannot show positive performance under conservative assumptions, it is unlikely that a more complex model will reliably “create” profitability from the same inputs.

## What the system does
Every trading day:
1. **Before 10:31 ET**, we summarize what happened:
   - **Premarket** price moves (04:00–09:15 ET)
   - **First hour after open** behavior (09:30–10:30 ET)
2. At **10:31 ET**, the model predicts each symbol’s expected move from **10:31 → 14:00**.
3. We build a portfolio (up to 9 positions) using those predictions.
4. We **close all positions at 14:00 ET** (no overnight risk).

## Universe (what we trade)
We restrict to 9 highly liquid symbols:

`AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA`

Why these:
- High liquidity → more realistic execution
- Good data coverage
- Mix of single stocks + broad ETFs (SPY/QQQ) for market context

## What data we use (plain English)
### Premarket (04:00–09:15 ET)
We convert premarket behavior into **three simple returns**:
- 04:00–06:00
- 06:00–08:00
- 08:00–09:15

We also track how much premarket data exists (bar counts) so we can distinguish:
- “no change” vs “no prints”

### Regular Trading Hours (09:30–10:30 ET)
We compute a standard set of market descriptors from 1‑minute bars, such as:
- short-term returns and volatility
- volume activity
- basic technical/context indicators
- market context using SPY/QQQ and “relative” performance

## What the model predicts
For each (day, symbol), the model predicts:

> **Expected return from 10:31 to 14:00**

This produces one forecast per symbol per day. The first version uses **Ridge regression** (a stable, conservative model).

## How we convert predictions into trades
### Trading schedule
- **Entry:** 10:31 (use the 10:31 bar open)
- **Exit:** 14:00 (use the 14:00 bar close)
- One entry + one exit per symbol per day (no re-entry in v0)

### Portfolio sizing (simple and consistent)
- Target **gross exposure = 10% of NAV** (capital at risk intraday)
- Up to **9 total open positions** (max one position per symbol)
- **Not required to be dollar-neutral** (net long/short may vary)
- **Bigger prediction ⇒ bigger position** (sizes proportional to predicted edge)

### Trading costs (conservative)
We assume **10 bps one-way cost** per trade.

“bps” = **basis points**:
- 1 bps = 0.01%
- 10 bps = 0.10%

This “all‑in” cost is meant to cover:
- bid/ask spread
- slippage
- commissions/fees

We only trade when the predicted move is large enough to beat costs.

## How we measure success (what “good” looks like)
We care first about **profitability after costs**, not a single ratio like Sharpe.

Primary success metric:
- **Net CAGR** (annualized growth) computed from the daily net P&L series

Risk gate:
- **Max drawdown ≤ 15%**

Generalization requirement:
- Must pass on **Validation** *and* **Dev Test** (two separate out-of-sample periods) before we look at 2025 holdout.

## Why we are doing this now
Baseline B is a fast “truth test”:
- **If it passes:** we have evidence the features contain tradable signal. Then it makes sense to invest in stronger models (e.g., tree ensembles, neural nets) and paper-trading validation.
- **If it fails:** we avoid spending months on complex approaches and instead focus on improving features, changing the prediction target, or adjusting the trading design.

## Known limitations (important for interpretation)
- Shorting is modeled without borrow/locate constraints (acceptable for a first pass on liquid names).
- Premarket liquidity can be thin for some names on some days; we explicitly track data availability.
- This is a backtest‑style evaluation; real trading will require additional execution checks (fills, slippage, halts).

## Deliverables (what management will receive)
1. A baseline model package (trained Ridge + scaler + feature schema).
2. A performance report on:
   - VAL period
   - DEV_TEST period
   - (Holdout 2025 only if the gates pass)
3. A data-quality / missing-data report (how many days were skipped and why).

## Current Status (initial results)
The backtest now correctly includes the **mandatory 14:00 exit**, meaning each traded position is a full **round-trip** (enter + exit) each day.

Results with **10 bps one-way** all‑in costs:
- **VAL:** Net CAGR **-2.71%**, MaxDD **1.83%** (risk gate passes, profit gate fails)
- **DEV_TEST:** Net CAGR **-4.43%**, MaxDD **3.31%** (risk gate passes, profit gate fails)

Interpretation:
- The signal is **not strong enough to survive costs**, and it also **does not generalize** (DEV_TEST is negative even **before** costs on the trades we selected).
- Implied break-even cost on VAL is ~**2.5 bps one‑way**; above that, net performance is expected to be negative without stronger features/targets.

Cost intuition (important):
- “10 bps one-way” is charged on the **traded notional**, not on the entire portfolio.
- At full exposure (`G=10%`), a full daily round-trip costs about **2 bps of NAV per day** (≈**5.0% per year**).
- In this run the model did not always deploy the full 10% gross (edge threshold filters some days), so realized average turnover was ~**14–15%** and realized cost was ~**1.45 bps/day** (≈**3.6%/year**).

Two takeaways:
- “Cost drag 405% / 479%” means **costs are several times larger than the gross return** (because gross return is very small), not that we are paying 405% of NAV in fees.
- Because DEV_TEST gross is already negative, simply reducing costs is **not** sufficient; we need a different feature set, target, or trading design.

Next step: because Baseline B fails both VAL and DEV_TEST, we should **not** proceed to 2025 holdout for this design; we should revise features/targets/trading design before further investment.
