# Sleeve TSMOM — Management Presentation (Plain English)
**Date:** 2026-01-07  
**Status:** Proposal for kill-test validation (not approved for trading)

---

## 1) What a “Future” Is (Simple Definition)
A **futures contract** is an exchange-traded agreement to buy or sell something (e.g., the S&P 500 index, oil, gold, EUR/USD) at a standardized size.

Important points:
- Futures trade on regulated exchanges (e.g., CME).
- They are **liquid** and typically cheap to trade versus many alternatives.
- You do **not** pay the full notional value upfront; you post margin, and profits/losses are settled daily.
- We use **micro futures**, which are smaller versions designed for flexible sizing (better for our risk limits).

In our system, a “futures position” is simply a way to gain (or reduce) exposure to a market with controlled risk and transparent pricing.

---

## 2) Strategy in One Sentence
**TSMOM (Time-Series Momentum)** tries to capture medium-term trends across many markets by going **long** markets that have been rising over the last ~12 months and **short** markets that have been falling, with strict risk limits and weekly rebalancing.

---

## 3) Why We’re Doing This (Lessons from ORB)
Our Opening Range Breakout (ORB) test showed a key problem: some intraday signals look OK in a narrow slice of time but fail to generalize and are sensitive to regime changes.

TSMOM is meant to address that:
- **Longer horizon** (weeks/months vs minutes) is typically more stable.
- **Breadth** (many markets) reduces the chance we are “lucky” in one instrument.
- **Lower turnover** (weekly rebalance) reduces cost drag.

This is a research direction, not a promise of performance.

---

## 4) What We Trade (Universe)
We trade a small, diversified set across economic “buckets”:

**Micro futures (Databento data):**
- Equity indices: `MES` (S&P 500), `MNQ` (Nasdaq-100), `M2K` (Russell 2000), `MYM` (Dow)
- Commodities: `MGC` (Gold), `MCL` (Crude Oil)
- FX: `M6E` (EUR/USD), `M6J` (JPY/USD)

**Bond exposure (ETFs, Polygon data):**
- `TLT` (long-term US Treasuries), `IEF` (intermediate US Treasuries)

Why include bonds: Treasuries often behave differently from equities/commodities in stress regimes, so they improve diversification.

---

## 5) How the Signal Works (No Math)
Once per week:
- For each market, look at whether its price over the last ~12 months is **up or down**.
- If up → we want to be **long**.
- If down → we want to be **short**.
- If we don’t have enough history → we stay **flat** (no position).

This is intentionally simple to avoid “overfitting” (designing rules that only work in the past).

---

## 6) Holding Period & Trading Frequency
- **Holding period:** typically **days to weeks** (not intraday).
- **Rebalance frequency:** **weekly** (usually Monday).
- We do not chase every small change; we only trade when the position meaningfully changes.

This keeps costs under control and reduces “noise trading.”

---

## 7) Risk Management (“Crash Governor”)
This sleeve is built with multiple layers of risk control:

1) **Portfolio volatility target**
- We target a conservative annualized volatility level for the sleeve (8%).
- If markets become more volatile, positions automatically shrink.

2) **Diversification by design**
- Risk is split across four buckets: equities, commodities, FX, and rates.
- This reduces dependence on a single market regime.

3) **Hard exposure limits**
- Caps prevent excessive leverage or concentration in any one instrument or bucket.

4) **Stress-cost test**
- We re-run results with pessimistic execution costs (2× slippage) to ensure the strategy isn’t “paper-only.”

These controls are intended to prevent the sleeve from becoming a hidden “risk-on bet” in market stress.

---

## 8) What Success Looks Like (Promotion Gate)
We will not “believe” the idea until it passes the same discipline used in previous sleeves:
- Out-of-sample walk-forward testing (year-long windows, not short cherry-picked periods).
- Risk-adjusted performance above our threshold.
- Does not depend on one instrument “getting lucky.”
- Still holds up under worse execution assumptions.

If it fails: we kill it (we do not tweak parameters until it passes).

---

## 9) What Can Go Wrong (Plain Risks)
Even a well-known strategy can fail in practice:
- Trends can disappear or reverse quickly (whipsaw).
- Correlations can spike in crises (diversification can temporarily fail).
- Futures rolling (moving from one contract to the next) can introduce costs and complexity.
- Short samples (a few years) can be misleading versus decades of academic studies.

This is why we insist on pre-registered rules + walk-forward testing + stress assumptions.

---

## 10) What We Need From Management
1) Approval of the universe and the fact we’re using **micro futures + bond ETFs**.
2) Approval of the validation process (pre-registered rules, no tuning).
3) Budget decision for historical data (Databento) to cover the required instruments and window.

Reference implementation spec: `dsp100k/docs/SLEEVE_TSMOM_MINIMAL_SPEC.md`

