# PRESENTATION_ORB_MINIMAL.md — Sleeve ORB (Opening Range Breakout, Micro Futures)

## Executive Summary (non-technical)
Sleeve ORB is a **simple intraday trading sleeve** that trades **micro futures** on the US equity indices:
- **MES** = Micro S&P 500 futures
- **MNQ** = Micro Nasdaq-100 futures

The idea is straightforward:
1. The **first 30 minutes** after the stock market opens often defines an important **price range**.
2. If price **breaks out** above/below that range, we take a position in the direction of the move.
3. We keep risk tightly controlled and **never hold overnight**.

This sleeve is currently **research / kill-test only** (not promoted to production).

---

## 1) Universe (what we trade)
We trade **Micro Futures** (not stocks, not options):
- **Instrument type:** CME-listed, highly liquid, exchange-traded futures (via IBKR)
- **Contracts:** `MES` and `MNQ` (micros only)
- **Why micros:** They allow **small, granular sizing** so the strategy can respect risk limits at realistic sleeve NAV (minis are typically too large for our per-trade risk budget).

What we do *not* trade in this sleeve:
- Individual equities
- Options
- Overnight futures sessions (Globex/ETH)

---

## 2) What the sleeve evaluates (Opening Range of the underlying equity market)
Even though futures trade nearly 24 hours, this sleeve only uses the **regular US stock market open** as its reference point:
- **RTH session:** 09:30–16:00 ET
- **Opening Range:** 09:30–10:00 ET (first 30 minutes)

We compute:
- **Opening Range High** = highest price in that 30-minute window
- **Opening Range Low** = lowest price in that 30-minute window

Trade decision (plain English):
- If price breaks **above** the range → we consider a **long** trade.
- If price breaks **below** the range → we consider a **short** trade.

This is a **reactive** strategy (responds to an observed breakout), not a predictive/ML model.

---

## 3) Time horizon (how long we hold risk)
This is an **intraday** sleeve:
- **Typical holding time:** minutes to a few hours
- **Entry window:** after the opening range completes (from ~10:00 ET) and only until early afternoon
- **Hard rule:** **flat by 15:55 ET** (no overnight positions)

Bottom line: the sleeve is designed so that **no position is held longer than the same trading day** (far less than a week).

---

## 4) Risk governor (how we limit losses during fast markets / crashes)
ORB is designed with multiple, layered protections intended to prevent “runaway” losses:

### A) Small, predefined risk per trade
- Position size is set so that the **maximum planned loss** (if the stop is hit) is a small fraction of sleeve NAV (e.g., ~0.20% per symbol per trade).
- If the required stop would be too wide to respect the risk cap, the trade is **skipped** (we do not over-size).

### B) Hard stop-loss + mandatory end-of-day exit
- Every position has a **stop-loss** defined at entry (automatic exit if the move reverses).
- Regardless of profit/loss, all positions are **closed by 15:55 ET**.

### C) Low trade frequency (limits churn)
- Maximum **one trade per symbol per day** (no “revenge trading” loops).

### D) “Circuit breaker” style controls
The broader DSP-100K risk framework can enforce safety rules such as:
- **Stop trading for the day** after a defined loss threshold
- **Flatten immediately** if risk limits are breached
- **Skip/limit trading around major scheduled events** (e.g., FOMC/CPI/NFP) where volatility spikes are common

### E) Conservative evaluation before any deployment
Before production consideration, the sleeve must pass:
- Out-of-sample walk-forward testing
- Net-of-costs results (including conservative slippage assumptions)
- Drawdown and stability gates across both MES and MNQ

---

## What management should expect next
If approved to proceed with the kill-test:
- We will produce a short report answering:
  - “Does it make money after conservative costs?”
  - “Is the drawdown acceptable?”
  - “Is it stable across years and across both MES and MNQ?”
- If it fails any gate, we **kill it** and move on (no production build-out).

