# SPEC_SLEEVE_C_TAIL_HEDGE.md — Sleeve C (Tail Hedge) v1.0

Status: **Paper trading (manual execution)** — automation blocked until DSP options execution is implemented.

## Purpose
Sleeve C is portfolio insurance: it is designed to lose a small, controlled premium in normal markets and pay out convexly in sharp equity drawdowns.

This sleeve is intentionally *not* expected to have high Sharpe in normal times. Its job is drawdown reduction and “survivability”.

## Instrument / Universe
- **Underlying**: `SPY`
- **Instrument**: **SPY put debit spreads** (vertical spreads)

## Structure (Pre-Registered)
- Buy **25-delta put**, sell **10-delta put**
- Target DTE: **30–45 days**
- Roll trigger: **10 DTE**
- Annual premium budget: **1.25% of portfolio NAV**
- Max concurrent spreads: **5**

These parameters are configured in `dsp100k/config/dsp100k.yaml` under `sleeve_c`.

## Sizing (Budget-Based)
- Budget is capped by `annual_budget_pct` (default 1.25%/year).
- Each new trade is limited to **≤ ~2 months of premium** (operational guardrail; prevents “spending the whole year in one trade”).
- Quantity is integer spreads only.

## Execution Model (Current Constraint)
The DSP executor currently supports only `STK`/`ETF` orders; **options execution is not implemented**. As a result:
- Sleeve C is **manual execution only** (in TWS) for now.
- We still produce a deterministic daily recommendation using IBKR option greeks/quotes.

## Monitoring
- Daily monitor script (source of truth): `dsp100k/scripts/sleeve_c_daily_monitor.py`
- Log output: `dsp100k/data/sleeve_c/paper_trading/sleeve_c_log.csv`
- The central digest includes Sleeve C status: `dsp100k/scripts/daily_digest.py`

## Risks / Non-Goals
- This is **not** a “VRP alpha” strategy; it is **tail convexity**.
- Put spreads can be expensive in high vol regimes; the budget cap is the main control.
- The hedge can underperform in slow drawdowns or choppy markets.

