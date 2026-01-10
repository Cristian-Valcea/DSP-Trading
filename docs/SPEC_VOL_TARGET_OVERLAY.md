# SPEC_VOL_TARGET_OVERLAY.md — Portfolio Volatility Targeting Overlay

**Version**: 1.0  
**Date**: 2026-01-10  
**Status**: Pre-Registered (Frozen after approval)  
**Audience**: DSP-100K engineering + management reviewers  

## 1. Goal
Add a **portfolio-level risk multiplier** that scales exposure up/down based on realized volatility, improving drawdowns and stabilizing risk across all sleeves (DM, VRP-CS, VRP-ERP).

This is **not** a new alpha sleeve. It is a *risk layer* applied to sleeves’ target notionals/allocations.

## 2. High-Level Mechanism
Compute a daily scalar:

`portfolio_risk_multiplier = clip(target_vol / realized_vol, min_mult, max_mult)`

Apply it to sleeves that represent **directional risk**:
- DM (risk assets when not in SHY/cash)
- VRP-ERP (SPY exposure)

Do **not** apply it to:
- VRP-CS (VX calendar spread) by default (it is already a spread/relative-value trade with different risk geometry).  
  Optional: allow a small multiplier band later (v1.1) after observing paper behavior.

## 3. Inputs & Data
### 3.1 Proxy Instrument
Use **SPY** close-to-close returns as a conservative proxy for overall portfolio risk:
- Symbol: `SPY`
- Frequency: daily close
- Source: Polygon (already subscribed)

Rationale: DM and VRP-ERP are equity-linked; SPY is the simplest robust risk proxy. This avoids the fragility of multi-asset covariance estimation.

### 3.2 Volatility Estimator
- Lookback: **21 trading days**
- Estimator: standard deviation of daily log returns, annualized by √252

`realized_vol = stdev(log_returns_21d) * sqrt(252)`

## 4. Parameters (Pre-Registered)
| Parameter | Default | Notes |
|---|---:|---|
| `target_vol` | 0.10 | 10% annualized target portfolio vol |
| `min_mult` | 0.25 | Never scale below 25% (avoid “dead portfolio”) |
| `max_mult` | 1.50 | Never lever beyond 1.5× |
| `lookback_days` | 21 | 1 month realized vol |
| `rebalance_band` | 0.10 | Only rebalance if multiplier changes by ≥0.10 |
| `min_days_between_changes` | 2 | Prevent whipsaw |
| `state_path` | `data/vol_target_overlay_state.json` | Persist last multiplier and timestamp |

## 5. Rebalance Rule
Compute new multiplier daily after the market close (or before the execution window using last close).

Rebalance is triggered only if BOTH:
1. `abs(new_mult - last_mult) >= rebalance_band`
2. At least `min_days_between_changes` have passed since the last multiplier update

This prevents churn from small volatility fluctuations.

## 6. How the Multiplier is Applied
### 6.1 VRP-ERP (SPY)
Current VRP-ERP logic outputs a target SPY share count based on regime.

Overlay application:
`target_shares_overlay = floor(target_shares_base * portfolio_risk_multiplier)`

### 6.2 Sleeve DM (ETF Dual Momentum)
DM produces weights across ETFs (including SHY cash).

Overlay application:
- Multiply **risky weights** by multiplier and renormalize with cash buffer, or
- Apply multiplier to the sleeve NAV allocation (recommended implementation path):
  `dm_alloc_usd = dm_alloc_usd_base * portfolio_risk_multiplier`

DM already has internal risk-off behavior (SHY). Overlay should still reduce gross in high-vol regimes.

### 6.3 VRP-CS (VX calendar spread)
Default (v1.0): **no scaling**.
- The spread is not linear equity beta and can widen in crises for reasons unrelated to SPY vol.
- Scaling it by SPY vol risks removing the edge.

## 7. Safety Constraints
Regardless of multiplier:
- Respect existing RiskManager hard stops (drawdown/margin caps).
- If RiskManager is in **CRITICAL** state, force multiplier to `min_mult` (or 0.0 if your live policy is “halt”).

## 8. Output Contract
The overlay module must expose:
- `get_multiplier(as_of_date) -> float`
- `should_rebalance(as_of_date, new_mult) -> bool`
- persistent state read/write

## 9. Validation (Non-Optimization)
We do **not** tune parameters from backtests.

We validate only that:
- multiplier is stable (doesn’t change daily)
- it binds in high-vol periods (e.g., Mar 2020)
- it never exceeds bounds

## 10. Deliverables
Code:
- `dsp100k/src/dsp/risk/vol_target_overlay.py`

Integration notes:
- Apply multiplier inside portfolio orchestration (DM + VRP-ERP planning stage).

---

**Approval to Freeze (v1.0):** ____________________  **Date:** __________

