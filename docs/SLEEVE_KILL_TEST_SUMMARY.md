# DSP-100K Sleeve Kill-Test Summary

**Date**: 2026-01-07
**Author**: Claude (consolidated from session logs)
**Status**: Sleeve DM promoted to paper trading; all others KILLED

---

## Executive Summary

| Sleeve | Strategy | Test Period | Sharpe | CAGR | Max DD | Cost Model | Verdict |
|--------|----------|-------------|--------|------|--------|------------|---------|
| **Sleeve A** | S&P 100 momentum L/S | 2018-2024 | -0.01 | n/a | n/a | 5 bps + $0.005/sh | **KILLED** (survivorship bias) |
| **Sleeve B (L/S)** | Sector ETF L/S (3×3) | 2022-2024 | -0.03 | -0.5% | -12% | 5 bps + $0.005/sh | **KILLED** (no edge) |
| **Sleeve B (Long)** | Sector ETF Long-only | 2022-2024 | 0.90 | 8.2% | -18% | 5 bps + $0.005/sh | **KILLED** (SPY β=0.86) |
| **Sleeve DM** | ETF Dual Momentum | 2022-2024 | 0.87 | 7.1% | -14% | 5 bps + $0.005/sh | **LIVE** ✅ |
| **Sleeve DM** | ETF Dual Momentum | 2018-2024 | 0.55 | 4.8% | -22% | 5 bps + $0.005/sh | **LIVE** ✅ |
| **Sleeve IM** | Intraday ML (LogReg) | Q3'23-Q4'24 | -2.25 | -48.6% | n/a | 10 bps/side | **KILLED** (0/6 folds) |
| **DQN (Gate 2.6)** | Deep RL intraday | 2024 val | see below | -79.7% | n/a | 10 bps/side | **KILLED** (no signal) |
| **Sleeve ORB** | Futures opening range breakout | 2022-2025 | 0.23 | 0.4%* | -1.7% | 1 tick + $1.24/RT | **KILLED** (low Sharpe, 2/6 folds) |

*Annualized from $1,403 total profit over 3.25 years

---

## Methodology Notes

### Sharpe Ratio Calculation
- **Annualization**: Daily returns × √252
- **Sleeve DM/B**: Monthly rebalance backtests with daily mark-to-market
- **Sleeve IM**: Per-trade Sharpe over 3-month test windows (not annualized in the usual sense—see fold details)
- **DQN**: Episode-level Sharpe (not directly comparable to monthly strategies)

### Cost Models
| Strategy | Per-Trade Cost | Implied Annual Drag |
|----------|----------------|---------------------|
| Sleeve DM | 5 bps slippage + $0.005/share commission | ~30-50 bps (12 rebalances × ~300% turnover/yr) |
| Sleeve B | 5 bps slippage + $0.005/share commission | ~30-50 bps |
| Sleeve IM | 10 bps/side (20 bps round-trip) | ~4000 bps (200+ trades/yr × 20 bps) |
| DQN | 10 bps/side on position changes | ~2000-5000 bps (high churn) |

### Currency
All figures in **USD**. Paper trading account is EUR-denominated (€1,050,917 NLV); USD positions are unhedged FX exposure.

---

## Sleeve A: S&P 100 Momentum L/S — KILLED

**Strategy**: Long top-decile momentum stocks from S&P 100, short bottom-decile
**Universe**: ~100 large-cap US stocks (OEX constituents)
**Signal**: 12-1 momentum (12-month return, skip most recent month)
**Rebalance**: Monthly

### Why Killed
**Survivorship bias**: Backtest used *current* S&P 100 constituents applied retroactively. Stocks that were removed (delisted, bankrupt, acquired) are missing from the short bucket, artificially inflating historical returns.

| Metric | Value | Notes |
|--------|-------|-------|
| Sharpe (2018-2024) | -0.01 | Effectively random after costs |
| Data issue | Critical | Delisted tickers not in backtest universe |

**Recommendation**: Would require point-in-time constituent data to fix. Not worth pursuing.

---

## Sleeve B: Sector ETF Momentum — KILLED

**Strategy**: Monthly sector rotation using SPDR sector ETFs
**Universe**: XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY (11 sectors)
**Signal**: 12-1 momentum
**Configurations Tested**:
1. **L/S**: Long top 3, short bottom 3
2. **Long-only**: Long top 3 only

### Results (2022-2024, 5 bps + $0.005/sh)

| Config | Sharpe | CAGR | Max DD | SPY Corr | Verdict |
|--------|--------|------|--------|----------|---------|
| L/S (3×3) | -0.03 | -0.5% | -12% | 0.41 | ❌ Can't beat cash |
| Long-only | 0.90 | 8.2% | -18% | **0.86** | ❌ Just expensive SPY beta |

### Why Killed
- **L/S**: Negative Sharpe—sector spreads don't have momentum alpha
- **Long-only**: High SPY correlation (0.86) means no diversification benefit. Just pay ETF fees to get SPY exposure with sector timing noise.

---

## Sleeve DM: ETF Dual Momentum — LIVE ✅

**Strategy**: Gary Antonacci-style dual momentum across asset classes
**Universe**: SPY, EFA, EEM, IEF, TLT, TIP, GLD, PDBC, UUP (risky) + SHY (cash)
**Signal**: 12-1 momentum; hold top 3 with momentum > 0, else 100% cash
**Vol Target**: 8% annualized (realizes ~6% due to conservative estimator)
**Rebalance**: Monthly (first trading day)

### Results

| Period | Sharpe | CAGR | Max DD | SPY Corr | Verdict |
|--------|--------|------|--------|----------|---------|
| 2022-2024 | 0.87 | 7.1% | -14% | 0.52 | ✅ PASS |
| 2018-2024 | 0.55 | 4.8% | -22% | 0.48 | ✅ PASS |

### Cost Assumptions
- **Slippage**: 5 bps per trade
- **Commission**: $0.005/share
- **Turnover**: ~300% annually (full portfolio rotation every 4 months on average)
- **Implied cost drag**: ~45 bps/year (not "~5 bps/year" as previously stated)

### Why It Works
1. **Asset-class momentum** is more robust than single-stock momentum
2. **ETFs don't delist** → no survivorship bias
3. **Monthly rebalance** → low turnover, low costs
4. **Cash option (SHY)** → avoids holding declining assets
5. **40+ years of academic backing** (Antonacci, Faber, AQR)

### Paper Trading Status (2026-01-05)
Positions established via `--force` flag (missed Jan 2 rebalance):

| Symbol | Shares | Avg Cost (USD) | Value (USD) | Weight |
|--------|--------|----------------|-------------|--------|
| EFA | 127 | $98.14 | $12,464 | 15.2% |
| EEM | 220 | $56.85 | $12,506 | 15.3% |
| GLD | 30 | $408.48 | $12,255 | 14.9% |
| SHY | 540 | $82.91 | $44,769 | 54.6% |
| **Total** | | | **$81,993** | 100% |

**Scale factor**: 0.087 (risk_mgr=0.92 × manual=0.095)
**Effective allocation**: ~7.8% of €1.05M NAV
**FX exposure**: USD positions, unhedged to EUR base

---

## Sleeve IM: Intraday ML — KILLED

**Strategy**: Supervised learning (logistic regression) on premarket + first-hour features
**Universe**: AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA
**Entry**: 10:31 ET (after feature cutoff)
**Exit**: 14:00 ET (mandatory flatten)
**Signal**: Binary classifier predicting positive/negative return

### Walk-Forward Results (6 folds, Q3'23–Q4'24)

| Fold | Test Period | Sharpe | Net Return | Accuracy | Trades |
|------|-------------|--------|------------|----------|--------|
| 1 | Q3 2023 | -2.64 | -59.8% | 56.0% | 352 |
| 2 | Q4 2023 | -1.94 | -38.1% | 50.8% | 290 |
| 3 | Q1 2024 | -1.75 | -32.8% | 48.1% | 253 |
| 4 | Q2 2024 | -3.14 | -65.0% | 49.9% | 314 |
| 5 | Q3 2024 | -1.77 | -47.4% | 54.8% | 291 |
| 6 | Q4 2024 | -2.28 | -48.7% | 54.1% | 318 |
| **Mean** | | **-2.25** | **-48.6%** | **52.3%** | 303 |

### Cost Model
- **Per-trade cost**: 10 bps/side (20 bps round-trip)
- **Rationale**: Includes spread (3-5 bps), market impact (2-3 bps), slippage (2-3 bps)

### Why Killed
1. **Accuracy ~52%** is barely above coin-flip
2. **Gross edge ~4.8 bps/trade** < 20 bps round-trip cost
3. **0/6 folds pass** kill test (need Sharpe ≥ 0)
4. **No regime** where signal works—consistent failure across all market conditions

---

## DQN (Gate 2.6): Deep RL Intraday — KILLED

**Strategy**: Double DQN with prioritized replay learning minute-by-minute positions
**Universe**: Same 9 symbols as Sleeve IM
**Trading window**: 10:31–14:00 ET (209 steps/episode)
**Training**: 500 episodes, 20-episode eval every 100 episodes

### Cost Model
- **Turnover cost**: 10 bps (one-way) on position changes
- **Frequency**: 1-minute decisions → up to 209 trades/day

### Results (Validation Set, 2024)

| Config | Metric | Value | Notes |
|--------|--------|-------|-------|
| No threshold | Mean P&L/episode | **-$0.797** | 1,079 trades/episode |
| No threshold | Trades/episode | 1,079 | Excessive churn |
| τ=0.06 | Mean P&L/episode | -$0.124 | 148 trades |
| τ=0.08 | Mean P&L/episode | -$0.035 | 43 trades |
| τ=0.10 | Mean P&L/episode | -$0.006 | 6.5 trades |
| τ=0.12 | Mean P&L/episode | -$0.001 | 0.6 trades (≈FLAT) |

**Note on "Sharpe -209 to -287"**: These are episode-level Sharpe ratios computed as `mean(episode_returns) / std(episode_returns)`, where each "episode" is a 209-step trading session. The extreme negative values reflect:
1. Highly negative mean returns (costs dominate)
2. Low variance in returns (consistently losing)

This is **not** the same as annualized Sharpe on daily returns. A more interpretable metric:
- **CAGR equivalent**: ~-80% annualized (if extrapolated, though nonsensical for intraday)
- **Break-even trades**: Would need ~2 bps cost (vs 10 bps actual) to approach Sharpe=0

### Why Killed
1. **No signal exists**: Conviction threshold sweep shows higher conviction ≠ better returns
2. **All baselines fail**: Even momentum, mean-reversion, RSI lose money on this data
3. **Cost structure impossible**: 10 bps × 1000+ trades/day = certain loss
4. **Model learned noise**: Filtering out low-conviction trades approaches FLAT, not profit

---

## Sleeve ORB: Futures Opening Range Breakout — KILLED

**Strategy**: Opening range breakout on MES/MNQ micro E-mini futures
**Universe**: MES (Micro S&P 500), MNQ (Micro Nasdaq 100)
**Signal**: Breakout above/below first 30 minutes (09:30-09:59 ET) with 2-tick buffer
**Stop**: max(1.0×OR_Width, 0.20×ATR_14d)
**Target**: 2R (2× stop distance)
**Risk**: 20 bps per trade
**Data**: Databento GLBX.MDP3 1-minute OHLCV (2022-2025, 320K bars each)

### Results (6-Fold Walk-Forward, 2022-2025)

| Period | Sharpe | CAGR | Trades | Win% | Verdict |
|--------|--------|------|--------|------|---------|
| Full OOS (2022-2025) | 0.23 | 0.4% | 285 | 44.4% | ❌ FAIL |

### Per-Fold Performance

| Fold | Test Period | Trades | Net P&L | Sharpe | Win% | Pass |
|------|-------------|--------|---------|--------|------|------|
| 1 | 2022-07-01 to 2022-09-30 | 56 | -$34 | -0.03 | 44.6% | ❌ |
| 2 | 2023-01-03 to 2023-03-31 | 64 | -$187 | -0.16 | 40.6% | ❌ |
| 3 | 2023-07-03 to 2023-09-29 | 65 | **+$1,672** | **1.74** | 49.2% | ✅ |
| 4 | 2024-01-02 to 2024-03-28 | 40 | -$612 | -0.79 | 37.5% | ❌ |
| 5 | 2024-07-01 to 2024-09-30 | 36 | **+$827** | **1.19** | 52.8% | ✅ |
| 6 | 2025-01-02 to 2025-03-31 | 24 | -$265 | -0.54 | 41.7% | ❌ |

**Folds Passing**: 2/6 (33.3%) — **FAILS threshold of ≥4/6**

### Per-Symbol Breakdown

| Symbol | Net P&L | Status |
|--------|---------|--------|
| MES | +$2,992 | ✅ Profitable |
| MNQ | -$1,590 | ❌ Unprofitable |

**Both symbols profitable?** NO — **FAILS kill-test criterion**

### Why Killed

1. **Low Sharpe Ratio** (0.23 < 0.5 threshold) — Not risk-adjusted profitable
2. **Fold Consistency** (2/6 < 4/6 threshold) — Only 33% of OOS folds passed
3. **MNQ Unprofitable** (-$1,590) — Strategy doesn't generalize across both instruments
4. **High Transaction Costs** — 46% of gross P&L consumed by costs ($530 commission + $655 slippage)
5. **Regime Dependency** — Only works in sustained trend regimes (Folds 3, 5), fails in choppy markets

### Cost Structure

| Component | Assumption | Total | % of Gross P&L |
|-----------|------------|-------|----------------|
| Commission | $1.24/RT | $530 | 20.5% |
| Slippage | 1 tick/side | $655 | 25.3% |
| **Total** | | **$1,185** | **45.8%** |

**Gross P&L**: $2,588 → **Net P&L**: $1,403 (after costs)

### Kill-Test Criteria Failures

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| Sharpe > 0.5 | >0.5 | **0.23** | ❌ FAIL (-54% shortfall) |
| ≥4/6 folds pass | ≥4/6 | **2/6** | ❌ FAIL (-50% shortfall) |
| Both symbols profitable | Both >$0 | MNQ -$1,590 | ❌ FAIL |
| Net P&L > $0 | >$0 | +$1,403 | ✅ PASS |
| Win rate > 35% | >35% | 44.4% | ✅ PASS |
| Max DD ≥ -15% | ≥-15% | -1.7% | ✅ PASS |

**Verdict**: 🔴 **DO NOT TRADE** — Fails 3 of 6 criteria

**Detailed Report**: [SLEEVE_ORB_KILL_TEST_RESULTS.md](./SLEEVE_ORB_KILL_TEST_RESULTS.md)

---

## Appendix: Kill-Test Criteria

For a strategy to pass kill tests:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Sharpe (primary window) | ≥ 0.3 | Must beat risk-free meaningfully |
| Sharpe (extended window) | ≥ 0.0 | Must not lose money over longer period |
| Max Drawdown | ≤ -30% | Must be tolerable for real capital |
| SPY Correlation | ≤ 0.7 | Must provide diversification |
| Walk-forward pass rate | ≥ 50% | Must generalize across regimes |
| Both symbols profitable | Both >$0 | (For multi-instrument strategies) |

**Only Sleeve DM passes all criteria.**

---

## Files Reference

| Sleeve | Backtest Script | Results |
|--------|-----------------|---------|
| Sleeve DM | `dsp100k/src/dsp/backtest/etf_dual_momentum.py` | (inline output) |
| Sleeve B | `dsp100k/src/dsp/backtest/etf_sector_momentum.py` | (inline output) |
| Sleeve IM | `dsp100k/scripts/sleeve_im/walk_forward_validation.py` | `dsp100k/data/sleeve_im/walk_forward_results.json` |
| DQN | `dsp100k/scripts/dqn/train.py`, `evaluate.py` | `dsp100k/docs/GATE_2_REPORT.md` |
| **Sleeve ORB** | `dsp100k/src/dsp/backtest/orb_futures.py` | `dsp100k/data/orb/walk_forward_results.json` |

---

*Last updated: 2026-01-07*
