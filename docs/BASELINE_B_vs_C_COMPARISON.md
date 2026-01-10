# Baseline B vs Baseline C: Strategy Comparison Report

**Date:** January 4, 2026
**Purpose:** Document the trading design differences and results between two supervised learning approaches using premarket + RTH features

---

## Executive Summary

Both strategies attempt to extract tradable signal from **premarket activity (4:00–9:15 AM) combined with early RTH context (9:30–10:30 AM)**. Neither strategy produces positive returns—even before transaction costs.

| Metric | Baseline B | Baseline C |
|--------|------------|------------|
| **Verdict** | KILLED | KILLED |
| **Gross profitable?** | VAL: Yes (+0.44%) / DEV: No (-0.38%) | No (both splits negative) |
| **Net profitable?** | No | No |
| **Core failure** | Costs erode thin edge | No edge exists at any horizon |

---

## 1. Trading Design Differences

### Baseline B: Single Daily Trade

**Entry:** 10:31 AM
**Exit:** 2:00 PM (mandatory flatten)
**Holding period:** ~3.5 hours
**Re-entry:** Not allowed
**Overnight:** Never (always flat by close)

**Philosophy:** One clean shot per day. If premarket + first hour tells us something about the rest of the day, capture it in a single 10:31→14:00 trade.

### Baseline C: Multiple Intraday Decisions with Overnight Option

**Rebalance times:** 10:31, 11:31, 12:31, 2:00 PM
**Re-entry:** Allowed (can exit at 11:31, re-enter at 12:31)
**Overnight:** Allowed (2:00 PM position held until next day 10:31)
**Holding periods:** 1 hour, 1 hour, 1.5 hours, or overnight (~20 hours)

**Philosophy:** If the single 3.5-hour window is too noisy, perhaps:
- Shorter 1-hour intervals have cleaner signal
- Ability to exit winners early and re-enter improves timing
- Holding overnight captures momentum that persists

---

## 2. What Each Strategy Tests

| Question | Baseline B | Baseline C |
|----------|------------|------------|
| Is there a 3.5-hour directional signal? | Yes | N/A |
| Is there a 1-hour directional signal? | N/A | Yes |
| Does overnight holding help? | N/A | Yes |
| Does re-entry improve returns? | N/A | Yes |
| Can smarter exits salvage thin edge? | N/A | Yes |

---

## 3. Performance Results

### Validation Period (Jan–Jun 2024)

| Metric | Baseline B | Baseline C | Interpretation |
|--------|------------|------------|----------------|
| **Gross Return** | +0.44% | -0.58% | B had small positive edge; C lost money even gross |
| **Net Return** | -1.34% | -5.54% | Both unprofitable after costs |
| **Net CAGR** | -2.71% | -10.93% | C lost 4× more annually |
| **Max Drawdown** | 1.83% | 5.40% | C had 3× worse drawdown |
| **Gross Sharpe** | +0.53 | -0.92 | B positive but weak; C negative |
| **Net Sharpe** | -1.61 | -8.58 | Both deeply negative |
| **Turnover** | 14% | 40% | C traded 3× more |
| **Cost Drag** | 4.1× | 8.5× | Costs vs gross return multiple |
| **Win Rate** | 32% | 20% | C won fewer intervals |

### Development Test Period (Jul–Dec 2024)

| Metric | Baseline B | Baseline C | Interpretation |
|--------|------------|------------|----------------|
| **Gross Return** | -0.38% | -0.87% | Both lost money gross |
| **Net Return** | -2.22% | -6.42% | C lost 3× more |
| **Net CAGR** | -4.43% | -12.53% | C annualized loss 3× worse |
| **Max Drawdown** | 3.31% | 6.76% | C had 2× worse drawdown |
| **Gross Sharpe** | -0.40 | -0.84 | Both negative; C 2× worse |
| **Net Sharpe** | -2.32 | -6.11 | Both deeply negative |
| **Turnover** | 15% | 44% | C traded 3× more |
| **Cost Drag** | 4.8× | 6.4× | Both high; C higher |
| **Win Rate** | 33% | 22% | C won fewer intervals |

---

## 4. Model Predictive Power (R² Scores)

### Baseline B (Single Horizon: 10:31→14:00)

| Split | R² |
|-------|-----|
| Train | +0.015 |
| Val | -0.036 |

Interpretation: Model learned something on training data (R² = 1.5%), but it was noise—validation R² is negative.

### Baseline C (Four Horizons)

| Interval | Train R² | Val R² |
|----------|----------|--------|
| 10:31→11:31 | +0.031 | -0.043 |
| 11:31→12:31 | +0.034 | -0.072 |
| 12:31→14:00 | +0.055 | -0.085 |
| 14:00→next 10:31 | +0.019 | -0.093 |

Interpretation: All four models show the same pattern—modest in-sample fit that completely fails out-of-sample. The overnight interval (14:00→10:31) has the worst validation R² (-9.3%), suggesting overnight holding adds noise, not signal.

---

## 5. Why Baseline C Performed Worse

### 1. Higher Turnover = Higher Costs

- Baseline B: One entry + one exit per day = 14-15% daily turnover
- Baseline C: Up to 4 rebalances per day = 40-44% daily turnover

With 10 bps one-way costs, Baseline C's 3× higher turnover means 3× higher transaction costs.

### 2. More Decisions = More Chances to Be Wrong

Baseline B makes 1 prediction per symbol per day.
Baseline C makes 4 predictions per symbol per day.

Since the underlying signal doesn't exist (all validation R² are negative), more predictions means more wrong trades.

### 3. Overnight Holding Added Risk, Not Return

The 14:00→next 10:31 interval:
- Longest holding period (~20 hours including overnight)
- Highest exposure to overnight gaps, news, macro events
- **Worst validation R²** (-9.3%)

Holding overnight didn't capture momentum—it added uncompensated risk.

### 4. Shorter Horizons Were Noisier, Not Cleaner

The hope was that 1-hour intervals might have cleaner signal than 3.5 hours. The opposite was true:
- 10:31→11:31 (1 hour): Val R² = -4.3%
- 12:31→14:00 (1.5 hours): Val R² = -8.5%

Shorter windows had worse (more negative) R², not better.

---

## 6. What We Learned

### Signal Assessment

The premarket + first-hour RTH feature family was tested across:

| Dimension Tested | Result |
|------------------|--------|
| Single 3.5-hour holding (B) | No net edge |
| Multiple 1-hour holdings (C) | No edge at any horizon |
| Overnight holding (C) | Adds risk, not return |
| Re-entry flexibility (C) | More trading, more losses |
| Ridge regression pooled model | Overfits; negative OOS R² |

**Conclusion:** The features contain no tradable signal. This is not a "costs are too high" problem—**gross returns are negative** in most tests.

### Comparison of Failure Modes

| Baseline B | Baseline C |
|------------|------------|
| Small positive gross edge on VAL | Gross negative on both splits |
| Costs killed a thin edge | No edge to kill |
| Controlled turnover (14%) | Excessive turnover (40%+) |
| Conservative single-trade approach | Aggressive multi-trade approach |
| Lost less money | Lost more money |

---

## 7. Cost Sensitivity Analysis

Even if we reduced costs to zero:

| Split | Baseline B Gross | Baseline C Gross |
|-------|------------------|------------------|
| VAL | +0.44% | -0.58% |
| DEV_TEST | -0.38% | -0.87% |

- Baseline B would have made a tiny profit on VAL but lost on DEV_TEST
- Baseline C would have lost on both splits regardless of costs

**Costs are not the problem.** The underlying predictions are wrong.

---

## 8. Symbol-Level Breakdown

### Baseline B: Symbol P&L (DEV_TEST)

| Symbol | Net P&L | Notes |
|--------|---------|-------|
| NVDA | +0.25% | Best performer |
| AAPL | +0.07% | Slight positive |
| QQQ | +0.07% | Slight positive |
| TSLA | +0.02% | Near flat |
| SPY | -0.07% | Slight negative |
| AMZN | -0.12% | Negative |
| GOOGL | -0.16% | Negative |
| META | -0.22% | Negative |
| MSFT | -0.22% | Worst performer |

### Baseline C: Symbol P&L (DEV_TEST)

| Symbol | Net P&L | Notes |
|--------|---------|-------|
| TSLA | +0.61% | Best performer (flipped from B) |
| GOOGL | +0.12% | Positive (was negative in B) |
| MSFT | +0.03% | Near flat (was worst in B) |
| AMZN | +0.03% | Near flat |
| QQQ | -0.01% | Near flat |
| SPY | -0.03% | Near flat |
| AAPL | -0.06% | Slight negative |
| META | -0.22% | Negative |
| NVDA | -1.34% | Worst performer (was best in B) |

**Observation:** Symbol rankings flip between B and C, suggesting the "signal" is noise—there's no consistent pattern that persists across different trading designs.

---

## 9. Recommendations

### For This Feature Family

**Stop.** Further optimization (alpha tuning, different model types, additional horizons) is unlikely to help. The validation R² values are all negative—the features do not predict returns.

### For Future Strategy Development

1. **Test gross profitability first.** If gross returns are negative, no amount of cost reduction helps.

2. **Prefer simpler designs initially.** Baseline B (one trade per day) lost less than Baseline C (four trades per day). Complexity did not help.

3. **Watch for overfitting.** Both strategies showed positive training R² but negative validation R². In-sample performance was meaningless.

4. **Turnover is costly.** At 10 bps one-way cost, 40% daily turnover means 8% annual cost drag. This is fatal for strategies with thin edges.

5. **Overnight holding requires a reason.** The overnight interval performed worst. Don't hold overnight just because you can.

---

## Appendix: Strategy Specifications Summary

| Specification | Baseline B | Baseline C |
|---------------|------------|------------|
| Universe | 9 symbols (AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA) | Same |
| Features | 30-dim RTH state + 6 premarket + 9 symbol one-hot = 45 dims | Same |
| Model | Ridge regression (pooled, alpha=1.0) | 4× Ridge (one per interval) |
| Entry times | 10:31 only | 10:31, 11:31, 12:31, 14:00 |
| Exit times | 14:00 only (mandatory) | Any rebalance time, or next-day 10:31 |
| Re-entry | No | Yes |
| Overnight | Never | Allowed |
| Gross exposure target | 10% | 10% |
| Cost assumption | 10 bps one-way | 10 bps one-way |
| Standardization | Train-only mean/std | Train-only mean/std |
| Kill gates | Net CAGR > 0, MaxDD ≤ 15%, Trade days ≥ 30 | Same |

---

*Document generated: January 4, 2026*
