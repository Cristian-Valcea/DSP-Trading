# Trading Strategy Education: A Systematic Framework

**Date**: January 8, 2026
**Purpose**: Move from trial-and-error to systematic strategy development
**Audience**: Retail systematic trader seeking academic/practitioner foundation

---

## Table of Contents

1. [Why Trial-and-Error Fails](#1-why-trial-and-error-fails)
2. [The Academic Foundations](#2-the-academic-foundations)
3. [Strategy Taxonomy](#3-strategy-taxonomy)
4. [What Actually Works (2024-2025 Evidence)](#4-what-actually-works-2024-2025-evidence)
5. [Recommended Reading List](#5-recommended-reading-list)
6. [Implications for Your Portfolio](#6-implications-for-your-portfolio)

---

## 1. Why Trial-and-Error Fails

### The Strategy Decay Problem

Academic research shows that **published trading strategies lose approximately half their Sharpe ratio after publication** (Falck and Rej, 2022). This means:

- A strategy showing Sharpe 1.0 in backtests may only deliver Sharpe 0.5 live
- Strategies based on market anomalies get arbitraged away
- **The more people know about an edge, the smaller it becomes**

> "Analyses of published investment strategies demonstrate that return predictability deteriorates significantly after publication."
> — [Gresham Systematic Strategies Report 2025](https://www.greshamllc.com/media/kycp0t30/systematic-report_0525_v1b.pdf)

### What Trial-and-Error Misses

When you test strategies one-by-one (VRP → killed, TSMOM → killed, ORB → killed), you're missing:

1. **Common failure modes** - WHY do they fail? Is it the same reason?
2. **Strategy classification** - What TYPE of edge was claimed?
3. **Structural vs. behavioral** - Is the edge structural (will persist) or behavioral (will decay)?
4. **Capacity constraints** - Does the strategy even work at your scale?

### A Better Framework: Source of Edge

Before testing ANY strategy, ask:

| Edge Type | Source | Persistence | Example |
|-----------|--------|-------------|---------|
| **Risk Premium** | Compensation for bearing risk | High | Equity risk premium, carry |
| **Behavioral** | Investor mistakes | Medium (decays) | Momentum, value |
| **Structural** | Market mechanics | Medium | Rebalancing flows, index changes |
| **Informational** | Better/faster data | Low (unless maintained) | News sentiment, satellite data |
| **Liquidity** | Providing liquidity | High (if you can bear it) | Market making |

**Your killed strategies:**
- **VRP**: Risk premium (should persist) → But VIX futures don't reliably decay (structural flaw)
- **TSMOM**: Behavioral (momentum) → Works academically, but drawdowns exceed retail tolerance
- **ORB**: Structural (opening mechanics) → Too regime-dependent, edge too thin after costs
- **Sleeve A/B**: Behavioral → Survivorship bias, no actual edge

---

## 2. The Academic Foundations

### Foundational Papers (Must-Read)

#### Factor Investing Origins

1. **Fama & French (1993)** - "Common Risk Factors in Stock Returns"
   - Introduced the 3-factor model (Market, Size, Value)
   - Foundation of modern factor investing
   - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2287202)

2. **Carhart (1997)** - "On Persistence in Mutual Fund Performance"
   - Added Momentum as the 4th factor
   - Showed momentum explains fund performance

3. **Fama & French (2015)** - "A Five-Factor Model"
   - Added Profitability (Quality) and Investment factors
   - Current standard for academic factor models

#### Momentum Research

4. **Jegadeesh & Titman (1993)** - "Returns to Buying Winners and Selling Losers"
   - Original momentum paper
   - ~1% monthly returns over 3-12 month holding periods
   - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=225132)

5. **Moskowitz, Ooi & Pedersen (2012)** - "Time Series Momentum"
   - **THE** foundational paper for trend-following
   - Documents TSMOM across 58 liquid futures
   - Shows 12-month lookback, 1-month hold works across asset classes
   - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)
   - [Free Data](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data)

6. **Asness, Moskowitz & Pedersen (2013)** - "Value and Momentum Everywhere"
   - Shows momentum and value work across all asset classes
   - Documents negative correlation between factors
   - [AQR Research](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere)
   - [Free Data](https://www.aqr.com/Insights/Datasets/Value-and-Momentum-Everywhere-Factors-Monthly)

#### Modern Machine Learning

7. **López de Prado (2018)** - "Advances in Financial Machine Learning"
   - How to properly apply ML to finance
   - Backtesting pitfalls (overfitting, leakage)
   - Feature importance and purged cross-validation

8. **Azevedo, Hoegner & Velikov (2024)** - "Expected Returns on ML Strategies"
   - ML strategies retain ~43% of backtest Sharpe after costs/decay
   - LSTM strategies show Sharpe 0.84 net of costs
   - [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406)

### Key Authors to Follow

| Author | Affiliation | Known For |
|--------|-------------|-----------|
| **Cliff Asness** | AQR | Factor investing, value/momentum |
| **Lasse Pedersen** | AQR/Copenhagen | Time series momentum, betting against beta |
| **Tobias Moskowitz** | AQR/Yale | Time series momentum |
| **Marcos López de Prado** | Cornell/ADIA | ML in finance |
| **Robert Shiller** | Yale | Value (CAPE), behavioral |
| **Eugene Fama** | Chicago | EMH, factor models |
| **Kenneth French** | Dartmouth | Factor data, models |
| **Ernest Chan** | QTS Capital | Retail quant trading books |
| **Robert Carver** | Ex-AHL | Systematic trading books |

### Research Libraries (Free Access)

1. **[AQR Research](https://www.aqr.com/Insights/Research)** - Premier factor research
2. **[AQR Datasets](https://www.aqr.com/Insights/Datasets)** - Free factor return data
3. **[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)** - Factor returns since 1926
4. **[SSRN](https://papers.ssrn.com/)** - Pre-prints of all finance research
5. **[Two Sigma Venn](https://www.venn.twosigma.com/insights)** - Monthly factor reports

---

## 3. Strategy Taxonomy

### Category 1: Factor Investing

**What it is**: Systematically harvesting risk premiums from well-documented factors.

**The Big Five Factors**:

| Factor | Definition | Historical Premium | Why It Exists |
|--------|------------|-------------------|---------------|
| **Value** | Buy cheap, sell expensive (P/B, P/E) | ~2.9% annually | Behavioral overreaction |
| **Momentum** | Buy winners, sell losers (12-1 returns) | ~7-8% annually | Underreaction, herding |
| **Quality** | Buy profitable, low-debt companies | ~3.2% annually | Risk compensation |
| **Low Volatility** | Buy low-beta stocks | ~2-3% annually | Leverage constraints |
| **Size** | Buy small caps | ~2-3% annually | Illiquidity premium |

**2024-2025 Performance** ([Two Sigma Venn](https://www.venn.twosigma.com/insights/may-2025-factor-performance)):
- Momentum: Strong (+3.43% in May 2025 alone)
- Value: Divergent (strong outside US, weak in US)
- Quality: Mixed (weak in US, strong in Europe)
- Low Volatility: Defensive, outperformed during drawdowns

**Implementation for Retail**:
- **ETFs**: MTUM (momentum), VTV (value), QUAL (quality), USMV (low vol)
- **Your Sleeve DM**: Multi-factor approach via dual momentum

### Category 2: Trend Following / Time Series Momentum

**What it is**: Go long assets with positive momentum, short assets with negative momentum.

**Academic Evidence**:
- Moskowitz et al. (2012): Sharpe ~1.0 across 58 futures
- Hurst, Ooi & Pedersen (2017): Trend works over 100+ years
- Managed futures/CTAs: ~$300B industry built on this

**Why It Works**:
- Behavioral: Investors underreact to news, then overreact
- Structural: Hedgers pay speculators to take risk

**Why Your TSMOM Failed**:
- **The strategy works** (Sharpe 4.8 in your backtest!)
- **Drawdowns too large** for retail (-44.8% max DD)
- Academic papers use leverage up to 5x with institutional risk tolerance

**Implementation Options**:
- Reduce leverage (accept lower returns for lower drawdowns)
- Use managed futures ETFs: KMLM, DBMF, CTA
- Apply to ETFs instead of futures (your Sleeve DM approach)

### Category 3: Mean Reversion

**What it is**: Bet that prices revert to a mean (short-term).

**Types**:
- **Statistical Arbitrage**: Pairs trading, cointegration
- **Short-term reversal**: 1-week momentum reversal
- **Bollinger Bands**: Price returns to moving average

**Academic Evidence**:
- Works at very short horizons (intraday to ~1 week)
- Decays quickly as holding period increases
- Heavily capacity-constrained

**Why It's Hard for Retail**:
- Requires fast execution
- Edge is thin, costs matter enormously
- Easily arbitraged by HFTs

### Category 4: Volatility Strategies

**What it is**: Harvest volatility risk premium or trade vol term structure.

**Types**:
- **VRP (Variance Risk Premium)**: Sell volatility (implied > realized)
- **VIX Term Structure**: Short front-month, long back-month
- **Dispersion**: Sell index vol, buy single-stock vol

**Academic Evidence**:
- VRP exists (~3-4% annually)
- But tail risk is enormous (March 2020: VIX 82)

**Why Your VRP Failed**:
- VIX futures don't reliably decay to spot
- Roll returns averaged **-2.58%** (negative!)
- Tail losses dwarf premium collected

**Reality Check**:
> "For both retail and institutional investors, selling volatility is the most successful strategy... but style effects are persistent and cannot be fully explained by systematic risk exposure."
> — [Management Science, 2023](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.4916)

The edge exists, but implementation is treacherous.

### Category 5: Machine Learning Approaches

**What it is**: Use ML to predict returns or optimize execution.

**Academic Evidence (2024-2025)**:
- LSTM strategies: Net Sharpe ~0.84 ([Azevedo et al. 2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406))
- Reinforcement learning (PPO): 21.3% gains, 70.5% win rate
- BUT: 57% performance reduction from backtest to live

**Why ML Strategies Fail for Retail**:
- Overfitting (curve-fitting historical noise)
- Data leakage (future information in features)
- Regime changes (model breaks when market changes)
- Transaction costs (models ignore real costs)

**Your DQN Experience**:
- You correctly identified: "costs dominate"
- ML models optimize for prediction, not P&L after costs
- Academic evidence supports your kill decision

### Category 6: Other Strategies

| Strategy | What It Is | Works For Retail? |
|----------|------------|-------------------|
| **Market Making** | Provide liquidity, earn bid-ask | No (requires infrastructure) |
| **Statistical Arbitrage** | Pairs trading, mean reversion | Difficult (capacity, speed) |
| **Event-Driven** | Trade around earnings, M&A | Possible (requires research) |
| **Index Arbitrage** | Exploit index rebalancing | No (too competitive) |
| **Carry** | Long high-yield, short low-yield | Yes (via ETFs) |

---

## 4. What Actually Works (2024-2025 Evidence)

### Strategies with Documented Edge (Retail Accessible)

#### 1. Multi-Factor ETF Portfolios
**Evidence**: [Aberdeen 2024](https://www.aberdeeninvestments.com/en-us/institutional/insights-and-research/io-2024-multi-factor-why-it-takes-value-quality-momentum)
- Combining Value, Quality, and Momentum reduces drawdowns
- Individual factors have low/negative correlations
- **Your Sleeve DM**: This is what you're doing!

#### 2. Dual/Cross-Sectional Momentum
**Evidence**: [Quantpedia](https://quantpedia.com/strategies/time-series-momentum-effect/)
- 12-month lookback, 1-month hold
- Sharpe 0.5-1.0 depending on universe
- **Your Sleeve DM**: Antonacci-style, working in production

#### 3. Dip-Buying During Volatility
**Evidence**: [CNBC 2025](https://www.cnbc.com/2025/12/31/retail-investors-dip-buying-taco-trade-strong-2025.html)
- Retail investors' second-best year since 1990s
- JPMorgan: Retail single-stock portfolios outperformed AI baskets
- Requires conviction and cash reserves

#### 4. The Wheel Strategy (Options)
**Evidence**: [Hyrotrader 2025](https://www.hyrotrader.com/blog/most-profitable-trading-strategy/)
- Sell cash-secured puts → Get assigned → Sell covered calls
- Systematic premium collection
- Works if you're willing to own the stock

#### 5. Opening Range Breakout (With Filters)
**Evidence**: [Zarattini et al. 2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284)
- 5-minute ORB on "Stocks in Play" (gapped stocks)
- Significant alpha after costs
- Requires active monitoring

### What's NOT Working (2024-2025)

| Strategy | Problem | Evidence |
|----------|---------|----------|
| **Pure VRP** | Roll returns negative | Your kill-test |
| **TSMOM with leverage** | Drawdowns too large | Your kill-test |
| **ML price prediction** | Overfitting, costs | Your DQN kill-test |
| **Sector rotation L/S** | No edge vs. SPY | Your Sleeve B kill-test |
| **Single-stock momentum** | Survivorship bias | Your Sleeve A kill-test |

### The Uncomfortable Truth

> "67% of retail investor accounts lose money when spread betting and/or trading CFDs."
> — [IG International](https://www.ig.com/en/trading-strategies/_how-to-become-a-better-trader-in-2025-241230)

> "Approximately 75% of retail crypto investors lost money on Bitcoin across 95 countries from 2015 to 2022."
> — Bank for International Settlements

**Most retail traders lose money. The ones who win:**
1. Use systematic approaches (not discretionary)
2. Focus on risk management (position sizing, stops)
3. Keep strategies simple (fewer parameters to overfit)
4. Diversify across uncorrelated strategies
5. Accept market returns when no edge exists

---

## 5. Recommended Reading List

### Tier 1: Essential Books (Start Here)

1. **"Quantitative Trading" by Ernest Chan** (2008)
   - Best intro for retail systematic traders
   - MatLab/Excel examples
   - Strategy evaluation framework

2. **"Systematic Trading" by Robert Carver** (2015)
   - Ex-AHL systematic trader
   - Position sizing, risk management
   - Practical implementation focus

3. **"Trading Evolved" by Andreas Clenow** (2019)
   - Python-based backtesting
   - Both futures and equities
   - Full source code provided

4. **"Dual Momentum Investing" by Gary Antonacci** (2014)
   - The strategy behind your Sleeve DM
   - Cross-sectional + time-series momentum
   - Simple, documented edge

### Tier 2: Intermediate Books

5. **"Algorithmic Trading" by Ernest Chan** (2013)
   - Mean reversion, momentum strategies
   - Statistical arbitrage
   - Risk management

6. **"Inside the Black Box" by Rishi Narang** (2009, updated 2014)
   - How quant funds actually work
   - HFT, alpha generation
   - Great for understanding the landscape

7. **"Active Portfolio Management" by Grinold & Kahn** (1999)
   - The "Bible" of quantitative investing
   - Information ratio, transfer coefficient
   - Academic but essential

### Tier 3: Advanced/ML

8. **"Advances in Financial Machine Learning" by López de Prado** (2018)
   - Proper ML methodology for finance
   - Purged cross-validation
   - Feature importance

9. **"Machine Learning for Asset Managers" by López de Prado** (2020)
   - Shorter, more practical
   - Portfolio construction with ML

10. **"Machine Learning for Algorithmic Trading" by Stefan Jansen** (2020)
    - Python/scikit-learn/TensorFlow
    - End-to-end ML trading systems

### Key Papers (Free on SSRN)

| Paper | Year | Topic | Link |
|-------|------|-------|------|
| Time Series Momentum | 2012 | TSMOM foundations | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463) |
| Value and Momentum Everywhere | 2013 | Factor universality | [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere) |
| Profitable Day Trading | 2024 | ORB with filters | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284) |
| Expected Returns on ML | 2024 | ML strategy decay | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406) |
| Factor Momentum Everywhere | 2018 | Meta-momentum | [AQR](https://www.aqr.com/Insights/Research/Working-Paper/Factor-Momentum-Everywhere) |

### Data Sources (Free)

| Source | Data | Link |
|--------|------|------|
| **AQR** | Factor returns, momentum data | [Datasets](https://www.aqr.com/Insights/Datasets) |
| **Ken French** | Fama-French factors since 1926 | [Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) |
| **FRED** | Economic indicators, rates | [fred.stlouisfed.org](https://fred.stlouisfed.org/) |
| **Yahoo Finance** | Price data, ETF data | [finance.yahoo.com](https://finance.yahoo.com/) |
| **Quandl/Nasdaq** | Alternative data (some paid) | [data.nasdaq.com](https://data.nasdaq.com/) |

### Online Resources

1. **[Quantpedia](https://quantpedia.com/)** - Strategy encyclopedia
2. **[Alpha Architect](https://alphaarchitect.com/)** - Research summaries
3. **[QuantStart](https://www.quantstart.com/)** - Tutorials, articles
4. **[PyQuant News](https://www.pyquantnews.com/)** - Python quant resources
5. **[Quantified Strategies](https://www.quantifiedstrategies.com/)** - Backtested strategies

---

## 6. Implications for Your Portfolio

### Why Your Killed Strategies Failed

| Strategy | Claimed Edge | Actual Problem |
|----------|--------------|----------------|
| **Sleeve A** | Momentum (stocks) | Survivorship bias in backtest |
| **Sleeve B** | Sector rotation | Just SPY beta (0.86 correlation) |
| **TSMOM** | Time series momentum | Edge exists, but DD too high for retail |
| **VRP** | Volatility risk premium | Structural flaw (VX doesn't decay) |
| **ORB** | Opening mechanics | Too regime-dependent, thin edge |
| **DQN** | ML prediction | Costs dominate, overfitting |
| **Sleeve IM** | Intraday ML | Walk-forward validation failed |

### Why Sleeve DM Works

Your surviving strategy (ETF Dual Momentum) has these characteristics:

1. **Documented academic edge** - Antonacci, AQR research
2. **Simple implementation** - 12-1 momentum, top 3, monthly rebalance
3. **Low costs** - ETFs, monthly trading (not daily)
4. **Diversified** - Multiple asset classes, not single stocks
5. **Behavioral edge** - Momentum persists across markets
6. **Appropriate drawdowns** - Max DD ~20% (tolerable for retail)

### Framework for Future Strategy Selection

Before testing ANY new strategy, answer:

**1. What is the claimed edge?**
- Risk premium (structural, should persist)
- Behavioral (may decay post-publication)
- Informational (requires maintenance)
- Structural (depends on market mechanics)

**2. Is there academic evidence?**
- Peer-reviewed papers?
- Out-of-sample validation?
- Multiple independent confirmations?

**3. What's the expected decay?**
- Published strategies lose ~50% Sharpe
- Factor in realistic transaction costs
- Account for implementation shortfall

**4. Does it fit your constraints?**
- Capital size ($100k-$1M sweet spot)
- Time availability (daily vs. monthly)
- Risk tolerance (max drawdown you can stomach)
- Execution capabilities (can you trade futures?)

**5. Is it diversifying?**
- Correlation to existing strategies (Sleeve DM)?
- Does it add a new risk factor?
- Is it just disguised beta?

### Recommended Next Steps

1. **Scale what works** - Sleeve DM at 7.8% could go to 15-20%
2. **Add uncorrelated factors** - Consider:
   - Quality factor ETF (QUAL)
   - Low volatility (USMV)
   - International value
3. **Study before testing** - Read Carver's book before next strategy
4. **Lower the bar** - Consider Sharpe 0.3-0.5 if uncorrelated

### The Meta-Lesson

Your kill-testing process is **correct**. Most strategies fail. The problem wasn't the testing - it was the selection of strategies to test.

Going forward:
- Start with academic evidence
- Check for documented edge
- Verify the edge type (structural vs. behavioral)
- Only then implement and test

**One good strategy (Sleeve DM) is better than ten killed strategies.**

---

## Summary

### What You Should Remember

1. **Factor premiums exist** (Momentum, Value, Quality, Low Vol, Carry)
2. **TSMOM works academically** but requires institutional risk tolerance
3. **ML strategies lose ~57%** of backtest performance in live trading
4. **Published strategies decay by ~50%** in Sharpe ratio
5. **Simplicity wins** for retail (fewer parameters, less overfitting)
6. **Your Sleeve DM** is exactly what the academic literature supports

### Reading Priority

1. **This week**: Antonacci "Dual Momentum Investing" (understand your strategy deeply)
2. **This month**: Carver "Systematic Trading" (risk management framework)
3. **Q1 2026**: Chan "Quantitative Trading" (broader strategy evaluation)
4. **Ongoing**: AQR research papers, Two Sigma factor reports

---

**Document Version**: 1.0
**Created**: January 8, 2026
**Purpose**: Educational foundation for systematic strategy development

---

## Sources

### Academic Papers
- [Moskowitz, Ooi & Pedersen - Time Series Momentum (2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)
- [Asness, Moskowitz & Pedersen - Value and Momentum Everywhere (2013)](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere)
- [Zarattini, Barbon & Aziz - Profitable Day Trading (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284)
- [Azevedo, Hoegner & Velikov - Expected Returns on ML Strategies (2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406)

### Industry Research
- [AQR Research Library](https://www.aqr.com/Insights/Research)
- [AQR Datasets](https://www.aqr.com/Insights/Datasets)
- [Two Sigma Venn Factor Reports](https://www.venn.twosigma.com/insights)
- [Gresham Systematic Strategies Report 2025](https://www.greshamllc.com/media/kycp0t30/systematic-report_0525_v1b.pdf)

### Factor Performance
- [JP Morgan Factor Views Q4 2025](https://am.jpmorgan.com/us/en/asset-management/adv/insights/portfolio-insights/asset-class-views/factor/)
- [Aberdeen Multi-Factor Research 2024](https://www.aberdeeninvestments.com/en-us/institutional/insights-and-research/io-2024-multi-factor-why-it-takes-value-quality-momentum)
- [Alpha Architect Momentum Research](https://alphaarchitect.com/momentum-factor-investing/)

### Retail Trading Evidence
- [CNBC - Retail Investors 2025](https://www.cnbc.com/2025/12/31/retail-investors-dip-buying-taco-trade-strong-2025.html)
- [Hyrotrader - Most Profitable Strategies 2025](https://www.hyrotrader.com/blog/most-profitable-trading-strategy/)
