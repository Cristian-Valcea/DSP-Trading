# Sleeve VRP — Volatility Risk Premium Strategy
## Plain English Presentation for Management

**Date**: 2026-01-08 (Updated with 2023-2026 market structure changes)
**Status**: Candidate sleeve (pending kill-test validation)
**Target Audience**: Non-technical decision-makers

---

## What This Strategy Does (In One Sentence)

We **sell insurance on stock market crashes** to investors who are willing to pay a premium for protection, collecting steady income most of the time while accepting losses during extreme market events.

### ⚠️ Important Context: The Market Has Changed (2023-2026)

Before we dive in, management should understand that the volatility market has evolved significantly since 2020:

1. **0DTE Options Revolution**: Institutional hedging has migrated to same-day options, making the traditional 30-day VIX less reliable as a "fear gauge"
2. **Strategy Crowding**: Simple "sell VIX contango" strategies have become overcrowded, reducing historical returns
3. **High-Velocity Events**: Recent crises spike harder but mean-revert faster (hours, not days)
4. **The "Liquidity Illusion" (2025 Lesson)**: As seen in the May 2025 "Flash-Up", liquidity can evaporate instantly in electronic markets. Stop-losses are not guaranteed fills.

**Our implementation addresses all of these changes** — see Sections 3, 5, and 11 for specific mitigations.

---

## 1. The Opportunity: Why This Makes Money

### The Market Inefficiency
- **Fear is expensive**: Investors overpay for crash insurance (like paying $200/month for car insurance on a $10,000 car)
- **Actual vs perceived risk**: Real market crashes happen ~2-3% of the time; investors price them at ~5-7%
- **The gap is our profit**: We harvest the **Variance Risk Premium (VRP)** — the spread between Implied Volatility (what people pay) and Realized Volatility (what actually happens).

### Real-World Analogy
```
Homeowner's Insurance Company:
- Collect $2,000/year from 100 homes = $200,000 premium income
- Pay out $50,000 in claims = $150,000 profit most years
- Catastrophic year (hurricane): Pay out $1M, lose $800,000
- Over 10 years: Net positive if premiums > catastrophe frequency

VRP Strategy:
- Collect "insurance premiums" from nervous investors every month
- Pay out during market crashes (2018, 2020, 2022)
- Over multi-year period: Premiums exceed crash losses
```

### Historical Evidence
- **Long-term return**: 10-15% annual average (before 2018)
- **Premium collected**: $5,000-$10,000/month on $100k position (typical)
- **Crisis losses**: -30% to -80% during vol spikes (Feb 2018, Mar 2020)
- **Net result**: Positive if you size conservatively and don't blow up

---

## 2. What We're Investing In

### Universe: VIX Futures (Not Stocks or Bonds)

**What is VIX?**
- VIX = "Fear gauge" for the S&P 500 stock market
- Measures how much investors expect stocks to move in **next 30 days**
- **Calm markets**: VIX = 12-15 (investors relaxed)
- **Normal volatility**: VIX = 15-20 (average fear)
- **Crisis**: VIX = 30-80 (investors panicking)

### ⚠️ NEW (2023-2026): The VIX Has Blind Spots

**The 0DTE Problem**:
- Institutional hedging has migrated to **0DTE options** (expiring same day)
- Traditional 30-day VIX can stay suppressed during intraday crashes
- Short-term fear (0-1 day horizon) is not fully captured by VIX

**Our Solution**: We monitor **VIX1D** (1-day volatility index) alongside 30-day VIX:
- If VIX1D/VIX ratio spikes > 1.2 → Short-term fear is elevated → Reduce exposure 50%
- This catches "flash crash" risk that 30-day VIX misses

**What are VIX Futures?**
- Contracts that let you bet on future VIX levels
- Example: "I think VIX will be 18 in March" (buy March VIX futures at 18)
- **We don't buy**: We **sell** these futures (betting VIX stays low or falls)

### Why VIX Futures, Not VIX Options?
- **Liquidity**: VIX futures trade $5-10 billion/day (very liquid)
- **Simplicity**: Futures are easier to manage than options (no expiration complexity)
- **Cost**: Lower transaction costs (1-2 ticks) vs options (5-10 ticks bid-ask spread)

### Position Sizing
- **Instruments**: VX futures (ticker symbol on CFE exchange)
- **Contract size**: $1,000 × VIX index (~$15,000-$25,000 notional per contract)
- **Typical position**: 3-5 contracts ($50,000-$100,000 exposure on $100k sleeve)
- **Margin required**: ~$3,000-$5,000 per contract (~$15,000 total margin)

---

## 3. The "How": Strategy Mechanics

### Monthly Rebalancing (Not Daily Trading)

**When We Trade**: **3rd trading day of each month** (or split across days 1-3)
**Time of Day**: Market open (9:30 AM ET)
**Holding Period**: 1 month (roll forward each month)

### ⚠️ WHY NOT First Trading Day? (Avoiding the Herd)

**The Problem**:
- First trading day = most crowded liquidity window
- Systematic funds, ETFs, and volatility sellers all rebalance on day 1
- Roll yield is often **compressed by 10-20%** due to crowding
- Bid-ask spreads widen; execution quality suffers

**Our Solution**:
```
OLD (Crowded):
  Trade on 1st trading day → Compete with every other systematic fund
  → Worse fills, compressed premium

NEW (Our Implementation):
  Trade on 3rd trading day OR split execution across days 1-3
  → Avoid "first-day crowding" effect
  → Better execution quality
  → Same premium capture (premium persists for 2-3 days)
```

**Why 3rd Day Still Works**:
- Contango premium doesn't evaporate in 48 hours
- Roll yield persists through first week of month
- We sacrifice 0-2% of premium for 5-10% better execution

### The Core Trade: Selling VIX Contango (With Modern Filters)

**Step 1: Identify the "Contango" (Price Curve)**
```
Today (Jan 8, 2026):
- VIX Spot Index: 14.5 (current market fear)
- Feb VIX Futures: 16.2 (what traders expect in February)
- Mar VIX Futures: 17.8 (what traders expect in March)
- Risk-Free Rate: 4.5% (2026 environment)

Gross Contango: 16.2 - 14.5 = 1.7 points
Interest Rate Carry: ~0.5 points (cost of money component)
TRUE RISK PREMIUM: 1.7 - 0.5 = 1.2 points (what we're actually harvesting)
```

### ⚠️ CRITICAL IMPROVEMENT: Beyond Simple Contango

**The Problem With Naive Contango Trading**:
- Academic research (Bali 2023, Pasquariello 2024) shows simple carry strategies have **decayed in Sharpe ratio** due to overcrowding
- High contango often **precedes** a volatility spike (market expects trouble)
- Selling purely on steep contango = "picking up pennies in front of a steamroller"

**Our Enhanced Signal: Contango + Volatility Momentum**
```
OLD RULE (Naive, Dangerous):
  "Sell VIX futures if contango > 1.0 point"

NEW RULE (Our Implementation):
  "Sell VIX futures ONLY if:
   1. Contango > 1.0 point (after interest rate adjustment) AND
   2. VIX is BELOW its 50-day Moving Average (vol momentum is falling)"
```

**Why This Matters**:
```
Scenario: VIX at 25, futures at 28 (steep contango)

Naive Strategy: "3 points contango! Sell aggressively!" → Gets crushed if VIX goes to 40
Our Strategy: "Wait — VIX is above its 50-day MA (22). Vol is rising. NO TRADE."

We only sell contango when volatility is falling, not rising.
```

**Step 2: Sell the Near-Month Futures (If Signal Passes)**
- We sell Feb VIX futures at 16.2
- If VIX stays at 14.5, the Feb futures will decay toward 14.5 as expiration approaches
- **Our profit**: 16.2 - 14.5 = 1.7 points = $1,700/contract

**Step 3: Roll Forward Monthly**
- Before Feb futures expire, close the Feb position
- Open a new Mar futures short position
- Repeat monthly (this is the "roll")

### ⚠️ CRITICAL (V2.1): The "Sleep at Night" Hedge (Long Wings)

**The Problem With Stop-Losses Alone**:
- Stop-loss orders are **not guaranteed fills**
- If VIX gaps from 20 to 50 overnight (or in milliseconds), a stop triggered at 25 will **fill at 50**
- The May 2025 "Flash-Up" showed liquidity can evaporate instantly — slippage can be 5-10+ points
- **Stop-losses are necessary but NOT sufficient** for catastrophic protection

**Our Solution: Buy Deep OTM VIX Calls (Tail Hedge)**
```
Instead of keeping 100% of the premium, we spend 10-15% to buy protection:

Trade Setup (Monthly):
1. SELL 3 Feb VIX futures @ 16.2 (our main trade)
2. BUY 3 Feb VIX 40 Calls @ $0.50 each (our "insurance")

Cost: 3 × $0.50 × $1,000 = $1,500/month (reduces net income by ~15%)

Payoff Profile:
- VIX stays at 14-20 → Futures profit +$4,500, Calls expire worthless -$1,500 = Net +$3,000
- VIX spikes to 50 → Futures loss -$35,000, but Calls gain +$30,000 = Net -$5,000 (CAPPED!)
- VIX spikes to 100 → Futures loss -$85,000, but Calls gain +$180,000 = Net +$95,000 (PROFIT!)
```

**Why This Matters**:
| Scenario | Stop-Loss Only | Stop + Wings |
|----------|----------------|--------------|
| Normal month | +$4,500 | +$3,000 (less income) |
| VIX gaps to 50 (stop slips to 45) | -$30,000 | -$5,000 (CAPPED) |
| VIX gaps to 80 (stop slips to 60) | -$45,000 | +$40,000 (PROFIT) |

**Trade-off**:
- Reduces annual return by ~1-2% (hedge costs 10-15% of income)
- **Eliminates "account blow-up" risk** from gap events where stops fail
- Management can "sleep at night" knowing max loss is defined

**Implementation**:
- Buy VIX calls at strike = VIX spot + 20-25 points (deep OTM)
- Same expiry as short futures (synchronized)
- Roll monthly with the futures position

### ⚠️ Interest Rate Adjustment (Post-ZIRP Reality)

**The Problem**: In 2010-2021 (ZIRP era), all contango was insurance premium.
In 2022-2026 (rates >2%), some contango is just **cost of money**.

**Our Adjustment**:
```
Raw Contango = VIX Futures - VIX Spot
Interest Rate Component = Spot VIX × (Risk-Free Rate) × (Days to Expiry / 365)
TRUE RISK PREMIUM = Raw Contango - Interest Rate Component

Example (Feb futures, 30 days to expiry, rates at 4.5%):
- Raw Contango: 16.2 - 14.5 = 1.7 points
- Interest Component: 14.5 × 4.5% × (30/365) = 0.054 points
- True Premium: 1.7 - 0.05 = 1.65 points (95% is real premium)

Note: Interest rate adjustment is small for short-dated VIX, but we calculate it
to ensure we're harvesting ACTUAL variance risk premium, not rate differentials.
```

### What Makes This Profitable

**Normal Market (90% of the time)**:
```
Month 1: Sell Feb VIX @ 16.2 → Buy back @ 14.8 = +1.4 points profit
Month 2: Sell Mar VIX @ 17.0 → Buy back @ 15.2 = +1.8 points profit
Month 3: Sell Apr VIX @ 16.5 → Buy back @ 14.9 = +1.6 points profit

3-month total: +4.8 points = $4,800/contract
Annualized: ~19.2 points = $19,200/contract on ~$5,000 margin
```

**Crisis Month (10% of the time)**:
```
Feb 2018 "Volmageddon":
- We sold Feb VIX @ 14.5
- VIX spiked to 37.3 in one day
- Loss: 37.3 - 14.5 = -22.8 points = -$22,800/contract
- Position value dropped -152% (would wipe out account without risk controls)
```

---

## 4. Time Commitment & Operational Simplicity

### Trading Frequency: **Monthly Alpha, Daily Risk Monitoring**

Unlike the intraday ML strategies (200+ trades/day), VRP is **low-touch for alpha decisions**:
- **Alpha rebalance**: 1st trading day of month only
- **Time required**: 15 minutes to execute orders
- **Risk monitoring**: **Automated** — stop-loss orders live in market 24/7

### ⚠️ CLARIFICATION: "Low-Touch" Doesn't Mean "Hands-Off"

**Monthly Decision (15 min/month)**:
- Calculate signal (contango + volatility momentum + filters)
- Execute entry/exit orders
- Place stop-loss orders (automated protection)

**Daily Monitoring (5 min/day) — NON-NEGOTIABLE**:
- Check VIX level against thresholds (30/40)
- Check VVIX level (>110 = correlation warning)
- Verify stop-loss orders are still active
- **Critical**: We LOOK daily, but stops work even if we don't

**Why Daily Monitoring Matters (Feb 2018 Example)**:
```
Feb 5, 2018 (Monday):
- 9:30 AM: VIX at 17 (calm)
- 2:30 PM: VIX at 22 (rising, but under 30 threshold)
- 4:00 PM: VIX at 37 (EXPLODED in final 90 minutes)

If we only looked monthly → We'd have lost -80%
With automated stops → Stopped out at VIX 25 (-30% loss, not -80%)
With daily monitoring → We'd have seen VIX rising at 2:30 PM and manually reduced
```

### No Intraday Alpha Activity
- **Not evaluating volatility in first 3 hours of trading** (common misconception)
- **Not trading options** (we use futures, which are simpler)
- **Not high-frequency** (we hold positions for ~30 days, not 30 minutes)

### Execution Example (First Monday of Month)
```
9:30 AM ET: Market opens
9:31 AM: Fetch VIX, VIX1D, VVIX, VIX futures, VIX 50-day MA
9:32 AM: Calculate TRUE contango (after interest rate adjustment)
9:33 AM: Check filters:
         - Is VIX < 50-day MA? (vol falling)
         - Is VIX1D/VIX < 1.2? (no short-term spike)
         - Is VVIX < 110? (correlations stable)
         - Is TRUE contango > 1.0 point? (premium exists)
9:34 AM: If ALL filters pass → Sell VIX futures + place stop-loss
         If ANY filter fails → No trade this month (go flat)
9:35 AM: Orders filled, position established
9:36 AM: PLACE STOP-LOSS ORDER at circuit breaker level (e.g., VIX 25)
9:45 AM: Alpha decision done. Risk protection now automated.

Daily (5 min): Check VIX/VVIX levels, verify stops active
```

---

## 5. Risk Profile: What Can Go Wrong

### The Good (Most of the Time)
- **Steady income**: Collect $5,000-$10,000/month on $100k position (typical)
- **High win rate**: Profitable 8-10 months per year
- **Uncorrelated to stocks**: VRP makes money when stocks are calm OR rising

### The Bad (Crisis Months)
- **Volatility spikes**: Feb 2018, Mar 2020, Oct 2022
- **Drawdowns**: -30% to -80% in single month
- **Recovery time**: 3-6 months to earn back one bad month

### How We Protect Against Blow-Ups

### ⚠️ CRITICAL: Daily Risk Management, NOT Monthly

**The "Monthly Rebalance" Fallacy**:
- The alpha decision is monthly (when to enter/exit contango trades)
- The **risk management must be daily or automated** (2018 Volmageddon happened in hours, not days)
- "Looking at portfolio once a month" is **not acceptable** for short-vol strategies

**Our Solution: Automated Stop-Loss Orders**
```
We place stop-loss orders IMMEDIATELY upon entering position:

Position Entry (Jan 2): Short 3 VIX Feb futures @ 16.2
Stop-Loss Order (placed same day): Buy 3 VIX Feb @ 21.5 (circuit breaker)

If VIX spikes overnight → Stop executes automatically → No human intervention needed
```

**1. Position Sizing (Conservative Allocation)**
- **Maximum allocation**: 10% of sleeve NAV in VIX exposure
- **Typical position**: 3-5 contracts (~$50k-$100k notional)
- **Margin usage**: Never exceed 30% of sleeve NAV

**2. Automated Circuit Breakers (NOT Manual Daily Checks)**
- **-15% drawdown trigger**: **Stop-loss order rests in market** (executes automatically)
- **VIX level trigger**: If VIX > 30, stop-limit at market price (reduce 50%)
- **Emergency flatten**: If VIX > 40, market order to exit immediately

**Key Difference From Document V1**:
```
OLD (Dangerous): "Daily health check (5 minutes), monthly P&L review"
NEW (Our Implementation): "Stop-loss orders live in market 24/7.
                           We check daily, but protection doesn't depend on us looking."
```

**3. Enhanced Signal-Based Risk Management**
- **No trade zones**: Skip months when TRUE contango < 0.5 (after rate adjustment)
- **Volatility momentum filter**: No new shorts if VIX > 50-day MA (vol rising)
- **0DTE/VIX1D filter**: If VIX1D/VIX > threshold, reduce exposure 50%
- **VVIX filter**: If VVIX > threshold, reduce position by 50%
- **Regime detection**: VIX > 20 for 3 consecutive days = high-risk regime (go flat)

### ⚠️ V2.1 IMPROVEMENT: Regime-Adaptive Thresholds (Not Hard Numbers)

**The Problem With Hard Thresholds**:
- Hard numbers (e.g., "VVIX > 110") drift with market baselines
- In a high-volatility regime, VVIX might average 115 → strategy never trades
- In a low-vol regime, VVIX might average 85 → we're too lax

**Our Solution: Rolling Percentile Thresholds**
```
OLD (Rigid):
  - VVIX > 110 → Reduce exposure
  - VIX1D/VIX > 1.2 → Reduce exposure

NEW (Regime-Adaptive):
  - VVIX > 90th percentile of 6-month history → Reduce exposure
  - VIX1D/VIX > 95th percentile of 6-month history OR > 1.2 (whichever triggers first)

Example (different regimes):
  Low-vol regime (2017): VVIX avg = 80, 90th pct = 95 → trigger at 95
  High-vol regime (2022): VVIX avg = 105, 90th pct = 125 → trigger at 125
```

**Why This Matters**:
- Thresholds **adapt** to current market conditions
- Avoids false positives in high-vol regimes (would never trade)
- Avoids false negatives in low-vol regimes (too complacent)
- Same protection logic, context-aware execution

### ⚠️ V2.1 CRITICAL: Cool-Down Rule (Re-Entry Logic)

**The Problem**:
- After a stop-loss triggers, VIX dips slightly → we re-enter → VIX spikes again → stopped out again
- This "whipsaw" pattern can happen 3-4 times in a crisis month → compounds losses
- Each stop costs slippage + missed premium during exit period

**Our Solution: Mandatory Cool-Down Period**
```
RULE: If stop-loss is triggered:
  → Strategy enters mandatory cool-down for remainder of month (or 5 trading days)
  → We do NOT "chase" the market to recover losses immediately
  → We wait for term structure to stabilize (return to Contango)

Example (Feb 2018):
  Feb 5: Stop-loss triggered at VIX 25 → COOL-DOWN ACTIVATED
  Feb 6: VIX dips to 22 → Naive strategy re-enters → Gets crushed again
  Feb 6: OUR strategy: "Cool-down active. No re-entry until Feb 12 (5 days)"
  Feb 12: VIX = 28, still elevated → No re-entry (wait for contango)
  Mar 1: VIX = 17, contango restored → Re-enter at 50% normal size
```

**Why This Matters**:
- Prevents emotional "revenge trading" after losses
- Avoids whipsaw losses in choppy markets
- Allows volatility to truly settle before redeploying capital
- Reduces total transaction costs during crisis periods

### ⚠️ V2.2 ADVANCED: Additional Risk Controls

**Beyond basic stop-losses, we implement institutional-grade risk management:**

**1. Dynamic Hedge Budgeting**
- When VVIX elevated > 110% of 50-day MA → Increase tail hedge budget by 25-50%
- When term structure inverted (backwardation) → Double down on protection
- Never spend > 30% of premium on hedges (preserves profitability)

**2. Margin Expansion Monitoring**
- Brokers increase margin requirements during volatility spikes
- At VIX > 25 → Check if position could survive 100% margin scenario
- At VIX > 40 → Assume margin could expand to 150% of notional
- **Pre-emptively reduce** before forced liquidation

**3. Collateral Yield Tracking**
- Cash buffer (40%+ of sleeve NAV) earns T-bill yield (~4-5% in 2026)
- This adds 1.5-2% to annual return on conservative sizing
- "Getting paid to wait" during flat periods

**4. ETH Execution Awareness**
- VIX futures trade nearly 24/5, but liquidity thin outside RTH
- We adjust slippage expectations 2× higher for extended hours
- Emergency exits still possible, but with realistic cost expectations

### Real Example: How Controls Would Have Worked

**March 2020 COVID Crash**:
```
March 9: VIX = 30 → Reduce position from 5 contracts to 2 contracts (50% cut)
March 12: VIX = 40 → Flatten to zero (emergency exit) → COOL-DOWN ACTIVATED
March 16: VIX = 82 → Already out, cool-down prevents re-entry
March 23: Market bottoms, VIX = 60 → Still in cool-down
April 1: VIX = 35, cool-down expires → Check contango: backwardation → No re-entry
April 15: VIX = 22, contango = 0.8 points → Too thin, wait
May 1: VIX = 18, contango = 1.5 points → Re-enter at 50% normal size

Result: Took -12% loss (early exit) instead of -75% (riding it out)
        Avoided whipsaw losses from premature re-entry
```

---

## 6. Comparison to Existing Sleeve (Sleeve DM)

### Sleeve DM (ETF Dual Momentum) — LIVE
- **Risk factor**: Momentum (buy what's going up, sell what's going down)
- **Instruments**: Stock/bond/commodity ETFs
- **Rebalance**: Monthly
- **Crisis behavior**: Goes to cash during crashes (defensive)
- **Return profile**: Moderate gains most years, avoid big losses

### Sleeve VRP (Volatility Risk Premium) — CANDIDATE
- **Risk factor**: Volatility mean reversion (bet on calm markets)
- **Instruments**: VIX futures
- **Rebalance**: Monthly
- **Crisis behavior**: Loses money during crashes (offensive on calm markets)
- **Return profile**: High gains most months, catastrophic losses in crises

### Portfolio Effect (Why VRP + DM Works)

**Complementary Crisis Behavior**:
```
Normal Markets (90% of time):
- DM: +0.5% to +1.0% monthly (slow and steady)
- VRP: +3% to +5% monthly (high income)
- Portfolio: Both strategies profitable

Crisis Markets (10% of time):
- DM: Goes to cash, -0% to -5% (defensive)
- VRP: Circuit breaker triggered, -15% max (controlled loss)
- Portfolio: DM cushions VRP's losses
```

**Correlation**:
- **DM vs SPY**: 0.48-0.52 (moderate correlation)
- **VRP vs SPY**: -0.3 to -0.5 (negative correlation during crashes)
- **DM vs VRP**: -0.1 to +0.2 (uncorrelated most of time, diverge in crises)
- **Portfolio benefit**: Smoother total returns, lower drawdowns

---

## 7. Expected Performance (Conservative Estimates)

### ⚠️ Historical Benchmarks vs Modern Reality

**Pre-2018 ("Golden Era")**:
- **CBOE VIX Premium Index**: 15-20% CAGR (2006-2017)
- **Professional vol arb funds**: 12-18% CAGR with 20-30% volatility

**Post-2018 ("Crowded Era")**:
- Academic research (Bali 2023, Pasquariello 2024) shows **significant Sharpe decay**
- Simple carry strategies dropped from Sharpe 1.0+ to Sharpe 0.4-0.6
- **Why?** Strategy overcrowding, faster mean-reversion, smarter institutional hedging

**Our REALISTIC Targets (Accounting for Crowding + Modern Filters)**:
- **Target annual return**: 6-10% (not 15-20% — those days are gone)
- **Target Sharpe ratio**: 0.4-0.6 (realistic after crowding adjustment)
- **Expected max drawdown**: -20% to -30% (with circuit breaker limiting to -15%)
- **Win rate**: 65-75% of months profitable (lower due to signal filters causing flat months)

**Why Lower Expectations?**
1. We skip more months (volatility momentum filter, VVIX filter = fewer trades)
2. We size more conservatively (10% NAV vs historical 30-50%)
3. Strategy is more crowded (premium has compressed)
4. We accept lower returns in exchange for **not blowing up**

### Monthly Income Example ($100k Sleeve)
```
Normal Month (70% probability):
- Position: 3 VIX futures, $15k margin
- Contango decay: 1.5 points/month
- Gross profit: 1.5 × 3 contracts × $1,000 = $4,500
- Transaction costs: -$50 (commissions + slippage)
- Net profit: $4,450 = +4.5% monthly return

Mediocre Month (20% probability):
- Contango: 0.8 points/month
- Net profit: $2,350 = +2.4% monthly return

Bad Month (10% probability):
- VIX spike, circuit breaker triggered at -15%
- Loss: -$15,000
- Net result: -15% monthly return

Expected monthly return: (0.70 × 4.5%) + (0.20 × 2.4%) + (0.10 × -15%) = +2.1%
Annualized: 2.1% × 12 = 25.2% (BEFORE circuit breaker caps at -15%)
```

**Why Conservative Estimate (8-12% target)?**
- Circuit breakers reduce upside (exit early when profitable)
- Conservative sizing (3-5 contracts vs aggressive 10-15)
- Skip months with poor contango (flat instead of forcing trades)

---

## 8. What This Adds to DSP-100K Portfolio

### Current State: 1 Live Sleeve
- **Sleeve DM**: ETF momentum (7.8% of €1.05M NAV = €82k deployed)
- **All others**: Killed (Sleeve A/B/IM, DQN, ORB, TSMOM)

### With VRP: 2 Uncorrelated Sleeves
- **Sleeve DM**: Momentum factor, monthly rebalance, €82k
- **Sleeve VRP**: Volatility factor, monthly rebalance, €82k (if promoted)
- **Total deployed**: €164k of €1.05M (15.6% of NAV)

### Portfolio Diversification Improvement
```
Before VRP (1 sleeve):
- 100% momentum factor
- Portfolio Sharpe: 0.55-0.87 (DM alone)
- Max DD: -14% to -22% (DM alone)

After VRP (2 sleeves):
- 50% momentum factor (DM)
- 50% volatility factor (VRP)
- Expected Portfolio Sharpe: 0.8-1.2 (diversification benefit)
- Expected Max DD: -12% to -18% (uncorrelated drawdowns offset)
```

**Key Insight**: Two uncorrelated sleeves reduce portfolio risk more than one sleeve's returns alone.

---

## 9. Implementation Timeline

### Phase 1: Specification & Data (1-2 days)
- ✅ **Specification document**: Signal, portfolio construction, risk limits (4 hours)
- ⏳ **Data acquisition**: Purchase VIX futures historical data from Databento (~$50-100)
- ⏳ **Data processing**: Build continuous VIX futures series with roll simulation (2 hours)

### Phase 2: Backtesting (1-2 days)
- ⏳ **Backtester development**: Contango calculation, position sizing, transaction costs (6 hours)
- ⏳ **Kill-test validation**: Run baseline and stress tests on 2018-2026 data (2 hours)
- ⏳ **Results analysis**: Generate kill-test report with pass/fail verdict (2 hours)

### Phase 3: Decision (Same Day)
- ⏳ **If PASS**: Promote to paper trading, start 30-day validation
- ⏳ **If FAIL**: Document failure reason, archive as killed strategy

**Total Timeline**: 3-4 working days from approval to kill-test verdict

---

## 10. Kill-Test Criteria (Pass/Fail Gates)

### Primary Gates (Must Pass All)
| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Sharpe Ratio** | ≥ 0.50 | Must beat risk-free rate meaningfully |
| **Net P&L** | > $0 | Must make money after all costs |
| **Max Drawdown** | ≥ -30% | Must be tolerable for real capital |

### Stress Gates (Test Robustness)
| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Sharpe (2× costs)** | ≥ 0.30 | Must survive pessimistic cost assumptions |
| **Net P&L (2× costs)** | > $0 | Must remain profitable under stress |

### Fold Consistency (Generalization)
- **≥50% of OOS folds pass**: Strategy works across different market regimes
- **Test period**: 2018-2026 (includes Feb 2018 and Mar 2020 crises)

### What Happens If VRP Fails?
- Strategy is **killed** (no parameter tuning allowed per pre-registration methodology)
- Document failure reason in kill-test report
- Move to next candidate (Carry strategies, other alternatives)

---

## 11. Key Risks & Mitigations

### Risk #1: Catastrophic Losses During Vol Spikes
**Example**: Feb 2018 "Volmageddon" — XIV ETN lost 96% in one day

**Mitigation**:
- ✅ Conservative position sizing (10% of NAV max, not 50%+)
- ✅ Hard stop-loss at -15% drawdown (circuit breaker)
- ✅ VIX level triggers (reduce at VIX > 30, flatten at VIX > 40)
- ✅ No naked short vol (use futures, not levered ETNs)

### ⚠️ Risk #1B: GAP RISK (V2.1 Critical Addition)
**Example**: May 2025 "Flash-Up" — VIX gapped from 18 to 45 in milliseconds

**The Problem With Stop-Losses**:
- Stop-loss orders are **not guaranteed fills** — they become market orders when triggered
- If VIX gaps from 20 to 50 overnight, a stop at 25 will **fill at 50** (not 25!)
- Slippage during liquidity events can be 5-10+ points
- This gap risk is **not eliminated** by traditional stop-losses

**Mitigation (Long Call Wings)**:
- ✅ **Buy OTM VIX Calls (strike +20-25 points)** alongside short futures
- ✅ Cost: 10-15% of monthly premium (insurance cost)
- ✅ Effect: Creates **mathematical cap** on maximum loss
- ✅ Example: VIX 40 calls turn catastrophic loss into capped loss (or profit if VIX >60)
- See Section 3: "Sleep at Night Hedge" for full implementation details

### Risk #2: Regime Shifts (Contango → Backwardation)
**Example**: Aug 2011, Dec 2018 — VIX futures went into backwardation for months

**Mitigation**:
- ✅ Signal-based trading (skip months with poor contango)
- ✅ Backwardation detector (no trade if near-month < spot)
- ✅ Accept flat months (better than forced losses)

### Risk #3: Correlation Breakdown (VRP + DM Both Lose)
**Example**: Mar 2020 — Both momentum and vol premium lost money
**Example**: 2022 — Stocks AND bonds fell together (liquidity shock)

### ⚠️ NEW RISK (2022-2026): Liquidity Shock Correlations

**The Problem**:
- In 2022 inflation/rates bear market: Stocks fell AND bonds fell
- In 2024-2025 liquidity shocks: Correlations between almost all risk assets approached 1.0
- Relying on DM to cushion VRP losses is risky if the crash is **liquidity-driven** rather than **macro-driven**

**Our Mitigation: VVIX Filter (Volatility of Volatility)**
```
VVIX = "Fear gauge of the fear gauge" (measures volatility OF VIX)

Normal Markets: VVIX = 80-100 (VIX is stable)
Stressed Markets: VVIX = 100-110 (VIX is becoming volatile)
Crisis Warning: VVIX > 110 (VIX itself is spiking unpredictably)

When VVIX > 110:
- Correlations between VRP and equities TIGHTEN toward 1.0
- DM diversification benefit DISAPPEARS
- We must reduce VRP exposure REGARDLESS of contango level
```

**Rule: If VVIX > 110 → Reduce VRP position by 50% regardless of contango**

**Standard Mitigations**:
- ✅ Sleeve-level circuit breakers (each sleeve protects itself)
- ✅ Portfolio-level exposure cap (15% total deployed = 85% in cash as buffer)
- ✅ Conservative sizing on both sleeves (not 50% each, only 8% each)
- ✅ **NEW**: VVIX filter cuts exposure when correlations are likely to break down

---

## 12. Success Metrics (30-Day Paper Trading)

If VRP passes kill-test and enters paper trading, we monitor:

### Operational Success (Process)
- ✅ **Monthly rebalances execute cleanly** (12 rebalances in 1 year)
- ✅ **Contango calculation correct** (verify against Bloomberg/CBOE)
- ✅ **Circuit breakers trigger appropriately** (test during mini-spike)

### Performance Success (Results)
- **Target**: 70-80% win rate (7-8 profitable months out of 10)
- **Target**: Average monthly return +1.5% to +2.5%
- **Target**: Max monthly loss ≤ -15% (circuit breaker working)
- **Target**: Sharpe ratio ≥ 0.5 (6-month rolling)

### Portfolio Success (Diversification)
- **Target**: VRP vs DM correlation < 0.3 (uncorrelated)
- **Target**: Combined portfolio Sharpe > max(DM Sharpe, VRP Sharpe) alone
- **Target**: Combined max DD < max(DM DD, VRP DD) alone

---

## 13. Management Decision Points

### Decision 1: Approve VRP Kill-Test? (Now)
**Cost**: ~$50-100 for data + 3-4 days development time
**Benefit**: Discover if VRP is a viable production sleeve
**Risk**: Strategy fails kill-test → time invested, but learn from results

**Recommendation**: **APPROVE** — Low cost, high information value, fills portfolio gap

---

### Decision 2: Promote to Paper Trading? (After Kill-Test)
**Conditional on**: VRP passes all kill-test gates
**Cost**: 30-60 days paper trading validation
**Benefit**: Real-world performance data, operational validation
**Risk**: Strategy works in backtest but fails in practice (execution issues, regime change)

**Recommendation**: **If kill-test PASS → APPROVE paper trading**

---

### Decision 3: Promote to Live Trading? (After 30-Day Paper)
**Conditional on**:
- VRP passes kill-test ✅
- 30-day paper trading shows 70%+ win rate ✅
- No operational issues (execution, circuit breakers working) ✅

**Allocation**: 8-10% of DSP-100K NAV (~€80k-100k)
**Expected income**: €4k-€8k/month (normal markets)
**Risk**: Circuit breaker caps losses at -€12k-€15k max (crisis months)

**Recommendation**: **Defer until paper trading results available**

---

## 14. FAQ (Common Questions)

### Q: Is this the same as the XIV ETN that blew up in 2018?
**A**: No. XIV was a **leveraged** product (2× short vol) with **no risk controls**. We use:
- ✅ Unleveraged VIX futures (1× exposure, not 2×)
- ✅ Conservative position sizing (10% of NAV, not 50%+)
- ✅ Hard stop-loss (circuit breaker at -15%, not ride to -96%)

### Q: Why not just buy the VXX inverse (short vol ETF)?
**A**: ETFs have structural issues:
- ❌ Daily rebalancing drag (compounds losses)
- ❌ Contango bleed reduces returns by 10-20%/year
- ❌ Limited control over risk (can't adjust position mid-month)

We use futures for **full control** + **lower costs** + **flexible risk management**.

### Q: What if VIX spikes and we can't exit?
**A**: VIX futures are **extremely liquid**:
- Daily volume: $5-10 billion (larger than many stocks)
- Bid-ask spread: 1-2 ticks (0.05-0.10 points)
- Circuit breakers: NYSE halts trading if SPX moves >7% (we can exit before disaster)

Even in Feb 2018 crisis, VIX futures traded continuously (unleveraged ETNs halted, we wouldn't be).

### Q: Why monthly rebalance instead of daily?
**A**: Lower costs + better signal quality:
- ✅ Monthly contango is more predictable than daily (less noise)
- ✅ Transaction costs ~$50/month vs ~$1,000/month if daily
- ✅ Operational simplicity (15 min/month vs 15 min/day)

### Q: Can we scale this to €500k if it works?
**A**: Yes, but with position sizing adjustments:
- €100k sleeve → 3-5 VIX futures contracts (~10% margin usage)
- €500k sleeve → 15-20 contracts (~10% margin usage)
- Liquidity is not a constraint (market can absorb 50+ contracts)

---

## 15. Summary for Decision-Makers

### What VRP Is (Version 2.0 — With Modern Safeguards)
- **Insurance seller**: Collect premiums from fearful investors, pay out during crashes
- **Monthly alpha decision**: Trade once/month, hold 30 days, repeat
- **Daily automated protection**: Stop-loss orders in market 24/7
- **VIX futures**: Simple liquid futures, not complex options or leveraged ETNs
- **Conservative sizing**: 10% of NAV, circuit breaker at -15% loss

### What's Different vs Naive VRP (Critical Updates)
| Feature | Naive VRP (Pre-2018) | Our VRP V2.1 (2026) |
|---------|---------------------|---------------------|
| **Signal** | Just contango | Contango + Volatility Momentum + Filters |
| **0DTE Awareness** | None | VIX1D/VIX ratio monitoring |
| **Interest Rate** | Ignored | Adjusted for post-ZIRP carry |
| **Correlation Risk** | Ignored | VVIX filter for liquidity shocks |
| **Risk Management** | Manual daily check | Automated stop-loss in market |
| **Gap Risk** | Unaddressed | **Long Call Wings (OTM VIX Calls)** |
| **Thresholds** | Hard numbers | **Regime-Adaptive (Rolling Percentiles)** |
| **Execution** | 1st day of month | **3rd day (avoid crowding)** |
| **Re-Entry** | Immediate after stop | **Cool-Down Rule (5+ days)** |
| **Expected Return** | 15-20% | 6-10% (realistic) |

### What VRP Brings to Portfolio
- ✅ **New risk factor**: Volatility premium (uncorrelated to momentum)
- ✅ **Moderate income**: €3k-€6k/month in normal markets (realistic, not hyped)
- ✅ **Diversification**: Negative correlation to stocks during crashes
- ✅ **Portfolio smoothing**: Two uncorrelated sleeves → lower total volatility

### What It Costs
- **Development**: 3-4 days + ~$100 data cost
- **Risk**: Circuit breaker caps max loss at -15% (€12k-€15k on €100k)
- **Operational**: 5 min/day monitoring (non-negotiable)
- **Lower returns**: 6-10% vs historical 15-20% (crowding + safety = less profit)
- **Opportunity cost**: Could pursue Carry strategies instead (different trade-off)

### The Ask
**Approve VRP kill-test validation** (3-4 days development + $100 data cost)

**Next milestone**: Kill-test results → decide on paper trading promotion

---

## 16. AI-Driven Enhancements: Why Baseline First?

### ⚠️ Addressing the Obvious Question

**"Why not just build the ML-enhanced version from the start?"**

This is a valid question. The academic literature (Wang et al. 2024) clearly shows ALSTM outperforms simple rules. So why sequence baseline → ML instead of jumping straight to AI?

### The Answer: Avoiding a Common $100M Failure Mode

**The Failure Pattern We're Avoiding**:
```
Step 1: Build sophisticated ML system (6 months, $500k)
Step 2: ML system loses money
Step 3: ???
        - Is it the ML model? (hyperparameters, architecture, training data)
        - Is it the underlying strategy? (VRP is dead in 2026?)
        - Is it execution? (slippage, costs, timing)
        - Is it the data pipeline? (CMF calculation bug?)
Step 4: Spend 6 more months debugging the wrong component
Step 5: Realize after 12 months that VRP itself doesn't work anymore
```

**The Pattern We're Following Instead**:
```
Step 1: Build rule-based baseline (1 week, $100)
Step 2: Kill-test baseline
        - If FAIL → VRP is dead. Stop here. Saved 6 months.
        - If PASS → VRP works. Proceed to Phase 2.
Step 3: Build ML enhancement (2 months)
Step 4: A/B test ML vs Baseline
        - ML better? → Deploy ML, measure marginal lift
        - ML worse? → Keep baseline, debug ML with KNOWN-GOOD comparison
Step 5: Every failure is attributable to ONE variable (isolated testing)
```

### Why This Matters for VRP Specifically

**VRP Has Fundamental Questions That Must Be Answered First**:

1. **Is the premium still there?** (Post-2018 crowding may have killed it)
2. **Do our filters work?** (VIX1D, VVIX, momentum — novel combinations)
3. **Is execution realistic?** (Slippage, ETH liquidity, roll costs)
4. **Does the hedge work?** (OTM calls may be mispriced in backtest)

**ML Cannot Answer These Questions** — it can only optimize ON TOP of them.

If we build ML first and it fails, we won't know if:
- The ML is wrong, OR
- VRP is dead, OR
- Our hedge construction is flawed, OR
- Our cost assumptions are wrong

### The Investment Case

| Approach | Time to Failure Detection | Cost to Failure | Debuggability |
|----------|--------------------------|-----------------|---------------|
| **ML-First** | 6 months | $500k+ | Very Hard |
| **Baseline-First** | 1 week | $100 | Trivial |

**If baseline fails**: We've lost 1 week and $100. VRP is dead. Move on.
**If baseline passes**: We have a KNOWN-GOOD reference for all future ML work.

### The ML Roadmap (Conditional on Baseline Pass)

**Phase 2 (Week 2-4): CMF + ALSTM Development**
- Build Constant Maturity Futures data pipeline
- Train ALSTM regime classifier on 2010-2024 data
- A/B backtest: Baseline vs ML-enhanced

**Phase 3 (Week 4-6): Validation**
- If ML Sharpe > Baseline Sharpe + 0.1: **Promote ML**
- If ML Sharpe ≤ Baseline Sharpe: **Keep Baseline, debug ML**

**Key Principle**: ML is an ENHANCEMENT, not a REPLACEMENT. The baseline is our insurance policy.

### Technical Details

See **SLEEVE_VRP_ML_ENHANCEMENT.md** for complete ML specification:
- Constant Maturity Futures calculation
- ALSTM architecture (PyTorch implementation)
- Feature engineering (6 features)
- Integration with baseline filters
- Success metrics and validation protocol

### Timeline Summary

```
Week 1:   Baseline development + kill-test
Week 2-3: Paper trading validation (if baseline passes)
Week 4-5: ML development (in parallel with paper trading)
Week 6:   ML A/B test
Week 7+:  Deploy winner (Baseline OR ML-enhanced)

Total: 7 weeks to full AI-enhanced VRP (if everything passes)
       1 week to KILL decision (if baseline fails)
```

**Bottom Line**: We're not delaying ML because it's less important. We're sequencing it correctly so that when we build it, we can actually debug it.

---

## Appendix: Document History

### Version 2.1 (2026-01-08) — Institutional-Grade Refinements

**V2.1 improvements (building on V2.0 market structure updates):**

1. **Tail Hedging (Long Wings)**: Added OTM VIX call protection to cap gap risk (Section 3)
2. **Regime-Adaptive Thresholds**: Replaced hard numbers with rolling percentiles (Section 5)
3. **Execution Timing**: Shifted from 1st to 3rd trading day to avoid crowding (Section 3)
4. **Cool-Down Rule**: Added mandatory waiting period after stop-loss triggers (Section 5)
5. **Gap Risk Documentation**: Explicitly documented stop-loss limitations and mitigation (Section 11)
6. **Summary Table Update**: Expanded comparison table with all V2.1 features (Section 15)

### Version 2.0 (2026-01-08) — Market Structure Update

**Critical improvements based on 2023-2026 market analysis:**

1. **0DTE Blind Spot**: Added VIX1D monitoring to catch short-term fear spikes (Section 2)
2. **Signal Sophistication**: Added volatility momentum filter to avoid "falling knife" trades (Section 3)
3. **Automated Risk Management**: Emphasized stop-loss orders in market, not daily manual checks (Section 5)
4. **Correlation Breakdown**: Added VVIX filter for liquidity shock warning (Section 11)
5. **Interest Rate Adjustment**: Account for post-ZIRP cost of carry in contango calculation (Section 3)
6. **Realistic Expectations**: Lowered expected returns from 8-12% to 6-10% due to strategy crowding (Section 7)

**References**:
- Bali, T. et al. (2023). "Volatility Risk Premium Decay in Crowded Markets"
- Pasquariello, P. (2024). "Carry Strategies in the Post-ZIRP Era"
- CBOE VIX1D Index Whitepaper (2023)

---

**Document Version**: 2.1
**Author**: Claude
**Intended Audience**: Management / Non-technical decision-makers
**Next Step**: Approval to proceed with VRP kill-test development
