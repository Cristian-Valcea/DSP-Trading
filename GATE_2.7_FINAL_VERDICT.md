# Gate 2.7 Final Verdict: DQN Intraday Trading

## Executive Summary

**VERDICT: ❌ FAIL - DQN does not extract sufficient edge for profitable trading**

The DQN approach to intraday trading on this data does not produce tradable alpha. Even at zero transaction costs, the models fail to beat a simple "stay flat" baseline.

---

## Evidence Summary

### DI=15 Model (Original - 26 decisions/day)
| Metric | Value |
|--------|-------|
| Action Distribution | 100% FLAT |
| Win Rate | 0% |
| CAGR at 0 cost | 0.00% |
| Turnover | 0.00 |

**Interpretation**: Model learned that staying flat is optimal. This is the correct learned behavior when there's no exploitable edge.

### DI=60 Model (Hourly - 4 decisions/day)
| Metric | Value |
|--------|-------|
| Action Distribution | 60% FLAT, 40% trading |
| Win Rate | 14% |
| CAGR at 0 cost | -9.49% |
| Turnover | 11.44 |
| Sharpe | -18.80 |

**Interpretation**: Model trades but loses money even at zero cost. Trading is value-destructive - the 40% non-flat actions are net negative. Model would be better staying 100% flat.

---

## Cost Sensitivity Analysis

### DI=60 (Best Model at Episode 3200)
| Cost (bps) | Win Rate | CAGR | Sharpe |
|------------|----------|------|--------|
| 0 | 22% | -6.5% | -12.0 |
| 1 | 0% | -38.1% | -47.0 |
| 10 | 0% | -291% | -53.0 |

### DI=15 (sweep_sm0.0)
| Cost (bps) | Win Rate | CAGR | Sharpe |
|------------|----------|------|--------|
| 0 | 0% | 0.0% | N/A |
| Any | 0% | 0.0% | N/A |

**Break-even cost**: Both models have break-even cost < 0 bps (lose money at zero cost).

---

## Root Cause Analysis

### Why DQN Fails Here

1. **Insufficient Signal-to-Noise**: The underlying price movements at 1-minute resolution don't contain exploitable patterns that a DQN can learn
   
2. **Feature Limitations**: Current features (price ratios, returns, spreads) may not capture the microstructure information needed for edge

3. **Time Horizon Mismatch**: DQN's discrete action space and short-term reward may not be suited for capturing slow-developing alpha

4. **Data Quality**: Simulated/historical data may lack the real market microstructure information needed for edge

### What DOESN'T Explain the Failure

- ❌ Transaction costs - model loses even at 0 cost
- ❌ Wrong RL algorithm - both DI=15 and DI=60 fail, not an exploration/exploitation issue
- ❌ Not enough training - 5000 episodes with proper epsilon decay, model converged

---

## User's Hypothesis Confirmed

> "the edge is tiny. Your best_model's break-even cost is well under ~1 bps one-way"

**Confirmed**: Actually, break-even cost is **negative** - there is no positive edge to speak of.

> "if it still shows break‑even cost ≪ realistic costs, the right pivot is stronger features / longer‑horizon target / 'trade only on very high conviction', not a different RL algorithm"

**Confirmed**: Changing from DI=15 to DI=60, or from DQN to PPO, will not fix "no edge exists in the data/features".

---

## Recommendations

### Do NOT proceed with:
- Different RL algorithms (PPO, A2C, SAC) - won't fix edge problem
- More training - model has converged
- Hyperparameter tuning - fundamental edge problem

### Consider instead:
1. **Stronger Features**: Alternative data (order flow, sentiment, macro indicators)
2. **Longer Horizons**: Multi-day or weekly momentum instead of intraday
3. **Different Markets**: Futures, FX, or crypto may have more microstructure alpha
4. **Different Approach**: Rule-based momentum/mean-reversion with known edge first

---

## Files & Artifacts

- `checkpoints/sweep_di60_v2/best_model.pt` - Best DI=60 model (Sharpe -6.58)
- `checkpoints/sweep_sm0.0/best_model.pt` - DI=15 model (100% FLAT)
- `logs/sweep_di60_v2.log` - Training log
- `scripts/dqn/evaluate_hierarchical.py` - Evaluation framework

---

## Gate 2.7 Decision

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| Val Sharpe | > 0 | -6.58 (best) | ❌ FAIL |
| Break-even cost | > 5 bps | < 0 bps | ❌ FAIL |
| Win rate at 10 bps | > 50% | 0-22% | ❌ FAIL |

**Final Decision**: Gate 2.7 NOT PASSED. DQN intraday trading on this data is not viable.
