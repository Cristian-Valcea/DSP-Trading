# Gate 2.7 Final Verdict: DQN Intraday Trading

## Executive Summary

**VERDICT: ❌ FAIL - DQN does not extract sufficient edge for profitable trading**

The DQN approach to intraday trading on this data does not produce tradable alpha. At zero transaction costs, models show marginal/negative CAGR. At realistic costs (10 bps), all models lose significantly.

---

## Evidence Summary (Corrected)

### Cost Sensitivity Analysis

Evaluated on all dates in each split (VAL and DEV):

#### DI=60 Model (`checkpoints/sweep_di60_v2/best_model.pt`)
| Cost (bps) | CAGR (VAL) | CAGR (DEV) |
|------------|------------|------------|
| 0          | -0.86%     | +0.01%     |
| 10         | -7.85%     | -7.09%     |

#### DI=15 Model (`checkpoints/sweep_di15_fixed_v3/best_model.pt`)
| Cost (bps) | CAGR (VAL) | CAGR (DEV) |
|------------|------------|------------|
| 0          | -0.33%     | +0.18%     |
| 10         | -4.85%     | -4.99%     |

**Key Observations**:
- At 0 cost: Both models hover around break-even (±0.3-0.9% CAGR)
- At 10 bps: Both models lose ~5-8% annually
- Break-even cost is approximately 0-1 bps (far below realistic 10 bps)

---

## Interpretation

### What the Numbers Mean

1. **Near-zero edge at 0 cost**: The models extract essentially no alpha from the data. DEV shows slight positive, VAL shows slight negative - consistent with random noise around zero.

2. **Cost-dominated at realistic levels**: At 10 bps one-way cost, both models lose 5-8% annually. The turnover required to express any signal is too expensive.

3. **DI=15 vs DI=60**: DI=15 is slightly less bad at 10 bps (-4.85% vs -7.85%) due to lower turnover, but neither is viable.

---

## Root Cause Analysis

### Why DQN Fails Here

1. **Insufficient Signal-to-Noise**: The underlying price movements at 1-minute resolution don't contain exploitable patterns that a DQN can learn

2. **Feature Limitations**: Current features (price ratios, returns, spreads) may not capture the microstructure information needed for edge

3. **Edge << Costs**: Even if there's a tiny learnable structure, the edge per unit turnover is well under 1 bps - any realistic execution cost erases it

### What DOESN'T Explain the Failure

- ❌ Wrong RL algorithm - the edge is ~0, switching to PPO won't help
- ❌ Not enough training - models converged
- ❌ Hyperparameter tuning - fundamental signal problem

---

## User's Hypothesis Confirmed

> "the edge is tiny. Your best_model's break-even cost is well under ~1 bps one-way"

**Confirmed**: Break-even cost is approximately 0-1 bps. At realistic 10 bps costs, models lose 5-8% annually.

> "if it still shows break‑even cost ≪ realistic costs, the right pivot is stronger features / longer‑horizon target / 'trade only on very high conviction', not a different RL algorithm"

**Confirmed**: Changing from DI=15 to DI=60, or from DQN to PPO, will not fix "edge too small vs costs".

---

## Recommendations

### Do NOT proceed with:
- Different RL algorithms (PPO, A2C, SAC) - won't fix edge problem
- More training - models have converged
- Hyperparameter tuning - fundamental edge problem

### Consider instead:
1. **Stronger Features**: Alternative data (order flow, sentiment, macro indicators)
2. **Longer Horizons**: Multi-day or weekly momentum instead of intraday
3. **Different Markets**: Futures, FX, or crypto may have more microstructure alpha
4. **Different Approach**: Rule-based momentum/mean-reversion with known edge first

---

## Files & Artifacts

- `checkpoints/sweep_di60_v2/best_model.pt` - Best DI=60 model (episode 3200)
- `checkpoints/sweep_di15_fixed_v3/best_model.pt` - Best DI=15 model
- `logs/sweep_di60_v2.log` - DI=60 training log
- `scripts/dqn/evaluate_hierarchical.py` - Evaluation framework

---

## Gate 2.7 Decision

| Criterion | Threshold | DI=15 Result | DI=60 Result | Status |
|-----------|-----------|--------------|--------------|--------|
| CAGR @ 0 cost | > 0% | -0.33% (VAL) | -0.86% (VAL) | ❌ FAIL |
| CAGR @ 10 bps | > 0% | -4.85% (VAL) | -7.85% (VAL) | ❌ FAIL |
| Break-even cost | > 5 bps | ~0-1 bps | ~0-1 bps | ❌ FAIL |

**Final Decision**: Gate 2.7 NOT PASSED. DQN intraday trading on this data is not viable at realistic transaction costs.
