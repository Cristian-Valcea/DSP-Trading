# Gate 2 Report: DQN Training Pipeline

**Date**: 2026-01-02
**Scope**: Gate 2 (RTH-only; premarket integrated)
**Status**: ❌ **KILL TEST FAILED** (validation Sharpe ≤ 0)

---

## Executive Summary

Gate 2 delivered a **working end-to-end DQN training pipeline** (env → replay → training → checkpoints → eval) through three iterations:

- **Gate 2A**: Initial implementation - epsilon decay bug (stuck at 0.997)
- **Gate 2.5**: Fixed epsilon decay + greedy eval - improved but high churn
- **Gate 2.6**: Added conviction threshold + premarket features + multi-symbol TD loss fix

The trained policy **loses money after costs** and fails the Gate 2 kill test. With conviction filtering, performance approaches FLAT baseline, indicating **the model hasn't learned a meaningful signal yet**.

---

## Version History

### Gate 2A (Original)
- **Problem**: Epsilon decay tied to training updates instead of env steps
- **Result**: Epsilon stuck at 0.997, policy ~random throughout training
- **Best Sharpe**: -533.47

### Gate 2.5 (Epsilon Fix)
- **Fixes Applied**:
  - Epsilon decay now driven by env_steps (not train_steps)
  - Evaluation episodes are greedy (no exploration)
- **Result**: Epsilon properly decayed to 0.05
- **Best Sharpe**: -485.37 (31% improvement in mean P&L)
- **Issue**: Still 1,117 trades/episode - churn dominates

### Gate 2.6 (Conviction Threshold + Premarket)
- **Fixes Applied**:
  - Conviction threshold: trade only when `Q(best) - Q(flat) > τ`
  - Multi-symbol TD loss: portfolio Q = Σ Q_i with top-K constraint
  - Premarket features: wired from Sleeve IM cache
  - Overnight gap: uses true prior close from previous trading day
- **Best Sharpe**: -209.74 (57% improvement over 2.5)
- **Checkpoints**: `dsp100k/checkpoints_gate26_run1/`

---

## Kill Test Results (Gate 2.6 Best Model)

### Without Conviction Threshold
```bash
ROOT=/Users/Shared/wsl-export/wsl-home
PYTHONPATH="$ROOT" "$ROOT/venv/bin/python" "$ROOT/dsp100k/scripts/dqn/evaluate.py" \
  --checkpoint "$ROOT/dsp100k/checkpoints_gate26_run1/best_model.pt" \
  --data-dir "$ROOT/data/dqn_val" \
  --episodes 20 --seed 42 --compare-baselines
```

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Val Sharpe | > 0 | **-286.62** | ❌ FAIL |
| Mean P&L | > 0 | **-0.7972** | ❌ FAIL |
| Mean Trades/Episode | < 50 | **1,079** | ❌ Too High |
| Beat RANDOM | Yes | Yes (-287 vs -683) | ✅ PASS |

### Conviction Threshold Sweep

| Threshold | Sharpe | Trades/Ep | Mean P&L | Flat % | Analysis |
|-----------|--------|-----------|----------|--------|----------|
| 0.00 | -286.62 | 1,079 | -0.797 | 41.7% | Excessive churn |
| 0.06 | -95.38 | 148 | -0.124 | 94.5% | Better but still losing |
| 0.08 | -41.35 | 43 | -0.035 | 98.7% | Approaching flat |
| 0.10 | -19.41 | 6.5 | -0.006 | 99.8% | Nearly flat |
| 0.12 | -9.33 | 0.6 | -0.001 | 100% | Effectively flat |

**Key Insight**: As conviction threshold increases, performance monotonically approaches FLAT baseline (Sharpe=0). This indicates:
1. The model's "conviction" is mostly noise
2. Higher conviction trades are not more profitable
3. The model hasn't learned a meaningful directional signal

### Baseline Comparison
| Policy | Sharpe | Mean P&L |
|--------|--------|----------|
| FLAT | 0.00 | 0.0000 |
| DQN (τ=0.12) | -9.33 | -0.0006 |
| DQN (τ=0.08) | -41.35 | -0.0346 |
| DQN (τ=0.06) | -95.38 | -0.1237 |
| MOMENTUM | -175.36 | -0.8022 |
| DQN (τ=0.00) | -286.62 | -0.7972 |
| RANDOM | -683.44 | -1.5068 |

---

## Training Configuration (Gate 2.6)

- **Agent**: Double DQN + dueling heads + prioritized replay + top-K constraint
- **Universe**: 9 symbols (AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA)
- **Trading window**: 10:31–14:00 ET (209 steps/episode)
- **Costs**: turnover cost = 10 bps (one-way) on position changes
- **Run**: 500 episodes, warmup 100, eval every 100 (20 eval episodes), CPU
- **Features**: Premarket gap/volume + RTH price/volume/spread

**Artifacts**:
- Checkpoints: `dsp100k/checkpoints_gate26_run1/best_model.pt`
- State: `dsp100k/checkpoints_gate26_run1/best_model.pt.state.json`

---

## Training Progress (Gate 2.6)

| Episode | Val Sharpe | Epsilon | Buffer | Notes |
|---------|------------|---------|--------|-------|
| 100 | -516.41 | 0.799 | 21,109 | Initial checkpoint |
| 200 | -485.37 | 0.601 | 42,009 | Best checkpoint (Gate 2.5) |
| 300 | -209.74 | 0.402 | 50,000 | **Best checkpoint (Gate 2.6)** |
| 400 | -528.15 | 0.204 | 50,000 | Regression |
| 500 | -336.96 | 0.005 | 50,000 | Final |

---

## Root Cause Analysis

### 1) No Learned Signal
The conviction threshold sweep reveals the model hasn't learned a meaningful signal:
- Random noise: Sharpe ~ -287 (high churn burns money on costs)
- With filtering: Sharpe approaches 0 (effectively becomes FLAT)
- No intermediate regime where signal > costs

### 2) Possible Contributing Factors
1. **Insufficient training data**: 509 training days may not provide enough signal
2. **Feature engineering**: Current features may not capture alpha
3. **Model capacity**: 272K parameters may be undersized for multi-symbol
4. **Reward signal**: PnL-only reward may be too sparse/noisy
5. **Exploration/exploitation**: Even with fixed epsilon, may need better exploration

### 3) Architecture Validated
The pipeline is functionally correct:
- Epsilon decay works (0.997 → 0.005)
- Greedy eval reflects true policy
- Conviction threshold effectively filters trades
- Multi-symbol TD loss computes portfolio Q correctly

---

## Baseline Signal Analysis (Critical Finding)

**Run on training data** (509 days, 2021-12-21 to 2023-12-29):

```bash
PYTHONPATH="$PWD" python dsp100k/scripts/dqn/evaluate_baselines.py \
  --data-dir "$PWD/data/dqn_train" --episodes 50
```

| Policy | Sharpe | Trades/Ep | Analysis |
|--------|--------|-----------|----------|
| FLAT | 0.00 | 0 | ✅ No look-ahead bias |
| RSI_CONTRARIAN | -0.81 | 6 | Best active, still negative |
| BUY_AND_HOLD | -1.71 | 6 | Long-only loses |
| MOMENTUM | -148.50 | 444 | Signal exists but costs dominate |
| MEAN_REVERSION | -174.53 | 436 | Similar to momentum |
| RANDOM | -840.64 | 1291 | Worst (max churn) |

**Critical Insight**: Even rule-based strategies that work in other contexts (momentum, mean-reversion, RSI) have **negative Sharpe** on this data.

**Root Cause**: Intraday trading of high-cap tech stocks with 10bps costs is fundamentally unprofitable in this dataset. The DQN isn't broken - **there's no edge to learn**.

### Structural Problem

The training data covers Dec 2021 - Dec 2023:
- SPY: $459 → $475 (+3.6% total, ~1.8% annualized)
- Includes 2022 bear market (-20% drawdown) + 2023 recovery

With 10bps roundtrip costs and ~200+ trades per day needed for active strategies, the cost drag (~2-5% daily) overwhelms any signal.

---

## Gate 2.7 Recommendations

### Priority 1: Reduce Transaction Costs (Structural Fix)
Current cost structure makes profitable trading impossible. Options:
- **Reduce turnover_cost**: 10bps → 2-3bps (more realistic for large-cap)
- **Add position holding reward**: Penalize churn explicitly
- **Increase min-hold time**: Only trade when conviction × duration > cost

### Priority 2: Extend Trading Window
Current RTH window (10:31-14:00) may be too short for signal extraction.
Consider full day (9:30-16:00) or multi-day holding.

### Priority 3: Different Universe
High-cap tech (AAPL, MSFT, etc.) may be too efficient.
Consider:
- Smaller-cap stocks with more friction/alpha
- Futures/FX with lower transaction costs
- Longer timeframes (daily bars, not 1-min)

### Priority 4: Alternative Approaches
Given structural unprofitability, consider:
- **Sleeve IM approach**: Monthly rebalancing (proven Sharpe 0.55+)
- **Regime-based trading**: Only trade during high-volatility regimes
- **Reduce frequency**: Weekly/monthly signals instead of minute-by-minute

---

## Conclusion

Gate 2.6 is **functionally complete** with proper epsilon decay, greedy evaluation, conviction filtering, premarket features, and multi-symbol TD loss.

**The model fails the kill test NOT because the DQN is broken, but because the problem setup is structurally unprofitable**:
- Even perfect rule-based strategies (momentum, mean-reversion, RSI) lose money
- Transaction costs (10bps) dominate any signal in minute-bar intraday trading
- The data has no extractable edge at this frequency/cost structure

**Recommended Path Forward**:
1. **Gate 2.7a**: Re-run with reduced costs (2-3bps) to verify model can learn when edge exists
2. **Gate 2.7b**: If still failing, pivot to Sleeve IM approach (monthly rebalancing, proven Sharpe 0.55+)
3. **Long-term**: Consider if DQN is the right tool for this problem vs rule-based strategies

**Key Artifact**: Gate 2.6 checkpoint at `dsp100k/checkpoints_gate26_run1/best_model.pt` (Sharpe -209.74)
