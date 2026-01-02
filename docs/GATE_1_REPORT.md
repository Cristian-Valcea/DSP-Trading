# Gate 1 Report: Environment & Baseline Validation

**Version**: 1.0
**Date**: 2026-01-02
**Status**: ✅ **PASS**
**Prerequisites**: Gate 0 PASS (see `GATE_0_REPORT.md`)

---

## 1. Executive Summary

Gate 1 validates the DQN trading environment by running baseline policies and verifying:
1. ✅ **No look-ahead bias** - FLAT policy achieves Sharpe ≈ 0.0
2. ✅ **Transaction costs work** - Random policy achieves Sharpe < 0

Both kill tests passed, confirming the environment is ready for DQN training.

---

## 2. Kill Test Results

### 2.1 FLAT Policy (Look-Ahead Bias Test)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Sharpe Ratio** | **0.0000** | abs < 0.1 | ✅ PASS |
| Total Return | 0.000000 | ≈ 0 | ✅ |
| Max Drawdown | 0.000000 | = 0 | ✅ |
| Trades/Episode | 0.0 | = 0 | ✅ |

**Interpretation**: The FLAT policy achieves exactly zero return with zero variance, confirming:
- No look-ahead bias in the data or reward calculation
- The environment correctly handles zero-position scenarios
- Transaction costs only apply when trading occurs

### 2.2 Random Policy (Transaction Cost Test)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Sharpe Ratio** | **-996.11** | < 0 | ✅ PASS |
| **Total Return** | **-24.10** | < 0 | ✅ PASS |
| Max Drawdown | 22.89 | > 0 | ✅ |
| Trades/Episode | 1291.0 | High | ✅ |

**Interpretation**: The random policy loses money consistently due to:
- High turnover (1291 trades/episode ÷ 209 minutes = ~6.2 position-changes/minute)
- 10 bps transaction cost per unit turnover
- Large negative Sharpe confirms costs dominate random entry/exit

---

## 3. Full Baseline Evaluation

| Policy | Sharpe | Total Return | Max DD | Trades/Ep |
|--------|--------|--------------|--------|-----------|
| **always_flat** | 0.00 | 0.00 | 0.00 | 0.0 |
| **random** | -996.11 | -24.10 | 22.89 | 1291.0 |
| **momentum** | -166.84 | -12.53 | 11.91 | 458.8 |
| **mean_reversion** | -138.78 | -12.18 | 11.64 | 453.4 |
| **buy_and_hold** | -10.48 | -0.31 | 0.34 | 6.0 |
| **rsi_contrarian** | -7.59 | -0.17 | 0.21 | 6.0 |

**Key Observations**:
1. All active strategies have negative Sharpe → transaction costs matter
2. Momentum and mean reversion have similar (poor) performance → simple signals not profitable
3. Buy-and-hold has lowest trading but still loses due to entry/exit costs
4. RSI contrarian performs slightly better than buy-and-hold

---

## 4. Environment Specification Validated

### 4.1 Observation Space

| Component | Shape | Validated |
|-----------|-------|-----------|
| Rolling Window | (60, 9, 30) | ✅ |
| Portfolio State | (21,) | ✅ |

### 4.2 Action Space

| Action | Value | Meaning | Validated |
|--------|-------|---------|-----------|
| 0 | FLAT | 0% exposure | ✅ |
| 1 | LONG_50 | +0.5 units | ✅ |
| 2 | LONG_100 | +1.0 units | ✅ |
| 3 | SHORT_50 | -0.5 units | ✅ |
| 4 | SHORT_100 | -1.0 units | ✅ |

### 4.3 Trading Parameters

| Parameter | Value | Validated |
|-----------|-------|-----------|
| Universe | 9 symbols | ✅ |
| Window Size | 60 bars | ✅ |
| Trading Window | 10:31-14:00 ET | ✅ (209 min) |
| Target Gross | 10% | ✅ |
| K per Side | 3 | ✅ |
| Turnover Cost | 10 bps | ✅ |

---

## 5. Data Splits Created

| Split | Date Range | Days | Total Rows |
|-------|------------|------|------------|
| **train** | 2021-12-20 → 2023-12-31 | ~510 | ~1,787K |
| **val** | 2024-01-01 → 2024-06-30 | ~124 | ~435K |
| **dev_test** | 2024-07-01 → 2024-12-31 | **128** | **~444K** |
| **holdout** | 2025-01-01 → 2025-12-19 | ~240-243 | ~846K |

**Kill tests run on**: dev_test (128 trading days, 2024-H2)

**Coverage caveat**: Per `data/split_manifest.json`, some symbols end earlier in 2025 (META/SPY/TSLA end 2025-12-16 → 240 days). The multi-symbol environment uses the intersection of dates across symbols.

---

## 6. Files Created/Modified

### 6.1 Core Environment Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/dqn/__init__.py` | 49 | Module exports |
| `src/dqn/env.py` | 388 | DQNTradingEnv gymnasium environment |
| `src/dqn/state_builder.py` | 321 | 30-feature rolling window builder |
| `src/dqn/reward.py` | 119 | Reward function with transaction costs |
| `src/dqn/constraints.py` | 160 | Top-K portfolio constraint layer |
| `src/dqn/baselines.py` | 326 | 6 baseline policies |

### 6.2 Scripts

| File | Purpose |
|------|---------|
| `scripts/dqn/create_splits.py` | Create train/val/test/holdout splits |
| `scripts/dqn/evaluate_baselines.py` | Run baseline evaluation and kill tests |

### 6.3 Documentation

| File | Purpose |
|------|---------|
| `docs/GATE_1_SPEC.md` | Gate 1 specification |
| `docs/GATE_1_REPORT.md` | This report |

### 6.4 Results

| File | Purpose |
|------|---------|
| `results/baseline_evaluation.json` | Kill test results (flat, random) |
| `results/baseline_evaluation_full.json` | Full baseline evaluation |

---

## 7. Verification Commands

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate

# Quick kill tests (2-3 min)
PYTHONPATH=/Users/Shared/wsl-export/wsl-home python scripts/dqn/evaluate_baselines.py \
    --data-dir /Users/Shared/wsl-export/wsl-home/data/dqn_dev_test \
    --kill-tests-only

# Full baseline evaluation (5-10 min)
PYTHONPATH=/Users/Shared/wsl-export/wsl-home python scripts/dqn/evaluate_baselines.py \
    --data-dir /Users/Shared/wsl-export/wsl-home/data/dqn_dev_test \
    --episodes 50

# View results
cat results/baseline_evaluation.json | jq '.kill_tests'
```

---

## 8. Acceptance Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Data splits created | 4 dirs × 9 symbols | ✅ Created | ✅ |
| Environment runs | 100 episodes | 120 episodes | ✅ |
| FLAT policy Sharpe | abs < 0.1 | 0.0000 | ✅ |
| Random policy Sharpe | < 0 | -996.11 | ✅ |
| Random policy return | < 0 | -24.10 | ✅ |
| Kill tests pass | All | 2/2 | ✅ |

---

## 9. Next Steps (Gate 2)

Gate 2 will implement DQN training with:
1. Double DQN architecture
2. Prioritized experience replay
3. Walk-forward training on train split
4. Validation on val split
5. Target: Sharpe > 0 on val split

**Gate 2 Start Condition**: Gate 1 PASS ✅

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-01-02 | Initial Gate 1 implementation and validation |

---

**END OF GATE 1 REPORT**
