# PLAN_DQN.md — Intraday DQN Implementation Plan

**Version**: 1.0
**Date**: 2026-01-02
**Reference**: [SPEC_DQN.md](SPEC_DQN.md)

---

## Overview

This document outlines the phased implementation plan for the Intraday DQN strategy. Each phase has explicit kill tests — if any phase fails its kill test, we stop and either pivot or kill the project.

**Timeline Estimate**: 4-6 weeks (assuming no major blockers)

**Gate 0 Result**: ✅ PASS — see `GATE_0_REPORT.md`

---

## Phase 0: Data Sanity (Gate 0)

**Duration**: 3-5 days
**Objective**: Verify all data is clean, split-adjusted, and complete

### 0.1 Tasks

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Verify stage1_raw split adjustment | P0 | 2 | Data |
| Audit GOOGL prices around July 2022 split | P0 | 1 | Data |
| Check for price outliers (>50% 1-min moves) | P0 | 2 | Data |
| Compute missing bar statistics per symbol | P0 | 2 | Data |
| Backfill premarket 2021-2022 (if needed) | P1 | 8-12 | Data |
| Backfill 2025 RTH + premarket | P1 | 4-6 | Data |
| Create unified data loader | P0 | 4 | Eng |

### 0.2 Deliverables

```
repo-root/
├── data/
│   ├── stage1_raw/                         # source (RTH 1-min bars)
│   ├── dqn_train/{symbol}_train.parquet
│   ├── dqn_val/{symbol}_val.parquet
│   ├── dqn_dev_test/{symbol}_dev_test.parquet
│   ├── dqn_holdout/{symbol}_holdout.parquet
│   └── split_manifest.json                 # ranges + per-symbol coverage
└── dsp100k/
    ├── data/
    │   └── data_quality_report.json         # Gate 0 output
    ├── scripts/
    │   ├── dqn/
    │   │   ├── verify_splits.py
    │   │   ├── audit_missing_bars.py
    │   │   ├── backfill_rth.py
    │   │   ├── create_splits.py
    │   │   └── evaluate_baselines.py
    │   └── sleeve_im/
    │       └── backfill_data.py             # premarket backfill (JSON cache)
    └── docs/
        └── GATE_0_REPORT.md
```

### 0.3 Kill Test

| Check | Threshold | Action |
|-------|-----------|--------|
| Missing RTH bars | < 1% per symbol-day | Pass |
| Split adjustment correct | No 20× discontinuity across GOOGL July 2022 split (pre-split should be ~$110 if adjusted, not ~$2200) | Pass |
| Price outliers | 0 unexplained >50% moves | Pass |
| Premarket coverage | > 95% trading days | Pass |

**Kill if**: Any P0 check fails and cannot be fixed within 2 days

### 0.4 Commands

```bash
# Verify split adjustment
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/dqn/verify_splits.py --symbol GOOGL --date 2022-07-14

# Audit missing bars
python scripts/dqn/audit_missing_bars.py --symbols all --output data/data_quality_report.json

# Backfill premarket (if needed)
python scripts/sleeve_im/backfill_data.py \
    --symbols AAPL,AMZN,GOOGL,META,MSFT,NVDA,QQQ,SPY,TSLA \
    --start 2021-12-20 \
    --end 2022-12-31

# Backfill missing RTH days (if needed)
python scripts/dqn/backfill_rth.py --symbols META --dry-run
python scripts/dqn/backfill_rth.py --symbols META
```

---

## Phase 1: Environment & Baseline (Gate 1)

**Duration**: 5-7 days
**Objective**: Build trading environment and validate with baseline policies

### 1.1 Tasks

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Implement DQNTradingEnv | P0 | 12 | Eng |
| Implement state builder (30 features) | P0 | 8 | Eng |
| Implement morning summary compression | P1 | 4 | Eng |
| Implement reward function | P0 | 4 | Eng |
| Implement top-K constraint layer | P0 | 4 | Eng |
| Create baseline policies (FLAT, Random, Momentum) | P0 | 4 | Eng |
| Run baseline policy evaluation | P0 | 2 | Eng |
| Document environment interface | P1 | 2 | Docs |

### 1.2 Deliverables

```
dsp100k/
├── src/dqn/
│   ├── __init__.py
│   ├── env.py              # DQNTradingEnv
│   ├── state_builder.py    # 30-feature state construction
│   ├── reward.py           # Reward function
│   ├── constraints.py      # Top-K portfolio constraint
│   └── baselines.py        # Baseline policies
└── results/
    └── baseline_evaluation.json
```

**Note**: `SPEC_DQN.md` includes a `morning_summary` component, but Gate 1 only requires `rolling_window` + `portfolio_state` for kill-test validation. Premarket/morning-summary wiring can be added in Gate 2.

### 1.3 Environment Interface

```python
class DQNTradingEnv(gym.Env):
    """
    Intraday DQN environment for multi-symbol trading.

    Observation Space:
        - rolling_window: (window_size, num_symbols, 30)
        - portfolio_state: (21,) positions + entry log-returns + {gross, net, daily_pnl}

    Action Space:
        - (num_symbols,) discrete actions (0-4) per symbol

    Reward:
        - Per-step log returns minus turnover cost
    """

    def __init__(
        self,
        data_dir: str,
        symbols: list[str],
        window_size: int = 60,
        target_gross: float = 0.10,
        k_per_side: int = 3,
        turnover_cost: float = 0.0010,
        start_minute: int = 61,   # 10:31 ET
        end_minute: int = 270,    # 14:00 ET
    ):
        ...

    def reset(self, date: Optional[str] = None) -> np.ndarray:
        """Reset to start of trading day."""
        ...

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute action and return (obs, reward, done, info).

        Args:
            action: (9,) array of actions per symbol

        Returns:
            obs: Next state
            reward: Portfolio reward
            done: True if end of day
            info: Dict with positions, P&L, etc.
        """
        ...
```

### 1.4 Kill Test

| Policy | Expected Sharpe | Tolerance | Result |
|--------|-----------------|-----------|--------|
| Always FLAT | 0.0 | ±0.05 | Must pass |
| Random | < 0 | — | Must pass |
| Momentum (5-min) | -0.3 to +0.3 | ±0.3 | Informational |
| Mean Reversion | -0.3 to +0.3 | ±0.3 | Informational |

**Kill if**:
- Always-FLAT Sharpe significantly ≠ 0 (indicates look-ahead bias)
- Random policy has Sharpe > 0 (indicates bug)

### 1.5 Commands

```bash
# Evaluate baseline policies
ROOT=/Users/Shared/wsl-export/wsl-home
cd "$ROOT/dsp100k"
source "$ROOT/venv/bin/activate"

PYTHONPATH="$ROOT" python scripts/dqn/evaluate_baselines.py \
    --data-dir "$ROOT/data/dqn_dev_test" \
    --kill-tests-only \
    --episodes 20 \
    --seed 42 \
    --output "$ROOT/dsp100k/results/baseline_evaluation.json"

# Expected output:
# {
#   "always_flat": {"sharpe": 0.02, "return": 0.0, "dd": 0.0},
#   "random": {"sharpe": -0.73, "return": -0.12, "dd": 0.08},
#   "momentum_5m": {"sharpe": 0.15, "return": 0.03, "dd": 0.05}
# }
```

---

## Phase 2: DQN Training (Gate 2)

**Duration**: 7-10 days
**Objective**: Train DQN and validate via walk-forward

### 2.1 Tasks

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Implement DQN model (shared trunk) | P0 | 6 | ML |
| Implement replay buffer | P0 | 4 | ML |
| Implement training loop | P0 | 6 | ML |
| Add symbol-specific heads (optional) | P2 | 4 | ML |
| Set up walk-forward validation | P0 | 4 | ML |
| Run 3 folds × 3 seeds training | P0 | 8 | ML |
| Analyze results and compute metrics | P0 | 4 | ML |
| Hyperparameter tuning (if promising) | P1 | 8 | ML |

### 2.2 Deliverables

```
dsp100k/
├── src/dqn/
│   ├── model.py            # DQN architecture
│   ├── replay_buffer.py    # Experience replay
│   ├── trainer.py          # Training loop
│   └── walk_forward.py     # Walk-forward validation
├── models/dqn/
│   ├── fold1_seed42/
│   │   ├── checkpoint_best.pt
│   │   └── training_log.json
│   ├── fold1_seed123/
│   ├── fold1_seed456/
│   └── ... (9 total)
└── results/
    ├── walk_forward_results.json
    └── GATE2_REPORT.md
```

### 2.3 Walk-Forward Configuration

```yaml
# config/walk_forward.yaml
folds:
  - name: fold1
    train_start: "2021-12-20"
    train_end: "2022-12-31"
    val_start: "2023-01-01"
    val_end: "2023-06-30"

  - name: fold2
    train_start: "2021-12-20"
    train_end: "2023-06-30"
    val_start: "2023-07-01"
    val_end: "2023-12-31"

  - name: fold3
    train_start: "2021-12-20"
    train_end: "2023-12-31"
    val_start: "2024-01-01"
    val_end: "2024-06-30"

dev_test:
  start: "2024-07-01"
  end: "2024-12-31"

seeds: [42, 123, 456]
```

### 2.4 Kill Test

| Metric | Threshold | Aggregation |
|--------|-----------|-------------|
| Sharpe Ratio | ≥ 0.2 | Mean across 9 runs |
| Max Drawdown | ≤ 15% | Any single run |
| Net Return | > 0% | Mean across 9 runs |
| Fold Consistency | ≥ 2/3 | Folds with positive Sharpe |

**Kill if**:
- Mean Sharpe < 0.2
- Any run has DD > 15%
- Mean return < 0%
- Fewer than 2/3 folds are profitable

### 2.5 Commands

```bash
# Train single fold
python scripts/dqn/train.py \
    --config config/walk_forward.yaml \
    --fold fold1 \
    --seed 42 \
    --output models/dqn/fold1_seed42

# Train all folds (parallel)
python scripts/dqn/train_all_folds.py \
    --config config/walk_forward.yaml \
    --output models/dqn/

# Evaluate on dev test
python scripts/dqn/evaluate_dev_test.py \
    --models models/dqn/ \
    --test-start 2024-07-01 \
    --test-end 2024-12-31 \
    --output results/walk_forward_results.json
```

### 2.6 Training Monitoring

```bash
# TensorBoard
tensorboard --logdir models/dqn/

# Key metrics to monitor:
# - loss/td_error
# - metrics/episode_return
# - metrics/sharpe_rolling_20d
# - exploration/epsilon
```

---

## Phase 3: Holdout Validation (Gate 3)

**Duration**: 2-3 days
**Objective**: Final validation on untouched 2025 data

### 3.1 Preconditions

- [ ] Gate 2 fully passed (all metrics met)
- [ ] No code changes after Gate 2 completion
- [ ] Model selection finalized (best fold/seed combo)

### 3.2 Tasks

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Select best model from Gate 2 | P0 | 1 | ML |
| Run inference on 2025 holdout | P0 | 2 | ML |
| Compute holdout metrics | P0 | 1 | ML |
| Generate final report | P0 | 2 | Docs |
| Decision: Deploy or Kill | P0 | 1 | Lead |

### 3.3 Deliverables

```
dsp100k/
├── results/
│   ├── holdout_2025_results.json
│   └── GATE3_FINAL_REPORT.md
└── models/dqn/
    └── production/
        └── best_model.pt
```

### 3.4 Kill Test

| Metric | Threshold | Action |
|--------|-----------|--------|
| Sharpe | ≥ 0.0 | Pass if ≥ 0, kill if < 0 |
| Net Return | > -5% | Pass if > -5%, kill if worse |
| Max Drawdown | ≤ 20% | Pass if ≤ 20%, kill if exceeded |

**Kill if**: Any metric fails threshold

### 3.5 Commands

```bash
# Run holdout evaluation (ONLY after Gate 2 passes)
python scripts/dqn/evaluate_holdout.py \
    --model models/dqn/production/best_model.pt \
    --data ../data/dqn_holdout \
    --output results/holdout_2025_results.json

# Generate final report
python scripts/dqn/generate_report.py \
    --gate2 results/walk_forward_results.json \
    --gate3 results/holdout_2025_results.json \
    --output results/GATE3_FINAL_REPORT.md
```

---

## Phase 4: Paper Trading (If Gates Pass)

**Duration**: 2-4 weeks
**Objective**: Validate live execution

### 4.1 Tasks

| Task | Priority | Est. Hours | Owner |
|------|----------|------------|-------|
| Integrate with DSP CLI | P0 | 8 | Eng |
| Implement order generation | P0 | 4 | Eng |
| Add real-time state builder | P0 | 6 | Eng |
| Set up monitoring dashboard | P1 | 4 | Eng |
| Run 2 weeks paper trading | P0 | — | Ops |
| Compare paper vs backtest | P0 | 4 | ML |

### 4.2 Paper Trading Checklist

- [ ] Model loads correctly in production environment
- [ ] State builder uses live Polygon data
- [ ] Orders generate at correct times (10:31-13:59 ET, force-flat at 14:00)
- [ ] Top-K constraint enforced
- [ ] Position sizing correct (w_max = 1.67%)
- [ ] Monitoring shows expected behavior

### 4.3 Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Execution fill rate | > 95% | Orders filled vs submitted |
| Slippage vs backtest | < 10 bps | Real vs assumed costs |
| P&L tracking accuracy | ±0.5% | Paper vs manual calc |
| System uptime | > 99% | No crashes during RTH |

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Data not split-adjusted | High | Medium | Verify in Gate 0 before proceeding |
| Insufficient sample size | High | Low | 1.9M transitions should be adequate |
| Overfitting to train period | High | Medium | Walk-forward + holdout prevents |
| Transaction costs exceed edge | High | Medium | Conservative 10 bps assumption |
| Model doesn't generalize to 2025 | High | Medium | Gate 3 holdout validation |
| Backfill takes too long | Medium | Medium | Parallelize, use existing sleeve_im data |
| Environment bugs (look-ahead) | High | Low | Gate 1 baseline sanity checks |

---

## Timeline Summary

```
Week 1: Phase 0 (Data Sanity)
        ├── Day 1-2: Verify splits, audit missing bars
        ├── Day 3-4: Backfill premarket if needed
        └── Day 5: Data quality report, Gate 0 decision

Week 2: Phase 1 (Environment & Baseline)
        ├── Day 1-3: Build DQNTradingEnv, state builder
        ├── Day 4-5: Implement baselines, run evaluation
        └── Day 5: Gate 1 decision

Week 3-4: Phase 2 (DQN Training)
        ├── Day 1-3: Implement DQN, replay buffer, trainer
        ├── Day 4-7: Train 3 folds × 3 seeds
        ├── Day 8-9: Analyze results, compute metrics
        └── Day 10: Gate 2 decision

Week 5: Phase 3 (Holdout Validation)
        ├── Day 1: Select best model
        ├── Day 2: Run 2025 holdout
        └── Day 3: Final report, Gate 3 decision

Week 6+: Phase 4 (Paper Trading)
        └── 2-4 weeks live validation
```

---

## Decision Tree

```
                    ┌──────────────┐
                    │  Gate 0      │
                    │ Data Sanity  │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         ✅ Pass                    ❌ Fail
              │                         │
              ▼                    Fix data or KILL
        ┌──────────────┐
        │  Gate 1      │
        │  Baselines   │
        └──────┬───────┘
               │
  ┌────────────┴────────────┐
  │                         │
✅ Pass                  ❌ Fail
  │                         │
  ▼                    Debug env or KILL
┌──────────────┐
│  Gate 2      │
│  Walk-Fwd    │
└──────┬───────┘
       │
┌──────┴──────────────────┐
│                         │
✅ Sharpe ≥ 0.2        ❌ Sharpe < 0.2
DD ≤ 15%               or DD > 15%
│                         │
▼                    KILL PROJECT
┌──────────────┐
│  Gate 3      │
│  Holdout     │
└──────┬───────┘
       │
┌──────┴──────────────────┐
│                         │
✅ Sharpe ≥ 0          ❌ Sharpe < 0
│                         │
▼                    KILL PROJECT
┌──────────────┐
│  Phase 4     │
│ Paper Trade  │
└──────────────┘
```

---

## Appendix: File Structure

### Current (Post Gate 1)

```
repo-root/
├── data/
│   ├── stage1_raw/                  # source (RTH 1-min bars)
│   ├── dqn_train/
│   ├── dqn_val/
│   ├── dqn_dev_test/
│   ├── dqn_holdout/
│   └── split_manifest.json
└── dsp100k/
    ├── data/
    │   ├── sleeve_im/minute_bars/    # premarket JSON cache (Polygon)
    │   └── data_quality_report.json  # Gate 0 output
    ├── docs/
    │   ├── SPEC_DQN.md
    │   ├── PLAN_DQN.md
    │   ├── GATE_0_REPORT.md
    │   ├── GATE_1_SPEC.md
    │   └── GATE_1_REPORT.md
    ├── results/
    │   ├── baseline_evaluation.json
    │   └── baseline_evaluation_full.json
    ├── scripts/dqn/
    │   ├── verify_splits.py
    │   ├── audit_missing_bars.py
    │   ├── backfill_rth.py
    │   ├── create_splits.py
    │   └── evaluate_baselines.py
    └── src/dqn/
        ├── __init__.py
        ├── env.py
        ├── state_builder.py
        ├── reward.py
        ├── constraints.py
        └── baselines.py
```

### Planned (Gate 2+)

```
dsp100k/
├── src/dqn/
│   ├── model.py
│   ├── replay_buffer.py
│   ├── trainer.py
│   └── walk_forward.py
└── scripts/dqn/
    ├── train.py
    ├── train_all_folds.py
    ├── evaluate_dev_test.py
    ├── evaluate_holdout.py
    └── generate_report.py
```

---

**END OF IMPLEMENTATION PLAN**
