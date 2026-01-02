# Gate 2 Specification: DQN Training Pipeline

**Version**: 1.0
**Date**: 2026-01-02
**Prerequisites**: Gate 1 PASS (environment validated with kill tests)
**Scope**: 2A (RTH-only data, premarket deferred to Gate 2.5)

---

## 1. Objective

Train a Double DQN agent on the validated trading environment to achieve:
- **Kill Test**: Sharpe > 0 on validation split (beat always-flat baseline)
- **Stretch Goal**: Sharpe > 0.5 on validation split

---

## 2. Architecture

### 2.1 Double DQN with Dueling Networks

```
Input: (batch, window=60, symbols=9, features=30) + (batch, portfolio=21)
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Market Encoder (per-symbol CNN → LSTM)                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Conv1D(30→64, k=3) → Conv1D(64→128, k=3) → LSTM(128)   │  │
│  │ Output: (batch, symbols=9, hidden=128)                 │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Cross-Symbol Attention                                      │
│  MultiHeadAttention(embed=128, heads=4)                      │
│  Output: (batch, symbols=9, hidden=128)                      │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Portfolio Integration                                       │
│  Concat(market_encoding, portfolio_state) → MLP(256→128)     │
│  Output: (batch, hidden=128)                                 │
└──────────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Dueling Heads (per symbol)                                  │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │ Value: MLP(128→1)   │  │ Advantage: MLP(128→5)       │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
│  Q(s,a) = V(s) + A(s,a) - mean(A(s,:))                      │
│  Output: (batch, symbols=9, actions=5)                       │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Top-K Constraint (Agent-Side)

**Critical**: Apply top-K using actual Q-values, NOT in environment.

```python
def select_action_with_topk(q_values: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Apply top-K constraint using Q-value conviction.

    Args:
        q_values: (num_symbols, num_actions) Q-values
        k: Max positions per side

    Returns:
        actions: (num_symbols,) selected actions after constraint
    """
    # Get greedy actions per symbol
    greedy_actions = np.argmax(q_values, axis=1)

    # Compute conviction = Q(greedy) - Q(flat)
    conviction = q_values[np.arange(len(greedy_actions)), greedy_actions] - q_values[:, 0]

    # Separate longs and shorts
    long_mask = greedy_actions >= 1  # LONG_50 or LONG_100
    short_mask = greedy_actions >= 3  # SHORT_50 or SHORT_100

    # Keep top-K longs by conviction
    long_indices = np.where(long_mask)[0]
    if len(long_indices) > k:
        long_convictions = conviction[long_indices]
        keep_longs = long_indices[np.argsort(long_convictions)[-k:]]
        for i in long_indices:
            if i not in keep_longs:
                greedy_actions[i] = 0  # Force FLAT

    # Keep top-K shorts by conviction
    short_indices = np.where(short_mask)[0]
    if len(short_indices) > k:
        short_convictions = conviction[short_indices]
        keep_shorts = short_indices[np.argsort(short_convictions)[-k:]]
        for i in short_indices:
            if i not in keep_shorts:
                greedy_actions[i] = 0  # Force FLAT

    return greedy_actions
```

---

## 3. Training Configuration

### 3.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 3e-4 | Standard for DQN |
| **Batch Size** | 256 | Balance memory/speed |
| **Replay Buffer Size** | 500,000 | ~2 days of transitions |
| **Target Update Freq** | 1,000 steps | Stability |
| **Gamma (discount)** | 0.99 | Long-horizon intraday |
| **Epsilon Start** | 1.0 | Full exploration |
| **Epsilon End** | 0.05 | Minimum exploration |
| **Epsilon Decay Steps** | 100,000 | ~0.5 days |
| **PER Alpha** | 0.6 | Prioritization strength |
| **PER Beta Start** | 0.4 | IS correction |
| **PER Beta End** | 1.0 | Full correction |
| **Gradient Clip** | 10.0 | Stability |

### 3.2 Training Schedule

| Phase | Episodes | Purpose |
|-------|----------|---------|
| **Warmup** | 1,000 | Fill replay buffer |
| **Training** | 50,000 | Main learning |
| **Evaluation** | Every 1,000 | Check val performance |

### 3.3 Early Stopping

Stop training if:
- Val Sharpe > 1.0 (strong signal found)
- Val Sharpe plateaus for 10 eval cycles
- Train loss diverges (NaN or > 100)

---

## 4. Data Pipeline

### 4.1 Splits (from Gate 1)

| Split | Date Range | Purpose | Touch? |
|-------|------------|---------|--------|
| **train** | 2021-12-20 → 2023-12-31 | Training | ✅ Yes |
| **val** | 2024-01-01 → 2024-06-30 | Validation | ✅ Yes |
| **dev_test** | 2024-07-01 → 2024-12-31 | Debugging | ⚠️ Sparingly |
| **holdout** | 2025-01-01 → 2025-12-19 | Final test | ❌ NO |

### 4.2 Episode Sampling

```python
# Training: Random date from train split
date = random.choice(train_dates)

# Validation: Sequential dates from val split
for date in val_dates:
    run_episode(date)
```

---

## 5. Prioritized Experience Replay

### 5.1 Priority Calculation

```python
# TD-error based priority
td_error = |r + γ * Q_target(s', argmax_a Q(s', a)) - Q(s, a)|
priority = (td_error + ε) ** α

# Where:
#   ε = 1e-6 (small constant)
#   α = 0.6 (prioritization strength)
```

### 5.2 Importance Sampling Correction

```python
# Correct for biased sampling
weight = (N * P(i)) ** (-β)
weight = weight / max(weights)  # Normalize

# β anneals from 0.4 → 1.0 over training
```

---

## 6. File Structure

```
dsp100k/
├── src/dqn/
│   ├── __init__.py      # Updated exports
│   ├── model.py         # NEW: Dueling DDQN network
│   ├── replay_buffer.py # NEW: PER implementation
│   ├── agent.py         # NEW: DQN agent with top-K
│   ├── trainer.py       # NEW: Training loop
│   ├── env.py           # MODIFIED: apply_constraint default=False
│   ├── state_builder.py
│   ├── reward.py
│   ├── constraints.py
│   └── baselines.py
├── scripts/dqn/
│   ├── train.py         # NEW: Training script
│   ├── evaluate.py      # NEW: Evaluation script
│   ├── create_splits.py
│   └── evaluate_baselines.py
├── checkpoints/         # NEW: Model checkpoints
│   └── .gitkeep
├── results/
│   ├── baseline_evaluation.json
│   └── training_metrics.json  # NEW
└── docs/
    ├── GATE_2_SPEC.md   # This file
    └── GATE_2_REPORT.md # To be created
```

---

## 7. Commands

```bash
ROOT=/Users/Shared/wsl-export/wsl-home
cd "$ROOT/dsp100k"
source "$ROOT/venv/bin/activate"

# 1. Train DQN agent
PYTHONPATH="$ROOT" python scripts/dqn/train.py \
    --train-dir "$ROOT/data/dqn_train" \
    --val-dir "$ROOT/data/dqn_val" \
    --checkpoint-dir checkpoints \
    --episodes 50000 \
    --eval-freq 1000

# 2. Evaluate best checkpoint
PYTHONPATH="$ROOT" python scripts/dqn/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir "$ROOT/data/dqn_val" \
    --episodes 100

# 3. (Optional) Debug on dev_test
PYTHONPATH="$ROOT" python scripts/dqn/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-dir "$ROOT/data/dqn_dev_test" \
    --episodes 50
```

---

## 8. Kill Tests

### 8.1 Must Pass

| Test | Condition | Interpretation |
|------|-----------|----------------|
| **Val Sharpe > 0** | `sharpe_val > 0` | Better than always-flat |
| **Val Return > 0** | `return_val > 0` | Net profitable |
| **No Collapse** | `action_entropy > 0.5` | Agent uses multiple actions |

### 8.2 Warning Flags

| Flag | Condition | Action |
|------|-----------|--------|
| **Overfit** | `sharpe_train > 2 * sharpe_val` | Increase regularization |
| **High Turnover** | `trades_per_ep > 200` | Add turnover penalty |
| **Position Bias** | `long_ratio > 0.8 or < 0.2` | Check data balance |

---

## 9. Acceptance Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| Model trains without error | 50k episodes | ⬜ |
| Val Sharpe > 0 | Kill test | ⬜ |
| Val Return > 0 | Kill test | ⬜ |
| Action entropy > 0.5 | Kill test | ⬜ |
| Checkpoint saved | `checkpoints/best_model.pt` | ⬜ |
| Training metrics logged | `results/training_metrics.json` | ⬜ |
| Gate 2 report created | `docs/GATE_2_REPORT.md` | ⬜ |

---

## 10. Estimated Timeline

| Task | Hours |
|------|-------|
| Implement model.py | 2 |
| Implement replay_buffer.py | 1.5 |
| Implement agent.py | 1.5 |
| Implement trainer.py | 2 |
| Create train.py script | 1 |
| Create evaluate.py script | 0.5 |
| Training run (50k episodes) | 4-8 (GPU dependent) |
| Evaluation and report | 1 |
| **Total** | **13-17 hours** |

---

## 11. Changelog

| Date | Change |
|------|--------|
| 2026-01-02 | Initial Gate 2 specification (scope 2A: RTH-only) |

---

**END OF GATE 2 SPECIFICATION**
