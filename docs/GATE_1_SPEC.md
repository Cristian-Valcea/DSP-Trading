# Gate 1 Specification: Environment & Baseline Validation

**Version**: 1.0
**Date**: 2026-01-02
**Prerequisites**: Gate 0 PASS (see `GATE_0_REPORT.md`)
**Reference**: [SPEC_DQN.md](SPEC_DQN.md), [PLAN_DQN.md](PLAN_DQN.md)

---

## 1. Objective

Build and validate the DQN trading environment with baseline policies to ensure:
1. No look-ahead bias (FLAT policy Sharpe ≈ 0)
2. Transaction costs are correctly applied (Random policy Sharpe < 0)
3. Environment correctly handles all edge cases

---

## 2. Data Artifacts Required

### 2.1 Dataset Split

| Dataset | Date Range | Purpose | Location |
|---------|------------|---------|----------|
| **train** | 2021-12-20 to 2023-12-31 | Training folds 1-2 | `data/dqn_train/` |
| **val** | 2024-01-01 to 2024-06-30 | Validation | `data/dqn_val/` |
| **dev_test** | 2024-07-01 to 2024-12-31 | Dev/debugging | `data/dqn_dev_test/` |
| **holdout** | 2025-01-01 to 2025-12-19 | Final test (DO NOT TOUCH) | `data/dqn_holdout/` |

### 2.2 Split Script Output

```bash
python scripts/dqn/create_splits.py --source data/stage1_raw --output data/

# Creates:
# data/dqn_train/{symbol}_train.parquet     (2021-12-20 to 2023-12-31)
# data/dqn_val/{symbol}_val.parquet         (2024-01-01 to 2024-06-30)
# data/dqn_dev_test/{symbol}_dev_test.parquet  (2024-07-01 to 2024-12-31)
# data/dqn_holdout/{symbol}_holdout.parquet    (2025-01-01 to 2025-12-19)
# data/split_manifest.json                     (metadata)
```

---

## 3. Environment Specification

### 3.1 DQNTradingEnv Interface

```python
class DQNTradingEnv(gym.Env):
    """
    Intraday DQN environment for multi-symbol trading.

    Observation Space:
        Dict with:
        - 'rolling_window': (window_size, num_symbols, 30)
        - 'portfolio_state': (21,) = 9 positions + 9 entry_returns + 3 aggregates

    Action Space:
        MultiDiscrete([5] * 9) - 5 actions per symbol
        After top-K constraint: effective (9,) array

    Reward:
        Sum of per-symbol (position × log_return - turnover_cost)
    """

    def __init__(
        self,
        data_dir: str,
        symbols: list[str] = None,  # Defaults to DQN_UNIVERSE
        window_size: int = 60,
        target_gross: float = 0.10,
        k_per_side: int = 3,
        turnover_cost: float = 0.0010,  # 10 bps
        start_minute: int = 61,   # 10:31 ET (minute 61 of 390 RTH)
        end_minute: int = 270,    # 14:00 ET (minute 270 of 390 RTH)
    ):
        ...

    def reset(self, seed=None, options=None):
        """Reset to start of a random (or specified) trading day."""
        ...

    def step(self, action):
        """Execute action, return (obs, reward, terminated, truncated, info)."""
        ...
```

### 3.2 Observation Components

#### Rolling Window (30 features per symbol per bar)

| # | Feature | Computation |
|---|---------|-------------|
| 1 | `log_return_1m` | log(close/prev_close) |
| 2 | `log_return_5m` | 5-bar cumulative |
| 3 | `log_return_15m` | 15-bar cumulative |
| 4 | `log_return_30m` | 30-bar cumulative |
| 5 | `volume_ratio` | volume / SMA(volume, 20) |
| 6 | `dollar_volume` | close × volume (scaled) |
| 7 | `bar_range_bps` | (high-low)/close × 10000 |
| 8 | `vwap_deviation` | (close - vwap) / vwap |
| 9 | `rsi_14` | 14-bar RSI |
| 10 | `ema_ratio_20_60` | EMA(20) / EMA(60) |
| 11 | `atr_14` | ATR(14) / close |
| 12 | `high_low_range` | (high-low) / close |
| 13 | `close_vs_high` | (close-low) / (high-low) |
| 14 | `time_sin` | sin(2π × minute / 390) |
| 15 | `time_cos` | cos(2π × minute / 390) |
| 16 | `day_of_week_sin` | sin(2π × dow / 5) |
| 17 | `day_of_week_cos` | cos(2π × dow / 5) |
| 18 | `overnight_gap` | (open - prev_close) / prev_close |
| 19 | `premarket_return` | Premarket cumulative (if avail, else 0) |
| 20 | `premarket_volume_ratio` | Premarket vol ratio (if avail, else 0) |
| 21 | `spy_return_1m` | SPY 1-min return |
| 22 | `spy_return_15m` | SPY 15-min return |
| 23 | `qqq_return_1m` | QQQ 1-min return |
| 24 | `qqq_return_15m` | QQQ 15-min return (new, replaces sector) |
| 25 | `realized_vol_5m` | std of 5 1-min returns |
| 26 | `realized_vol_15m` | std of 15 1-min returns |
| 27 | `return_vs_spy_1m` | symbol - SPY 1-min return |
| 28 | `return_vs_spy_15m` | symbol - SPY 15-min return |
| 29 | `return_vs_qqq_1m` | symbol - QQQ 1-min return |
| 30 | `return_vs_qqq_15m` | symbol - QQQ 15-min return |

#### Portfolio State (21 dims)

| Dim | Description | Range |
|-----|-------------|-------|
| 0-8 | Position per symbol | {-1, -0.5, 0, +0.5, +1} |
| 9-17 | Entry log-return per symbol | (-inf, +inf), 0 if flat |
| 18 | Gross exposure / G | [0, 1] typical |
| 19 | Net exposure | [-1, +1] typical |
| 20 | Daily P&L (scaled) | (-inf, +inf) |

### 3.3 Action Space

| Action | Meaning | Weight |
|--------|---------|--------|
| 0 | FLAT | 0% |
| 1 | LONG_50 | +0.83% |
| 2 | LONG_100 | +1.67% |
| 3 | SHORT_50 | -0.83% |
| 4 | SHORT_100 | -1.67% |

**Constraint Layer**: Top-K (default K=3) selection per side based on Q-value conviction.

### 3.4 Reward Function

```python
reward_t = sum over symbols:
    position_{t-1}[i] × log_return_t[i] - turnover_cost × |position_t[i] - position_{t-1}[i]|
```

- Uses PREVIOUS position to avoid look-ahead
- Transaction cost = 10 bps one-way

---

## 4. Baseline Policies

### 4.1 Always-FLAT (Critical Sanity Check)

```python
class FlatPolicy:
    """Always return action 0 (FLAT) for all symbols."""
    def act(self, obs):
        return np.zeros(9, dtype=int)
```

**Expected**: Sharpe ≈ 0.0 (±0.05)
**Purpose**: Detect look-ahead bias or data leakage

### 4.2 Random Policy

```python
class RandomPolicy:
    """Sample actions uniformly at random."""
    def act(self, obs):
        return np.random.randint(0, 5, size=9)
```

**Expected**: Sharpe ≈ -0.5 to -1.0
**Purpose**: Verify transaction costs dominate

### 4.3 Momentum Policy (Informational)

```python
class MomentumPolicy:
    """Go long if 5-min return > 0, short if < 0."""
    def act(self, obs):
        returns_5m = obs['rolling_window'][-1, :, 1]  # Feature index 1
        actions = np.where(returns_5m > 0, 2, np.where(returns_5m < 0, 4, 0))
        return actions
```

**Expected**: Sharpe ≈ -0.3 to +0.3
**Purpose**: Simple baseline reference

### 4.4 Mean Reversion Policy (Informational)

```python
class MeanReversionPolicy:
    """Go short if 5-min return > 0, long if < 0."""
    def act(self, obs):
        returns_5m = obs['rolling_window'][-1, :, 1]
        actions = np.where(returns_5m > 0, 4, np.where(returns_5m < 0, 2, 0))
        return actions
```

**Expected**: Sharpe ≈ -0.3 to +0.3
**Purpose**: Simple baseline reference

---

## 5. Kill Tests

### 5.1 Critical Tests (Must Pass)

| Test | Condition | Action if Fail |
|------|-----------|----------------|
| FLAT Sharpe | abs(sharpe) < 0.1 | **KILL** - Look-ahead bias detected |
| Random Sharpe | sharpe < 0 | **KILL** - Transaction costs not working |
| Random P&L | return < 0 | **KILL** - Costs not applied correctly |

### 5.2 Warning Tests (Investigate if Fail)

| Test | Condition | Action if Fail |
|------|-----------|----------------|
| Momentum Sharpe | sharpe ∈ [-0.5, 0.5] | Investigate - unexpected signal |
| Mean Rev Sharpe | sharpe ∈ [-0.5, 0.5] | Investigate - unexpected signal |

---

## 6. File Structure

```
dsp100k/
├── src/dqn/
│   ├── __init__.py
│   ├── env.py              # DQNTradingEnv
│   ├── state_builder.py    # Feature computation
│   ├── reward.py           # Reward calculation
│   ├── constraints.py      # Top-K constraint layer
│   └── baselines.py        # Baseline policies
├── scripts/dqn/
│   ├── create_splits.py    # Create train/val/test splits
│   └── evaluate_baselines.py  # Run baseline evaluation
├── tests/
│   └── test_dqn_env.py     # Environment unit tests
├── data/
│   ├── dqn_train/          # Training data
│   ├── dqn_val/            # Validation data
│   ├── dqn_dev_test/       # Dev test data
│   └── dqn_holdout/        # Holdout (DO NOT TOUCH)
└── results/
    └── baseline_evaluation.json
```

---

## 7. Commands

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate

# 1. Create train/val/test splits
python scripts/dqn/create_splits.py --source ../data/stage1_raw --output ../data

# 2. Run environment tests
pytest tests/test_dqn_env.py -v

# 3. Evaluate baseline policies
python scripts/dqn/evaluate_baselines.py \
    --data-dir ../data/dqn_dev_test \
    --output results/baseline_evaluation.json

# Expected output:
# {
#   "always_flat": {"sharpe": 0.02, "return": 0.0, "max_dd": 0.0},
#   "random": {"sharpe": -0.73, "return": -0.12, "max_dd": 0.08},
#   "momentum": {"sharpe": 0.15, "return": 0.03, "max_dd": 0.05},
#   "mean_reversion": {"sharpe": -0.10, "return": -0.02, "max_dd": 0.04}
# }
```

---

## 8. Acceptance Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| Data splits created | 4 directories with 9 symbols each | ⬜ |
| Environment runs without error | 100 episodes complete | ⬜ |
| FLAT policy Sharpe | abs(sharpe) < 0.1 | ⬜ |
| Random policy Sharpe | sharpe < 0 | ⬜ |
| Unit tests pass | 100% | ⬜ |
| Baseline evaluation saved | JSON with all 4 policies | ⬜ |

---

## 9. Estimated Timeline

| Task | Hours |
|------|-------|
| Create splits script | 2 |
| Implement state_builder.py | 6 |
| Implement env.py | 8 |
| Implement reward.py | 2 |
| Implement constraints.py | 2 |
| Implement baselines.py | 2 |
| Implement evaluate_baselines.py | 2 |
| Unit tests | 4 |
| Integration testing | 4 |
| **Total** | **32 hours** |

---

**END OF GATE 1 SPECIFICATION**
