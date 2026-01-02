# Gate 2.7 Plan: Training Mechanics Fix + Churn Control

**Date**: 2026-01-02
**Status**: PLANNING
**Goal**: Fix under-training bug and add proper churn control before concluding "no edge"

---

## Executive Summary

Gate 2.6 showed the DQN pipeline is *functionally correct*, but the training loop has severe under-training issues:

1. **Only ~400 gradient updates** in a 500-episode run (1 update per episode)
2. **Target network never updates** (target_update_freq=1000 but only ~400 train_steps)
3. **No switch margin** - model churns because any tiny Q-value difference triggers position change

We cannot conclude "no edge" until these issues are fixed and we run to 1-3M env steps.

---

## Root Cause Analysis

### Issue 1: Training Updates Per Episode

**Current Code** (`trainer.py:167-170`):
```python
# Train agent (after warmup)
if ep >= self.config.warmup_episodes:
    loss = self.agent.train_step()  # <-- ONE update per episode!
```

**Problem**:
- 500 episodes × 1 update = ~400 gradient updates (after 100 warmup)
- Typical DQN needs 100K-1M+ updates
- Model barely learns anything

**Fix**: Train N times per episode, where N scales with episode length:
```python
# Train multiple times per episode (N updates)
updates_per_episode = max(1, stats.num_steps // 4)  # ~50 updates for 200-step episode
for _ in range(updates_per_episode):
    loss = self.agent.train_step()
```

### Issue 2: Target Network Never Updates

**Current Code** (`agent.py:345-347`):
```python
# Soft update target network
if self.steps % self.target_update_freq == 0:  # target_update_freq=1000
    self.soft_update_target()
```

**Problem**:
- `self.steps` = train_steps = ~400 after 500 episodes
- `target_update_freq = 1000`
- Target net never updates! (400 % 1000 ≠ 0)

**Fix Options**:
1. Reduce `target_update_freq` to 100-200
2. Use soft update every step with τ=0.005 (already implemented, but never called)
3. Both: soft update every step + hard update every 1000

### Issue 3: No Switch Margin (Churn Control)

**Current behavior**: Model changes position whenever `Q(new) > Q(current)` by any amount.

**Problem**: Tiny Q-value noise causes position flip-flopping, burning transaction costs.

**Fix**: Add switch margin δ - only change position if:
```
Q(new) - Q(current) > δ
```

Where δ compensates for the round-trip cost of switching.

---

## Gate 2.7 Implementation Plan

### Phase 2.7a: Training Mechanics Fix

**Changes to `trainer.py`**:

```python
# In train() method, replace single train_step with:
if ep >= self.config.warmup_episodes:
    # Multiple gradient updates per episode
    updates_per_episode = self.config.updates_per_episode
    if updates_per_episode == "auto":
        updates_per_episode = max(1, stats.num_steps // 4)

    for update_idx in range(updates_per_episode):
        loss = self.agent.train_step()
        if loss is not None and self.writer and update_idx == 0:
            self.writer.add_scalar("train/loss", loss, self.total_steps)
```

**Changes to `agent.py`**:

```python
# Reduce target_update_freq from 1000 to 100
target_update_freq: int = 100,  # Was 1000

# OR: Update target every train step with soft update (already have tau=0.005)
def train_step(self):
    ...
    # Soft update target network EVERY step
    self.soft_update_target()  # Move outside the if-block
```

**New TrainingConfig parameters**:
```python
@dataclass
class TrainingConfig:
    ...
    updates_per_episode: int | str = "auto"  # "auto" = num_steps // 4
```

### Phase 2.7b: Switch Margin (Churn Control)

**Changes to `agent.py`**:

Add `switch_margin` parameter (distinct from `conviction_threshold`):

```python
def __init__(self, ..., switch_margin: float = 0.0, ...):
    self.switch_margin = switch_margin

def select_action(self, ..., current_positions: np.ndarray | None = None):
    """
    If current_positions provided, apply switch margin:
    Only change position if Q(new) - Q(current) > switch_margin
    """
    ...
    # After selecting greedy actions
    if current_positions is not None and self.switch_margin > 0:
        for i in range(self.num_symbols):
            current_action = self._position_to_action(current_positions[i])
            new_action = actions[i]
            if new_action != current_action:
                q_current = q_values[i, current_action]
                q_new = q_values[i, new_action]
                if q_new - q_current < self.switch_margin:
                    actions[i] = current_action  # Keep current position
```

**Rationale**:
- `conviction_threshold`: Filters trades where model isn't confident vs FLAT
- `switch_margin`: Filters trades where improvement over current position is marginal
- Both can coexist as complementary filters

### Phase 2.7c: Premarket Data Backfill (2021-2022)

**Scope**: Backfill premarket data for 9 symbols from 2021-12-20 to 2022-12-31

**Symbols**: AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA

**Data Requirements**:
- Premarket session: 04:00-09:30 ET
- Fields: premarket_return, premarket_volume, premarket_vol_ratio
- Source: Polygon.io aggregates API

**Implementation**:
1. Create `scripts/dqn/backfill_premarket.py`
2. Fetch premarket bars for each date/symbol
3. Compute premarket_return = (premarket_close - prior_close) / prior_close
4. Compute premarket_vol_ratio = premarket_volume / avg_premarket_volume_20d
5. Store in `data/dqn_premarket_cache/` (parquet files per symbol)

**Integration**:
- Update `env.py` to load premarket cache
- Pass to `state_builder.reset()` as `premarket_data` dict

### Phase 2.7d: Prior Close Tracking (Overnight Gap)

**Current Issue**: `overnight_gap` feature uses `first_open / first_close` which is wrong.

**Correct Calculation**:
```
overnight_gap = (today_open - yesterday_close) / yesterday_close
```

**Fix**:
1. Track `prior_close_by_symbol` in environment
2. Pass to `state_builder.reset()`
3. Already handled in `state_builder.py:226-229` if `prior_close` is provided

**Changes to `env.py`**:
```python
def reset(self):
    ...
    # Load prior trading day's close
    prior_close_by_symbol = self._get_prior_close(self.current_date)
    self.state_builder.reset(
        self.day_data,
        premarket_data=premarket_data,
        prior_close_by_symbol=prior_close_by_symbol,
    )
```

---

## Training Configuration (Gate 2.7)

```python
# Agent parameters
DQNAgent(
    # Existing
    num_symbols=9,
    k_per_side=3,
    epsilon_decay_steps=500_000,  # Increased from 100K

    # Fixed
    target_update_freq=100,  # Was 1000 (never triggered)

    # New
    switch_margin=0.002,  # ~2bps equivalent in Q-value space
)

# Training parameters
TrainingConfig(
    num_episodes=5000,  # More episodes
    warmup_episodes=100,
    updates_per_episode="auto",  # ~50 updates per 200-step episode
    eval_freq=500,
    eval_episodes=20,
    patience=20,  # More patience
)
```

**Expected env_steps**: ~5000 episodes × 200 steps = 1M env_steps
**Expected train_steps**: ~5000 episodes × 50 updates = 250K train_steps
**Expected target updates**: 250K / 100 = 2500 target updates

---

## Success Criteria

### Gate 2.7 Kill Test

| Metric | Target | Gate 2.6 | Notes |
|--------|--------|----------|-------|
| **Val Sharpe** | > 0 | -209.74 | Must be positive to pass |
| **Train Steps** | > 100K | ~400 | 250x increase expected |
| **Target Updates** | > 1000 | 0 | Actually updating now |
| **Epsilon Decay** | 1.0 → 0.05 | 0.997 → 0.005 | Should complete |

### Diagnostic Metrics

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| Loss trend | Decreasing | Flat or increasing |
| Q-values | Growing, then stabilizing | Constant or exploding |
| Entropy | Decreasing (policy sharpening) | Constant (not learning) |
| Action distribution | Concentrated | Uniform |

---

## Implementation Order

1. **Training mechanics** (Phase 2.7a) - Critical, do first
   - Multiple updates per episode
   - Fix target_update_freq

2. **Switch margin** (Phase 2.7b) - Important for cost control
   - Add switch_margin parameter
   - Wire through to action selection

3. **Premarket backfill** (Phase 2.7c) - Data completeness
   - Fetch 2021-2022 premarket data
   - Create premarket cache

4. **Prior close tracking** (Phase 2.7d) - Data correctness
   - Fix overnight_gap calculation
   - Track prior close per symbol

5. **Run Gate 2.7 training** - Full run to 1M+ env_steps

---

## Estimated Timeline

| Phase | Effort | Priority |
|-------|--------|----------|
| 2.7a Training mechanics | 1-2 hours | P0 |
| 2.7b Switch margin | 1 hour | P1 |
| 2.7c Premarket backfill | 2-3 hours | P1 |
| 2.7d Prior close | 30 min | P2 |
| Training run | 4-8 hours (GPU) | - |

**Total**: ~6 hours implementation + 4-8 hours training

---

## Decision Point

After Gate 2.7 completes:

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| Sharpe > 0.5 | Signal exists, model learns | Scale up, hyperparameter tune |
| Sharpe 0-0.5 | Weak signal, marginal | Consider lower costs or longer holding |
| Sharpe < 0 | No learnable signal | Pivot to Sleeve DM approach |

**Key insight**: Only after 250K+ train steps with proper target updates can we conclude whether the problem is learnable.

---

## Files to Modify

| File | Changes |
|------|---------|
| `dsp100k/src/dqn/trainer.py` | Multiple updates per episode |
| `dsp100k/src/dqn/agent.py` | target_update_freq, switch_margin |
| `dsp100k/src/dqn/env.py` | Prior close tracking, premarket loading |
| `dsp100k/scripts/dqn/backfill_premarket.py` | NEW - premarket data fetcher |
| `dsp100k/scripts/dqn/train.py` | Update config for Gate 2.7 |

---

## Appendix: Training Mechanics Math

**Gate 2.6 (broken)**:
- 500 episodes × 1 update/episode = 500 train_steps
- 500 train_steps / 1000 target_freq = 0 target updates
- Model: ~randomly initialized

**Gate 2.7 (fixed)**:
- 5000 episodes × 50 updates/episode = 250,000 train_steps
- 250,000 train_steps / 100 target_freq = 2,500 target updates
- Model: Actually learns from experience

This is a **500x increase** in gradient updates and **∞ increase** in target updates (0 → 2500).
