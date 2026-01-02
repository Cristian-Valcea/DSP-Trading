# SPEC_DQN.md — Intraday DQN Technical Specification

**Version**: 1.0
**Date**: 2026-01-02
**Status**: Draft — Gate 1 PASS (see `GATE_1_REPORT.md`) — Ready for Gate 2

---

## 1. Executive Summary

This document specifies an intraday Deep Q-Network (DQN) strategy for the DSP-100K sleeve. The model makes 1-minute decisions from 10:31–13:59 ET and forces flat at 14:00 ET across 9 liquid symbols, selecting the top-K highest-conviction positions per side while respecting a gross exposure budget.

**Key Parameters**:
- **G** = 10% target gross exposure (of sleeve NAV)
- **K** = 3 positions per side (default)
- **w_max** = G / (2K) = 1.67% max weight per symbol

---

## 2. Universe

### 2.1 Symbols (9 Total)

| Symbol | Sector | Avg Daily Volume | Notes |
|--------|--------|------------------|-------|
| AAPL | Tech | ~80M | Mega-cap, highly liquid |
| AMZN | Consumer | ~40M | Mega-cap |
| GOOGL | Tech | ~25M | 20:1 split July 2022 — verify adjusted |
| META | Tech | ~20M | Ticker changed from FB Oct 2021 |
| MSFT | Tech | ~25M | Mega-cap |
| NVDA | Tech | ~50M | High volatility, GPU leader |
| QQQ | ETF | ~50M | Nasdaq-100 ETF, benchmark |
| SPY | ETF | ~80M | S&P 500 ETF, market proxy |
| TSLA | Consumer | ~100M | High volatility |

### 2.2 Why These 9?

1. **Liquidity**: All have >$1B daily dollar volume → minimal market impact
2. **Spread**: Tight bid-ask (1-3 bps) → low transaction costs
3. **Coverage**: Mix of tech, consumer, ETFs for diversification
4. **Existing Data**: Premarket data already backfilled for 2023-2024

---

## 3. Trading Window

### 3.1 Decision Cadence

| Time (ET) | Activity |
|-----------|----------|
| 04:00-09:30 | **Feature Window** — Aggregate premarket summary |
| 09:30-10:30 | **Observation Only** — Collect first-hour RTH data |
| 10:31-13:59 | **Active Trading** — DQN decisions every 1 minute |
| 14:00 | **Mandatory Flat** — Force flatten and end-of-day |

### 3.2 Why 10:31 Start?

1. **Avoid Opening Volatility**: First 60 minutes have widest spreads, most noise
2. **Morning Context**: One full hour of RTH data to condition the model
3. **Feature Stability**: Technical indicators (RSI, VWAP) need warm-up period

### 3.3 Why 14:00 Stop?

1. **Avoid Late-Day Risk**: Skip the last 2 hours of RTH (close-auction dynamics, news risk)
2. **Simpler Execution**: No MOC/MOO logic; hard stop at 14:00 is easier to enforce
3. **Cleaner Learning Target**: Fixed intraday horizon reduces label/exit ambiguity

---

## 4. State Representation

### 4.1 State Tensor Shape

```
State = [rolling_window, morning_summary, portfolio_state]

Dimensions:
- rolling_window: (N, num_symbols, F) where N ∈ {60, 120}, num_symbols = 9, F = 30 features
- morning_summary: (num_symbols, M) compressed embedding of premarket/first-hour (optional; see below)
- portfolio_state: (2 × num_symbols + 3,) = (21,) for positions, per-symbol entry log-returns, exposures, P&L
```

**Design choice**: `rolling_window` contains **market-derived features only** (price/volume/time/cross-asset). All portfolio/context features live in `portfolio_state` (current snapshot).

**Implementation note (Gate 1)**: The environment observation currently contains `rolling_window` and `portfolio_state`. `morning_summary` is specified here for the final model, but is not yet wired into `DQNTradingEnv` (it can be added in Gate 2 when premarket integration lands).

### 4.2 Rolling Window Features (30 columns per bar)

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 1 | `log_return_1m` | log(close/prev_close) | Price |
| 2 | `log_return_5m` | 5-bar cumulative return | Price |
| 3 | `log_return_15m` | 15-bar cumulative return | Price |
| 4 | `volume_ratio` | volume / 20-bar SMA(volume) | Volume |
| 5 | `dollar_volume` | close × volume (scaled) | Volume |
| 6 | `bar_range_bps` | (high - low) / close × 10000 | Microstructure proxy |
| 7 | `vwap_deviation` | (close - rolling_vwap) / rolling_vwap | Price |
| 8 | `rsi_14` | 14-bar RSI | Technical |
| 9 | `ema_ratio_20_60` | EMA(20) / EMA(60) | Technical |
| 10 | `atr_14` | 14-bar ATR / close | Volatility |
| 11 | `high_low_range` | (high - low) / close | Volatility |
| 12 | `close_vs_high` | (close - low) / (high - low) | Price position |
| 13 | `time_sin` | sin(2π × minute / 390) | Time encoding |
| 14 | `time_cos` | cos(2π × minute / 390) | Time encoding |
| 15 | `day_of_week_sin` | sin(2π × dow / 5) | Calendar |
| 16 | `day_of_week_cos` | cos(2π × dow / 5) | Calendar |
| 17 | `overnight_gap` | (open - prev_close) / prev_close | Gap |
| 18 | `premarket_return` | Premarket price change | Extended hours |
| 19 | `premarket_volume_ratio` | Premarket vol / 20d avg | Extended hours |
| 20 | `spy_return_1m` | SPY 1-min return | Cross-asset |
| 21 | `spy_return_15m` | SPY 15-min return | Cross-asset |
| 22 | `qqq_return_1m` | QQQ 1-min return | Cross-asset |
| 23 | `sector_momentum` | Sector ETF return | Cross-asset |
| 24 | `log_return_30m` | 30-bar cumulative return | Price |
| 25 | `realized_vol_5m` | Std of 1-min returns (last 5 bars) | Volatility |
| 26 | `realized_vol_15m` | Std of 1-min returns (last 15 bars) | Volatility |
| 27 | `return_vs_spy_1m` | symbol 1m return − SPY 1m return | Cross-asset |
| 28 | `return_vs_spy_15m` | symbol 15m return − SPY 15m return | Cross-asset |
| 29 | `return_vs_qqq_1m` | symbol 1m return − QQQ 1m return | Cross-asset |
| 30 | `return_vs_qqq_15m` | symbol 15m return − QQQ 15m return | Cross-asset |

**Symbol identity** is provided to the network via `symbol_id` conditioning (Section 6); it is not part of the 30 rolling features.

### 4.3 Morning Summary Embedding

Compress premarket + first-hour (04:00-10:30) into fixed-size vector:

```python
morning_summary = {
    'premarket_return': float,      # 04:00 → 09:30 return
    'premarket_volatility': float,  # Std of 1-min returns
    'premarket_volume_zscore': float,  # vs 20d avg
    'first_hour_return': float,     # 09:30 → 10:30 return
    'first_hour_volatility': float,
    'overnight_gap': float,
    'vwap_to_open': float,          # Premarket VWAP vs open
}
# Concatenate → (7,) vector per symbol, or embed via small MLP
```

### 4.4 Portfolio State

```python
portfolio_state = {
    'positions': np.array of shape (9,),  # {-1, -0.5, 0, +0.5, +1} per symbol
    # Per-symbol log-return since entry (0 if FLAT):
    # entry_log_return[i] = 0 if positions[i] == 0 else log(current_price[i] / entry_price[i])
    'entry_log_return': np.array of shape (9,),
    'gross_exposure': float,  # Current gross / G
    'net_exposure': float,    # Current net
    'daily_pnl': float,       # Realized + unrealized
}
```

---

## 5. Action Space

### 5.1 Per-Symbol Actions

| Action | Position | Weight |
|--------|----------|--------|
| 0: FLAT | 0 | 0% |
| 1: LONG_50 | +0.5 | +w_max/2 = +0.83% |
| 2: LONG_100 | +1.0 | +w_max = +1.67% |
| 3: SHORT_50 | -0.5 | -w_max/2 = -0.83% |
| 4: SHORT_100 | -1.0 | -w_max = -1.67% |

### 5.2 Portfolio Constraint Layer

The DQN outputs independent Q-values for each (symbol, action) pair:

```python
Q(s, a) → shape (num_symbols, num_actions) = (9, 5)
```

Portfolio construction:

1. **Select Best Action Per Symbol**: `a_i* = argmax_a Q(s, i, a)`
2. **Compute Conviction Score**: `q_i* = max_a Q(s, i, a) - Q(s, i, FLAT)`
3. **Rank by Conviction**: Sort symbols by |q_i*|
4. **Apply Top-K**: Keep top-K longs and top-K shorts
5. **Force Others to FLAT**: Symbols not in top-K → action = FLAT

```python
def apply_topk_constraint(q_values, k=3):
    """
    q_values: (num_symbols, num_actions)
    Returns: (num_symbols,) final actions
    """
    num_symbols = q_values.shape[0]
    best_actions = q_values.argmax(axis=1)  # (num_symbols,)
    conviction = q_values.max(axis=1) - q_values[:, 0]  # vs FLAT

    # Separate longs and shorts
    long_candidates = np.where((best_actions == 1) | (best_actions == 2))[0]
    short_candidates = np.where((best_actions == 3) | (best_actions == 4))[0]

    # Top-K selection
    long_order = np.argsort(conviction[long_candidates])
    short_order = np.argsort(conviction[short_candidates])
    long_indices = long_candidates[long_order[-k:]]
    short_indices = short_candidates[short_order[-k:]]

    # Build final actions
    final_actions = np.zeros(num_symbols, dtype=int)  # All FLAT
    final_actions[long_indices] = best_actions[long_indices]
    final_actions[short_indices] = best_actions[short_indices]

    return final_actions
```

### 5.3 Position Sizing

With G = 10% and K = 3:

| Position | Weight | Dollar Value (on $100K sleeve) |
|----------|--------|--------------------------------|
| LONG_100 | +1.67% | +$1,667 |
| LONG_50 | +0.83% | +$833 |
| FLAT | 0% | $0 |
| SHORT_50 | -0.83% | -$833 |
| SHORT_100 | -1.67% | -$1,667 |

**Max Gross Exposure**: K × w_max × 2 = 3 × 1.67% × 2 = **10%**

---

## 6. Network Architecture

### 6.1 Shared Trunk

```python
class DQNTrunk(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, num_symbols=9, symbol_embed_dim=8):
        super().__init__()
        self.symbol_embedding = nn.Embedding(num_symbols, symbol_embed_dim)
        self.fc1 = nn.Linear(state_dim + symbol_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state, symbol_id):
        # state: (batch, state_dim)
        # symbol_id: (batch,) integer

        # Symbol identity conditioning (recommended): learned embedding
        # (One-hot is acceptable for a quick baseline)
        symbol_embed = self.symbol_embedding(symbol_id)  # (batch, embed_dim)
        x = torch.cat([state, symbol_embed], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
```

### 6.2 Per-Symbol Heads (Optional)

```python
class DQNWithHeads(nn.Module):
    def __init__(self, trunk, num_symbols=9, num_actions=5, hidden_dim=256):
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            ) for _ in range(num_symbols)
        ])

    def forward(self, state, symbol_id):
        trunk_out = self.trunk(state, symbol_id)
        # Use appropriate head based on symbol_id
        q_values = self.heads[symbol_id](trunk_out)
        return q_values
```

### 6.3 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_dim` | 256 | Trunk hidden size |
| `num_layers` | 3 | Trunk depth |
| `learning_rate` | 1e-4 | Adam optimizer |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration |
| `epsilon_end` | 0.05 | Final exploration |
| `epsilon_decay` | 100,000 | Steps to decay |
| `target_update_freq` | 1000 | Target network update |
| `batch_size` | 256 | Training batch size |
| `replay_buffer_size` | 500,000 | Experience replay |
| `min_replay_size` | 10,000 | Before training starts |

---

## 7. Reward Function

### 7.1 Per-Step Reward

```python
def compute_reward(position_t, position_t1, price_t, price_t1, turnover_cost=0.0010):
    """
    Dense per-step reward.

    Args:
        position_t: Previous position {-1, -0.5, 0, +0.5, +1}
        position_t1: New position
        price_t: Previous price
        price_t1: Current price
        turnover_cost: One-way transaction cost (10 bps default)

    Returns:
        reward: float
    """
    # Log return
    log_return = np.log(price_t1 / price_t)

    # P&L from position
    pnl = position_t * log_return  # Use previous position

    # Turnover cost
    turnover = abs(position_t1 - position_t)
    cost = turnover * turnover_cost

    # Net reward
    reward = pnl - cost

    return reward
```

### 7.2 Aggregation

Portfolio-level reward is sum of per-symbol rewards:

```python
total_reward = sum(reward_i * w_max for i in range(num_symbols))
```

### 7.3 No Terminal Bonus

- Dense rewards only (every step)
- No special EOD bonus/penalty
- Mandatory flatten at 14:00 handled by environment

---

## 8. Training Configuration

### 8.1 Data Split

Data availability is constrained by `data/stage1_raw/` coverage (starts 2021-12-20).

| Split | Nominal Range | Directory | Notes |
|------|---------------|-----------|-------|
| train | 2021-12-20 → 2023-12-31 | `data/dqn_train/` | Used for main training |
| val | 2024-01-01 → 2024-06-30 | `data/dqn_val/` | Used for model selection |
| dev_test | 2024-07-01 → 2024-12-31 | `data/dqn_dev_test/` | Debug + final pre-holdout check |
| holdout | 2025-01-01 → 2025-12-19 | `data/dqn_holdout/` | DO NOT TOUCH until Gate 3 |

**Important**: Per `data/split_manifest.json`, not all symbols have full coverage through 2025-12-19 (META/SPY/TSLA end 2025-12-16). For multi-symbol training/evaluation, the environment uses the intersection of available dates across symbols.

**Sample size accounting**:
- Environment steps/day: `end_minute - start_minute = 270 - 61 = 209`
- Total action decisions ≈ `(#days × 209 × 9)` (9 per-step actions), even though the environment emits one transition per minute.

### 8.2 Walk-Forward Validation

Fold boundaries must respect the available history start (2021-12-20).

| Fold | Train | Validation | Dev Test |
|------|-------|------------|----------|
| 1 | 2021-12-20 → 2022-12-31 | 2023-01-01 → 2023-06-30 | 2024-07-01 → 2024-12-31 |
| 2 | 2021-12-20 → 2023-06-30 | 2023-07-01 → 2023-12-31 | 2024-07-01 → 2024-12-31 |
| 3 | 2021-12-20 → 2023-12-31 | 2024-01-01 → 2024-06-30 | 2024-07-01 → 2024-12-31 |

Each fold trained with 3 random seeds → 9 total runs for variance estimation.

### 8.3 Training Loop

```python
def train_dqn(env, model, target_model, replay_buffer, config):
    state = env.reset()

    for step in range(config.total_steps):
        # Epsilon-greedy action selection
        epsilon = get_epsilon(step, config)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(state)
            action = apply_topk_constraint(q_values, k=config.K)

        # Environment step
        next_state, reward, done, info = env.step(action)

        # Store transition
        replay_buffer.add(state, action, reward, next_state, done)

        # Sample and train
        if len(replay_buffer) >= config.min_replay_size:
            batch = replay_buffer.sample(config.batch_size)
            loss = compute_dqn_loss(model, target_model, batch, config.gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if step % config.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        state = next_state
        if done:
            state = env.reset()
```

---

## 9. Kill Tests

### 9.1 Gate 0: Data Sanity

| Check | Threshold | Action if Fail |
|-------|-----------|----------------|
| Missing bars in RTH | < 1% per symbol-day | Fill forward or exclude day |
| Split adjustment | GOOGL July 2022 correct | Re-download if wrong |
| Premarket coverage | > 95% trading days | Backfill gaps |
| Price outliers | No > 50% 1-min moves | Flag and review |

### 9.2 Gate 1: Baseline Policies

Before training DQN, verify environment with deterministic policies:

| Policy | Expected Sharpe | Purpose |
|--------|-----------------|---------|
| Always FLAT | ~0.0 | Verify no look-ahead bias |
| Random | ~-0.5 to -1.0 | Transaction costs dominate |
| Momentum (5-min) | -0.3 to +0.3 | Simple baseline |
| Mean Reversion | -0.3 to +0.3 | Simple baseline |

**Kill if**: Always-FLAT has Sharpe significantly ≠ 0 (indicates data leakage)

### 9.3 Gate 2: DQN Walk-Forward

Run 3 folds × 3 seeds = 9 configurations on 2021-2024 data:

| Metric | Threshold | Kill Condition |
|--------|-----------|----------------|
| Sharpe Ratio | ≥ 0.2 | Mean across seeds < 0.2 |
| Max Drawdown | ≤ 15% | Any run > 15% |
| Net Return | > 0% | Mean < 0% |
| Fold Consistency | ≥ 2/3 pass | < 2/3 folds positive |

**Kill if**: Mean Sharpe < 0.2 OR any DD > 15% OR < 2/3 folds profitable

### 9.4 Gate 3: Holdout (2025)

**Only touch if Gate 2 passes completely.**

| Metric | Threshold | Action |
|--------|-----------|--------|
| Sharpe | ≥ 0.0 | Kill if negative |
| Return | > -5% | Kill if worse |
| DD | ≤ 20% | Kill if exceeded |

**Kill if**: Sharpe < 0 on 2025 holdout

---

## 10. Execution & Deployment

### 10.1 Order Generation

```python
def generate_orders(current_positions, target_positions, prices, nav):
    """
    Generate orders to move from current to target positions.

    Args:
        current_positions: dict {symbol: weight}
        target_positions: dict {symbol: weight}
        prices: dict {symbol: price}
        nav: Total portfolio value

    Returns:
        orders: list of Order objects
    """
    orders = []
    for symbol in UNIVERSE:
        current_weight = current_positions.get(symbol, 0)
        target_weight = target_positions.get(symbol, 0)

        if abs(target_weight - current_weight) > 0.001:  # 0.1% threshold
            dollar_change = (target_weight - current_weight) * nav
            shares = int(dollar_change / prices[symbol])

            if shares != 0:
                orders.append(Order(
                    symbol=symbol,
                    side='BUY' if shares > 0 else 'SELL',
                    quantity=abs(shares),
                    order_type='MKT'  # or LIMIT with spread buffer
                ))

    return orders
```

### 10.2 Risk Checks

Before execution:

1. **Gross Exposure**: Verify ≤ G × 1.05 (5% buffer)
2. **Position Limits**: Verify each |w_i| ≤ w_max × 1.05
3. **Order Size**: Minimum $500, maximum $10K per order
4. **Rate Limit**: Max 10 orders per minute

### 10.3 Monitoring

Real-time metrics:

- P&L (realized + unrealized)
- Gross/net exposure
- Number of positions
- Turnover (daily)
- Sharpe (rolling 20-day)

---

## 11. Appendix

### A. Symbol ID Mapping

```python
SYMBOL_TO_ID = {
    'AAPL': 0,
    'AMZN': 1,
    'GOOGL': 2,
    'META': 3,
    'MSFT': 4,
    'NVDA': 5,
    'QQQ': 6,
    'SPY': 7,
    'TSLA': 8,
}
```

### B. Configuration File Template

```yaml
# config/dqn_intraday.yaml
universe:
  symbols: [AAPL, AMZN, GOOGL, META, MSFT, NVDA, QQQ, SPY, TSLA]

portfolio:
  target_gross_exposure: 0.10  # G = 10%
  positions_per_side: 3        # K = 3
  w_max: 0.0167                # G / (2K)

trading:
  start_minute: 61             # 10:31 ET (minute 61 of RTH)
  end_minute: 270              # 14:00 ET (minute 270 of RTH)

model:
  hidden_dim: 256
  num_layers: 3
  learning_rate: 0.0001
  gamma: 0.99

costs:
  turnover_cost_bps: 10        # 10 bps one-way

kill_tests:
  gate2_min_sharpe: 0.2
  gate2_max_dd: 0.15
  gate3_min_sharpe: 0.0
```

### C. Data Requirements Summary

| Data Type | Date Range | Source | Status |
|-----------|------------|--------|--------|
| RTH 1-min bars | 2021-2024 | stage1_raw | ✅ Exists |
| RTH 1-min bars | 2025 | Polygon | ⏳ Backfill needed |
| Premarket 1-min | 2021-2022 | `data/sleeve_im/minute_bars/` (Polygon cache) | ⏳ Backfill needed |
| Premarket 1-min | 2023-2024 | `data/sleeve_im/minute_bars/` (Polygon cache) | ✅ Exists |
| Premarket 1-min | 2025 | `data/sleeve_im/minute_bars/` (Polygon cache) | ⏳ Backfill needed |

**Estimated Backfill**: ~6,800 new JSON files

---

**END OF SPECIFICATION**
