# NN Objective (DQN Sleeve): Profit First, Risk as Gates

## Conclusion
Optimizing a single metric (especially Sharpe) is brittle in trading RL: it can be **gamed** (e.g., “always FLAT”), it under-penalizes **tail risk** (drawdowns), and it is **noisy** on small evaluation samples. The most robust approach is a **hierarchy of objectives**:

1. **Primary objective:** maximize a profit metric (CAGR proxy).
2. **Hard gates (constraints):** reject any model that violates non‑negotiable risk and realism limits (drawdown, turnover/cost drag, minimum activity, etc.).
3. **Tie‑breaker:** use Sharpe (or a similar risk-adjusted score) only to choose between models that already pass the hard gates and are close on profit.

This “profit-first, risk-as-constraints” structure aligns RL training with real financial objectives without letting the agent win by doing nothing.

## Why A Single Metric Fails
- **Sharpe can select “do nothing”.** If costs are realistic and edge is weak, “always FLAT” can beat noisy trading on Sharpe.
- **Sharpe ignores tail structure.** Drawdowns and skew/kurtosis matter; Sharpe assumes something close to normal.
- **Small-sample noise.** With limited eval days/episodes, Sharpe swings wildly and can mislead checkpoint selection.

## Metrics (Evaluation-Time)
Compute these on **VAL** (and then **DEV_TEST**) from the episode/day outcomes.

### Definitions (Code-Aligned)
This section makes the metrics concrete and consistent with the current implementation in `src/dqn/env.py` + `src/dqn/reward.py`.

#### Episode, step, and action semantics
- One **episode** = one trading day.
- One **step** = one minute in the trading window (currently ~10:31 → 14:00 ET).
- Per symbol, the agent selects one discrete action:
  - `0: FLAT → position = 0.0`
  - `1: LONG_50 → position = +0.5`
  - `2: LONG_100 → position = +1.0`
  - `3: SHORT_50 → position = -0.5`
  - `4: SHORT_100 → position = -1.0`

#### Prices and returns
For each symbol `i` and minute `t`:
- `lr_{i,t} = log(P_{i,t} / P_{i,t-1})`

#### Portfolio weights (how to interpret “position”)
The code tracks a per-symbol maximum weight `w_max` (default: `target_gross / (2 * k_per_side)`), and positions are *multipliers* in `{-1, -0.5, 0, +0.5, +1}`.

To interpret returns in portfolio terms, define:
- `w_{i,t} = w_max * position_{i,t}`
- Gross exposure at time `t`: `gross_t = sum_i |w_{i,t}|`
- Net exposure at time `t`: `net_t = sum_i w_{i,t}`

#### Reward as implemented (dense, per-step)
Per symbol:
- `reward_{i,t} = position_{i,t-1} * lr_{i,t} - turnover_cost * |position_{i,t} - position_{i,t-1}|`

Portfolio reward (per step) is the sum across symbols.

Important scale note:
- The environment currently sums rewards across symbols **without multiplying by `w_max`**. Therefore the raw `daily_pnl` is best treated as a *return-like score in “position units”*.
- If you want a portfolio log-return estimate (for CAGR/drawdown in % terms), use:
  - `r_d = w_max * daily_pnl_d`
  - This interpretation matches the idea that positions are scaled into real portfolio weights via `w_max`.

#### Turnover (code-aligned definition for gates)
Define daily turnover in portfolio-weight units:
- `turnover_d = sum_t sum_i |w_{i,t} - w_{i,t-1}|`
- Annualized turnover: `annual_turnover = 252 * mean(turnover_d)`

This is the standard “how many times per year you trade your capital” definition.

#### Activity (to prevent “always FLAT”)
Define activity on a per-step basis:
- `activity_pct = 100 * mean_t[ gross_t > 0 ]`

Optionally define a stricter version that avoids counting tiny exposure:
- `activity_pct_strict = 100 * mean_t[ gross_t >= 0.25 * target_gross ]`

### Profit / CAGR proxy (primary)
Use a daily return series from evaluation episodes and annualize:

- Let `r_d` be the daily **log-return** estimate for episode/day `d` (see “Reward as implemented” above).
- `CAGR_proxy = exp(252 * mean(r_d)) - 1`.

Notes for this codebase:
- `daily_pnl` in the environment is an accumulated reward. For percent-like performance numbers, treat `r_d = w_max * daily_pnl_d`.

### Max drawdown (hard gate)
Build an equity curve from the daily log-returns:

- `E_0 = 1`
- `E_{d+1} = E_d * exp(r_d)`
- `MaxDD = max_peak_to_trough_decline(E)`

Gate example: `MaxDD <= 15%` (tuneable).

### Turnover / cost drag (hard gate)
Control “high-churn noise trading” by gating on either turnover or cost drag:

- Use `annual_turnover` as defined above.
  - Gate example: `annual_turnover <= T_max`.
- Annual cost drag can be approximated as:
  - `annual_cost_drag ≈ annual_turnover * cost_bps_one_way`
  - Gate example: `annual_cost_drag <= X%`.

### Minimum activity (hard gate)
Prevent the trivial optimum (“always FLAT”) from being selected as “best”:

- Use `activity_pct` (or `activity_pct_strict`) as defined above.
- Gate example: `activity_pct >= 5%` (tuneable).

### Sharpe (sanity gate + tie-breaker)
Sharpe is still useful as:
- a **sanity gate** (don’t accept negative risk-adjusted returns), and/or
- a **tie-breaker** between similarly profitable candidates.

Example usage:
- Gate: `Sharpe >= 0` (on VAL and DEV_TEST)
- Tie-break: among gate-passing models, prefer higher Sharpe when CAGR is within a small band.

## Model Selection Logic (Hierarchical)
1. Evaluate candidate checkpoints on **VAL** with enough episodes to reduce noise (e.g., 20–50+).
2. Reject if any hard gate fails (`MaxDD`, `turnover/cost-drag`, `activity`, and optionally `Sharpe >= 0`).
3. Among survivors, choose the checkpoint with highest `CAGR_proxy`.
4. If two checkpoints are within a tolerance on CAGR, use Sharpe as tie-breaker.
5. Only after passing on VAL, repeat the same gating on **DEV_TEST**. Proceed only if both pass.
6. Treat 2025 as a true holdout: do not use it for tuning/selection; only evaluate once the design is locked.

## Implementation Configuration
Define thresholds in a `success_criteria.yaml` file so criteria are explicit and adjustable per phase.

```yaml
primary_metric: 'cagr_proxy'
tie_breaker_metric: 'sharpe_ratio'

hard_gates:
  max_drawdown_pct:
    direction: 'max'
    threshold: 15.0
  annual_turnover:
    direction: 'max'
    threshold: 250.0
  activity_pct:
    direction: 'min'
    threshold: 5.0
  sharpe_ratio:
    direction: 'min'
    threshold: 0.0
```

## Clarifying Note on “Entropy”
In this project, “entropy” is best treated as a diagnostic (how mixed the action distribution is), not a success metric. A rising entropy can simply mean the policy is changing its action mix; it does not imply the agent is learning profitable structure.
