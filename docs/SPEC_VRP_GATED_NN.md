# SPEC_VRP_GATED_NN.md — VRP-Gated Weekly Direction Model

**Status**: Draft specification
**Date**: 2026-01-09
**Dependency**: `SPEC_VRP_REGIME_GATE.md` (implemented)

---

## 1. Executive Summary

This spec defines a weekly equity direction model (SPY/QQQ) that is **gated by the VRP regime classifier**. The NN only trades when the regime gate is OPEN; it reduces/exits when REDUCE; it stays flat when CLOSED.

**Key Insight**: The regime gate has been validated to catch 100% of COVID crisis days and 91% of Volmageddon days. By gating NN trades, we can let a simple direction model operate in favorable conditions while automatically protecting capital during crises.

---

## 2. Model Objective

### 2.1 Target Variable
Predict the **sign of next-week's return** for SPY and QQQ:
```
y = sign(P_{t+5} / P_t - 1)
```
Where:
- `P_t` = Friday close price
- `P_{t+5}` = Following Friday close price
- `y ∈ {-1, 0, +1}` (DOWN, FLAT, UP)

**Note**: We use 3-class instead of binary to allow a "no signal" output when confidence is low.

### 2.2 Trading Horizon
| Aspect | Specification |
|--------|---------------|
| Entry | Monday open (or first trading day of week) |
| Exit | Friday close (or regime flip, whichever first) |
| Holding period | 1-5 trading days |
| Rebalance | Weekly |

### 2.3 Why Weekly (Not Intraday)?
1. **Transaction costs**: Fewer trades = costs don't dominate returns
2. **Signal quality**: Weekly returns have higher signal-to-noise than minute returns
3. **Regime alignment**: VRP indicators are daily/weekly signals
4. **Academic support**: LSTM direction models show ~0.84 Sharpe at daily+ frequencies

---

## 3. Gate Integration

### 3.1 VRPRegimeGate Dependency
The model imports and uses:
```python
from dsp.regime import VRPRegimeGate, GateState

gate = VRPRegimeGate()
state = gate.update(vix=..., vvix=..., vx_f1=...)
```

### 3.2 Trading Rules by Gate State

| Gate State | Model Action | Position Limit |
|------------|--------------|----------------|
| **OPEN** | Trade NN signal freely | 100% of target size |
| **REDUCE** | Only reduce/exit positions | 50% max, no new positions |
| **CLOSED** | Must be flat | 0% (exit any positions) |

### 3.3 State Transition Handling
- **OPEN → REDUCE**: Exit 50% of position by Tuesday close
- **OPEN → CLOSED**: Exit 100% of position immediately (MOC order)
- **REDUCE → CLOSED**: Exit remaining position immediately
- **CLOSED → REDUCE**: May build to 50% position
- **REDUCE → OPEN**: May build to 100% position

### 3.4 Gate Check Frequency
- **Daily at close**: Update gate with latest VIX/VVIX/VX data
- **Intraday on VIX spike**: If VIX > 30 intraday, force gate check

---

## 4. Model Architecture

### 4.1 Candidate Architectures (To Be Validated)

**Option A: LSTM Direction Classifier**
```
Input (features) → LSTM(64) → Dense(32) → Softmax(3)
Output: P(DOWN), P(FLAT), P(UP)
```

**Option B: Transformer with Cross-Asset Attention**
```
Input (multi-asset features) → Transformer(heads=4) → Dense(3)
Output: P(DOWN), P(FLAT), P(UP)
```

**Option C: Gradient Boosting Baseline**
```
XGBoost/LightGBM classifier
Input: Engineered features → Output: Class probabilities
```

### 4.2 Decision Rule
```python
def get_position(probs, confidence_threshold=0.55):
    """
    probs: [P(DOWN), P(FLAT), P(UP)]
    Returns: -1 (short), 0 (flat), +1 (long)
    """
    max_prob = max(probs)
    if max_prob < confidence_threshold:
        return 0  # No signal

    if probs[2] == max_prob:  # UP
        return +1
    elif probs[0] == max_prob:  # DOWN
        return -1
    else:
        return 0  # FLAT class won
```

---

## 5. Feature Set

### 5.1 Price-Based Features (Per Asset)
| Feature | Description | Lookback |
|---------|-------------|----------|
| `ret_5d` | 5-day return | 5 days |
| `ret_10d` | 10-day return | 10 days |
| `ret_20d` | 20-day return (1 month) | 20 days |
| `ret_60d` | 60-day return (3 months) | 60 days |
| `vol_20d` | 20-day realized volatility | 20 days |
| `vol_ratio` | vol_5d / vol_20d | 5/20 days |

### 5.2 Cross-Asset Features
| Feature | Description |
|---------|-------------|
| `spy_qqq_corr_20d` | Rolling correlation SPY/QQQ |
| `spy_tlt_corr_20d` | Rolling correlation SPY/TLT |
| `equity_bond_spread` | SPY ret_20d - TLT ret_20d |

### 5.3 Volatility Regime Features
| Feature | Description |
|---------|-------------|
| `vix_level` | VIX spot (normalized) |
| `vix_percentile` | VIX percentile rank (252d) |
| `vvix_level` | VVIX (normalized) |
| `contango` | VX_F1 - VIX |
| `regime_score` | From VRPRegimeGate.compute_score() |

### 5.4 Macro/Calendar Features
| Feature | Description |
|---------|-------------|
| `fomc_week` | Binary: FOMC meeting this week |
| `opex_week` | Binary: Options expiration week |
| `month_end` | Binary: Last week of month |
| `quarter_end` | Binary: Last week of quarter |

---

## 6. Training Protocol

### 6.1 Data Split
| Set | Period | Purpose |
|-----|--------|---------|
| TRAIN | 2015-2020 | Model fitting |
| VAL | 2021-2022 | Hyperparameter tuning |
| TEST | 2023-2024 | Final evaluation |
| HOLDOUT | 2025+ | True out-of-sample |

### 6.2 Training Procedure
1. **Feature engineering**: Compute all features for TRAIN/VAL/TEST
2. **Class balancing**: Use class weights or SMOTE (directional predictions often imbalanced)
3. **Cross-validation**: 5-fold time-series CV on TRAIN
4. **Hyperparameter search**: Grid search on VAL performance
5. **Final model**: Retrain on TRAIN+VAL, evaluate on TEST

### 6.3 Loss Function
```python
# Weighted cross-entropy (penalize wrong direction more than flat)
weights = {-1: 1.5, 0: 1.0, +1: 1.5}
loss = CrossEntropyLoss(weight=weights)
```

---

## 7. Backtest Specification

### 7.1 Backtest Parameters
```yaml
start_date: "2018-01-01"
end_date: "2025-12-31"
initial_capital: 100000
position_size: 0.10  # 10% per position
transaction_cost: 0.0010  # 10bps round-trip
slippage: 0.0005  # 5bps
```

### 7.2 Backtest Variants
1. **Ungated**: NN trades every week regardless of regime
2. **Gated (default thresholds)**: NN gated by VRPRegimeGate
3. **Gated (conservative)**: Higher CLOSED threshold (-0.2 instead of -0.3)

### 7.3 Comparison Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe Ratio | ≥ 0.50 | Minimum viable |
| Max Drawdown | ≤ 20% | Risk control |
| Win Rate | ≥ 52% | Better than random |
| Gated Sharpe > Ungated | Yes | Gate adds value |
| Gated MaxDD < Ungated | Yes | Gate reduces tail risk |

---

## 8. Kill Criteria

The gated NN fails if ANY of these conditions are true:

| Criterion | Threshold | Checked On |
|-----------|-----------|------------|
| Sharpe Ratio | < 0.50 | TEST set |
| Max Drawdown | > 25% | TEST set |
| Win Rate | < 50% | TEST set |
| Gated vs Ungated Sharpe | Gated worse | Comparison |
| Walk-forward folds | < 2/3 pass | Cross-validation |

---

## 9. Implementation Roadmap

### Phase 1: Baseline Model (Current)
- [x] Implement VRPRegimeGate (`src/dsp/regime/vrp_regime_gate.py`)
- [x] Validate gate on historical data (100% COVID detection)
- [ ] Build feature engineering pipeline
- [ ] Train baseline LSTM on SPY direction

### Phase 2: Gated Backtest
- [ ] Implement gated backtester (`src/dsp/backtest/nn_vrp_gated.py`)
- [ ] Run ungated vs gated comparison
- [ ] Evaluate kill criteria

### Phase 3: Production Integration (If Passes)
- [ ] Daily feature pipeline
- [ ] Weekly signal generation
- [ ] Integration with DSP-100K allocation

---

## 10. Data Requirements

### 10.1 Already Available
| Data | Location | Coverage |
|------|----------|----------|
| VIX spot | `data/vrp/indices/VIX_spot.parquet` | 2004-2026 |
| VVIX | `data/vrp/indices/VVIX.parquet` | 2011-2026 |
| VX F1 | `data/vrp/futures/VX_F1_CBOE.parquet` | 2013-2026 |

### 10.2 To Be Acquired
| Data | Source | Priority |
|------|--------|----------|
| SPY daily OHLCV | Polygon/Yahoo | HIGH |
| QQQ daily OHLCV | Polygon/Yahoo | HIGH |
| TLT daily OHLCV | Polygon/Yahoo | MEDIUM |
| FOMC meeting dates | Fed calendar | LOW |

---

## 11. Open Questions

1. **Single vs Multi-Asset**: Should we predict SPY and QQQ jointly or separately?
2. **Position Sizing**: Fixed 10% or volatility-targeted?
3. **Partial Signals**: What if model says UP for SPY but DOWN for QQQ?
4. **Intraweek Exits**: Should we exit mid-week if regime flips to CLOSED?

---

## Appendix A: Gate State Distribution (2013-2026)

From validation backtest:
- **OPEN**: 71.7% of days
- **REDUCE**: 22.1% of days
- **CLOSED**: 6.2% of days

This means the NN can trade freely ~72% of the time, providing ample opportunity while staying protected during the ~6% of crisis days.
