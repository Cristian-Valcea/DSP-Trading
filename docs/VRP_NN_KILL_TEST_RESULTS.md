# VRP-Gated Weekly NN Kill Test Results

**Date**: 2026-01-09
**Status**: FAILED - Gate hurts performance

---

## Executive Summary

The VRP Regime Gate was tested as a position scaling mechanism for a weekly SPY direction model (logistic regression). **The gate consistently HURT performance** compared to ungated trading across all test splits.

**Recommendation**: Do NOT use VRP Regime Gate for position sizing. Explore alternative uses (e.g., execution timing, vol-targeting) or abandon the gate entirely.

---

## 1. Background

### VRP Factor Blocker
The original plan was to implement `src/dsp/vrp/factor.py` using the Babiak et al. VRP Factor methodology. This is **BLOCKED** due to:

1. **Data Cost**: OptionMetrics firm-level options data costs $5-15k/year
2. **Strategy Already Failed**: The underlying VRP futures strategy (`vrp_futures.py`) already failed kill-test with Sharpe -0.13

### Pivot to Weekly NN Gate Test
Instead of implementing the VRP Factor, we tested whether the existing VRP Regime Gate (VIX/VVIX/contango-based) could improve a weekly direction model.

---

## 2. Methodology

### Model
- **Type**: Logistic Regression (multinomial)
- **Target**: Weekly SPY return direction (-1=DOWN, 0=FLAT, +1=UP)
- **Features**: 18 features (SPY/QQQ/TLT returns, volatility, cross-asset correlations, VRP gate score)

### Gate Mechanism
```python
exposure = predicted_direction * gate_numeric
# gate_numeric: 1.0 (OPEN), 0.5 (REDUCE), 0.0 (CLOSED)
```

### Data Splits
| Split | Period | Weeks |
|-------|--------|-------|
| Train | 2015-2020 | ~273 |
| Val | 2021-2022 | ~89 |
| Test | 2023-2024 | ~90 |
| Holdout | 2025+ | ~47 |

---

## 3. Kill Test Results

### Per-Split Performance

| Split | Gated Sharpe | Ungated Sharpe | Gated P&L | Ungated P&L | Accuracy |
|-------|--------------|----------------|-----------|-------------|----------|
| Train | 0.51 | **1.19** | 25.1% | **91.0%** | 50.2% |
| Val | -0.08 | **0.07** | -1.7% | **2.0%** | 40.4% |
| Test | 0.87 | **1.12** | 16.9% | **24.6%** | 45.6% |
| Holdout | -0.51 | **0.42** | -4.7% | **6.6%** | 42.6% |

### Kill Criteria Assessment

| Criterion | Target | Gated | Ungated | Pass? |
|-----------|--------|-------|---------|-------|
| Test Sharpe | >= 0.50 | 0.87 | 1.12 | Both pass |
| Test Net P&L | > 0 | 16.9% | 24.6% | Both pass |
| Holdout Sharpe | >= 0.50 | -0.51 | 0.42 | Both fail |
| Gated > Ungated | Yes | N/A | N/A | **FAIL** |

### Verdict: **GATE FAILS KILL TEST**

The gate **reduces performance** in every single split:
- Train: -57% Sharpe reduction (1.19 → 0.51)
- Val: From marginally positive to negative
- Test: -23% Sharpe reduction (1.12 → 0.87)
- Holdout: From positive to negative

---

## 4. Analysis

### Why the Gate Hurts Performance

1. **Opportunity Cost During Crises**: When gate=CLOSED during market stress, the model cannot express SHORT positions that would profit from the decline

2. **Gate Correlation with Returns**: The gate closes during high VIX regimes, which are often followed by mean-reversion rallies

3. **Asymmetric Scaling**: Reducing position size (0.5x) during REDUCE periods doesn't help if the model's prediction is correct

4. **Sample Bias**: Gate-closed periods (6.2% of days) include some of the most profitable trading opportunities for a direction model

### Feature Importance Issue

The model likely learned to use gate_score as a feature, but the gate_numeric scaling mechanism fights against this:
- If gate_score predicts volatility → model adjusts predictions
- But then gate_numeric reduces exposure → double-adjustment

---

## 5. Recommendations

### Option A: Abandon Gate for NN
Use the weekly NN model **ungated**:
- Test Sharpe: 1.12 (passes)
- Holdout Sharpe: 0.42 (marginal)
- Requires out-of-sample validation post-2025

### Option B: Gate for Execution Only
Use gate to adjust:
- Order timing (avoid high-vol periods)
- Position sizing via vol-targeting (not binary gate)
- Stop-loss placement

### Option C: Different Gate Application
Instead of scaling positions, use gate to:
- Skip trading entirely when CLOSED (not scale to 0)
- Adjust confidence thresholds
- Change holding period

### Option D: Abandon VRP Approach
Focus on other sleeves:
- Sleeve DM: Already LIVE (Sharpe 0.55-0.87)
- Sleeve TSMOM: Data ready, awaiting backtest

---

## 6. Files Reference

| File | Purpose |
|------|---------|
| `src/dsp/ml/vrp_gated_nn.py` | Model training + evaluation |
| `src/dsp/features/weekly_direction_features.py` | Feature engineering |
| `src/dsp/regime/vrp_regime_gate.py` | Gate implementation |
| `data/vrp/nn_features.parquet` | Training data (575 weeks) |
| `data/vrp/models/vrp_nn_evaluation.json` | Full evaluation results |

---

## 7. Conclusion

**The VRP Regime Gate as a position-scaling mechanism for the weekly direction NN FAILS the kill test.**

The gate consistently reduces performance vs ungated trading. This does not mean the gate is useless - it successfully identifies crisis regimes (100% COVID detection) - but using it to scale positions in a direction-prediction model is counterproductive.

**Next Steps**:
1. Test ungated weekly NN with proper walk-forward validation
2. Or pivot to other sleeves (TSMOM data is ready)
3. Consider gate for non-position-sizing applications

---

*Generated: 2026-01-09*
