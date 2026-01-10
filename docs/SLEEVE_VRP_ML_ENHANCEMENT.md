# SLEEVE_VRP_ML_ENHANCEMENT.md — AI-Enhanced VRP (Phase 2 Roadmap)

**Version**: 1.0
**Date**: 2026-01-08
**Status**: Research Candidate (Post-Baseline)
**References**: Wang et al. (2024), Nguyen (2025), Cho et al. (2025)

---

## 1. Executive Summary

This document specifies optional ML enhancements for the VRP strategy, to be implemented **only after** the baseline rule-based VRP passes its kill-test. The ML layer acts as a **regime filter** to avoid entering short VIX positions during high-risk periods.

**Key Principle**: ML enhances but does not replace the rule-based system. All baseline filters remain active.

---

## 2. Core Insight: The "Constant Maturity" Edge

### 2.1 Problem with Raw Futures

Traditional VRP strategies trade specific contracts (e.g., "Short Feb VIX"). This introduces noise because the "Feb contract" changes character as it approaches expiration:

- **Day 30**: High roll yield, behaves like "30-day vol"
- **Day 5**: Low roll yield, behaves like "5-day vol"
- **Signal quality degrades** as contract approaches expiry

### 2.2 Solution: Constant Maturity Futures (CMF)

Construct synthetic **Constant Maturity Futures** data by interpolating between front and back month contracts:

```python
def calculate_cmf_30(
    f1_price: float,
    f2_price: float,
    days_to_f1: int,
    days_to_f2: int
) -> float:
    """
    Calculate 30-day Constant Maturity Future price.

    Linear interpolation between F1 and F2 to create synthetic
    "30-day constant maturity future" price.

    Args:
        f1_price: Front-month futures price
        f2_price: Second-month futures price
        days_to_f1: Days until front-month expiration
        days_to_f2: Days until second-month expiration

    Returns:
        Synthetic 30-day CMF price
    """
    TARGET_DAYS = 30

    # Linear interpolation weight
    # When days_to_f1=30, w=1.0 (use F1 only)
    # When days_to_f1=0, w approaches 0 (use mostly F2)
    w = (days_to_f2 - TARGET_DAYS) / (days_to_f2 - days_to_f1)
    w = max(0.0, min(1.0, w))  # Clamp to [0, 1]

    cmf_30 = w * f1_price + (1 - w) * f2_price

    return cmf_30
```

### 2.3 CMF-Derived Features

```python
def calculate_cmf_features(
    f1_price: float,
    f2_price: float,
    days_to_f1: int,
    days_to_f2: int,
    vix_spot: float
) -> dict:
    """
    Calculate all CMF-derived features for ML model.

    Returns:
        Dict with cmf_30, roll_yield, spot_vol_spread
    """
    cmf_30 = calculate_cmf_30(f1_price, f2_price, days_to_f1, days_to_f2)

    # Roll Yield: Expected return from contango decay
    # Annualized: (F1 - VIX) / VIX * (365 / days_to_f1)
    if days_to_f1 > 0 and vix_spot > 0:
        roll_yield = (f1_price - vix_spot) / vix_spot * (365 / days_to_f1)
    else:
        roll_yield = 0.0

    # CMF Roll Yield: More stable version using CMF
    if cmf_30 > 0 and vix_spot > 0:
        cmf_roll_yield = (cmf_30 - vix_spot) / vix_spot * (365 / 30)
    else:
        cmf_roll_yield = 0.0

    # Spot-Vol Spread: Difference between spot and constant maturity
    spot_vol_spread = vix_spot - cmf_30

    return {
        'cmf_30': cmf_30,
        'roll_yield': roll_yield,
        'cmf_roll_yield': cmf_roll_yield,
        'spot_vol_spread': spot_vol_spread
    }
```

---

## 3. Neural Network Architecture: ALSTM

Recent literature (Wang et al., PLOS One 2024) identifies **Attention-LSTM (ALSTM)** as the superior architecture for VIX regime forecasting.

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      ALSTM Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT LAYER (6 features)                                   │
│  ├── Term Structure Slope: F6 - F1 (back - front)          │
│  ├── CMF Roll Yield: (CMF_30 - VIX) / VIX * 12             │
│  ├── Spot-Vol Spread: VIX_spot - CMF_30                    │
│  ├── VVIX: Volatility of VIX index                         │
│  ├── US 10Y Yield: Risk-free rate proxy                    │
│  └── DXY: Dollar strength (flight-to-safety signal)        │
│                                                              │
│  ▼                                                          │
│                                                              │
│  LSTM LAYER (64 units, 20-day lookback)                     │
│  └── Captures volatility clustering patterns                │
│                                                              │
│  ▼                                                          │
│                                                              │
│  ATTENTION LAYER (self-attention)                           │
│  └── Weights past days dynamically                          │
│      (e.g., higher weight on Fed meeting days)              │
│                                                              │
│  ▼                                                          │
│                                                              │
│  DENSE LAYER (32 units, ReLU)                               │
│                                                              │
│  ▼                                                          │
│                                                              │
│  OUTPUT LAYER (2 classes, softmax)                          │
│  ├── Class 0: Safe Regime (OK to short VIX)                │
│  └── Class 1: Risk Regime (avoid short VIX)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Feature Engineering

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ALSTMFeatures:
    """Features for ALSTM regime classifier."""

    # Term structure
    term_slope: float         # F6 - F1 (wider term spread = more backwardation risk)
    cmf_roll_yield: float     # Annualized CMF roll yield
    spot_vol_spread: float    # VIX_spot - CMF_30

    # Volatility of volatility
    vvix: float               # CBOE VVIX index

    # Macro factors
    us_10y_yield: float       # US 10-year Treasury yield
    dxy: float                # Dollar Index

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.term_slope,
            self.cmf_roll_yield,
            self.spot_vol_spread,
            self.vvix,
            self.us_10y_yield,
            self.dxy
        ])

def build_alstm_features(
    market_data: dict,
    lookback_days: int = 20
) -> np.ndarray:
    """
    Build feature matrix for ALSTM model.

    Args:
        market_data: Dict with historical data for past `lookback_days`
        lookback_days: Number of past days to include

    Returns:
        Shape (lookback_days, 6) feature matrix
    """
    features = []

    for i in range(lookback_days):
        day_data = market_data['history'][-(lookback_days - i)]

        # Calculate CMF features
        cmf = calculate_cmf_features(
            day_data['f1_price'],
            day_data['f2_price'],
            day_data['days_to_f1'],
            day_data['days_to_f2'],
            day_data['vix_spot']
        )

        row = ALSTMFeatures(
            term_slope=day_data['f6_price'] - day_data['f1_price'],
            cmf_roll_yield=cmf['cmf_roll_yield'],
            spot_vol_spread=cmf['spot_vol_spread'],
            vvix=day_data['vvix'],
            us_10y_yield=day_data['us_10y'],
            dxy=day_data['dxy']
        )
        features.append(row.to_array())

    return np.array(features)
```

### 3.3 Model Definition (PyTorch)

```python
import torch
import torch.nn as nn

class ALSTMRegimeClassifier(nn.Module):
    """
    Attention-LSTM for VIX regime classification.

    Based on Wang et al. (2024) architecture.
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Regime probabilities of shape (batch, 2)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)

        # Weighted sum (context vector)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)

        # Classification
        output = self.fc(context)  # (batch, 2)

        return output

    def predict_regime(self, x: torch.Tensor, threshold: float = 0.6) -> int:
        """
        Predict regime with configurable threshold.

        Args:
            x: Input features
            threshold: Probability threshold for risk regime

        Returns:
            0 = Safe, 1 = Risk
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            risk_prob = probs[0, 1].item()  # P(Risk)
            return 1 if risk_prob >= threshold else 0
```

---

## 4. Integration with Baseline VRP

### 4.1 ML as Additional Filter

The ML model acts as an **additional entry filter**, not a replacement for rule-based filters:

```python
def check_entry_filters_with_ml(
    adjusted_contango: float,
    vix_spot: float,
    vix_50d_ma: float,
    vix1d: float,
    vvix: float,
    vvix_90th_pct: float,
    vix1d_vix_95th_pct: float,
    cool_down_active: bool,
    market_data: dict,
    ml_model: ALSTMRegimeClassifier = None,
    ml_threshold: float = 0.6
) -> tuple[bool, str]:
    """
    Enhanced entry filter with optional ML overlay.

    ML model only blocks entry when BOTH:
    1. Rule-based filters pass
    2. ML predicts high-risk regime

    Args:
        ... (same as baseline check_entry_filters)
        ml_model: Trained ALSTM model (None to skip ML)
        ml_threshold: P(Risk) threshold for blocking entry

    Returns:
        (can_enter, reason)
    """
    # STEP 1: Run ALL baseline filters first
    can_enter, reason = check_entry_filters(
        adjusted_contango,
        vix_spot,
        vix_50d_ma,
        vix1d,
        vvix,
        vvix_90th_pct,
        vix1d_vix_95th_pct,
        cool_down_active
    )

    # If baseline rejects, no need for ML check
    if not can_enter:
        return can_enter, reason

    # STEP 2: ML overlay (only if baseline passes)
    if ml_model is not None:
        # Build features
        features = build_alstm_features(market_data, lookback_days=20)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Predict regime
        ml_model.eval()
        with torch.no_grad():
            probs = ml_model(x)
            risk_prob = probs[0, 1].item()

        if risk_prob >= ml_threshold:
            return False, f"ML Model predicts High Risk ({risk_prob:.1%} >= {ml_threshold:.0%})"

    return True, "All filters pass (including ML)"
```

### 4.2 ML Filter Placement in Entry Logic

```
Entry Decision Flow (with ML):
─────────────────────────────────────────────────────────────────────

  START
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  RULE-BASED FILTERS (Must ALL Pass)                            │
│  ├── F1: Adjusted Contango >= 0.5                              │
│  ├── F2: VIX < 50-day MA (vol falling)                        │
│  ├── F3: VIX1D/VIX < 95th percentile                          │
│  ├── F4: VVIX < 90th percentile                               │
│  ├── F5: Cool-down inactive                                    │
│  └── F6: VIX < 25 (not high-vol regime)                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼ All Pass?
    │
   NO ─────────────────────────────────────────────────▶ NO ENTRY
    │
   YES
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ML FILTER (Optional - Phase 2)                                 │
│  ├── Build 20-day feature matrix                               │
│  ├── Run ALSTM forward pass                                    │
│  └── If P(Risk) >= 60%: Block entry                            │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼ Safe Regime?
    │
   NO ─────────────────────────────────────────────────▶ NO ENTRY
    │
   YES
    │
    ▼
  ENTER POSITION
```

---

## 5. Implementation Plan

### Phase 1: Feature Engineering (After Baseline Kill-Test)

**Objective**: Build CMF data pipeline and feature store

| Task | Deliverable | Duration |
|------|-------------|----------|
| Implement `calculate_cmf_30()` | Tested function | 2 hours |
| Implement `calculate_cmf_features()` | Tested function | 2 hours |
| Build CMF historical series | 2015-2025 CMF data | 4 hours |
| Integrate macro features (10Y, DXY) | Feature pipeline | 2 hours |
| Create feature store schema | Parquet files | 2 hours |

**Success Criteria**:
- CMF series validated against raw contracts (interpolation correct)
- All features available for backtest period
- Feature store documented

### Phase 2: Model Training

**Objective**: Train and validate ALSTM regime classifier

| Task | Deliverable | Duration |
|------|-------------|----------|
| Define target variable | Crisis dates labeling | 4 hours |
| Train/val/test split | Walk-forward setup | 2 hours |
| Hyperparameter tuning | Best model | 8 hours |
| Model validation | OOS metrics | 4 hours |
| Threshold calibration | Optimal threshold | 2 hours |

**Target Variable Definition**:
```python
def label_crisis_regime(
    returns: pd.Series,
    lookforward_days: int = 5,
    threshold_pct: float = -0.10
) -> pd.Series:
    """
    Label crisis regime based on forward-looking short VIX returns.

    Risk = 1 if worst return in next 5 days <= -10%
    """
    worst_forward = returns.rolling(lookforward_days).min().shift(-lookforward_days)
    return (worst_forward <= threshold_pct).astype(int)
```

**Success Criteria**:
- OOS Precision >= 60% (when predicting Risk)
- OOS Recall >= 50% (catch at least half of crises)
- False Positive Rate <= 30% (don't block too many good trades)

### Phase 3: Integration & Validation

**Objective**: Integrate ML filter and measure marginal value

| Task | Deliverable | Duration |
|------|-------------|----------|
| Integrate with entry filters | Updated `check_entry_filters()` | 4 hours |
| A/B backtest (ML vs no-ML) | Comparison report | 4 hours |
| Sensitivity analysis | Threshold sweep results | 4 hours |
| Production deployment | Feature flag system | 4 hours |
| Monitoring dashboard | ML metrics panel | 4 hours |

**Validation Protocol**:
```python
def run_ml_ablation_study(
    baseline_config: BacktestConfig,
    ml_model: ALSTMRegimeClassifier,
    data: dict,
    thresholds: list[float] = [0.5, 0.6, 0.7, 0.8]
) -> pd.DataFrame:
    """
    Compare baseline vs ML-enhanced performance.

    Tests multiple ML thresholds to find optimal setting.
    """
    results = []

    # Run baseline (no ML)
    baseline_results = run_backtest(baseline_config, data, ml_model=None)
    results.append({
        'variant': 'Baseline (No ML)',
        'threshold': None,
        **calculate_metrics(baseline_results)
    })

    # Run with ML at various thresholds
    for threshold in thresholds:
        ml_results = run_backtest(
            baseline_config, data,
            ml_model=ml_model,
            ml_threshold=threshold
        )
        results.append({
            'variant': f'ML @ {threshold:.0%}',
            'threshold': threshold,
            **calculate_metrics(ml_results)
        })

    return pd.DataFrame(results)
```

**Success Criteria (ML vs Baseline)**:
- Sharpe improvement >= 10% (marginal value)
- Drawdown improvement >= 5% (risk reduction)
- Win rate improvement >= 5% (better trade selection)
- Statistical significance p < 0.05 (bootstrap test)

---

## 6. Data Requirements

### 6.1 Additional Data for ML

| Data | Source | Frequency | Historical Depth |
|------|--------|-----------|------------------|
| VIX Futures F1-F6 | Databento/CBOE | Daily OHLC | 2015-present |
| US 10Y Yield | FRED (DGS10) | Daily | 2015-present |
| Dollar Index (DXY) | Investing.com/FRED | Daily | 2015-present |

### 6.2 Feature Store Schema

```python
@dataclass
class ALSTMFeatureRow:
    """Daily feature row for ML model."""
    date: datetime

    # CMF features
    cmf_30: float
    cmf_roll_yield: float
    spot_vol_spread: float

    # Term structure
    f1_price: float
    f6_price: float
    term_slope: float

    # Volatility
    vix_spot: float
    vvix: float

    # Macro
    us_10y_yield: float
    dxy: float

    # Target (for training)
    risk_regime: int  # 0=Safe, 1=Risk (labeled post-hoc)
```

---

## 7. Risk Controls for ML Layer

### 7.1 Model Staleness Check

```python
def check_model_staleness(
    model_train_date: datetime,
    max_age_days: int = 90
) -> bool:
    """
    Ensure model is not stale.

    Retrain required every 90 days or after major vol events.
    """
    age = (datetime.now() - model_train_date).days
    if age > max_age_days:
        logger.warning(f"ML model is {age} days old (max {max_age_days})")
        return True
    return False
```

### 7.2 Feature Drift Detection

```python
def detect_feature_drift(
    current_features: np.ndarray,
    training_distribution: dict,
    threshold: float = 3.0
) -> bool:
    """
    Detect if current features are out-of-distribution.

    Uses z-score relative to training mean/std.
    """
    for i, (mean, std) in enumerate(zip(
        training_distribution['means'],
        training_distribution['stds']
    )):
        z_score = abs(current_features[i] - mean) / std
        if z_score > threshold:
            logger.warning(
                f"Feature {i} drift detected: z-score {z_score:.1f} > {threshold}"
            )
            return True
    return False
```

### 7.3 Graceful Degradation

```python
def get_ml_prediction_safe(
    ml_model: ALSTMRegimeClassifier,
    market_data: dict,
    ml_threshold: float = 0.6
) -> tuple[bool, str]:
    """
    Get ML prediction with graceful degradation.

    Returns (should_block, reason)
    """
    try:
        # Check model staleness
        if check_model_staleness(ml_model.train_date):
            return False, "ML model stale - defaulting to allow"

        # Build features
        features = build_alstm_features(market_data)

        # Check feature drift
        if detect_feature_drift(features[-1], ml_model.training_dist):
            return False, "Feature drift detected - defaulting to allow"

        # Run inference
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        ml_model.eval()
        with torch.no_grad():
            probs = ml_model(x)
            risk_prob = probs[0, 1].item()

        if risk_prob >= ml_threshold:
            return True, f"ML Risk: {risk_prob:.1%}"
        else:
            return False, f"ML Safe: {risk_prob:.1%}"

    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        return False, f"ML error - defaulting to allow: {e}"
```

---

## 8. Success Metrics

### 8.1 ML Model Quality

| Metric | Target | Measurement |
|--------|--------|-------------|
| OOS Precision (Risk class) | >= 60% | True positives / predicted positives |
| OOS Recall (Risk class) | >= 50% | True positives / actual positives |
| OOS AUC-ROC | >= 0.70 | Area under ROC curve |
| False Positive Rate | <= 30% | False positives / actual negatives |

### 8.2 Strategy Improvement

| Metric | Target | Comparison |
|--------|--------|------------|
| Sharpe Improvement | >= +10% | (ML_Sharpe - Baseline_Sharpe) / Baseline_Sharpe |
| Max DD Improvement | >= +5% | (Baseline_DD - ML_DD) / abs(Baseline_DD) |
| Win Rate Improvement | >= +5% | ML_WinRate - Baseline_WinRate |
| Statistical Significance | p < 0.05 | Bootstrap difference test |

---

## 9. Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Phase 0**: Baseline VRP | 2 weeks | None (in progress) |
| **Phase 1**: Feature Engineering | 1 week | Baseline kill-test pass |
| **Phase 2**: Model Training | 2 weeks | Phase 1 complete |
| **Phase 3**: Integration | 1 week | Phase 2 complete |
| **Phase 4**: Validation | 1 week | Phase 3 complete |
| **Total** | ~7 weeks | After baseline approval |

---

## 10. References

1. **Wang et al. (2024)** - "ALSTM for VIX Forecasting", PLOS One
   - Attention-LSTM architecture for volatility prediction
   - Demonstrates superiority over standard LSTM and GARCH

2. **Nguyen (2025)** - "Machine Learning in Volatility Trading"
   - Feature engineering for VIX term structure
   - Constant Maturity Futures methodology

3. **Cho et al. (2025)** - "Regime Detection in VIX Markets"
   - Regime classification frameworks
   - Integration with rule-based strategies

---

## Appendix A: Deleted Files Reference

This document consolidates and replaces the following problematic files:

1. **`SLEEVE_VRP_ML_STRATEGY_FUCKED.md`** (DELETED)
   - Corrupted: Content repeated 3 times
   - Incomplete: Cut off after Section 3
   - Broken LaTeX notation

2. **` SLEEVE_VRP_ML_STRATEGY.md`** (DELETED - note leading space in filename)
   - Incomplete: Only 45 lines
   - Missing Phases 2-3 implementation details

---

**Document Version**: 1.0
**Status**: Research Candidate (Implement after baseline VRP passes kill-test)
**Next Step**: Complete baseline VRP kill-test, then proceed to Phase 1
