# Sleeve NN: Premarket Features → First Hour Direction

**Date**: 2026-01-02
**Status**: Proposal (post Sleeve IM kill)

---

## Core Idea

Use **out-of-RTH features** (overnight + premarket) to predict **direction 1 hour after open** (09:30 → 10:30 ET).

**Why this might work better than Sleeve IM**:
1. **Shorter prediction window**: 1 hour vs 4.5 hours (less noise)
2. **Stronger signal**: Overnight gaps and premarket activity often set the tone for first hour
3. **More actionable**: Trade at open, exit at 10:30 - clean execution window
4. **Information asymmetry**: Most retail trades later in day; premarket is institutional-heavy

---

## Target Variable

```
y = sign(price_10:30 - price_09:30)

Binary classification:
  1 = first hour up (price_10:30 > price_09:30)
  0 = first hour down/flat
```

**Alternative targets** (to explore):
- Magnitude-weighted: `y = 1 if return > +0.3%, else 0 if return < -0.3%, else exclude`
- Multi-class: Strong up / Weak up / Flat / Weak down / Strong down

---

## Feature Categories

### 1. Overnight Gap Features
```python
# Gap from prior close to premarket
overnight_gap = (premarket_first_trade - prior_close) / prior_close
overnight_gap_to_open = (open_price - prior_close) / prior_close

# Gap direction persistence (does gap fill or extend?)
gap_direction = sign(overnight_gap)
```

### 2. Premarket Activity Features
```python
# Volume and liquidity
premarket_volume_total          # Total shares 04:00-09:30
premarket_volume_last_30min     # Shares 09:00-09:30 (institutional positioning)
premarket_dollar_volume         # $ volume (size-weighted)
premarket_trade_count           # Number of prints (activity level)

# Price action
premarket_range = (premarket_high - premarket_low) / premarket_vwap
premarket_return = (premarket_close - premarket_open) / premarket_open
premarket_vwap_vs_close = (premarket_vwap - prior_close) / prior_close

# Momentum within premarket
premarket_trend = slope of price from 04:00 to 09:30 (normalized)
last_30min_momentum = (price_09:30 - price_09:00) / price_09:00
```

### 3. Volatility Features
```python
# Realized volatility
premarket_volatility = std(1-min returns) * sqrt(minutes)
overnight_atr = average true range of premarket bars

# Relative to recent history
vol_ratio = premarket_volatility / trailing_5day_premarket_vol
range_percentile = where does today's premarket range rank vs last 20 days
```

### 4. Time/Calendar Features
```python
# Day of week (one-hot or cyclical encoding)
day_of_week_sin = sin(2 * pi * dow / 5)
day_of_week_cos = cos(2 * pi * dow / 5)

# Month effects
month_sin = sin(2 * pi * month / 12)
month_cos = cos(2 * pi * month / 12)

# Special days
is_monday = 1 if Monday else 0  # Weekend gap effect
is_friday = 1 if Friday else 0  # Position squaring
is_month_end = 1 if last 3 trading days of month else 0
is_quarter_end = 1 if last week of quarter else 0
days_to_earnings = days until next earnings (if known)
```

### 5. Cross-Asset Context (Optional)
```python
# Futures overnight action
es_overnight_return = (ES_09:30 - ES_prior_close) / ES_prior_close
nq_overnight_return = same for NQ futures

# VIX level
vix_level = VIX close prior day
vix_term_structure = VIX / VIX3M (contango/backwardation)

# Sector rotation (if trading single stocks)
sector_etf_premarket_return = XLK premarket return for tech stocks, etc.
```

### 6. Technical Context
```python
# Where is price relative to recent history
price_vs_20d_sma = (price - sma_20) / sma_20
price_vs_50d_sma = (price - sma_50) / sma_50
rsi_14 = RSI computed on prior day closes

# Recent momentum
return_1d = prior day return
return_5d = 5-day return
return_20d = 20-day return
```

---

## Neural Network Architecture

### Option A: Simple MLP (Baseline)
```python
import torch
import torch.nn as nn

class PremarketMLP(nn.Module):
    def __init__(self, input_dim=30, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

### Option B: Attention-Based (If Sequence Matters)
```python
class PremarketTransformer(nn.Module):
    """
    Treat premarket as sequence of 5-min bars (66 bars from 04:00-09:30).
    Use attention to weight which periods matter most.
    """
    def __init__(self, bar_features=5, seq_len=66, d_model=32, nhead=4):
        super().__init__()
        self.embed = nn.Linear(bar_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, bar_features)
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)
```

### Option C: Hybrid (Aggregated + Sequence)
```python
class HybridNN(nn.Module):
    """
    Combines:
    1. Aggregated features (overnight gap, premarket stats)
    2. Sequence model for premarket bar-by-bar data
    """
    def __init__(self, agg_dim=20, seq_len=66, bar_dim=5):
        super().__init__()
        # Sequence branch (LSTM or Transformer)
        self.lstm = nn.LSTM(bar_dim, 32, batch_first=True)

        # Aggregated branch
        self.agg_net = nn.Sequential(
            nn.Linear(agg_dim, 32),
            nn.ReLU(),
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, agg_features, bar_sequence):
        # agg_features: (batch, agg_dim)
        # bar_sequence: (batch, seq_len, bar_dim)
        _, (h_n, _) = self.lstm(bar_sequence)
        seq_out = h_n[-1]  # Last hidden state
        agg_out = self.agg_net(agg_features)
        combined = torch.cat([seq_out, agg_out], dim=1)
        return self.classifier(combined)
```

---

## Training Setup

### Loss Function
```python
# Binary cross-entropy (baseline)
criterion = nn.BCELoss()

# Focal loss (if class imbalance)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

### Regularization
```python
# L2 regularization via weight_decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Dropout already in architecture

# Early stopping based on validation loss
patience = 10
```

### Data Augmentation (Time Series)
```python
# Noise injection
def add_noise(features, std=0.01):
    return features + torch.randn_like(features) * std

# Feature dropout (randomly zero some features)
def feature_dropout(features, p=0.1):
    mask = torch.bernoulli(torch.ones_like(features) * (1 - p))
    return features * mask
```

---

## Evaluation Framework

### Metrics
```python
# Classification
accuracy = (pred_class == true_class).mean()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# Trading
def trading_metrics(predictions, returns, threshold=0.5, cost_bps=10):
    """
    predictions: probability of up
    returns: actual 09:30→10:30 returns
    """
    trades = predictions > threshold
    gross_pnl = (trades * returns).sum()
    costs = trades.sum() * 2 * cost_bps / 10000
    net_pnl = gross_pnl - costs

    # Sharpe (daily)
    daily_returns = np.where(trades, returns - 2 * cost_bps / 10000, 0)
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    return {
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'sharpe': sharpe,
        'trade_rate': trades.mean(),
    }
```

### Walk-Forward Validation
Same 6-fold structure as Sleeve IM to ensure consistency.

### Calibration
```python
# Reliability diagram
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)

# Brier score
brier = ((y_pred - y_true) ** 2).mean()
```

---

## Data Requirements

### Existing Data (Can Reuse)
- Premarket bars from Polygon backfill (`dsp100k/data/cache/`)
- RTH bars from `../data/stage1_raw/` (for 09:30, 10:30 prices)

### New Data Needed
- Prior day close (for overnight gap) - available in RTH data
- Futures data (ES, NQ) if using cross-asset features
- VIX data if using volatility context

---

## Kill Tests (Adjusted for Shorter Window)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Sharpe** | ≥ 0.5 | Higher bar since shorter window = more trades |
| **Accuracy** | > 52% | Slightly above coin flip (costs are lower for 1hr) |
| **Profit Factor** | > 1.2 | Winners meaningfully exceed losers |
| **Net Return** | > 0% | Must be profitable after costs |

### Cost Assumptions for 1-Hour Trade
- **Entry**: Market order at 09:30 open → ~5 bps slippage
- **Exit**: Market order at 10:30 → ~5 bps slippage
- **Total round-trip**: ~10 bps (half of Sleeve IM's 4.5-hour window)

---

## Implementation Plan

### Phase 1: Feature Engineering (1-2 days)
1. Build premarket feature extractor from existing cache
2. Compute overnight gaps from RTH data
3. Create train/val/test datasets with new target (09:30→10:30)

### Phase 2: Baseline MLP (1 day)
1. Train simple MLP on aggregated features
2. Run kill tests
3. If fails, analyze which features have predictive power

### Phase 3: Advanced Architectures (if baseline shows promise)
1. Add sequence model for bar-by-bar data
2. Add cross-asset features
3. Hyperparameter tuning

### Phase 4: Walk-Forward Validation
1. 6-fold validation same as Sleeve IM
2. Consistency check across regimes
3. Final kill decision

---

## Why This Might Succeed Where Sleeve IM Failed

| Factor | Sleeve IM | Sleeve NN |
|--------|-----------|-----------|
| **Prediction Window** | 4.5 hours | 1 hour |
| **Signal Source** | Same-day premarket | Overnight + premarket |
| **Model Complexity** | Logistic regression | Neural network |
| **Feature Richness** | 9 aggregated features | 30+ features + sequences |
| **Cost Impact** | 20 bps round-trip | ~10 bps round-trip |
| **Required Edge** | ~5 bps/trade | ~2.5 bps/trade |

The shorter window and richer features give more room for the model to find patterns, and lower execution costs mean smaller edge is sufficient.

---

## Risks

1. **Overfitting**: NN can memorize noise; need strong regularization
2. **Regime change**: Premarket patterns may shift over time
3. **Data leakage**: Must be careful with time-based features
4. **Execution**: 09:30 open can be volatile; execution quality varies

---

## Next Step

Want me to implement Phase 1 (feature engineering for the new 09:30→10:30 target)?
