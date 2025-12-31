# Implementation Plan: Sleeve IM (Intraday ML Long/Short)

**Version**: 1.0
**Date**: 2025-12-31
**Status**: PLANNING
**Specification**: `SPEC_INTRADAY_ML.md`
**Parent System**: DSP-100K

---

## Executive Summary

This document provides the engineering implementation plan for Sleeve IM, an intraday machine learning long/short equity strategy within the DSP-100K portfolio framework.

**Strategy Overview**:
- Uses morning price/volume patterns [01:30-10:30 ET] to predict afternoon returns
- Enters at 11:30 ET, exits flat via MOC orders by 15:50 ET
- Dollar-neutral long/short (3 longs + 3 shorts)
- Starts in shadow mode (signal generation only, no orders)

**Key Dependencies**:
- Polygon.io API (Starter tier minimum, $29/mo)
- IBKR integration (existing DSP-100K infrastructure)
- Neural network model (PyTorch)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Structure](#2-file-structure)
3. [Configuration Schema](#3-configuration-schema)
4. [Implementation Phases](#4-implementation-phases)
5. [Phase 1: Infrastructure](#phase-1-infrastructure-2-weeks)
6. [Phase 2: Data Pipeline](#phase-2-data-pipeline-2-weeks)
7. [Phase 3: Feature Engineering + Baseline](#phase-3-feature-engineering--baseline-2-weeks)
8. [Phase 4: Neural Network](#phase-4-neural-network-2-weeks)
9. [Phase 5: Paper Trading](#phase-5-paper-trading-4-weeks)
10. [Phase 6: Live Trading](#phase-6-live-trading-ongoing)
11. [Risk Management Integration](#risk-management-integration)
12. [Testing Strategy](#testing-strategy)
13. [Dependencies](#dependencies)

---

## 1. Architecture Overview

### 1.1 System Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DSP-100K Orchestrator                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Sleeve DM  │  │  Sleeve A   │  │  Sleeve B   │  │  Sleeve IM  │ │
│  │  (Monthly)  │  │  (KILLED)   │  │  (KILLED)   │  │   (Daily)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│        │                                                   │         │
│        └───────────────────┬───────────────────────────────┘         │
│                            │                                         │
│                   ┌────────┴────────┐                               │
│                   │  Risk Manager   │                               │
│                   └────────┬────────┘                               │
│                            │                                         │
│                   ┌────────┴────────┐                               │
│                   │ Order Executor  │                               │
│                   └────────┬────────┘                               │
│                            │                                         │
│                   ┌────────┴────────┐                               │
│                   │   IBKR Client   │                               │
│                   └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Sleeve IM Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Sleeve IM                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  01:30 ET                                                            │
│    │                                                                 │
│    ▼                                                                 │
│  ┌─────────────────┐                                                │
│  │ Polygon Fetcher │──── Minute bars (OHLCV + trade_count)          │
│  └────────┬────────┘                                                │
│           │                                                          │
│  10:30 ET │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │ Feature Builder │──── 54 features per symbol                     │
│  └────────┬────────┘                                                │
│           │                                                          │
│  10:31 ET │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  ML Inference   │──── Neural network (64-32-1)                   │
│  └────────┬────────┘                                                │
│           │                                                          │
│  11:25 ET │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │ Position Sizer  │──── Top-3 long, Top-3 short                    │
│  └────────┬────────┘                                                │
│           │                                                          │
│  11:30 ET │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │ Order Generator │──── Marketable limit orders                    │
│  └────────┬────────┘                                                │
│           │                                                          │
│  15:45 ET │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │   MOC Submitter │──── Market-on-close exits                      │
│  └─────────────────┘                                                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Daily Timeline

| Time (ET) | Event | Component |
|-----------|-------|-----------|
| 01:30 | Start data collection | PolygonFetcher |
| 10:30 | Feature window closes | FeatureBuilder |
| 10:31-11:25 | Feature computation + inference | MLEngine |
| 11:25-11:30 | Position sizing + pre-trade checks | PositionSizer |
| 11:30 | Entry execution | OrderExecutor |
| 11:35-15:40 | Intraday risk monitoring (every 5 min) | RiskMonitor |
| 15:45 | Submit MOC exits | MOCSubmitter |
| 16:00 | Verify flat | Reconciliation |

---

## 2. File Structure

```
dsp100k/
├── src/
│   └── dsp/
│       ├── sleeves/
│       │   ├── __init__.py             # Add SleeveIM export
│       │   ├── sleeve_dm.py            # Existing (reference)
│       │   └── sleeve_im.py            # NEW: Main sleeve class
│       │
│       ├── data/
│       │   ├── polygon_fetcher.py      # NEW: Polygon minute bars
│       │   └── minute_bar.py           # NEW: MinuteBar dataclass
│       │
│       ├── ml/                         # NEW: ML module
│       │   ├── __init__.py
│       │   ├── features.py             # Feature engineering
│       │   ├── model.py                # Neural network model
│       │   ├── inference.py            # Inference engine
│       │   └── training.py             # Training pipeline
│       │
│       ├── utils/
│       │   └── config.py               # Add SleeveIMConfig
│       │
│       └── execution/
│           └── orchestrator.py         # Add Sleeve IM integration
│
├── config/
│   ├── dsp100k_with_im.yaml           # NEW: Config with Sleeve IM enabled
│   └── universes/
│       └── sleeve_im_universe.yaml     # NEW: Sleeve IM universe
│
├── models/
│   └── sleeve_im/                      # NEW: ML model artifacts
│       ├── model_v1.pt                 # Trained model weights
│       └── scaler_v1.pkl               # Feature scaler
│
├── data/
│   └── sleeve_im/                      # NEW: Historical data cache
│       ├── minute_bars/                # Daily minute bar parquet files
│       └── features/                   # Precomputed features
│
├── tests/
│   └── sleeve_im/                      # NEW: Test suite
│       ├── test_polygon_fetcher.py
│       ├── test_features.py
│       ├── test_model.py
│       └── test_sleeve_im.py
│
└── scripts/
    └── sleeve_im/                      # NEW: Utility scripts
        ├── backfill_data.py            # Historical data backfill
        ├── train_model.py              # Model training
        ├── backtest.py                 # Backtest simulation
        └── shadow_mode.py              # Shadow mode runner
```

---

## 3. Configuration Schema

### 3.1 SleeveIMConfig Dataclass

Add to `dsp100k/src/dsp/utils/config.py`:

```python
@dataclass
class SleeveIMConfig:
    """Sleeve IM (Intraday ML Long/Short) configuration."""
    enabled: bool = False

    # Portfolio construction
    top_k: int = 3                          # Per-side: 3 long + 3 short
    edge_threshold: float = 0.02            # Trade if p >= 0.52 (long) or p <= 0.48 (short)
    target_gross_exposure: float = 0.00     # Start in shadow mode (0% exposure)
    dollar_neutral: bool = True
    max_net_exposure_pct_gross: float = 0.10  # Net <= 10% of gross

    # Time windows (ET)
    feature_window_start: str = "01:30"
    feature_window_end: str = "10:30"
    entry_time: str = "11:30"
    moc_submit_time: str = "15:45"

    # Risk limits
    max_single_name_pct: float = 0.03       # 3% of total NAV
    max_sleeve_gross_pct: float = 0.15      # 15% of total NAV
    drawdown_warning: float = 0.10          # 10% DD -> reduce exposure
    drawdown_hard_stop: float = 0.15        # 15% DD -> halt
    daily_loss_limit: float = 0.01          # 1% daily loss -> no new trades
    per_name_loss_limit: float = 0.02       # 2% per-name loss -> exit

    # Execution
    entry_slippage_cap_bps: int = 20
    order_timeout_seconds: int = 60

    # Data quality
    max_synthetic_bar_pct: float = 0.70
    min_premarket_dollar_volume: float = 500_000
    min_adv_dollar: float = 50_000_000
    min_market_cap: float = 10_000_000_000

    # Model
    model_path: str = "models/sleeve_im/model_v1.pt"
    scaler_path: str = "models/sleeve_im/scaler_v1.pkl"

    # Universe
    universe: List[str] = field(default_factory=lambda: [
        # Magnificent 7 (most liquid premarket)
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        # Benchmark ETFs (for context features)
        "SPY", "QQQ",
    ])

    # Polygon API
    polygon_api_key_env: str = "POLYGON_API_KEY"  # Env var name
```

### 3.2 Example Config YAML

`config/dsp100k_with_im.yaml`:

```yaml
general:
  nlv_target: 100000
  cash_buffer: 0.10
  margin_cap: 0.60
  risk_scale: 1.0
  allow_external_positions: false

  # Sleeve allocations
  sleeve_dm_nav_pct: 0.85          # 85% to Dual Momentum
  sleeve_im_nav_pct: 0.15          # 15% to Intraday ML (when enabled)

sleeve_dm:
  enabled: true
  # ... (existing config)

sleeve_im:
  enabled: false                   # Start disabled, enable for shadow mode
  top_k: 3
  edge_threshold: 0.02
  target_gross_exposure: 0.00      # Shadow mode: 0% exposure
  dollar_neutral: true
  max_net_exposure_pct_gross: 0.10

  # Universe (start conservative)
  universe:
    - AAPL
    - MSFT
    - NVDA
    - AMZN
    - GOOGL
    - META
    - TSLA
    - SPY
    - QQQ

# ... rest of config
```

---

## 4. Implementation Phases

| Phase | Duration | Focus | Gate |
|-------|----------|-------|------|
| **Phase 1** | 2 weeks | Infrastructure (config, skeleton) | Code compiles, tests pass |
| **Phase 2** | 2 weeks | Data pipeline (Polygon → minute bars) | Data quality metrics OK |
| **Phase 3** | 2 weeks | Features + baseline model | Sharpe > 0.2 (logistic) |
| **Phase 4** | 2 weeks | Neural network + tuning | Sharpe > 0.4 |
| **Phase 5** | 4 weeks | Paper trading (20+ days) | Matches backtest ± 20% |
| **Phase 6** | Ongoing | Live trading (scale up) | Risk metrics healthy |

---

## Phase 1: Infrastructure (2 weeks)

### Goals
- Add Sleeve IM configuration to DSP-100K
- Create skeleton files for all components
- Update orchestrator to recognize Sleeve IM
- Set up test infrastructure

### Tasks

#### 1.1 Configuration (Day 1-2)

- [ ] Add `SleeveIMConfig` dataclass to `config.py`
- [ ] Update `Config` dataclass to include `sleeve_im` field
- [ ] Add `sleeve_im_nav_pct` to `GeneralConfig`
- [ ] Update `_apply_env_overrides()` for Sleeve IM settings
- [ ] Update `load_config()` to parse Sleeve IM section
- [ ] Add environment variable `DSP_SLEEVE_IM_ENABLED`

#### 1.2 Skeleton Files (Day 3-5)

- [ ] Create `sleeve_im.py` with `SleeveIM` class skeleton
- [ ] Create `polygon_fetcher.py` with `PolygonFetcher` class skeleton
- [ ] Create `minute_bar.py` with `MinuteBar` dataclass
- [ ] Create `ml/__init__.py`, `features.py`, `model.py`, `inference.py`
- [ ] Update `sleeves/__init__.py` to export `SleeveIM`

#### 1.3 Orchestrator Integration (Day 6-8)

- [ ] Add `_sleeve_im: Optional[SleeveIM]` to `DailyOrchestrator`
- [ ] Add `sleeve_im_orders` to `DailyPlan`
- [ ] Add `sleeve_im_report` to `ExecutionResult`
- [ ] Add Sleeve IM initialization in `initialize()`
- [ ] Add Sleeve IM to daily execution flow (with enable check)

#### 1.4 Test Infrastructure (Day 9-10)

- [ ] Create `tests/sleeve_im/` directory
- [ ] Add `conftest.py` with common fixtures
- [ ] Create placeholder test files
- [ ] Verify all imports work
- [ ] Add to CI pipeline

### Phase 1 Deliverables

| Deliverable | File | Status |
|-------------|------|--------|
| SleeveIMConfig | `config.py` | 🔲 |
| SleeveIM skeleton | `sleeve_im.py` | 🔲 |
| PolygonFetcher skeleton | `polygon_fetcher.py` | 🔲 |
| MinuteBar dataclass | `minute_bar.py` | 🔲 |
| ML module skeleton | `ml/*.py` | 🔲 |
| Orchestrator updates | `orchestrator.py` | 🔲 |
| Test infrastructure | `tests/sleeve_im/` | 🔲 |

### Phase 1 Gate Criteria

✅ All skeleton files compile without errors
✅ `python -c "from dsp.sleeves import SleeveIM"` succeeds
✅ Config loads with `sleeve_im` section
✅ All placeholder tests pass
✅ Orchestrator initializes with Sleeve IM disabled

---

## Phase 2: Data Pipeline (2 weeks)

### Goals
- Implement Polygon.io data fetcher
- Build minute bar construction with carry-forward
- Create data quality monitoring
- Backfill historical data (2020-2024)

### Tasks

#### 2.1 Polygon API Client (Day 1-4)

- [ ] Set up Polygon API authentication (env var `POLYGON_API_KEY`)
- [ ] Implement `get_aggregates()` for 1-minute bars
- [ ] Handle extended hours data (as available on tier)
- [ ] Implement rate limiting (5 req/min for Starter)
- [ ] Add caching layer (local parquet files)
- [ ] Handle API errors gracefully

```python
# Example API call
class PolygonFetcher:
    async def get_minute_bars(
        self,
        symbol: str,
        date: date,
        start_time: time = time(1, 30),
        end_time: time = time(10, 30),
    ) -> List[MinuteBar]:
        """
        Fetch 1-minute bars from Polygon.io.
        """
        ...
```

#### 2.2 Minute Bar Construction (Day 5-7)

- [ ] Implement `MinuteBar` dataclass with all fields
- [ ] Build complete time grid [01:30, 10:30] ET (541 minutes)
- [ ] Implement carry-forward logic for sparse premarket
- [ ] Track synthetic bar metadata (`is_synthetic`, `seconds_since_last_trade`)
- [ ] Handle gaps > 1 hour gracefully

```python
@dataclass
class MinuteBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int
    is_synthetic: bool
    seconds_since_last_trade: float
    last_trade_timestamp: Optional[datetime]
```

#### 2.3 Data Quality Monitoring (Day 8-10)

- [ ] Calculate synthetic bar percentage per symbol-day
- [ ] Detect suspicious patterns (constant prices, zero volume)
- [ ] Flag outlier prices (> 3 std from rolling mean)
- [ ] Generate daily data quality report
- [ ] Alert on data quality below threshold

#### 2.4 Historical Backfill (Day 11-14)

- [ ] Script to backfill 2020-2024 data
- [ ] Store in parquet format: `data/sleeve_im/minute_bars/{symbol}/{YYYY-MM-DD}.parquet`
- [ ] Handle rate limits (batch with delays)
- [ ] Verify data completeness
- [ ] Generate backfill report

### Phase 2 Deliverables

| Deliverable | File | Status |
|-------------|------|--------|
| Polygon fetcher | `polygon_fetcher.py` | 🔲 |
| Minute bar builder | `minute_bar.py` | 🔲 |
| Data quality monitor | `data_quality.py` | 🔲 |
| Backfill script | `scripts/sleeve_im/backfill_data.py` | 🔲 |
| Historical data | `data/sleeve_im/minute_bars/` | 🔲 |

### Phase 2 Gate Criteria

✅ Can fetch minute bars for all universe symbols
✅ Synthetic bar % < 70% for trading hours (04:00-10:30)
✅ 2020-2024 data backfilled for all symbols
✅ Data quality metrics dashboard working
✅ Unit tests for fetcher and bar construction pass

---

## Phase 3: Feature Engineering + Baseline (2 weeks)

### Goals
- Implement all 54 features from spec
- Build training/validation pipeline
- Train baseline logistic regression
- Achieve Sharpe > 0.2 in backtest

### Tasks

#### 3.1 Feature Engineering (Day 1-6)

Implement all feature groups from SPEC_INTRADAY_ML.md Section 5:

```python
def compute_features(bars: List[MinuteBar], spy_bars: List[MinuteBar]) -> Dict[str, float]:
    """
    Compute 54 features for one symbol-day.
    """
    features = {}

    # Group 1: Returns (4 features)
    features["ret_overnight"] = ...
    features["ret_premarket"] = ...
    features["ret_first_hour"] = ...
    features["ret_total_feature_window"] = ...

    # Group 2: Volume (6 features)
    features["volume_premarket_total"] = ...
    # ... etc

    # Group 3: Volatility (4 features)
    # Group 4: Price patterns (6 features)
    # Group 5: Volume patterns (4 features)
    # Group 6: Momentum (4 features)
    # Group 7: Relative (4 features)
    # Group 8: Microstructure (6 features)
    # Group 9: Staleness (6 features)
    # Group 10: Time/calendar (10 features)

    return features
```

#### 3.2 Label Computation (Day 7-8)

- [ ] Compute afternoon return: `close / price_at_1130 - 1`
- [ ] Create binary label: `1` if return > 0, else `0`
- [ ] Handle edge cases (halts, missing close)

#### 3.3 Training Pipeline (Day 9-12)

- [ ] Create train/val/test splits (2020-2022 / 2023 / 2024)
- [ ] Implement feature standardization (sklearn StandardScaler)
- [ ] Train logistic regression baseline
- [ ] Calculate performance metrics (Sharpe, accuracy, profit factor)
- [ ] Implement walk-forward validation

#### 3.4 Baseline Model Evaluation (Day 13-14)

- [ ] Run backtest simulation with transaction costs (10 bps per side)
- [ ] Calculate all metrics from Section 14 (Kill Test Criteria)
- [ ] Document feature importance
- [ ] Identify features to drop/add

### Phase 3 Deliverables

| Deliverable | File | Status |
|-------------|------|--------|
| Feature pipeline | `ml/features.py` | 🔲 |
| Label computation | `ml/labels.py` | 🔲 |
| Training pipeline | `ml/training.py` | 🔲 |
| Baseline model | `models/sleeve_im/baseline_logistic.pkl` | 🔲 |
| Backtest script | `scripts/sleeve_im/backtest.py` | 🔲 |

### Phase 3 Gate Criteria

✅ All 54 features compute correctly
✅ Train/val/test splits created
✅ Baseline logistic regression trained
✅ Backtest Sharpe > 0.2 (2022-2024)
✅ Feature importance documented

---

## Phase 4: Neural Network (2 weeks)

### Goals
- Implement neural network model
- Hyperparameter tuning
- Achieve Sharpe > 0.4 in backtest

### Tasks

#### 4.1 Neural Network Implementation (Day 1-5)

```python
import torch
import torch.nn as nn

class SleeveIMModel(nn.Module):
    """
    Neural network for Sleeve IM predictions.
    """
    def __init__(
        self,
        input_dim: int = 54,
        hidden_layers: List[int] = [64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

#### 4.2 Training Loop (Day 6-8)

- [ ] Implement training loop with early stopping
- [ ] Use BCE loss with class weights (handle imbalance)
- [ ] Implement learning rate scheduling
- [ ] Save best model checkpoint
- [ ] Track training metrics (loss, accuracy)

#### 4.3 Hyperparameter Search (Day 9-11)

Search space:
- Hidden layers: [32], [64, 32], [128, 64, 32]
- Dropout: 0.1, 0.2, 0.3
- Learning rate: 1e-4, 1e-3, 1e-2
- Batch size: 32, 64, 128

- [ ] Implement grid search or Optuna optimization
- [ ] Use 2023 as validation year
- [ ] Select best hyperparameters

#### 4.4 Final Model Training (Day 12-14)

- [ ] Train on 2020-2023, test on 2024
- [ ] Run full backtest with transaction costs
- [ ] Verify all kill test criteria met
- [ ] Document model performance

### Phase 4 Deliverables

| Deliverable | File | Status |
|-------------|------|--------|
| NN model class | `ml/model.py` | 🔲 |
| Training loop | `ml/training.py` | 🔲 |
| Hyperparameter search | `scripts/sleeve_im/hp_search.py` | 🔲 |
| Trained model | `models/sleeve_im/model_v1.pt` | 🔲 |
| Feature scaler | `models/sleeve_im/scaler_v1.pkl` | 🔲 |

### Phase 4 Gate Criteria

✅ Neural network model implemented
✅ Hyperparameters optimized
✅ Backtest Sharpe > 0.4 (2022-2024)
✅ Max drawdown < 15%
✅ Avg trade after costs > 5 bps
✅ Model artifacts saved

---

## Phase 5: Paper Trading (4 weeks)

### Goals
- Full integration with DSP-100K orchestrator
- Shadow mode first (signals only)
- Paper trading for 20+ days
- Verify backtest correlation > 0.7

### Tasks

#### 5.1 Full SleeveIM Implementation (Week 1)

- [ ] Complete `SleeveIM` class with all methods
- [ ] Implement `generate_signals()` method
- [ ] Implement `generate_adjustment()` method
- [ ] Implement `get_target_orders()` method
- [ ] Add position tracking

```python
class SleeveIM:
    async def generate_signals(
        self,
        *,
        as_of_date: date,
    ) -> List[SleeveIMSignal]:
        """
        Generate trading signals for the day.

        Called at 10:31 ET after feature window closes.
        """
        ...

    async def generate_adjustment(
        self,
        *,
        sleeve_nav: float,
        prices: Dict[str, float],
        signals: List[SleeveIMSignal],
    ) -> SleeveIMAdjustment:
        """
        Convert signals to position targets.

        Called at 11:25 ET before entry.
        """
        ...
```

#### 5.2 Shadow Mode (Week 1-2)

- [ ] Run with `target_gross_exposure: 0.00`
- [ ] Log all signals and hypothetical trades
- [ ] Verify timing (feature cutoff, entry timing)
- [ ] Compare signals to backtest expectations
- [ ] Fix any discrepancies

#### 5.3 Paper Trading Integration (Week 2-3)

- [ ] Enable Sleeve IM in orchestrator
- [ ] Set `target_gross_exposure: 0.10` (10% of sleeve NAV)
- [ ] Execute through IBKR paper account
- [ ] Submit MOC exits by 15:50 ET
- [ ] Verify positions flat by 16:00 ET

#### 5.4 Monitoring & Validation (Week 3-4)

- [ ] Track daily P&L
- [ ] Calculate correlation with backtest
- [ ] Monitor execution slippage
- [ ] Track fill rates
- [ ] Document operational issues

### Phase 5 Deliverables

| Deliverable | File | Status |
|-------------|------|--------|
| Complete SleeveIM | `sleeve_im.py` | 🔲 |
| Shadow mode script | `scripts/sleeve_im/shadow_mode.py` | 🔲 |
| Paper trading log | `data/sleeve_im/paper_trading_log.csv` | 🔲 |
| Performance report | `reports/sleeve_im_paper_trading.md` | 🔲 |

### Phase 5 Gate Criteria

✅ 20+ paper trading days completed
✅ Execution slippage < 15 bps avg
✅ Fill rate > 95%
✅ Daily return correlation with backtest > 0.7
✅ 0 critical operational errors
✅ Risk governor functioning correctly

---

## Phase 6: Live Trading (Ongoing)

### Goals
- Gradual scale-up from 0% to target allocation
- Continuous monitoring
- Model retraining schedule

### Tasks

#### 6.1 Initial Live (Scale = 0.05)

- [ ] Set `target_gross_exposure: 0.05` (5% of sleeve NAV)
- [ ] Run for 10 trading days
- [ ] Verify no operational issues
- [ ] Compare to paper trading performance

#### 6.2 Scale-Up Schedule

| Week | Gross Exposure | Notes |
|------|----------------|-------|
| 1-2 | 5% | Initial validation |
| 3-4 | 7.5% | If no issues |
| 5-6 | 10% | If Sharpe > 0.3 |
| 7-8 | 12.5% | If Sharpe > 0.4 |
| 9+ | 15% (target) | Full allocation |

#### 6.3 Ongoing Monitoring

- [ ] Daily performance report
- [ ] Weekly risk metrics review
- [ ] Monthly model performance review
- [ ] Quarterly model retraining (if degradation)

#### 6.4 Kill Switches

Automatic halt triggers:
- Rolling 60-day Sharpe < -0.3
- Max drawdown > 15%
- 3 consecutive days of losses > 1%

---

## Risk Management Integration

### Integration with DSP-100K Risk Manager

```python
# In orchestrator.py

async def _run_sleeve_im(self) -> Tuple[List[Dict], Optional[SleeveIMAdjustment]]:
    """
    Run Sleeve IM signal generation and position sizing.
    """
    if not self.config.sleeve_im.enabled:
        return [], None

    # Check portfolio risk limits
    risk_status = await self._risk.check_portfolio_status()
    if risk_status.halted:
        logger.warning("Sleeve IM: Portfolio halted, skipping")
        return [], None

    # Check sleeve-specific limits
    sleeve_nav = self._get_sleeve_nav("IM")
    if sleeve_nav <= 0:
        return [], None

    # Generate signals
    signals = await self._sleeve_im.generate_signals(as_of_date=...)

    # Generate adjustment with risk scaling
    scale = risk_status.scale_factor * self.config.general.risk_scale
    adjustment = await self._sleeve_im.generate_adjustment(
        sleeve_nav=sleeve_nav * scale,
        prices=self._prices,
        signals=signals,
    )

    # Get orders
    orders = self._sleeve_im.get_target_orders(adjustment)

    return orders, adjustment
```

### Sleeve IM Risk Governor

See SPEC_INTRADAY_ML.md Section 10 for `SleeveIMRiskGovernor` implementation.

---

## Testing Strategy

### Unit Tests

```
tests/sleeve_im/
├── test_polygon_fetcher.py      # Data fetching
├── test_minute_bar.py           # Bar construction
├── test_features.py             # Feature engineering
├── test_model.py                # Model inference
├── test_sleeve_im.py            # Main sleeve logic
└── test_risk_governor.py        # Risk management
```

### Integration Tests

```
tests/sleeve_im/
├── test_integration_polygon.py  # Polygon API integration
├── test_integration_ibkr.py     # IBKR execution
└── test_integration_orchestrator.py  # Full orchestrator
```

### Backtest Validation

- Compare backtest results to paper trading
- Track correlation coefficient
- Document any discrepancies

---

## Dependencies

### Python Packages

```
# Add to requirements.txt
polygon-api-client>=1.13.0    # Polygon.io client
torch>=2.0.0                   # PyTorch for NN
scikit-learn>=1.3.0           # Baseline model, scaler
pyarrow>=14.0.0               # Parquet file handling
```

### External Services

| Service | Tier | Monthly Cost | Purpose |
|---------|------|--------------|---------|
| Polygon.io | Starter | $29 | Minute bars (15-min delay OK) |
| IBKR | Existing | $0 | Execution |

### Environment Variables

```bash
export POLYGON_API_KEY="your_api_key"
export DSP_SLEEVE_IM_ENABLED="true"
```

---

## Timeline Summary

| Phase | Start | End | Duration |
|-------|-------|-----|----------|
| Phase 1: Infrastructure | Week 1 | Week 2 | 2 weeks |
| Phase 2: Data Pipeline | Week 3 | Week 4 | 2 weeks |
| Phase 3: Features + Baseline | Week 5 | Week 6 | 2 weeks |
| Phase 4: Neural Network | Week 7 | Week 8 | 2 weeks |
| Phase 5: Paper Trading | Week 9 | Week 12 | 4 weeks |
| Phase 6: Live Trading | Week 13+ | Ongoing | - |

**Total Time to Live Trading**: ~12 weeks (3 months)

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-31 | Claude | Initial plan |

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Strategy Owner | | | |
| Engineering Lead | | | |
| Risk Manager | | | |
