# Downloaded Market Data Inventory

**Purpose**: Track all purchased/downloaded historical market data for backtesting and research

**Last Updated**: 2026-01-07

---

## Databento - CME Micro Futures (GLBX.MDP3)

**Purchase Date**: 2026-01-07
**Provider**: Databento (https://databento.com)
**Dataset**: GLBX.MDP3 (CME Globex Level 1 Market Data)
**Format**: DBN + zstd compression
**Storage Location**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/databento/GLBX-20260107-5JE6UNGTJF/`

### **Instruments Covered**

| Symbol | Name | Contract Type | Date Range | Bars Count |
|--------|------|---------------|------------|------------|
| **MES.FUT** | Micro E-mini S&P 500 | Continuous (front-month roll) | 2022-01-05 to 2025-03-31 | 320,128 |
| **MNQ.FUT** | Micro E-mini Nasdaq-100 | Continuous (front-month roll) | 2022-01-05 to 2025-03-31 | 320,130 |

### **Data Specifications**

- **Schema**: ohlcv-1m (1-minute OHLCV bars)
- **Trading Days**: 834 days (2022-2025)
- **Session**: RTH only (09:30-16:00 ET)
- **Columns**: timestamp, open, high, low, close, volume, contract
- **Quality**: Zero missing bars, complete continuous series

### **Conversion to Parquet**

**Script**: `dsp100k/src/dsp/data/databento_orb_importer.py` (395 lines)
**Output Location**: `dsp100k/data/orb/`
**Output Files**:
- `MES_1min_2022-01-01_2025-03-31.parquet` (320,128 bars)
- `MNQ_1min_2022-01-01_2025-03-31.parquet` (320,130 bars)

**Parquet Schema**:
```
timestamp: datetime64[ns] (ET timezone-aware)
open: float64
high: float64
low: float64
close: float64
volume: int64
contract: object (e.g., "MESH5", "MNQM5")
```

### **Usage**

**Primary Use Case**: Sleeve ORB (Opening Range Breakout) backtest
**Backtester**: `dsp100k/src/dsp/backtest/orb_futures.py`
**Results**: See `dsp100k/docs/SLEEVE_ORB_KILL_TEST_RESULTS.md`
**Verdict**: Strategy killed (Sharpe 0.23, 2/6 folds passing)

**Commands**:
```bash
# Convert DBN to Parquet (already done)
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
PYTHONPATH=src python -m dsp.data.databento_orb_importer \
  --input-dir data/databento/GLBX-20260107-5JE6UNGTJF \
  --output-dir data/orb

# Run ORB backtest (already completed)
PYTHONPATH=src python -m dsp.backtest.orb_futures \
  --data-dir data/orb \
  --slippage 1 \
  --output data/orb/walk_forward_results.json
```

### **Cost Information**

- **Dataset Size**: ~200MB compressed DBN files
- **Pricing Model**: Pay-per-use (Databento)
- **Cost**: [User to fill in actual cost]
- **License**: Single-user research license

### **Data Quality Validation**

✅ **Completeness**: 834/834 trading days present (100%)
✅ **Continuity**: Zero gaps in 1-minute bars during RTH
✅ **Integrity**: No NaN values, all OHLC relationships valid (high ≥ low, etc.)
✅ **Contract Rolls**: Clean front-month transitions at proper roll dates
✅ **Volume**: Non-zero volume on all trading bars

### **Key Insights from Data**

**MES (Micro S&P 500)**:
- Average daily bars: ~384 bars (390 minutes RTH ÷ 1 min)
- Typical OR width: 4-8 points ($20-40 per contract)
- ATR(14): ~50 points ($250 per contract)

**MNQ (Micro Nasdaq-100)**:
- Average daily bars: ~384 bars
- Typical OR width: 15-30 points ($30-60 per contract)
- ATR(14): ~180 points ($360 per contract)

### **Historical Context**

**Period Coverage**:
- **2022**: High volatility (Fed hiking cycle begins)
- **2023**: Market recovery, trend formation
- **2024**: Consolidation and rotation
- **2025 Q1**: Recent market conditions

**Market Regimes Captured**:
- Trending markets: 2023 H2 (Folds 3, 5 profitable)
- Choppy markets: 2022 H2, 2023 Q1, 2024 Q1, 2025 Q1 (Folds 1, 2, 4, 6 unprofitable)

---

## Data Inventory Summary

| Dataset | Instruments | Date Range | Bars | Status | Primary Use |
|---------|-------------|------------|------|--------|-------------|
| **Databento GLBX.MDP3** | MES, MNQ | 2022-2025 | 320K each | ✅ Complete | Sleeve ORB (killed) |

---

## Future Data Needs

### **For V15 Enhancement**
- [ ] MGC (Micro Gold) 1-minute data (if pursuing metals sleeve)
- [ ] MCL (Micro Crude Oil) 1-minute data (if pursuing energy sleeve)
- [ ] M6E (Micro Euro FX) 1-minute data (if pursuing FX sleeve)

### **For Equities ORB Research** (if pursued)
- [ ] SPY constituents 1-minute data (2022-2025)
- [ ] High-liquidity mega-caps (AAPL, MSFT, NVDA, GOOGL, AMZN, etc.)
- [ ] Estimated volume: ~200 stocks × 320K bars = 64M bars

### **For Volatility Risk Premium (VRP) Sleeve** (next candidate)
- [ ] VIX futures chain 1-minute data
- [ ] SPX options implied volatility surface (if pursuing vol surface arbitrage)
- [ ] VVIX (VIX of VIX) for regime detection

---

## Data Management Notes

### **Storage Locations**
- **Raw DBN Files**: `dsp100k/data/databento/` (keep for audit trail)
- **Converted Parquet**: `dsp100k/data/orb/` (working format)
- **Backup**: [User to specify backup location if any]

### **Retention Policy**
- **Raw DBN**: Permanent retention (source of truth)
- **Parquet**: Permanent retention (working files)
- **Backtest Results**: Permanent retention in `dsp100k/data/orb/walk_forward_results.json`

### **Access & Licensing**
- **License Type**: Single-user research license (non-redistributable)
- **Permitted Uses**: Backtesting, research, algorithm development
- **Prohibited Uses**: Redistribution, commercial resale of raw data

---

## Related Documentation

- **ORB Implementation**: [SLEEVE_ORB_IMPLEMENTATION_STATUS.md](./SLEEVE_ORB_IMPLEMENTATION_STATUS.md)
- **Kill Test Results**: [SLEEVE_ORB_KILL_TEST_RESULTS.md](./SLEEVE_ORB_KILL_TEST_RESULTS.md)
- **Kill Test Summary**: [SLEEVE_KILL_TEST_SUMMARY.md](./SLEEVE_KILL_TEST_SUMMARY.md)
- **Data Importer**: `dsp100k/src/dsp/data/databento_orb_importer.py`
- **Backtester**: `dsp100k/src/dsp/backtest/orb_futures.py`

---

**Maintained By**: Development Team
**Contact**: [Update with appropriate contact if needed]
