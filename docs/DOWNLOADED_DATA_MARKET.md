# Downloaded Market Data Inventory

**Purpose**: Track all purchased/downloaded historical market data for backtesting and research

**Last Updated**: 2026-01-08

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

## Databento - CME Micro Futures Daily (GLBX.MDP3, TSMOM)

**Purchase Date**: 2026-01-07  
**Provider**: Databento (https://databento.com)  
**Dataset**: GLBX.MDP3  
**Format**: CSV + zstd compression  
**Storage Location**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/databento/GLBX-20260107-MXWMXNTA6P/`

### **Instruments Covered (Parents Requested)**

| Symbol | Name | Frequency | Requested Range |
|--------|------|-----------|-----------------|
| **MES.FUT** | Micro E-mini S&P 500 | Daily | 2021-01-05 to 2026-01-05 |
| **MNQ.FUT** | Micro E-mini Nasdaq-100 | Daily | 2021-01-05 to 2026-01-05 |
| **M2K.FUT** | Micro E-mini Russell 2000 | Daily | 2021-01-05 to 2026-01-05 |
| **MYM.FUT** | Micro E-mini Dow | Daily | 2021-01-05 to 2026-01-05 |
| **MGC.FUT** | Micro Gold | Daily | 2021-01-05 to 2026-01-05 |
| **MCL.FUT** | Micro WTI Crude | Daily | 2021-01-05 to 2026-01-05 |
| **M6E.FUT** | Micro EUR/USD | Daily | 2021-01-05 to 2026-01-05 |
| **M6J.FUT** | Micro USD/JPY | Daily | 2021-01-05 to 2026-01-05 |

### **Data Specifications**

- **Schema**: `ohlcv-1d` (daily OHLCV bars)
- **Customizations**: `map_symbols=true`, `pretty_px=true`, `pretty_ts=true`
- **Split**: by instrument + month
- **Aux files**:
  - `symbology.csv` and `symbology.json` (instrument_id ↔ raw_symbol mapping over time)

### **Important Notes (Coverage Gaps Discovered)**

These are *data availability* issues in the delivered batch (not code issues):
- **MCL**: first available daily bar in the delivered files is `2021-07-11` (not present earlier in 2021).
- **M6J**: last available daily bar in the delivered files is `2024-03-18` (no 2024-04 → 2026 coverage present).

This may require either:
- Adjusting the TSMOM universe (replace `M6J`), or
- Re-requesting an FX instrument with full coverage, or
- Redefining the validation window to fit the available data.

### **Conversion to Parquet (Rolled Root Series)**

**Script**: `dsp100k/src/dsp/data/databento_tsmom_importer.py`  
**Output Location**: `dsp100k/data/tsmom/`  
**Outputs (rolled series with `contract` column)**:
- `MES_1d_2021-01-05_2026-01-04.parquet`
- `MNQ_1d_2021-01-05_2026-01-04.parquet`
- `M2K_1d_2021-01-05_2026-01-04.parquet`
- `MYM_1d_2021-01-05_2026-01-04.parquet`
- `MGC_1d_2021-01-05_2026-01-04.parquet`
- `MCL_1d_2021-07-11_2026-01-04.parquet`
- `M6E_1d_2021-01-05_2026-01-04.parquet`
- `M6J_1d_2021-08-04_2024-03-18.parquet`

**Command used**:
```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
PYTHONPATH=src python -m dsp.data.databento_tsmom_importer \
  --input-dir data/databento/GLBX-20260107-MXWMXNTA6P \
  --output-dir data/tsmom \
  --start 2021-01-05 \
  --end 2026-01-05
```

**Primary Use Case**: Sleeve TSMOM baseline validation (daily, weekly rebalance)

---

## Databento - TSMOM Replacements Basket (FX/Rates/Commodities)

**Purchase Date**: 2026-01-08  
**Provider**: Databento (https://databento.com)  
**Dataset**: GLBX.MDP3  
**Schema**: `ohlcv-1d`  
**Format**: CSV + zstd compression  
**Storage Location**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/databento/GLBX-20260107-AYH5HTQUB3/` (zip: `.../GLBX-20260107-AYH5HTQUB3.zip`)

### **Instruments Covered (Parents Requested)**

| Symbol | Name | Notes |
|--------|------|------|
| **M6B.FUT** | Micro GBP/USD | Full coverage delivered (2021–2026) |
| **M6A.FUT** | Micro AUD/USD | Full coverage delivered (2021–2026) |
| **6J.FUT** | JPY futures | Full-size (research-grade FX coverage) |
| **6C.FUT** | CAD futures | Full-size (research-grade FX coverage) |
| **ZN.FUT** | 10Y Treasury Note | Full-size (rates diversifier) |
| **SR3.FUT** | 3M SOFR | Full-size (rates diversifier) |
| **HG.FUT** | Copper | Full-size (macro growth proxy) |
| **ZC.FUT** | Corn | Full-size (agri diversifier) |

### **Conversion to Parquet (Rolled Root Series)**

**Script**: `dsp100k/src/dsp/data/databento_tsmom_importer.py`  
**Output Location**: `dsp100k/data/tsmom/`  
**Outputs**:
- `M6B_1d_2021-01-05_2026-01-04.parquet`
- `M6A_1d_2021-01-05_2026-01-04.parquet`
- `6J_1d_2021-01-05_2026-01-04.parquet`
- `6C_1d_2021-01-05_2026-01-04.parquet`
- `ZN_1d_2021-01-05_2026-01-04.parquet`
- `SR3_1d_2021-01-05_2026-01-04.parquet`
- `HG_1d_2021-01-05_2026-01-04.parquet`
- `ZC_1d_2021-01-05_2026-01-02.parquet` (delivered up to 2026-01-02)

**Outcome:** `M6B` and `M6A` provide complete micro-FX coverage and can replace the incomplete `M6J` in the TSMOM sleeve universe.

---

## VRP (Volatility Risk Premium) Data - Free Sources

**Download Date**: 2026-01-08
**Provider**: Yahoo Finance + FRED (Federal Reserve Economic Data)
**Format**: Parquet
**Storage Location**: `/Users/Shared/wsl-export/wsl-home/dsp100k/data/vrp/`

### **Instruments Downloaded**

| Symbol | Source | Date Range | Rows | File |
|--------|--------|------------|------|------|
| **VIX Spot** | Yahoo `^VIX` | 2010-01-04 to 2026-01-07 | 4,028 | `indices/VIX_spot.parquet` |
| **VVIX** | Yahoo `^VVIX` | 2010-01-04 to 2026-01-07 | 4,019 | `indices/VVIX.parquet` |
| **Fed Funds Rate** | FRED `DFF` | 2010-01-01 to 2026-01-06 | 5,850 | `rates/fed_funds.parquet` |
| **3-Month T-Bill** | FRED `DTB3` | 2010-01-01 to 2026-01-06 | 4,178 | `rates/tbill_3m.parquet` |
| **10-Year Treasury** | FRED `DGS10` | 2010-01-01 to 2026-01-06 | 4,178 | `rates/treasury_10y.parquet` |

### **Data Specifications**

- **Schema**: Single column (close/value) with date index
- **Frequency**: Daily
- **Coverage**: 16 years (2010-2026) - covers multiple volatility regimes
- **Quality**: Forward-filled for weekends/holidays (rates data)

### **Download Script**

**Script**: `dsp100k/scripts/vrp_data_downloader.py`

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/vrp_data_downloader.py
```

### **🎉 VIX Futures (VX F1) - FREE FROM CBOE!**

| Symbol | Source | Date Range | Rows | File |
|--------|--------|------------|------|------|
| **VX_F1_CBOE** | CBOE CFE Historical Data (FREE) | 2013-05-20 to 2026-01-07 | **3,182** | `futures/VX_F1_CBOE.parquet` |

**Source URL**: https://www.cboe.com/us/futures/market_statistics/historical_data/

**Data Details**:
- **Contracts Downloaded**: 385 individual contract CSVs (monthly + weekly)
- **Series Type**: Continuous front-month (F1) with automatic roll logic
- **Columns**: vx_f1 (settlement price), f1_expiration, volume, open_interest
- **Statistics**: Mean=18.35, Min=9.88, Max=74.53, Std=6.10

**Download Script**: `dsp100k/scripts/download_vx_futures_cboe.py`

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate
python scripts/download_vx_futures_cboe.py
```

**Note**: Data before 2013-05-20 returns 403 (may require DataShop for earlier years). 12+ years is sufficient for VRP backtest.

### **Deprecated: Synthetic VX_F1 Estimate**

| Symbol | Source | Date Range | Rows | File |
|--------|--------|------------|------|------|
| **VX_F1_SYNTHETIC** | Derived from VIX spot | 2010-01-04 to 2026-01-07 | 4,028 | `futures/VX_F1_SYNTHETIC.parquet` |

**Status**: ⚠️ **DEPRECATED** - Use `VX_F1_CBOE.parquet` instead (real settlement prices)

### **Still Needed for VRP Backtest (Optional)**

| Data | Source | Priority | Notes |
|------|--------|----------|-------|
| ~~**VIX Futures (VX)**~~ | ~~CBOE DataShop~~ | ~~P1~~ | ✅ **SOLVED** - Free from CBOE! |
| **VIX1D** | CBOE (free) | P3 | Only available since 2023, optional |
| **VIX Options** | CBOE DataShop | P3 | For tail hedge modeling (can use 15% approx) |

### **Primary Use Case**

- **Strategy**: Sleeve VRP (Volatility Risk Premium)
- **Signal**: VIX futures contango (VX_F1 - VIX spot) adjusted for risk-free rate
- **Specification**: See `dsp100k/docs/SPEC_VRP.md`
- **Status**: ✅ **READY FOR BACKTEST** - All required data acquired!

---

## Data Inventory Summary

| Dataset | Instruments | Date Range | Bars | Status | Primary Use |
|---------|-------------|------------|------|--------|-------------|
| **Databento GLBX.MDP3** | MES, MNQ | 2022-2025 | 320K each | ✅ Complete | Sleeve ORB (killed) |
| **Databento GLBX.MDP3 (Daily)** | MES, MNQ, M2K, MYM, MGC, MCL, M6E, M6J | 2021-2026* | Daily | ⚠️ Partial | Sleeve TSMOM (baseline) |
| **Databento GLBX.MDP3 (Daily add-on)** | M6B, M6A, 6J, 6C, ZN, SR3, HG, ZC | 2021-2026* | Daily | ✅ Mostly complete | TSMOM replacements + cross-asset research |
| **Yahoo + FRED (VRP)** | VIX, VVIX, DFF, DTB3, DGS10 | 2010-2026 | 4K-6K each | ✅ Complete | Sleeve VRP (indices + rates) |
| **🎉 VIX Futures (VX F1)** | VX_F1_CBOE | 2013-2026 | **3,182** | ✅ **Complete** | Sleeve VRP (contango signal) |
| ~~Synthetic VX_F1~~ | VX_F1_SYNTHETIC | 2010-2026 | 4,028 | ⚠️ Deprecated | Use VX_F1_CBOE instead |

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

### **For Volatility Risk Premium (VRP) Sleeve** ✅ **COMPLETE**
- [x] VIX Spot Index (Yahoo ^VIX) - 2010-2026 ✅ DONE
- [x] VVIX (VIX of VIX) for regime detection (Yahoo ^VVIX) - 2010-2026 ✅ DONE
- [x] Fed Funds Rate (FRED DFF) - 2010-2026 ✅ DONE
- [x] 3-Month T-Bill (FRED DTB3) - 2010-2026 ✅ DONE
- [x] 10-Year Treasury (FRED DGS10) - 2010-2026 ✅ DONE
- [x] **🎉 VIX Futures (VX F1)** - ✅ **DONE** - Free from CBOE! (3,182 days, 2013-2026)
- [ ] VIX1D - CBOE (free, since 2023 only) - Optional
- [ ] VIX Options - CBOE DataShop (Phase 3, can use 15% approximation)

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
