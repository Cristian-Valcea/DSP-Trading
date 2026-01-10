# VRP Data Acquisition Checklist

**Purpose**: Track data requirements for VRP (Volatility Risk Premium) strategy backtest
**Date Created**: 2026-01-08
**Last Updated**: 2026-01-08
**Status**: ✅ **Phase 1 COMPLETE** - All required data downloaded

---

## Download Status

### ✅ Completed Downloads (2026-01-08)

| Data | Source | Date Range | Rows | File |
|------|--------|------------|------|------|
| **VIX Spot** | Yahoo `^VIX` | 2010-01-04 to 2026-01-07 | 4,028 | `data/vrp/indices/VIX_spot.parquet` |
| **VVIX** | Yahoo `^VVIX` | 2010-01-04 to 2026-01-07 | 4,019 | `data/vrp/indices/VVIX.parquet` |
| **Fed Funds Rate** | FRED `DFF` | 2010-01-01 to 2026-01-06 | 5,850 | `data/vrp/rates/fed_funds.parquet` |
| **3-Month T-Bill** | FRED `DTB3` | 2010-01-01 to 2026-01-06 | 4,178 | `data/vrp/rates/tbill_3m.parquet` |
| **10-Year Treasury** | FRED `DGS10` | 2010-01-01 to 2026-01-06 | 4,178 | `data/vrp/rates/treasury_10y.parquet` |
| **🎉 VIX Futures (VX F1)** | **CBOE FREE** | 2013-05-20 to 2026-01-07 | **3,182** | `data/vrp/futures/VX_F1_CBOE.parquet` |

**Download Scripts**:
- `scripts/vrp_data_downloader.py` - VIX indices and FRED rates
- `scripts/download_vx_futures_cboe.py` - **NEW** - VIX futures from CBOE (FREE!)

### Still Needed (Optional)

| Data | Source | Priority | Notes |
|------|--------|----------|-------|
| **VIX1D** | CBOE (free) | P2 | Only available since 2023, optional for VRP |
| **VIX Options** | CBOE DataShop | P3 | For tail hedge modeling (can use 15% approximation)

---

## Summary

The VRP strategy requires VIX futures and options data from **CBOE/CFE** (not CME).

**⚠️ IMPORTANT**: Databento does **NOT** offer CFE/CBOE VIX futures data. Alternative sources required.

---

## Data Sources Comparison

| Data | Databento | CBOE Direct | Polygon.io | Quandl/Nasdaq | Yahoo Finance |
|------|-----------|-------------|------------|---------------|---------------|
| VIX Futures (VX) | ❌ Not available | ✅ Paid API | ❌ No futures | ❌ **PREMIUM ONLY** | ❌ No futures |
| VIX Spot Index | ❌ | ✅ Free CSV | ✅ `I:VIX` | ❌ **PREMIUM ONLY** | ✅ `^VIX` |
| VIX1D | ❌ | ✅ Free CSV | ❓ Limited | ❓ | ❌ |
| VVIX | ❌ | ✅ Free CSV | ❓ | ❌ **PREMIUM ONLY** | ✅ `^VVIX` |
| VIX Options | ❌ | ✅ Paid | ❌ | ❓ | ❌ |

**⚠️ NOTE (2026-01-08)**: Nasdaq Data Link free tier does NOT include VIX data. All CBOE-related datasets require premium subscription.

---

## Recommended Data Sources

### Option A: CBOE DataShop (Official Source)

**URL**: https://datashop.cboe.com/

**Available Products**:
| Product | Coverage | Format | Cost |
|---------|----------|--------|------|
| VIX Futures Historical | 2004-present | CSV | $$ (subscription) |
| VIX Term Structure | Daily | CSV | $ |
| VIX Options | 2006-present | CSV | $$$ |
| VIX Spot Index | 1990-present | CSV | Free (delayed) |

**Pros**: Official source, highest quality, complete history
**Cons**: Requires subscription, may need institutional account

---

### Option B: Quandl/Nasdaq Data Link

**URL**: https://data.nasdaq.com/

**Available Datasets**:
| Dataset Code | Description | Cost |
|--------------|-------------|------|
| `SCF/VX` | VIX Futures Continuous | Free tier available |
| `CBOE/VIX` | VIX Spot Index | Free |
| `CBOE/VVIX` | VVIX Index | Free |

**Python Example**:
```python
import nasdaqdatalink

# VIX Futures (continuous front-month)
vx_futures = nasdaqdatalink.get("SCF/VX", start_date="2010-01-01")

# VIX Spot
vix_spot = nasdaqdatalink.get("CBOE/VIX", start_date="2010-01-01")

# VVIX
vvix = nasdaqdatalink.get("CBOE/VVIX", start_date="2010-01-01")
```

**Pros**: Easy API, free tier for some data, Python SDK
**Cons**: May need premium for full futures chain

---

### Option C: Yahoo Finance (Free, Limited)

**Available via yfinance**:
| Symbol | Description | Quality |
|--------|-------------|---------|
| `^VIX` | VIX Spot Index | ✅ Good |
| `^VVIX` | VVIX Index | ✅ Good |
| `VXF25.CBF`, `VXG25.CBF`, etc. | Individual VIX futures | ⚠️ Spotty |

**Python Example**:
```python
import yfinance as yf

# VIX Spot (works well)
vix = yf.download("^VIX", start="2010-01-01")

# VVIX (works well)
vvix = yf.download("^VVIX", start="2010-01-01")

# VIX Futures (individual contracts - may have gaps)
vxf25 = yf.download("VXF25.CBF", start="2024-01-01")
```

**Pros**: Free, easy to use
**Cons**: Futures data unreliable, no term structure

---

### Option D: Polygon.io (If Subscribed)

**Check availability**:
```python
# VIX Spot Index
GET /v2/aggs/ticker/I:VIX/range/1/day/2010-01-01/2026-01-08

# Note: Polygon may not have VIX futures
# Verify before relying on it
```

---

## Free Data Sources (Confirmed Working)

### Source 1: CBOE (Free Downloads)

**URL**: https://www.cboe.com/tradable_products/vix/

| Data | URL / Method | Format |
|------|--------------|--------|
| **VIX Spot** | `^VIX` historical | CSV |
| **VIX1D** | `^VIX1D` historical | CSV |
| **VVIX** | `^VVIX` historical | CSV |
| **VIX Term Structure** | CBOE term structure page | CSV |

**Note**: CBOE provides free historical index data, but not futures/options execution data.

---

### Source 2: FRED (Federal Reserve Economic Data)

**URL**: https://fred.stlouisfed.org/

| Data | FRED Series ID | Frequency |
|------|----------------|-----------|
| **Fed Funds Rate** | `DFF` | Daily |
| **3-Month T-Bill** | `DTB3` | Daily |
| **10-Year Treasury** | `DGS10` | Daily |

**Used For**:
- Adjusted contango calculation (subtract risk-free rate component)
- Collateral yield calculation

---

### Source 3: Polygon.io (If Already Subscribed)

If you have Polygon.io subscription:

| Data | Endpoint | Notes |
|------|----------|-------|
| **VIX Spot** | `/v2/aggs/ticker/I:VIX` | Index data |
| **VIX1D** | `/v2/aggs/ticker/I:VIX1D` | May not be available |
| **VVIX** | `/v2/aggs/ticker/I:VVIX` | May not be available |

**⚠️ Note**: Polygon may not have complete VIX index coverage. Verify before relying on it.

---

## Priority Order for Acquisition

### Phase 1: Minimum Viable Backtest (Start Here)

| Priority | Data | Source | Cost |
|----------|------|--------|------|
| **P1** | VIX Futures (VX) daily | Quandl `SCF/VX` OR CBOE DataShop | Free/$ |
| **P1** | VIX Spot Index | CBOE (free) or Yahoo `^VIX` | Free |
| **P1** | Fed Funds Rate | FRED `DFF` (free) | Free |

**With This You Can**:
- Calculate adjusted contango signal
- Build continuous futures series
- Run basic VRP backtest (no options hedge)

### Phase 2: Full Backtest

| Priority | Data | Source | Cost |
|----------|------|--------|------|
| **P2** | VIX1D Index | CBOE (free) | Free |
| **P2** | VVIX Index | CBOE (free) or Yahoo `^VVIX` | Free |
| **P2** | 3-Month T-Bill Rate | FRED `DTB3` (free) | Free |

**With This You Can**:
- Implement all entry/exit filters
- Add regime-adaptive thresholds
- Full signal generation

### Phase 3: Options Hedge (Optional)

| Priority | Data | Source | Cost |
|----------|------|--------|------|
| **P3** | VIX Options daily | CBOE DataShop | $$$ |

**With This You Can**:
- Realistic tail hedge construction
- Accurate hedge cost modeling
- Full trade structure simulation

**Alternative**: Use simplified model (15% of premium) until Phase 3

---

## Recommended Acquisition Strategy

### Step 1: Start with Free Data (TODAY)

```bash
# Download free indices from CBOE
# Visit: https://www.cboe.com/tradable_products/vix/
# Download VIX, VIX1D, VVIX historical data (CSV)

# Download rates from FRED
# Visit: https://fred.stlouisfed.org/series/DFF
# Download Fed Funds Rate (CSV)
```

### Step 2: Get VIX Futures via Quandl (RECOMMENDED)

```python
import nasdaqdatalink

# Set your API key (free tier available)
nasdaqdatalink.ApiConfig.api_key = "YOUR_API_KEY"

# VIX Futures (continuous front-month)
vx_futures = nasdaqdatalink.get("SCF/VX", start_date="2010-01-01")
vx_futures.to_csv("data/vrp/futures/VX_continuous.csv")

# VIX Spot (backup source)
vix_spot = nasdaqdatalink.get("CBOE/VIX", start_date="2010-01-01")
vix_spot.to_csv("data/vrp/indices/VIX_spot.csv")

# VVIX
vvix = nasdaqdatalink.get("CBOE/VVIX", start_date="2010-01-01")
vvix.to_csv("data/vrp/indices/VVIX.csv")
```

### Step 3: Yahoo Finance Backup (FREE)

```python
import yfinance as yf

# VIX Spot (works reliably)
vix = yf.download("^VIX", start="2010-01-01")
vix.to_csv("data/vrp/indices/VIX_yahoo.csv")

# VVIX (works reliably)
vvix = yf.download("^VVIX", start="2010-01-01")
vvix.to_csv("data/vrp/indices/VVIX_yahoo.csv")
```

### Step 4: FRED Rates (FREE)

```python
import pandas_datareader as pdr

# Fed Funds Rate
fed_funds = pdr.get_data_fred("DFF", start="2010-01-01")
fed_funds.to_csv("data/vrp/rates/fed_funds.csv")

# 3-Month T-Bill
tbill_3m = pdr.get_data_fred("DTB3", start="2010-01-01")
tbill_3m.to_csv("data/vrp/rates/tbill_3m.csv")

# 10-Year Treasury
treasury_10y = pdr.get_data_fred("DGS10", start="2010-01-01")
treasury_10y.to_csv("data/vrp/rates/treasury_10y.csv")
```

---

## Cost Summary

| Data | Source | Cost | Notes |
|------|--------|------|-------|
| VIX Spot | CBOE/Yahoo | Free | Reliable |
| VIX1D | CBOE | Free | Since 2023 only |
| VVIX | CBOE/Yahoo | Free | Reliable |
| Fed Funds, T-Bills | FRED | Free | Reliable |
| VIX Futures (continuous) | Quandl SCF/VX | Free tier | May need premium for full chain |
| VIX Futures (full chain) | CBOE DataShop | $$ | Official source |
| VIX Options | CBOE DataShop | $$$ | Phase 3 only |

**Total Minimum Cost**: $0 (using Quandl free tier + CBOE + FRED + Yahoo)

---

## Download Commands (After Acquisition)

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate

# Step 1: Download VIX Futures from Quandl (or CBOE DataShop CSV)
python scripts/vrp_data_downloader.py \
  --source quandl \
  --output-dir data/vrp/futures

# Step 2: Download free indices (VIX, VIX1D, VVIX)
python scripts/vrp_data_downloader.py \
  --source cboe \
  --symbols VIX VIX1D VVIX \
  --start 2010-01-01 \
  --output-dir data/vrp/indices

# Step 3: Download rates from FRED
python scripts/vrp_data_downloader.py \
  --source fred \
  --series DFF DTB3 DGS10 \
  --start 2010-01-01 \
  --output-dir data/vrp/rates

# Step 4: Convert all to Parquet
python scripts/vrp_data_converter.py \
  --input-dir data/vrp \
  --output-dir data/vrp
```

---

## Data Validation Checklist

After downloading, verify:

- [x] VX futures: 385 contracts downloaded (monthly + weekly 2012-2026) ✅
- [x] VX futures: Continuous F1 series built (3,182 days) ✅
- [x] VX futures: Settle prices valid (range: 9.88 - 74.53) ✅
- [x] VIX spot: Complete daily series (4,028 rows, 2010-2026) ✅
- [ ] VIX1D: Complete from 2023-present (NOT DOWNLOADED - optional)
- [x] VVIX: Complete from 2010-present (4,019 rows) ✅
- [x] Fed Funds: Complete daily series (5,850 rows) ✅
- [x] 3-Month T-Bill: Complete daily series (4,178 rows) ✅
- [x] 10-Year Treasury: Complete daily series (4,178 rows) ✅
- [x] Roll dates: Automatic roll logic implemented in continuous series builder ✅

---

## File Structure After Acquisition

```
dsp100k/data/vrp/
├── futures/
│   ├── VX_continuous_1d.parquet     # Rolled continuous series
│   ├── VX_individual_contracts/     # Raw per-contract files
│   │   ├── VXF10.parquet
│   │   ├── VXG10.parquet
│   │   └── ...
│   └── VX_term_structure.parquet    # F1, F2, F3 columns per day
├── options/                          # Phase 3
│   └── VIX_options_daily.parquet
├── indices/
│   ├── VIX_spot.parquet
│   ├── VIX1D.parquet
│   └── VVIX.parquet
└── rates/
    ├── fed_funds.parquet
    └── tbill_3m.parquet
```

---

## Next Steps

1. ~~**Sign up for Quandl/Nasdaq Data Link**: Get API key (free tier) at https://data.nasdaq.com/~~ → ✅ **DONE** (2026-01-08)
2. ~~**Download Free Data**: Get VIX/VIX1D/VVIX from CBOE, rates from FRED~~ → ✅ **DONE** (2026-01-08)
3. ~~**Create Data Downloader**: Write `scripts/vrp_data_downloader.py` for all sources~~ → ✅ **DONE**
4. ~~**Test Nasdaq Data Link API**: SCF/VX download attempted~~ → ❌ **BLOCKED** - Premium subscription required
5. ~~**OPTION A: Use Synthetic VX_F1** - Proceed with initial backtest using VIX + estimated contango~~ → **SUPERSEDED**
6. ~~**OPTION B: Purchase CBOE DataShop** - Official VIX futures historical data ($$$)~~ → **NOT NEEDED!**
7. ✅ **🎉 SOLVED: Download FREE VIX futures from CBOE!** → `scripts/download_vx_futures_cboe.py` (3,182 days of real data!)
8. ✅ **Build Continuous Series**: Automatic roll logic in downloader (front-month F1 series built)
9. ✅ **Validate Data**: Quality checks passed (VX F1: mean=18.35, min=9.88, max=74.53)
10. ✅ **Create VRP Backtester**: `src/dsp/backtest/vrp_futures.py` (576 lines)
11. ✅ **Run Baseline Backtest**: Kill-test completed with corrected P&L calculation

---

## 🔴 KILL-TEST VERDICT: FAIL (2026-01-08)

### Full Period Results (2014-2025)
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Sharpe Ratio** | -0.13 | ≥ 0.50 | ❌ FAIL |
| **Net Return** | -6.25% | > 0% | ❌ FAIL |
| **Max Drawdown** | -15.16% | ≥ -30% | ✅ PASS |

### OOS Fold Results
| Fold | Period | Sharpe | Return | MaxDD | Verdict |
|------|--------|--------|--------|-------|---------|
| **Fold 1** | 2023 | -1.05 | -4.44% | -5.86% | ❌ FAIL |
| **Fold 2** | 2024 | +1.56 | +1.06% | -0.23% | ✅ PASS |
| **Fold 3** | 2025 | -0.22 | -0.35% | -1.30% | ❌ FAIL |

**Folds Passed**: 1/3 (needs 2/3)

### Configuration
- Initial NAV: $100,000
- Max NAV%: 20% (allows 1 contract at VX~$15)
- Max Margin%: 30%
- Stop-Loss: -15%
- Cost Multiplier: 1.0x (baseline), 2.0x (stress)

### Why VRP Fails Kill-Test

1. **Entry Filter Too Strict**: Only 12.4% time in market (375 days out of 3,018)
2. **Stop-Loss Triggered Often**: Multiple stop-losses in 2014, 2020, 2022-2023
3. **Poor 2023 Performance**: VIX volatility spikes caused -4.44% loss
4. **Small Position Size**: 1 contract per trade limits profit capture in contango

### Recommendations

**DO NOT TRADE** this strategy as configured. Consider:
1. Different entry signals (looser filters for more exposure)
2. Tail hedge implementation (OTM calls to limit stop-loss damage)
3. Higher capital allocation (more contracts = better risk-adjusted returns)
4. Alternative VRP approach (e.g., VIX ETN pair trading like SVXY/UVXY)

---

## ⚠️ CRITICAL DISCOVERY: Nasdaq Free Tier Limitations (2026-01-08)

**Tested API Key**: Saved in `.env.local` as `NASDAQ_API_KEY`

**Datasets Tested** (ALL REQUIRE PREMIUM):
| Dataset | Result | Error |
|---------|--------|-------|
| `SCF/VX` (Stevens Continuous VIX Futures) | ❌ 403 Forbidden | Premium subscription required |
| `CBOE/VIX` (VIX Spot) | ❌ 403 Forbidden | Premium subscription required |
| `CHRIS/CBOE_VX1` (Front-month VIX) | ❌ 403 Forbidden | Premium subscription required |
| `CHRIS/CBOE_VX2` (Second-month VIX) | ❌ 403 Forbidden | Premium subscription required |

**Yahoo Finance VIX Futures**: Also tested and NOT available (404 Not Found for `VXF25.CBF`, etc.)

**Conclusion**: **There is NO free source for VIX futures daily data with full 2010-2026 coverage.**

---

## 🎉 SOLUTION: FREE VIX Futures from CBOE (2026-01-08)

**Discovery**: CBOE provides FREE historical VIX futures data via their CFE historical data page!

**URL**: https://www.cboe.com/us/futures/market_statistics/historical_data/

**How It Works**:
1. Individual contract CSVs available for each expiration (e.g., `VX_2024-01-17.csv`)
2. Contains: Trade Date, Futures name, Open, High, Low, Close, Settle, Change, Volume, OI
3. Download script builds continuous front-month (F1) series automatically

**Download Script**: `scripts/download_vx_futures_cboe.py`

**Data Acquired**:
| Metric | Value |
|--------|-------|
| **Contracts Downloaded** | 385 (monthly + weekly) |
| **Continuous F1 Days** | 3,182 |
| **Date Range** | 2013-05-20 to 2026-01-07 |
| **Mean Settle** | 18.35 |
| **Min Settle** | 9.88 |
| **Max Settle** | 74.53 |
| **File** | `data/vrp/futures/VX_F1_CBOE.parquet` |

**Limitation**: Data before 2013-05-20 returns 403 (may require DataShop subscription for earlier years). However, 12+ years of data is sufficient for robust VRP backtest.

---

## Deprecated: Synthetic VX_F1 Estimate

~~A synthetic VX_F1 series was created as a workaround~~ → **NO LONGER NEEDED**

**File**: `data/vrp/futures/VX_F1_SYNTHETIC.parquet` (kept for reference only)

**Note**: Use `VX_F1_CBOE.parquet` for all VRP backtest work - it contains REAL settlement prices from CBOE.

---

**Document Version**: 1.4
**Last Updated**: 2026-01-08
**Author**: Claude
**Related Files**:
- `SPEC_VRP.md` (full technical specification)
- `SLEEVE_VRP_PRESENTATION.md` (management presentation)
- `DOWNLOADED_DATA_MARKET.md` (data inventory)
