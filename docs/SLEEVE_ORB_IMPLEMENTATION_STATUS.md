# Sleeve ORB Implementation Status

**Date**: 2026-01-07
**Status**: 🔴 **KILLED** - Failed Kill-Test Criteria

---

## Summary

The Opening Range Breakout (ORB) strategy backtester for MES/MNQ micro futures was fully implemented and tested with 3.25 years of historical data (2022-2025). **The strategy failed kill-test criteria** with a Sharpe ratio of 0.23 (threshold: 0.5) and only 2 out of 6 folds passing.

## 🔴 KILL-TEST VERDICT

**Do Not Trade** - Failed 3 of 6 critical criteria:
- ❌ Sharpe 0.23 < 0.5 threshold (-54% shortfall)
- ❌ Only 2/6 folds passed (need ≥4/6)
- ❌ MNQ unprofitable (-$1,590)
- ✅ Net P&L +$1,403 (passed)
- ✅ Win rate 44.4% > 35% (passed)
- ✅ Max DD -1.7% (passed)

**Detailed Results**: See [SLEEVE_ORB_KILL_TEST_RESULTS.md](./SLEEVE_ORB_KILL_TEST_RESULTS.md)

## ~~🚫 BLOCKER: Historical 1-Minute Futures Data~~ ✅ RESOLVED

**Root Cause Identified (2026-01-06 18:33 ET)**:

The ORB backtest requires 1-minute intraday data for **2022-2024**, but:

1. **IBKR Limitation**: Cannot provide historical data for **expired** futures contracts
   - Current contracts (2026+) qualify fine: MESH6 ✅, MESM6 ✅, etc.
   - Expired contracts (2024 and earlier) fail: MESZ24 ❌, MESM24 ❌, MESH24 ❌
   - Error 200: "No security definition has been found for the request"
   - This is a fundamental IBKR limitation, not a permissions issue

2. **Polygon.io Limitation**: No futures data subscription
   - Current API key covers stocks only
   - Futures requires separate subscription (not yet purchased)

3. **V15 Data**: Only has **daily** OHLC bars (not suitable for ORB intraday)
   - data/v15/raw/MES.parquet: Daily bars from yfinance (ES=F)
   - ORB requires 1-minute bars for opening range calculation

## Data Source Options

### Option A: IBKR TWS/Gateway (FUTURE DATA ONLY)

**✅ Account DU8009825 HAS futures permissions** (confirmed 2026-01-06):
- V15 successfully trades MES, MNQ, M2K, MGC, MCL, M6E futures
- Current contracts qualify and trade successfully

**❌ IBKR cannot provide historical data for expired contracts**:
```
# Current contracts work:
MESH6 (Mar 2026) → ConId=750150186 ✅
MESM6 (Jun 2026) → Available ✅
MESU6 (Sep 2026) → Available ✅

# Expired contracts fail:
MESZ24 (Dec 2024) → Error 200 ❌
MESM24 (Jun 2024) → Error 200 ❌
MESH24 (Mar 2024) → Error 200 ❌
```

**IBKR is only suitable for**:
- Forward-looking backtests (using data from 2026 onward)
- Live trading execution
- NOT suitable for historical backtest data (2022-2024)

### Option B: Polygon.io / Massive.com (Requires Subscription)

**Current Status**: No futures data subscription.

**Subscription Options**:
1. **Polygon.io/Massive.com Futures**: https://massive.com/futures
2. **Databento**: https://databento.com/pricing (has CME micro futures, pay-as-you-go)
3. **FirstRate Data**: https://firstratedata.com (one-time purchase for historical)
4. **CME DataMine**: https://datamine.cmegroup.com (official CME source)

### Option C: Alternative Historical Data Providers

For one-time historical backtest data purchase:

| Provider | Coverage | Format | Approx. Cost |
|----------|----------|--------|--------------|
| FirstRate Data | CME micros, 1-min | CSV | ~$100-200 |
| Databento | CME micros, tick-level | Parquet | Pay-per-use |
| Kibot | Futures, 1-min | CSV | ~$50/symbol |
| QuantGo | CME, 1-min+ | API | Subscription |

## Completed Components

### 1. Configuration (`config/sleeve_orb.yaml`)
- All frozen parameters from spec v1.6
- Walk-forward fold definitions (6 folds, 2022-2025)
- Kill-test thresholds
- Contract specifications (MES, MNQ)

### 2. Event Calendar (`data/orb/skip_dates.csv`)
- FOMC dates with EARLY_FLATTEN action (13:55 ET)
- CPI release dates with SKIP_FULL_DAY action
- NFP release dates with SKIP_FULL_DAY action
- Quad witching dates with SKIP_FULL_DAY action
- Half-day dates (Black Friday) with SKIP_FULL_DAY action
- Coverage: 2022-2025

### 3. Backtester (`src/dsp/backtest/orb_futures.py`)
~1239 lines, complete implementation:

- **OR Construction**: 30 bars (09:30-09:59 ET), excludes 10:00 bar
- **Entry Logic**: OCO stop-market at OR±buffer (2 ticks)
- **Stop Sizing**: `max(1.0×OR_Width, 0.20×ATR_d)` where ATR is RTH-only
- **Target**: 2R (2×stop distance)
- **Position Sizing**: 20 bps risk, skip if qty < 1 (does NOT force min(1))
- **Pessimistic Fills**: Stop-first on same-bar, gap-through at bar.open
- **Cost Model**: Adverse tick slippage + $1.24 RT commission
- **Filters**: Compression (<20% avg), Exhaustion (>200% avg)
- **Walk-Forward**: 6 folds with exact dates from spec

### 4. Data Fetchers

**Polygon.io Fetcher** (`src/dsp/data/futures_fetcher.py`) - ~508 lines:
- **Ticker Format**: Base symbol + Month code + Year digit (e.g., MESH5)
- **Roll Logic**: 5 days before third Friday of expiration month
- **Back-Adjustment**: Additive (preserves point values)
- **Continuous Series**: Builds from per-contract data
- **Caching**: Parquet file output
- **Status**: Requires futures subscription (current key = stocks only)

**IBKR Fetcher** (`src/dsp/data/ibkr_futures_fetcher.py`) - ~475 lines:
- **Contract Format**: Symbol + YYYYMM expiry (e.g., MES 202503)
- **Roll Logic**: Same as Polygon (5 days before third Friday)
- **Back-Adjustment**: Additive (preserves point values)
- **RTH Only**: Fetches only regular trading hours data
- **Pacing**: Respects IBKR rate limits (2s between requests)
- **Error Logging**: Captures IBKR error codes (200/354/162) for diagnosis
- **Status**: Ready to use once futures permissions are enabled on account

---

## Next Steps

### Step 0: Acquire 1-Minute Futures Data (REQUIRED)

**Option A: Purchase from Third-Party Provider** (Recommended for backtest)
- FirstRate Data, Databento, or Kibot
- One-time purchase for 2022-2025 historical data
- Expected format: 1-minute OHLCV bars with timestamps

**Option B: Subscribe to Polygon.io/Massive.com Futures**
- Ongoing subscription
- API access for historical and real-time data

**Option C: Use IBKR for Forward-Looking Only**
- Start collecting 1-min data from current contracts (2026+)
- Build backtest dataset over time
- Not suitable for immediate historical backtest

### Step 1: Place Data Files

Once data is acquired, place parquet files in:
```
dsp100k/data/orb/
├── MES_1min_2022-01-01_2025-03-31.parquet
└── MNQ_1min_2022-01-01_2025-03-31.parquet
```

Expected columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`

### Step 2: Fetch Historical Data

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source ../venv/bin/activate

# Fetch MES and MNQ data (2022-01-01 to 2025-03-31)
PYTHONPATH=src python -m dsp.data.futures_fetcher \
  --symbols MES,MNQ \
  --start 2022-01-01 \
  --end 2025-03-31 \
  --output-dir data/orb
```

Expected output files:
- `data/orb/MES_1min_2022-01-01_2025-03-31.parquet`
- `data/orb/MNQ_1min_2022-01-01_2025-03-31.parquet`

### Step 3: Run Walk-Forward Backtest

```bash
# Baseline slippage (1 tick/side)
PYTHONPATH=src python -m dsp.backtest.orb_futures \
  --data-dir data/orb \
  --slippage 1 \
  --output data/orb/walk_forward_results.json

# Stress test (2 ticks/side)
PYTHONPATH=src python -m dsp.backtest.orb_futures \
  --data-dir data/orb \
  --slippage 2 \
  --output data/orb/walk_forward_results_stress.json
```

### Step 4: Evaluate Kill-Test Results

Kill-test criteria (all must pass):
- Net PnL > $0
- Mean Sharpe > 0.5
- Win Rate > 35%
- Max Drawdown ≥ -15%
- SPY Correlation < 0.7
- ≥4/6 folds pass
- Both MES and MNQ profitable

---

## File Structure

```
dsp100k/
├── config/
│   └── sleeve_orb.yaml           # Strategy configuration
├── data/orb/
│   ├── skip_dates.csv            # Event calendar
│   ├── MES_1min_*.parquet        # (after fetch)
│   ├── MNQ_1min_*.parquet        # (after fetch)
│   ├── walk_forward_results.json # (after backtest)
│   ├── trades.csv                # (after backtest)
│   └── equity.parquet            # (after backtest)
├── docs/
│   ├── SLEEVE_ORB_MINIMAL_SPEC.md       # Frozen spec (v1.6)
│   └── SLEEVE_ORB_IMPLEMENTATION_STATUS.md  # This file
└── src/dsp/
    ├── backtest/
    │   └── orb_futures.py        # ORB backtester
    └── data/
        └── futures_fetcher.py    # Polygon.io futures data fetcher
```

---

## Implementation Notes

### Polygon.io Ticker Format

For micro E-mini futures:
- **Base symbols**: MES (Micro S&P 500), MNQ (Micro Nasdaq 100)
- **Month codes**: H=Mar, M=Jun, U=Sep, Z=Dec (quarterly only)
- **Year digit**: Single digit (e.g., 5 for 2025)
- **Examples**: MESH5 (MES March 2025), MNQZ4 (MNQ December 2024)

Source: [Polygon.io Futures API](https://polygon.io/docs/rest/futures/aggregates)

### Roll Schedule

Micro E-mini futures roll 5 days before the third Friday of the expiration month:

| Month | Third Friday | Roll Date | Contract |
|-------|--------------|-----------|----------|
| Mar 2025 | 2025-03-21 | 2025-03-16 | H5 → M5 |
| Jun 2025 | 2025-06-20 | 2025-06-15 | M5 → U5 |
| Sep 2025 | 2025-09-19 | 2025-09-14 | U5 → Z5 |
| Dec 2025 | 2025-12-19 | 2025-12-14 | Z5 → H6 |

### ATR Calculation

ATR is calculated from RTH-only data to avoid overnight gap inflation:
- Period: 14 trading days
- True Range: Uses RTH high, low, and prior RTH close
- No globex/overnight data included

---

## Verification Checklist

Before running backtest, verify:

- [ ] `POLYGON_API_KEY` environment variable set
- [ ] Data files exist in `data/orb/`
- [ ] Data covers full backtest period (2022-01-01 to 2025-03-31)
- [ ] Skip dates CSV has correct format
- [ ] Virtual environment activated (`source ../venv/bin/activate`)

---

## References

- [SLEEVE_ORB_MINIMAL_SPEC.md](./SLEEVE_ORB_MINIMAL_SPEC.md) - Strategy specification (v1.6)
- [SLEEVE_KILL_TEST_SUMMARY.md](./SLEEVE_KILL_TEST_SUMMARY.md) - Kill-test results for other sleeves
- [Polygon.io Futures API](https://polygon.io/futures) - Data source documentation
