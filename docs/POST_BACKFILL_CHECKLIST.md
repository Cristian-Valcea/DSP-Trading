# Sleeve IM: Post-Backfill Checklist

**Created**: 2025-12-31
**Status**: ✅ **BACKFILL COMPLETE** (2026-01-01)

---

## Overview

This document tracks the steps to validate the Polygon minute-bar backfill for Sleeve IM and proceed to model training.

**Spec Decisions (Locked)**:
- **Feature Window (inputs)**: 04:00-10:30 ET (premarket + first hour)
- **Signal Time**: ~10:31 ET (after feature window closes)
- **Entry Window (trading)**: 11:30 ET → close (exit via MOC)
- **Target / Label**: binary sign of return (11:30 → close)
- **Symbols**: 9-symbol universe (QQQ, TSLA, AAPL, MSFT, NVDA, AMZN, GOOGL, META, SPY)
- **Date Range**: 2023-01-01 to 2024-12-31 (~502 trading days)

**Environment Prereqs**:
- Use Python **3.10+** (the system `python3` is 3.9 and will break imports like `pandas_market_calendars`)
- Activate the project venv before running any `dsp` imports:

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source venv/bin/activate
python --version
```

---

## Phase 1: Confirm Completeness

### 1.1 File Count Validation

Each symbol should have ~502 JSON files (one per trading day in 2023-2024).

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k

# Count files per symbol
for sym in QQQ TSLA AAPL MSFT NVDA AMZN GOOGL META SPY; do
  count=$(find "data/sleeve_im/minute_bars/${sym}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  echo "$sym: $count files"
done
```

**Expected**: 502 files per symbol for 2023-2024 (per `scripts/sleeve_im/backfill_data.py` holiday list)

To compute the expected count from the same calendar logic used by the backfill:

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source venv/bin/activate
python -c "
from datetime import date
import scripts.sleeve_im.backfill_data as b
days = b.get_trading_days_excluding_holidays(date(2023,1,1), date(2024,12,31))
print('expected_trading_days:', len(days))
print('first:', days[0], 'last:', days[-1])
"
```

### 1.2 Date Range Validation

Verify first and last dates are correct:

```bash
# First file (should be ~2023-01-03)
ls data/sleeve_im/minute_bars/QQQ/*.json | head -1

# Last file (should be ~2024-12-31)
ls data/sleeve_im/minute_bars/QQQ/*.json | tail -1
```

### 1.3 Time Window + Grid Validation

Important: the JSON cache files in `data/sleeve_im/minute_bars/` store **raw Polygon aggregates** (real bars only).
They are **not** a complete 1-minute grid, so `len(data["bars"])` will vary day-to-day.

**Grid sizes explained**:
- **Raw cache**: Variable length (only real bars from Polygon)
- **Backfill log "541 bars"**: The fetcher's default grid is 01:30-10:30 ET (541 minutes)
- **Sleeve IM spec "391 bars"**: The locked feature window is 04:00-10:30 ET (391 minutes)

Validate two things:
1) raw cache schema + timestamp bounds look sane
2) the **processed** grid (carry-forward applied) matches the Sleeve IM feature window (04:00-10:30 ET)

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source venv/bin/activate

# Raw cache sanity (schema + timestamp bounds)
python -c "
import json
from pathlib import Path

f = sorted(Path('data/sleeve_im/minute_bars/QQQ').glob('*.json'))[0]
data = json.load(open(f))
bars = data['bars']
req = {'timestamp','open','high','low','close','volume','trade_count'}
missing = [k for k in sorted(req) if any(k not in b for b in bars)]
print('file:', f.name)
print('raw_bars:', len(bars))
print('first_ts:', bars[0]['timestamp'])
print('last_ts:', bars[-1]['timestamp'])
print('missing_required_fields:', missing)
"
```

Processed grid validation (fixed bar count + synthetic % as used by data quality checks).  
Note: `PolygonFetcher.get_minute_bars()` currently fetches **prior close** from Polygon; plan on needing `POLYGON_API_KEY` (suggestion: cache prior close to avoid extra API calls during dataset build).

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source venv/bin/activate

python -c "
import asyncio
from datetime import date, time
from dsp.data.polygon_fetcher import PolygonConfig, PolygonFetcher

async def run():
    cfg = PolygonConfig.from_env()
    async with PolygonFetcher(cfg) as f:
        d = date(2024, 9, 20)
        bars = await f.get_minute_bars('QQQ', d, start_time=time(4,0), end_time=time(10,30))
        print('date:', d)
        print('total_bars:', bars.total_bars)          # expected 391 for 04:00-10:30 inclusive
        print('real_bars:', bars.real_bars)
        print('synthetic_pct:', round(bars.synthetic_pct * 100, 1))
        print('first_ts:', bars.bars[0].timestamp)
        print('last_ts:', bars.bars[-1].timestamp)

asyncio.run(run())
"
```

**Expected** (processed grid): `total_bars == 391` for 04:00-10:30 inclusive.

### 1.4 Log Scan (Errors / Quality Fails)

```bash
rg -n "Failed to fetch|Quality check FAILED|ERROR" /tmp/backfill_qqq_tsla.log || true
```

If you see failures, re-run backfill for a narrower range with `--force` (so you only pay for the missing dates):

```bash
cd /Users/Shared/wsl-export/wsl-home/dsp100k
source venv/bin/activate

# Example: re-try a month
python scripts/sleeve_im/backfill_data.py --symbols QQQ --start 2024-09-01 --end 2024-09-30 --force
```

### 1.5 Data Quality Results & Policy

**Backfill Quality Summary** (2026-01-01):

| Batch | Symbols | Quality Pass | Quality Fail | Fetch Errors |
|-------|---------|--------------|--------------|--------------|
| Batch 1 | QQQ, TSLA | 1,004 | 0 | 0 |
| Batch 2 | AAPL, MSFT, NVDA, AMZN, GOOGL, META, SPY | 3,096 | 417 | 0 |
| **Total** | 9 symbols | **4,100** | **417** | **0** |

**Quality fail rate**: 417 / 4,517 = **9.2%** of QC'd symbol-days

*(Note: 4,517 = days that were fetched and QC'd; 1 cached day was not re-QC'd, so total symbol-days = 4,518)*

**Common causes of quality warnings**:
- Low-volume trading days (day after Thanksgiving, etc.)
- Early market closes (half-days)
- Data gaps in Polygon's premarket coverage (especially pre-04:00 ET)
- Thin premarket liquidity for some symbols

---

#### **🔒 QUALITY FAIL POLICY DECISION (LOCKED)**

**Policy**: Treat quality-fail days as **"no-trade" days** (signal = 0, no position taken)

**Rationale**:
1. **Conservative**: Avoids making predictions on unreliable data
2. **Realistic**: In production, we wouldn't trade on days with bad data quality
3. **Simple**: No need for complex imputation or feature engineering
4. **Preserves integrity**: Training labels remain clean

**Implementation**:

The backfill script uses `DailyQualityReport._evaluate_tradability()` from `dsp/data/data_quality.py`, which applies **three hard-fail gates** (lines 104-120):

| Gate | Hard-Fail Threshold | Warning Threshold | Constant |
|------|---------------------|-------------------|----------|
| Synthetic % | > 70% | > 50% | `MAX_SYNTHETIC_PCT`, `SYNTHETIC_WARN_PCT` |
| Staleness | > 60 min (3600s) | > 30 min (1800s) | `MAX_STALENESS_SECONDS`, `STALENESS_WARN_SECONDS` |
| Premarket Volume | < 10,000 | < 50,000 | `MIN_PREMARKET_VOLUME`, `MIN_PREMARKET_VOLUME_WARN` |

The backfill log's "Quality check FAILED" is driven by these hard-fail gates (any one fails → `is_tradable=False`).

```python
# Reference: dsp/data/data_quality.py thresholds
MAX_SYNTHETIC_PCT = 0.70       # Hard fail if > 70%
MAX_STALENESS_SECONDS = 3600   # Hard fail if > 60 min
MIN_PREMARKET_VOLUME = 10_000  # Hard fail if < 10k shares

def apply_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark quality-fail days as no-trade (signal = 0).

    Uses the same hard-fail gates as dsp/data/data_quality.py.
    Note: MIN_PREMARKET_VOLUME gate is computed by assess_daily_quality()
    during feature build, not re-implemented here.
    """
    # Synthetic and staleness gates (volume gate applied elsewhere during QC)
    quality_fail_mask = (df['synthetic_pct'] > 0.70) | (df['staleness_max_min'] > 60)
    df.loc[quality_fail_mask, 'signal'] = 0
    df.loc[quality_fail_mask, 'quality_flag'] = 'no_trade'
    return df
```

> **Note**: The `MIN_PREMARKET_VOLUME` gate is computed by `assess_daily_quality()` during the feature build phase, which sets `is_tradable=False` on the `DailyQualityReport`. The snippet above handles only synthetic/staleness; the volume gate is applied via the full QC pipeline.

**Impact on training**:
- ~417 symbol-days excluded from active trading signals
- These days still appear in the dataset (for continuity) but with signal=0
- Backtest will show 0 P&L contribution from these days

---

#### Quality Fail Log Analysis

To extract the specific quality-fail days for review:

```bash
# Extract quality fail summary from backfill log
grep "Quality check FAILED" /tmp/backfill_remaining7.log | \
  sed 's/.*\[\(.*\)\] Quality check FAILED.*/\1/' | \
  sort | uniq -c | sort -rn | head -20
```

**✅ DONE**: Parsed `/tmp/backfill_remaining7.log` → `data/sleeve_im/quality_fail_days.csv` (417 rows)

#### Quality Fail Summary by Symbol (01:30-10:30 grid)

| Symbol | Fail Days | % of 502 | Notes |
|--------|-----------|----------|-------|
| META | 204 | 40.6% | Worst premarket data quality |
| MSFT | 114 | 22.7% | |
| GOOGL | 98 | 19.5% | |
| AMZN | 1 | 0.2% | |
| AAPL | 0 | 0% | Perfect quality |
| NVDA | 0 | 0% | Perfect quality |
| SPY | 0 | 0% | Perfect quality |
| QQQ | 0 | 0% | Perfect quality |
| TSLA | 0 | 0% | Perfect quality |

#### Worst Days (highest synthetic %)

| Symbol | Date | Synthetic % |
|--------|------|-------------|
| META | 2024-09-25 | 81.0% |
| META | 2024-12-31 | 80.0% |
| GOOGL | 2024-06-04 | 79.7% |
| MSFT | 2024-05-09 | 78.0% |

#### ⚠️ Important Caveat: Grid Mismatch

The quality stats above are computed on the **01:30-10:30 (541-bar) grid** used by the backfill script's default fetcher, **NOT** the locked **04:00-10:30 (391-bar)** feature window.

**Estimated fail counts on 04:00-10:30 window** (much lower):

| Symbol | 01:30-10:30 Fails | Est. 04:00-10:30 Fails |
|--------|-------------------|------------------------|
| META | 204 | ~79 |
| MSFT | 114 | ~14 |
| GOOGL | 98 | ~24 |
| AMZN | 1 | ~0 |

**Recommendation**: Re-run quality checks using the 04:00-10:30 window during feature build to get accurate fail counts for the locked spec. The current CSV is conservative (flags more days than necessary).

---

## Phase 2: Spec Alignment (LOCKED)

### 2.1 Feature Window Decision

| Option | Window | Pros | Cons |
|--------|--------|------|------|
| **Option A (LOCKED)** | 04:00-10:30 ET | Real data from ~04:00, includes key premarket | Misses 01:30-04:00 |
| Option B | 01:30-10:30 ET | Matches original spec | 99%+ synthetic before 04:00 |

**Decision**: Option A (04:00-10:30 ET) is locked. Polygon Starter tier only has premarket data from ~04:00 ET.

### 2.2 Code Alignment Check

Verify `sleeve_im.py`, `polygon_fetcher.py`, and the backfill script are consistent about the feature window.

Key facts (as currently implemented):
- `dsp.sleeves.sleeve_im.SleeveIM._fetch_minute_bars()` requests `04:00-10:30`
- `dsp.data.polygon_fetcher.PolygonFetcher.get_minute_bars()` defaults to `01:30-10:30` unless you pass times explicitly
- `scripts/sleeve_im/backfill_data.py` calls `get_minute_bars()` without passing times, so its logs may show a `541`-bar grid (01:30-10:30) even though the Sleeve IM spec is locked to 04:00-10:30
- `SleeveIMConfig.feature_window_start` still defaults to `01:30`; consider updating it to `04:00` to match the locked spec decision

Recommended consistency target:
- Treat Sleeve IM feature window as **04:00-10:30 ET** end-to-end (data QC denominator, features, and live fetch).

---

## Phase 3: Build Training Dataset

### 3.1 Feature Computation

```python
# Pseudo-code for feature pipeline
from datetime import time
import pandas as pd

def build_features(cache_dir: str, symbols: list, start: str, end: str) -> pd.DataFrame:
    """
    Build features from cached minute bars.

    Features (04:00-10:30 ET window):
    - Overnight gap: (first_bar.open - prev_close) / prev_close
    - Premarket return: (10:30_close - 04:00_open) / 04:00_open
    - Premarket volume: sum of volumes 04:00-10:30
    - Premarket VWAP deviation
    - Volatility (high-low range)
    - Momentum indicators
    """
    pass

def compute_labels(cache_dir: str, symbols: list, start: str, end: str) -> pd.DataFrame:
    """
    Compute target labels for the trading window.

    Target label = sign( close - entry ) for the trading window.
    Default: entry at 11:30 ET, exit at the close (MOC).
    """
    pass
```

### 3.2 Label Price Source ✅ DECIDED

**Decision (2026-01-01)**: Use `../data/stage1_raw/` (Polygon RTH parquet) for both prices:

| Price | Source | Field | Notes |
|-------|--------|-------|-------|
| **Entry** | `../data/stage1_raw/{symbol}_1min.parquet` | 11:30 bar `open` | First-pass proxy |
| **Exit** | `../data/stage1_raw/{symbol}_1min.parquet` | 15:59 bar `close` | First-pass proxy for MOC |

**Rationale**:
- Fast (data already exists locally)
- Sufficient for baseline model + kill tests
- After baseline passes kill tests, can upgrade exit to Polygon daily close for better MOC approximation

**Edge Case Handling**:

| Scenario | Action | Reason |
|----------|--------|--------|
| **Early close day** (no 15:59 bar) | Use last bar if 12:55-13:05 | Half-days close at 13:00 ET |
| **Normal day missing 15:59** (halt) | Skip day (no trade) | Incomplete trading session |
| **Missing 11:30 bar** (halt, etc.) | Skip day (no trade) | No reliable entry price |

**Implementation**:
```python
import pandas as pd
from datetime import time

# Known NYSE half-days (close at 13:00 ET)
HALF_DAYS = {
    # 2023
    "2023-07-03", "2023-11-24",
    # 2024
    "2024-07-03", "2024-11-29", "2024-12-24",
    # 2025
    "2025-07-03", "2025-11-28", "2025-12-24",
}

def get_label_prices(symbol: str, date: pd.Timestamp) -> tuple[float | None, float | None]:
    """
    Get entry (11:30 open) and exit (15:59 close or 13:00 on half-days) prices.

    Returns (None, None) if either price is unavailable (skip day).
    """
    # Note: filenames are lowercase, e.g., spy_1min.parquet
    df = pd.read_parquet(f"../data/stage1_raw/{symbol.lower()}_1min.parquet")
    day_data = df[df['timestamp'].dt.date == date.date()]

    if day_data.empty:
        return None, None

    date_str = date.strftime("%Y-%m-%d")
    is_half_day = date_str in HALF_DAYS

    # Entry: 11:30 bar open
    entry_bar = day_data[day_data['timestamp'].dt.time == time(11, 30)]
    entry_price = entry_bar['open'].iloc[0] if len(entry_bar) > 0 else None

    # Exit: 15:59 bar close (normal day) or ~13:00 (half-day)
    exit_bar = day_data[day_data['timestamp'].dt.time == time(15, 59)]
    if len(exit_bar) > 0:
        exit_price = exit_bar['close'].iloc[0]
    elif is_half_day:
        # Half-day: accept last bar if in 12:55-13:05 window
        last_bar = day_data.iloc[-1]
        last_time = last_bar['timestamp'].time()
        if time(12, 55) <= last_time <= time(13, 5):
            exit_price = last_bar['close']
        else:
            exit_price = None  # Unexpected half-day close time
    else:
        exit_price = None  # Normal day missing 15:59 → skip

    # Both prices required; skip day if either missing
    if entry_price is None or exit_price is None:
        return None, None

    return entry_price, exit_price
```

### 3.3 Train/Validation/Dev Test Split

| Split | Date Range | Purpose |
|-------|------------|---------|
| **Train** | 2023-01-01 to 2024-06-30 | Model training |
| **Validation** | 2024-07-01 to 2024-09-30 | Hyperparameter tuning |
| **Dev Test** | 2024-10-01 to 2024-12-31 | Final pre-holdout evaluation |

**Important**: Never use Dev Test for tuning decisions.

### 3.4 True Holdout: 2025 Data

> **Definition**: A "true holdout" is a block of time-series data (usually the most recent period) that you **never use for any model/feature/threshold decisions**. You look at it only once at the end to get an unbiased estimate of how the system generalizes.

**2025 as True Holdout**:
- Even though 2025 is the holdout, you still need the same input data for 2025 so you can run the final, untouched evaluation
- **You still need premarket/feature-window bars (04:00-10:30 ET) for 2025** — requires separate Polygon backfill

**⚠️ RTH Coverage Warning**:
> The claim "RTH bars through 2025-12-19" is **not true for all 9 symbols**. Actual coverage in `../data/stage1_raw/`:

| Symbol | Last Date | Notes |
|--------|-----------|-------|
| AAPL | 2025-12-19 | ✅ |
| AMZN | 2025-12-19 | ✅ |
| GOOGL | 2025-12-19 | ✅ |
| MSFT | 2025-12-19 | ✅ |
| NVDA | 2025-12-19 | ✅ |
| QQQ | 2025-12-19 | ✅ |
| META | 2025-12-16 | ⚠️ 3 days short |
| SPY | 2025-12-16 | ⚠️ 3 days short |
| TSLA | 2025-12-16 | ⚠️ 3 days short |

**Holdout End Date**: Use **2025-12-16** (minimum common coverage) OR patch missing days for META/SPY/TSLA.

**⚠️ Holiday Calendar Gotcha**:
> `dsp100k/scripts/sleeve_im/backfill_data.py`'s holiday list currently stops at 2024. For 2025, the script will try to fetch on holidays and log "fails" unless the holiday calendar is extended (or switched to a real market calendar like `pandas_market_calendars`).

**Recommendation**:
1. Extend the holiday calendar in `backfill_data.py` for 2025 before running backfill
2. Backfill 2025 premarket up to **2025-12-16** (min common RTH coverage)
3. Optionally patch RTH data for META/SPY/TSLA to 2025-12-19 before extending holdout end

| Dataset | Period | Data Source | Status |
|---------|--------|-------------|--------|
| **Train/Val/Dev Test** | 2023-01-01 to 2024-12-31 | Polygon (premarket) + `../data/stage1_raw/` (RTH) | 🔄 In Progress |
| **True Holdout** | 2025-01-01 to 2025-12-16 | Polygon (premarket) + `../data/stage1_raw/` (RTH) | ❌ Needs backfill |

---

## Phase 4: Research Gate (Kill Tests)

### 4.1 Baseline Model

Train a simple model (XGBoost or LightGBM) with the features:

```python
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective='binary:logistic',
)

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
```

### 4.2 Kill Test Criteria

| Metric | Threshold | Kill If |
|--------|-----------|---------|
| **Sharpe Ratio** | ≥ 0.5 | < 0.5 (no edge) |
| **Hit Rate** | > 52% | ≤ 50% (random) |
| **Max Drawdown** | < 15% | > 20% |
| **Correlation to SPY** | < 0.7 | > 0.85 (just beta) |

### 4.3 Backtest Requirements

- **Execution**: 1 entry (11:30 ET) + 1 exit (MOC/close)
- **Slippage/fees**: include a realistic per-trade cost model (start with 5–10 bps + commissions)
- **Position limits**: start conservative (e.g., ~3% NAV per name cap, ~10% NAV gross, net ≤10% of gross)
- **Rebalance**: daily at 11:30 ET (sizing computed after 10:30 window)

---

## Phase 5: Shadow → Paper → Scale

Note: these steps require Sleeve IM signal generation + execution wiring to be fully implemented; treat the CLI commands here as placeholders until then.

### 5.1 Shadow Mode (Week 1-2)

- Generate signals daily at 10:31 ET
- Log signals but don't execute
- Compare predicted vs actual returns
- Monitor: Hit rate, signal stability, data quality

```bash
# Daily shadow run
python -m dsp.cli --config config/dsp100k_sleeve_im_shadow.yaml plan
```

Note: `config/dsp100k_sleeve_im_shadow.yaml` is not in the repo yet; create it once Sleeve IM signal generation + orchestrator integration is runnable.

### 5.2 Paper Trading (Week 3-6)

- Execute via IBKR paper account
- Start with $50K notional
- Monitor: Fill rates, slippage, P&L

```bash
# Paper trading execution
python -m dsp.cli --config config/dsp100k_sleeve_im_paper.yaml run
```

Note: `config/dsp100k_sleeve_im_paper.yaml` is not in the repo yet; create it once Sleeve IM execution is implemented.

### 5.3 Scale to Live (Week 7+)

- Start with 25% of target allocation
- Scale up 25% per week if metrics hold
- Full allocation after 30 days of positive Sharpe

---

## Appendix: Symbol Universe

### Primary 9-Symbol Universe

| Symbol | Sector | Rationale |
|--------|--------|-----------|
| QQQ | Tech ETF | Broad tech exposure, high liquidity |
| TSLA | Auto/Tech | High vol, strong morning patterns |
| AAPL | Tech | Mega-cap, liquid, earnings catalyst |
| MSFT | Tech | Mega-cap, stable patterns |
| NVDA | Semis | AI leader, high premarket activity |
| AMZN | Retail/Cloud | Large-cap, premarket news sensitive |
| GOOGL | Tech | Mega-cap, earnings driven |
| META | Social | High vol, news sensitive |
| SPY | Broad ETF | Benchmark, overnight gap signal |

### Backfill Status

✅ **COMPLETE** (2026-01-01)

| Symbol | Files | First | Last | Status |
|--------|-------|-------|------|--------|
| QQQ | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| TSLA | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| AAPL | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| MSFT | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| NVDA | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| AMZN | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| GOOGL | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| META | 502 | 2023-01-03 | 2024-12-31 | ✅ |
| SPY | 502 | 2023-01-03 | 2024-12-31 | ✅ |

---

## Monitoring Commands

```bash
# Check backfill progress
tail -20 /tmp/backfill_qqq_tsla.log

# Count all symbol files
cd /Users/Shared/wsl-export/wsl-home/dsp100k
for sym in QQQ TSLA AAPL MSFT NVDA AMZN GOOGL META SPY; do
  count=$(find "data/sleeve_im/minute_bars/${sym}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  echo "$sym: $count"
done

# Check if backfill is running
ps aux | grep backfill | grep -v grep

# Sample a cached file
python - <<'PY'
import json
from pathlib import Path

f = next(Path("data/sleeve_im/minute_bars/QQQ").glob("*.json"))
data = json.load(open(f))
bars = data["bars"]

print("File:", f.name)
print("Raw bars:", len(bars))
print("First ts:", bars[0]["timestamp"])
print("Last ts:", bars[-1]["timestamp"])
print("Bar keys:", sorted(bars[0].keys()))
PY
```

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-31 | Created checklist, locked 04:00-10:30 ET feature window |
| 2025-12-31 | Added Section 3.4: True Holdout (2025 data) + holiday calendar gotcha |
| 2026-01-01 | Backfill complete: all 9 symbols 502/502 files |
| 2026-01-01 | Section 1.5 updated: Quality fail policy locked (no-trade days), stats: 4100 pass / 417 fail |
| 2026-01-01 | Section 1.3 clarified: grid sizes (541 fetcher default vs 391 spec window) |
| 2026-01-01 | Created `data/sleeve_im/quality_fail_days.csv` (417 rows) |
| 2026-01-01 | Added grid mismatch caveat: 01:30-10:30 QC vs 04:00-10:30 spec (est. ~117 true fails vs 417) |
| 2026-01-02 | Section 3.2: Label price source DECIDED - 11:30 bar open (entry), 15:59 close (exit) from `../data/stage1_raw/` |
| 2026-01-02 | Section 1.5: Fixed threshold documentation - three hard-fail gates from data_quality.py (70%/60m/10k vol) |
| 2026-01-02 | Section 3.2: Fixed filename pattern (`{symbol}_1min.parquet`), added edge case handling (early close, halts) |
| 2026-01-02 | Section 3.4: Fixed holdout coverage - META/SPY/TSLA stop at 2025-12-16, set holdout end = 2025-12-16 |
| 2026-01-02 | Section 3.2: Tightened early-close detection (explicit HALF_DAYS set, skip halts on normal days) |
| 2026-01-02 | Section 1.5: Clarified denominator (4,517 QC'd days vs 4,518 total symbol-days) |
