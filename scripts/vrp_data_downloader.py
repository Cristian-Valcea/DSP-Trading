#!/usr/bin/env python3
"""
VRP Data Downloader - Downloads all required data for VRP backtest

Downloads:
- VIX Spot Index (Yahoo ^VIX) - 2010-2026
- VVIX Index (Yahoo ^VVIX) - 2010-2026
- Fed Funds Rate (FRED DFF) - 2010-2026
- 3-Month T-Bill Rate (FRED DTB3) - 2010-2026
- 10-Year Treasury (FRED DGS10) - 2010-2026
- VIX Futures (Nasdaq Data Link SCF/VX) - 2010-2026

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source ../venv/bin/activate

    # Download free data only (Yahoo + FRED)
    python scripts/vrp_data_downloader.py

    # Download VIX futures (requires Nasdaq API key)
    NASDAQ_API_KEY=your_key python scripts/vrp_data_downloader.py --vx-futures
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Try to import pandas_datareader, fall back to yfinance for FRED
try:
    import pandas_datareader as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    print("Warning: pandas_datareader not installed, using yfinance for all data")

# Try to import nasdaq-data-link for Quandl
try:
    import nasdaqdatalink
    HAS_NASDAQ = True
except ImportError:
    HAS_NASDAQ = False


# Configuration
START_DATE = "2010-01-01"
END_DATE = "2026-01-08"
OUTPUT_DIR = Path("/Users/Shared/wsl-export/wsl-home/dsp100k/data/vrp")


def download_yahoo_index(symbol: str, name: str, output_dir: Path) -> pd.DataFrame:
    """Download index data from Yahoo Finance."""
    print(f"Downloading {name} ({symbol}) from Yahoo Finance...")

    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            print(f"  ERROR: No data returned for {symbol}")
            return pd.DataFrame()

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only Close column and rename
        df = df[['Close']].copy()
        df.columns = [name.lower()]
        df.index.name = 'date'

        # Save to parquet
        output_file = output_dir / f"{name}.parquet"
        df.to_parquet(output_file)

        print(f"  Saved: {output_file}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Rows: {len(df):,}")

        return df

    except Exception as e:
        print(f"  ERROR downloading {symbol}: {e}")
        return pd.DataFrame()


def download_fred_series(series_id: str, name: str, output_dir: Path) -> pd.DataFrame:
    """Download series from FRED."""
    print(f"Downloading {name} ({series_id}) from FRED...")

    try:
        if HAS_PDR:
            df = pdr.get_data_fred(series_id, start=START_DATE, end=END_DATE)
        else:
            # Fallback: try yfinance ticker mapping
            # FRED series don't have direct yfinance equivalents for rates
            print(f"  Warning: pandas_datareader not available, trying alternative...")
            # Use fredapi if available
            try:
                from fredapi import Fred
                fred = Fred(api_key=os.environ.get('FRED_API_KEY', ''))
                df = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
                df = pd.DataFrame(df, columns=[series_id])
            except:
                print(f"  ERROR: Cannot download FRED data without pandas_datareader or fredapi")
                return pd.DataFrame()

        if df.empty:
            print(f"  ERROR: No data returned for {series_id}")
            return pd.DataFrame()

        # Rename column
        df.columns = [name.lower()]
        df.index.name = 'date'

        # Forward fill missing values (weekends/holidays)
        df = df.ffill()

        # Save to parquet
        output_file = output_dir / f"{name}.parquet"
        df.to_parquet(output_file)

        print(f"  Saved: {output_file}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Rows: {len(df):,}")

        return df

    except Exception as e:
        print(f"  ERROR downloading {series_id}: {e}")
        return pd.DataFrame()


def download_vx_futures(output_dir: Path, api_key: str = None) -> pd.DataFrame:
    """Download VIX Futures continuous series from Nasdaq Data Link (Quandl)."""
    print("Downloading VIX Futures (SCF/VX) from Nasdaq Data Link...")

    if not HAS_NASDAQ:
        print("  ERROR: nasdaq-data-link not installed. Run: pip install nasdaq-data-link")
        return pd.DataFrame()

    # Get API key from parameter or environment
    key = api_key or os.environ.get('NASDAQ_API_KEY') or os.environ.get('QUANDL_API_KEY')
    if not key:
        print("  ERROR: No API key found. Set NASDAQ_API_KEY environment variable.")
        print("  Get your free API key from: https://data.nasdaq.com/account/profile")
        return pd.DataFrame()

    try:
        # Configure API key
        nasdaqdatalink.ApiConfig.api_key = key

        # Download SCF/VX - Stevens Continuous Futures VIX
        # This provides a continuous front-month VIX futures series
        print("  Fetching SCF/VX (continuous front-month VIX futures)...")
        df = nasdaqdatalink.get("SCF/VX", start_date=START_DATE, end_date=END_DATE)

        if df.empty:
            print("  ERROR: No data returned for SCF/VX")
            return pd.DataFrame()

        # Rename columns for clarity
        df.index.name = 'date'

        # Save to parquet
        output_file = output_dir / "VX_continuous.parquet"
        df.to_parquet(output_file)

        print(f"  Saved: {output_file}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")

        return df

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("  ERROR: Invalid API key. Check your NASDAQ_API_KEY.")
        elif "403" in error_msg or "Forbidden" in error_msg:
            print("  ERROR: Access denied. Your account may not have access to SCF/VX.")
            print("  Note: SCF/VX may require a premium subscription.")
        elif "404" in error_msg or "Not Found" in error_msg:
            print("  ERROR: Dataset not found. SCF/VX may have been renamed or removed.")
        else:
            print(f"  ERROR downloading VX futures: {e}")
        return pd.DataFrame()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download VRP backtest data")
    parser.add_argument("--vx-futures", action="store_true",
                       help="Download VIX futures from Nasdaq Data Link (requires NASDAQ_API_KEY)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Nasdaq Data Link API key (or set NASDAQ_API_KEY env var)")
    parser.add_argument("--only-vx", action="store_true",
                       help="Only download VIX futures (skip Yahoo/FRED data)")
    args = parser.parse_args()

    print("=" * 60)
    print("VRP Data Downloader")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("=" * 60)
    print()

    # Create output directories
    indices_dir = OUTPUT_DIR / "indices"
    rates_dir = OUTPUT_DIR / "rates"
    futures_dir = OUTPUT_DIR / "futures"

    indices_dir.mkdir(parents=True, exist_ok=True)
    rates_dir.mkdir(parents=True, exist_ok=True)
    futures_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Skip Yahoo/FRED if --only-vx specified
    if not args.only_vx:
        # Download VIX indices from Yahoo
        print("=" * 40)
        print("STEP 1: Downloading VIX Indices (Yahoo)")
        print("=" * 40)

        results['VIX'] = download_yahoo_index("^VIX", "VIX_spot", indices_dir)
        print()

        results['VVIX'] = download_yahoo_index("^VVIX", "VVIX", indices_dir)
        print()

        # Download FRED rates
        print("=" * 40)
        print("STEP 2: Downloading Interest Rates (FRED)")
        print("=" * 40)

        results['DFF'] = download_fred_series("DFF", "fed_funds", rates_dir)
        print()

        results['DTB3'] = download_fred_series("DTB3", "tbill_3m", rates_dir)
        print()

        results['DGS10'] = download_fred_series("DGS10", "treasury_10y", rates_dir)
        print()

    # Download VIX Futures if requested
    if args.vx_futures or args.only_vx:
        print("=" * 40)
        print("STEP 3: Downloading VIX Futures (Nasdaq)")
        print("=" * 40)

        results['VX'] = download_vx_futures(futures_dir, api_key=args.api_key)
        print()

    # Summary
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for name, df in results.items():
        if not df.empty:
            print(f"  ✅ {name}: {df.index.min().date()} to {df.index.max().date()} ({len(df):,} rows)")
        else:
            print(f"  ❌ {name}: FAILED")

    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Check what's still needed
    vx_downloaded = 'VX' in results and not results['VX'].empty
    print("=" * 60)
    print("STATUS")
    print("=" * 60)
    if vx_downloaded:
        print("  ✅ VIX Futures (VX): Downloaded from Nasdaq Data Link")
    else:
        print("  ⏳ VIX Futures (VX): Run with --vx-futures flag to download")
    print("  ⏳ VIX1D: Only available from CBOE (since 2023)")
    print("  ⏳ VIX Options: Requires CBOE DataShop (Phase 3)")
    print()

    return 0 if all(not df.empty for df in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
