#!/usr/bin/env python3
"""
Download VIX Futures (VX) Daily Data from CBOE - FREE SOURCE

Downloads individual contract CSVs from CBOE's free historical data page,
then builds a continuous front-month (F1) series for VRP backtest.

Data source: https://www.cboe.com/us/futures/market_statistics/historical_data/
URL pattern: https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_YYYY-MM-DD.csv

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source ../venv/bin/activate
    python scripts/download_vx_futures_cboe.py
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import requests

# Configuration
OUTPUT_DIR = Path("/Users/Shared/wsl-export/wsl-home/dsp100k/data/vrp/futures")
CONTRACT_DIR = OUTPUT_DIR / "vx_contracts"
BASE_URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX"

# VIX futures expiration: 3rd Wednesday of each month (30 days before SPX options expiration)
# We'll generate expected expiration dates for monthly contracts

def get_third_wednesday(year: int, month: int) -> datetime:
    """Get the third Wednesday of a given month."""
    first_day = datetime(year, month, 1)
    # Find first Wednesday
    days_until_wed = (2 - first_day.weekday()) % 7
    first_wed = first_day + timedelta(days=days_until_wed)
    # Third Wednesday is 14 days later
    third_wed = first_wed + timedelta(days=14)
    return third_wed


def generate_monthly_expirations(start_year: int, end_year: int) -> List[str]:
    """Generate list of monthly expiration dates in YYYY-MM-DD format."""
    expirations = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            exp_date = get_third_wednesday(year, month)
            # VIX settlement is 30 days before SPX options expiration
            # Adjust: VIX futures expire on Wednesday morning
            expirations.append(exp_date.strftime("%Y-%m-%d"))
    return expirations


def download_contract(exp_date: str, output_dir: Path) -> Optional[Path]:
    """Download a single VX contract CSV."""
    filename = f"VX_{exp_date}.csv"
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 100:
        return output_path

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            return output_path
        elif response.status_code == 404:
            # Contract doesn't exist for this date
            return None
        else:
            print(f"  Warning: {url} returned {response.status_code}")
            return None
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return None


def load_contract(filepath: Path) -> pd.DataFrame:
    """Load a contract CSV and clean it."""
    try:
        df = pd.read_csv(filepath)
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        # Keep only rows with valid settlement prices
        df = df[df['Settle'] > 0].copy()
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame()


def build_continuous_f1(contract_dir: Path) -> pd.DataFrame:
    """
    Build continuous front-month (F1) series from individual contracts.

    Logic: For each trade date, use the settlement price of the contract
    expiring soonest but not yet expired.
    """
    # Load all contract files
    contracts = {}
    for filepath in sorted(contract_dir.glob("VX_*.csv")):
        exp_date_str = filepath.stem.replace("VX_", "")
        try:
            exp_date = pd.Timestamp(exp_date_str)
            df = load_contract(filepath)
            if not df.empty:
                contracts[exp_date] = df
        except:
            continue

    if not contracts:
        return pd.DataFrame()

    print(f"  Loaded {len(contracts)} contracts")

    # Sort by expiration
    sorted_exps = sorted(contracts.keys())

    # Build daily series
    all_dates = set()
    for df in contracts.values():
        all_dates.update(df['Trade Date'].tolist())

    all_dates = sorted(all_dates)

    # For each date, find the front-month contract
    records = []
    for trade_date in all_dates:
        # Find contracts expiring AFTER this trade date
        valid_exps = [exp for exp in sorted_exps if exp > trade_date]

        if not valid_exps:
            continue

        # Front month = first valid expiration
        f1_exp = valid_exps[0]
        f1_df = contracts[f1_exp]

        # Get settlement price for this trade date
        row = f1_df[f1_df['Trade Date'] == trade_date]
        if len(row) == 1:
            records.append({
                'date': trade_date,
                'vx_f1': row['Settle'].values[0],
                'f1_expiration': f1_exp,
                'volume': row['Total Volume'].values[0],
                'open_interest': row['Open Interest'].values[0]
            })

    df_f1 = pd.DataFrame(records)
    df_f1.set_index('date', inplace=True)
    return df_f1


def main():
    print("=" * 60)
    print("VIX Futures (VX) Daily Data Downloader - CBOE FREE SOURCE")
    print("=" * 60)
    print()

    # Create output directories
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate expected expiration dates (2010-2026)
    print("Generating expiration dates for 2010-2026...")
    expirations = generate_monthly_expirations(2010, 2026)
    print(f"  Generated {len(expirations)} monthly expirations")
    print()

    # Download contracts
    print("Downloading VX contracts from CBOE...")
    downloaded = 0
    skipped = 0
    failed = 0

    for i, exp_date in enumerate(expirations):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(expirations)} contracts...")

        result = download_contract(exp_date, CONTRACT_DIR)
        if result:
            downloaded += 1
        elif result is None:
            failed += 1
        else:
            skipped += 1

        # Rate limiting - be nice to CBOE servers
        time.sleep(0.1)

    print(f"\n  Downloaded: {downloaded}")
    print(f"  Not found: {failed}")
    print(f"  Total: {downloaded + failed}")
    print()

    # Also try weekly contracts for more recent years (2020+)
    print("Downloading weekly VX contracts (2020-2026)...")
    weekly_count = 0
    for year in range(2020, 2027):
        # Check all Wednesdays
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31) if year < 2026 else datetime(2026, 12, 31)

        current = start
        while current <= end:
            if current.weekday() == 2:  # Wednesday
                exp_date = current.strftime("%Y-%m-%d")
                result = download_contract(exp_date, CONTRACT_DIR)
                if result:
                    weekly_count += 1
                time.sleep(0.05)
            current += timedelta(days=1)

    print(f"  Downloaded {weekly_count} additional weekly contracts")
    print()

    # Build continuous F1 series
    print("Building continuous front-month (F1) series...")
    df_f1 = build_continuous_f1(CONTRACT_DIR)

    if df_f1.empty:
        print("  ERROR: No data to build continuous series!")
        return 1

    # Save to parquet
    output_file = OUTPUT_DIR / "VX_F1_CBOE.parquet"
    df_f1.to_parquet(output_file)

    print(f"\n  Saved: {output_file}")
    print(f"  Date range: {df_f1.index.min().date()} to {df_f1.index.max().date()}")
    print(f"  Rows: {len(df_f1):,}")
    print(f"  Columns: {list(df_f1.columns)}")

    # Also save as CSV for inspection
    csv_file = OUTPUT_DIR / "VX_F1_CBOE.csv"
    df_f1.to_csv(csv_file)
    print(f"  CSV copy: {csv_file}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  VX F1 Settlement Prices:")
    print(f"    Mean: {df_f1['vx_f1'].mean():.2f}")
    print(f"    Min:  {df_f1['vx_f1'].min():.2f}")
    print(f"    Max:  {df_f1['vx_f1'].max():.2f}")
    print(f"    Std:  {df_f1['vx_f1'].std():.2f}")

    # Check coverage
    print(f"\n  Data coverage:")
    years = df_f1.index.year.unique()
    for year in sorted(years):
        year_data = df_f1[df_f1.index.year == year]
        print(f"    {year}: {len(year_data):4d} days ({year_data.index.min().date()} to {year_data.index.max().date()})")

    print("\n✅ VX Futures data acquisition COMPLETE!")
    print("   Source: CBOE (FREE)")
    print(f"   File: {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
