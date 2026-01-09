"""
VX Term Structure Builder

Builds continuous VX1, VX2, VX3, VX4 series from individual contract files.
This is critical data for VRP calendar spread strategies.

The key insight: we don't just short VIX (that failed). Instead we:
1. Harvest roll yield via calendar spreads (long VX2, short VX1)
2. Use regime gating to avoid spike losses
3. Size based on term structure slope
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_contract_expiry(filename: str) -> datetime:
    """Extract expiry date from filename like VX_2024-01-17.csv"""
    date_str = filename.replace("VX_", "").replace(".csv", "")
    return datetime.strptime(date_str, "%Y-%m-%d")


def load_all_contracts(contracts_dir: str) -> Dict[datetime, pd.DataFrame]:
    """
    Load all VX contract files.

    Returns:
        Dict mapping expiry date -> DataFrame with OHLC data
    """
    contracts = {}

    for filename in os.listdir(contracts_dir):
        if not filename.endswith(".csv") or not filename.startswith("VX_"):
            continue

        expiry = parse_contract_expiry(filename)
        filepath = os.path.join(contracts_dir, filename)

        try:
            df = pd.read_csv(filepath)
            df["Trade Date"] = pd.to_datetime(df["Trade Date"])
            df = df.set_index("Trade Date").sort_index()

            # Keep only useful columns
            df = df[["Settle", "Total Volume", "Open Interest"]].copy()
            df.columns = ["settle", "volume", "oi"]

            # Remove rows with 0 settle (no trading)
            df = df[df["settle"] > 0]

            if len(df) > 0:
                contracts[expiry] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return contracts


def get_front_months(
    trade_date: datetime,
    contracts: Dict[datetime, pd.DataFrame],
    n_months: int = 4
) -> List[Tuple[datetime, float]]:
    """
    Get the N front-month contracts for a given trade date.

    Returns:
        List of (expiry_date, settle_price) for VX1, VX2, VX3, etc.
    """
    # Find contracts expiring AFTER trade_date, sorted by expiry
    future_expiries = sorted([exp for exp in contracts.keys() if exp > trade_date])

    results = []
    for expiry in future_expiries[:n_months]:
        df = contracts[expiry]
        if trade_date in df.index:
            settle = df.loc[trade_date, "settle"]
            results.append((expiry, settle))
        elif trade_date < df.index.min():
            # Contract not trading yet on this date
            continue
        else:
            # Try to find closest prior date (for weekend gaps)
            prior_dates = df.index[df.index <= trade_date]
            if len(prior_dates) > 0:
                closest = prior_dates[-1]
                # Only use if within 5 trading days
                if (trade_date - closest).days <= 7:
                    settle = df.loc[closest, "settle"]
                    results.append((expiry, settle))

    return results


def build_term_structure(
    contracts_dir: str,
    start_date: str = "2013-06-01",
    end_date: str = "2025-12-31",
    n_months: int = 4
) -> pd.DataFrame:
    """
    Build continuous term structure series (VX1, VX2, VX3, VX4).

    Args:
        contracts_dir: Path to directory with VX_YYYY-MM-DD.csv files
        start_date: Start of output series
        end_date: End of output series
        n_months: Number of contract months to track

    Returns:
        DataFrame with columns: vx1, vx2, vx3, vx4, vx1_expiry, vx2_expiry, etc.
    """
    print("Loading all VX contracts...")
    contracts = load_all_contracts(contracts_dir)
    print(f"Loaded {len(contracts)} contracts")

    # Generate all trading days
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")

    records = []
    for trade_date in all_dates:
        front_months = get_front_months(trade_date.to_pydatetime(), contracts, n_months)

        if len(front_months) >= 2:  # Need at least VX1 and VX2
            record = {"date": trade_date}
            for i, (expiry, settle) in enumerate(front_months, 1):
                record[f"vx{i}"] = settle
                record[f"vx{i}_expiry"] = expiry
                record[f"vx{i}_dte"] = (expiry - trade_date.to_pydatetime()).days
            records.append(record)

    df = pd.DataFrame(records)
    df = df.set_index("date").sort_index()

    # Forward fill small gaps (holidays)
    df = df.ffill(limit=3)

    print(f"Built term structure: {len(df)} trading days")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def compute_term_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute useful features from the term structure.

    Features:
    - contango_1_2: (VX2 - VX1) / VX1 * 100 (% premium of VX2 over VX1)
    - contango_2_3: (VX3 - VX2) / VX2 * 100
    - term_slope: Linear regression slope of VX1-VX4 curve
    - roll_yield: Annualized roll yield if holding VX1
    - vix_basis: VX1 vs VIX spot (if available)
    """
    result = df.copy()

    # Ensure numeric columns are numeric
    for col in ["vx1", "vx2", "vx3", "vx4"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Contango between adjacent months
    if "vx1" in result.columns and "vx2" in result.columns:
        result["contango_1_2"] = (result["vx2"] - result["vx1"]) / result["vx1"] * 100

    if "vx2" in result.columns and "vx3" in result.columns:
        result["contango_2_3"] = (result["vx3"] - result["vx2"]) / result["vx2"] * 100

    if "vx3" in result.columns and "vx4" in result.columns:
        result["contango_3_4"] = (result["vx4"] - result["vx3"]) / result["vx3"] * 100

    # Average contango (term structure slope)
    contango_cols = ["contango_1_2", "contango_2_3", "contango_3_4"]
    contango_cols = [c for c in contango_cols if c in result.columns]
    if contango_cols:
        result["avg_contango"] = result[contango_cols].mean(axis=1, skipna=True)

    # Annualized roll yield (from VX1 decay to spot)
    if "contango_1_2" in result.columns and "vx1_dte" in result.columns:
        # Approximate: VX1 will converge to spot at expiry
        # Roll yield = (VX1 - implied_spot) / VX1 * (365 / DTE)
        # Simplified: use contango_1_2 as proxy
        dte = pd.to_numeric(result["vx1_dte"], errors="coerce").clip(lower=1)
        result["ann_roll_yield"] = result["contango_1_2"] * (365 / dte)

    return result


def save_term_structure(df: pd.DataFrame, output_dir: str) -> None:
    """Save term structure to parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full term structure
    output_path = os.path.join(output_dir, "vx_term_structure.parquet")
    df.to_parquet(output_path)
    print(f"Saved: {output_path}")

    # Also save individual series for convenience
    for col in ["vx1", "vx2", "vx3", "vx4"]:
        if col in df.columns:
            series_df = df[[col, f"{col}_expiry", f"{col}_dte"]].copy()
            series_path = os.path.join(output_dir, f"{col.upper()}_continuous.parquet")
            series_df.to_parquet(series_path)
            print(f"Saved: {series_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build VX term structure")
    parser.add_argument(
        "--contracts-dir",
        default="data/vrp/futures/vx_contracts",
        help="Directory with VX contract CSVs"
    )
    parser.add_argument(
        "--output-dir",
        default="data/vrp/term_structure",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--start", default="2013-06-01", help="Start date"
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="End date"
    )

    args = parser.parse_args()

    # Build term structure
    df = build_term_structure(
        contracts_dir=args.contracts_dir,
        start_date=args.start,
        end_date=args.end,
        n_months=4
    )

    # Compute features
    df = compute_term_structure_features(df)

    # Print summary stats
    print("\n" + "=" * 60)
    print("TERM STRUCTURE SUMMARY")
    print("=" * 60)

    if "contango_1_2" in df.columns:
        print(f"\nVX2-VX1 Contango (%):")
        print(f"  Mean: {df['contango_1_2'].mean():.2f}%")
        print(f"  Std:  {df['contango_1_2'].std():.2f}%")
        print(f"  Min:  {df['contango_1_2'].min():.2f}%")
        print(f"  Max:  {df['contango_1_2'].max():.2f}%")
        print(f"  % Positive: {(df['contango_1_2'] > 0).mean() * 100:.1f}%")

    if "avg_contango" in df.columns:
        print(f"\nAverage Contango (all months):")
        print(f"  Mean: {df['avg_contango'].mean():.2f}%")

    # Save
    save_term_structure(df, args.output_dir)

    print("\nDone!")
