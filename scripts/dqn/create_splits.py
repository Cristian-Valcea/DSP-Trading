#!/usr/bin/env python3
"""
Gate 1: Create train/val/dev_test/holdout data splits from stage1_raw.

This script creates temporally-separated data splits for DQN training:
- train: 2021-12-20 to 2023-12-31
- val: 2024-01-01 to 2024-06-30
- dev_test: 2024-07-01 to 2024-12-31
- holdout: 2025-01-01 to 2025-12-19 (DO NOT TOUCH until Gate 3)

Usage:
    python scripts/dqn/create_splits.py --source ../data/stage1_raw --output ../data
    python scripts/dqn/create_splits.py --dry-run  # Preview without writing
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# DQN Universe (9 symbols)
DQN_UNIVERSE = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "QQQ", "SPY", "TSLA"]

# Split definitions (inclusive dates)
SPLITS = {
    "train": {
        "start": "2021-12-20",
        "end": "2023-12-31",
        "description": "Training data for walk-forward folds",
    },
    "val": {
        "start": "2024-01-01",
        "end": "2024-06-30",
        "description": "Validation data for model selection",
    },
    "dev_test": {
        "start": "2024-07-01",
        "end": "2024-12-31",
        "description": "Development test set for debugging",
    },
    "holdout": {
        "start": "2025-01-01",
        "end": "2025-12-19",
        "description": "Holdout test set - DO NOT TOUCH until Gate 3",
    },
}


def load_symbol_data(source_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """Load minute data for a symbol from stage1_raw."""
    filepath = source_dir / f"{symbol.lower()}_1min.parquet"

    if not filepath.exists():
        print(f"  ⚠️  File not found: {filepath}")
        return None

    df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def filter_by_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter dataframe to date range (inclusive)."""
    mask = (df["timestamp"].dt.date >= pd.Timestamp(start).date()) & (
        df["timestamp"].dt.date <= pd.Timestamp(end).date()
    )
    return df[mask].copy()


def get_split_stats(df: pd.DataFrame) -> dict:
    """Compute statistics for a data split."""
    if df.empty:
        return {"rows": 0, "days": 0, "start": None, "end": None}

    return {
        "rows": len(df),
        "days": df["timestamp"].dt.date.nunique(),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }


def create_splits(
    source_dir: Path,
    output_dir: Path,
    symbols: list[str],
    dry_run: bool = False,
) -> dict:
    """Create train/val/dev_test/holdout splits for all symbols."""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "splits": {},
        "symbols": {},
    }

    # Create output directories
    if not dry_run:
        for split_name in SPLITS.keys():
            split_dir = output_dir / f"dqn_{split_name}"
            split_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gate 1: Creating Data Splits")
    print("=" * 70)

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be written\n")

    # Process each symbol
    for symbol in symbols:
        print(f"\n📊 Processing {symbol}...")

        df = load_symbol_data(source_dir, symbol)
        if df is None:
            manifest["symbols"][symbol] = {"error": "File not found"}
            continue

        symbol_stats = {"total_rows": len(df), "splits": {}}

        # Create each split
        for split_name, split_config in SPLITS.items():
            split_df = filter_by_date_range(
                df, split_config["start"], split_config["end"]
            )
            stats = get_split_stats(split_df)
            symbol_stats["splits"][split_name] = stats

            print(
                f"   {split_name:12s}: {stats['rows']:>8,} rows, {stats['days']:>4} days "
                f"({split_config['start']} → {split_config['end']})"
            )

            if not dry_run and not split_df.empty:
                output_path = (
                    output_dir / f"dqn_{split_name}" / f"{symbol.lower()}_{split_name}.parquet"
                )
                split_df.to_parquet(output_path, index=False)

        manifest["symbols"][symbol] = symbol_stats

    # Add split metadata to manifest
    for split_name, split_config in SPLITS.items():
        manifest["splits"][split_name] = {
            "start": split_config["start"],
            "end": split_config["end"],
            "description": split_config["description"],
        }

    # Save manifest
    if not dry_run:
        manifest_path = output_dir / "split_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n📄 Manifest saved to: {manifest_path}")

    return manifest


def print_summary(manifest: dict):
    """Print summary of created splits."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for split_name in SPLITS.keys():
        total_rows = 0
        total_days = 0
        for symbol, stats in manifest["symbols"].items():
            if "splits" in stats and split_name in stats["splits"]:
                total_rows += stats["splits"][split_name]["rows"]
                total_days = max(total_days, stats["splits"][split_name]["days"])

        print(f"\n{split_name.upper()}:")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Max days: {total_days}")

    # Count missing symbols
    missing = [s for s, stats in manifest["symbols"].items() if "error" in stats]
    if missing:
        print(f"\n⚠️  Missing symbols: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/dev_test/holdout splits for DQN"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="../data/stage1_raw",
        help="Source directory with *_1min.parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data",
        help="Output directory for split directories",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        help="Comma-separated symbols or 'all' for DQN universe",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview splits without writing files",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    source_dir = (script_dir / args.source).resolve()
    output_dir = (script_dir / args.output).resolve()

    # Parse symbols
    if args.symbols.lower() == "all":
        symbols = DQN_UNIVERSE
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Validate source directory
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        sys.exit(1)

    # Create splits
    manifest = create_splits(source_dir, output_dir, symbols, dry_run=args.dry_run)
    print_summary(manifest)

    # Final status
    missing = [s for s, stats in manifest["symbols"].items() if "error" in stats]
    if missing:
        print(f"\n❌ FAILED: {len(missing)} symbols missing")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: All {len(symbols)} symbols processed")
        if not args.dry_run:
            print(f"   Output: {output_dir}/dqn_*/")


if __name__ == "__main__":
    main()
