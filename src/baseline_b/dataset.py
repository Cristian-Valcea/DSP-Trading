"""
Dataset Generator for Baseline B

Generates (X, y) pairs for training/evaluation with comprehensive missing data tracking.

Labels:
- y(d,s) = log(P_exit / P_entry)
- P_entry = 10:31 bar open
- P_exit = 14:00 bar close

Index/time alignment:
- 09:30 = minute_idx 0
- 10:30 = minute_idx 60 (feature cutoff)
- 10:31 = minute_idx 61 (entry)
- 14:00 = minute_idx 270 (exit)
"""

from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data_loader import (
    SYMBOLS,
    get_bar_by_time,
    get_rth_day_data,
    get_trading_days,
    load_rth_data,
)
from .feature_builder import (
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureBuilder,
    get_prior_close,
)


@dataclass
class SkipStats:
    """Tracks reasons for skipped samples."""
    missing_entry_bar: int = 0
    missing_exit_bar: int = 0
    insufficient_rth_data: int = 0
    missing_premarket: int = 0
    other: int = 0

    def total(self) -> int:
        return (
            self.missing_entry_bar
            + self.missing_exit_bar
            + self.insufficient_rth_data
            + self.missing_premarket
            + self.other
        )

    def to_dict(self) -> dict:
        return {
            "missing_entry_bar": self.missing_entry_bar,
            "missing_exit_bar": self.missing_exit_bar,
            "insufficient_rth_data": self.insufficient_rth_data,
            "missing_premarket": self.missing_premarket,
            "other": self.other,
            "total_skipped": self.total(),
        }


@dataclass
class DatasetStats:
    """Statistics for generated dataset."""
    split: str
    total_trading_days: int = 0
    total_samples: int = 0
    samples_by_symbol: dict = field(default_factory=dict)
    skip_stats_by_symbol: dict = field(default_factory=dict)
    skip_stats_total: SkipStats = field(default_factory=SkipStats)
    premarket_bar_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "total_trading_days": self.total_trading_days,
            "total_samples": self.total_samples,
            "samples_by_symbol": self.samples_by_symbol,
            "skip_stats_total": self.skip_stats_total.to_dict(),
            "skip_stats_by_symbol": {
                s: stats.to_dict() for s, stats in self.skip_stats_by_symbol.items()
            },
        }


# Time constants
ENTRY_TIME = time(10, 31)
EXIT_TIME = time(14, 0)


class DatasetGenerator:
    """
    Generates (X, y) dataset for Baseline B supervised learning.

    Creates samples for each (day, symbol) with features computed at 10:30 ET
    and labels from 10:31 open to 14:00 close.
    """

    def __init__(
        self,
        symbols: list[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize dataset generator.

        Args:
            symbols: List of symbols (defaults to all 9)
            verbose: Show progress bars
        """
        self.symbols = symbols or SYMBOLS
        self.verbose = verbose
        self.feature_builder = FeatureBuilder(symbols=self.symbols)

    def _get_label(
        self,
        day_df: pd.DataFrame,
    ) -> tuple[Optional[float], str]:
        """
        Compute label y = log(P_exit / P_entry).

        Args:
            day_df: Single day RTH data

        Returns:
            Tuple of (label value or None, skip reason if None)
        """
        # Get entry bar (10:31)
        entry_bar = get_bar_by_time(day_df, ENTRY_TIME)
        if entry_bar is None:
            return None, "missing_entry_bar"

        # Get exit bar (14:00)
        exit_bar = get_bar_by_time(day_df, EXIT_TIME)
        if exit_bar is None:
            return None, "missing_exit_bar"

        # Compute log return
        p_entry = float(entry_bar["open"])
        p_exit = float(exit_bar["close"])

        if p_entry <= 0:
            return None, "invalid_entry_price"

        log_return = np.log(p_exit / p_entry)
        return log_return, ""

    def generate(self, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, DatasetStats]:
        """
        Generate dataset for a split.

        Args:
            split: One of train, val, dev_test, holdout

        Returns:
            Tuple of:
            - X: (N, NUM_FEATURES) feature array
            - y: (N,) label array
            - metadata: DataFrame with date, symbol for each sample
            - stats: DatasetStats with skip counts
        """
        stats = DatasetStats(split=split)

        # Get trading days
        trading_days = get_trading_days(split)
        stats.total_trading_days = len(trading_days)

        if self.verbose:
            print(f"Generating {split} dataset: {len(trading_days)} trading days, {len(self.symbols)} symbols")

        # Load all RTH data upfront
        rth_data = {}
        for symbol in self.symbols:
            rth_data[symbol] = load_rth_data(symbol, split)
            stats.samples_by_symbol[symbol] = 0
            stats.skip_stats_by_symbol[symbol] = SkipStats()

        # Generate samples
        X_list = []
        y_list = []
        metadata_list = []

        day_iter = tqdm(trading_days, desc=f"Processing {split}", disable=not self.verbose)

        for d in day_iter:
            # Get RTH data for this day for all symbols
            rth_day_data = {}
            for symbol in self.symbols:
                day_df = get_rth_day_data(rth_data[symbol], d)
                if not day_df.empty:
                    rth_day_data[symbol] = day_df

            # Process each symbol
            for symbol in self.symbols:
                if symbol not in rth_day_data:
                    stats.skip_stats_by_symbol[symbol].insufficient_rth_data += 1
                    stats.skip_stats_total.insufficient_rth_data += 1
                    continue

                day_df = rth_day_data[symbol]

                # Compute label
                label, skip_reason = self._get_label(day_df)
                if label is None:
                    if skip_reason == "missing_entry_bar":
                        stats.skip_stats_by_symbol[symbol].missing_entry_bar += 1
                        stats.skip_stats_total.missing_entry_bar += 1
                    elif skip_reason == "missing_exit_bar":
                        stats.skip_stats_by_symbol[symbol].missing_exit_bar += 1
                        stats.skip_stats_total.missing_exit_bar += 1
                    else:
                        stats.skip_stats_by_symbol[symbol].other += 1
                        stats.skip_stats_total.other += 1
                    continue

                # Get prior close for overnight gap feature
                prior_close = get_prior_close(rth_data[symbol], d)

                # Compute features
                features, diag = self.feature_builder.build_features(
                    rth_day_data, symbol, d, prior_close
                )

                if features is None:
                    reason = diag.get("skip_reason", "other")
                    if reason == "insufficient_rth_data":
                        stats.skip_stats_by_symbol[symbol].insufficient_rth_data += 1
                        stats.skip_stats_total.insufficient_rth_data += 1
                    else:
                        stats.skip_stats_by_symbol[symbol].other += 1
                        stats.skip_stats_total.other += 1
                    continue

                # Track premarket bar counts
                for seg in ["0400_0600", "0600_0800", "0800_0915"]:
                    key = f"bar_count_{seg}"
                    if key in diag:
                        if seg not in stats.premarket_bar_counts:
                            stats.premarket_bar_counts[seg] = []
                        stats.premarket_bar_counts[seg].append(diag[key])

                # Add sample
                X_list.append(features)
                y_list.append(label)
                metadata_list.append({"date": d, "symbol": symbol})

                stats.samples_by_symbol[symbol] += 1
                stats.total_samples += 1

        # Convert to arrays
        X = np.array(X_list, dtype=np.float32) if X_list else np.zeros((0, NUM_FEATURES), dtype=np.float32)
        y = np.array(y_list, dtype=np.float32) if y_list else np.zeros(0, dtype=np.float32)
        metadata = pd.DataFrame(metadata_list)

        if self.verbose:
            print(f"Generated {stats.total_samples} samples, skipped {stats.skip_stats_total.total()}")

        return X, y, metadata, stats


def generate_all_splits(
    output_dir: Path = None,
    symbols: list[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Generate datasets for all splits (except holdout by default).

    Args:
        output_dir: Optional directory to save arrays
        symbols: List of symbols
        verbose: Show progress

    Returns:
        Dict mapping split -> (X, y, metadata, stats)
    """
    generator = DatasetGenerator(symbols=symbols, verbose=verbose)

    results = {}
    for split in ["train", "val", "dev_test"]:
        X, y, metadata, stats = generator.generate(split)
        results[split] = (X, y, metadata, stats)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / f"X_{split}.npy", X)
            np.save(output_dir / f"y_{split}.npy", y)
            metadata.to_parquet(output_dir / f"metadata_{split}.parquet", index=False)

            # Save stats
            import json
            with open(output_dir / f"stats_{split}.json", "w") as f:
                json.dump(stats.to_dict(), f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import json

    print("=== Dataset Generator Validation ===")

    # Generate small test
    generator = DatasetGenerator(verbose=True)

    # Just test with val (smaller split)
    X, y, metadata, stats = generator.generate("val")

    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Samples: {stats.total_samples}")
    print(f"Trading days: {stats.total_trading_days}")
    print(f"Skip stats: {stats.skip_stats_total.to_dict()}")

    print("\nSamples by symbol:")
    for symbol, count in sorted(stats.samples_by_symbol.items()):
        print(f"  {symbol}: {count}")

    print("\nLabel statistics:")
    print(f"  Mean: {y.mean():.6f}")
    print(f"  Std: {y.std():.6f}")
    print(f"  Min: {y.min():.6f}")
    print(f"  Max: {y.max():.6f}")

    print("\n=== Validation Complete ===")
