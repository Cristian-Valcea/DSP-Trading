"""
Dataset Generator for Baseline C

Generates (X, y) pairs for 4 intervals with comprehensive missing data tracking.

Intervals:
- 10:31→11:31: y = log(P_11:31_open / P_10:31_open)
- 11:31→12:31: y = log(P_12:31_open / P_11:31_open)
- 12:31→14:00: y = log(P_14:00_open / P_12:31_open)
- 14:00→next_10:31: y = log(P_next_10:31_open / P_14:00_open) [overnight]

Key differences from Baseline B:
- 4 separate interval datasets (one per Ridge model)
- Overnight labels can span weekends/holidays
- Skip cross-split boundary overnight labels
- Skip early-close days entirely (no 14:00 bar)
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
    REBALANCE_TIMES,
    INTERVALS,
    get_bar_by_time,
    get_bar_open_price,
    get_rth_day_data,
    get_trading_days,
    is_early_close_day,
    load_rth_data,
)
from .feature_builder import (
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureBuilder,
    get_prior_close,
)


# Interval definitions for labels
INTERVAL_DEFS = [
    ("10:31->11:31", time(10, 31), time(11, 31), False),   # (name, start, end, is_overnight)
    ("11:31->12:31", time(11, 31), time(12, 31), False),
    ("12:31->14:00", time(12, 31), time(14, 0), False),
    ("14:00->next10:31", time(14, 0), time(10, 31), True),  # overnight
]


@dataclass
class SkipStats:
    """Tracks reasons for skipped samples."""
    missing_start_bar: int = 0
    missing_end_bar: int = 0
    insufficient_rth_data: int = 0
    missing_premarket: int = 0
    early_close_day: int = 0
    cross_split_boundary: int = 0
    missing_next_day: int = 0
    other: int = 0

    def total(self) -> int:
        return (
            self.missing_start_bar
            + self.missing_end_bar
            + self.insufficient_rth_data
            + self.missing_premarket
            + self.early_close_day
            + self.cross_split_boundary
            + self.missing_next_day
            + self.other
        )

    def to_dict(self) -> dict:
        return {
            "missing_start_bar": self.missing_start_bar,
            "missing_end_bar": self.missing_end_bar,
            "insufficient_rth_data": self.insufficient_rth_data,
            "missing_premarket": self.missing_premarket,
            "early_close_day": self.early_close_day,
            "cross_split_boundary": self.cross_split_boundary,
            "missing_next_day": self.missing_next_day,
            "other": self.other,
            "total_skipped": self.total(),
        }


@dataclass
class DatasetStats:
    """Statistics for generated dataset."""
    split: str
    interval: str
    total_trading_days: int = 0
    total_samples: int = 0
    samples_by_symbol: dict = field(default_factory=dict)
    skip_stats_by_symbol: dict = field(default_factory=dict)
    skip_stats_total: SkipStats = field(default_factory=SkipStats)
    premarket_bar_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "interval": self.interval,
            "total_trading_days": self.total_trading_days,
            "total_samples": self.total_samples,
            "samples_by_symbol": self.samples_by_symbol,
            "skip_stats_total": self.skip_stats_total.to_dict(),
            "skip_stats_by_symbol": {
                s: stats.to_dict() for s, stats in self.skip_stats_by_symbol.items()
            },
        }


class DatasetGenerator:
    """
    Generates (X, y) dataset for Baseline C supervised learning.

    Creates samples for each (day, symbol, interval) with features computed
    at the interval start time and labels from start_open to end_open.
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

    def _get_intraday_label(
        self,
        day_df: pd.DataFrame,
        start_time: time,
        end_time: time,
    ) -> tuple[Optional[float], str]:
        """
        Compute intraday label y = log(P_end_open / P_start_open).

        Args:
            day_df: Single day RTH data
            start_time: Interval start time
            end_time: Interval end time

        Returns:
            Tuple of (label value or None, skip reason if None)
        """
        # Get start bar
        p_start = get_bar_open_price(day_df, start_time)
        if p_start is None:
            return None, "missing_start_bar"

        # Get end bar
        p_end = get_bar_open_price(day_df, end_time)
        if p_end is None:
            return None, "missing_end_bar"

        if p_start <= 0:
            return None, "invalid_start_price"

        log_return = np.log(p_end / p_start)
        return log_return, ""

    def _get_overnight_label(
        self,
        current_day_df: pd.DataFrame,
        next_day_df: Optional[pd.DataFrame],
    ) -> tuple[Optional[float], str]:
        """
        Compute overnight label y = log(P_next_10:31_open / P_14:00_open).

        Args:
            current_day_df: Current day RTH data
            next_day_df: Next trading day RTH data (or None if not available)

        Returns:
            Tuple of (label value or None, skip reason if None)
        """
        # Get 14:00 open price from current day
        p_start = get_bar_open_price(current_day_df, time(14, 0))
        if p_start is None:
            return None, "missing_start_bar"

        if next_day_df is None:
            return None, "missing_next_day"

        # Get 10:31 open price from next day
        p_end = get_bar_open_price(next_day_df, time(10, 31))
        if p_end is None:
            return None, "missing_end_bar"

        if p_start <= 0:
            return None, "invalid_start_price"

        log_return = np.log(p_end / p_start)
        return log_return, ""

    def generate(
        self,
        split: str,
        interval_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, DatasetStats]:
        """
        Generate dataset for a split and specific interval.

        Args:
            split: One of train, val, dev_test, holdout
            interval_idx: 0-3 for the four intervals

        Returns:
            Tuple of:
            - X: (N, 45) feature array
            - y: (N,) label array
            - metadata: DataFrame with date, symbol, interval for each sample
            - stats: DatasetStats with skip counts
        """
        interval_name, start_time, end_time, is_overnight = INTERVAL_DEFS[interval_idx]
        stats = DatasetStats(split=split, interval=interval_name)

        # Get trading days for this split and next split (for boundary checking)
        trading_days = get_trading_days(split)
        trading_days_set = set(trading_days)
        stats.total_trading_days = len(trading_days)

        # For overnight labels, we need to know if next day crosses split boundary
        # Get next split's first day for boundary checking
        next_split_first_day = None
        if is_overnight:
            next_split_map = {"train": "val", "val": "dev_test", "dev_test": "holdout"}
            if split in next_split_map:
                try:
                    next_split_days = get_trading_days(next_split_map[split])
                    if next_split_days:
                        next_split_first_day = next_split_days[0]
                except FileNotFoundError:
                    pass

        if self.verbose:
            print(f"Generating {split}/{interval_name}: {len(trading_days)} trading days, {len(self.symbols)} symbols")

        # Load all RTH data upfront
        rth_data = {}
        for symbol in self.symbols:
            rth_data[symbol] = load_rth_data(symbol, split)
            stats.samples_by_symbol[symbol] = 0
            stats.skip_stats_by_symbol[symbol] = SkipStats()

        # For overnight labels, also load next split data
        next_split_rth_data = {}
        if is_overnight and split in next_split_map:
            try:
                for symbol in self.symbols:
                    next_split_rth_data[symbol] = load_rth_data(symbol, next_split_map[split])
            except FileNotFoundError:
                pass

        # Generate samples
        X_list = []
        y_list = []
        metadata_list = []

        day_iter = tqdm(trading_days, desc=f"Processing {split}/{interval_name}", disable=not self.verbose)

        for day_idx, d in enumerate(day_iter):
            # Get RTH data for this day for all symbols
            rth_day_data = {}
            for symbol in self.symbols:
                day_df = get_rth_day_data(rth_data[symbol], d)
                if not day_df.empty:
                    rth_day_data[symbol] = day_df

            # Check if this is an early close day (skip entirely per spec)
            # Early close means no 14:00 bar, which affects ALL intervals on that day
            sample_day_df = list(rth_day_data.values())[0] if rth_day_data else None
            if sample_day_df is not None and is_early_close_day(sample_day_df):
                for symbol in self.symbols:
                    stats.skip_stats_by_symbol[symbol].early_close_day += 1
                    stats.skip_stats_total.early_close_day += 1
                continue

            # For overnight interval, get next trading day data
            next_day_data = None
            if is_overnight:
                # Find next trading day
                next_day = None
                if day_idx + 1 < len(trading_days):
                    next_day = trading_days[day_idx + 1]
                elif next_split_first_day:
                    # Next day is in the next split - this is a cross-boundary label
                    for symbol in self.symbols:
                        stats.skip_stats_by_symbol[symbol].cross_split_boundary += 1
                        stats.skip_stats_total.cross_split_boundary += 1
                    continue

                if next_day:
                    # Check if next_day crosses into next split (boundary case)
                    if next_split_first_day and next_day >= next_split_first_day:
                        for symbol in self.symbols:
                            stats.skip_stats_by_symbol[symbol].cross_split_boundary += 1
                            stats.skip_stats_total.cross_split_boundary += 1
                        continue

                    next_day_data = {}
                    for symbol in self.symbols:
                        # First try current split data
                        nday_df = get_rth_day_data(rth_data[symbol], next_day)
                        if nday_df.empty and symbol in next_split_rth_data:
                            # Try next split data
                            nday_df = get_rth_day_data(next_split_rth_data[symbol], next_day)
                        if not nday_df.empty:
                            next_day_data[symbol] = nday_df

            # Process each symbol
            for symbol in self.symbols:
                if symbol not in rth_day_data:
                    stats.skip_stats_by_symbol[symbol].insufficient_rth_data += 1
                    stats.skip_stats_total.insufficient_rth_data += 1
                    continue

                day_df = rth_day_data[symbol]

                # Compute label based on interval type
                if is_overnight:
                    next_df = next_day_data.get(symbol) if next_day_data else None
                    label, skip_reason = self._get_overnight_label(day_df, next_df)
                else:
                    label, skip_reason = self._get_intraday_label(day_df, start_time, end_time)

                if label is None:
                    sym_stats = stats.skip_stats_by_symbol[symbol]
                    if skip_reason == "missing_start_bar":
                        sym_stats.missing_start_bar += 1
                        stats.skip_stats_total.missing_start_bar += 1
                    elif skip_reason == "missing_end_bar":
                        sym_stats.missing_end_bar += 1
                        stats.skip_stats_total.missing_end_bar += 1
                    elif skip_reason == "missing_next_day":
                        sym_stats.missing_next_day += 1
                        stats.skip_stats_total.missing_next_day += 1
                    else:
                        sym_stats.other += 1
                        stats.skip_stats_total.other += 1
                    continue

                # Get prior close for overnight gap feature
                prior_close = get_prior_close(rth_data[symbol], d)

                # Compute features at the interval start time
                features, diag = self.feature_builder.build_features(
                    rth_day_data, symbol, d, start_time, prior_close
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

                # Track premarket bar counts (only meaningful for first interval)
                if interval_idx == 0:
                    for seg in ["0400_0600", "0600_0800", "0800_0915"]:
                        key = f"bar_count_{seg}"
                        if key in diag:
                            if seg not in stats.premarket_bar_counts:
                                stats.premarket_bar_counts[seg] = []
                            stats.premarket_bar_counts[seg].append(diag[key])

                # Add sample
                X_list.append(features)
                y_list.append(label)
                metadata_list.append({
                    "date": d,
                    "symbol": symbol,
                    "interval": interval_name,
                    "rebalance_time": str(start_time),
                })

                stats.samples_by_symbol[symbol] += 1
                stats.total_samples += 1

        # Convert to arrays
        X = np.array(X_list, dtype=np.float32) if X_list else np.zeros((0, NUM_FEATURES), dtype=np.float32)
        y = np.array(y_list, dtype=np.float32) if y_list else np.zeros(0, dtype=np.float32)
        metadata = pd.DataFrame(metadata_list)

        if self.verbose:
            print(f"Generated {stats.total_samples} samples, skipped {stats.skip_stats_total.total()}")

        return X, y, metadata, stats

    def generate_all_intervals(
        self,
        split: str,
    ) -> dict:
        """
        Generate datasets for all 4 intervals for a split.

        Args:
            split: One of train, val, dev_test, holdout

        Returns:
            Dict mapping interval_idx -> (X, y, metadata, stats)
        """
        results = {}
        for interval_idx in range(4):
            X, y, metadata, stats = self.generate(split, interval_idx)
            results[interval_idx] = (X, y, metadata, stats)
        return results


def generate_all_splits(
    output_dir: Path = None,
    symbols: list[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Generate datasets for all splits and all intervals.

    Args:
        output_dir: Optional directory to save arrays
        symbols: List of symbols
        verbose: Show progress

    Returns:
        Dict mapping (split, interval_idx) -> (X, y, metadata, stats)
    """
    import json

    generator = DatasetGenerator(symbols=symbols, verbose=verbose)

    results = {}
    for split in ["train", "val", "dev_test"]:
        for interval_idx in range(4):
            X, y, metadata, stats = generator.generate(split, interval_idx)
            results[(split, interval_idx)] = (X, y, metadata, stats)

            if output_dir:
                output_dir = Path(output_dir)
                interval_name = INTERVAL_DEFS[interval_idx][0].replace(":", "").replace("->", "_to_")
                split_dir = output_dir / split / interval_name
                split_dir.mkdir(parents=True, exist_ok=True)

                np.save(split_dir / "X.npy", X)
                np.save(split_dir / "y.npy", y)
                metadata.to_parquet(split_dir / "metadata.parquet", index=False)

                with open(split_dir / "stats.json", "w") as f:
                    json.dump(stats.to_dict(), f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import json

    print("=== Baseline C Dataset Generator Validation ===")

    # Generate test for val split
    generator = DatasetGenerator(verbose=True)

    print("\n--- Testing all 4 intervals on VAL split ---")
    for interval_idx in range(4):
        interval_name = INTERVAL_DEFS[interval_idx][0]
        print(f"\n--- Interval {interval_idx}: {interval_name} ---")

        X, y, metadata, stats = generator.generate("val", interval_idx)

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        print(f"Samples: {stats.total_samples}")
        print(f"Trading days: {stats.total_trading_days}")
        print(f"Skip stats: {stats.skip_stats_total.to_dict()}")

        if len(y) > 0:
            print(f"Label stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")

    print("\n=== Validation Complete ===")
