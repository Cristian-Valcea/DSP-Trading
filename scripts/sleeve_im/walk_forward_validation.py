#!/usr/bin/env python3
"""
Walk-Forward Validation for Sleeve IM Baseline Model.

Instead of a single train/val/test split, this script:
1. Creates multiple rolling train/test folds within 2023-2024
2. Trains a fresh logistic regression on each train fold
3. Tests on the immediately following test fold
4. Aggregates results to see if signal is consistent or regime-dependent

Goal: Detect "Q3 worked / Q4 broke" patterns immediately, without
touching the 2025 holdout.

Fold Structure (6-month train, 3-month test):
  Fold 1: Train 2023-01-01 to 2023-06-30, Test 2023-07-01 to 2023-09-30
  Fold 2: Train 2023-04-01 to 2023-09-30, Test 2023-10-01 to 2023-12-31
  Fold 3: Train 2023-07-01 to 2023-12-31, Test 2024-01-01 to 2024-03-31
  Fold 4: Train 2023-10-01 to 2024-03-31, Test 2024-04-01 to 2024-06-30
  Fold 5: Train 2024-01-01 to 2024-06-30, Test 2024-07-01 to 2024-09-30
  Fold 6: Train 2024-04-01 to 2024-09-30, Test 2024-10-01 to 2024-12-31

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source venv/bin/activate
    python scripts/sleeve_im/walk_forward_validation.py
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "sleeve_im" / "feature_dataset.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sleeve_im" / "walk_forward_results.json"

# Feature columns (same as baseline_model.py)
FEATURE_COLS = [
    "synthetic_pct",
    "max_staleness_min",
    "premarket_return",
    "first_hour_return",
    "feature_window_return",
    "premarket_vol_ratio",
    "premarket_volatility",
    "max_bar_range_pct",
    "avg_staleness_sec",
]

LABEL_COL = "label_binary"
TRANSACTION_COST = 0.0010  # 10 bps per side

# Walk-forward fold definitions
# Each fold: (train_start, train_end, test_start, test_end)
FOLDS = [
    # Fold 1: Early 2023 train → Q3 2023 test
    (date(2023, 1, 1), date(2023, 6, 30), date(2023, 7, 1), date(2023, 9, 30)),
    # Fold 2: Mid 2023 train → Q4 2023 test
    (date(2023, 4, 1), date(2023, 9, 30), date(2023, 10, 1), date(2023, 12, 31)),
    # Fold 3: Late 2023 train → Q1 2024 test
    (date(2023, 7, 1), date(2023, 12, 31), date(2024, 1, 1), date(2024, 3, 31)),
    # Fold 4: Cross-year train → Q2 2024 test
    (date(2023, 10, 1), date(2024, 3, 31), date(2024, 4, 1), date(2024, 6, 30)),
    # Fold 5: Early 2024 train → Q3 2024 test (matches original Val)
    (date(2024, 1, 1), date(2024, 6, 30), date(2024, 7, 1), date(2024, 9, 30)),
    # Fold 6: Mid 2024 train → Q4 2024 test (matches original Dev Test)
    (date(2024, 4, 1), date(2024, 9, 30), date(2024, 10, 1), date(2024, 12, 31)),
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Sample counts
    n_train: int
    n_test: int

    # Classification metrics
    train_accuracy: float
    test_accuracy: float

    # Trading metrics (on test set)
    n_trades: int  # Number of predicted positive (model=1)
    gross_return: float  # Sum of returns when model=1
    net_return: float  # After 20 bps round-trip cost per trade
    sharpe_ratio: float
    win_rate: float
    profit_factor: float

    # Diagnostics
    label_balance: float  # % positive labels in test
    pred_positive_rate: float  # % predicted positive

    def passes_kill_tests(self) -> bool:
        """Check if fold passes basic kill tests."""
        return (
            self.sharpe_ratio >= 0.0 and
            self.test_accuracy > 0.50 and
            self.profit_factor > 1.0
        )


# =============================================================================
# Core Functions
# =============================================================================


def load_dataset() -> pd.DataFrame:
    """Load and prepare feature dataset."""
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def filter_tradable(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to tradable days with valid labels and features."""
    mask = (
        df["is_tradable"] &
        df["label_binary"].notna() &
        df["label_return"].notna()
    )
    for col in FEATURE_COLS:
        mask = mask & df[col].notna()
    return df[mask].copy()


def get_fold_data(
    df: pd.DataFrame,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract train and test data for a fold."""
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    return train, test


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_id: int,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
) -> FoldResult:
    """Train model on fold and evaluate on test."""

    # Prepare features
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values.astype(int)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[LABEL_COL].values.astype(int)
    returns_test = test_df["label_return"].values

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    # Classification metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    # Trading metrics (only on test set)
    trade_mask = test_pred == 1
    n_trades = trade_mask.sum()

    if n_trades > 0:
        traded_returns = returns_test[trade_mask]
        gross_return = traded_returns.sum()
        # 20 bps round-trip cost per trade
        net_return = gross_return - (n_trades * 2 * TRANSACTION_COST)

        # Daily returns for Sharpe (include 0 for no-trade days)
        daily_returns = np.where(trade_mask, returns_test - 2 * TRANSACTION_COST, 0.0)
        if np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Win rate and profit factor
        wins = traded_returns[traded_returns > 0]
        losses = traded_returns[traded_returns <= 0]
        win_rate = len(wins) / len(traded_returns) if len(traded_returns) > 0 else 0.0

        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.001
        profit_factor = total_wins / total_losses
    else:
        gross_return = 0.0
        net_return = 0.0
        sharpe = 0.0
        win_rate = 0.0
        profit_factor = 0.0

    return FoldResult(
        fold_id=fold_id,
        train_start=str(train_start),
        train_end=str(train_end),
        test_start=str(test_start),
        test_end=str(test_end),
        n_train=len(train_df),
        n_test=len(test_df),
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        n_trades=int(n_trades),
        gross_return=float(gross_return),
        net_return=float(net_return),
        sharpe_ratio=float(sharpe),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        label_balance=float(y_test.mean()),
        pred_positive_rate=float(test_pred.mean()),
    )


def print_fold_result(result: FoldResult) -> None:
    """Print results for a single fold."""
    status = "✅ PASS" if result.passes_kill_tests() else "❌ FAIL"

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Fold {result.fold_id}: {result.test_start} to {result.test_end} [{status}]")
    logger.info(f"{'=' * 70}")
    logger.info(f"Train: {result.train_start} to {result.train_end} ({result.n_train:,} samples)")
    logger.info(f"Test:  {result.test_start} to {result.test_end} ({result.n_test:,} samples)")
    logger.info(f"\nClassification:")
    logger.info(f"  Train Accuracy: {result.train_accuracy:.3f}")
    logger.info(f"  Test Accuracy:  {result.test_accuracy:.3f}")
    logger.info(f"  Label Balance:  {result.label_balance:.1%} positive")
    logger.info(f"  Pred Positive:  {result.pred_positive_rate:.1%}")
    logger.info(f"\nTrading (on test):")
    logger.info(f"  Trades:        {result.n_trades} ({result.n_trades/result.n_test*100:.1f}% of days)")
    logger.info(f"  Gross Return:  {result.gross_return:+.2%}")
    logger.info(f"  Net Return:    {result.net_return:+.2%} (after {TRANSACTION_COST:.0%} per side)")
    logger.info(f"  Sharpe Ratio:  {result.sharpe_ratio:.3f}")
    logger.info(f"  Win Rate:      {result.win_rate:.1%}")
    logger.info(f"  Profit Factor: {result.profit_factor:.3f}")


def print_summary(results: List[FoldResult]) -> None:
    """Print aggregated summary across all folds."""

    logger.info(f"\n{'=' * 70}")
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info(f"{'=' * 70}")

    # Summary table
    logger.info(f"\n{'Fold':>4} | {'Test Period':>21} | {'Sharpe':>7} | {'Net Ret':>8} | {'Acc':>5} | {'PF':>5} | {'Status':>6}")
    logger.info("-" * 70)

    for r in results:
        status = "PASS" if r.passes_kill_tests() else "FAIL"
        logger.info(
            f"{r.fold_id:>4} | {r.test_start} to {r.test_end[:7]} | "
            f"{r.sharpe_ratio:>7.2f} | {r.net_return:>+7.2%} | "
            f"{r.test_accuracy:>5.1%} | {r.profit_factor:>5.2f} | {status:>6}"
        )

    # Aggregate metrics
    sharpes = [r.sharpe_ratio for r in results]
    net_returns = [r.net_return for r in results]
    accuracies = [r.test_accuracy for r in results]
    n_pass = sum(1 for r in results if r.passes_kill_tests())

    logger.info("-" * 70)
    logger.info(f"{'Mean':>4} | {'':>21} | {np.mean(sharpes):>7.2f} | {np.mean(net_returns):>+7.2%} | "
                f"{np.mean(accuracies):>5.1%} |       |")
    logger.info(f"{'Std':>4} | {'':>21} | {np.std(sharpes):>7.2f} | {np.std(net_returns):>7.2%} | "
                f"{np.std(accuracies):>5.1%} |       |")

    # Consistency analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("CONSISTENCY ANALYSIS")
    logger.info(f"{'=' * 70}")

    logger.info(f"\nFolds Passing Kill Tests: {n_pass}/{len(results)} ({n_pass/len(results)*100:.0f}%)")

    # Check for regime dependence
    positive_sharpe_folds = [r.fold_id for r in results if r.sharpe_ratio > 0]
    negative_sharpe_folds = [r.fold_id for r in results if r.sharpe_ratio <= 0]

    logger.info(f"Positive Sharpe Folds: {positive_sharpe_folds}")
    logger.info(f"Negative Sharpe Folds: {negative_sharpe_folds}")

    # Sharpe consistency
    sharpe_std = np.std(sharpes)
    sharpe_range = max(sharpes) - min(sharpes)

    if sharpe_std > 1.0 or sharpe_range > 2.0:
        logger.info(f"\n⚠️  HIGH SHARPE VARIANCE: std={sharpe_std:.2f}, range={sharpe_range:.2f}")
        logger.info("   Signal is highly regime-dependent!")
    elif np.mean(sharpes) < 0:
        logger.info(f"\n⚠️  MEAN SHARPE NEGATIVE: {np.mean(sharpes):.2f}")
        logger.info("   No consistent positive signal across folds!")

    # Final verdict
    logger.info(f"\n{'=' * 70}")
    logger.info("KILL DECISION RECOMMENDATION")
    logger.info(f"{'=' * 70}")

    if n_pass == len(results):
        logger.info("\n🟢 ALL FOLDS PASS - Signal appears robust, proceed to 2025 holdout")
    elif n_pass >= len(results) * 0.67:  # 4/6 or better
        logger.info(f"\n🟡 MAJORITY PASS ({n_pass}/{len(results)}) - Marginal signal, investigate failing folds")
    elif n_pass >= 1:
        logger.info(f"\n🟠 FEW PASS ({n_pass}/{len(results)}) - Likely regime-fitting, not real signal")
    else:
        logger.info(f"\n🔴 NO FOLDS PASS - KILL Sleeve IM, no exploitable signal exists")

    # Net return consistency
    if all(r < 0 for r in net_returns):
        logger.info("\n🔴 ALL FOLDS HAVE NEGATIVE NET RETURNS")
        logger.info("   Transaction costs exceed any signal edge - KILL SLEEVE IM")


def main():
    """Run walk-forward validation."""
    logger.info("=" * 70)
    logger.info("Walk-Forward Validation for Sleeve IM Baseline")
    logger.info("=" * 70)

    # Load data
    df = load_dataset()
    df_tradable = filter_tradable(df)
    logger.info(f"Total tradable samples: {len(df_tradable):,}")

    # Run each fold
    results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(FOLDS, 1):
        logger.info(f"\n--- Processing Fold {i} ---")

        train_df, test_df = get_fold_data(df_tradable, train_start, train_end, test_start, test_end)

        if len(train_df) < 100 or len(test_df) < 50:
            logger.warning(f"Fold {i} has insufficient data: train={len(train_df)}, test={len(test_df)}")
            continue

        result = train_and_evaluate_fold(
            train_df, test_df, i,
            train_start, train_end, test_start, test_end
        )
        results.append(result)
        print_fold_result(result)

    # Print summary
    if results:
        print_summary(results)

        # Save results
        output_data = {
            "folds": [
                {
                    "fold_id": r.fold_id,
                    "train_period": f"{r.train_start} to {r.train_end}",
                    "test_period": f"{r.test_start} to {r.test_end}",
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "train_accuracy": r.train_accuracy,
                    "test_accuracy": r.test_accuracy,
                    "n_trades": r.n_trades,
                    "gross_return": r.gross_return,
                    "net_return": r.net_return,
                    "sharpe_ratio": r.sharpe_ratio,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "passes_kill_tests": r.passes_kill_tests(),
                }
                for r in results
            ],
            "summary": {
                "n_folds": len(results),
                "n_pass": sum(1 for r in results if r.passes_kill_tests()),
                "mean_sharpe": float(np.mean([r.sharpe_ratio for r in results])),
                "std_sharpe": float(np.std([r.sharpe_ratio for r in results])),
                "mean_net_return": float(np.mean([r.net_return for r in results])),
                "all_net_returns_negative": all(r.net_return < 0 for r in results),
            },
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
