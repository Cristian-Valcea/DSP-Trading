#!/usr/bin/env python3
"""
Phase 3: Baseline Model for Sleeve IM.

This script:
1. Loads the feature dataset from build_features.py
2. Splits into train/val/test by date
3. Trains a logistic regression baseline
4. Evaluates and computes kill-test metrics (Sharpe, accuracy, etc.)

Usage:
    cd /Users/Shared/wsl-export/wsl-home/dsp100k
    source venv/bin/activate
    python scripts/sleeve_im/baseline_model.py

Kill Test Criteria (from POST_BACKFILL_CHECKLIST.md):
- Sharpe ratio ≥ 0.0 (or 0.2 for "tradable")
- Accuracy > 50% (better than coin flip)
- Profit factor > 1.0 (winners > losers in dollar terms)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "sleeve_im" / "feature_dataset.parquet"

# Train/Val/Test splits (from POST_BACKFILL_CHECKLIST.md Section 3.3)
TRAIN_START = date(2023, 1, 1)
TRAIN_END = date(2024, 6, 30)

VAL_START = date(2024, 7, 1)
VAL_END = date(2024, 9, 30)

TEST_START = date(2024, 10, 1)
TEST_END = date(2024, 12, 31)

# Features to use (numeric features from feature_dataset)
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

# Label column
LABEL_COL = "label_binary"

# Transaction costs (one-way, in decimal)
TRANSACTION_COST = 0.0010  # 10 bps per side


# =============================================================================
# Data Loading and Splitting
# =============================================================================


def load_dataset() -> pd.DataFrame:
    """Load feature dataset."""
    logger.info(f"Loading dataset from {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test by date."""
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]

    logger.info(f"Train: {len(train):,} rows ({TRAIN_START} to {TRAIN_END})")
    logger.info(f"Val:   {len(val):,} rows ({VAL_START} to {VAL_END})")
    logger.info(f"Test:  {len(test):,} rows ({TEST_START} to {TEST_END})")

    return train, val, test


def filter_tradable(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to tradable days with valid labels."""
    mask = (
        df["is_tradable"]
        & df["label_binary"].notna()
        & df["label_return"].notna()
    )
    # Also filter for valid features (no NaN in feature columns)
    for col in FEATURE_COLS:
        mask = mask & df[col].notna()

    filtered = df[mask].copy()
    logger.info(f"  Filtered to {len(filtered):,} tradable rows with valid features")
    return filtered


# =============================================================================
# Model Training
# =============================================================================


def prepare_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Prepare feature matrices and fit scaler on training data."""

    X_train = train[FEATURE_COLS].values
    y_train = train[LABEL_COL].values.astype(int)

    X_val = val[FEATURE_COLS].values
    y_val = val[LABEL_COL].values.astype(int)

    X_test = test[FEATURE_COLS].values
    y_test = test[LABEL_COL].values.astype(int)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train logistic regression baseline."""
    logger.info("Training logistic regression...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# Evaluation Metrics
# =============================================================================


@dataclass
class EvalMetrics:
    """Evaluation metrics for a dataset."""

    split_name: str
    n_samples: int

    # Classification metrics
    accuracy: float
    precision_1: float
    recall_1: float
    f1_1: float

    # Trading metrics (simulated)
    total_return: float
    total_return_net: float  # After transaction costs
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = [
            f"\n{'=' * 60}",
            f"{self.split_name} Results (n={self.n_samples:,})",
            f"{'=' * 60}",
            f"\nClassification Metrics:",
            f"  Accuracy:     {self.accuracy:.3f}",
            f"  Precision(1): {self.precision_1:.3f}",
            f"  Recall(1):    {self.recall_1:.3f}",
            f"  F1(1):        {self.f1_1:.3f}",
            f"\nTrading Metrics:",
            f"  Total Return:     {self.total_return:+.2%}",
            f"  Total Return Net: {self.total_return_net:+.2%} (after {TRANSACTION_COST:.0%} tx cost)",
            f"  Sharpe Ratio:     {self.sharpe_ratio:.3f}",
            f"  Profit Factor:    {self.profit_factor:.3f}",
            f"  Win Rate:         {self.win_rate:.1%}",
            f"  Avg Win:          {self.avg_win:+.2%}",
            f"  Avg Loss:         {self.avg_loss:+.2%}",
        ]
        return "\n".join(lines)


def compute_trading_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute trading metrics assuming long-only strategy.

    Strategy: Go long when prediction=1, stay flat when prediction=0.
    """
    # Trade returns: actual return when we predicted 1, 0 otherwise
    trade_returns = np.where(predictions == 1, actual_returns, 0.0)

    # Apply transaction costs: 2 * cost per trade (round-trip)
    trade_costs = np.where(predictions == 1, 2 * TRANSACTION_COST, 0.0)
    trade_returns_net = trade_returns - trade_costs

    # Total returns
    total_return = np.sum(trade_returns)
    total_return_net = np.sum(trade_returns_net)

    # Sharpe ratio (annualized, assuming ~252 trading days)
    # Use net returns for Sharpe
    daily_returns = trade_returns_net
    if np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Win/loss stats (when we traded)
    traded_mask = predictions == 1
    traded_returns = actual_returns[traded_mask]

    if len(traded_returns) == 0:
        return total_return, total_return_net, 0.0, 0.0, 0.0, 0.0

    wins = traded_returns[traded_returns > 0]
    losses = traded_returns[traded_returns <= 0]

    win_rate = len(wins) / len(traded_returns) if len(traded_returns) > 0 else 0.0
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

    # Profit factor: gross wins / gross losses
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.001  # Avoid div by zero
    profit_factor = total_wins / total_losses

    return total_return, total_return_net, sharpe, profit_factor, win_rate, avg_win, avg_loss


def evaluate(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    split_name: str,
) -> EvalMetrics:
    """Evaluate model on a dataset."""
    predictions = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Classification metrics
    acc = accuracy_score(y, predictions)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)

    # Trading metrics
    total_ret, total_ret_net, sharpe, pf, win_rate, avg_win, avg_loss = compute_trading_metrics(
        predictions, returns
    )

    return EvalMetrics(
        split_name=split_name,
        n_samples=len(y),
        accuracy=acc,
        precision_1=report["1"]["precision"],
        recall_1=report["1"]["recall"],
        f1_1=report["1"]["f1-score"],
        total_return=total_ret,
        total_return_net=total_ret_net,
        sharpe_ratio=sharpe,
        profit_factor=pf,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


# =============================================================================
# Kill Tests
# =============================================================================


def run_kill_tests(metrics: EvalMetrics) -> Dict[str, bool]:
    """Run kill tests on evaluation metrics."""
    tests = {
        "sharpe_gte_0": metrics.sharpe_ratio >= 0.0,
        "sharpe_gte_0.2": metrics.sharpe_ratio >= 0.2,  # "Tradable" threshold
        "accuracy_gt_50%": metrics.accuracy > 0.50,
        "profit_factor_gt_1": metrics.profit_factor > 1.0,
        "positive_net_return": metrics.total_return_net > 0,
    }
    return tests


def print_kill_test_results(metrics: EvalMetrics) -> bool:
    """Print kill test results and return overall pass/fail."""
    tests = run_kill_tests(metrics)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"KILL TEST RESULTS ({metrics.split_name})")
    logger.info(f"{'=' * 60}")

    all_pass = True
    critical_pass = True

    for test_name, passed in tests.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_pass = False
            # Critical tests: sharpe >= 0, accuracy > 50%
            if test_name in ["sharpe_gte_0", "accuracy_gt_50%"]:
                critical_pass = False

    if critical_pass:
        logger.info("\n🟢 CRITICAL KILL TESTS PASSED - Model viable for further development")
    else:
        logger.info("\n🔴 CRITICAL KILL TESTS FAILED - Model needs work before proceeding")

    return critical_pass


# =============================================================================
# Feature Importance
# =============================================================================


def print_feature_importance(model: LogisticRegression, feature_names: List[str]):
    """Print feature importance from logistic regression coefficients."""
    logger.info(f"\n{'=' * 60}")
    logger.info("Feature Importance (Logistic Regression Coefficients)")
    logger.info(f"{'=' * 60}")

    coefs = model.coef_[0]
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    for name, coef in importance:
        direction = "+" if coef > 0 else "-"
        logger.info(f"  {name:25s}: {direction}{abs(coef):.4f}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Phase 3: Baseline Model for Sleeve IM")
    logger.info("=" * 60)

    # Load and split data
    df = load_dataset()
    train_raw, val_raw, test_raw = split_data(df)

    # Filter to tradable days with valid features
    logger.info("\nFiltering datasets...")
    train = filter_tradable(train_raw)
    val = filter_tradable(val_raw)
    test = filter_tradable(test_raw)

    # Prepare features
    logger.info("\nPreparing features...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_features(train, val, test)

    # Get actual returns for trading metrics
    returns_train = train["label_return"].values
    returns_val = val["label_return"].values
    returns_test = test["label_return"].values

    logger.info(f"\nFeatures: {FEATURE_COLS}")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Test samples: {len(X_test):,}")

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate on all splits
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    train_metrics = evaluate(model, X_train, y_train, returns_train, "TRAIN")
    val_metrics = evaluate(model, X_val, y_val, returns_val, "VALIDATION")
    test_metrics = evaluate(model, X_test, y_test, returns_test, "TEST (Dev Test)")

    print(train_metrics)
    print(val_metrics)
    print(test_metrics)

    # Feature importance
    print_feature_importance(model, FEATURE_COLS)

    # Kill tests on validation set (primary decision point)
    val_pass = print_kill_test_results(val_metrics)

    # Kill tests on test set (final check)
    test_pass = print_kill_test_results(test_metrics)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Validation Sharpe: {val_metrics.sharpe_ratio:.3f}")
    logger.info(f"Test Sharpe:       {test_metrics.sharpe_ratio:.3f}")
    logger.info(f"Validation Pass:   {'✅' if val_pass else '❌'}")
    logger.info(f"Test Pass:         {'✅' if test_pass else '❌'}")

    if val_pass and test_pass:
        logger.info("\n🎉 MODEL PASSES ALL CRITICAL KILL TESTS")
        logger.info("   Ready to proceed to Phase 4 (advanced model development)")
    elif val_pass:
        logger.info("\n⚠️  Model passes validation but fails on test set")
        logger.info("   Consider: overfitting, feature selection, or more data")
    else:
        logger.info("\n❌ MODEL FAILS KILL TESTS")
        logger.info("   Do not proceed to Phase 4 until baseline is fixed")

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    main()
