#!/usr/bin/env python3
"""
Walk-Forward Evaluation for Baseline B with Regime Features

Time-boxed final attempt:
1. Add 3 regime features (SPY trend, vol, momentum) → 48-dim total
2. Walk-forward: train on expanding window, test on next month
3. Kill criterion: >50% of folds must have positive CAGR

Walk-forward folds on VAL (2024-01-02 to 2024-06-28):
- Fold 1: Train on Jan data, test on Feb
- Fold 2: Train on Jan-Feb, test on Mar
- Fold 3: Train on Jan-Mar, test on Apr
- Fold 4: Train on Jan-Apr, test on May
- Fold 5: Train on Jan-May, test on Jun
"""

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline_b.data_loader import (
    SYMBOLS,
    get_bar_by_time,
    get_rth_day_data,
    get_trading_days,
    load_rth_data,
)
from baseline_b.feature_builder import FeatureBuilder, get_prior_close, NUM_FEATURES
from baseline_b.regime_features import RegimeFeatureBuilder, NUM_REGIME_FEATURES

# Constants
COST_BPS = 10.0
GROSS_EXPOSURE = 0.10


@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_samples: int
    test_samples: int
    cagr: float
    sharpe: float
    max_dd: float
    active_days: int
    total_days: int
    pass_cagr: bool


def generate_samples_with_regime(
    split: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    feature_builder: FeatureBuilder,
    regime_builder: RegimeFeatureBuilder,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate feature/label samples with regime features for a date range.

    Returns:
        X: Features (N, 48) - 45 base + 3 regime
        y: Labels (N,)
        metadata: DataFrame with date, symbol columns
    """
    # Load all RTH data for symbols
    rth_data = {}
    for symbol in symbols:
        try:
            rth_data[symbol] = load_rth_data(symbol, split)
        except FileNotFoundError:
            pass

    # Get trading days in range
    all_days = get_trading_days(split)
    days = [d for d in all_days if start_date <= d <= end_date]

    if verbose:
        print(f"Generating samples: {len(days)} days, {len(symbols)} symbols")

    X_list = []
    y_list = []
    meta_list = []

    iterator = tqdm(days, desc=f"Processing {split}") if verbose else days

    for d in iterator:
        # Get regime features for this day (same for all symbols)
        regime_features, _ = regime_builder.compute_regime_features(d, split)

        # Get RTH data for this day
        rth_day_data = {}
        for symbol in symbols:
            if symbol in rth_data:
                day_df = get_rth_day_data(rth_data[symbol], d)
                if not day_df.empty:
                    rth_day_data[symbol] = day_df

        for symbol in symbols:
            if symbol not in rth_day_data:
                continue

            day_df = rth_day_data[symbol]

            # Get prior close for overnight gap
            prior_close = None
            if symbol in rth_data:
                prior_close = get_prior_close(rth_data[symbol], d)

            # Build base features (45-dim)
            base_features, diag = feature_builder.build_features(
                rth_day_data, symbol, d, prior_close
            )
            if base_features is None:
                continue

            # Concatenate with regime features → 48-dim
            full_features = np.concatenate([base_features, regime_features])

            # Get label: log(P_1400_close / P_1031_open)
            from datetime import time
            entry_bar = get_bar_by_time(day_df, time(10, 31))
            exit_bar = get_bar_by_time(day_df, time(14, 0))

            if entry_bar is None or exit_bar is None:
                continue

            entry_price = entry_bar["open"]
            exit_price = exit_bar["close"]

            if entry_price <= 0 or exit_price <= 0:
                continue

            label = np.log(exit_price / entry_price)

            X_list.append(full_features)
            y_list.append(label)
            meta_list.append({"date": d, "symbol": symbol})

    if not X_list:
        return np.array([]), np.array([]), pd.DataFrame()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    metadata = pd.DataFrame(meta_list)

    return X, y, metadata


def run_fold(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    test_meta: pd.DataFrame,
    cost_decimal: float,
    gross_exposure: float,
    threshold_mult: float,
    weight_scale_k: float,
    weight_cap: float,
) -> dict:
    """
    Train model on train data, evaluate on test data.

    Returns dict with metrics.
    """
    # Train Ridge model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, train_y)

    # Predict on test
    X_test_scaled = scaler.transform(test_X)
    y_pred = model.predict(X_test_scaled)

    # Compute daily P&L (simplified v1 sizing)
    base_threshold = 2 * cost_decimal
    t_long = threshold_mult * base_threshold

    # Group by date
    test_meta = test_meta.copy()
    test_meta["y_pred"] = y_pred
    test_meta["y_true"] = test_y

    daily_returns = []
    daily_costs = []
    active_count = 0

    for d, group in test_meta.groupby("date"):
        preds = group["y_pred"].values
        actuals = group["y_true"].values

        # Compute scores (long-only)
        scores = np.maximum(preds - t_long, 0)

        if np.sum(scores) == 0:
            daily_returns.append(0.0)
            daily_costs.append(0.0)
            continue

        active_count += 1

        # v1 weights
        weights_raw = weight_scale_k * scores
        weights = np.clip(weights_raw, 0, weight_cap)

        total_gross = np.sum(weights)
        if total_gross > gross_exposure:
            weights = weights * (gross_exposure / total_gross)

        # P&L
        gross_return = np.sum(weights * actuals)
        cost = np.sum(weights) * 2 * cost_decimal
        net_return = gross_return - cost

        daily_returns.append(net_return)
        daily_costs.append(cost)

    if not daily_returns:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "active_days": 0,
            "total_days": 0,
        }

    daily_returns = np.array(daily_returns)
    total_days = len(daily_returns)

    # Metrics
    total_return = np.sum(daily_returns)
    cagr = np.mean(daily_returns) * 252

    daily_std = np.std(daily_returns)
    sharpe = (np.mean(daily_returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    # Max drawdown
    cumulative = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "active_days": active_count,
        "total_days": total_days,
    }


def run_walk_forward(
    split: str = "val",
    threshold_mult: float = 1.5,
    weight_scale_k: float = 50.0,
    weight_cap: float = 0.05,
    verbose: bool = True,
) -> list[FoldResult]:
    """
    Run walk-forward evaluation with expanding training window.

    VAL period: 2024-01-02 to 2024-06-28
    Folds:
    - Fold 1: Train Jan, Test Feb
    - Fold 2: Train Jan-Feb, Test Mar
    - Fold 3: Train Jan-Mar, Test Apr
    - Fold 4: Train Jan-Apr, Test May
    - Fold 5: Train Jan-May, Test Jun
    """
    # Define fold boundaries
    fold_boundaries = [
        # (train_start, train_end, test_start, test_end)
        (date(2024, 1, 2), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 1, 2), date(2024, 2, 29), date(2024, 3, 1), date(2024, 3, 29)),
        (date(2024, 1, 2), date(2024, 3, 29), date(2024, 4, 1), date(2024, 4, 30)),
        (date(2024, 1, 2), date(2024, 4, 30), date(2024, 5, 1), date(2024, 5, 31)),
        (date(2024, 1, 2), date(2024, 5, 31), date(2024, 6, 3), date(2024, 6, 28)),
    ]

    feature_builder = FeatureBuilder(symbols=SYMBOLS)
    regime_builder = RegimeFeatureBuilder()

    cost_decimal = COST_BPS / 10000

    results = []

    if verbose:
        print("=" * 80)
        print("Walk-Forward Evaluation with Regime Features")
        print("=" * 80)
        print(f"Features: 45 base + 3 regime = 48 total")
        print(f"Config: m_long={threshold_mult}, k={weight_scale_k}, w_cap={weight_cap}")
        print()

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries, 1):
        if verbose:
            print(f"Fold {fold_id}: Train [{train_start} → {train_end}], Test [{test_start} → {test_end}]")

        # Generate training data
        train_X, train_y, train_meta = generate_samples_with_regime(
            split, SYMBOLS, train_start, train_end,
            feature_builder, regime_builder, verbose=False
        )

        # Generate test data
        test_X, test_y, test_meta = generate_samples_with_regime(
            split, SYMBOLS, test_start, test_end,
            feature_builder, regime_builder, verbose=False
        )

        if len(train_X) == 0 or len(test_X) == 0:
            if verbose:
                print(f"  SKIP: Insufficient data (train={len(train_X)}, test={len(test_X)})")
            continue

        # Run fold
        metrics = run_fold(
            train_X, train_y, test_X, test_y, test_meta,
            cost_decimal, GROSS_EXPOSURE,
            threshold_mult, weight_scale_k, weight_cap
        )

        cagr_pct = metrics["cagr"] * 100
        pass_cagr = cagr_pct > 0

        result = FoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_samples=len(train_X),
            test_samples=len(test_X),
            cagr=cagr_pct,
            sharpe=metrics["sharpe"],
            max_dd=metrics["max_dd"] * 100,
            active_days=metrics["active_days"],
            total_days=metrics["total_days"],
            pass_cagr=pass_cagr,
        )
        results.append(result)

        if verbose:
            status = "✅" if pass_cagr else "❌"
            print(f"  Train: {len(train_X)} samples, Test: {len(test_X)} samples")
            print(f"  CAGR: {cagr_pct:+.2f}% | Sharpe: {metrics['sharpe']:.2f} | MaxDD: {metrics['max_dd']*100:.2f}% | Active: {metrics['active_days']}/{metrics['total_days']} | {status}")
            print()

    return results


def main():
    """Run walk-forward evaluation and report results."""
    print("=" * 80)
    print("BASELINE B - FINAL ATTEMPT: Regime Features + Walk-Forward")
    print("=" * 80)
    print()
    print("Kill criterion: >50% of folds must have positive CAGR")
    print()

    # Run with best v1 parameters
    results = run_walk_forward(
        split="val",
        threshold_mult=1.5,
        weight_scale_k=50.0,
        weight_cap=0.05,
        verbose=True,
    )

    # Summary
    print("=" * 80)
    print("WALK-FORWARD SUMMARY")
    print("=" * 80)

    if not results:
        print("❌ NO FOLDS COMPLETED")
        return 1

    n_folds = len(results)
    n_pass = sum(1 for r in results if r.pass_cagr)
    pass_rate = n_pass / n_folds

    print(f"{'Fold':>6} {'Test Period':>24} {'CAGR%':>10} {'Sharpe':>8} {'Active':>8} {'Pass':>6}")
    print("-" * 80)

    for r in results:
        status = "✅" if r.pass_cagr else "❌"
        print(f"{r.fold_id:>6} {str(r.test_start) + ' → ' + str(r.test_end):>24} "
              f"{r.cagr:>+9.2f}% {r.sharpe:>8.2f} {r.active_days:>4}/{r.total_days:<3} {status:>6}")

    print("-" * 80)
    print(f"Pass rate: {n_pass}/{n_folds} = {pass_rate*100:.0f}%")
    print()

    # Final verdict
    if pass_rate > 0.5:
        print("✅ VERDICT: >50% folds positive - PROCEED TO DEV_TEST")
        return 0
    else:
        print("❌ VERDICT: ≤50% folds positive - KILL BASELINE B PERMANENTLY")
        return 1


if __name__ == "__main__":
    sys.exit(main())
