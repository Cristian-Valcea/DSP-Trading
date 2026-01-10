#!/usr/bin/env python3
"""
Baseline B v1 DEV_TEST single-check runner.

This is the ONLY DEV_TEST run for v1 - no tuning allowed on this split!
Uses the best config from VAL sweep: m_long=1.5, k=50.0, w_cap=0.05
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baseline_b.train import load_model
from baseline_b.backtest import Backtester, COST_BPS, GROSS_EXPOSURE


def main():
    # Find the latest model
    checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints" / "baseline_b"

    if not checkpoints_dir.exists():
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return 1

    model_dirs = sorted(checkpoints_dir.glob("ridge_v0_*"))
    if not model_dirs:
        print("No model checkpoints found.")
        return 1

    model_dir = model_dirs[-1]
    print(f"Loading model from: {model_dir}")

    model, scaler, config = load_model(model_dir)
    print(f"Model loaded: alpha={config['config']['alpha']}")
    print()

    # Best config from VAL sweep
    print("=" * 60)
    print("Baseline B v1 - SINGLE DEV_TEST CHECK")
    print("=" * 60)
    print("Using best config from VAL sweep:")
    print("- sizing_mode: v1 (conviction-sized gross)")
    print("- direction: long_only")
    print("- threshold_mult_long: 1.5")
    print("- weight_scale_k: 50.0")
    print("- weight_cap: 5.0% per name")
    print("- gross_exposure cap: 10.0%")
    print()
    print("⚠️  THIS IS THE ONLY DEV_TEST RUN - NO TUNING ALLOWED!")
    print()

    # Run VAL first for reference
    print("=" * 60)
    print("VAL Results (reference)")
    print("=" * 60)

    backtester_val = Backtester(
        model=model,
        scaler=scaler,
        cost_bps=COST_BPS,
        gross_exposure=GROSS_EXPOSURE,
        direction="long_only",
        threshold_mult_long=1.5,
        sizing_mode="v1",
        weight_scale_k=50.0,
        weight_cap=0.05,
        verbose=False,
    )

    val_result = backtester_val.run("val")

    val_cagr = val_result.annualized_return * 100
    val_max_dd = val_result.max_drawdown * 100
    val_active = val_result.active_days
    val_activity = val_result.activity_pct * 100

    print(f"  CAGR: {val_cagr:.2f}%")
    print(f"  MaxDD: {val_max_dd:.2f}%")
    print(f"  Active days: {val_active} ({val_activity:.1f}%)")
    print()

    # Run DEV_TEST
    print("=" * 60)
    print("DEV_TEST Results (final validation)")
    print("=" * 60)

    backtester_dev = Backtester(
        model=model,
        scaler=scaler,
        cost_bps=COST_BPS,
        gross_exposure=GROSS_EXPOSURE,
        direction="long_only",
        threshold_mult_long=1.5,
        sizing_mode="v1",
        weight_scale_k=50.0,
        weight_cap=0.05,
        verbose=True,
    )

    result = backtester_dev.run("dev_test")

    # Summary
    print()
    print("=" * 60)
    print("KILL/PASS GATES")
    print("=" * 60)

    cagr = result.annualized_return * 100
    sharpe = result.sharpe_ratio
    max_dd = result.max_drawdown * 100
    active_days = result.active_days
    activity_pct = result.activity_pct * 100

    # Gate 1: CAGR > 0
    cagr_pass = cagr > 0
    print(f"  CAGR > 0: {cagr:.2f}% → {'✅ PASS' if cagr_pass else '❌ FAIL'}")

    # Gate 2: Max DD < 15%
    dd_pass = max_dd < 15
    print(f"  MaxDD < 15%: {max_dd:.2f}% → {'✅ PASS' if dd_pass else '❌ FAIL'}")

    # Gate 3: Active days >= 30 (or activity_pct >= 25%)
    activity_pass = active_days >= 30 or activity_pct >= 25
    print(f"  Activity (days≥30 or %≥25): {active_days} days ({activity_pct:.1f}%) → {'✅ PASS' if activity_pass else '❌ FAIL'}")

    print()
    overall = cagr_pass and dd_pass and activity_pass
    print(f"OVERALL: {'✅ PASS' if overall else '❌ KILL'}")

    # Comparison
    print()
    print("=" * 60)
    print("VAL vs DEV_TEST COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'VAL':>12} {'DEV_TEST':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'CAGR':<20} {val_cagr:>11.2f}% {cagr:>11.2f}% {cagr - val_cagr:>+11.2f}%")
    print(f"{'MaxDD':<20} {val_max_dd:>11.2f}% {max_dd:>11.2f}% {max_dd - val_max_dd:>+11.2f}%")
    print(f"{'Active days':<20} {val_active:>12} {active_days:>12} {active_days - val_active:>+12}")
    print(f"{'Activity %':<20} {val_activity:>11.1f}% {activity_pct:>11.1f}% {activity_pct - val_activity:>+11.1f}%")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
