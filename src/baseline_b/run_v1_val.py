#!/usr/bin/env python3
"""
Baseline B v1 VAL-only test runner.

Tests conviction-sized gross sizing with default parameters:
- sizing_mode: v1
- direction: long_only
- threshold_mult_long: 2.0 (require 2x cost to trade)
- weight_scale_k: 1.0
- weight_cap: 0.03 (3% per name)
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
        print("Please run train.py first to create a model.")
        return 1

    model_dirs = sorted(checkpoints_dir.glob("ridge_v0_*"))
    if not model_dirs:
        print("No model checkpoints found. Please run train.py first.")
        return 1

    model_dir = model_dirs[-1]  # Most recent
    print(f"Loading model from: {model_dir}")

    # Load model
    model, scaler, config = load_model(model_dir)
    print(f"Model loaded: alpha={config['config']['alpha']}")
    print()

    # v1 configuration
    print("=" * 60)
    print("Baseline B v1 Configuration")
    print("=" * 60)
    print("- sizing_mode: v1 (conviction-sized gross)")
    print("- direction: long_only")
    print("- threshold_mult_long: 2.0 (require ŷ > 2×round-trip cost)")
    print("- weight_scale_k: 1.0")
    print("- weight_cap: 3.0% per name")
    print("- gross_exposure cap: 10.0%")
    print()

    # Run backtest with v1 settings
    backtester = Backtester(
        model=model,
        scaler=scaler,
        cost_bps=COST_BPS,
        gross_exposure=GROSS_EXPOSURE,
        direction="long_only",
        threshold_mult_long=2.0,  # Require 2x cost threshold
        sizing_mode="v1",
        weight_scale_k=1.0,
        weight_cap=0.03,  # 3% per name
        verbose=True,
    )

    result = backtester.run("val")

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

    # Cost analysis
    print()
    print("=" * 60)
    print("COST ANALYSIS")
    print("=" * 60)
    total_gross = sum(abs(d.gross_return) for d in result.daily_results)
    total_cost = result.total_cost
    if total_gross > 0:
        cost_drag_pct = (total_cost / total_gross) * 100
        print(f"  Total gross P&L: {total_gross * 100:.4f}%")
        print(f"  Total cost: {total_cost * 100:.4f}%")
        print(f"  Cost as % of gross: {cost_drag_pct:.1f}%")
    else:
        print("  No gross P&L to analyze")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
