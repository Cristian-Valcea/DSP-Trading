#!/usr/bin/env python3
"""
Baseline B v1 VAL-only parameter sweep.

Sweeps over {m_long, k, w_cap} to find parameters that:
1. Pass activity gate (active_days >= 30 or activity_pct >= 25%)
2. Maximize net CAGR (must be > 0)
3. Keep MaxDD < 15%
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

    # Parameter grid (VAL-only, conservative first)
    # The key insight: k controls how much we scale the score to get weight
    # With scores typically in range [0, 0.01] (1% expected return edge),
    # we need k > 1 to get meaningful weights

    param_grid = [
        # Start with high k values since scores are small
        {"m_long": 2.0, "k": 10.0, "w_cap": 0.03},
        {"m_long": 2.0, "k": 20.0, "w_cap": 0.03},
        {"m_long": 2.0, "k": 50.0, "w_cap": 0.03},
        {"m_long": 2.0, "k": 100.0, "w_cap": 0.03},
        # Try lower threshold (more trades)
        {"m_long": 1.5, "k": 20.0, "w_cap": 0.03},
        {"m_long": 1.5, "k": 50.0, "w_cap": 0.03},
        {"m_long": 1.0, "k": 50.0, "w_cap": 0.03},
        # Higher weight cap
        {"m_long": 2.0, "k": 50.0, "w_cap": 0.05},
        {"m_long": 1.5, "k": 50.0, "w_cap": 0.05},
    ]

    print("=" * 80)
    print("Baseline B v1 VAL-Only Parameter Sweep")
    print("=" * 80)
    print(f"{'m_long':>8} {'k':>8} {'w_cap':>8} | {'CAGR%':>8} {'MaxDD%':>8} {'Active':>8} {'Act%':>8} {'Gross%':>8} | {'Pass':>6}")
    print("-" * 80)

    results = []

    for params in param_grid:
        backtester = Backtester(
            model=model,
            scaler=scaler,
            cost_bps=COST_BPS,
            gross_exposure=GROSS_EXPOSURE,
            direction="long_only",
            threshold_mult_long=params["m_long"],
            sizing_mode="v1",
            weight_scale_k=params["k"],
            weight_cap=params["w_cap"],
            verbose=False,
        )

        result = backtester.run("val")

        cagr = result.annualized_return * 100
        max_dd = result.max_drawdown * 100
        active_days = result.active_days
        activity_pct = result.activity_pct * 100
        avg_gross = result.avg_gross_exposure * 100

        # Gates
        cagr_pass = cagr > 0
        dd_pass = max_dd < 15
        activity_pass = active_days >= 30 or activity_pct >= 25

        overall = cagr_pass and dd_pass and activity_pass
        status = "✅" if overall else "❌"

        print(f"{params['m_long']:>8.1f} {params['k']:>8.1f} {params['w_cap']:>8.2f} | "
              f"{cagr:>8.2f} {max_dd:>8.2f} {active_days:>8} {activity_pct:>7.1f}% {avg_gross:>7.2f}% | {status:>6}")

        results.append({
            "params": params,
            "cagr": cagr,
            "max_dd": max_dd,
            "active_days": active_days,
            "activity_pct": activity_pct,
            "avg_gross": avg_gross,
            "overall": overall,
        })

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find best passing configuration
    passing = [r for r in results if r["overall"]]
    if passing:
        best = max(passing, key=lambda x: x["cagr"])
        print(f"✅ BEST PASSING CONFIG:")
        print(f"   m_long={best['params']['m_long']}, k={best['params']['k']}, w_cap={best['params']['w_cap']}")
        print(f"   CAGR={best['cagr']:.2f}%, MaxDD={best['max_dd']:.2f}%, Active={best['active_days']}, Activity={best['activity_pct']:.1f}%")
    else:
        # Find closest to passing
        print("❌ NO CONFIGS PASS ALL GATES")
        print()
        # Sort by CAGR among configs that pass activity
        activity_pass = [r for r in results if r["active_days"] >= 30 or r["activity_pct"] >= 25]
        if activity_pass:
            best_active = max(activity_pass, key=lambda x: x["cagr"])
            print(f"Best config passing ACTIVITY gate:")
            print(f"   m_long={best_active['params']['m_long']}, k={best_active['params']['k']}, w_cap={best_active['params']['w_cap']}")
            print(f"   CAGR={best_active['cagr']:.2f}%, MaxDD={best_active['max_dd']:.2f}%, Active={best_active['active_days']}")
        else:
            print("No configs pass activity gate - need higher k or lower m_long")
            # Find config with highest activity
            most_active = max(results, key=lambda x: x["active_days"])
            print(f"Most active config:")
            print(f"   m_long={most_active['params']['m_long']}, k={most_active['params']['k']}, w_cap={most_active['params']['w_cap']}")
            print(f"   Active days={most_active['active_days']}, Activity={most_active['activity_pct']:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
