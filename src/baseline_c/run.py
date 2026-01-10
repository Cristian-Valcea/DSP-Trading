#!/usr/bin/env python3
"""
CLI Entry Point for Baseline C

Multi-Time Supervised Learning with 4 rebalance intervals.

Usage:
    # Train model (4 Ridge models, one per interval)
    python -m baseline_c.run train

    # Train with custom alpha
    python -m baseline_c.run train --alpha 0.5

    # Evaluate on val and dev_test
    python -m baseline_c.run eval --model checkpoints/baseline_c/ridge_c_v0_xxx

    # Run holdout (only after val+dev_test pass)
    python -m baseline_c.run holdout --model checkpoints/baseline_c/ridge_c_v0_xxx

    # Full pipeline: train + eval
    python -m baseline_c.run pipeline

    # Dataset statistics (for all 4 intervals)
    python -m baseline_c.run stats
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_c.backtest import Backtester
from baseline_c.data_loader import SYMBOLS
from baseline_c.dataset import DatasetGenerator, INTERVAL_DEFS
from baseline_c.evaluate import Evaluator, run_full_evaluation, save_evaluation
from baseline_c.train import RidgeTrainer, TrainConfig, load_model, save_model


def cmd_train(args):
    """Train 4 Ridge regression models (one per interval)."""
    print("=" * 60)
    print("BASELINE C - TRAINING (4 INTERVAL MODELS)")
    print("=" * 60)
    print(f"Alpha: {args.alpha}")
    print(f"Symbols: {args.symbols or 'all 9'}")
    print()
    print("Intervals:")
    for i, (name, start, end, is_overnight) in enumerate(INTERVAL_DEFS):
        print(f"  {i}: {name} {'(overnight)' if is_overnight else ''}")
    print()

    # Parse symbols
    symbols = args.symbols.split(",") if args.symbols else None

    # Train
    config = TrainConfig(
        alpha=args.alpha,
        fit_intercept=True,
        symbols=symbols,
    )

    trainer = RidgeTrainer(config=config, verbose=True)
    result = trainer.train(include_val=not args.no_val)

    # Save model
    output_dir = Path(args.output_dir)
    run_id = args.run_id or f"ridge_c_v0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = save_model(result, output_dir, run_id)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {model_dir}")
    print()
    print("Per-Interval R² Summary:")
    for ir in result.interval_results:
        val_str = f"{ir.val_r2:.6f}" if ir.val_r2 is not None else "N/A"
        print(f"  {ir.interval_name}: Train R²={ir.train_r2:.6f}, Val R²={val_str}")

    return model_dir


def cmd_eval(args):
    """Evaluate model on val and dev_test."""
    model_dir = Path(args.model)

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        sys.exit(1)

    print("=" * 60)
    print("BASELINE C - EVALUATION")
    print("=" * 60)
    print(f"Model: {model_dir}")
    print()

    # Determine splits
    splits = args.splits.split(",") if args.splits else ["val", "dev_test"]

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "results"

    # Run evaluation
    results = run_full_evaluation(
        model_dir=model_dir,
        output_dir=output_dir,
        splits=splits,
        verbose=True,
    )

    # Check if all gates pass
    all_pass = all(r.all_gates_passed for r in results.values())

    print()
    print("=" * 60)
    if all_pass:
        print("✅ ALL KILL GATES PASSED - Strategy viable for holdout")
    else:
        print("❌ STRATEGY KILLED - One or more gates failed")
    print("=" * 60)

    return results


def cmd_holdout(args):
    """Run holdout evaluation (only after val+dev_test pass)."""
    model_dir = Path(args.model)

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        sys.exit(1)

    print("=" * 60)
    print("BASELINE C - HOLDOUT EVALUATION")
    print("=" * 60)
    print()
    print("⚠️  WARNING: Holdout should only be run after val+dev_test pass!")
    print()

    if not args.force:
        response = input("Continue with holdout evaluation? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "results"

    # Run holdout
    results = run_full_evaluation(
        model_dir=model_dir,
        output_dir=output_dir,
        splits=["holdout"],
        verbose=True,
    )

    return results


def cmd_pipeline(args):
    """Full pipeline: train + evaluate."""
    print("=" * 60)
    print("BASELINE C - FULL PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Train
    print("Step 1/2: Training 4 interval models...")
    print("-" * 40)

    # Parse symbols
    symbols = args.symbols.split(",") if args.symbols else None

    config = TrainConfig(
        alpha=args.alpha,
        fit_intercept=True,
        symbols=symbols,
    )

    trainer = RidgeTrainer(config=config, verbose=True)
    result = trainer.train(include_val=True)

    # Save model
    output_dir = Path(args.output_dir)
    run_id = args.run_id or f"ridge_c_v0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_dir = save_model(result, output_dir, run_id)

    print()
    print("Step 2/2: Evaluating...")
    print("-" * 40)

    # Step 2: Evaluate
    results_dir = model_dir / "results"
    eval_results = run_full_evaluation(
        model_dir=model_dir,
        output_dir=results_dir,
        splits=["val", "dev_test"],
        verbose=True,
    )

    # Final summary
    all_pass = all(r.all_gates_passed for r in eval_results.values())

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model: {model_dir}")
    print()

    if all_pass:
        print("✅ ALL KILL GATES PASSED")
        print()
        print("Next steps:")
        print(f"  1. Review results in {results_dir}")
        print(f"  2. Run holdout: python -m baseline_c.run holdout --model {model_dir}")
    else:
        print("❌ STRATEGY KILLED")
        print()
        print("The strategy did not pass kill gates.")
        print("Recommendation: Do not proceed to holdout.")

    return model_dir, eval_results


def cmd_stats(args):
    """Show dataset statistics for all 4 intervals."""
    print("=" * 60)
    print("BASELINE C - DATASET STATISTICS (4 INTERVALS)")
    print("=" * 60)
    print()

    splits = args.splits.split(",") if args.splits else ["train", "val", "dev_test"]
    symbols = args.symbols.split(",") if args.symbols else None

    generator = DatasetGenerator(symbols=symbols, verbose=False)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split.upper()}")
        print("=" * 60)

        total_samples = 0
        for interval_idx in range(4):
            interval_name = INTERVAL_DEFS[interval_idx][0]
            X, y, metadata, stats = generator.generate(split, interval_idx)

            total_samples += stats.total_samples

            print(f"\n--- Interval {interval_idx}: {interval_name} ---")
            print(f"Samples: {stats.total_samples}")
            print(f"Trading days: {stats.total_trading_days}")
            print(f"Skipped (insufficient_data): {stats.skip_stats_total.insufficient_rth_data}")
            print(f"Skipped (missing_next_open): {stats.skip_stats_total.missing_next_open}")

            if stats.total_samples > 0:
                print(f"Label mean: {y.mean():.6f}")
                print(f"Label std: {y.std():.6f}")
                print(f"Label range: [{y.min():.6f}, {y.max():.6f}]")

        print(f"\n--- {split.upper()} TOTAL ---")
        print(f"Total samples across all 4 intervals: {total_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline C: Multi-Time Supervised Learning (4 Intervals)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train 4 Ridge regression models")
    train_parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0)")
    train_parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: all 9)")
    train_parser.add_argument("--output-dir", type=str, default="dsp100k/checkpoints/baseline_c",
                              help="Output directory for model")
    train_parser.add_argument("--run-id", type=str, help="Run identifier (default: timestamp)")
    train_parser.add_argument("--no-val", action="store_true", help="Skip validation split")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model on val/dev_test")
    eval_parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    eval_parser.add_argument("--splits", type=str, default="val,dev_test",
                             help="Comma-separated splits (default: val,dev_test)")
    eval_parser.add_argument("--output-dir", type=str, help="Output directory for results")

    # Holdout command
    holdout_parser = subparsers.add_parser("holdout", help="Run holdout evaluation")
    holdout_parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    holdout_parser.add_argument("--output-dir", type=str, help="Output directory for results")
    holdout_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Full pipeline: train + eval")
    pipeline_parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0)")
    pipeline_parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: all 9)")
    pipeline_parser.add_argument("--output-dir", type=str, default="dsp100k/checkpoints/baseline_c",
                                 help="Output directory for model")
    pipeline_parser.add_argument("--run-id", type=str, help="Run identifier (default: timestamp)")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics for all intervals")
    stats_parser.add_argument("--splits", type=str, default="train,val,dev_test",
                              help="Comma-separated splits")
    stats_parser.add_argument("--symbols", type=str, help="Comma-separated symbols")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "holdout":
        cmd_holdout(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
