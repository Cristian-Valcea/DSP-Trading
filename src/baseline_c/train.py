"""
Train Module for Baseline C

4 Ridge regression models (one per interval) with standardization.

Per spec §6:
- 4 pooled Ridge regressors (one per interval)
- Standardize using train split only (mean/std)
- Fixed alpha for v0 (can be tuned later)

Intervals:
- Ridge(10:31→11:31)
- Ridge(11:31→12:31)
- Ridge(12:31→14:00)
- Ridge(14:00→next 10:31)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .dataset import DatasetGenerator, DatasetStats, INTERVAL_DEFS
from .feature_builder import FEATURE_NAMES, NUM_FEATURES


@dataclass
class TrainConfig:
    """Training configuration."""
    alpha: float = 1.0  # Ridge regularization (v0 default)
    fit_intercept: bool = True
    symbols: list[str] = None  # None = all 9 symbols


@dataclass
class IntervalTrainResult:
    """Training result for a single interval."""
    interval_idx: int
    interval_name: str
    model: Ridge
    scaler: StandardScaler
    train_stats: DatasetStats
    val_stats: Optional[DatasetStats]

    # Training metrics
    train_r2: float
    train_rmse: float
    train_samples: int
    val_r2: Optional[float] = None
    val_rmse: Optional[float] = None
    val_samples: int = 0

    # Feature coefficients
    feature_coefficients: dict = None

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "interval_idx": self.interval_idx,
            "interval_name": self.interval_name,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "metrics": {
                "train_r2": self.train_r2,
                "train_rmse": self.train_rmse,
                "val_r2": self.val_r2,
                "val_rmse": self.val_rmse,
            },
            "train_stats": self.train_stats.to_dict(),
            "val_stats": self.val_stats.to_dict() if self.val_stats else None,
            "feature_coefficients": self.feature_coefficients,
        }


@dataclass
class TrainResult:
    """Training result container for all 4 intervals."""
    config: TrainConfig
    interval_results: list[IntervalTrainResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "config": {
                "alpha": self.config.alpha,
                "fit_intercept": self.config.fit_intercept,
                "symbols": self.config.symbols,
            },
            "intervals": [r.to_dict() for r in self.interval_results],
            "summary": {
                "total_train_samples": sum(r.train_samples for r in self.interval_results),
                "total_val_samples": sum(r.val_samples for r in self.interval_results),
                "avg_train_r2": np.mean([r.train_r2 for r in self.interval_results]),
                "avg_val_r2": np.mean([r.val_r2 for r in self.interval_results if r.val_r2 is not None]) if any(r.val_r2 is not None for r in self.interval_results) else None,
            },
        }


class RidgeTrainer:
    """
    Ridge regression trainer for Baseline C.

    Trains 4 separate Ridge models (one per interval):
    - Each interval has its own scaler (fit on train)
    - Each interval has its own model

    Note: While we could share a scaler across intervals (features are the same),
    keeping them separate provides flexibility for future modifications.
    """

    def __init__(self, config: TrainConfig = None, verbose: bool = True):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            verbose: Show progress
        """
        self.config = config or TrainConfig()
        self.verbose = verbose
        self.models: list[Ridge] = [None, None, None, None]
        self.scalers: list[StandardScaler] = [None, None, None, None]

    def train(
        self,
        include_val: bool = True,
    ) -> TrainResult:
        """
        Train 4 Ridge regression models (one per interval).

        Args:
            include_val: Also generate and evaluate on val split

        Returns:
            TrainResult with all 4 interval results
        """
        if self.verbose:
            print("=" * 60)
            print("Training Baseline C Ridge Regression (4 Intervals)")
            print("=" * 60)
            print(f"Config: alpha={self.config.alpha}, fit_intercept={self.config.fit_intercept}")
            print()

        generator = DatasetGenerator(
            symbols=self.config.symbols,
            verbose=self.verbose,
        )

        result = TrainResult(config=self.config)

        # Train each interval's model
        for interval_idx in range(4):
            interval_name = INTERVAL_DEFS[interval_idx][0]

            if self.verbose:
                print("=" * 60)
                print(f"Training Interval {interval_idx}: {interval_name}")
                print("=" * 60)

            # Generate training data for this interval
            X_train, y_train, meta_train, train_stats = generator.generate("train", interval_idx)

            if len(X_train) == 0:
                if self.verbose:
                    print(f"WARNING: No training samples for interval {interval_name}!")
                continue

            if self.verbose:
                print(f"Train samples: {len(X_train)}")

            # Fit scaler on train data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[interval_idx] = scaler

            # Fit Ridge regression
            model = Ridge(
                alpha=self.config.alpha,
                fit_intercept=self.config.fit_intercept,
            )
            model.fit(X_train_scaled, y_train)
            self.models[interval_idx] = model

            # Train metrics
            y_train_pred = model.predict(X_train_scaled)
            train_r2 = self._compute_r2(y_train, y_train_pred)
            train_rmse = self._compute_rmse(y_train, y_train_pred)

            if self.verbose:
                print(f"Train R²: {train_r2:.6f}")
                print(f"Train RMSE: {train_rmse:.6f}")

            # Feature coefficients
            feature_coefficients = {
                name: float(coef)
                for name, coef in zip(FEATURE_NAMES, model.coef_)
            }

            # Validation metrics
            val_r2, val_rmse, val_stats, val_samples = None, None, None, 0
            if include_val:
                X_val, y_val, meta_val, val_stats = generator.generate("val", interval_idx)

                if len(X_val) > 0:
                    val_samples = len(X_val)
                    X_val_scaled = scaler.transform(X_val)

                    y_val_pred = model.predict(X_val_scaled)
                    val_r2 = self._compute_r2(y_val, y_val_pred)
                    val_rmse = self._compute_rmse(y_val, y_val_pred)

                    if self.verbose:
                        print(f"Val samples: {val_samples}")
                        print(f"Val R²: {val_r2:.6f}")
                        print(f"Val RMSE: {val_rmse:.6f}")

            # Top 5 features for this interval
            if self.verbose:
                print(f"\nTop 5 features by |coefficient| for {interval_name}:")
                sorted_features = sorted(
                    feature_coefficients.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                for i, (name, coef) in enumerate(sorted_features[:5], 1):
                    print(f"  {i}. {name}: {coef:+.6f}")
                print()

            # Store interval result
            interval_result = IntervalTrainResult(
                interval_idx=interval_idx,
                interval_name=interval_name,
                model=model,
                scaler=scaler,
                train_stats=train_stats,
                val_stats=val_stats,
                train_r2=train_r2,
                train_rmse=train_rmse,
                train_samples=len(X_train),
                val_r2=val_r2,
                val_rmse=val_rmse,
                val_samples=val_samples,
                feature_coefficients=feature_coefficients,
            )
            result.interval_results.append(interval_result)

        # Summary
        if self.verbose:
            print("=" * 60)
            print("Training Summary (All Intervals)")
            print("=" * 60)
            print(f"{'Interval':<20} {'Train R²':>10} {'Val R²':>10} {'Train N':>10} {'Val N':>10}")
            print("-" * 60)
            for ir in result.interval_results:
                val_r2_str = f"{ir.val_r2:.6f}" if ir.val_r2 is not None else "N/A"
                print(f"{ir.interval_name:<20} {ir.train_r2:>10.6f} {val_r2_str:>10} {ir.train_samples:>10} {ir.val_samples:>10}")
            print()

        return result

    def predict(self, X: np.ndarray, interval_idx: int) -> np.ndarray:
        """
        Make predictions for a specific interval.

        Args:
            X: Feature array (N, 45)
            interval_idx: 0-3 for the four intervals

        Returns:
            Predictions (N,)
        """
        if self.models[interval_idx] is None or self.scalers[interval_idx] is None:
            raise RuntimeError(f"Model for interval {interval_idx} not trained!")

        X_scaled = self.scalers[interval_idx].transform(X)
        return self.models[interval_idx].predict(X_scaled)

    @staticmethod
    def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


def save_model(
    result: TrainResult,
    output_dir: Path,
    run_id: str = None,
) -> Path:
    """
    Save trained models and artifacts.

    Args:
        result: Training result
        output_dir: Base output directory
        run_id: Optional run identifier (defaults to timestamp)

    Returns:
        Path to saved model directory
    """
    if run_id is None:
        run_id = f"ridge_c_v0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_dir = Path(output_dir) / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save each interval's model and scaler
    for ir in result.interval_results:
        interval_subdir = model_dir / f"interval_{ir.interval_idx}"
        interval_subdir.mkdir(exist_ok=True)

        joblib.dump(ir.model, interval_subdir / "ridge_model.joblib")
        joblib.dump(ir.scaler, interval_subdir / "scaler.joblib")

    # Save config and metrics
    with open(model_dir / "train_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    # Save feature names
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    # Save interval definitions
    with open(model_dir / "interval_defs.json", "w") as f:
        json.dump(
            [{"idx": i, "name": d[0], "start": str(d[1]), "end": str(d[2]), "is_overnight": d[3]}
             for i, d in enumerate(INTERVAL_DEFS)],
            f, indent=2
        )

    print(f"Model saved to: {model_dir}")
    return model_dir


def load_model(model_dir: Path) -> tuple[list[Ridge], list[StandardScaler], dict]:
    """
    Load trained models and artifacts.

    Args:
        model_dir: Path to saved model directory

    Returns:
        Tuple of (models list, scalers list, config_dict)
    """
    model_dir = Path(model_dir)

    models = []
    scalers = []

    for interval_idx in range(4):
        interval_subdir = model_dir / f"interval_{interval_idx}"
        if interval_subdir.exists():
            model = joblib.load(interval_subdir / "ridge_model.joblib")
            scaler = joblib.load(interval_subdir / "scaler.joblib")
        else:
            model = None
            scaler = None
        models.append(model)
        scalers.append(scaler)

    with open(model_dir / "train_result.json", "r") as f:
        config_dict = json.load(f)

    return models, scalers, config_dict


if __name__ == "__main__":
    print("=== Ridge Trainer Validation (Baseline C) ===")

    # Train with default config
    trainer = RidgeTrainer(
        config=TrainConfig(alpha=1.0),
        verbose=True,
    )

    result = trainer.train(include_val=True)

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Summary table
    for ir in result.interval_results:
        print(f"{ir.interval_name}: Train R²={ir.train_r2:.6f}, Val R²={ir.val_r2:.6f if ir.val_r2 else 'N/A'}")

    # Save model
    output_dir = Path(__file__).parent.parent.parent / "checkpoints" / "baseline_c"
    model_dir = save_model(result, output_dir)

    # Test loading
    print("\nTesting model loading...")
    loaded_models, loaded_scalers, loaded_config = load_model(model_dir)
    print(f"Loaded {len([m for m in loaded_models if m is not None])} models")
    print(f"Loaded {len([s for s in loaded_scalers if s is not None])} scalers")
    print(f"Config keys: {list(loaded_config.keys())}")

    print("\n=== Validation Complete ===")
