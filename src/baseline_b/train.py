"""
Train Module for Baseline B

Ridge regression training with standardization.

Per spec §7:
- Ridge regression on pooled data
- Standardize using train split only (mean/std)
- Fixed alpha=1.0 for v0
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .dataset import DatasetGenerator, DatasetStats
from .feature_builder import FEATURE_NAMES, NUM_FEATURES


@dataclass
class TrainConfig:
    """Training configuration."""
    alpha: float = 1.0  # Ridge regularization (v0 default)
    fit_intercept: bool = True
    symbols: list[str] = None  # None = all 9 symbols


@dataclass
class TrainResult:
    """Training result container."""
    model: Ridge
    scaler: StandardScaler
    train_stats: DatasetStats
    val_stats: Optional[DatasetStats]
    config: TrainConfig

    # Training metrics
    train_r2: float
    train_rmse: float
    val_r2: Optional[float] = None
    val_rmse: Optional[float] = None

    # Feature coefficients
    feature_coefficients: dict = None

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "config": {
                "alpha": self.config.alpha,
                "fit_intercept": self.config.fit_intercept,
                "symbols": self.config.symbols,
            },
            "train_stats": self.train_stats.to_dict(),
            "val_stats": self.val_stats.to_dict() if self.val_stats else None,
            "metrics": {
                "train_r2": self.train_r2,
                "train_rmse": self.train_rmse,
                "val_r2": self.val_r2,
                "val_rmse": self.val_rmse,
            },
            "feature_coefficients": self.feature_coefficients,
        }


class RidgeTrainer:
    """
    Ridge regression trainer for Baseline B.

    Handles:
    - Dataset generation for train/val splits
    - Feature standardization (train-only mean/std)
    - Ridge regression fitting
    - Model evaluation and metrics
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
        self.model: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None

    def train(
        self,
        include_val: bool = True,
    ) -> TrainResult:
        """
        Train Ridge regression model.

        Args:
            include_val: Also generate and evaluate on val split

        Returns:
            TrainResult with model, scaler, and metrics
        """
        if self.verbose:
            print("=" * 60)
            print("Training Baseline B Ridge Regression")
            print("=" * 60)
            print(f"Config: alpha={self.config.alpha}, fit_intercept={self.config.fit_intercept}")
            print()

        # Generate training data
        generator = DatasetGenerator(
            symbols=self.config.symbols,
            verbose=self.verbose,
        )

        if self.verbose:
            print("Generating train dataset...")
        X_train, y_train, meta_train, train_stats = generator.generate("train")

        if len(X_train) == 0:
            raise ValueError("No training samples generated!")

        if self.verbose:
            print(f"Train samples: {len(X_train)}")
            print()

        # Fit scaler on train data only (per spec §5.4)
        if self.verbose:
            print("Fitting StandardScaler on train data...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.verbose:
            print(f"Feature means (train): min={self.scaler.mean_.min():.4f}, max={self.scaler.mean_.max():.4f}")
            print(f"Feature stds (train): min={self.scaler.scale_.min():.4f}, max={self.scaler.scale_.max():.4f}")
            print()

        # Fit Ridge regression
        if self.verbose:
            print(f"Fitting Ridge regression (alpha={self.config.alpha})...")
        self.model = Ridge(
            alpha=self.config.alpha,
            fit_intercept=self.config.fit_intercept,
        )
        self.model.fit(X_train_scaled, y_train)

        # Train metrics
        y_train_pred = self.model.predict(X_train_scaled)
        train_r2 = self._compute_r2(y_train, y_train_pred)
        train_rmse = self._compute_rmse(y_train, y_train_pred)

        if self.verbose:
            print(f"Train R²: {train_r2:.6f}")
            print(f"Train RMSE: {train_rmse:.6f}")
            print()

        # Feature coefficients
        feature_coefficients = {
            name: float(coef)
            for name, coef in zip(FEATURE_NAMES, self.model.coef_)
        }

        # Validation metrics
        val_r2, val_rmse, val_stats = None, None, None
        if include_val:
            if self.verbose:
                print("Generating val dataset...")
            X_val, y_val, meta_val, val_stats = generator.generate("val")

            if len(X_val) > 0:
                # Apply train scaler to val data
                X_val_scaled = self.scaler.transform(X_val)

                y_val_pred = self.model.predict(X_val_scaled)
                val_r2 = self._compute_r2(y_val, y_val_pred)
                val_rmse = self._compute_rmse(y_val, y_val_pred)

                if self.verbose:
                    print(f"Val samples: {len(X_val)}")
                    print(f"Val R²: {val_r2:.6f}")
                    print(f"Val RMSE: {val_rmse:.6f}")
                    print()

        # Top features by absolute coefficient
        if self.verbose:
            print("Top 10 features by |coefficient|:")
            sorted_features = sorted(
                feature_coefficients.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for i, (name, coef) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {name}: {coef:+.6f}")
            print()

        return TrainResult(
            model=self.model,
            scaler=self.scaler,
            train_stats=train_stats,
            val_stats=val_stats,
            config=self.config,
            train_r2=train_r2,
            train_rmse=train_rmse,
            val_r2=val_r2,
            val_rmse=val_rmse,
            feature_coefficients=feature_coefficients,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature array (N, NUM_FEATURES)

        Returns:
            Predictions (N,)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained! Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

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
    Save trained model and artifacts.

    Args:
        result: Training result
        output_dir: Base output directory
        run_id: Optional run identifier (defaults to timestamp)

    Returns:
        Path to saved model directory
    """
    if run_id is None:
        run_id = f"ridge_v0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_dir = Path(output_dir) / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(result.model, model_dir / "ridge_model.joblib")

    # Save scaler
    joblib.dump(result.scaler, model_dir / "scaler.joblib")

    # Save config and metrics
    with open(model_dir / "train_result.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    # Save feature names
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"Model saved to: {model_dir}")
    return model_dir


def load_model(model_dir: Path) -> tuple[Ridge, StandardScaler, dict]:
    """
    Load trained model and artifacts.

    Args:
        model_dir: Path to saved model directory

    Returns:
        Tuple of (model, scaler, config_dict)
    """
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "ridge_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")

    with open(model_dir / "train_result.json", "r") as f:
        config_dict = json.load(f)

    return model, scaler, config_dict


if __name__ == "__main__":
    print("=== Ridge Trainer Validation ===")

    # Train with default config
    trainer = RidgeTrainer(
        config=TrainConfig(alpha=1.0),
        verbose=True,
    )

    result = trainer.train(include_val=True)

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Train R²: {result.train_r2:.6f}")
    print(f"Train RMSE: {result.train_rmse:.6f}")
    if result.val_r2 is not None:
        print(f"Val R²: {result.val_r2:.6f}")
        print(f"Val RMSE: {result.val_rmse:.6f}")

    # Save model
    output_dir = Path(__file__).parent.parent.parent / "checkpoints" / "baseline_b"
    model_dir = save_model(result, output_dir)

    # Test loading
    print("\nTesting model loading...")
    loaded_model, loaded_scaler, loaded_config = load_model(model_dir)
    print(f"Loaded model type: {type(loaded_model).__name__}")
    print(f"Loaded scaler type: {type(loaded_scaler).__name__}")
    print(f"Loaded config keys: {list(loaded_config.keys())}")

    print("\n=== Validation Complete ===")
