"""
Train and backtest the weekly SPY/QQQ direction NN gated by the VRP regime score.

Output:
- Trained logistic regression model (saved to data/vrp/models/)
- Backtest report comparing gated vs ungated performance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_PATH = Path("data/vrp/nn_features.parquet")
MODEL_DIR = Path("data/vrp/models")


@dataclass
class Split:
    name: str
    start: str
    end: str


SPLITS = [
    Split("train", "2015-01-01", "2020-12-31"),
    Split("val", "2021-01-01", "2022-12-31"),
    Split("test", "2023-01-01", "2024-12-31"),
    Split("holdout", "2025-01-01", "2026-12-31"),
]


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURE_PATH)
    df = df.dropna()
    return df


def build_datasets(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result = {}
    for split in SPLITS:
        mask = (df.index >= split.start) & (df.index <= split.end)
        result[split.name] = df.loc[mask]
    return result


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["label"]
    drop_cols = ["label", "gate_state", "gate_numeric", "target_return"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, target


def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=2000,
        verbose=0,
        random_state=42,
    )
    model.fit(X_scaled, y)
    return model, scaler


def evaluate(model: LogisticRegression, scaler: StandardScaler, df: pd.DataFrame) -> Dict[str, Any]:
    X, y = prepare_xy(df)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)
    accuracy = accuracy_score(y, preds)

    # Compute weekly return using SPY target_return, scaled by predictions
    returns = []
    for i, idx in enumerate(df.index):
        direction = preds[i]
        actual_return = df.iloc[i]["target_return"]
        gate_mult = df.iloc[i]["gate_numeric"]
        exposure = {1: 1.0, 0: 0.0, -1: -1.0}[direction]
        returns.append(exposure * gate_mult * actual_return)
    ret_series = pd.Series(returns, index=df.index)
    sharpe = ret_series.mean() / ret_series.std() * np.sqrt(52) if ret_series.std() > 0 else 0.0
    net_pnl = ret_series.sum()
    return {
        "accuracy": accuracy,
        "sharpe": float(sharpe),
        "net_pnl": float(net_pnl),
        "returns": ret_series,
        "preds": preds,
        "probs": probs,
    }


def gated_vs_ungated(model: LogisticRegression, scaler: StandardScaler, df: pd.DataFrame) -> Dict[str, float]:
    optimistic = evaluate(model, scaler, df)
    ungated_df = df.copy()
    ungated_df["gate_numeric"] = 1.0
    ungated = evaluate(model, scaler, ungated_df)
    return {
        "gated_sharpe": optimistic["sharpe"],
        "ungated_sharpe": ungated["sharpe"],
        "gated_pnl": optimistic["net_pnl"],
        "ungated_pnl": ungated["net_pnl"],
    }


def persist_model(model: LogisticRegression, scaler: StandardScaler) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "vrp_nn_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "vrp_nn_scaler.pkl")
    logger.info("Saved model+scaler to %s", MODEL_DIR)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_features()
    splits = build_datasets(df)
    train_df = splits["train"]
    if train_df.empty:
        logger.error("Train split empty")
        return 1
    X_train, y_train = prepare_xy(train_df)
    model, scaler = train_model(X_train, y_train)
    persist_model(model, scaler)

    results = {}
    for name, split_df in splits.items():
        if split_df.empty:
            continue
        res = evaluate(model, scaler, split_df)
        res["gated_vs_ungated"] = gated_vs_ungated(model, scaler, split_df)
        results[name] = {
            "sharpe": res["sharpe"],
            "net_pnl": res["net_pnl"],
            "accuracy": res["accuracy"],
            "gated_vs_ungated": res["gated_vs_ungated"],
        }
    out_path = MODEL_DIR / "vrp_nn_evaluation.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Saved evaluation summary to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
