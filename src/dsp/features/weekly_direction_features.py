"""
Engine to compute weekly features and targets for the VRP-gated NN.

Outputs weekly dataset with:
- SPY/QQQ returns, vol stats, cross-asset spreads, VRP gate state
- Gate scores and states from VRPRegimeGate
- Target label (DOWN/FLAT/UP) for next-week SPY return

Usage:
  from src.dsp.features.weekly_direction_features import build_weekly_dataset
  df = build_weekly_dataset()
  df.to_parquet("data/vrp/nn_features.parquet")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..regime.vrp_regime_gate import GateState, VRPRegimeGate

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/vrp")
EQUITY_DIR = DATA_DIR / "equities"
INDICES_DIR = DATA_DIR / "indices"
FUTURES_DIR = DATA_DIR / "futures"


class DirectionLabel(Enum):
    DOWN = -1
    FLAT = 0
    UP = 1


@dataclass
class WeeklySample:
    date: pd.Timestamp
    features: Dict[str, float]
    label: DirectionLabel
    gate_state: GateState
    gate_score: float


def load_equity(symbol: str) -> pd.DataFrame:
    path = EQUITY_DIR / f"{symbol}_daily.parquet"
    df = pd.read_parquet(path)
    # Normalize to date-only index for joins
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index).tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index = df.index.normalize()  # Remove time component, keep UTC
    df = df.sort_index()
    return df


def load_gate_data() -> pd.DataFrame:
    vix = pd.read_parquet(INDICES_DIR / "VIX_spot.parquet")
    vvix = pd.read_parquet(INDICES_DIR / "VVIX.parquet")
    vx = pd.read_parquet(FUTURES_DIR / "VX_F1_CBOE.parquet")
    df = (
        vix.rename(columns={"vix_spot": "vix"})
        .join(vvix, how="inner")
        .join(vx[["vx_f1"]], how="inner")
        .dropna()
    )
    df.index = pd.to_datetime(df.index).tz_localize("UTC").normalize()
    return df


def run_gate(df: pd.DataFrame) -> pd.DataFrame:
    gate = VRPRegimeGate()
    rows = []
    for ts, row in df.iterrows():
        state = gate.update(
            vix=float(row["vix"]),
            vvix=float(row["vvix"]),
            vx_f1=float(row["vx_f1"]),
            as_of_date=ts.date(),
        )
        rows.append(
            {
                "date": ts,
                "gate_state": state.value,
                "gate_score": gate.last_score,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def compute_lookback_returns(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change(periods=window).rename(f"ret_{window}")


def compute_volatility(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std().rename(f"vol_{window}")


def weekly_target(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    fridays = df.index[df.index.weekday == 4]
    records = {}
    for friday in fridays:
        monday = friday - pd.Timedelta(days=4)
        if monday not in df.index:
            continue
        start = df.loc[monday]["spy_close"]
        end = df.loc[friday]["spy_close"]
        ret = (end / start) - 1.0
        if ret > 0.005:
            label = DirectionLabel.UP
        elif ret < -0.005:
            label = DirectionLabel.DOWN
        else:
            label = DirectionLabel.FLAT
        records[friday] = (label, ret)
    return pd.Series(records)


def build_weekly_dataset() -> pd.DataFrame:
    spy = load_equity("SPY")
    qqq = load_equity("QQQ")
    tlt = load_equity("TLT")
    gate_data = run_gate(load_gate_data())
    print("gate data sample", gate_data.head())
    print("gate rows", len(gate_data))

    df = (
        spy[["close"]]
        .rename(columns={"close": "spy_close"})
        .join(qqq["close"].rename("qqq_close"), how="inner")
        .join(tlt["close"].rename("tlt_close"), how="inner")
        .dropna()
    )
    print("pre-gate df length", len(df))
    print("weekday sample", df.index.weekday[:5])
    df["spy_ret5"] = compute_lookback_returns(df["spy_close"], 5)
    df["spy_ret10"] = compute_lookback_returns(df["spy_close"], 10)
    df["spy_ret20"] = compute_lookback_returns(df["spy_close"], 20)
    df["spy_ret60"] = compute_lookback_returns(df["spy_close"], 60)
    df["qqq_ret5"] = compute_lookback_returns(df["qqq_close"], 5)
    df["qqq_ret10"] = compute_lookback_returns(df["qqq_close"], 10)
    df["qqq_ret20"] = compute_lookback_returns(df["qqq_close"], 20)
    df["tlt_ret5"] = compute_lookback_returns(df["tlt_close"], 5)
    df["tlt_ret10"] = compute_lookback_returns(df["tlt_close"], 10)
    df["tlt_ret20"] = compute_lookback_returns(df["tlt_close"], 20)

    df["vol_spy_20"] = compute_volatility(df["spy_close"], 20)
    df["vol_qqq_20"] = compute_volatility(df["qqq_close"], 20)
    df["vol_ratio"] = df["vol_qqq_20"] / df["vol_spy_20"]
    df["cross_corr"] = df["spy_ret5"].rolling(20).corr(df["qqq_ret5"])
    df["eq_bond_spread"] = df["spy_ret20"] - df["tlt_ret20"]

    df = df.join(gate_data, how="left").ffill()
    print("after join length", len(df))
    print(df.head())
    df = df.dropna()
    print("after dropna", len(df))

    weekly_rows: List[WeeklySample] = []
    target_series = weekly_target(df)
    mondays = df.index[df.index.weekday == 0]
    print("Mondays", len(mondays), "Targets", len(target_series), "index sample", df.index[:5])
    for dt in mondays:  # Monday
        features = df.loc[dt].to_dict()
        friday = dt + pd.Timedelta(days=4)
        if friday not in target_series:
            continue
        label, target_return = target_series[friday]
        weekly_rows.append(
            WeeklySample(
                date=friday,
                features=features,
                label=label,
                gate_state=GateState(features["gate_state"]),
                gate_score=features["gate_score"],
            )
        )
    records = []
    for sample in weekly_rows:
        row = sample.features.copy()
        row["label"] = int(sample.label.value)
        row["gate_state"] = sample.gate_state.value
        row["gate_numeric"] = {"OPEN": 1.0, "REDUCE": 0.5, "CLOSED": 0.0}[sample.gate_state.value]
        row["date"] = sample.date
        row["target_return"] = float(
            target_series[sample.date][1]
        )
        if row["gate_state"] not in GateState.__members__:
            continue
        records.append(row)
    logger.info("Built %d weekly samples / %d rows", len(weekly_rows), len(records))
    if not records:
        raise RuntimeError("No weekly records generated")
    out = pd.DataFrame(records).set_index("date")
    out = out.sort_index()
    return out


def save_features(path: Path) -> None:
    df = build_weekly_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Saved weekly features to %s (%d rows)", path, len(df))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build weekly direction features.")
    parser.add_argument(
        "--output",
        default="data/vrp/nn_features.parquet",
        help="Output parquet path.",
    )
    args = parser.parse_args()
    save_features(Path(args.output))
