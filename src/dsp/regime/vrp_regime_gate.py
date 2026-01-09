"""
VRP Regime Gate — Regime classifier for gating NN trading strategies.

This module provides a regime gate based on VIX/VVIX/contango indicators.
Instead of trading VRP directly (which failed kill-tests), we use VRP
indicators to classify market regimes and gate when NN strategies trade.

Gate States:
    OPEN:   Favorable regime → NN trades normally
    REDUCE: Uncertain regime → NN can only reduce/close positions
    CLOSED: Crisis regime → NN must stay flat

Usage:
    gate = VRPRegimeGate()
    state = gate.update(vix=15.0, vvix=85.0, vx_f1=17.5)
    if state == GateState.OPEN:
        # NN can trade freely
    elif state == GateState.REDUCE:
        # NN can only reduce exposure
    else:  # CLOSED
        # NN must be flat

See: dsp100k/docs/SPEC_VRP_REGIME_GATE.md for full specification.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GateState(Enum):
    """Gate state enum."""
    OPEN = "OPEN"       # NN trades freely
    REDUCE = "REDUCE"   # NN can only reduce exposure
    CLOSED = "CLOSED"   # NN must be flat


@dataclass
class RegimeGateConfig:
    """Configuration for VRP regime gate.

    Attributes:
        open_threshold: Regime score above which gate is OPEN (default: 0.2)
        closed_threshold: Regime score below which gate is CLOSED (default: -0.3)
        hysteresis: Buffer to prevent flip-flopping at boundaries (default: 0.1)
        min_days_in_state: Minimum days before state can change (default: 2)

        vix_range: (min, max) for VIX normalization (default: 12, 35)
        vvix_range: (min, max) for VVIX normalization (default: 80, 140)
        contango_scale: Divisor for contango normalization (default: 5.0)
    """
    open_threshold: float = 0.2
    closed_threshold: float = -0.3
    hysteresis: float = 0.1
    min_days_in_state: int = 2

    # Normalization ranges (calibrated from historical data)
    vix_range: Tuple[float, float] = (12.0, 35.0)
    vvix_range: Tuple[float, float] = (80.0, 140.0)
    contango_scale: float = 5.0


@dataclass
class GateSnapshot:
    """Point-in-time snapshot of gate state."""
    date: date
    state: GateState
    score: float
    vix: float
    vvix: float
    vx_f1: float
    contango: float
    days_in_state: int

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "state": self.state.value,
            "score": round(self.score, 4),
            "vix": round(self.vix, 2),
            "vvix": round(self.vvix, 2),
            "vx_f1": round(self.vx_f1, 2),
            "contango": round(self.contango, 2),
            "days_in_state": self.days_in_state,
        }


class VRPRegimeGate:
    """VRP-based regime gate for NN trading strategies.

    Computes a regime score from VIX/VVIX/contango and maintains
    a state machine with hysteresis to prevent flip-flopping.

    The gate protects NN strategies by:
    - Allowing full trading when regime is favorable (OPEN)
    - Restricting to position reduction when uncertain (REDUCE)
    - Forcing flat positions during crises (CLOSED)
    """

    def __init__(self, config: Optional[RegimeGateConfig] = None):
        self.config = config or RegimeGateConfig()
        self.current_state = GateState.OPEN
        self.days_in_state = 0
        self.last_score = 0.0
        self.last_snapshot: Optional[GateSnapshot] = None
        self._history: list[GateSnapshot] = []

    def compute_score(self, vix: float, vvix: float, vx_f1: float) -> float:
        """Compute regime score from -1 (crisis) to +1 (risk-on).

        Components (equal-weighted):
            1. VIX level: lower = better (less fear)
            2. VVIX level: lower = better (stable vol expectations)
            3. Contango: higher = better (complacency, VX > VIX)

        Args:
            vix: VIX spot level
            vvix: VVIX level (vol-of-vol)
            vx_f1: Front-month VIX futures price

        Returns:
            Regime score in [-1, +1] range
        """
        vix_min, vix_max = self.config.vix_range
        vvix_min, vvix_max = self.config.vvix_range

        # VIX score: 12→+1, 35→-1
        vix_clipped = np.clip(vix, vix_min, vix_max)
        vix_score = 1.0 - 2.0 * (vix_clipped - vix_min) / (vix_max - vix_min)

        # VVIX score: 80→+1, 140→-1
        vvix_clipped = np.clip(vvix, vvix_min, vvix_max)
        vvix_score = 1.0 - 2.0 * (vvix_clipped - vvix_min) / (vvix_max - vvix_min)

        # Contango score: +5 → +1, -5 → -1
        contango = vx_f1 - vix
        contango_score = np.clip(contango / self.config.contango_scale, -1, 1)

        # Equal-weighted average
        score = (vix_score + vvix_score + contango_score) / 3.0

        return float(score)

    def update(
        self,
        vix: float,
        vvix: float,
        vx_f1: float,
        as_of_date: Optional[date] = None,
    ) -> GateState:
        """Update gate state based on new market data.

        Applies hysteresis and minimum-days-in-state logic to prevent
        excessive flip-flopping at regime boundaries.

        Args:
            vix: VIX spot level
            vvix: VVIX level
            vx_f1: Front-month VIX futures price
            as_of_date: Date for this observation (default: today)

        Returns:
            Current gate state after update
        """
        score = self.compute_score(vix, vvix, vx_f1)
        self.last_score = score

        # Determine target state based on score and hysteresis
        target_state = self._compute_target_state(score)

        # Apply state transition if allowed
        if target_state != self.current_state:
            if self.days_in_state >= self.config.min_days_in_state:
                logger.info(
                    f"Regime gate transition: {self.current_state.value} → {target_state.value} "
                    f"(score={score:.3f}, days_in_prev={self.days_in_state})"
                )
                self.current_state = target_state
                self.days_in_state = 0
            else:
                logger.debug(
                    f"Regime gate transition blocked: need {self.config.min_days_in_state} days, "
                    f"have {self.days_in_state}"
                )

        self.days_in_state += 1

        # Record snapshot
        snapshot = GateSnapshot(
            date=as_of_date or date.today(),
            state=self.current_state,
            score=score,
            vix=vix,
            vvix=vvix,
            vx_f1=vx_f1,
            contango=vx_f1 - vix,
            days_in_state=self.days_in_state,
        )
        self.last_snapshot = snapshot
        self._history.append(snapshot)

        return self.current_state

    def _compute_target_state(self, score: float) -> GateState:
        """Compute target state with hysteresis.

        The hysteresis prevents flip-flopping at threshold boundaries:
        - To exit OPEN, score must drop below (open_threshold - hysteresis)
        - To enter OPEN from below, score must rise above open_threshold
        - Same logic applies at closed_threshold boundary
        """
        cfg = self.config

        if self.current_state == GateState.OPEN:
            # In OPEN: need to drop below threshold - hysteresis to leave
            if score < cfg.closed_threshold:
                return GateState.CLOSED
            elif score < cfg.open_threshold - cfg.hysteresis:
                return GateState.REDUCE
            return GateState.OPEN

        elif self.current_state == GateState.REDUCE:
            # In REDUCE: can go either direction
            if score >= cfg.open_threshold:
                return GateState.OPEN
            elif score < cfg.closed_threshold:
                return GateState.CLOSED
            return GateState.REDUCE

        else:  # CLOSED
            # In CLOSED: need to rise above threshold + hysteresis to leave
            if score >= cfg.open_threshold:
                return GateState.OPEN
            elif score >= cfg.closed_threshold + cfg.hysteresis:
                return GateState.REDUCE
            return GateState.CLOSED

    def is_trading_allowed(self, action: str = "any") -> bool:
        """Check if trading action is allowed under current gate state.

        Args:
            action: One of "open_new", "reduce", "close", "any"

        Returns:
            True if action is allowed, False otherwise
        """
        if self.current_state == GateState.OPEN:
            return True
        elif self.current_state == GateState.REDUCE:
            return action in ("reduce", "close")
        else:  # CLOSED
            return action == "close"

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on gate state.

        Returns:
            1.0 for OPEN, 0.5 for REDUCE, 0.0 for CLOSED
        """
        if self.current_state == GateState.OPEN:
            return 1.0
        elif self.current_state == GateState.REDUCE:
            return 0.5
        else:
            return 0.0

    def get_status(self) -> dict:
        """Get current gate status as dict."""
        return {
            "state": self.current_state.value,
            "score": round(self.last_score, 4),
            "days_in_state": self.days_in_state,
            "trading_allowed": self.is_trading_allowed(),
            "position_multiplier": self.get_position_multiplier(),
        }

    def get_history_df(self) -> pd.DataFrame:
        """Get gate history as DataFrame."""
        if not self._history:
            return pd.DataFrame()

        records = [s.to_dict() for s in self._history]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df

    def save_state(self, path: Path) -> None:
        """Save current state to JSON file."""
        state = {
            "current_state": self.current_state.value,
            "days_in_state": self.days_in_state,
            "last_score": self.last_score,
            "last_snapshot": self.last_snapshot.to_dict() if self.last_snapshot else None,
            "config": {
                "open_threshold": self.config.open_threshold,
                "closed_threshold": self.config.closed_threshold,
                "hysteresis": self.config.hysteresis,
                "min_days_in_state": self.config.min_days_in_state,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved gate state to {path}")

    def load_state(self, path: Path) -> None:
        """Load state from JSON file."""
        with open(path) as f:
            state = json.load(f)

        self.current_state = GateState(state["current_state"])
        self.days_in_state = state["days_in_state"]
        self.last_score = state["last_score"]

        if state.get("last_snapshot"):
            snap = state["last_snapshot"]
            self.last_snapshot = GateSnapshot(
                date=date.fromisoformat(snap["date"]),
                state=GateState(snap["state"]),
                score=snap["score"],
                vix=snap["vix"],
                vvix=snap["vvix"],
                vx_f1=snap["vx_f1"],
                contango=snap["contango"],
                days_in_state=snap["days_in_state"],
            )

        logger.info(f"Loaded gate state from {path}: {self.current_state.value}")


def load_vrp_data(data_dir: Path) -> pd.DataFrame:
    """Load VIX/VVIX/VX data and merge into single DataFrame.

    Args:
        data_dir: Path to dsp100k/data/vrp directory

    Returns:
        DataFrame with columns: vix, vvix, vx_f1, indexed by date
    """
    vix = pd.read_parquet(data_dir / "indices" / "VIX_spot.parquet")
    vvix = pd.read_parquet(data_dir / "indices" / "VVIX.parquet")
    vx_f1 = pd.read_parquet(data_dir / "futures" / "VX_F1_CBOE.parquet")

    # Standardize column names
    vix = vix.rename(columns={"vix_spot": "vix"})
    vx_f1 = vx_f1[["vx_f1"]]

    # Merge on date index
    df = vix.join(vvix, how="inner").join(vx_f1, how="inner")

    return df


def backtest_gate(
    df: pd.DataFrame,
    config: Optional[RegimeGateConfig] = None,
) -> pd.DataFrame:
    """Run gate logic over historical data.

    Args:
        df: DataFrame with vix, vvix, vx_f1 columns
        config: Gate configuration

    Returns:
        DataFrame with added columns: score, state, days_in_state
    """
    gate = VRPRegimeGate(config)

    results = []
    for dt, row in df.iterrows():
        state = gate.update(
            vix=row["vix"],
            vvix=row["vvix"],
            vx_f1=row["vx_f1"],
            as_of_date=dt.date() if hasattr(dt, "date") else dt,
        )
        results.append({
            "date": dt,
            "vix": row["vix"],
            "vvix": row["vvix"],
            "vx_f1": row["vx_f1"],
            "score": gate.last_score,
            "state": state.value,
            "days_in_state": gate.days_in_state,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.set_index("date")

    return result_df


if __name__ == "__main__":
    # Quick validation
    import sys

    logging.basicConfig(level=logging.INFO)

    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "vrp"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    print("Loading VRP data...")
    df = load_vrp_data(data_dir)
    print(f"Loaded {len(df)} rows: {df.index.min()} to {df.index.max()}")

    print("\nRunning backtest...")
    result = backtest_gate(df)

    print("\n=== GATE STATE DISTRIBUTION ===")
    print(result["state"].value_counts(normalize=True).round(3) * 100)

    print("\n=== SCORE STATISTICS ===")
    print(result["score"].describe().round(3))

    print("\n=== LATEST STATE ===")
    latest = result.iloc[-1]
    print(f"Date: {result.index[-1]}")
    print(f"State: {latest['state']}")
    print(f"Score: {latest['score']:.3f}")
    print(f"VIX: {latest['vix']:.1f}, VVIX: {latest['vvix']:.1f}, VX_F1: {latest['vx_f1']:.1f}")
