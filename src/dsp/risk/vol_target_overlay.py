"""
Volatility targeting overlay for DSP-100K.

This module computes a portfolio-level risk multiplier that can be applied to
directional sleeves (e.g., DM, VRP-ERP) to stabilize risk.

See: dsp100k/docs/SPEC_VOL_TARGET_OVERLAY.md
"""

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from ..data.fetcher import DataFetcher


@dataclass(frozen=True)
class VolTargetOverlayConfig:
    """Pre-registered configuration for the volatility targeting overlay."""

    proxy_symbol: str = "SPY"
    lookback_days: int = 21

    target_vol: float = 0.10  # 10% annualized
    min_mult: float = 0.25
    max_mult: float = 1.50

    rebalance_band: float = 0.10
    min_days_between_changes: int = 2

    state_path: Path = Path("data/vol_target_overlay_state.json")


@dataclass
class VolTargetOverlayState:
    """Persisted overlay state."""

    last_multiplier: float = 1.0
    last_update_date: Optional[date] = None

    def to_dict(self) -> dict:
        return {
            "last_multiplier": float(self.last_multiplier),
            "last_update_date": self.last_update_date.isoformat() if self.last_update_date else None,
            "updated_at": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def from_dict(data: dict) -> "VolTargetOverlayState":
        last_update = data.get("last_update_date")
        return VolTargetOverlayState(
            last_multiplier=float(data.get("last_multiplier", 1.0)),
            last_update_date=date.fromisoformat(last_update) if last_update else None,
        )


class VolTargetOverlay:
    """Computes and persists a volatility targeting multiplier."""

    def __init__(self, fetcher: DataFetcher, config: Optional[VolTargetOverlayConfig] = None):
        self.fetcher = fetcher
        self.config = config or VolTargetOverlayConfig()
        self.state = self._load_state()

    async def compute_multiplier(self, as_of_date: Optional[date] = None) -> float:
        """
        Compute the raw (un-smoothed) multiplier for the given date.

        Uses realized volatility of `proxy_symbol` over `lookback_days`.
        """
        if as_of_date is None:
            as_of_date = self.fetcher.calendar.get_latest_complete_session()

        realized_vol = await self.fetcher.get_volatility(
            self.config.proxy_symbol,
            lookback_days=self.config.lookback_days,
            end_date=as_of_date,
        )

        if not realized_vol or realized_vol <= 0:
            return 1.0

        raw = self.config.target_vol / realized_vol
        return float(min(self.config.max_mult, max(self.config.min_mult, raw)))

    def should_rebalance(self, as_of_date: date, new_multiplier: float) -> bool:
        """
        Decide whether to adopt a new multiplier.

        Rules:
        - only change if delta >= rebalance_band
        - and at least min_days_between_changes since last update
        """
        if new_multiplier <= 0:
            return False

        if abs(new_multiplier - self.state.last_multiplier) < self.config.rebalance_band:
            return False

        if self.state.last_update_date is None:
            return True

        days_since = (as_of_date - self.state.last_update_date).days
        return days_since >= self.config.min_days_between_changes

    def adopt_multiplier(self, as_of_date: date, new_multiplier: float) -> None:
        """Persist the new multiplier as the active one."""
        self.state.last_multiplier = float(new_multiplier)
        self.state.last_update_date = as_of_date
        self._save_state()

    def get_active_multiplier(self) -> float:
        """Return the currently active (persisted) multiplier."""
        return float(self.state.last_multiplier)

    def _load_state(self) -> VolTargetOverlayState:
        try:
            if self.config.state_path.exists():
                with self.config.state_path.open("r", encoding="utf-8") as f:
                    return VolTargetOverlayState.from_dict(json.load(f))
        except Exception:
            # Fail-safe: do not block trading; default to 1.0.
            return VolTargetOverlayState()

        return VolTargetOverlayState()

    def _save_state(self) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config.state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2, sort_keys=True)
        tmp_path.replace(self.config.state_path)

