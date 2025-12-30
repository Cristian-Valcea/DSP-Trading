"""
Sleeve A: Equity Long/Short (12-1 Momentum).

============================================================
                    NOT IMPLEMENTED
============================================================

This file is a placeholder. Sleeve A implementation is pending.

Per SPEC_DSP_100K.md Section 2.1, Sleeve A should implement:
- 12-1 month momentum ranking (skip most recent month)
- Top 20 longs, bottom 20 shorts
- Volatility targeting (5% annualized)
- Monthly rebalancing
- Single-name cap at 4% of NLV
- Sector cap at 20% gross
- Short squeeze protection (20% gap triggers cover)
- Beta limit (|β| ≤ 0.10)
- SPY hedge up to 20% of NLV

Required Components:
1. MomentumRanker - Calculate 12-1 momentum scores
2. PositionSizer - Volatility-targeted position sizing
3. RiskMonitor - Short squeeze detection, beta hedging
4. SleeveAManager - Orchestrate rebalancing and risk controls

Dependencies:
- Universe filtering (min price $10, min ADV $20M)
- Sector classification data
- Historical returns (12+ months)
- Real-time beta calculation vs SPY

Kill Criteria (from spec):
- Rolling 90-day Sharpe < 0 for 5 consecutive days
- Drawdown > 15% triggers flatten

============================================================
                  IMPLEMENTATION STATUS
============================================================
Status: NOT STARTED
Priority: HIGH (Core revenue driver)
Estimated effort: 3-5 days
Blocking dependencies: None (can use existing data infrastructure)

============================================================
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from ..utils.config import SleeveAConfig

logger = logging.getLogger(__name__)


class SleeveANotImplementedError(NotImplementedError):
    """Raised when Sleeve A functionality is called but not yet implemented."""
    pass


@dataclass
class MomentumSignal:
    """Momentum signal for a single stock (placeholder)."""
    symbol: str
    ret_12m_skip: float  # 12-month return skipping most recent month
    rank: int            # 1 = highest momentum
    sector: str
    as_of_date: date


@dataclass
class SleeveAPosition:
    """Position in Sleeve A portfolio (placeholder)."""
    symbol: str
    shares: int
    side: str           # "long" or "short"
    weight: float       # % of NLV
    sector: str
    entry_price: float
    current_price: float
    unrealized_pnl: float


@dataclass
class SleeveAAdjustment:
    """Adjustment plan for Sleeve A (placeholder)."""
    as_of_date: date
    long_targets: List[MomentumSignal]
    short_targets: List[MomentumSignal]
    positions: List[SleeveAPosition]
    estimated_turnover: float
    rebalance_needed: bool
    beta_hedge_shares: int  # SPY shares for beta hedge


class SleeveA:
    """
    Sleeve A: Equity Long/Short Momentum.

    ============================================================
                        NOT IMPLEMENTED
    ============================================================

    This class is a placeholder stub. All methods raise
    SleeveANotImplementedError to make it clear that the
    functionality is pending implementation.
    """

    def __init__(self, config: SleeveAConfig):
        """
        Initialize Sleeve A manager.

        Args:
            config: Sleeve A configuration
        """
        self.config = config
        self._warn_not_implemented()

    def _warn_not_implemented(self) -> None:
        """Log warning that Sleeve A is not implemented."""
        logger.warning(
            "⚠️  SLEEVE A NOT IMPLEMENTED - This is a placeholder. "
            "Equity L/S momentum strategy requires full implementation."
        )

    async def compute_signals(
        self,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, MomentumSignal]:
        """
        Compute momentum signals for universe.

        NOT IMPLEMENTED - raises SleeveANotImplementedError.
        """
        raise SleeveANotImplementedError(
            "SleeveA.compute_signals() not implemented. "
            "Requires: universe filtering, 12-1 momentum calculation, sector classification."
        )

    async def calculate_targets(
        self,
        sleeve_nav: float,
        as_of_date: Optional[date] = None,
    ) -> SleeveAAdjustment:
        """
        Calculate target positions for rebalancing.

        NOT IMPLEMENTED - raises SleeveANotImplementedError.
        """
        raise SleeveANotImplementedError(
            "SleeveA.calculate_targets() not implemented. "
            "Requires: position sizing, sector caps, vol targeting, beta hedging."
        )

    async def check_short_squeeze(
        self,
        positions: Dict[str, SleeveAPosition],
    ) -> List[str]:
        """
        Check for short squeeze conditions (20% gap).

        NOT IMPLEMENTED - raises SleeveANotImplementedError.
        """
        raise SleeveANotImplementedError(
            "SleeveA.check_short_squeeze() not implemented. "
            "Requires: real-time gap monitoring for short positions."
        )

    async def calculate_beta_hedge(
        self,
        positions: Dict[str, SleeveAPosition],
    ) -> int:
        """
        Calculate SPY hedge shares to achieve beta neutrality.

        NOT IMPLEMENTED - raises SleeveANotImplementedError.
        """
        raise SleeveANotImplementedError(
            "SleeveA.calculate_beta_hedge() not implemented. "
            "Requires: rolling beta calculation, hedge ratio optimization."
        )

    async def check_kill_criteria(self) -> bool:
        """
        Check if Sleeve A should be killed (Sharpe < 0 or DD > 15%).

        NOT IMPLEMENTED - raises SleeveANotImplementedError.
        """
        raise SleeveANotImplementedError(
            "SleeveA.check_kill_criteria() not implemented. "
            "Requires: rolling 90-day Sharpe, drawdown tracking."
        )

    def get_status(self) -> Dict:
        """
        Get current status of Sleeve A.

        Returns placeholder status indicating not implemented.
        """
        return {
            "implemented": False,
            "status": "NOT_IMPLEMENTED",
            "message": "Sleeve A (Equity L/S Momentum) pending implementation",
            "enabled": self.config.enabled,
            "config": {
                "vol_target": self.config.vol_target,
                "n_long": self.config.n_long,
                "n_short": self.config.n_short,
                "max_weight_per_name": self.config.max_weight_per_name,
                "max_sector_gross": self.config.max_sector_gross,
            },
        }


# Export placeholder classes
__all__ = [
    "SleeveA",
    "SleeveANotImplementedError",
    "MomentumSignal",
    "SleeveAPosition",
    "SleeveAAdjustment",
]
