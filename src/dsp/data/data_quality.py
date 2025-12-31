"""
Data quality monitoring for Sleeve IM minute bar data.

This module provides real-time and batch data quality assessment,
including synthetic bar detection, outlier flagging, and data completeness metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, time, datetime
from typing import Dict, List, Optional, Tuple

from .minute_bar import DailyMinuteBars, MinuteBar

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Thresholds (from Sleeve IM spec)
# =============================================================================

# Synthetic bar limits
MAX_SYNTHETIC_PCT = 0.70  # 70% max synthetic bars
SYNTHETIC_WARN_PCT = 0.50  # 50% synthetic bar warning threshold

# Volume thresholds
MIN_PREMARKET_VOLUME = 10_000  # Minimum premarket volume for tradability
MIN_PREMARKET_VOLUME_WARN = 50_000  # Warning level for low premarket volume

# Staleness thresholds
MAX_STALENESS_SECONDS = 3600  # 1 hour max staleness (60 minutes)
STALENESS_WARN_SECONDS = 1800  # 30 minutes staleness warning

# Gap detection
MAX_GAP_PCT = 0.10  # 10% max overnight gap before flagging
EXTREME_GAP_PCT = 0.20  # 20% gap is extreme


# =============================================================================
# Quality Report Data Classes
# =============================================================================


@dataclass
class BarQualityMetrics:
    """Quality metrics for a single minute bar."""

    timestamp: datetime
    is_synthetic: bool
    staleness_seconds: float
    volume: int
    trade_count: int

    # Flags
    is_stale: bool = False  # Exceeds staleness threshold
    is_low_liquidity: bool = False  # Low volume/trades for real bar
    is_outlier: bool = False  # Price outlier detected


@dataclass
class DailyQualityReport:
    """Data quality report for a single symbol-day."""

    symbol: str
    trading_date: date

    # Bar counts
    total_bars: int = 0
    real_bars: int = 0
    synthetic_bars: int = 0
    synthetic_pct: float = 0.0

    # Staleness metrics
    max_staleness_seconds: float = 0.0
    avg_staleness_seconds: float = 0.0
    stale_bar_count: int = 0

    # Volume metrics
    total_volume: int = 0
    premarket_volume: int = 0
    rth_volume: int = 0

    # Price metrics
    overnight_gap_pct: float = 0.0
    max_bar_range_pct: float = 0.0
    outlier_count: int = 0

    # Quality flags
    is_tradable: bool = False
    flags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived flags."""
        self._evaluate_tradability()

    def _evaluate_tradability(self):
        """Evaluate overall tradability based on quality metrics."""
        self.flags = []
        self.is_tradable = True

        # Check synthetic bar percentage
        if self.synthetic_pct > MAX_SYNTHETIC_PCT:
            self.flags.append(f"HIGH_SYNTHETIC: {self.synthetic_pct:.1%} > {MAX_SYNTHETIC_PCT:.0%}")
            self.is_tradable = False
        elif self.synthetic_pct > SYNTHETIC_WARN_PCT:
            self.flags.append(f"WARN_SYNTHETIC: {self.synthetic_pct:.1%}")

        # Check premarket volume
        if self.premarket_volume < MIN_PREMARKET_VOLUME:
            self.flags.append(f"LOW_PREMARKET_VOL: {self.premarket_volume:,} < {MIN_PREMARKET_VOLUME:,}")
            self.is_tradable = False
        elif self.premarket_volume < MIN_PREMARKET_VOLUME_WARN:
            self.flags.append(f"WARN_PREMARKET_VOL: {self.premarket_volume:,}")

        # Check staleness
        if self.max_staleness_seconds > MAX_STALENESS_SECONDS:
            self.flags.append(f"HIGH_STALENESS: {self.max_staleness_seconds:.0f}s > {MAX_STALENESS_SECONDS}s")
            self.is_tradable = False
        elif self.max_staleness_seconds > STALENESS_WARN_SECONDS:
            self.flags.append(f"WARN_STALENESS: {self.max_staleness_seconds:.0f}s")

        # Check overnight gap
        if abs(self.overnight_gap_pct) > EXTREME_GAP_PCT:
            self.flags.append(f"EXTREME_GAP: {self.overnight_gap_pct:.1%}")
        elif abs(self.overnight_gap_pct) > MAX_GAP_PCT:
            self.flags.append(f"WARN_GAP: {self.overnight_gap_pct:.1%}")

        # Check outliers
        if self.outlier_count > 0:
            self.flags.append(f"OUTLIERS: {self.outlier_count}")


@dataclass
class MultiSymbolQualityReport:
    """Aggregated quality report for multiple symbols on a day."""

    trading_date: date
    reports: Dict[str, DailyQualityReport] = field(default_factory=dict)

    @property
    def tradable_symbols(self) -> List[str]:
        """List of symbols passing all quality gates."""
        return [s for s, r in self.reports.items() if r.is_tradable]

    @property
    def failed_symbols(self) -> List[str]:
        """List of symbols failing quality gates."""
        return [s for s, r in self.reports.items() if not r.is_tradable]

    @property
    def overall_tradable(self) -> bool:
        """True if all symbols are tradable."""
        return len(self.failed_symbols) == 0

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Data Quality Report for {self.trading_date}",
            f"=" * 50,
            f"Symbols: {len(self.reports)}",
            f"Tradable: {len(self.tradable_symbols)}",
            f"Failed: {len(self.failed_symbols)}",
        ]

        if self.failed_symbols:
            lines.append(f"\nFailed Symbols:")
            for symbol in self.failed_symbols:
                report = self.reports[symbol]
                flags_str = ", ".join(report.flags[:3])  # First 3 flags
                lines.append(f"  {symbol}: {flags_str}")

        return "\n".join(lines)


# =============================================================================
# Quality Assessment Functions
# =============================================================================


def assess_daily_quality(daily_bars: DailyMinuteBars) -> DailyQualityReport:
    """
    Assess data quality for a single symbol-day.

    Args:
        daily_bars: DailyMinuteBars instance with all bars

    Returns:
        DailyQualityReport with quality metrics and flags
    """
    report = DailyQualityReport(
        symbol=daily_bars.symbol,
        trading_date=daily_bars.date,
        total_bars=daily_bars.total_bars,
        real_bars=daily_bars.real_bars,
        synthetic_bars=daily_bars.synthetic_bars,
        synthetic_pct=daily_bars.synthetic_pct,
        total_volume=daily_bars.total_volume,
        premarket_volume=daily_bars.premarket_volume,
        rth_volume=daily_bars.rth_volume,
    )

    # Calculate staleness metrics
    if daily_bars.bars:
        staleness_values = [b.seconds_since_last_trade for b in daily_bars.bars]
        # Filter out inf values for averaging
        finite_staleness = [s for s in staleness_values if s != float("inf")]

        report.max_staleness_seconds = max(
            s for s in staleness_values if s != float("inf")
        ) if finite_staleness else 0.0
        report.avg_staleness_seconds = (
            sum(finite_staleness) / len(finite_staleness) if finite_staleness else 0.0
        )
        report.stale_bar_count = sum(
            1 for s in staleness_values
            if s != float("inf") and s > STALENESS_WARN_SECONDS
        )

    # Calculate overnight gap
    if daily_bars.prior_close > 0 and daily_bars.first_real_price > 0:
        report.overnight_gap_pct = (
            daily_bars.first_real_price / daily_bars.prior_close - 1
        )

    # Calculate max bar range
    if daily_bars.bars:
        bar_ranges = [b.bar_range_pct for b in daily_bars.bars if not b.is_synthetic]
        report.max_bar_range_pct = max(bar_ranges) if bar_ranges else 0.0

    # Evaluate tradability (populates flags)
    report._evaluate_tradability()

    return report


def assess_multi_symbol_quality(
    bars_by_symbol: Dict[str, DailyMinuteBars],
    trading_date: date,
) -> MultiSymbolQualityReport:
    """
    Assess data quality for multiple symbols on a day.

    Args:
        bars_by_symbol: Dict mapping symbol to DailyMinuteBars
        trading_date: The trading date

    Returns:
        MultiSymbolQualityReport with all symbol reports
    """
    report = MultiSymbolQualityReport(trading_date=trading_date)

    for symbol, daily_bars in bars_by_symbol.items():
        report.reports[symbol] = assess_daily_quality(daily_bars)

    return report


# =============================================================================
# Quality Monitoring Functions
# =============================================================================


class DataQualityMonitor:
    """
    Real-time data quality monitoring for Sleeve IM.

    Tracks quality metrics over time and alerts on degradation.
    """

    def __init__(
        self,
        synthetic_threshold: float = MAX_SYNTHETIC_PCT,
        volume_threshold: int = MIN_PREMARKET_VOLUME,
        staleness_threshold: float = MAX_STALENESS_SECONDS,
    ):
        """
        Initialize quality monitor.

        Args:
            synthetic_threshold: Max allowed synthetic bar percentage
            volume_threshold: Min required premarket volume
            staleness_threshold: Max allowed staleness in seconds
        """
        self.synthetic_threshold = synthetic_threshold
        self.volume_threshold = volume_threshold
        self.staleness_threshold = staleness_threshold

        # Historical reports for trend analysis
        self._history: Dict[Tuple[str, date], DailyQualityReport] = {}

    def check(
        self, daily_bars: DailyMinuteBars
    ) -> Tuple[bool, DailyQualityReport]:
        """
        Check if data passes quality gates.

        Args:
            daily_bars: DailyMinuteBars to check

        Returns:
            Tuple of (passes_gate, quality_report)
        """
        report = assess_daily_quality(daily_bars)

        # Store in history
        key = (daily_bars.symbol, daily_bars.date)
        self._history[key] = report

        # Log quality assessment
        if report.is_tradable:
            logger.info(
                "Quality check PASSED: %s %s (synthetic=%.1f%%, premarket_vol=%d)",
                daily_bars.symbol,
                daily_bars.date,
                report.synthetic_pct * 100,
                report.premarket_volume,
            )
        else:
            logger.warning(
                "Quality check FAILED: %s %s - %s",
                daily_bars.symbol,
                daily_bars.date,
                ", ".join(report.flags),
            )

        return report.is_tradable, report

    def check_multi(
        self, bars_by_symbol: Dict[str, DailyMinuteBars], trading_date: date
    ) -> Tuple[bool, MultiSymbolQualityReport]:
        """
        Check quality for multiple symbols.

        Args:
            bars_by_symbol: Dict mapping symbol to DailyMinuteBars
            trading_date: The trading date

        Returns:
            Tuple of (all_pass, multi_report)
        """
        report = assess_multi_symbol_quality(bars_by_symbol, trading_date)

        # Store individual reports in history
        for symbol, daily_report in report.reports.items():
            self._history[(symbol, trading_date)] = daily_report

        # Log summary
        if report.overall_tradable:
            logger.info(
                "Multi-symbol quality check PASSED: %d symbols on %s",
                len(report.reports),
                trading_date,
            )
        else:
            logger.warning(
                "Multi-symbol quality check FAILED: %d/%d symbols failed on %s: %s",
                len(report.failed_symbols),
                len(report.reports),
                trading_date,
                ", ".join(report.failed_symbols),
            )

        return report.overall_tradable, report

    def get_history(
        self, symbol: str, days: int = 30
    ) -> List[DailyQualityReport]:
        """
        Get quality history for a symbol.

        Args:
            symbol: Stock symbol
            days: Number of recent days to return

        Returns:
            List of DailyQualityReport ordered by date (newest first)
        """
        reports = [
            report
            for (s, d), report in self._history.items()
            if s == symbol
        ]
        reports.sort(key=lambda r: r.trading_date, reverse=True)
        return reports[:days]

    def get_quality_trend(
        self, symbol: str, days: int = 30
    ) -> Dict[str, List[float]]:
        """
        Get quality metric trends for a symbol.

        Args:
            symbol: Stock symbol
            days: Number of recent days

        Returns:
            Dict with metric name → list of values (oldest first)
        """
        history = self.get_history(symbol, days)
        history.reverse()  # Oldest first

        return {
            "synthetic_pct": [r.synthetic_pct for r in history],
            "premarket_volume": [float(r.premarket_volume) for r in history],
            "max_staleness": [r.max_staleness_seconds for r in history],
            "overnight_gap": [abs(r.overnight_gap_pct) for r in history],
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def is_tradable(daily_bars: DailyMinuteBars) -> bool:
    """
    Quick check if data is tradable.

    Args:
        daily_bars: DailyMinuteBars to check

    Returns:
        True if data passes all quality gates
    """
    report = assess_daily_quality(daily_bars)
    return report.is_tradable


def get_quality_summary(daily_bars: DailyMinuteBars) -> str:
    """
    Get human-readable quality summary.

    Args:
        daily_bars: DailyMinuteBars to summarize

    Returns:
        Multi-line summary string
    """
    report = assess_daily_quality(daily_bars)

    lines = [
        f"Quality Summary: {report.symbol} {report.trading_date}",
        f"  Bars: {report.total_bars} ({report.real_bars} real, {report.synthetic_bars} synthetic)",
        f"  Synthetic %: {report.synthetic_pct:.1%}",
        f"  Premarket Volume: {report.premarket_volume:,}",
        f"  Max Staleness: {report.max_staleness_seconds:.0f}s",
        f"  Overnight Gap: {report.overnight_gap_pct:+.2%}",
        f"  Tradable: {'YES' if report.is_tradable else 'NO'}",
    ]

    if report.flags:
        lines.append(f"  Flags: {', '.join(report.flags)}")

    return "\n".join(lines)
