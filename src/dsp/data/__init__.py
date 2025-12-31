"""Data pipeline module for DSP-100K."""

from .fetcher import DataFetcher
from .cache import DataCache
from .validation import DataValidator, DataQualityReport

# Sleeve IM minute bar data structures
from .minute_bar import (
    RawMinuteBar,
    MinuteBar,
    DailyMinuteBars,
    build_time_grid,
    build_minute_bars,
)
from .polygon_fetcher import (
    PolygonConfig,
    PolygonFetcher,
    fetch_minute_bars,
)
from .data_quality import (
    DailyQualityReport,
    MultiSymbolQualityReport,
    DataQualityMonitor,
    assess_daily_quality,
    assess_multi_symbol_quality,
    is_tradable,
    get_quality_summary,
)

__all__ = [
    # Core data pipeline
    "DataFetcher",
    "DataCache",
    "DataValidator",
    "DataQualityReport",
    # Sleeve IM minute bars
    "RawMinuteBar",
    "MinuteBar",
    "DailyMinuteBars",
    "build_time_grid",
    "build_minute_bars",
    # Polygon.io fetcher
    "PolygonConfig",
    "PolygonFetcher",
    "fetch_minute_bars",
    # Data quality monitoring
    "DailyQualityReport",
    "MultiSymbolQualityReport",
    "DataQualityMonitor",
    "assess_daily_quality",
    "assess_multi_symbol_quality",
    "is_tradable",
    "get_quality_summary",
]
