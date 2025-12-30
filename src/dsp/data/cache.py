"""
Local data cache with staleness detection for DSP-100K.

Provides persistent storage for market data with automatic
expiration and integrity checking.
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for a cached item."""
    key: str
    created_at: datetime
    expires_at: Optional[datetime]
    checksum: str
    size_bytes: int
    source: str  # "IBKR" | "POLYGON" | "DERIVED"


class DataCache:
    """
    Local file-based cache for market data.

    Features:
    - Persistent storage in Parquet format for DataFrames
    - JSON storage for metadata and small objects
    - Automatic staleness detection
    - Integrity checking via checksums
    - Configurable TTL by data type
    """

    # Default TTLs by data type (hours)
    DEFAULT_TTLS = {
        "daily_bars": 24,           # Refresh daily after close
        "adjusted_close": 24,       # Refresh daily
        "intraday_bars": 1,         # Short-lived
        "quotes": 0,                # Never cache (real-time)
        "positions": 0,             # Never cache (real-time)
        "account": 0,               # Never cache (real-time)
        "option_chain": 4,          # Refresh periodically
        "metadata": 168,            # Weekly refresh
        "signals": 24,              # Refresh daily
        "returns": 24,              # Refresh daily
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        custom_ttls: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory for cache files (default: ~/.dsp100k/cache)
            custom_ttls: Override default TTLs (hours) by data type
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".dsp100k" / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories by data type
        self.bars_dir = self.cache_dir / "bars"
        self.signals_dir = self.cache_dir / "signals"
        self.metadata_dir = self.cache_dir / "metadata"

        for d in [self.bars_dir, self.signals_dir, self.metadata_dir]:
            d.mkdir(exist_ok=True)

        # Index file for tracking cache entries
        self.index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict] = self._load_index()

        # TTL configuration
        self.ttls = {**self.DEFAULT_TTLS, **(custom_ttls or {})}

    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load cache index: {e}")
                return {}
        return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._index, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save cache index: {e}")

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()[:16]

    def _get_cache_key(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        date_key: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> str:
        """
        Generate a unique cache key.

        Args:
            data_type: Type of data (daily_bars, signals, etc.)
            symbol: Symbol or universe identifier
            date_key: Date string (YYYYMMDD) or range
            extra: Additional key component
        """
        parts = [data_type]
        if symbol:
            parts.append(symbol)
        if date_key:
            parts.append(date_key)
        if extra:
            parts.append(extra)
        return "_".join(parts)

    def _get_file_path(self, key: str, data_type: str) -> Path:
        """Get file path for a cache key."""
        if "bars" in data_type or "returns" in data_type:
            base_dir = self.bars_dir
            ext = ".parquet"
        elif "signals" in data_type:
            base_dir = self.signals_dir
            ext = ".parquet"
        else:
            base_dir = self.metadata_dir
            ext = ".json"

        return base_dir / f"{key}{ext}"

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached daily bars for a symbol.

        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data if cached and fresh, None otherwise
        """
        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        key = self._get_cache_key("daily_bars", symbol, date_key)

        return self._get_dataframe(key, "daily_bars")

    def put_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        df: pd.DataFrame,
        source: str = "IBKR",
    ) -> bool:
        """
        Cache daily bars for a symbol.

        Args:
            symbol: Symbol
            start_date: Start date
            end_date: End date
            df: DataFrame with OHLCV data
            source: Data source identifier

        Returns:
            True if successfully cached
        """
        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        key = self._get_cache_key("daily_bars", symbol, date_key)

        return self._put_dataframe(key, "daily_bars", df, source)

    def get_universe_bars(
        self,
        universe_id: str,
        end_date: date,
        lookback_days: int,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get cached bars for an entire universe.

        Args:
            universe_id: Universe identifier (e.g., "sleeve_b_etfs")
            end_date: End date
            lookback_days: Number of days of history

        Returns:
            Dict mapping symbols to DataFrames if all cached, None otherwise
        """
        start_date = end_date - timedelta(days=lookback_days)
        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        key = self._get_cache_key("universe_bars", universe_id, date_key)

        # Check if universe metadata exists and is fresh
        metadata = self._get_json(f"{key}_meta")
        if metadata is None:
            return None

        # Load all symbol data
        result = {}
        for symbol in metadata.get("symbols", []):
            df = self.get_daily_bars(symbol, start_date, end_date)
            if df is None:
                return None  # Missing data for at least one symbol
            result[symbol] = df

        return result

    def put_universe_bars(
        self,
        universe_id: str,
        end_date: date,
        lookback_days: int,
        data: Dict[str, pd.DataFrame],
        source: str = "IBKR",
    ) -> bool:
        """
        Cache bars for an entire universe.

        Args:
            universe_id: Universe identifier
            end_date: End date
            lookback_days: Number of days of history
            data: Dict mapping symbols to DataFrames
            source: Data source identifier

        Returns:
            True if successfully cached
        """
        start_date = end_date - timedelta(days=lookback_days)
        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        key = self._get_cache_key("universe_bars", universe_id, date_key)

        # Cache each symbol's data
        for symbol, df in data.items():
            if not self.put_daily_bars(symbol, start_date, end_date, df, source):
                return False

        # Cache universe metadata
        metadata = {
            "universe_id": universe_id,
            "symbols": list(data.keys()),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lookback_days": lookback_days,
        }
        return self._put_json(f"{key}_meta", metadata, "metadata")

    def get_signals(
        self,
        signal_type: str,
        as_of_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached signals.

        Args:
            signal_type: Type of signal (e.g., "sleeve_b_trend")
            as_of_date: Signal calculation date

        Returns:
            DataFrame with signals if cached and fresh, None otherwise
        """
        date_key = as_of_date.strftime("%Y%m%d")
        key = self._get_cache_key("signals", signal_type, date_key)

        return self._get_dataframe(key, "signals")

    def put_signals(
        self,
        signal_type: str,
        as_of_date: date,
        df: pd.DataFrame,
    ) -> bool:
        """
        Cache signals.

        Args:
            signal_type: Type of signal
            as_of_date: Signal calculation date
            df: DataFrame with signals

        Returns:
            True if successfully cached
        """
        date_key = as_of_date.strftime("%Y%m%d")
        key = self._get_cache_key("signals", signal_type, date_key)

        return self._put_dataframe(key, "signals", df, "DERIVED")

    def _get_dataframe(
        self,
        key: str,
        data_type: str,
    ) -> Optional[pd.DataFrame]:
        """Get a cached DataFrame."""
        if not self._is_fresh(key, data_type):
            return None

        file_path = self._get_file_path(key, data_type)
        if not file_path.exists():
            return None

        try:
            df = pd.read_parquet(file_path)

            # Verify checksum
            entry = self._index.get(key)
            if entry:
                actual_checksum = self._compute_checksum(df.to_parquet())
                if actual_checksum != entry.get("checksum"):
                    logger.warning(f"Checksum mismatch for {key}, invalidating cache")
                    self.invalidate(key)
                    return None

            return df
        except Exception as e:
            logger.error(f"Failed to read cached DataFrame {key}: {e}")
            return None

    def _put_dataframe(
        self,
        key: str,
        data_type: str,
        df: pd.DataFrame,
        source: str,
    ) -> bool:
        """Cache a DataFrame."""
        file_path = self._get_file_path(key, data_type)

        try:
            # Convert to parquet bytes for checksum
            parquet_bytes = df.to_parquet()
            checksum = self._compute_checksum(parquet_bytes)

            # Write file
            df.to_parquet(file_path)

            # Update index
            ttl_hours = self.ttls.get(data_type, 24)
            expires_at = None
            if ttl_hours > 0:
                expires_at = datetime.now() + timedelta(hours=ttl_hours)

            self._index[key] = {
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "checksum": checksum,
                "size_bytes": len(parquet_bytes),
                "source": source,
                "data_type": data_type,
                "file_path": str(file_path),
            }
            self._save_index()

            logger.debug(f"Cached {key} ({len(parquet_bytes)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache DataFrame {key}: {e}")
            return False

    def _get_json(self, key: str) -> Optional[Dict]:
        """Get a cached JSON object."""
        if not self._is_fresh(key, "metadata"):
            return None

        file_path = self._get_file_path(key, "metadata")
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read cached JSON {key}: {e}")
            return None

    def _put_json(
        self,
        key: str,
        data: Dict,
        data_type: str,
    ) -> bool:
        """Cache a JSON object."""
        file_path = self._get_file_path(key, data_type)

        try:
            json_str = json.dumps(data, indent=2, default=str)
            checksum = self._compute_checksum(json_str.encode())

            with open(file_path, "w") as f:
                f.write(json_str)

            ttl_hours = self.ttls.get(data_type, 24)
            expires_at = None
            if ttl_hours > 0:
                expires_at = datetime.now() + timedelta(hours=ttl_hours)

            self._index[key] = {
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else None,
                "checksum": checksum,
                "size_bytes": len(json_str),
                "source": "DERIVED",
                "data_type": data_type,
                "file_path": str(file_path),
            }
            self._save_index()

            return True

        except Exception as e:
            logger.error(f"Failed to cache JSON {key}: {e}")
            return False

    def _is_fresh(self, key: str, data_type: str) -> bool:
        """Check if a cache entry is still fresh."""
        entry = self._index.get(key)
        if entry is None:
            return False

        # Check TTL
        expires_at_str = entry.get("expires_at")
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now() > expires_at:
                logger.debug(f"Cache entry {key} has expired")
                return False

        # Check file exists
        file_path = entry.get("file_path")
        if file_path and not Path(file_path).exists():
            logger.debug(f"Cache file missing for {key}")
            return False

        return True

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was invalidated
        """
        if key not in self._index:
            return False

        entry = self._index.pop(key)
        file_path = entry.get("file_path")

        if file_path and Path(file_path).exists():
            try:
                Path(file_path).unlink()
            except OSError as e:
                logger.warning(f"Failed to delete cache file {file_path}: {e}")

        self._save_index()
        logger.info(f"Invalidated cache entry: {key}")
        return True

    def invalidate_symbol(self, symbol: str) -> int:
        """
        Invalidate all cache entries for a symbol.

        Args:
            symbol: Symbol to invalidate

        Returns:
            Number of entries invalidated
        """
        keys_to_invalidate = [
            k for k in self._index.keys()
            if symbol in k
        ]

        count = 0
        for key in keys_to_invalidate:
            if self.invalidate(key):
                count += 1

        return count

    def clear_expired(self) -> int:
        """
        Remove all expired cache entries.

        Returns:
            Number of entries cleared
        """
        now = datetime.now()
        expired_keys = []

        for key, entry in self._index.items():
            expires_at_str = entry.get("expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if now > expires_at:
                    expired_keys.append(key)

        count = 0
        for key in expired_keys:
            if self.invalidate(key):
                count += 1

        if count > 0:
            logger.info(f"Cleared {count} expired cache entries")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        total_size = 0
        by_type: Dict[str, int] = {}
        fresh_count = 0
        stale_count = 0

        for key, entry in self._index.items():
            size = entry.get("size_bytes", 0)
            total_size += size

            data_type = entry.get("data_type", "unknown")
            by_type[data_type] = by_type.get(data_type, 0) + 1

            if self._is_fresh(key, data_type):
                fresh_count += 1
            else:
                stale_count += 1

        return {
            "total_entries": len(self._index),
            "fresh_entries": fresh_count,
            "stale_entries": stale_count,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type,
            "cache_dir": str(self.cache_dir),
        }
