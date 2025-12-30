"""
Logging configuration for DSP-100K.

Provides structured logging with JSON output for audit trail.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class AuditLogger:
    """
    Audit logger for decision tracking.

    Writes to JSONL file for reconstructibility.
    """

    def __init__(self, log_dir: Path):
        # Keep audit logs under a dedicated subdirectory so they're easy to ignore
        # in git and easy to rotate/ship separately from app logs.
        self.log_dir = log_dir if log_dir.name == "audit" else (log_dir / "audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        self.log_path = self.log_dir / f"audit_{date_str}.jsonl"

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an audit event."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **data,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()  # Immediate flush for crash safety

    def log_signal(
        self,
        sleeve: str,
        symbol: str,
        signal: float,
        score: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log a signal generation event."""
        self.log(
            "SIGNAL",
            {
                "sleeve": sleeve,
                "symbol": symbol,
                "signal": signal,
                "score": score,
                "reason": reason,
            },
        )

    def log_order(
        self,
        sleeve: str,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log an order intent event."""
        self.log(
            "ORDER_INTENT",
            {
                "sleeve": sleeve,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "limit_price": limit_price,
                "reason": reason,
            },
        )

    def log_fill(
        self,
        sleeve: str,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        commission: float,
        slippage_bps: Optional[float] = None,
    ) -> None:
        """Log a fill event."""
        self.log(
            "FILL",
            {
                "sleeve": sleeve,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "commission": commission,
                "slippage_bps": slippage_bps,
            },
        )

    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """Log a risk management event."""
        self.log(
            f"RISK_{event_type}",
            details,
        )

    def log_suppression(
        self,
        sleeve: str,
        symbol: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log when a trade is suppressed."""
        self.log(
            "TRADE_SUPPRESSED",
            {
                "sleeve": sleeve,
                "symbol": symbol,
                "reason": reason,
                "details": details or {},
            },
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        log_dir = Path(os.getenv("DSP_LOG_DIR", "logs"))
        _audit_logger = AuditLogger(log_dir)
    return _audit_logger


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (default: logs/)
        json_format: Use JSON formatting for logs
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create log directory
    log_path = Path(log_dir or os.getenv("DSP_LOG_DIR", "logs"))
    log_path.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(console_handler)

    # File handler (always JSON for parsing)
    date_str = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(log_path / f"dsp_{date_str}.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Initialize audit logger
    global _audit_logger
    _audit_logger = AuditLogger(log_path)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
