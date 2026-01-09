"""
Fetch daily OHLCV series from Polygon.io for eq/futures needed by VRP NN.

Usage:
  python -m dsp.data.equity_daily_fetcher --symbols SPY,QQQ,TLT --start 2010-01-01 --end 2026-01-05

Outputs per-symbol parquet to data/vrp/equities/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_ENDPOINT = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"


def polygon_key() -> str:
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY must be set")
    return key


def fetch_daily(ticker: str, start: date, end: date, limit: int = 5000) -> pd.DataFrame:
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": polygon_key(),
    }
    url = API_ENDPOINT.format(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        return pd.DataFrame()
    records = []
    for row in data["results"]:
        records.append(
            {
                "timestamp": pd.to_datetime(row["t"], unit="ms", utc=True),
                "open": row["o"],
                "high": row["h"],
                "low": row["l"],
                "close": row["c"],
                "volume": row["v"],
            }
        )
    df = pd.DataFrame(records).set_index("timestamp")
    return df


def persist(df: pd.DataFrame, symbol: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}_daily.parquet"
    df.to_parquet(path)
    logger.info("Wrote %s rows to %s", len(df), path)
    return path


def run_fetcher(symbols: List[str], start: date, end: date, output_dir: Path) -> None:
    for symbol in symbols:
        logger.info("Fetching daily bars for %s %s to %s", symbol, start, end)
        df = fetch_daily(symbol, start, end)
        if df.empty:
            logger.warning("No data returned for %s", symbol)
            continue
        persist(df, symbol, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily data from Polygon.")
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers.")
    parser.add_argument("--start", default="2010-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"), help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/vrp/equities", help="Output directory.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    run_fetcher(symbols, start, end, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
