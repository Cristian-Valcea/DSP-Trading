#!/usr/bin/env python3
"""
Fetch foreign short-rate series from FRED for the ETF carry sleeve.

This script downloads FRED series via the public fredgraph CSV endpoint
(no API key required) and writes both per-series and combined parquet files.

Default output:
  dsp100k/data/carry/rates/
    IR3TIB01EZM156N.parquet
    IR3TIB01JPM156N.parquet
    IR3TIB01GBM156N.parquet
    IR3TIB01AUM156N.parquet
    carry_rates.parquet

Usage:
  python dsp100k/scripts/fetch_carry_rates_fred.py --start 2010-01-01 --end 2026-01-10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    column_name: str


DEFAULT_SERIES: List[SeriesSpec] = [
    SeriesSpec("IR3TIB01EZM156N", "eur_3m"),
    SeriesSpec("IR3TIB01JPM156N", "jpy_3m"),
    SeriesSpec("IR3TIB01GBM156N", "gbp_3m"),
    SeriesSpec("IR3TIB01AUM156N", "aud_3m"),
]


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def fetch_fred_series(series_id: str) -> pd.Series:
    """
    Fetch a FRED series as a pandas Series indexed by date.
    Uses the fredgraph CSV endpoint and treats '.' as missing.
    """
    resp = requests.get(FREDGRAPH_URL, params={"id": series_id}, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(resp.text))

    # FRED CSV headers vary by endpoint/version. Common forms:
    # - DATE,<series_id>
    # - observation_date,<series_id>
    date_col = None
    for candidate in ("DATE", "observation_date"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None or series_id not in df.columns:
        raise ValueError(f"Unexpected FRED CSV schema for {series_id}: {df.columns.tolist()}")

    value_col = series_id
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert(None)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    s = df.set_index(date_col)[value_col].sort_index()
    s.name = series_id
    return s


def build_daily_frame(
    series_map: Dict[str, pd.Series],
    series_specs: Iterable[SeriesSpec],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Build a daily dataframe (calendar-day) with forward-filled rates.
    """
    idx = pd.date_range(start=start.isoformat(), end=end.isoformat(), freq="D")
    out = pd.DataFrame(index=idx)

    for spec in series_specs:
        s = series_map[spec.series_id]
        out[spec.column_name] = s.reindex(idx).ffill()

    return out


def write_outputs(
    out_dir: Path,
    series_map: Dict[str, pd.Series],
    series_specs: List[SeriesSpec],
    daily: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-series outputs (original frequency, as published by FRED)
    for spec in series_specs:
        df = series_map[spec.series_id].to_frame(name=spec.column_name)
        df.to_parquet(out_dir / f"{spec.series_id}.parquet")

    # Combined daily output (forward-filled)
    daily.to_parquet(out_dir / "carry_rates.parquet")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch foreign short-rate series for carry sleeve (FRED).")
    parser.add_argument("--start", type=_parse_iso_date, default=date(2010, 1, 1))
    parser.add_argument("--end", type=_parse_iso_date, default=date.today())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dsp100k/data/carry/rates"),
        help="Output directory for parquet files",
    )
    args = parser.parse_args(argv)

    series_map: Dict[str, pd.Series] = {}
    for spec in DEFAULT_SERIES:
        series_map[spec.series_id] = fetch_fred_series(spec.series_id)

    daily = build_daily_frame(series_map, DEFAULT_SERIES, args.start, args.end)
    write_outputs(args.out_dir, series_map, DEFAULT_SERIES, daily)

    print(f"Wrote carry rate series to: {args.out_dir}")
    print("Files:")
    for spec in DEFAULT_SERIES:
        print(f"  - {spec.series_id}.parquet ({spec.column_name})")
    print("  - carry_rates.parquet (daily, forward-filled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
