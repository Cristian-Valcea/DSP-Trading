"""
Databento OHLCV-1d Importer for Sleeve TSMOM (Cross-Asset Trend)

Converts Databento CSV+zstd OHLCV-1d files into per-root-symbol daily parquet
series suitable for the TSMOM backtest:
  - One parquet per root (e.g., MES, MCL, M6E)
  - DatetimeIndex (UTC, at 00:00:00) derived from ts_event
  - Columns: open, high, low, close, volume, contract

The Databento batch directory (job) is expected to contain:
  - many monthly files split by instrument, e.g.
      glbx-mdp3-20210105-20210131.ohlcv-1d.MESH1.csv.zst
  - optional spread instruments (contain '-') which we ignore
  - symbology.csv (recommended) mapping instrument_id <-> raw_symbol

Roll handling (pre-registered in spec v1.0)
------------------------------------------
We build a continuous per-root series by explicitly switching contracts using a
volume-led roll rule:
  - For each root on each date, define the "front" and "next" contracts as the
    two nearest expiries among contracts that have a bar on that date.
  - Compute V5 = 5-day moving average of daily volume for each contract.
  - If V5(next) > V5(front) for 3 consecutive trading days, roll at the close of
    the 3rd day (effective from the next trading day).
  - Guardrail: do not roll more than once per week per root.

This importer does NOT create a back-adjusted continuous price series. It
creates a *rolled* series with an explicit `contract` column. PnL calculations
should use contract-specific returns to avoid artificial roll gaps.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import zstandard as zstd

logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^glbx-mdp3-(?P<start>\d{8})-(?P<end>\d{8})\.ohlcv-1d\.(?P<inst>.+)\.csv\.zst$"
)

MONTH_CODE_TO_MONTH = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


@dataclass(frozen=True)
class ContractKey:
    root: str
    month: int
    year: int

    @property
    def sort_key(self) -> int:
        return self.year * 12 + self.month


def _parse_contract_symbol(symbol: str) -> Optional[ContractKey]:
    """
    Parse a futures contract symbol like MESH1, MCLZ4, M6EU5 into a sortable key.

    Assumptions:
    - Last char is the year digit (0-9) for years in the 2020s.
    - Second-to-last char is the month code.
    - Everything before that is the root symbol.
    """
    symbol = symbol.strip()
    if len(symbol) < 3:
        return None
    year_digit = symbol[-1]
    month_code = symbol[-2]
    root = symbol[:-2]
    if not year_digit.isdigit():
        return None
    if month_code not in MONTH_CODE_TO_MONTH:
        return None
    month = MONTH_CODE_TO_MONTH[month_code]
    year = 2020 + int(year_digit)  # valid for 2021-2026 window
    return ContractKey(root=root, month=month, year=year)


def _iter_ohlcv_files(input_dir: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yield (instrument_string, path) for all OHLCV-1d csv.zst files in the job dir.
    instrument_string is the part after ".ohlcv-1d." (e.g. MESH1 or MESH1-MESM1).
    """
    for p in input_dir.glob("*.csv.zst"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        yield m.group("inst"), p


def _read_csv_zst(path: Path) -> pd.DataFrame:
    """
    Read a Databento OHLCV-1d CSV+zstd file.
    Returns a DataFrame with UTC DatetimeIndex and required columns.
    """
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as f, dctx.stream_reader(f) as reader:
        raw = reader.read()

    df = pd.read_csv(
        io.BytesIO(raw),
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
    )
    if df.empty:
        return df

    # Databento "pretty_ts" yields ISO timestamps in UTC.
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("ts_event").sort_index()
    df["volume"] = df["volume"].astype("int64", errors="ignore")
    return df


def _collect_contract_dfs(
    input_dir: Path,
    roots: Sequence[str],
    start: Optional[date],
    end: Optional[date],
) -> Dict[str, pd.DataFrame]:
    """
    Read all outright contract files for the requested roots and return:
      {contract_symbol: daily_df}
    """
    roots_set = set(roots)
    parts: Dict[str, List[pd.DataFrame]] = {}

    for inst, path in _iter_ohlcv_files(input_dir):
        # Ignore spreads (contain '-') and other synthetic combos.
        if "-" in inst:
            continue
        key = _parse_contract_symbol(inst)
        if key is None:
            continue
        if key.root not in roots_set:
            continue

        df = _read_csv_zst(path)
        if df.empty:
            continue

        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz=timezone.utc)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz=timezone.utc)]
        if df.empty:
            continue

        parts.setdefault(inst, []).append(df)

    out: Dict[str, pd.DataFrame] = {}
    for contract, dfs in parts.items():
        df = pd.concat(dfs, axis=0).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[contract] = df
    return out


def _root_to_contracts(contract_dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    by_root: Dict[str, List[str]] = {}
    for contract in contract_dfs.keys():
        key = _parse_contract_symbol(contract)
        if key is None:
            continue
        by_root.setdefault(key.root, []).append(contract)

    # Sort each root's contracts by expiry month ordering.
    for root, contracts in by_root.items():
        contracts.sort(key=lambda c: _parse_contract_symbol(c).sort_key)  # type: ignore[union-attr]
    return by_root


def _trading_days_union(frames: Iterable[pd.DataFrame]) -> List[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for df in frames:
        dates.update(df.index)
    return sorted(dates)


def _roll_continuous_series_for_root(
    root: str,
    contracts: Sequence[str],
    contract_dfs: Dict[str, pd.DataFrame],
    min_v_days: int = 5,
    consec_days: int = 3,
    min_days_between_rolls: int = 5,
) -> pd.DataFrame:
    """
    Produce a rolled daily series for a given root symbol with a `contract` column.
    """
    # Pre-compute V5 (rolling average volume) per contract.
    v5: Dict[str, pd.Series] = {}
    for c in contracts:
        df = contract_dfs[c]
        v5[c] = df["volume"].rolling(min_v_days).mean()

    calendar = _trading_days_union(contract_dfs[c] for c in contracts)
    if not calendar:
        return pd.DataFrame()

    current: Optional[str] = None
    last_roll_day: Optional[pd.Timestamp] = None
    consec = 0
    pending_roll_to: Optional[str] = None

    rows = []

    for i, ts in enumerate(calendar):
        # Determine eligible contracts for today (have a bar).
        eligible = [c for c in contracts if ts in contract_dfs[c].index]
        if not eligible:
            continue

        # Sort eligible by expiry order and pick front/next.
        eligible_sorted = sorted(
            eligible, key=lambda c: _parse_contract_symbol(c).sort_key  # type: ignore[union-attr]
        )
        front = eligible_sorted[0]
        next_c = eligible_sorted[1] if len(eligible_sorted) > 1 else None

        # If we have a scheduled roll effective today, apply it.
        if pending_roll_to is not None:
            current = pending_roll_to
            pending_roll_to = None
            consec = 0

        # Ensure current is a valid contract for today.
        if current is None or ts not in contract_dfs[current].index:
            current = front
            consec = 0

        # Decide whether to increment roll trigger counter.
        can_roll = True
        if last_roll_day is not None:
            if (ts - last_roll_day).days < min_days_between_rolls:
                can_roll = False

        if can_roll and next_c is not None and current == front:
            v_front = v5[front].get(ts)
            v_next = v5[next_c].get(ts)
            if pd.notna(v_front) and pd.notna(v_next) and float(v_next) > float(v_front):
                consec += 1
            else:
                consec = 0
        else:
            consec = 0

        # Trigger roll at close of today (effective next trading day).
        if can_roll and next_c is not None and current == front and consec >= consec_days:
            if i + 1 < len(calendar):
                pending_roll_to = next_c
                last_roll_day = ts
            consec = 0

        # Emit today's bar from the current contract.
        bar = contract_dfs[current].loc[ts]
        rows.append(
            {
                "ts_event": ts,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": int(bar["volume"]),
                "contract": str(bar["symbol"]) if "symbol" in bar else current,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("ts_event").sort_index()
    out.index = out.index.tz_convert("UTC")
    return out


def build_rolled_series(
    input_dir: Path,
    roots: Sequence[str],
    start: Optional[date],
    end: Optional[date],
) -> Dict[str, pd.DataFrame]:
    contract_dfs = _collect_contract_dfs(input_dir=input_dir, roots=roots, start=start, end=end)
    by_root = _root_to_contracts(contract_dfs)

    rolled: Dict[str, pd.DataFrame] = {}
    for root in roots:
        contracts = by_root.get(root, [])
        if not contracts:
            logger.warning("No contracts found for root=%s", root)
            rolled[root] = pd.DataFrame()
            continue
        logger.info("Root=%s contracts=%d", root, len(contracts))
        rolled[root] = _roll_continuous_series_for_root(root=root, contracts=contracts, contract_dfs=contract_dfs)
    return rolled


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Databento OHLCV-1d CSV+zstd for TSMOM.")
    p.add_argument(
        "--input-dir",
        required=True,
        help="Databento batch download directory (contains *.csv.zst and symbology.csv).",
    )
    p.add_argument(
        "--roots",
        default="MES,MNQ,M2K,MYM,MGC,MCL,M6E,M6J",
        help="Comma-separated futures root symbols to build (default: spec v1.0 set).",
    )
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional).")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional).")
    p.add_argument(
        "--output-dir",
        default="data/tsmom",
        help="Output directory for rolled parquet series.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    logger.info("Importing Databento OHLCV-1d from %s", input_dir)
    rolled = build_rolled_series(input_dir=input_dir, roots=roots, start=start, end=end)

    for root, df in rolled.items():
        if df.empty:
            logger.warning("Root=%s produced empty series; skipping write", root)
            continue
        out_name = f"{root}_1d_{df.index.min().date()}_{df.index.max().date()}.parquet"
        out_path = output_dir / out_name
        df.to_parquet(out_path)
        logger.info("Wrote %s rows=%d", out_path, len(df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

