"""
Databento OHLCV-1m Importer for Sleeve ORB (MES/MNQ)

Converts Databento DBN+zstd OHLCV-1m files into the parquet format expected by
`dsp.backtest.orb_futures`:
  - One parquet file per base symbol (MES, MNQ)
  - DatetimeIndex (tz-aware UTC)
  - Columns: open, high, low, close, volume (+ optional contract)

This importer builds a continuous series from per-contract micro futures using
the pre-registered ORB roll schedule:
  - Quarterly contracts (H, M, U, Z)
  - Roll N=5 calendar days before the 3rd Friday of the expiration month
  - Additive back-adjustment to remove roll gaps

Input assumptions
-----------------
The Databento batch job is expected to be:
  - dataset: GLBX.MDP3
  - schema: ohlcv-1m
  - encoding: dbn
  - compression: zstd
  - split by instrument + month (recommended)

Files typically look like:
  glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.MESH2.dbn.zst
  glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.MNQZ4.dbn.zst
and may also include spread instruments (with '-') which we ignore.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pytz
import zstandard as zstd
from databento_dbn import DBNDecoder

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# Quarterly expiration months for equity index futures
QUARTERLY_MONTHS = [3, 6, 9, 12]  # H, M, U, Z
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

# Pre-registered roll rule (spec v1.6+)
DAYS_BEFORE_EXPIRY_ROLL = 5


@dataclass(frozen=True)
class ContractInfo:
    base_symbol: str  # "MES" or "MNQ"
    year: int
    month: int  # 3/6/9/12

    @property
    def ticker(self) -> str:
        code = MONTH_CODES[self.month]
        year_digit = self.year % 10
        return f"{self.base_symbol}{code}{year_digit}"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _third_friday(year: int, month: int) -> date:
    first_day = date(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7  # Monday=0
    first_friday = first_day + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)


def get_front_contract(base_symbol: str, as_of_date: date) -> ContractInfo:
    year = as_of_date.year
    month = as_of_date.month

    # next quarterly month
    for m in QUARTERLY_MONTHS:
        if m >= month:
            exp_month = m
            exp_year = year
            break
    else:
        exp_month = QUARTERLY_MONTHS[0]
        exp_year = year + 1

    third_friday = _third_friday(exp_year, exp_month)
    roll_date = third_friday - timedelta(days=DAYS_BEFORE_EXPIRY_ROLL)

    if as_of_date >= roll_date:
        idx = QUARTERLY_MONTHS.index(exp_month)
        if idx == len(QUARTERLY_MONTHS) - 1:
            exp_month = QUARTERLY_MONTHS[0]
            exp_year += 1
        else:
            exp_month = QUARTERLY_MONTHS[idx + 1]

    return ContractInfo(base_symbol=base_symbol, year=exp_year, month=exp_month)


def get_contract_schedule(
    base_symbol: str, start_date: date, end_date: date
) -> List[Tuple[ContractInfo, date, date]]:
    """
    Return [(contract, segment_start, segment_end)] where each segment is the
    front contract between roll dates (inclusive).
    """
    schedule: List[Tuple[ContractInfo, date, date]] = []
    current_date = start_date

    while current_date <= end_date:
        contract = get_front_contract(base_symbol, current_date)
        third_friday = _third_friday(contract.year, contract.month)
        roll_date = third_friday - timedelta(days=DAYS_BEFORE_EXPIRY_ROLL)
        segment_end = min(roll_date - timedelta(days=1), end_date)

        if segment_end >= current_date:
            schedule.append((contract, current_date, segment_end))

        current_date = roll_date

    return schedule


FILENAME_RE = re.compile(
    r"^glbx-mdp3-(?P<start>\d{8})-(?P<end>\d{8})\.ohlcv-1m\.(?P<inst>.+)\.dbn\.zst$"
)


def build_file_index(input_dir: Path) -> Dict[str, List[Path]]:
    """
    Map instrument string (e.g., 'MESH2') -> list of dbn.zst files.
    Spread instruments containing '-' are included in the index but should be filtered by caller.
    """
    index: Dict[str, List[Path]] = {}
    for p in input_dir.glob("*.dbn.zst"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        inst = m.group("inst")
        index.setdefault(inst, []).append(p)

    for inst, files in index.items():
        files.sort()
    return index


def read_dbn_ohlcv_1m(path: Path) -> pd.DataFrame:
    """
    Read a Databento DBN (zstd-compressed) OHLCV-1m file into a DataFrame.
    Returns a DataFrame indexed by UTC timestamp with columns:
      open, high, low, close, volume
    """
    dctx = zstd.ZstdDecompressor()
    dec = DBNDecoder(ts_out=True)

    with path.open("rb") as f, dctx.stream_reader(f) as reader:
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                break
            dec.write(chunk)

    recs = dec.decode()
    if not recs:
        return pd.DataFrame()

    rows = []
    for r in recs:
        # First element is Metadata; data records are OHLCVMsg
        if type(r).__name__ != "OHLCVMsg":
            continue
        rows.append(
            {
                # Databento DBN stores prices as fixed-point integers; use the decoded
                # "pretty_*" accessors (float) to get real price levels.
                "timestamp": r.pretty_ts_event if hasattr(r, "pretty_ts_event") else None,
                "ts_event": r.ts_event,
                "open": float(r.pretty_open) if hasattr(r, "pretty_open") else float(r.open),
                "high": float(r.pretty_high) if hasattr(r, "pretty_high") else float(r.high),
                "low": float(r.pretty_low) if hasattr(r, "pretty_low") else float(r.low),
                "close": float(r.pretty_close) if hasattr(r, "pretty_close") else float(r.close),
                "volume": int(r.volume),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Prefer vendor-decoded timestamps if available; fall back to ts_event.
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.drop(columns=["ts_event"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def filter_rth_et(df_utc: pd.DataFrame) -> pd.DataFrame:
    """Filter to RTH 09:30-16:00 ET using UTC-indexed bars."""
    if df_utc.empty:
        return df_utc
    idx_et = df_utc.index.tz_convert(ET)
    mask = (idx_et.time >= datetime.strptime("09:30", "%H:%M").time()) & (
        idx_et.time < datetime.strptime("16:00", "%H:%M").time()
    )
    return df_utc.loc[mask]


def slice_by_et_dates(df_utc: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Slice UTC-indexed bars by ET calendar dates (inclusive)."""
    if df_utc.empty:
        return df_utc
    idx_et = df_utc.index.tz_convert(ET)
    mask = (idx_et.date >= start) & (idx_et.date <= end)
    return df_utc.loc[mask]


def build_continuous_from_databento(
    base_symbol: str,
    file_index: Dict[str, List[Path]],
    start_date: date,
    end_date: date,
    rth_only: bool = True,
) -> pd.DataFrame:
    """
    Build a continuous, back-adjusted OHLCV-1m series for base_symbol over [start_date, end_date].
    """
    schedule = get_contract_schedule(base_symbol, start_date, end_date)

    all_parts: List[pd.DataFrame] = []
    cumulative_adjustment = 0.0
    prev_close: Optional[float] = None

    for contract, seg_start, seg_end in schedule:
        ticker = contract.ticker
        if "-" in ticker:
            continue

        files = file_index.get(ticker, [])
        if not files:
            logger.warning("No Databento files found for %s (segment %s -> %s)", ticker, seg_start, seg_end)
            continue

        # Load and concat all monthly parts for this contract
        contract_df = []
        for p in files:
            df = read_dbn_ohlcv_1m(p)
            if df.empty:
                continue
            contract_df.append(df)

        if not contract_df:
            logger.warning("No bars decoded for %s", ticker)
            continue

        df = pd.concat(contract_df).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Slice to ET date segment and optionally to RTH only
        df = slice_by_et_dates(df, seg_start, seg_end)
        if rth_only:
            df = filter_rth_et(df)
        if df.empty:
            logger.warning("No bars in segment after slicing for %s (%s -> %s)", ticker, seg_start, seg_end)
            continue

        # Additive back-adjustment to remove roll gaps
        if prev_close is not None:
            first_close = float(df.iloc[0]["close"])
            gap = first_close - prev_close
            cumulative_adjustment += gap

        if cumulative_adjustment != 0.0:
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] - cumulative_adjustment

        df["contract"] = ticker
        all_parts.append(df)
        prev_close = float(df.iloc[-1]["close"])

    if not all_parts:
        return pd.DataFrame()

    out = pd.concat(all_parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Final slice to requested range in case of any drift
    out = slice_by_et_dates(out, start_date, end_date)
    if rth_only:
        out = filter_rth_et(out)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Databento OHLCV-1m for ORB (MES/MNQ)")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Databento batch download directory containing *.dbn.zst files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/orb",
        help="Output directory for ORB parquet files (default: data/orb)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="MES,MNQ",
        help="Comma-separated base symbols to build (default: MES,MNQ)",
    )
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2025-03-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--rth-only",
        action="store_true",
        default=True,
        help="Keep only RTH bars (09:30-16:00 ET). Default: true.",
    )
    parser.add_argument(
        "--include-eth",
        action="store_true",
        default=False,
        help="Keep all bars (RTH+ETH). Overrides --rth-only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output parquet files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    start = parse_date(args.start)
    end = parse_date(args.end)
    if start > end:
        raise ValueError("start must be <= end")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    rth_only = (not args.include_eth) and args.rth_only

    logger.info("Indexing Databento files in %s", input_dir)
    file_index = build_file_index(input_dir)
    logger.info("Indexed %d instruments", len(file_index))

    for sym in symbols:
        out_path = output_dir / f"{sym}_1min_{start}_{end}.parquet"
        if out_path.exists() and not args.overwrite:
            logger.info("Skipping %s (exists): %s", sym, out_path)
            continue

        logger.info("Building continuous series for %s (%s -> %s), rth_only=%s", sym, start, end, rth_only)
        df = build_continuous_from_databento(
            base_symbol=sym,
            file_index=file_index,
            start_date=start,
            end_date=end,
            rth_only=rth_only,
        )
        if df.empty:
            raise RuntimeError(f"No data produced for {sym}. Check input files and date range.")

        df.to_parquet(out_path)
        logger.info("Wrote %s rows to %s", len(df), out_path)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
