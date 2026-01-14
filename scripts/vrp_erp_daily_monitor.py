#!/usr/bin/env python3
"""
VRP-ERP Daily Position Monitor

Daily monitoring script for VRP-ERP paper trading.
Tracks VIX regime, SPY position, drift, and rebalancing signals.

Usage:
    python scripts/vrp_erp_daily_monitor.py
    python scripts/vrp_erp_daily_monitor.py --live  # Fetch live quotes from IBKR

Output:
    Prints formatted status report
    Appends to data/vrp/paper_trading/vrp_erp_log.csv
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsp.risk.vol_target_overlay import VolTargetOverlay, VolTargetOverlayConfig


@dataclass
class ERPConfig:
    """VRP-ERP configuration."""
    # Base allocation
    base_allocation: float = 10000.0  # $10k for paper trading

    # VIX regime thresholds
    calm_threshold: float = 15.0
    elevated_threshold: float = 20.0
    high_threshold: float = 30.0

    # Regime multipliers
    regime_multipliers: dict = None

    # Rebalancing rules
    min_days_in_regime: int = 2
    drift_threshold: float = 0.05  # 5%
    vix_buffer: float = 0.5  # Buffer at regime boundaries

    def __post_init__(self):
        if self.regime_multipliers is None:
            self.regime_multipliers = {
                "CALM": 1.00,
                "ELEVATED": 0.75,
                "HIGH": 0.50,
                "CRISIS": 0.25,
            }


@dataclass
class RegimeState:
    """Current regime state tracking."""
    regime: str
    vix: float
    days_in_regime: int = 1
    last_transition_date: Optional[date] = None

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "vix": self.vix,
            "days_in_regime": self.days_in_regime,
            "last_transition_date": self.last_transition_date.isoformat() if self.last_transition_date else None,
        }


@dataclass
class SPYPosition:
    """Current SPY position."""
    shares: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)


def classify_regime(vix: float, config: ERPConfig) -> str:
    """Classify VIX regime."""
    if vix < config.calm_threshold:
        return "CALM"
    elif vix < config.elevated_threshold:
        return "ELEVATED"
    elif vix < config.high_threshold:
        return "HIGH"
    else:
        return "CRISIS"


def calculate_target_position(
    vix: float,
    spy_price: float,
    config: ERPConfig,
    vol_mult: float = 1.0,
) -> Tuple[str, float, int, float, float]:
    """Calculate target SPY position based on VIX regime and vol-target overlay.

    Args:
        vix: Current VIX spot value
        spy_price: Current SPY price
        config: ERP configuration
        vol_mult: Vol-targeting overlay multiplier (from VolTargetOverlay)

    Returns:
        Tuple of (regime, target_value, target_shares, regime_multiplier, vol_mult)
    """
    regime = classify_regime(vix, config)
    regime_mult = config.regime_multipliers[regime]
    # Per SPEC_VOL_TARGET_OVERLAY.md Section 6.1: apply vol_mult to SPY target shares
    combined_mult = regime_mult * vol_mult
    target_value = config.base_allocation * combined_mult
    target_shares = int(target_value / spy_price)

    return regime, target_value, target_shares, regime_mult, vol_mult


def calculate_drift(current_shares: int, target_shares: int) -> float:
    """Calculate drift percentage."""
    if target_shares == 0:
        return 0.0 if current_shares == 0 else 1.0
    return abs(current_shares - target_shares) / target_shares


def load_regime_state(state_path: Path) -> Optional[RegimeState]:
    """Load previous regime state from file."""
    if not state_path.exists():
        return None

    with open(state_path) as f:
        data = json.load(f)

    return RegimeState(
        regime=data["regime"],
        vix=data["vix"],
        days_in_regime=data["days_in_regime"],
        last_transition_date=date.fromisoformat(data["last_transition_date"]) if data.get("last_transition_date") else None,
    )


def save_regime_state(state: RegimeState, state_path: Path):
    """Save regime state to file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, 'w') as f:
        json.dump(state.to_dict(), f, indent=2)


def get_manual_quotes() -> Tuple[float, float]:
    """Prompt user for manual quote entry."""
    print("\n--- Enter Current Market Data ---")

    try:
        vix = float(input("VIX Spot: "))
        spy_price = float(input("SPY Price: "))
    except (ValueError, EOFError):
        print("Invalid input. Using placeholder values.")
        return 14.8, 595.0

    return vix, spy_price


def get_vix_from_state() -> Optional[float]:
    """Try to load VIX from the regime state file (updated by digest/monitor)."""
    state_path = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "vrp_erp_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                data = json.load(f)
            vix = data.get("vix")
            if vix and float(vix) > 0:
                return float(vix)
        except Exception:
            pass
    return None


def get_live_quotes_ibkr(non_interactive: bool = False) -> Optional[Tuple[float, float, SPYPosition]]:
    """Fetch live quotes and position from IBKR.

    Args:
        non_interactive: If True, avoid prompting for input (use state file for VIX).
    """
    try:
        from ib_insync import IB, Stock
        import math

        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=201, timeout=10)

        # Request SPY quotes
        spy_contract = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(spy_contract)
        [ticker] = ib.reqTickers(spy_contract)

        # Try multiple price sources (market may be closed)
        spy_price = None
        if ticker.midpoint() and not math.isnan(ticker.midpoint()):
            spy_price = ticker.midpoint()
        elif ticker.last and not math.isnan(ticker.last):
            spy_price = ticker.last
        elif ticker.close and not math.isnan(ticker.close):
            spy_price = ticker.close

        # Get current position - aggregate all SPY lots
        positions = ib.positions()
        spy_pos = SPYPosition()
        total_shares = 0
        total_cost = 0.0
        for pos in positions:
            if pos.contract.symbol == 'SPY' and pos.contract.secType == 'STK':
                shares = int(pos.position)
                total_shares += shares
                # Weighted average cost calculation
                if shares != 0:
                    total_cost += shares * pos.avgCost
        spy_pos.shares = total_shares
        if total_shares != 0:
            spy_pos.avg_cost = total_cost / total_shares
        if spy_price:
            spy_pos.current_price = spy_price

        ib.disconnect()

        # If no valid price, fall back to manual entry or state
        if spy_price is None or math.isnan(spy_price):
            print("\n⚠️  Market closed - no live SPY quote available.")
            print(f"SPY Position: {spy_pos.shares} shares")
            if non_interactive:
                print("❌ Cannot proceed in non-interactive mode without SPY price.")
                return None
            print("\n--- Enter Current Market Data ---")
            spy_price = float(input("SPY Price (check Yahoo Finance): "))
            vix = float(input("VIX Spot: "))
        else:
            print(f"\nSPY Price: ${spy_price:.2f}")
            print(f"SPY Position: {spy_pos.shares} shares")

            # Try to get VIX from state file first (for non-interactive mode)
            state_vix = get_vix_from_state()
            if non_interactive:
                if state_vix:
                    print(f"VIX (from state): {state_vix:.2f}")
                    vix = state_vix
                else:
                    print("❌ Cannot proceed in non-interactive mode without VIX in state file.")
                    return None
            else:
                if state_vix:
                    vix_input = input(f"Enter current VIX [{state_vix:.2f}]: ").strip()
                    vix = float(vix_input) if vix_input else state_vix
                else:
                    vix = float(input("Enter current VIX: "))

        spy_pos.current_price = spy_price
        return vix, spy_price, spy_pos

    except Exception as e:
        print(f"IBKR connection failed: {e}")
        return None


def check_rebalance_signal(
    current_regime: str,
    previous_state: Optional[RegimeState],
    drift: float,
    config: ERPConfig,
) -> Tuple[bool, str]:
    """Check if rebalancing is needed.

    Returns:
        Tuple of (should_rebalance, reason)
    """
    reasons = []

    # Check regime change with 2-day confirmation
    if previous_state:
        if current_regime != previous_state.regime:
            if previous_state.days_in_regime >= config.min_days_in_regime:
                # Regime held long enough, transition confirmed
                reasons.append(f"Regime changed: {previous_state.regime} → {current_regime} (confirmed)")
            else:
                # Still in transition period
                return False, f"Regime changing ({previous_state.days_in_regime}/{config.min_days_in_regime} days)"

    # Check drift threshold
    if drift > config.drift_threshold:
        reasons.append(f"Drift {drift:.1%} exceeds {config.drift_threshold:.0%} threshold")

    if reasons:
        return True, "; ".join(reasons)

    return False, "No rebalancing needed"


def print_report(
    vix: float,
    spy_price: float,
    current_position: SPYPosition,
    config: ERPConfig,
    previous_state: Optional[RegimeState],
    vol_mult: float = 1.0,
):
    """Print formatted monitoring report."""
    regime, target_value, target_shares, regime_mult, vol_mult = calculate_target_position(
        vix, spy_price, config, vol_mult
    )
    drift = calculate_drift(current_position.shares, target_shares)

    # Check rebalance signal
    should_rebalance, rebalance_reason = check_rebalance_signal(
        regime, previous_state, drift, config
    )

    print("\n" + "=" * 60)
    print("  VRP-ERP DAILY MONITORING REPORT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # VIX Regime
    print("\n--- VIX REGIME ---")
    print(f"VIX Spot:    {vix:.2f}")
    print(f"Regime:      {regime} ({regime_mult:.0%} regime exposure)")

    if previous_state:
        if regime == previous_state.regime:
            print(f"Days in Regime: {previous_state.days_in_regime + 1}")
        else:
            print(f"Regime Change: {previous_state.regime} → {regime}")

    # Regime Thresholds Reference
    print(f"\n  Thresholds: CALM (<{config.calm_threshold}), ELEVATED ({config.calm_threshold}-{config.elevated_threshold}), HIGH ({config.elevated_threshold}-{config.high_threshold}), CRISIS (>{config.high_threshold})")

    # SPY Position
    print("\n--- SPY POSITION ---")
    print(f"SPY Price:       ${spy_price:.2f}")
    print(f"Current Shares:  {current_position.shares}")
    print(f"Current Value:   ${current_position.market_value:,.2f}")

    if current_position.shares > 0 and current_position.avg_cost > 0:
        print(f"Avg Cost:        ${current_position.avg_cost:.2f}")
        print(f"Unrealized P&L:  ${current_position.pnl:+,.2f}")

    # Vol-Targeting Overlay
    print("\n--- VOL-TARGET OVERLAY ---")
    print(f"Vol Multiplier:  {vol_mult:.2f} (from realized SPY vol)")
    print(f"Combined Mult:   {regime_mult * vol_mult:.2f} (regime × vol_target)")

    # Target vs Actual
    print("\n--- TARGET vs ACTUAL ---")
    print(f"Base Allocation: ${config.base_allocation:,.2f}")
    print(f"Target Value:    ${target_value:,.2f} ({regime_mult * vol_mult:.0%} combined)")
    print(f"Target Shares:   {target_shares}")
    print(f"Actual Shares:   {current_position.shares}")
    print(f"Drift:           {drift:.1%}")

    # Delta needed
    delta_shares = target_shares - current_position.shares
    if delta_shares != 0:
        action = "BUY" if delta_shares > 0 else "SELL"
        print(f"\nDelta:           {action} {abs(delta_shares)} shares")

    # Rebalancing Signal
    print("\n--- REBALANCING ---")
    if should_rebalance:
        print(f"SIGNAL: REBALANCE NEEDED")
        print(f"Reason: {rebalance_reason}")
        if delta_shares != 0:
            print(f"\n  >>> {action} {abs(delta_shares)} SPY shares <<<")
    else:
        print(f"Status: {rebalance_reason}")

    # Action Summary
    print("\n--- ACTION REQUIRED ---")
    if should_rebalance and delta_shares != 0:
        action_list = [
            f"{action} {abs(delta_shares)} SPY shares",
            "Use LMT order at MID price (SPY is liquid - avoid systematic overpay)",
            "Best timing: 09:35-10:15 ET or 15:30-15:45 ET",
        ]
        for item in action_list:
            print(f"  -> {item}")
    else:
        print("  None - Position is on target")

    print("\n" + "=" * 60)

    return regime, target_shares, drift, should_rebalance


def append_to_log(
    vix: float,
    spy_price: float,
    regime: str,
    target_shares: int,
    actual_shares: int,
    drift: float,
    should_rebalance: bool,
    log_path: Path,
    fill_price: Optional[float] = None,
    avg_cost: Optional[float] = None,
):
    """Append daily record to CSV log.

    Args:
        fill_price: Actual fill price if trade executed (for Day 1 / rebalance days)
        avg_cost: Average cost basis from IBKR position
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create header if file doesn't exist
    write_header = not log_path.exists()

    # Calculate actual dollar exposure
    actual_exposure = actual_shares * spy_price if spy_price > 0 else 0.0

    with open(log_path, 'a') as f:
        if write_header:
            f.write("date,time,vix,spy_price,regime,target_shares,actual_shares,")
            f.write("actual_exposure_usd,avg_cost,fill_price,drift_pct,rebalance_signal,action_taken,notes\n")

        row = [
            date.today().isoformat(),
            datetime.now().strftime("%H:%M:%S"),
            f"{vix:.2f}",
            f"{spy_price:.2f}",
            regime,
            str(target_shares),
            str(actual_shares),
            f"{actual_exposure:.2f}",
            f"{avg_cost:.2f}" if avg_cost else "",
            f"{fill_price:.2f}" if fill_price else "",
            f"{drift:.2%}",
            str(should_rebalance),
            "",  # action_taken - to be filled manually
            "",  # notes
        ]
        f.write(",".join(row) + "\n")

    print(f"\nLog appended to: {log_path}")


def get_vol_multiplier() -> float:
    """Get the current vol-targeting overlay multiplier.

    Uses persisted state from VolTargetOverlay. If state doesn't exist,
    returns 1.0 (no scaling).
    """
    def _compute_from_local_spy() -> float:
        """
        Fallback: compute multiplier from local SPY daily parquet.

        This prevents "silent no-vol-target" if orchestrator hasn't written state yet.
        Matches SPEC_VOL_TARGET_OVERLAY.md v1.0 defaults:
        - 21 trading day lookback
        - 10% target vol
        - bounds [0.25, 1.50]
        """
        spy_path = PROJECT_ROOT / "data" / "vrp" / "equities" / "SPY_daily.parquet"
        if not spy_path.exists():
            return 1.0

        df = pd.read_parquet(spy_path)
        if df is None or df.empty or "close" not in df.columns:
            return 1.0

        closes = df["close"].astype(float).dropna()
        if len(closes) < 30:
            return 1.0

        # Use last 21 trading days of log returns
        log_rets = (closes / closes.shift(1)).apply(lambda x: float("nan") if x <= 0 else np.log(x)).dropna()
        lookback = log_rets.tail(21)
        if len(lookback) < 10:
            return 1.0

        realized_vol = float(lookback.std() * np.sqrt(252)) if lookback.std() and lookback.std() > 0 else 0.0
        if realized_vol <= 0:
            return 1.0

        target_vol = 0.10
        raw = target_vol / realized_vol
        return float(min(1.50, max(0.25, raw)))

    try:
        # Read persisted multiplier directly from state file
        state_path = PROJECT_ROOT / "data" / "vol_target_overlay_state.json"
        if state_path.exists():
            with open(state_path) as f:
                data = json.load(f)
            mult = float(data.get("last_multiplier", 1.0))
            print(f"📊 Vol-Target Overlay: loaded multiplier {mult:.2f} from state")
            return mult
        else:
            mult = _compute_from_local_spy()
            print(f"📊 Vol-Target Overlay: no state file, computed multiplier {mult:.2f} from local SPY")
            return mult
    except Exception as e:
        mult = _compute_from_local_spy()
        print(f"⚠️  Vol-Target Overlay: error loading state ({e}), using local SPY multiplier {mult:.2f}")
        return mult


def main():
    parser = argparse.ArgumentParser(description="VRP-ERP Daily Position Monitor")
    parser.add_argument('--live', action='store_true', help="Fetch live quotes from IBKR")
    parser.add_argument('--base', type=float, default=10000, help="Base allocation (default: $10,000)")
    parser.add_argument('--no-log', action='store_true', help="Skip appending to daily log")
    parser.add_argument('--no-vol-target', action='store_true', help="Disable vol-targeting overlay")
    parser.add_argument('--non-interactive', action='store_true', help="Non-interactive mode (use state file for VIX, fail if unavailable)")
    # Manual value arguments (for non-interactive use, e.g., from UI)
    parser.add_argument('--vix', type=float, help="Manual VIX spot value")
    parser.add_argument('--spy', type=float, help="Manual SPY price")
    parser.add_argument('--shares', type=int, default=0, help="Current SPY shares held")
    args = parser.parse_args()

    # Initialize config
    config = ERPConfig(base_allocation=args.base)

    # Load previous state
    state_path = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "vrp_erp_state.json"
    previous_state = load_regime_state(state_path)

    # Get vol-targeting multiplier (from shared state with orchestrator)
    vol_mult = 1.0 if args.no_vol_target else get_vol_multiplier()

    # Get market data - priority: CLI args > live > interactive prompt
    current_position = SPYPosition()

    if args.vix is not None and args.spy is not None:
        # Manual values provided via CLI
        print(f"Using provided values: VIX={args.vix}, SPY={args.spy}, Shares={args.shares}")
        vix, spy_price = args.vix, args.spy
        current_position.shares = args.shares
        current_position.current_price = spy_price
    elif args.live:
        print("Fetching live quotes from IBKR...")
        result = get_live_quotes_ibkr(non_interactive=args.non_interactive)
        if result:
            vix, spy_price, current_position = result
        else:
            if args.non_interactive:
                print("❌ Failed to get live quotes in non-interactive mode. Exiting.")
                sys.exit(1)
            print("Failed to get live quotes. Falling back to manual entry.")
            vix, spy_price = get_manual_quotes()
            current_position.shares = int(input("Current SPY shares (0 if none): ") or "0")
    else:
        if args.non_interactive:
            print("❌ Non-interactive mode requires --live or explicit --vix/--spy values.")
            sys.exit(1)
        vix, spy_price = get_manual_quotes()
        current_position.shares = int(input("Current SPY shares (0 if none): ") or "0")

    current_position.current_price = spy_price

    # Print report
    regime, target_shares, drift, should_rebalance = print_report(
        vix, spy_price, current_position, config, previous_state, vol_mult
    )

    # Update regime state
    if previous_state and regime == previous_state.regime:
        new_state = RegimeState(
            regime=regime,
            vix=vix,
            days_in_regime=previous_state.days_in_regime + 1,
            last_transition_date=previous_state.last_transition_date,
        )
    else:
        new_state = RegimeState(
            regime=regime,
            vix=vix,
            days_in_regime=1,
            last_transition_date=date.today(),
        )

    save_regime_state(new_state, state_path)

    # Append to daily log
    if not args.no_log:
        log_path = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "vrp_erp_log.csv"
        append_to_log(
            vix, spy_price, regime, target_shares,
            current_position.shares, drift, should_rebalance, log_path,
            fill_price=None,  # Fill manually after trade execution
            avg_cost=current_position.avg_cost if current_position.avg_cost > 0 else None,
        )


if __name__ == "__main__":
    main()
