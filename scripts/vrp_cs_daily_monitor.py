#!/usr/bin/env python3
"""
VRP-CS Daily Position Monitor

Daily monitoring script for VRP Calendar Spread paper trading.
Tracks spread P&L, exit triggers, regime gate, and roll dates.

Usage:
    python scripts/vrp_cs_daily_monitor.py
    python scripts/vrp_cs_daily_monitor.py --live  # Fetch live quotes from IBKR

Output:
    Prints formatted status report
    Appends to data/vrp/paper_trading/daily_log.csv
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class PositionConfig:
    """Current VRP-CS position configuration."""
    # Contract identifiers
    short_contract: str = "VXMF6"  # Front month (short)
    long_contract: str = "VXMG6"   # Back month (long)

    # Entry prices (from 2026-01-09)
    short_entry_price: float = 16.24
    long_entry_price: float = 17.87

    # Entry metadata
    entry_date: str = "2026-01-09"
    entry_spread: float = 1.63  # VX2 - VX1 = 17.87 - 16.24

    # Exit triggers (from spec)
    stop_loss_spread: float = 2.04   # 25% wider than entry
    take_profit_spread: float = 0.82  # 50% narrower than entry

    # Contract specs
    multiplier: float = 100.0  # VXM = $100 per point
    position_size: int = 1     # 1 spread
    quantity: int = 1          # Alias for position_size (from config JSON)

    # Roll schedule
    vx1_expiry: str = "2026-01-21"  # Jan expiry (Wednesday)
    roll_by_date: str = "2026-01-14"  # 5 trading days before


@dataclass
class MarketData:
    """Current market data snapshot."""
    vix: float
    vx1: float  # Front month price
    vx2: float  # Back month price
    vvix: Optional[float] = None
    timestamp: Optional[datetime] = None

    @property
    def spread(self) -> float:
        """Calendar spread = VX2 - VX1"""
        return self.vx2 - self.vx1

    @property
    def contango_pct(self) -> float:
        """Contango percentage."""
        if self.vix == 0:
            return 0.0
        return 100.0 * (self.vx1 - self.vix) / self.vix


def load_position_config(config_path: Optional[Path] = None) -> PositionConfig:
    """Load position config from file or use defaults."""
    default_path = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "position_config.json"
    config_path = config_path or default_path

    if config_path and config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        return PositionConfig(**data)
    return PositionConfig()


def get_manual_quotes() -> MarketData:
    """Prompt user for manual quote entry."""
    print("\n--- Enter Current Market Data ---")
    print("(Enter current prices from TWS or CBOE website)")

    try:
        vix = float(input("VIX Spot: "))
        vx1 = float(input("VX1 (Front Month) Price: "))
        vx2 = float(input("VX2 (Back Month) Price: "))
        vvix_str = input("VVIX (optional, press Enter to skip): ").strip()
        vvix = float(vvix_str) if vvix_str else None
    except (ValueError, EOFError):
        print("Invalid input. Using placeholder values.")
        return MarketData(vix=15.0, vx1=16.25, vx2=17.83, vvix=85.0)

    return MarketData(
        vix=vix,
        vx1=vx1,
        vx2=vx2,
        vvix=vvix,
        timestamp=datetime.now()
    )


def get_live_quotes_ibkr() -> Optional[MarketData]:
    """Fetch live quotes from IBKR (requires connection)."""
    try:
        from ib_insync import IB, Future

        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=200, timeout=10)

        # Request VIX index (not always available)
        # For VX futures, use CFE exchange
        vxm_f6 = Future(symbol='VXM', lastTradeDateOrContractMonth='202601', exchange='CFE')
        vxm_g6 = Future(symbol='VXM', lastTradeDateOrContractMonth='202602', exchange='CFE')

        ib.qualifyContracts(vxm_f6, vxm_g6)

        # Get market data
        [ticker_f6] = ib.reqTickers(vxm_f6)
        [ticker_g6] = ib.reqTickers(vxm_g6)

        vx1 = ticker_f6.midpoint() if ticker_f6.midpoint() else ticker_f6.last
        vx2 = ticker_g6.midpoint() if ticker_g6.midpoint() else ticker_g6.last

        ib.disconnect()

        # Approximate VIX from VX1 (VX1 typically 0.5-2 points above VIX in contango)
        vix_approx = vx1 - 1.0

        return MarketData(
            vix=vix_approx,
            vx1=vx1,
            vx2=vx2,
            timestamp=datetime.now()
        )
    except Exception as e:
        print(f"IBKR connection failed: {e}")
        return None


def calculate_pnl(config: PositionConfig, market: MarketData) -> dict:
    """Calculate P&L for current position."""
    # Spread P&L: We're long VX2, short VX1
    # Entry: bought VX2 at 17.87, sold VX1 at 16.24
    # Current: VX2 at market.vx2, VX1 at market.vx1

    vx1_pnl = (config.short_entry_price - market.vx1) * config.multiplier * config.position_size
    vx2_pnl = (market.vx2 - config.long_entry_price) * config.multiplier * config.position_size
    total_pnl = vx1_pnl + vx2_pnl

    # Alternative calculation via spread change
    entry_spread = config.entry_spread  # 1.63
    current_spread = market.spread
    spread_change = current_spread - entry_spread
    spread_pnl = -spread_change * config.multiplier * config.position_size  # Negative because we want spread to narrow

    return {
        "vx1_pnl": vx1_pnl,
        "vx2_pnl": vx2_pnl,
        "total_pnl": total_pnl,
        "spread_pnl": spread_pnl,
        "entry_spread": entry_spread,
        "current_spread": current_spread,
        "spread_change": spread_change,
    }


def check_exit_triggers(config: PositionConfig, market: MarketData) -> dict:
    """Check if any exit triggers are hit.

    For VRP calendar spread (short front, long back):
    - We PROFIT when spread WIDENS (front drops faster than back)
    - We LOSE when spread NARROWS (front rises faster than back)

    Therefore:
    - Stop-loss: spread narrows below threshold (loss)
    - Take-profit: spread widens above threshold (profit)
    """
    current_spread = market.spread

    # Stop-loss when spread NARROWS (we're losing)
    stop_loss_hit = current_spread <= config.stop_loss_spread
    # Take-profit when spread WIDENS (we're profiting)
    take_profit_hit = current_spread >= config.take_profit_spread

    # Distance to triggers
    stop_loss_distance = current_spread - config.stop_loss_spread  # Positive = safe
    take_profit_distance = config.take_profit_spread - current_spread  # Positive = not yet hit

    return {
        "stop_loss_hit": stop_loss_hit,
        "take_profit_hit": take_profit_hit,
        "stop_loss_distance": stop_loss_distance,
        "take_profit_distance": take_profit_distance,
        "stop_loss_level": config.stop_loss_spread,
        "take_profit_level": config.take_profit_spread,
    }


def auto_execute_close(config: PositionConfig, reason: str) -> bool:
    """Auto-execute close with MKT order when trigger hit.

    Returns True if order submitted successfully.
    """
    import subprocess

    print("\n" + "!" * 60)
    print(f"  AUTO-EXECUTING CLOSE: {reason}")
    print("!" * 60)

    # Build command for vrp_cs_trade.py close with MKT order
    cmd = [
        "python", str(PROJECT_ROOT / "scripts" / "vrp_cs_trade.py"),
        "close",
        "--live",
        "--confirm", "YES",
        "--order-type", "MKT",  # MARKET order for immediate fill
        "--override-checks",    # Don't block on position mismatch
    ]

    print(f"Executing: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        if result.returncode == 0:
            print("\n✅ AUTO-CLOSE ORDER SUBMITTED SUCCESSFULLY")
            return True
        else:
            print(f"\n❌ AUTO-CLOSE FAILED (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print("\n❌ AUTO-CLOSE TIMED OUT (30s)")
        return False
    except Exception as e:
        print(f"\n❌ AUTO-CLOSE ERROR: {e}")
        return False


def check_roll_status(config: PositionConfig) -> dict:
    """Check roll timing."""
    today = date.today()
    roll_date = date.fromisoformat(config.roll_by_date)
    expiry_date = date.fromisoformat(config.vx1_expiry)

    days_to_roll = (roll_date - today).days
    days_to_expiry = (expiry_date - today).days

    roll_due = days_to_roll <= 0
    roll_warning = days_to_roll <= 2 and not roll_due

    return {
        "days_to_roll": days_to_roll,
        "days_to_expiry": days_to_expiry,
        "roll_date": config.roll_by_date,
        "expiry_date": config.vx1_expiry,
        "roll_due": roll_due,
        "roll_warning": roll_warning,
    }


def check_regime_gate(market: MarketData) -> dict:
    """Check VRP Regime Gate status."""
    try:
        from dsp.regime.vrp_regime_gate import VRPRegimeGate, GateState

        gate = VRPRegimeGate()

        # Use VVIX if available, otherwise approximate
        vvix = market.vvix if market.vvix else 85.0

        state = gate.update(
            vix=market.vix,
            vvix=vvix,
            vx_f1=market.vx1,
        )

        return {
            "state": state.value,
            "score": gate.last_score,
            "trading_allowed": gate.is_trading_allowed(),
            "position_multiplier": gate.get_position_multiplier(),
            "vix": market.vix,
            "vvix": vvix,
            "contango": market.vx1 - market.vix,
        }
    except Exception as e:
        return {
            "state": "UNKNOWN",
            "error": str(e),
            "vix": market.vix,
            "contango": market.vx1 - market.vix,
        }


def print_report(config: PositionConfig, market: MarketData):
    """Print formatted monitoring report."""
    pnl = calculate_pnl(config, market)
    triggers = check_exit_triggers(config, market)
    roll = check_roll_status(config)
    gate = check_regime_gate(market)

    print("\n" + "=" * 60)
    print("  VRP-CS DAILY MONITORING REPORT")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    # Position Summary
    print("\n--- POSITION ---")
    print(f"Short: {config.short_contract} @ {config.short_entry_price:.2f} (current: {market.vx1:.2f})")
    print(f"Long:  {config.long_contract} @ {config.long_entry_price:.2f} (current: {market.vx2:.2f})")
    print(f"Entry Date: {config.entry_date}")

    # Spread & P&L
    print("\n--- SPREAD & P&L ---")
    print(f"Entry Spread:   {pnl['entry_spread']:.2f}")
    print(f"Current Spread: {pnl['current_spread']:.2f} ({pnl['spread_change']:+.2f})")
    print(f"Unrealized P&L: ${pnl['total_pnl']:+.2f}")

    # Exit Triggers
    # For VRP-CS: stop-loss when spread narrows (<), take-profit when spread widens (>)
    print("\n--- EXIT TRIGGERS ---")
    sl_status = "HIT!" if triggers['stop_loss_hit'] else f"OK (distance: {triggers['stop_loss_distance']:.2f})"
    tp_status = "HIT!" if triggers['take_profit_hit'] else f"OK (distance: {triggers['take_profit_distance']:.2f})"
    print(f"Stop-Loss  (<{triggers['stop_loss_level']:.2f}): {sl_status}")
    print(f"Take-Profit (>{triggers['take_profit_level']:.2f}): {tp_status}")

    if triggers['stop_loss_hit']:
        print("\n  >>> STOP-LOSS TRIGGERED - EXIT POSITION IMMEDIATELY <<<")
    if triggers['take_profit_hit']:
        print("\n  >>> TAKE-PROFIT TRIGGERED - CLOSE FOR PROFIT <<<")

    # Roll Status
    print("\n--- ROLL STATUS ---")
    print(f"VX1 Expiry:    {roll['expiry_date']} ({roll['days_to_expiry']} days)")
    print(f"Roll By:       {roll['roll_date']} ({roll['days_to_roll']} days)")

    if roll['roll_due']:
        print("\n  >>> ROLL IS DUE - EXECUTE ROLL TODAY <<<")
    elif roll['roll_warning']:
        print(f"\n  [!] Roll approaching in {roll['days_to_roll']} days")

    # Regime Gate
    print("\n--- REGIME GATE ---")
    print(f"State:    {gate.get('state', 'UNKNOWN')}")
    if 'score' in gate:
        print(f"Score:    {gate['score']:.3f}")
    print(f"VIX:      {gate.get('vix', 'N/A')}")
    print(f"Contango: {gate.get('contango', 0):.2f} ({market.contango_pct:.1f}%)")

    if gate.get('state') == 'CLOSED':
        print("\n  >>> GATE CLOSED - EXIT ALL POSITIONS <<<")
    elif gate.get('state') == 'REDUCE':
        print("\n  [!] GATE IN REDUCE MODE - Monitor closely")

    # Action Summary
    print("\n--- ACTION REQUIRED ---")
    actions = []
    if triggers['stop_loss_hit']:
        actions.append("EXIT: Stop-loss triggered")
    if triggers['take_profit_hit']:
        actions.append("EXIT: Take-profit triggered")
    if gate.get('state') == 'CLOSED':
        actions.append("EXIT: Gate closed")
    if roll['roll_due']:
        actions.append("ROLL: Execute roll today")
    elif roll['roll_warning']:
        actions.append(f"PREPARE: Roll in {roll['days_to_roll']} days")

    if actions:
        for action in actions:
            print(f"  -> {action}")
    else:
        print("  None - Position is healthy")

    print("\n" + "=" * 60)


def append_to_log(config: PositionConfig, market: MarketData, log_path: Path):
    """Append daily record to CSV log."""
    pnl = calculate_pnl(config, market)
    triggers = check_exit_triggers(config, market)
    roll = check_roll_status(config)
    gate = check_regime_gate(market)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create header if file doesn't exist
    write_header = not log_path.exists()

    with open(log_path, 'a') as f:
        if write_header:
            f.write("date,time,vix,vx1,vx2,spread,entry_spread,pnl_usd,")
            f.write("gate_state,gate_score,contango_pct,days_to_roll,roll_by,")
            f.write("stop_loss_level,take_profit_level,stop_loss_hit,take_profit_hit,notes\n")

        row = [
            date.today().isoformat(),
            datetime.now().strftime("%H:%M:%S"),
            f"{market.vix:.2f}",
            f"{market.vx1:.2f}",
            f"{market.vx2:.2f}",
            f"{market.spread:.2f}",
            f"{pnl['entry_spread']:.2f}",
            f"{pnl['total_pnl']:.2f}",
            gate.get('state', 'UNKNOWN'),
            f"{gate.get('score', 0):.3f}",
            f"{market.contango_pct:.2f}",
            str(roll['days_to_roll']),
            roll['roll_date'],
            f"{triggers['stop_loss_level']:.2f}",
            f"{triggers['take_profit_level']:.2f}",
            str(triggers['stop_loss_hit']),
            str(triggers['take_profit_hit']),
            "",  # notes placeholder
        ]
        f.write(",".join(row) + "\n")

    print(f"\nLog appended to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="VRP-CS Daily Position Monitor")
    parser.add_argument('--live', action='store_true', help="Fetch live quotes from IBKR")
    parser.add_argument('--config', type=Path, help="Position config JSON file (default: data/vrp/paper_trading/position_config.json)")
    parser.add_argument('--no-log', action='store_true', help="Skip appending to daily log")
    # Manual value arguments (for non-interactive use, e.g., from UI)
    parser.add_argument('--vix', type=float, help="Manual VIX spot value")
    parser.add_argument('--vx1', type=float, help="Manual VX1 (front month) price")
    parser.add_argument('--vx2', type=float, help="Manual VX2 (back month) price")
    parser.add_argument('--vvix', type=float, help="Manual VVIX value (optional)")
    # Auto-execution on trigger
    parser.add_argument('--auto-exit', action='store_true',
                        help="Auto-execute MKT close order when stop-loss or take-profit triggers hit")
    args = parser.parse_args()

    # Load position config
    config = load_position_config(args.config)

    # Get market data - priority: CLI args > live > interactive prompt
    if args.vix is not None and args.vx1 is not None and args.vx2 is not None:
        # Manual values provided via CLI
        print(f"Using provided values: VIX={args.vix}, VX1={args.vx1}, VX2={args.vx2}")
        market = MarketData(
            vix=args.vix,
            vx1=args.vx1,
            vx2=args.vx2,
            vvix=args.vvix,
            timestamp=datetime.now()
        )
    elif args.live:
        print("Fetching live quotes from IBKR...")
        market = get_live_quotes_ibkr()
        if market is None:
            print("Failed to get live quotes. Falling back to manual entry.")
            market = get_manual_quotes()
    else:
        market = get_manual_quotes()

    # Print report
    print_report(config, market)

    # Check triggers for auto-exit
    triggers = check_exit_triggers(config, market)

    # Auto-execute close if trigger hit and --auto-exit enabled
    if args.auto_exit:
        if triggers['stop_loss_hit']:
            auto_execute_close(config, "STOP-LOSS HIT")
        elif triggers['take_profit_hit']:
            auto_execute_close(config, "TAKE-PROFIT HIT")
        else:
            print("\n[auto-exit] No triggers hit - position maintained")

    # Append to daily log
    if not args.no_log:
        log_path = PROJECT_ROOT / "data" / "vrp" / "paper_trading" / "daily_log.csv"
        append_to_log(config, market, log_path)


if __name__ == "__main__":
    main()
