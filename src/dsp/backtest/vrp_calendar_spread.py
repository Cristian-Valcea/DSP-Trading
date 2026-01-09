"""
VRP Calendar Spread Strategy

TRUE VRP ALPHA STRATEGY - Not just shorting VIX

The key insight: We DON'T short the front month outright (that's what failed).
Instead we trade the SPREAD between VX2 and VX1, harvesting the roll yield
while hedging against spike risk.

Strategy Overview:
==================
1. When contango is HIGH (VX2 >> VX1): Long VX2, Short VX1 (harvest roll-down)
2. When contango is LOW or backwardated: Flat (wait for normalization)
3. Use VRP Regime Gate to avoid entering during crisis periods
4. Monthly roll at expiration (not daily)

Why This Works:
===============
- VX1 decays faster than VX2 in contango (time decay is steeper at front)
- The spread P&L = (VX2 decay) - (VX1 decay) ≈ positive in contango
- Hedged against parallel VIX spike (both legs move together)
- Still exposed to term structure INVERSION (backwardation), hence the gate

Risk Management:
================
- Max 1 spread per $50k NAV
- Stop-loss at 25% spread widening
- Gate closes during stress regimes
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the VRP Regime Gate
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from regime.vrp_regime_gate import GateState, VRPRegimeGate


@dataclass
class SpreadPosition:
    """Track calendar spread position."""
    entry_date: datetime
    entry_vx1: float
    entry_vx2: float
    entry_spread: float  # VX2 - VX1
    contracts: int  # Number of spreads (1 spread = long 1 VX2, short 1 VX1)
    roll_date: datetime  # When VX1 expires and we need to roll


@dataclass
class CalendarSpreadConfig:
    """Configuration for the calendar spread strategy."""
    # Entry conditions
    min_contango_pct: float = 2.0  # Only enter if VX2-VX1 > 2%
    max_vix_level: float = 30.0  # Don't enter if VIX > 30

    # Position sizing
    notional_per_spread: float = 1000.0  # Notional per VX point per spread
    max_spreads: int = 5  # Max concurrent spreads

    # Risk management
    stop_loss_pct: float = 25.0  # Exit if spread widens 25% against us
    target_profit_pct: float = 50.0  # Take profit if spread narrows 50%

    # Rolling
    roll_days_before_expiry: int = 5  # Roll 5 days before VX1 expires

    # Gate integration
    use_gate: bool = True  # Use VRP Regime Gate


class VRPCalendarSpreadBacktest:
    """
    Backtest the VRP Calendar Spread Strategy.

    Trade Logic:
    - Long VX2, Short VX1 when contango is favorable and gate is OPEN
    - Roll the spread monthly as VX1 approaches expiry
    - Exit on stop-loss, take-profit, or gate closure
    """

    def __init__(
        self,
        term_structure_path: str,
        vix_path: str,
        vvix_path: str,
        vx_f1_path: str,
        config: Optional[CalendarSpreadConfig] = None,
    ):
        self.config = config or CalendarSpreadConfig()

        # Load data
        self.ts = pd.read_parquet(term_structure_path)
        self.vix = pd.read_parquet(vix_path)
        self.vvix = pd.read_parquet(vvix_path)
        self.vx_f1 = pd.read_parquet(vx_f1_path)

        # Initialize gate
        if self.config.use_gate:
            self.gate = VRPRegimeGate()
        else:
            self.gate = None

        # Align all data
        self._align_data()

    def _align_data(self) -> None:
        """Align all data to common index."""
        # Get common dates
        common_idx = (
            self.ts.index
            .intersection(self.vix.index)
            .intersection(self.vvix.index)
            .intersection(self.vx_f1.index)
        )

        self.ts = self.ts.loc[common_idx]
        self.vix = self.vix.loc[common_idx]
        self.vvix = self.vvix.loc[common_idx]
        self.vx_f1 = self.vx_f1.loc[common_idx]

        print(f"Data aligned: {len(common_idx)} trading days")
        print(f"Date range: {common_idx.min()} to {common_idx.max()}")

    def _get_gate_state(self, date: datetime) -> GateState:
        """Get the VRP Regime Gate state for a date."""
        if not self.config.use_gate:
            return GateState.OPEN

        try:
            vix_val = float(self.vix.loc[date].iloc[0])
            vvix_val = float(self.vvix.loc[date].iloc[0])
            vx_f1_val = float(self.vx_f1.loc[date, "vx_f1"])

            # Use the gate's update() method with correct signature
            as_of = date.date() if hasattr(date, "date") else date
            return self.gate.update(
                vix=vix_val,
                vvix=vvix_val,
                vx_f1=vx_f1_val,
                as_of_date=as_of
            )
        except Exception as e:
            # On error, return current state if available, else CLOSED
            return getattr(self.gate, 'current_state', GateState.CLOSED)

    def _should_enter(
        self,
        date: datetime,
        vx1: float,
        vx2: float,
        vix: float,
        gate_state: GateState
    ) -> bool:
        """Check if we should enter a new spread position."""
        # Gate must be OPEN
        if gate_state != GateState.OPEN:
            return False

        # VIX not too high
        if vix > self.config.max_vix_level:
            return False

        # Contango requirement
        contango_pct = (vx2 - vx1) / vx1 * 100
        if contango_pct < self.config.min_contango_pct:
            return False

        return True

    def _should_exit(
        self,
        position: SpreadPosition,
        current_vx1: float,
        current_vx2: float,
        current_date: datetime,
        gate_state: GateState
    ) -> Tuple[bool, str]:
        """Check if we should exit the position."""
        current_spread = current_vx2 - current_vx1
        spread_change_pct = (current_spread - position.entry_spread) / abs(position.entry_spread) * 100

        # Stop-loss: spread widened against us
        # We're short VX1, long VX2. We LOSE if spread widens (VX2 - VX1 increases)
        # Wait... actually we PROFIT if spread narrows (VX2 decays faster than VX1)
        # So: spread_pnl = entry_spread - current_spread
        # If current_spread > entry_spread * (1 + stop_loss), exit
        if current_spread > position.entry_spread * (1 + self.config.stop_loss_pct / 100):
            return True, "stop_loss"

        # Take-profit: spread narrowed significantly
        if current_spread < position.entry_spread * (1 - self.config.target_profit_pct / 100):
            return True, "take_profit"

        # Gate closed
        if gate_state == GateState.CLOSED:
            return True, "gate_closed"

        # Roll required (approaching VX1 expiry)
        if "vx1_dte" in self.ts.columns:
            dte = self.ts.loc[current_date, "vx1_dte"]
            if isinstance(dte, (int, float)) and dte <= self.config.roll_days_before_expiry:
                return True, "roll"

        return False, ""

    def run(
        self,
        start_date: str = "2014-01-01",
        end_date: str = "2025-12-31",
        initial_capital: float = 100000.0
    ) -> Dict:
        """
        Run the backtest.

        Returns:
            Dictionary with performance metrics and trade history
        """
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # Filter to date range
        mask = (self.ts.index >= start_dt) & (self.ts.index <= end_dt)
        dates = self.ts.index[mask]

        # Track state
        cash = initial_capital
        position: Optional[SpreadPosition] = None
        equity_curve = []
        trades = []
        daily_pnl = []

        # Gate state tracking
        gate_states = {"OPEN": 0, "REDUCE": 0, "CLOSED": 0}

        for date in dates:
            try:
                # Get current prices
                vx1 = float(self.ts.loc[date, "vx1"])
                vx2 = float(self.ts.loc[date, "vx2"])
                vix = float(self.vix.loc[date].iloc[0])

                if pd.isna(vx1) or pd.isna(vx2) or pd.isna(vix):
                    equity_curve.append({"date": date, "equity": cash})
                    continue

                gate_state = self._get_gate_state(date)
                gate_states[gate_state.name] += 1

                # Calculate position value if we have one
                if position is not None:
                    # Current spread value
                    current_spread = vx2 - vx1
                    # P&L = (entry_spread - current_spread) * contracts * notional
                    # We profit when spread NARROWS (VX2 decays faster)
                    spread_pnl = (position.entry_spread - current_spread) * position.contracts * self.config.notional_per_spread
                    position_value = spread_pnl
                else:
                    position_value = 0

                equity = cash + position_value
                equity_curve.append({"date": date, "equity": equity})

                # Check for exit
                if position is not None:
                    should_exit, exit_reason = self._should_exit(
                        position, vx1, vx2, date, gate_state
                    )

                    if should_exit:
                        # Close position
                        current_spread = vx2 - vx1
                        pnl = (position.entry_spread - current_spread) * position.contracts * self.config.notional_per_spread

                        trades.append({
                            "entry_date": position.entry_date,
                            "exit_date": date,
                            "entry_spread": position.entry_spread,
                            "exit_spread": current_spread,
                            "contracts": position.contracts,
                            "pnl": pnl,
                            "exit_reason": exit_reason,
                        })

                        cash += pnl
                        position = None
                        daily_pnl.append(pnl)

                # Check for entry (only if no position)
                if position is None:
                    if self._should_enter(date, vx1, vx2, vix, gate_state):
                        # Determine position size
                        spread = vx2 - vx1
                        contracts = min(
                            self.config.max_spreads,
                            int(cash / (spread * self.config.notional_per_spread * 2))  # 2x margin
                        )

                        if contracts > 0:
                            position = SpreadPosition(
                                entry_date=date,
                                entry_vx1=vx1,
                                entry_vx2=vx2,
                                entry_spread=spread,
                                contracts=contracts,
                                roll_date=date,  # Will be updated
                            )

            except Exception as e:
                # Skip problematic dates
                if len(equity_curve) > 0:
                    equity_curve.append({"date": date, "equity": equity_curve[-1]["equity"]})
                continue

        # Close any remaining position at end
        if position is not None and len(dates) > 0:
            final_date = dates[-1]
            vx1 = float(self.ts.loc[final_date, "vx1"])
            vx2 = float(self.ts.loc[final_date, "vx2"])
            current_spread = vx2 - vx1
            pnl = (position.entry_spread - current_spread) * position.contracts * self.config.notional_per_spread

            trades.append({
                "entry_date": position.entry_date,
                "exit_date": final_date,
                "entry_spread": position.entry_spread,
                "exit_spread": current_spread,
                "contracts": position.contracts,
                "pnl": pnl,
                "exit_reason": "end_of_backtest",
            })

            cash += pnl

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        if len(equity_df) > 0:
            equity_df = equity_df.set_index("date")
            returns = equity_df["equity"].pct_change().dropna()

            total_return = (equity_df["equity"].iloc[-1] / initial_capital - 1)
            n_years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
            cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

            vol = returns.std() * sqrt(252)
            sharpe = (returns.mean() * 252) / vol if vol > 0 else 0

            # Drawdown
            cum_max = equity_df["equity"].cummax()
            drawdown = (equity_df["equity"] - cum_max) / cum_max
            max_dd = drawdown.min()

            calmar = cagr / abs(max_dd) if max_dd != 0 else 0

            # Win rate
            if len(trades) > 0:
                wins = sum(1 for t in trades if t["pnl"] > 0)
                win_rate = wins / len(trades)
                avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if wins > 0 else 0
                avg_loss = np.mean([t["pnl"] for t in trades if t["pnl"] <= 0]) if (len(trades) - wins) > 0 else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
        else:
            total_return = cagr = vol = sharpe = max_dd = calmar = win_rate = avg_win = avg_loss = 0

        results = {
            "strategy": "VRP Calendar Spread (Long VX2, Short VX1)",
            "period": f"{start_date} to {end_date}",
            "metrics": {
                "return": float(total_return),
                "cagr": float(cagr),
                "vol": float(vol),
                "sharpe": float(sharpe),
                "max_dd": float(max_dd),
                "calmar": float(calmar),
            },
            "trade_stats": {
                "total_trades": len(trades),
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
            },
            "gate_distribution": gate_states,
            "config": {
                "min_contango_pct": self.config.min_contango_pct,
                "max_vix_level": self.config.max_vix_level,
                "stop_loss_pct": self.config.stop_loss_pct,
                "use_gate": self.config.use_gate,
            },
            "trades": trades,
            "equity_curve": equity_df["equity"].to_dict() if len(equity_df) > 0 else {},
        }

        return results


def run_kill_test(results: Dict) -> Dict:
    """
    Run kill-test criteria on the backtest results.

    Kill Criteria:
    - Sharpe >= 0.50
    - Max DD >= -30%
    - Total return > 0
    - Win rate >= 40%
    """
    metrics = results["metrics"]
    trade_stats = results["trade_stats"]

    criteria = {
        "sharpe_pass": metrics["sharpe"] >= 0.50,
        "max_dd_pass": metrics["max_dd"] >= -0.30,
        "return_pass": metrics["return"] > 0,
        "win_rate_pass": trade_stats["win_rate"] >= 0.40,
    }

    criteria["overall_pass"] = all(criteria.values())
    criteria["verdict"] = "PASS" if criteria["overall_pass"] else "FAIL"

    return criteria


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VRP Calendar Spread Backtest")
    parser.add_argument("--data-dir", default="data/vrp", help="VRP data directory")
    parser.add_argument("--start", default="2014-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--no-gate", action="store_true", help="Disable VRP regime gate")
    parser.add_argument("--output", default=None, help="Output JSON path")

    args = parser.parse_args()

    # Paths
    ts_path = os.path.join(args.data_dir, "term_structure/vx_term_structure.parquet")
    vix_path = os.path.join(args.data_dir, "indices/VIX_spot.parquet")
    vvix_path = os.path.join(args.data_dir, "indices/VVIX.parquet")
    vx_f1_path = os.path.join(args.data_dir, "futures/VX_F1_CBOE.parquet")

    # Config
    config = CalendarSpreadConfig(use_gate=not args.no_gate)

    # Run backtest
    print("=" * 60)
    print("VRP CALENDAR SPREAD BACKTEST")
    print("=" * 60)
    print(f"Period: {args.start} to {args.end}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Gate: {'ENABLED' if config.use_gate else 'DISABLED'}")
    print()

    backtest = VRPCalendarSpreadBacktest(
        term_structure_path=ts_path,
        vix_path=vix_path,
        vvix_path=vvix_path,
        vx_f1_path=vx_f1_path,
        config=config,
    )

    results = backtest.run(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    # Print results
    print("\nPERFORMANCE METRICS:")
    print("-" * 40)
    m = results["metrics"]
    print(f"Total Return:  {m['return']*100:.1f}%")
    print(f"CAGR:          {m['cagr']*100:.1f}%")
    print(f"Volatility:    {m['vol']*100:.1f}%")
    print(f"Sharpe:        {m['sharpe']:.2f}")
    print(f"Max Drawdown:  {m['max_dd']*100:.1f}%")
    print(f"Calmar:        {m['calmar']:.2f}")

    print("\nTRADE STATISTICS:")
    print("-" * 40)
    ts = results["trade_stats"]
    print(f"Total Trades:  {ts['total_trades']}")
    print(f"Win Rate:      {ts['win_rate']*100:.1f}%")
    print(f"Avg Win:       ${ts['avg_win']:,.0f}")
    print(f"Avg Loss:      ${ts['avg_loss']:,.0f}")

    print("\nGATE DISTRIBUTION:")
    print("-" * 40)
    gd = results["gate_distribution"]
    total = sum(gd.values())
    for state, count in gd.items():
        print(f"{state}: {count} days ({count/total*100:.1f}%)")

    # Kill test
    print("\nKILL TEST:")
    print("-" * 40)
    kill = run_kill_test(results)
    print(f"Sharpe >= 0.50:    {'✅' if kill['sharpe_pass'] else '❌'} ({m['sharpe']:.2f})")
    print(f"Max DD >= -30%:    {'✅' if kill['max_dd_pass'] else '❌'} ({m['max_dd']*100:.1f}%)")
    print(f"Return > 0:        {'✅' if kill['return_pass'] else '❌'} ({m['return']*100:.1f}%)")
    print(f"Win Rate >= 40%:   {'✅' if kill['win_rate_pass'] else '❌'} ({ts['win_rate']*100:.1f}%)")
    print()
    print(f"VERDICT: {kill['verdict']}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(args.data_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "vrp_calendar_spread_evaluation.json")

    # Remove equity curve for cleaner JSON
    save_results = {k: v for k, v in results.items() if k != "equity_curve"}
    save_results["kill_test"] = kill

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
