"""
CLI for DSP-100K.

Command-line interface for running the Diversified Systematic Portfolio.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .execution.orchestrator import DailyOrchestrator
from .utils.config import load_config, ConfigValidationError, STRICT_CONFIG
from .utils.logging import setup_logging

console = Console()


@click.group()
@click.option("--config", "-c", default="config/dsp100k.yaml", help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--strict", is_flag=True, help="Enable strict config validation (fail on unknown keys)")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool, strict: bool) -> None:
    """DSP-100K: Diversified Systematic Portfolio Manager."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose
    ctx.obj["strict"] = strict

    # Setup logging
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, json_format=False)


@cli.command()
@click.option("--force", is_flag=True, help="Force execution even if market is closed")
@click.pass_context
def run(ctx: click.Context, force: bool) -> None:
    """Run the daily trading cycle."""
    config_path = ctx.obj["config_path"]
    strict = ctx.obj.get("strict", False)

    console.print(f"[bold blue]DSP-100K Daily Execution[/bold blue]")
    console.print(f"Config: {config_path}")
    console.print(f"Force: {force}")
    if strict or STRICT_CONFIG:
        console.print(f"[yellow]Strict config mode: enabled[/yellow]")
    console.print()

    try:
        config = load_config(config_path, strict=strict)
    except ConfigValidationError as e:
        console.print(f"[bold red]Config validation failed (strict mode):[/bold red]")
        console.print(f"  [red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    async def execute():
        orchestrator = DailyOrchestrator(config)

        try:
            # Initialize
            console.print("[yellow]Initializing...[/yellow]")
            if not await orchestrator.initialize():
                console.print("[bold red]Initialization failed[/bold red]")
                return False

            console.print("[green]✓ Initialized successfully[/green]")

            # Run daily execution
            console.print("[yellow]Running daily execution...[/yellow]")
            result = await orchestrator.run_daily(force=force)

            # Display results
            if result.success:
                console.print()
                console.print("[bold green]✓ Daily execution complete[/bold green]")

                # Results table
                table = Table(title="Execution Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Date", str(result.as_of_date))
                table.add_row("Status", "SUCCESS" if result.success else "FAILED")

                if result.sleeve_b_report:
                    table.add_row(
                        "Sleeve B Orders",
                        f"{result.sleeve_b_report.orders_filled}/{result.sleeve_b_report.total_orders}",
                    )
                if result.sleeve_c_report:
                    table.add_row(
                        "Sleeve C Orders",
                        f"{result.sleeve_c_report.orders_filled}/{result.sleeve_c_report.total_orders}",
                    )

                table.add_row("Total Commission", f"${result.total_commission:.2f}")
                table.add_row("Avg Slippage", f"{result.total_slippage_bps:.1f} bps")
                table.add_row("Risk Level", result.risk_status.level.value)
                table.add_row("Scale Factor", f"{result.risk_status.scale_factor:.2f}")

                console.print(table)
                return True
            else:
                console.print()
                console.print("[bold red]✗ Daily execution failed[/bold red]")
                for error in result.errors:
                    console.print(f"  [red]• {error}[/red]")
                return False

        finally:
            await orchestrator.shutdown()

    success = asyncio.run(execute())
    sys.exit(0 if success else 1)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current system status."""
    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    async def get_status():
        orchestrator = DailyOrchestrator(config)

        try:
            if not await orchestrator.initialize():
                console.print("[bold red]Failed to connect to IBKR[/bold red]")
                return

            status = orchestrator.get_status()
            risk_status = await orchestrator._risk.get_status()

            # Status table
            table = Table(title="DSP-100K System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")

            table.add_row("State", status["state"])
            table.add_row(
                "IBKR Connection",
                "[green]Connected[/green]" if status["ibkr_connected"] else "[red]Disconnected[/red]",
            )
            table.add_row("Positions", str(status["positions_count"]))

            if status["last_execution"]["date"]:
                table.add_row("Last Execution", status["last_execution"]["date"])
                table.add_row(
                    "Last Result",
                    "[green]Success[/green]"
                    if status["last_execution"]["success"]
                    else "[red]Failed[/red]",
                )

            console.print(table)

            # Risk status table
            risk_table = Table(title="Risk Status")
            risk_table.add_column("Metric", style="cyan")
            risk_table.add_column("Value", style="green")

            risk_table.add_row("Risk Level", risk_status.level.value)
            risk_table.add_row("Scale Factor", f"{risk_status.scale_factor:.2f}")
            risk_table.add_row(
                "Double Strike",
                "[red]ACTIVE[/red]" if risk_status.double_strike_active else "[green]Inactive[/green]",
            )

            # Drawdown
            dd = risk_status.drawdown
            dd_color = "green" if dd.drawdown_pct < 0.06 else "yellow" if dd.drawdown_pct < 0.10 else "red"
            risk_table.add_row("Drawdown", f"[{dd_color}]{dd.drawdown_pct:.1%}[/{dd_color}]")
            risk_table.add_row("Peak NAV", f"${dd.peak_nav:,.2f}")
            risk_table.add_row("Current NAV", f"${dd.current_nav:,.2f}")

            # Margin
            margin = risk_status.margin
            margin_color = "green" if margin.utilization_pct < 0.40 else "yellow" if margin.utilization_pct < 0.60 else "red"
            risk_table.add_row("Margin Util", f"[{margin_color}]{margin.utilization_pct:.1%}[/{margin_color}]")

            # Volatility
            vol = risk_status.volatility
            vol_color = "green" if vol.portfolio_vol < 0.07 else "yellow" if vol.portfolio_vol < 0.08 else "red"
            risk_table.add_row("Portfolio Vol", f"[{vol_color}]{vol.portfolio_vol:.1%}[/{vol_color}]")

            console.print(risk_table)

            # Alerts
            if risk_status.alerts:
                alert_table = Table(title="Active Alerts")
                alert_table.add_column("Type", style="cyan")
                alert_table.add_column("Level", style="yellow")
                alert_table.add_column("Message", style="white")

                for alert in risk_status.alerts:
                    level_color = {
                        "normal": "green",
                        "elevated": "yellow",
                        "high": "orange3",
                        "critical": "red",
                    }.get(alert.level.value, "white")
                    alert_table.add_row(
                        alert.alert_type.value,
                        f"[{level_color}]{alert.level.value.upper()}[/{level_color}]",
                        alert.message,
                    )

                console.print(alert_table)

        finally:
            await orchestrator.shutdown()

    asyncio.run(get_status())


@cli.command()
@click.pass_context
def positions(ctx: click.Context) -> None:
    """Show current positions."""
    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    async def show_positions():
        orchestrator = DailyOrchestrator(config)

        try:
            if not await orchestrator.initialize():
                console.print("[bold red]Failed to connect to IBKR[/bold red]")
                return

            positions = orchestrator._positions

            if not positions:
                console.print("[yellow]No positions found[/yellow]")
                return

            # Positions table
            table = Table(title="Current Positions")
            table.add_column("Symbol", style="cyan")
            table.add_column("Quantity", justify="right")
            table.add_column("Avg Cost", justify="right", style="green")
            table.add_column("Market Value", justify="right", style="blue")
            table.add_column("Unrealized P&L", justify="right")

            total_value = 0.0
            total_pnl = 0.0

            for symbol, position in sorted(positions.items()):
                pnl_color = "green" if position.unrealized_pnl >= 0 else "red"
                table.add_row(
                    symbol,
                    f"{position.quantity:,.0f}",
                    f"${position.avg_cost:.2f}",
                    f"${position.market_value:,.2f}",
                    f"[{pnl_color}]${position.unrealized_pnl:,.2f}[/{pnl_color}]",
                )
                total_value += position.market_value
                total_pnl += position.unrealized_pnl

            # Summary row
            table.add_section()
            pnl_color = "green" if total_pnl >= 0 else "red"
            table.add_row(
                "[bold]Total[/bold]",
                "",
                "",
                f"[bold]${total_value:,.2f}[/bold]",
                f"[bold {pnl_color}]${total_pnl:,.2f}[/bold {pnl_color}]",
            )

            console.print(table)

        finally:
            await orchestrator.shutdown()

    asyncio.run(show_positions())


@cli.command()
@click.option("--symbols", "-s", help="Comma-separated list of symbols")
@click.pass_context
def signals(ctx: click.Context, symbols: Optional[str]) -> None:
    """Show current trend signals for Sleeve B."""
    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    async def show_signals():
        orchestrator = DailyOrchestrator(config)

        try:
            if not await orchestrator.initialize():
                console.print("[bold red]Failed to connect to IBKR[/bold red]")
                return

            # Get symbols
            if symbols:
                symbol_list = [s.strip() for s in symbols.split(",")]
            else:
                symbol_list = orchestrator._sleeve_b.symbols

            console.print(f"[yellow]Fetching signals for {len(symbol_list)} symbols...[/yellow]")

            # Signals table
            table = Table(title="Sleeve B Trend Signals")
            table.add_column("Symbol", style="cyan")
            table.add_column("1M Return", justify="right")
            table.add_column("3M Return", justify="right")
            table.add_column("12M Skip", justify="right")
            table.add_column("Composite", justify="right", style="bold")
            table.add_column("Signal", justify="center")

            for symbol in sorted(symbol_list):
                signal_data = await orchestrator._fetcher.get_trend_signals(symbol)

                if signal_data is None:
                    table.add_row(symbol, "-", "-", "-", "-", "[red]NO DATA[/red]")
                    continue

                # Color based on value
                def format_return(val: float) -> str:
                    color = "green" if val > 0 else "red"
                    return f"[{color}]{val:+.1%}[/{color}]"

                composite = signal_data["composite_signal"]
                signal_color = "green" if composite > 0 else "red" if composite < 0 else "yellow"
                signal_text = "LONG" if composite > 0.02 else "SHORT" if composite < -0.02 else "FLAT"

                table.add_row(
                    symbol,
                    format_return(signal_data["ret_1m"]),
                    format_return(signal_data["ret_3m"]),
                    format_return(signal_data["ret_12m_skip"]),
                    format_return(composite),
                    f"[{signal_color}]{signal_text}[/{signal_color}]",
                )

            console.print(table)

        finally:
            await orchestrator.shutdown()

    asyncio.run(show_signals())


@cli.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate configuration and system setup."""
    config_path = ctx.obj["config_path"]
    strict = ctx.obj.get("strict", False)

    console.print("[bold blue]DSP-100K System Validation[/bold blue]")
    if strict or STRICT_CONFIG:
        console.print("[yellow]Strict config mode: enabled[/yellow]")
    console.print()

    errors = []
    warnings = []

    # Check config file exists
    if not Path(config_path).exists():
        errors.append(f"Config file not found: {config_path}")
    else:
        console.print(f"[green]✓[/green] Config file found: {config_path}")

        try:
            config = load_config(config_path, strict=strict)
            console.print("[green]✓[/green] Config file parsed successfully")

            # Validate risk settings
            if config.risk.dd_hard_stop > 0.15:
                warnings.append(f"Hard stop at {config.risk.dd_hard_stop:.0%} is above recommended 10%")

            if config.risk.margin_cap > 0.70:
                warnings.append(f"Margin cap at {config.risk.margin_cap:.0%} is above recommended 60%")

            # Validate sleeve settings
            if config.sleeve_b.vol_target > 0.05:
                warnings.append(f"Sleeve B vol target {config.sleeve_b.vol_target:.1%} is aggressive")

        except Exception as e:
            errors.append(f"Config validation failed: {e}")

    # Check IBKR connectivity
    console.print()
    console.print("[yellow]Checking IBKR connectivity...[/yellow]")

    async def check_ibkr():
        try:
            from .ibkr import IBKRClient

            client = IBKRClient(
                host=config.ibkr.host,
                port=config.ibkr.port,
                client_id=config.ibkr.client_id,
            )

            connected = await client.connect()
            if connected:
                console.print("[green]✓[/green] IBKR connection successful")
                summary = await client.get_account_summary()
                console.print(f"  Account: {summary.account_id}")
                console.print(f"  NLV: ${summary.nlv:,.2f}")
                await client.disconnect()
            else:
                errors.append("Failed to connect to IBKR")

        except Exception as e:
            errors.append(f"IBKR connectivity check failed: {e}")

    try:
        asyncio.run(check_ibkr())
    except Exception as e:
        errors.append(f"IBKR check error: {e}")

    # Summary
    console.print()
    console.print("[bold]Validation Summary[/bold]")

    if warnings:
        console.print()
        console.print("[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]⚠[/yellow] {w}")

    if errors:
        console.print()
        console.print("[red]Errors:[/red]")
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
        console.print()
        console.print("[bold red]Validation FAILED[/bold red]")
        sys.exit(1)
    else:
        console.print()
        console.print("[bold green]Validation PASSED[/bold green]")


@cli.command()
@click.option("--output", "-o", default=None, help="Output file for plan JSON (default: stdout)")
@click.option("--force", is_flag=True, help="Force plan generation even if market is closed")
@click.pass_context
def plan(ctx: click.Context, output: Optional[str], force: bool) -> None:
    """
    Generate execution plan without placing trades (dry-run mode).

    This command connects to IBKR to fetch current prices and positions,
    generates the daily execution plan, and exports it as JSON without
    actually placing any orders.

    Use this to review what orders WOULD be placed before running 'run'.
    """
    config_path = ctx.obj["config_path"]
    strict = ctx.obj.get("strict", False)

    console.print("[bold blue]DSP-100K Dry-Run Plan Generation[/bold blue]")
    console.print(f"Config: {config_path}")
    console.print(f"Strict mode: {strict or STRICT_CONFIG}")
    console.print()

    try:
        config = load_config(config_path, strict=strict)
    except ConfigValidationError as e:
        console.print(f"[bold red]Config validation failed (strict mode):[/bold red]")
        console.print(f"  [red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error loading config:[/bold red] {e}")
        sys.exit(1)

    async def generate_plan():
        orchestrator = DailyOrchestrator(config)

        try:
            console.print("[yellow]Connecting to IBKR...[/yellow]")
            if not await orchestrator.initialize():
                console.print("[bold red]Initialization failed[/bold red]")
                return None

            console.print("[green]✓ Connected to IBKR[/green]")

            # Get account summary and risk status
            summary = await orchestrator._ibkr.get_account_summary()
            risk_status = await orchestrator._risk.get_status()

            console.print(f"  Account NLV: ${summary.nlv:,.2f}")
            console.print(f"  Risk Level: {risk_status.level.value}")
            console.print(f"  Scale Factor: {risk_status.scale_factor:.2f}")
            console.print()

            # Check trading halt
            if orchestrator._risk.is_trading_halted:
                console.print("[bold red]⚠️  Trading is HALTED due to risk breach[/bold red]")
                return {
                    "status": "halted",
                    "reason": "Trading halted - risk breach",
                    "risk_status": {
                        "level": risk_status.level.value,
                        "scale_factor": risk_status.scale_factor,
                        "double_strike_active": risk_status.double_strike_active,
                    },
                    "orders": [],
                    "generated_at": datetime.now().isoformat(),
                }

            # Generate the plan
            console.print("[yellow]Generating execution plan...[/yellow]")
            today = orchestrator.calendar.get_latest_complete_session()

            daily_plan = await orchestrator._generate_plan(today, summary, risk_status)

            # Build output structure
            plan_output = {
                "status": "ok",
                "dry_run": True,
                "as_of_date": str(daily_plan.as_of_date),
                "generated_at": datetime.now().isoformat(),
                "account": {
                    "nlv": summary.nlv,
                    "buying_power": getattr(summary, "buying_power", 0),
                },
                "risk_status": {
                    "level": risk_status.level.value,
                    "scale_factor": daily_plan.scale_factor,
                    "double_strike_active": risk_status.double_strike_active,
                    "drawdown_pct": risk_status.drawdown.drawdown_pct if risk_status.drawdown else 0,
                },
                "sleeve_b": {
                    "enabled": config.sleeve_b.enabled,
                    "order_count": len(daily_plan.sleeve_b_orders),
                    "orders": daily_plan.sleeve_b_orders,
                },
                "sleeve_c": {
                    "enabled": config.sleeve_c.enabled,
                    "auto_disabled": not orchestrator._sleeve_c_enabled,
                    "order_count": len(daily_plan.sleeve_c_orders),
                    "orders": daily_plan.sleeve_c_orders,
                },
                "estimated_turnover": daily_plan.estimated_turnover,
            }

            console.print("[green]✓ Plan generated successfully[/green]")
            return plan_output

        finally:
            await orchestrator.shutdown()

    plan_data = asyncio.run(generate_plan())

    if plan_data is None:
        sys.exit(1)

    # Display summary
    console.print()
    console.print("[bold]Execution Plan Summary[/bold]")

    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Orders")

    # Sleeve B
    sleeve_b = plan_data["sleeve_b"]
    table.add_row(
        "Sleeve B",
        "[green]Enabled[/green]" if sleeve_b["enabled"] else "[dim]Disabled[/dim]",
        str(sleeve_b["order_count"]),
    )

    # Sleeve C
    sleeve_c = plan_data["sleeve_c"]
    status = "[red]Auto-disabled[/red]" if sleeve_c["auto_disabled"] else (
        "[green]Enabled[/green]" if sleeve_c["enabled"] else "[dim]Disabled[/dim]"
    )
    table.add_row("Sleeve C", status, str(sleeve_c["order_count"]))

    console.print(table)

    # Show order details
    if sleeve_b["orders"]:
        console.print()
        console.print("[bold]Sleeve B Orders:[/bold]")
        for order in sleeve_b["orders"]:
            side = order.get("side", "?")
            qty = order.get("quantity", 0)
            symbol = order.get("symbol", "?")
            reason = order.get("reason", "")
            side_color = "green" if side == "BUY" else "red"
            console.print(f"  [{side_color}]{side}[/{side_color}] {qty} {symbol} - {reason}")

    if sleeve_c["orders"]:
        console.print()
        console.print("[bold]Sleeve C Orders:[/bold]")
        for order in sleeve_c["orders"]:
            side = order.get("side", "?")
            qty = order.get("quantity", 0)
            symbol = order.get("symbol", "?")
            console.print(f"  {side} {qty} {symbol}")

    # Output to file or stdout
    json_output = json.dumps(plan_data, indent=2, default=str)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(json_output)
        console.print()
        console.print(f"[green]✓ Plan saved to {output}[/green]")
    else:
        console.print()
        console.print("[bold]Full Plan JSON:[/bold]")
        console.print(json_output)

    console.print()
    console.print("[bold yellow]⚠️  DRY RUN - No orders were placed[/bold yellow]")


def main():
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
