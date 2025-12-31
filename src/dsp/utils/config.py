"""
Configuration management for DSP-100K.

Loads configuration from YAML files with environment variable overrides.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# When True, fail on any unknown config keys instead of just warning
STRICT_CONFIG = os.getenv("DSP_STRICT_CONFIG", "false").lower() == "true"


class ConfigValidationError(ValueError):
    """Raised when config validation fails in strict mode."""
    pass

# Equity ETFs that are NOT allowed in Sleeve B (non-equity trend sleeve)
DISALLOWED_EQUITY_ETFS = {"SPY", "QQQ", "IWM", "EFA", "EEM", "VTI", "VOO", "DIA"}


@dataclass
class IBKRConfig:
    """IBKR connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout_s: int = 10
    historical_duration: str = "2 Y"
    historical_bar_size: str = "1 day"


@dataclass
class SleeveAConfig:
    """Sleeve A (Equity Momentum + SPY hedge) configuration."""
    enabled: bool = True
    vol_target: float = 0.05  # 5% annualized (cap, no leverage)
    n_long: int = 20
    n_short: int = 20
    max_weight_per_name: float = 0.04  # 4% of NLV
    max_sector_gross: float = 0.20  # 20% sector cap
    rebalance_frequency: str = "monthly"
    trade_band: float = 0.0025  # Don't trade if change < 0.25%
    hedge_trade_band: float = 0.05  # Don't adjust SPY hedge if change < 5% of sleeve NAV

    # Risk controls
    short_squeeze_gap: float = 0.20  # 20% gap triggers cover
    max_loss_per_name: float = 0.0075  # 0.75% of NLV
    beta_limit: float = 0.60  # Reduce beta to <= 0.60 net (not market-neutral)
    spy_hedge_cap: float = 0.20  # 20% of NLV

    # Universe filters
    min_price: float = 10.0
    min_adv: float = 20_000_000  # $20M ADV

    # Kill criteria
    kill_sharpe_threshold: float = 0.0
    kill_drawdown_threshold: float = 0.15


@dataclass
class SleeveBConfig:
    """Sleeve B (Cross-Asset Trend) configuration."""
    enabled: bool = True
    vol_target: float = 0.035  # 3.5% annualized
    rebalance_frequency: str = "weekly"
    turnover_cap: float = 0.50  # 50% weekly cap
    deadband: float = 0.25  # Score deadband
    single_name_cap: float = 0.15  # 15% max per ETF (alias for max_weight_per_etf)
    rebal_threshold: float = 0.05  # 5% threshold to trigger rebalance

    # Signal weights (multi-horizon)
    weight_1m: float = 0.25
    weight_3m: float = 0.50
    weight_12m: float = 0.25

    # Universe - list of non-equity ETF symbols
    universe: List[str] = field(default_factory=lambda: [
        # Bonds
        "TLT", "IEF", "LQD", "HYG", "EMB", "TIP",
        # Commodities
        "GLD", "SLV", "USO", "UNG", "DBA", "DBB",
        # Currencies
        "UUP", "FXE", "FXY", "FXB",
        # Volatility
        "VIXY",
    ])

    # Kill criteria
    spy_correlation_limit: float = 0.70
    spy_correlation_days: int = 60


@dataclass
class SleeveDMConfig:
    """Sleeve DM (ETF Dual Momentum) configuration."""
    enabled: bool = False

    # Universe
    risky_universe: List[str] = field(default_factory=lambda: [
        "SPY", "EFA", "EEM", "IEF", "TLT", "TIP", "GLD", "PDBC", "UUP",
    ])
    cash_symbol: str = "SHY"

    # Signal
    top_k: int = 3
    min_momentum: float = 0.0  # Require momentum > 0 to hold (else cash)
    momentum_window_days: int = 252
    momentum_skip_days: int = 21

    # Vol targeting (conservative estimator, matches backtest)
    vol_target: float = 0.08
    vol_lookback_days: int = 63
    max_leverage: float = 1.5

    # Execution / sizing
    rebalance_frequency: str = "monthly"
    lookback_days: int = 420  # Calendar days to fetch (>= 12m + skip + vol cushion)
    min_order_notional: float = 100.0


@dataclass
class SleeveCConfig:
    """Sleeve C (SPY Hedge) configuration."""
    enabled: bool = True
    annual_budget_pct: float = 0.0125  # 1.25% of NLV
    target_dte_min: int = 30  # Minimum DTE for new spreads
    target_dte_max: int = 45  # Maximum DTE for new spreads
    roll_dte_trigger: int = 10  # Roll when DTE falls below this

    # Delta targets
    long_delta_target: float = -0.25  # ~25 delta put
    short_delta_target: float = -0.10  # ~10 delta put
    delta_drift_min: float = -0.35
    delta_drift_max: float = -0.15

    # Structure
    underlying: str = "SPY"
    max_spreads: int = 5

    # Backwards-compatible aliases used by some modules.
    @property
    def annual_budget(self) -> float:
        return self.annual_budget_pct

    @property
    def roll_dte(self) -> int:
        return self.roll_dte_trigger


@dataclass
class SleeveIMConfig:
    """Sleeve IM (Intraday ML Long/Short) configuration."""
    enabled: bool = False

    # Portfolio construction
    top_k: int = 3                          # Per-side: 3 long + 3 short
    edge_threshold: float = 0.02            # Trade if p >= 0.52 (long) or p <= 0.48 (short)
    target_gross_exposure: float = 0.00     # Start in shadow mode (0% exposure)
    dollar_neutral: bool = True
    max_net_exposure_pct_gross: float = 0.10  # Net <= 10% of gross

    # Time windows (ET)
    feature_window_start: str = "01:30"
    feature_window_end: str = "10:30"
    entry_time: str = "11:30"
    moc_submit_time: str = "15:45"

    # Risk limits
    max_single_name_pct: float = 0.03       # 3% of total NAV
    max_sleeve_gross_pct: float = 0.15      # 15% of total NAV
    drawdown_warning: float = 0.10          # 10% DD -> reduce exposure
    drawdown_hard_stop: float = 0.15        # 15% DD -> halt
    daily_loss_limit: float = 0.01          # 1% daily loss -> no new trades
    per_name_loss_limit: float = 0.02       # 2% per-name loss -> exit

    # Execution
    entry_slippage_cap_bps: int = 20
    order_timeout_seconds: int = 60

    # Data quality
    max_synthetic_bar_pct: float = 0.70
    min_premarket_dollar_volume: float = 500_000
    min_adv_dollar: float = 50_000_000
    min_market_cap: float = 10_000_000_000

    # Model paths
    model_path: str = "models/sleeve_im/model_v1.pt"
    scaler_path: str = "models/sleeve_im/scaler_v1.pkl"

    # Universe (most liquid premarket names)
    universe: List[str] = field(default_factory=lambda: [
        # Magnificent 7
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        # Benchmark ETFs (for context features)
        "SPY", "QQQ",
    ])

    # Polygon API
    polygon_api_key_env: str = "POLYGON_API_KEY"  # Env var name


@dataclass
class RiskConfig:
    """Portfolio-level risk configuration."""
    # Volatility targeting
    vol_target: float = 0.07  # 7% portfolio volatility target
    vol_cap: float = 0.08  # 8% hard cap
    sleeve_a_vol_target: float = 0.05  # 5% for Sleeve A
    sleeve_b_vol_target: float = 0.035  # 3.5% for Sleeve B

    # Drawdown controls
    dd_warning: float = 0.06  # 6% triggers scale-down / strike
    dd_hard_stop: float = 0.10  # 10% triggers flatten to cash

    # Margin controls
    margin_cap: float = 0.60  # 60% margin utilization cap

    # Correlation limits
    sleeve_correlation_limit: float = 0.60  # A-B correlation limit
    sleeve_correlation_window: int = 30  # days


@dataclass
class ExecutionConfig:
    """Order execution configuration."""
    window_start: str = "09:35"
    window_end: str = "10:15"
    order_timeout_s: int = 120
    max_slippage_bps: int = 50  # 50 bps max slippage
    retry_count: int = 2


@dataclass
class TransactionCostConfig:
    """Transaction cost model configuration."""
    stock_slippage_bps: float = 10.0
    stock_commission_per_share: float = 0.005
    etf_slippage_bps: float = 5.0
    option_commission_per_contract: float = 0.65
    option_friction_pct: float = 0.05  # 5% of premium


@dataclass
class GeneralConfig:
    """General portfolio configuration."""
    nlv_target: float = 100_000
    cash_buffer: float = 0.10  # 10% cash buffer
    margin_cap: float = 0.60  # 60% margin cap
    risk_scale: float = 1.0  # 0.25 for small-size live
    allow_external_positions: bool = False  # Fail closed if account has FUT/OPT/etc.
    sleeve_a_nav_pct: float = 0.60
    sleeve_b_nav_pct: float = 0.30
    sleeve_dm_nav_pct: float = 0.0
    sleeve_im_nav_pct: float = 0.0  # Intraday ML sleeve allocation


@dataclass
class Config:
    """Complete DSP-100K configuration."""
    general: GeneralConfig = field(default_factory=GeneralConfig)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    sleeve_a: SleeveAConfig = field(default_factory=SleeveAConfig)
    sleeve_b: SleeveBConfig = field(default_factory=SleeveBConfig)
    sleeve_dm: SleeveDMConfig = field(default_factory=SleeveDMConfig)
    sleeve_im: SleeveIMConfig = field(default_factory=SleeveIMConfig)
    sleeve_c: SleeveCConfig = field(default_factory=SleeveCConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)

    # Universe data (loaded separately)
    sleeve_a_universe: List[Dict[str, Any]] = field(default_factory=list)
    sleeve_b_universe: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


# Backwards-compatible name used by the orchestrator.
DSPConfig = Config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""

    # IBKR overrides
    if os.getenv("DSP_IBKR_HOST"):
        config.ibkr.host = os.getenv("DSP_IBKR_HOST")
    if os.getenv("DSP_IBKR_PORT"):
        config.ibkr.port = int(os.getenv("DSP_IBKR_PORT"))
    if os.getenv("DSP_IBKR_CLIENT_ID"):
        config.ibkr.client_id = int(os.getenv("DSP_IBKR_CLIENT_ID"))

    # General overrides
    if os.getenv("DSP_RISK_SCALE"):
        config.general.risk_scale = float(os.getenv("DSP_RISK_SCALE"))
    if os.getenv("DSP_ALLOW_EXTERNAL_POSITIONS"):
        config.general.allow_external_positions = (
            os.getenv("DSP_ALLOW_EXTERNAL_POSITIONS").lower() == "true"
        )
    if os.getenv("DSP_CASH_BUFFER"):
        config.general.cash_buffer = float(os.getenv("DSP_CASH_BUFFER"))
    if os.getenv("DSP_MARGIN_CAP"):
        config.general.margin_cap = float(os.getenv("DSP_MARGIN_CAP"))

    # Sleeve enable/disable
    if os.getenv("DSP_SLEEVE_A_ENABLED"):
        config.sleeve_a.enabled = os.getenv("DSP_SLEEVE_A_ENABLED").lower() == "true"
    if os.getenv("DSP_SLEEVE_B_ENABLED"):
        config.sleeve_b.enabled = os.getenv("DSP_SLEEVE_B_ENABLED").lower() == "true"
    if os.getenv("DSP_SLEEVE_C_ENABLED"):
        config.sleeve_c.enabled = os.getenv("DSP_SLEEVE_C_ENABLED").lower() == "true"
    if os.getenv("DSP_SLEEVE_DM_ENABLED"):
        config.sleeve_dm.enabled = os.getenv("DSP_SLEEVE_DM_ENABLED").lower() == "true"
    if os.getenv("DSP_SLEEVE_IM_ENABLED"):
        config.sleeve_im.enabled = os.getenv("DSP_SLEEVE_IM_ENABLED").lower() == "true"

    # Sleeve IM specific overrides
    if os.getenv("DSP_SLEEVE_IM_EXPOSURE"):
        config.sleeve_im.target_gross_exposure = float(os.getenv("DSP_SLEEVE_IM_EXPOSURE"))

    return config


def _validate_sleeve_b_universe(universe: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Validate Sleeve B universe does not contain equity ETFs.

    Raises ValueError if any disallowed symbols are present.
    This is a hard fail - do not "warn and continue".
    """
    if not universe:
        return

    symbols = set()
    for asset_class, etfs in universe.items():
        for etf in etfs:
            if "symbol" in etf:
                symbols.add(etf["symbol"].upper())

    banned = sorted(symbols & DISALLOWED_EQUITY_ETFS)
    if banned:
        raise ValueError(
            f"Sleeve B v1 may NOT include equity index ETFs to avoid duplicating "
            f"equity beta from Sleeve A. Found: {banned}. "
            f"Remove these from the universe config."
        )


def _dataclass_from_dict(cls, data: Dict[str, Any], *, section: str, strict: bool = False):
    """Create a dataclass instance from a dictionary, warning on ignored keys.

    Args:
        cls: The dataclass type to instantiate
        data: Dictionary of config values
        section: Name of the config section (for error messages)
        strict: If True, raise ConfigValidationError on unknown keys

    Returns:
        Instance of cls with values from data

    Raises:
        ConfigValidationError: If strict=True and unknown keys are present
    """
    if data is None:
        return cls()

    # Get valid field names
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}

    # Filter to only valid fields
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    ignored = sorted(set(data.keys()) - valid_fields)
    if ignored:
        if strict or STRICT_CONFIG:
            raise ConfigValidationError(
                f"Unknown config keys in [{section}]: {ignored}. "
                f"Set DSP_STRICT_CONFIG=false to ignore (not recommended)."
            )
        logger.warning("Ignored unknown config keys in %s: %s", section, ignored)

    return cls(**filtered)


def load_config(
    config_path: Optional[str] = None,
    universe_path: Optional[str] = None,
    sleeve_a_universe_path: Optional[str] = None,
    strict: Optional[bool] = None,
) -> Config:
    """
    Load configuration from YAML files.

    Args:
        config_path: Path to main config YAML (default: config/dsp100k.yaml)
        universe_path: Path to Sleeve B universe YAML (default: config/universes/sleeve_b.yaml)
        strict: If True, fail on unknown config keys. If None, uses DSP_STRICT_CONFIG env var.

    Returns:
        Config object with all settings loaded

    Raises:
        ConfigValidationError: If strict mode enabled and unknown keys found
        ValueError: If configuration validation fails
        FileNotFoundError: If config files not found
    """
    # Use explicit strict param, otherwise fall back to env var
    strict_mode = strict if strict is not None else STRICT_CONFIG
    # Default paths
    base_dir = Path(__file__).parent.parent.parent.parent  # dsp100k/

    if config_path is None:
        config_path = os.getenv("DSP_CONFIG_PATH", str(base_dir / "config" / "dsp100k.yaml"))

    if universe_path is None:
        universe_path = str(base_dir / "config" / "universes" / "sleeve_b.yaml")

    if sleeve_a_universe_path is None:
        sleeve_a_universe_path = str(base_dir / "config" / "universes" / "sleeve_a_universe.yaml")

    # Load main config
    config_data = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    allowed_top_level = {
        "general",
        "ibkr",
        "sleeve_a",
        "sleeve_b",
        "sleeve_dm",
        "sleeve_im",
        "sleeve_c",
        "risk",
        "execution",
        "transaction_costs",
    }
    ignored_top_level = sorted(set(config_data.keys()) - allowed_top_level)
    if ignored_top_level:
        if strict_mode:
            raise ConfigValidationError(
                f"Unknown top-level config sections: {ignored_top_level}. "
                f"Set DSP_STRICT_CONFIG=false to ignore (not recommended)."
            )
        logger.warning("Ignored unknown top-level config sections: %s", ignored_top_level)

    # Build config object
    config = Config(
        general=_dataclass_from_dict(GeneralConfig, config_data.get("general"), section="general", strict=strict_mode),
        ibkr=_dataclass_from_dict(IBKRConfig, config_data.get("ibkr"), section="ibkr", strict=strict_mode),
        sleeve_a=_dataclass_from_dict(SleeveAConfig, config_data.get("sleeve_a"), section="sleeve_a", strict=strict_mode),
        sleeve_b=_dataclass_from_dict(SleeveBConfig, config_data.get("sleeve_b"), section="sleeve_b", strict=strict_mode),
        sleeve_dm=_dataclass_from_dict(SleeveDMConfig, config_data.get("sleeve_dm"), section="sleeve_dm", strict=strict_mode),
        sleeve_im=_dataclass_from_dict(SleeveIMConfig, config_data.get("sleeve_im"), section="sleeve_im", strict=strict_mode),
        sleeve_c=_dataclass_from_dict(SleeveCConfig, config_data.get("sleeve_c"), section="sleeve_c", strict=strict_mode),
        risk=_dataclass_from_dict(RiskConfig, config_data.get("risk"), section="risk", strict=strict_mode),
        execution=_dataclass_from_dict(ExecutionConfig, config_data.get("execution"), section="execution", strict=strict_mode),
        transaction_costs=_dataclass_from_dict(
            TransactionCostConfig, config_data.get("transaction_costs"), section="transaction_costs", strict=strict_mode
        ),
    )

    # Load Sleeve A universe (static, versioned YAML with sectors).
    if Path(sleeve_a_universe_path).exists():
        with open(sleeve_a_universe_path) as f:
            sleeve_a_data = yaml.safe_load(f) or {}
            sleeve_a_universe = (sleeve_a_data.get("sleeve_a_universe") or {}).get("symbols") or []
            if not isinstance(sleeve_a_universe, list):
                raise ValueError(
                    f"Invalid Sleeve A universe format in {sleeve_a_universe_path}: "
                    f"expected sleeve_a_universe.symbols to be a list"
                )
            # Validate minimal schema early (fail-fast).
            for entry in sleeve_a_universe:
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Invalid Sleeve A universe entry in {sleeve_a_universe_path}: {entry!r} (expected mapping)"
                    )
                if not entry.get("symbol") or not entry.get("sector"):
                    raise ValueError(
                        f"Invalid Sleeve A universe entry in {sleeve_a_universe_path}: "
                        f"missing required keys 'symbol' and/or 'sector': {entry!r}"
                    )
            config.sleeve_a_universe = sleeve_a_universe

    # Load Sleeve B universe
    if Path(universe_path).exists():
        with open(universe_path) as f:
            universe_data = yaml.safe_load(f) or {}
            config.sleeve_b_universe = universe_data.get("sleeve_b_universe", {})

    # Validate Sleeve B universe (hard fail on equity ETFs)
    _validate_sleeve_b_universe(config.sleeve_b_universe)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    from dataclasses import asdict

    # Convert to dict, excluding universe (saved separately)
    data = {
        "general": asdict(config.general),
        "ibkr": asdict(config.ibkr),
        "sleeve_a": asdict(config.sleeve_a),
        "sleeve_b": asdict(config.sleeve_b),
        "sleeve_dm": asdict(config.sleeve_dm),
        "sleeve_im": asdict(config.sleeve_im),
        "sleeve_c": asdict(config.sleeve_c),
        "risk": asdict(config.risk),
        "execution": asdict(config.execution),
        "transaction_costs": asdict(config.transaction_costs),
    }

    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
