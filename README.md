# DSP-100K: Diversified Systematic Portfolio

A production-ready algorithmic portfolio management system targeting $100,000 with Reg-T margin, implementing a three-sleeve diversified strategy.

## Overview

DSP-100K implements a systematic approach to portfolio management with three independent return drivers:

| Sleeve | Strategy | Allocation | Volatility Target |
|--------|----------|------------|-------------------|
| **Sleeve A** | Equity L/S (12-1 Momentum) | 30% NAV | 5% |
| **Sleeve B** | Cross-Asset Trend ETFs | 30% NAV | 3.5% |
| **Sleeve C** | SPY Put-Spread Hedge | 1.25% annual budget | N/A |

### Key Features

- **Risk Management**: 7% portfolio vol target, 8% cap, 6% DD warning, 10% hard stop
- **Double-Strike Protection**: Automatic 50% deleveraging after 2 drawdown breaches in 365 days
- **Margin Control**: 60% utilization cap with what-if checking
- **IBKR Integration**: Full broker integration via ib_insync
- **Execution Window**: 09:35-10:15 ET with marketable limit orders

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/dsp100k.git
cd dsp100k

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or for development
pip install -e ".[dev]"
```

## Configuration

Copy and modify the default configuration:

```bash
cp config/dsp100k.yaml config/my_config.yaml
```

Key configuration sections:

```yaml
# IBKR connection
ibkr:
  host: "127.0.0.1"
  port: 7497  # Paper trading
  client_id: 100

# Risk settings
risk:
  vol_target: 0.07      # 7% portfolio volatility
  dd_warning: 0.06      # 6% drawdown warning
  dd_hard_stop: 0.10    # 10% hard stop
  margin_cap: 0.60      # 60% margin cap

# Execution window
execution:
  window_start: "09:35"
  window_end: "10:15"
```

## Usage

### Validate Setup

```bash
# Check configuration and IBKR connectivity
dsp --config config/dsp100k.yaml validate
```

### Run Daily Execution

```bash
# Run the daily trading cycle
dsp run

# Force execution (even if market closed)
dsp run --force
```

### Check Status

```bash
# Show system status and risk metrics
dsp status

# Show current positions
dsp positions

# Show Sleeve B trend signals
dsp signals
```

### Verbose Mode

```bash
# Enable debug logging
dsp -v run
```

## Architecture

```
dsp100k/
├── src/dsp/
│   ├── cli.py              # CLI entry point
│   ├── ibkr/               # IBKR client and models
│   ├── data/               # Data fetching and caching
│   ├── sleeves/            # Sleeve B and C implementations
│   ├── risk/               # Risk management
│   ├── execution/          # Order execution and orchestration
│   └── utils/              # Configuration, logging, time utilities
├── config/
│   └── dsp100k.yaml        # Default configuration
└── tests/                  # Test suite
```

## Sleeve Strategies

### Sleeve B: Cross-Asset Trend

Non-equity ETF trend-following with multi-horizon signals:
- **Signal**: 0.25 × 1M + 0.50 × 3M + 0.25 × 12M (skip most recent month)
- **Universe**: TLT, GLD, USO, UUP, etc. (no equity ETFs)
- **Sizing**: Inverse-volatility weighting to 3.5% vol target
- **Caps**: 15% single-name maximum

### Sleeve C: Put-Spread Hedge

SPY put spreads for tail protection:
- **Structure**: 25-delta long put, 10-delta short put
- **Budget**: 1.25% annual (premium paid)
- **Roll**: At 10 DTE, target 30-45 DTE
- **Sizing**: Budget-based, max 5 spreads

## Risk Controls

| Control | Threshold | Action |
|---------|-----------|--------|
| Portfolio Vol | > 8% | Scale positions to hit cap |
| Drawdown | > 6% | Strike 1 logged |
| Drawdown | > 10% | **HALT TRADING** - liquidate to cash |
| Double Strike | 2 in 365 days | 50% exposure for 30 days |
| Margin | > 60% | Block new exposure |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
isort src tests

# Type checking
mypy src

# Lint
ruff src tests
```

## Paper Trading Checklist

Before running with real capital:

1. [ ] Run paper trading for minimum 2 weeks
2. [ ] Verify fill rates and slippage match expectations
3. [ ] Confirm EOD flat rates > 95%
4. [ ] Test drawdown and circuit breaker behavior
5. [ ] Verify option chain fetching works correctly
6. [ ] Review audit logs for any anomalies

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## License

MIT License - see LICENSE file for details.
