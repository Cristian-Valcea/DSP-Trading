# DSP-100K Control UI Specification

**Version**: 1.0
**Date**: 2026-01-10
**Status**: Draft

---

## 1. Overview

This document specifies a new page `/ui/dsp100k` for the existing Control UI infrastructure (`src/control_ui/`). The page provides a unified dashboard for monitoring and operating the DSP-100K multi-sleeve paper trading system.

### 1.1 Context

The Control UI already supports:
- **Paper Trading** (`/ui/paper`) — DQN process-isolation system
- **V15** (`/ui/v15`) — Managed futures trend-following
- **Data Acquisition** (`/ui/data`) — Market data fetching
- **Training** (`/ui/training`) — Neural network training

DSP-100K is a **different architecture** — it's a Python-based orchestrated multi-sleeve system, not a process-isolation supervised system. The UI needs to reflect this difference.

### 1.2 Design Principles

1. **Reuse existing UI patterns** — Same CSS, layout, and conventions as V15 page
2. **Sleeve-centric layout** — Each sleeve (DM, VRP-ERP, VRP-CS) as a distinct panel
3. **Operational focus** — Show "what do I need to do today?" prominently
4. **Read-before-write** — Show current state before any action buttons
5. **Fail-closed visibility** — Surface alerts and warnings prominently

---

## 2. Page Structure

### 2.1 Header

```
📊 DSP-100K Multi-Sleeve Portfolio
[Mock Mode] [Token input] [← Home]
```

### 2.2 Main Grid (2-column layout)

| Left Column | Right Column |
|-------------|--------------|
| **Alerts Panel** (full width) | |
| Sleeve DM Panel | Sleeve VRP-ERP Panel |
| Sleeve VRP-CS Panel | Daily Digest Panel |
| Output Console (full width) | |

---

## 3. Panel Specifications

### 3.1 Alerts Panel (Priority 0 — Top of Page)

**Purpose**: Surface all alerts from `daily_digest.py` prominently.

**Data Source**: Parse `data/daily_digest/digest_YYYY-MM-DD.md` for `## Alerts` section.

**Display**:
```
┌─────────────────────────────────────────────────────────────────┐
│ ALERTS                                                          │
│ ⚠️ Vol-target state is 4 days stale — run orchestrator to update│
│ ⚠️ VRP-CS roll-by is tomorrow (2026-01-14) — prepare runbook    │
│ 🔴 VRP-ERP drift >= 5%: 6.2%                                    │
│ ─────────────────────────────────────────────────────────────── │
│ [🔄 Refresh Digest]                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Severity Colors**:
- 🔴 Red chip: Critical (roll due today, unexpected positions)
- ⚠️ Yellow chip: Warning (drift, staleness, roll approaching)
- ✅ Green chip: "None" when no alerts

### 3.2 Sleeve DM Panel

**Purpose**: ETF Dual Momentum status and rebalance controls.

**Data Sources**:
- `data/vol_target_overlay_state.json` → multiplier, last_update_date
- IBKR positions (via `/api/dsp/positions?sleeve=dm`)
- Next rebalance date (first trading day of month)

**Display**:
```
┌─────────────────────────────────────────────────────────────────┐
│ SLEEVE DM (ETF Dual Momentum)                                   │
│ Status: 🟢 LIVE                                                 │
│ Vol Multiplier: 0.87 (last updated 2 days ago)                  │
│ Next Rebalance: Feb 3, 2026 (first trading day of month)        │
│ ─────────────────────────────────────────────────────────────── │
│ Positions:                                                       │
│   EFA   127 shares   $12,464   15.2%                            │
│   EEM   220 shares   $12,506   15.3%                            │
│   GLD    30 shares   $12,255   14.9%                            │
│   SHY   540 shares   $44,769   54.6%                            │
│ ─────────────────────────────────────────────────────────────── │
│ [👁️ Dry-Run Rebalance] [▶️ Execute Rebalance]                   │
│ (Only enabled on first trading day of month)                    │
└─────────────────────────────────────────────────────────────────┘
```

**Actions**:
- `Dry-Run Rebalance`: Run `dsp.cli plan` → show proposed orders
- `Execute Rebalance`: Run `dsp.cli run` with confirmation modal

### 3.3 Sleeve VRP-ERP Panel

**Purpose**: VIX-gated SPY exposure status and daily monitoring.

**Data Sources**:
- `data/vrp/paper_trading/vrp_erp_log.csv` → last row
- IBKR SPY position (via `/api/dsp/positions?symbol=SPY`)
- VIX spot from local parquet

**Display**:
```
┌─────────────────────────────────────────────────────────────────┐
│ SLEEVE VRP-ERP (VIX-Gated SPY)                                  │
│ Status: 🟢 LIVE (since Jan 12, 2026)                            │
│ ─────────────────────────────────────────────────────────────── │
│ Regime: RISK_ON (VIX: 16.24)                                    │
│ Target Shares: 42                                                │
│ Actual Shares: 40                                                │
│ Drift: 4.8%                                                      │
│ Vol Multiplier: 0.87 (applied)                                  │
│ ─────────────────────────────────────────────────────────────── │
│ [🔄 Refresh Quotes] [👁️ Dry-Run] [▶️ Execute] [📊 Run Monitor]  │
└─────────────────────────────────────────────────────────────────┘
```

**Actions**:
- `Refresh Quotes`: Fetch live VIX and SPY quotes from IBKR
- `Dry-Run`: Run `vrp_erp_daily_monitor.py --live --base 10000` (show output only)
- `Execute`: Place SPY order via IBKR (with confirmation)
- `Run Monitor`: Open terminal/log with monitor output

### 3.4 Sleeve VRP-CS Panel

**Purpose**: VIX Calendar Spread position monitoring and roll controls.

**Data Sources**:
- `data/vrp/paper_trading/daily_log.csv` → last row
- IBKR VXM positions

**Display**:
```
┌─────────────────────────────────────────────────────────────────┐
│ SLEEVE VRP-CS (Calendar Spread)                                 │
│ Status: 🟢 LIVE (Entry: Jan 9, 2026)                            │
│ ─────────────────────────────────────────────────────────────── │
│ Position: Short VXMF6 @ 16.24 | Long VXMG6 @ 17.87              │
│ Entry Spread: 1.63 | Current Spread: 1.58 (-0.05)               │
│ Unrealized P&L: +$5.00                                           │
│ ─────────────────────────────────────────────────────────────── │
│ Gate: OPEN (score: 0.623)                                        │
│ Stop-Loss: 2.04 | Take-Profit: 0.82                             │
│ ─────────────────────────────────────────────────────────────── │
│ Roll-By: Jan 14, 2026 (4 days) [⚠️ Prepare runbook]             │
│ ─────────────────────────────────────────────────────────────── │
│ [🔄 Refresh] [📊 Run Monitor] [📄 View Runbook]                 │
└─────────────────────────────────────────────────────────────────┘
```

**Actions**:
- `Refresh`: Fetch live VX quotes and update spread
- `Run Monitor`: Run `vrp_cs_daily_monitor.py --live`
- `View Runbook`: Open `docs/VRP_CS_ROLL_RUNBOOK.md` in new tab

**Roll Status Colors**:
- 🔴 Red: Roll due today (days <= 0)
- ⚠️ Yellow: Roll in 1-2 days
- 🟢 Green: Roll > 2 days away

### 3.5 Daily Digest Panel

**Purpose**: Show latest daily digest content and generation controls.

**Data Source**: `data/daily_digest/digest_YYYY-MM-DD.md`

**Display**:
```
┌─────────────────────────────────────────────────────────────────┐
│ DAILY DIGEST                                                     │
│ Last Generated: 2026-01-10 16:05:32                             │
│ ─────────────────────────────────────────────────────────────── │
│ ## Vol-Target Overlay                                            │
│ - Multiplier: 0.87 (from state file, 2 days ago)                │
│                                                                  │
│ ## Vol Indices                                                   │
│ - VIX: 16.24 | VVIX: 85.32                                      │
│ ─────────────────────────────────────────────────────────────── │
│ [🔄 Generate (Offline)] [🔄 Generate (Live)] [📄 View Full]     │
└─────────────────────────────────────────────────────────────────┘
```

**Actions**:
- `Generate (Offline)`: Run `daily_digest.py`
- `Generate (Live)`: Run `daily_digest.py --live`
- `View Full`: Open digest markdown in new tab/modal

### 3.6 Output Console Panel

**Purpose**: Show command output and job progress.

Same pattern as V15: scrollable monospace console with timestamps.

---

## 4. API Endpoints

### 4.1 New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ui/dsp100k` | Serve the DSP-100K page |
| GET | `/api/dsp/alerts` | Parse and return current alerts |
| GET | `/api/dsp/sleeves` | Return all sleeve statuses |
| GET | `/api/dsp/sleeve/{name}` | Return specific sleeve status |
| GET | `/api/dsp/positions` | Query IBKR positions (filtered) |
| GET | `/api/dsp/digest/latest` | Return latest digest content |
| POST | `/api/dsp/digest/generate` | Generate new digest |
| POST | `/api/dsp/run/{action}` | Run a sleeve action |
| GET | `/api/dsp/job/{job_id}` | Poll job status |

### 4.2 Action Types for `/api/dsp/run/{action}`

| Action | Description | Command |
|--------|-------------|---------|
| `dm_plan` | DM dry-run | `dsp.cli plan` |
| `dm_run` | DM execute | `dsp.cli run` |
| `vrp_erp_monitor` | VRP-ERP monitor | `vrp_erp_daily_monitor.py --live` |
| `vrp_erp_execute` | VRP-ERP execute | Custom order placement |
| `vrp_cs_monitor` | VRP-CS monitor | `vrp_cs_daily_monitor.py --live` |
| `digest_offline` | Generate digest | `daily_digest.py` |
| `digest_live` | Generate digest (live) | `daily_digest.py --live` |

---

## 5. Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)

1. **Add home page link**: Update `/home` route to include DSP-100K card
2. **Create template**: `src/control_ui/templates/dsp100k.html`
3. **Add route**: `/ui/dsp100k` in `server.py`
4. **Implement basic layout**: Header, grid, panels with static placeholders

### Phase 2: Read-Only Data (2-3 hours)

1. **`/api/dsp/alerts`**: Parse digest for alerts section
2. **`/api/dsp/sleeves`**: Aggregate status from logs and state files
3. **`/api/dsp/positions`**: Proxy to IBKR for DSP symbols
4. **`/api/dsp/digest/latest`**: Read and return digest content
5. **Wire panels**: JavaScript to fetch and render data

### Phase 3: Actions (2-3 hours)

1. **Job execution framework**: Reuse V15 job pattern
2. **Implement actions**: DM plan/run, VRP-ERP monitor, VRP-CS monitor
3. **Confirmation modals**: Same pattern as V15
4. **Output console**: Wire up job output streaming

### Phase 4: Polish (1-2 hours)

1. **Refresh intervals**: Auto-refresh status every 30s
2. **Roll countdown**: Highlight roll proximity warnings
3. **Error handling**: Graceful degradation when data missing
4. **Documentation**: Update USER_MANUAL_DSP.md with UI instructions

---

## 6. Dependencies

### 6.1 Existing Infrastructure (Reuse)

- `src/control_ui/server.py` — FastAPI server
- `src/control_ui/templates/*.html` — Jinja2 templates
- CSS variables and component patterns from V15

### 6.2 DSP-100K Scripts (Already Exist)

- `dsp100k/scripts/daily_digest.py`
- `dsp100k/scripts/vrp_cs_daily_monitor.py`
- `dsp100k/scripts/vrp_erp_daily_monitor.py`
- `dsp100k/src/dsp/cli.py` (DM orchestrator)

### 6.3 Data Files (Already Exist)

- `dsp100k/data/daily_digest/digest_*.md`
- `dsp100k/data/vrp/paper_trading/daily_log.csv`
- `dsp100k/data/vrp/paper_trading/vrp_erp_log.csv`
- `dsp100k/data/vol_target_overlay_state.json`

---

## 7. Open Questions

1. **IBKR Connection**: Should DSP-100K UI connect to same IBKR as V15, or separate client ID?
   - **Recommendation**: Separate client ID (e.g., 901) to avoid conflicts

2. **Execution Approach**: Should VRP-ERP/VRP-CS execution go through orchestrator or direct IBKR calls?
   - **Recommendation**: Direct IBKR calls via existing wrapper (simpler, more transparent)

3. **Mock Mode**: Should DSP-100K have its own mock data like V15?
   - **Recommendation**: Yes, for offline testing without IBKR

---

## 8. Acceptance Criteria

### 8.1 Must Have (Phase 1-2)

- [ ] `/ui/dsp100k` page renders with correct layout
- [ ] Alerts panel shows current alerts from digest
- [ ] All three sleeve panels show current status
- [ ] Daily digest content is displayed
- [ ] Navigation from home page works

### 8.2 Should Have (Phase 3)

- [ ] DM dry-run and execute work
- [ ] VRP-ERP monitor runs and shows output
- [ ] VRP-CS monitor runs and shows output
- [ ] Digest generation works from UI
- [ ] Confirmation modals for destructive actions

### 8.3 Nice to Have (Phase 4)

- [ ] Auto-refresh every 30 seconds
- [ ] Roll countdown with color coding
- [ ] Mock mode for offline testing
- [ ] Direct link to runbook PDF/print view

---

*Last updated: 2026-01-10*
