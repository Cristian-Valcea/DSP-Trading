# SPEC_VRP_FACTOR_SPEC.md — VRP Factor Signal Inspired by Babiak et al. (SSRN 4618943)
**Version**: 1.0  
**Date**: 2026-01-09  
**Status**: Research specification (adds factor-based gate to VRP sleeve)  
**Author**: Claude  

## 1. Objective
Use the “common bad volatility risk premium” factor documented in Babiak, Bevilacqua, Baruník & Ellington (SSRN 4618943) to time and size the existing VRP sleeve. The paper shows that firm-level VRP decomposes into a dominant bad-premium factor that is priced cross-sectionally and predicts aggregate market returns up to two years. Rather than keep short VIX futures on autopilot, we can treat that factor as a regime gate: when the bad-premium factor is high, systemic crowding is already baked into equity option prices and the VRP counterparty (vol sellers) is more fragile; when the factor is low, the premium is less exploited and VRP is safer.

## 2. Key Insights from the Paper
| Finding | Relevance to the sleeve |
|---|---|
| The first principal component of firm-level bad VRP explains ~80% of daily bad-premium variation | There is a single, observable “crowding” indicator instead of thousands of ad-hoc signals. |
| Stocks with the lowest exposure to the common **bad** VRP factor outperform by ~7.3% pa vs. the highest exposure quintile, conditional on other known factors | We can interpret a high bad-VRP factor as a leading warning that risk premiums are being paid for aggressively, so shorting volatility is hazardous. |
| The same common bad factor predicts aggregate equity returns 6–24 months ahead | The factor is a pre-registered, forward-looking signal that is materially different from purely backward-looking VIX levels, making it useful as a gating overlay rather than an optimization lever. |

## 3. Signal Construction
### 3.1 Data Inputs
- **Firm-level options**: OptionMetrics (U.S. single-name options) or equivalent to cover ~500 liquid S&P/Russell names. Needed fields: mid prices for calls & puts within 23–37 day maturity, implied variance, delta, volume, open interest.  
- **Realized variance**: 5-minute returns (Kibot or Polygon 5-min data) to compute daily realized total/good/bad variance (RVT, RVG, RVB).  
- **Risk-free rate**: Daily Fed funds / SOFR to convert to variance price.

### 3.2 Factor Computations
1. **Implied variance (IV)**: replicate Equations (2)–(3) in the paper to compute `IVT`, `IVG`, `IVB` via numeric integration of out-of-the-money calls/puts.  
2. **Realized variance (RV)**: sum squared 5-min returns, then split positive/negative legs (Equation (4)).  
3. **Firm-level VRP**: `VRPX_{i,t} = sqrt(IVX_{i,t}) - sqrt(30 × RVX_{i,t})`, for `X ∈ {T,G,B}` per Equation (5).  
4. **Common Factor**: Run PCA on the pooled `VRPB` cross-section daily. The first principal component (or average of loadings if PCA is noisy) becomes the **VRP Bad-Factor Score**. Normalize it to [-1, +1] via rolling z-score (lookback 260 days).  
5. **Long/Short Exposures (optional)**: Regress single-stock returns onto the factor (Fama-MacBeth) to ensure exposures match published 7.3% spread. Use this validation for factor integrity rather than as a trading signal.

## 4. Integration with VRP Sleeve
### 4.1 Factor-Based Gate
Add a new entry test in `SPEC_VRP` (Section 2.2) between the contango filter and the VVIX filter:

```python
if vrp_bad_factor_score > VRP_BAD_FACTOR_HEDGE_LEVEL:
    return False, "Bad VRP factor elevated"
elif vrp_bad_factor_score > VRP_BAD_FACTOR_REDUCE_LEVEL:
    gate_state = GateState.REDUCE  # only reduce existing exposure
else:
    gate_state = GateState.OPEN
```

Choose thresholds based on empirical percentiles:
- `HEDGE_LEVEL` ≈ 0.8 (top decile) → disallow new trades and require call-hedge ratio ≥ 100% (paid via OTM calls).  
- `REDUCE_LEVEL` ≈ 0.4 (top quartile) → cap new positions at 50% of normal size and tighten stop-loss to -10%.  

### 4.2 Sizing Adjustment
When the factor lies between `REDUCE_LEVEL` and `HEDGE_LEVEL`, scale VRP position size `(contracts_by_nav × (1 - scaled_factor))`, where `scaled_factor` maps the normalized score to [0,1]. This keeps risk contribution low while still participating in the premium when there is mild crowding.

### 4.3 Hedge Overlay
Use the same score to size the option hedge:  
`hedge_ratio = min(1.0, 0.25 + 0.75 × (vrp_bad_factor_score / HEDGE_LEVEL))`. When the factor surges, the hedge approaches full coverage, mimicking the paper’s focus on “bad” volatility (downside heavy). Keep the hedge budget ≤75 bps per annum by scaling notionals in calmer states.

## 5. Infrastructure & Deliverables
1. **Data acquisition**  
   - OptionMetrics (or Databento’s US options if available) for 500+ names.  
   - Kibot 5-min returns (fallback to Polygon 1-min if licensing prevents 5-min).  
2. **Processing pipeline (new module `src/dsp/vrp/factor.py`)**  
   - Builders for IV decomposition + RV splits.  
   - PCA engine (PyTorch/sklearn) that writes `vrp_bad_factor_score` to parquet.  
   - Validation tests: PCA variance explained ≥60%, cross-sectional spread ~7% between quintiles.  
3. **Gate integration**  
   - Extend `src/dsp/backtest/vrp_futures.py` to consume the score, apply gating logic, and generate `gate_state` time series.  
   - Add gating knobs to config `config/sleeve_vrp.yaml`.  
4. **Backtest deliverables**  
   - Baseline (no factor) vs gated TSM results, comparing Sharpe, drawdown, and hedge cost.  
   - Factor efficiency report for management (similar to `SLEEVE_ORB_KILL_TEST_RESULTS`).

## 6. Validation & Kill Criteria
Add the following additional kill criteria once the factor is enabled:
| Criterion | Threshold |
|---|---|
| Factor alignment | `VRP_BAD_FACTOR_SCORE` correlation with VIX/MR returns should exceed 0.6 historically. |
| Hedge cost vs premium | Stress test with 2× hedge cost must still leave Sharpe ≥ 0.4. |
| Factor gate value-add | Gated backtest must beat ungated backtest in net Sharpe and max DD. Otherwise treat factor as non-robust and revert to baseline `SPEC_VRP`. |

## 7. Risks & Limitations
- **Data cost**: OptionMetrics + high-frequency data ≈ $5k–15k/year. Requires procurement approval.  
- **Computation**: PCA on 500+ series requires daily pipeline; add caching.  
- **Look-ahead caution**: Factor is built from current-day implied data. Ensure we use only end-of-day options (post-market) so next-day trades do not peek ahead.  
- **Model drift**: If factor suddenly collapses (explained variance drops), gate should disable (fallback to baseline `SPEC_VRP` with only contango/VIX filters).

## 8. Next Steps
1. Acquire OptionMetrics + realized variance data.  
2. Prototype `vrp_bad_factor_score` on 2000–2025 history.  
3. Confirm scoring replicates key paper facts: 60–80% variance explained, 7% cross-sectional spread, predictive power at 6–24 months.  
4. Wire the gate into `src/dsp/backtest/vrp_futures.py`.  
5. Run gated vs baseline kill-test; keep gated variant only if it strictly improves Sharpe/maxDD.

