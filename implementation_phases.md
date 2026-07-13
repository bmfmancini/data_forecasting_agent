# Statistical Improvements Implementation Plan

This document contains the phased engineering roadmap derived from the statistical methodology review in [report.md](report.md).

## Implementation status

- **R1 / Phase 1 — in progress:** typed fit statuses and metric contracts are implemented; unavailable evidence is no longer encoded as zero; central MAE/RMSE/MAPE/WAPE/MASE conventions are active; degraded models are excluded from ranking.
- **Next R1 work:** complete synthetic regression fixtures, move adapter dictionaries fully onto `ForecastAdapterResult`, and add fitted-configuration provenance before starting common rolling-origin backtesting.

## Phased implementation roadmap

The phases below are dependency ordered. Each phase should be independently releasable behind a feature flag where it changes report output or model selection. Do not add new forecasting families until Phase 4 is complete; otherwise new models will inherit the current evaluation defects.

### Phase 0 — Freeze contracts and add regression fixtures

**Goal:** Establish observable current behavior and define the replacement interfaces before changing model logic.

**Implementation:**

1. Add deterministic fixture series covering:
   - constant and near-constant data;
   - random walk and stationary AR data;
   - additive and multiplicative seasonal data;
   - trend without seasonality;
   - zeros, negative values, missing timestamps, and duplicate timestamps;
   - short series with fewer than two seasonal cycles;
   - structural breaks and isolated anomalies.
2. Introduce typed result objects, without yet migrating all callers:
   - `ForecastFitStatus`: `ok`, `degraded`, `failed`, `not_estimable`;
   - `ForecastPrediction`: origin, horizon, timestamps, actuals, point predictions, lower/upper bounds;
   - `BacktestFoldResult`: train/test boundaries, predictions, errors, fit status, warnings, fitted configuration;
   - `ModelEvaluation`: fold results, aggregate metrics, interval metrics, diagnostics, and provenance.
3. Define explicit distinctions between:
   - fitted residuals/innovations;
   - one-step-ahead backtest errors;
   - multi-step forecast errors.
4. Snapshot current API/report schemas so migrations remain backward compatible.
5. Add structured logging fields for model name, fold, order/configuration, fallback state, and failure reason.

**Primary files:** `backend/schemas.py`, a new `backend/forecasting/contracts.py`, test fixtures under `tests/`, and pipeline/report schema tests.

**Exit criteria:** Typed contracts are tested and serializable; fixture generation is deterministic; no production behavior has changed; existing tests pass.

### Phase 1 — Honest model adapters and centralized metrics

**Goal:** Stop failed models from looking perfect and make every reported metric mathematically consistent.

**Implementation:**

1. Remove metric calculation from ARIMA, SARIMA, Holt-Winters, and EWMA adapters. Adapters should fit and predict; the evaluation layer should score.
2. Replace every zero-on-exception path with an explicit non-`ok` status and unavailable metrics.
3. Preserve complete fitted configurations when refitting:
   - ARIMA/SARIMA order and seasonal order;
   - intercept, constant, or trend configuration;
   - transformation and inverse-transformation metadata;
   - Holt-Winters trend, damping, seasonal type, and initialization;
   - EWMA/SES estimated alpha and initialization.
4. Create one central metric module with documented conventions:
   - MAE and RMSE;
   - MASE with one configured denominator convention;
   - WAPE only when its aggregate denominator is meaningful;
   - optional sMAPE with an explicit formula;
   - MAPE marked unavailable for zeros or inappropriate signed targets.
5. Include `n_evaluated`, missing count, and metric availability/reason with every score.
6. Keep baseline models in the same prediction/result contract.
7. Change `_has_required_metrics` and all comparison code to require `status == "ok"`; finiteness alone is insufficient.

**Primary files:** `forecasting/metrics.py`, all files in `forecasting/*_model.py`, `services/baseline_service.py`, `agents/forecasting_agent.py`, `schemas.py`.

**Tests:** Exact metric unit tests, zero/negative-target cases, adapter failure tests, refit-configuration tests, and a regression test proving a failed model cannot win.

**Exit criteria:** No failure produces zero error; every successful model is scored by the same functions; WAPE/MASE are populated where valid; failed candidates are absent from ranking but visible in reports.

### Phase 2 — Common rolling-origin backtesting

**Goal:** Produce valid apples-to-apples out-of-sample evidence for every model and baseline.

**Implementation:**

1. Replace the existing mislabeled helper with a backtesting service that creates splits once and reuses them for all candidates.
2. Support expanding-window validation first, with configuration for:
   - initial training size;
   - forecast horizon;
   - step size;
   - maximum number of origins;
   - optional gap between train and validation periods.
3. Use the requested production horizon where data permits. If it does not, shorten the validation horizon transparently and mark which horizons are unsupported.
4. Calculate metrics by horizon and pooled across folds; retain fold-level results.
5. Reserve an optional final untouched test window when enough data exists. Use rolling folds for tuning and the final window once for the release-quality estimate.
6. Fit preprocessing and all model choices using training observations only within each fold.
7. Make runtime limits explicit: cap candidate complexity/origins according to series length and service budget, but apply identical folds to all surviving models.
8. Keep the old terminal-holdout path behind a temporary compatibility flag and label it accurately.

**Primary files:** replace or supersede `utils/validation.py` with `forecasting/backtesting.py`; update `forecasting_agent.py`, `pipeline_service.py`, baselines, report models, and visualization inputs.

**Tests:** Split-boundary tests, no-future-data/leakage tests, identical-fold tests across all candidates, irregular-index tests, horizon aggregation tests, and deterministic repeated-run tests.

**Exit criteria:** Every displayed model metric comes from identical folds; fold boundaries are auditable; no test value affects fold preprocessing or configuration; the UI/report identifies validation design and sample size.

### Phase 3 — Residual diagnostics and uncertainty calibration

**Goal:** Make residual review operational and stop presenting heuristic bands as calibrated 95% prediction intervals.

**Implementation:**

1. Return fitted innovations where supported and pooled backtest errors from Phase 2. Never mix them under one `residuals` name.
2. Apply diagnostics to appropriate error types:
   - bias/mean error and confidence interval;
   - residual/error ACF;
   - Ljung-Box at relevant lags, with fitted AR/MA degrees-of-freedom adjustment for ARIMA-family innovations;
   - variance by horizon;
   - distribution/tail diagnostics as interval-assumption evidence, not a point-forecast pass/fail gate.
3. Preserve holdout interval bounds from ARIMA/SARIMA.
4. Replace Holt-Winters intervals with fitted-model simulation or residual/bootstrap intervals; document whether parameter uncertainty is included.
5. Replace pandas EWMA with properly fitted SES/state-space behavior and model/simulation-based intervals. Estimate alpha on each training fold. Retain the expected flat SES multi-step point forecast.
6. Calculate empirical coverage, average width, and interval/Winkler score by horizon. Add weighted interval score later if multiple nominal coverage levels are emitted.
7. Rename all user-facing uncertainty ranges “prediction intervals,” not confidence intervals.
8. Suppress a nominal “95%” claim when coverage cannot be evaluated; label such output model-based or experimental.

**Primary files:** `utils/statistical_analysis.py`, `forecasting/holt_winters.py`, `forecasting/ewma_model.py`, ARIMA/SARIMA prediction contracts, statistical review rules, reports and charts.

**Tests:** Synthetic coverage tests with broad tolerances, interval ordering/finite-value tests, width-by-horizon tests, diagnostics reachability tests, and tests confirming Shapiro results do not reject a point forecast by themselves.

**Exit criteria:** Residual diagnostics are populated for successful forecasts; interval coverage is reported when estimable; no heuristic band is labeled calibrated; statistical review consumes real diagnostics.

### Phase 4 — Seasonality, stationarity, anomalies, and leakage-safe preprocessing

**Goal:** Replace assumed/overinterpreted diagnostics with explicit evidence states and fold-safe transformations.

**Implementation:**

1. Replace the single `seasonal_period` meaning with:
   - observed timestamp frequency;
   - frequency-implied candidate periods;
   - data-derived candidate periods;
   - seasonality strength/evidence;
   - selected model period and selection provenance.
2. Permit 12 as a monthly candidate prior when metadata supports it, but never equate it with detected seasonality.
3. Use detrended spectral evidence and robust STL seasonal strength; account for harmonics rather than treating the largest periodogram peak as definitive.
4. Set and record `auto_arima` differencing options explicitly: nonseasonal test, seasonal test (the installed default is OCSB), differencing orders, and warnings.
5. Add ADF/KPSS constant and trend specifications as appropriate, with a decision matrix that can return stationary, trend-stationary, difference-stationary, conflicting, or not estimable.
6. Replace iid OLS trend significance with effect size plus autocorrelation-robust inference or a suitable nonparametric trend method.
7. Detect anomalies on detrended/seasonally adjusted residuals using robust MAD/Hampel-style rules. Keep user-confirmed events distinct from errors.
8. Replace the current uncalibrated CUSUM threshold crossing list with a calibrated change-point method and minimum segment/spacing rules. Analyze variance breaks separately.
9. Make imputation, clipping, transformation-lambda estimation, and additive/multiplicative seasonal selection train-fold operations. Implement inverse transformation and bias adjustment.
10. Return `not_estimable` rather than inventing period 2 when requested STL seasonality lacks enough cycles. A separately labeled nonseasonal trend smoother may still be returned.

**Primary files:** `utils/statistical.py`, `utils/data_cleaning.py`, `utils/preflight.py`, `agents/statistical_analysis_agent.py`, `agents/model_selection_agent.py`, schemas and prompts.

**Tests:** Known seasonal/nonseasonal simulations, harmonic-period cases, trend-stationary versus random-walk cases, anomaly-versus-seasonal-peak cases, transformation leakage tests, inverse-transform tests, and short-series capability tests.

**Exit criteria:** Unknown frequency does not manufacture seasonality; every diagnostic has `ok`/`not_estimable`/`disabled`/`failed` status; preprocessing is fitted inside folds; model selection can proceed without converting absent evidence into positive evidence.

### Phase 5 — Deterministic selection policy and bounded LLM roles

**Goal:** Make Python the source of statistical decisions and use the LLM for context, critique, and explanation.

**Implementation:**

1. Introduce a deterministic selection policy that:
   - excludes failed, degraded-by-policy, and assumption-invalid candidates;
   - requires identical-fold evidence;
   - applies user/domain loss preferences when supplied;
   - ranks using configured out-of-sample point and interval metrics;
   - recognizes statistically/practically negligible differences;
   - prefers the simpler model when evidence is effectively tied;
   - retains naive/seasonal-naive when no complex model adds demonstrated value.
2. Remove token-based remediation decisions such as `APPLY_IQR` and `APPLY_BOXCOX`. The LLM may propose them; deterministic code must test prerequisites and measure backtest impact.
3. Pass versioned typed evidence to the LLM, including status, assumptions, sample size, folds, metrics, uncertainty, warnings, and provenance.
4. Require structured LLM output with claim-to-evidence references and uncertainty labels.
5. Add a deterministic output validator for invented metrics, unsupported conclusions, contradictory model names, and recommendations violating target constraints.
6. Use the LLM to ask high-value questions about units, decision loss, horizon, holidays, interventions, censoring/stockouts, future covariates, aggregation, and allowable values.
7. Keep the statistical review agent as a critic, but prevent it from overriding numerical policy without a typed, code-recognized reason.

**Primary files:** `agents/model_selection_agent.py`, `agents/statistical_analysis_agent.py`, `agents/statistical_review_agent.py`, prompts, schemas, and pipeline orchestration.

**Tests:** Deterministic selection tables, tie/simplicity tests, baseline-retention tests, unsupported-claim tests, malformed LLM output tests, LLM outage tests, and reproducibility tests proving the selected model does not change with narrative wording.

**Exit criteria:** The same numerical evidence and policy always produce the same model; the system works without an LLM; every LLM claim is traceable or explicitly labeled as inference; user context can change the loss policy but prose cannot silently change scores.

### Phase 6 — Model coverage and advanced workflows

**Goal:** Expand capability only after the evaluation and governance foundation is trustworthy.

**Suggested order:**

1. ETS state-space candidates including no trend, damped trend, and admissible additive/multiplicative combinations.
2. Theta as a strong low-cost benchmark.
3. Dynamic regression/ARIMAX with holidays, interventions, and known future covariates.
4. Fourier regression plus ARIMA errors or another multiple-seasonality method.
5. Simple and validation-weighted forecast combinations.
6. Intermittent-demand methods when target characteristics justify them.
7. Hierarchical/grouped reconciliation when related series are introduced.
8. Count/nonnegative distributions and forecast constraints.

Every addition must implement the common adapter contract, use the same Phase 2 folds, provide supported uncertainty output, declare capability constraints, and beat or complement the reference baselines before production selection.

**Exit criteria:** Each new family has simulation/fixture tests, common-fold benchmarks, calibrated or honestly labeled intervals, runtime limits, and reportable assumptions.

### Phase 7 — Monitoring and production calibration

**Goal:** Detect when historical validation no longer represents production behavior.

**Implementation:**

1. Store forecasts, issue timestamps, horizons, model versions, intervals, and eventual actuals.
2. Monitor error and interval coverage by horizon, series, and model version.
3. Track drift in level, variance, seasonality, missingness, and covariate availability.
4. Define retraining, reselection, fallback, and alert thresholds.
5. Compare champion versus challenger models without exposing production decisions to unvalidated challengers.
6. Record overrides and user-confirmed events for later analysis.

**Exit criteria:** Forecast quality and coverage are observable after deployment; threshold breaches trigger documented actions; model/report versions are reproducible from stored provenance.

## Suggested delivery slices

For practical project management, the phases can be grouped into four releases:

| Release | Included phases | User-visible outcome |
|---|---|---|
| **R1: Honest scoring** | 0–1 | Failed models cannot win; metrics and statuses are consistent. |
| **R2: Trustworthy comparison** | 2–3 | Models use identical rolling folds; residual and interval evidence becomes real. |
| **R3: Defensible automation** | 4–5 | Seasonality/preprocessing are evidence-based; selection is deterministic and LLM claims are bounded. |
| **R4: Broader capability** | 6–7 | New model families and production monitoring build on a validated foundation. |

## Cross-phase engineering rules

- Preserve old API fields during a deprecation window, but attach explicit availability/status metadata immediately.
- Version backtest configuration, metric definitions, model configuration, preprocessing, prompts, and selection policy.
- Prefer typed objects over nested unvalidated dictionaries.
- Keep numerical computation independent of LLM availability.
- Use feature flags for selection-policy and report-schema changes; shadow-run new evaluation before it selects production forecasts.
- Do not compare results produced under different fold definitions or metric versions in the same ranking table.
- Treat performance budgets as part of statistical design: reducing origins or candidates must be visible in provenance.
- Require a test demonstrating no future-data leakage for every new preprocessing or model-selection feature.
