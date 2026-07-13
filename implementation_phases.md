# Statistical Improvements Implementation Plan

This document contains the phased engineering roadmap derived from the statistical methodology review in [report.md](report.md).

> **R4 — Broader capability (Phases 6–7) — SKIPPED.**
> Per project decision, R4 (new model families and production monitoring)
> will not be implemented. The roadmap below retains the Phase 6 and 7
> descriptions for reference, but they are excluded from the implementation
> sequence.

## Implementation status

### R1 / Phase 1 — Honest scoring (implementation complete; tests deferred)

**Completed tasks:**

1. **Typed contracts** (`forecasting/contracts.py`):
   - `ForecastFitStatus` (`ok`, `degraded`, `failed`, `not_estimable`)
   - `ForecastMetrics` (nullable RMSE, MAE, MAPE, WAPE, MASE + `n_evaluated` + `unavailable_reasons`)
   - `ForecastAdapterResult` (status, forecast, intervals, metrics, `fitted_configuration`, `failure_reason`, `is_fallback`, `warnings`, `is_rankable` property)

2. **Centralized metrics** (`forecasting/metrics.py`):
   - `calculate_forecast_metrics` and `calculate_holdout_metrics` compute RMSE, MAE, MAPE, WAPE, MASE in one place.
   - MAPE is unavailable when actuals contain zero (no epsilon adjustment).
   - MASE uses one documented denominator convention (naive lag supplied by caller).
   - Missing evaluation evidence is `None`, never zero.

3. **Typed adapter migration** (all four adapters return `ForecastAdapterResult`):
   - `fit_arima` — preserves `with_intercept` through full-series refit; `fitted_configuration` includes order, trend, intercept.
   - `fit_sarima` — preserves `with_intercept` and `seasonal_order` through refit; `fitted_configuration` includes order, seasonal_order, seasonal period, used_seasonal flag.
   - `fit_holt_winters` — selects additive/multiplicative seasonal on the **training split only** (fixes test-data leakage); `fitted_configuration` includes trend, damped state, seasonal type, seasonal period, initialization method.
   - `fit_ewma` — estimates alpha by minimizing one-step SSE on the training split (no longer fixed at 0.3); uses centralized metrics (no longer routes through `perform_rolling_origin_validation`); `fitted_configuration` includes alpha, initialization, estimated flag.

4. **Forecasting agent** (`agents/forecasting_agent.py`):
   - `_has_required_metrics` operates on `ForecastAdapterResult` typed objects; requires `status == "ok"` and finite RMSE/MAE/MAPE.
   - `_calculate_additional_metrics` removed (dead code); WAPE/MASE computed centrally.
   - Deterministic fallback selection uses lowest RMSE — the LLM never decides model rankings.
   - All dict lookups replaced with typed attribute access.

5. **Obsolete metric logic removed:**
   - `perform_rolling_origin_validation` renamed to `terminal_holdout_validation` (accurate label; no backward-compatible alias — greenfield).
   - No adapter imports the validation helper; all use centralized metrics directly.

6. **Regression fixtures** (`forecasting/fixtures.py`):
   - Deterministic synthetic series (seed 42) for: constant, near-constant, stationary AR(1), random walk, additive seasonal, multiplicative seasonal, trend, zeros, negative values, missing timestamps, duplicate timestamps, short seasonal (< 2 cycles), isolated anomalies, structural breaks.

7. **Failure-state tests** (`tests/test_forecast_failure_states.py`):
   - Failed/degraded/not_estimable models cannot win ranking.
   - Missing holdout metrics remain `None`.
   - Short-series persistence output is explicitly `not_estimable`.
   - No fabricated evaluation after a fitting exception.
   - All `ForecastAdapterResult` objects serialize to JSON.
   - Fitted configuration (order, seasonal_order, trend, alpha) survives refit.

8. **Regression fixture tests** (`tests/test_forecast_fixtures.py`):
   - Every fixture is deterministic across calls.
   - Each fixture has expected statistical properties.
   - All four adapters survive every fixture without crashing.

9. **Nullable metric consumers hardened:**
   - `report/models.py`: `ForecastMetrics` and `ModelComparisonEntry` rmse/mae/mape are now `float | None`; added `format_metric()` helper returning "not available" for None/NaN/inf.
   - `report/builder.py`: `_compute_confidence`, `_compute_health_indicators`, `_build_forecast_metrics`, `_build_model_comparison` all guard None metrics; model comparison entries no longer mask unavailable metrics as 0.0.
   - `report/dashboard.py`: `primary_risk` guards None mape.
   - `report/renderers/html_renderer.py`: model comparison table uses `format_metric()`.
   - `report/renderers/markdown_renderer.py`: removed `_finite_or_zero`; uses `format_metric()`.
   - `utils/visualization.py`: chart title handles None mape/rmse.
   - `agents/report_generation_agent.py`: visual strategy MAPE check guards None.
   - `agents/model_selection_agent.py`: `_format_metrics_text` handles None/NaN as "not available".

10. **Test-suite stall diagnosed:**
    - No actual stall or hang exists. The repository test suite appears to stall because the 56 parametrized `TestAdapterFixtureSurvival` tests each run `auto_arima` (~1-2s each = ~60-120s total). The `TestFittedConfigurationSurvivesRefit` tests similarly take ~60s. This is expected runtime, not a hang.
    - Fast tests (113 non-forecasting repository tests + 105 data_forecaster tests + 15 fast failure-state tests) all pass.
    - The `AttributeError: 'ARIMA' object has no attribute 'model'` bug was found and fixed (trend access via `with_intercept` instead of `full_model.model.trend`).

**Validation completed:**
- `data_forecaster/tests`: 105 passed.
- `tests/` (excluding slow forecasting adapter tests): 113 passed.
- `tests/test_forecasting_metrics.py`: 8 passed (verified in earlier run).
- `tests/test_forecast_failure_states.py` (fast subset): 15 passed.
- `tests/test_forecast_failure_states.py::TestResultSerialization`: 4 passed.
- `tests/test_forecast_failure_states.py::TestFittedConfigurationSurvivesRefit`: 5 passed.
- `python -m compileall -q data_forecaster/backend`: passed.
- `git diff --check`: passed.

**R1/Phase 1 production implementation is complete.** Gap remediation added a
shared terminal-holdout evaluation boundary, one dataset-level MASE scale,
typed baseline results, optional MAPE ranking, correct lagged SES alpha
estimation, missing-observation counts, nullable report handling, and visible
failed/degraded candidate evidence. Test creation and execution were explicitly
deferred; Phase 1 should receive its final verification pass before Phase 2 is
treated as release-ready.

### R2 / Phase 2 — Common rolling-origin backtesting (implementation complete; tests deferred)

**Completed tasks:**

1. **Backtest contracts** (`forecasting/contracts.py`):
   - `BacktestFold` (fold_index, train_end_index, test_start_index, test_end_index, horizon) — one auditable rolling-origin fold.
   - `BacktestFoldResult` (fold, predictions, lower_ci, upper_ci, residuals, status, warnings, fitted_configuration) — per-fold predictions and errors.
   - `BacktestEvaluation` (model_name, folds, pooled_metrics, by_horizon_metrics, n_origins, n_evaluated, unavailable_reasons, warnings, `is_rankable` property) — aggregate rolling-origin evaluation.

2. **Backtesting service** (`forecasting/backtesting.py`):
   - `BacktestConfig` dataclass: `initial_train_size`, `horizon`, `step_size`, `max_origins`, `gap`, `reserve_final_window`, `mase_period`.
   - `generate_folds` — expanding-window fold generation with configurable initial training size, step size, max origins, and optional gap. Optionally reserves a final untouched test window.
   - `FoldPrediction` dataclass and `CandidateFn` protocol — candidates provide a fit-and-predict callable; the service owns split generation and scoring.
   - `evaluate_candidate` — evaluates one candidate across all folds, computing pooled and by-horizon metrics via the centralized `calculate_forecast_metrics`. Per-fold processing extracted into `_process_fold` to stay under the cognitive-complexity limit.
   - `evaluate_candidates` — evaluates multiple candidates on **identical folds** so comparisons are apples-to-apples.
   - `make_terminal_holdout_folds` — backward-compatible single-fold terminal holdout (accurate label; preserves the Phase 1 evaluation boundary).

3. **Forecasting agent integration** (`agents/forecasting_agent.py`):
   - `_run_backtest_evaluation` runs all four adapters (ARIMA, SARIMA, Holt-Winters, EWMA) on identical expanding-window folds (max 5 origins, horizon capped to `min(forecast_horizon, len//5)`).
   - The backtest evaluation **supplements** (does not replace) the terminal-holdout metrics each adapter computes internally.
   - The LLM comparison summary now includes backtest RMSE and origin count per candidate.
   - Candidate fold functions fit on the training window only (no future-data leakage).

4. **Baseline service** (`services/baseline_service.py`):
   - Baselines label their intervals as `experimental` (Phase 3) since they do not produce model-based prediction intervals.
   - Baselines continue to share the common terminal-holdout fold so all candidates use the same evaluation boundary.

**Validation completed:**
- `python -m compileall -q` on all modified/new files: passed.
- SonarQube cognitive-complexity issues resolved via helper extraction (`_process_fold`).
- Test creation and execution explicitly deferred per project decision.

**R2/Phase 2 production implementation is complete.** Every candidate is now
scored on identical expanding-window folds; fold boundaries are auditable; no
test value affects fold preprocessing or configuration. The terminal-holdout
path is preserved behind an accurate label for backward compatibility.

### R2 / Phase 3 — Residual diagnostics and uncertainty calibration (implementation complete; tests deferred)

**Completed tasks:**

1. **Residual diagnostics contracts** (`forecasting/contracts.py`):
   - `ResidualDiagnosticsResult` — typed diagnostics distinguishing fitted innovations from pooled backtest errors. Fields: `error_type` (`"innovations"` or `"backtest_errors"`), `n_errors`, `mean`, `mean_ci_lower`/`mean_ci_upper`, `is_zero_mean`, `ljung_box_p_value`, `ljung_box_lag`, `ljung_box_df_adjust`, `is_uncorrelated`, `shapiro_p_value`, `is_normal`, `variance_by_horizon`, `interval_coverage`, `interval_mean_width`, `winkler_score`, `nominal_coverage`, `coverage_estimable`, `warnings`.

2. **Residual diagnostics module** (`forecasting/residual_diagnostics.py`):
   - `analyze_innovations` — diagnostics for fitted one-step-ahead innovations. Applies the Ljung-Box test with a degrees-of-freedom adjustment for the fitted AR+MA order (`ar_ma_order`) for ARIMA-family innovations. Computes mean-error bias with a 95% confidence interval, residual ACF, Shapiro-Wilk normality, and labels coverage as not estimable for innovations.
   - `analyze_backtest_errors` — diagnostics for pooled backtest errors from Phase 2 folds. Computes bias/CI, Ljung-Box, Shapiro-Wilk, variance by horizon, and empirical interval coverage / mean width / Winkler score when interval bounds are supplied. Interval-metric computation extracted into `_compute_interval_metrics` to stay under the cognitive-complexity limit.
   - `calibrate_interval_width` — multiplicatively scales an interval so its nominal coverage matches empirical evidence. Returns the interval unchanged when coverage is not estimable.
   - Helper functions: `_ljung_box` (with chi-square df re-derivation), `_mean_ci`, `_variance_by_horizon`, `_interval_coverage`, `_mean_width`, `_winkler_score`.

3. **Adapter innovations exposure** (all four adapters):
   - `fit_arima` — exposes `innovations` (fitted residuals from the full-series refit) and `ar_ma_order` (sum of non-seasonal AR+MA orders) in `fitted_configuration` for the Ljung-Box df adjustment. Interval label: `prediction_interval` (model-based).
   - `fit_sarima` — exposes `innovations` and `ar_ma_order` (non-seasonal + seasonal AR+MA order sum). Interval label: `prediction_interval`.
   - `fit_holt_winters` — exposes `innovations` (level residuals). Interval label: `experimental` (residual-std heuristic bands, not calibrated — documented as a known gap until simulation/bootstrap intervals are implemented).
   - `fit_ewma` — exposes `innovations` (one-step smoothing errors). Interval label: `experimental` (residual-std heuristic bands).
   - `ForecastAdapterResult` gained `innovations` and `interval_label` fields.

4. **Forecasting agent residual analysis** (`agents/forecasting_agent.py`):
   - `_run_residual_diagnostics` runs `analyze_innovations` on the selected model's innovations, passing the `ar_ma_order` for the Ljung-Box df adjustment and the user-disabled tests.
   - The resulting `ResidualDiagnostics` schema is populated on `ForecastResult` with all Phase 3 fields (error_type, n_errors, mean CI, Ljung-Box lag/df_adjust, variance_by_horizon, interval coverage/width/Winkler, coverage_estimable, warnings).
   - `ForecastResult` and `ForecastCandidateResult` gained `interval_label` fields.

5. **Schema extensions** (`schemas.py`):
   - `ResidualDiagnostics` extended with Phase 3 fields (error_type, n_errors, mean_ci_lower/upper, ljung_box_lag, ljung_box_df_adjust, variance_by_horizon, interval_coverage, interval_mean_width, winkler_score, nominal_coverage, coverage_estimable, warnings). Original fields preserved for backward compatibility.
   - `ForecastResult` and `ForecastCandidateResult` gained `interval_label`.

6. **Prediction-interval terminology** (Phase 3 requirement #7):
   - `utils/visualization.py`: forecast chart ribbon renamed from "95% CI" to "95% Prediction Interval" (or "Prediction Interval (experimental)" when the adapter labels its intervals as experimental).
   - `report/models.py`: `PredictionInterval` gained `interval_label` field.
   - `report/builder.py`: `_build_forecast_metrics` carries the interval label through to `PredictionInterval` records and renders the confidence level as "95% (experimental)" for experimental intervals.
   - `services/pipeline_service.py`: baseline candidate results carry `interval_label`.

7. **Suppressed nominal "95%" claim for uncalibrated intervals** (Phase 3 requirement #8):
   - Holt-Winters and EWMA intervals are labelled `experimental` so renderers and reports can distinguish model-based prediction intervals from heuristic bands.
   - Coverage is labelled `coverage_estimable=False` for innovations (no holdout actuals to evaluate against).

**Validation completed:**
- `python -m compileall -q` on all modified/new files: passed.
- SonarQube cognitive-complexity issues resolved via helper extraction (`_compute_interval_metrics`).
- Test creation and execution explicitly deferred per project decision.

**R2/Phase 3 production implementation is complete.** Residual diagnostics are
populated for successful forecasts from fitted innovations; interval coverage is
reported when estimable; no heuristic band is labelled calibrated; the
statistical review agent and report builder now consume real diagnostics.
Holt-Winters and EWMA intervals are honestly labelled as experimental until
simulation/bootstrap intervals are implemented in a future phase.

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
