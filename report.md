# Expert Statistical Methodology Review

**Project:** Data Forecasting Agent
**Review date:** 2026-07-12
**Scope:** Current repository implementation of time-series cleaning, diagnostics, model selection, validation, forecasting, uncertainty intervals, and LLM-assisted interpretation. Facebook Prophet is intentionally out of scope.

## Executive assessment

The platform has a sensible initial architecture: deterministic Python performs the numerical work, while LLM agents interpret results, select among ARIMA, SARIMA, Holt-Winters, and EWMA, review consistency, and generate narrative output. It also contains useful ingredients—ADF and KPSS tests, STL, ACF/PACF, a periodogram, Ljung-Box tests, simple baselines, outlier checks, change-point heuristics, and forecast intervals.

However, the current pipeline is not yet statistically reliable enough for automated model ranking or decision-grade uncertainty statements. The most important problems are:

1. Model error estimates are based on inconsistent holdout windows, making cross-model comparison potentially invalid.
2. The function named `perform_rolling_origin_validation` performs only one terminal holdout, not rolling-origin validation.
3. WAPE, MASE, and residual diagnostics are effectively dead code because model adapters do not return the required `y_train`, `y_test`, or `residuals` fields.
4. Failed fits and unavailable evaluations are frequently represented as zero error, which can make a failed model appear perfect.
5. Holt-Winters and EWMA intervals are heuristic bands, not properly calibrated forecast/prediction intervals.
6. Seasonality is usually assumed from frequency (and defaults to 12) rather than established statistically; this can force seasonal model selection where no seasonal signal exists.
7. Several tests are applied to the raw series where detrending, differencing, lag selection, or multiple-testing control is needed for valid interpretation.
8. The LLM is allowed to influence remediation and model choice before it receives consistently computed out-of-sample evidence. Numerical decisions should be deterministic; the LLM should explain, challenge, and collect context.

The existing `statistical_methodology_review.md` should not be treated as an accurate specification of the code. For example, it says all models use residual-standard-deviation intervals, says EWMA alpha is optimized, and describes residual analysis as operational. Those claims do not match the current implementation.

## Methods currently implemented

### Data preparation

The repository supports timestamp auditing, duplicate detection/resolution, regular-frequency reindexing, forward fill, time interpolation, seasonal-decomposition imputation, IQR and Z-score outlier detection/clipping, optional removal, Savitzky-Golay/rolling smoothing, and Box-Cox transformation. Preflight logic exposes several cleaning choices to the user.

This is broader than the external methodology document's statement that missing observations are simply dropped. Individual model adapters still call `dropna()`, which silently compresses time if unresolved gaps remain. For a time series, deleting missing values without restoring the regular time grid changes lag meaning and is generally unsafe.

### Statistical profiling

Implemented diagnostics include:

- ADF unit-root test (`autolag="AIC"`, constant-only specification).
- KPSS stationarity test (`regression="c"`).
- OLS linear trend significance.
- STL decomposition with a supplied period.
- ACF and PACF.
- Periodogram dominant frequency.
- Ljung-Box white-noise test at one selected lag.
- IQR and Z-score outlier rules.
- Rolling mean/standard-deviation correlation as a variance-stability heuristic.
- A custom CUSUM-like change-point heuristic.
- Residual mean t-test, Ljung-Box test, and Shapiro-Wilk test (implemented, but normally not reached due to missing residual output).

### Forecasting models

- **ARIMA:** `pmdarima.auto_arima` on a training portion, using AIC and a stepwise search; its selected order is refit on the full series.
- **SARIMA:** seasonal `auto_arima`, with a supplied seasonal period; falls back to a nonseasonal configuration if fewer than two cycles exist.
- **Holt-Winters:** additive trend; additive versus multiplicative seasonality is selected by in-sample AIC when the full series is positive and contains at least two assumed cycles.
- **EWMA:** fixed `alpha=0.3`; all horizons receive the last exponentially weighted mean, so it is essentially a smoothed-level benchmark rather than a model of future dynamics.
- **Baselines:** naive, seasonal naive, historical mean, and drift forecasts.

### Model selection and LLM review

The LLM receives deterministic statistical summaries and can select a model, with a heuristic fallback. A later statistical-review agent combines deterministic flags and an LLM critic. This separation is directionally good, but model ranking and remediation need stronger deterministic gates.

## Critical correctness findings

### 1. Validation results are not comparable across models

ARIMA, SARIMA, Holt-Winters, baselines, and EWMA do not consistently evaluate exactly the same origins and horizons. ARIMA/SARIMA/Holt-Winters use:

```python
max(int(n * 0.8), n - forecast_horizon)
```

EWMA uses exactly the last `forecast_horizon` observations. These are equal only in some datasets. Cross-model ranking is valid only when every candidate is evaluated on identical observations, horizons, preprocessing fitted on training data only, and preferably identical rolling origins.

**Required fix:** create one backtesting service that generates splits once and passes them to every candidate and baseline. Report per-horizon and aggregate errors over multiple expanding-window origins. Keep the final untouched test window separate from tuning/model selection if the report claims unbiased performance.

### 2. “Rolling-origin validation” is mislabeled

`perform_rolling_origin_validation` creates one train/test split. It neither rolls nor evaluates multiple origins. This overstates robustness and makes results unusually dependent on the last window.

**Required fix:** implement expanding-window or sliding-window evaluation with configurable initial window, step, horizon, and number of origins. Rename the current function to `terminal_holdout_validation` until that is done.

### 3. WAPE and MASE are never calculated for the forecasting models

`run_forecasting_agent` calculates them only if a model result contains `y_test`, but ARIMA, SARIMA, Holt-Winters, and EWMA return no `y_test` or `y_train`. Consequently these metrics remain absent/NaN. This also undermines model selection, whose stated metric priority begins with MASE and WAPE.

**Required fix:** make validation return a common typed result containing fold-level actuals and predictions, then calculate all metrics centrally. Do not make model adapters calculate their own metrics.

### 4. Residual diagnostics are normally unreachable

The forecasting agent calls `analyze_residuals` only when the selected result contains a pandas `residuals` series. None of the four adapters returns one. Thus the residual review flags cannot validate residual autocorrelation or normality in normal operation.

**Required fix:** return in-sample innovations where meaningful and, more importantly, pooled one-step-ahead backtest errors. Label them separately. Diagnostics based only on in-sample fitted residuals can look too optimistic.

### 5. Failure is encoded as perfect performance

Several exception paths return `rmse = mae = mape = 0.0`; short ARIMA series also return zero error. Zero means a perfect forecast and can win model ranking or suppress warnings. The fallback ARIMA/SARIMA orders can also be fit after auto-selection failed, without marking the result degraded.

**Required fix:** represent unavailable metrics as `None`/NaN plus explicit `status`, `failure_reason`, and `is_fallback`. Exclude failed or unevaluated candidates from ranking. A persistence fallback must be evaluated honestly when a test set exists.

### 6. MAPE is numerically and conceptually unsafe

Adding `1e-8` to each denominator makes values at or near zero produce arbitrarily huge errors and treats negative actuals awkwardly. The baseline service instead drops zero actuals, so MAPE is inconsistent across the same comparison table.

**Required fix:** use one central metric implementation. Prefer MAE/RMSE plus MASE and WAPE when the business denominator is meaningful. Add sMAPE only with its convention documented. Mark MAPE undefined when zeros are present; do not silently alter denominators.

### 7. Forecast intervals are not uniformly valid

ARIMA/SARIMA use model-based intervals, which is appropriate subject to model assumptions. Holt-Winters uses `forecast ± 1.96 * residual_sd * sqrt(h)`. That is not the forecast-error variance formula for fitted ETS models and ignores parameter, state, trend, and seasonal uncertainty. EWMA uses a constant-width residual band at every horizon, which likewise is not a calibrated multi-step prediction interval. The document's blanket statement that these are “95% confidence intervals” is inaccurate; these should be prediction intervals, and nominal 95% coverage has not been tested.

**Required fix:** use a state-space ETS implementation with simulated/analytic prediction intervals, or bootstrap forecast errors. For EWMA, use a fitted simple-exponential-smoothing state-space model or explicitly call the bands heuristic. Backtest empirical coverage and interval score at every horizon.

### 8. Seasonal period handling can manufacture seasonality

The statistical agent accepts a default `seasonal_period=12` and returns it even when the periodogram disagrees or no seasonal evidence exists. Model-selection heuristics interpret any period greater than one as detected seasonality. Holt-Winters defaults unknown frequency to 12; SARIMA similarly uses 12 through the statistical result. Daily data is forced to 7 and weekly to 52, while valid alternatives (business-week cycles, annual daily seasonality, multiple seasonalities) are ignored.

**Required fix:** distinguish `frequency_implied_period`, `candidate_periods`, and `seasonality_detected`. Test seasonal strength after detrending, validate candidate periods through backtesting, and allow “none/unknown.” Never map unknown frequency to 12 silently.

### 9. Full-series information leaks into validation configuration

Holt-Winters chooses additive versus multiplicative seasonality by comparing models fitted to the full series, then evaluates that choice on a preceding holdout. This exposes test observations to configuration selection. Cleaning/remediation can create the same risk if clipping, Box-Cox parameters, smoothing, or imputation are estimated before splitting.

**Required fix:** fit every preprocessing choice and model hyperparameter within each training fold. Refit the chosen pipeline on all observations only after selection.

### 10. Several diagnostics are statistically overinterpreted

- ADF uses only a constant term, while KPSS also tests level stationarity. Trending series need explicit trend-stationarity specifications and a decision matrix for concordant/discordant ADF-KPSS results.
- Linear trend significance on autocorrelated observations uses invalid iid OLS standard errors; long series can make negligible slopes “significant.”
- ACF significance uses `±1.96/sqrt(n)` independently at many lags and does not control family-wise error.
- Ljung-Box is evaluated at a single arbitrary lag. For fitted ARIMA residuals, degrees of freedom should account for fitted AR/MA parameters.
- Shapiro-Wilk normality is not a core requirement for unbiased point forecasts and becomes hypersensitive for large samples. Tail behavior and interval coverage matter more.
- The variance-stability correlation is a heuristic, not a formal heteroskedasticity test.
- Raw-series IQR/Z-score rules confuse trend and seasonality with anomalies. A high seasonal peak can be valid rather than anomalous.
- The CUSUM implementation compares an unstandardized cumulative sum against `2 * raw_series_sd`; repeated exceedances become many “change points.” It is not a calibrated structural-break test.

**Required fix:** test anomalies on robust STL residuals; add appropriate lag/parameter handling; report effect size and uncertainty alongside p-values; label heuristics honestly; and use established break tests or libraries with minimum segment length and penalty selection.

### 11. Small-sample and edge-case handling is insufficient

STL falls back to period 2 even when there are not enough observations for the requested seasonal structure, which yields a decomposition but not evidence for the original cycle. ACF/PACF can receive nonpositive lag limits on very short series. Shapiro and unit-root tests have minimum-length and degeneracy constraints. Constant-series logic labels an externally supplied seasonal period despite no variation.

**Required fix:** define capability thresholds per test/model, return “not estimable,” and propagate that state into the LLM prompt and report. Never translate skipped or failed tests into affirmative evidence.

## Missing tests and methods, prioritized

### Priority 0 — required before adding more forecasting models

1. **Common time-series cross-validation:** expanding-window origins, identical splits, horizon-specific scores, and an untouched final test set.
2. **Calibrated uncertainty evaluation:** empirical coverage, average interval width, Winkler/interval score, and preferably weighted interval score.
3. **Central metric layer:** MAE, RMSE, MASE, WAPE where valid, documented sMAPE, and optional RMSSE. Include sample count and uncertainty (bootstrap intervals) for metric differences.
4. **Naive benchmarks as first-class candidates:** seasonal naive should be the minimum standard. Add relative skill scores versus naive and seasonal naive.
5. **Operational residual diagnostics:** backtest errors and fitted innovations, Ljung-Box across relevant lags with model degrees-of-freedom adjustment, residual ACF, bias, and variance by forecast horizon.

### Priority 1 — major improvements to statistical validity

1. **Seasonality strength and validation:** robust STL seasonal strength, detrended spectral analysis, candidate-period validation, and tests such as OCSB/Canova-Hansen for seasonal differencing when SARIMA is considered.
2. **Transformation selection:** Guerrero or likelihood-based Box-Cox lambda, Yeo-Johnson for nonpositive data, bias-adjusted inverse transformations, and transformation fitting inside each fold.
3. **Structural breaks:** established methods such as PELT, binary segmentation, Bai-Perron-style multiple breaks, or CUSUM tests with calibrated boundaries. Model regimes rather than merely clipping them.
4. **Robust anomaly detection:** STL residuals with MAD/Hampel or generalized ESD; classify additive outliers, level shifts, temporary changes, and missingness separately.
5. **Heteroskedasticity:** residual plots plus ARCH LM tests when relevant. If conditional variance matters, consider ARIMA/ETS mean models with GARCH-style variance models.
6. **Monotonic trend tests:** Mann-Kendall with autocorrelation correction and Sen slope where linear OLS trend is inappropriate.
7. **Long-memory/intermittent demand diagnostics:** consider Croston/SBA/TSB for intermittent nonnegative demand; do not use MAPE there.

### Priority 2 — model coverage

1. **ETS state-space model selection:** error/trend/seasonal combinations, damped trend, and admissibility constraints. This is a more principled replacement for the current fixed additive-trend Holt-Winters path.
2. **Theta method:** a strong, inexpensive univariate benchmark.
3. **Dynamic regression / ARIMAX:** holidays, promotions, weather, prices, interventions, and known future covariates often matter more than adding another univariate algorithm.
4. **Multiple-seasonality models:** TBATS/BATS, dynamic harmonic regression with Fourier terms plus ARIMA errors, or MSTL-based approaches for hourly/daily/weekly mixtures.
5. **Ensembles:** simple or validation-weighted combinations. Combination forecasts are often more stable than selecting one winner.
6. **Intervention and causal-impact support:** pulses, steps, ramps, calendar effects, and explicit pre/post intervention analysis.
7. **Hierarchical/grouped reconciliation:** bottom-up, top-down, and MinT when users forecast related totals and subseries.
8. **Count/nonnegative constraints and distributions:** Poisson/negative-binomial or transformed models; prevent impossible negative forecasts where the domain forbids them.

R-squared and in-sample AIC/BIC should not be added as generic forecast-accuracy metrics. AIC/AICc/BIC are useful for comparing models fitted to the same training data and likelihood family, especially within a model class; they do not replace out-of-sample forecast evaluation. R-squared is usually misleading for trending time series and is not a forecast metric.

## Model-specific assessment

### ARIMA

The implementation correctly separates order discovery on training data from final refitting and uses model-derived intervals. Improvements needed: use AICc for small samples where available; expose drift/constant behavior; validate differencing choices with complementary tests; enforce convergence/invertibility checks; record selected order and diagnostics in the result; and compare against naive forecasts on common folds. The assumption is not that raw observations must be stationary—rather, the differenced regression error process must be adequately stationary and residuals approximately uncorrelated. Normal residuals are primarily needed for conventional Gaussian interval accuracy, not point forecasting.

### SARIMA

The two-cycle minimum is only a bare fitting threshold, not evidence of reliable seasonal estimation; three to five cycles is a safer practical warning threshold, depending on noise and model complexity. Frequency alone must not establish seasonality. Seasonal differencing and seasonal terms should be selected and checked for over-differencing. A fallback with seasonal period one should be labeled ARIMA, not reported as substantive SARIMA performance.

### Holt-Winters

The current model always includes an additive, undamped trend. That will extrapolate indefinitely and can be unstable at longer horizons. Add no-trend and damped-trend candidates, ETS state-space selection, positivity/domain checks for multiplicative forms, and calibrated intervals. Additive versus multiplicative seasonal choice must occur inside each training fold, not on the full sample.

### EWMA

`alpha=0.3` is fixed despite the methodology document saying it is optimized. The implementation emits the same value at every horizon, so it is best presented as a simple exponential smoothing benchmark. Estimate alpha by likelihood/SSE on training data or use a state-space SES implementation. Include naive forecasts, which may outperform the lagged smoothed level after sudden changes.

## How to leverage the LLM better

### Keep these decisions deterministic

The LLM should not decide whether to clip data, apply Box-Cox, declare a period real, select a winning model, or accept a failed diagnostic from prose tokens such as `APPLY_IQR`. These operations should follow typed, auditable rules based on training-only data and common backtests. The LLM can propose an action, but code should validate prerequisites and quantify the effect before accepting it.

### High-value LLM roles

1. **Context elicitation:** ask about the target meaning, units, data-generating cadence, forecast decision, loss asymmetry, known future covariates, holidays, stockouts/censoring, aggregation, allowable negative values, and intervention dates. These facts often determine the correct statistical method.
2. **Assumption-aware explanation:** translate deterministic diagnostics into plain language, including null hypotheses, limitations, effect sizes, and what is inconclusive. Avoid saying “stationary” solely because one p-value crosses 0.05.
3. **Contradiction detection:** compare frequency metadata, detected periods, domain calendars, forecast constraints, fold metrics, residual diagnostics, and interval coverage using a typed evidence object.
4. **Analysis planning:** generate a proposed candidate set and diagnostic plan, but let deterministic policy approve it. Example: multiple seasonality plus known promotions should trigger Fourier/ARIMAX candidates rather than a narrative-only warning.
5. **Data issue classification:** use user descriptions and metadata to distinguish true anomalies from promotions, shutdowns, sensor replacements, stockouts, or regime changes. Never infer this from values alone.
6. **Sensitivity narratives:** explain how conclusions change under alternate periods, transformations, anomaly treatments, cutoff dates, and forecast horizons.
7. **Decision-focused reporting:** report expected error in domain units, skill against baseline, calibrated uncertainty, downside/upside scenarios, and actionable limitations rather than generic model definitions.

### Recommended LLM contract

Pass a versioned structured object containing test status (`passed`, `failed`, `not_estimable`, `disabled`), statistic, p-value, effect size, sample size, assumptions, fold-level metrics, interval coverage, model warnings, and provenance. Require structured output with claim-to-evidence references. Run a deterministic validator that rejects unsupported claims, invented numbers, contradictory model names, or recommendations that violate domain constraints.

The final model choice should be computed by policy—for example, exclude failed fits and poorly calibrated models, then minimize a user-selected loss or rank by MASE/WIS across common folds. The LLM should explain that choice and surface close alternatives, not create the ranking.

## Recommended implementation sequence

1. Build a single backtesting and metric service and migrate all models/baselines to it.
2. Replace zero-on-error behavior with explicit unavailable/degraded result states.
3. Return and distinguish innovations, fitted residuals, and out-of-sample errors; activate residual diagnostics.
4. Add MASE/WAPE and interval scores centrally, with consistent zero handling and per-horizon results.
5. Separate frequency-implied candidate periods from statistically supported seasonality.
6. Replace heuristic Holt-Winters/EWMA bands with ETS/state-space or bootstrap prediction intervals and measure coverage.
7. Make preprocessing a fold-fitted pipeline; add inverse-transform and bias correction.
8. Make naive and seasonal-naive forecasts production candidates and calculate skill scores.
9. Add damped ETS, Theta, dynamic regression, multiple-seasonality support, and ensembles according to dataset characteristics.
10. Convert LLM exchanges to typed evidence and recommendation schemas with deterministic validation.

## Acceptance criteria for a statistically trustworthy release

- Every candidate and baseline is evaluated on identical, timestamp-preserving folds.
- No test observation influences preprocessing, hyperparameter choice, period choice, or model form.
- Failed/unevaluated models cannot receive finite performance scores or win selection.
- Point metrics include MAE/RMSE and scale-free skill (preferably MASE); percentage metrics clearly define zero/negative behavior.
- Prediction intervals have measured out-of-sample coverage and interval score by horizon.
- Seasonal claims require evidence beyond timestamp frequency.
- Residual diagnostics are populated from actual returned errors and interpreted with appropriate lags/degrees of freedom.
- Every report statement can be traced to a typed numerical result, user-provided context, or an explicitly labeled inference.
- The selected model beats or meaningfully complements naive/seasonal-naive performance; otherwise the simple baseline is retained.
- Reports distinguish statistical significance, practical significance, uncertainty, and “not estimable.”

## Bottom line

The platform has a strong foundation for an AI-assisted time-series analysis product, but the next engineering effort should improve evaluation integrity rather than expand the model catalog. Common rolling-origin backtesting, honest failure states, operational residual diagnostics, validated seasonality, and calibrated prediction intervals will yield a much larger reliability gain than adding another forecasting algorithm. Once numerical evidence is centralized and typed, the LLM can be used exceptionally well as a context collector, skeptical reviewer, and decision-oriented explainer while Python remains the source of statistical truth.

---

## Independent expert assessment

**Reviewer:** Independent statistician / time-series forecasting specialist
**Date:** 2026-07-12
**Basis:** Code inspection of `backend/forecasting/`, `backend/utils/statistical.py`, `backend/utils/statistical_analysis.py`, `backend/utils/validation.py`, `backend/agents/forecasting_agent.py`, `backend/agents/statistical_analysis_agent.py`, `backend/agents/model_selection_agent.py`, and `backend/forecasting/metrics.py`, cross-referenced against the review above.

This section records where I agree with the preceding review, where I think it overstates or mischaracterises the implementation, and where its recommendations need refinement. I verified each claim against the current source before recording it here.

### Claims I agree with (and the code evidence)

1. **Inconsistent holdout windows across models (Finding 1).** Confirmed. `fit_arima` and `fit_sarima` use `split = max(int(len(series) * 0.8), len(series) - forecast_horizon)`; `fit_holt_winters` uses the same expression; `fit_ewma` routes through `perform_rolling_origin_validation`, which uses `split = max(1, len(clean_series) - forecast_horizon)`. These coincide only when `0.8n <= n - h`, i.e. `h <= 0.2n`. For longer horizons the EWMA test window is strictly shorter than the ARIMA/SARIMA/Holt-Winters test window, so per-model RMSE/MAE/MAPE are not computed on the same observations. Cross-model ranking on these numbers is not apples-to-apples. The fix proposed — one backtesting service that emits identical splits — is correct and should be Priority 0.

2. **`perform_rolling_origin_validation` is mislabeled (Finding 2).** Confirmed verbatim. The function in `utils/validation.py` performs a single terminal holdout split and returns one set of metrics. There is no loop over origins, no expanding/sliding window, and no per-fold aggregation. The name is misleading and the suggested rename to `terminal_holdout_validation` is appropriate until a real rolling-origin implementation exists.

3. **WAPE/MASE are effectively dead code (Finding 3).** Confirmed. `_calculate_additional_metrics` in `forecasting_agent.py` is gated on `"y_test" in results_store[name]`. None of `fit_arima`, `fit_sarima`, `fit_holt_winters`, or `fit_ewma` returns `y_test` or `y_train` (grep confirms only `ewma_model.py` uses the token `residuals`, and only for its own CI band). So the MASE/WAPE branch never fires for the four core models, and `all_metrics` ends up with `WAPE=NaN, MASE=NaN` for every candidate. This directly undermines `_METRIC_PRIORITY = ("MASE", "WAPE", ...)` in `model_selection_agent.py`, which is stated to rank on MASE first. The recommendation to compute all metrics centrally from a common typed fold result is the right structural fix.

4. **Residual diagnostics are unreachable in normal operation (Finding 4).** Confirmed. `forecasting_agent.py` only calls `analyze_residuals` when `isinstance(res["residuals"], pd.Series)`. No adapter returns such a key. The residual pipeline (`ttest_1samp`, `acorr_ljungbox`, `shapiro`) in `utils/statistical_analysis.py` is therefore never exercised on real model output. The fix — return in-sample innovations and pooled one-step-ahead backtest errors separately — is sound.

5. **Failure encoded as zero error (Finding 5).** Confirmed and, if anything, understated. `fit_arima` returns `rmse=mae=mape=0.0` for series shorter than 3 points and on every `except` branch in `_calculate_metrics`. `fit_sarima` does the same. `fit_holt_winters` sets `rmse = mae = mape = 0.0` in its `except` block. Zero is the *best possible* score, so a crashed model can silently win `_has_required_metrics` filtering (which only checks finiteness, not positivity) and appear at the top of the comparison chart. The proposed `status`/`failure_reason`/`is_fallback` result state is the correct remedy; I would additionally filter on `status == "ok"` rather than `np.isfinite(rmse)`.

6. **MAPE denominator handling is unsafe and inconsistent (Finding 6).** Confirmed. `metrics.py` and `validation.py` both add `1e-8` to the denominator; the baseline service (per the review) drops zero actuals. Two different MAPE conventions in the same comparison table is a real bug. The `1e-8` epsilon produces arbitrarily large percentage errors for near-zero actuals and is meaningless for negative observations. Centralising MAPE (and preferably deprecating it in favour of MASE/WAPE/sMAPE with a documented convention) is the right call.

7. **Holt-Winters interval formula is not the ETS forecast-error variance (Finding 7).** Confirmed. `holt_winters.py` uses `forecast ± 1.96 * resid_std * sqrt(h)`. The `sqrt(h)` growth is a rough heuristic; the true multi-step prediction variance for an ETS/AAN/AAM model includes state, parameter, and seasonal-error terms and is not `sigma^2 * h`. The review's recommendation to use `statsmodels.tsa.holtwinters.ExponentialSmoothing` with `initialization_method` and the state-space simulation intervals (or a bootstrap) is correct. Note `statsmodels` 0.14.2 does expose `simulate` on the fitted Holt-Winters result, which makes a bootstrap interval straightforward to add without changing the model class.

8. **EWMA intervals are a constant-width band (Finding 7, EWMA part).** Confirmed. `ewma_model.py` uses `f ± 1.96 * std_residuals` with no `h` growth at all, so the band is the same width at every horizon. For a simple exponential smoothing model the 1-step prediction variance is `sigma^2 * alpha/(2-alpha)` (for the equivalent ARIMA(0,1,1) representation) and multi-step variance grows; a constant band understates uncertainty at longer horizons. The review's suggestion to either fit a state-space SES or explicitly label the band as heuristic is reasonable.

9. **Seasonal period defaults to 12 and is propagated without statistical confirmation (Finding 8).** Confirmed. `_infer_seasonal_period` returns 12 for any unrecognised frequency, and `run_statistical_agent` returns `inferred_period = seasonal_period` (the caller-supplied default) even when the periodogram disagrees — it only logs a mismatch. `_heuristic_preference` in `model_selection_agent.py` then treats `sp > 1` as "seasonality detected" and prefers SARIMA. So an unknown-frequency series with no seasonal signal is pushed toward SARIMA purely by the default. Separating `frequency_implied_period`, `candidate_periods`, and `seasonality_detected` is the correct fix.

10. **Holt-Winters additive/multiplicative selection leaks test data (Finding 9).** Confirmed. `fit_holt_winters` fits both `seasonal="mul"` and `seasonal="add"` on the *full series* and picks the lower AIC, then evaluates that choice on the preceding holdout. The test observations therefore influence the model form. The fix — choose seasonal type inside each training fold — is correct and aligns with standard nested-cross-validation practice.

11. **ADF/KPSS specification mismatch (Finding 10, first bullet).** Confirmed. `run_adf_test` calls `adfuller(values, autolag="AIC")` with no `regression` argument, so it defaults to `'c'` (constant only). `run_kpss_test` uses `regression="c"`. Neither tests trend stationarity (`regression='ct'`). For a trending series, ADF with only a constant is misspecified and will fail to reject the unit root too often, while KPSS with only a constant will reject stationarity — producing a confusing "both say non-stationary" result that is really an artefact of the specification. A concordant/discordant decision matrix plus a `ct` variant for trending series is a genuine improvement.

12. **Ljung-Box at a single arbitrary lag (Finding 10, fourth bullet).** Confirmed. `run_white_noise_test` uses `lags = min(10, len(series) // 5)` (one lag value), and `analyze_residuals` uses `lag = min(10, max(1, len(residual_values) // 5))`. For fitted ARIMA residuals the degrees of freedom should be `lag - (p + q)`; neither call subtracts fitted-parameter count. The recommendation to evaluate across relevant lags with a DoF adjustment is statistically correct.

13. **CUSUM is not a calibrated break test (Finding 10, eighth bullet).** Confirmed. `detect_change_points` compares an unstandardised cumulative sum against `2 * series.std()`. This threshold has no distributional basis (the standard Brownian-bridge CUSUM boundary is `±sqrt(n) * sigma` at the boundary, not a flat `2*sigma`), and repeated threshold crossings are reported as distinct change points. Using `ruptures` (PELT/BinSeg) or `statsmodels.tsa.stattools.breakvar` with a penalty is the right direction.

14. **STL period-2 fallback masks insufficient data (Finding 11).** Confirmed. `run_stl_decomposition` sets `period = max(period, 2)` and, if `len(values) < 2*period`, silently falls back to `period=2`. This returns a decomposition but not evidence for the requested cycle. Returning "not estimable" and propagating that state is better.

### Claims I partially agree with but think need refinement

1. **"EWMA is essentially a smoothed-level benchmark" (Model-specific assessment, EWMA).** This is accurate for the current implementation (`alpha=0.3` fixed, same value at every horizon), but the framing implies EWMA is *inherently* a weak benchmark. Simple exponential smoothing is a legitimate model with an ARIMA(0,1,1) equivalence and optimal one-step-ahead properties under squared-error loss; the weakness here is the fixed `alpha` and the flat multi-step output, not the method. The fix (estimate `alpha` by SSE/likelihood on training data, or use the state-space SES in `statsmodels.tsa.holtwinters`) recovers a respectable benchmark. I would phrase the recommendation as "fit SES properly" rather than "EWMA is just a benchmark."

2. **"Use AICc for small samples where available" (ARIMA assessment).** Directionally right, but `pmdarima.auto_arima` already supports `information_criterion="aicc"` directly — the current code uses `"aic"`. The concrete fix is a one-line change: `information_criterion="aicc"` (or `"oob"` for very short series). The review could have been more actionable here.

3. **"Three to five cycles is a safer practical warning threshold" for SARIMA (SARIMA assessment).** This is a reasonable rule of thumb, but the right threshold depends on the seasonal signal-to-noise ratio and the model order. A fixed "≥3 cycles" gate can refuse legitimate monthly series with 3 years of data (36 points, 3 cycles) that SARIMA handles well. I would frame this as a *warning* ("seasonal estimates are uncertain below ~3-5 cycles") rather than a hard gate, and pair it with a seasonal-strength test (e.g. STL seasonal strength ≥ 0.3-0.4) before committing to seasonal terms.

4. **"R-squared is usually misleading for trending time series" (Priority 2 note).** Correct, but the stronger statement is that in-sample fit metrics (including AIC/BIC) should never be used for *cross-model* forecast ranking when models are fitted on different training windows or belong to different likelihood families. AIC is valid for comparing ARIMA orders on the *same* training set; it is not valid for comparing ARIMA vs Holt-Winters vs EWMA. The review states this two paragraphs later but the ordering risks being misread as "AIC is fine for ranking." I would lead with: "Out-of-sample metrics on identical folds are the only valid cross-model ranking criterion; AIC/BIC are within-family model-selection tools only."

5. **"Add OCSB/Canova-Hansen for seasonal differencing when SARIMA is considered" (Priority 1.1).** Correct in principle, but `pmdarima.auto_arima` with `seasonal=True` already performs seasonal differencing selection via its internal Canova-Hansen/OCSB test when `test='ch'` or `test='ocsb'` is passed. The current code does not set `test`, so it defaults to `'ch'` for `D` selection. The actionable fix is to expose `seasonal_test` and report which test was used, not necessarily to reimplement the test from scratch.

### Claims I think do not entirely make sense

1. **"The LLM is allowed to influence remediation and model choice before it receives consistently computed out-of-sample evidence" (Executive assessment, item 8).** This is framed as an ordering bug, but the deeper issue is not ordering — it is that the LLM is making *numerical* decisions at all. Even if out-of-sample metrics were consistent, letting an LLM pick the winner from a prose comparison summary is statistically wrong; the ranking should be a deterministic policy over typed metrics. Reordering the pipeline (metrics first, then LLM) is necessary but not sufficient. The review's own "Recommended LLM contract" section says this correctly ("the final model choice should be computed by policy"), so the executive summary and the contract section are slightly inconsistent in emphasis.

2. **"Never map unknown frequency to 12 silently" (Finding 8 required fix).** The word "silently" is the real problem, not the value 12. A monthly default is a reasonable prior for business time series; the bug is that the default is returned as `seasonality_detected=True` without a test. The fix should be: keep 12 as a *candidate* period, but require a seasonal-strength or spectral test to promote it to `seasonality_detected`. Banning the default entirely would break the common case where the user uploads monthly data without a frequency hint.

3. **"STL falls back to period 2 … which yields a decomposition but not evidence for the original cycle" (Finding 11).** This is true but the implied fix — refuse to decompose — loses useful information. A better behaviour is to decompose with the requested period if `len >= 2*period`, otherwise return the trend component only (which STL can produce without a seasonal cycle) and mark `seasonal = "not estimable"`. The review's "return not estimable" is right for the *seasonal* component but should not suppress the trend/residual decomposition.

4. **"Shapiro-Wilk normality is not a core requirement for unbiased point forecasts" (Finding 10, fifth bullet).** This is correct but slightly misdirected. The reason Shapiro-Wilk is in the code is for *interval* validity (Gaussian prediction intervals assume normal innovations), not point-forecast bias. The fix is not to drop it but to scope it: report it only as an interval-assumption check, use a robust normality test (e.g. Anderson-Darling or a Jarque-Bera with DoF correction) for large samples, and down-weight it for `n > 5000` where it is hypersensitive. The review's "tail behavior and interval coverage matter more" is the right emphasis but reads as "remove Shapiro" rather than "repurpose it."

5. **"Prefer `collections.abc.Sequence` over `list` in signatures" is listed as a Google-style rule in the project instructions but the review does not address it.** This is a code-style point, not a statistical one, and the review correctly ignores it. I note it only to flag that the review's statistical scope is appropriate and should not be expanded to cover style.

6. **The review treats "naive and seasonal-naive as first-class candidates" as Priority 0.4.** I agree they should be evaluated, but calling them "first-class candidates" risks implying they should be *selectable* as the production model. For most business series a naive forecast winning is a signal that the fitted models are broken, not a desirable outcome. The right framing is: naive/seasonal-naive are *reference baselines* used to compute skill scores (MASE is literally MAE/naive-MAE); a model that cannot beat seasonal-naive on common folds should be flagged as "no added value" rather than "the naive is the winner." The review's acceptance criterion ("the selected model beats or meaningfully complements naive/seasonal-naive performance; otherwise the simple baseline is retained") captures this, but the Priority 0 wording is looser.

### Additional issues the review does not raise

1. **`fit_arima` refits with `pm.ARIMA(order=order).fit(series)` but does not pass `seasonal_order`** — so the full-series refit for SARIMA's fallback path is correct, but for ARIMA the refit loses any drift/constant flag the training fit may have selected. This can change the forecast level. The review mentions "expose drift/constant behavior" but not this specific refit inconsistency.

2. **`calculate_holdout_metrics` calls `model.predict(n_periods=len(test), return_conf_int=True)` and discards the intervals.** For interval-coverage evaluation (Priority 0.2) the holdout intervals are already computed and thrown away. A one-line change to return them would give free empirical coverage data for ARIMA/SARIMA.

3. **`_calculate_additional_metrics` computes MASE with `y_train.shift(seasonal_period)` but the forecasting agent never passes `y_train`/`y_test`, so the MASE denominator logic is untested.** Even after the proposed fix, the MASE fallback (`np.diff(y_train)`) for short series uses a non-seasonal naive, which changes the metric's meaning. The MASE convention should be fixed (always seasonal-naive denominator, or always non-seasonal-naive, documented) rather than switched based on `y_train.shape[0] > seasonal_period`.

4. **`run_statistical_agent` sets `inferred_period = seasonal_period` and only logs periodogram mismatches when `abs(pg_period - seasonal_period) > 2`.** A 2-period tolerance is arbitrary; for monthly data (period 12) a periodogram peak at 6 (biannual) or 4 (quarterly) is silently ignored. The tolerance should be relative (e.g. within 15% of the candidate) or the periodogram should contribute a *candidate* rather than a validation gate.

### Summary judgement

The preceding review is statistically literate and, on inspection of the code, overwhelmingly accurate on the facts. The eleven "critical correctness findings" are all real and verified. The areas where I diverge are matters of emphasis and framing, not of fact:

- The core problem is not LLM ordering but LLM-as-decision-maker; deterministic policy should rank, the LLM should explain.
- The seasonal-period default of 12 is a reasonable prior that needs a statistical gate, not a ban.
- EWMA/SES is a legitimate model when fitted properly; the implementation is the problem, not the method.
- AICc, seasonal-test selection, and interval bootstrap are mostly one-line or small changes given the existing `pmdarima`/`statsmodels` APIs; the review sometimes presents them as larger efforts than they are.

The recommended implementation sequence is sound and should be followed in roughly the order given. The single highest-leverage change is item 1 (a common backtesting service with identical folds) because it simultaneously fixes Findings 1, 2, 3, 4, and 9, and enables the honest failure states of Finding 5. I would prioritise that above everything else.

## Author reconciliation after independent review

The independent assessment materially strengthens the review. Its verification of all eleven correctness findings supports leaving those findings and their priorities intact. I accept the following refinements:

- State more directly that deterministic policy—not the LLM—must rank models. Giving an LLM consistent metrics is necessary, but it still should not make the numerical decision.
- Treat 12 as a permissible candidate period or prior for apparently monthly business data, never as evidence that seasonality was detected.
- Describe EWMA/SES as a legitimate forecasting method that is under-fitted here, rather than implying that the method itself is merely a weak benchmark.
- Make the AICc recommendation concrete: `pmdarima.auto_arima` supports `information_criterion="aicc"`.
- Treat three-to-five observed seasonal cycles as an uncertainty warning, not a universal exclusion rule.
- State first that cross-family ranking must use out-of-sample results on identical folds; reserve AIC/AICc/BIC for suitable within-family comparisons on the same training sample.
- Preserve trend estimation when the requested seasonal component is not estimable, while explicitly marking seasonal decomposition unavailable.
- Retain distributional residual checks as secondary interval-assumption diagnostics, not point-forecast acceptance tests.
- Clarify that naive and seasonal-naive forecasts are mandatory references and should also be deployable when no complex model adds demonstrated skill. A baseline winning is a useful result and a pipeline warning, not grounds to deploy a worse complex model.
- Add the specific ARIMA refit issue: the refit preserves `order` but not the training model's intercept/trend configuration, so the final fitted model need not be the selected specification.
- Preserve ARIMA/SARIMA holdout interval outputs so coverage and interval scores can be calculated rather than discarding those intervals.
- Define one MASE denominator convention in advance instead of switching from seasonal-naive to one-step-naive based on sample length.
- Replace the arbitrary absolute periodogram tolerance with explicit candidate-period evidence, including harmonics and relative tolerance where useful.

### Corrections to the independent assessment

Several statements in the independent assessment require correction before they become implementation guidance:

1. **A flat SES multi-step point forecast is not a defect.** Proper simple exponential smoothing has a constant point forecast at every horizon, equal to the final estimated level. The current implementation's problems are the fixed rather than estimated `alpha`, use of `pandas.ewm` rather than a fully specified fitted innovations/state-space model, weak validation, and uncalibrated intervals. Replacing it with properly fitted SES will ordinarily retain flat multi-step point forecasts.

2. **The quoted EWMA/SES variance formula is not the relevant multi-step prediction variance.** The assessment states that the one-step variance is `sigma² * alpha / (2-alpha)` and then argues that multi-step variance grows. That expression can describe variance of a smoothed level under particular assumptions; it is not the standard one-step future-observation forecast-error variance for an innovations SES model. Under the usual SES/ARIMA(0,1,1)-without-constant formulation, forecast-error variance depends on the innovation definition and grows with horizon (commonly proportional to `1 + (h-1)alpha²` when `sigma²` denotes innovation variance). The implementation should obtain intervals from the fitted model or simulation rather than hard-code either formula.

3. **`auto_arima` does not default seasonal differencing selection to Canova-Hansen in the installed API.** Its signature defaults to `seasonal_test="ocsb"`; `test="kpss"` controls non-seasonal differencing. The concrete recommendation is still good—set and record these arguments explicitly—but the assessment's claim that the current default is CH is incorrect.

4. **STL cannot cleanly return a standalone STL “trend component” while declaring its seasonal component unestimated without choosing a seasonal smoother/period.** A short-series fallback may use a separate trend smoother or nonseasonal model, but it must not label that output as an STL decomposition for the requested period. Return the requested STL result as `not_estimable` and, if useful, return a separately labeled trend estimate.

5. **`breakvar` is not a general replacement for the current mean-level change-point heuristic.** It is aimed at variance stability. PELT/BinSeg or calibrated CUSUM procedures can address level/regime changes; variance-break diagnostics should be a separate analysis.

6. **Simulation support is available but does not make calibrated intervals automatic.** The installed Statsmodels API exposes `simulate` on Holt-Winters results, so simulation is a practical implementation route. Coverage must still be evaluated out of sample, and parameter uncertainty or residual resampling choices must be documented.

### Revised priority conclusion

The independent reviewer and original review agree on the central decision: implement the common backtesting service first. That service should emit typed fold-level actuals, point predictions, interval bounds, errors, preprocessing provenance, and fit status. Once those objects exist, central metrics, residual diagnostics, interval coverage, honest failure handling, baseline skill, and deterministic model selection become parts of one coherent correction rather than isolated patches.

## Implementation roadmap

The code implementation phases, delivery slices, testing expectations, and cross-phase engineering rules have been moved to [implementation_phases.md](implementation_phases.md).
