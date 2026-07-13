# Statistical Improvements — Verification Remaining

The production implementation for Phases 1–5 is complete. This file contains only verification work intentionally deferred because the local machine is not suitable for the full forecasting test suite.

## Deferred verification

Before release, run the complete unit and integration suite on appropriately provisioned hardware and add focused coverage for:

- failure states and nullable metrics;
- identical rolling-origin folds across complex models and baselines;
- fold-safe imputation, clipping, Box-Cox fitting, and inverse transformation;
- requested, evaluated, and unsupported horizons;
- failed-origin exclusion and one-based horizon aggregation;
- out-of-sample residual diagnostics and interval coverage by horizon;
- bootstrap interval ordering and reproducibility;
- deterministic loss selection, simplicity ties, and baseline retention;
- typed diagnostic statuses for short, constant, seasonal, and nonseasonal series;
- malformed LLM narratives, invented claims, and complete LLM outages;
- forced-model behavior and typed statistical-review overrides;
- end-to-end report and visualization handling of unavailable metrics and intervals.

## Completed production behavior

- Rolling-origin metrics are authoritative and carry auditable validation provenance.
- Complex candidates and simple baselines use common folds.
- Failed folds cannot contaminate pooled scores.
- Model selection is deterministic, honors the configured loss, and can retain a baseline.
- LLM output is advisory, validated, and cannot trigger data mutations.
- Statistical analysis uses one typed evidence pipeline with explicit statuses and warnings.
- ARIMA/SARIMA differencing tests are explicit and recorded.
- Residual diagnostics prefer out-of-sample forecast errors and score intervals by horizon.
- SES uses a fitted state-space model; SES and Holt-Winters use bootstrap prediction intervals.
- Empirical interval calibration is applied only when rolling evidence is available.
- IQR clipping is fitted within each training fold when explicitly requested.
- A skew-triggered Box-Cox ARIMA pipeline is compared on the same folds and inverted to the original target scale.
- High-value forecast context is captured during preflight and attached to selection evidence.
- Holt-Winters consumes the typed seasonal period and treats period 1 as nonseasonal; it no longer independently defaults unknown frequency to 12.
- Holt-Winters selects no-trend, additive-trend, damped-trend, and admissible seasonal forms by training-window AICc.
- Rolling Holt-Winters folds and the production refit use the same model-form selector.
- Holt-Winters configuration records the requested/used seasonal period, selection scope, criterion, initialization, and parameter-uncertainty limitation.

## Explicitly skipped scope

- Additional model families such as ETS variants, Theta, Prophet, ARIMAX, Fourier regression, intermittent-demand, hierarchical, and ensemble methods.
- Production monitoring, champion/challenger operation, drift alerts, and automatic retraining.
