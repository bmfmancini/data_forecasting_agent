# Remaining Statistical Improvements

This file contains only unfinished work from the statistical methodology review. Completed implementation history is available in Git and is intentionally not repeated here.

The project is greenfield, so these changes do not require compatibility aliases, deprecated schemas, or migration paths.

## Deferred verification debt

Unit and integration tests were intentionally skipped because of local hardware constraints. Before a release, run the existing suite and add focused coverage for:

- failure states and nullable metrics;
- identical rolling-origin folds across every candidate;
- prevention of future-data leakage;
- horizon aggregation and unsupported horizons;
- interval coverage and ordering;
- deterministic selection, ties, and baseline retention;
- diagnostic evidence states and short/constant series;
- LLM outage and malformed-output behavior.

## Phase 2 — Finish authoritative rolling-origin evaluation

The rolling-origin engine exists, but it is not yet the single source of displayed metrics and model selection.

1. Replace terminal-holdout candidate metrics in `forecasting_agent.py` with pooled rolling-origin metrics for ranking, reports, and `all_metrics`.
2. Run Naive, Seasonal Naive, Mean, and Drift baselines on the same generated folds as the complex models. Do not overwrite rolling scores later with terminal-holdout baseline scores.
3. Make fold fitting call the production adapters or a shared fit/configuration layer so ARIMA bounds, SARIMA settings, Holt-Winters seasonal form, and EWMA alpha match production behavior.
4. Preserve fold prediction intervals from ARIMA and SARIMA.
5. Surface the validation design in output provenance: initial training size, requested/evaluated horizon, unsupported horizons, step, gap, origin cap, successful origins, failed origins, and evaluated observations.
6. If `reserve_final_window` is enabled, evaluate that window once and expose it separately from rolling tuning evidence. Otherwise remove the unused option.
7. Remove the terminal-holdout compatibility path once no production caller depends on it.
8. Ensure runtime caps are explicit in provenance rather than silently reducing horizons or origins.

Exit criteria: every candidate and baseline is ranked using identical rolling folds, and every displayed comparison metric identifies its evaluation design and sample size.

## Phase 3 — Finish uncertainty calibration

Residual diagnostic utilities exist, but production orchestration still primarily analyzes fitted innovations.

1. Feed pooled rolling-origin forecast errors into residual diagnostics for the selected candidate.
2. Supply fold actuals and preserved interval bounds so empirical coverage, width, and Winkler score are calculated.
3. Report diagnostics and interval scores by forecast horizon where sample size permits.
4. Use fitted-model simulation or residual/bootstrap intervals for Holt-Winters.
5. Use a fitted SES/state-space implementation with model- or simulation-based intervals for EWMA/SES.
6. Apply interval calibration only from out-of-sample evidence and record the calibration sample and method.
7. Document whether parameter uncertainty is represented.
8. Do not display a nominal “95%” label for unavailable or experimental intervals.

Exit criteria: selected-model diagnostics use out-of-sample errors when available, interval coverage is operational, and heuristic bands are either replaced or clearly non-nominal.

## Phase 4 — Finish evidence-based diagnostics and fold-safe preprocessing

Typed diagnostic and preprocessing components exist, but legacy diagnostics still run alongside them and transformations are not connected to backtesting.

1. Remove the parallel legacy diagnostic path from `statistical_analysis_agent.py`; make typed evidence the sole downstream input.
2. Propagate `ok`, `not_estimable`, `disabled`, and `failed` statuses through schemas, prompts, selection, review, and reports instead of flattening them into booleans/defaults.
3. Never restore a default period such as 12 when evidence selects period 1 or seasonality is not estimable. Constant series must report no seasonality.
4. Set and record `auto_arima` differencing configuration explicitly, including nonseasonal test, seasonal test, differencing limits/orders, and warnings.
5. Fit imputation, clipping, Box-Cox/log parameters, and seasonal-form choices inside each training fold only.
6. Apply inverse transformations to forecasts and intervals, including an explicit retransformation bias policy.
7. Compare transformed and untransformed pipelines using the same rolling folds; enable a transformation only when it improves the configured loss and satisfies target constraints.
8. Ensure unknown frequency, insufficient cycles, and failed diagnostics cannot become positive seasonality evidence.

Exit criteria: one typed diagnostic pipeline drives decisions, and every data-dependent preprocessing parameter is estimated inside its training fold.

## Phase 5 — Finish deterministic selection and bounded LLM behavior

The deterministic policy exists, but it is not yet authoritative during the normal first forecast pass.

1. Invoke deterministic selection after common rolling-origin evidence is available during every run, not only during retry/review flows.
2. Prefer rolling-origin evidence over terminal-holdout evidence in `CandidateEvidence` and require comparable fold provenance.
3. Pass the configured user/domain loss into ranking rather than using a hard-coded global metric priority.
4. Allow a baseline to be the final production selection and generate its full-history forecast through the same result contract.
5. Remove `APPLY_IQR`, `APPLY_ZSCORE`, `APPLY_BOXCOX`, and similar token-triggered mutations. LLM suggestions must be tested by deterministic backtesting before use.
6. Invoke `validate_llm_output` on every narrative response and attach validation warnings to the result/report.
7. Replace prose-only LLM contracts with structured claims containing evidence references and uncertainty labels.
8. Keep statistical review advisory: it may request a code-recognized retry but cannot override numerical ranking through prose.
9. Add structured context capture for units, decision loss, horizon, interventions, censoring/stockouts, known future covariates, aggregation, and allowable forecast values.
10. Ensure the final `ModelSelectionResult` records the actual deterministic model and evidence rather than the provisional LLM choice.

Exit criteria: identical numerical evidence and policy always select the same model, the pipeline works without an LLM, and narrative text cannot mutate data or silently change rankings.

## Explicitly skipped scope

The following broader capabilities remain intentionally out of scope and are not implementation phases:

- additional model families such as ETS variants, Theta, Prophet, ARIMAX, Fourier regression, intermittent-demand, hierarchical, and ensemble methods;
- production monitoring, champion/challenger operation, drift alerts, and automatic retraining.

## Engineering rules for remaining work

- Prefer typed contracts over nested unvalidated dictionaries.
- Keep numerical computation independent of LLM availability.
- Never rank metrics produced from different fold definitions.
- Record performance-driven reductions in candidates, origins, or horizon.
- Treat missing or failed evidence as unavailable, never as zero or affirmative evidence.
- Require leakage tests for every preprocessing or model-selection feature before release.
