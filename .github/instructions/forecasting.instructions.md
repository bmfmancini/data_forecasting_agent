---
name: statistical-forecasting-instructions
applyTo:
  - "**/backend/forecasting/**/*.py"
description: |
  Guidelines for implementing statistical forecasting models in the `backend/forecasting` package.
  
  ### General Requirements
  - All functions must use **type hints** for parameters and return values.
  - Follow **PEP8** formatting (4‑space indent, max line length 88).
  - Imports must be ordered: standard library → third‑party → local modules; no wildcard imports.
  - Use the project's logging configuration via `utils.logging_config.get_logger`.
  - Raise custom exceptions from `backend.exceptions` for validation errors (e.g., insufficient data length).
  - Write docstrings in **Google style** describing parameters, return dict keys, and possible exceptions.
  - Ensure every public function has accompanying unit tests.
  
  ### Model Functions
  - **`fit_arima(series, forecast_horizon)`**
    - Accepts a `pd.Series` (numeric, cleaned with `dropna().astype(float)`).
    - Performs an 80 %/20 % train‑test split for metric calculation (RMSE, MAE, MAPE).
    - Uses `pmdarima.auto_arima` (non‑seasonal) with `stepwise=True` and AIC selection.
    - Returns a dict with keys: `forecast`, `lower_ci`, `upper_ci`, `rmse`, `mae`, `mape`.
    - Logs selected order with `logger.info`.
  
  - **`fit_holt_winters(series, forecast_horizon)`**
    - Determines seasonal period via `_infer_seasonal_period` (monthly → 12, quarterly → 4, weekly → 52, daily → 7, default 12).
    - Uses `statsmodels.tsa.holtwinters.ExponentialSmoothing` with additive trend and either additive or multiplicative seasonal component based on AIC.
    - Computes metrics on the same train‑test split as ARIMA.
    - Constructs 95 % confidence intervals using residual standard deviation.
    - Returns the same dict structure as ARIMA.
  
  - **`fit_sarima(series, forecast_horizon, seasonal_period=12)`**
    - Validates that the series length is at least two full seasonal cycles; otherwise falls back to non‑seasonal ARIMA and logs a warning.
    - Calls `pmdarima.auto_arima` with `seasonal=True` and `m=seasonal_period`.
    - Provides the same metric dict as the other models.
  
  ### Helper Functions
  - **`_infer_seasonal_period(series)`**
    - Inspects `series.index.freq` to infer appropriate seasonal period.
    - Returns an integer (default 12).
  
  ### Error Handling & Logging
  - Wrap model fitting and metric calculation in `try/except` blocks.
  - On exception, log a warning (`logger.warning`) and set metric values to `float('nan')`.
  - Do not let exceptions propagate to the caller unless they are custom validation errors.
  
  ### Testing Guidance
  - Unit tests should cover:
    - Correct handling of missing values.
    - Proper train‑test split logic.
    - Return dict contains all required keys with correct types.
    - Logging of selected model order.
    - Fallback behavior in `fit_sarima` when series is too short.
---
