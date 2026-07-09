# Statistical Methodology Review

---

**Project:** Data Forecasting Agent
**Date:** 2026-07-09
**Author:** GitHub Copilot

---

## 1. Introduction

### 1.1 Purpose
This document provides a detailed review of the statistical methodologies implemented in the **Data Forecasting Agent** project. It is intended for statisticians, data scientists, and reviewers to validate the robustness, assumptions, and implementation of the forecasting models.

### 1.2 Scope
This review covers:
- Statistical models used for forecasting.
- Data preprocessing techniques.
- Model selection criteria.
- Validation metrics and residual analysis.
- Strengths and limitations of the implemented methodologies.
- Recommendations for improvement.

### 1.3 Audience
- **Statisticians:** For validation of methodological rigor.
- **Data Scientists:** For understanding model implementation and assumptions.
- **Reviewers:** For assessing the suitability of the models for forecasting tasks.

---

## 2. Overview of Statistical Models
The project implements the following statistical forecasting models:

1. **ARIMA (AutoRegressive Integrated Moving Average)**
2. **SARIMA (Seasonal ARIMA)**
3. **Holt-Winters Exponential Smoothing**
4. **EWMA (Exponentially Weighted Moving Average)**

---

## 3. Detailed Methodology

### 3.1 Data Preprocessing

#### 3.1.1 Handling Missing Values
- Missing values in the input time series are handled using `series.dropna().astype(float)`. This ensures that the series is clean and numeric before model fitting.

#### 3.1.2 Train-Test Split
- An 80%/20% train-test split is used for model validation. The split is performed as follows:
  ```python
  split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
  ```
- This ensures that the test set is large enough to evaluate the forecast accuracy.

#### 3.1.3 Seasonal Period Inference
- The seasonal period is inferred using the `_infer_seasonal_period` function, which inspects the `series.index.freq` attribute:
  - Monthly data → 12
  - Quarterly data → 4
  - Weekly data → 52
  - Daily data → 7
  - Default → 12

---

### 3.2 ARIMA (AutoRegressive Integrated Moving Average)

#### 3.2.1 Mathematical Formulation
The ARIMA model is defined by three parameters: `(p, d, q)`:
- **p:** Order of the autoregressive (AR) term.
- **d:** Degree of differencing required to make the time series stationary.
- **q:** Order of the moving average (MA) term.

The model is represented as:
$$
\phi(B)(1-B)^d y_t = \theta(B) \epsilon_t
$$
where:
- $ B $ is the backshift operator.
- $ \phi(B) $ is the AR polynomial.
- $ \theta(B) $ is the MA polynomial.
- $ \epsilon_t $ is white noise.

#### 3.2.2 Assumptions
- The time series is stationary or can be made stationary through differencing.
- Residuals are normally distributed with mean zero and constant variance.

#### 3.2.3 Model Selection
- The `pmdarima.auto_arima` function is used with `stepwise=True` and AIC selection to automatically determine the optimal `(p, d, q)` parameters.

#### 3.2.4 Validation Metrics
- **RMSE (Root Mean Squared Error):** Measures the square root of the average squared differences between predicted and observed values.
- **MAE (Mean Absolute Error):** Measures the average absolute differences between predicted and observed values.
- **MAPE (Mean Absolute Percentage Error):** Measures the average absolute percentage differences between predicted and observed values.

#### 3.2.5 Implementation Details
- **Library:** `pmdarima`
- **Key Function:** `fit_arima(series, forecast_horizon)`
- **Output:** A dictionary containing `forecast`, `lower_ci`, `upper_ci`, `rmse`, `mae`, and `mape`.

#### 3.2.6 Strengths and Limitations
- **Strengths:**
  - Automated parameter selection.
  - Works well for non-seasonal time series.
- **Limitations:**
  - Assumes stationarity.
  - Sensitive to outliers.

---

### 3.3 SARIMA (Seasonal ARIMA)

#### 3.3.1 Mathematical Formulation
The SARIMA model extends ARIMA by adding seasonal terms: `(p, d, q)(P, D, Q)[m]`:
- **P:** Seasonal AR term.
- **D:** Seasonal differencing term.
- **Q:** Seasonal MA term.
- **m:** Seasonal period (e.g., 12 for monthly data).

The model is represented as:
$$
\phi(B)\Phi(B^m)(1-B)^d(1-B^m)^D y_t = \theta(B)\Theta(B^m) \epsilon_t
$$
where:
- $ \Phi(B^m) $ is the seasonal AR polynomial.
- $ \Theta(B^m) $ is the seasonal MA polynomial.

#### 3.3.2 Assumptions
- The time series exhibits seasonality.
- The seasonal component is stationary or can be made stationary through differencing.

#### 3.3.3 Model Selection
- The `pmdarima.auto_arima` function is used with `seasonal=True` and `m=seasonal_period` to automatically determine the optimal `(p, d, q)(P, D, Q)[m]` parameters.

#### 3.3.4 Validation Metrics
- Same as ARIMA: RMSE, MAE, MAPE.

#### 3.3.5 Implementation Details
- **Library:** `pmdarima`
- **Key Function:** `fit_sarima(series, forecast_horizon, seasonal_period=12)`
- **Output:** A dictionary containing `forecast`, `lower_ci`, `upper_ci`, `rmse`, `mae`, and `mape`.
- **Fallback:** If the series length is less than two full seasonal cycles, the model falls back to non-seasonal ARIMA.

#### 3.3.6 Strengths and Limitations
- **Strengths:**
  - Captures both non-seasonal and seasonal patterns.
  - Automated parameter selection.
- **Limitations:**
  - Computationally intensive.
  - Requires sufficient data for seasonal patterns.

---

### 3.4 Holt-Winters Exponential Smoothing

#### 3.4.1 Mathematical Formulation
The Holt-Winters model is defined by three components:
- **Level ($ \ell_t $):** Smoothing of the series.
- **Trend ($ b_t $):** Smoothing of the trend.
- **Seasonality ($ s_t $):** Smoothing of the seasonal component.

The model supports two variants:
1. **Additive Seasonality:**
   $$
   \hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)}
   $$
2. **Multiplicative Seasonality:**
   $$
   \hat{y}_{t+h|t} = (\ell_t + h b_t) s_{t+h-m(k+1)}
   $$

#### 3.4.2 Assumptions
- The time series exhibits trend and/or seasonality.
- The seasonal component is either additive or multiplicative.

#### 3.4.3 Model Selection
- The `statsmodels.tsa.holtwinters.ExponentialSmoothing` function is used with additive or multiplicative seasonality based on AIC.

#### 3.4.4 Validation Metrics
- Same as ARIMA: RMSE, MAE, MAPE.

#### 3.4.5 Implementation Details
- **Library:** `statsmodels`
- **Key Function:** `fit_holt_winters(series, forecast_horizon)`
- **Output:** A dictionary containing `forecast`, `lower_ci`, `upper_ci`, `rmse`, `mae`, and `mape`.
- **Confidence Intervals:** Constructed using residual standard deviation.

#### 3.4.6 Strengths and Limitations
- **Strengths:**
  - Captures trend and seasonality.
  - Works well for short-term forecasting.
- **Limitations:**
  - Assumes linear trend.
  - Sensitive to initial values.

---

### 3.5 EWMA (Exponentially Weighted Moving Average)

#### 3.5.1 Mathematical Formulation
The EWMA model assigns exponentially decreasing weights to past observations:
$$
\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t
$$
where:
- $ \alpha $ is the smoothing factor ($ 0 < \alpha < 1 $).

#### 3.5.2 Assumptions
- The time series is stationary.
- Recent observations are more relevant than older ones.

#### 3.5.3 Model Selection
- The smoothing factor $ \alpha $ is typically set based on domain knowledge or optimized using grid search.

#### 3.5.4 Validation Metrics
- Same as ARIMA: RMSE, MAE, MAPE.

#### 3.5.5 Implementation Details
- **Library:** Custom implementation (or `pandas`)
- **Key Function:** `fit_ewma(series, forecast_horizon)`
- **Output:** A dictionary containing `forecast`, `lower_ci`, `upper_ci`, `rmse`, `mae`, and `mape`.

#### 3.5.6 Strengths and Limitations
- **Strengths:**
  - Simple and computationally efficient.
  - Works well for stationary time series.
- **Limitations:**
  - Does not capture trend or seasonality.
  - Sensitive to the choice of $ \alpha $.

---

## 4. Model Validation

### 4.1 Metrics
The following metrics are used to validate model performance:
- **RMSE (Root Mean Squared Error):**
  $$
  RMSE = \sqrt{\frac{1}{n} \sum_{t=1}^n (y_t - \hat{y}_t)^2}
  $$
- **MAE (Mean Absolute Error):**
  $$
  MAE = \frac{1}{n} \sum_{t=1}^n |y_t - \hat{y}_t|
  $$
- **MAPE (Mean Absolute Percentage Error):**
  $$
  MAPE = \frac{100}{n} \sum_{t=1}^n \left| \frac{y_t - \hat{y}_t}{y_t} \right|
  $$

### 4.2 Confidence Intervals
- 95% confidence intervals are constructed for all forecasts using residual standard deviation.

### 4.3 Residual Analysis
- Residuals are logged and analyzed for patterns. If residuals exhibit structure, the model is reconsidered.

---

## 5. Error Handling and Logging

### 5.1 Error Handling
- All model fitting and metric calculations are wrapped in `try/except` blocks.
- On exception, a warning is logged (`logger.warning`), and metric values are set to `float('nan')`.
- Custom exceptions from `backend.exceptions` are raised for validation errors (e.g., insufficient data length).

### 5.2 Logging
- The project uses `utils.logging_config.get_logger` for consistent logging.
- Key events logged:
  - Model selection (e.g., ARIMA order).
  - Errors and warnings.
  - Fallback behavior (e.g., SARIMA to ARIMA).

---

## 6. Strengths and Limitations of the Methodology

### 6.1 Strengths
- **Automated Model Selection:** Uses `auto_arima` and AIC for optimal parameter selection.
- **Seasonal and Non-Seasonal Support:** Implements both ARIMA and SARIMA for flexibility.
- **Robust Error Handling:** Graceful fallback mechanisms and logging.
- **Validation Metrics:** Comprehensive metrics (RMSE, MAE, MAPE) for model evaluation.

### 6.2 Limitations
- **Assumptions:** Models assume stationarity, linearity, or specific seasonal patterns.
- **Data Requirements:** SARIMA requires at least two full seasonal cycles for reliable results.
- **Computational Complexity:** SARIMA and Holt-Winters can be computationally intensive.
- **Sensitivity to Outliers:** Models may perform poorly in the presence of outliers.

---

## 7. Recommendations for Improvement

1. **Enhance Model Robustness:**
   - Incorporate outlier detection and handling (e.g., using robust statistics).
   - Explore hybrid models (e.g., ARIMA + machine learning).

2. **Improve Validation:**
   - Add cross-validation for time series data.
   - Include additional metrics (e.g., R², AIC, BIC).

3. **Expand Model Support:**
   - Add support for Prophet, LSTM, or other modern forecasting techniques.

4. **Optimize Computational Efficiency:**
   - Implement parallel model fitting for large datasets.

5. **Enhance Interpretability:**
   - Provide visualizations of model components (e.g., trend, seasonality).

---

## 8. References

1. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*. OTexts.
2. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
3. `pmdarima` Documentation: [https://alkaline-ml.com/pmdarima/](https://alkaline-ml.com/pmdarima/)
4. `statsmodels` Documentation: [https://www.statsmodels.org/](https://www.statsmodels.org/)
5. Pandas Documentation: [https://pandas.pydata.org/](https://pandas.pydata.org/)

---

## 9. Appendix

### 9.1 Code Snippets

#### 9.1.1 ARIMA Implementation
```python
from pmdarima import auto_arima
import pandas as pd

def fit_arima(series: pd.Series, forecast_horizon: int) -> dict:
    series = series.dropna().astype(float)
    split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
    train, test = series[:split], series[split:]
    
    model = auto_arima(train, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=forecast_horizon)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test, forecast[:len(test)]))
    mae = mean_absolute_error(test, forecast[:len(test)])
    mape = mean_absolute_percentage_error(test, forecast[:len(test)])
    
    return {
        "forecast": forecast,
        "lower_ci": forecast - 1.96 * rmse,
        "upper_ci": forecast + 1.96 * rmse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }
```

#### 9.1.2 Holt-Winters Implementation
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_holt_winters(series: pd.Series, forecast_horizon: int) -> dict:
    series = series.dropna().astype(float)
    split = max(int(len(series) * 0.8), len(series) - forecast_horizon)
    train, test = series[:split], series[split:]
    
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=_infer_seasonal_period(series),
    ).fit()
    
    forecast = model.forecast(forecast_horizon)
    residuals = test - model.fittedvalues
    rmse = np.sqrt(mean_squared_error(test, model.fittedvalues))
    
    return {
        "forecast": forecast,
        "lower_ci": forecast - 1.96 * rmse,
        "upper_ci": forecast + 1.96 * rmse,
        "rmse": rmse,
        "mae": mean_absolute_error(test, model.fittedvalues),
        "mape": mean_absolute_percentage_error(test, model.fittedvalues),
    }
```

---

**End of Document.**