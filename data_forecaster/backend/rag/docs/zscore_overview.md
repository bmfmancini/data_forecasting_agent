# Z-Score Outlier Detection Overview

## What is Z-Score?
The **z‑score** (standard score) measures how many standard deviations a data point is away from the mean of a distribution:

```
z = (x - μ) / σ
```
- **x** – the data point value
- **μ** – mean of the series
- **σ** – standard deviation of the series

A high absolute z‑score indicates that the point is far from the typical range of the data and may be considered an outlier.

## Why Use Z‑Score for Outlier Detection?
- **Assumes Normal Distribution** – Works best when the data is approximately normally distributed (low skewness and kurtosis).
- **Simple Threshold** – A common threshold is `|z| > 3`, which corresponds to a 99.7% confidence interval for a normal distribution.
- **Scale‑Invariant** – Works regardless of the magnitude of the data because it normalizes by the standard deviation.

## When to Prefer Z‑Score Over IQR
| Situation | Recommended Method |
|-----------|-------------------|
| Data is roughly symmetric with low skewness/kurtosis | **Z‑Score** |
| Data has heavy tails or is highly skewed | **IQR** |
| You need a conservative approach that flags fewer points | **Z‑Score** |
| You need a method that works well for normally distributed series | **Z‑Score** |

## Implementation in the Forecasting Agent
The project now includes two outlier detection utilities in `backend/utils/statistical.py`:

```python
from utils.statistical import detect_outliers_zscore, apply_zscore_clipping
```
- `detect_outliers_zscore(series, threshold=3.0)` returns a dictionary with:
  - `count`, `ratio`, `mean`, `std`
  - `lower_bound`, `upper_bound`
  - Human‑readable `interpretation`
- `apply_zscore_clipping(series, threshold=3.0)` clips values to the calculated bounds (Winsorization).

## Decision Logic in the Agent
The statistical analysis agent now:
1. Computes both IQR and Z‑Score outlier metrics.
2. Calculates **skewness** and **kurtosis** of the series.
3. Chooses Z‑Score when:
   - `|skewness| < 1.0` **and** `|kurtosis| < 3.0`
   - Z‑Score detects **fewer or equal** outliers compared to IQR.
4. Otherwise, it falls back to IQR.

The LLM can still override this heuristic by explicitly returning `APPLY_ZSCORE` or `APPLY_IQR` in its response.

## How to Use It
```python
# Detect outliers
result = detect_outliers_zscore(series)
print(result["interpretation"])

# Apply clipping if needed
clean_series = apply_zscore_clipping(series)
```

## References
- **Standard Score** – https://en.wikipedia.org/wiki/Standard_score
- **Outlier Detection** – https://en.wikipedia.org/wiki/Outlier
- **Statistical Analysis Agent** – See `backend/agents/statistical_analysis_agent.py`
