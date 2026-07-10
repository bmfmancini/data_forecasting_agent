"""Statistical utilities for performing residual analysis.

Provides a single function to run a battery of tests on model residuals
to check for whiteness (zero mean, no autocorrelation, normally distributed).
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import shapiro, ttest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox

from schemas import ResidualDiagnostics

_ZERO_MEAN_P_THRESHOLD = 0.05
_AUTOCORRELATION_P_THRESHOLD = 0.05
_NORMALITY_P_THRESHOLD = 0.05


def analyze_residuals(residuals: pd.Series) -> ResidualDiagnostics:
    """Perform a series of diagnostic tests on model residuals.

    Args:
        residuals: The difference between actual and predicted values.

    Returns:
        A :class:`ResidualDiagnostics` object with test results.
    """
    # 1. Zero Mean Check (One-sample t-test)
    # The null hypothesis is that the expected value (mean) of the sample is zero.
    # We want a high p-value to fail to reject the null.
    mean = residuals.mean()
    is_zero_mean = ttest_1samp(residuals, 0).pvalue >= _ZERO_MEAN_P_THRESHOLD

    # 2. Autocorrelation (Ljung-Box Test)
    # The null hypothesis is that the data are independently distributed.
    # We want a high p-value to fail to reject the null.
    ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    ljung_box_p_value = ljung_box_result["lb_pvalue"].iloc[0]
    is_uncorrelated = ljung_box_p_value >= _AUTOCORRELATION_P_THRESHOLD

    # 3. Normality (Shapiro-Wilk Test)
    # The null hypothesis is that the data was drawn from a normal distribution.
    # We want a high p-value to fail to reject the null.
    shapiro_stat, shapiro_p = shapiro(residuals)
    is_normal = shapiro_p >= _NORMALITY_P_THRESHOLD

    return ResidualDiagnostics(
        mean=mean,
        is_zero_mean=is_zero_mean,
        ljung_box_p_value=ljung_box_p_value,
        is_uncorrelated=is_uncorrelated,
        shapiro_wilk_p_value=shapiro_p,
        is_normal=is_normal,
    )