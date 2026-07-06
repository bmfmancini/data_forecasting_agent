"""Unit tests for data_cleaning.py module."""

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_forecaster.backend.utils.data_cleaning import (
    apply_iqr_clipping,
    apply_zscore_clipping,
    audit_series,
    detect_outliers_iqr,
    detect_outliers_zscore,
    impute_missing,
    reindex_series,
    resolve_duplicates,
    smooth_series,
    treat_outliers,
    validate_schema,
    _infer_seasonal_period,
)


@pytest.fixture
def sample_series():
    """Fixture for a sample time series."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.normal(loc=100, scale=10, size=100).clip(0, 200)
    return pd.Series(values, index=dates)


@pytest.fixture
def series_with_missing():
    """Fixture for a time series with missing values."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.normal(loc=100, scale=10, size=100).clip(0, 200)
    series = pd.Series(values, index=dates)
    series.iloc[10:20] = np.nan
    series.iloc[50:55] = np.nan
    return series


@pytest.fixture
def series_with_duplicates():
    """Fixture for a time series with duplicate timestamps."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.normal(loc=100, scale=10, size=100).clip(0, 200)
    series = pd.Series(values, index=dates)
    # Add duplicates
    duplicate_dates = pd.date_range(start="2023-01-10", periods=5, freq="D")
    duplicate_values = np.random.normal(loc=100, scale=10, size=5).clip(0, 200)
    duplicate_series = pd.Series(duplicate_values, index=duplicate_dates)
    return pd.concat([series, duplicate_series])


@pytest.fixture
def series_with_outliers():
    """Fixture for a time series with outliers."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.normal(loc=100, scale=10, size=100).clip(0, 200)
    series = pd.Series(values, index=dates)
    # Add outliers
    series.iloc[10] = 500
    series.iloc[50] = -300
    return series


@pytest.fixture
def irregular_series():
    """Fixture for an irregular time series."""
    dates = pd.to_datetime([
        "2023-01-01", "2023-01-02", "2023-01-04", "2023-01-05", "2023-01-07",
        "2023-01-08", "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-15"
    ])
    values = np.random.normal(loc=100, scale=10, size=10).clip(0, 200)
    return pd.Series(values, index=dates)


# Test audit_series
class TestAuditSeries:
    def test_audit_series(self, sample_series):
        """Test audit_series with a regular series."""
        result = audit_series(sample_series)
        assert result["length"] == 100
        assert result["missing"] == 0
        assert result["duplicate_timestamps"] == 0
        assert result["irregular"] is False
        assert result["freq"] == "D"
        assert isinstance(result["outlier_counts"], dict)

    def test_audit_series_with_missing(self, series_with_missing):
        """Test audit_series with a series containing missing values."""
        result = audit_series(series_with_missing)
        assert result["missing"] > 0

    def test_audit_series_with_duplicates(self, series_with_duplicates):
        """Test audit_series with a series containing duplicate timestamps."""
        result = audit_series(series_with_duplicates)
        assert result["duplicate_timestamps"] > 0

    def test_audit_series_irregular(self, irregular_series):
        """Test audit_series with an irregular series."""
        result = audit_series(irregular_series)
        assert result["irregular"] is True

    def test_audit_series_invalid_index(self):
        """Test audit_series with an invalid index."""
        series = pd.Series([1, 2, 3], index=[1, 2, 3])
        with pytest.raises(ValueError, match="Series index must be a pandas DatetimeIndex"):
            audit_series(series)


# Test reindex_series
class TestReindexSeries:
    def test_reindex_series(self, sample_series):
        """Test reindex_series with a regular series."""
        reindexed = reindex_series(sample_series, "D")
        assert len(reindexed) == len(sample_series)
        assert reindexed.index.freq == to_offset("D")

    def test_reindex_series_irregular(self, irregular_series):
        """Test reindex_series with an irregular series."""
        reindexed = reindex_series(irregular_series, "D")
        assert len(reindexed) > len(irregular_series)
        assert reindexed.index.freq == to_offset("D")

    def test_reindex_series_invalid_index(self):
        """Test reindex_series with an invalid index."""
        series = pd.Series([1, 2, 3], index=[1, 2, 3])
        with pytest.raises(ValueError, match="Series index must be a pandas DatetimeIndex"):
            reindex_series(series, "D")


# Test impute_missing
class TestImputeMissing:
    def test_impute_missing_forward_fill(self, series_with_missing):
        """Test impute_missing with forward-fill method."""
        imputed = impute_missing(series_with_missing, "forward-fill")
        assert imputed.isna().sum() == 0

    def test_impute_missing_interpolate(self, series_with_missing):
        """Test impute_missing with interpolate method."""
        imputed = impute_missing(series_with_missing, "interpolate")
        assert imputed.isna().sum() == 0

    def test_impute_missing_seasonal_decompose(self, series_with_missing):
        """Test impute_missing with seasonal-decompose method."""
        imputed = impute_missing(series_with_missing, "seasonal-decompose")
        assert imputed.isna().sum() == 0

    def test_impute_missing_unsupported_method(self, series_with_missing):
        """Test impute_missing with an unsupported method."""
        with pytest.raises(ValueError, match="Unsupported imputation method"):
            impute_missing(series_with_missing, "unsupported-method")


# Test detect_outliers_iqr
class TestDetectOutliersIQR:
    def test_detect_outliers_iqr(self, sample_series):
        """Test detect_outliers_iqr with a normal series."""
        result = detect_outliers_iqr(sample_series)
        assert isinstance(result["count"], int)
        assert isinstance(result["ratio"], float)
        assert isinstance(result["lower_bound"], float)
        assert isinstance(result["upper_bound"], float)
        assert isinstance(result["interpretation"], str)

    def test_detect_outliers_iqr_with_outliers(self, series_with_outliers):
        """Test detect_outliers_iqr with a series containing outliers."""
        result = detect_outliers_iqr(series_with_outliers)
        assert result["count"] > 0


# Test detect_outliers_zscore
class TestDetectOutliersZScore:
    def test_detect_outliers_zscore(self, sample_series):
        """Test detect_outliers_zscore with a normal series."""
        result = detect_outliers_zscore(sample_series)
        assert isinstance(result["count"], int)
        assert isinstance(result["ratio"], float)
        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)
        assert isinstance(result["lower_bound"], float)
        assert isinstance(result["upper_bound"], float)
        assert isinstance(result["interpretation"], str)

    def test_detect_outliers_zscore_with_outliers(self, series_with_outliers):
        """Test detect_outliers_zscore with a series containing outliers."""
        result = detect_outliers_zscore(series_with_outliers)
        assert result["count"] > 0

    def test_detect_outliers_zscore_constant_series(self):
        """Test detect_outliers_zscore with a constant series."""
        series = pd.Series([10, 10, 10], index=pd.date_range(start="2023-01-01", periods=3, freq="D"))
        result = detect_outliers_zscore(series)
        assert result["count"] == 0
        assert "constant series" in result["interpretation"]


# Test apply_iqr_clipping
class TestApplyIQRClipping:
    def test_apply_iqr_clipping(self, series_with_outliers):
        """Test apply_iqr_clipping with a series containing outliers."""
        clipped = apply_iqr_clipping(series_with_outliers)
        outliers = series_with_outliers[(series_with_outliers < clipped.min()) | (series_with_outliers > clipped.max())]
        assert len(outliers) > 0
        assert (clipped >= clipped.min()).all() and (clipped <= clipped.max()).all()


# Test apply_zscore_clipping
class TestApplyZScoreClipping:
    def test_apply_zscore_clipping(self, series_with_outliers):
        """Test apply_zscore_clipping with a series containing outliers."""
        clipped = apply_zscore_clipping(series_with_outliers)
        assert (clipped >= clipped.min()).all() and (clipped <= clipped.max()).all()

    def test_apply_zscore_clipping_constant_series(self):
        """Test apply_zscore_clipping with a constant series."""
        series = pd.Series([10, 10, 10], index=pd.date_range(start="2023-01-01", periods=3, freq="D"))
        clipped = apply_zscore_clipping(series)
        assert (clipped == series).all()


# Test treat_outliers
class TestTreatOutliers:
    def test_treat_outliers_clip(self, series_with_outliers):
        """Test treat_outliers with clip strategy."""
        treated = treat_outliers(series_with_outliers, "clip")
        assert (treated >= treated.min()).all() and (treated <= treated.max()).all()

    def test_treat_outliers_winsorize(self, series_with_outliers):
        """Test treat_outliers with winsorize strategy."""
        treated = treat_outliers(series_with_outliers, "winsorize")
        assert (treated >= treated.min()).all() and (treated <= treated.max()).all()

    def test_treat_outliers_zscore_clip(self, series_with_outliers):
        """Test treat_outliers with zscore_clip strategy."""
        treated = treat_outliers(series_with_outliers, "zscore_clip")
        assert (treated >= treated.min()).all() and (treated <= treated.max()).all()

    def test_treat_outliers_remove(self, series_with_outliers):
        """Test treat_outliers with remove strategy."""
        treated = treat_outliers(series_with_outliers, "remove")
        assert treated.isna().sum() > 0

    def test_treat_outliers_none(self, series_with_outliers):
        """Test treat_outliers with none strategy."""
        treated = treat_outliers(series_with_outliers, "none")
        assert (treated == series_with_outliers).all()

    def test_treat_outliers_unsupported_strategy(self, series_with_outliers):
        """Test treat_outliers with an unsupported strategy."""
        with pytest.raises(ValueError, match="Unsupported outlier strategy"):
            treat_outliers(series_with_outliers, "unsupported-strategy")


# Test resolve_duplicates
class TestResolveDuplicates:
    def test_resolve_duplicates_keep_first(self, series_with_duplicates):
        """Test resolve_duplicates with keep-first strategy."""
        resolved = resolve_duplicates(series_with_duplicates, "keep-first")
        assert not resolved.index.has_duplicates

    def test_resolve_duplicates_latest(self, series_with_duplicates):
        """Test resolve_duplicates with latest strategy."""
        resolved = resolve_duplicates(series_with_duplicates, "latest")
        assert not resolved.index.has_duplicates

    def test_resolve_duplicates_mean(self, series_with_duplicates):
        """Test resolve_duplicates with mean strategy."""
        resolved = resolve_duplicates(series_with_duplicates, "mean")
        assert not resolved.index.has_duplicates

    def test_resolve_duplicates_no_duplicates(self, sample_series):
        """Test resolve_duplicates with a series without duplicates."""
        resolved = resolve_duplicates(sample_series, "keep-first")
        assert (resolved == sample_series).all()

    def test_resolve_duplicates_unsupported_strategy(self, series_with_duplicates):
        """Test resolve_duplicates with an unsupported strategy."""
        with pytest.raises(ValueError, match="Unsupported duplicate strategy"):
            resolve_duplicates(series_with_duplicates, "unsupported-strategy")


# Test smooth_series
class TestSmoothSeries:
    def test_smooth_series_ewma(self, sample_series):
        """Test smooth_series with ewma method."""
        smoothed = smooth_series(sample_series, "ewma", span=6)
        assert len(smoothed) == len(sample_series)

    def test_smooth_series_savgol(self, sample_series):
        """Test smooth_series with savgol method."""
        smoothed = smooth_series(sample_series, "savgol", window=11, polyorder=2)
        assert len(smoothed) == len(sample_series)

    def test_smooth_series_none(self, sample_series):
        """Test smooth_series with none method."""
        smoothed = smooth_series(sample_series, "none")
        assert (smoothed == sample_series).all()

    def test_smooth_series_unsupported_method(self, sample_series):
        """Test smooth_series with an unsupported method."""
        with pytest.raises(ValueError, match="Unsupported smoothing method"):
            smooth_series(sample_series, "unsupported-method")


# Test validate_schema
class TestValidateSchema:
    def test_validate_schema(self, sample_series):
        """Test validate_schema with a valid schema."""
        config = {
            "expected_freq": "D",
            "max_missing_rate": 0.05,
            "min_value": 0,
            "max_value": 200,
        }
        result = validate_schema(sample_series, config)
        assert result["freq_regular"] is True
        assert result["missing_below_threshold"] is True
        assert result["values_in_range"] is True
        assert result["no_duplicates"] is True
        assert result["index_monotonic"] is True

    def test_validate_schema_invalid_freq(self, irregular_series):
        """Test validate_schema with an invalid frequency."""
        config = {
            "expected_freq": "D",
            "max_missing_rate": 0.05,
            "min_value": 0,
            "max_value": 200,
        }
        result = validate_schema(irregular_series, config)
        assert result["freq_regular"] is False

    def test_validate_schema_missing_above_threshold(self, series_with_missing):
        """Test validate_schema with missing values above threshold."""
        config = {
            "expected_freq": "D",
            "max_missing_rate": 0.01,
            "min_value": 0,
            "max_value": 200,
        }
        result = validate_schema(series_with_missing, config)
        assert result["missing_below_threshold"] is False

    def test_validate_schema_out_of_range(self, series_with_outliers):
        """Test validate_schema with values out of range."""
        config = {
            "expected_freq": "D",
            "max_missing_rate": 0.05,
            "min_value": 0,
            "max_value": 200,
        }
        result = validate_schema(series_with_outliers, config)
        assert result["values_in_range"] is False


# Test _infer_seasonal_period
class TestInferSeasonalPeriod:
    def test_infer_seasonal_period_daily(self):
        """Test _infer_seasonal_period with daily frequency."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        series = pd.Series(range(100), index=dates)
        assert _infer_seasonal_period(series) == 7

    def test_infer_seasonal_period_weekly(self):
        """Test _infer_seasonal_period with weekly frequency."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="W")
        series = pd.Series(range(100), index=dates)
        assert _infer_seasonal_period(series) == 52

    def test_infer_seasonal_period_monthly(self):
        """Test _infer_seasonal_period with monthly frequency."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="MS")
        series = pd.Series(range(100), index=dates)
        assert _infer_seasonal_period(series) == 12

    def test_infer_seasonal_period_quarterly(self):
        """Test _infer_seasonal_period with quarterly frequency."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="QS")
        series = pd.Series(range(100), index=dates)
        assert _infer_seasonal_period(series) == 4

    def test_infer_seasonal_period_unknown(self):
        """Test _infer_seasonal_period with unknown frequency."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
        series = pd.Series(range(100), index=dates)
        assert _infer_seasonal_period(series) == 12

    def test_infer_seasonal_period_no_freq(self):
        """Test _infer_seasonal_period with no frequency."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04"])
        series = pd.Series(range(3), index=dates)
        assert _infer_seasonal_period(series) == 12