"""Tests for z-score outlier detection and clipping functions."""

import numpy as np
import pandas as pd
import pytest

from utils.data_cleaning import detect_outliers_zscore, apply_zscore_clipping


class TestZScoreOutliers:
    """Test z-score outlier detection and clipping functions."""

    def test_detect_outliers_zscore_normal_distribution(self):
        """Test z-score outlier detection with normally distributed data."""
        # Create normally distributed data with a few outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        # Add a few outliers
        data_with_outliers = np.concatenate([normal_data, [5.0, -5.0, 6.0]])

        series = pd.Series(data_with_outliers)
        result = detect_outliers_zscore(series)

        # Should detect the 3 outliers we added
        assert result["count"] >= 3
        assert result["lower_bound"] < -3
        assert result["upper_bound"] > 3
        assert "Z-score" in result["interpretation"]

    def test_detect_outliers_zscore_constant_series(self):
        """Test z-score outlier detection with constant series."""
        series = pd.Series([5.0] * 10)
        result = detect_outliers_zscore(series)

        # Should detect no outliers in constant series
        assert result["count"] == 0
        assert result["std"] == 0
        assert "constant series" in result["interpretation"]

    def test_detect_outliers_zscore_threshold(self):
        """Test z-score outlier detection with custom threshold."""
        # Create data with outliers that are only detected with lower threshold
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        # Add outliers at 2.5 standard deviations
        data_with_outliers = np.concatenate([normal_data, [2.5, -2.5]])

        series = pd.Series(data_with_outliers)

        # With default threshold (3.0), should detect 0 outliers
        result_default = detect_outliers_zscore(series)

        # With lower threshold (2.0), should detect 2 outliers
        result_lower = detect_outliers_zscore(series, threshold=2.0)

        assert result_default["count"] == 0
        assert result_lower["count"] >= 2

    def test_apply_zscore_clipping_normal_data(self):
        """Test z-score clipping with normally distributed data."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        # Add extreme outliers
        data_with_outliers = np.concatenate([normal_data, [10.0, -10.0]])

        series = pd.Series(data_with_outliers)
        clipped_series = apply_zscore_clipping(series)

        # Check that extreme values were clipped
        assert clipped_series.max() < 10.0
        assert clipped_series.min() > -10.0
        # Most values should remain unchanged
        assert np.sum(np.abs(clipped_series.values[:100] - normal_data) < 0.01) > 90

    def test_apply_zscore_clipping_constant_series(self):
        """Test z-score clipping with constant series."""
        series = pd.Series([5.0] * 10)
        clipped_series = apply_zscore_clipping(series)

        # Constant series should remain unchanged
        pd.testing.assert_series_equal(series, clipped_series)

    def test_apply_zscore_clipping_custom_threshold(self):
        """Test z-score clipping with custom threshold."""
        # Create data with moderate outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        data_with_outliers = np.concatenate([normal_data, [2.5, -2.5]])

        series = pd.Series(data_with_outliers)

        # With default threshold, these won't be clipped
        clipped_default = apply_zscore_clipping(series)

        # With lower threshold, these should be clipped
        clipped_custom = apply_zscore_clipping(series, threshold=2.0)

        # Values should be more constrained with lower threshold
        assert clipped_custom.max() <= clipped_default.max()
        assert clipped_custom.min() >= clipped_default.min()
