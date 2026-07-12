"""Unit tests for the executive report business rules module."""

from __future__ import annotations

import pytest

from report.rules import (
    CONFIDENCE_DEDUCTIONS,
    CONFIDENCE_LABELS,
    confidence_label,
    data_quality_rating,
    mape_quality,
)


class TestConfidenceLabel:
    """Tests for the confidence_label function."""

    def test_high_label(self) -> None:
        assert confidence_label(100) == "High"
        assert confidence_label(75) == "High"

    def test_medium_label(self) -> None:
        assert confidence_label(74) == "Medium"
        assert confidence_label(50) == "Medium"

    def test_low_label(self) -> None:
        assert confidence_label(49) == "Low"
        assert confidence_label(0) == "Low"

    def test_boundary_high(self) -> None:
        assert confidence_label(CONFIDENCE_LABELS["high_min"]) == "High"

    def test_boundary_medium(self) -> None:
        assert confidence_label(CONFIDENCE_LABELS["medium_min"]) == "Medium"
        assert confidence_label(CONFIDENCE_LABELS["medium_min"] - 1) == "Low"


class TestDataQualityRating:
    """Tests for the data_quality_rating function."""

    def test_good_rating(self) -> None:
        assert data_quality_rating(0, 0, 0, 0, True) == "Good"

    def test_good_requires_regular(self) -> None:
        assert data_quality_rating(0, 0, 0, 0, False) != "Good"

    def test_fair_rating(self) -> None:
        assert data_quality_rating(3, 1, 2, 2, True) == "Fair"

    def test_poor_rating_high_missing(self) -> None:
        assert data_quality_rating(100, 0, 0, 0, True) == "Poor"

    def test_poor_rating_high_duplicates(self) -> None:
        assert data_quality_rating(0, 50, 0, 0, True) == "Poor"

    def test_poor_rating_high_gaps(self) -> None:
        assert data_quality_rating(0, 0, 50, 0, True) == "Poor"

    def test_poor_rating_many_issues(self) -> None:
        assert data_quality_rating(0, 0, 0, 10, True) == "Poor"


class TestMapeQuality:
    """Tests for the mape_quality function."""

    def test_high_quality(self) -> None:
        assert "high" in mape_quality(5.0)

    def test_acceptable_quality(self) -> None:
        assert "acceptable" in mape_quality(15.0)

    def test_moderate_quality(self) -> None:
        assert "moderate" in mape_quality(30.0)

    def test_low_quality(self) -> None:
        assert "low" in mape_quality(60.0)

    def test_boundary(self) -> None:
        assert "high" in mape_quality(9.9)
        assert "acceptable" in mape_quality(10.0)


class TestConfidenceDeductions:
    """Tests that confidence deduction constants are defined."""

    def test_all_deductions_present(self) -> None:
        expected_keys = [
            "mape_above_20",
            "mape_above_10",
            "mape_above_5",
            "non_stationary_adf",
            "white_noise",
            "outlier_ratio_high",
            "missing_data",
            "review_warn",
            "review_fail",
            "structural_breaks",
        ]
        for key in expected_keys:
            assert key in CONFIDENCE_DEDUCTIONS
            assert CONFIDENCE_DEDUCTIONS[key] > 0

    def test_mape_deductions_ordered(self) -> None:
        assert (
            CONFIDENCE_DEDUCTIONS["mape_above_20"]
            >= CONFIDENCE_DEDUCTIONS["mape_above_10"]
        )
        assert (
            CONFIDENCE_DEDUCTIONS["mape_above_10"]
            >= CONFIDENCE_DEDUCTIONS["mape_above_5"]
        )

    def test_review_fail_greater_than_warn(self) -> None:
        assert (
            CONFIDENCE_DEDUCTIONS["review_fail"]
            > CONFIDENCE_DEDUCTIONS["review_warn"]
        )
