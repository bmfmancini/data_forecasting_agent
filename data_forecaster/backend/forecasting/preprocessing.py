"""Fold-safe preprocessing transformations with inverse support.

Imputation, clipping, transformation-lambda estimation, and
additive/multiplicative seasonal selection are train-fold operations. This
module provides transformations that are fitted on training data only and
can be inverted on predictions to return to the original scale.

Each transform follows the :class:`PreprocessingTransform` contract from
:mod:`forecasting.contracts`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson

from core.logging_config import get_logger
from forecasting.contracts import PreprocessingTransform
from utils.data_cleaning import smooth_series

logger = get_logger(__name__)

_MIN_BOXCOX_LENGTH = 5
_EPSILON = 1e-8


class FoldSafeImputer:
    """Impute a training window without consulting validation observations."""

    def __init__(self, method: str = "interpolate") -> None:
        self.method = (
            method
            if method in {"interpolate", "forward-fill", "drop"}
            else "interpolate"
        )
        self.leading_value: float | None = None

    def fit(self, train: pd.Series) -> FoldSafeImputer:
        """Record the last observed training value for boundary-safe filling."""
        observed = train.dropna().astype(float)
        self.leading_value = float(observed.iloc[-1]) if not observed.empty else None
        return self

    def transform_training(self, train: pd.Series) -> pd.Series:
        """Fill using training observations only."""
        values = train.astype(float).copy()
        if self.method == "drop":
            return values.dropna()
        if self.method == "forward-fill":
            return values.ffill().bfill()
        return values.interpolate(limit_direction="both").ffill().bfill()

    def transform_future_inputs(self, values: pd.Series) -> pd.Series:
        """Fill future covariate-like values using the fitted training boundary."""
        result = values.astype(float).copy()
        if self.method == "drop" or self.leading_value is None:
            return result
        return result.fillna(self.leading_value)


def smooth_training_series(series: pd.Series, method: str) -> pd.Series:
    """Apply optional smoothing only to a model's training observations."""
    if not method or method == "none":
        return series
    return smooth_series(series, method)


class FoldSafeOutlierTreatment:
    """Fit clipping/removal thresholds using training observations only."""

    def __init__(self, strategy: str = "none") -> None:
        self.strategy = strategy
        self.lower: float | None = None
        self.upper: float | None = None

    def fit(self, train: pd.Series) -> FoldSafeOutlierTreatment:
        """Estimate IQR or z-score bounds from the training window."""
        observed = train.dropna().astype(float)
        if observed.empty:
            return self
        if self.strategy in {"clip", "remove"}:
            q1, q3 = float(observed.quantile(0.25)), float(observed.quantile(0.75))
            spread = q3 - q1
            self.lower, self.upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        elif self.strategy == "zscore_clip":
            mean, std = float(observed.mean()), float(observed.std(ddof=0))
            if np.isfinite(std) and std > 0:
                self.lower, self.upper = mean - 3.0 * std, mean + 3.0 * std
        return self

    def transform_training(self, train: pd.Series) -> pd.Series:
        """Apply fitted bounds without changing validation observations."""
        if self.lower is None or self.upper is None:
            return train
        if self.strategy == "remove":
            return train.where(train.between(self.lower, self.upper))
        return train.clip(lower=self.lower, upper=self.upper)


class BoxCoxTransform:
    """Fold-safe Box-Cox transformation with inverse support.

    The lambda parameter is estimated on training data only. The shift
    required to make the series strictly positive is also fitted on
    training data and applied to test/prediction data.

    Attributes:
        transform: The :class:`PreprocessingTransform` metadata.
    """

    def __init__(self) -> None:
        self.transform = PreprocessingTransform(name="boxcox")

    def fit(self, train: pd.Series) -> BoxCoxTransform:
        """Fit the Box-Cox transform on training data.

        Args:
            train: Training series (will be shifted to positive).

        Returns:
            Self for chaining.
        """
        values = train.dropna().astype(float).values
        if len(values) < _MIN_BOXCOX_LENGTH:
            logger.warning(
                "Box-Cox fit: training series too short (n=%d).", len(values)
            )
            self.transform.is_fitted = False
            return self

        shift = 0.0
        min_val = float(np.min(values))
        if min_val <= 0:
            shift = abs(min_val) + 1.0

        shifted = values + shift
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, lam = boxcox(shifted)
            self.transform.lambda_value = float(lam)
            self.transform.shift = shift
            self.transform.is_fitted = True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Box-Cox lambda estimation failed: %s", exc)
            self.transform.is_fitted = False
        return self

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Apply the fitted Box-Cox transform to a series.

        Args:
            series: Series to transform.

        Returns:
            Transformed series (unchanged if not fitted).
        """
        if not self.transform.is_fitted or self.transform.lambda_value is None:
            return series
        values = series.astype(float).values + self.transform.shift
        lam = self.transform.lambda_value
        if abs(lam) < _EPSILON:
            # log transform
            result = np.log(np.maximum(values, _EPSILON))
        else:
            result = (np.maximum(values, _EPSILON) ** lam - 1) / lam
        return pd.Series(result, index=series.index)

    def inverse_transform(self, values: np.ndarray | pd.Series) -> np.ndarray:
        """Invert the Box-Cox transform on predictions.

        Args:
            values: Transformed predictions.

        Returns:
            Predictions on the original scale.
        """
        if not self.transform.is_fitted or self.transform.lambda_value is None:
            return np.asarray(values, dtype=float)
        arr = np.asarray(values, dtype=float)
        lam = self.transform.lambda_value
        if abs(lam) < _EPSILON:
            result = np.exp(arr)
        else:
            result = (arr * lam + 1) ** (1 / lam)
        return result - self.transform.shift


class LogTransform:
    """Fold-safe log transformation with inverse support.

    The shift required to make the series strictly positive is fitted on
    training data.
    """

    def __init__(self) -> None:
        self.transform = PreprocessingTransform(name="log")

    def fit(self, train: pd.Series) -> LogTransform:
        """Fit the log transform on training data.

        Args:
            train: Training series.

        Returns:
            Self for chaining.
        """
        values = train.dropna().astype(float).values
        if len(values) == 0:
            self.transform.is_fitted = False
            return self
        shift = 0.0
        min_val = float(np.min(values))
        if min_val <= 0:
            shift = abs(min_val) + 1.0
        self.transform.shift = shift
        self.transform.is_fitted = True
        return self

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Apply the fitted log transform to a series.

        Args:
            series: Series to transform.

        Returns:
            Transformed series.
        """
        if not self.transform.is_fitted:
            return series
        values = np.maximum(
            series.astype(float).values + self.transform.shift, _EPSILON
        )
        return pd.Series(np.log(values), index=series.index)

    def inverse_transform(self, values: np.ndarray | pd.Series) -> np.ndarray:
        """Invert the log transform on predictions.

        Args:
            values: Transformed predictions.

        Returns:
            Predictions on the original scale.
        """
        if not self.transform.is_fitted:
            return np.asarray(values, dtype=float)
        return np.exp(np.asarray(values, dtype=float)) - self.transform.shift


class YeoJohnsonTransform:
    """Fold-safe Yeo-Johnson transform for targets containing nonpositive values."""

    def __init__(self) -> None:
        self.transform = PreprocessingTransform(name="yeojohnson")

    def fit(self, train: pd.Series) -> YeoJohnsonTransform:
        values = train.dropna().astype(float).to_numpy()
        if values.size < _MIN_BOXCOX_LENGTH or np.all(values == values[0]):
            return self
        try:
            _, lam = yeojohnson(values)
            self.transform.lambda_value = float(lam)
            self.transform.is_fitted = True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Yeo-Johnson lambda estimation failed: %s", exc)
        return self

    def transform_series(self, series: pd.Series) -> pd.Series:
        if not self.transform.is_fitted or self.transform.lambda_value is None:
            return series
        return pd.Series(
            yeojohnson(series.astype(float).to_numpy(), self.transform.lambda_value),
            index=series.index,
        )

    def inverse_transform(self, values: np.ndarray | pd.Series) -> np.ndarray:
        if not self.transform.is_fitted or self.transform.lambda_value is None:
            return np.asarray(values, dtype=float)
        transformed = np.asarray(values, dtype=float)
        lam = self.transform.lambda_value
        result = np.empty_like(transformed)
        positive = transformed >= 0
        if abs(lam) < _EPSILON:
            result[positive] = np.expm1(transformed[positive])
        else:
            result[positive] = (
                np.power(
                    np.maximum(lam * transformed[positive] + 1.0, _EPSILON), 1.0 / lam
                )
                - 1.0
            )
        if abs(lam - 2.0) < _EPSILON:
            result[~positive] = 1.0 - np.exp(-transformed[~positive])
        else:
            result[~positive] = 1.0 - np.power(
                np.maximum(1.0 - (2.0 - lam) * transformed[~positive], _EPSILON),
                1.0 / (2.0 - lam),
            )
        return result


def bias_adjusted_inverse(
    transform: Any,
    predictions: np.ndarray | pd.Series | list[float],
    transformed_residuals: np.ndarray | pd.Series | list[float],
    *,
    seed: int = 42,
    simulations: int = 1000,
) -> np.ndarray:
    """Apply a deterministic residual-smearing retransformation correction."""
    point = np.asarray(predictions, dtype=float)
    residuals = np.asarray(transformed_residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return transform.inverse_transform(point)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(residuals, size=(simulations, point.size), replace=True)
    original_scale = transform.inverse_transform(point[None, :] + sampled)
    return np.mean(original_scale, axis=0)


class IQRClipping:
    """Fold-safe IQR clipping (winsorization) with training-fitted bounds.

    The lower and upper bounds are computed on training data only and
    applied to test/prediction data to prevent leakage.
    """

    def __init__(self, multiplier: float = 1.5) -> None:
        self.multiplier = multiplier
        self.lower_bound: float = -np.inf
        self.upper_bound: float = np.inf
        self.is_fitted = False

    def fit(self, train: pd.Series) -> IQRClipping:
        """Fit IQR bounds on training data.

        Args:
            train: Training series.

        Returns:
            Self for chaining.
        """
        values = train.dropna().astype(float).values
        if len(values) < 4:
            self.is_fitted = False
            return self
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        self.lower_bound = q1 - self.multiplier * iqr
        self.upper_bound = q3 + self.multiplier * iqr
        self.is_fitted = True
        return self

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Clip the series to the fitted bounds.

        Args:
            series: Series to clip.

        Returns:
            Clipped series.
        """
        if not self.is_fitted:
            return series
        return series.clip(lower=self.lower_bound, upper=self.upper_bound)


def fit_transform_on_train(
    train: pd.Series,
    *,
    apply_boxcox: bool = False,
    apply_iqr_clip: bool = False,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.Series, list[Any]]:
    """Fit and apply preprocessing transformations on training data only.

    This is the fold-safe entry point used by backtesting folds. Each
    transform is fitted on the training window and can be applied to the
    test window without leakage.

    Args:
        train:           Training series.
        apply_boxcox:    Whether to apply a Box-Cox transform.
        apply_iqr_clip:  Whether to apply IQR clipping.
        iqr_multiplier:  IQR multiplier for clipping.

    Returns:
        Tuple of (transformed_train, list_of_fitted_transforms).
    """
    transformed = train.copy()
    transforms: list[Any] = []

    if apply_iqr_clip:
        clipper = IQRClipping(multiplier=iqr_multiplier)
        clipper.fit(transformed)
        if clipper.is_fitted:
            transformed = clipper.transform_series(transformed)
            transforms.append(clipper)

    if apply_boxcox:
        bc = BoxCoxTransform()
        bc.fit(transformed)
        if bc.transform.is_fitted:
            transformed = bc.transform_series(transformed)
            transforms.append(bc)

    return transformed, transforms


def apply_transforms_to_test(
    test: pd.Series,
    transforms: list[Any],
) -> pd.Series:
    """Apply training-fitted transforms to test data.

    Args:
        test:      Test series.
        transforms: List of fitted transforms from :func:`fit_transform_on_train`.

    Returns:
        Transformed test series.
    """
    result = test.copy()
    for transform in transforms:
        if hasattr(transform, "transform_series"):
            result = transform.transform_series(result)
    return result


def inverse_transform_predictions(
    predictions: np.ndarray | pd.Series,
    transforms: list[Any],
) -> np.ndarray:
    """Invert all transforms on predictions to return to the original scale.

    Transforms are inverted in reverse order (last applied, first inverted).

    Args:
        predictions: Model predictions on the transformed scale.
        transforms:  List of fitted transforms.

    Returns:
        Predictions on the original scale.
    """
    result = np.asarray(predictions, dtype=float)
    for transform in reversed(transforms):
        if hasattr(transform, "inverse_transform"):
            result = transform.inverse_transform(result)
    return result
