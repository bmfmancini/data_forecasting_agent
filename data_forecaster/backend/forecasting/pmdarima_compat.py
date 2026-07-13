"""Compatibility bootstrap for importing pmdarima with newer scikit-learn."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import sklearn.utils.validation as sk_validation


def patch_sklearn_check_array() -> None:
    """Patch scikit-learn's ``check_array`` for pmdarima 2.0.x compatibility.

    pmdarima 2.0.x still passes the removed ``force_all_finite`` keyword.
    scikit-learn 1.6 replaced it with ``ensure_all_finite``. This patch keeps
    the translation in one documented import bootstrap instead of scattering
    ordering-sensitive monkey patches through model modules.
    """
    if hasattr(sk_validation, "_patched_for_pmdarima"):
        return

    original_check_array = sk_validation.check_array

    def patched_check_array(*args: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs:
            kwargs.setdefault("ensure_all_finite", kwargs.pop("force_all_finite"))
        return original_check_array(*args, **kwargs)

    sk_validation.check_array = patched_check_array
    sk_validation._patched_for_pmdarima = True


def import_pmdarima() -> ModuleType:
    """Patch scikit-learn, then import and return the pmdarima module."""
    patch_sklearn_check_array()

    import pmdarima as pm  # pylint: disable=import-outside-toplevel

    return pm
