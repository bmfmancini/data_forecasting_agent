"""Tests for ordering-sensitive compatibility bootstrap modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from forecasting import pmdarima_compat


def test_pmdarima_patch_translates_removed_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """The compatibility patch maps pmdarima's old keyword before delegation."""
    calls: list[dict[str, object]] = []

    def fake_check_array(*args: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return "ok"

    monkeypatch.delattr(
        pmdarima_compat.sk_validation,
        "_patched_for_pmdarima",
        raising=False,
    )
    monkeypatch.setattr(pmdarima_compat.sk_validation, "check_array", fake_check_array)

    pmdarima_compat.patch_sklearn_check_array()
    result = pmdarima_compat.sk_validation.check_array(
        [1, 2, 3],
        force_all_finite=False,
    )

    assert result == "ok"
    assert calls == [{"ensure_all_finite": False}]


def test_matplotlib_uses_noninteractive_backend() -> None:
    """Server-side visualization bootstrap configures Agg before pyplot import."""
    source = Path("data_forecaster/backend/utils/matplotlib_backend.py").read_text()

    assert 'matplotlib.use("Agg")' in source
    assert source.index('matplotlib.use("Agg")') < source.index("matplotlib.pyplot")
