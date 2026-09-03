"""Unit tests for the forecasting model registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = str(Path(__file__).resolve().parent.parent / "data_forecaster" / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core import config as settings  # noqa: E402
from core.database import init_database  # noqa: E402
from forecasting import registry  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Temp backend DB with the model_config seed rows."""
    db_path = str(tmp_path / "backend.db")
    monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
    init_database()
    return db_path


class TestEnabledModels:
    """Registry reads of the enabled set."""

    def test_all_enabled_by_default(self, db):
        assert set(registry.get_enabled_models(db)) == set(registry.MODEL_NAMES)

    def test_disable_excludes_model(self, db):
        registry.set_model_enabled("Prophet", False, db_path=db)

        enabled = registry.get_enabled_models(db)

        assert "Prophet" not in enabled
        assert len(enabled) == 4

    def test_missing_table_falls_back_to_all(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "empty.db")
        monkeypatch.setattr(settings, "BACKEND_DB_PATH", db_path)
        # No init_database — model_config table does not exist.
        assert set(registry.get_enabled_models(db_path)) == set(registry.MODEL_NAMES)


class TestSetModelEnabled:
    """Enable/disable writes and the last-model guard."""

    def test_unknown_model_rejected(self, db):
        with pytest.raises(ValueError, match="Unknown model"):
            registry.set_model_enabled("LSTM", True, db_path=db)

    def test_cannot_disable_last_enabled_model(self, db):
        for name in registry.MODEL_NAMES:
            if name != "EWMA":
                registry.set_model_enabled(name, False, db_path=db)

        with pytest.raises(ValueError, match="last enabled model"):
            registry.set_model_enabled("EWMA", False, db_path=db)

        assert registry.get_enabled_models(db) == ("EWMA",)

    def test_reenable_after_disable(self, db):
        registry.set_model_enabled("ARIMA", False, db_path=db)
        registry.set_model_enabled("ARIMA", True, db_path=db)

        assert "ARIMA" in registry.get_enabled_models(db)


class TestFitFunctions:
    """Fit-loop triples honour enabled state and per-model kwargs."""

    def test_disabled_model_not_returned(self, db):
        registry.set_model_enabled("Prophet", False, db_path=db)

        triples = registry.get_fit_functions(
            {"seasonal_period": 12, "freq": "MS"}, db_path=db
        )

        names = [name for name, _, _ in triples]
        assert "Prophet" not in names
        assert len(triples) == 4

    def test_model_specific_kwargs(self, db):
        triples = dict(
            (name, kwargs)
            for name, _, kwargs in registry.get_fit_functions(
                {"seasonal_period": 7, "freq": "D"}, db_path=db
            )
        )

        assert triples["SARIMA"] == {"seasonal_period": 7}
        assert triples["Prophet"] == {"freq": "D"}
        assert triples["ARIMA"] == {}


class TestListModelStates:
    """Admin UI state listing."""

    def test_states_reflect_enabled(self, db):
        registry.set_model_enabled("SARIMA", False, db_path=db)

        states = {m["name"]: m["enabled"] for m in registry.list_model_states(db)}

        assert states["SARIMA"] is False
        assert states["ARIMA"] is True
        assert len(states) == 5
