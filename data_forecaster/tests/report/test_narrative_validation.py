"""Regression tests for report context validation and bounded correction."""
from types import SimpleNamespace

from report import narrative
from report.models import Assumption, Risk


class FakePrompt:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __or__(self, llm):
        return self

    def invoke(self, inputs):
        self.calls.append(inputs)
        return SimpleNamespace(content=next(self.responses))


def generate(monkeypatch, section, responses, context=None, name="assumption"):
    monkeypatch.setattr(narrative, "extract_token_usage", lambda *a, **k: {"total_tokens": 1})
    monkeypatch.setattr(narrative, "estimate_input_text", lambda *a: "")
    prompt = FakePrompt(responses)
    fallbacks = []
    usage = {"total_tokens": 0}
    result = narrative._generate_section(object(), prompt, section, name, usage, "", fallbacks, context)
    return result, fallbacks, prompt.calls, usage


def test_declared_holiday_context_passes_validation(monkeypatch):
    result, fallbacks, calls, _ = generate(monkeypatch,
        Assumption(assumption="Conditions remain stable.", consequence_if_false="Reassess."),
        ["The holiday calendar remains stable."],
        {"known_context": {"holidays_country": "CA", "covariates": []}},
    )
    assert result == "The holiday calendar remains stable."
    assert not fallbacks
    assert len(calls) == 1


def test_holiday_in_assumption_seed_is_valid():
    assert not narrative._unsupported_assumption_claims("The holiday calendar persists.", {"assumption": "The Canada holiday calendar persists."})


def test_empty_context_and_instruction_text_do_not_authorize_claims():
    assert narrative._unsupported_assumption_claims("The holiday affects usage.", {
        "assumption": "Stable conditions.",
        "business_context": {"known_context": {"holidays_country": None, "covariates": []}, "dated_context": {"interpretation_rule": "Discuss holiday covariates", "dated_events": []}},
    })


def test_supported_model_in_recommendation_evidence_is_allowed(monkeypatch):
    risk = Risk(category="Model", severity="Medium", description="SARIMA cannot ingest holidays.", potential_impact="Future events may differ.", mitigation="Compare Prophet or Dynamic Regression.")
    _, fallbacks, calls, _ = generate(monkeypatch, risk, ["Compare Prophet or Dynamic Regression."], name="risk")
    assert not fallbacks
    assert len(calls) == 1


def test_failed_validation_gets_one_correction(monkeypatch):
    result, fallbacks, calls, usage = generate(monkeypatch,
        Assumption(assumption="Conditions remain stable.", consequence_if_false="Reassess."),
        ["The holiday affects demand.", "Conditions remain stable."],
    )
    assert result == "Conditions remain stable."
    assert not fallbacks
    assert len(calls) == 2
    assert "REVISION REQUIRED" in calls[1]["section_json"]
    assert usage["total_tokens"] == 2


def test_persistent_invalid_claim_still_falls_back(monkeypatch):
    result, fallbacks, calls, _ = generate(monkeypatch,
        Assumption(assumption="Conditions remain stable.", consequence_if_false="Reassess."),
        ["The holiday affects demand."] * 2,
    )
    assert "holiday" not in result
    assert fallbacks == ["assumption"]
    assert len(calls) == 2


def test_break_sequencing_accepts_confirmation_but_rejects_immediate_changes():
    evidence = {"mitigation": "Validate candidate break dates first."}
    assert not narrative._unsupported_change_point_sequencing("Validate the breaks. After confirmation, compare segmentation.", evidence)
    assert narrative._unsupported_change_point_sequencing("Apply segmentation immediately.", evidence)


def test_holiday_calendar_does_not_authorize_undeclared_covariates():
    assert narrative._unsupported_assumption_claims("The price signal stays stable.", {
        "assumption": "Conditions remain stable.",
        "business_context": {"known_context": {"holidays_country": "CA", "covariates": []}},
    })
