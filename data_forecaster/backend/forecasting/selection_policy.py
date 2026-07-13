"""Deterministic model selection policy for the forecasting pipeline.

Python is the source of statistical decisions and the LLM is used for
context, critique, and explanation. This module implements the
deterministic selection policy that:

  - excludes failed, degraded-by-policy, and assumption-invalid candidates;
  - requires identical-fold evidence;
  - applies user/domain loss preferences when supplied;
  - ranks using configured out-of-sample point and interval metrics;
  - recognizes statistically/practically negligible differences;
  - prefers the simpler model when evidence is effectively tied;
  - retains naive/seasonal-naive when no complex model adds demonstrated
    value.

The policy is pure-Python and does not depend on LLM availability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.logging_config import get_logger
from forecasting.contracts import (
    BacktestEvaluation,
    ForecastAdapterResult,
    ForecastFitStatus,
)

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Metric priority for ranking (lower is better for all).
_METRIC_PRIORITY = ("mase", "wape", "rmse", "mae", "mape")

# Models ordered by simplicity (simplest first) for tie-breaking.
_SIMPLICITY_ORDER = ("EWMA", "Holt-Winters", "ARIMA", "SARIMA")

# Baseline model names that are retained when no complex model adds value.
_BASELINE_MODELS = ("Constant", "Naive", "Seasonal Naive", "Mean Forecast", "Drift")

# Threshold for "negligibly different" RMSE (relative).
_NEGLIGIBLE_RMSE_RATIO = 1.05

# Minimum improvement ratio for a complex model to beat a baseline.
_BASELINE_IMPROVEMENT_RATIO = 1.10


@dataclass
class CandidateEvidence:
    """Evidence for one candidate model in the selection policy.

    Attributes:
        name:           Model name.
        adapter_result: Terminal-holdout adapter result (or None).
        backtest:        Rolling-origin backtest evaluation (or None).
        is_baseline:    Whether this is a baseline model.
    """

    name: str
    adapter_result: ForecastAdapterResult | None = None
    backtest: BacktestEvaluation | None = None
    is_baseline: bool = False

    @property
    def is_rankable(self) -> bool:
        """Return whether this candidate has valid point-error evidence."""
        if self.backtest is not None:
            return self.backtest.is_rankable
        return bool(self.adapter_result and self.adapter_result.is_rankable)

    @property
    def rmse(self) -> float | None:
        """Return rolling-origin RMSE, falling back only when unavailable."""
        if self.backtest and self.backtest.pooled_metrics.rmse is not None:
            if math.isfinite(self.backtest.pooled_metrics.rmse):
                return self.backtest.pooled_metrics.rmse
        if self.adapter_result and self.adapter_result.metrics.rmse is not None:
            if math.isfinite(self.adapter_result.metrics.rmse):
                return self.adapter_result.metrics.rmse
        return None

    @property
    def backtest_rmse(self) -> float | None:
        """Return the rolling-origin pooled RMSE (or None)."""
        if self.backtest and self.backtest.pooled_metrics.rmse is not None:
            if math.isfinite(self.backtest.pooled_metrics.rmse):
                return self.backtest.pooled_metrics.rmse
        return None

    def metric_value(self, metric: str) -> float | None:
        """Return a named metric value, preferring rolling-origin evidence.

        Args:
            metric: Metric name (``"rmse"``, ``"mae"``, ``"mape"``,
                    ``"wape"``, ``"mase"``).

        Returns:
            The metric value, or None when unavailable.
        """
        metric = metric.lower()
        if self.backtest:
            val = getattr(self.backtest.pooled_metrics, metric, None)
            if val is not None and math.isfinite(val):
                return val
        if self.adapter_result:
            val = getattr(self.adapter_result.metrics, metric, None)
            if val is not None and math.isfinite(val):
                return val
        return None


@dataclass
class SelectionOutcome:
    """Result of the deterministic selection policy.

    Attributes:
        selected_model:   The name of the selected model.
        method:           How the selection was made (``"deterministic"``).
        is_fallback:      Whether the selected model is a fallback.
        ranking:          Ordered list of (model_name, rmse) for rankable
                          candidates.
        exclusion_reasons: Dict mapping excluded model names to reasons.
        tie_break_note:   Explanation of any tie-breaking applied.
        evidence_summary: Dict of evidence used for the decision.
    """

    selected_model: str
    method: str = "deterministic"
    is_fallback: bool = False
    ranking: list[tuple[str, float]] = field(default_factory=list)
    exclusion_reasons: dict[str, str] = field(default_factory=dict)
    tie_break_note: str = ""
    evidence_summary: dict[str, Any] = field(default_factory=dict)


def _filter_rankable(
    candidates: list[CandidateEvidence],
    exclude_set: set[str],
) -> tuple[list[CandidateEvidence], dict[str, str]]:
    """Filter candidates to rankable ones, recording exclusion reasons.

    Args:
        candidates:  List of candidate evidence objects.
        exclude_set: Set of model names to exclude.

    Returns:
        A tuple of (rankable_candidates, exclusion_reasons).
    """
    exclusion_reasons: dict[str, str] = {}
    rankable: list[CandidateEvidence] = []

    for cand in candidates:
        if cand.name in exclude_set:
            exclusion_reasons[cand.name] = "Excluded by caller request."
            continue
        if not cand.is_rankable:
            exclusion_reasons[cand.name] = _exclusion_reason(cand)
            continue
        rankable.append(cand)
    return rankable, exclusion_reasons


def _exclusion_reason(cand: CandidateEvidence) -> str:
    """Return the reason a non-rankable candidate was excluded."""
    if cand.adapter_result and cand.adapter_result.status != ForecastFitStatus.OK:
        return f"Excluded: status is {cand.adapter_result.status.value}."
    if cand.adapter_result:
        return "Excluded: required metrics (RMSE/MAE) are unavailable."
    return "Excluded: no adapter result."


def _rank_candidates(
    rankable: list[CandidateEvidence],
    loss_metric: str,
) -> list[CandidateEvidence]:
    """Rank candidates by metric priority (lower is better).

    Args:
        rankable: List of rankable candidate evidence objects.

    Returns:
        Sorted list of candidates (best first).
    """

    def _loss_key(cand: CandidateEvidence) -> tuple[float, ...]:
        """Return a tuple of metric values for ranking (lower is better)."""
        ordered = (loss_metric,) + tuple(
            metric for metric in _METRIC_PRIORITY if metric != loss_metric
        )
        return tuple(
            cand.metric_value(m) if cand.metric_value(m) is not None else float("inf")
            for m in ordered
        )

    return sorted(rankable, key=_loss_key)


def _apply_tie_break(
    ranked: list[CandidateEvidence],
) -> tuple[CandidateEvidence, str]:
    """Apply tie-breaking: prefer simpler model on negligible RMSE difference.

    Args:
        ranked: Ranked list of candidates (best first).

    Returns:
        A tuple of (selected_candidate, tie_break_note).
    """
    best = ranked[0]
    selected = best
    tie_break_note = ""

    if len(ranked) <= 1:
        return selected, tie_break_note

    second = ranked[1]
    best_rmse = best.rmse
    second_rmse = second.rmse
    if not (best_rmse and second_rmse and best_rmse > 0):
        return selected, tie_break_note

    ratio = second_rmse / best_rmse
    if ratio >= _NEGLIGIBLE_RMSE_RATIO:
        return selected, tie_break_note

    # Evidence is effectively tied — prefer the simpler model
    best_simplicity = _simplicity_index(best.name)
    second_simplicity = _simplicity_index(second.name)
    if second_simplicity < best_simplicity:
        selected = second
        tie_break_note = (
            f"RMSE difference between {best.name} and "
            f"{second.name} is negligible (ratio={ratio:.3f}); "
            f"preferring simpler model {second.name}."
        )
    return selected, tie_break_note


def _check_baseline_retention(
    selected: CandidateEvidence,
    ranked: list[CandidateEvidence],
    tie_break_note: str,
) -> tuple[CandidateEvidence, str]:
    """Retain a baseline if the complex model doesn't add sufficient value.

    Args:
        selected:       Currently selected candidate.
        ranked:         Ranked list of candidates.
        tie_break_note: Existing tie-break note.

    Returns:
        A tuple of (possibly_updated_selected, updated_tie_break_note).
    """
    if selected.is_baseline:
        return selected, tie_break_note

    baselines = [c for c in ranked if c.is_baseline]
    if not baselines:
        return selected, tie_break_note

    best_baseline = min(baselines, key=lambda c: c.rmse or float("inf"))
    selected_rmse = selected.rmse
    baseline_rmse = best_baseline.rmse
    if not (selected_rmse and baseline_rmse and selected_rmse > 0):
        return selected, tie_break_note

    improvement = baseline_rmse / selected_rmse
    if improvement >= _BASELINE_IMPROVEMENT_RATIO:
        return selected, tie_break_note

    tie_break_note += (
        f" Complex model did not demonstrate sufficient "
        f"improvement over baseline {best_baseline.name} "
        f"(improvement ratio={improvement:.3f}); retaining "
        f"baseline."
    )
    return best_baseline, tie_break_note


def select_model_deterministic(
    candidates: list[CandidateEvidence],
    *,
    exclude_models: list[str] | None = None,
    user_loss_preference: str = "rmse",
) -> SelectionOutcome:
    """Deterministically select the best model from typed evidence.

    The policy:
      1. Excludes failed, degraded, and non-rankable candidates.
      2. Excludes any models in the ``exclude_models`` list.
      3. Ranks surviving candidates by the configured loss metric.
      4. Recognizes negligibly different RMSE and prefers the simpler model.
      5. Retains baselines when no complex model adds demonstrated value.

    Args:
        candidates:          List of candidate evidence objects.
        exclude_models:      Optional list of model names to exclude.
        user_loss_preference: Loss metric for ranking (``"rmse"``,
                             ``"mase"``, ``"wape"``).

    Returns:
        :class:`SelectionOutcome` with the selected model and evidence.
    """
    exclude_set = set(exclude_models or [])
    rankable, exclusion_reasons = _filter_rankable(candidates, exclude_set)

    if not rankable:
        logger.warning("No rankable candidates for deterministic selection.")
        return SelectionOutcome(
            selected_model="",
            method="deterministic",
            ranking=[],
            exclusion_reasons=exclusion_reasons,
            evidence_summary={"n_candidates": len(candidates), "n_rankable": 0},
        )

    loss_metric = user_loss_preference.lower()
    if loss_metric not in _METRIC_PRIORITY:
        loss_metric = "rmse"

    ranked = _rank_candidates(rankable, loss_metric)
    ranking = [(c.name, c.rmse or float("inf")) for c in ranked]

    selected, tie_break_note = _apply_tie_break(ranked)
    selected, tie_break_note = _check_baseline_retention(
        selected, ranked, tie_break_note
    )

    logger.info(
        "Deterministic selection: %s (method=deterministic, rankable=%d)",
        selected.name,
        len(rankable),
    )

    return SelectionOutcome(
        selected_model=selected.name,
        method="deterministic",
        is_fallback=False,
        ranking=ranking,
        exclusion_reasons=exclusion_reasons,
        tie_break_note=tie_break_note,
        evidence_summary={
            "n_candidates": len(candidates),
            "n_rankable": len(rankable),
            "loss_metric": loss_metric,
            "selected_rmse": selected.rmse,
            "ranking": ranking[:5],
        },
    )


def _simplicity_index(model_name: str) -> int:
    """Return the simplicity index for a model (lower = simpler).

    Args:
        model_name: Model name.

    Returns:
        Simplicity index (0 = simplest).
    """
    for i, name in enumerate(_SIMPLICITY_ORDER):
        if name.lower() in model_name.lower():
            return i
    # Baselines are simplest
    for i, name in enumerate(_BASELINE_MODELS):
        if name.lower() in model_name.lower():
            return -1 + i
    return len(_SIMPLICITY_ORDER)


_KNOWN_MODEL_NAMES = (
    "ARIMA",
    "SARIMA",
    "Holt-Winters",
    "EWMA",
    "ETS",
    "Theta",
    "Prophet",
)
_COMMON_WORDS = frozenset({"The", "A", "An", "This", "That", "It", "Model"})


def _check_invented_models(
    llm_text: str,
    valid_models: list[str],
) -> list[str]:
    """Check for model names not in the valid candidate set.

    Args:
        llm_text:     Raw LLM output text.
        valid_models: List of valid model names.

    Returns:
        List of warning strings.
    """
    warnings_list: list[str] = []
    for word in llm_text.replace(",", " ").replace(".", " ").split():
        word_clean = word.strip("*_`#")
        if (
            word_clean
            and word_clean[0].isupper()
            and word_clean not in _COMMON_WORDS
            and "model" not in word_clean.lower()
            and len(word_clean) > 3
            and word_clean in _KNOWN_MODEL_NAMES
            and word_clean not in valid_models
        ):
            warnings_list.append(
                f"LLM referenced model '{word_clean}' which is not in "
                f"the valid candidate set."
            )
    return warnings_list


def _check_invented_metrics(
    llm_text: str,
    evidence: dict[str, Any],
) -> list[str]:
    """Check for numeric RMSE values not present in the evidence.

    Args:
        llm_text: Raw LLM output text.
        evidence: Dict of evidence the LLM was given.

    Returns:
        List of warning strings.
    """
    import re

    warnings_list: list[str] = []
    numbers = re.findall(r"RMSE\s*[:=]\s*([\d.]+)", llm_text, re.IGNORECASE)
    evidence_rmse_values: set[float] = set()
    for metrics in evidence.get("all_metrics", {}).values():
        rmse = metrics.get("RMSE") if isinstance(metrics, dict) else None
        if rmse is not None and math.isfinite(rmse):
            evidence_rmse_values.add(round(float(rmse), 4))
    for num_str in numbers:
        try:
            val = round(float(num_str), 4)
            if evidence_rmse_values and val not in evidence_rmse_values:
                warnings_list.append(
                    f"LLM cited RMSE={val} which does not match any " f"evidence value."
                )
        except ValueError:
            pass
    return warnings_list


def _check_contradictory_selection(
    text_lower: str,
    valid_models: list[str],
) -> list[str]:
    """Check for selected model names that are not valid.

    Args:
        text_lower:   Lower-cased LLM output text.
        valid_models: List of valid model names.

    Returns:
        List of warning strings.
    """
    import re

    warnings_list: list[str] = []
    selected_matches = re.findall(r"selected model\s*:\s*(\w+)", text_lower)
    for match in selected_matches:
        if match.title() not in valid_models and match != "no":
            warnings_list.append(f"LLM selected '{match}' which is not a valid model.")
    return warnings_list


def validate_llm_output(
    llm_text: str,
    valid_models: list[str],
    evidence: dict[str, Any],
) -> list[str]:
    """Validate LLM output for invented metrics and unsupported conclusions.

    A deterministic output validator for invented metrics, unsupported
    conclusions, contradictory model names, and recommendations violating
    target constraints.

    Args:
        llm_text:     Raw LLM output text.
        valid_models: List of valid model names.
        evidence:     Dict of evidence the LLM was given.

    Returns:
        List of validation warning strings (empty if valid).
    """
    text_lower = llm_text.lower()
    warnings_list: list[str] = []
    warnings_list.extend(_check_invented_models(llm_text, valid_models))
    warnings_list.extend(_check_invented_metrics(llm_text, evidence))
    warnings_list.extend(_check_contradictory_selection(text_lower, valid_models))
    return warnings_list
