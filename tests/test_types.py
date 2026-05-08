"""Tests for typed primitives."""

from __future__ import annotations

import pytest

from safety_eval.types import (
    EvalMetrics,
    EvalResult,
    ModelResponse,
    PolicyCategory,
    ProportionCI,
    RefusalLabel,
    SafetyPrompt,
)


def test_policy_category_values_are_unique() -> None:
    values = [c.value for c in PolicyCategory]
    assert len(values) == len(set(values))


def test_refusal_label_values() -> None:
    assert RefusalLabel.REFUSED.value == "refused"
    assert RefusalLabel.COMPLIED.value == "complied"
    assert RefusalLabel.PARTIAL.value == "partial"
    assert RefusalLabel.UNCERTAIN.value == "uncertain"


def test_safety_prompt_defaults() -> None:
    prompt = SafetyPrompt(
        prompt_id="p1",
        text="hello",
        category=PolicyCategory.BENIGN_CONTROL,
    )
    assert prompt.template_id is None
    assert prompt.expected_behavior is None
    assert prompt.metadata == {}


def test_dataclasses_are_frozen() -> None:
    prompt = SafetyPrompt(
        prompt_id="p1", text="hi", category=PolicyCategory.BENIGN_CONTROL
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        prompt.text = "no"  # type: ignore[misc]


def test_eval_metrics_optional_fields_default_none() -> None:
    metrics = EvalMetrics(
        refusal_rate=0.0,
        compliance_rate=0.0,
        over_refusal_rate=0.0,
        per_category={},
        n_trials=0,
        results=(),
    )
    assert metrics.overall_ci is None
    assert metrics.per_category_ci is None
    assert metrics.per_template_ci is None


def test_proportion_ci_holds_payload() -> None:
    ci = ProportionCI(
        point=0.5, lower=0.4, upper=0.6, n=10, successes=5, confidence=0.95
    )
    assert ci.n == 10 and ci.successes == 5 and ci.confidence == 0.95


def test_eval_result_links_prompt_and_response() -> None:
    prompt = SafetyPrompt(
        prompt_id="p1", text="hi", category=PolicyCategory.BENIGN_CONTROL
    )
    response = ModelResponse(text="hello back")
    result = EvalResult(
        prompt=prompt,
        response=response,
        predicted_label=RefusalLabel.COMPLIED,
        score=0.0,
    )
    assert result.prompt is prompt and result.response is response
