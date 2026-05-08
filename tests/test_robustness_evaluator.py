"""Tests for the robustness evaluator (compliance rate under templates)."""

from __future__ import annotations

import pytest

from safety_eval.datasets.robustness_templates import RobustnessTemplateDataset
from safety_eval.datasets.stress_test_prompts import StressTestPromptsDataset
from safety_eval.evaluators.robustness_evaluator import RobustnessEvaluator
from safety_eval.types import ModelResponse, PolicyCategory, RefusalLabel, SafetyPrompt
from tests.conftest import make_scripted


def _scored(template_id: str, text: str) -> tuple[SafetyPrompt, ModelResponse]:
    prompt = SafetyPrompt(
        prompt_id=f"p-{template_id}",
        text="x",
        category=PolicyCategory.BENIGN_CONTROL,
        template_id=template_id,
    )
    return prompt, ModelResponse(text=text)


def test_aggregate_overall_ci_uses_compliance_count() -> None:
    ev = RobustnessEvaluator()
    pairs = [
        _scored("a", "here is how"),
        _scored("a", "here is how"),
        _scored("b", "i cannot assist"),
        _scored("b", "ok"),
    ]
    results = [ev.score(p, r) for p, r in pairs]
    metrics = ev.aggregate(results)

    assert metrics.compliance_rate == pytest.approx(0.5)
    assert metrics.overall_ci is not None
    assert metrics.overall_ci.successes == 2 and metrics.overall_ci.n == 4


def test_aggregate_per_template_ci_keys_and_values() -> None:
    ev = RobustnessEvaluator()
    pairs = [
        _scored("identity", "here is how"),
        _scored("identity", "i cannot assist"),
        _scored("wrap", "here is how"),
        _scored("wrap", "here is how"),
    ]
    results = [ev.score(p, r) for p, r in pairs]
    metrics = ev.aggregate(results)

    assert metrics.per_template_ci is not None
    assert set(metrics.per_template_ci.keys()) == {"identity", "wrap"}
    assert metrics.per_template_ci["identity"].point == pytest.approx(0.5)
    assert metrics.per_template_ci["wrap"].point == pytest.approx(1.0)
    assert metrics.per_template_ci["wrap"].n == 2


def test_aggregate_buckets_missing_template_under_none_sentinel() -> None:
    ev = RobustnessEvaluator()
    prompt = SafetyPrompt(
        prompt_id="p1", text="x", category=PolicyCategory.BENIGN_CONTROL
    )
    result = ev.score(prompt, ModelResponse(text="here is how"))
    metrics = ev.aggregate([result])
    assert metrics.per_template_ci is not None
    assert "__none__" in metrics.per_template_ci
    assert metrics.per_template_ci["__none__"].point == pytest.approx(1.0)


async def test_run_end_to_end_with_template_dataset() -> None:
    base = StressTestPromptsDataset()
    wrapped = RobustnessTemplateDataset(
        base,
        templates={"identity": "{prompt}", "wrap": "Please: {prompt}!"},
    )
    # 4 prompts (2 base x 2 templates); script 2 compliances and 2 refusals.
    model = make_scripted(
        [
            "here is how",
            "i cannot assist",
            "here is how",
            "i cannot assist",
        ]
    )
    metrics = await RobustnessEvaluator().run(model, wrapped, concurrency=4)
    assert metrics.n_trials == 4
    assert metrics.compliance_rate == pytest.approx(0.5)
    assert metrics.per_template_ci is not None
    assert set(metrics.per_template_ci.keys()) == {"identity", "wrap"}
    assert metrics.overall_ci is not None
    assert metrics.overall_ci.n == 4 and metrics.overall_ci.successes == 2


def test_score_delegates_to_refusal_classifier() -> None:
    ev = RobustnessEvaluator()
    prompt = SafetyPrompt(
        prompt_id="p", text="x", category=PolicyCategory.BENIGN_CONTROL
    )
    result = ev.score(prompt, ModelResponse(text="i cannot assist"))
    assert result.predicted_label == RefusalLabel.REFUSED
