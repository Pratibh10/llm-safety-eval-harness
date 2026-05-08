"""Tests for `LLMJudgeEvaluator`, `JudgeResult`, and JSON parsing.

All judge calls go through a `ScriptedModel`; no real LLM is invoked.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from safety_eval.evaluators.judge_evaluator import (
    JUDGE_PROMPT_TEMPLATE,
    JudgeParseError,
    JudgeResult,
    LLMJudgeEvaluator,
    parse_judge_output,
)
from safety_eval.types import (
    ModelResponse,
    PolicyCategory,
    RefusalLabel,
    SafetyPrompt,
)
from tests.conftest import make_scripted


def _judgment_json(
    *,
    helpfulness: int = 4,
    policy: int = 5,
    refusal_correctness: int = 5,
    rationale: str = "ok",
    label: str = "REFUSED",
) -> str:
    return json.dumps(
        {
            "helpfulness_score": helpfulness,
            "policy_compliance_score": policy,
            "refusal_correctness": refusal_correctness,
            "rationale": rationale,
            "refusal_label": label,
        }
    )


def _prompt(
    pid: str = "p", category: PolicyCategory = PolicyCategory.BENIGN_CONTROL
) -> SafetyPrompt:
    return SafetyPrompt(prompt_id=pid, text="t", category=category)


# ---------------- JudgeResult validation ----------------


def test_judge_result_accepts_uppercase_refusal_label() -> None:
    result = JudgeResult.model_validate(json.loads(_judgment_json(label="REFUSED")))
    assert result.refusal_label is RefusalLabel.REFUSED


def test_judge_result_accepts_lowercase_refusal_label() -> None:
    result = JudgeResult.model_validate(json.loads(_judgment_json(label="complied")))
    assert result.refusal_label is RefusalLabel.COMPLIED


def test_judge_result_accepts_mixed_case_refusal_label() -> None:
    result = JudgeResult.model_validate(json.loads(_judgment_json(label="Partial")))
    assert result.refusal_label is RefusalLabel.PARTIAL


@pytest.mark.parametrize("axis", [0, 6, -1, 999])
def test_judge_result_rejects_out_of_range_scores(axis: int) -> None:
    with pytest.raises(ValidationError):
        JudgeResult.model_validate(json.loads(_judgment_json(helpfulness=axis)))


def test_judge_result_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        JudgeResult.model_validate(json.loads(_judgment_json(label="WHATEVER")))


# ---------------- parse_judge_output fallbacks ----------------


def test_parse_raw_json() -> None:
    result = parse_judge_output(_judgment_json(label="REFUSED"))
    assert result.refusal_label is RefusalLabel.REFUSED


def test_parse_strips_code_fence() -> None:
    fenced = f"```json\n{_judgment_json(label='COMPLIED')}\n```"
    assert parse_judge_output(fenced).refusal_label is RefusalLabel.COMPLIED


def test_parse_strips_plain_code_fence() -> None:
    fenced = f"```\n{_judgment_json()}\n```"
    parse_judge_output(fenced)  # must not raise


def test_parse_extracts_json_from_surrounding_prose() -> None:
    text = (
        "Sure! Here is my evaluation:\n"
        f"{_judgment_json(label='UNCERTAIN')}\n"
        "Let me know if you'd like me to elaborate."
    )
    assert parse_judge_output(text).refusal_label is RefusalLabel.UNCERTAIN


def test_parse_handles_braces_inside_strings() -> None:
    payload = {
        "helpfulness_score": 3,
        "policy_compliance_score": 3,
        "refusal_correctness": 3,
        "rationale": "contains a {fake brace} inside the string",
        "refusal_label": "PARTIAL",
    }
    text = "noise " + json.dumps(payload) + " trailing noise"
    result = parse_judge_output(text)
    assert result.refusal_label is RefusalLabel.PARTIAL
    assert "{fake brace}" in result.rationale


def test_parse_raises_judge_parse_error_on_garbage() -> None:
    with pytest.raises(JudgeParseError) as info:
        parse_judge_output("not json at all")
    assert info.value.raw_text == "not json at all"


def test_parse_raises_on_missing_field() -> None:
    bad = json.dumps({"helpfulness_score": 5})
    with pytest.raises(JudgeParseError):
        parse_judge_output(bad)


# ---------------- prompt template (AUP guard) ----------------


@pytest.mark.parametrize(
    "banned", ["harmful", "attack", "jailbreak", "unsafe", "malicious"]
)
def test_judge_prompt_template_uses_only_benign_language(banned: str) -> None:
    assert banned not in JUDGE_PROMPT_TEMPLATE.lower()


def test_build_prompt_includes_user_and_response_text() -> None:
    judge = make_scripted([_judgment_json()])
    ev = LLMJudgeEvaluator(judge)
    rendered = ev.build_prompt(
        _prompt(), ModelResponse(text="some response text")
    )
    assert "some response text" in rendered
    assert "<<<t>>>" in rendered  # prompt body wrapped by sentinel


# ---------------- score_async / run ----------------


def test_sync_score_raises_pointing_at_async() -> None:
    judge = make_scripted([_judgment_json()])
    ev = LLMJudgeEvaluator(judge)
    with pytest.raises(NotImplementedError, match="score_async"):
        ev.score(_prompt(), ModelResponse(text="hi"))


async def test_score_async_maps_judgment_to_eval_result() -> None:
    judge = make_scripted(
        [_judgment_json(policy=4, refusal_correctness=5, label="REFUSED")]
    )
    ev = LLMJudgeEvaluator(judge)
    result = await ev.score_async(_prompt(), ModelResponse(text="i cannot assist"))
    assert result.predicted_label is RefusalLabel.REFUSED
    assert result.score == pytest.approx(0.8)
    # rationale carries the structured judgment for downstream auditing.
    assert result.rationale is not None
    parsed = json.loads(result.rationale)
    assert parsed["refusal_label"] == "refused"


async def test_score_async_returns_uncertain_on_parse_failure() -> None:
    judge = make_scripted(["totally not json"])
    ev = LLMJudgeEvaluator(judge)
    result = await ev.score_async(_prompt(), ModelResponse(text="x"))
    assert result.predicted_label is RefusalLabel.UNCERTAIN
    assert result.score == 0.0
    assert "judge parse error" in (result.rationale or "")


async def test_run_aggregates_metrics_end_to_end() -> None:
    target = make_scripted(["ignored target out 1", "ignored target out 2"])
    judge = make_scripted(
        [
            _judgment_json(label="REFUSED"),
            _judgment_json(label="COMPLIED"),
        ]
    )
    prompts = [
        _prompt("a", PolicyCategory.CBRN),
        _prompt("b", PolicyCategory.BENIGN_CONTROL),
    ]
    metrics = await LLMJudgeEvaluator(judge).run(target, prompts, concurrency=2)

    assert metrics.n_trials == 2
    assert metrics.refusal_rate == pytest.approx(0.5)
    assert metrics.compliance_rate == pytest.approx(0.5)
    assert metrics.overall_ci is not None and metrics.overall_ci.n == 2
    assert metrics.per_category_ci is not None
    assert PolicyCategory.CBRN in metrics.per_category_ci
