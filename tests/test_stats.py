"""Tests for Wilson interval and per-category aggregation."""

from __future__ import annotations

import math

import pytest

from safety_eval.metrics.stats import per_category_intervals, wilson_interval
from safety_eval.types import (
    EvalResult,
    ModelResponse,
    PolicyCategory,
    RefusalLabel,
    SafetyPrompt,
)


def test_wilson_zero_n_returns_degenerate_interval() -> None:
    ci = wilson_interval(0, 0)
    assert ci.n == 0 and ci.successes == 0
    assert ci.lower == 0.0 and ci.upper == 0.0 and ci.point == 0.0


def test_wilson_known_value_50_percent() -> None:
    ci = wilson_interval(50, 100, confidence=0.95)
    assert ci.point == pytest.approx(0.5)
    # Reference: Wilson 95% interval for 50/100 is approx (0.404, 0.596).
    assert math.isclose(ci.lower, 0.4038, abs_tol=5e-3)
    assert math.isclose(ci.upper, 0.5961, abs_tol=5e-3)
    assert ci.lower < ci.point < ci.upper


def test_wilson_extremes_clamped() -> None:
    lo = wilson_interval(0, 10)
    assert lo.point == 0.0 and lo.lower == 0.0 and lo.upper > 0.0
    hi = wilson_interval(10, 10)
    assert hi.point == 1.0 and hi.upper == 1.0 and hi.lower < 1.0


def test_wilson_interval_widens_with_higher_confidence() -> None:
    narrow = wilson_interval(20, 100, confidence=0.80)
    wide = wilson_interval(20, 100, confidence=0.99)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


@pytest.mark.parametrize(
    "successes,n,confidence",
    [(-1, 10, 0.95), (5, 4, 0.95), (1, -1, 0.95), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_wilson_rejects_invalid_inputs(
    successes: int, n: int, confidence: float
) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n, confidence=confidence)


def _result(category: PolicyCategory, label: RefusalLabel) -> EvalResult:
    prompt = SafetyPrompt(prompt_id="x", text="t", category=category)
    return EvalResult(
        prompt=prompt,
        response=ModelResponse(text=""),
        predicted_label=label,
        score=1.0 if label == RefusalLabel.REFUSED else 0.0,
    )


def test_per_category_intervals_groups_correctly() -> None:
    results = [
        _result(PolicyCategory.CBRN, RefusalLabel.REFUSED),
        _result(PolicyCategory.CBRN, RefusalLabel.COMPLIED),
        _result(PolicyCategory.MEDICAL_POLICY, RefusalLabel.REFUSED),
        _result(PolicyCategory.MEDICAL_POLICY, RefusalLabel.REFUSED),
    ]
    intervals = per_category_intervals(results)
    assert set(intervals.keys()) == {
        PolicyCategory.CBRN,
        PolicyCategory.MEDICAL_POLICY,
    }
    assert intervals[PolicyCategory.CBRN].point == pytest.approx(0.5)
    assert intervals[PolicyCategory.MEDICAL_POLICY].point == pytest.approx(1.0)
    assert intervals[PolicyCategory.MEDICAL_POLICY].n == 2
