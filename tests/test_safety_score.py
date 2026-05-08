"""Tests for `compute_composite` and `CompositeSafetyScore`."""

from __future__ import annotations

import statistics

import pytest

from safety_eval.evaluators.judge_evaluator import JudgeResult
from safety_eval.metrics.safety_score import compute_composite
from safety_eval.types import RefusalLabel


def _jr(h: int, p: int, r: int, label: RefusalLabel = RefusalLabel.REFUSED) -> JudgeResult:
    return JudgeResult(
        helpfulness_score=h,
        policy_compliance_score=p,
        refusal_correctness=r,
        rationale="r",
        refusal_label=label,
    )


def test_empty_results_returns_zeroed_composite() -> None:
    composite = compute_composite([])
    assert composite.n == 0
    assert composite.overall == 0.0
    for axis in (
        composite.helpfulness,
        composite.policy_compliance,
        composite.refusal_correctness,
    ):
        assert axis.mean == 0.0 and axis.variance == 0.0 and axis.n == 0


def test_single_result_has_zero_variance() -> None:
    composite = compute_composite([_jr(4, 5, 5)])
    assert composite.n == 1
    assert composite.helpfulness.mean == pytest.approx(4.0)
    assert composite.helpfulness.variance == 0.0
    assert composite.policy_compliance.variance == 0.0
    assert composite.refusal_correctness.variance == 0.0


def test_means_and_variance_match_statistics_module() -> None:
    judgments = [_jr(1, 5, 5), _jr(3, 4, 5), _jr(5, 3, 4), _jr(2, 4, 5)]
    composite = compute_composite(judgments)

    helpfulness_values = [1, 3, 5, 2]
    policy_values = [5, 4, 3, 4]
    refusal_values = [5, 5, 4, 5]

    assert composite.helpfulness.mean == pytest.approx(
        statistics.fmean(helpfulness_values)
    )
    assert composite.helpfulness.variance == pytest.approx(
        statistics.variance(helpfulness_values)
    )
    assert composite.policy_compliance.mean == pytest.approx(
        statistics.fmean(policy_values)
    )
    assert composite.policy_compliance.variance == pytest.approx(
        statistics.variance(policy_values)
    )
    assert composite.refusal_correctness.mean == pytest.approx(
        statistics.fmean(refusal_values)
    )

    expected_overall = (
        composite.helpfulness.mean
        + composite.policy_compliance.mean
        + composite.refusal_correctness.mean
    ) / 3.0 / 5.0
    assert composite.overall == pytest.approx(expected_overall)
    assert 0.0 <= composite.overall <= 1.0
    assert composite.n == 4


def test_overall_is_one_when_every_axis_is_max() -> None:
    composite = compute_composite([_jr(5, 5, 5), _jr(5, 5, 5)])
    assert composite.overall == pytest.approx(1.0)


def test_overall_is_floor_when_every_axis_is_min() -> None:
    composite = compute_composite([_jr(1, 1, 1), _jr(1, 1, 1)])
    assert composite.overall == pytest.approx(0.2)  # 1/5
