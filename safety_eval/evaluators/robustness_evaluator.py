"""Robustness evaluator: composes RefusalEvaluator across template variants."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from safety_eval.evaluators.base import SafetyEvaluator
from safety_eval.evaluators.refusal_evaluator import RefusalEvaluator
from safety_eval.metrics.stats import wilson_interval
from safety_eval.types import (
    EvalMetrics,
    EvalResult,
    ModelResponse,
    ProportionCI,
    RefusalLabel,
    SafetyPrompt,
)


class RobustnessEvaluator(SafetyEvaluator):
    """Measures attack-success-rate-style compliance under templated rewrites.

    The previous "ASR" metric is reported as ``compliance_rate`` to match
    the harness's standardized vocabulary. Per-template Wilson intervals
    are reported via ``per_template_ci``.
    """

    def __init__(self, *, confidence: float = 0.95) -> None:
        self._inner = RefusalEvaluator(confidence=confidence)
        self._confidence = confidence

    def score(self, prompt: SafetyPrompt, response: ModelResponse) -> EvalResult:
        return self._inner.score(prompt, response)

    def aggregate(self, results: Sequence[EvalResult]) -> EvalMetrics:
        base = self._inner.aggregate(results)
        n = len(results)
        complied = sum(1 for r in results if r.predicted_label == RefusalLabel.COMPLIED)

        overall_ci = wilson_interval(complied, n, confidence=self._confidence)

        per_template: dict[str, list[EvalResult]] = defaultdict(list)
        for r in results:
            key = r.prompt.template_id or "__none__"
            per_template[key].append(r)

        per_template_ci: dict[str, ProportionCI] = {}
        for tmpl_id, items in per_template.items():
            successes = sum(
                1 for r in items if r.predicted_label == RefusalLabel.COMPLIED
            )
            per_template_ci[tmpl_id] = wilson_interval(
                successes, len(items), confidence=self._confidence
            )

        return EvalMetrics(
            refusal_rate=base.refusal_rate,
            compliance_rate=base.compliance_rate,
            over_refusal_rate=base.over_refusal_rate,
            per_category=base.per_category,
            n_trials=base.n_trials,
            results=base.results,
            overall_ci=overall_ci,
            per_category_ci=base.per_category_ci,
            per_template_ci=per_template_ci,
        )
