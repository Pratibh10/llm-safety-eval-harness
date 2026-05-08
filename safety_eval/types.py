"""Core typed primitives for the safety evaluation harness.

Everything here is strictly typed and immutable where reasonable. The
harness uses these dataclasses as the unit of exchange between datasets,
model adapters, and evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class PolicyCategory(str, Enum):
    """Coarse policy buckets evaluated by the harness."""

    CBRN = "cbrn"
    CYBER_OFFENSE = "cyber_offense"
    SELF_HARM = "self_harm"
    CSAM = "csam"
    MANIPULATION = "manipulation"
    AUTONOMY = "autonomy"
    BENIGN_CONTROL = "benign_control"
    MEDICAL_POLICY = "medical_policy"


class RefusalLabel(str, Enum):
    """Predicted/expected behavior label for a single response."""

    REFUSED = "refused"
    COMPLIED = "complied"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ProportionCI:
    """Confidence interval for a binomial proportion (e.g. refusal rate)."""

    point: float
    lower: float
    upper: float
    n: int
    successes: int
    confidence: float


@dataclass(frozen=True)
class SafetyPrompt:
    """A single evaluation prompt with provenance metadata."""

    prompt_id: str
    text: str
    category: PolicyCategory
    template_id: Optional[str] = None
    expected_behavior: Optional[RefusalLabel] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response returned by a `ModelAdapter`."""

    text: str
    raw: Any = None
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass(frozen=True)
class EvalResult:
    """Outcome of scoring a single (prompt, response) pair."""

    prompt: SafetyPrompt
    response: ModelResponse
    predicted_label: RefusalLabel
    score: float
    rationale: Optional[str] = None


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregated metrics produced by an evaluator's `aggregate()` call."""

    refusal_rate: float
    compliance_rate: float
    over_refusal_rate: float
    per_category: Mapping[PolicyCategory, float]
    n_trials: int
    results: Sequence[EvalResult]
    overall_ci: Optional[ProportionCI] = None
    per_category_ci: Optional[Mapping[PolicyCategory, ProportionCI]] = None
    per_template_ci: Optional[Mapping[str, ProportionCI]] = None
