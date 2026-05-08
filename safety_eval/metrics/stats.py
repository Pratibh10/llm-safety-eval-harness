"""Statistical helpers for binomial proportion intervals."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist
from typing import Mapping, Sequence

from safety_eval.types import EvalResult, PolicyCategory, ProportionCI, RefusalLabel


def _z_for_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    alpha = 1.0 - confidence
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson_interval(
    successes: int, n: int, *, confidence: float = 0.95
) -> ProportionCI:
    """Wilson score interval for a binomial proportion.

    Falls back to a degenerate interval at the point estimate when ``n == 0``.
    """
    if successes < 0:
        raise ValueError(f"successes must be >= 0, got {successes}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")

    if n == 0:
        return ProportionCI(
            point=0.0,
            lower=0.0,
            upper=0.0,
            n=0,
            successes=0,
            confidence=confidence,
        )

    z = _z_for_confidence(confidence)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return ProportionCI(
        point=p,
        lower=lower,
        upper=upper,
        n=n,
        successes=successes,
        confidence=confidence,
    )


def per_category_intervals(
    results: Sequence[EvalResult],
    *,
    success_label: RefusalLabel = RefusalLabel.REFUSED,
    confidence: float = 0.95,
) -> Mapping[PolicyCategory, ProportionCI]:
    """Compute a Wilson interval per ``PolicyCategory`` over ``results``."""
    buckets: dict[PolicyCategory, list[EvalResult]] = defaultdict(list)
    for r in results:
        buckets[r.prompt.category].append(r)

    out: dict[PolicyCategory, ProportionCI] = {}
    for category, items in buckets.items():
        successes = sum(1 for r in items if r.predicted_label == success_label)
        out[category] = wilson_interval(successes, len(items), confidence=confidence)
    return out
