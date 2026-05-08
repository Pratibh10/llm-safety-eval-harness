"""Abstract base for safety evaluators."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from safety_eval.datasets.base import SafetyDataset
from safety_eval.models.base import ModelAdapter
from safety_eval.types import EvalMetrics, EvalResult, ModelResponse, SafetyPrompt


class SafetyEvaluator(ABC):
    """Base evaluator: scores individual (prompt, response) pairs and aggregates."""

    @abstractmethod
    def score(self, prompt: SafetyPrompt, response: ModelResponse) -> EvalResult:
        """Score a single prompt/response pair."""

    @abstractmethod
    def aggregate(self, results: Sequence[EvalResult]) -> EvalMetrics:
        """Aggregate per-item results into harness metrics."""

    async def run(
        self,
        model: ModelAdapter,
        dataset: SafetyDataset | Iterable[SafetyPrompt],
        *,
        concurrency: int = 8,
        **gen_kwargs: object,
    ) -> EvalMetrics:
        """Run the evaluator against `model` over every prompt in `dataset`."""
        prompts: list[SafetyPrompt] = list(dataset)
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _one(prompt: SafetyPrompt) -> EvalResult:
            async with semaphore:
                response = await model.generate(prompt.text, **gen_kwargs)
            return self.score(prompt, response)

        results = await asyncio.gather(*(_one(p) for p in prompts))
        return self.aggregate(results)
