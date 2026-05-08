"""Abstract base for model adapters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence

from safety_eval.types import ModelResponse


class ModelAdapter(ABC):
    """Adapter wrapping any LLM backend in a uniform async interface."""

    name: str = "unnamed-model"

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: object) -> ModelResponse:
        """Generate a single response for a prompt."""

    async def generate_batch(
        self, prompts: Sequence[str], **kwargs: object
    ) -> list[ModelResponse]:
        """Default batch implementation runs `generate` concurrently."""
        return list(await asyncio.gather(*(self.generate(p, **kwargs) for p in prompts)))

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Yield incremental text chunks for a prompt."""
