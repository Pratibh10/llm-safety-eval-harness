"""Shared fixtures and a deterministic mock model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Sequence

import pytest

from safety_eval.models.base import ModelAdapter
from safety_eval.types import ModelResponse


@dataclass
class ScriptedModel(ModelAdapter):
    """Deterministic adapter that returns canned responses.

    ``responder`` may be a mapping-like callable from prompt -> text, or a
    list of fixed responses returned in order.
    """

    responder: Callable[[str], str] = field(default=lambda _: "ok")
    name: str = "scripted-mock"
    calls: list[str] = field(default_factory=list)

    async def generate(self, prompt: str, **kwargs: object) -> ModelResponse:
        self.calls.append(prompt)
        text = self.responder(prompt)
        return ModelResponse(text=text, raw=None, stop_reason="end_turn")

    async def stream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        async def _agen() -> AsyncIterator[str]:
            text = self.responder(prompt)
            for chunk in text.split(" "):
                yield chunk
        return _agen()


def make_scripted(responses: Sequence[str]) -> ScriptedModel:
    """Build a ScriptedModel that yields ``responses`` round-robin by prompt order."""
    iterator = iter(responses)

    def _responder(_prompt: str) -> str:
        try:
            return next(iterator)
        except StopIteration:  # pragma: no cover - safety net
            return ""

    return ScriptedModel(responder=_responder)


@pytest.fixture
def scripted_model_factory() -> Callable[[Sequence[str]], ScriptedModel]:
    return make_scripted
