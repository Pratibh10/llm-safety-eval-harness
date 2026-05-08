"""Concrete `ModelAdapter` backed by the official `anthropic` SDK.

The SDK is imported lazily so the package can be used in test
environments (with an injected mock client) without `anthropic`
installed.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from safety_eval.models.base import ModelAdapter
from safety_eval.types import ModelResponse

DEFAULT_MODEL = "claude-3-haiku-20240307"
DEFAULT_MAX_TOKENS = 1024


def _extract_text(content: Any) -> str:
    """Concatenate text from a `Message.content` block list."""
    if content is None:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class AnthropicModel(ModelAdapter):
    """`ModelAdapter` that calls Anthropic's Messages API asynchronously."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"
        if client is not None:
            self._client = client
        else:
            from anthropic import AsyncAnthropic  # lazy import

            self._client = AsyncAnthropic(api_key=api_key)

    @property
    def client(self) -> Any:
        return self._client

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
        model = str(kwargs.pop("model", self.model))
        message = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        usage = getattr(message, "usage", None)
        return ModelResponse(
            text=_extract_text(getattr(message, "content", None)),
            raw=message,
            stop_reason=getattr(message, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
        )

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
        model = str(kwargs.pop("model", self.model))
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk
