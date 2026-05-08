"""Tests for `AnthropicModel`. The Anthropic SDK is fully mocked — no network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from safety_eval.models.anthropic_model import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AnthropicModel,
)


def _fake_message(
    text: str = "hello",
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 7,
    output_tokens: int = 11,
) -> SimpleNamespace:
    """Build an object that quacks like an Anthropic `Message`."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


def _make_client(message: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=message)
    return client


def test_init_defaults_and_name() -> None:
    client = _make_client(_fake_message())
    model = AnthropicModel(client=client)
    assert model.model == DEFAULT_MODEL
    assert model.max_tokens == DEFAULT_MAX_TOKENS
    assert model.name == f"anthropic:{DEFAULT_MODEL}"
    assert model.client is client


async def test_generate_maps_response_and_forwards_kwargs() -> None:
    client = _make_client(_fake_message(text="world", stop_reason="stop"))
    model = AnthropicModel(client=client, model="custom-model", max_tokens=256)

    resp = await model.generate("hi there", temperature=0.2)

    client.messages.create.assert_awaited_once()
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "custom-model"
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"] == [{"role": "user", "content": "hi there"}]

    assert resp.text == "world"
    assert resp.stop_reason == "stop"
    assert resp.input_tokens == 7
    assert resp.output_tokens == 11
    assert resp.raw is not None


async def test_generate_per_call_overrides_defaults() -> None:
    client = _make_client(_fake_message())
    model = AnthropicModel(client=client, model="default-model", max_tokens=128)
    await model.generate("x", model="other-model", max_tokens=999)
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "other-model"
    assert kwargs["max_tokens"] == 999


async def test_generate_concatenates_multi_block_content() -> None:
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello "),
            SimpleNamespace(type="text", text="world"),
        ],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )
    client = _make_client(message)
    model = AnthropicModel(client=client)
    resp = await model.generate("p")
    assert resp.text == "hello world"


async def test_generate_handles_missing_usage() -> None:
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        usage=None,
    )
    client = _make_client(message)
    model = AnthropicModel(client=client)
    resp = await model.generate("p")
    assert resp.input_tokens is None and resp.output_tokens is None


async def test_generate_batch_calls_create_per_prompt() -> None:
    client = _make_client(_fake_message())
    model = AnthropicModel(client=client)
    out = await model.generate_batch(["one", "two", "three"])
    assert len(out) == 3
    assert client.messages.create.await_count == 3
    seen_prompts = [
        call.kwargs["messages"][0]["content"]
        for call in client.messages.create.await_args_list
    ]
    assert seen_prompts == ["one", "two", "three"]


async def test_generate_propagates_sdk_errors() -> None:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("boom"))
    model = AnthropicModel(client=client)
    with pytest.raises(RuntimeError, match="boom"):
        await model.generate("p")


def test_lazy_import_when_no_client_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no client is given, the SDK is imported and instantiated with the api_key."""
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, api_key: str | None = None) -> None:
            captured["api_key"] = api_key

    fake_module = SimpleNamespace(AsyncAnthropic=_FakeAsyncAnthropic)
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)
    AnthropicModel(api_key="sk-test")
    assert captured["api_key"] == "sk-test"
