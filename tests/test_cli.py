"""Tests for the Typer CLI. Network calls and chart rendering are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from safety_eval import cli as cli_module
from safety_eval.cli import app
from safety_eval.types import ModelResponse


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fake_generate() -> AsyncMock:
    """`AsyncMock` standing in for `AnthropicModel.generate`."""
    return AsyncMock(return_value=ModelResponse(text="i cannot assist", raw=None))


@pytest.fixture
def patched_environment(
    monkeypatch: pytest.MonkeyPatch, fake_generate: AsyncMock
) -> AsyncMock:
    """Patch the SDK call site and the chart renderer so no network/disk-art is hit."""
    monkeypatch.setattr(cli_module.AnthropicModel, "generate", fake_generate)
    # Skip the real chart so tests don't depend on matplotlib rendering.
    monkeypatch.setattr(
        cli_module.SafetyReporter,
        "render_chart",
        lambda self, metrics, output_path, **kw: output_path.write_bytes(b""),
    )
    # Stop AsyncAnthropic from being instantiated even with a dummy key.
    monkeypatch.setattr(
        cli_module.AnthropicModel,
        "__init__",
        lambda self, api_key=None, *, model="test-model", max_tokens=1024, client=None: setattr(
            self, "model", model
        )
        or setattr(self, "max_tokens", max_tokens)
        or setattr(self, "name", f"anthropic:{model}")
        or setattr(self, "_client", object()),
    )
    return fake_generate


def test_run_refusal_evaluator_writes_report(
    runner: CliRunner,
    tmp_path: Path,
    patched_environment: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "anthropic",
            "--evaluator",
            "refusal",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "chart.png").exists()
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Safety Evaluation Report" in md
    assert "Refusal rate" in md
    # Stress dataset has 2 prompts, both should be classified REFUSED.
    assert "**Trials:** 2" in md
    # Rich summary printed to stdout.
    assert "Refusal rate" in result.stdout
    # No banned safety vocabulary in the user-facing summary.
    for banned in ("harmful", "attack", "jailbreak"):
        assert banned not in md.lower()
    # generate() was awaited once per dataset prompt.
    assert patched_environment.await_count == 2


def test_run_robustness_evaluator_includes_template_section(
    runner: CliRunner,
    tmp_path: Path,
    patched_environment: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    # Have the model "comply" so the template table populates with non-zero rates.
    patched_environment.return_value = ModelResponse(text="here is how to do it")

    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "anthropic",
            "--evaluator",
            "robustness",
            "--output-dir",
            str(tmp_path),
            "--name",
            "robust-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Per-template compliance rate" in md
    assert "| identity |" in md
    assert "robust-run" in md


def test_run_warns_when_api_key_missing(
    runner: CliRunner,
    tmp_path: Path,
    patched_environment: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "run",
            "--evaluator",
            "refusal",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "ANTHROPIC_API_KEY" in result.stdout


def test_run_requires_evaluator(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_run_rejects_invalid_evaluator(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--evaluator",
            "not-a-real-evaluator",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0


def test_run_with_no_args_shows_help(runner: CliRunner) -> None:
    result = runner.invoke(app, [])
    # `no_args_is_help=True` produces help output (to stdout or stderr).
    output = result.stdout + (result.stderr if hasattr(result, "stderr") else "")
    assert "Usage" in output
