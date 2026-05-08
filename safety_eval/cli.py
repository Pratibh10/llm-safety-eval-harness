"""Typer CLI orchestrating dataset, model, evaluator, and reporter."""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from safety_eval.datasets.robustness_templates import RobustnessTemplateDataset
from safety_eval.datasets.stress_test_prompts import StressTestPromptsDataset
from safety_eval.evaluators.base import SafetyEvaluator
from safety_eval.evaluators.refusal_evaluator import RefusalEvaluator
from safety_eval.evaluators.robustness_evaluator import RobustnessEvaluator
from safety_eval.models.anthropic_model import AnthropicModel
from safety_eval.models.base import ModelAdapter
from safety_eval.reporter import SafetyReporter
from safety_eval.types import EvalMetrics


class ModelChoice(str, Enum):
    ANTHROPIC = "anthropic"


class EvaluatorChoice(str, Enum):
    REFUSAL = "refusal"
    ROBUSTNESS = "robustness"


app = typer.Typer(
    name="safety-eval",
    no_args_is_help=True,
    add_completion=False,
    help="Run benign-data safety evaluations and emit a markdown + chart report.",
)


@app.callback()
def _root() -> None:
    """Force multi-command mode so `safety-eval run ...` works as documented."""


def _build_model(choice: ModelChoice, *, console: Console) -> ModelAdapter:
    if choice is ModelChoice.ANTHROPIC:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print(
                "[yellow]warning:[/] ANTHROPIC_API_KEY is not set; "
                "using a placeholder key. Real API calls will fail.",
                highlight=False,
            )
            api_key = "dummy-dry-run-key"
        return AnthropicModel(api_key=api_key)
    raise typer.BadParameter(f"unsupported model: {choice}")


def _build_evaluator(choice: EvaluatorChoice) -> SafetyEvaluator:
    if choice is EvaluatorChoice.REFUSAL:
        return RefusalEvaluator()
    if choice is EvaluatorChoice.ROBUSTNESS:
        return RobustnessEvaluator()
    raise typer.BadParameter(f"unsupported evaluator: {choice}")


def _build_dataset(choice: EvaluatorChoice):
    base = StressTestPromptsDataset()
    if choice is EvaluatorChoice.ROBUSTNESS:
        return RobustnessTemplateDataset(base)
    return base


def _print_summary(
    console: Console,
    *,
    metrics: EvalMetrics,
    model_name: str,
    eval_name: str,
    md_path: Path,
    chart_path: Path,
) -> None:
    table = Table(title=f"{eval_name} — {model_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("95% Wilson CI", justify="right")

    overall = metrics.overall_ci
    overall_ci_str = (
        f"[{overall.lower:.3f}, {overall.upper:.3f}]" if overall is not None else "—"
    )
    table.add_row("Trials", str(metrics.n_trials), "—")
    table.add_row("Refusal rate", f"{metrics.refusal_rate:.3f}", overall_ci_str)
    table.add_row("Compliance rate", f"{metrics.compliance_rate:.3f}", "—")
    table.add_row("Over-refusal rate", f"{metrics.over_refusal_rate:.3f}", "—")

    console.print(table)
    console.print(f"Report:  [cyan]{md_path}[/]", highlight=False)
    console.print(f"Chart:   [cyan]{chart_path}[/]", highlight=False)


@app.command("run")
def run(
    model: ModelChoice = typer.Option(ModelChoice.ANTHROPIC, "--model"),
    evaluator: EvaluatorChoice = typer.Option(..., "--evaluator"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    concurrency: int = typer.Option(4, "--concurrency", min=1, max=64),
    eval_name: str = typer.Option("safety-eval", "--name"),
) -> None:
    """Execute an evaluation against the bundled benign mock dataset."""
    console = Console()

    model_instance = _build_model(model, console=console)
    evaluator_instance = _build_evaluator(evaluator)
    dataset = _build_dataset(evaluator)

    console.print(
        f"Running [bold]{evaluator.value}[/] evaluator against "
        f"[bold]{model_instance.name}[/] over {len(list(dataset))} prompts...",
        highlight=False,
    )
    metrics = asyncio.run(
        evaluator_instance.run(model_instance, dataset, concurrency=concurrency)
    )

    paths = SafetyReporter().save_report(
        metrics,
        output_dir=output_dir,
        model_name=model_instance.name,
        eval_name=eval_name,
    )
    _print_summary(
        console,
        metrics=metrics,
        model_name=model_instance.name,
        eval_name=eval_name,
        md_path=paths.markdown,
        chart_path=paths.chart,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
