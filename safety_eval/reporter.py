"""Markdown + chart reporter for `EvalMetrics`.

Output is written to a single ``output_dir``:

- ``report.md`` — professional markdown summary.
- ``chart.png`` — bar chart of per-category rates with asymmetric Wilson
  error bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / servers
import matplotlib.pyplot as plt  # noqa: E402  (must follow `use("Agg")`)

from safety_eval.types import EvalMetrics, PolicyCategory, ProportionCI


@dataclass(frozen=True)
class ReportPaths:
    markdown: Path
    chart: Path


def _fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _fmt_ci(ci: ProportionCI | None, decimals: int) -> str:
    if ci is None:
        return "—"
    return (
        f"[{_fmt(ci.lower, decimals)}, {_fmt(ci.upper, decimals)}] "
        f"(n={ci.n}, conf={ci.confidence:.2f})"
    )


class SafetyReporter:
    """Render `EvalMetrics` to disk as markdown + chart."""

    def __init__(self, *, decimals: int = 3) -> None:
        if decimals < 0:
            raise ValueError("decimals must be non-negative")
        self.decimals = decimals

    def save_report(
        self,
        metrics: EvalMetrics,
        output_dir: Path,
        model_name: str,
        eval_name: str,
    ) -> ReportPaths:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / "report.md"
        chart_path = output_dir / "chart.png"

        md_path.write_text(
            self.render_markdown(metrics, model_name=model_name, eval_name=eval_name),
            encoding="utf-8",
        )
        self.render_chart(metrics, chart_path, eval_name=eval_name)
        return ReportPaths(markdown=md_path, chart=chart_path)

    # ----- markdown -----

    def render_markdown(
        self, metrics: EvalMetrics, *, model_name: str, eval_name: str
    ) -> str:
        d = self.decimals
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        lines: list[str] = [
            "# Safety Evaluation Report",
            "",
            f"- **Model:** {model_name}",
            f"- **Evaluator:** {eval_name}",
            f"- **Trials:** {metrics.n_trials}",
            f"- **Generated:** {timestamp}",
            "",
            "## Overall rates",
            "",
            "| Metric | Rate | 95% Wilson CI |",
            "| --- | --- | --- |",
            f"| Refusal rate | {_fmt(metrics.refusal_rate, d)} | "
            f"{_fmt_ci(metrics.overall_ci, d)} |",
            f"| Compliance rate | {_fmt(metrics.compliance_rate, d)} | — |",
            f"| Over-refusal rate | {_fmt(metrics.over_refusal_rate, d)} | — |",
            "",
            "## Per-category refusal rate",
            "",
        ]
        lines.extend(self._render_category_table(metrics))

        if metrics.per_template_ci:
            lines.append("")
            lines.append("## Per-template compliance rate")
            lines.append("")
            lines.extend(self._render_template_table(metrics.per_template_ci))

        lines.extend(
            [
                "",
                "## Methodology",
                "",
                "Confidence intervals are computed using the Wilson score "
                "interval, which performs reliably for small sample sizes and "
                "for proportions near 0 or 1. Per-category and per-template "
                "intervals are computed independently over their respective "
                "subsets of trials.",
                "",
            ]
        )
        return "\n".join(lines)

    def _render_category_table(self, metrics: EvalMetrics) -> list[str]:
        d = self.decimals
        rows: list[str] = [
            "| Category | n | Rate | 95% Wilson CI |",
            "| --- | --- | --- | --- |",
        ]
        per_cat_ci = metrics.per_category_ci or {}
        # Stable ordering by category value for deterministic snapshots.
        categories = sorted(
            set(metrics.per_category) | set(per_cat_ci),
            key=lambda c: c.value,
        )
        if not categories:
            rows.append("| _no data_ | 0 | — | — |")
            return rows
        for category in categories:
            rate = metrics.per_category.get(category, 0.0)
            ci = per_cat_ci.get(category)
            n = ci.n if ci is not None else 0
            rows.append(
                f"| {category.value} | {n} | {_fmt(rate, d)} | {_fmt_ci(ci, d)} |"
            )
        return rows

    def _render_template_table(
        self, per_template: Mapping[str, ProportionCI]
    ) -> list[str]:
        d = self.decimals
        rows: list[str] = [
            "| Template | n | Rate | 95% Wilson CI |",
            "| --- | --- | --- | --- |",
        ]
        for tmpl_id in sorted(per_template):
            ci = per_template[tmpl_id]
            rows.append(
                f"| {tmpl_id} | {ci.n} | {_fmt(ci.point, d)} | {_fmt_ci(ci, d)} |"
            )
        return rows

    # ----- chart -----

    def render_chart(
        self,
        metrics: EvalMetrics,
        output_path: Path,
        *,
        eval_name: str = "evaluation",
    ) -> None:
        per_cat_ci = metrics.per_category_ci or {}
        # Stable ordering for deterministic visual output.
        categories: Sequence[PolicyCategory] = sorted(
            per_cat_ci.keys(), key=lambda c: c.value
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            if categories:
                labels = [c.value for c in categories]
                points = [per_cat_ci[c].point for c in categories]
                lower_err = [per_cat_ci[c].point - per_cat_ci[c].lower for c in categories]
                upper_err = [per_cat_ci[c].upper - per_cat_ci[c].point for c in categories]
                ax.bar(
                    labels,
                    points,
                    yerr=[lower_err, upper_err],
                    capsize=4,
                    color="#4C78A8",
                    edgecolor="black",
                )
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha="right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No per-category data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Refusal rate")
            ax.set_title(f"{eval_name} — refusal rate by policy category")
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)
