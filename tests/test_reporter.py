"""Tests for `SafetyReporter`. Disk and matplotlib are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from safety_eval.reporter import SafetyReporter
from safety_eval.types import (
    EvalMetrics,
    EvalResult,
    ModelResponse,
    PolicyCategory,
    ProportionCI,
    RefusalLabel,
    SafetyPrompt,
)


def _ci(point: float, lower: float, upper: float, n: int) -> ProportionCI:
    return ProportionCI(
        point=point,
        lower=lower,
        upper=upper,
        n=n,
        successes=int(round(point * n)),
        confidence=0.95,
    )


def _eval_result(
    category: PolicyCategory, label: RefusalLabel = RefusalLabel.REFUSED
) -> EvalResult:
    prompt = SafetyPrompt(prompt_id="p", text="t", category=category)
    return EvalResult(
        prompt=prompt,
        response=ModelResponse(text=""),
        predicted_label=label,
        score=1.0 if label == RefusalLabel.REFUSED else 0.0,
    )


def _metrics(*, with_template: bool = False) -> EvalMetrics:
    per_category = {
        PolicyCategory.CBRN: 0.917,
        PolicyCategory.MEDICAL_POLICY: 0.500,
    }
    per_category_ci = {
        PolicyCategory.CBRN: _ci(0.917, 0.646, 0.985, 12),
        PolicyCategory.MEDICAL_POLICY: _ci(0.500, 0.404, 0.596, 100),
    }
    per_template_ci = (
        {
            "identity": _ci(0.250, 0.100, 0.500, 20),
            "wrap": _ci(0.750, 0.500, 0.900, 20),
        }
        if with_template
        else None
    )
    return EvalMetrics(
        refusal_rate=0.7,
        compliance_rate=0.2,
        over_refusal_rate=0.05,
        per_category=per_category,
        n_trials=112,
        results=(
            _eval_result(PolicyCategory.CBRN),
            _eval_result(PolicyCategory.MEDICAL_POLICY, RefusalLabel.COMPLIED),
        ),
        overall_ci=_ci(0.7, 0.61, 0.78, 112),
        per_category_ci=per_category_ci,
        per_template_ci=per_template_ci,
    )


# ----- markdown -----


def test_render_markdown_includes_metadata_and_rates() -> None:
    md = SafetyReporter().render_markdown(
        _metrics(), model_name="anthropic:test-model", eval_name="refusal"
    )
    assert "# Safety Evaluation Report" in md
    assert "anthropic:test-model" in md
    assert "refusal" in md
    assert "**Trials:** 112" in md
    assert "0.700" in md  # refusal rate formatted with 3 decimals
    assert "0.200" in md  # compliance rate
    assert "[0.610, 0.780]" in md  # overall Wilson CI


def test_render_markdown_emits_sorted_per_category_table() -> None:
    md = SafetyReporter().render_markdown(
        _metrics(), model_name="m", eval_name="e"
    )
    assert "| Category | n | Rate | 95% Wilson CI |" in md
    assert "| cbrn | 12 |" in md
    assert "| medical_policy | 100 |" in md
    cbrn_idx = md.index("cbrn")
    med_idx = md.index("medical_policy")
    assert cbrn_idx < med_idx  # alphabetical by enum value


def test_render_markdown_includes_template_section_only_when_present() -> None:
    md_no = SafetyReporter().render_markdown(
        _metrics(with_template=False), model_name="m", eval_name="e"
    )
    assert "Per-template" not in md_no

    md_yes = SafetyReporter().render_markdown(
        _metrics(with_template=True), model_name="m", eval_name="e"
    )
    assert "## Per-template compliance rate" in md_yes
    assert "| identity |" in md_yes and "| wrap |" in md_yes


@pytest.mark.parametrize(
    "banned", ["harmful", "attack", "jailbreak", "unsafe", "malicious"]
)
def test_markdown_uses_only_benign_language(banned: str) -> None:
    md = SafetyReporter().render_markdown(
        _metrics(with_template=True), model_name="m", eval_name="refusal"
    )
    assert banned not in md.lower()


def test_render_markdown_handles_empty_per_category() -> None:
    metrics = EvalMetrics(
        refusal_rate=0.0,
        compliance_rate=0.0,
        over_refusal_rate=0.0,
        per_category={},
        n_trials=0,
        results=(),
        overall_ci=None,
        per_category_ci=None,
    )
    md = SafetyReporter().render_markdown(metrics, model_name="m", eval_name="e")
    assert "_no data_" in md


# ----- chart -----


def test_render_chart_passes_asymmetric_wilson_error_bars(tmp_path: Path) -> None:
    metrics = _metrics()
    captured: dict[str, object] = {}

    real_subplots = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots

    def _fake_subplots(*args: object, **kwargs: object):
        fig, ax = real_subplots(*args, **kwargs)
        original_bar = ax.bar

        def _spy_bar(*a: object, **kw: object):
            captured["bar_args"] = a
            captured["bar_kwargs"] = kw
            return original_bar(*a, **kw)

        ax.bar = _spy_bar  # type: ignore[method-assign]
        return fig, ax

    with (
        patch("safety_eval.reporter.plt.subplots", side_effect=_fake_subplots),
        patch("matplotlib.figure.Figure.savefig") as savefig,
    ):
        SafetyReporter().render_chart(
            metrics, tmp_path / "chart.png", eval_name="refusal"
        )

    yerr = captured["bar_kwargs"]["yerr"]  # type: ignore[index]
    assert isinstance(yerr, list) and len(yerr) == 2  # [lower_err, upper_err]
    lower, upper = yerr
    assert len(lower) == len(upper) == 2  # one entry per category

    # Categories sorted alphabetically: cbrn first, medical_policy second.
    cbrn_ci = metrics.per_category_ci[PolicyCategory.CBRN]
    med_ci = metrics.per_category_ci[PolicyCategory.MEDICAL_POLICY]
    assert lower[0] == pytest.approx(cbrn_ci.point - cbrn_ci.lower)
    assert upper[0] == pytest.approx(cbrn_ci.upper - cbrn_ci.point)
    assert lower[1] == pytest.approx(med_ci.point - med_ci.lower)
    assert upper[1] == pytest.approx(med_ci.upper - med_ci.point)

    # Asymmetric for the CBRN bar (point near 1.0).
    assert lower[0] != pytest.approx(upper[0])

    savefig.assert_called_once()


def test_render_chart_handles_empty_per_category_ci(tmp_path: Path) -> None:
    metrics = EvalMetrics(
        refusal_rate=0.0,
        compliance_rate=0.0,
        over_refusal_rate=0.0,
        per_category={},
        n_trials=0,
        results=(),
        overall_ci=None,
        per_category_ci=None,
    )
    with patch("matplotlib.figure.Figure.savefig") as savefig:
        SafetyReporter().render_chart(metrics, tmp_path / "chart.png")
    savefig.assert_called_once()


# ----- save_report end to end -----


def test_save_report_creates_dir_and_invokes_chart_save(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "out"
    assert not nested.exists()

    with patch("matplotlib.figure.Figure.savefig") as savefig:
        paths = SafetyReporter().save_report(
            _metrics(),
            output_dir=nested,
            model_name="anthropic:claude-x",
            eval_name="refusal",
        )

    assert nested.is_dir()
    assert paths.markdown == nested / "report.md"
    assert paths.chart == nested / "chart.png"
    assert paths.markdown.read_text(encoding="utf-8").startswith(
        "# Safety Evaluation Report"
    )
    savefig.assert_called_once_with(paths.chart, dpi=150)


def test_save_report_writes_markdown_with_expected_content(tmp_path: Path) -> None:
    """Markdown is written to disk; chart save is mocked so no PNG is rendered."""
    with patch("matplotlib.figure.Figure.savefig") as savefig:
        paths = SafetyReporter().save_report(
            _metrics(),
            output_dir=tmp_path,
            model_name="anthropic:claude-x",
            eval_name="refusal",
        )

    md = paths.markdown.read_text(encoding="utf-8")
    assert "anthropic:claude-x" in md
    assert "## Overall rates" in md
    assert "## Per-category refusal rate" in md
    # Chart was "saved" exactly once via the mocked Figure.savefig.
    savefig.assert_called_once()


def test_decimals_negative_rejected() -> None:
    with pytest.raises(ValueError):
        SafetyReporter(decimals=-1)
