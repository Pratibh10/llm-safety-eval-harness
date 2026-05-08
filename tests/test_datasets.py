"""Tests for dataset implementations."""

from __future__ import annotations

import pytest

from safety_eval.datasets.robustness_templates import (
    MOCK_TEMPLATES,
    RobustnessTemplateDataset,
)
from safety_eval.datasets.stress_test_prompts import StressTestPromptsDataset
from safety_eval.types import PolicyCategory, RefusalLabel


def test_stress_test_dataset_loads_two_mock_items() -> None:
    ds = StressTestPromptsDataset()
    items = list(ds)
    assert len(ds) == 2
    assert {p.prompt_id for p in items} == {"id1", "id2"}

    by_id = {p.prompt_id: p for p in items}
    assert by_id["id1"].text == "Say apple"
    assert by_id["id1"].category == PolicyCategory.MEDICAL_POLICY
    assert by_id["id1"].expected_behavior == RefusalLabel.REFUSED
    assert by_id["id2"].text == "Say banana"
    assert by_id["id2"].category == PolicyCategory.CBRN
    assert by_id["id2"].expected_behavior == RefusalLabel.COMPLIED


def test_stress_test_dataset_filter_by_category() -> None:
    ds = StressTestPromptsDataset()
    cbrn = ds.filter_by_category(PolicyCategory.CBRN)
    assert len(cbrn) == 1 and cbrn[0].prompt_id == "id2"
    empty = ds.filter_by_category(PolicyCategory.CYBER_OFFENSE)
    assert empty == []


def test_stress_test_dataset_sample_is_deterministic() -> None:
    ds = StressTestPromptsDataset()
    a = ds.sample(1, seed=7)
    b = ds.sample(1, seed=7)
    assert [p.prompt_id for p in a] == [p.prompt_id for p in b]
    # k larger than dataset returns the full set.
    assert len(ds.sample(99)) == len(ds)


def test_robustness_template_dataset_uses_identity_template() -> None:
    base = StressTestPromptsDataset()
    wrapped = RobustnessTemplateDataset(base)
    assert wrapped.templates == MOCK_TEMPLATES
    items = list(wrapped)
    assert len(items) == 2  # one template * two base items
    for item in items:
        assert item.template_id == "identity"
        assert item.metadata["base_prompt_id"] in {"id1", "id2"}
        assert item.text in {"Say apple", "Say banana"}


def test_robustness_template_dataset_multi_template_expansion() -> None:
    base = StressTestPromptsDataset()
    wrapped = RobustnessTemplateDataset(
        base,
        templates={"identity": "{prompt}", "wrap": "Please: {prompt}!"},
    )
    items = list(wrapped)
    assert len(items) == 4
    template_ids = {i.template_id for i in items}
    assert template_ids == {"identity", "wrap"}
    wrapped_text = next(i.text for i in items if i.template_id == "wrap")
    assert wrapped_text.startswith("Please:") and wrapped_text.endswith("!")


def test_robustness_template_dataset_rejects_empty_templates() -> None:
    with pytest.raises(ValueError):
        RobustnessTemplateDataset(StressTestPromptsDataset(), templates={})
