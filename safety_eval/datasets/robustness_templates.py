"""Robustness wrapper that applies templates to a base safety dataset.

Only one mock template is shipped (``"{prompt}"``) so the wrapper is a
no-op transformation by default. Real templates would live in a config
file alongside this module.
"""

from __future__ import annotations

from typing import Final, Sequence

from safety_eval.datasets.base import SafetyDataset
from safety_eval.types import SafetyPrompt

MOCK_TEMPLATES: Final[dict[str, str]] = {
    "identity": "{prompt}",
}


class RobustnessTemplateDataset(SafetyDataset):
    """Wraps a base dataset and yields one prompt per (item, template) pair."""

    def __init__(
        self,
        base: SafetyDataset,
        templates: dict[str, str] | None = None,
    ) -> None:
        self._base = base
        self._templates: dict[str, str] = (
            dict(templates) if templates is not None else dict(MOCK_TEMPLATES)
        )
        if not self._templates:
            raise ValueError("RobustnessTemplateDataset requires at least one template")

    @property
    def templates(self) -> dict[str, str]:
        return dict(self._templates)

    def load(self) -> Sequence[SafetyPrompt]:
        base_items = list(self._base)
        rendered: list[SafetyPrompt] = []
        for template_id, template in self._templates.items():
            for item in base_items:
                rendered.append(
                    SafetyPrompt(
                        prompt_id=f"{item.prompt_id}::{template_id}",
                        text=template.format(prompt=item.text),
                        category=item.category,
                        template_id=template_id,
                        expected_behavior=item.expected_behavior,
                        metadata={
                            **dict(item.metadata),
                            "base_prompt_id": item.prompt_id,
                        },
                    )
                )
        self._items = rendered
        return rendered
