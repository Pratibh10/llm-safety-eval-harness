"""Abstract dataset interface."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Iterator, Sequence

from safety_eval.types import PolicyCategory, SafetyPrompt


class SafetyDataset(ABC):
    """Iterable collection of `SafetyPrompt` items with category filtering."""

    _items: Sequence[SafetyPrompt]

    @abstractmethod
    def load(self) -> Sequence[SafetyPrompt]:
        """Materialize and cache the dataset items, returning them."""

    def __iter__(self) -> Iterator[SafetyPrompt]:
        return iter(self._ensure_loaded())

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def filter_by_category(self, category: PolicyCategory) -> list[SafetyPrompt]:
        return [p for p in self._ensure_loaded() if p.category == category]

    def sample(self, k: int, seed: int | None = None) -> list[SafetyPrompt]:
        items = list(self._ensure_loaded())
        if k >= len(items):
            return items
        rng = random.Random(seed)
        return rng.sample(items, k)

    def _ensure_loaded(self) -> Sequence[SafetyPrompt]:
        if not hasattr(self, "_items") or self._items is None:
            self._items = self.load()
        return self._items
