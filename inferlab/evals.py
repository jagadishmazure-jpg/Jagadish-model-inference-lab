"""Golden eval set: 10 synthetic prompt/keyword pairs.

run() scores a pipeline function (prompt -> text) by keyword hit-rate;
compare() runs every model through FakeBackend and ranks by score, then
cost — so routing decisions can be validated, not just asserted.
All synthetic; no external data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clients import FakeBackend
from .models import ModelCard, ModelRegistry


@dataclass(frozen=True)
class GoldenItem:
    prompt: str
    expected_keywords: tuple[str, ...]


GOLDEN_SET: tuple[GoldenItem, ...] = (
    GoldenItem("What is 2 + 2?", ("4", "four")),
    GoldenItem("Name the capital of France.", ("paris",)),
    GoldenItem("Summarize: the cat sat on the mat.", ("cat", "mat")),
    GoldenItem("Translate 'hello' to Spanish.", ("hola",)),
    GoldenItem("What color is the sky on a clear day?", ("blue",)),
    GoldenItem("List two primary colors.", ("red", "blue", "yellow")),
    GoldenItem("Is water wet? Answer yes or no.", ("yes",)),
    GoldenItem("What is the opposite of hot?", ("cold",)),
    GoldenItem("How many days in a week?", ("7", "seven")),
    GoldenItem("Name a planet.", ("earth", "mars", "jupiter", "venus",
                                  "mercury", "saturn", "uranus", "neptune")),
)


class GoldenSet:
    def __init__(self, items: tuple[GoldenItem, ...] = GOLDEN_SET) -> None:
        self.items = list(items)

    def score(self, pipeline_fn) -> dict:
        """pipeline_fn(prompt) -> text. Returns hits/total and per-item detail."""
        hits = 0
        detail = []
        for item in self.items:
            text = pipeline_fn(item.prompt).lower()
            hit = any(re.search(r"\b" + re.escape(kw.lower()) + r"\b", text)
                      for kw in item.expected_keywords)
            hits += int(hit)
            detail.append({"prompt": item.prompt, "hit": hit})
        return {"hits": hits, "total": len(self.items),
                "hit_rate": hits / len(self.items) if self.items else 0.0,
                "detail": detail}

    def compare(self, registry: ModelRegistry | None = None) -> list[dict]:
        """Run each registered model via FakeBackend; rank by score then cost."""
        registry = registry or ModelRegistry()
        backend = FakeBackend(registry)
        rows = []
        for card in registry.list():
            result = self.score(
                lambda p, m=card.name: backend.complete(p, m).text)
            # Reference cost: 100 in / 50 out tokens for ranking.
            ref_cost = card.cost_usd(100, 50)
            rows.append({"model": card.name, "license": card.license,
                         "hit_rate": result["hit_rate"],
                         "ref_cost_usd": ref_cost})
        rows.sort(key=lambda r: (-r["hit_rate"], r["ref_cost_usd"]))
        return rows
