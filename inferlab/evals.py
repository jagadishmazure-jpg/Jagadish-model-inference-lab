"""Golden eval set: 16 synthetic prompt/keyword pairs.

Each item's task class (``simple`` or ``complex``) is assigned by the same
complexity estimator the router uses, so scores line up with routing
decisions. In the shipped set that gives 9 simple and 7 complex items
("How many days in a week?" counts as complex because of "how").

``score()`` grades any pipeline (prompt -> text) by keyword hit rate.
``compare()`` runs every model and reports the hit rate per task class next
to the reference-call cost, which is what ``EvalInformedRouter`` routes on.
Offline, ``compare()`` uses FakeBackend, whose quality is **simulated** (see
``clients.FakeBackend``); point it at real backends to get real numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clients import Backend, FakeBackend
from .models import ModelRegistry
from .router import estimate_complexity


@dataclass(frozen=True)
class GoldenItem:
    prompt: str
    expected_keywords: tuple[str, ...]

    @property
    def task_class(self) -> str:
        return estimate_complexity(self.prompt)


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
    GoldenItem("Explain why a cache speeds up repeated reads.", ("memory", "latency")),
    GoldenItem("Compare a queue and a topic for event delivery.", ("consumer", "subscribers")),
    GoldenItem("Explain the tradeoff between consistency and availability.", ("partition",)),
    GoldenItem("Why do retries need jitter?", ("thundering", "herd")),
    GoldenItem("Explain how a circuit breaker protects a caller.", ("open", "failures")),
    GoldenItem("Evaluate when to fine-tune instead of prompting.", ("data", "examples")),
)
ITEM_BY_PROMPT = {item.prompt: item for item in GOLDEN_SET}


class GoldenSet:
    def __init__(self, items: tuple[GoldenItem, ...] = GOLDEN_SET) -> None:
        self.items = list(items)

    def score(self, pipeline_fn, task_class: str | None = None) -> dict:
        items = [i for i in self.items if task_class in (None, i.task_class)]
        hits = 0
        detail = []
        for item in items:
            text = pipeline_fn(item.prompt).lower()
            hit = any(re.search(r"\b" + re.escape(kw.lower()) + r"\b", text)
                      for kw in item.expected_keywords)
            hits += int(hit)
            detail.append({"prompt": item.prompt, "class": item.task_class, "hit": hit})
        return {"hits": hits, "total": len(items),
                "hit_rate": hits / len(items) if items else 0.0, "detail": detail}

    def compare(self, registry: ModelRegistry | None = None,
                backend: Backend | None = None) -> list[dict]:
        """Hit rate per task class and reference cost for every model."""
        registry = registry or ModelRegistry()
        backend = backend or FakeBackend(registry)
        rows = []
        for card in registry.list():
            def fn(p, m=card.name):
                return backend.complete(p, m).text
            row = {"model": card.name, "weights": card.weights,
                   "ref_cost_usd": card.blended_ref_cost(),
                   "backend": backend.name}
            for cls in ("simple", "complex"):
                r = self.score(fn, cls)
                row[cls] = r["hit_rate"]
                row[f"{cls}_n"] = r["total"]
            rows.append(row)
        rows.sort(key=lambda r: (-(r["simple"] + r["complex"]), r["ref_cost_usd"], r["model"]))
        return rows


def scores_table(rows: list[dict]) -> dict[str, dict[str, float]]:
    """{model: {"simple": rate, "complex": rate}} for EvalInformedRouter."""
    return {r["model"]: {"simple": r["simple"], "complex": r["complex"]} for r in rows}
