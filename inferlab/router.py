"""Policy routing: pick the right model for each prompt.

Two routers:

* ``PolicyRouter``: static policies (cheapest | balanced | quality | fastest)
  over the routing cards' tiers and prices.
* ``EvalInformedRouter``: picks the cheapest model whose measured eval score
  for this kind of task (simple or complex) meets a quality floor. The scores
  come from ``evals.GoldenSet.compare()``, so routing follows evidence, not
  tier labels.

A small complexity estimator (length + question-word heuristics) decides
whether a prompt is simple or complex. Every decision cites its reason
(``RouteDecision.reason``) so routing is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import PRICES_AS_OF, ModelRegistry

POLICIES = ("cheapest", "balanced", "quality", "fastest")

_QUESTION_WORDS = ("why", "how", "explain", "compare", "analyze",
                   "design", "prove", "evaluate", "tradeoff", "trade-off")


@dataclass(frozen=True)
class RouteDecision:
    model_name: str
    backend_name: str
    reason: str
    complexity: str  # "simple" | "complex"


def estimate_complexity(prompt: str) -> str:
    """Heuristic: long prompts or analytical question-words -> complex."""
    lowered = prompt.lower()
    if len(prompt) > 300:
        return "complex"
    if any(w in lowered for w in _QUESTION_WORDS):
        return "complex"
    if prompt.count("?") >= 2:
        return "complex"
    return "simple"


# Map backend name per provider for the shipped cards.
_PROVIDER_BACKEND = {"openai": "openai", "ollama": "ollama"}


class PolicyRouter:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def route(self, prompt: str, policy: str) -> RouteDecision:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        complexity = estimate_complexity(prompt)

        if policy == "cheapest":
            card = self.registry.cheapest()
            reason = (f"policy=cheapest -> lowest reference-call cost "
                      f"(${card.blended_ref_cost():.6f} per 100 in/50 out tokens, "
                      f"{card.price_basis} price, as of {PRICES_AS_OF})")
        elif policy == "fastest":
            card = min(self.registry.list(),
                       key=lambda c: (c.latency_tier, c.blended_ref_cost()))
            reason = (f"policy=fastest -> lowest latency tier "
                      f"(tier {card.latency_tier})")
        elif policy == "quality":
            tier3 = self.registry.list_by(quality_tier=3)
            pool = tier3 or self.registry.list()
            card = min(pool, key=lambda c: (c.blended_ref_cost(),
                                            c.latency_tier))
            reason = (f"policy=quality -> highest quality tier "
                      f"(tier {card.quality_tier})")
        else:  # balanced
            if complexity == "complex":
                pool = [c for c in self.registry.list()
                        if c.quality_tier >= 2] or self.registry.list()
                card = min(pool, key=lambda c: (c.blended_ref_cost(),
                                               c.latency_tier))
                reason = ("policy=balanced + complex prompt -> "
                          "quality tier >= 2 at lowest price")
            else:
                card = self.registry.cheapest()
                reason = ("policy=balanced + simple prompt -> "
                          "cheapest model suffices")

        backend = _PROVIDER_BACKEND.get(card.provider, card.provider)
        return RouteDecision(model_name=card.name, backend_name=backend,
                             reason=reason, complexity=complexity)


class EvalInformedRouter:
    """Cheapest model whose eval score for this task class meets the floor.

    ``scores`` is ``{model: {"simple": rate, "complex": rate}}`` (see
    ``evals.scores_table``). If no model meets the floor, the best-scoring
    model is used and the reason says the floor was not met.
    """

    def __init__(self, registry: ModelRegistry, scores: dict[str, dict[str, float]],
                 quality_floor: float = 0.8) -> None:
        self.registry = registry
        self.scores = scores
        self.quality_floor = quality_floor

    def route(self, prompt: str) -> RouteDecision:
        cls = estimate_complexity(prompt)
        cards = [c for c in self.registry.list() if c.name in self.scores]
        if not cards:
            raise ValueError("no eval scores for any registered model")
        passing = [c for c in cards if self.scores[c.name][cls] >= self.quality_floor]
        if passing:
            card = min(passing, key=lambda c: (c.blended_ref_cost(), c.latency_tier, c.name))
            reason = (f"eval-informed: {cls} task, floor {self.quality_floor:.2f} -> "
                      f"cheapest passing model scored {self.scores[card.name][cls]:.2f} "
                      f"at ${card.blended_ref_cost():.6f}/ref call")
        else:
            card = max(cards, key=lambda c: (self.scores[c.name][cls], -c.blended_ref_cost()))
            reason = (f"eval-informed: {cls} task, no model meets floor "
                      f"{self.quality_floor:.2f}; best available scored "
                      f"{self.scores[card.name][cls]:.2f}")
        backend = _PROVIDER_BACKEND.get(card.provider, card.provider)
        return RouteDecision(model_name=card.name, backend_name=backend,
                             reason=reason, complexity=cls)
