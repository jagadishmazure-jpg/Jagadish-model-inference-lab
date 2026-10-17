"""Policy routing: pick the right model for each prompt.

Policies: cheapest | balanced | quality | fastest.
A tiny complexity estimator (length + question-word heuristics) decides
whether a prompt is simple or complex; the policy then maps to a tier.
Every decision cites its reason so routing is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import ModelCard, ModelRegistry

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
            reason = (f"policy=cheapest -> lowest blended price "
                      f"(${card.price_in_per_1k}/$"
                      f"{card.price_out_per_1k} per 1K in/out)")
        elif policy == "fastest":
            card = min(self.registry.list(),
                       key=lambda c: (c.latency_tier, c.price_in_per_1k))
            reason = (f"policy=fastest -> lowest latency tier "
                      f"(tier {card.latency_tier})")
        elif policy == "quality":
            tier3 = self.registry.list_by(quality_tier=3)
            pool = tier3 or self.registry.list()
            card = min(pool, key=lambda c: (c.price_in_per_1k,
                                            c.latency_tier))
            reason = (f"policy=quality -> highest quality tier "
                      f"(tier {card.quality_tier})")
        else:  # balanced
            if complexity == "complex":
                pool = [c for c in self.registry.list()
                        if c.quality_tier >= 2] or self.registry.list()
                card = min(pool, key=lambda c: (c.price_in_per_1k,
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
