"""Model registry with model cards: name, provider, open/closed licensing,
context window, price per 1K tokens (in/out), quality tier and latency tier.

Quality tier: 1 = small/fast/cheap, 3 = frontier-class.
Latency tier: 1 = fastest, 3 = slowest.
Prices are illustrative list prices for routing math, not live quotes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelCard:
    name: str
    provider: str            # e.g. "openai", "ollama"
    license: str             # "open" | "closed"
    context_window: int      # tokens
    price_in_per_1k: float   # USD per 1K input tokens
    price_out_per_1k: float  # USD per 1K output tokens
    quality_tier: int        # 1..3
    latency_tier: int        # 1..3 (1 = fastest)
    notes: str = ""

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1000.0) * self.price_in_per_1k + (
            tokens_out / 1000.0
        ) * self.price_out_per_1k


_SHIPPED_CARDS = [
    ModelCard(
        name="gpt-4o-mini", provider="openai", license="closed",
        context_window=128_000, price_in_per_1k=0.00015,
        price_out_per_1k=0.0006, quality_tier=1, latency_tier=1,
        notes="Cheap fast OpenAI model; default for simple tasks.",
    ),
    ModelCard(
        name="gpt-4o", provider="openai", license="closed",
        context_window=128_000, price_in_per_1k=0.0025,
        price_out_per_1k=0.010, quality_tier=3, latency_tier=2,
        notes="Frontier-class OpenAI model; complex reasoning.",
    ),
    ModelCard(
        name="llama-3.1-8b", provider="ollama", license="open",
        context_window=128_000, price_in_per_1k=0.0,
        price_out_per_1k=0.0, quality_tier=1, latency_tier=2,
        notes="Meta open weights via Ollama; local, free at inference.",
    ),
    ModelCard(
        name="mistral-7b-instruct", provider="ollama", license="open",
        context_window=32_768, price_in_per_1k=0.0,
        price_out_per_1k=0.0, quality_tier=1, latency_tier=2,
        notes="Mistral open weights via Ollama; strong instruction following.",
    ),
    ModelCard(
        name="qwen2.5-7b", provider="ollama", license="open",
        context_window=32_768, price_in_per_1k=0.0,
        price_out_per_1k=0.0, quality_tier=2, latency_tier=2,
        notes="Alibaba open weights via Ollama; good multilingual/code.",
    ),
]


class ModelRegistry:
    """Lookup of shipped model cards."""

    def __init__(self, cards: list[ModelCard] | None = None) -> None:
        cards = list(cards) if cards is not None else list(_SHIPPED_CARDS)
        self._cards: dict[str, ModelCard] = {c.name: c for c in cards}

    def get(self, name: str) -> ModelCard:
        try:
            return self._cards[name]
        except KeyError:
            raise KeyError(f"unknown model: {name!r}") from None

    def list(self) -> list[ModelCard]:
        return list(self._cards.values())

    def list_by(self, *, provider: str | None = None,
                license: str | None = None,
                quality_tier: int | None = None) -> list[ModelCard]:
        out = self.list()
        if provider is not None:
            out = [c for c in out if c.provider == provider]
        if license is not None:
            out = [c for c in out if c.license == license]
        if quality_tier is not None:
            out = [c for c in out if c.quality_tier == quality_tier]
        return out

    def cheapest(self, *, license: str | None = None) -> ModelCard:
        """Card with the lowest blended price (open models = $0)."""
        cards = self.list_by(license=license) if license else self.list()
        return min(cards, key=lambda c: (c.price_in_per_1k + c.price_out_per_1k,
                                        c.latency_tier))
