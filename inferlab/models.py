"""Routing cards: the facts the router, cost tracker and eval harness need
about each model (price, context window, quality and latency tiers).

These are *routing cards*, not full model cards. The model cards with intended
use, limitations, eval results and NIST AI RMF / EU AI Act notes are built in
``inferlab.model_cards`` from these routing cards plus a real eval run.

Prices
------
* Hosted (closed) models: list prices per 1K tokens, copied from the vendor
  pricing page on ``PRICES_AS_OF`` (source in ``PRICE_SOURCE``). Refresh them
  when the vendor reprices; the as-of date is printed next to every cost.
* Local (open-weights) models are **not free**. Their price per 1K tokens is an
  amortised infrastructure cost: an hourly GPU VM price divided by the tokens
  that VM actually serves in an hour (throughput x utilisation). The defaults
  in ``DEFAULT_LOCAL_INFRA`` are stated assumptions, not measurements; pass
  your own ``LocalInfra`` to see how the routing changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRICES_AS_OF = "2026-10-17"
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing (standard tier, short context)"


@dataclass(frozen=True)
class LocalInfra:
    """Assumptions for amortising a self-hosted GPU into a price per token."""

    usd_per_hour: float = 0.60      # assumed GPU VM price (illustrative)
    prefill_tok_s: float = 4000.0   # input tokens/s the server processes (batched)
    decode_tok_s: float = 400.0     # output tokens/s across all requests (batched)
    utilisation: float = 0.50       # share of paid hours doing useful work

    def __post_init__(self) -> None:
        if self.usd_per_hour < 0:
            raise ValueError("usd_per_hour must be >= 0")
        if self.prefill_tok_s <= 0 or self.decode_tok_s <= 0:
            raise ValueError("throughput must be > 0")
        if not 0 < self.utilisation <= 1:
            raise ValueError("utilisation must be in (0, 1]")

    def price_per_1k(self) -> tuple[float, float]:
        """(input, output) USD per 1K tokens."""
        per_hour_in = self.prefill_tok_s * 3600 * self.utilisation
        per_hour_out = self.decode_tok_s * 3600 * self.utilisation
        return (self.usd_per_hour / per_hour_in * 1000,
                self.usd_per_hour / per_hour_out * 1000)


DEFAULT_LOCAL_INFRA = LocalInfra()


@dataclass(frozen=True)
class RoutingCard:
    name: str
    provider: str            # "openai" | "ollama"
    weights: str             # "open" | "closed"
    license_name: str        # e.g. "Apache-2.0", "proprietary (API terms)"
    context_window: int      # tokens
    price_in_per_1k: float   # USD per 1K input tokens
    price_out_per_1k: float  # USD per 1K output tokens
    quality_tier: int        # 1..3 (3 = frontier-class)
    latency_tier: int        # 1..3 (1 = fastest)
    price_basis: str = "list"  # "list" (vendor price) | "amortised" (local infra)
    notes: str = ""
    token_multipliers: dict[str, float] = field(default_factory=dict)
    # Per-language token multipliers vs English (1.0): some languages need
    # more tokens for the same content (tokenizer fertility). Defaults are
    # illustrative, not measurements -- measure per tokenizer. Documented
    # example: Hindi needs roughly 2x the tokens of English ("hi": 2.0).

    def multiplier_for(self, language: str) -> float:
        """Token multiplier for a language code; English and unknown -> 1.0."""
        if language == "en":
            return 1.0
        return self.token_multipliers.get(language, 1.0)

    def cost_usd(self, tokens_in: int, tokens_out: int,
                 language: str = "en") -> float:
        """Price a call. Pass ``language`` only for English-equivalent token
        estimates; real tokenizer counts must use the default "en"."""
        m = self.multiplier_for(language)
        return (tokens_in * m / 1000.0) * self.price_in_per_1k + (
            tokens_out * m / 1000.0) * self.price_out_per_1k

    def blended_ref_cost(self, language: str = "en") -> float:
        """Cost of a reference call (100 input, 50 output tokens)."""
        return self.cost_usd(100, 50, language=language)


def _local(name: str, license_name: str, context: int, quality: int,
           infra: LocalInfra, notes: str) -> RoutingCard:
    p_in, p_out = infra.price_per_1k()
    return RoutingCard(name=name, provider="ollama", weights="open",
                       license_name=license_name, context_window=context,
                       price_in_per_1k=p_in, price_out_per_1k=p_out,
                       quality_tier=quality, latency_tier=2,
                       price_basis="amortised", notes=notes,
                       token_multipliers={"hi": 2.0})


def shipped_cards(infra: LocalInfra = DEFAULT_LOCAL_INFRA) -> list[RoutingCard]:
    return [
        RoutingCard(name="gpt-4o-mini", provider="openai", weights="closed",
                    license_name="proprietary (API terms)", context_window=128_000,
                    price_in_per_1k=0.00015, price_out_per_1k=0.0006,
                    quality_tier=1, latency_tier=1,
                    notes="Small hosted model; default for simple tasks.",
                    token_multipliers={"hi": 2.0}),
        RoutingCard(name="gpt-4o", provider="openai", weights="closed",
                    license_name="proprietary (API terms)", context_window=128_000,
                    price_in_per_1k=0.0025, price_out_per_1k=0.010,
                    quality_tier=3, latency_tier=2,
                    notes="Larger hosted model; complex reasoning.",
                    token_multipliers={"hi": 2.0}),
        _local("llama-3.1-8b", "Llama 3.1 Community License", 128_000, 1, infra,
               "Open weights served locally by Ollama."),
        _local("mistral-7b-instruct", "Apache-2.0", 32_768, 1, infra,
               "Open weights served locally by Ollama."),
        _local("qwen2.5-7b", "Apache-2.0", 32_768, 2, infra,
               "Open weights served locally by Ollama."),
    ]


class ModelRegistry:
    """Lookup of routing cards."""

    def __init__(self, cards: list[RoutingCard] | None = None) -> None:
        cards = list(cards) if cards is not None else shipped_cards()
        self._cards: dict[str, RoutingCard] = {c.name: c for c in cards}

    def get(self, name: str) -> RoutingCard:
        try:
            return self._cards[name]
        except KeyError:
            raise KeyError(f"unknown model: {name!r}") from None

    def list(self) -> list[RoutingCard]:
        return list(self._cards.values())

    def list_by(self, *, provider: str | None = None, weights: str | None = None,
                quality_tier: int | None = None) -> list[RoutingCard]:
        out = self.list()
        if provider is not None:
            out = [c for c in out if c.provider == provider]
        if weights is not None:
            out = [c for c in out if c.weights == weights]
        if quality_tier is not None:
            out = [c for c in out if c.quality_tier == quality_tier]
        return out

    def cheapest(self, *, weights: str | None = None) -> RoutingCard:
        """Card with the lowest reference-call cost (ties: faster, then name)."""
        cards = self.list_by(weights=weights) if weights else self.list()
        return min(cards, key=lambda c: (c.blended_ref_cost(), c.latency_tier, c.name))


def routing_table(infra: LocalInfra = DEFAULT_LOCAL_INFRA) -> str:
    """Plain-text table of the shipped routing cards (used in docs)."""
    lines = [f"prices as of {PRICES_AS_OF}; reference call = 100 in / 50 out tokens"]
    for c in shipped_cards(infra):
        lines.append(
            f"{c.name:20} {c.weights:6} {c.context_window:>7,} ctx  {c.price_basis:9} "
            f"${c.price_in_per_1k * 1000:.3f} in / ${c.price_out_per_1k * 1000:.3f} out per 1M  "
            f"ref call ${c.blended_ref_cost():.6f}  q{c.quality_tier} l{c.latency_tier}")
    return "\n".join(lines)


def utilisation_sweep(model: str = "llama-3.1-8b", against: str = "gpt-4o-mini",
                      levels: tuple[float, ...] = (0.05, 0.25, 0.5, 1.0)) -> str:
    """How the amortised local price moves with GPU utilisation."""
    ref = ModelRegistry().get(against).blended_ref_cost()
    out = []
    for u in levels:
        cost = ModelRegistry(shipped_cards(LocalInfra(utilisation=u))).get(model).blended_ref_cost()
        out.append(f"utilisation {u:>4.0%}: {model} ${cost:.6f} per reference call "
                   f"({against} ${ref:.6f})")
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    import sys
    print(utilisation_sweep() if "--sweep" in sys.argv else routing_table())
