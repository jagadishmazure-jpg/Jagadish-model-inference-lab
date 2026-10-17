# Model cards

A model card is the unit of routing truth: everything the router, cost
tracker, and eval harness need to know about a model in one place.

## Shipped cards

| name | provider | license | context | $/1K in | $/1K out | quality | latency |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini | openai | closed | 128K | 0.00015 | 0.0006 | 1 | 1 |
| gpt-4o | openai | closed | 128K | 0.0025 | 0.010 | 3 | 2 |
| llama-3.1-8b | ollama | open | 128K | 0 | 0 | 1 | 2 |
| mistral-7b-instruct | ollama | open | 32K | 0 | 0 | 1 | 2 |
| qwen2.5-7b | ollama | open | 32K | 0 | 0 | 2 | 2 |

- **quality tier** 1 = small/fast/cheap, 3 = frontier-class.
- **latency tier** 1 = fastest, 3 = slowest.
- **open models price at $0** for routing math — the real cost is your own
  compute (local GPU or a cloud VM), which this lab does not meter.
- Prices are illustrative list prices for routing decisions, not live quotes.
  Refresh them when providers reprice.

## How to add a card

```python
from inferlab.models import ModelCard, ModelRegistry

registry = ModelRegistry()
# registries are immutable snapshots; build a new one with your card added:
cards = registry.list() + [
    ModelCard(
        name="my-model", provider="ollama", license="open",
        context_window=32768,
        price_in_per_1k=0.0, price_out_per_1k=0.0,
        quality_tier=2, latency_tier=2,
        notes="Why this model earns a place in the fleet.",
    )
]
mine = ModelRegistry(cards)
```

Then map the provider to a backend in `router._PROVIDER_BACKEND`
(or pass an explicit backend when building your `FallbackChain`).
