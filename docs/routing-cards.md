# Routing cards

A **routing card** (`inferlab.models.RoutingCard`) is the operational record the
router, fallback chain and cost tracker read: provider, weights, licence,
context window, price per token and two coarse tiers. It is not a model card;
the governance-facing model cards are in [model-cards.md](model-cards.md).

Earlier versions of this repo called these "model cards". I renamed them
because a model card (Mitchell et al., 2019) documents intended use,
limitations and evaluation, which a price sheet does not.

## Shipped routing cards

<!-- run: python -m inferlab.models -->
```text
prices as of 2026-10-17; reference call = 100 in / 50 out tokens
gpt-4o-mini          closed 128,000 ctx  list      $0.150 in / $0.600 out per 1M  ref call $0.000045  q1 l1
gpt-4o               closed 128,000 ctx  list      $2.500 in / $10.000 out per 1M  ref call $0.000750  q3 l2
llama-3.1-8b         open   128,000 ctx  amortised $0.083 in / $0.833 out per 1M  ref call $0.000050  q1 l2
mistral-7b-instruct  open    32,768 ctx  amortised $0.083 in / $0.833 out per 1M  ref call $0.000050  q1 l2
qwen2.5-7b           open    32,768 ctx  amortised $0.083 in / $0.833 out per 1M  ref call $0.000050  q2 l2
```
<!-- /run -->

- **List prices** for the hosted models were checked on the date above against
  the provider's public pricing page (standard tier, short context). Prices
  change; the date is part of the card so a stale price is visible.
- **Local models are not free.** Their price is an amortised estimate from
  `LocalInfra`: GPU VM $/hour divided by tokens served per hour at a given
  utilisation. The defaults (0.60 USD/h, 4,000 prefill tok/s, 400 decode
  tok/s, 50% utilisation) are assumptions, not measurements. At low
  utilisation a local model costs more per call than a cheap hosted one:

<!-- run: python -m inferlab.models --sweep -->
```text
utilisation   5%: llama-3.1-8b $0.000500 per reference call (gpt-4o-mini $0.000045)
utilisation  25%: llama-3.1-8b $0.000100 per reference call (gpt-4o-mini $0.000045)
utilisation  50%: llama-3.1-8b $0.000050 per reference call (gpt-4o-mini $0.000045)
utilisation 100%: llama-3.1-8b $0.000025 per reference call (gpt-4o-mini $0.000045)
```
<!-- /run -->

## Add a routing card

```python
from inferlab.models import ModelRegistry, RoutingCard, shipped_cards

cards = shipped_cards() + [
    RoutingCard(name="my-model", provider="ollama", weights="open",
                license_name="Apache-2.0", context_window=32_768,
                price_in_per_1k=0.0001, price_out_per_1k=0.0009,
                quality_tier=2, latency_tier=2, price_basis="amortised",
                notes="Why this model earns a place in the fleet."),
]
registry = ModelRegistry(cards)
```

Then add golden-set scores for it (`inferlab.evals`) and a model card entry
(`inferlab.model_cards`); `validate()` fails if a required field is empty.
