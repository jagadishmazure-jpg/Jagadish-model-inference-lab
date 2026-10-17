# Jagadish-model-inference-lab

One client, many models. Every agent pays per token with no visibility — this lab gives you a unified inference interface over **open and closed models**, with policy routing, fallback chains, and a per-call cost/latency ledger.

## The problem

An agentic system makes hundreds of model calls per task. Without plumbing:

- every call goes to the same expensive model, even trivial ones,
- a provider outage kills the whole pipeline,
- nobody knows what a session cost until the invoice arrives.

## What this gives you

| Module | What it does |
|---|---|
| `models.py` | `ModelRegistry` with model cards — price, context window, quality/latency tiers, open vs closed |
| `clients.py` | One `Backend` interface: `OpenAIBackend` (closed), `OllamaBackend` (open weights, local), `FakeBackend` (deterministic, for tests) |
| `router.py` | `PolicyRouter` — `cheapest` / `balanced` / `quality` / `fastest` policies with a cited reason per decision |
| `fallback.py` | `FallbackChain` — ordered hops; a down provider degrades to the next, and the record shows which hop served |
| `tracker.py` | `CallTracker` — per-call tokens, TTFT/TPOT, cost; `check_budget()` blocks *before* a call that would breach the limit |
| `evals.py` | `GoldenSet` — 10 synthetic prompt/keyword pairs; `compare()` ranks models by score, then cost |

## Quickstart

```python
from inferlab.models import ModelRegistry
from inferlab.clients import FakeBackend
from inferlab.router import PolicyRouter
from inferlab.fallback import FallbackChain
from inferlab.tracker import CallTracker

registry = ModelRegistry()
router = PolicyRouter(registry)
tracker = CallTracker()

decision = router.route("What is 2+2?", "cheapest")
print(decision)  # model_name, backend_name, reason_cited

chain = FallbackChain([
    (FakeBackend(registry), decision.model_name),
])
completion = chain.complete("What is 2+2?")
tracker.log(model=decision.model_name, backend="fake",
            card=registry.get(decision.model_name), completion=completion)
print(tracker.summary())
```

## Live runs

Backends read credentials **from the environment at call time** and degrade gracefully when absent. No keys are stored anywhere in this repo.

**Closed models (OpenAI):**
```bash
export OPENAI_API_KEY="sk-..."
python -m inferlab.live_demo --model gpt-4o-mini --policy balanced
```

**Open models (Ollama, local, free):**
```bash
# install from https://ollama.com, then:
ollama pull llama3.1
python -m inferlab.live_demo --model llama-3.1-8b --policy cheapest
```

Tests never touch the network: they use only `FakeBackend`.

## Production path

- Swap `FakeBackend` for `OpenAIBackend`/`OllamaBackend` behind the same interface.
- Put `CallTracker` behind your agent loop; enforce `check_budget()` per session/user.
- Feed `compare()` results back into router policy weights.
- Secrets via a vault (Azure Key Vault / AWS Secrets Manager), never env files in CI.

## Synthetic data note

All eval prompts and model prices are synthetic/illustrative. Prices are list-price approximations for routing math, not live quotes.

## Docs

- [docs/model-cards.md](docs/model-cards.md) — the shipped cards and how to add one.
