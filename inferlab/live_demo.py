"""Live demo CLI: route -> fallback -> track against REAL backends.

Credentials come from the environment at call time; missing backends
degrade gracefully with a clear message. Never used by tests.

Usage:
    export OPENAI_API_KEY="sk-..."            # for closed models
    # ollama pull llama3.1                       # for open models (local)
    python -m inferlab.live_demo --model gpt-4o-mini --policy balanced \\
        --prompt "Summarize RAG in one sentence." --budget 0.50
"""
from __future__ import annotations

import argparse

from .clients import BackendUnavailable, FakeBackend, OllamaBackend, OpenAIBackend
from .fallback import FallbackChain
from .models import ModelRegistry
from .router import POLICIES, PolicyRouter
from .tracker import BudgetExceeded, CallTracker


def main() -> None:
    args = parse_args()
    registry = ModelRegistry()
    card = registry.get(args.model)
    router = PolicyRouter(registry)
    decision = router.route(args.prompt, args.policy)
    print(f"route: {decision.model_name} via {decision.backend_name}")
    print(f"reason: {decision.reason}")

    backends = {"openai": OpenAIBackend(), "ollama": OllamaBackend(),
                "fake": FakeBackend(registry)}
    primary = backends.get(decision.backend_name, backends["fake"])
    # Fallback: requested model first, then cheapest open model locally.
    chain = FallbackChain([
        (primary, args.model),
        (backends["ollama"], "llama-3.1-8b"),
        (backends["fake"], "gpt-4o-mini"),
    ])

    tracker = CallTracker()
    tracker.check_budget(args.budget, projected_cost_usd=0.05)
    try:
        completion = chain.complete(args.prompt, max_tokens=args.max_tokens)
    except BackendUnavailable as exc:
        print(f"unavailable: {exc}")
        return
    served = chain.last_record
    tracker.log(model=served.served_by_model,
                backend=served.served_by_backend,
                card=registry.get(served.served_by_model),
                completion=completion)
    print(f"served by: {served.served_by_backend}/{served.served_by_model}")
    print(f"answer: {completion.text}")
    print(tracker.summary())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="inferlab live demo")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--policy", default="balanced", choices=POLICIES)
    p.add_argument("--prompt", default="What is 2+2?")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--budget", type=float, default=1.0,
                   help="session budget in USD")
    return p.parse_args()


if __name__ == "__main__":
    main()
