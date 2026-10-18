"""Routing cards, eval-informed routing and a resilient fallback chain, offline.

Quality numbers come from FakeBackend and are SIMULATED (they test the routing
logic, not the models). Prices: list prices as of the date shown; local models
use an amortised GPU cost from stated assumptions.
"""
from inferlab.clients import FakeBackend, RateLimited, ScriptedBackend, TransientBackendError
from inferlab.evals import GoldenSet, scores_table
from inferlab.fallback import FallbackChain
from inferlab.generation import BALANCED, GREEDY, best_of_n, generate, self_consistency
from inferlab.models import PRICES_AS_OF, LocalInfra, ModelRegistry, shipped_cards
from inferlab.resilience import CircuitBreaker, RetryPolicy
from inferlab.router import EvalInformedRouter, PolicyRouter
from inferlab.tracker import CallTracker

reg = ModelRegistry()
print(f"Routing cards (prices as of {PRICES_AS_OF}; local = amortised infra):")
for c in reg.list():
    print(f"  {c.name:20s} {c.price_basis:9s} ${c.price_in_per_1k * 1000:.3f} in / "
          f"${c.price_out_per_1k * 1000:.3f} out per 1M tokens")

idle = ModelRegistry(shipped_cards(LocalInfra(utilisation=0.05)))
print(f"\nSame GPU at 5% utilisation: llama-3.1-8b ${idle.get('llama-3.1-8b').blended_ref_cost():.6f} "
      f"vs gpt-4o-mini ${idle.get('gpt-4o-mini').blended_ref_cost():.6f} per reference call")

rows = GoldenSet().compare(reg)
print("\nGolden-set hit rate per task class (SIMULATED by FakeBackend):")
for r in rows:
    print(f"  {r['model']:20s} simple {r['simple']:.2f} (n={r['simple_n']})  "
          f"complex {r['complex']:.2f} (n={r['complex_n']})  ref cost ${r['ref_cost_usd']:.6f}")

router = EvalInformedRouter(reg, scores_table(rows), quality_floor=0.8)
for prompt in ["What is 2+2?", "Explain the tradeoff between cost and quality in routing."]:
    d = router.route(prompt)
    print(f"\nroute {prompt!r}\n  -> {d.model_name} ({d.reason})")
print("\nstatic policy=cheapest ->", PolicyRouter(reg).route("What is 2+2?", "cheapest").reason)

print("\nResilience: hop 1 rate-limited then 503s, hop 2 serves")
waits = []
hop1 = ScriptedBackend("provider-a", [RateLimited("429", retry_after_s=1.0), TransientBackendError("503")])
chain = FallbackChain([(hop1, "gpt-4o"), (FakeBackend(reg), "gpt-4o-mini")],
                      retry=RetryPolicy(max_attempts=3),
                      breakers={"provider-a": CircuitBreaker(failure_threshold=3, cooldown_s=30,
                                                             clock=lambda: 0.0)},
                      sleep=waits.append)
completion = chain.complete("What is 2+2?")
for a in chain.last_record.attempts:
    print(f"  {a['backend']}/{a['model']} attempt {a.get('attempt', '-')}: {a['outcome']}")
print(f"  waited {[round(w, 3) for w in waits]} s (Retry-After honoured, then jittered backoff)")
print("  breaker for provider-a:", chain.breakers["provider-a"].state)
chain.complete("Name a planet.")
print("  next call:", [a["outcome"] for a in chain.last_record.attempts])

t = CallTracker()
rec = chain.last_record
t.log(model=rec.served_by_model, backend=rec.served_by_backend, card=reg.get(rec.served_by_model),
      completion=completion)
print("\n" + t.summary())

# Generation controls: greedy is repeatable, sampling is seeded; best-of-N and self-consistency spend N calls.
gen_backend = ScriptedBackend("vote", ["Paris", "paris ", "Lyon", "Paris", "Marseille"])
answer, votes = self_consistency(gen_backend, "capital of France?", "gpt-4o-mini", n=5, config=BALANCED)
print(f"Self-consistency over 5 samples: {answer!r}, votes {votes}")
fake = FakeBackend()
same = generate(fake, "summarise the incident", "gpt-4o-mini", GREEDY).text == generate(
    fake, "summarise the incident", "gpt-4o-mini", GREEDY).text
winner, samples = best_of_n(fake, "summarise the incident", "gpt-4o-mini", n=4, config=BALANCED,
                            score_fn=lambda c: -c.tokens_out)
print(f"Greedy repeatable: {same}; best-of-4 picked the shortest of {[c.tokens_out for c in samples]} output tokens "
      f"({winner.tokens_out}), paying for {sum(c.tokens_in + c.tokens_out for c in samples)} tokens in total")
