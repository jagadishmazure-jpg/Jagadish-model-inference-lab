"""Deterministic tests for inferlab. Real backends are never called:
OpenAI/Ollama tests stub the network; resilience tests use ScriptedBackend
and injected clocks, so nothing sleeps.
"""
import io
import urllib.error
import urllib.request

import pytest

from inferlab.clients import (
    BackendUnavailable,
    Completion,
    FakeBackend,
    OllamaBackend,
    OpenAIBackend,
    RateLimited,
    ScriptedBackend,
    TransientBackendError,
)
from inferlab.evals import GOLDEN_SET, GoldenSet, scores_table
from inferlab.fallback import FallbackChain
from inferlab.model_cards import REQUIRED, build_cards, render_markdown, validate
from inferlab.models import PRICES_AS_OF, LocalInfra, ModelRegistry, RoutingCard, shipped_cards
from inferlab.resilience import CircuitBreaker, RetryPolicy, parse_retry_after
from inferlab.router import EvalInformedRouter, PolicyRouter, estimate_complexity
from inferlab.tracker import BudgetExceeded, CallTracker


@pytest.fixture()
def registry():
    return ModelRegistry()


# ---------- routing cards ----------

def test_all_five_cards_shipped(registry):
    assert {c.name for c in registry.list()} == {
        "gpt-4o-mini", "gpt-4o", "llama-3.1-8b", "mistral-7b-instruct", "qwen2.5-7b"}


def test_card_fields(registry):
    card = registry.get("gpt-4o")
    assert isinstance(card, RoutingCard)
    assert card.provider == "openai" and card.weights == "closed"
    assert card.context_window == 128_000 and card.quality_tier == 3
    assert card.cost_usd(1000, 1000) == pytest.approx(0.0125)
    assert card.price_basis == "list"


def test_prices_have_an_as_of_date():
    assert PRICES_AS_OF == "2026-10-17"


def test_local_models_are_not_free(registry):
    for card in registry.list_by(weights="open"):
        assert card.price_basis == "amortised"
        assert card.price_in_per_1k > 0 and card.price_out_per_1k > 0


def test_local_price_is_amortised_infra():
    infra = LocalInfra(usd_per_hour=0.60, prefill_tok_s=4000, decode_tok_s=400, utilisation=0.5)
    p_in, p_out = infra.price_per_1k()
    assert p_out == pytest.approx(0.60 / (400 * 3600 * 0.5) * 1000)
    assert p_in == pytest.approx(0.60 / (4000 * 3600 * 0.5) * 1000)


def test_idle_gpu_makes_local_more_expensive_than_hosted():
    idle = ModelRegistry(shipped_cards(LocalInfra(utilisation=0.05)))
    assert idle.get("llama-3.1-8b").blended_ref_cost() > idle.get("gpt-4o-mini").blended_ref_cost()


@pytest.mark.parametrize("kw", [{"usd_per_hour": -1}, {"decode_tok_s": 0}, {"utilisation": 0}])
def test_bad_infra_assumptions_rejected(kw):
    with pytest.raises(ValueError):
        LocalInfra(**kw)


def test_weights_filter(registry):
    assert {c.name for c in registry.list_by(weights="open")} == {
        "llama-3.1-8b", "mistral-7b-instruct", "qwen2.5-7b"}
    assert len(registry.list_by(provider="openai")) == 2


def test_unknown_model_raises(registry):
    with pytest.raises(KeyError):
        registry.get("gpt-99")


def test_cheapest_uses_reference_cost(registry):
    card = registry.cheapest()
    assert card.blended_ref_cost() == min(c.blended_ref_cost() for c in registry.list())


# ---------- clients ----------

def test_fake_backend_deterministic(registry):
    backend = FakeBackend(registry)
    assert backend.complete("hello", "gpt-4o-mini") == backend.complete("hello", "gpt-4o-mini")


def test_fake_backend_latency_follows_card_tier(registry):
    backend = FakeBackend(registry)
    fast, slow = backend.complete("hi", "gpt-4o-mini"), backend.complete("hi", "gpt-4o")
    assert fast.ttft_ms < slow.ttft_ms and fast.tpot_ms < slow.tpot_ms


def test_openai_backend_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable, match="OPENAI_API_KEY"):
        OpenAIBackend().complete("hi", "gpt-4o-mini")


def _http_error(code, headers=None):
    from email.message import Message
    msg = Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError("https://x", code, "err", msg, io.BytesIO(b""))


def test_openai_429_becomes_rate_limited_with_retry_after(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    def boom(*a, **k):
        raise _http_error(429, {"Retry-After": "2"})
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RateLimited) as exc:
        OpenAIBackend().complete("hi", "gpt-4o-mini")
    assert exc.value.retry_after_s == 2.0


def test_openai_5xx_is_transient(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    def boom(*a, **k):
        raise _http_error(503)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(TransientBackendError):
        OpenAIBackend().complete("hi", "gpt-4o-mini")


def test_openai_network_error_unavailable(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    def boom(*a, **k):
        raise urllib.error.URLError("no route")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(BackendUnavailable, match="failed"):
        OpenAIBackend().complete("hi", "gpt-4o-mini")


def test_ollama_backend_unavailable_when_down(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(BackendUnavailable, match="Ollama not reachable"):
        OllamaBackend().complete("hi", "llama-3.1-8b")


# ---------- resilience ----------

def test_parse_retry_after():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


def test_backoff_is_bounded_and_jittered():
    policy = RetryPolicy(max_attempts=5, base_s=0.5, cap_s=2.0)
    delays = [policy.delay(n, TransientBackendError("x")) for n in range(1, 5)]
    assert all(0 <= d <= min(2.0, 0.5 * 2 ** (n - 1)) for n, d in zip(range(1, 5), delays))
    assert policy.delay(5, TransientBackendError("x")) is None


def test_non_transient_errors_are_not_retried():
    assert RetryPolicy().delay(1, BackendUnavailable("no key")) is None


def test_long_retry_after_gives_up_on_the_hop():
    assert RetryPolicy(max_retry_after_s=5).delay(1, RateLimited("429", 30)) is None
    assert RetryPolicy(max_retry_after_s=5).delay(1, RateLimited("429", 2)) == 2


def test_circuit_breaker_opens_then_half_opens():
    now = [0.0]
    b = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=lambda: now[0])
    b.record_failure()
    assert b.state == "closed"
    b.record_failure()
    assert b.state == "open" and not b.allow()
    now[0] = 10.0
    assert b.state == "half_open" and b.allow()
    b.record_failure()          # trial call failed: open again
    assert b.state == "open"
    now[0] = 20.0
    b.record_success()
    assert b.state == "closed" and b.failures == 0


# ---------- fallback chain ----------

def test_retry_then_success_on_same_hop(registry):
    flaky = ScriptedBackend("flaky", [TransientBackendError("503"), "ok text"])
    waits = []
    chain = FallbackChain([(flaky, "gpt-4o")], retry=RetryPolicy(max_attempts=3), sleep=waits.append)
    assert chain.complete("hi").text == "ok text"
    assert [a["outcome"] for a in chain.last_record.attempts] == ["transient", "ok"]
    assert len(waits) == 1


def test_rate_limit_honours_retry_after():
    rl = ScriptedBackend("rl", [RateLimited("429", retry_after_s=1.5), "fine"])
    waits = []
    chain = FallbackChain([(rl, "gpt-4o")], retry=RetryPolicy(max_attempts=2), sleep=waits.append)
    chain.complete("hi")
    assert waits == [1.5] and chain.last_record.waited_s == 1.5


def test_fallback_to_second_hop_when_first_down(registry):
    down = ScriptedBackend("down", [BackendUnavailable("down")])
    chain = FallbackChain([(down, "gpt-4o"), (FakeBackend(registry), "gpt-4o-mini")])
    chain.complete("hello")
    rec = chain.last_record
    assert (rec.served_by_backend, rec.served_by_model) == ("fake", "gpt-4o-mini")
    assert [a["outcome"] for a in rec.attempts] == ["unavailable", "ok"]


def test_invalid_output_falls_through(registry):
    empty = ScriptedBackend("empty", ["   "])
    chain = FallbackChain([(empty, "gpt-4o"), (FakeBackend(registry), "gpt-4o-mini")])
    chain.complete("hello")
    assert chain.last_record.attempts[0]["outcome"] == "invalid_output"


def test_open_breaker_skips_hop_without_calling(registry):
    down = ScriptedBackend("down", [TransientBackendError("503")])
    breakers = {"down": CircuitBreaker(failure_threshold=2, cooldown_s=60, clock=lambda: 0.0)}
    chain = FallbackChain([(down, "gpt-4o"), (FakeBackend(registry), "gpt-4o-mini")],
                          retry=RetryPolicy(max_attempts=5), breakers=breakers, sleep=lambda s: None)
    chain.complete("a")
    assert down.calls == 2                     # breaker opened after 2 failures, retries stopped
    chain.complete("b")
    assert down.calls == 2                     # skipped while open
    assert chain.last_record.attempts[0]["outcome"] == "circuit_open"


def test_all_hops_failing_raises():
    chain = FallbackChain([(ScriptedBackend("a", [BackendUnavailable("x")]), "gpt-4o"),
                           (ScriptedBackend("b", [TimeoutError("slow")]), "gpt-4o-mini")])
    with pytest.raises(BackendUnavailable, match="all 2 fallback hops failed"):
        chain.complete("hello")


def test_programming_errors_are_not_swallowed():
    chain = FallbackChain([(ScriptedBackend("bug", [KeyError("oops")]), "gpt-4o")])
    with pytest.raises(KeyError):
        chain.complete("hello")


def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        FallbackChain([])


# ---------- routers ----------

def test_cheapest_policy_reason_cites_price_and_date(registry):
    d = PolicyRouter(registry).route("What is 2+2?", "cheapest")
    assert "cheapest" in d.reason and PRICES_AS_OF in d.reason
    assert d.complexity == "simple"


def test_quality_policy_picks_tier3(registry):
    d = PolicyRouter(registry).route("Explain the economic factors.", "quality")
    assert registry.get(d.model_name).quality_tier == 3


def test_fastest_policy_picks_lowest_latency_tier(registry):
    assert registry.get(PolicyRouter(registry).route("hi", "fastest").model_name).latency_tier == 1


def test_balanced_complex_goes_quality(registry):
    d = PolicyRouter(registry).route("Explain consistency versus availability.", "balanced")
    assert registry.get(d.model_name).quality_tier >= 2 and d.complexity == "complex"


def test_unknown_policy_raises(registry):
    with pytest.raises(ValueError, match="unknown policy"):
        PolicyRouter(registry).route("hi", "luxury")


def test_complexity_estimator():
    assert estimate_complexity("What is 2+2?") == "simple"
    assert estimate_complexity("Why is the sky blue? Explain.") == "complex"
    assert estimate_complexity("x" * 400) == "complex"


def test_eval_informed_router_picks_cheapest_passing_model(registry):
    scores = {"gpt-4o-mini": {"simple": 0.95, "complex": 0.3},
              "gpt-4o": {"simple": 1.0, "complex": 0.95},
              "qwen2.5-7b": {"simple": 0.9, "complex": 0.85}}
    r = EvalInformedRouter(registry, scores, quality_floor=0.8)
    assert r.route("What is 2+2?").model_name == "gpt-4o-mini"
    complex_route = r.route("Explain the tradeoff between cost and quality.")
    assert complex_route.model_name == "qwen2.5-7b"
    assert "0.85" in complex_route.reason


def test_eval_informed_router_reports_when_floor_not_met(registry):
    r = EvalInformedRouter(registry, {"gpt-4o-mini": {"simple": 0.5, "complex": 0.1},
                                      "gpt-4o": {"simple": 0.6, "complex": 0.4}}, quality_floor=0.9)
    d = r.route("What is 2+2?")
    assert d.model_name == "gpt-4o" and "no model meets floor" in d.reason


# ---------- evals ----------

def test_golden_set_size_and_classes():
    assert len(GOLDEN_SET) == 16
    classes = [i.task_class for i in GOLDEN_SET]
    assert classes.count("simple") == 9 and classes.count("complex") == 7


def test_score_with_perfect_pipeline():
    result = GoldenSet().score(lambda p: " ".join(
        kw for item in GOLDEN_SET if item.prompt == p for kw in item.expected_keywords))
    assert result["hits"] == 16 and result["hit_rate"] == 1.0


def test_compare_is_deterministic_and_complete(registry):
    rows = GoldenSet().compare(registry)
    assert rows == GoldenSet().compare(registry)
    assert len(rows) == 5 and all(r["backend"] == "fake" for r in rows)


def test_simulated_quality_follows_tier(registry):
    table = scores_table(GoldenSet().compare(registry))
    assert table["gpt-4o"]["complex"] > table["gpt-4o-mini"]["complex"]


# ---------- model cards ----------

def test_model_cards_have_every_required_field(registry):
    cards = build_cards(registry)
    assert len(cards) == 5
    assert all(validate(c) == [] for c in cards)
    assert set(REQUIRED) <= set(vars(cards[0]))


def test_model_cards_mark_simulated_eval_and_cite_price_date(registry):
    text = render_markdown(build_cards(registry))
    assert "simulated by FakeBackend" in text
    assert PRICES_AS_OF in text
    assert "NIST AI RMF, Measure" in text and "Art. 50" in text


def test_validate_flags_empty_fields(registry):
    card = build_cards(registry)[0]
    card.limitations = []
    assert validate(card) == ["limitations"]


# ---------- tracker ----------

def test_cost_math(registry):
    tracker = CallTracker()
    comp = Completion("t", tokens_in=2000, tokens_out=1000, ttft_ms=1.0, tpot_ms=1.0)
    rec = tracker.log(model="gpt-4o-mini", backend="openai", card=registry.get("gpt-4o-mini"),
                      completion=comp)
    assert rec.cost_usd == pytest.approx(2 * 0.00015 + 1 * 0.0006)


def test_local_model_cost_is_non_zero(registry):
    tracker = CallTracker()
    comp = Completion("t", tokens_in=5000, tokens_out=500, ttft_ms=1.0, tpot_ms=1.0)
    rec = tracker.log(model="llama-3.1-8b", backend="ollama", card=registry.get("llama-3.1-8b"),
                      completion=comp)
    assert rec.cost_usd > 0


def test_budget_blocks_before_call(registry):
    tracker = CallTracker()
    comp = Completion("t", tokens_in=1000, tokens_out=1000, ttft_ms=1.0, tpot_ms=1.0)
    tracker.log(model="gpt-4o", backend="openai", card=registry.get("gpt-4o"), completion=comp)
    with pytest.raises(BudgetExceeded, match="breached"):
        tracker.check_budget(limit_usd=tracker.session_cost() + 0.0001, projected_cost_usd=0.01)


def test_summary_and_totals(registry):
    tracker = CallTracker()
    comp = Completion("t", tokens_in=100, tokens_out=50, ttft_ms=120.0, tpot_ms=18.0)
    for _ in range(2):
        tracker.log(model="gpt-4o-mini", backend="openai", card=registry.get("gpt-4o-mini"),
                    completion=comp)
    assert "TOTAL" in tracker.summary()
    assert tracker.totals_by_model()["gpt-4o-mini"]["calls"] == 2
