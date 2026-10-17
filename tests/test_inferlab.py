"""Deterministic tests for inferlab. Real backends are never touched:
OpenAI/Ollama tests assert graceful degradation with the network stubbed.
"""
import urllib.error
import urllib.request

import pytest

from inferlab.clients import (BackendUnavailable, Completion, FakeBackend,
                              OllamaBackend, OpenAIBackend)
from inferlab.evals import GOLDEN_SET, GoldenSet
from inferlab.fallback import FallbackChain
from inferlab.models import ModelCard, ModelRegistry
from inferlab.router import PolicyRouter, estimate_complexity
from inferlab.tracker import BudgetExceeded, CallTracker


@pytest.fixture()
def registry():
    return ModelRegistry()


# ---------- registry ----------

def test_all_five_cards_shipped(registry):
    names = {c.name for c in registry.list()}
    assert names == {"gpt-4o-mini", "gpt-4o", "llama-3.1-8b",
                     "mistral-7b-instruct", "qwen2.5-7b"}


def test_card_fields(registry):
    card = registry.get("gpt-4o")
    assert card.provider == "openai"
    assert card.license == "closed"
    assert card.context_window == 128_000
    assert card.quality_tier == 3
    assert card.cost_usd(1000, 1000) == pytest.approx(0.0125)


def test_open_filter(registry):
    cards = registry.list_by(license="open")
    assert {c.name for c in cards} == {"llama-3.1-8b", "mistral-7b-instruct",
                                      "qwen2.5-7b"}
    assert all(c.price_in_per_1k == 0.0 for c in cards)


def test_provider_filter(registry):
    assert len(registry.list_by(provider="openai")) == 2
    assert len(registry.list_by(provider="ollama")) == 3


def test_unknown_model_raises(registry):
    with pytest.raises(KeyError):
        registry.get("gpt-99")


def test_cheapest_is_free_open_model(registry):
    card = registry.cheapest()
    assert card.price_in_per_1k == 0.0 and card.price_out_per_1k == 0.0


# ---------- clients ----------

def test_fake_backend_deterministic(registry):
    backend = FakeBackend(registry)
    a = backend.complete("hello", "gpt-4o-mini")
    b = backend.complete("hello", "gpt-4o-mini")
    assert a == b


def test_fake_backend_completion_shape(registry):
    c = FakeBackend(registry).complete("hello world", "llama-3.1-8b")
    assert isinstance(c, Completion)
    assert c.tokens_in > 0 and c.tokens_out > 0
    assert c.ttft_ms > 0 and c.tpot_ms > 0
    assert c.text


def test_fake_backend_latency_follows_card_tier(registry):
    backend = FakeBackend(registry)
    fast = backend.complete("hi", "gpt-4o-mini")   # latency tier 1
    slow = backend.complete("hi", "gpt-4o")       # latency tier 2
    assert fast.ttft_ms < slow.ttft_ms
    assert fast.tpot_ms < slow.tpot_ms


def test_openai_backend_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable, match="OPENAI_API_KEY"):
        OpenAIBackend().complete("hi", "gpt-4o-mini")


def test_openai_backend_unavailable_on_network_error(monkeypatch):
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


# ---------- router ----------

def test_cheapest_policy_picks_free_tier1(registry):
    d = PolicyRouter(registry).route("What is 2+2?", "cheapest")
    card = registry.get(d.model_name)
    assert card.price_in_per_1k == 0.0
    assert "cheapest" in d.reason
    assert d.complexity == "simple"


def test_quality_policy_picks_tier3(registry):
    d = PolicyRouter(registry).route(
        "Why did the Roman empire fall? Explain the economic factors.",
        "quality")
    assert registry.get(d.model_name).quality_tier == 3
    assert "quality" in d.reason


def test_fastest_policy_picks_lowest_latency_tier(registry):
    d = PolicyRouter(registry).route("hi", "fastest")
    assert registry.get(d.model_name).latency_tier == 1


def test_balanced_simple_goes_cheap(registry):
    d = PolicyRouter(registry).route("What is 2+2?", "balanced")
    assert registry.get(d.model_name).price_in_per_1k == 0.0


def test_balanced_complex_goes_quality(registry):
    d = PolicyRouter(registry).route(
        "Explain the tradeoff between consistency and availability in "
        "distributed systems, with examples.", "balanced")
    assert registry.get(d.model_name).quality_tier >= 2
    assert d.complexity == "complex"


def test_unknown_policy_raises(registry):
    with pytest.raises(ValueError, match="unknown policy"):
        PolicyRouter(registry).route("hi", "luxury")


def test_complexity_estimator():
    assert estimate_complexity("What is 2+2?") == "simple"
    assert estimate_complexity("Why is the sky blue? Explain.") == "complex"
    assert estimate_complexity("x" * 400) == "complex"


# ---------- fallback ----------

class _Down:
    name = "down"
    def complete(self, prompt, model, max_tokens=256):
        raise BackendUnavailable("down")


class _Flaky:
    name = "flaky"
    def complete(self, prompt, model, max_tokens=256):
        raise TimeoutError("slow")


def test_fallback_uses_second_hop(registry):
    chain = FallbackChain([(_Down(), "gpt-4o"),
                           (FakeBackend(registry), "gpt-4o-mini")])
    c = chain.complete("hello")
    assert isinstance(c, Completion)
    rec = chain.last_record
    assert rec.served_by_backend == "fake"
    assert rec.served_by_model == "gpt-4o-mini"
    assert [a["outcome"] for a in rec.attempts] == ["unavailable", "ok"]


def test_fallback_all_down_raises(registry):
    chain = FallbackChain([(_Down(), "gpt-4o"), (_Flaky(), "gpt-4o-mini")])
    with pytest.raises(BackendUnavailable, match="all 2 fallback hops"):
        chain.complete("hello")
    assert [a["outcome"] for a in chain.last_record.attempts] == [
        "unavailable", "timeout"]


def test_fallback_first_hop_serves(registry):
    chain = FallbackChain([(FakeBackend(registry), "llama-3.1-8b"),
                           (_Down(), "gpt-4o")])
    chain.complete("hello")
    assert chain.last_record.served_by_model == "llama-3.1-8b"
    assert len(chain.last_record.attempts) == 1


def test_empty_chain_rejected():
    with pytest.raises(ValueError):
        FallbackChain([])


# ---------- tracker ----------

def test_cost_math(registry):
    tracker = CallTracker()
    card = registry.get("gpt-4o-mini")  # 0.00015 / 0.0006 per 1K
    comp = Completion("t", tokens_in=2000, tokens_out=1000,
                      ttft_ms=1.0, tpot_ms=1.0)
    rec = tracker.log(model="gpt-4o-mini", backend="openai", card=card,
                      completion=comp)
    assert rec.cost_usd == pytest.approx(2 * 0.00015 + 1 * 0.0006)
    assert tracker.session_cost() == pytest.approx(rec.cost_usd)


def test_open_model_costs_zero(registry):
    tracker = CallTracker()
    card = registry.get("llama-3.1-8b")
    comp = Completion("t", tokens_in=5000, tokens_out=500,
                      ttft_ms=1.0, tpot_ms=1.0)
    rec = tracker.log(model="llama-3.1-8b", backend="ollama", card=card,
                      completion=comp)
    assert rec.cost_usd == 0.0


def test_budget_blocks_before_call(registry):
    tracker = CallTracker()
    card = registry.get("gpt-4o")
    comp = Completion("t", tokens_in=1000, tokens_out=1000,
                      ttft_ms=1.0, tpot_ms=1.0)
    tracker.log(model="gpt-4o", backend="openai", card=card, completion=comp)
    spent = tracker.session_cost()
    with pytest.raises(BudgetExceeded, match="breached"):
        tracker.check_budget(limit_usd=spent + 0.0001,
                             projected_cost_usd=0.01)


def test_budget_allows_under_limit(registry):
    tracker = CallTracker()
    tracker.check_budget(limit_usd=1.0, projected_cost_usd=0.01)  # no raise


def test_summary_table(registry):
    tracker = CallTracker()
    card = registry.get("gpt-4o-mini")
    comp = Completion("t", tokens_in=100, tokens_out=50,
                      ttft_ms=120.0, tpot_ms=18.0)
    tracker.log(model="gpt-4o-mini", backend="openai", card=card,
                completion=comp)
    table = tracker.summary()
    assert "gpt-4o-mini" in table and "TOTAL" in table


def test_totals_by_model(registry):
    tracker = CallTracker()
    card = registry.get("gpt-4o-mini")
    comp = Completion("t", tokens_in=100, tokens_out=50,
                      ttft_ms=1.0, tpot_ms=1.0)
    tracker.log(model="gpt-4o-mini", backend="openai", card=card,
                completion=comp)
    tracker.log(model="gpt-4o-mini", backend="openai", card=card,
                completion=comp)
    totals = tracker.totals_by_model()
    assert totals["gpt-4o-mini"]["calls"] == 2
    assert totals["gpt-4o-mini"]["tokens_in"] == 200


# ---------- evals ----------

def test_golden_set_has_ten_items():
    assert len(GOLDEN_SET) == 10


def test_score_with_perfect_pipeline():
    gs = GoldenSet()
    result = gs.score(lambda p: " ".join(
        kw for item in GOLDEN_SET if item.prompt == p
        for kw in item.expected_keywords))
    assert result["hits"] == 10
    assert result["hit_rate"] == 1.0


def test_score_with_fake_backend_is_deterministic(registry):
    gs = GoldenSet()
    backend = FakeBackend(registry)
    r1 = gs.score(lambda p: backend.complete(p, "gpt-4o-mini").text)
    r2 = gs.score(lambda p: backend.complete(p, "gpt-4o-mini").text)
    assert r1["hit_rate"] == r2["hit_rate"]
    assert r1["detail"] == r2["detail"]


def test_compare_ranks_all_models(registry):
    rows = GoldenSet().compare(registry)
    assert [r["model"] for r in rows] == sorted(
        [r["model"] for r in rows],
        key=lambda m: (-next(x["hit_rate"] for x in rows
                             if x["model"] == m),
                       next(x["ref_cost_usd"] for x in rows
                            if x["model"] == m)))
    assert len(rows) == 5


def test_compare_tiebreak_prefers_free(registry):
    # With FakeBackend all scores tie; free open models must rank first.
    rows = GoldenSet().compare(registry)
    assert len({r["hit_rate"] for r in rows}) == 1  # tie across models
    free_idx = [i for i, r in enumerate(rows) if r["ref_cost_usd"] == 0.0]
    paid_idx = [i for i, r in enumerate(rows) if r["ref_cost_usd"] > 0.0]
    assert free_idx and paid_idx
    assert max(free_idx) < min(paid_idx)
    # and within paid, cheaper (gpt-4o-mini) precedes gpt-4o
    paid_models = [rows[i]["model"] for i in paid_idx]
    assert paid_models == ["gpt-4o-mini", "gpt-4o"]


def test_compare_stable_across_runs(registry):
    assert GoldenSet().compare(registry) == GoldenSet().compare(registry)
