"""Deterministic tests for test-time generation controls (sampling configs,
best-of-N, self-consistency) and per-language token multipliers.

FakeBackend and ScriptedBackend only -- no network, no keys. Every test is
seeded or scripted, so nothing flakes.
"""
import pytest

from inferlab.clients import Completion, FakeBackend, ScriptedBackend
from inferlab.generation import (
    BALANCED,
    CREATIVE,
    GREEDY,
    PROFILES,
    SamplingConfig,
    best_of_n,
    generate,
    profile,
    sample_n,
    self_consistency,
)
from inferlab.models import ModelRegistry
from inferlab.tracker import BudgetExceeded, CallTracker


@pytest.fixture()
def backend():
    return FakeBackend()


@pytest.fixture()
def registry():
    return ModelRegistry()


# ---------- sampling configs & profiles ----------

def test_profile_shapes():
    assert GREEDY.temperature == 0.0
    assert BALANCED.temperature == 0.7
    assert CREATIVE.temperature == 1.2
    assert profile("greedy") is GREEDY
    assert profile("balanced") is BALANCED
    assert profile("creative") is CREATIVE
    assert set(PROFILES) == {"greedy", "balanced", "creative"}


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        profile("maximum-overdrive")


@pytest.mark.parametrize("kwargs", [
    {"temperature": -0.1},
    {"top_p": 0.0},
    {"top_p": 1.5},
    {"top_k": -1},
    {"stop": ("",)},
])
def test_sampling_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_stop_list_is_coerced_to_tuple():
    assert SamplingConfig(stop=["a", "b"]).stop == ("a", "b")


# ---------- greedy determinism ----------

def test_greedy_is_deterministic_across_seeds(backend):
    prompt = "Summarize the refund policy."
    texts = [generate(backend, prompt, "gpt-4o-mini", GREEDY, seed=s).text
             for s in range(5)]
    assert len(set(texts)) == 1


def test_greedy_matches_direct_backend_call(backend):
    prompt = "Summarize the refund policy."
    got = generate(backend, prompt, "gpt-4o-mini", GREEDY, seed=99)
    want = backend.complete(prompt, "gpt-4o-mini")
    assert got == want


# ---------- seeded sampling ----------

def test_sampled_draws_vary_by_seed_but_reproduce(backend):
    prompt = "Explain the refund policy in detail."
    samples = [generate(backend, prompt, "gpt-4o-mini", BALANCED, seed=s)
               for s in range(12)]
    assert len({c.text for c in samples}) >= 2  # the fake has 3 canned replies
    again = generate(backend, prompt, "gpt-4o-mini", BALANCED, seed=4)
    assert again.text == samples[4].text


def test_sampled_tokens_count_the_original_prompt(backend):
    prompt = "Explain the refund policy in detail."
    completion = generate(backend, prompt, "gpt-4o-mini", BALANCED, seed=7)
    assert completion.tokens_in == max(1, len(prompt) // 4)


def test_stop_sequences_truncate():
    backend = ScriptedBackend("stop", ["alpha STOP beta"])
    completion = generate(backend, "p", "gpt-4o-mini",
                          SamplingConfig(temperature=0.5, stop=("STOP",)),
                          seed=0)
    assert completion.text == "alpha "
    assert completion.tokens_out == max(1, len("alpha ") // 4)


def test_sample_n_rejects_bad_n(backend):
    with pytest.raises(ValueError):
        sample_n(backend, "p", "gpt-4o-mini", 0, BALANCED)


# ---------- best-of-N ----------

def test_best_of_n_picks_highest_score():
    backend = ScriptedBackend("bon", ["short", "a much longer response here",
                                      "mid"])
    winner, samples = best_of_n(backend, "p", "gpt-4o-mini", n=3,
                                config=BALANCED,
                                score_fn=lambda c: len(c.text), seed=0)
    assert winner.text == "a much longer response here"
    assert len(samples) == 3


def test_best_of_n_tie_goes_to_earliest_sample():
    backend = ScriptedBackend("tie", ["aa", "bb"])
    winner, _ = best_of_n(backend, "p", "gpt-4o-mini", n=2, config=BALANCED,
                          score_fn=lambda c: len(c.text), seed=0)
    assert winner.text == "aa"


# ---------- self-consistency ----------

def test_self_consistency_picks_majority():
    backend = ScriptedBackend("vote", ["alpha", "beta", "alpha", "alpha",
                                       "beta"])
    winner, counts = self_consistency(backend, "p", "gpt-4o-mini", n=5,
                                      config=BALANCED, seed=0)
    assert winner == "alpha"
    assert counts == {"alpha": 3, "beta": 2}


def test_self_consistency_normalizes_by_default():
    backend = ScriptedBackend("norm", ["Yes", "yes", "YES", "no"])
    winner, counts = self_consistency(backend, "p", "gpt-4o-mini", n=4,
                                      config=BALANCED, seed=0)
    assert winner == "yes"
    assert counts == {"yes": 3, "no": 1}


def test_self_consistency_tie_goes_to_first_seen():
    backend = ScriptedBackend("tie2", ["beta", "alpha", "beta", "alpha"])
    winner, counts = self_consistency(backend, "p", "gpt-4o-mini", n=4,
                                      config=BALANCED, seed=0)
    assert counts == {"alpha": 2, "beta": 2}
    assert winner == "beta"  # "beta" was seen first


# ---------- per-language token multipliers ----------

def test_all_shipped_cards_carry_hindi_multiplier(registry):
    for card in registry.list():
        assert card.token_multipliers["hi"] == 2.0
        assert card.multiplier_for("en") == 1.0
        assert card.multiplier_for("fr") == 1.0  # unknown -> 1.0


def test_cost_usd_applies_language_multiplier(registry):
    card = registry.get("gpt-4o-mini")
    english = card.cost_usd(1000, 2000)
    assert card.cost_usd(1000, 2000, language="en") == pytest.approx(english)
    assert card.cost_usd(1000, 2000, language="hi") == pytest.approx(2 * english)
    assert card.blended_ref_cost("hi") == pytest.approx(
        2 * card.blended_ref_cost())


def _ref_completion():
    return Completion(text="ok", tokens_in=100, tokens_out=50,
                      ttft_ms=10.0, tpot_ms=5.0)


def test_tracker_log_applies_language_multiplier(registry):
    tracker = CallTracker()
    card = registry.get("gpt-4o-mini")
    english = tracker.log(model="gpt-4o-mini", backend="fake", card=card,
                          completion=_ref_completion(), language="en")
    hindi = tracker.log(model="gpt-4o-mini", backend="fake", card=card,
                        completion=_ref_completion(), language="hi")
    assert hindi.cost_usd == pytest.approx(2 * english.cost_usd)


def test_budget_bitten_by_hindi_multiplier(registry):
    card = registry.get("gpt-4o-mini")
    ref_en = card.blended_ref_cost("en")
    limit = 3 * ref_en

    tracker = CallTracker()
    tracker.log(model="gpt-4o-mini", backend="fake", card=card,
                completion=_ref_completion(), language="hi")  # spends 2x
    with pytest.raises(BudgetExceeded):
        tracker.check_budget(limit, card.blended_ref_cost("hi"))

    tracker_en = CallTracker()
    tracker_en.log(model="gpt-4o-mini", backend="fake", card=card,
                   completion=_ref_completion(), language="en")
    tracker_en.check_budget(limit, card.blended_ref_cost("en"))  # no raise


def test_provider_reported_tokens_are_not_multiplied(registry):
    # Real usage counts already reflect the tokenizer; only heuristic estimates get the multiplier.
    card = registry.get("gpt-4o-mini")
    reported = Completion(text="ok", tokens_in=100, tokens_out=50, ttft_ms=10.0, tpot_ms=5.0,
                          tokens_estimated=False)
    tracker = CallTracker()
    hi = tracker.log(model="gpt-4o-mini", backend="openai", card=card, completion=reported, language="hi")
    en = tracker.log(model="gpt-4o-mini", backend="openai", card=card, completion=reported, language="en")
    assert hi.cost_usd == pytest.approx(en.cost_usd)


def test_fake_backend_marks_tokens_as_estimated():
    from inferlab.clients import FakeBackend

    assert FakeBackend().complete("hello", "gpt-4o-mini").tokens_estimated is True
