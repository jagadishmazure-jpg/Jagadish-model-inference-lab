"""Test-time generation controls: sampling configs and multi-sample selectors.

Sampling (temperature / top-p / top-k / stop sequences) and test-time compute
(best-of-N, self-consistency) are generation-side levers: they change what a
model produces and what it costs, without touching routing. Everything here
is deterministic and seeded so tests never flake.

How sampling is simulated with FakeBackend: the fake is hash-deterministic
per (model, prompt), so independent draws are simulated by salting the prompt
with the sample seed. The salt is a harness artifact, not user content:
token accounting always uses the ORIGINAL prompt, and golden-set keyword
matching is bypassed for temperature > 0. Sampling is for open-ended
generation; evals should use the greedy profile. Live backends that accept
native sampling parameters should thread SamplingConfig through instead of
salting (OpenAIBackend/OllamaBackend do not take sampling params yet).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable

from .clients import Backend, Completion, _estimate_tokens

__all__ = [
    "BALANCED",
    "CREATIVE",
    "GREEDY",
    "PROFILES",
    "SamplingConfig",
    "best_of_n",
    "generate",
    "profile",
    "sample_n",
    "self_consistency",
]


@dataclass(frozen=True)
class SamplingConfig:
    """Per-call generation parameters.

    temperature 0.0 = greedy (fully deterministic; the seed is ignored).
    top_k = 0 disables top-k filtering. stop sequences truncate the text at
    their first occurrence.
    """

    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    stop: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if any(not s for s in self.stop):
            raise ValueError("stop sequences must be non-empty strings")
        object.__setattr__(self, "stop", tuple(self.stop))


# Per-task profiles: pick by task shape, not by model.
GREEDY = SamplingConfig(temperature=0.0, top_p=1.0, top_k=0)
BALANCED = SamplingConfig(temperature=0.7, top_p=0.9, top_k=50)
CREATIVE = SamplingConfig(temperature=1.2, top_p=0.95, top_k=100)

PROFILES: dict[str, SamplingConfig] = {
    "greedy": GREEDY,
    "balanced": BALANCED,
    "creative": CREATIVE,
}


def profile(name: str) -> SamplingConfig:
    """Look up a sampling profile by name."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown sampling profile: {name!r} (known: {sorted(PROFILES)})"
        ) from None


def _salted(prompt: str, seed: int) -> str:
    # Harness artifact only: makes each seeded draw hash differently in
    # FakeBackend. Never counted in token accounting (see generate()).
    return f"{prompt}\n<sample seed={seed}>"


def _apply_stop(completion: Completion, stop: tuple[str, ...]) -> Completion:
    text = completion.text
    for marker in stop:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    if text == completion.text:
        return completion
    return dataclasses.replace(
        completion, text=text, tokens_out=_estimate_tokens(text))


def generate(backend: Backend, prompt: str, model: str,
             config: SamplingConfig = GREEDY, seed: int = 0,
             max_tokens: int = 256) -> Completion:
    """Run one generation.

    temperature 0.0 is greedy: the seed is ignored and the result is bit
    identical across runs (and equal to backend.complete(prompt, model)).
    temperature > 0 draws a deterministic sample: the same seed always
    reproduces the same sample, different seeds (almost surely) differ.
    """
    if config.temperature == 0.0:
        completion = backend.complete(prompt, model, max_tokens=max_tokens)
    else:
        completion = backend.complete(_salted(prompt, seed), model,
                                      max_tokens=max_tokens)
        # The salt is not user content: re-count input tokens on the prompt.
        completion = dataclasses.replace(
            completion, tokens_in=_estimate_tokens(prompt))
    return _apply_stop(completion, config.stop)


def sample_n(backend: Backend, prompt: str, model: str, n: int,
             config: SamplingConfig = GREEDY, seed: int = 0,
             max_tokens: int = 256) -> list[Completion]:
    """Draw n deterministic samples (seeds seed .. seed+n-1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [generate(backend, prompt, model, config, seed=seed + i,
                     max_tokens=max_tokens)
            for i in range(n)]


def best_of_n(backend: Backend, prompt: str, model: str, n: int,
              config: SamplingConfig,
              score_fn: Callable[[Completion], float],
              seed: int = 0,
              max_tokens: int = 256) -> tuple[Completion, list[Completion]]:
    """Draw n samples and return (winner, all samples).

    The winner is the highest score_fn(sample); ties go to the earliest
    sample so the result is deterministic.
    """
    samples = sample_n(backend, prompt, model, n, config, seed, max_tokens)
    best_idx = max(range(len(samples)),
                   key=lambda i: (score_fn(samples[i]), -i))
    return samples[best_idx], samples


def _default_normalize(text: str) -> str:
    return text.strip().casefold()


def self_consistency(backend: Backend, prompt: str, model: str, n: int,
                     config: SamplingConfig, seed: int = 0,
                     max_tokens: int = 256,
                     normalize: Callable[[str], str] | None = None
                     ) -> tuple[str, dict[str, int]]:
    """Draw n samples and take a majority vote over normalized answers.

    Returns (winning answer, vote counts). The default normalization is
    strip + casefold. Ties go to the answer seen first in sample order,
    so the result is deterministic.
    """
    norm = normalize or _default_normalize
    samples = sample_n(backend, prompt, model, n, config, seed, max_tokens)
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for i, completion in enumerate(samples):
        answer = norm(completion.text)
        counts[answer] = counts.get(answer, 0) + 1
        first_seen.setdefault(answer, i)
    winner = max(counts, key=lambda a: (counts[a], -first_seen[a]))
    ordered = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return winner, ordered
