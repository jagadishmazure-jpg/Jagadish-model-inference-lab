"""Backend clients behind one interface.

Backends read credentials/endpoints from the environment AT CALL TIME and
degrade gracefully (raise BackendUnavailable) when the dependency is
absent. Tests use ONLY FakeBackend — no network, no keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .models import ModelRegistry, ModelCard


class BackendUnavailable(RuntimeError):
    """A backend cannot serve right now (no key, no server, timeout)."""


@dataclass(frozen=True)
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    ttft_ms: float   # time to first token
    tpot_ms: float   # time per output token


class Backend:
    name: str = "base"

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        raise NotImplementedError


def _estimate_tokens(text: str) -> int:
    # Crude char-based estimate; real tokenizers live in the live path.
    return max(1, len(text) // 4)


class OpenAIBackend(Backend):
    """Closed-source backend: OpenAI Chat Completions over HTTPS (urllib)."""

    name = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise BackendUnavailable(
                "OPENAI_API_KEY is not set; cannot reach OpenAI"
            )
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailable(f"OpenAI request failed: {exc}") from exc
        choice = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        # The API does not report ttft/tpot; live runs measure them.
        return Completion(
            text=choice,
            tokens_in=usage.get("prompt_tokens", _estimate_tokens(prompt)),
            tokens_out=usage.get("completion_tokens",
                                 _estimate_tokens(choice)),
            ttft_ms=0.0, tpot_ms=0.0,
        )


class OllamaBackend(Backend):
    """Open-weights backend: local Ollama server."""

    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        self.host = host or os.environ.get("OLLAMA_HOST",
                                           "http://localhost:11434")

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        body = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailable(
                f"Ollama not reachable at {self.host}: {exc}"
            ) from exc
        text = payload.get("response", "")
        return Completion(
            text=text,
            tokens_in=payload.get("prompt_eval_count",
                                  _estimate_tokens(prompt)),
            tokens_out=payload.get("eval_count", _estimate_tokens(text)),
            ttft_ms=0.0, tpot_ms=0.0,
        )


class FakeBackend(Backend):
    """Deterministic stand-in for tests and offline runs.

    Responses are canned per prompt-hash so runs are reproducible; latency
    and token counts are derived from the model card so routing/cost math
    exercises realistic numbers.
    """

    name = "fake"

    # Simulated per-tier latency (ms)
    _TTFT = {1: 120.0, 2: 350.0, 3: 900.0}
    _TPOT = {1: 18.0, 2: 32.0, 3: 55.0}

    _RESPONSES = [
        "Here is a concise answer based on the request.",
        "The key points are: first, the input was understood; second, "
        "the answer follows the expected format.",
        "Answer: the requested information is summarized below in brief.",
    ]

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def complete(self, prompt: str, model: str, max_tokens: int = 256) -> Completion:
        card = self.registry.get(model)
        digest = hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()
        text = self._RESPONSES[int(digest, 16) % len(self._RESPONSES)]
        tokens_in = _estimate_tokens(prompt)
        tokens_out = min(_estimate_tokens(text), max_tokens)
        return Completion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_ms=self._TTFT[card.latency_tier],
            tpot_ms=self._TPOT[card.latency_tier],
        )
