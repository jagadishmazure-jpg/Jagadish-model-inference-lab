"""Fallback chains with retries, per-provider circuit breakers, rate-limit
handling and output validation.

For each hop in order:
1. If that provider's circuit breaker is open, skip the hop without calling it.
2. Call it. Transient errors (timeouts, 5xx, 429) are retried on the same hop
   with ``RetryPolicy`` backoff; a 429's Retry-After is honoured.
3. A completion that fails ``validator`` (empty text by default) counts as a
   failure, so a broken answer falls through to the next hop like an outage.
4. Every attempt is recorded, so the failover path is auditable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .clients import Backend, Completion
from .resilience import BackendUnavailable, CircuitBreaker, RateLimited, RetryPolicy, TransientBackendError


def non_empty(completion: Completion) -> bool:
    return bool(completion.text.strip())


@dataclass
class FallbackRecord:
    served_by_backend: str
    served_by_model: str
    attempts: list[dict] = field(default_factory=list)
    waited_s: float = 0.0


class FallbackChain:
    def __init__(self, hops: list[tuple[Backend, str]], *,
                 retry: RetryPolicy | None = None,
                 breakers: dict[str, CircuitBreaker] | None = None,
                 validator: Callable[[Completion], bool] = non_empty,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not hops:
            raise ValueError("fallback chain needs at least one hop")
        self.hops = list(hops)
        self.retry = retry or RetryPolicy(max_attempts=1)
        self.breakers = breakers if breakers is not None else {}
        self.validator = validator
        self.sleep = sleep
        self.last_record: FallbackRecord | None = None

    def _breaker(self, backend: Backend) -> CircuitBreaker:
        return self.breakers.setdefault(backend.name, CircuitBreaker())

    def complete(self, prompt: str, max_tokens: int = 256) -> Completion:
        record = FallbackRecord(served_by_backend="", served_by_model="")
        self.last_record = record
        for backend, model in self.hops:
            breaker = self._breaker(backend)
            if not breaker.allow():
                record.attempts.append({"backend": backend.name, "model": model,
                                        "outcome": "circuit_open"})
                continue
            attempt = 0
            while True:
                attempt += 1
                try:
                    completion = backend.complete(prompt, model, max_tokens=max_tokens)
                except Exception as exc:  # classified below
                    outcome = _classify(exc)
                    record.attempts.append({"backend": backend.name, "model": model,
                                            "outcome": outcome, "attempt": attempt})
                    if outcome == "error":
                        raise
                    breaker.record_failure()
                    wait = self.retry.delay(attempt, exc) if breaker.allow() else None
                    if wait is None:
                        break
                    record.waited_s += wait
                    self.sleep(wait)
                    continue
                if not self.validator(completion):
                    breaker.record_failure()
                    record.attempts.append({"backend": backend.name, "model": model,
                                            "outcome": "invalid_output", "attempt": attempt})
                    break
                breaker.record_success()
                record.attempts.append({"backend": backend.name, "model": model,
                                        "outcome": "ok", "attempt": attempt})
                record.served_by_backend = backend.name
                record.served_by_model = model
                return completion
        raise BackendUnavailable(
            f"all {len(self.hops)} fallback hops failed: "
            + ", ".join(f"{a['backend']}/{a['model']}={a['outcome']}" for a in record.attempts))


def _classify(exc: Exception) -> str:
    if isinstance(exc, RateLimited):
        return "rate_limited"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, TransientBackendError):
        return "transient"
    if isinstance(exc, BackendUnavailable):
        return "unavailable"
    return "error"  # a bug, not an outage: re-raised, never silently failed over
