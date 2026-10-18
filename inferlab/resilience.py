"""Retries with backoff, per-provider circuit breakers, and rate-limit errors.

Everything takes an injectable clock / sleep / random source so tests and the
demo are deterministic and never actually sleep.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable


class BackendUnavailable(RuntimeError):
    """A backend cannot serve right now (no key, no server, all hops failed)."""


class TransientBackendError(BackendUnavailable):
    """Worth retrying on the same hop: a timeout or a 5xx from the provider."""


class RateLimited(TransientBackendError):
    """HTTP 429. ``retry_after_s`` is the provider's Retry-After, if it sent one."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class CircuitOpen(BackendUnavailable):
    """The breaker for this provider is open; the hop is skipped without a call."""


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After in seconds (the HTTP-date form is not supported here)."""
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter, for transient errors only.

    Delay before retry n (1-based) is ``uniform(0, min(cap, base * 2**(n-1)))``.
    A 429 with Retry-After waits exactly that long, unless it is longer than
    ``max_retry_after_s``, in which case the hop gives up and the chain moves on.
    """

    max_attempts: int = 3
    base_s: float = 0.5
    cap_s: float = 8.0
    max_retry_after_s: float = 10.0
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def delay(self, attempt: int, error: Exception) -> float | None:
        """Seconds to wait before the next attempt, or None to stop retrying."""
        if attempt >= self.max_attempts or not isinstance(error, (TransientBackendError, TimeoutError)):
            return None
        if isinstance(error, RateLimited) and error.retry_after_s is not None:
            if error.retry_after_s > self.max_retry_after_s:
                return None
            return error.retry_after_s
        return self.rng.uniform(0, min(self.cap_s, self.base_s * 2 ** (attempt - 1)))


class CircuitBreaker:
    """Closed -> open after ``failure_threshold`` consecutive failures.

    While open, calls are refused for ``cooldown_s``. After that one trial call
    is let through (half-open): success closes the breaker, failure re-opens it.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self._clock() - self.opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        if self.state == "half_open":
            self.opened_at = self._clock()
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self._clock()
