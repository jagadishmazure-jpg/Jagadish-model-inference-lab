"""Fallback chains: ordered (backend, model) hops with graceful degradation.

Try each hop in order; on BackendUnavailable or TimeoutError move to the
next. The record shows which hop served, so failover is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clients import Backend, BackendUnavailable, Completion


@dataclass
class FallbackRecord:
    served_by_backend: str
    served_by_model: str
    attempts: list[dict] = field(default_factory=list)
    # attempts: [{"backend":..., "model":..., "outcome": "ok"|"unavailable"|"timeout"}]


class FallbackChain:
    def __init__(self, hops: list[tuple[Backend, str]]) -> None:
        if not hops:
            raise ValueError("fallback chain needs at least one hop")
        self.hops = list(hops)
        self.last_record: FallbackRecord | None = None

    def complete(self, prompt: str, max_tokens: int = 256) -> Completion:
        record = FallbackRecord(served_by_backend="", served_by_model="")
        for backend, model in self.hops:
            try:
                completion = backend.complete(prompt, model,
                                              max_tokens=max_tokens)
            except BackendUnavailable:
                record.attempts.append({"backend": backend.name,
                                        "model": model,
                                        "outcome": "unavailable"})
                continue
            except TimeoutError:
                record.attempts.append({"backend": backend.name,
                                        "model": model,
                                        "outcome": "timeout"})
                continue
            record.attempts.append({"backend": backend.name, "model": model,
                                    "outcome": "ok"})
            record.served_by_backend = backend.name
            record.served_by_model = model
            self.last_record = record
            return completion
        self.last_record = record
        raise BackendUnavailable(
            f"all {len(self.hops)} fallback hops unavailable: "
            + ", ".join(f"{a['backend']}/{a['model']}" for a in record.attempts)
        )
