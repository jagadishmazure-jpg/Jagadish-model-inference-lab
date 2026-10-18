"""Per-call cost/latency ledger with budget enforcement.

Every inference call is logged (model, backend, tokens, ttft, tpot,
cost). check_budget() projects the next call's cost and raises
BudgetExceeded BEFORE the call when it would breach the limit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .clients import Completion
from .models import RoutingCard


class BudgetExceeded(RuntimeError):
    """Projected spend would breach the session budget."""


@dataclass
class CallRecord:
    timestamp: float
    model: str
    backend: str
    tokens_in: int
    tokens_out: int
    ttft_ms: float
    tpot_ms: float
    cost_usd: float


class CallTracker:
    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def log(self, *, model: str, backend: str, card: RoutingCard,
            completion: Completion) -> CallRecord:
        record = CallRecord(
            timestamp=time.time(),
            model=model,
            backend=backend,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            ttft_ms=completion.ttft_ms,
            tpot_ms=completion.tpot_ms,
            cost_usd=card.cost_usd(completion.tokens_in,
                                   completion.tokens_out),
        )
        self.calls.append(record)
        return record

    def session_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def check_budget(self, limit_usd: float, projected_cost_usd: float) -> None:
        """Raise BudgetExceeded if limit would be breached by the next call."""
        if self.session_cost() + projected_cost_usd > limit_usd:
            raise BudgetExceeded(
                f"budget ${limit_usd:.4f} would be breached: "
                f"spent ${self.session_cost():.4f} + "
                f"projected ${projected_cost_usd:.4f}"
            )

    def summary(self) -> str:
        lines = ["model | backend | in | out | ttft_ms | tpot_ms | cost_usd"]
        for c in self.calls:
            lines.append(
                f"{c.model} | {c.backend} | {c.tokens_in} | {c.tokens_out} | "
                f"{c.ttft_ms:.1f} | {c.tpot_ms:.1f} | ${c.cost_usd:.6f}"
            )
        lines.append(f"TOTAL | | | | | | ${self.session_cost():.6f}")
        return "\n".join(lines)

    def totals_by_model(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for c in self.calls:
            agg = out.setdefault(c.model, {"calls": 0, "cost_usd": 0.0,
                                           "tokens_in": 0, "tokens_out": 0})
            agg["calls"] += 1
            agg["cost_usd"] += c.cost_usd
            agg["tokens_in"] += c.tokens_in
            agg["tokens_out"] += c.tokens_out
        return out
