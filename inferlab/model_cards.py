"""Model cards for the models in the routing fleet.

A routing card (``models.RoutingCard``) is what the router needs. A model card
is what a reviewer needs: what the model is used for *in this system*, what it
must not be used for, its known limitations, the eval results behind the
routing, and the governance notes. The fields follow the Model Cards for Model
Reporting structure (model details, intended use, factors, metrics, evaluation
data, caveats), with sections mapped to the NIST AI RMF functions and to EU AI
Act transparency duties.

Cards are generated from the routing cards plus a real eval run, so the
numbers are never typed by hand; ``validate()`` fails if a required field is
empty. Run ``python -m inferlab.model_cards`` to print them.

Scope note: I did not train any of these models. Vendor facts (context window,
license) come from the routing cards; everything about intended use and risk
is about *this* routing system, as the deployer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .evals import GoldenSet
from .models import PRICE_SOURCE, PRICES_AS_OF, ModelRegistry, RoutingCard

CARD_VERSION = "1.0"

REQUIRED = ("name", "intended_use", "out_of_scope", "limitations", "evaluation",
            "nist_ai_rmf", "eu_ai_act", "price_note")

_INTENDED = {
    1: "Short, low-risk tasks routed as 'simple': lookups, reformatting, one-step answers.",
    2: "Simple tasks and moderately complex explanations when it passes the eval floor.",
    3: "Complex reasoning and explanations; the escalation target when cheaper models fail the floor.",
}

_COMMON_OUT_OF_SCOPE = [
    "Decisions about people (credit, hiring, medical, legal) without human review.",
    "Any task the eval set does not cover, until it has its own golden items.",
    "Handling secrets or unredacted personal data (redact before routing).",
]

_COMMON_LIMITS = [
    "Can produce fluent but wrong answers; the eval here is keyword-based.",
    "Offline eval scores are simulated by FakeBackend and must be re-run on the real model.",
]

_NIST = {
    "Govern": "Owner: the platform team; routing cards and model cards are versioned in git and reviewed in pull requests.",
    "Map": "Use is limited to the task classes above; out-of-scope uses are listed and refused by policy, not by the model.",
    "Measure": "Golden-set hit rate per task class, cost per reference call, latency tier; re-measured on every change.",
    "Manage": "Quality floor in EvalInformedRouter, fallback chain with circuit breakers, per-session budget, incident = roll back the card.",
}

_EU_AI_ACT = [
    "Role: deployer of general-purpose AI models supplied by their providers.",
    "Art. 50 transparency: tell users they are interacting with an AI system, and label AI-generated text where it is published.",
    "Risk class of this use: not high-risk as scoped (no Annex III use such as credit scoring or hiring); re-assess if the use changes.",
    "Keep provider documentation (GPAI technical docs and usage policy) on file and follow it.",
]


@dataclass
class ModelCardDoc:
    name: str
    provider: str
    weights: str
    license_name: str
    context_window: int
    intended_use: str
    out_of_scope: list[str]
    limitations: list[str]
    evaluation: dict
    nist_ai_rmf: dict[str, str]
    eu_ai_act: list[str]
    price_note: str
    version: str = CARD_VERSION
    extra: dict = field(default_factory=dict)


def _price_note(card: RoutingCard) -> str:
    if card.price_basis == "amortised":
        return (f"Amortised local infrastructure estimate: ${card.price_in_per_1k * 1000:.3f} in / "
                f"${card.price_out_per_1k * 1000:.3f} out per 1M tokens (assumptions in models.LocalInfra).")
    return (f"List price ${card.price_in_per_1k * 1000:.2f} in / ${card.price_out_per_1k * 1000:.2f} out "
            f"per 1M tokens, as of {PRICES_AS_OF} ({PRICE_SOURCE}).")


def build_cards(registry: ModelRegistry | None = None,
                eval_rows: list[dict] | None = None) -> list[ModelCardDoc]:
    registry = registry or ModelRegistry()
    rows = {r["model"]: r for r in (eval_rows or GoldenSet().compare(registry))}
    cards = []
    for c in registry.list():
        r = rows[c.name]
        limits = list(_COMMON_LIMITS)
        if c.weights == "open":
            limits.append("Self-hosted: quality depends on the quantisation and server build you run.")
        if c.context_window < 100_000:
            limits.append(f"Context window {c.context_window:,} tokens; long documents need chunking.")
        cards.append(ModelCardDoc(
            name=c.name, provider=c.provider, weights=c.weights,
            license_name=c.license_name, context_window=c.context_window,
            intended_use=_INTENDED[c.quality_tier],
            out_of_scope=list(_COMMON_OUT_OF_SCOPE),
            limitations=limits,
            evaluation={"backend": r["backend"], "simulated": r["backend"] == "fake",
                        "simple": round(r["simple"], 3), "simple_n": r["simple_n"],
                        "complex": round(r["complex"], 3), "complex_n": r["complex_n"]},
            nist_ai_rmf=dict(_NIST), eu_ai_act=list(_EU_AI_ACT),
            price_note=_price_note(c)))
    return cards


def validate(card: ModelCardDoc) -> list[str]:
    """Names of required fields that are missing or empty."""
    return [f for f in REQUIRED if not getattr(card, f)]


def render_markdown(cards: list[ModelCardDoc]) -> str:
    out = []
    for c in cards:
        e = c.evaluation
        sim = " (simulated by FakeBackend)" if e["simulated"] else ""
        out += [f"### {c.name} (card v{c.version})", "",
                f"- **Model details:** {c.provider}, {c.weights} weights, license {c.license_name}, "
                f"context {c.context_window:,} tokens",
                f"- **Intended use here:** {c.intended_use}",
                "- **Out of scope:** " + " ".join(c.out_of_scope),
                "- **Limitations:** " + " ".join(c.limitations),
                f"- **Evaluation{sim}:** simple {e['simple']:.3f} on {e['simple_n']} items, "
                f"complex {e['complex']:.3f} on {e['complex_n']} items",
                f"- **Cost:** {c.price_note}", ""]
    out += ["### Governance notes (same for every card)", ""]
    out += [f"- **NIST AI RMF, {k}:** {v}" for k, v in _NIST.items()]
    out += [f"- **EU AI Act:** {x}" for x in _EU_AI_ACT]
    return "\n".join(out)


if __name__ == "__main__":
    cards = build_cards()
    bad = {c.name: validate(c) for c in cards if validate(c)}
    if bad:
        raise SystemExit(f"invalid cards: {bad}")
    print(render_markdown(cards))
