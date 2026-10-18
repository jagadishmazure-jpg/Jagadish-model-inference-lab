# Model cards

Each model the router can pick has a model card built by
`inferlab.model_cards`. The fields follow the model-card idea (Mitchell et
al., 2019): model details, intended use, out-of-scope use, limitations,
evaluation results and cost. Each card also records how the deployment maps
to the NIST AI RMF functions (Govern, Map, Measure, Manage) and to the EU AI
Act transparency duties for a deployer.

This is documentation for a learning lab, not legal advice. The evaluation
numbers are **simulated** by `FakeBackend` (a seeded accuracy per quality
tier), so they show the mechanics of eval-informed routing, not the quality of
the real models. Re-run `inferlab.evals` against real backends before using
any number here.

`validate()` rejects a card with a missing required field; the test suite
runs it on every shipped card.

<!-- run: python -m inferlab.model_cards -->
```text
### gpt-4o-mini (card v1.0)

- **Model details:** openai, closed weights, license proprietary (API terms), context 128,000 tokens
- **Intended use here:** Short, low-risk tasks routed as 'simple': lookups, reformatting, one-step answers.
- **Out of scope:** Decisions about people (credit, hiring, medical, legal) without human review. Any task the eval set does not cover, until it has its own golden items. Handling secrets or unredacted personal data (redact before routing).
- **Limitations:** Can produce fluent but wrong answers; the eval here is keyword-based. Offline eval scores are simulated by FakeBackend and must be re-run on the real model.
- **Evaluation (simulated by FakeBackend):** simple 0.667 on 9 items, complex 0.286 on 7 items
- **Cost:** List price $0.15 in / $0.60 out per 1M tokens, as of 2026-10-17 (https://developers.openai.com/api/docs/pricing (standard tier, short context)).

### gpt-4o (card v1.0)

- **Model details:** openai, closed weights, license proprietary (API terms), context 128,000 tokens
- **Intended use here:** Complex reasoning and explanations; the escalation target when cheaper models fail the floor.
- **Out of scope:** Decisions about people (credit, hiring, medical, legal) without human review. Any task the eval set does not cover, until it has its own golden items. Handling secrets or unredacted personal data (redact before routing).
- **Limitations:** Can produce fluent but wrong answers; the eval here is keyword-based. Offline eval scores are simulated by FakeBackend and must be re-run on the real model.
- **Evaluation (simulated by FakeBackend):** simple 1.000 on 9 items, complex 1.000 on 7 items
- **Cost:** List price $2.50 in / $10.00 out per 1M tokens, as of 2026-10-17 (https://developers.openai.com/api/docs/pricing (standard tier, short context)).

### llama-3.1-8b (card v1.0)

- **Model details:** ollama, open weights, license Llama 3.1 Community License, context 128,000 tokens
- **Intended use here:** Short, low-risk tasks routed as 'simple': lookups, reformatting, one-step answers.
- **Out of scope:** Decisions about people (credit, hiring, medical, legal) without human review. Any task the eval set does not cover, until it has its own golden items. Handling secrets or unredacted personal data (redact before routing).
- **Limitations:** Can produce fluent but wrong answers; the eval here is keyword-based. Offline eval scores are simulated by FakeBackend and must be re-run on the real model. Self-hosted: quality depends on the quantisation and server build you run.
- **Evaluation (simulated by FakeBackend):** simple 0.889 on 9 items, complex 0.429 on 7 items
- **Cost:** Amortised local infrastructure estimate: $0.083 in / $0.833 out per 1M tokens (assumptions in models.LocalInfra).

### mistral-7b-instruct (card v1.0)

- **Model details:** ollama, open weights, license Apache-2.0, context 32,768 tokens
- **Intended use here:** Short, low-risk tasks routed as 'simple': lookups, reformatting, one-step answers.
- **Out of scope:** Decisions about people (credit, hiring, medical, legal) without human review. Any task the eval set does not cover, until it has its own golden items. Handling secrets or unredacted personal data (redact before routing).
- **Limitations:** Can produce fluent but wrong answers; the eval here is keyword-based. Offline eval scores are simulated by FakeBackend and must be re-run on the real model. Self-hosted: quality depends on the quantisation and server build you run. Context window 32,768 tokens; long documents need chunking.
- **Evaluation (simulated by FakeBackend):** simple 0.667 on 9 items, complex 0.429 on 7 items
- **Cost:** Amortised local infrastructure estimate: $0.083 in / $0.833 out per 1M tokens (assumptions in models.LocalInfra).

### qwen2.5-7b (card v1.0)

- **Model details:** ollama, open weights, license Apache-2.0, context 32,768 tokens
- **Intended use here:** Simple tasks and moderately complex explanations when it passes the eval floor.
- **Out of scope:** Decisions about people (credit, hiring, medical, legal) without human review. Any task the eval set does not cover, until it has its own golden items. Handling secrets or unredacted personal data (redact before routing).
- **Limitations:** Can produce fluent but wrong answers; the eval here is keyword-based. Offline eval scores are simulated by FakeBackend and must be re-run on the real model. Self-hosted: quality depends on the quantisation and server build you run. Context window 32,768 tokens; long documents need chunking.
- **Evaluation (simulated by FakeBackend):** simple 0.889 on 9 items, complex 0.714 on 7 items
- **Cost:** Amortised local infrastructure estimate: $0.083 in / $0.833 out per 1M tokens (assumptions in models.LocalInfra).

### Governance notes (same for every card)

- **NIST AI RMF, Govern:** Owner: the platform team; routing cards and model cards are versioned in git and reviewed in pull requests.
- **NIST AI RMF, Map:** Use is limited to the task classes above; out-of-scope uses are listed and refused by policy, not by the model.
- **NIST AI RMF, Measure:** Golden-set hit rate per task class, cost per reference call, latency tier; re-measured on every change.
- **NIST AI RMF, Manage:** Quality floor in EvalInformedRouter, fallback chain with circuit breakers, per-session budget, incident = roll back the card.
- **EU AI Act:** Role: deployer of general-purpose AI models supplied by their providers.
- **EU AI Act:** Art. 50 transparency: tell users they are interacting with an AI system, and label AI-generated text where it is published.
- **EU AI Act:** Risk class of this use: not high-risk as scoped (no Annex III use such as credit scoring or hiring); re-assess if the use changes.
- **EU AI Act:** Keep provider documentation (GPAI technical docs and usage policy) on file and follow it.
```
<!-- /run -->
