# Interview points

Short, honest talking points for Jagadish-model-inference-lab. Every number quoted is from the
real run shown in the README.

- Open models are not free: at 5% GPU utilisation the amortised cost per call is about 11 times gpt-4o-mini's list price.
- Route on measured quality per task class with a floor, then on cost; the reason string shows the evidence.
- Retry-After first, then full-jitter backoff, then a breaker; and never retry a programming error.
- I renamed my 'model cards' to routing cards, because a model card documents use, limits and evals, and added real ones.

## Likely follow-up questions

- How would you avoid overfitting routing to a small golden set? (Hold-out items and per-class sample-size minimums.)
- What if the fallback model is much weaker? (Lower the task's ambition or fail visibly; do not silently degrade high-risk tasks.)
- Where does the EU AI Act bite here? (Transparency to users; the use case decides the risk class, not the router.)
