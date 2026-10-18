# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| Hosted models | Azure OpenAI or Azure AI Foundry deployments | Use your contract prices in the cards |
| Gateway, rate limits, retries | Azure API Management with the GenAI gateway policies (token limits, backend pools, circuit breaker) | Keep app-side breakers for non-APIM backends |
| Local models | vLLM or Ollama on Container Apps GPU or AKS | Measure utilisation to set LocalInfra |
| Golden-set evals | Azure AI Foundry evaluations, run per model on each change |  |
| Tracking | Application Insights custom metrics per call |  |

## Steps

1. Run the golden set against each real backend and replace the simulated scores.
2. Grow the golden set per task class until the confidence intervals separate the candidates.
3. Measure real utilisation of the GPU pool and set `LocalInfra` from it.
4. Put hosted calls behind API Management and keep the fallback chain for cross-provider failover.

## Guardrails to keep when going live

- Quality floor per task class; a cheap model that fails it is not chosen.
- Retry-After is honoured and capped; backoff is jittered so clients do not retry in step.
- Circuit breakers stop calls to a failing provider and probe it again after a cool-down.
- Empty or invalid output does not count as success; the chain moves to the next hop.
- Session budget check before each call.
- Credentials are read from the environment at call time and never logged.

## What to measure before calling it production-ready

- Hit rate per task class per model on the golden set, with sample size.
- Cost per successful call, including retries and fallbacks.
- Breaker open time per provider; share of calls served by a fallback.
