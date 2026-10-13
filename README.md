# NTG smart router

Python 3.11+. This rebuilds the existing `ntg.router.engine.UnifiedNTGRouter`; it does not introduce a second router. The old implementation depended on missing modules and persisted quota estimates. No database, rate-limit tables, usage counters, or availability probes are used here.

```powershell
python -m pip install -r requirements.txt
python -m ntg --status
python -m ntg "Explain async Python briefly" --provider groq --model openai/gpt-oss-120b
python -m ntg "Hello" --provider openrouter --model openrouter/free
```

The local `.env` contains the 24 supplied credentials; `.env.example` contains placeholders. Both `.env` and the original credential text are ignored by Git. Do not commit either. Numbered keys can be added without code changes. Environment variables override `.env`.

The configured accounts have billing disabled, as confirmed by the user. All accounts participate without a local free-plan declaration. Their default pools now contain current free-tier text-chat models checked against official documentation and live catalogs. See [the provider review](docs/provider-model-review.md), [model pools](docs/free-model-pools.json), and [per-account catalog evidence](docs/model-catalog-audit.json).

Set `PROVIDER_N_MODELS` to comma-separated native identifiers in the desired fallback order. Requests stay within these configured pools. Explicit unlisted or deprecated model IDs are not added automatically; with normal fallback, a configured alternative is attempted. Provider APIs enforce actual access and quota, and `.env` must be reviewed again when catalogs change.

Use `PROVIDER_N_QUOTA_GROUP` to identify keys sharing a project, organization or account. Gemini and Groq limits usually apply per model within that shared group. Cohere chat limits are per model, while its monthly trial cap is shared. OpenRouter account-wide free caps and Cloudflare task/daily limits are shared. Explicit quota scopes in provider errors override defaults. Cloudflare keys are grouped automatically by account ID. Additional keys sharing an account do not create more allowance.

```python
import asyncio
from ntg.router import UnifiedNTGRouter, RoutingError

async def main():
    router = UnifiedNTGRouter()
    try:
        result = await router.acompletion(
            [{"role": "user", "content": "Hello"}],
            providers=["groq"], model="openai/gpt-oss-120b",
            max_tokens=128, max_attempts=5, deadline_seconds=30)
        print(result.response.choices[0].message.content)
        print(result.provider, result.account_id, result.model, result.attempts)
    except RoutingError as error:
        print(error.details)
    print(router.availability())
    await router.reenable("groq-1")  # session exhaustion/server blocks; cannot clear invalid credentials
    await router.reload()            # reload configuration and clear operational state; requires idle router

asyncio.run(main())
```

Keep one router per event loop so requests share operational cooldowns. CLI quota cooldowns start fresh, while routing positions persist across commands. New requests rotate **provider → account → model**, giving eligible providers equal starting turns, accounts equal turns within each provider, and models equal turns within each account. The initial provider order is Gemini, Groq, OpenRouter, Cloudflare, Cohere. After a failed attempt, fallback remains **next model → next account → next provider**. A requested configured model starts first. Automatic model fallback is enabled by default; use `allow_model_fallback=False` or CLI `--exact-model` to preserve an exact model. `fallback_models=[...]` / `--fallback-model` narrows the pool. Explicit provider restrictions always apply, so `--provider groq` never calls Gemini or another provider. Without `--provider`, exhausted providers can fall through to the next one.

The CLI stores only rotation positions in `.ntg-routing.json` beside the env file, with a process lock for simultaneous commands. It contains no credentials, messages or quota counters and is ignored by Git. API users can share this behavior using `routing_state_path=".ntg-routing.json"`; otherwise a single router keeps its rotation in memory. Explicit model/provider selections constrain balancing. Failed or unavailable routes can make actual completed-call counts uneven.

The default attempt budget covers all selected routes within the overall deadline. Use `--attempts` for a smaller cap or `--deadline` to change the default 60-second deadline. Transport timeouts are capped at 15 seconds per attempt. Slow requests may reach the deadline before exhausting the pool.

The router validates its public parameter names and text-only message format. Model availability, parameter support, and context/output limits are validated by the provider API, rather than a potentially stale local catalog. Parameters are never silently removed and messages are never truncated. No output-token cap is sent by default; providers use their own defaults and model limits. Pass `--max-tokens` or `max_tokens` only when you want an explicit cap. `input_tokens` is retained for custom capability checks but the default adapter does not prefilter by token estimates. Tools must be caller-supplied function tools.

For streaming, pass `stream=True`, iterate `result.response`, and **always close it** in `finally` with `await result.response.aclose()`. The first chunk is fetched within the fallback cycle; after a chunk is returned there is no replay. In-flight reservations last through consumption/close. Latency for streams is time to first chunk; subsequent chunks share the same overall deadline.

Availability reports `eligible`, `unknown` (configured but untried), `temporarily_blocked`, or `disabled`. Reset times are Unix timestamps. Retry-After, reset headers and structured RetryInfo are used when supplied. Missing short-term resets use bounded jittered backoff; daily/monthly exhaustion without a reset suspends the scope for this session. Invalid keys need reload. Failures contain sanitized categories and HTTP status, never raw exception text or credentials. No remaining quota is calculated or displayed. Restart learns availability again through ordinary requests. Provider availability and quota depend on the account and service.

Validation: `python -m unittest discover -s tests -v` covers hierarchical fallback, per-model and shared quota scopes, exact/provider restrictions, more than six attempts, deadline cleanup, cooldown expiry, and stream failure without replay. All 24 configured keys retrieved model catalogs successfully. Catalog evidence alone is not a live completion or remaining-quota guarantee.

Official references checked 2026-10-13:

- [Gemini billing](https://ai.google.dev/gemini-api/docs/billing/) and [models](https://ai.google.dev/gemini-api/docs/models)
- [Groq Free plan billing](https://console.groq.com/docs/billing-faqs), [models](https://console.groq.com/docs/models), [rate-limit headers](https://console.groq.com/docs/rate-limits)
- [OpenRouter free variants](https://openrouter.ai/docs/guides/routing/model-variants/free)
- [Cloudflare pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) and [FP8 model profile](https://developers.cloudflare.com/workers-ai/models/llama-3.1-8b-instruct-fp8/)
- [Cohere pricing/trial restrictions](https://cohere.com/pricing) and [models](https://docs.cohere.com/docs/models)
- [LiteLLM Cohere v2](https://docs.litellm.ai/docs/providers/cohere) and [Cloudflare](https://docs.litellm.ai/docs/providers/cloudflare_workers)
