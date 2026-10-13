# NTG smart router

Python 3.11+. This rebuilds the existing `ntg.router.engine.UnifiedNTGRouter`; it does not introduce a second router. The old implementation depended on missing modules and persisted quota estimates. No database, rate-limit tables, usage counters, or availability probes are used here.

```powershell
python -m pip install -r requirements.txt
python -m ntg --status
python -m ntg "Explain async Python briefly" --provider groq --model llama-3.3-70b-versatile
python -m ntg "Hello" --provider openrouter --model openrouter/free
```

The local `.env` contains the 24 supplied credentials; `.env.example` contains placeholders. Both `.env` and the original credential text are ignored by Git. Do not commit either. Numbered keys can be added without code changes. Environment variables override `.env`.

**Before inference**, confirm the account in its provider console and set each `PROVIDER_N_FREE_PLAN`: Gemini `free_project` (billing disabled), Groq `free_plan`, Cloudflare `workers_free` (no paid Workers plan or AI Gateway billing), Cohere `trial_noncommercial`, OpenRouter `free_models`. Defaults are `unknown`, which excludes the route. The supplied text's “free tier” labels are not billing verification. Cohere trial keys are not for production/commercial use. Groq is distinct from xAI Grok; no Grok allowance is assumed.

Set `PROVIDER_N_MODELS` to comma-separated **native model identifiers** whose free access you have verified in current official documentation and your account. Suggested identifiers in the example are supported interface examples, not proof of eligibility. Cloudflare's current example is `@cf/meta/llama-3.1-8b-instruct-fp8`; its earlier non-FP8 identifier is unavailable in the current catalog. OpenRouter is configured locally with `openrouter/free`, which automatically selects an available free model. Explicit `:free` variants are also supported. The free router has a local 4,096 output-token policy ceiling; selected providers still validate their own limits. Every OpenRouter attempt checks its public model metadata for zero prices and sets zero prompt/completion price ceilings, disables upstream provider fallback, and requires parameter support. Paid variants, paid routers such as `openrouter/auto`, BYOK and plugins are not enabled. Metadata retrieval is not an inference probe and stores no quota state.

Model APIs generally cannot prove the account's billing plan. Strict free-only enforcement therefore depends on the explicit plan declaration and provider-side billing controls. Unverified accounts and models remain excluded; changing a plan outside this program invalidates its declaration. Never mark a paid plan as free merely because it includes an allowance. Model discovery and successful calls do not establish free eligibility.

Use `PROVIDER_N_QUOTA_GROUP` to assign the same stable project/organization/team identifier to keys sharing limits. Cloudflare keys are automatically grouped by their account ID. When a provider exposes a model-specific quota, that group/model is blocked; otherwise the shared group is conservatively blocked. Incorrect grouping can cause unnecessary failures. These identifiers are configuration, not quota tables.

```python
import asyncio
from ntg.router import UnifiedNTGRouter, RoutingError

async def main():
    router = UnifiedNTGRouter()
    try:
        result = await router.acompletion(
            [{"role": "user", "content": "Hello"}],
            providers=["groq"], model="llama-3.3-70b-versatile",
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

Keep one router per event loop for fair distribution and shared operational state. CLI invocations start fresh. Explicit models are preserved across accounts. To permit model changes use `allow_model_fallback=True, fallback_models=[...]` or CLI `--fallback-model MODEL` (repeatable). With no model, all configured compatible routes participate from the first request. Provider restrictions always apply. A qualified model uses the public provider name, e.g. `cohere/command-a-03-2025`; transport uses LiteLLM's `cohere_chat/` prefix internally.

Supported caller parameters are validated, never silently removed. Tools/schema support and context/output limits use LiteLLM's pinned bundled metadata, request-time OpenRouter model API metadata, or the documented Cloudflare FP8 profile (32,000 context, a conservative 4,096 output policy ceiling). Unknown capabilities are excluded. Multimodal inputs are currently excluded. Text token counting is an estimate; `input_tokens` can supply a stricter caller count. Provider context errors remain authoritative. No messages are truncated. Default output limit is 256; pass an explicit limit when needed. Tools must be caller-supplied function tools, not provider paid built-ins.

For streaming, pass `stream=True`, iterate `result.response`, and **always close it** in `finally` with `await result.response.aclose()`. The first chunk is fetched within the fallback cycle; after a chunk is returned there is no replay. In-flight reservations last through consumption/close. Latency for streams is time to first chunk; subsequent chunks share the same overall deadline.

Availability reports `eligible`, `unknown` (configured but untried), `temporarily_blocked`, or `disabled`. Reset times are Unix timestamps. Retry-After, reset headers and structured RetryInfo are used when supplied. Missing short-term resets use bounded jittered backoff; daily/monthly exhaustion without a reset suspends the scope for this session. Invalid keys need reload. Failures contain sanitized categories and HTTP status, never raw exception text or credentials. No remaining quota is calculated or displayed. Restart learns availability again through ordinary requests. Free service availability and complete allowance consumption are not guaranteed.

Validation: temporary inline simulations passed sequential/concurrent distribution, unequal model pools, bounded fallback, shared quota scopes, exhaustion/manual re-enable, request deadlines, stream closure and no replay. A mocked real LiteLLM Cloudflare call preserved HTTP 429/Retry-After and made exactly one request. Syntax and whitespace checks passed. No live inference calls or test files were used.

Official references checked 2026-10-13:

- [Gemini billing](https://ai.google.dev/gemini-api/docs/billing/) and [models](https://ai.google.dev/gemini-api/docs/models)
- [Groq Free plan billing](https://console.groq.com/docs/billing-faqs), [models](https://console.groq.com/docs/models), [rate-limit headers](https://console.groq.com/docs/rate-limits)
- [OpenRouter free variants](https://openrouter.ai/docs/guides/routing/model-variants/free)
- [Cloudflare pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) and [FP8 model profile](https://developers.cloudflare.com/workers-ai/models/llama-3.1-8b-instruct-fp8/)
- [Cohere pricing/trial restrictions](https://cohere.com/pricing) and [models](https://docs.cohere.com/docs/models)
- [LiteLLM Cohere v2](https://docs.litellm.ai/docs/providers/cohere) and [Cloudflare](https://docs.litellm.ai/docs/providers/cloudflare_workers)
