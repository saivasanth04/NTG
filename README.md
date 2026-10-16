# NTG model routing foundation

A Python interface and local OpenAI-compatible chat gateway built on LiteLLM. It filters free-use evidence, capabilities and context before routing; balances accounts; coordinates bounded concurrency; and applies scoped cooldowns and bounded fallback. It contains no agents or tool execution.

Read [feasibility and all feature decisions](docs/feasibility.md), [architecture](docs/architecture.md), and [validation results](docs/validation.md).

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e '.[test]'
# Keep the existing .env. For a new installation only:
# Copy-Item .env.example .env
.\.venv\Scripts\python -m ntg discover
.\.venv\Scripts\python -m ntg smoke
.\.venv\Scripts\python -m pytest -q
```

Discovery writes secret-free metadata to ignored `runtime/catalog.json`; it makes metadata requests, not paid inference probes. Smoke sends one small chat request with up to three attempts, and prints only routing/usage metadata. For POSIX use `.venv/bin/python`.

## Configure free access safely

`PROVIDER_API_KEY_1`, `_2`, ... are detected without a fixed count for GEMINI, GROQ, OPENROUTER, CLOUDFLARE and COHERE. Unnumbered `PROVIDER_API_KEY` is also accepted as slot 1; configure one spelling per slot. Each account may have `PROVIDER_1_MODELS` (comma-separated allowlist), `PROVIDER_1_QUOTA_GROUP` and `PROVIDER_1_FREE_VERIFIED`. Cloudflare also requires `CLOUDFLARE_ACCOUNT_ID_1`. Environment variables override `.env` values. Values are never exported into provider-global environment state. See [.env.example](.env.example).

Use the **same quota group** for keys that share project/organization/owner limits. If unsure, omit groups; the default shares provider capacity conservatively. Existing supplied quota groups are respected; the software cannot verify that they are truly independent.

OpenRouter's explicit `:free` models are automatically eligible only when live metadata reports zero for every pricing field; the transport also requests zero-price upstream routing. `openrouter/free` is excluded because its downstream model/capabilities are opaque. Listing, successful authentication and zero account balance do not establish other providers' free eligibility.

Other providers stay blocked for inference until both of these are true:

1. Verify in the provider dashboard that the key belongs to a free-only / evaluation plan with billing disabled and that your use is allowed. Set that account's `PROVIDER_1_FREE_VERIFIED=true`.
2. Add a documented model entry to `policy.json` with `source`, `reviewed_on`, `capabilities`, `context`, `max_output` and `free_eligible`. See [policy.example.json](policy.example.json); recheck the linked model documentation before enabling it. An API catalog cannot verify the operator's attestation.

Model policy evidence must be reviewed within 30 days. Expired entries cannot broaden eligibility or capabilities. Unknown pricing or stale model evidence fails closed.

Do not enable paid keys merely because they have trial credits, an included allowance, or a consumer subscription. Configure provider-side spending controls as well. `.env` is ignored; never commit it. This project does not edit existing credentials or account plans.

## Python interface

```python
import asyncio
from ntg import RoutingService, ChatRequest, RoutingError

async def main():
    router = await RoutingService.from_env()  # create once and share among agents
    try:
        response = await router.chat(ChatRequest(
            messages=[{"role": "user", "content": "Write a Python add function."}],
            max_tokens=256,
            deadline_seconds=45,
            # tools=[...], tool_choice="auto",
            # response_format={"type": "json_schema", "json_schema": {...}},
        ))
        print(response["choices"][0]["message"]["content"])
        print(response["routing"])
    except RoutingError as exc:
        print(exc.public())  # sanitized error and fallback history

asyncio.run(main())
```

`model="auto"` selects the compatible pool. A raw provider model ID constrains fallback to that ID across accounts; an ID from `/v1/models` (`account-alias/model-id`) pins the deployment. Required tools and JSON/schema capabilities are inferred from request fields. `required_capabilities` can impose additional constraints. `input_tokens` can increase the conservative local context bound. UTF-8 byte sizing intentionally underuses context to avoid optimistic model-independent token estimates. Streaming, arbitrary provider parameters and non-chat operations return errors or are absent; this is a documented chat-completions subset.

## Gateway

Set a random secret in your current shell (or privately add it to `.env`):

```powershell
$env:NTG_GATEWAY_KEY = & .\.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))"
.\.venv\Scripts\python -m ntg serve
```

The server listens at `127.0.0.1:8000`, one worker. Authenticated routes: `POST /v1/chat/completions`, `GET /v1/models`, `GET /status`. `/healthz` is public process health. Access logs, content logs and automatic API docs are disabled. Do not enable debug logging in application dependencies. Prompts still go to the selected provider under its data policy.

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                api_key=os.environ["NTG_GATEWAY_KEY"], max_retries=0, timeout=60)
result = client.chat.completions.create(model="auto", messages=[
    {"role": "user", "content": "Explain Python generators."}], max_tokens=256)
print(result.choices[0].message.content)
```

Disable client retries to keep a single retry owner. An OpenAI SDK timeout should exceed `deadline_seconds` (default 45s). Custom deadline/capability fields can be passed via SDK `extra_body`. Bodies are limited to 1 MiB. Put TLS/authenticated network controls in front of any remote deployment. One gateway secret grants all local routes; tenant isolation and virtual keys are deferred.

## Operations and extension

Defaults: four active attempts, sixteen waiting callers, one concurrent attempt per quota group, three attempts per request, 20s per attempt, 45s overall request deadline. Configure these through `RoutingService(...)` / `from_env(...)` and inject the service into `create_app(service, token)`. Scope-specific cooldowns and Retry-After/reset handling do not assume fixed provider quotas. Unknown quota allows conservative probing, not an assertion of spare capacity.

Catalog evidence expires after 24h. Restart a long-running service to refresh discovery; the CLI discovery report does not hot-reload the running service. Avoid frequent refreshes. Runtime state is in memory and lost on restart. Single-process coordination is intentional: do not run multiple workers or replicas without shared atomic state. The queue is bounded admission with fair deployment selection, not a durable FIFO job queue.

Responses include provider, model, account alias, total latency, usage and attempt history. `cost_usd` remains unknown unless a trustworthy provider value is present; no invoice verification is performed. Status exposes safe model metadata, counters, cooldowns and latency EWMA. No prompt/response caching or content logging is configured.

Run the optional bounded live validation with:

```powershell
.\.venv\Scripts\python scripts\validate_live.py
```

It submits chat/tools/schema probes plus two concurrent chat callers (up to three attempts each) and saves metadata to `runtime/live-validation.json`. It does not execute returned tools. Controlled failure tests need no API keys. See [architecture extensions](docs/architecture.md) for embeddings, images and eventual distributed gateway migration.
