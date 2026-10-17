# NTG routing and coding agents

A Python interface and local OpenAI-compatible chat gateway built on LiteLLM. It filters free-use evidence, capabilities and context before routing; balances accounts; coordinates bounded concurrency; and applies scoped cooldowns and bounded fallback. The system plans coding tasks and generates, compiles, executes, repairs and reviews a cumulative Python application inside task workspaces.

See [guide.md](guide.md) for the Milestone 12 implementation contract and validation evidence.

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
# Keep the existing .env. For a new installation only:
# Copy-Item .env.example .env
.\.venv\Scripts\python -m ntg discover
.\.venv\Scripts\python -m ntg smoke
.\.venv\Scripts\python LLMCALL.py
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

Structured output is the system default for Python and gateway requests. Without
a response format (or with `null`), completions contain `{"answer": "..."}`.
The default schema lives in `ntg/types.py`. The router selects models advertising
schema support, validates responses, and falls back on invalid output.
Tool-free agents parse this JSON into an `AgentResult`. Tool-enabled agents use
a system-defined JSON action schema to request filesystem operations, then return
a final answer. Neither schema needs to be defined by the caller.
Supply a custom `response_format` to require different fields, or explicitly use
`{"type": "text"}` for plain text. The OpenAI-compatible response envelope stays
unchanged; the JSON object is encoded in `choices[0].message.content`.

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

## Coding workflow and task graphs (Milestones 6–12)

Run `python LLMCALL.py` in the activated virtual environment. It builds a standard-library expense-tracker CLI in app.py using Planner → dependency scheduler → Coder → compile → Tester → Reviewer. Ordinary Agent and ReActAgent remain available for structured chat and bounded tool use.

Plans contain one through six task dictionaries with id, description and dependencies. IDs remain the contiguous set T1 through Tn, but tasks may appear in any array order. Duplicate IDs or dependencies, unknown prerequisites, self-dependencies and cycles are rejected before coding. The Planner has no tools.

TaskGraph is a small deterministic Python scheduler. It selects pending tasks with all prerequisites passed; PlannedWorkflow runs the lowest-numbered ready task and recomputes readiness. Execution is sequential. For T1 → {T2,T3} → T4, execution is T1,T2,T3,T4. No framework, extra scheduler agent, parallel workers or branch merging is introduced.

The artifact is cumulative app.py: T3 extends T2's approved snapshot, and T4 receives both siblings' features. Each task receives the original goal, current task, direct dependencies, all completed tasks and latest approved source/hash. Intermediate tasks need implement only current requirements while preserving completed work; the final executed task must satisfy the entire goal. Dependencies govern readiness rather than limiting source context.

Each task has its own workspace. Coder initially supplies complete source; repairs use host-issued source anchors against the actual current file. Repairs receive exact execution evidence, current source and previous feedback, and must make the smallest change. Stale, overlapping, overly broad, empty and newline-damaging edits are rejected. Existing filesystem write_file remains create-only; host-controlled replacements use expected source hashes and atomic replacement.

Tester runs python -m py_compile before functional execution. Compile failures skip functional checks. Host-defined checks require exit codes and stdout fragments; the LLMCALL example checks no-argument help and --help. Both must mention add, list and total. Tester feedback uses actual stderr/stdout, not an LLM diagnosis. An optional subparser alone does not print help: a missing-help failure points to parse_args() and the no-command branch. Configured smoke checks are not exhaustive proof of expense-tracker correctness.

Reviewer inspects the tested source and returns approved/issues JSON. Every issue blocks approval. Verified completion requires matching workspace and source identities across execution and review. Passing tests and completed features are preserved during repair; regressions and ineffective repeated failures trigger rollback. At most two repairs follow an initial candidate. No scheduler-level retries or automatic replanning are added.

Execution is local under your user account, without Docker or a sandbox. Workspace path guards constrain filesystem tools but are not process isolation. Tester limits direct processes to 20 seconds and captures at most 32 KiB per output stream. Source is limited to 64 KiB. Planner has 60 seconds, each CodingWorkflow 300 seconds, and the complete planned run 1,200 seconds. Router owns bounded provider retries.

Task graph states are pending, running, passed, failed, blocked and cancelled. A task passes only after verified tests/review and fresh source-hash validation. The first failure stops dispatch: unexecuted descendants become blocked, and independent pending work becomes cancelled. External cancellation propagates after cleanup. An unverified run exits nonzero and never advertises a final successful project.

LLMCALL prints elapsed liveness every 15 seconds while awaiting the workflow, then plan declarations, actual execution_order, all graph states and reasons, revision evidence and artifact locations. Run metadata includes plan.json, task_<id>.json for attempted tasks, and result.json with task_graph for every planned task. Plan hashing retains declaration order; actual execution order is separate. Task workspaces contain immutable revision source snapshots, diffs, tester evidence, raw captures and review reports. These can contain generated source and task data.

Validation used inline harnesses in temporary workspaces without adding permanent tests/examples: unordered diamond, chain and independent roots; invalid graphs and state transitions; real compilation/execution with deterministic generation/review; cumulative sibling preservation; failure propagation; cancellation and deadlines; result identity/tampering; and metadata failures. Live provider outcomes are reported separately from deterministic validation because quotas, availability and generated plan/code quality vary.

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

Milestone 12 validation criteria and implementation details are recorded in
[guide.md](guide.md). Live runs use the configured free provider pool; availability
and quotas can change between requests.
