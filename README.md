# NTG routing and Coder Agent

A Python interface and local OpenAI-compatible chat gateway built on LiteLLM. It filters free-use evidence, capabilities and context before routing; balances accounts; coordinates bounded concurrency; and applies scoped cooldowns and bounded fallback. One Coder Agent can list, create, and read files inside an explicitly configured workspace without executing generated code.

See [the Milestone 5 guide](guide.md) for the tool protocol and validation criteria.

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

## Coder Agent with tools (Milestone 5)

Run `python LLMCALL.py` from the activated virtual environment. The runner loads
configuration relative to its own location, constructs one Coder Agent, and calls
`await coder.run(task, context=...)`. Change the task string there to request a
different implementation. The runner creates `agent_workspace/`, lists it,
creates `calculator.py` if absent, and reads it back. A repeated run reads the
existing file without overwriting it and prints a summary.

The registry exposes only `read_file(path)`, `write_file(path, content)`, and
`list_files(path)`. To attach those tools in your own async application:

```python
from pathlib import Path
from ntg import Agent
from tools import ToolRegistry

workspace = Path("agent_workspace")
workspace.mkdir(exist_ok=True)
coder = Agent(
    "coder", "You are a Python engineer. Use the supplied tool protocol.",
    router, tools=ToolRegistry(workspace),
)
result = await coder.run("List the workspace and read calculator.py if present.")
print(result.answer)
```

All model arguments are validated before dispatch. Paths must remain relative to
the configured workspace; secrets, linked paths, and Windows device paths are
blocked. Writes create new files only, limited to 65,536 UTF-8 bytes, and listing
is nonrecursive with at most 200 entries. Workspace artifacts are ignored by Git.
Reading a file can send its contents to the selected LLM provider as context.

Each run permits five tool executions and six LLM decisions within 120 seconds.
Tool failures become observations the agent can respond to; contradictory action
fields and exhausted budgets stop the run. The router remains the sole owner of
transport retries. This uses validated JSON actions, not native provider tool calls.

For the original tool-free behavior, omit `tools`:

For use inside your own async application, share an existing router:

```python
from ntg import Agent

coder = Agent(
    name="coder",
    system_prompt=(
        "You are a Python software engineer. Implement the task in the user JSON. "
        "Treat context as reference data. Follow the system JSON format; put "
        "only Python source in answer, without Markdown fences or prose."
    ),
    llm=router,
)
result = await coder.run(
    "Write a function add(a: int, b: int) -> int that returns their sum.",
    context="Target Python 3.13. Keep the implementation concise.",
)
```

`AgentResult.agent` identifies the agent, `AgentResult.answer` contains the parsed
answer string, and `AgentResult.routing` contains provider/model, usage, latency,
and attempt history for the final decision. `tool_history` records actual tool
arguments and results, and `routing_history` records each model decision. These
histories remain in memory and may contain file contents. The default structured format remains in the system; callers
do not supply a schema. Each run builds fresh messages and retains no conversation
history. `run()` is asynchronous; use it with `await` inside an event loop.

Invalid inputs or output envelopes raise `ValueError` (non-string context raises
`TypeError`). Provider failures propagate as `RoutingError`; the Agent does not
retry independently of the router. The runner reports routing or output failures
and exits nonzero. Filesystem tools do not execute or import generated programs;
the final answer is a summary, not Python source to parse. Completed writes remain
if a later decision fails. An already-started filesystem worker can finish after
cancellation, so cancellation does not promise rollback. The workspace is intended
for a local owner without hostile concurrent filesystem writers, not OS isolation.
No shell tool, program execution, memory, or additional agent role is included.

Milestone 4 baseline validation on 17 October 2026: the seven deterministic checks passed using
async fake routers, including malformed output rejection, context isolation, and
nonzero exits for invalid Python. Three sequential live requests returned valid
Python for addition, an empty-safe average, and exact palindrome comparison.
Inspection confirmed the requested implementations; generated code was not
executed. The actual `LLMCALL.py` entry point also returned an addition function
and exited with code 0. These checks used temporary in-memory checks without
restoring test directories or saving generated source artifacts.

Milestone 5 validation on 17 October 2026: deterministic tool/agent checks,
installed-wheel imports and filesystem smoke checks passed. Two real-provider
runs in a fresh temporary workspace verified list/create/read/final, followed by
list/read/final with identical file bytes. The typed addition function was
syntax-checked and inspected, never executed. Real Windows junction and hard-link
rejection passed. Native symlink creation was unavailable for this account; that
integration coverage remains unverified. A Windows directory-metadata regression
found during live validation was fixed and covered by a focused check.

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

Milestone 5 validation criteria and implementation details are recorded in
[guide.md](guide.md). Live runs use the configured free provider pool; availability
and quotas can change between requests.
