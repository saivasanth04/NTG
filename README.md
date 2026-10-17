# Adaptive project execution

`python LLMCALL.py` now uses ProjectWorkflow: the existing Planner, Coder, Tester, Reviewer and optional Researcher operate on a tracked multi-file project. The previous single-file PlannedWorkflow API is retained through `python LLMCALL.py --legacy`.

```text
requirements -> small plan -> ready task / component expansion
    -> model-budgeted packet -> source read/search/edit steps
    -> compile all tracked Python -> ordered acceptance/regression checks
    -> current-source review -> approved project manifest
    -> integration + final requirement review -> focused repair or success
```

## Run a project

```powershell
python LLMCALL.py
python LLMCALL.py --goal-file goal.md --checks-file acceptance.json --repeatable-checks --max-calls 2000 --run-timeout 7200
python LLMCALL.py --resume project_<saved_id> --goal-file goal.md --checks-file acceptance.json --repeatable-checks --max-calls 2000 --run-timeout 7200
python LLMCALL.py --legacy
```

The goal file and acceptance file are user inputs; NTG does not create those example files. `--goal` accepts a short goal directly. A checks file is a JSON list using `{"name":"help","arguments":["--help"],"exit_code":0,"stdout_contains":["usage:"]}` per entry. Arguments contain only app.py argument tokens, never a shell command. In project mode `exit_code: -1` means any nonzero exit. Host-provided checks remain authoritative; project tasks add acceptance checks without replacing earlier obligations. A full CLI invocation accidentally put in a check name is normalized only when unambiguous, and contradictory exit expectations are rejected before dispatch.

Python standard-library projects with an app.py entrypoint are supported. Component scopes may include validated relative directories ending in `/`; expansion assigns stable IDs such as T2.1. Planner-proposed helper-file scope amendments are validated and recorded; direct coder writes outside its declared scope are rejected. Source files and mutable runtime data (`runtime_files`, such as expenses.json) are declared separately. Every component must pass its own integration barrier after its children pass.

## Context and source handling

Host artifacts remain streamed and have no application byte ceiling. Project source can exceed the old 64 KiB single-file bound: models read bounded exact line ranges, search tracked files and page through file listings. Host patches check the current whole-file SHA-256 and replace only declared ranges atomically. New files are create-only, small initial scaffolds can be extended, and stale source is refreshed before each role request. Reviewer approval/rejection must refer to current inspected files; approval requires reading all changed line ranges. Final review must inspect deferred requirement ranges. Workspace guards are retained; local Python execution remains the existing unsandboxed execution model.

Each model candidate has distinct known input, combined-context and output limits. Budget calculation uses the same prepared packet as transport, including JSON fallback instructions, tools and schemas. Already cached/local tokenizers or an injected model counter are used when available; uncertain counters have explicit provenance and margins. Admission never downloads Hugging Face tokenizers; a missing local tokenizer uses a conservative byte estimate without disabling the account. Optional trusted model policy `tokenizer_file` points to a local tokenizer JSON. Actual context errors adapt the request and do not cool down the account. Truncated output is rejected. Existing provider/account health fallback is preserved.

The host rebuilds context between hops rather than accumulating the conversation. Requirements are archived exactly with IDs/hashes, ingested in bounded passes, and split further when planning cannot fit. Tasks can request missing source, requirements, dependency contracts or saved evidence. Read operations against `artifact:<name>` use character offsets to inspect full evidence without flooding prompts. Repeated operations, code repairs, model calls, task expansions and run deadlines have separate budgets. Hitting a run budget persists a checkpoint rather than losing approved work.

## Verification and checkpoint behavior

Library-only leaves can receive explicit compile-only coverage until the CLI exists; final entrypoint and integration gates remain mandatory. Tests compile tracked Python files and execute the entire ordered obligation sequence in a disposable filesystem copy, preventing mutable data from contaminating later runs. Immutable source changes and untracked Python files are rejected. Task approvals bind test/review evidence to a sorted file/hash manifest and project revision. Final integration and requirement reviews can create targeted repair tasks; passing individual startup checks alone does not complete the project.

Schema-2 `project_state.json` references requirement, routing, revision, test and review artifacts. `result.json` contains `manifest`, `project_revision`, execution order and verification. Resume checks task definitions, approval/test artifact hashes, source hashes and configuration. Completed projects do not repeat model calls. Interrupted candidates are discarded in favor of approved source unless a complete, hash-verified passing test record matches the candidate. Such a candidate resumes through fresh testing and review, preserving its changed-line inspection obligations. Test/integration evidence identifies its project revision so old failures are not presented as current results. Automatic repetition of interrupted checks requires the caller's repeatability declaration; the default disposable expense-tracker demo declares its own checks repeatable. Schema-1 legacy runs remain inspectable through the old path and are not silently promoted into resumable projects.

Operational call/time limits can be changed through CLI options; parent expansion defaults to 512 and can be changed with `--max-expansions`. These are execution budgets, not storage limits or promises of model competence. A synthetic 1000-task scheduling test is not evidence that 1000 generated tasks complete successfully. General large-project quality, cost and latency require workload-specific evaluation.

For the approved design and detailed implementation/validation record, see [FEASIBILITY.md](FEASIBILITY.md) and [the implementation plan](plan/feature-adaptive-project-workflow-1.md).

---

## Repository intelligence (Milestone 15)

Project mode derives a local Python AST index from the active tracked project. It extracts classes, nested/async functions, imports and calls, with source hashes and line ranges. Local imports and unambiguous lexical calls receive resolved relationships; dynamic calls, shadowed names, wildcard imports and ambiguous targets remain explicitly unresolved. The index never imports or executes source.

Coder and Reviewer can issue `repo_search` (`text` is a literal query; optional `path` filters a tracked file) and `repo_dependencies` (`path` selects a tracked file; `text` is null). Both use `start: 0` initially, then `next_offset`; unused `end`/`sha256` fields are null. Dependency pages label incoming/outgoing imports/calls. Results identify the project revision and source hashes. A source change invalidates pagination; restart at zero.

Index metadata guides existing exact `read` operations; it does not satisfy source inspection or approve a change. Query observations are bounded by current context budgets and persisted as `repository_query` evidence. The host refreshes candidate metadata after edits, reuses unchanged per-file extraction, and keeps approved index references separate from active candidate snapshots. Invalid/missing optional cache artifacts rebuild without weakening approval/test/source checks. Existing schema-2 checkpoints remain compatible.

`RepositoryIndex` is also exported from `ntg` for host use: construct it, call `refresh(project_files)`, then `search("name")` or `dependencies("module.py")`. JSON snapshots live in host metadata; no database or extra dependency is required. Only tracked project files are indexed; runtime data, credentials, sibling workspaces and the NTG installation are excluded. A separate host-controlled import policy is still needed to onboard an arbitrary checkout.

Static analysis is conservative, not a runtime call graph or correctness proof. Python parsing needs memory for one file; errors are recorded with partial coverage. Existing compile/test/integration/review gates still decide completion. See [guide.md](guide.md) for the implementation requirements and verification record.

## Unified cross-provider routing

RoutingService is the global logical pool; LiteLLMTransport uses litellm.acompletion for one selected deployment and disables SDK retries. This project does not instantiate a separate LiteLLM Router with provider-local aliases. Agent requests use model="auto". Explicit raw model/deployment IDs still intentionally restrict selection for callers who request a pin.

Pricing approval is no longer a routing prerequisite, as explicitly requested by the operator. All configured accounts and configured/catalog models join the global pool. Free-price metadata is informational; no FREE_VERIFIED flag, reviewed free-access attestation, catalog age, or zero-price OpenRouter ceiling is required. Configured model lists are preferences rather than allowlists.

The corrected request flow is:

```text
role preference + mandatory capabilities
    -> all configured deployment records
    -> protocol / known context / output / explicit-pin checks
    -> record exclusions and runtime cooldown skips
    -> score healthy compatible candidates, spreading fallback across providers
    -> reserve account/owner-group capacity
    -> one LiteLLM attempt
    -> validate output or classify failure and apply scoped cooldown
    -> next untried compatible deployment, preferably another provider
    -> success, genuine request deadline/cancellation, or exhausted global pool
```

Priority (lower first), optional preferred_provider/preferred_model, advisory model_class, recent health, provider/account rotation and measured latency rank compatible candidates. Preferences never lock an agent to a provider. Planner/Reviewer prefer reasoning, Coder and repairs prefer coding, Researcher prefers fast, and ordinary chat/ReAct prefer general. These are quality preferences, not invented capability guarantees. Explicit protocol capabilities, tools and known context/output limits remain compatibility checks. Unknown context limits permit runtime attempts. JSON/schema output can use prompted JSON with mandatory host validation when native support is not advertised. Tester executes Python locally and has no provider assignment.

Optional model policy entries may specify priority, model_classes, capabilities and known context/output limits without pricing attestations or review-date approval gates.

Each unique compatible deployment is attempted at most once per request. The legacy max_attempts setting no longer truncates coverage of a larger compatible pool; the effective traversal can expand to the pool size. Existing admission, per-owner concurrency, per-attempt timeout and caller/agent deadlines remain hard limits. Request deadline or queue-full errors do not claim that every provider has exhausted quota.

History distinguishes attempted, skipped, excluded, provider_exhausted and pool_exhausted events. It records provider/account/model, safe outcome/status, category, scope and latency; skips also show evidence origin and remaining cooldown. No prompts, headers containing credentials, keys or raw upstream exception bodies are logged. Categories include ACCOUNT_EXHAUSTED, PROVIDER_EXHAUSTED, MODEL_UNAVAILABLE, TEMPORARY_FAILURE, GLOBAL_POOL_EXHAUSTED, INELIGIBLE and INVALID_REQUEST.

Known quota observations seed short account cooldowns, not a guessed daily-reset counter. Retry-After and reset headers take precedence (relative durations and epoch-second/millisecond reset timestamps are normalized); absent reset information uses a short recheck interval. Temporary provider failures use short provider cooldowns; account/model failures stay scoped. Owner quota-group sharing remains enforced so duplicate/shared keys do not manufacture capacity. Last failure and current cooldown/availability are inspectable through status(). Cooldowns expire; an inaccessible deployment is not permanently filtered out.

quota_exhausted is returned only when all compatible candidates are known quota-exhausted. Mixed rate limits, unavailable models or outages produce global_pool_exhausted with evidence. Truly malformed requests remain terminal; they are not replayed across providers.

Deterministic validation used the actual RoutingService, PlannerAgent, PlannedWorkflow, Coder, compile/Tester/Reviewer and ProjectState with an injected transport. Scenario: five OpenRouter accounts had observed quota exhaustion, Groq account 1 returned 429, and a Gemini account succeeded. The Planner history recorded all five quota skips, provider exhaustion, Groq's actual rate-limit attempt and Gemini success. Orchestration continued to verified completion even with max_attempts=1. Additional inline checks covered all-quota versus mixed outages, 401/402/403/404/429/5xx failover, one-provider account rotation, shared owner groups, cooldown expiry, capability exclusions, output validation and terminal malformed requests. An additional invocation of the actual LLMCALL.main entry function with that real global router and injected upstream responses exited 0, with matching passed project_state.json/result.json. Planner used Gemini after the Groq 429; Coder and Reviewer continued dynamically. LiteLLM transport translation was also checked for all five providers with the SDK call mocked. This historical check proved routing behavior using injected responses. Pricing gates described in earlier verification records have since been removed at the operator's request. No test files or infrastructure were added.

Pricing-gate removal verification (17 October 2026): a bounded live RoutingService request succeeded through Cloudflare account cloudflare-5, model @cf/meta/llama-3.2-1b-instruct. All five providers had eligible candidates without FREE_VERIFIED flags. Deterministic checks also verified cross-provider quota fallback, absent OpenRouter max_price, host JSON validation, configured/catalog model union, and isolation of free-model quota observations from paid models. All ntg modules compiled. This small live routing check does not claim a new complete LLMCALL workflow run.

## Shared project state (Milestone 14)

ProjectState is the per-run host owner of the goal, validated plan, one TaskGraph, task executions and their revision records, active stage/candidate, latest approved artifact and terminal outcome. Existing reports are composed rather than copied into a second history. Public result objects are detached terminal views. Separate runs own separate state; no database, resume mechanism or global state singleton is introduced.

Each metadata workspace now contains project_state.json, allocated before planning so planning failures are inspectable too. The versioned UTF-8 snapshot contains task states, progress, file/evidence references and hashes, agent/test/review/research indexes, errors and approvals. It excludes source bodies, raw pages, full conversations and routing objects. Raw execution captures, source revisions and existing reports remain separate artifacts. Relative file paths are paired with workspace IDs and content hashes.

Snapshots are updated after meaningful host transitions with guarded expected-hash replacement and monotonic state_revision. Host-owned state, plans, revision evidence and final results have no application byte-size ceiling. JSON is serialized incrementally, hashes are computed in chunks, and fsynced temporary files are published atomically; model filesystem tools retain create-only writes and 64 KiB files. Host artifact reads, writes and seals use the same workspace path protections. A conflict/write failure stops dispatch and returns nonzero; the previous complete snapshot remains readable. Atomicity is per file, not a transaction across state and result.json. Reloading JSON is inspection, not automatic resumption or renewed approval.

Pure context builders provide narrow copied views: Planner receives goal/research policy; Researcher receives current technical questions; Coder receives task requirements and verified cumulative source; repairs receive current source once, exact relevant active/rolled-back evidence, passing obligations and at most three compact failure-history entries; Reviewer receives current code, requirements and compact execution evidence. Tester receives host entrypoint/source identity and configured checks. No agent prompt receives the whole state or unrelated plan/report/routing histories.

Context limits are 24 KiB non-source metadata, 160 KiB exact relevant repair evidence, 64 KiB current source and the existing 8 KiB research payload. Required over-budget data is not silently truncated: dispatch blocks with context_budget_exceeded, and evidence remains in referenced artifacts. Prior source and duplicated stdout/stderr are removed from repair context while current file contents and actual failure evidence are retained.

LLMCALL.py reports stage, task, revision and derived completed/total counts during its existing liveness updates, then prints the project state path. Final state/result identities and verification must agree before the run is accepted as successful. See guide.md for implementation status and actual validation evidence.

# NTG routing and coding agents

A Python interface and local OpenAI-compatible chat gateway built on LiteLLM. It checks protocol compatibility and known context limits before routing; balances accounts; coordinates bounded concurrency; and applies scoped cooldowns and bounded fallback. The system plans coding tasks and generates, compiles, executes, repairs and reviews a cumulative Python application inside task workspaces.

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

Discovery writes secret-free metadata to ignored `runtime/catalog.json`; it makes metadata requests, not paid inference probes. Smoke sends one small chat request with deadline-bounded global fallback, and prints only routing/usage metadata. For POSIX use `.venv/bin/python`.

## Configure providers

`PROVIDER_API_KEY_1`, `_2`, ... are detected for GEMINI, GROQ, OPENROUTER, CLOUDFLARE and COHERE. Unnumbered keys are accepted as slot 1. Each account may have `PROVIDER_1_MODELS` (preferred models) and `PROVIDER_1_QUOTA_GROUP`. Cloudflare requires `CLOUDFLARE_ACCOUNT_ID_1`. Environment variables override `.env` values.

All configured accounts are enabled without free-price verification. Discovery includes configured and catalog models; configured models rank ahead of other catalog entries. Legacy FREE_VERIFIED and free_eligible values are informational and do not gate requests. OpenRouter requests have no max_price restriction. Free-model quota observations apply only to free deployments, never paid models on the same account. Actual authentication, quota and service errors still drive temporary cooldowns and fallback.

Use the same quota group for keys sharing project/organization limits. `.env` remains ignored; credentials are never logged or modified.

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

## Researcher Agent (Milestone 13)

Planner tasks now declare `research_questions` (zero through three short questions). Ordinary local tasks use `[]`; legacy three-field plans retain their existing hashes and skip research. A ready task with questions runs Researcher once before Coder. Coder receives compact source-linked findings; its repairs reuse the same evidence. Reviewer sees the reference evidence in its specification. Tests and source/review identity checks remain mandatory.

`ResearcherAgent(llm, ResearchTools(search_backend, allowed_hosts))` uses a host-injected async `search_backend(query, limit=3)` returning at most three `{"title": str, "url": str}` candidates. Import ResearchTools from `tools.research`. The backend can be an explicitly curated documentation lookup or an authorized search-service adapter; no undocumented search endpoint or provider is selected automatically. Use the same RoutingService instance when constructing agents in an application. ResearchTools is intentionally absent from the Coder/ReAct filesystem registry.

Configure exact official documentation hostnames in `allowed_hosts`. Retrieval permits public HTTPS on port 443, rejects credentials, all query strings and fragments, checks DNS addresses, and connects to a checked IP while retaining original-hostname TLS certificate verification. No proxy environment is used. Redirects are revalidated. The first implementation requests HTTP/1.0/identity encoding and rejects compressed, chunked, binary, oversized or unsupported responses; unsupported sites fail visibly rather than bypass limits.

Budgets per task: 60 seconds total; at most three searches and three document fetch attempts, ten seconds per retrieval request, 64 KiB per response body, 6,000 extracted characters per document and 12,000 total, one schema-constrained summary call with max_tokens 2048, and an 8 KiB compact evidence context. Model findings reference host-issued source IDs; URLs, UTC fetch times and content hashes come from retrieval. Raw pages are neither persisted nor passed to Coder. A citation establishes provenance, not independent proof of a claim.

ResearchResult validates question coverage, source references and size. Missing backend, invalid evidence, unanswered questions, timeouts or storage failures block coding, block descendants and cancel independent pending work. Research cannot write files, execute commands or change task approval. Cancellation propagates and closes active connections.

Pass `researcher=` to PlannedWorkflow, or to the asynchronous `LLMCALL.main(researcher=...)` entry function when embedding it. `python LLMCALL.py` remains a local expense-tracker demo with no search backend configured. Its Planner uses an explicit local-work policy requiring empty research questions; no requested evidence is discarded after planning. Providing a Researcher enables the research-capable plan schema. General PlannerAgent callers still allow research by default and required research without a backend remains blocked. Configure an authorized backend explicitly for research tasks; no extra package or credentials are added by this milestone.

Run metadata adds `research_<task_id>.json` for researched tasks. Task summaries record skipped status or the report, its relative artifact path and SHA-256 of the persisted UTF-8 JSON. Final verification checks plan/task/question identities, report hash and final stored report bytes. Graph execution_order includes tasks dispatched into research even when they never reach Coder; those tasks have no generated revisions.

Deterministic inline checks cover conditional research, schema/hash compatibility, fabricated-source rejection, evidence tampering, real compile/test/review handoff, failure propagation, byte/URL/redirect restrictions and cancellation. Live retrieval/model outcomes are recorded in guide.md separately from these deterministic checks. Current application scope remains standard-library single-file programs; the Stripe example does not add payment execution or production checkout integration.

## Coding workflow and task graphs (Milestones 6Ã¢â‚¬â€œ12)

Run `python LLMCALL.py` in the activated virtual environment. It builds a standard-library expense-tracker CLI in app.py using Planner Ã¢â€ â€™ dependency scheduler Ã¢â€ â€™ Coder Ã¢â€ â€™ compile Ã¢â€ â€™ Tester Ã¢â€ â€™ Reviewer. Ordinary Agent and ReActAgent remain available for structured chat and bounded tool use.

Plans contain one through six task dictionaries with id, description and dependencies. IDs remain the contiguous set T1 through Tn, but tasks may appear in any array order. Duplicate IDs or dependencies, unknown prerequisites, self-dependencies and cycles are rejected before coding. The Planner has no tools.

TaskGraph is a small deterministic Python scheduler. It selects pending tasks with all prerequisites passed; PlannedWorkflow runs the lowest-numbered ready task and recomputes readiness. Execution is sequential. For T1 Ã¢â€ â€™ {T2,T3} Ã¢â€ â€™ T4, execution is T1,T2,T3,T4. No framework, extra scheduler agent, parallel workers or branch merging is introduced.

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

Defaults: four active attempts, sixteen waiting callers, one concurrent attempt per quota group, legacy max_attempts=3 with full compatible-pool traversal, 20s per attempt, 45s overall request deadline. Configure these through `RoutingService(...)` / `from_env(...)` and inject the service into `create_app(service, token)`. Scope-specific cooldowns and Retry-After/reset handling do not assume fixed provider quotas. Unknown quota allows conservative probing, not an assertion of spare capacity.

Catalog evidence expires after 24h. Restart a long-running service to refresh discovery; the CLI discovery report does not hot-reload the running service. Avoid frequent refreshes. Runtime state is in memory and lost on restart. Single-process coordination is intentional: do not run multiple workers or replicas without shared atomic state. The queue is bounded admission with fair deployment selection, not a durable FIFO job queue.

Responses include provider, model, account alias, total latency, usage and attempt history. `cost_usd` remains unknown unless a trustworthy provider value is present; no invoice verification is performed. Status exposes safe model metadata, counters, cooldowns and latency EWMA. No prompt/response caching or content logging is configured.

Milestone 12 validation criteria and implementation details are recorded in
[guide.md](guide.md). Live runs use the configured provider pool; availability
and quotas can change between requests.

### Structured-output provider compatibility

Some OpenRouter upstream grammar compilers reject JSON Schema uniqueItems rather than ignoring unsupported constraints. Transport sends a copied schema with that keyword omitted for OpenRouter, while the router validates responses against the untouched full host schema. Duplicate research questions and citations are still rejected. Explicit grammar/unimplemented-key HTTP 400 errors are deployment capability failures eligible for bounded fallback; other invalid-request errors remain terminal. Host structured-output validation remains mandatory; pricing verification is no longer a routing gate.

Recovery verification on 17 October 2026: the real LLMCALL.py run exited 0 with all three planned tasks compiled, tested and approved. Result metadata is in workspaces/project_474108939ec54ac69d332ed29588e968; final code is workspaces/project_1d1b4f405fa74d8aa5402f6477dd4cb5/app.py. Additional temporary-copy checks verified add/list/total, persistence and invalid amount rejection. A separate live Researcher fetched official Python documentation and produced a verified source-linked summary on its first model attempt. See guide.md for the detailed recovery record.

Earlier Milestone 14 verification checkpoint (superseded by live completion below): local state, projection, repair, research, storage, concurrency, cancellation and actual fixture-backed LLMCALL.main checks passed. Full live verification is pending at the user's request because all five configured OpenRouter accounts report no free requests remaining. The live runner now reports quota_exhausted immediately and persists a blocked project state instead of waiting for a planning timeout. guide.md remains In progress; no live-success claim is made for Milestone 14.


Milestone 14 storage correction: plan.json previously used the model tool's 64 KiB limit and included one exclusion record per catalog deployment. A deterministic 317 KiB reproduction failed with too_large. The router now aggregates excluded/skipped diagnostics by provider/account/reason, records deployment_count and marks model_is_example, while preserving each actual API attempt. Host-owned plans, reports and results use the separate bounded artifact registry for writing, reading and sealing. Storage errors expose safe filename/error codes instead of only plan_storage_failed. Task failure reasons retain their task prefix rather than collapsing into workflow_error.

Additional live recovery findings: arbitrary chat-capable translation models could be selected for code review. Discovery now adds advisory coding/reasoning/fast hints from model names; these rank candidates without excluding any provider or model. The Reviewer prompt requires tracing real save/validation/control-flow code and quoting relevant source before reporting defects. LiteLLM structured dictionary content is normalized to JSON text and still host-validated. LLMCALL.py configures UTF-8 output so Unicode findings cannot terminate Windows result reporting.


Milestone 14 is Completed. The actual LLMCALL.py process exited 0 with all three tasks compiled, tested and approved. Run metadata: workspaces/project_4db303aceae34809a3daec83b17cfd48; final source: workspaces/project_dc62e5d6190c4026841fb11e49c77409/app.py. State/result read-back, final source hash and independent temporary-copy checks for help, add/list/total, persistence and invalid amounts passed. The persisted result.json is 100479 bytes, beyond the former 64 KiB failure boundary. See guide.md for the completion record.


Host storage scaling: the temporary 512 KiB state / 2 MiB artifact caps have been removed. ArtifactFiles streams JSON via iterencode and hashes in 64 KiB chunks, checks the expected old hash before replacement, preserves create-only publication, and cleans partial temporary files on failure. Snapshots still contain references rather than raw source/history bodies. A 100052911-byte artifact (about 1000 times the earlier result) wrote successfully with 251002 bytes peak serializer allocations; larger-than-2-MiB state create/update/read-back, injected disk-full preservation, stale revisions/hashes, Unicode and path guards passed. Actual capacity is bounded by disk, memory and the filesystem; this does not promise unlimited projects.

Prompt budgets are separate from storage capacity. Model file tools and current source/context checks remain bounded; LLMCALL.py currently plans at most six tasks and builds one cumulative app.py. Projects beyond that execution scope require incremental multi-file task decomposition and relevant-source retrieval; removing disk limits alone does not implement those capabilities. No full conversation or large artifact is automatically inserted into agent prompts. JSON read-back still parses the selected JSON document in memory; this change streams serialization and hashing, not arbitrary-size JSON parsing.


Tester Python invocations explicitly enable UTF-8 (`-X utf8`) for both compile and execution. A live rerun exposed cp1252 failure while printing Unicode argparse help; the unchanged affected source then passed the real Tester with parent UTF-8 disabled. The unsuccessful storage-validation run project_e8622644a75e4311a0a0c3dff0954ee4 retained readable matching failed state/result records, demonstrating failure persistence without a byte-limit error.


The live run with streaming host storage exited 0: workspaces/project_0c0e09a233504db6944f00e426f37dc4/result.json. All three tasks passed; final app is workspaces/project_6c82b0bd393449f493266390adf9af1b/app.py. Independent state/result/hash and temporary-copy functional checks passed, including non-finite amount rejection. Milestone 14 remains Completed.
