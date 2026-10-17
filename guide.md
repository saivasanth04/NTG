---
goal: Add one bounded Researcher Agent before coding tasks that require external technical evidence
version: 1.0
date_created: 2026-10-17
status: Completed
tags: [milestone-13, researcher, evidence, task-graph]
---

# Introduction

![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

Milestone 13 adds one Researcher Agent to the existing dependency scheduler. Research is conditional, read-only, bounded, and supplied as compact structured context to Coder. Implemented on 17 October 2026; the contract and validation record below describe the delivered capability.

```text
Planner -> validated task graph -> next ready task
                                      |
                             research questions?
                              /             \
                            none            present
                             |                 |
                             |          Researcher
                             |          retrieve + summarize
                             |                 |
                             +-------> Coder <-+
                                         |
                              compile -> Tester -> Reviewer
                                         |
                              next ready task or failure
```

Current repository baseline:

- `ntg/planner_agent.py` produces one through six validated tasks; `ntg/types.py:plan_response_format()` defines their structured schema.
- `ntg/task_graph.py` requires exactly id, description and dependencies. It validates unordered DAGs, selects numerically ordered ready tasks, and guards task states.
- `ntg/planned_workflow.py` executes one ready task at a time, carries the latest approved cumulative app.py into the next task, and stops on failure. Results include graph states and execution order.
- `ntg/coding_workflow.py` manages bounded targeted repairs, compilation, functional checks, and review tied to source hashes. Keep this intact.
- RoutingService provides chat completions, not a web-search service. Existing model tools are workspace filesystem tools; none retrieves technical documentation. The Researcher needs actual retrieval capability, not a prompt asking an LLM to recall current documentation.
- The current demo supports standard-library, single-file applications. Stripe checkout illustrates a research need; production payment integration, SDK installation, web servers and payment execution are outside this milestone.

## 1. Requirements & Constraints

- **REQ-001**: Add one ResearcherAgent that receives goal, current task and explicit research questions, collects external technical documentation through host-controlled retrieval, and returns validated evidence. It does not implement code, execute programs, modify files or approve tasks.
- **REQ-002**: Extend new Planner task JSON with `research_questions`: an array of zero through three distinct, nonblank questions, each at most 300 characters. An empty array means skip research. Do not add a redundant needs_research boolean or infer research with keyword matching.
- **REQ-003**: Research happens once per dispatched task before Coder and remains within the overall planned-run deadline. Coder, Tester, Reviewer and repair attempts receive no web tools. Reuse the same evidence through that task's repairs; do not research again on every revision.
- **REQ-004**: A successful research result must contain retrieved source identities, fetch timestamps and compact findings that reference those identities. Never present a search snippet, remembered URL or model assertion as a fetched document.
- **REQ-005**: Required research that is unavailable, invalid, timed out or insufficient blocks coding and fails the scheduled task. Record the research reason, block dependent tasks, cancel remaining independent tasks under existing fail-fast policy, and retain a nonzero exit. Do not silently proceed with guesses.
- **REQ-006**: Preserve source handoff, dependency verification and final result checks. Research completion is not task completion; tests and Reviewer must still verify the generated source.
- **CON-001**: Keep one through six tasks, sequential execution, existing two-repair limit, Planner 60 seconds, CodingWorkflow 300 seconds and total workflow 1,200 seconds. Give each Researcher run 60 seconds including retrieval and summarization; all retrieval and model calls consume that one budget.
- **CON-002**: Use existing httpx and JSON-schema dependencies plus the standard library. No framework, new research subagents, vector database, browser automation, persistent cache, background process, recursive crawler, new sandbox or Docker.
- **CON-003**: Retrieval is bounded to three questions, one search request per question, three candidates per search, and three fetched documents total per task. Limit each fetched body to 64 KiB and each extracted text to 6,000 characters; aggregate extracted input at most 12,000 characters. Model summary output max_tokens=2048 and serialized Coder research context at most 8 KiB UTF-8.
- **SEC-001**: Hosts and retrieval backends are selected by application configuration, never by model permission. Use HTTPS, port 443, no URL credentials, no URL fragments or secret query parameters; block local/private/link-local/loopback/multicast/reserved destinations and redirects to them. Validate every URL and redirect before access. Never forward provider credentials or workspace contents to documentation sites.
- **SEC-002**: Treat retrieved text as untrusted reference data, clearly delimited from system instructions. Ignore embedded instructions to execute code, reveal secrets, change roles or bypass output contracts. Fetched code examples are not executable tool actions.
- **PAT-001**: Keep retrieval, summarization and workflow dispatch separate. Host code records provenance; the LLM supplies only findings and unanswered questions referencing host-issued document IDs. A source reference establishes provenance, not proof that a claim is semantically correct.

### Task schema and backward compatibility

New Planner output includes all four fields:

```json
{
  "id": "T2",
  "description": "Implement an HTTP API client using documented request fields.",
  "dependencies": ["T1"],
  "research_questions": [
    "Which endpoint and request fields does the official API documentation require?"
  ]
}
```

Ordinary local tasks such as arithmetic, expense summation and no-argument argparse help should use an empty list. Every question must be relevant to the task's implementation and acceptance requirements. Planner must stay within current application constraints; requesting research does not authorize new application capabilities.

For legacy three-field task dictionaries, validation treats missing research_questions as an empty list without mutating the supplied dictionaries or changing their historical plan hashes. New strict provider schema requires the fourth field. Host validation accepts exactly the old or new shape, never arbitrary extra fields. Store/read the optional value with task.get('research_questions', []). Hash explicit questions when present. Validate questions in TaskGraph as well as Planner so direct graph construction cannot bypass bounds.

### Research output contract

Host ResearchResult fields:

```json
{
  "status": "completed",
  "task_id": "T2",
  "plan_sha256": "host-computed-plan-hash",
  "questions": ["Which request fields are required?"],
  "findings": [
    {
      "question_index": 0,
      "summary": "Concise implementation guidance grounded in the retrieved document.",
      "source_ids": ["S1"]
    }
  ],
  "sources": [
    {
      "id": "S1",
      "url": "https://docs.example.com/api/reference",
      "retrieved_at": "2026-10-17T10:00:00Z",
      "content_sha256": "host-computed-document-hash"
    }
  ],
  "open_questions": [],
  "reason": null
}
```

The example host above is a placeholder domain, not an implementation backend or an actual retrieved source. Do not create artifacts claiming these example values were fetched.

Use statuses completed, blocked and failed. Persist skipped research in task metadata as status skipped, without making an LLM or network call. Define ResearchResult.verified to require completed status, matching task/plan/question identities, one through three successful host fetch records, one through six findings, valid question indexes, nonempty source references, and coverage of every required question with no open questions. Each finding summary is at most 600 characters; source IDs must exist in the host manifest. Model output contains only findings and open_questions; status, URLs, timestamps, hashes and routing evidence come from host code. Never let the model self-certify successful retrieval.

If evidence contradicts itself or does not answer a question, return an open question and block coding with research_incomplete. Do not turn an arbitrary fact into a completed finding to satisfy schema coverage. Retain version/date information in summaries when the retrieved document provides it; a fetch timestamp alone does not establish document currency.

### Retrieval boundary

Add `tools/research.py` with `ResearchTools(search_backend, allowed_hosts)`:

```python
await tools.search(query, limit=3)  # validated candidates: title, url
await tools.fetch(url)             # host-issued source ID and bounded extracted text
```

search_backend is a host-injected async callable with the documented result contract, not an LLM and not a guessed public endpoint. An actual backend integration must be explicitly configured; do not invent an endpoint, scrape search-engine HTML or claim chat completions include search. With no backend, questions requiring search return research_unavailable before Coder. Existing no-research tasks continue to work without a backend. Deterministic injected backends cover offline validation. A live research claim requires a real configured backend and successfully fetched documents.

Search receives only bounded questions and relevant public product/version names, not goal text wholesale, generated code, files, credentials or model routing history. Build queries deterministically from those questions; do not add an extra model query-planning loop. Filter candidates to a host-configured exact hostname allowlist of official technical documentation; prefer product/version-specific primary documentation. Do not accept hostname suffix lookalikes.

Use async httpx with trust_env=False, follow_redirects=False, bounded streaming, at most two explicitly validated redirects and 10 seconds per request, capped by the remaining Researcher deadline. Do not forward authentication between requests. Fetch public text/html or text/plain documents only; reject binary, PDFs, downloads and compressed responses unless decoded-size limits can be enforced (request identity encoding for this first version). Extract readable HTML text using a small standard-library HTMLParser, excluding script/style content. Search results are candidates only; do not cite unfetched snippets.

Validate resolved addresses immediately before connecting. Use a transport that connects to those validated public addresses while retaining hostname TLS verification, or a trusted egress service that enforces equivalent destination restrictions. A DNS check followed by an unrelated fresh resolver lookup is insufficient against rebinding. If this cannot be guaranteed, fail closed with research_fetch_unavailable rather than claiming SSRF-safe arbitrary URL retrieval. Keep the implementation explicit and small; never weaken address checks to make a live smoke run pass.

## 2. Implementation Steps

### Implementation Phase 1 — Extend task and research contracts

- **GOAL-001**: Declare research needs and validate compact evidence without changing ordinary chat behavior.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Update ntg/types.py:plan_response_format with required research_questions for new Planner responses. Update ntg/planner_agent.py instructions and validate_plan, plus ntg/task_graph.py exact-field validation, to accept legacy three-field inputs and enforce question count, length, uniqueness and blank rejection. Preserve existing DAG validation and legacy plan hashing. | Yes | 2026-10-17 |
| TASK-002 | Add internal research_response_format in ntg/types.py for findings and open_questions only. Add ResearchResult in ntg/researcher_agent.py with host provenance, verified and to_dict methods. Enforce IDs, coverage, lengths and the 8 KiB context bound; reject oversize results rather than cutting JSON or silently dropping sources. | Yes | 2026-10-17 |

Phase completion: legacy expense-tracker plans remain valid and skip research; new plans hash explicit questions; invalid research fields or fabricated source IDs cannot verify.

### Implementation Phase 2 — Retrieve and summarize

Depends on Phase 1.

- **GOAL-002**: Collect bounded primary documentation through one controlled interface and summarize it once.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-003 | Add tools/research.py implementing ResearchTools, injected search contract, exact host allowlist, public HTTPS checks, safe transport boundary, redirect and byte limits, timeouts, HTML text extraction, UTC fetch times and SHA-256 of fetched bytes. Keep it outside the Coder/ReAct filesystem registry; no model gets a general HTTP tool. | Yes | 2026-10-17 |
| TASK-004 | Add ResearcherAgent(llm, research_tools) with async run(goal, task, plan_sha256). Validate task input, deterministically search each question, fetch up to three distinct approved documents and assign S1–S3 IDs. Use one model call with a strict summarization schema and remaining deadline. Validate references against the actual retrieval manifest. No automatic model retries beyond Router's existing transport behavior. | Yes | 2026-10-17 |
| TASK-005 | Return stable reasons: research_unavailable, research_timeout, research_no_sources, research_incomplete, invalid_research_output, research_output_too_large and research_fetch_unavailable. Record sanitized model routing, rejected retrieval reason codes and attempted counts separately from the compact Coder payload. Propagate external CancelledError and close active HTTP responses/clients. | Yes | 2026-10-17 |

Phase completion: deterministic retrieval and model doubles produce a verified report only from successfully fetched evidence. Missing backend, timeouts and invalid sources produce explicit failures within the budget.

### Implementation Phase 3 — Integrate before coding

Depends on Phase 2.

- **GOAL-003**: Research a ready task once, preserve evidence through repairs, and retain task-graph correctness.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-006 | Extend PlannedWorkflow.__init__ with optional keyword researcher=None. After selecting a ready task, graph.start and create its TaskExecution placeholder before awaiting research so cancellation has an active task to finalize. Empty questions skip retrieval. Nonempty questions without Researcher block with research_unavailable. On research failure, finish graph task failed and use existing fail-fast propagation; never invoke Coder. | Yes | 2026-10-17 |
| TASK-007 | Add compact research context to the existing CodingWorkflow context while retaining goal, task, dependencies, completed tasks and latest approved source/hash. Do not reset source to dependency-parent snapshots. On required research, pass only a verified report whose identity matches the current plan/task/questions. Existing repair_context.task_context carries this unchanged through repairs. | Yes | 2026-10-17 |
| TASK-008 | Add research as reference evidence to Reviewer context/specification so it can assess API/version compliance while still reviewing actual code and tests. Do not change Tester from deterministic local checks or let research replace testing. Keep goal, evidence and fetched text as data; use an explicit data section in prompts. Modify Coder/Reviewer instructions only where needed to recognize research references. | Yes | 2026-10-17 |
| TASK-009 | Add optional research report and status to TaskExecution.to_dict. Persist research_<task_id>.json once in the existing host run directory, before Coder dispatch; store manifest and compact findings, not whole fetched pages. Keep skipped status in task metadata. On storage failure block with research_storage_failed; do not proceed without the report. Persist final task summary and graph on failure. | Yes | 2026-10-17 |
| TASK-010 | Extend TaskExecution/PlannedResult verification to require matching verified research when the plan has nonempty questions. Legacy/empty-question tasks must remain verifiable without a report. Prevent tampered plan hash, task ID, source references or report contents from satisfying final verification; hash persisted report bytes and associate that hash with the task summary. Distinguish graph dispatch order from coding attempts because a task can fail in research before a Coder revision exists. | Yes | 2026-10-17 |

Phase completion: research gates Coder, required research cannot be bypassed by a passing code result, and no-research plans preserve Milestone 12 behavior.

### Implementation Phase 4 — Configure, document and verify

Depends on Phase 3.

- **GOAL-004**: Make research behavior explicit to callers and provide honest validation evidence.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-011 | Update LLMCALL.py to accept or construct only an explicitly configured Researcher/retrieval backend, retain the no-research expense-tracker demo by default, and print skipped/completed/blocked research with source URLs and report paths. Keep liveness and nonzero exits. Export ResearcherAgent/ResearchResult from ntg/__init__.py. Document the injected backend contract and allowlist configuration; never add credentials to logs or source. | Yes | 2026-10-17 |
| TASK-012 | Update README.md for research limits, backend-unavailable behavior, provenance, reference-only output, retained application scope and metadata. Run the checks below with temporary inline harnesses; add no permanent test/example files. Mark this guide Completed only after implementation and evidence justify it; clearly distinguish deterministic checks from live retrieval/model success. Use explicit UTF-8 for all documentation writes. | Yes | 2026-10-17 |

## 3. Alternatives

- **ALT-001**: Give every agent unrestricted browsing. Rejected because it duplicates research and makes sources, cost and tool behavior hard to control.
- **ALT-002**: Ask Researcher to answer from model memory with URLs. Rejected because remembered content is not retrieved current evidence.
- **ALT-003**: Add research as independent generated graph nodes. Deferred; a conditional pre-Coder phase is enough and preserves current task IDs and graph semantics.
- **ALT-004**: Build a general autonomous search loop, persistent knowledge base or browser crawler. Deferred; deterministic bounded retrieval and one summarization call are sufficient.

## 4. Dependencies

- **DEP-001**: Milestone 12 TaskGraph, PlannedWorkflow, plan hashing and result verification.
- **DEP-002**: Existing RoutingService, ChatRequest and JSON-schema validation for summarization; these do not supply external search.
- **DEP-003**: Existing httpx plus a host-configured async search backend and controlled public-document fetch transport. Without these, required research is unavailable; local no-research tasks remain supported.
- **DEP-004**: Existing CodingWorkflow/Reviewer handoff, workspace metadata storage, targeted repairs and local compile/execution gates.

## 5. Files

- **FILE-001**: ntg/researcher_agent.py — new bounded retrieval orchestration, summarization and ResearchResult.
- **FILE-002**: tools/research.py — new host-controlled retrieval interface and provenance capture.
- **FILE-003**: ntg/types.py — task research questions and internal summary schema.
- **FILE-004**: ntg/planner_agent.py and ntg/task_graph.py — research declaration and compatible validation.
- **FILE-005**: ntg/planned_workflow.py — conditional research phase, metadata, failure propagation and verification.
- **FILE-006**: ntg/coder_agent.py and ntg/reviewer_agent.py — evidence handling instructions/context only as needed; retain execution/repair contracts.
- **FILE-007**: LLMCALL.py, ntg/__init__.py, README.md and guide.md — configuration, exports, reporting and validation record.

## 6. Testing

Use deterministic injected search/fetch/model doubles and temporary directories first. Real external requests must use an explicitly configured approved backend; do not mark fixture data as live retrieval. No permanent test files are required.

- **TEST-001**: Legacy and new empty-question plans perform zero search, fetch or Researcher model calls. Existing diamond scheduling, cumulative sibling preservation, compile/tests/review and final verification still pass.
- **TEST-002**: Reject too many, duplicate, blank, nonstring and oversized questions. Verify legacy hash remains unchanged and explicit questions alter plan identity. Test direct TaskGraph construction as well as Planner parsing.
- **TEST-003**: Required research calls Researcher once before Coder, passes compact verified findings to initial generation and every repair, and exposes reference evidence to Reviewer. Coder receives no raw page bodies or retrieval tools.
- **TEST-004**: Reject invented source IDs, missing question coverage, empty findings, unresolved questions, oversized JSON, malformed output and model-supplied provenance fields. Recheck mutated report identity/content in final verification and artifact association.
- **TEST-005**: Missing backend, no sources, rate limits, response timeout, summary timeout and metadata failure block Coder. Verify dependent graph tasks blocked, independent pending tasks cancelled, useful reasons persisted, no false final project and nonzero CLI exit.
- **TEST-006**: Test private IPv4/IPv6, localhost, credentials, ports, non-HTTPS, hostname lookalikes, query secrets, redirects, DNS rebinding boundary, oversized/encoded bodies, binary types and streaming cutoff. No disallowed destination may be contacted; unsupported safe transport must fail closed.
- **TEST-007**: Search/fetch counts, aggregate text, UTF-8 context bytes and the 60-second total are enforced even with slow readers or repeated candidates. Cancellation closes connections, unwinds the active task and propagates without orphan work.
- **TEST-008**: Inject a document containing instructions to reveal credentials or write files. Verify roles have no such retrieval capabilities, prompts delimit data, the summary cannot contain executable tool actions, and workspace/code cannot be changed by research.
- **TEST-009**: Persisted report timestamps, URLs and hashes must originate from actual host fetch records. No raw pages or secrets appear in task context/logs. Source failures and insufficient evidence remain visible; a citation is not treated as independent proof of factual correctness.
- **TEST-010**: With an explicitly configured live backend, fetch at least one official technical document and produce a small source-linked report. Do not execute payment calls, use live Stripe credentials or claim Stripe integration was built. If live configuration is unavailable, record that limitation rather than claiming live success.

## 7. Risks & Assumptions

- **RISK-001**: A Planner can omit needed questions or request unnecessary research. Instructions and validation establish a first explicit gate, not perfect semantic classification; record questions and improve from observed failures rather than add another classifier agent.
- **RISK-002**: Official pages can be outdated, unavailable or contradictory. Record fetch times and versions where present, preserve open questions, and block insufficient required research.
- **RISK-003**: Small summaries can omit detail, and citations can support a different claim than the model states. Keep questions narrow, preserve source links, and retain Reviewer/test gates; do not claim provenance alone proves correctness.
- **RISK-004**: Research adds elapsed time and another failure boundary. Count it inside the existing overall deadline; do not increase repair counts or hide retrieval failures as application bugs.
- **ASSUMPTION-001**: The first version continues to build standard-library single-file applications sequentially. Research is a capability addition, not authorization for SDK installation, paid services, browser actions or production integration.
- **ASSUMPTION-002**: The host configures trusted documentation domains, backend access and safe transport. No paid search service or provider credential is selected automatically by this guide.

## 8. Related Specifications / Further Reading

Local baseline references: ntg/planner_agent.py, ntg/task_graph.py, ntg/planned_workflow.py, ntg/coding_workflow.py, ntg/coder_agent.py, ntg/tester_agent.py, ntg/reviewer_agent.py, tools/registry.py and README.md.

This file replaces the prior completed Milestone 12 guide as the next-work specification. Milestone 13 is now implemented with the existing execution safeguards retained. The record below distinguishes deterministic checks, real retrieval, and model availability.

### Implementation and validation record — 17 October 2026

Added ResearcherAgent/ResearchResult, internal structured findings schema, backward-compatible research_questions validation, host-injected ResearchTools, public-IP-pinned TLS fetching, optional scheduler research gating, compact Coder/Reviewer evidence, persisted research reports and hashes, final report identity/content verification, and CLI research reporting. ResearchTools is not in the filesystem/model registry. No framework, provider subscription, permanent tests/examples or dependency was added.

Deterministic inline checks passed for legacy plan/hash preservation; invalid question rejection; empty-question skip; conditional one-time research; actual local compile/execution/review handoff; unordered diamond/source preservation; fabricated references, unresolved questions, oversized UTF-8 evidence and extra model fields; tampered report identity/content; unavailable backend; safe backend exception handling; research storage failure; dependency failure propagation; timeout and external cancellation with graph persistence. A real CodingWorkflow compile-failure/repair run reused identical research context and exact failure evidence, made no additional retrieval/model research calls, and passed final tests/review.

Mocked transport checks exercised public-address validation before connection, IP-pinned connection with original TLS hostname, disallowed redirects, URL credentials/query/port/host restrictions, body limits, binary/compressed/chunked/malformed rejection, script/style exclusion, host timestamps/content hashes and connection closure on cancellation. Request counts and aggregate text limits were checked using deterministic sources. Metadata and generated candidates stayed inside temporary directories and were cleaned up.

Live smoke: an explicitly configured host-curated lookup returned the official Python __main__ documentation URL. The real controlled HTTPS fetch succeeded and produced bounded extracted text. The subsequent real routed model summary returned research_unavailable and a blocked/unverified result. Live model research completion is therefore not claimed. No production integration or payment request was executed. A general live search-service backend remains host-supplied through the documented injection interface, not bundled or guessed.

CLI configuration: ordinary python LLMCALL.py uses no backend and supports local no-research tasks. Applications can construct ResearchTools(search_backend, allowed_hosts), ResearcherAgent(shared_llm, tools), and PlannedWorkflow(planner, workflow, researcher=researcher). For embedding the existing demo, LLMCALL.main(researcher=researcher) accepts the configured role. Required research without a backend is deliberately blocked rather than replaced by model memory.
