# NTG feasibility study: large tasks with bounded model context

Date: 2026-10-17
Status: Architecture implemented in project mode; validation and limitations are recorded below.

## Decision

Use the existing Planner, task graph, Coder, Tester, Reviewer, Researcher and ProjectState. Add hierarchical task expansion, model-aware context selection, multi-file targeted edits and explicit integration gates. Execute writes serially first. No additional agent roles, framework, vector database, service or sandbox is needed.

A project may exceed any individual model's context window. Each model request must still fit a usable deployment and include enough exact information to perform its current step. Smaller models can work on narrower tasks; splitting does not create a capability that the selected model actually lacks.

The recommended combination is semantic task decomposition + just-in-time source retrieval + bounded multi-hop execution + deterministic project integration. Larger-context model fallback remains useful when a tightly coupled change cannot be reduced safely.

## Scope of the evidence

This report is based on the current repository, official LiteLLM token-counting documentation and Anthropic's context-engineering guidance. Inline, local measurements made no inference requests and changed no runtime source files. It is a feasibility assessment, not proof that 1000-task projects or arbitrary small models already work.

The preceding storage changes allow large host artifacts; they do not enlarge model context, add durable resumption or enable multi-file coding.

## Bottlenecks identified before implementation

| Area | Current behavior | Required change |
|---|---|---|
| ntg/types.py:plan_response_format | At most six tasks; IDs restricted to T1..T6 | Keep each planning response small, but allow many tasks accumulated through validated expansion |
| ntg/task_graph.py:TaskGraph | Six-task cap, numeric ID assumptions, one running task | Support stable IDs, parent/child tasks and host-validated graph amendments; retain serial execution initially |
| ntg/planner_agent.py:PlannerAgent | Single-file app.py prompt and one flat plan | First produce components, interfaces and acceptance criteria; expand a ready component into small leaf tasks when needed |
| ntg/context.py:coder_context | Entire prior source plus all completed task descriptions; fixed source/metadata budgets | Supply current leaf requirements, relevant contracts and exact source ranges; refer to unrelated completed work by ID/hash |
| ntg/coder_agent.py:run/repair | Whole-file generation; repairs include full source plus repeated line anchors; 64 KiB generated-source cap | Create small files or edit requested ranges/symbols against exact file hashes; retain output limits per operation rather than project-file limits |
| ntg/engine.py:_eligible | Counts serialized UTF-8 bytes as tokens; mostly fixed messages across candidates | Candidate-specific token estimates and separate input/context/output limits; request-size failures trigger packet adaptation |
| ntg/discovery.py:deployment | Some context limits unknown; different provider limit semantics share one context field | Track input, combined-context and output limits separately with provenance/uncertainty; keep unknown metadata attemptable |
| ntg/failures.py:classify | ContextWindowExceededError becomes a temporary unsupported-capability deployment cooldown | Return a size-specific adaptation result; a too-large request is not evidence that the deployment is globally unhealthy |
| ntg/react_agent.py:run | Repeated action/observation calls exist, but all observations accumulate in messages | Rebuild a bounded packet between calls; retain authoritative findings and artifact references in state |
| tools/filesystem.py / tools/registry.py | Whole-file reads, bounded writes and immediate listings; no range reads or search; listings fail beyond 200 entries | Add paginated listing, bounded source-range reads/search and host-applied guarded edits; keep workspace validation |
| ntg/coding_workflow.py | One cumulative file, limited repairs and per-task deadline | Manage a candidate project revision and changed-file set; compile/test/review leaves and integrate components |
| ntg/tester_agent.py | Compiles one entrypoint and executes configured CLI checks | Compile changed Python files and run leaf acceptance, dependency regression and final integration checks |
| ntg/project_state.py / ntg/planned_workflow.py | Source and approvals identify one file/hash; research tied to whole plan hash | Track task versions, project manifests and project revision hashes; preserve provenance across plan amendments |
| ntg/reviewer_agent.py | Reviews a whole single source file | Review exact changed ranges plus dependencies/contracts and test evidence; separately assess integration completeness |

Increasing task count alone would keep resending and rewriting a growing app.py. Increasing byte limits alone would move the failure to the provider. More agents alone would add coordination work without resolving either limitation.

## Feasibility findings

| Approach | Assessment | Recommended role |
|---|---|---|
| Route every large request to a larger-context model | Feasible when available, but dependent on capacity and still limited in output size | Escalation for indivisible work |
| Split by arbitrary token/line count | Fits the transport but can separate definitions, callers and invariants | Use only for navigable source retrieval, not task planning |
| Split by feature, interface and observable acceptance condition | Feasible; requires explicit dependency and integration contracts | Primary planning mechanism |
| Multi-hop read/edit/test with refreshed context | Feasible using the existing ReAct concepts and repair loop | Primary execution mechanism |
| Have many models generate pieces and concatenate them | Unsafe consolidation: conflicting APIs/imports and missing behavior remain undetected | Reject |
| Apply guarded edits to a canonical workspace, then compile/test integration | Feasible and relatively simple | Primary consolidation mechanism |
| Unlimited recursive planning or history | Can grow indefinitely or lose the original requirements | Reject; require measurable progress and resumable budgets |

A model lacking enough context can often be given a narrower task. A model lacking native structured output may use NTG's existing JSON-instruction plus host-validation path. A model lacking a required modality, or repeatedly failing code correctness, needs compatible model escalation or a different task formulation. Text splitting cannot make a text-only model understand unavailable image input. Decomposition also increases API calls, so it does not cure provider quotas.

## Local measurements

1. The current TaskGraph rejected a ten-task input with invalid_task_count. Therefore large plans are not currently supported merely because artifact storage can grow.
2. A repeated Python snippet produced a current byte-based input estimate of 34163. LiteLLM token_counter with an illustrative GPT-4 tokenizer counted 12007 input tokens. Against a fixture 16384-token context, 512 output tokens and an illustrative 1024-token margin, the tokenizer-based packet fit while current routing excluded it. This fixture was not a real deployment and that tokenizer must not be assumed exact for other providers.
3. A small illustrative packet containing a task, function contract, required checks and exact function text counted 90 tokens with the same tokenizer. That shows packet size reduction is mechanically possible, not that this packet alone proves semantic correctness or model success.

## Proposed execution flow

```text
User goal / document references
    -> persist requirements and acceptance IDs
    -> Planner: component outline + interface contracts
    -> expand a ready component into leaf tasks
    -> retrieve current exact code, callers and dependency contracts
    -> build a model-budgeted packet
    -> global compatible provider/account routing
    -> Coder requests missing evidence or returns a targeted edit
    -> host verifies hashes and applies edit to candidate project revision
    -> compile changed Python files
    -> leaf acceptance checks + affected dependency regressions
    -> Reviewer checks concrete changes and evidence
    -> commit approved project manifest; record completed acceptance IDs
    -> continue ready leaves
    -> component integration checks
    -> whole-project acceptance checks and final review
```

The router owns deployment selection and transport fallback. The orchestration layer owns decomposition and semantics. The router must not silently rewrite the user's task to make a request fit.

## Model-aware context admission

For each candidate deployment, check separately:

```text
estimated_input <= known_max_input
requested_output <= known_max_output
estimated_input + requested_output + model_overhead_margin <= known_combined_context
```

Apply only limits whose provider semantics are known. Input-only limits are not combined-context limits. For reasoning models, account for documented reasoning/output consumption and provider accounting rules rather than adding the same reserve blindly twice.

Count the provider-facing packet, including system messages, retrieved text, tools, JSON schema and transport-added JSON instructions. Count structured patch output needs too: a short prompt does not guarantee its requested output will fit.

Use the matching LiteLLM tokenizer when available. A fallback tokenizer is an estimate: retain its provenance, use an explicit conservative margin and refine with runtime evidence. Never label an unknown context window unlimited or block an account solely because metadata is missing. Bound a trial request when metadata remains unknown, and adapt when the provider reports an actual context limit.

The existing global routing pool remains intact. Every fallback must recheck the packet against the next candidate. Prefer models suited to the role using observed outcomes as well as advisory labels; name hints are not validated coding competence.

When a request does not fit:

1. Remove duplicate and optional history, keeping exact required source and requirements.
2. Retrieve a smaller semantically sufficient set of source ranges.
3. Try a healthy compatible deployment with enough context/output capacity.
4. If still oversized, ask Planner to split the current task or dependency analysis into smaller verified steps.
5. If the task is genuinely indivisible, persist the exact missing capacity/dependency reason and a resumable checkpoint rather than cycling unchanged requests.

A context error is specific to the request/model pairing. Record the attempt and any credible reported size constraint. Do not consume all repair attempts or mark the entire account unavailable for a packet-size problem. Provider rate-limit/authentication failures still use existing runtime cooldown/fallback logic.

## Task decomposition contracts

Each leaf needs: stable task ID, parent ID, task version, requirement IDs, prerequisites, allowed write paths, exact input artifact hashes, interface contracts and observable acceptance checks. Keep the parent requirements authoritative; summaries carry links back to exact requirements.

For an expense tracker, suitable leaves are persistence functions, amount validation, add behavior, list behavior, total behavior and CLI integration. Define expense schema and public function signatures before independent implementation. A tiny function may be one leaf; a cross-cutting interface change may need several ordered leaves.

Do not ask for thousands of tasks in one LLM response. Expand only ready, oversized components. Validate unique IDs, acyclicity, dependency preservation and requirement coverage in the host. IDs must no longer depend on the six-task regex or numeric sorting convention. Persist deterministic dispatch order separately.

Parent completion requires all mandatory children plus integration acceptance; it is not inferred from child summaries. Version a plan amendment instead of silently changing the whole-plan hash under approved work. Research/approval reuse must match the relevant task requirements and source revision, with invalidation when those inputs change.

A huge requirements document also needs staged ingestion: preserve the original, assign exact section references and build an acceptance index in bounded passes. Giving the Planner the entire oversized document would simply move the same context failure upstream.

## Multi-hop execution and bounded memory

A hop is a visible operation or validated result, not a demand for hidden chain-of-thought. Reuse Coder/ReAct behavior to request a source range, inspect callers, produce a patch, observe compilation/test evidence and continue.

Each new call gets a freshly built packet containing the current objective, relevant contracts, exact source spans, recent unresolved evidence and a small progress note. Historical source and raw logs remain in artifacts. Summary notes are navigation aids; before editing, the host retrieves exact current text and hashes again.

Provide a context-insufficient result so the Coder can request a definition, caller, schema or contract before guessing. Permit bounded contextual reads on escalation. Repeated reads with unchanged evidence, identical rejected edits, non-shrinking decomposition or repeated equivalent failures trigger a different retrieval/model strategy.

Use separate counters for routing attempts, evidence hops, code repairs and task expansions. An overflow or a new piece of evidence is not automatically a failed code repair. All still have configurable run budgets, no-progress detection and persisted checkpoints; raising a single repair limit is not the solution.

## Safe consolidation and verification

Consolidation means applying verified edits to one canonical project revision. Keep serial writers initially. A patch declares the file, expected SHA-256, targeted span/symbol and replacement; the host rejects stale, overlapping or out-of-scope edits and publishes atomically. Large files remain on disk; only necessary slices enter prompts. Add small-file creation and validated directory creation without sending a whole project as one completion.

Replace final_source_sha256 as the sole identity with a project revision derived from a sorted manifest of workspace-relative paths and hashes. Tests and reviews bind to that revision and changed-file set. Preserve prior approved artifacts and apply the existing rollback principle to the full affected change set.

Run local leaf checks first, then affected integration/regression checks. A final host acceptance pass must verify every original requirement, including behaviors spanning several leaves. An integration failure becomes a focused repair task with exact failing evidence, not an instruction to regenerate the whole project.

Multi-file compile checks, controlled test commands and task-specific acceptance evidence are necessary. The present default no-argument/--help checks alone cannot establish whole-project correctness. Keep the user's existing local execution approach; no sandbox is proposed.

## Implementation sequence

1. Model budgets: separate model limit semantics, count actual wire packets, expose typed context/output adaptation, retain global provider fallback. Validate boundaries, unknown metadata and different limits on fallback.
2. Source navigation and edits: paginated listing, bounded range/search operations, hash-guarded patches for large files. Validate edits in files larger than a prompt, stale hashes, rollback and unchanged regions.
3. Multi-file project identity: manifests, changed-file compile/test/review bindings and cumulative approved revisions. Preserve the small app.py path as a regression scenario.
4. Hierarchical graph: bounded incremental expansion, stable IDs, parent acceptance, amendment versions and requirement coverage. Do not simply change maxItems from six to 1000.
5. Multi-hop execution and durable resumption: task checkpoints, compact operational memory and no-progress rules. Saved JSON inspection is currently not automatic resumption; implement and verify that separately.
6. Scale evaluation: synthetic graph/packet checks, actual integration projects, injected provider/context failures, restarts and measured small-model task success.

Implement each stage behind the existing public workflow where practical. No new specialized agent is required. Use temporary verification harnesses rather than adding unnecessary permanent test files.

## Acceptance criteria for claiming large-project support

- A synthetic 1000-leaf dependency graph schedules deterministically and validates amendments without 1000-task prompts; this tests scheduling only.
- A project larger than the smallest model window is edited via exact ranges; no request exceeds its measured/estimated budget after provider-specific overhead.
- Output truncation is detected and leads to a smaller edit/task, never acceptance of partial source or blind continuation of broken JSON.
- Cross-provider fallback rechecks context and preserves native capability requirements; a too-small candidate does not cause a global quota error.
- Shared interfaces, cross-file imports and integration behavior are tested; a deliberate API mismatch is caught and repaired.
- Restart resumes from validated task/project checkpoints without treating incomplete work as approved or rerunning destructive actions blindly.
- Identical failures/expansions stop producing duplicate API work; original requirement coverage cannot disappear through summaries.
- Real representative projects complete with recorded task success, integration regressions, calls, tokens, latency and rework rates. Repeat with small models and mixed-provider capacity. A storage load test or one smoke run does not meet these criteria alone.

## External evidence and limits

[LiteLLM token usage documentation](https://docs.litellm.ai/docs/completion/token_usage) documents token_counter, model metadata and fallback-tokenizer behavior. These support budget estimation, but do not guarantee exact counts for every configured model or account-specific limits.

[Anthropic context engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) describes selective context, just-in-time retrieval and maintaining useful state across longer tasks. The NTG design above is an engineering recommendation derived from that guidance and local code inspection, not an externally proven NTG implementation.

Conclusion: feasible with the existing agents, but requires orchestration, context and project-identity changes together. Proceed with adaptive context and targeted multi-file work first, then hierarchical planning and resumable multi-hop execution. Do not promise that additional hops make every model capable of every task or guarantee arbitrary 1000x project completion.


## Implementation record

Project mode is implemented through ntg/project_workflow.py, ntg/project_protocol.py, ntg/budget.py and additions to the existing agents, project.py, TaskGraph and ProjectState. LLMCALL.py selects project mode by default and retains --legacy. No new agent role, framework, service, database or sandbox was added.

The implemented path includes requirement ingestion/adaptive splitting, per-candidate wire-packet budget checks, explicit size failure handling, source range/search/list operations, guarded edits, directory-scoped parent expansion, project manifests, immutable approvals, isolated ordered checks, current-source review, parent/global integration repair and validated checkpoint restoration. Optional local tokenizer files and cached fallback estimates avoid model-tokenizer downloads during routing. Approximate tokenizer results remain labeled estimates.

Local inline validation passed: 1000-task deterministic chain scheduling; nested parent expansion and integration barriers; edits beyond 64 KiB preserving unrelated bytes; stale-edit/source-tamper rejection; two-module compilation/execution/review; a deliberate cross-file failure converted to a focused integration repair; planning packets split without losing requirement text/hashes; interruption and approved checkpoint reuse; no-call completed resume; immutable/mutable data separation; fresh fixtures across repeated checks; exact wire overhead, context fallback without cooldown and truncated-output rejection. The legacy PlannedWorkflow path also passed actual compile/test/review with injected responses. No permanent test files were added. Live verification: the default expense-tracker project completed all four tasks (including a focused integration repair), compile/functional checks, integration and final review. Its metadata is workspaces/project_ee0095a386724cc08db03a87d02f283b; final source is project_195154b82f7e4abfba210d984d6ebcf9, revision ea48ea5a140f36665b098a891aee22e3d64de3e9d5a937808a6488d77a797ef2. The run required debugging/resumption; it is not an uninterrupted fresh-run benchmark. A subsequent LLMCALL.py --resume exited 0 without new model calls, and a separate fresh host integration check passed with state/result/source hashes agreeing. An inline interruption regression also verified that a fully tested candidate resumes through fresh testing/review without coder regeneration, preserving changed-line review requirements. Exact initial changed-source slices are supplied before review, source versus saved-report addressing is explicit, and absent future-task files receive task-scope correction rather than an impossible inspect-missing-file loop. Inline regressions verified current versus stale integration evidence and future-task review scope.

Limits: Python/app.py remains the execution target. Initial planning responses stay small, while graph expansion grows the project. Resource budgets remain configurable. Unknown tokenizers are estimated conservatively. Model judgement and complete large-project semantic reliability are not established by synthetic scheduling or one demo. Caller-defined acceptance/integration checks are necessary for real workloads; local subprocess execution is not a sandbox.

### Additional live multi-file verification (2026-10-18)

The two-module arithmetic/CLI project also completed after a call-budget pause and resume: run project_d9a635e45402489ba9512144cc9bbe51, final source project_bbb60bc63cd14348948547d4660c70d0, revision 5c6711cd249f191dd22024ba36be008b95c25ca5cf7494dae5ba4ebd5103be53. Execution order was T1.1 (library), T1.2 (CLI), T1 (parent integration). All passed with final requirement review. Separate fresh subprocesses confirmed CLI output 5 and add(12,7) == 19 / add(-2,3) == 1; manifest/result/state hashes and revisions agreed. This is live multi-file recovery evidence, not an uninterrupted performance or large-project competence benchmark.
