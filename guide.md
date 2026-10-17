---
goal: Build a validated task graph and deterministic dependency scheduler on the current agent workflow
version: 1.0
date_created: 2026-10-17
status: Completed
tags: [milestone-12, task-graph, scheduler, dependencies]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Completed-brightgreen)

Milestone 12 replaces list-order execution with dependency-driven scheduling. Milestone 12 is implemented. This document records the implementation contract and validation evidence. Use the existing Planner, Coder, compile validation, Tester, Reviewer, workspaces, and bounded repair loop. Add a small Python task graph, not another agent or orchestration framework.

The current repository already represents dependencies as task dictionaries. The missing part is graph validation and scheduling independently of the Planner's output order.

```text
       T1
      /  \
     T2  T3
      \  /
       T4
```

After T1 passes, T2 and T3 are both eligible. Execute them sequentially in numeric ID order for this milestone. T4 becomes eligible only after both pass. A graph does not require concurrent execution.

Current implementation findings:

- `ntg/planner_agent.py:validate_plan()` requires tasks in contiguous T1, T2, ... order and permits dependencies only on preceding entries. This rejects valid unordered DAGs and avoids cycles indirectly rather than detecting them explicitly.
- `ntg/planned_workflow.py:PlannedWorkflow.run()` iterates `enumerate(plan.tasks)`, checks dependencies, and stops at the first failure. It does not select among ready tasks.
- `PlannedResult.verified` compares execution IDs directly to Planner list order. This must change when execution order differs from plan order.
- Each task produces an approved cumulative single-file snapshot. The next task receives the latest approved source, not separate dependency branches. Preserve this contract.
- `LLMCALL.py` prints the plan and revision evidence after completion and emits liveness updates while running. Keep that behavior and add graph outcomes to the final output.

## 1. Requirements & Constraints

- **REQ-001**: Keep the Planner JSON shape unchanged: each task has exactly `id`, `description`, and `dependencies`. Keep one through six tasks and IDs T1 through Tn, contiguous as a set; allow any input array order and references to later array entries.
- **REQ-002**: Before allocating coding workspaces or invoking Coder, reject duplicate IDs, empty descriptions, unknown dependencies, duplicate dependency edges, self-dependencies, and cycles. Preserve the public `invalid_plan_output` boundary; make a safe specific validation reason available in planning metadata.
- **REQ-003**: A task is ready exactly when its state is `pending` and every declared dependency has state `passed`. Only a verified Coder/Tester/Reviewer result may transition a task to `passed`.
- **REQ-004**: Select the numerically smallest ready task, execute it once through the existing CodingWorkflow, update its state, and recompute readiness. Never depend on input array order or ID arithmetic as a dependency rule.
- **REQ-005**: Preserve cumulative source by passing the latest approved snapshot, all verified completed task descriptions, and the current task's dependency IDs into each execution. Dependencies control eligibility; they do not restrict source context to parent tasks.
- **REQ-006**: Record every task's final state, including unexecuted tasks, plus actual execution order, dependency IDs, source hashes, stop reason, and workspace links. Require all tasks to pass before reporting overall success.
- **CON-001**: Execute at most one task at a time. Do not use gather, threads, worker pools, branch merging, automatic replanning, persistent resume, or scheduler-level retries. Retain first-failure termination for compatibility.
- **CON-002**: Preserve one workspace per task, in-place targeted repairs within that workspace, immutable revision snapshots, syntax validation before functional execution, raw stdout/stderr, evidence-based feedback, regression rollback, repeated-failure detection, and Reviewer approval tied to the tested source hash.
- **CON-003**: Keep Planner's 60-second limit, CodingWorkflow's 300-second limit and at most two repairs, and PlannedWorkflow's overall 1,200-second deadline. Router retains transport retries; the scheduler adds none.
- **SEC-001**: IDs and dependency names are data. Validate before using IDs in host metadata filenames. Preserve workspace path guards, host-owned entrypoint selection, and approved-source hash verification. No sandbox or Docker is introduced.
- **GUD-001**: Keep public constructor/run interfaces and ordinary structured chat/tool/review outputs compatible. Document any additive fields in result metadata. Do not add permanent test files, examples, or dependencies.
- **PAT-001**: Keep graph logic deterministic and free of LLM, filesystem, subprocess, and network calls. PlannedWorkflow owns execution and artifacts; TaskGraph owns validation, readiness, and task states.

### Graph and state contract

Create `ntg/task_graph.py`. Continue using validated dictionaries rather than adding a Task dataclass merely to duplicate the existing representation.

```python
TaskGraph(tasks)
graph.ready()                    # list of copied task dictionaries, numeric ID order
graph.start(task_id)              # pending -> running, only if ready
```

The actual public methods must be:

```python
graph.finish(task_id, *, passed: bool)  # running -> passed or failed
graph.stop(reason, failed_task_id=None)
graph.to_dict()                       # copied state, no live mutable references
```

Use exactly these per-task states:

| State | Meaning |
|---|---|
| pending | Validated but not started |
| running | The single active CodingWorkflow execution |
| passed | Host verified execution and Reviewer approval |
| failed | Executed but did not produce a verified result, including blocked execution |
| blocked | Not executed; a dependency failed or became blocked |
| cancelled | Not executed because fail-fast policy, cancellation, deadline, or infrastructure stop ended this run |

Store a separate `reason` for each terminal failure/blocked/cancelled state. Keep TaskExecution's existing result status (`passed`, `failed`, `blocked`) intact; graph state describes scheduling, not replacement of the execution report.

`stop()` requires no running task: mark the active task failed first if execution was interrupted. Mark pending descendants of failed tasks blocked using dependency edges, then mark remaining pending tasks cancelled. A failed dependency's descendants remain blocked even when another independent branch is cancelled. This explicitly distinguishes dependency failure from fail-fast policy.

### Source handling at the join

For the diamond, actual execution is T1, T2, T3, T4. T3 extends the verified T2 snapshot, which already contains T1. T4 extends T3 and therefore receives both branches' completed features. Pass direct dependency IDs and completed task records separately so provenance remains clear.

Do not initialize T3 only from T1: that would discard T2 from the cumulative application. Do not concatenate parent sources or claim branch isolation. Independent eligibility does not imply independent files in the current single-file architecture.

## 2. Implementation Steps

### Implementation Phase 1 — Validate and model the graph

- **GOAL-001**: Accept valid unordered DAGs and reject invalid plans before coding begins.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Add `ntg/task_graph.py`. Validate the full task set, contiguous ID set, required fields, nonempty descriptions, unique dependencies, known IDs, and no self-edges. Deep-copy input. Build ID lookup, dependency sets, and reverse edges. Use Kahn's algorithm on copies of indegree counts to reject cycles, including disconnected cycles. Keep at most six tasks. Define stable internal reason codes such as `duplicate_task_id`, `unknown_dependency`, `self_dependency`, `duplicate_dependency`, `cyclic_dependencies`, and `invalid_task_ids`. | Yes | 2026-10-17 |
| TASK-002 | In `ntg/planner_agent.py:validate_plan()`, retain JSON-schema validation, replace array-position restrictions with TaskGraph validation, and preserve the returned task array order for existing plan hashing. Expose safe specific reasons without breaking callers that catch `ValueError('invalid_plan_output')`; for example attach a `validation_reason` attribute to that exception. Avoid circular imports: task_graph must not import planner_agent or schema types. | Yes | 2026-10-17 |
| TASK-003 | Update `PlannerAgent.SYSTEM_PROMPT` to request explicit prerequisite edges and allow branching, multiple roots, and joins. Remove statements requiring dependencies on earlier array entries or making the last array item inherently the final task. Retain runnable cumulative app.py, observable task requirements, and existing bounds. Keep `plan_response_format()` JSON shape and ID bounds in `ntg/types.py`. | Yes | 2026-10-17 |
| TASK-004 | Implement `ready`, `start`, `finish`, `stop`, and `to_dict` with guarded state transitions. Sort readiness by numeric ID. Reject a second active task, unknown IDs, starting nonready tasks, completing unstarted tasks, and attempts to overwrite terminal results. Return copies from inspection methods. | Yes | 2026-10-17 |

Phase completion: an unordered diamond validates; its ready sequence is T1, then T2/T3, then T4. Invalid graphs fail before Coder or workspace allocation.

### Implementation Phase 2 — Integrate dependency scheduling

Depends on Phase 1.

- **GOAL-002**: Drive the existing workflow using verified dependency completion and preserve approved code at every join.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-005 | In `PlannedWorkflow.run()`, construct the graph after validating the plan and before creating execution workspaces. Replace the `for enumerate(plan.tasks)` loop with recompute-ready/select-one/execute/update. If all tasks passed, finish successfully. If pending work exists but nothing is ready and nothing is running, fail with `scheduler_no_progress`; do not spin or call the LLM. | Yes | 2026-10-17 |
| TASK-006 | Keep `_approved_source()` immediately before dispatch and after successful execution. Pass latest cumulative source/hash, direct dependencies, plan hash, and all verified completed tasks to Coder. Set acceptance_scope to whole-goal completion when this selected task is the last unfinished task, not when it occupies the last position in the Planner array. Preserve completed features at every step. | Yes | 2026-10-17 |
| TASK-007 | Call graph.start before dispatch. Transition to passed only after `TaskExecution.verified` succeeds, including fresh approved-source verification. Otherwise finish failed, call stop to classify descendants and independent remaining tasks, and terminate without executing blocked work. Retain task-level repair behavior unchanged. | Yes | 2026-10-17 |
| TASK-008 | Handle workflow timeout, external cancellation, and artifact failures consistently. On interruption, terminate/await the active child through existing cancellation behavior, mark any running graph task failed with the interruption reason, classify pending states, and attempt bounded metadata finalization. Re-raise external CancelledError after cleanup; never report cancellation as success. Do not use an unbounded shield that defeats deadlines. | Yes | 2026-10-17 |

Scheduling outline, deliberately sequential:

```python
while True:
    if all(s == 'passed' for s in graph.states.values()):
        break
    ready = graph.ready()
    if not ready:
        # No running task is expected in this sequential loop.
        stop_with_reason('scheduler_no_progress')
        break
    task = ready[0]
    graph.start(task['id'])
    execution = await execute_existing_workflow(task, latest_approved_source)
    # This helper represents existing dispatch + fresh source-hash validation.
    graph.finish(task['id'], passed=execution.verified)
    if not execution.verified:
        graph.stop(execution.result.reason, failed_task_id=task['id'])
        break
    latest_approved_source = read_and_verify_approved_source(execution)
```

Adapt the outline to current functions; it is not a request for new execution helper classes. Fresh source validation must occur before the success transition as specified in TASK-007.

### Implementation Phase 3 — Make graph outcomes inspectable

Depends on Phase 2.

- **GOAL-003**: Report dependency outcomes and verify success without relying on Planner array order.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-009 | Extend PlannedResult with additive `task_graph` and `execution_order` snapshots. Keep `tasks` as attempted TaskExecution objects in actual execution order; never fabricate revisions for blocked/cancelled tasks. `to_dict()` records graph states, reasons, dependencies, ordered attempted IDs, and current completion evidence in result.json. Keep plan.json and task_<id>.json artifact conventions. Persist failure graphs too. | Yes | 2026-10-17 |
| TASK-010 | Update PlannedResult.verified: validate plan/hash; require each planned ID exactly once in executions, matching unique execution_order, all task results verified, graph passed state for every ID, and each dependency appearing earlier in execution_order. Compare graph edges to the validated plan rather than trusting serialized state. Final project/hash must match the last actually executed verified snapshot. Clear final success fields on failed/blocked result or metadata persistence failure. | Yes | 2026-10-17 |
| TASK-011 | Update `LLMCALL.py` to print actual execution order and final task graph states/reasons alongside existing plan, revision evidence, metadata paths, liveness, and exit status. Print plan entries as declarations, not implied execution order. Print dependencies of blocked tasks clearly. Preserve nonzero exit for every unverified run. Export TaskGraph from `ntg/__init__.py` only if intended as a public API. | Yes | 2026-10-17 |
| TASK-012 | Update README.md with graph scheduling, sequential cumulative-source semantics, example diamond readiness, failure policy, bounds and metadata fields. Execute the validation checklist below without adding permanent test files. Change this guide's status to Completed only after implementation and evidence satisfy the checklist; distinguish deterministic verification from live LLM success. | Yes | 2026-10-17 |

## 3. Alternatives

- **ALT-001**: Keep the ordered list and add dependency checks only. Rejected because input order still determines execution and valid unordered plans cannot run.
- **ALT-002**: Execute ready tasks concurrently in separate workspaces. Deferred because the current artifact is cumulative app.py; branch merging, conflict resolution and post-merge validation are separate work.
- **ALT-003**: Add LangGraph, a scheduler agent, database, or message broker. Rejected because a bounded in-memory DAG and Python loop cover this milestone.

## 4. Dependencies

- **DEP-001**: Existing PlannerAgent, PlanResult and schema validation; preserve plan mutation/hash checks.
- **DEP-002**: Existing CodingWorkflow and TaskExecution verification, compile-first testing, bounded minimal repairs, Reviewer, and WorkspaceManager.
- **DEP-003**: Python standard library for graph traversal and state snapshots. Existing jsonschema remains; no package installation is required.

## 5. Files

- **FILE-001**: `ntg/task_graph.py` — new pure graph validation, readiness and state transitions.
- **FILE-002**: `ntg/planner_agent.py` — graph-aware validation and Planner instructions.
- **FILE-003**: `ntg/planned_workflow.py` — scheduler integration, cumulative handoff, graph results and final verification.
- **FILE-004**: `LLMCALL.py` — graph outcome reporting with existing liveness and exit semantics.
- **FILE-005**: `README.md` and `guide.md` — implementation contract, evidence and status.
- **FILE-006**: `ntg/types.py` and `ntg/__init__.py` — inspect schema compatibility and optional export; change only if needed. Coder, Tester, Reviewer and filesystem interfaces remain unchanged.

## 6. Testing

Use inline Python harnesses and temporary directories. Deterministic doubles may replace LLM roles; use real compile/execution checks for source-handoff verification. Do not add permanent test/example files or alter historical failed project artifacts.

- **TEST-001**: Validate single-task, chain, diamond, disconnected DAG, multiple-root and unordered-input plans. For `[T4, T3, T1, T2]` with diamond dependencies, actual execution must be `[T1, T2, T3, T4]`. T4 must not become ready after T2 alone.
- **TEST-002**: Reject duplicate/missing IDs, nonexistent references, repeated dependency edges, self-loops, two-node cycles, disconnected cycles, invalid fields and task-count violations before any Coder call or coding workspace allocation. Verify safe specific reason metadata and unchanged public invalid_plan_output behavior.
- **TEST-003**: Reject invalid state transitions, duplicate dispatch, second active execution and external mutation of graph input/returned snapshots. Recompute readiness after every successful task.
- **TEST-004**: Fail T2 in the diamond: T1 passed, T2 failed, T4 blocked, T3 cancelled under fail-fast policy. T3 and T4 receive no execution calls. Repeat with both siblings complete and T4 failing; no successful final artifact is advertised.
- **TEST-005**: Use deterministic Coder results that add distinguishable features at T1, T2 and T3; inspect T3 context to confirm it contains T2's approved implementation. Verify T4's final source contains both sibling features and passes real compilation and functional checks.
- **TEST-006**: Treat status='passed' with invalid review/source identities as a failure. Tamper with an approved artifact before dispatch and reject it. Test mismatched graph edges, duplicate execution IDs, wrong execution order and wrong final hash against PlannedResult.verified.
- **TEST-007**: Exercise execution timeout, overall deadline, external cancellation, unavailable workspace and metadata storage failure. Verify no hanging pending loop, no orphan active execution, no descendant dispatch and no false success. CancelledError propagates after cleanup.
- **TEST-008**: Run existing inline repair regression checks: syntax failure blocks functional execution, exact evidence reaches repairs, missing no-argument help is diagnosed, targeted edits preserve passing --help, identical failures stop, and rollback restores current source. Graph changes must not weaken these safeguards.
- **TEST-009**: Inspect result.json for all planned states and actual order, and task summaries for only attempted tasks. A successful graph requires every result verified. A normal `python LLMCALL.py` smoke run must show liveness and graph outcomes; record actual provider outcomes separately because free-provider availability and plan quality are nondeterministic.

## 7. Risks & Assumptions

- **RISK-001**: Removing list-order restrictions without updating final verification can reject successful DAG executions or advertise the wrong final workspace. TASK-010 covers both.
- **RISK-002**: Using only dependency-parent source for sibling tasks loses approved code in the single-file model. TASK-006 explicitly carries the global latest approved snapshot.
- **RISK-003**: Plan edges do not prove implementation correctness. Keep compilation, configured behavioral checks and Reviewer approval; do not present graph completion as exhaustive functional testing.
- **RISK-004**: Fail-fast scheduling intentionally cancels independent pending work after any task failure. This milestone does not promise maximum throughput or continuation of unaffected branches.
- **ASSUMPTION-001**: Existing support remains one standard-library app.py, one through six tasks, sequential execution and host-selected task workspaces. Parallel branches and multi-file merge semantics require a later explicit design.
- **ASSUMPTION-002**: The input order remains part of the plan hash for compatibility. The scheduler's execution order is a separate record; equivalent DAGs with different input arrays may retain different hashes.

## 8. Related Specifications / Further Reading

Local implementation references: `ntg/planner_agent.py`, `ntg/planned_workflow.py`, `ntg/coding_workflow.py`, `ntg/coder_agent.py`, `ntg/tester_agent.py`, `ntg/reviewer_agent.py`, and `README.md`.

The existing default structured-output contract, workspace path boundaries, local execution policy and bounded repair safeguards remain the baseline. This guide replaces the prior Milestone 11 implementation guide as the next-work specification; the implementation and validation record below describes the delivered milestone.

### Implementation and verification record — 17 October 2026

Implemented TaskGraph, unordered DAG validation, numeric ready-task selection, sequential scheduling, cumulative source handoff, guarded states, dependency-failure propagation, additive graph metadata, topology-aware final verification and CLI graph reporting. Existing repair, execution and review interfaces remain unchanged.

Inline harnesses passed for unordered diamond, chain and multiple roots; duplicate/unknown/self/cyclic/repeated-edge validation; invalid transitions and snapshot isolation; real py_compile and process execution with deterministic Coder/Reviewer doubles; cumulative sibling-source preservation; final result/order/edge tampering; source modification after review; unverified passed results; dependency blocking and fail-fast cancellation; external cancellation and overall timeout; plan/result storage failures and persisted result graphs. A compile-error replay confirmed no functional execution occurred. No permanent test files were added.

Live smoke: python LLMCALL.py emitted liveness, then stopped safely with planning_failed / invalid_plan_output and an empty execution order. No coding task ran. This is not a successful live goal completion; deterministic scheduler and source-handoff checks establish the implemented milestone independently of provider-generated plan quality.

During documentation editing a Windows encoding failure truncated README.md. Its tracked setup/routing/gateway content was restored and its obsolete agent section replaced with the current workflow documentation; the earlier uncommitted README wording could not be restored verbatim.
