---
goal: Implement adaptive multi-file task execution from FEASIBILITY.md
version: 1.0
date_created: 2026-10-17
status: Completed
tags: [context, project-state, orchestration]
---
# Introduction

![Status: Completed](https://img.shields.io/badge/status-Completed-green)

Implement the approved FEASIBILITY.md using the existing agent roles. Preserve the legacy single-file API.

## 1. Requirements & Constraints

- **REQ-001**: Budget provider-facing packets per candidate; adapt context without disabling cross-provider routing.
- **REQ-002**: Support bounded task expansion, ranged source retrieval, guarded multi-file edits, integration gates and restart checkpoints.
- **SEC-001**: Preserve workspace path validation and hash checks; never accept truncated output as source.
- **CON-001**: No new agent roles, external frameworks, services, sandbox or permanent test files.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Establish token admission and project operations.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | Add ntg/budget.py and shared transport packet preparation; distinguish input/context/output metadata and context-size failures in engine.py, discovery.py, failures.py and types.py. | Yes | 2026-10-18 |
| TASK-002 | Extend ntg/project.py with source range/search, tracked manifests and atomic hash-guarded line edits; extend task_graph.py for stable IDs and validated expansion. | Yes | 2026-10-18 |

### Implementation Phase 2

- GOAL-002: Integrate existing roles and durable execution.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-003 | Add project protocols to Planner/Coder/Tester/Reviewer, a project-mode projection to ProjectState, and ntg/project_workflow.py for adaptive serial orchestration and checkpoint restoration. Depends on TASK-001 and TASK-002. | Yes | 2026-10-18 |
| TASK-004 | Update LLMCALL.py to exercise project mode with goal/resume/legacy CLI options; document boundaries and run inline regressions plus live execution. Depends on TASK-003. | Yes | 2026-10-18 |

## 3. Alternatives

- **ALT-001**: Raising context/file limits alone does not support semantic decomposition; rejected.
- **ALT-002**: Parallel writers and additional agents add coordination risk; defer.

## 4. Dependencies

- **DEP-001**: Existing LiteLLM, jsonschema, Python standard library and configured credentials.

## 5. Files

- **FILE-001**: ntg/budget.py, ntg/project.py, ntg/project_workflow.py and existing routing/agent/state modules.
- **FILE-002**: LLMCALL.py, README.md and FEASIBILITY.md.

## 6. Testing

- **TEST-001**: Token budget boundaries, transport-added schema overhead, unknown metadata, context failures and truncation.
- **TEST-002**: 1000-task DAG, parent expansion, cycle rejection, source files larger than context, multi-file manifests and stale edits.
- **TEST-003**: Injected end-to-end multi-hop execution, component integration, rollback, restart validation and legacy compilation/behavior.
- **TEST-004**: Actual LLMCALL.py exit 0 and persisted state/result/project hash agreement.

## 7. Risks & Assumptions

- **RISK-001**: Smaller models can still fail semantic tasks. Runtime evidence, escalation and bounded no-progress rules are required.
- **ASSUMPTION-001**: Python standard-library projects remain the supported execution target; local execution is not a sandbox.

## 8. Related Specifications / Further Reading

[Approved feasibility study](../FEASIBILITY.md)

## Completion evidence

Default expense-tracker run and completed resume passed; final source/result/state revisions agree. See FEASIBILITY.md for exact project identifiers, inline regressions and limits. The default run required debugging and resumption; arbitrary large-project semantic reliability is not claimed.

A second live two-module project passed child tasks, the parent integration gate and final review after resumption. Independent CLI/library behavior and persisted hash/revision checks passed.
