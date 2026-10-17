---
goal: Add revision-aware Python repository intelligence to the existing project workflow
version: 1.0
date_created: 2026-10-18
status: Completed
tags: [milestone-15, repository-intelligence, ast, context]
---

# Introduction

![Status: Completed](https://img.shields.io/badge/status-Completed-green)

Milestone 15 gives existing agents searchable information about project files, classes, functions, imports and calls. Build a small Python AST index, then retrieve exact relevant source through existing bounded read operations. The index is derived evidence; source hashes, tests, review and ProjectState remain authoritative.

This replaces the historical Milestone 14 guide. The index and agent integration are implemented. The verification record below distinguishes deterministic checks, live model operations and complete workflow runs.

```text
Tracked ProjectFiles + current manifest
                 |
        File inventory / hashes
                 |
         ast.parse + visitor
                 |
   Symbols + imports + call sites
                 |
 Conservative relationship resolution
                 |
 Searchable, revision-bound local index
                 |
 Bounded agent query -> exact source read
                 |
 Existing edit -> compile -> test -> review
```

## 1. Requirements & Constraints

### Current baseline

LLMCALL.py defaults to ProjectWorkflow.run in ntg/project_workflow.py. It already supports multiple tracked files, hierarchical tasks, serial candidate edits, integration gates, compile/test/review and validated resumption. Preserve this; do not restore the old six-task/single-file restriction.

ProjectFiles in ntg/project.py owns paths, runtime_paths, manifest(), revision(), listing(), read(), search(), edit() and clone_to(). Existing search is literal source-text search, not a symbol/dependency index.

ProjectWorkflow._observe dispatches source/artifact/requirement/task reads and search/list. _packet builds bounded context; _evidence enforces read-only review and changed-line inspection. These are the integration seams. tools/registry.py is not the default project-mode dispatcher.

ntg/project_protocol.py defines STEP and separate Coder/Reviewer action allowlists. ProjectState schema 2 stores source identities and evidence references. ntg/budget.py budgets provider-facing packets. Reuse these mechanisms.

Verified historical baselines: workspaces/project_ee0095a386724cc08db03a87d02f283b/result.json (expense tracker) and workspaces/project_d9a635e45402489ba9512144cc9bbe51/result.json (two modules). Preserve historical artifacts and the --legacy path.

- **REQ-001**: Index only active ProjectFiles.paths. Record tracked paths, hashes, sizes and language; parse .py files only. Exclude runtime_paths. Never crawl the NTG installation, .env, virtual environments, workspace siblings or host metadata.
- **REQ-002**: Extract classes, sync/async functions, nested definitions, methods, imports and call sites with ast. Preserve lexical ownership and exact source locations. Never execute/import indexed code.
- **REQ-003**: Resolve local imports, aliases, relative imports and unambiguous lexical calls conservatively. Mark external, ambiguous and dynamic relationships explicitly unresolved. A syntactic call is not a proven runtime edge.
- **REQ-004**: Provide deterministic metadata search and incoming/outgoing relationship queries with pagination, hashes and project revision. Agents receive selected records rather than the entire index.
- **REQ-005**: Reuse per-file extraction by content hash and parser version. Reparse changed/new files, remove deleted records, and recompute relationship resolution when the manifest changes. Candidate indexes must reflect candidate source.
- **REQ-006**: Index hits lead to exact ProjectFiles.read operations. Metadata, signatures and docstrings do not satisfy source inspection or changed-range review requirements.
- **REQ-007**: Parse/encoding failures produce file diagnostics and remove that file's old semantic records from the current view. Continue with other files. Existing compile validation remains authoritative.
- **REQ-008**: Persist an optional hash-verified index artifact reference identifying source revision and index schema. Missing/stale/invalid derived caches rebuild without discarding valid approvals or falsely completing tasks.
- **SEC-001**: Use existing guarded source/storage APIs and regular-file validation. Verify source identities before/after extraction. Reject traversal, linked escapes, foreign paths and files outside the tracked project.
- **SEC-002**: Source/docstrings/index metadata are untrusted data, never system instructions. Index artifacts are host-owned; agents cannot write them or project_state.json through query operations.
- **CON-001**: Add one focused module, ntg/repository.py. Use standard-library ast, tokenize, hashlib and existing helpers. No embeddings, vector database, language server, parser package, service, framework, agent, sandbox or permanent test files.
- **CON-002**: Preserve routing/fallback, context budgets, edit guards, task scopes, compile/test evidence, integration, review and repeatability checks. The code dependency graph is separate from TaskGraph scheduling.
- **CON-003**: Query limits are prompt/computation budgets, not artifact storage ceilings. Partial responses expose continuation and coverage diagnostics. Never label incomplete analysis complete.

### Data model

Use JSON-compatible dictionaries and one RepositoryIndex class. Persist extracted records, not AST objects or source bodies:

```text
schema_version: 1
parser_version: host-selected extraction version
python_version: interpreter major/minor
project_revision: ProjectFiles.revision(manifest)
manifest: relative path -> SHA-256
files: path -> {sha256, size_bytes, language, parse_status,
                symbols, imports, calls, diagnostics}
relationships: resolved/unresolved import and call references
coverage: {inventory_count, python_count, parsed_count, diagnostic_count}
```

Symbols store id, path, source_sha256, name, qualified_name, kind, owner_id, lineno, end_lineno, col_offset and end_col_offset. IDs combine path, lexical qualified name and definition line/column, and must be paired with source hash. Moving a definition can change its ID. Include decorators in the retrievable source range. Distinguish duplicate/nested definitions and methods through lexical ownership and location.

Keep signatures compact: parameter names/kinds and bounded annotations/default representations, not complete bodies. An optional first docstring line has an explicit truncation marker. AST columns are UTF-8 byte offsets, not Python string indices.

Imports store module, imported name, alias, relative level, owner and line range. Calls store enclosing symbol/module, bounded syntactic callee text, source range, resolution status, optional target_id and reason. Do not capture full argument values or unrelated source.

### Relationship rules

Map pkg/mod.py to pkg.mod and pkg/__init__.py to pkg. Resolve relative imports from the importing package and level. Handle tracked namespace-package directories conservatively. Detect module/package name collisions rather than silently selecting a target. External imports remain references; do not download dependencies.

Resolve a direct call only when the lexical/imported target is unique and no relevant rebinding or shadowing makes it uncertain. Resolve module_alias.function only when the alias and local target are unambiguous. Function parameters/local assignments can shadow imports or definitions and must be considered.

Record instance.method, self.method, factory()(), getattr, dynamic imports, monkey patches, conditional bindings, wildcard imports and arbitrary re-export chains conservatively as unresolved where certainty is unavailable. Keep syntactic evidence and a reason. Do not guess targets by matching method names globally.

Cycles are valid index information. app -> helpers -> app must terminate without changing TaskGraph. These relationships do not prove runtime reachability, execution order or correctness.

### Minimal interface

```python
class RepositoryIndex:
    def refresh(self, project): ...
    def search(self, query, *, path=None, offset=0, limit=10): ...
    def dependencies(self, path, *, offset=0, limit=10): ...
    def snapshot(self): ...
    @classmethod
    def from_snapshot(cls, snapshot): ...
```

refresh synchronizes against the verified current manifest and returns coverage. The class receives no router credentials or live ProjectState.

search uses case-insensitive literal matching over paths, symbol names/qualified names and import/callee names. Rank exact names first, then qualified-name matches, prefixes and substrings; break ties by path, line, kind and id. Defer fuzzy/semantic search.

dependencies returns flat records labelled direction, relationship kind and resolution status: outgoing imports/calls and incoming resolved references for the requested file. Include unresolved outgoing references; do not claim complete incoming dynamic-call coverage.

Queries return project_revision, records, next_offset and coverage/diagnostics. Records include path/hash and source ranges where applicable. Default page size is 10, maximum 20; query length maximum 200 characters. Also respect existing _chars/model response budgets. Explicitly mark omitted metadata and continuation; never silently drop results. A continuation from an older index revision returns index_revision_changed and restarts at offset 0.

### Agent protocol

Add repo_search and repo_dependencies to STEP and the Coder/Reviewer read-only action allowlists:

```json
{
  "action": "repo_search",
  "path": null,
  "start": 0,
  "end": null,
  "sha256": null,
  "text": "add",
  "note": "Locate the definition and import references",
  "issues": []
}
```

repo_search uses text as a literal query, optional path as a tracked-file filter and start as zero-based offset. repo_dependencies requires path, uses start as offset and text=null. Unused fields are null. Validate types/ranges on the host. Record page revision alongside continuation in host observations, and reject pagination after source changes without trusting a model-supplied revision.

Example: repo_search("add") -> arithmetic.py definition/app.py import -> read arithmetic.py exact lines -> hash-guarded edit -> refresh -> compile/test -> repo_dependencies("arithmetic.py") -> inspect callers. Source paths stay plain; artifact: addresses saved host reports only.

## 2. Implementation Steps

### Implementation Phase 1

- GOAL-001: Build guarded inventory and deterministic per-file AST extraction.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add RepositoryIndex in ntg/repository.py. Inventory ProjectFiles.paths/manifest excluding runtime_paths. Read guarded regular-file bytes, detect Python encoding with tokenize, hash bytes and verify against the manifest. Retry a racing read once; then return source_changed without publishing mismatched records. | Yes | 2026-10-18 |
| TASK-002 | Depends on TASK-001. Implement an ast.NodeVisitor retaining module/class/function scope, nested/async definitions, decorators, imports and calls. Catch SyntaxError, UnicodeError, LookupError, ValueError and RecursionError as per-file diagnostics. Use deterministic IDs and bounded metadata; execute no source. | Yes | 2026-10-18 |

Phase exit: extraction matches temporary fixtures, including aliases, nested scopes, async functions and invalid Python. Every record has current identity and exact location.

### Implementation Phase 2

- GOAL-002: Make metadata searchable and incrementally correct.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-003 | Depends on TASK-002. Implement module/import resolution, shadow-aware conservative direct-call resolution, incoming/outgoing relationships and unresolved reasons. Ordinary queries must not recursively traverse cycles. | Yes | 2026-10-18 |
| TASK-004 | Depends on TASK-003. Implement refresh/search/dependencies/snapshot/from_snapshot. Cache extraction by hash/parser version, invalidate changed/deleted files, rebuild relationship resolution after inventory changes and sort/page deterministically. Validate cached paths, hashes and schema before reuse. | Yes | 2026-10-18 |

Phase exit: edits/deletions/syntax errors cannot expose stale definitions or edges. Identical revision/query pairs produce identical pages. Workspaces share no mutable relationship state.

### Implementation Phase 3

- GOAL-003: Integrate bounded queries with existing agents and state.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Depends on TASK-004. Extend ntg/project_protocol.py role schemas, CoderAgent.project_step, ReviewerAgent.review_project and ProjectWorkflow._observe/_evidence with read-only queries. Their metadata does not count as actual source inspection; keep all changed-range and action authorization checks. | Yes | 2026-10-18 |
| TASK-006 | Depends on TASK-005. Manage index lifecycle in ProjectWorkflow._new/_resume/_execute_task, successful edit handling, _packet and approval/rollback. Refresh on the actual candidate before queries. Add compact revision/coverage hints and bounded metadata ranked by task paths and literal identifiers. Reuse existing _step and ntg/budget.py adaptation; never serialize the entire index. | Yes | 2026-10-18 |
| TASK-007 | Depends on TASK-006. Save snapshots using _record. Store an optional project.repository_index reference with artifact hash/source revision/schema; active candidate references stay under active state until approval. Modify ntg/project_state.py only if validation/projection needs it. Old schema-2 states without the field remain valid. Rebuild bad derived caches while retaining strict approval/source/test artifact verification. | Yes | 2026-10-18 |

Phase exit: agents find definitions, read exact code, edit, requery callers and resume with current results. Tester remains deterministic. Planner may receive a bounded approved-source inventory summary when source exists; empty initial projects remain valid.

### Implementation Phase 4

- GOAL-004: Verify actual repository-aware execution and document boundaries.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-008 | Depends on TASK-007. Run all inline cases in section 6, then real LLMCALL.py default and multi-file runs. Inspect repository operations, source hashes and final state/result. Fix workflow defects rather than application files manually. Record capacity/time pauses honestly. | Yes | 2026-10-18 |
| TASK-009 | Depends on TASK-008. Update README.md with queries, static-analysis limits, cache/resume behavior and evidence. Mark this guide Completed only after acceptance passes. Preserve FEASIBILITY.md/history. No new CLI mode or dependency is required. | Yes | 2026-10-18 |

## 3. Alternatives

- **ALT-001**: Recommended: one in-process AST index, JSON snapshots and literal metadata search backed by exact source reads. Fits existing serial orchestration and host state.
- **ALT-002**: Reparse everything before every prompt. Defer to initial construction/cache recovery only; repeated full parsing and prompt injection waste resources.
- **ALT-003**: Embeddings/vector database/LSP/type-driven call graphs. Defer until simpler search has measured shortcomings; these add dependencies and complexity beyond this milestone.

## 4. Dependencies

- **DEP-001**: Python >=3.11 ast/tokenize and existing guarded ProjectFiles/ArtifactFiles interfaces. No pyproject.toml package change.
- **DEP-002**: ProjectWorkflow, schema-2 state, structured role validation, token admission and source/test/review hashes. Existing inline/live baselines remain valid.

## 5. Files

- **FILE-001**: New ntg/repository.py: extraction, resolution, refresh, queries and snapshot validation.
- **FILE-002**: ntg/project_workflow.py: candidate lifecycle, dispatch, bounded context metadata and host artifact references.
- **FILE-003**: ntg/project_protocol.py, ntg/coder_agent.py and ntg/reviewer_agent.py: explicit read-only actions and relevant-source instructions.
- **FILE-004**: ntg/project_state.py and ntg/__init__.py only if validation/public exports need additions. ntg/project.py remains authoritative for source operations; avoid unrelated legacy changes.
- **FILE-005**: README.md and guide.md: usage/status/limits/evidence. LLMCALL.py uses the integrated workflow unchanged unless minimal reporting needs an addition.

## 6. Testing

Use inline Python harnesses and TemporaryDirectory fixtures; add no permanent test files. Injected responses verify orchestration; real provider runs separately verify model use.

- **TEST-001**: Extract top-level/nested/decorated/async functions, classes/methods, aliases/imports and calls. Assert ownership, unique IDs and exact source spans. Include non-ASCII code and valid encoding cookies.
- **TEST-002**: Resolve two-module from arithmetic import add and package-relative/aliased imports. Shadowed names, dynamic/instance calls, wildcard imports and module collisions remain unresolved with reasons. Import cycles terminate deterministically.
- **TEST-003**: Rename/edit a definition; old symbols/targets disappear and new records match hashes. Unchanged files are not reparsed. Remove tracked files and change local/external resolution through inventory changes; no stale edges survive.
- **TEST-004**: Invalid syntax, invalid encoding and parser recursion failures produce diagnostics, not stale records. Other files stay searchable. py_compile still returns actual syntax failures to repairs. Partial index coverage cannot mark code correct.
- **TEST-005**: Reject traversal, foreign paths, linked escapes, unknown files and untracked .env/metadata access. Indexing has no import side effects. Source/index text cannot grant edit rights or modify host artifacts.
- **TEST-006**: Page over 1,000 synthetic functions/files without putting all records in a prompt. Verify stable sorting, response budgets, continuation, coverage, stale-cursor rejection and current revision. This tests indexing, not semantic success on 1,000 tasks.
- **TEST-007**: Inject queries -> exact reads -> edit -> refreshed queries -> review. Metadata cannot bypass source inspection, stale hashes fail, candidate relationships stay separate until approval and rollback restores approved relationships. Reuse parse records across identical clones without mutable edge leakage.
- **TEST-008**: Resume with valid/missing/stale/corrupt optional index caches. Existing schema-2 checkpoints work; derived caches rebuild. Tampered approved source or required approval/test artifacts remain rejected. Completed resume makes no new model calls.
- **TEST-009**: Run python LLMCALL.py successfully with all existing default gates. Separately run a live goal creating arithmetic.py/add and app.py importing it and printing 5; explicitly request finding the definition and caller through repository queries before final review. Verify actual repo_search/repo_dependencies operations and current-hash source reads in artifacts, two source modules, correct execution and matching state/result/manifest identities. A green run without repository-query evidence does not prove this milestone. Resumption is allowed and must be disclosed.
- **TEST-010**: Run system syntax checks and git diff --check; preserve legacy behavior through the existing injected baseline. Record fixture parsed-file/cache-reuse counts, result sizes and unresolved relationships. Never expose keys or credential-bearing requests.

Completion requires all checks and live repository-query evidence. Change Planned -> In progress at implementation start, then Completed only after verification. API capacity failures leave live verification pending; fixtures alone cannot establish completion.

## 7. Risks & Assumptions

- **RISK-001**: Python is dynamic. This index supplies syntactic evidence, not a precise runtime call graph or guaranteed semantic understanding. Unknown targets remain unknown.
- **RISK-002**: Parsing needs memory for one source file. Surface parser/resource diagnostics and explicit partial coverage; do not truncate source before parsing. Avoid retrying the same failed hash repeatedly within a run. Catchable errors do not make catastrophic process memory exhaustion safe. Keep text search/source reads available.
- **RISK-003**: Edits can invalidate relationships and cursors. Synchronize before queries and identify returned revisions/hashes. Metadata never authorizes a patch or approval by itself.
- **RISK-004**: Extra metadata can waste context. Keep summaries bounded, page results and retain packet adaptation. Evaluate retrieval quality with actual tasks rather than assuming AST indexing guarantees better reasoning.
- **ASSUMPTION-001**: Starting scope is the active tracked Python project. Importing an arbitrary existing checkout needs a separate host-controlled inventory policy; do not automatically crawl NTG or all workspaces.
- **ASSUMPTION-002**: Existing agents, serial execution and local unsandboxed Python checks remain. Queries add no external research or shell access.

## 8. Related Specifications / Further Reading

- [Adaptive implementation and verification](FEASIBILITY.md)
- [Completed adaptive-workflow plan](plan/feature-adaptive-project-workflow-1.md)
- [Current project workflow](ntg/project_workflow.py)
- [Tracked source interface](ntg/project.py)
- [Structured role protocol](ntg/project_protocol.py)
- [Shared project state](ntg/project_state.py)

The milestone is complete when agents can locate relevant code, inspect local relationships, retrieve exact source, change it safely and receive refreshed repository information without loading the entire codebase into each prompt.

## Implementation and verification record (2026-10-18)

Implemented ntg/repository.py and exported RepositoryIndex from ntg. ProjectWorkflow owns candidate-aware index refresh, bounded queries and optional artifact references; STEP exposes read-only queries to Coder/Reviewer. Existing source-read, compile/test, review, task-graph and routing gates remain. No new dependency, service, agent, sandbox or permanent test file was added.

Inline verification passed for extraction/locations, async/nested scopes, aliases and relative imports, parameter/pattern/attribute shadowing, ambiguous module/package names, import cycles, Unicode/encoding failures, parser recursion diagnostics, unchanged-hash reuse, edit/delete/syntax invalidation, candidate rollback, stale cursors, query response budgets, missing/corrupt cache recovery, source tamper rejection, actual Windows junction escape rejection, no import side effects, role action schemas and changed-range review enforcement. A 1,000-function fixture produced paginated bounded results; an unchanged five-file fixture reused all five files without reparsing. This measures indexing, not large-project semantic reliability. Legacy PlannedWorkflow also passed compile/execution/review and schema-1 persistence through injected responses.

Live multi-file run: workspaces/project_e735ec37fa2849b1824f110e65ceaf2e/result.json passed a one-task two-module project with real Planner/Coder/Reviewer responses. Final source project_3eb4d44868134fada4329a3464746b88 has revision e6a111dda9c07e35a2966e28d7abfdc7a6a2e9badef6c3a0a8cb47ddd7038e5f. The initial workflow used bounded source reads and did not itself choose repository queries. A subsequent explicit live Reviewer verification in the same host workflow requested repo_search and repo_dependencies, followed by exact reads of both modules; fresh compilation/execution passed. Persisted repository_query artifacts contain three search records and two dependency records, including app.py incoming references, all bound to the final revision. This was host-directed query verification, not evidence that models always choose queries autonomously. Completed resumption rebuilt the final parser-version index without new model calls and preserved source/result/state identities.

Historical default expense-tracker project_ee0095a386724cc08db03a87d02f283b also resumed successfully with LLMCALL.py exit 0 and passed a fresh host integration check. Fresh default verification first exposed recursive scaffolding planning; Planner instructions now explicitly keep package initialization within concrete module leaves and restrict decomposition to the parent contract. A second fresh run paused on upstream invalid_request responses, then progressed after resumption. The run subsequently completed after resumption, including two focused repairs and final review. Capacity/safety limits were not bypassed, and generated application files were not edited manually.


Final default workflow verification: workspaces/project_d951217768894e98b312d6c71f9e06da/result.json is passed/verified; LLMCALL.py --resume exited 0 with T1 through T5 passed. Final source project_faa4ced5643a489b9bbaa8fbe05564b7 has revision bbe6425d8e772ac6dc28aa6e2879036b6a6646828b2ccdea378477128c60d9dd. A fresh host expense integration check passed. Completed resumption used zero model calls, validated approvals and preserved source/result/state revision agreement. The persisted index matches that source revision and reports two inventoried/parsed Python files with zero diagnostics. This was a recovered run, not uninterrupted first-attempt success. Earlier failed attempts remain available as evidence.

The verification also exposed an evidence-retrieval inefficiency: small character reads of final-review artifacts could omit the actual issues. Such reads now include bounded, explicitly excerpted real review findings; an inline regression verified this without altering generated application code. The implementation retains unknown static relationships and explicit partial coverage rather than inventing dependencies or treating index metadata as correctness proof. All system files parsed and git diff --check passed. Milestone 15 is Completed.
