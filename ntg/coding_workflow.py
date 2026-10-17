"""Sequential Coder/Tester/Reviewer coordination with immutable candidate revisions."""
import asyncio
import hashlib
import difflib
import json
from dataclasses import dataclass, field

from .tester_agent import TestReport, TesterAgent
from .project_state import ProjectState
from .context import repair_context as build_repair_context
from .reviewer_agent import ReviewReport
from .types import RoutingError
from .project import validate_entrypoint
from .workspace import Workspace


@dataclass
class RevisionRecord:
    revision: int
    project_id: str
    source_sha256: str = ""
    generation_routing: dict = field(default_factory=dict)
    report: TestReport | None = None
    generation_error: str | None = None
    review: ReviewReport | None = None
    workspace: Workspace | None = field(default=None, repr=False)
    base_source_sha256: str | None = None
    source_snapshot: str | None = None
    changes: str = ""
    failure_type: str | None = None
    failure_signature: str | None = None
    failure_changed: bool | None = None
    repeated_failure: bool = False
    progress: str = ""
    rolled_back: bool = False
    capture_files: list[str] = field(default_factory=list)
    repair_proposal: dict | None = None

    def summary(self):
        return {
            "revision": self.revision, "project_id": self.project_id,
            "source_sha256": self.source_sha256, "base_source_sha256": self.base_source_sha256,
            "source_snapshot": self.source_snapshot, "changes": self.changes,
            "test_status": self.report.status if self.report else None,
            "review_status": self.review.status if self.review else None,
            "generation_error": self.generation_error, "generation_routing": self.generation_routing,
            "failure_type": self.failure_type, "failure_signature": self.failure_signature,
            "failure_changed": self.failure_changed, "repeated_failure": self.repeated_failure,
            "progress": self.progress, "rolled_back": self.rolled_back,
            "capture_files": self.capture_files, "repair_proposal": self.repair_proposal,
        }


@dataclass
class WorkflowResult:
    status: str
    final_project_id: str | None
    revisions: list[RevisionRecord]
    reason: str
    review_required: bool = False

    @property
    def verified(self):
        if self.status != "passed" or not self.revisions:
            return False
        record = self.revisions[-1]
        report = record.report
        tested = (self.final_project_id == record.project_id and report is not None
                  and report.verified and report.project_id == record.project_id
                  and report.source_sha256 == record.source_sha256)
        if not tested or not self.review_required:
            return bool(tested)
        review = record.review
        return (review is not None and review.verified
                and review.project_id == record.project_id
                and review.source_sha256 == record.source_sha256)


class CodingWorkflow:
    RUN_TIMEOUT = 300

    def __init__(self, coder, tester, workspace_manager, max_repairs=2, *, reviewer=None, entrypoint="calculator.py"):
        if type(max_repairs) is not int or not 0 <= max_repairs <= 2:
            raise ValueError("max_repairs must be an integer from zero through two")
        self.entrypoint = validate_entrypoint(entrypoint)
        if any(role.entrypoint != self.entrypoint for role in (coder, tester, reviewer) if role is not None):
            raise ValueError("entrypoint_mismatch")
        self.coder, self.tester, self.workspace_manager = coder, tester, workspace_manager
        self.max_repairs, self.reviewer = max_repairs, reviewer

    def _read(self, workspace):
        if workspace.tools.validate_workspace() != workspace.path:
            raise ValueError("workspace_unavailable")
        read = workspace.tools.execute("read_file", {"path": self.entrypoint})
        if not read["ok"]:
            raise ValueError("source_unavailable")
        source = read["data"]["content"]
        return source, hashlib.sha256(source.encode("utf-8")).hexdigest()

    @staticmethod
    def _save(workspace, path, data):
        try:
            if isinstance(data, str):
                result = workspace.artifacts.execute("write_file", {"path": path, "content": data})
                if not result["ok"]:
                    raise ValueError("revision_storage_failed")
            else:
                workspace.artifacts.write_json(path, data)
        except (OSError, ValueError):
            raise ValueError("revision_storage_failed") from None

    def _persist(self, record, source):
        workspace = record.workspace
        prefix = f"revision_{record.revision:03}"
        if source is not None and record.source_snapshot is None:
            record.source_snapshot = prefix + "_source.txt"
            self._save(workspace, record.source_snapshot, source)
        if record.report is not None:
            self._save(workspace, prefix + "_tester.json", record.report.to_dict(compact=True))
            evidence = [("compile", record.report.compilation)]
            evidence += [(f"check_{i}", c["evidence"]) for i, c in enumerate(record.report.checks, 1)]
            for name, result in evidence:
                if result is None:
                    continue
                for stream in ("stdout", "stderr"):
                    text = result[stream] if isinstance(result, dict) else getattr(result, stream)
                    if text:
                        path = f"{prefix}_{name}_{stream}.txt"
                        self._save(workspace, path, text)
                        record.capture_files.append(path)
        if record.review is not None:
            self._save(workspace, prefix + "_reviewer.json", record.review.to_dict())
        self._save(workspace, prefix + ".json", record.summary())

    @staticmethod
    def _passed(report):
        if report is None:
            return set()
        result = {"compile"} if report.compilation and report.compilation.passed else set()
        from .tester_agent import check_passed
        return result | {c["name"] for c in report.checks if check_passed(c)}

    async def run(self, task, *, context=None, project_state=None):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be a nonempty string")
        if context is not None and not isinstance(context, str):
            raise TypeError("Context must be a string or None")
        owner = project_state or ProjectState.standalone(task, self.reviewer is not None)
        if project_state is None:
            owner.entrypoint = self.entrypoint
        if owner.active is None or owner.active.result.revisions:
            raise ValueError("invalid_state_transition")
        history = owner.active.result.revisions
        seen_sources, seen_failures = set(), set()
        workspace, active_report, last_attempt, rejection = None, None, None, None
        source = None

        def persist(record, candidate):
            self._persist(record, candidate)
            owner.revision_finished(record, self.entrypoint)

        def record_interruption(reason):
            if history and not history[-1].progress:
                record = history[-1]
                record.progress, record.failure_type = "interrupted", reason
                record.failure_changed = (history[-2].failure_type != reason) if len(history) > 1 else None
                try:
                    persist(record, source)
                except (ValueError, OSError):
                    pass  # An unavailable/full workspace cannot promise durable evidence.

        def finish(status, project_id, reason):
            # Canonical reports describe the active source; rejected proposals have numbered reports.
            if workspace is not None and active_report is not None:
                try:
                    if self._read(workspace)[1] == active_report.source_sha256:
                        self._save(workspace, "tester_report.json", active_report.to_dict(compact=True))
                    if status == "passed" and history[-1].review is not None:
                        self._save(workspace, "reviewer_report.json", history[-1].review.to_dict())
                except ValueError:
                    status, project_id, reason = "blocked", None, "report_storage_failed"
            return WorkflowResult(status, project_id, history, reason, self.reviewer is not None)

        timeout = asyncio.timeout(self.RUN_TIMEOUT)
        try:
            async with timeout:
                workspace = self.workspace_manager.create()
                for index in range(self.max_repairs + 1):
                    record = RevisionRecord(index + 1, workspace.project_id, workspace=workspace)
                    base_source, base_hash = self._read(workspace) if index else (None, None)
                    source = base_source
                    record.base_source_sha256 = base_hash
                    record.source_sha256 = base_hash or ""
                    previous_report = active_report
                    owner.register_revision(owner.current_task, record)
                    try:
                        if index == 0:
                            generated = await self.coder.run(task, context)
                        else:
                            repair_context = build_repair_context(
                                context, base_hash, self.tester.describe_checks(), active_report,
                                last_attempt, history[:-1], rejection)
                            generated = await self.coder.repair(task, base_source, repair_context)
                    except RoutingError as exc:
                        record.generation_error = exc.code
                        record.failure_type, record.progress = "generation_error", "provider_failure"
                        record.failure_changed = (last_attempt.failure_type != record.failure_type) if last_attempt else None
                        record.generation_routing = {"history": exc.history}
                        persist(record, base_source)
                        return finish("failed", None, "generation_failed")
                    except ValueError as exc:
                        if str(exc) in ("context_budget_exceeded", "state_storage_failed", "state_conflict"):
                            raise
                        record.generation_error = str(exc)
                        record.repair_proposal = getattr(exc, "repair_proposal", None)
                        record.failure_type = "invalid_repair" if index else "invalid_generation"
                        record.failure_changed = (last_attempt.failure_type != record.failure_type) if last_attempt else None
                        record.failure_signature = hashlib.sha256(str(exc).encode()).hexdigest()
                        record.repeated_failure = record.failure_signature in seen_failures
                        record.progress = "rejected_patch"
                        persist(record, base_source)
                        if not index:
                            return finish("failed", None, "generation_failed")
                        if record.repeated_failure:
                            return finish("failed", None, "repair_stalled")
                        seen_failures.add(record.failure_signature)
                        rejection = {"error": str(exc), "proposal": record.repair_proposal,
                                     "instruction": "Use supplied anchor IDs against the unchanged current file. Correct the rejected proposal while preserving unrelated lines."}
                        last_attempt = record
                        continue
                    source = generated.answer
                    record.generation_routing = generated.routing
                    record.source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
                    if base_source is not None:
                        record.changes = "\n".join(difflib.unified_diff(
                            base_source.splitlines(), source.splitlines(),
                            fromfile="before/" + self.entrypoint, tofile="after/" + self.entrypoint, lineterm="",
                        ))
                    if record.source_sha256 in seen_sources:
                        record.progress, record.failure_type = "stalled", "source_cycle"
                        record.failure_changed = (last_attempt.failure_type != record.failure_type) if last_attempt else None
                        persist(record, source)
                        return finish("failed", None, "repair_cycle_detected")
                    seen_sources.add(record.source_sha256)
                    if index == 0:
                        self._save(workspace, self.entrypoint, source)
                    else:
                        workspace.tools.replace_file(self.entrypoint, source, base_hash)
                    record.source_snapshot = f"revision_{record.revision:03}_source.txt"
                    self._save(workspace, record.source_snapshot, source)
                    if isinstance(self.tester, TesterAgent):
                        report = await self.tester.run(workspace, {
                            "entrypoint": self.entrypoint, "source_sha256": record.source_sha256},
                            stage_hook=owner.set_stage)
                    else:
                        owner.set_stage("compiling")
                        report = await self.tester.run(workspace, task)
                        if report.compilation and report.compilation.passed:
                            owner.set_stage("testing")
                    record.report = report
                    if report.project_id != record.project_id or report.source_sha256 != record.source_sha256:
                        record.failure_type, record.progress = "artifact_mismatch", "blocked"
                        persist(record, source)
                        return finish("blocked", None, "test_artifact_mismatch")
                    record.failure_type, record.failure_signature = report.failure_type, report.failure_signature
                    if report.verified and self.reviewer is not None:
                        owner.set_stage("reviewing")
                        review = await self.reviewer.run(workspace, task, report)
                        record.review = review
                        if review.status not in ("approved", "rejected", "blocked"):
                            persist(record, source)
                            return finish("blocked", None, "invalid_review_output")
                        if review.project_id != record.project_id or review.source_sha256 != record.source_sha256:
                            record.failure_type, record.progress = "artifact_mismatch", "blocked"
                            persist(record, source)
                            return finish("blocked", None, "review_artifact_mismatch")
                        if review.status == "rejected":
                            record.failure_type = "review_rejected"
                            record.failure_signature = hashlib.sha256(json.dumps(review.issues, sort_keys=True).encode()).hexdigest()
                    record.failure_changed = (record.failure_type != last_attempt.failure_type) if last_attempt else None
                    record.repeated_failure = bool(record.failure_signature and record.failure_signature in seen_failures)
                    regressions = sorted(self._passed(previous_report) - self._passed(report))
                    if regressions and report.status != "blocked":
                        workspace.tools.replace_file(self.entrypoint, base_source, record.source_sha256)
                        record.rolled_back, record.progress = True, "regression_rejected"
                        rejection = {"source": source, "regressed_checks": regressions,
                                     "instruction": "Candidate rolled back. Repair the current file; do not reapply the rejected proposal."}
                    else:
                        active_report, rejection = report, None
                        record.progress = "checks_improved" if previous_report and report.score > previous_report.score else "initial_candidate"
                    stalled = record.repeated_failure and (not previous_report or report.score <= previous_report.score)
                    no_progress = (previous_report is not None and not record.rolled_back
                                   and not report.verified and report.score <= previous_report.score)
                    if (stalled or no_progress) and base_source is not None and not record.rolled_back:
                        workspace.tools.replace_file(self.entrypoint, base_source, record.source_sha256)
                        active_report, record.rolled_back = previous_report, True
                    if stalled or no_progress:
                        record.progress = "stalled" if stalled else "no_observed_progress"
                    if report.verified and (self.reviewer is None or record.review.verified):
                        record.progress = "verified"
                    persist(record, source)
                    if report.status == "blocked":
                        return finish("blocked", None, report.feedback)
                    if record.review is not None and record.review.status == "blocked":
                        return finish("blocked", None, record.review.error or "review_unavailable")
                    if report.verified and (self.reviewer is None or record.review.verified):
                        return finish("passed", workspace.project_id, "tested_and_reviewed" if self.reviewer else "verified")
                    if record.progress == "stalled":
                        return finish("failed", None, "repair_stalled")
                    if no_progress:
                        return finish("failed", None, "no_observed_progress")
                    if record.failure_signature:
                        seen_failures.add(record.failure_signature)
                    last_attempt = record
                return finish("failed", None, "repair_limit_exceeded")
        except ValueError as exc:
            if str(exc) not in ("workspace_unavailable", "workspace_allocation_failed", "source_unavailable",
                                "source_changed", "revision_storage_failed", "path_not_allowed",
                                "context_budget_exceeded", "state_storage_failed", "state_conflict"):
                raise
            record_interruption(str(exc))
            return finish("blocked", None, str(exc))
        except asyncio.CancelledError:
            record_interruption("cancelled")
            raise
        except OSError:
            record_interruption("workspace_io_error")
            return finish("blocked", None, "workspace_io_error")
        except TimeoutError:
            if not timeout.expired():
                raise
            record_interruption("workflow_deadline_exceeded")
            return finish("failed", None, "workflow_deadline_exceeded")
