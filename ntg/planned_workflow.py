"""Dependency scheduling over cumulative, approved single-file snapshots."""
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from copy import deepcopy

from .coding_workflow import WorkflowResult
from .planner_agent import PlanResult
from .types import RoutingError
from .project_state import ProjectState
from .context import coder_context, researcher_context
from .researcher_agent import ResearchResult


@dataclass
class TaskExecution:
    task_id: str
    status: str
    result: WorkflowResult
    approved_source_sha256: str | None = None
    research: ResearchResult | None = None
    research_sha256: str | None = None
    research_questions: list[str] = field(default_factory=list)
    research_artifact: str | None = None

    @property
    def research_verified(self):
        if not self.research_questions:
            return (self.research is None and self.research_sha256 is None
                    and self.research_artifact is None)
        return (self.research is not None and self.research.verified
                and self.research.task_id == self.task_id
                and self.research.questions == self.research_questions
                and self.research_artifact == f"research_{self.task_id}.json"
                and self.research_sha256 == hashlib.sha256(json.dumps(
                    self.research.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")).hexdigest())


    @property
    def verified(self):
        return (self.status == "passed" and self.research_verified and self.result.review_required
                and self.result.verified and self.approved_source_sha256 is not None
                and self.approved_source_sha256 == self.result.revisions[-1].source_sha256)

    def to_dict(self):
        return {
            "task_id": self.task_id, "status": self.status,
            "research": self.research.to_dict() if self.research else {"status": "skipped"},
            "research_sha256": self.research_sha256,
            "research_artifact": self.research_artifact,
            "approved_source_sha256": self.approved_source_sha256,
            "final_project_id": self.result.final_project_id,
            "reason": self.result.reason,
            "verified": self.verified,
            "revisions": [r.summary() for r in self.result.revisions],
        }


@dataclass
class PlannedResult:
    status: str
    goal: str
    plan: PlanResult | None = None
    run_project_id: str | None = None
    tasks: list[TaskExecution] = field(default_factory=list)
    final_project_id: str | None = None
    final_source_sha256: str | None = None
    reason: str = ""
    planning_error: dict | None = None
    task_graph: dict = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    project_state_path: str | None = None
    state_revision: int | None = None

    @property
    def verified(self):
        if self.status != "passed" or self.plan is None or not self.tasks:
            return False
        try:
            self.plan.validate()
        except ValueError:
            return False
        ids = [t.task_id for t in self.tasks]
        planned = {t["id"]: t for t in self.plan.tasks}
        if (self.goal != self.plan.goal or len(ids) != len(set(ids))
                or set(ids) != set(planned) or ids != self.execution_order
                or set(self.task_graph) != set(planned)):
            return False
        completed = set()
        for task in self.tasks:
            node = self.task_graph.get(task.task_id)
            deps = planned[task.task_id]["dependencies"]
            if (not isinstance(node, dict) or node.get("state") != "passed"
                    or node.get("reason") is not None
                    or node.get("dependencies") != deps
                    or not set(deps) <= completed or not task.verified
                    or task.research_questions != planned[task.task_id].get("research_questions", [])
                    or (task.research is not None and task.research.plan_sha256 != self.plan.plan_sha256)):
                return False
            completed.add(task.task_id)
        last = self.tasks[-1]
        return (self.final_project_id == last.result.final_project_id
                and self.final_source_sha256 == last.approved_source_sha256)

    def to_dict(self):
        return {
            "status": self.status, "goal": self.goal,
            "plan_sha256": self.plan.plan_sha256 if self.plan else None,
            "run_project_id": self.run_project_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "final_project_id": self.final_project_id,
            "final_source_sha256": self.final_source_sha256,
            "reason": self.reason, "verified": self.verified,
            "planning_error": self.planning_error,
            "task_graph": self.task_graph, "execution_order": self.execution_order,
            "project_state_path": self.project_state_path, "state_revision": self.state_revision,
        }


class PlannedWorkflow:
    RUN_TIMEOUT = 1200

    def __init__(self, planner, workflow, *, researcher=None):
        if (workflow.reviewer is None or workflow.entrypoint != "app.py"
                or type(workflow.max_repairs) is not int or not 0 <= workflow.max_repairs <= 2):
            raise ValueError("unsupported_planned_workflow_configuration")
        self.planner, self.workflow = planner, workflow
        self.researcher = researcher

    @staticmethod
    def _store(workspace, path, data):
        try:
            workspace.artifacts.write_json(path, data)
            return True
        except (OSError, ValueError) as exc:
            from tools.filesystem import ToolError
            code = exc.code if isinstance(exc, ToolError) else (
                "disk_full" if getattr(exc, "errno", None) == 28 else "io_error" if isinstance(exc, OSError) else "invalid_json")
            raise ValueError("metadata_storage_failed_" + path.replace(".", "_") + "_" + code) from None

    def _approved_source(self, result):
        if not result.verified:
            raise ValueError("approved_artifact_unavailable")
        record = result.revisions[-1]
        workspace = record.workspace
        if (workspace is None or workspace.project_id != record.project_id
                or workspace.tools.validate_workspace() != workspace.path):
            raise ValueError("approved_artifact_unavailable")
        read = workspace.tools.execute("read_file", {"path": self.workflow.entrypoint})
        if not read["ok"]:
            raise ValueError("approved_artifact_unavailable")
        source = read["data"]["content"]
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != record.source_sha256:
            raise ValueError("approved_artifact_unavailable")
        return source, digest

    @staticmethod
    def _result(project):
        return PlannedResult(
            status=project.status, goal=project.goal, plan=deepcopy(project.plan),
            run_project_id=project.metadata.project_id if project.metadata else None,
            tasks=deepcopy(project.tasks), final_project_id=project.final_project_id,
            final_source_sha256=project.final_source_sha256,
            reason=project.reason or "all_tasks_tested_and_reviewed",
            planning_error=deepcopy(project.planning_error),
            task_graph=project.graph.to_dict() if project.graph else {},
            execution_order=project.execution_order,
            project_state_path="project_state.json" if project.metadata else None,
            state_revision=project.state_revision,
        )

    async def run(self, goal):
        project = ProjectState(goal)
        self.project_state = project  # Read-only progress access; execution uses local ownership.
        cancelled = False
        timeout = asyncio.timeout(self.RUN_TIMEOUT)
        try:
            async with timeout:
                project.bind(self.workflow.workspace_manager.create())
                try:
                    plan = await self.planner.run(project.goal)
                    plan.validate()
                    if plan.goal != project.goal:
                        raise ValueError("invalid_plan_output")
                except RoutingError as exc:
                    project.planning_error = exc.public()
                    project.add_error(exc.code)
                    raise ValueError("planning_failed") from None
                except ValueError as exc:
                    project.planning_error = {"code": str(exc)}
                    project.add_error(str(exc))
                    if hasattr(exc, "validation_reason"):
                        project.planning_error["validation_reason"] = exc.validation_reason
                    raise ValueError("planning_timeout" if str(exc) == "planning_timeout" else "planning_failed") from None
                project.attach_plan(plan)
                if not self._store(project.metadata, "plan.json", plan.to_dict()):
                    raise ValueError("plan_storage_failed")
                project.seal(project.metadata, "plan.json")
                while not all(s == "passed" for s in project.graph.states.values()):
                    project.plan.validate()
                    ready = project.graph.ready()
                    if not ready:
                        raise ValueError("scheduler_no_progress")
                    task = ready[0]
                    source, digest = project.approved_source()
                    execution = TaskExecution(task["id"], "failed", WorkflowResult(
                        "failed", None, [], "execution_interrupted", review_required=True),
                        research_questions=list(task.get("research_questions", [])))
                    project.start_task(task["id"], execution)
                    if execution.research_questions:
                        view = researcher_context(project, task)
                        execution.research = (await self.researcher.run(
                            "Research current task technical questions", task, view["plan_sha256"])
                            if self.researcher else ResearchResult(
                                "blocked", task["id"], plan.plan_sha256,
                                execution.research_questions, reason="research_unavailable"))
                        report = execution.research
                        execution.research_artifact = f"research_{task['id']}.json"
                        data = report.to_dict()
                        if not self._store(project.metadata, execution.research_artifact, data):
                            raise ValueError("research_storage_failed")
                        if (not report.verified or report.task_id != task["id"]
                                or report.plan_sha256 != project.plan.plan_sha256
                                or report.questions != execution.research_questions):
                            raise ValueError(report.reason or "invalid_research_output")
                        execution.research_sha256 = hashlib.sha256(json.dumps(
                            data, ensure_ascii=False, indent=2).encode("utf-8")).hexdigest()
                        project.seal(project.metadata, execution.research_artifact, execution.research_sha256)
                        project.set_stage("coding")
                    specification, context = coder_context(project, task, source, digest,
                                                            self.workflow.tester.describe_checks())
                    result = await self.workflow.run(json.dumps(specification, ensure_ascii=False),
                        context=json.dumps(context, ensure_ascii=False), project_state=project)
                    execution.result, execution.status = result, result.status
                    if result.verified:
                        _, execution.approved_source_sha256 = self._approved_source(result)
                    if result.status == "passed" and not execution.verified:
                        execution.status = "blocked"
                        execution.result.reason = "task_not_verified"
                    project.finish_task(task["id"])
                    if not self._store(project.metadata, f"task_{task['id']}.json", {
                            "plan_sha256": project.plan.plan_sha256, **execution.to_dict()}):
                        raise ValueError("task_summary_storage_failed")
                    if not execution.verified:
                        project.finish("blocked" if execution.status == "blocked" else "failed",
                                       f"{task['id']}: {execution.result.reason}")
                        break
                else:
                    for finished in project.tasks:
                        if finished.research_artifact:
                            if project.metadata.artifacts.file_sha256(finished.research_artifact) != finished.research_sha256:
                                raise ValueError("research_artifact_mismatch")
                    project.finish("passed")
        except asyncio.CancelledError:
            cancelled = True
            project.finish("cancelled", "cancelled")
        except TimeoutError:
            if not timeout.expired():
                raise
            project.finish("failed", "planned_workflow_deadline_exceeded")
        except (ValueError, OSError) as exc:
            code = str(exc) if isinstance(exc, ValueError) else "workspace_io_error"
            if not code or len(code) > 100 or not all(c.isalnum() or c in "_: " for c in code):
                code = "workflow_error"
            project.finish("blocked", code)
        # Terminal snapshots and DTOs are views of the same canonical records.
        try:
            if project.metadata is not None:
                for execution in project.tasks:
                    path = f"task_{execution.task_id}.json"
                    try:
                        project.metadata.artifacts.file_sha256(path)
                    except FileNotFoundError:
                        self._store(project.metadata, path, {
                            "plan_sha256": project.plan.plan_sha256, **execution.to_dict()})
                project.checkpoint()
                result = self._result(project)
                if project.status == "passed" and not result.verified:
                    raise ValueError("state_storage_failed")
                if not self._store(project.metadata, "result.json", result.to_dict()):
                    raise ValueError("state_storage_failed")
                if project.store.files.file_sha256("project_state.json") != project.store.digest:
                    raise ValueError("state_storage_failed")
                saved_state = project.store.files.read_json("project_state.json")
                saved_result = project.metadata.artifacts.read_json("result.json")
                for key in ("status", "goal", "run_project_id", "state_revision", "final_project_id", "final_source_sha256"):
                    if saved_state[key] != saved_result[key]:
                        raise ValueError("state_storage_failed")
        except (ValueError, OSError):
            project.storage_failed()
            try:
                project.checkpoint()
            except (ValueError, OSError):
                pass  # The last complete state remains readable; success is never returned.
            if project.metadata:
                try:
                    failed_result = self._result(project).to_dict()
                    try:
                        old_digest = project.metadata.artifacts.file_sha256("result.json")
                    except FileNotFoundError:
                        old_digest = None
                    project.metadata.artifacts.write_json("result.json", failed_result, old_digest)
                except (ValueError, OSError):
                    pass
        finally:
            if cancelled:
                raise asyncio.CancelledError
        return self._result(project)
