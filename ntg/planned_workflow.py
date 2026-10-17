"""Dependency scheduling over cumulative, approved single-file snapshots."""
import asyncio
import hashlib
import json
from dataclasses import dataclass, field

from .coding_workflow import WorkflowResult
from .planner_agent import PlanResult
from .types import RoutingError
from .task_graph import TaskGraph


@dataclass
class TaskExecution:
    task_id: str
    status: str
    result: WorkflowResult
    approved_source_sha256: str | None = None

    @property
    def verified(self):
        return (self.status == "passed" and self.result.review_required
                and self.result.verified and self.approved_source_sha256 is not None
                and self.approved_source_sha256 == self.result.revisions[-1].source_sha256)

    def to_dict(self):
        return {
            "task_id": self.task_id, "status": self.status,
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
                    or not set(deps) <= completed or not task.verified):
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
        }


class PlannedWorkflow:
    RUN_TIMEOUT = 1200

    def __init__(self, planner, workflow):
        if (workflow.reviewer is None or workflow.entrypoint != "app.py"
                or type(workflow.max_repairs) is not int or not 0 <= workflow.max_repairs <= 2):
            raise ValueError("unsupported_planned_workflow_configuration")
        self.planner, self.workflow = planner, workflow

    @staticmethod
    def _store(workspace, path, data):
        try:
            result = workspace.tools.execute("write_file", {
                "path": path, "content": json.dumps(data, ensure_ascii=False, indent=2),
            })
            return result["ok"]
        except (OSError, ValueError):
            return False

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

    async def run(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("Goal must be a nonempty string")
        state = PlannedResult("blocked", goal)
        metadata = None
        active = None
        graph = None
        cancelled = False
        timeout = asyncio.timeout(self.RUN_TIMEOUT)
        try:
            async with timeout:
                try:
                    plan = await self.planner.run(goal)
                    plan.validate()
                    if plan.goal != goal:
                        raise ValueError("invalid_plan_output")
                except RoutingError as exc:
                    state.reason = "planning_failed"
                    state.planning_error = exc.public()
                    return state
                except ValueError as exc:
                    state.reason = "planning_timeout" if str(exc) == "planning_timeout" else "planning_failed"
                    state.planning_error = {"code": str(exc)}
                    if hasattr(exc, "validation_reason"):
                        state.planning_error["validation_reason"] = exc.validation_reason
                    return state
                state.plan = plan
                graph = TaskGraph(plan.tasks)
                metadata = self.workflow.workspace_manager.create()
                state.run_project_id = metadata.project_id
                if not self._store(metadata, "plan.json", plan.to_dict()):
                    raise ValueError("plan_storage_failed")
                prior_source, prior_hash = None, None
                completed = []
                while not all(status == "passed" for status in graph.states.values()):
                    plan.validate()
                    ready = graph.ready()
                    if not ready:
                        state.status, state.reason = "failed", "scheduler_no_progress"
                        break
                    task = ready[0]
                    # Recheck the last artifact immediately before dispatch.
                    if state.tasks:
                        prior_source, prior_hash = self._approved_source(state.tasks[-1].result)
                    specification = json.dumps({
                        "original_goal": goal, "current_task": task,
                        "required_checks": self.workflow.tester.describe_checks(),
                        "completed_tasks": completed,
                        "acceptance_scope": (
                            "Complete the current task and preserve all completed features. "
                            + ("This is the final task: satisfy the whole original goal."
                               if len(completed) == len(plan.tasks) - 1 else
                               "This is an intermediate task: future tasks need not be implemented yet.")
                        ),
                    }, ensure_ascii=False)
                    context = json.dumps({
                        "plan_sha256": plan.plan_sha256, "tasks": plan.tasks,
                        "current_task_id": task["id"], "completed_tasks": completed,
                        "dependencies": task["dependencies"],
                        "prior_source": prior_source, "prior_source_sha256": prior_hash,
                    }, ensure_ascii=False)
                    execution = TaskExecution(task["id"], "failed", WorkflowResult(
                        "failed", None, [], "execution_interrupted", review_required=True,
                    ))
                    graph.start(task["id"])
                    state.execution_order.append(task["id"])
                    state.tasks.append(execution)
                    active = execution
                    result = await self.workflow.run(specification, context=context)
                    execution.result, execution.status = result, result.status
                    if result.verified:
                        try:
                            prior_source, prior_hash = self._approved_source(result)
                            execution.approved_source_sha256 = prior_hash
                        except ValueError:
                            execution.status = "blocked"
                            state.status, state.reason = "blocked", "approved_artifact_unavailable"
                    elif result.status == "passed":
                        execution.status = "blocked"
                        state.status, state.reason = "blocked", "task_not_verified"
                    else:
                        state.status = result.status
                        state.reason = f"{task['id']}: {result.reason}"
                    graph.finish(task["id"], passed=execution.verified)
                    summary = {"plan_sha256": plan.plan_sha256, **execution.to_dict()}
                    stored = self._store(metadata, f"task_{task['id']}.json", summary)
                    active = None
                    if not stored:
                        state.status, state.reason = "blocked", "task_summary_storage_failed"
                        break
                    if execution.status != "passed":
                        break
                    completed.append(task.copy())
                else:
                    state.status, state.reason = "passed", "all_tasks_tested_and_reviewed"
                    state.final_project_id = state.tasks[-1].result.final_project_id
                    state.final_source_sha256 = state.tasks[-1].approved_source_sha256
        except TimeoutError:
            if not timeout.expired():
                raise
            state.status, state.reason = "failed", "planned_workflow_deadline_exceeded"
            if active is not None:
                active.status = "failed"
                active.result.reason = state.reason
        except asyncio.CancelledError:
            cancelled = True
            state.status, state.reason = "failed", "cancelled"
        except ValueError as exc:
            if str(exc) not in ("workspace_unavailable", "workspace_allocation_failed",
                                "approved_artifact_unavailable", "invalid_plan_output", "plan_storage_failed"):
                raise
            state.status, state.reason = "blocked", str(exc)
        except OSError:
            state.status, state.reason = "blocked", "workspace_io_error"
        if graph is not None:
            if active is not None:
                if graph.states[active.task_id] == "running":
                    active.status = "failed"
                    active.result.reason = state.reason
                    graph.finish(active.task_id, passed=False)
            if state.status != "passed":
                graph.stop(state.reason)
                state.final_project_id = state.final_source_sha256 = None
            state.task_graph = graph.to_dict()
        try:
            if metadata is not None:
                if active is not None:
                    if not self._store(metadata, f"task_{active.task_id}.json", {
                        "plan_sha256": state.plan.plan_sha256, **active.to_dict(),
                    }):
                        state.status, state.reason = "blocked", "task_summary_storage_failed"
                if not self._store(metadata, "result.json", state.to_dict()):
                    state.status, state.reason = "blocked", "result_storage_failed"
                    state.final_project_id = state.final_source_sha256 = None
        finally:
            if cancelled:
                raise asyncio.CancelledError
        if state.status != "passed":
            state.final_project_id = state.final_source_sha256 = None
        return state
