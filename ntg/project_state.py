"""Per-run host ownership; snapshots contain references, never prompts or source bodies."""
import hashlib
import json
from copy import deepcopy

from .task_graph import TaskGraph
from tools.filesystem import ArtifactFiles, ToolError


class StateStore:
    def __init__(self, workspace):
        self.workspace = workspace
        self.files = ArtifactFiles(workspace.path)
        self.digest = None
        self.revision = -1

    def save(self, state):
        snapshot = state.snapshot()
        if (snapshot["run_project_id"] != self.workspace.project_id
                or type(snapshot["state_revision"]) is not int
                or snapshot["state_revision"] <= self.revision):
            raise ValueError("state_conflict")
        try:
            if self.workspace.tools.validate_workspace() != self.files.validate_root():
                raise ValueError("state_conflict")
            if self.digest is not None:
                if self.files.file_sha256("project_state.json") != self.digest:
                    raise ValueError("state_conflict")
                data = self.files.read_json("project_state.json")
                if (data["state_revision"] != self.revision
                        or data["run_project_id"] != self.workspace.project_id):
                    raise ValueError("state_conflict")
            saved = self.files.write_json("project_state.json", snapshot, self.digest)
        except (OSError, ToolError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("state_conflict" if str(exc) in ("state_conflict", "source_changed") else "state_storage_failed") from None
        self.digest, self.revision = saved["sha256"], snapshot["state_revision"]



class ProjectState:
    def __init__(self, goal):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("invalid_goal")
        self.goal, self.plan, self.graph = goal, None, None
        self.project = None
        self.entrypoint = "app.py"
        self.tasks = []  # Each canonical TaskExecution is registered exactly once.
        self.status, self.stage, self.reason = "initialized", "planning", None
        self.current_task, self.current_revision = None, None
        self.latest_approved_source = self.candidate = self.last_attempt = None
        self.metadata, self.store, self.planning_error = None, None, None
        self.state_revision, self.errors = 0, []
        self._sealed = {}
        self.final_project_id = self.final_source_sha256 = None

    @classmethod
    def standalone(cls, goal, review_required):
        # Imports are local to avoid coupling data modules during package import.
        from .planner_agent import PlanResult
        from .planned_workflow import TaskExecution
        from .coding_workflow import WorkflowResult
        state = cls(goal)
        state.status = "running"
        state.attach_plan(PlanResult(goal, [{"id": "T1", "description": goal[:2000], "dependencies": []}]))
        state.start_task("T1", TaskExecution("T1", "failed", WorkflowResult(
            "failed", None, [], "execution_interrupted", review_required)))
        return state

    def _touch(self):
        if self.status in ("passed", "failed", "blocked", "cancelled"):
            raise ValueError("state_finalized")
        if self.plan is not None:
            self.plan.validate()
            if self.plan.goal != self.goal:
                raise ValueError("invalid_plan_output")
        self.state_revision += 1

    @property
    def execution_order(self):
        return [t.task_id for t in self.tasks]

    @property
    def active(self):
        return next((t for t in self.tasks if t.task_id == self.current_task), None)

    @property
    def completed(self):
        if self.graph is None:
            return []
        return [deepcopy(t) for t in self.plan.tasks if self.graph.states[t["id"]] == "passed"]

    def bind(self, workspace):
        if self.metadata is not None:
            raise ValueError("invalid_state_transition")
        self._touch()
        self.metadata, self.store, self.status = workspace, StateStore(workspace), "running"
        self.checkpoint()

    def seal(self, workspace, path, expected=None):
        try:
            digest = workspace.artifacts.file_sha256(path)
        except (OSError, ToolError):
            raise ValueError("artifact_unavailable") from None
        key = (workspace.project_id, path)
        if (expected is not None and digest != expected) or (key in self._sealed and self._sealed[key] != digest):
            raise ValueError("artifact_mismatch")
        self._sealed[key] = digest

    def checkpoint(self):
        if self.store is not None:
            self.store.save(self)

    def attach_plan(self, plan):
        if self.plan is not None or plan.goal != self.goal or self.current_task:
            raise ValueError("invalid_state_transition")
        plan.validate()
        self._touch()
        self.plan, self.graph = plan, TaskGraph(plan.tasks)
        self.checkpoint()

    def start_task(self, task_id, execution):
        if (self.graph is None or self.current_task or task_id != execution.task_id
                or any(s == "failed" for s in self.graph.states.values())
                or task_id not in {t["id"] for t in self.graph.ready()}):
            raise ValueError("invalid_state_transition")
        self._touch()
        self.graph.start(task_id)
        self.tasks.append(execution)
        self.current_task, self.current_revision, self.candidate = task_id, None, None
        self.stage = "researching" if execution.research_questions else "coding"
        self.checkpoint()

    def set_stage(self, stage, revision=None):
        allowed = {
            "planning": {"coding", "researching"},
            "researching": {"researching", "coding"},
            "coding": {"coding", "compiling", "repairing"},
            "compiling": {"testing", "repairing", "coding"},
            "testing": {"testing", "reviewing", "repairing", "coding"},
            "reviewing": {"reviewing", "repairing", "coding"},
            "repairing": {"repairing", "compiling", "coding"},
        }
        if not self.current_task or not isinstance(stage, str) or stage not in allowed.get(self.stage, set()):
            raise ValueError("invalid_state_transition")
        if revision is not None and (type(revision) is not int or not 1 <= revision <= 3):
            raise ValueError("invalid_revision")
        self._touch()
        self.stage = stage
        if revision is not None:
            self.current_revision = revision
        self.checkpoint()

    def register_revision(self, task_id, record):
        if self.active is None or task_id != self.current_task:
            raise ValueError("invalid_state_transition")
        revisions = self.active.result.revisions
        if record.revision != len(revisions) + 1 or record in revisions:
            raise ValueError("invalid_revision")
        self._touch()
        revisions.append(record)
        self.current_revision = record.revision
        self.stage = "coding" if record.revision == 1 else "repairing"
        self.checkpoint()

    @staticmethod
    def source_ref(record, path="app.py", *, actual=False):
        if record.workspace is None or record.workspace.project_id != record.project_id:
            raise ValueError("approved_artifact_unavailable")
        if record.workspace.tools.validate_workspace() != record.workspace.path:
            raise ValueError("approved_artifact_unavailable")
        read = record.workspace.tools.execute("read_file", {"path": path})
        if not read["ok"]:
            raise ValueError("approved_artifact_unavailable")
        source = read["data"]["content"]
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if not actual and digest != record.source_sha256:
            raise ValueError("approved_artifact_unavailable")
        return {"task_id": None, "revision": record.revision, "project_id": record.project_id,
                "path": path, "sha256": digest}, source

    def revision_finished(self, record, entrypoint):
        if self.active is None or not any(r is record for r in self.active.result.revisions):
            raise ValueError("invalid_revision")
        self._touch()
        if record.source_snapshot:
            self.last_attempt = {"task_id": self.current_task, "revision": record.revision,
                                 "project_id": record.project_id, "path": record.source_snapshot,
                                 "sha256": record.source_sha256}
        try:
            self.candidate, _ = self.source_ref(record, entrypoint, actual=True)
            self.candidate["task_id"] = self.current_task
            matching = next((r for r in reversed(self.active.result.revisions)
                             if r.source_sha256 == self.candidate["sha256"]), None)
            self.candidate["revision"] = matching.revision if matching else None
        except ValueError:
            self.candidate = None
        if record.workspace:
            if record.source_snapshot:
                self.seal(record.workspace, record.source_snapshot, record.source_sha256)
            for kind, report in (("tester", record.report), ("reviewer", record.review)):
                if report is not None:
                    self.seal(record.workspace, f"revision_{record.revision:03}_{kind}.json")
            for path in record.capture_files:
                self.seal(record.workspace, path)
        if record.failure_type:
            self.add_error(record.failure_type, record.revision)
        self.checkpoint()

    def add_error(self, code, revision=None):
        # Stable codes only; raw exception strings and execution output live in artifacts.
        if self.status in ("passed", "failed", "blocked", "cancelled"):
            raise ValueError("state_finalized")
        self.state_revision += 1
        if not isinstance(code, str) or len(code) > 100 or not all(c.isalnum() or c in "_:" for c in code):
            code = "execution_error"
        entry = {"code": code, "stage": self.stage, "task_id": self.current_task,
                 "revision": revision or self.current_revision, "artifact": deepcopy(self.last_attempt), "count": 1}
        match = next((e for e in self.errors if all(e[k] == entry[k] for k in entry if k != "count")), None)
        if match:
            match["count"] += 1
        elif len(self.errors) < 64:
            self.errors.append(entry)

    def finish_task(self, task_id):
        execution = self.active
        if execution is None or task_id != self.current_task:
            raise ValueError("invalid_state_transition")
        passed = execution.verified
        approved = None
        if passed:
            approved, _ = self.source_ref(execution.result.revisions[-1])
            approved["task_id"] = task_id
            if approved["sha256"] != execution.approved_source_sha256:
                raise ValueError("approved_artifact_unavailable")
        self._touch()
        self.graph.finish(task_id, passed=passed)
        if approved:
            self.latest_approved_source = approved
        self.current_task = self.current_revision = None
        self.checkpoint()

    def approved_source(self):
        if self.latest_approved_source is None:
            return None, None
        execution = next(t for t in self.tasks if t.task_id == self.latest_approved_source["task_id"])
        if not execution.verified:
            raise ValueError("approved_artifact_unavailable")
        ref, source = self.source_ref(execution.result.revisions[-1])
        if ref["sha256"] != self.latest_approved_source["sha256"]:
            raise ValueError("approved_artifact_unavailable")
        return source, ref["sha256"]

    def finish(self, status, reason=None):
        if not isinstance(status, str) or status not in ("passed", "failed", "blocked", "cancelled"):
            raise ValueError("invalid_state_transition")
        if status == "passed":
            if (self.graph is None or not self.tasks or self.current_task
                    or not all(s == "passed" for s in self.graph.states.values())
                    or not all(t.verified for t in self.tasks)):
                raise ValueError("task_not_verified")
            planned = {t["id"]: t for t in self.plan.tasks}
            if len(self.tasks) != len(planned) or set(self.execution_order) != set(planned):
                raise ValueError("task_not_verified")
            completed = set()
            graph = self.graph.to_dict()
            for execution in self.tasks:
                task = planned[execution.task_id]
                if (execution.task_id in completed or not set(task["dependencies"]) <= completed
                        or graph[execution.task_id]["dependencies"] != task["dependencies"]
                        or execution.research_questions != task.get("research_questions", [])
                        or (execution.research and execution.research.plan_sha256 != self.plan.plan_sha256)):
                    raise ValueError("task_not_verified")
                completed.add(execution.task_id)
            self.approved_source()
        if status != "passed" and self.status not in ("passed", "failed", "blocked", "cancelled"):
            # A mutated plan is itself a failure; do not prevent terminal cleanup.
            self.state_revision += 1
        else:
            self._touch()
        if self.active is not None:
            self.active.status = "blocked" if status == "blocked" else "failed"
            self.active.result.reason = reason or status
            self.graph.finish(self.current_task, passed=False)
        if status != "passed":
            self.add_error(reason or status)
            if self.graph is not None:
                self.graph.stop(reason or status)
        self.status, self.stage, self.reason = status, "finished", None if status == "passed" else reason
        self.current_task = self.current_revision = None
        self.final_project_id = self.latest_approved_source["project_id"] if status == "passed" else None
        self.final_source_sha256 = self.latest_approved_source["sha256"] if status == "passed" else None

    def storage_failed(self):
        # Terminal I/O failure is the sole permitted correction after finish().
        self.status, self.reason, self.stage = "blocked", "state_storage_failed", "finished"
        self.final_project_id = self.final_source_sha256 = None
        self.current_task = self.current_revision = None
        self.state_revision += 1

    def snapshot(self):
        if self.project is not None:
            return {"schema_version": 2, "state_revision": self.state_revision,
                    "run_project_id": self.metadata.project_id, "status": self.status, "stage": self.stage,
                    "current_task": self.current_task, "current_revision": self.current_revision,
                    "reason": self.reason, "tasks": self.graph.tasks if self.graph else [],
                    "task_graph": self.graph.to_dict() if self.graph else {},
                    "project": deepcopy(self.project)}
        files, agents, tests, reviews, research, approvals = [], [], [], [], [], []
        def artifact(workspace, path, kind, task_id=None, revision=None):
            try:
                digest = workspace.artifacts.file_sha256(path)
            except (OSError, ToolError):
                if (workspace.project_id, path) in self._sealed:
                    raise ValueError("artifact_unavailable") from None
                return None
            item = {"task_id": task_id, "revision": revision, "project_id": workspace.project_id,
                    "path": path, "kind": kind, "sha256": digest}
            if (workspace.project_id, path) in self._sealed and self._sealed[(workspace.project_id, path)] != item["sha256"]:
                raise ValueError("artifact_mismatch")
            files.append(item)
            return item
        if self.metadata and self.plan:
            ref = artifact(self.metadata, "plan.json", "plan")
            agents.append({"role": "planner", "status": "completed", "artifact": ref})
        elif self.planning_error:
            agents.append({"role": "planner", "status": "blocked", "artifact": None})
        for task in self.tasks:
            if task.research:
                ref = artifact(self.metadata, task.research_artifact, "research", task.task_id) if self.metadata and task.research_artifact else None
                research.append({"task_id": task.task_id, "status": task.research.status,
                                 "verified": task.research_verified, "artifact": ref})
                agents.append({"role": "researcher", **research[-1]})
            for record in task.result.revisions:
                base = {"task_id": task.task_id, "revision": record.revision, "source_sha256": record.source_sha256}
                if record.workspace:
                    if record.source_snapshot:
                        ref = artifact(record.workspace, record.source_snapshot, "source", task.task_id, record.revision)
                        agents.append({"role": "coder", **base, "status": record.progress, "artifact": ref})
                    for kind, report, index in (("tester", record.report, tests), ("reviewer", record.review, reviews)):
                        if report is not None:
                            ref = artifact(record.workspace, f"revision_{record.revision:03}_{kind}.json", kind, task.task_id, record.revision)
                            index.append({**base, "status": report.status, "verified": report.verified, "artifact": ref})
                            agents.append({"role": kind, **index[-1]})
                    for path in record.capture_files:
                        artifact(record.workspace, path, "capture", task.task_id, record.revision)
            if task.result.revisions and task.result.revisions[-1].workspace:
                last = task.result.revisions[-1]
                current = artifact(last.workspace, self.entrypoint, "entrypoint", task.task_id, last.revision)
                if current:
                    matching = next((r for r in reversed(task.result.revisions)
                                     if r.source_sha256 == current["sha256"]), None)
                    current["revision"] = matching.revision if matching else None
            if task.verified:
                approvals.append({"task_id": task.task_id, "project_id": task.result.final_project_id,
                                  "source_sha256": task.approved_source_sha256})
        graph = self.graph.to_dict() if self.graph else {}
        tasks = [{**deepcopy(t), **graph[t["id"]]} for t in self.plan.tasks] if self.plan else []
        for item in files:
            item["approved"] = any(a["task_id"] == item["task_id"] and a["source_sha256"] == item["sha256"] for a in approvals)
        return deepcopy({"schema_version": 1, "state_revision": self.state_revision,
            "run_project_id": self.metadata.project_id if self.metadata else None,
            "goal": self.goal, "status": self.status, "stage": self.stage, "reason": self.reason,
            "planning_error": self.planning_error,
            "current_task": self.current_task, "current_revision": self.current_revision,
            "plan_sha256": self.plan.plan_sha256 if self.plan else None,
            "tasks": tasks, "execution_order": self.execution_order, "files": files,
            "agent_results": agents, "test_results": tests, "review_results": reviews,
            "research_results": research, "errors": self.errors, "approvals": approvals,
            "latest_approved_source": self.latest_approved_source, "candidate": self.candidate,
            "last_attempt": self.last_attempt, "final_project_id": self.final_project_id,
            "final_source_sha256": self.final_source_sha256})
