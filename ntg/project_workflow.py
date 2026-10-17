"""Adaptive serial project orchestration over existing agent roles and guarded files."""
import asyncio
import hashlib
import json
import time
import uuid
from copy import deepcopy

from .project import ProjectFiles
from .project_state import ProjectState, StateStore
from .task_graph import TaskGraph
from .types import RoutingError
from .tester_agent import TesterAgent


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class ProjectWorkflow:
    def __init__(self, planner, coder, tester, reviewer, workspace_manager, *, researcher=None,
                 max_hops=32, max_repairs=3, max_expansions=512, max_depth=6,
                 max_tasks=10000, max_calls=200, run_timeout=1200,
                 checks_repeatable=False, integration_check=None):
        if min(max_hops, max_repairs, max_expansions, max_depth, max_tasks, max_calls, run_timeout) <= 0:
            raise ValueError("invalid_project_limits")
        self.planner, self.coder, self.tester, self.reviewer = planner, coder, tester, reviewer
        self.manager, self.researcher = workspace_manager, researcher
        self.max_hops, self.max_repairs = max_hops, max_repairs
        self.max_expansions, self.max_depth, self.max_tasks = max_expansions, max_depth, max_tasks
        self.max_calls, self.run_timeout = max_calls, run_timeout
        self.checks_repeatable, self.integration_check = checks_repeatable, integration_check
        semantic_config = {"entrypoint": tester.entrypoint, "checks": tester.describe_checks(),
                           "integration": getattr(integration_check, "__qualname__", None)}
        self.config = digest(semantic_config)
        self._previous_configs = {digest({**semantic_config, "repeatable": value}) for value in (True, False)}
        self.project_state = None
        self._calls = 0
        self._chars = 12000
        self._resume_candidate = None
        self._repository_index = None
        self._repository_cursors = {}

    def _save(self):
        state = self.project_state
        reference = state.project.get("repository_index")
        if hasattr(self, "source") and (not reference or reference.get("project_revision") != state.project["project_revision"]):
            self._repository(self.source, persist=True)
        state.project["calls"] = self._total_calls
        state.state_revision += 1
        state.store.save(state)

    def _record(self, kind, data):
        state = self.project_state
        number = state.project["artifact_sequence"] + 1
        path = f"{kind}_{number:06}_{uuid.uuid4().hex[:12]}.json"
        saved = state.metadata.artifacts.write_json(path, data)
        state.project["artifact_sequence"] = number
        state.project.setdefault("evidence", {})[path] = saved["sha256"]
        return {"path": path, "sha256": saved["sha256"]}

    def _load_ref(self, ref):
        files = self.project_state.metadata.artifacts
        if files.file_sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("checkpoint_artifact_mismatch")
        return files.read_json(ref["path"])

    def _new(self, goal):
        self._repository_index, self._repository_cursors = None, {}
        state = ProjectState(goal)
        state.metadata = self.manager.create()
        state.store = StateStore(state.metadata)
        source = ProjectFiles(self.manager.create())
        goal_file = state.metadata.artifacts.execute("write_file", {"path": "goal.txt", "content": goal})
        if not goal_file["ok"]:
            raise ValueError("goal_storage_failed")
        state.project = {"config": self.config, "goal_sha256": hashlib.sha256(goal.encode()).hexdigest(),
            "source_project_id": source.workspace.project_id, "manifest": {}, "project_revision": ProjectFiles.revision({}),
            "planning_cursor": 0, "planning_tasks": [], "requirements": [], "expansions": 0,
            "approvals": {}, "dispatch_order": [], "calls": 0, "artifact_sequence": 0,
            "active": None, "integration": None}
        # Preserve every requirement character; each ingestion packet stays bounded.
        for offset in range(0, len(goal), 6000):
            text = goal[offset:offset+6000]
            state.project["requirements"].append({"id": f"R{offset//6000+1}", "start": offset, "end": offset+len(text),
                                                     "sha256": hashlib.sha256(text.encode()).hexdigest()})
        state.status, state.stage = "running", "planning"
        self.project_state, self.source = state, source
        self._total_calls = 0
        state.project["host_checks"] = self._record("host_checks", self.tester.describe_checks())
        self._save()

    def _resume(self, project_id):
        metadata = self.manager.open(project_id)
        saved = metadata.artifacts.read_json("project_state.json")
        if (saved.get("schema_version") != 2 or saved.get("run_project_id") != project_id
                or saved["project"]["config"] not in {self.config, *self._previous_configs}):
            raise ValueError("checkpoint_configuration_mismatch")
        goal = metadata.artifacts.execute("read_file", {"path": "goal.txt"})
        if not goal["ok"] or hashlib.sha256(goal["data"]["content"].encode()).hexdigest() != saved["project"]["goal_sha256"]:
            raise ValueError("checkpoint_goal_mismatch")
        state = ProjectState(goal["data"]["content"])
        state.metadata, state.store = metadata, StateStore(metadata)
        state.project = saved["project"]
        state.project["config"] = self.config
        state.project["repeatable_checks"] = self.checks_repeatable
        state.state_revision = saved["state_revision"]
        state.store.revision = state.state_revision
        state.store.digest = metadata.artifacts.file_sha256("project_state.json")
        self.project_state = state
        self._total_calls = state.project["calls"]
        cursor = 0
        for requirement in state.project["requirements"]:
            start, end = requirement["start"], requirement["end"]
            if (start != cursor or end <= start or end > len(state.goal)
                    or hashlib.sha256(state.goal[start:end].encode()).hexdigest() != requirement["sha256"]):
                raise ValueError("checkpoint_requirement_mismatch")
            cursor = end
        if cursor != len(state.goal):
            raise ValueError("checkpoint_requirement_mismatch")
        self.source = ProjectFiles(self.manager.open(state.project["source_project_id"]), state.project["manifest"])
        if self.source.manifest() != state.project["manifest"] or ProjectFiles.revision(self.source.manifest()) != state.project["project_revision"]:
            raise ValueError("checkpoint_source_mismatch")
        if saved["tasks"]:
            state.graph = TaskGraph(saved["tasks"])
            self.source.runtime_paths.update(p for task in saved["tasks"] for p in task.get("runtime_files", []))
            states = {k: v["state"] for k, v in saved["task_graph"].items()}
            for task_id, status in states.items():
                if status == "passed":
                    approval = state.project["approvals"].get(task_id)
                    if not approval:
                        raise ValueError("checkpoint_approval_missing")
                    evidence = self._load_ref(approval)
                    task = next(t for t in saved["tasks"] if t["id"] == task_id)
                    if (evidence["review"]["action"] != "approve" or evidence["review"]["issues"]
                            or evidence["test"]["status"] != "passed" or evidence["task_sha256"] != digest(task)
                            or evidence["project_revision"] != ProjectFiles.revision(evidence["manifest"])
                            or evidence["test"]["project_revision"] != evidence["project_revision"]):
                        raise ValueError("checkpoint_approval_invalid")
                    execution = self._load_ref(evidence["test"]["artifact"])
                    if (execution["status"] != "passed" or execution["project_revision"] != evidence["project_revision"]
                            or any(not check["passed"] for check in execution["checks"])):
                        raise ValueError("checkpoint_test_evidence_invalid")
                else:
                    states[task_id] = "pending"
            state.graph.restore(states)
        active = state.project.get("active")
        if active and active.get("checks_started") and not self.checks_repeatable:
            raise ValueError("checkpoint_execution_not_repeatable")
        self._resume_candidate = None
        if active and active.get("latest_test", {}).get("status") == "passed":
            try:
                latest = active["latest_test"]
                execution = self._load_ref(latest["artifact"])
                task = next(t for t in state.graph.tasks if t["id"] == active["task_id"])
                paths = set(latest.get("manifest", state.project["manifest"]))
                paths.update(c["name"][8:] for c in execution["checks"] if c["name"].startswith("compile:"))
                workspace = self.manager.open(active["candidate_id"])
                for path in task["files"]:
                    if not path.endswith("/") and path not in task.get("runtime_files", []) and workspace.artifacts._artifacts()._resolve(path).is_file():
                        paths.add(path)
                candidate = ProjectFiles(workspace, paths)
                candidate.runtime_paths.update(p for t in state.graph.tasks for p in t.get("runtime_files", []))
                candidate.verify_python_inventory()
                candidate_revision = ProjectFiles.revision(candidate.manifest(refresh=True))
                integration_ok = (task["id"] not in state.project.get("integration_tasks", []) or not self.integration_check
                                  or execution.get("integration", {}).get("status") == "passed")
                if (execution["status"] == "passed" and all(c["passed"] for c in execution["checks"])
                        and candidate_revision == latest["project_revision"] == execution["project_revision"] and integration_ok):
                    candidate.changed_ranges = deepcopy(latest.get("changed_ranges", {}))
                    for path, sha in candidate.manifest().items():
                        if sha != state.project["manifest"].get(path) and path not in candidate.changed_ranges:
                            # Older checkpoints lack edit spans: review the entire changed file.
                            with candidate.files._resolve(path).open(encoding="utf-8") as stream:
                                count = sum(1 for _ in stream)
                            candidate.changed_ranges[path] = [(1, max(1, count))]
                    self._resume_candidate = (task["id"], candidate)
            except (OSError, ValueError, KeyError, StopIteration):
                # Unapproved evidence never overrides the immutable approved-source checkpoint.
                self._resume_candidate = None
        # Only a completely tested/hash-verified candidate may resume at fresh validation/review.
        if state.project["dispatch_order"]:
            last = self._load_ref(state.project["approvals"][state.project["dispatch_order"][-1]])
            if last["manifest"] != state.project["manifest"]:
                raise ValueError("checkpoint_approved_source_mismatch")
        if saved["status"] == "passed":
            if (state.graph is None or any(v != "passed" for v in state.graph.states.values())
                    or self.tester.entrypoint not in self.source.paths):
                raise ValueError("checkpoint_completion_invalid")
            reviews = dict(state.project.get("final_reviews", {}))
            # Recover the explicit final-review references from early schema-2 checkpoints.
            if not reviews:
                for path, sha in state.project.get("evidence", {}).items():
                    if path.startswith("final_review_"):
                        ref = {"path": path, "sha256": sha}
                        data = self._load_ref(ref)
                        if data["project_revision"] == state.project["project_revision"]:
                            reviews[data["requirement"]] = ref
            if set(reviews) != {r["id"] for r in state.project["requirements"]}:
                raise ValueError("checkpoint_final_review_missing")
            for ref in reviews.values():
                data = self._load_ref(ref)
                if (data["project_revision"] != state.project["project_revision"]
                        or data["review"]["action"] != "approve" or data["review"]["issues"]):
                    raise ValueError("checkpoint_final_review_invalid")
            state.project["final_reviews"] = reviews
            integration = state.project.get("integration", {})
            if integration.get("status") != "passed":
                raise ValueError("checkpoint_integration_invalid")
            if integration.get("artifact") and self._load_ref(integration["artifact"]).get("status") != "passed":
                raise ValueError("checkpoint_integration_invalid")
        state.project["active"] = None
        state.status, state.stage = ("passed", "finished") if saved["status"] == "passed" else ("running", "planning" if state.graph is None else "coding")
        state.current_task = state.current_revision = None
        self._repository_index, self._repository_cursors = None, {}
        self._repository(self.source, persist=True)
        self._save()

    async def _invoke(self, method, packet, *, output=2048):
        if self._calls >= self.max_calls:
            raise ValueError("run_call_budget_reached")
        self._calls += 1
        self._total_calls += 1
        try:
            result, routing = await method(packet, max_tokens=output)
        except RoutingError as exc:
            self._record("routing_failure", {"code": exc.code, "history": exc.history})
            raise
        self._record("routing", routing)
        return result

    def _requirements(self, ids):
        return [{**r, "text": self.project_state.goal[r["start"]:r["end"]]}
                for r in self.project_state.project["requirements"] if r["id"] in ids]

    @staticmethod
    def _size_failure(exc):
        attempts = [h for h in exc.history if h.get("attempted")]
        return exc.code in ("context_capacity_exceeded", "output_truncated") or bool(attempts) and all(
            h.get("outcome") in ("context_exceeded", "output_truncated") for h in attempts)

    async def _planning_call(self, packet):
        index = self._repository(self.source)
        packet = {**packet, "repository": {"project_revision": index.project_revision,
                  "coverage": index.coverage, "files": self.source.listing(0, 10)}}
        last = None
        for output in (4096, 2048, 1024, 512):
            try:
                tasks = await self._invoke(self.planner.plan_project, packet, output=output)
                for task in tasks:
                    self._validate_task(task)
                return tasks
            except RoutingError as exc:
                if not self._size_failure(exc):
                    raise
                last = exc
            except ValueError as exc:
                code = str(exc)
                if not code or len(code) > 100 or not all(c.isalnum() or c == "_" for c in code):
                    raise
                self._record("plan_validation_failure", {"code": code})
                packet = {**packet, "validation_error": code,
                          "instruction": "Correct the task graph/check contract; preserve requirements and return the full corrected small plan."}
                last = exc
        raise last

    async def _plan(self):
        state = self.project_state
        while state.project["planning_cursor"] < len(state.project["requirements"]):
            req = state.project["requirements"][state.project["planning_cursor"]]
            prior = state.project["planning_tasks"]
            previous = [{"id": t["id"], "contract": t["contract"]} for t in prior[-3:]]
            try:
                tasks = await self._planning_call({"requirements": self._requirements([req["id"]]),
                    "preceding_requirement_context": state.goal[max(0, req["start"]-400):req["start"]],
                    "existing_contracts": previous, "research_enabled": self.researcher is not None,
                    "required_checks": self.tester.describe_checks()})
            except RoutingError as exc:
                if not self._size_failure(exc):
                    raise
                if req["end"]-req["start"] < 600:
                    raise ValueError("planning_protocol_exceeds_available_context") from None
                midpoint = (req["start"]+req["end"])//2
                parts = []
                for index, (start, end) in enumerate(((req["start"], midpoint), (midpoint, req["end"])), 1):
                    parts.append({"id": req["id"]+f".{index}", "start": start, "end": end,
                                  "sha256": hashlib.sha256(state.goal[start:end].encode()).hexdigest()})
                cursor = state.project["planning_cursor"]
                state.project["requirements"][cursor:cursor+1] = parts
                self._save()
                continue
            mapping = {t["id"]: f"T{len(prior)+i+1}" for i, t in enumerate(tasks)}
            previous_roots = [t["id"] for t in prior if not any(t["id"] in n["dependencies"] for n in prior)]
            for task in tasks:
                task = deepcopy(task)
                task["id"] = mapping[task["id"]]
                task["dependencies"] = [mapping[d] for d in task["dependencies"]] or previous_roots
                task["requirement_ids"] = [req["id"]]
                self._validate_task(task)
                prior.append(task)
            if len(prior) > self.max_tasks:
                raise ValueError("project_task_budget_reached")
            state.project["planning_cursor"] += 1
            self._save()
        state.graph = TaskGraph(state.project.pop("planning_tasks", []))
        state.stage = "coding"
        self._save()

    def _validate_task(self, task):
        import shlex
        for path in task.get("runtime_files", []):
            self.source.files._resolve(path)
            if not path.endswith((".json", ".csv", ".txt", ".log", ".db", ".sqlite")):
                raise ValueError("invalid_runtime_data_path")
        for path in task["files"]:
            if path.endswith("/"):
                self.source.files._resolve(path + "scope_check.py")
                if task["kind"] == "leaf":
                    task["kind"] = "component"
                continue
            self.source.files._resolve(path)
            if not path.endswith((".py", ".txt", ".md", ".json", ".toml")):
                raise ValueError("unsupported_project_file")
        # Normalize only an unambiguous full CLI invocation placed in a check label.
        # Tokens remain literal subprocess arguments; no shell command is executed.
        for check in task["checks"]:
            if not check["arguments"]:
                try:
                    tokens = shlex.split(check["name"])
                except ValueError:
                    tokens = []
                if len(tokens) >= 2 and tokens[0] in ("python", "python3", "python.exe") and tokens[1] == self.tester.entrypoint:
                    check["arguments"] = tokens[2:]
        if task["checks"]:
            TesterAgent(entrypoint=self.tester.entrypoint, checks=task["checks"])
        obligations = {}
        for check in self.tester.describe_checks() + task["checks"]:
            key = tuple(check["arguments"])
            previous = obligations.get(key)
            current = check["exit_code"]
            if previous is not None and previous != current:
                if previous == -1 and current != 0:
                    pass
                elif current == -1 and previous != 0:
                    current = previous
                else:
                    raise ValueError("contradictory_task_checks")
            obligations[key] = current

    async def _expand(self, task):
        state = self.project_state
        if task["id"].count(".") >= self.max_depth or state.project["expansions"] >= self.max_expansions:
            raise ValueError("task_expansion_budget_reached")
        children = await self._planning_call({"parent": task,
            "requirements": self._requirements(task["requirement_ids"]),
            "research_enabled": self.researcher is not None, "instruction": "Split this task into smaller implementable leaves. Preserve the parent contract and allowed paths."})
        if len(children) < 2 or any(t["description"].strip() == task["description"].strip() for t in children):
            raise ValueError("decomposition_no_progress")
        mapping = {t["id"]: task["id"] + "." + str(i+1) for i, t in enumerate(children)}
        additions = []
        for child in children:
            self._validate_task(child)
            additions.extend(path for path in child["files"] if not self._in_scope(task, path))
        if additions:
            # A generated plan's ownership list is amendable, not a new user permission wall.
            # Only the Planner can propose this; all paths still pass the workspace guards.
            previous = list(task["files"])
            task["files"] = list(dict.fromkeys(previous + additions))
            state.graph.amend_pending(task)
            self._record("scope_amendment", {"parent": task["id"], "before": previous, "after": task["files"]})
        for child in children:
            child["id"] = mapping[child["id"]]
            child["dependencies"] = [mapping[d] for d in child["dependencies"]]
            child["requirement_ids"] = list(task["requirement_ids"])
            self._validate_task(child)
        if len(state.graph.tasks) + len(children) > self.max_tasks:
            raise ValueError("project_task_budget_reached")
        state.graph.expand(task["id"], children)
        state.project["expansions"] += 1
        self._record("plan_amendment", {"parent": task["id"], "children": children})
        self._save()

    def _observe(self, project, operation):
        action, path = operation["action"], operation.get("path")
        if action in ("repo_search", "repo_dependencies"):
            if any(operation.get(k) is not None for k in ("end", "sha256")):
                raise ValueError("invalid_repository_operation")
            offset = operation.get("start") if operation.get("start") is not None else 0
            if type(offset) is not int or offset < 0:
                raise ValueError("invalid_repository_range")
            index = self._repository(project, persist=True)
            key = (project.workspace.project_id, action, path, operation.get("text"))
            previous = self._repository_cursors.get(key)
            if offset and (previous is None or previous != (index.project_revision, offset)):
                raise ValueError("index_revision_changed")
            if action == "repo_search":
                result = index.search(operation.get("text"), path=path, offset=offset)
            else:
                if operation.get("text") is not None:
                    raise ValueError("invalid_repository_operation")
                result = index.dependencies(path, offset=offset)
            # Keep pages within the same source-observation budget as exact reads.
            maximum = max(1000, self._chars//2)
            while len(json.dumps(result, ensure_ascii=False)) > maximum and len(result["records"]) > 1:
                result["records"].pop()
                result["next_offset"] = offset+len(result["records"])
            if len(json.dumps(result, ensure_ascii=False)) > maximum:
                if result["records"]:
                    row = result["records"][0]
                    result["records"] = [{k:row[k] for k in ("path", "source_sha256", "kind", "lineno", "end_lineno", "resolution", "target_path") if k in row}]
                    result["omitted_fields"] = True
                result["diagnostics"] = []
                result["diagnostics_more"] = bool(index.coverage["diagnostic_count"])
                if result["records"] and len(json.dumps(result, ensure_ascii=False)) > maximum:
                    result["records"][0].pop("target_path", None)
            self._repository_cursors[key] = (index.project_revision, result["next_offset"])
            self._record("repository_query", {"operation": operation, "result": result})
            return result
        if action == "read":
            if path and path.startswith("artifact:"):
                artifact = path[9:]
                sha = self.project_state.project.get("evidence", {}).get(artifact)
                if not sha:
                    if artifact in project.paths:
                        return {"error": "source_path_requires_plain_read", "instruction": "Use the suggested source read without artifact: prefix.",
                                "suggested_operation": {**operation, "path": artifact}}
                    raise ValueError("unknown_evidence")
                data = self._load_ref({"path": artifact, "sha256": sha})
                text = json.dumps(data, ensure_ascii=False, indent=2)
                offset = max(0, operation.get("start") or 0)
                count = min(max(1000, self._chars//2), max(1, (operation.get("end") or offset+4000)-offset))
                summary = {k:data[k] for k in ("status", "failure", "persistence", "persistence_evidence", "rejected_inputs_preserved_data")
                           if isinstance(data, dict) and k in data}
                if isinstance(data, dict) and isinstance(data.get("review"), dict):
                    review = data["review"]
                    summary["review"] = {"action": review.get("action"),
                        "issues": [{"file": i["file"], "severity": i["severity"], "description": i["description"][:400],
                                    "excerpted": len(i["description"]) > 400} for i in review.get("issues", [])[:2]],
                        "more_issues": len(review.get("issues", [])) > 2}
                return {"path": path, "sha256": sha, "summary": summary, "text": text[offset:offset+count],
                        "next_offset": offset+count if offset+count < len(text) else None}
            if path and path.startswith("task:"):
                task_id = path[5:]
                task = next((t for t in self.project_state.graph.tasks if t["id"] == task_id), None)
                if task is None:
                    raise ValueError("unknown_task")
                text = json.dumps(task, ensure_ascii=False)
                start = max(0, operation.get("start") or 0)
                end = min(start+max(1000, self._chars//2), operation.get("end") or start+2000)
                return {"task_id": task_id, "state": self.project_state.graph.states[task_id],
                        "text": text[start:end], "next_offset": end if end < len(text) else None}
            if path and path.startswith("requirement:"):
                result = self._requirements([path[12:]])
                if not result:
                    raise ValueError("unknown_requirement")
                requirement = result[0]
                start = max(0, operation.get("start") or 0)
                end = min(start+max(1000, self._chars//2), operation.get("end") or start+2000, len(requirement["text"]))
                if end < start:
                    raise ValueError("invalid_requirement_range")
                return {"requirement_id": requirement["id"], "sha256": requirement["sha256"],
                        "start_offset": start, "end_offset": end, "total_chars": len(requirement["text"]),
                        "text": requirement["text"][start:end], "next_offset": end if end < len(requirement["text"]) else None}
            start, end = operation.get("start") or 1, operation.get("end") or 80
            return project.read(path, start, min(200, max(1, end-start+1)), chars=max(1000, self._chars//2))
        if action == "search":
            return project.search(operation.get("text"))
        if action == "list":
            return project.listing(operation.get("start") or 0)
        raise ValueError("invalid_observation")

    def _repository(self, project, persist=False):
        from .repository import RepositoryIndex
        if self._repository_index is None:
            reference = self.project_state.project.get("repository_index")
            try:
                self._repository_index = RepositoryIndex.from_snapshot(self._load_ref(reference)) if reference else RepositoryIndex()
                self._repository_index.refresh(project)
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
                # Optional derived cache never replaces required source/approval checks.
                self._repository_index = RepositoryIndex()
                self.project_state.project.pop("repository_index", None)
        index = self._repository_index
        index.refresh(project)
        if persist:
            state = self.project_state
            active = state.project.get("active")
            owner = state.project if project.workspace.project_id == state.project["source_project_id"] else active
            if owner is not None:
                ref = owner.get("repository_index")
                if not ref or ref.get("project_revision") != index.project_revision:
                    ref = self._record("repository_index", index.snapshot())
                    owner["repository_index"] = {**ref, "project_revision": index.project_revision, "schema_version": index.SCHEMA}
        return index

    @staticmethod
    def _in_scope(task, path):
        return isinstance(path, str) and (path in task["files"] or any(
            scope.endswith("/") and path.startswith(scope) for scope in task["files"]))

    def _packet(self, task, project, observations, feedback):
        current = project.manifest()
        for observation in observations:
            result = observation.get("result")
            if isinstance(result, dict) and "records" in result and result.get("project_revision") != ProjectFiles.revision(current):
                observation["result"] = {"error": "index_revision_changed", "instruction": "Restart repository query at start=0 against current source."}
            if not isinstance(result, dict) or "lines" not in result:
                continue
            path = result.get("path")
            if path in current and current[path] != result.get("sha256"):
                # A successful edit invalidates earlier excerpts for every role, not just Coder.
                operation = observation["request"]
                observation["result"] = self._observe(project, operation)
                observation["refreshed_for_current_revision"] = True
        reqs = self._requirements(task["requirement_ids"])
        # Exact requirement text is inline when small; large documents remain addressable by ID.
        requirements = reqs if sum(len(r["text"]) for r in reqs) <= self._chars//2 else [{k:v for k,v in r.items() if k != "text"} for r in reqs]
        dependencies = [t for t in self.project_state.graph.tasks if t["id"] in task["dependencies"]]
        displayed_task = task
        if len(json.dumps(task, ensure_ascii=False)) > max(2400, self._chars//2):
            cache = self.project_state.project.setdefault("task_contracts", {})
            key = digest(task)
            if key not in cache:
                cache[key] = self._record("task_contract", task)
            displayed_task = {"id": task["id"], "description": task["description"], "kind": task["kind"],
                              "files": task["files"][:12], "full_task_artifact": cache[key],
                              "metadata_deferred": True}
        host_checks = self.tester.describe_checks()
        inline_checks = []
        for check in host_checks[:4]:
            if len(json.dumps(check)) <= 1000:
                inline_checks.append(check)
        revision = ProjectFiles.revision(current)
        latest_test = (self.project_state.project.get("active") or {}).get("latest_test")
        integration = self.project_state.project.get("integration")
        if latest_test and latest_test.get("project_revision") == revision and latest_test.get("integration"):
            integration = {**latest_test["integration"], "project_revision": revision, "matches_current_source": True}
        elif integration:
            bound_revision = integration.get("project_revision", self.project_state.project["project_revision"])
            integration = {**integration, "project_revision": bound_revision, "matches_current_source": bound_revision == revision}
        index = self._repository(project)
        relevant = []
        for path in task["files"][:2]:
            for symbol in index.files.get(path, {}).get("symbols", [])[:3]:
                relevant.append({"path": path, "source_sha256": symbol["source_sha256"],
                    "kind": symbol["kind"], "name": symbol["name"][:120],
                    "name_truncated": len(symbol["name"]) > 120,
                    "lineno": symbol["lineno"], "end_lineno": symbol["end_lineno"]})
        return {"task": displayed_task, "requirements": requirements,
            "repository": {"project_revision": index.project_revision, "coverage": index.coverage,
                "relevant_symbols": relevant,
                "queries": "repo_search text=literal; repo_dependencies path=tracked_file; start=0. Metadata is not source inspection. Use read on returned plain paths/lines."},
            "review_scope": "Review this task's contract and acceptance now. Original requirements describe the whole project; pending sibling/future tasks are checked at their own gates and final review.",
            "read_requirements_with": "read path=requirement:R1 (use actual ID)",
            "dependency_contracts": [{"id": t["id"], "contract": t["contract"]} for t in dependencies[:2]],
            "more_dependencies": len(dependencies)>4, "read_contracts_with": "read path=task:T1 (use actual ID)",
            "files": project.listing(0, 20), "observations": observations[-2:], "feedback": feedback,
            "changed_ranges": project.changed_ranges,
            "may_repair_existing_project_files": task["id"] in self.project_state.project.get("integration_tasks", []),
            "required_checks": [] if self.tester.entrypoint not in project.paths and not self._in_scope(task, self.tester.entrypoint) else inline_checks,
            "all_host_checks_artifact": self.project_state.project.get("host_checks"),
            "host_checks_deferred": len(inline_checks) != len(host_checks),
            "research_artifact": self.project_state.project.get("task_research", {}).get(task["id"]),
            "test_evidence": {**latest_test, "matches_current_source": latest_test.get("project_revision") == revision} if latest_test else None,
            "integration_evidence": integration,
            "scope": "Verify the whole original requirements for final_review; otherwise implement current task and preserve dependencies, without requiring future tasks."}

    async def _step(self, role, task, project, observations, feedback):
        method = self.coder.project_step if role == "coder" else self.reviewer.review_project
        for output in (2048, 1024, 512, 256):
            try:
                packet = self._packet(task, project, observations, feedback)
                packet["response_budget_tokens"] = output
                packet["edit_budget_guidance"] = "Keep replacement code below half this token budget; create a small scaffold then append/edit in later steps."
                return await self._invoke(method, packet, output=output)
            except RoutingError as exc:
                if not self._size_failure(exc):
                    raise
                self._chars = max(2000, self._chars//2)
                # Re-read a smaller exact range instead of slicing code or error strings silently.
                if observations and observations[-1].get("request", {}).get("action") == "read":
                    op = observations[-1]["request"]
                    observations[:] = [{"request": op, "result": self._observe(project, op)}]
        raise ValueError("task_requires_decomposition")

    @staticmethod
    def _first_uncovered(intervals, start, end):
        cursor = start
        for first, last in sorted(intervals):
            if first > cursor:
                break
            cursor = max(cursor, last)
            if cursor >= end:
                return None
        return cursor if cursor < end else None

    async def _evidence(self, task, project, observations, *, review=False, feedback=None):
        signatures = set()
        manifest = project.manifest()
        if review:
            # Supply an exact initial source slice rather than requiring a model call
            # merely to discover that an edit acknowledgement is not source evidence.
            changed_paths = [p for p, sha in manifest.items() if self.source.manifest().get(p) != sha]
            for path in changed_paths[:2]:
                ranges = project.changed_ranges.get(path, [(1, 80)])
                first, last = ranges[0] if ranges else (1, 80)
                operation = {"action": "read", "path": path, "start": first, "end": min(last, first+79)}
                observations.append({"request": operation, "result": self._observe(project, operation)})
            observations[:] = observations[-2:]
        inspected = {o["result"].get("path") for o in observations
                     if isinstance(o.get("result"), dict) and "lines" in o["result"]
                     and manifest.get(o["result"].get("path")) == o["result"].get("sha256")}
        changed = {p for p,h in manifest.items() if self.source.manifest().get(p) != h}
        read_lines, read_requirements = {}, {}
        for hop in range(self.max_hops):
            step = await self._step("reviewer" if review else "coder", task, project, observations, feedback)
            if review:
                inspected.update(o["result"]["path"] for o in observations
                                 if isinstance(o.get("result"), dict) and "lines" in o["result"]
                                 and manifest.get(o["result"].get("path")) == o["result"].get("sha256"))
                reqs = self._requirements(task["requirement_ids"])
                if sum(len(r["text"]) for r in reqs) <= self._chars//2:
                    for requirement in reqs:
                        read_requirements.setdefault(requirement["id"], []).append((0, len(requirement["text"])))
                for observation in observations[-2:]:
                    result = observation.get("result")
                    if not isinstance(result, dict):
                        continue
                    if "lines" in result and manifest.get(result.get("path")) == result.get("sha256"):
                        lines = result["lines"]
                        if lines:
                            read_lines.setdefault(result["path"], []).append((lines[0]["line"], lines[-1]["line"]+1))
                        elif result.get("next_line") is None and (observation["request"].get("start") or 1) == 1:
                            read_lines.setdefault(result["path"], []).append((1, 2))
                    if "requirement_id" in result:
                        read_requirements.setdefault(result["requirement_id"], []).append((result["start_offset"], result["end_offset"]))
            self._record("review_step" if review else "coder_step", {"task_id": task["id"], "operation": step})
            if self.project_state.project.get("active"):
                self.project_state.project["active"]["hop"] = hop+1
            action = step["action"]
            if action == "create" and not review:
                if step.get("sha256") is not None:
                    raise ValueError("invalid_create")
                step.update(action="edit", start=1, end=0)
                action = "edit"
            evidence_key = [{"request": {k:o.get("request", {}).get(k) for k in ("action", "path", "start", "end")},
                             "sha256": o.get("result", {}).get("sha256") if isinstance(o.get("result"), dict) else None}
                            for o in observations[-2:]]
            marker = digest({"step": {k:v for k,v in step.items() if k not in ("note", "issues")},
                             "revision": project.manifest(), "evidence": evidence_key})
            if action in ("read", "search", "list", "edit", "repo_search", "repo_dependencies"):
                if marker in signatures:
                    raise ValueError("repeated_operation_no_progress")
                signatures.add(marker)
            if action in ("read", "search", "list", "repo_search", "repo_dependencies"):
                try:
                    result = self._observe(project, step)
                except (OSError, ValueError) as exc:
                    result = {"error": str(exc) if isinstance(exc, ValueError) else "file_unavailable"}
                if isinstance(result, dict) and "lines" in result:
                    inspected.add(result["path"])
                    if review and feedback and "uninspected_changed_files" in feedback:
                        feedback = {"uninspected_changed_files": sorted(changed-inspected)}
                observations.append({"request": step, "result": result})
                observations[:] = observations[-2:]
                if review and feedback and any(k in feedback for k in ("unread_changed_range", "unread_requirement_range")):
                    feedback = {k:v for k,v in feedback.items() if k not in ("unread_changed_range", "unread_requirement_range")}
                    feedback["read_result_supplied"] = True
                    feedback["instruction"] = "Inspect the new exact observation and decide, or request another range if needed."
            elif action == "edit" and not review:
                try:
                    integration_scope = task["id"] in self.project_state.project.get("integration_tasks", []) and step["path"] in project.paths
                    if not self._in_scope(task, step["path"]) and not integration_scope:
                        raise ValueError("edit_outside_task_scope:allowed=" + ",".join(task["files"]))
                    result = project.edit(step["path"], step["sha256"], step["start"], step["end"], step["text"])
                except (OSError, ValueError) as exc:
                    result = {"error": str(exc) if isinstance(exc, ValueError) else "edit_io_error"}
                    if step["path"] in project.paths:
                        result["current_source"] = project.read(step["path"], 1, 80, chars=max(1000, self._chars//2))
                observations.append({"request": {"action": "edit", "path": step["path"]}, "result": result})
                observations[:] = observations[-2:]
                checks = self._checks(task)
                if ("error" not in result and self.checks_repeatable
                        and self.tester.entrypoint in project.paths
                        and {p for p in task["files"] if p.endswith(".py")} <= project.paths):
                    # Validation is host-owned: do not spend more LLM calls waiting for "done".
                    self.project_state.stage = "testing"
                    self.project_state.project["active"]["checks_started"] = True
                    self._save()
                    probe = await self.tester.test_project(project, checks)
                    ref = self._record("edit_validation", probe)
                    self.project_state.project["active"]["latest_test"] = {"artifact": ref, "status": probe["status"],
                        "project_revision": probe["project_revision"], "manifest": project.manifest(),
                        "changed_ranges": deepcopy(project.changed_ranges)}
                    self.project_state.stage = "coding"
                    if probe["status"] == "passed":
                        if task["id"] in self.project_state.project.get("integration_tasks", []) and self.integration_check:
                            integration = await self.integration_check(project)
                            integration_ref = self._record("integration_diagnostic", integration)
                            if integration.get("status") != "passed":
                                feedback = {"integration_failure": integration_ref,
                                    "summary": {k:v for k,v in integration.items() if k != "checks"},
                                    "read_more": "read path=artifact:"+integration_ref["path"]}
                                continue
                        return {"action": "done", "note": "Host edit validation passed"}
                    feedback = self._feedback(probe, ref)
            elif action == "split":
                raise ValueError("task_requires_decomposition")
            elif action == "done" and not review:
                return step
            elif review and action in ("approve", "reject"):
                deferred = {i["file"] for i in step["issues"] if i["file"] not in project.paths and not self._in_scope(task, i["file"])}
                if deferred and task["id"] != "final_review":
                    feedback = {"outside_current_task_scope": sorted(deferred),
                                "instruction": "These absent files are outside this task's implementation scope. Review the current task contract; whole-project completeness is checked at integration/final review."}
                    continue
                uninspected_issues = {i["file"] for i in step["issues"]} - inspected
                if uninspected_issues:
                    feedback = {"uninspected_changed_files": sorted(uninspected_issues),
                                "instruction": "Inspect the current file before asserting a defect."}
                    continue
                if action == "approve" and changed - inspected:
                    feedback = {"uninspected_changed_files": sorted(changed - inspected), "instruction": "Read these changed files before approval."}
                    continue
                if action == "approve":
                    missing = None
                    for path in sorted(changed):
                        for first, last in project.changed_ranges.get(path, []):
                            gap = self._first_uncovered(read_lines.get(path, []), first, last+1)
                            if gap is not None:
                                missing = {"action": "read", "path": path, "start": gap, "end": min(last, gap+79)}
                                break
                        if missing:
                            break
                    if missing:
                        feedback = {"unread_changed_range": missing, "instruction": "Inspect this exact changed range before approval."}
                        continue
                    if task["id"] == "final_review":
                        for requirement in self._requirements(task["requirement_ids"]):
                            gap = self._first_uncovered(read_requirements.get(requirement["id"], []), 0, len(requirement["text"]))
                            if gap is not None:
                                missing = {"action": "read", "path": "requirement:"+requirement["id"],
                                           "start": gap, "end": min(len(requirement["text"]), gap+2000)}
                                break
                        if missing:
                            feedback = {"unread_requirement_range": missing, "instruction": "Read the remaining exact requirements before final approval."}
                            continue
                if (action == "approve") != (not step["issues"]):
                    raise ValueError("invalid_review_output")
                if any(i["file"] not in project.paths for i in step["issues"]):
                    raise ValueError("invalid_review_file")
                return step
            else:
                raise ValueError("invalid_role_action")
            self._save()
        raise ValueError("task_requires_decomposition" if not review else "review_hop_budget_reached")

    @staticmethod
    def _feedback(report, ref):
        failed = []
        for check in report["checks"]:
            if check["passed"]:
                continue
            failed.append({"name": check["name"], "exit_code": check["exit_code"],
                "stdout_excerpt": check["stdout"][-1500:], "stderr_excerpt": check["stderr"][-2500:],
                "excerpted": len(check["stdout"])>1500 or len(check["stderr"])>2500})
        return {"project_revision": report["project_revision"], "status": report["status"],
                "failed_checks": failed[:2], "full_evidence": ref,
                "read_more": "read path=artifact:" + ref["path"] + "; start/end are character offsets"}

    @staticmethod
    def _failure_signature(feedback):
        import re
        material = {k:v for k,v in feedback.items() if k not in ("project_revision", "full_evidence", "read_more")}
        encoded = json.dumps(material, sort_keys=True, ensure_ascii=False)
        # Disposable test-root names and fresh evidence filenames are not new failure types.
        encoded = re.sub(r'ntg_check_project_[^/\\\\" ]+', '<test_root>', encoded)
        encoded = re.sub(r'project_[0-9a-f]{32}', '<project>', encoded)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _checks(self, task, project=None):
        library_only = project is not None and self.tester.entrypoint not in project.paths and not self._in_scope(task, self.tester.entrypoint)
        checks = {} if library_only else {"host:"+c["name"]: {**c, "name": "host:"+c["name"]} for c in self.tester.describe_checks()}
        for done in self.project_state.project["dispatch_order"]:
            previous = next(t for t in self.project_state.graph.tasks if t["id"] == done)
            checks.update({done+":"+c["name"]: {**c, "name": done+":"+c["name"]} for c in previous["checks"]})
        checks.update({task["id"]+":"+c["name"]: {**c, "name": task["id"]+":"+c["name"]} for c in task["checks"]})
        return list(checks.values())

    async def _execute_task(self, task):
        state = self.project_state
        resumed_candidate = self._resume_candidate is not None and self._resume_candidate[0] == task["id"]
        candidate = self._resume_candidate[1] if resumed_candidate else self.source.clone_to(self.manager.create())
        self._resume_candidate = None
        candidate.runtime_paths.update(p for t in state.graph.tasks for p in t.get("runtime_files", []))
        state.project["active"] = {"task_id": task["id"], "candidate_id": candidate.workspace.project_id, "checks_started": False}
        state.current_task, state.stage = task["id"], "coding"
        state.graph.start(task["id"])
        observations, feedback, seen = [], None, set()
        previous_failure_type = None
        # Initial observations are bounded source slices, not the whole project.
        for path in task["files"][:2]:
            if path in candidate.paths:
                op = {"action": "read", "path": path, "start": 1, "end": 80}
                observations.append({"request": op, "result": self._observe(candidate, op)})
        self._save()
        if task.get("research_questions"):
            from .researcher_agent import ResearchResult
            if not self.researcher:
                raise ValueError("research_unavailable")
            state.stage = "researching"
            cache_key = digest({"task": task, "project_revision": state.project["project_revision"]})
            cached = state.project.setdefault("research_cache", {}).get(cache_key)
            if cached:
                result = ResearchResult(**self._load_ref(cached))
            else:
                result = await self.researcher.run("Research current task",
                    {k:task[k] for k in ("id", "description", "dependencies", "research_questions")}, digest(task))
                cached = self._record("research", result.to_dict())
            if not result.verified or result.plan_sha256 != digest(task) or result.task_id != task["id"]:
                raise ValueError("research_failed")
            state.project["research_cache"][cache_key] = cached
            state.project.setdefault("task_research", {})[task["id"]] = cached
            feedback = {"research_evidence": result.context()}
            self._save()
        if task["id"] in state.project.get("integration_tasks", []) and self.integration_check:
            diagnostic = await self.integration_check(self.source)
            diagnostic_ref = self._record("integration_diagnostic", diagnostic)
            feedback = {"integration_failure": diagnostic_ref,
                        "summary": {k:v for k,v in diagnostic.items() if k != "checks"},
                        "read_more": "read path=artifact:"+diagnostic_ref["path"]}
        for revision in range(self.max_repairs+1):
            state.current_revision = revision+1
            state.stage = "repairing" if revision else "coding"
            self._save()
            already_implemented = False
            if (revision == 0 and (task["checks"] or resumed_candidate) and self.checks_repeatable
                    and self.tester.entrypoint in candidate.paths
                    and {p for p in task["files"] if p.endswith(".py")} <= candidate.paths):
                state.project["active"]["checks_started"] = True
                self._save()
                existing = await self.tester.test_project(candidate, self._checks(task))
                self._record("existing_implementation_check", existing)
                already_implemented = existing["status"] == "passed"
                if already_implemented and task["id"] in state.project.get("integration_tasks", []) and self.integration_check:
                    already_implemented = (await self.integration_check(candidate)).get("status") == "passed"
            if (task["kind"] != "integration" or revision) and not already_implemented:
                await self._evidence(task, candidate, observations, feedback=feedback)
            if self.tester.entrypoint not in candidate.paths and self._in_scope(task, self.tester.entrypoint):
                feedback = {"failure": "entrypoint_missing", "path": self.tester.entrypoint}
                continue
            state.stage = "testing"
            state.project["active"]["checks_started"] = True
            self._save()
            # Preserve the complete ordered obligation sequence and its data fixtures.
            report = await self.tester.test_project(candidate, self._checks(task, candidate))
            if (report["status"] == "passed" and task["id"] in state.project.get("integration_tasks", [])
                    and self.integration_check):
                integration = await self.integration_check(candidate)
                integration_ref = self._record("integration_diagnostic", integration)
                report["integration"] = {"status": integration.get("status"), "artifact": integration_ref,
                    "summary": {k:v for k,v in integration.items() if k != "checks"}}
                if integration.get("status") != "passed":
                    report["status"] = "failed"
            ref = self._record("test", report)
            state.project["active"]["latest_test"] = {"artifact": ref, "status": report["status"],
                                                      "project_revision": report["project_revision"], "manifest": candidate.manifest(),
                                                      "changed_ranges": deepcopy(candidate.changed_ranges),
                                                      "integration": report.get("integration")}
            state.stage = "reviewing"
            self._save()
            if report["status"] == "passed":
                review = await self._evidence(task, candidate, observations, review=True,
                    feedback={"test_report": ref, "status": "passed", "coverage": report["coverage"],
                              "checks": [c["name"] for c in report["checks"]]})
                if review["action"] == "approve":
                    manifest = candidate.manifest(refresh=True)
                    if ProjectFiles.revision(manifest) != report["project_revision"]:
                        raise ValueError("approval_source_changed")
                    approval = self._record("approval", {"task_id": task["id"], "task_sha256": digest(task),
                        "manifest": manifest, "project_revision": report["project_revision"], "review": review,
                        "test": {"status": "passed", "coverage": report["coverage"],
                                 "project_revision": report["project_revision"], "artifact": ref}})
                    state.project["approvals"][task["id"]] = approval
                    self.source = candidate
                    state.project.update(source_project_id=candidate.workspace.project_id, manifest=manifest,
                                         project_revision=report["project_revision"], active=None)
                    state.graph.finish(task["id"], passed=True)
                    state.project["dispatch_order"].append(task["id"])
                    state.current_task = state.current_revision = None
                    self._save()
                    return
                feedback = {"review_findings": review["issues"], "project_revision": report["project_revision"]}
            else:
                feedback = self._feedback(report, ref)
                if report.get("integration"):
                    feedback["integration"] = report["integration"]
            signature = self._failure_signature(feedback)
            failure_type = "review_rejected" if report["status"] == "passed" else (
                "syntax_error" if any(c["name"].startswith("compile:") and not c["passed"] for c in report["checks"]) else "functional_mismatch")
            self._record("revision", {"task_id": task["id"], "revision": revision+1,
                "project_revision": ProjectFiles.revision(candidate.manifest()), "test_artifact": ref,
                "changed_files": [p for p,h in candidate.manifest().items() if self.source.manifest().get(p) != h],
                "failure_type": failure_type, "failure_changed": previous_failure_type != failure_type if previous_failure_type else None,
                "failure_signature": signature, "repeated_failure": signature in seen})
            previous_failure_type = failure_type
            if signature in seen:
                raise ValueError("repair_no_progress")
            seen.add(signature)
            # Keep exact active failure evidence; the Coder may request new exact source slices.
            observations.clear()
            state.project["last_failure"] = self._record("repair_feedback", feedback)
        raise ValueError("repair_budget_reached")

    async def _final_review(self):
        state = self.project_state
        for requirement in state.project["requirements"]:
            owners = [t for t in state.graph.tasks if requirement["id"] in t["requirement_ids"]]
            task = {"id": "final_review", "description": "Verify the complete original requirement, including cross-file behavior.",
                    "dependencies": [t["id"] for t in owners], "kind": "integration",
                    "files": sorted(self.source.paths)[:12], "contract": "Preserve all completed component contracts.",
                    "acceptance": ["All original behavior in this requirement is implemented and integrated."],
                    "checks": [], "research_questions": [], "requirement_ids": [requirement["id"]]}
            observations = []
            for path in task["files"][:2]:
                op = {"action": "read", "path": path, "start": 1, "end": 80}
                observations.append({"request": op, "result": self._observe(self.source, op)})
            review = await self._evidence(task, self.source, observations, review=True,
                feedback={"integration": state.project["integration"], "approved_task_count": len(owners)})
            ref = self._record("final_review", {"requirement": requirement["id"], "review": review,
                                                       "project_revision": state.project["project_revision"]})
            if review["action"] == "reject":
                return requirement, review, ref
            state.project.setdefault("final_reviews", {})[requirement["id"]] = ref
        return None

    def _integration_repair(self, requirement, issues, ref):
        state = self.project_state
        count = state.project.get("integration_repairs", 0)
        if count >= self.max_repairs:
            raise ValueError("integration_repair_budget_reached")
        tasks = state.graph.tasks
        roots = [int(t["id"][1:].split(".")[0]) for t in tasks]
        task = {"id": f"T{max(roots)+1}", "description": "Repair concrete whole-project integration failures.",
            "dependencies": [t["id"] for t in tasks if not any(t["id"] in n["dependencies"] for n in tasks)],
            "kind": "leaf", "files": sorted({i["file"] for i in issues}) or sorted(self.source.paths)[:12],
            "contract": "Preserve all approved APIs. Read exact failure evidence at artifact:" + ref["path"],
            "acceptance": ["Resolve every concrete integration finding; preserve all earlier checks."],
            "checks": [], "research_questions": [], "requirement_ids": [requirement["id"]]}
        self._validate_task(task)
        updated = TaskGraph(tasks+[task])
        statuses = updated.states
        statuses.update(state.graph.states)
        updated.restore(statuses)
        state.graph = updated
        state.project["integration_repairs"] = count+1
        state.project.setdefault("integration_tasks", []).append(task["id"])
        self._save()

    def _parent_repair(self, parent):
        """Insert a focused child before re-running an existing parent integration gate."""
        state = self.project_state
        repairs = state.project.setdefault("parent_repairs", {})
        if repairs.get(parent["id"], 0) >= self.max_repairs:
            raise ValueError("parent_repair_budget_reached")
        tasks = state.graph.tasks
        if len(tasks) >= self.max_tasks:
            raise ValueError("project_task_budget_reached")
        children = [t for t in tasks if t.get("parent") == parent["id"]]
        suffix = max([int(t["id"].split(".")[-1]) for t in children] or [0])+1
        repair = deepcopy(parent)
        repair.update(id=parent["id"]+f".{suffix}", parent=parent["id"], kind="leaf",
                      description="Repair concrete parent integration failures while preserving existing contracts.")
        evidence = state.project.get("last_failure")
        if evidence:
            repair["contract"] = parent["contract"] + " Read exact failures at artifact:" + evidence["path"]
        amended_parent = deepcopy(parent)
        amended_parent["dependencies"] = parent["dependencies"]+[repair["id"]]
        graph = TaskGraph([amended_parent if t["id"] == parent["id"] else t for t in tasks]+[repair])
        statuses = graph.states
        statuses.update(state.graph.states)
        graph.restore(statuses)
        state.graph = graph
        repairs[parent["id"]] = repairs.get(parent["id"], 0)+1
        self._record("integration_child_repair", {"parent": parent["id"], "repair": repair})
        self._save()

    async def run(self, goal=None, *, resume=None):
        self._calls, self._chars = 0, 12000
        if resume:
            if goal is not None:
                saved = self.manager.open(resume).artifacts.read_json("project_state.json")
                if hashlib.sha256(goal.encode()).hexdigest() != saved.get("project", {}).get("goal_sha256"):
                    raise ValueError("checkpoint_goal_mismatch")
            self._resume(resume)
        else:
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("invalid_goal")
            self._new(goal)
        state = self.project_state
        if state.status == "passed":
            return self.result()
        try:
            async with asyncio.timeout(self.run_timeout):
                if state.graph is None:
                    await self._plan()
                while True:
                    while not all(v == "passed" for v in state.graph.states.values()):
                        ready = state.graph.ready()
                        if not ready:
                            raise ValueError("scheduler_no_progress")
                        task = ready[0]
                        original = deepcopy(task)
                        self._validate_task(task)
                        if task != original:
                            state.graph.amend_pending(task)
                            self._record("check_contract_correction", {"before": original["checks"], "after": task["checks"]})
                            self._save()
                        if task["kind"] == "component":
                            await self._expand(task)
                            continue
                        try:
                            await self._execute_task(task)
                        except ValueError as exc:
                            if str(exc) not in ("task_requires_decomposition", "repair_no_progress",
                                               "repair_budget_reached", "repeated_operation_no_progress"):
                                raise
                            # Discard the unapproved candidate before expanding; dependencies stay approved.
                            states = state.graph.states
                            states[task["id"]] = "pending"
                            state.graph.restore(states)
                            state.project["active"] = None
                            state.current_task = None
                            if task["kind"] == "integration":
                                self._parent_repair(task)
                            else:
                                await self._expand(task)
                    state.stage = "integration"
                    if self.tester.entrypoint not in self.source.paths:
                        failure = {"failure": "entrypoint_missing", "path": self.tester.entrypoint}
                        ref = self._record("integration_failure", failure)
                        self._integration_repair(state.project["requirements"][0],
                            [{"file": self.tester.entrypoint, "description": "Create the required project entrypoint", "severity": "high"}], ref)
                        continue
                    if self.integration_check:
                        before = self.source.manifest()
                        integration = await self.integration_check(self.source)
                        integration_ref = self._record("integration", integration)
                        if before != self.source.manifest():
                            raise ValueError("integration_source_changed")
                        state.project["integration"] = {"status": integration.get("status"), "artifact": integration_ref,
                                                        "project_revision": state.project["project_revision"]}
                        if integration.get("status") != "passed":
                            ref = self._record("integration_failure", integration)
                            issue = {"severity": "high", "file": self.tester.entrypoint,
                                     "description": "Host integration checks failed; read artifact:" + ref["path"]}
                            self._integration_repair(state.project["requirements"][0], [issue], ref)
                            continue
                    else:
                        state.project["integration"] = {"status": "passed", "scope": "accumulated_task_checks_and_reviews"}
                    # Final review verifies requirement coverage in small groups, not one giant prompt.
                    covered = {rid for t in state.graph.tasks if state.graph.states[t["id"]] == "passed" for rid in t["requirement_ids"]}
                    if covered != {r["id"] for r in state.project["requirements"]}:
                        raise ValueError("requirement_coverage_missing")
                    rejected = await self._final_review()
                    if rejected:
                        requirement, review, ref = rejected
                        self._integration_repair(requirement, review["issues"], ref)
                        continue
                    break
                state.status, state.reason, state.stage = "passed", None, "finished"
        except asyncio.CancelledError:
            state.status, state.reason = "paused", "cancelled"
            self._save()
            raise
        except (ValueError, RoutingError, OSError, TimeoutError) as exc:
            state.status = "paused"
            code = exc.code if isinstance(exc, RoutingError) else str(exc) if isinstance(exc, ValueError) else "run_timeout" if isinstance(exc, TimeoutError) else "workspace_io_error"
            state.reason = code if len(code) < 120 and all(c.isalnum() or c in "_:" for c in code) else "project_execution_failed"
        self._save()
        return self.result()

    def result(self):
        state = self.project_state
        manifest = self.source.manifest(refresh=True)
        verified = (state.status == "passed" and manifest == state.project["manifest"]
                    and ProjectFiles.revision(manifest) == state.project["project_revision"]
                    and all(v == "passed" for v in state.graph.states.values()))
        result = {"status": state.status, "verified": verified, "reason": state.reason or "all_tasks_tested_and_reviewed",
            "run_project_id": state.metadata.project_id, "final_project_id": self.source.workspace.project_id if verified else None,
            "project_revision": state.project["project_revision"], "manifest": manifest,
            "task_graph": state.graph.to_dict() if state.graph else {}, "execution_order": list(state.project["dispatch_order"]),
            "state_revision": state.state_revision, "project_state_path": str(state.metadata.path / "project_state.json"),
            "calls": state.project["calls"], "expansions": state.project["expansions"],
            "last_failure": deepcopy(state.project.get("last_failure"))}
        files = state.metadata.artifacts
        try:
            expected = files.file_sha256("result.json")
        except FileNotFoundError:
            expected = None
        files.write_json("result.json", result, expected)
        saved = files.read_json("project_state.json")
        if saved["state_revision"] != result["state_revision"] or saved["status"] != result["status"] or saved["project"]["project_revision"] != result["project_revision"]:
            raise ValueError("state_result_mismatch")
        return result
