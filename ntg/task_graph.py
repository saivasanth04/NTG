"""Bounded deterministic task graph; no execution or artifact ownership."""
from copy import deepcopy
import re


class TaskGraph:
    def __init__(self, tasks):
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("invalid_task_count")
        self._tasks = {}
        for task in tasks:
            if (not isinstance(task, dict) or not {"id", "description", "dependencies"} <= set(task)
                    or set(task) - {"id", "description", "dependencies", "research_questions", "kind", "files", "runtime_files", "contract", "acceptance", "parent", "checks", "requirement_ids"}
                    or not isinstance(task["id"], str)
                    or not re.fullmatch(r"T[1-9][0-9]*(?:\.[1-9][0-9]*)*", task["id"])
                    or not isinstance(task["description"], str) or not task["description"].strip()
                    or len(task["description"]) > 2000
                    or not isinstance(task["dependencies"], list)
                    or any(not isinstance(d, str) for d in task["dependencies"])):
                raise ValueError("invalid_task_fields")
            questions = task.get("research_questions", [])
            if (not isinstance(questions, list) or len(questions) > 3
                    or any(not isinstance(q, str) or not q.strip() or len(q) > 300 for q in questions)
                    or len({q.strip().casefold() for q in questions}) != len(questions)):
                raise ValueError("invalid_research_questions")
            name = task["id"]
            if name in self._tasks:
                raise ValueError("duplicate_task_id")
            self._tasks[name] = deepcopy(task)
        for name, task in self._tasks.items():
            deps = task["dependencies"]
            if len(deps) != len(set(deps)):
                raise ValueError("duplicate_dependency")
            if name in deps:
                raise ValueError("self_dependency")
            if not set(deps) <= self._tasks.keys():
                raise ValueError("unknown_dependency")
        remaining, visited = set(self._tasks), set()
        while remaining:
            ready = {n for n in remaining if set(self._tasks[n]["dependencies"]) <= visited}
            if not ready:
                raise ValueError("cyclic_dependencies")
            visited.update(ready)
            remaining.difference_update(ready)
        self._states = dict.fromkeys(self._tasks, "pending")
        self._reasons = dict.fromkeys(self._tasks, None)

    @property
    def states(self):
        return self._states.copy()

    def ready(self):
        return [deepcopy(self._tasks[n]) for n in sorted(self._tasks, key=lambda n: tuple(int(p) for p in n[1:].split(".")))
                if self._states[n] == "pending"
                and all(self._states[d] == "passed" for d in self._tasks[n]["dependencies"])]

    def start(self, task_id):
        if "running" in self._states.values() or task_id not in {t["id"] for t in self.ready()}:
            raise ValueError("task_not_ready")
        self._states[task_id] = "running"

    def finish(self, task_id, *, passed):
        if type(passed) is not bool or self._states.get(task_id) != "running":
            raise ValueError("invalid_task_transition")
        self._states[task_id] = "passed" if passed else "failed"

    def stop(self, reason, failed_task_id=None):
        if ("running" in self._states.values() or not isinstance(reason, str) or not reason
                or (failed_task_id is not None and self._states.get(failed_task_id) != "failed")):
            raise ValueError("invalid_graph_stop")
        for name, status in self._states.items():
            if status == "failed" and self._reasons[name] is None:
                self._reasons[name] = reason
        while True:
            blocked = [n for n, status in self._states.items() if status == "pending"
                       and any(self._states[d] in ("failed", "blocked")
                               for d in self._tasks[n]["dependencies"])]
            if not blocked:
                break
            for name in blocked:
                self._states[name], self._reasons[name] = "blocked", "dependency_failed"
        for name, status in self._states.items():
            if status == "pending":
                self._states[name], self._reasons[name] = "cancelled", reason

    @property
    def tasks(self):
        return deepcopy(list(self._tasks.values()))

    def amend_pending(self, task):
        name = task["id"]
        if self._states.get(name) != "pending" or task["dependencies"] != self._tasks[name]["dependencies"]:
            raise ValueError("invalid_graph_amendment")
        validated = TaskGraph([task if t["id"] == name else t for t in self.tasks])
        self._tasks = validated._tasks

    def expand(self, task_id, children):
        """Amend only a pending parent; it becomes an integration barrier for its children."""
        if self._states.get(task_id) != "pending" or not children:
            raise ValueError("invalid_graph_expansion")
        if any(t.get("parent") == task_id for t in self._tasks.values()):
            raise ValueError("already_expanded")
        parent = deepcopy(self._tasks[task_id])
        child_ids = [t["id"] for t in children]
        if (len(child_ids) != len(set(child_ids)) or set(child_ids) & self._tasks.keys()
                or any(not n.startswith(task_id + ".") for n in child_ids)):
            raise ValueError("invalid_child_ids")
        additions = []
        for child in children:
            item = deepcopy(child)
            if not set(item["dependencies"]) <= set(child_ids):
                raise ValueError("invalid_child_dependencies")
            item["dependencies"] = list(dict.fromkeys(parent["dependencies"] + item["dependencies"]))
            item["parent"] = task_id
            additions.append(item)
        parent["dependencies"] = child_ids
        parent["kind"] = "integration"
        amended = [parent if t["id"] == task_id else t for t in self.tasks] + additions
        validated = TaskGraph(amended)
        # Validate before mutation: old completed evidence and statuses remain authoritative.
        validated._states.update(self._states)
        validated._reasons.update(self._reasons)
        self._tasks, self._states, self._reasons = validated._tasks, validated._states, validated._reasons

    def restore(self, states, reasons=None):
        if set(states) != set(self._tasks) or any(v not in ("pending", "running", "passed", "failed", "blocked", "cancelled") for v in states.values()):
            raise ValueError("invalid_graph_checkpoint")
        if sum(v == "running" for v in states.values()) > 1:
            raise ValueError("invalid_graph_checkpoint")
        for name, status in states.items():
            if status in ("passed", "running") and any(states[d] != "passed" for d in self._tasks[name]["dependencies"]):
                raise ValueError("invalid_graph_checkpoint")
        self._states = dict(states)
        self._reasons = {n: (reasons or {}).get(n) for n in states}

    def to_dict(self):
        return {name: {"dependencies": list(self._tasks[name]["dependencies"]),
                       "state": self._states[name], "reason": self._reasons[name]}
                for name in sorted(self._tasks, key=lambda n: tuple(int(p) for p in n[1:].split(".")))}
