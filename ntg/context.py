"""Detached role views with explicit source, metadata and evidence budgets."""
import hashlib
import json
from copy import deepcopy

from .tester_agent import check_passed


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _bounded(value, limit=24576):
    if _size(value) > limit:
        raise ValueError("context_budget_exceeded")
    return deepcopy(value)


def planner_context(goal, *, research_enabled=True):
    return _bounded({"goal": goal, "research_enabled": research_enabled})


def researcher_context(state, task):
    return _bounded({"task_id": task["id"], "plan_sha256": state.plan.plan_sha256,
                     "questions": task.get("research_questions", [])})


def coder_context(state, task, source, digest, checks):
    if source is not None and (len(source.encode("utf-8")) > 65536
            or hashlib.sha256(source.encode("utf-8")).hexdigest() != digest):
        raise ValueError("approved_artifact_unavailable")
    completed = state.completed
    specification = {"original_goal": state.goal, "current_task": deepcopy(task),
                     "required_checks": deepcopy(checks), "completed_tasks": completed,
                     "acceptance_scope": "Complete this task and preserve all completed features. " + (
                         "This is the final task: satisfy the whole original goal."
                         if len(completed) == len(state.plan.tasks) - 1 else
                         "Intermediate task: future work need not be implemented yet.")}
    if state.active and state.active.research:
        report = state.active.research
        if not state.active.research_verified or report.plan_sha256 != state.plan.plan_sha256:
            raise ValueError("invalid_research_output")
        evidence = report.context()
        if _size(evidence) > 8192:
            raise ValueError("context_budget_exceeded")
        specification["research_evidence"] = evidence
    # Only source/hash are in the initial Coder context. Requirements live in task once.
    _bounded(specification)
    return deepcopy(specification), {"prior_source": source, "prior_source_sha256": digest}


def _failure(report):
    if report is None:
        return None
    data = report.to_dict()
    # feedback includes a duplicate of execution streams; retain diagnosis only here.
    return {"project_id": data["project_id"], "source_sha256": data["source_sha256"],
            "status": data["status"], "failure_type": data["failure_type"],
            "failure_signature": data["failure_signature"],
            "source_context": data["source_context"],
            "feedback": data["feedback"].split("\nActual stderr:", 1)[0],
            "compilation": data["compilation"] if not report.compilation or not report.compilation.passed else None,
            "checks": [c for c in data["checks"] if not check_passed(c)]}


def repair_context(task_context, digest, checks, active_report, last_attempt, history, rejection):
    # Never propagate initial prior_source or complete conversation/history into repairs.
    evidence = {"execution_evidence": _failure(active_report),
                "last_attempt_evidence": _failure(last_attempt.report)
                if last_attempt and last_attempt.rolled_back and last_attempt.report is not active_report else None}
    _bounded(evidence, 160 * 1024)
    review = last_attempt.review if last_attempt else None
    metadata = {"current_source_sha256": digest, "checks": deepcopy(checks),
                "passing_checks": [c["name"] for c in active_report.checks if check_passed(c)] if active_report else [],
                "review_findings": {"status": review.status, "issues": deepcopy(review.issues),
                                    "source_sha256": review.source_sha256,
                                    "matches_current_source": review.source_sha256 == digest} if review else None,
                "repair_history": [{"revision": r.revision, "source_sha256": r.source_sha256,
                    "failure_type": r.failure_type, "failure_signature": r.failure_signature,
                    "progress": r.progress, "rolled_back": r.rolled_back,
                    "changed_line_count": sum(line.startswith(("+", "-")) for line in r.changes.splitlines())}
                    for r in history[-3:]],
                "rejected_proposal": {k: deepcopy(v) for k, v in (rejection or {}).items() if k != "source"}}
    if rejection and "source" in rejection and last_attempt:
        metadata["rejected_changes"] = last_attempt.changes
    # Preserve standalone contextual requirements, but never resend prior source.
    if task_context:
        try:
            supplied = json.loads(task_context)
        except (ValueError, TypeError):
            supplied = {"requirements": task_context}
        if isinstance(supplied, dict):
            relevant = {k: supplied[k] for k in (
                "requirements", "research_evidence", "original_goal", "current_task",
                "completed_tasks", "acceptance_scope") if k in supplied}
            if relevant:
                metadata["task_context"] = relevant
    # Planned requirements/research are already the task argument to CoderAgent.repair.
    _bounded(metadata)
    return {**metadata, **evidence}


def reviewer_context(specification, source, report, entrypoint):
    try:
        requirements = json.loads(specification)
    except (ValueError, TypeError):
        requirements = specification
    _bounded(requirements)
    if len(source.encode("utf-8")) > 65536:
        raise ValueError("context_budget_exceeded")
    data = report.to_dict(compact=True)
    data.pop("program", None)  # same evidence is already in checks
    return {"specification": deepcopy(requirements), "file": entrypoint, "source": source,
            "project_id": report.project_id, "source_sha256": report.source_sha256,
            "execution_evidence": _bounded(data)}


def role_context(role, *args, **kwargs):
    builders = {"planner": planner_context, "researcher": researcher_context,
                "coder": coder_context, "repair": repair_context, "reviewer": reviewer_context}
    if not isinstance(role, str) or role not in builders:
        raise ValueError("invalid_context_role")
    return builders[role](*args, **kwargs)
