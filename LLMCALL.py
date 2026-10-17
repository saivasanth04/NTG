import asyncio
import time
from pathlib import Path

from ntg import (
    CoderAgent, CodingWorkflow, PlannerAgent, PlannedWorkflow,
    ReviewerAgent, RoutingError, RoutingService, TesterAgent, WorkspaceManager,
)


async def run_with_progress(planned, goal, *, resume=None):
    """Show liveness while preserving workflow deadlines and cancellation."""
    started = time.monotonic()
    pending = asyncio.create_task(planned.run(goal, resume=resume) if resume else planned.run(goal))
    print("Running Planner -> optional Researcher -> Coder -> compile -> Tester -> Reviewer...", flush=True)
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=15)
            if done:
                return await pending
            state = getattr(planned, "project_state", None)
            detail = ""
            if state is not None:
                statuses = state.graph.states if state.graph else {}
                done_count = sum(s == "passed" for s in statuses.values())
                detail = (f"stage={state.stage}; task={state.current_task}; "
                          f"revision={state.current_revision}; completed={done_count}/{len(statuses)}; ")
            print(f"Workflow active ({time.monotonic() - started:.0f}s); {detail}"
                  "waiting for bounded model calls or execution checks.", flush=True)
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def legacy_main(*, researcher=None):
    root = Path(__file__).resolve().parent
    try:
        llm = await RoutingService.from_env(
            root / ".env", root / "policy.json", max_attempts=8, attempt_timeout=20,
        )
        workflow = CodingWorkflow(
            CoderAgent(llm, entrypoint="app.py"), TesterAgent(llm, entrypoint="app.py", checks=[
                {"name": "no_arguments", "arguments": [], "exit_code": 0,
                 "stdout_contains": ["usage:", "add", "list", "total"]},
                {"name": "help", "arguments": ["--help"], "exit_code": 0,
                 "stdout_contains": ["usage:", "add", "list", "total"]},
            ]),
            WorkspaceManager(root / "workspaces"), max_repairs=2,
            reviewer=ReviewerAgent(llm, entrypoint="app.py"), entrypoint="app.py",
        )
        result = await run_with_progress(PlannedWorkflow(PlannerAgent(llm, research_enabled=researcher is not None), workflow, researcher=researcher),
            "Build a standard-library command-line expense tracker in app.py using "
            "argparse commands add, list, total; persist expenses in a JSON file in the "
            "current project; reject nonpositive or invalid amounts; no arguments "
            "prints help and exits zero."
        )
    except RoutingError as exc:
        print(f"failed: {exc.code}")
        return 1
    except ValueError as exc:
        print(f"blocked: {exc}")
        return 1

    if result.planning_error:
        print(f"Planning detail: {result.planning_error}")
    if result.plan:
        for task in result.plan.tasks:
            print(f"{task['id']}: {task['description']} (dependencies: {task['dependencies']})")
    print(f"Execution order: {result.execution_order}")
    for task_id, node in result.task_graph.items():
        print(f"  {task_id}: {node['state']}; dependencies={node['dependencies']}; "
              f"reason={node['reason']}")
    for task in result.tasks:
        print(f"Task {task.task_id}: {task.status}")
        print(f"  Research: {task.research.status if task.research else 'skipped'}")
        if task.research:
            print(f"    Reason: {task.research.reason}")
            for source in task.research.sources:
                print(f"    {source['id']}: {source['url']}")
            if task.research_artifact:
                print(f"    Report: {root / 'workspaces' / result.run_project_id / task.research_artifact}")
        for revision in task.result.revisions:
            project = root / "workspaces" / revision.project_id
            artifact = project / revision.source_snapshot if revision.source_snapshot else project
            print(f"  Revision {revision.revision}: {artifact}")
            print(f"    Changes: {revision.progress}; failure={revision.failure_type}; "
                  f"failure_changed={revision.failure_changed}; rolled_back={revision.rolled_back}")
            if revision.generation_error:
                print(f"    Generation: {revision.generation_error}")
            if revision.report:
                print(f"    Tester: {revision.report.status}: {revision.report.feedback}")
            if revision.review:
                print(f"    Reviewer: {revision.review.status}")
                for issue in revision.review.issues:
                    print(f"      {issue['severity']}: {issue['description']}")
                if revision.review.error:
                    print(f"      {revision.review.error}")
    if result.run_project_id and result.project_state_path:
        print(f"Project state: {root / 'workspaces' / result.run_project_id / result.project_state_path}")
    if result.run_project_id:
        print(f"Run metadata: {root / 'workspaces' / result.run_project_id}")
    if result.verified:
        print(f"Final project: {root / 'workspaces' / result.final_project_id}")
    print(f"{result.status}: {result.reason}")
    return 0 if result.verified else 1


DEFAULT_GOAL = (
    "Build a standard-library command-line expense tracker with app.py entrypoint. "
    "Use argparse commands add, list, total; add AMOUNT accepts a positive finite number "
    "with no required description argument. Persist records with an amount field in "
    "expenses.json in the current directory. Reject zero, negative, nonnumeric, NaN and "
    "infinite amounts with nonzero exit. No arguments prints help and exits zero. "
    "list prints all amounts; total prints their sum. Use modules when helpful."
)


async def verify_expense_project(project):
    """Host integration checks run in a disposable copy, preserving approved source/data."""
    import tempfile
    import json
    from dataclasses import asdict
    from ntg import ExecutionEvidence
    runner = TesterAgent(entrypoint="app.py")
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="ntg_integration_") as tmp:
        clone = project.clone_to(WorkspaceManager(Path(tmp) / "workspaces").create())
        async def call(args, expected=None, contains=()):
            import sys
            evidence = ExecutionEvidence(clone.workspace.project_id, clone.revision(clone.manifest()),
                [sys.executable, "-B", "-X", "utf8", str(clone.files._regular("app.py")), *args])
            await runner._execute(clone.workspace.path, evidence)
            passed = (not evidence.timed_out and not evidence.infrastructure_error and not evidence.output_limited
                      and (evidence.exit_code == expected if expected is not None else evidence.exit_code != 0)
                      and all(fragment in evidence.stdout for fragment in contains))
            outcomes.append({"arguments": args, "passed": passed, **asdict(evidence)})
            return passed
        for args in ([], ["--help"]):
            await call(args, 0, ["usage:", "add", "list", "total"])
        await call(["add", "12.50"], 0)
        await call(["add", "7.50"], 0)
        await call(["list"], 0, ["12.5", "7.5"])
        await call(["total"], 0, ["20"])
        path = clone.workspace.path / "expenses.json"
        data = path.read_bytes() if path.exists() else None
        persistence = False
        persistence_evidence = {"expected": "two JSON records with amount fields summing to 20", "observed": "missing or invalid JSON"}
        try:
            entries = json.loads(data)
            persistence_evidence["observed"] = {"type": type(entries).__name__,
                "count": len(entries) if isinstance(entries, list) else None,
                "sample": entries[:2] if isinstance(entries, list) else None}
            persistence = isinstance(entries, list) and len(entries) == 2 and sum(e["amount"] for e in entries) == 20
        except (ValueError, TypeError, KeyError):
            pass
        for amount in ("0", "-1", "invalid", "nan", "inf"):
            await call(["add", amount])
        unchanged = data is not None and path.exists() and path.read_bytes() == data
        return {"status": "passed" if persistence and unchanged and all(x["passed"] for x in outcomes) else "failed",
                "persistence": persistence, "persistence_evidence": persistence_evidence,
                "rejected_inputs_preserved_data": unchanged, "checks": outcomes}


async def main(*, researcher=None, goal=None, resume=None, legacy=False, checks_override=None, repeatable_checks=False, max_calls=200, run_timeout=1200, max_expansions=512):
    if legacy:
        return await legacy_main(researcher=researcher)
    from ntg.project_workflow import ProjectWorkflow
    if goal is not None and not goal.strip():
        print("blocked: invalid_goal")
        return 1
    if min(max_calls, run_timeout, max_expansions) <= 0:
        print("blocked: invalid_project_limits")
        return 1
    root = Path(__file__).resolve().parent
    try:
        llm = await RoutingService.from_env(root / ".env", root / "policy.json", max_attempts=8, attempt_timeout=20)
        checks = [{"name": "startup", "arguments": [], "exit_code": 0, "stdout_contains": []}]
        demo = goal is None or goal == DEFAULT_GOAL
        if resume and goal is None:
            metadata = WorkspaceManager(root / "workspaces").open(resume)
            saved_goal = metadata.artifacts.execute("read_file", {"path": "goal.txt"})
            demo = saved_goal["ok"] and saved_goal["data"]["content"] == DEFAULT_GOAL
        if demo:
            checks = [{"name": "no_arguments", "arguments": [], "exit_code": 0, "stdout_contains": ["usage:", "add", "list", "total"]},
                      {"name": "help", "arguments": ["--help"], "exit_code": 0, "stdout_contains": ["usage:", "add", "list", "total"]}]
        if checks_override is not None:
            checks = checks_override
        workflow = ProjectWorkflow(PlannerAgent(llm, research_enabled=researcher is not None),
            CoderAgent(llm, entrypoint="app.py"), TesterAgent(entrypoint="app.py", checks=checks, project_mode=True),
            ReviewerAgent(llm, entrypoint="app.py"), WorkspaceManager(root / "workspaces"),
            researcher=researcher, checks_repeatable=(demo and checks_override is None) or repeatable_checks,
            integration_check=verify_expense_project if demo else None,
            max_calls=max_calls, run_timeout=run_timeout, max_expansions=max_expansions)
        result = await run_with_progress(workflow, goal if resume else goal or DEFAULT_GOAL, resume=resume)
    except (RoutingError, ValueError, OSError) as exc:
        print(f"blocked: {exc.code if isinstance(exc, RoutingError) else 'workspace_io_error' if isinstance(exc, OSError) else exc}")
        return 1
    print(f"Execution order: {result['execution_order']}")
    for task_id, node in result["task_graph"].items():
        print(f"  {task_id}: {node['state']}; dependencies={node['dependencies']}")
    print(f"Project state: {result['project_state_path']}")
    print(f"Run metadata: {root / 'workspaces' / result['run_project_id']}")
    if result["verified"]:
        print(f"Final project: {root / 'workspaces' / result['final_project_id']}")
        print(f"Project revision: {result['project_revision']}; files={len(result['manifest'])}")
    if result.get("last_failure") and not result["verified"]:
        print(f"Failure evidence: {root / 'workspaces' / result['run_project_id'] / result['last_failure']['path']}")
    print(f"{result['status']}: {result['reason']}")
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    import sys
    import argparse
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="NTG adaptive project workflow")
    selected = parser.add_mutually_exclusive_group()
    selected.add_argument("--goal")
    selected.add_argument("--goal-file", type=Path)
    parser.add_argument("--resume", help="Run project_<id> from its validated checkpoint")
    parser.add_argument("--legacy", action="store_true", help="Use the previous single-file workflow")
    parser.add_argument("--checks-file", type=Path, help="Host-defined JSON list of app.py acceptance checks")
    parser.add_argument("--repeatable-checks", action="store_true", help="Declare supplied checks safe to repeat after interruption")
    parser.add_argument("--max-calls", type=int, default=200, help="Model calls allowed in this invocation before checkpointing")
    parser.add_argument("--run-timeout", type=int, default=1200, help="Seconds allowed in this invocation before checkpointing")
    parser.add_argument("--max-expansions", type=int, default=512, help="Parent-task expansion budget across the project")
    args = parser.parse_args()
    if args.legacy and (args.resume or args.goal or args.goal_file or args.checks_file):
        parser.error("--legacy runs the original fixed demo; use project mode for goal/checkpoint options")
    try:
        goal = args.goal_file.read_text(encoding="utf-8") if args.goal_file else args.goal
        checks = __import__("json").loads(args.checks_file.read_text(encoding="utf-8")) if args.checks_file else None
    except (OSError, UnicodeError, ValueError):
        parser.error("goal/checks input must be a readable UTF-8 file; checks must contain valid JSON")
    raise SystemExit(asyncio.run(main(goal=goal, resume=args.resume, legacy=args.legacy, checks_override=checks, repeatable_checks=args.repeatable_checks,
        max_calls=args.max_calls, run_timeout=args.run_timeout, max_expansions=args.max_expansions)))
