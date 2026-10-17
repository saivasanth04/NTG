import asyncio
import time
from pathlib import Path

from ntg import (
    CoderAgent, CodingWorkflow, PlannerAgent, PlannedWorkflow,
    ReviewerAgent, RoutingError, RoutingService, TesterAgent, WorkspaceManager,
)


async def run_with_progress(planned, goal):
    """Show liveness while preserving workflow deadlines and cancellation."""
    started = time.monotonic()
    pending = asyncio.create_task(planned.run(goal))
    print("Running Planner -> optional Researcher -> Coder -> compile -> Tester -> Reviewer...", flush=True)
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=15)
            if done:
                return await pending
            print(f"Workflow active ({time.monotonic() - started:.0f}s); "
                  "waiting for bounded model calls or execution checks.", flush=True)
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def main(*, researcher=None):
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
        result = await run_with_progress(PlannedWorkflow(PlannerAgent(llm), workflow, researcher=researcher),
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
    if result.run_project_id:
        print(f"Run metadata: {root / 'workspaces' / result.run_project_id}")
    if result.verified:
        print(f"Final project: {root / 'workspaces' / result.final_project_id}")
    print(f"{result.status}: {result.reason}")
    return 0 if result.verified else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
