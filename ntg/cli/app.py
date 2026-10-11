"""CLI application runner for the NTG Coding Agent and Multi-Provider Smart Router."""

from __future__ import annotations

import sys
from typing import List, Optional

from ntg.cli.diagnostics import print_account_status, print_banner


def configure_console_encoding() -> None:
    """Ensure UTF-8 output encoding for Windows console (handles emojis cleanly)."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def run_cli(argv: Optional[List[str]] = None) -> None:
    """Run interactive or argument-based multi-provider router and coding agent CLI."""
    from ntg.agent.code_intelligence import CodeIntelligence
    from ntg.agent.orchestrator import ArchitectureAwareAgent
    from ntg.agent.verifier import CodeVerifier
    from ntg.router.engine import UnifiedNTGRouter
    from ntg.router.state import StateManager

    configure_console_encoding()
    args = list(argv) if argv is not None else sys.argv[1:]

    # Check for reset-state flag
    if "--reset-state" in args:
        StateManager().reset_all()
        print("Persisted NTG state has been reset.")
        args = [a for a in args if a != "--reset-state"]

    target_dir = None
    if "--dir" in args or "-d" in args:
        flag = "--dir" if "--dir" in args else "-d"
        idx = args.index(flag)
        if idx + 1 < len(args):
            target_dir = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

    # Check for refresh-knowledge flag
    if "--refresh-knowledge" in args:
        print("Refreshing Graphify knowledge graph and Codebase Memory MCP index...")
        intel = CodeIntelligence(target_dir or ".")
        sync_res = intel.refresh_knowledge()
        print(f"Knowledge sync complete (synced={sync_res.get('synced')}):")
        print(f"  Graphify: {sync_res.get('graphify')}")
        print(f"  Codebase Memory: {sync_res.get('memory')}")
        return

    # Check for verify flag
    if "--verify" in args:
        verifier = CodeVerifier(target_dir or ".")
        res = verifier.verify_syntax()
        print(
            f"Syntax verification {'PASSED' if res['passed'] else 'FAILED'} "
            f"({len(res['checked_files'])} files checked)."
        )
        for err in res.get("errors", []):
            print(f"  ! {err['file']}: {err['error']}")
        return

    router = UnifiedNTGRouter()

    # Check for help flag
    if "--help" in args or "-h" in args:
        print_banner(deployments=router.deployments, default_model=router.default_model)
        print("Usage: python main.py [OPTIONS] [PROMPT]\n")
        print("Options:")
        print("  -d, --dir <path>        Target repository/directory location to analyze and query (defaults to '.')")
        print("  --model <name>          Route to specific model group (auto, groq, openrouter, gemini, etc.)")
        print("  --capability <caps>     Filter by comma-separated capabilities (coding, vision, reasoning, etc.)")
        print("  --plan                  Generate an architecture-aware change plan using Graphify + Codebase Memory")
        print("  --router-only           Bypass repository analysis and send prompt directly to the smart router")
        print("  --verify                Verify Python syntax across the repository")
        print("  --refresh-knowledge     Update Graphify knowledge graph and Codebase Memory MCP index")
        print("  --status                Display detailed status table of all deployments and quotas")
        print("  --rediscover            Force authoritative discovery of models across provider accounts")
        print("  --reset-state           Reset persisted routing state, circuit breakers, and metrics")
        print("  -h, --help              Show this help message and exit\n")
        return

    # Check for status flag
    if "--status" in args:
        print_banner(deployments=router.deployments, default_model=router.default_model)
        print_account_status(router.deployments)
        return

    # Check for rediscover flag
    if "--rediscover" in args:
        print("Re-probing models via provider APIs...")
        res = router.rediscover_models(force=True)
        print("Rediscovery completed:")
        for k, v in res.get("success", {}).items():
            print(f"  + {k}: {v} models discovered")
        for k, v in res.get("failed", {}).items():
            print(f"  ! {k}: {v}")
        return

    plan_mode = False
    if "--plan" in args:
        plan_mode = True
        args = [a for a in args if a != "--plan"]

    router_only = False
    if "--router-only" in args or "--direct" in args:
        router_only = True
        args = [a for a in args if a not in ("--router-only", "--direct")]

    strict_mode = True
    if "--non-strict" in args:
        strict_mode = False
        args = [a for a in args if a != "--non-strict"]

    selected_model = None
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            selected_model = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

    selected_capability = None
    if "--capability" in args or "--capabilities" in args:
        flag = "--capability" if "--capability" in args else "--capabilities"
        idx = args.index(flag)
        if idx + 1 < len(args):
            raw_caps = args[idx + 1]
            selected_capability = [c.strip() for c in raw_caps.split(",") if c.strip()]
            args = args[:idx] + args[idx + 2 :]

    print_banner(
        deployments=router.deployments,
        default_model=router.default_model,
    )

    if args:
        prompt = " ".join(args).strip()
    else:
        try:
            prompt = input("Enter your prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return

    if not prompt:
        print("No prompt supplied.")
        return

    if router_only and not plan_mode:
        router.ask(prompt, model=selected_model, capabilities=selected_capability)
        return

    effective_repo_root = target_dir or "."
    agent = ArchitectureAwareAgent(repo_root=effective_repo_root, router=router)
    result = agent.handle_request(
        request=prompt,
        mode="coding" if plan_mode else None,
        model=selected_model,
        capabilities=selected_capability,
        auto_build_graph=True,
        strict=strict_mode,
        include_diagnostics=True,
    )

    if result.get("request_type") == "coding":
        print("\n=== Architecture-Aware Change Plan ===")
        print(
            f"Status: {result['status']} (complete={result.get('complete')}, "
            f"approved={result['approved']}, model_invoked={result.get('model_invoked')})\n"
        )
        print(result["plan"])
        return

    print(f"\n=== Directory Query Answer ({result['repo_root']}) ===")
    print(
        f"Verification: status={result['status']}, complete={result['complete']}, "
        f"model_invoked={result.get('model_invoked')}, "
        f"retrieval_blocked={result.get('retrieval_blocked')}, "
        f"errors={len(result.get('errors', []))}\n"
    )
    print(result["answer"])


main = run_cli
