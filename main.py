"""Main CLI entry point for the Multi-Provider LiteLLM NTG Router."""

from __future__ import annotations

import sys

# Ensure UTF-8 output encoding for Windows console (handles emojis cleanly)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ntg import (
    ArchitectureAwareAgent,
    CodeIntelligence,
    CodeVerifier,
    UnifiedNTGRouter,
    print_account_status,
    print_banner,
)


def main() -> None:
    """Run interactive or argument-based multi-provider router and coding agent CLI."""
    args = sys.argv[1:]

    # Check for reset-state flag
    if "--reset-state" in args:
        from ntg.state import StateManager

        StateManager().reset_all()
        print("Persisted NTG state has been reset.")
        args = [a for a in args if a != "--reset-state"]

    router = UnifiedNTGRouter()

    # Check for help flag
    if "--help" in args or "-h" in args:
        print_banner(deployments=router.deployments, default_model=router.default_model)
        print("Usage: python main.py [OPTIONS] [PROMPT]\n")
        print("Options:")
        print("  --model <name>          Route to specific model group (auto, groq, openrouter, gemini, etc.)")
        print("  --capability <caps>     Filter by comma-separated capabilities (coding, vision, reasoning, etc.)")
        print("  --plan                  Generate an architecture-aware change plan using Graphify + Codebase Memory")
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

    # Check for refresh-knowledge flag
    if "--refresh-knowledge" in args:
        print("Refreshing Graphify knowledge graph and Codebase Memory MCP index...")
        intel = CodeIntelligence(".")
        sync_res = intel.refresh_knowledge()
        print(f"Knowledge sync complete (synced={sync_res.get('synced')}):")
        print(f"  Graphify: {sync_res.get('graphify')}")
        print(f"  Codebase Memory: {sync_res.get('memory')}")
        return

    # Check for verify flag
    if "--verify" in args:
        verifier = CodeVerifier(".")
        res = verifier.verify_syntax()
        print(
            f"Syntax verification {'PASSED' if res['passed'] else 'FAILED'} "
            f"({len(res['checked_files'])} files checked)."
        )
        for err in res.get("errors", []):
            print(f"  ! {err['file']}: {err['error']}")
        return

    plan_mode = False
    if "--plan" in args:
        plan_mode = True
        args = [a for a in args if a != "--plan"]

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

    # Allow prompt from CLI args or interactive input
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

    if plan_mode:
        agent = ArchitectureAwareAgent(repo_root=".", router=router)
        plan = agent.create_plan(
            request=prompt,
            model=selected_model,
            capabilities=selected_capability,
            auto_build_graph=True,
        )
        print("\n=== Architecture-Aware Change Plan ===")
        print(f"Status: {plan['status']} (approved={plan['approved']})\n")
        print(plan["plan"])
        return

    router.ask(prompt, model=selected_model, capabilities=selected_capability)


if __name__ == "__main__":
    main()