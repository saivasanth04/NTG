"""Main CLI entry point for the Multi-Provider LiteLLM NTG Router."""

from __future__ import annotations

import sys

# Ensure UTF-8 output encoding for Windows console (handles emojis cleanly)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ntg import UnifiedNTGRouter, print_account_status, print_banner


def main() -> None:
    """Run interactive or argument-based multi-provider router CLI."""
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

    router.ask(prompt, model=selected_model, capabilities=selected_capability)


if __name__ == "__main__":
    main()