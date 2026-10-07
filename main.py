"""Main CLI entry point for the Multi-Provider LiteLLM NTG Router."""

import sys

# Ensure UTF-8 output encoding for Windows console (handles emojis cleanly)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ntg import UnifiedNTGRouter, print_banner


def main():
    """Run interactive or argument-based multi-provider router CLI."""
    router = UnifiedNTGRouter()

    print_banner(
        deployments=router.deployments,
        default_model=router.default_model,
    )

    args = sys.argv[1:]
    selected_model = None

    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            selected_model = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

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

    router.ask(prompt, model=selected_model)


if __name__ == "__main__":
    main()