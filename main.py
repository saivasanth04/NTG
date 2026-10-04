"""Main CLI entry point for the Dynamic OpenRouter Fallback Router."""

import sys

# Ensure UTF-8 output encoding for Windows console (handles emojis cleanly)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ntg import (
    Account,
    DEFAULT_KEYS,
    OpenRouterFallbackRouter,
)


def main():
    """Run interactive or argument-based OpenRouter fallback router CLI."""
    accounts = [
        Account(name=name, api_key=key, order=i + 1)
        for i, (name, key) in enumerate(DEFAULT_KEYS.items())
    ]

    # Ensure 5 account slots exist
    while len(accounts) < 5:
        idx = len(accounts) + 1
        accounts.append(
            Account(
                name=f"account_{idx}",
                api_key="",
                order=idx,
                available=False,
            )
        )

    router = OpenRouterFallbackRouter(accounts)

    # Allow prompt from CLI args or interactive input
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
    else:
        try:
            prompt = input("Enter your prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return

    if not prompt:
        print("No prompt supplied.")
        return

    router.ask(prompt)


if __name__ == "__main__":
    main()