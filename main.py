"""Main CLI entry point for the OpenRouter Multi-Account Fallback Router."""

import sys
from ntg import (
    Account,
    DEFAULT_KEYS,
    LITELLM_MODEL_NAME,
    OPENROUTER_FREE_MODEL,
    OpenRouterFallbackRouter,
    print_account_status,
)
from ntg.diagnostics import print_divider


def main():
    """Run interactive or argument-based OpenRouter multi-account router CLI."""
    accounts = [
        Account(name=f"account_{i+1}", api_key=key, order=i+1)
        for i, key in enumerate(DEFAULT_KEYS)
    ]
    router = OpenRouterFallbackRouter(accounts)

    print_divider("LITELLM + OPENROUTER FREE ROUTER")
    print(f"OpenRouter model   : {OPENROUTER_FREE_MODEL}")
    print(f"LiteLLM model      : {LITELLM_MODEL_NAME}")
    print(f"Configured accounts: {len(accounts)}")
    print("Fallback strategy  : Priority failover with rate-limit classification\n")

    print_account_status(accounts)

    # Allow prompt from CLI args or interactive input
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
        print(f"\nEnter your prompt: {prompt}")
    else:
        try:
            prompt = input("\nEnter your prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return

    if not prompt:
        print("No prompt supplied.")
        return

    router.ask(prompt)
    print_account_status(accounts)


if __name__ == "__main__":
    main()
