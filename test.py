"""Test entry point and backwards-compatible interface for NTG Router."""

import sys
from typing import Any, Optional

from ntg import (
    Account,
    DEFAULT_KEYS,
    LITELLM_MODEL_NAME,
    OPENROUTER_FREE_MODEL,
    OpenRouterFallbackRouter,
    print_account_status as _print_status,
)

# Initialize default accounts and router instance
ACCOUNTS = [
    Account(name=f"account_{i+1}", api_key=key, order=i+1)
    for i, key in enumerate(DEFAULT_KEYS)
]
router_manager = OpenRouterFallbackRouter(ACCOUNTS)
router = router_manager.router


def ask(prompt: str) -> Optional[Any]:
    """Execute completion request with account fallback."""
    return router_manager.ask(prompt)


def print_account_status() -> None:
    """Print current status of accounts."""
    _print_status(ACCOUNTS)


def main():
    """CLI runner matching original test.py behavior."""
    from main import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()