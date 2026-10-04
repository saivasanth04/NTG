"""Diagnostic console tables and status formatting."""

from typing import Any, Dict, List, Optional

from ntg.models import Account, utc_string


def print_divider(title: Optional[str] = None, width: int = 70) -> None:
    """Print standard section divider."""
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def print_banner(accounts_count: int = 5) -> None:
    """Print application banner."""
    print_divider("NTG — DYNAMIC OPENROUTER FALLBACK ROUTER")
    print(f"Deployments  : {accounts_count} OpenRouter accounts")
    print("Logical Model: openrouter-free (LiteLLM managed)")
    print("Routing      : LiteLLM deployment selection & dynamic failover")
    print("Quota Mode   : Reactive real OpenRouter error handling")
    print("State        : Dynamic runtime self-adapting\n")


def print_account_status(accounts: List[Account]) -> None:
    """Displays formatted status table of all configured accounts."""
    for acc in accounts:
        acc.refresh()

    print_divider("ACCOUNT STATUS")
    for acc in accounts:
        if not acc.api_key or not acc.api_key.strip():
            status_str = "NO KEY"
        elif acc.is_available():
            status_str = "ACTIVE"
        else:
            status_str = "COOLING DOWN"

        print(f"\n{acc.name} (Order: {acc.order})")
        print(f"  Status          : {status_str}")
        print(f"  Attempts        : {acc.attempts}")
        print(f"  Successes       : {acc.successes}")
        print(f"  Failures        : {acc.failures}")
        print(f"  Rate Limits     : {acc.rate_limits}")
        if acc.remaining_cooldown > 0:
            print(f"  Block remaining : {acc.remaining_cooldown:.1f}s")
        if acc.blocked_reason:
            print(f"  Block reason    : {acc.blocked_reason}")


def print_request_execution(
    account: Account,
    response_model: Optional[str] = None,
) -> None:
    """Prints execution summary after a request."""
    print(f"\nFulfilling account : {account.name}")
    print(f"Total attempts     : {account.attempts}")
    print(f"Account successes  : {account.successes}")
    if response_model:
        print(f"Response model     : {response_model}")


def print_rate_limit_details(info: Dict[str, Any]) -> None:
    """Prints diagnostic rate-limit information extracted from response headers."""
    print_divider("OPENROUTER RATE LIMIT INFORMATION")
    fields = [
        ("category", "Classification"),
        ("limit_source", "Limit source"),
        ("provider_name", "Provider"),
        ("limit", "Limit"),
        ("remaining", "Remaining"),
        ("reset", "Reset"),
    ]
    for key, label in fields:
        print(f"{label:<16}: {info.get(key)}")
    if info.get("reset_timestamp"):
        print(f"{'Reset UTC':<16}: {utc_string(info['reset_timestamp'])}")
    if info.get("remedy"):
        print(f"{'Remedy':<16}: {info['remedy']}")