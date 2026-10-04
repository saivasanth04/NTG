"""Diagnostic tables, rate-limit inspect tools, and formatted console banners."""

from typing import Any, Dict, List, Optional

from ntg.models import Account, utc_string


def print_divider(title: Optional[str] = None, width: int = 70) -> None:
    """Print standard section divider."""
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def print_account_status(accounts: List[Account]) -> None:
    """Displays formatted status table of all configured accounts."""
    for acc in accounts:
        acc.refresh()

    print_divider("ACCOUNT STATUS")
    for acc in accounts:
        print(f"\n{acc.name}")
        print(f"  Available       : {acc.available}")
        print(f"  Attempts        : {acc.attempts}")
        print(f"  Successes       : {acc.successes}")
        print(f"  Failures        : {acc.failures}")
        print(f"  429s            : {acc.rate_limits}")
        if acc.remaining_cooldown > 0:
            print(f"  Block remaining : {acc.remaining_cooldown:.1f}s")
        if acc.blocked_reason:
            print(f"  Block reason    : {acc.blocked_reason}")


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
