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
    print_divider("NTG — OPENROUTER FREE GATEWAY")
    print(f"Accounts : {accounts_count}")
    print("RPM      : 100 configured")
    print("RPD      : 250 configured")
    print("Routing  : capacity-aware")
    print("Retries  : disabled")
    print("State    : persistent\n")


def print_account_status(accounts: List[Account]) -> None:
    """Displays formatted status table of all configured accounts."""
    for acc in accounts:
        acc.refresh()

    print_divider("ACCOUNT STATUS")
    for acc in accounts:
        rpm = acc.rpm_info()
        rpd = acc.rpd_info()

        print(f"\n{acc.name}")
        print(f"  Available       : {acc.available}")
        print(f"  RPM             : {int(rpm['used'])}/{int(rpm['max'])}")
        print(f"  RPM remaining   : {int(rpm['remaining'])}")
        print(f"  RPD             : {int(rpd['used'])}/{int(rpd['max'])}")
        print(f"  RPD remaining   : {int(rpd['remaining'])}")
        if acc.remaining_cooldown > 0:
            print(f"  Block remaining : {acc.remaining_cooldown:.1f}s")
        if acc.blocked_reason:
            print(f"  Block reason    : {acc.blocked_reason}")


def print_request_execution(
    account: Account,
    response_model: Optional[str] = None,
) -> None:
    """Prints request selection and quota details during a request."""
    rpm = account.rpm_info()
    rpd = account.rpd_info()

    print(f"\nSelected account : {account.name}")
    print(f"RPM              : {int(rpm['used'])}/{int(rpm['max'])}")
    print(f"RPD              : {int(rpd['used'])}/{int(rpd['max'])}")
    print(f"RPM remaining    : {int(rpm['remaining'])}")
    print(f"RPD remaining    : {int(rpd['remaining'])}")
    if response_model:
        print(f"\nResponse model   : {response_model}")


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