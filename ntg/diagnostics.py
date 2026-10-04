"""Diagnostic console tables and status formatting."""

from typing import Any, Dict, List, Optional

from ntg.config import LITELLM_MODEL_NAME
from ntg.models import Deployment, utc_string


def print_divider(title: Optional[str] = None, width: int = 70) -> None:
    """Print standard section divider."""
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def print_banner(openrouter_count: int = 5, gemini_count: int = 0) -> None:
    """Print application banner."""
    total = openrouter_count + gemini_count
    print_divider("NTG — UNIFIED OPENROUTER + GEMINI SMART ROUTER")
    print(f"Deployments  : {total} total ({openrouter_count} OpenRouter accounts, {gemini_count} Gemini models)")
    print(f"Logical Model: {LITELLM_MODEL_NAME} (LiteLLM managed)")
    print("Routing      : LiteLLM deployment selection & dynamic failover")
    print("Quota Mode   : Reactive real provider error handling")
    print("State        : Dynamic runtime self-adapting\n")


def print_account_status(deployments: List[Deployment]) -> None:
    """Displays formatted status table of all configured deployments (OpenRouter & Gemini)."""
    for dep in deployments:
        dep.refresh()

    print_divider("DEPLOYMENT STATUS")
    for dep in deployments:
        if not dep.api_key or not dep.api_key.strip():
            status_str = "NO KEY"
        elif dep.is_available():
            status_str = "ACTIVE"
        else:
            status_str = "COOLING DOWN"

        print(f"\n{dep.display_name} (Provider: {dep.provider.title()}, Order: {dep.order})")
        print(f"  Status          : {status_str}")
        print(f"  Attempts        : {dep.attempts}")
        print(f"  Successes       : {dep.successes}")
        print(f"  Failures        : {dep.failures}")
        print(f"  Rate Limits     : {dep.rate_limits}")
        if dep.remaining_cooldown > 0:
            print(f"  Block remaining : {dep.remaining_cooldown:.1f}s")
        if dep.blocked_reason:
            print(f"  Block reason    : {dep.blocked_reason}")


def print_request_execution(
    deployment: Deployment,
    response_model: Optional[str] = None,
) -> None:
    """Prints execution summary after a request."""
    print(f"\nFulfilling deployment: {deployment.display_name}")
    print(f"Provider             : {deployment.provider.title()}")
    print(f"Total attempts       : {deployment.attempts}")
    print(f"Deployment successes : {deployment.successes}")
    if response_model:
        print(f"Response model       : {response_model}")


def print_rate_limit_details(info: Dict[str, Any]) -> None:
    """Prints diagnostic rate-limit information extracted from response metadata."""
    print_divider("RATE LIMIT / DIAGNOSTIC DETAILS")
    fields = [
        ("category", "Classification"),
        ("limit_source", "Limit source"),
        ("provider_name", "Provider"),
        ("limit", "Limit"),
        ("remaining", "Remaining"),
        ("reset", "Reset"),
    ]
    for key, label in fields:
        val = info.get(key)
        if val is not None:
            print(f"{label:<16}: {val}")
    if info.get("reset_timestamp"):
        print(f"{'Reset UTC':<16}: {utc_string(info['reset_timestamp'])}")
    if info.get("remedy"):
        print(f"{'Remedy':<16}: {info['remedy']}")