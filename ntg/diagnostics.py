"""Diagnostic console tables and status formatting."""

from typing import Any, Dict, List, Optional

from ntg.config import DEFAULT_LOGICAL_MODEL
from ntg.models import Deployment, utc_string


def print_divider(title: Optional[str] = None, width: int = 70) -> None:
    """Print standard section divider."""
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


def print_banner(
    deployments: Optional[List[Deployment]] = None,
    openrouter_count: int = 0,
    groq_count: int = 0,
    nvidia_count: int = 0,
    cohere_count: int = 0,
    gemini_count: int = 0,
    gemini_accounts_count: int = 0,
    default_model: str = DEFAULT_LOGICAL_MODEL,
) -> None:
    """Print application banner with multi-provider summary."""
    if deployments is not None:
        openrouter_count = len([d for d in deployments if d.provider == "openrouter"])
        groq_count = len([d for d in deployments if d.provider == "groq"])
        nvidia_count = len([d for d in deployments if d.provider == "nvidia"])
        cohere_count = len([d for d in deployments if d.provider == "cohere"])
        gemini_deps = [d for d in deployments if d.provider == "gemini"]
        gemini_count = len(gemini_deps)
        gemini_accounts_count = len({d.account for d in gemini_deps})
        total = len(deployments)
    else:
        total = openrouter_count + groq_count + nvidia_count + cohere_count + gemini_count

    parts = []
    if openrouter_count > 0:
        parts.append(f"{openrouter_count} OpenRouter")
    if groq_count > 0:
        parts.append(f"{groq_count} Groq")
    if nvidia_count > 0:
        parts.append(f"{nvidia_count} NVIDIA")
    if cohere_count > 0:
        parts.append(f"{cohere_count} Cohere")
    if gemini_count > 0:
        gemini_label = f"{gemini_count} Gemini"
        if gemini_accounts_count > 0:
            gemini_label += f" across {gemini_accounts_count} accounts"
        parts.append(gemini_label)

    breakdown = ", ".join(parts) if parts else "No active providers"

    print_divider("NTG — MULTI-PROVIDER LITELLM SMART ROUTER")
    print(f"Deployments   : {total} total ({breakdown})")
    print(f"Default Model : {default_model}")
    print("Routing Engine: LiteLLM Router (load-balancing & dynamic failover)")
    print("Isolation     : Per-deployment health state & cooldown")
    print("Security      : Zero secrets in console diagnostics\n")


def print_account_status(deployments: List[Deployment]) -> None:
    """Displays formatted status table of all configured deployments across providers."""
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

        print(f"\n{dep.display_name}")
        print(f"  Provider        : {dep.provider.title()}")
        print(f"  Logical Model   : {dep.logical_model}")
        print(f"  Underlying Model: {dep.model}")
        print(f"  Account         : {dep.account}")
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
    print(f"Logical model        : {deployment.logical_model}")
    print(f"Underlying model     : {deployment.model}")
    print(f"Account              : {deployment.account}")
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