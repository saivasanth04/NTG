"""Diagnostic console tables and status formatting."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ntg.config import DEFAULT_LOGICAL_MODEL
from ntg.models import CircuitState, Deployment, sanitize_secret, utc_string


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
    print("Architecture  : NTG (capabilities, quotas, state) + LiteLLM (routing & failover)")
    print("Routing Engine: LiteLLM Router (simple-shuffle & bounded failover)")
    print("Reliability   : Circuit Breaker (HEALTHY -> OPEN -> HALF_OPEN -> HEALTHY)")
    print("Security      : Zero secrets in console diagnostics\n")


def print_account_status(deployments: List[Deployment]) -> None:
    """Displays formatted status table of all configured deployments across providers."""
    now = time.time()
    for dep in deployments:
        dep.refresh(now)

    print_divider("DEPLOYMENT STATUS")
    for dep in deployments:
        if not dep.api_key or not dep.api_key.strip():
            status_str = "NO KEY"
        elif dep.circuit_state == CircuitState.AUTH_FAILED:
            status_str = "AUTH_FAILED (QUARANTINED)"
        elif dep.circuit_state == CircuitState.QUARANTINED:
            status_str = "QUARANTINED (AUTH)"
        elif dep.circuit_state == CircuitState.OPEN:
            status_str = f"OPEN (COOLING DOWN, {dep.remaining_cooldown:.1f}s)"
        elif dep.circuit_state == CircuitState.HALF_OPEN:
            status_str = "HALF_OPEN (PROBING)"
        else:
            status_str = "HEALTHY (ACTIVE)"

        if dep.is_stale:
            status_str += f" [STALE: {dep.stale_reason or 'Cached'}]"

        caps_tags = ", ".join(dep.capabilities.to_tags()) or "general"

        print(f"\n{dep.display_name}")
        print(f"  Provider        : {dep.provider.title()}")
        print(f"  Logical Model   : {dep.logical_model}")
        print(f"  Underlying Model: {dep.model}")
        print(f"  Account         : {dep.account}")
        print(f"  Circuit State   : {status_str}")
        print(f"  Capabilities    : {caps_tags} (context: {dep.capabilities.context_window:,})")
        print(
            f"  Telemetry       : User reqs: {dep.metrics.user_requests} | Upstream attempts: {dep.metrics.upstream_attempts} | Success: {dep.metrics.successes} | Fail: {dep.metrics.failures}"
        )
        print(
            f"  Errors Breakdown: Rate Limits: {dep.metrics.rate_limits} | Auth Failures: {dep.metrics.auth_failures}"
        )
        if dep.quota.quota_scope:
            scope_str = f"  Quota Scope     : {dep.quota.quota_scope}"
            if dep.quota.limit_type:
                scope_str += f" (limit type: {dep.quota.limit_type})"
            print(scope_str)

        quota_parts = []
        if dep.rpm_limit is not None or dep.rpm_remaining is not None:
            rem = dep.rpm_remaining if dep.rpm_remaining is not None else "?"
            lim = dep.rpm_limit if dep.rpm_limit is not None else "?"
            quota_parts.append(f"RPM: {rem}/{lim}")
        if dep.rpd_limit is not None or dep.rpd_remaining is not None:
            rem = dep.rpd_remaining if dep.rpd_remaining is not None else "?"
            lim = dep.rpd_limit if dep.rpd_limit is not None else "?"
            quota_parts.append(f"RPD: {rem}/{lim}")
        if quota_parts:
            print(f"  Quota Status    : {', '.join(quota_parts)}")

        if dep.remaining_cooldown > 0:
            print(f"  Cooldown left   : {dep.remaining_cooldown:.1f}s")
        if dep.blocked_reason:
            print(f"  Last reason     : {sanitize_secret(dep.blocked_reason, dep.api_key)}")


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
    print(f"Circuit State        : {deployment.circuit_state.value}")
    print(
        f"Telemetry            : User reqs: {deployment.metrics.user_requests} | Upstream attempts: {deployment.metrics.upstream_attempts} | Success: {deployment.metrics.successes} | Fail: {deployment.metrics.failures}"
    )
    if response_model:
        print(f"Response model       : {response_model}")


def print_rate_limit_details(info: Dict[str, Any]) -> None:
    """Prints diagnostic rate-limit information extracted from response metadata."""
    print_divider("RATE LIMIT / DIAGNOSTIC DETAILS")
    fields = [
        ("category", "Classification"),
        ("limit_source", "Limit source"),
        ("provider_name", "Provider"),
        ("limit_type", "Limit type"),
        ("quota_scope", "Quota scope"),
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