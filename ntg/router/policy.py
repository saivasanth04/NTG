"""Deterministic routing policy. Planning never acquires resources or mutates health."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ntg.core.models import CircuitState, Deployment, QuotaScope
from ntg.router.requirements import RequestRequirements

CAPABILITY_GROUPS = frozenset(("coding", "reasoning", "vision", "tool_calling",
                               "structured_output", "streaming", "audio_input", "audio_output"))
GLOBAL_GROUPS = frozenset(("auto", "ntg-auto"))


def matches(deployment: Deployment, group: str) -> bool:
    if group in GLOBAL_GROUPS:
        return True
    if group in CAPABILITY_GROUPS:
        value = getattr(deployment.capabilities, group)
        # Coding/reasoning describe task suitability rather than a wire protocol.
        return value is not False if group in ("coding", "reasoning") else value is True
    return group in (deployment.id, deployment.model, deployment.litellm_model,
                     deployment.logical_model, deployment.provider)


def compatible(deployment: Deployment, requirements: RequestRequirements) -> bool:
    required = requirements.capabilities
    actual = deployment.capabilities
    for name in CAPABILITY_GROUPS:
        if getattr(required, name) is True:
            value = getattr(actual, name)
            if name == "coding" or name == "reasoning" and name not in requirements.strict_requirements:
                if value is False:
                    return False
            elif value is not True:
                return False
    return actual.context_window >= required.context_window


def quota_peers(deployment: Deployment, deployments: Sequence[Deployment]) -> list[Deployment]:
    """Never share credentials or limits across providers; unknown scope stays local."""
    scope = deployment.quota.quota_scope
    result = []
    for other in deployments:
        if other.provider != deployment.provider:
            continue
        same_account = other.account == deployment.account or other.api_key == deployment.api_key
        if (other.id == deployment.id
            or scope == QuotaScope.PROVIDER
            or scope == QuotaScope.ACCOUNT and same_account
            or scope == QuotaScope.MODEL and other.model == deployment.model
            or scope == QuotaScope.ACCOUNT_MODEL and same_account and other.model == deployment.model):
            result.append(other)
    return result


@dataclass(frozen=True)
class DeploymentScore:
    deployment_id: str
    capability_fit: float
    quota_headroom: float
    health: float
    latency: float
    account_load: float
    weight: float

    @property
    def total(self) -> float:
        return (25 * self.capability_fit + 25 * self.quota_headroom + 25 * self.health
                + 10 * self.latency + 10 * self.account_load + 5 * self.weight)


def score(deployment: Deployment, requirements: RequestRequirements,
          account_attempts: Mapping[tuple[str, str], int],
          inflight: Mapping[tuple[str, str], int]) -> DeploymentScore:
    caps = deployment.capabilities
    preferences = [name for name in ("coding", "reasoning")
                   if getattr(requirements.capabilities, name) is True
                   or name == "coding" and requirements.coding_preferred]
    fit = sum(1 if getattr(caps, name) is True else 0.5 if getattr(caps, name) is None else 0
              for name in preferences) / len(preferences) if preferences else 1.0
    fractions = [max(0.0, min(1.0, remaining / limit))
                 for remaining, limit in ((deployment.quota.rpm_remaining, deployment.quota.rpm_limit),
                                          (deployment.quota.rpd_remaining, deployment.quota.rpd_limit))
                 if remaining is not None and limit and limit > 0]
    headroom = min(fractions) if fractions else 0.5
    health = 1 / (1 + deployment.consecutive_failures)
    if deployment.circuit_state == CircuitState.HALF_OPEN:
        health *= 0.5
    if deployment.is_stale:
        health *= 0.8
    latency = 1 / (1 + (deployment.latency_ewma or 1.0))
    account = (deployment.provider, deployment.account)
    # Account counters are process-local, so old persisted usage cannot starve a key.
    load = 1 / (1 + account_attempts.get(account, 0) + 2 * inflight.get(account, 0))
    weight = deployment.weight if math.isfinite(deployment.weight) else 1.0
    return DeploymentScore(deployment.id, fit, headroom, health, latency, load,
                           max(0.0, weight) / (1 + max(0.0, weight)))


def rank(deployments: Sequence[Deployment], requirements: RequestRequirements,
         account_attempts: Mapping[tuple[str, str], int],
         inflight: Mapping[tuple[str, str], int]) -> list[Deployment]:
    accounts = {(deployment.provider, deployment.account) for deployment in deployments}
    floor = min((account_attempts.get(account, 0) for account in accounts), default=0)
    # Preserve useful distribution pressure after millions of requests instead
    # of letting every account's load component converge to zero.
    relative_attempts = {account: account_attempts.get(account, 0) - floor for account in accounts}
    return sorted(deployments, key=lambda d: (-score(d, requirements, relative_attempts, inflight).total, d.id))
