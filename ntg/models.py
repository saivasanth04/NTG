"""Provider-neutral data models, capability representation, and metrics."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional

from ntg.config import (
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_OPENROUTER,
    OPENROUTER_ACTUAL_MODEL,
    OPENROUTER_LITELLM_MODEL,
)
from ntg.state import CircuitState


def utc_string(timestamp: Optional[float] = None) -> str:
    """Format Unix timestamp (or current time) as an ISO-8601 UTC string."""
    ts = timestamp if timestamp is not None else time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


@dataclass
class ModelCapabilities:
    """Normalized capabilities for an LLM deployment."""

    coding: bool = False
    reasoning: bool = False
    vision: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    streaming: bool = True
    context_window: int = 4096

    def satisfies(self, required: Optional["ModelCapabilities"]) -> bool:
        """Check if deployment meets all required capabilities and minimum context window."""
        if required is None:
            return True
        if required.coding and not self.coding:
            return False
        if required.reasoning and not self.reasoning:
            return False
        if required.vision and not self.vision:
            return False
        if required.tool_calling and not self.tool_calling:
            return False
        if required.structured_output and not self.structured_output:
            return False
        if required.streaming and not self.streaming:
            return False
        if required.context_window > self.context_window:
            return False
        return True

    def to_tags(self) -> List[str]:
        """Convert active capabilities to LiteLLM tags for routing filters."""
        tags = []
        if self.coding:
            tags.append("coding")
        if self.reasoning:
            tags.append("reasoning")
        if self.vision:
            tags.append("vision")
        if self.tool_calling:
            tags.append("tool_calling")
        if self.structured_output:
            tags.append("structured_output")
        if self.streaming:
            tags.append("streaming")
        return tags


@dataclass
class QuotaInfo:
    """Quota tracking metadata separated from health state."""

    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    rpm_remaining: Optional[int] = None
    rpd_remaining: Optional[int] = None
    reset_at: Optional[float] = None
    retry_after: Optional[float] = None
    quota_scope: str = "deployment"  # "deployment", "account", or "provider"
    limit_type: Optional[str] = None  # "rpm", "rpd", "quota", "temporary"
    last_updated: float = 0.0


@dataclass
class DeploymentMetrics:
    """Accurate separation of user requests and upstream attempts."""

    user_requests: int = 0
    upstream_attempts: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0
    auth_errors: int = 0


@dataclass
class Deployment:
    """Provider-neutral deployment representation with capabilities, quota, and metrics."""

    id: str
    provider: str
    account: str
    model: str
    logical_model: str
    litellm_model: str
    api_key: str
    api_base: Optional[str] = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    quota: QuotaInfo = field(default_factory=QuotaInfo)
    metrics: DeploymentMetrics = field(default_factory=DeploymentMetrics)
    circuit_state: CircuitState = CircuitState.HEALTHY
    state_reason: Optional[str] = None
    state_updated_at: float = 0.0
    half_open_probes: int = 0
    consecutive_failures: int = 0
    weight: float = 1.0

    @property
    def deployment_id(self) -> str:
        """Alias for id."""
        return self.id

    @property
    def name(self) -> str:
        """Alias for model name (backward compatibility)."""
        return self.model

    @property
    def account_name(self) -> str:
        """Alias for account (backward compatibility)."""
        return self.account

    @property
    def order(self) -> int:
        """Legacy order attribute."""
        return 1

    @property
    def available(self) -> bool:
        """True if deployment is eligible for requests (HEALTHY or HALF_OPEN)."""
        return self.is_eligible()

    @property
    def blocked_until(self) -> float:
        """Timestamp until which deployment is in cooldown."""
        return self.quota.reset_at or 0.0

    @blocked_until.setter
    def blocked_until(self, val: float) -> None:
        self.quota.reset_at = val

    @property
    def cooldown_until(self) -> float:
        """Timestamp until which deployment is in cooldown."""
        return self.quota.reset_at or 0.0

    @cooldown_until.setter
    def cooldown_until(self, val: float) -> None:
        self.quota.reset_at = val

    @property
    def blocked_reason(self) -> Optional[str]:
        """Reason deployment is in cooldown or quarantine."""
        return self.state_reason

    @blocked_reason.setter
    def blocked_reason(self, val: Optional[str]) -> None:
        self.state_reason = val

    @property
    def attempts(self) -> int:
        """Upstream attempts count."""
        return self.metrics.upstream_attempts

    @property
    def successes(self) -> int:
        """Success count."""
        return self.metrics.successes

    @property
    def failures(self) -> int:
        """Failure count."""
        return self.metrics.failures

    @property
    def rate_limits(self) -> int:
        """Rate limit count."""
        return self.metrics.rate_limits

    @property
    def remaining_cooldown(self) -> float:
        """Returns the number of seconds remaining in explicit cooldown."""
        if self.quota.reset_at and self.quota.reset_at > 0:
            return max(0.0, self.quota.reset_at - time.time())
        return 0.0

    @property
    def display_name(self) -> str:
        """Formatted human-readable deployment label without exposing secrets."""
        return f"{self.provider.title()} {self.account} ({self.model})"

    def is_eligible(self, now: Optional[float] = None) -> bool:
        """Check if deployment is eligible to receive traffic."""
        if not self.api_key or not self.api_key.strip():
            return False
        if self.circuit_state == CircuitState.QUARANTINED:
            return False
        if self.circuit_state == CircuitState.OPEN:
            current_time = now if now is not None else time.time()
            if self.quota.reset_at and current_time >= self.quota.reset_at:
                return True
            return False
        return True

    def is_available(self, now: Optional[float] = None) -> bool:
        """Alias for is_eligible."""
        return self.is_eligible(now)

    def record_failure(self, max_failures: int = 3) -> None:
        """Record upstream failure and update circuit breaker state."""
        self.metrics.upstream_attempts += 1
        self.metrics.failures += 1
        self.consecutive_failures += 1
        now = time.time()
        self.state_updated_at = now

        if self.circuit_state == CircuitState.HALF_OPEN:
            self.circuit_state = CircuitState.OPEN
            self.state_reason = "Probe request failed while in HALF_OPEN"
        elif self.consecutive_failures >= max_failures:
            self.circuit_state = CircuitState.OPEN
            self.state_reason = f"Consecutive failure threshold reached ({self.consecutive_failures})"

    def record_success(self) -> None:
        """Record upstream success and restore circuit to HEALTHY."""
        self.consecutive_failures = 0
        now = time.time()
        self.state_updated_at = now
        if self.circuit_state == CircuitState.HALF_OPEN:
            self.circuit_state = CircuitState.HEALTHY
            self.state_reason = "Probe succeeded; recovered to HEALTHY"
            self.half_open_probes = 0

    def refresh(self, now: Optional[float] = None) -> bool:
        """Evaluate circuit breaker recovery if cooldown elapsed."""
        current_time = now if now is not None else time.time()
        if self.circuit_state == CircuitState.OPEN:
            if self.quota.reset_at and current_time >= self.quota.reset_at:
                self.circuit_state = CircuitState.HALF_OPEN
                self.half_open_probes = 1
                self.state_reason = "Cooldown elapsed; probing deployment health"
                self.state_updated_at = current_time
        return self.is_eligible(current_time)

    def to_litellm_dict(self, group_override: Optional[str] = None) -> Dict[str, Any]:
        """Converts deployment into LiteLLM router dictionary format with tags and weights."""
        group_name = group_override if group_override else self.logical_model
        litellm_params: Dict[str, Any] = {
            "model": self.litellm_model,
            "api_key": self.api_key,
            "tags": self.capabilities.to_tags(),
            "weight": self.weight,
            "metadata": {
                "deployment_id": self.id,
                "provider": self.provider,
                "account": self.account,
                "model": self.model,
                "logical_model": self.logical_model,
            },
        }
        if self.api_base:
            litellm_params["api_base"] = self.api_base

        return {
            "model_name": group_name,
            "litellm_params": litellm_params,
            "model_info": {
                "id": self.id,
                "provider": self.provider,
                "account": self.account,
                "model": self.model,
                "logical_model": self.logical_model,
                "tags": self.capabilities.to_tags(),
            },
        }

    def to_deployment(self) -> Dict[str, Any]:
        """Backward-compatible alias for to_litellm_dict."""
        return self.to_litellm_dict()


class Account(Deployment):
    """Backwards-compatible OpenRouter account deployment."""

    def __init__(
        self,
        name: str,
        api_key: str,
        order: int = 1,
        available: bool = True,
        blocked_until: float = 0.0,
        blocked_reason: Optional[str] = None,
        attempts: int = 0,
        successes: int = 0,
        failures: int = 0,
        rate_limits: int = 0,
    ):
        super().__init__(
            id=f"openrouter-{name}",
            provider=MODEL_GROUP_OPENROUTER,
            account=name,
            model=OPENROUTER_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_OPENROUTER,
            litellm_model=OPENROUTER_LITELLM_MODEL,
            api_key=api_key,
            capabilities=ModelCapabilities(
                coding=True,
                reasoning=True,
                tool_calling=True,
                structured_output=True,
                streaming=True,
                context_window=32768,
            ),
        )


class GeminiDeployment(Deployment):
    """Backwards-compatible Gemini account/model deployment."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        account_name: str = "account_1",
        order: int = 1,
        available: bool = True,
        blocked_until: float = 0.0,
        blocked_reason: Optional[str] = None,
        attempts: int = 0,
        successes: int = 0,
        failures: int = 0,
        rate_limits: int = 0,
    ):
        super().__init__(
            id=f"gemini-{account_name}-{model_name}",
            provider=MODEL_GROUP_GEMINI,
            account=account_name,
            model=model_name,
            logical_model=MODEL_GROUP_GEMINI,
            litellm_model=f"gemini/{model_name}",
            api_key=api_key,
            capabilities=ModelCapabilities(
                coding=True,
                reasoning="pro" in model_name.lower() or "flash" in model_name.lower(),
                vision=True,
                tool_calling=True,
                structured_output=True,
                streaming=True,
                context_window=1000000,
            ),
        )