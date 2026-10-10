"""Provider-neutral domain models, circuit state, capability representation, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ntg.core.config import (
    CIRCUIT_BREAKER_HALF_OPEN_PROBES,
    CIRCUIT_BREAKER_MAX_FAILURES,
    CIRCUIT_BREAKER_RECOVERY_TIME,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_OPENROUTER,
    OPENROUTER_ACTUAL_MODEL,
    OPENROUTER_LITELLM_MODEL,
)
from ntg.core.utils import (
    parse_duration_string,
    parse_reset_header,
    sanitize_secret,
    utc_string,
)


class CircuitState(str, Enum):
    """Lifecycle states of a deployment's circuit breaker."""

    HEALTHY = "HEALTHY"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    QUARANTINED = "QUARANTINED"
    AUTH_FAILED = "AUTH_FAILED"


@dataclass
class ModelCapabilities:
    """Normalized capabilities for an LLM deployment.

    Fields:
      coding: True if verified capable of code generation/comprehension,
              False if known incapable, None if unknown.
      reasoning: True if verified capable of deep reasoning/chain-of-thought,
              False if known incapable, None if unknown.
      vision: True if verified capable of image/visual inputs,
              False if known text-only, None if unknown.
      tool_calling: True if verified capable of function/tool calling,
              False if known unsupported, None if unknown.
      structured_output: True if verified capable of JSON schema / structured output,
              False if known unsupported, None if unknown.
      streaming: True if verified capable of streaming responses,
              False if known unsupported, None if unknown.
      context_window: Maximum input tokens supported by the model.
    """

    coding: Optional[bool] = None
    reasoning: Optional[bool] = None
    vision: Optional[bool] = None
    tool_calling: Optional[bool] = None
    structured_output: Optional[bool] = None
    streaming: Optional[bool] = None
    context_window: int = 4096

    def satisfies(self, required: Optional["ModelCapabilities"]) -> bool:
        """Check if deployment meets all required capabilities and minimum context window."""
        if required is None:
            return True
        if required.coding is True and self.coding is not True:
            return False
        if required.reasoning is True and self.reasoning is not True:
            return False
        if required.vision is True and self.vision is not True:
            return False
        if required.tool_calling is True and self.tool_calling is not True:
            return False
        if required.structured_output is True and self.structured_output is not True:
            return False
        if required.streaming is True and self.streaming is not True:
            return False
        if required.context_window and self.context_window < required.context_window:
            return False
        return True

    def to_tags(self) -> List[str]:
        """Convert active capabilities to LiteLLM tags for routing filters."""
        tags = []
        if self.coding is True:
            tags.append("coding")
        if self.reasoning is True:
            tags.append("reasoning")
        if self.vision is True:
            tags.append("vision")
        if self.tool_calling is True:
            tags.append("tool_calling")
        if self.structured_output is True:
            tags.append("structured_output")
        if self.streaming is True:
            tags.append("streaming")
        return tags

    def to_dict(self) -> Dict[str, Any]:
        """Return capabilities as a dictionary representation."""
        return {
            "coding": self.coding,
            "reasoning": self.reasoning,
            "vision": self.vision,
            "tool_calling": self.tool_calling,
            "structured_output": self.structured_output,
            "streaming": self.streaming,
            "context_window": self.context_window,
        }


class QuotaScope:
    """Allowed quota scope values representing known boundary of limits."""

    PROVIDER = "provider"
    ACCOUNT = "account"
    MODEL = "model"
    ACCOUNT_MODEL = "account_model"
    UNKNOWN = "unknown"


@dataclass
class QuotaInfo:
    """Normalized quota tracking metadata separated from health and circuit state."""

    rpm_limit: Optional[int] = None
    rpm_remaining: Optional[int] = None
    rpd_limit: Optional[int] = None
    rpd_remaining: Optional[int] = None
    reset_at: Optional[float] = None
    retry_after: Optional[float] = None
    quota_scope: str = QuotaScope.UNKNOWN
    limit_type: Optional[str] = None  # "rpm", "rpd", "tpm", "quota", "temporary"
    last_updated: float = 0.0

    def update(
        self,
        rpm_limit: Optional[int] = None,
        rpm_remaining: Optional[int] = None,
        rpd_limit: Optional[int] = None,
        rpd_remaining: Optional[int] = None,
        reset_at: Optional[float] = None,
        retry_after: Optional[float] = None,
        quota_scope: Optional[str] = None,
        limit_type: Optional[str] = None,
    ) -> None:
        """Update quota metadata fields if non-None."""
        if rpm_limit is not None:
            self.rpm_limit = rpm_limit
        if rpm_remaining is not None:
            self.rpm_remaining = rpm_remaining
        if rpd_limit is not None:
            self.rpd_limit = rpd_limit
        if rpd_remaining is not None:
            self.rpd_remaining = rpd_remaining
        if reset_at is not None:
            self.reset_at = reset_at
        if retry_after is not None:
            self.retry_after = retry_after
        if quota_scope is not None:
            self.quota_scope = quota_scope
        if limit_type is not None:
            self.limit_type = limit_type
        self.last_updated = time.time()

    def update_from_headers(
        self,
        headers: Any,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Parse HTTP response headers to update normalized quota without inventing values."""
        if not headers:
            return

        now = time.time()

        def _get_h(names: List[str]) -> Optional[str]:
            if not isinstance(headers, dict) and not hasattr(headers, "items"):
                return None
            try:
                d = dict(headers)
            except Exception:
                return None
            for n in names:
                if n in d and d[n] is not None:
                    return str(d[n]).strip()
                n_low = n.lower()
                for k, v in d.items():
                    if str(k).lower() == n_low and v is not None:
                        return str(v).strip()
            return None

        # RPM limits
        raw_rpm_limit = _get_h(["x-ratelimit-limit-requests", "x-ratelimit-limit"])
        if raw_rpm_limit and raw_rpm_limit.isdigit():
            self.rpm_limit = int(raw_rpm_limit)

        raw_rpm_rem = _get_h(["x-ratelimit-remaining-requests", "x-ratelimit-remaining"])
        if raw_rpm_rem and raw_rpm_rem.isdigit():
            self.rpm_remaining = int(raw_rpm_rem)

        # RPD limits (if provided)
        raw_rpd_limit = _get_h(["x-ratelimit-limit-requests-day", "x-ratelimit-limit-day"])
        if raw_rpd_limit and raw_rpd_limit.isdigit():
            self.rpd_limit = int(raw_rpd_limit)

        raw_rpd_rem = _get_h(["x-ratelimit-remaining-requests-day", "x-ratelimit-remaining-day"])
        if raw_rpd_rem and raw_rpd_rem.isdigit():
            self.rpd_remaining = int(raw_rpd_rem)

        # Retry-After / Reset (Priority 1: Retry-After, Priority 2: Authoritative reset headers)
        raw_retry = _get_h(["retry-after"])
        raw_reset = _get_h([
            "x-ratelimit-reset",
            "ratelimit-reset",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
            "x-goog-ratelimit-reset",
        ])

        target = raw_retry or raw_reset
        if target:
            ts, cd = parse_reset_header(target)
            if cd is not None and ts is not None:
                self.retry_after = cd
                self.reset_at = ts

        # Quota Scope determination
        p_low = (provider or "").lower()
        if p_low == "groq":
            self.quota_scope = QuotaScope.ACCOUNT_MODEL
        elif p_low in ("openrouter", "cohere", "nvidia"):
            self.quota_scope = QuotaScope.ACCOUNT
        elif p_low == "gemini":
            self.quota_scope = QuotaScope.ACCOUNT_MODEL
        elif self.quota_scope == QuotaScope.UNKNOWN:
            self.quota_scope = QuotaScope.UNKNOWN

        if any(v is not None for v in (raw_rpm_limit, raw_rpm_rem, raw_rpd_limit, raw_rpd_rem, raw_retry, raw_reset)):
            self.last_updated = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert quota info to a dictionary representation."""
        return {
            "rpm_limit": self.rpm_limit,
            "rpm_remaining": self.rpm_remaining,
            "rpd_limit": self.rpd_limit,
            "rpd_remaining": self.rpd_remaining,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after,
            "quota_scope": self.quota_scope,
            "limit_type": self.limit_type,
            "last_updated": self.last_updated,
        }


class DeploymentMetrics:
    """Accurate separation of user requests and upstream attempts.

    Thread-safe atomic counters to prevent race conditions under concurrent requests.
    """

    def __init__(
        self,
        user_requests: int = 0,
        upstream_attempts: int = 0,
        successes: int = 0,
        failures: int = 0,
        rate_limits: int = 0,
        auth_failures: int = 0,
        auth_errors: Optional[int] = None,
    ):
        self._lock = threading.Lock()
        self._user_requests = int(user_requests)
        self._upstream_attempts = int(upstream_attempts)
        self._successes = int(successes)
        self._failures = int(failures)
        self._rate_limits = int(rate_limits)
        if auth_errors is not None and auth_failures == 0:
            self._auth_failures = int(auth_errors)
        else:
            self._auth_failures = int(auth_failures)

    @property
    def user_requests(self) -> int:
        with self._lock:
            return self._user_requests

    @user_requests.setter
    def user_requests(self, val: int) -> None:
        with self._lock:
            self._user_requests = int(val)

    @property
    def upstream_attempts(self) -> int:
        with self._lock:
            return self._upstream_attempts

    @upstream_attempts.setter
    def upstream_attempts(self, val: int) -> None:
        with self._lock:
            self._upstream_attempts = int(val)

    @property
    def successes(self) -> int:
        with self._lock:
            return self._successes

    @successes.setter
    def successes(self, val: int) -> None:
        with self._lock:
            self._successes = int(val)

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @failures.setter
    def failures(self, val: int) -> None:
        with self._lock:
            self._failures = int(val)

    @property
    def rate_limits(self) -> int:
        with self._lock:
            return self._rate_limits

    @rate_limits.setter
    def rate_limits(self, val: int) -> None:
        with self._lock:
            self._rate_limits = int(val)

    @property
    def auth_failures(self) -> int:
        with self._lock:
            return self._auth_failures

    @auth_failures.setter
    def auth_failures(self, val: int) -> None:
        with self._lock:
            self._auth_failures = int(val)

    @property
    def auth_errors(self) -> int:
        return self.auth_failures

    @auth_errors.setter
    def auth_errors(self, val: int) -> None:
        self.auth_failures = val

    def increment_user_requests(self, count: int = 1) -> int:
        with self._lock:
            self._user_requests += count
            return self._user_requests

    def record_attempt(self, count: int = 1) -> int:
        with self._lock:
            self._upstream_attempts += count
            return self._upstream_attempts

    def record_success(self, count: int = 1) -> int:
        with self._lock:
            self._successes += count
            return self._successes

    def record_failure(self, count: int = 1) -> int:
        with self._lock:
            self._failures += count
            return self._failures

    def record_rate_limit(self, count: int = 1) -> int:
        with self._lock:
            self._rate_limits += count
            return self._rate_limits

    def record_auth_failure(self, count: int = 1) -> int:
        with self._lock:
            self._auth_failures += count
            return self._auth_failures

    def to_dict(self) -> Dict[str, int]:
        with self._lock:
            return {
                "user_requests": self._user_requests,
                "upstream_attempts": self._upstream_attempts,
                "successes": self._successes,
                "failures": self._failures,
                "rate_limits": self._rate_limits,
                "auth_failures": self._auth_failures,
                "auth_errors": self._auth_failures,
            }

    def __repr__(self) -> str:
        return (
            f"DeploymentMetrics(user_requests={self.user_requests}, "
            f"upstream_attempts={self.upstream_attempts}, successes={self.successes}, "
            f"failures={self.failures}, rate_limits={self.rate_limits}, "
            f"auth_failures={self.auth_failures})"
        )


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
    circuit_open_until: float = 0.0
    half_open_probes: int = 0
    max_half_open_probes: int = CIRCUIT_BREAKER_HALF_OPEN_PROBES
    consecutive_failures: int = 0
    max_consecutive_failures: int = CIRCUIT_BREAKER_MAX_FAILURES
    recovery_time: float = CIRCUIT_BREAKER_RECOVERY_TIME
    weight: float = 1.0
    auth_failed_key_hash: Optional[str] = None
    last_auth_probe_at: float = 0.0
    auth_probe_backoff: float = 300.0
    is_stale: bool = False
    stale_reason: Optional[str] = None
    last_discovered_at: float = 0.0
    _active_probe_request_ids: Set[str] = field(default_factory=set, repr=False, compare=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @property
    def _active_probe_request_id(self) -> Optional[str]:
        """Backwards-compatibility property returning active probe request ID."""
        with self._state_lock:
            return next(iter(self._active_probe_request_ids)) if self._active_probe_request_ids else None

    @_active_probe_request_id.setter
    def _active_probe_request_id(self, val: Optional[str]) -> None:
        with self._state_lock:
            if val is None:
                self._active_probe_request_ids.clear()
            else:
                self._active_probe_request_ids = {val}
            self.half_open_probes = len(self._active_probe_request_ids)

    @property
    def is_quarantined(self) -> bool:
        """True if deployment is quarantined due to auth or permanent failure."""
        return self.circuit_state in (CircuitState.QUARANTINED, CircuitState.AUTH_FAILED)

    @property
    def is_auth_failed(self) -> bool:
        """True if deployment specifically failed authentication."""
        return self.circuit_state == CircuitState.AUTH_FAILED

    @property
    def rpm_limit(self) -> Optional[int]:
        """Requests per minute limit."""
        return self.quota.rpm_limit

    @property
    def rpm_remaining(self) -> Optional[int]:
        """Requests per minute remaining."""
        return self.quota.rpm_remaining

    @property
    def rpd_limit(self) -> Optional[int]:
        """Requests per day limit."""
        return self.quota.rpd_limit

    @property
    def rpd_remaining(self) -> Optional[int]:
        """Requests per day remaining."""
        return self.quota.rpd_remaining

    @property
    def reset_at(self) -> Optional[float]:
        """Timestamp when current quota/cooldown resets."""
        return self.quota.reset_at

    @property
    def retry_after(self) -> Optional[float]:
        """Seconds to wait before retrying."""
        return self.quota.retry_after

    @property
    def quota_scope(self) -> str:
        """Known scope of the quota limit ('provider', 'account', 'model', 'account_model', 'unknown')."""
        return self.quota.quota_scope

    def update_quota(
        self,
        rpm_limit: Optional[int] = None,
        rpm_remaining: Optional[int] = None,
        rpd_limit: Optional[int] = None,
        rpd_remaining: Optional[int] = None,
        reset_at: Optional[float] = None,
        retry_after: Optional[float] = None,
        quota_scope: Optional[str] = None,
        limit_type: Optional[str] = None,
    ) -> None:
        """Update normalized quota metadata from provider responses or error diagnostics."""
        self.quota.update(
            rpm_limit=rpm_limit,
            rpm_remaining=rpm_remaining,
            rpd_limit=rpd_limit,
            rpd_remaining=rpd_remaining,
            reset_at=reset_at,
            retry_after=retry_after,
            quota_scope=quota_scope,
            limit_type=limit_type,
        )

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
        """Timestamp until which deployment is in cooldown or circuit breaker OPEN."""
        return max(self.circuit_open_until or 0.0, self.quota.reset_at or 0.0)

    @blocked_until.setter
    def blocked_until(self, val: float) -> None:
        self.circuit_open_until = val

    @property
    def cooldown_until(self) -> float:
        """Timestamp until which deployment is in cooldown or circuit breaker OPEN."""
        return max(self.circuit_open_until or 0.0, self.quota.reset_at or 0.0)

    @cooldown_until.setter
    def cooldown_until(self, val: float) -> None:
        self.circuit_open_until = val

    @property
    def blocked_reason(self) -> Optional[str]:
        """Reason deployment is in cooldown or quarantine."""
        return self.state_reason

    @blocked_reason.setter
    def blocked_reason(self, val: Optional[str]) -> None:
        self.state_reason = val

    def record_user_request(self, count: int = 1) -> int:
        """Atomically record one user request on this deployment."""
        return self.metrics.increment_user_requests(count)

    def record_attempt(self, count: int = 1) -> int:
        """Atomically record one upstream provider attempt."""
        return self.metrics.record_attempt(count)

    @property
    def user_requests(self) -> int:
        """User requests count."""
        return self.metrics.user_requests

    @property
    def upstream_attempts(self) -> int:
        """Upstream attempts count."""
        return self.metrics.upstream_attempts

    @property
    def attempts(self) -> int:
        """Alias for upstream_attempts count."""
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
    def auth_failures(self) -> int:
        """Authentication failures count."""
        return self.metrics.auth_failures

    @property
    def auth_errors(self) -> int:
        """Alias for auth_failures."""
        return self.metrics.auth_failures

    @property
    def remaining_cooldown(self) -> float:
        """Returns the number of seconds remaining in explicit cooldown or circuit breaker OPEN."""
        now = time.time()
        expiry = max(self.circuit_open_until or 0.0, self.quota.reset_at or 0.0)
        if expiry > 0:
            return max(0.0, expiry - now)
        return 0.0

    @property
    def display_name(self) -> str:
        """Formatted human-readable deployment label without exposing secrets."""
        return f"{self.provider.title()} {self.account} ({self.model})"

    def is_eligible(self, now: Optional[float] = None) -> bool:
        """Check if deployment is eligible to receive traffic.

        Behavior:
        - HEALTHY: eligible normally.
        - OPEN: temporarily excluded after repeated qualifying failures.
        - HALF_OPEN: allows controlled probing.
        - AUTH_FAILED / QUARANTINED: excluded until revalidated/reset.
        """
        if not self.api_key or not self.api_key.strip():
            return False
        if self.circuit_state in (CircuitState.QUARANTINED, CircuitState.AUTH_FAILED):
            return False
        current_time = now if now is not None else time.time()
        if self.quota.reset_at and current_time < self.quota.reset_at:
            return False
        if self.circuit_state == CircuitState.OPEN:
            return False
        if self.circuit_state == CircuitState.HALF_OPEN:
            with self._state_lock:
                if len(self._active_probe_request_ids) >= self.max_half_open_probes:
                    return False
            return True
        return True

    def is_available(self, now: Optional[float] = None) -> bool:
        """Alias for is_eligible."""
        return self.is_eligible(now)

    def claim_half_open_probe(self, request_id: Optional[str] = None) -> bool:
        """Atomically claim a controlled probe slot in HALF_OPEN state.

        Returns True if the probe was successfully claimed or already claimed by this request_id,
        False if not in HALF_OPEN or probe capacity is exhausted.
        """
        with self._state_lock:
            if self.circuit_state != CircuitState.HALF_OPEN:
                return False
            req_key = request_id or "__anonymous__"
            if req_key in self._active_probe_request_ids:
                return True
            if len(self._active_probe_request_ids) >= self.max_half_open_probes:
                return False
            self._active_probe_request_ids.add(req_key)
            self.half_open_probes = len(self._active_probe_request_ids)
            return True

    def release_half_open_probe(self, request_id: Optional[str] = None) -> None:
        """Release a claimed probe slot if the probe request was not dispatched."""
        with self._state_lock:
            req_key = request_id or "__anonymous__"
            if req_key in self._active_probe_request_ids:
                self._active_probe_request_ids.remove(req_key)
            self.half_open_probes = len(self._active_probe_request_ids)

    def admit_to_request_pool(self, now: Optional[float] = None, request_id: Optional[str] = None) -> bool:
        """Atomically admit deployment into an active request pool.

        For HALF_OPEN deployments, atomically claims a single probe slot tied to request_id before admission.
        Returns False if probe cannot be claimed, circuit is OPEN, quota is exhausted, or credentials missing.
        """
        current_time = now if now is not None else time.time()
        self.refresh(current_time)
        with self._state_lock:
            if not self.api_key or not self.api_key.strip():
                return False
            if self.circuit_state in (CircuitState.QUARANTINED, CircuitState.AUTH_FAILED):
                return False
            if self.quota.reset_at and current_time < self.quota.reset_at:
                return False
            if self.circuit_state == CircuitState.OPEN:
                return False
            if self.circuit_state == CircuitState.HALF_OPEN:
                return self.claim_half_open_probe(request_id=request_id)
            return True

    def transition_to_healthy(
        self,
        reason: str = "Recovered to HEALTHY",
        request_id: Optional[str] = None,
    ) -> None:
        """Atomically transition deployment to HEALTHY state under _state_lock."""
        with self._state_lock:
            self.consecutive_failures = 0
            self.circuit_open_until = 0.0
            if request_id:
                self._active_probe_request_ids.discard(request_id)
            else:
                self._active_probe_request_ids.clear()
            self.half_open_probes = len(self._active_probe_request_ids)
            self.circuit_state = CircuitState.HEALTHY
            self.state_reason = reason
            self.state_updated_at = time.time()
            self.auth_failed_key_hash = None
            self.auth_probe_backoff = 300.0

    def transition_to_open(
        self,
        cooldown_seconds: float,
        reason: str,
        reset_at: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Atomically trip circuit breaker to OPEN state under _state_lock."""
        with self._state_lock:
            now = time.time()
            self.circuit_state = CircuitState.OPEN
            self._active_probe_request_ids.clear()
            self.half_open_probes = 0
            if reset_at is not None and reset_at > 0:
                self.quota.reset_at = reset_at
                self.quota.retry_after = cooldown_seconds
                self.circuit_open_until = max(now + cooldown_seconds, reset_at)
            else:
                self.circuit_open_until = now + cooldown_seconds
            self.state_reason = reason
            self.state_updated_at = now

    def transition_to_half_open(
        self,
        reason: str = "Recovery condition reached; entering HALF_OPEN for controlled probe",
    ) -> None:
        """Atomically transition circuit breaker to HALF_OPEN under _state_lock."""
        with self._state_lock:
            self.circuit_state = CircuitState.HALF_OPEN
            self._active_probe_request_ids.clear()
            self.half_open_probes = 0
            self.state_reason = reason
            self.state_updated_at = time.time()

    def transition_to_quarantine(
        self,
        reason: str,
        state: CircuitState = CircuitState.QUARANTINED,
        failed_key: Optional[str] = None,
    ) -> None:
        """Atomically quarantine deployment under _state_lock."""
        with self._state_lock:
            self.circuit_state = state
            self._active_probe_request_ids.clear()
            self.half_open_probes = 0
            key_to_hash = failed_key or self.api_key or ""
            if key_to_hash:
                self.auth_failed_key_hash = hashlib.sha256(key_to_hash.encode("utf-8")).hexdigest()
            self.state_reason = sanitize_secret(reason, self.api_key)
            self.state_updated_at = time.time()

    def record_failure(
        self,
        max_failures: Optional[int] = None,
        recovery_time: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Record qualifying upstream failure and update circuit breaker state."""
        threshold = max_failures if max_failures is not None else self.max_consecutive_failures
        cooldown = recovery_time if recovery_time is not None else self.recovery_time

        with self._state_lock:
            now = time.time()
            if request_id:
                self._active_probe_request_ids.discard(request_id)
                self.half_open_probes = len(self._active_probe_request_ids)

            if self.circuit_state == CircuitState.HALF_OPEN:
                backoff_cooldown = cooldown * 2.0
                open_until = now + backoff_cooldown
                self.transition_to_open(
                    cooldown_seconds=backoff_cooldown,
                    reason=f"Probe request failed while in HALF_OPEN; circuit OPEN until {utc_string(open_until)}",
                    request_id=request_id,
                )
            else:
                self.consecutive_failures += 1
                self.state_updated_at = now
                if self.consecutive_failures >= threshold:
                    open_until = now + cooldown
                    self.transition_to_open(
                        cooldown_seconds=cooldown,
                        reason=f"Consecutive failure threshold reached ({self.consecutive_failures}); circuit OPEN until {utc_string(open_until)}",
                        request_id=request_id,
                    )

    def record_success(self, request_id: Optional[str] = None) -> None:
        """Record upstream success and restore circuit to HEALTHY."""
        self.transition_to_healthy(
            reason="Probe succeeded; recovered to HEALTHY",
            request_id=request_id,
        )

    def refresh(self, now: Optional[float] = None) -> bool:
        """Evaluate circuit breaker, quota recovery, and dynamic key replacement."""
        current_time = now if now is not None else time.time()

        with self._state_lock:
            if self.circuit_state == CircuitState.AUTH_FAILED and self.auth_failed_key_hash:
                curr_hash = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
                if curr_hash != self.auth_failed_key_hash:
                    self.transition_to_healthy(reason="API key updated; authentication quarantine lifted")

            if self.quota.reset_at and current_time >= self.quota.reset_at:
                self.quota.reset_at = None

            if self.circuit_state == CircuitState.OPEN:
                is_circuit_cooled = bool(not self.circuit_open_until or current_time >= self.circuit_open_until)
                is_quota_cooled = bool(not self.quota.reset_at or current_time >= self.quota.reset_at)
                is_fallback_cooled = bool(
                    (self.circuit_open_until or self.quota.reset_at)
                    or (current_time - self.state_updated_at) >= self.recovery_time
                )

                if is_circuit_cooled and is_quota_cooled and is_fallback_cooled:
                    self.transition_to_half_open(reason="Recovery condition reached; entering HALF_OPEN for controlled probe")

            return self.is_eligible(current_time)

    def routing_config_fingerprint(self) -> Tuple[Any, ...]:
        """Return a tuple capturing all routing-relevant configuration and state.

        Changes to credentials, weights, API base, circuit state, quota reset timestamps,
        tags, or context window immediately change this fingerprint, triggering pool refresh.
        Never exposes raw secrets.
        """
        key_hash = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest() if self.api_key else ""
        return (
            self.id,
            self.provider,
            self.account,
            self.model,
            self.litellm_model,
            key_hash,
            self.api_base,
            self.weight,
            self.circuit_state.value,
            self.circuit_open_until,
            self.quota.reset_at,
            self.quota.rpm_remaining,
            self.quota.rpd_remaining,
            self.available,
            self.is_stale,
            tuple(sorted(self.capabilities.to_tags())),
            self.capabilities.context_window,
        )

    def to_litellm_dict(self, group_override: Optional[str] = None) -> Dict[str, Any]:
        """Convert deployment into LiteLLM router dictionary format with tags and weights."""
        group_name = group_override if group_override else self.logical_model
        litellm_params: Dict[str, Any] = {
            "model": self.litellm_model,
            "api_key": self.api_key,
            "tags": self.capabilities.to_tags(),
            "weight": self.weight,
            "model_info": {
                "id": self.id,
                "provider": self.provider,
                "account": self.account,
                "model": self.model,
                "logical_model": self.logical_model,
            },
            "metadata": {
                "deployment_id": self.id,
                "provider": self.provider,
                "account": self.account,
                "model": self.model,
                "logical_model": self.logical_model,
                "capabilities": self.capabilities.to_dict(),
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
                "capabilities": self.capabilities.to_dict(),
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
        capabilities: Optional[ModelCapabilities] = None,
    ):
        caps = capabilities or ModelCapabilities(
            streaming=True,
            context_window=4096,
        )
        super().__init__(
            id=f"openrouter-{name}",
            provider=MODEL_GROUP_OPENROUTER,
            account=name,
            model=OPENROUTER_ACTUAL_MODEL,
            logical_model=MODEL_GROUP_OPENROUTER,
            litellm_model=OPENROUTER_LITELLM_MODEL,
            api_key=api_key,
            capabilities=caps,
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
        capabilities: Optional[ModelCapabilities] = None,
        is_stale: bool = False,
        stale_reason: Optional[str] = None,
    ):
        caps = capabilities or ModelCapabilities(
            streaming=True,
            context_window=4096,
        )
        super().__init__(
            id=f"gemini-{account_name}-{model_name}",
            provider=MODEL_GROUP_GEMINI,
            account=account_name,
            model=model_name,
            logical_model=MODEL_GROUP_GEMINI,
            litellm_model=f"gemini/{model_name}",
            api_key=api_key,
            capabilities=caps,
            is_stale=is_stale,
            stale_reason=stale_reason,
        )
