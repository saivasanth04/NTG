"""LiteLLM-based multi-provider smart router with NTG intelligence."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

import litellm
from litellm import RetryPolicy, Router
from litellm.integrations.custom_logger import CustomLogger

from ntg.config import (
    DEFAULT_LOGICAL_MODEL,
    DEFAULT_NUM_RETRIES,
    MAX_RETRY_BUDGET,
    MODEL_GROUP_AUTO,
    MODEL_GROUP_COHERE,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NTG_AUTO,
    MODEL_GROUP_NVIDIA,
    MODEL_GROUP_OPENROUTER,
    PROVIDER_FALLBACKS,
)
from ntg.diagnostics import (
    print_account_status,
    print_divider,
    print_rate_limit_details,
    print_request_execution,
)
from ntg.discovery import build_all_deployments
from ntg.exceptions import (
    CATEGORY_AUTH_ERROR,
    CATEGORY_DAILY_QUOTA,
    CATEGORY_PROVIDER_LIMIT,
    CATEGORY_REQUEST_ERROR,
    CATEGORY_SERVER_ERROR,
    CATEGORY_UNKNOWN_LIMIT,
    parse_provider_error,
)
from ntg.models import CircuitState, Deployment, ModelCapabilities, QuotaScope, sanitize_secret, utc_string
from ntg.state import StateManager


class NTGTelemetryLogger(CustomLogger):
    """LiteLLM custom logger callback for telemetry, circuit breaker updates, and quota intelligence.

    Extracts per-attempt deployment metadata directly from event kwargs to ensure
    100% thread safety and eliminate shared mutable request state.
    """

    def __init__(self, router: UnifiedNTGRouter):
        super().__init__()
        self.router = router

    def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any = None) -> None:
        """LiteLLM pre-API call callback: atomically claim HALF_OPEN probe (Rule 3, 11)."""
        kw = kwargs or {}
        dep_id = kw.get("litellm_params", {}).get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kw.get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kw.get("litellm_params", {}).get("metadata", {}).get("deployment_id")
        if not dep_id:
            dep_id = kw.get("metadata", {}).get("deployment_id")
        if not dep_id and isinstance(model, str):
            dep_id = model

        deployment = self.router.deployment_map.get(dep_id)
        if deployment and deployment.circuit_state == CircuitState.HALF_OPEN:
            deployment.claim_half_open_probe()

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Handle upstream success callback."""
        dep_id = kwargs.get("litellm_params", {}).get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kwargs.get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kwargs.get("litellm_params", {}).get("metadata", {}).get("deployment_id")
        if not dep_id:
            dep_id = kwargs.get("metadata", {}).get("deployment_id")

        deployment = getattr(response_obj, "_ntg_deployment", None)
        if not deployment and dep_id:
            deployment = self.router.deployment_map.get(dep_id)

        if not deployment and dep_id:
            for dep in self.router.deployments:
                if dep.id == dep_id or dep.model == dep_id or dep.litellm_model == dep_id:
                    deployment = dep
                    break

        if not deployment:
            model_arg = kwargs.get("model") or kwargs.get("litellm_params", {}).get("model")
            if model_arg:
                if model_arg in self.router.deployment_map:
                    deployment = self.router.deployment_map[model_arg]
                else:
                    for dep in self.router.deployments:
                        if dep.id == model_arg or dep.model == model_arg or dep.litellm_model == model_arg:
                            deployment = dep
                            break

        if deployment:
            deployment.metrics.record_attempt()
            deployment.metrics.record_success()
            deployment.record_success()

            # Parse headers from response or kwargs to update normalized quota metadata
            headers = None
            if response_obj is not None:
                headers = getattr(response_obj, "_response_headers", None) or getattr(response_obj, "headers", None)
            if not headers:
                headers = kwargs.get("response_headers")

            if headers:
                deployment.quota.update_from_headers(
                    headers,
                    provider=deployment.provider,
                    model=deployment.model,
                )
                self.router.state_manager.record_quota(deployment.id, deployment.quota)

            # Record healthy circuit state and update persisted state
            self.router.state_manager.record_circuit_state(deployment.id, CircuitState.HEALTHY)
            self.router.state_manager.record_metrics(deployment.id, deployment.metrics)

            # Request-local deployment attribution stamped directly onto response instance
            if response_obj is not None:
                try:
                    setattr(response_obj, "_ntg_deployment", deployment)
                except Exception:
                    pass

    def log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Handle upstream failure callback with granular error classification."""
        dep_id = kwargs.get("litellm_params", {}).get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kwargs.get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kwargs.get("litellm_params", {}).get("metadata", {}).get("deployment_id")
        if not dep_id:
            dep_id = kwargs.get("metadata", {}).get("deployment_id")

        exc = kwargs.get("exception")
        if not exc and isinstance(response_obj, Exception):
            exc = response_obj

        if not dep_id and exc and hasattr(exc, "model_id"):
            dep_id = getattr(exc, "model_id")

        deployment = self.router.deployment_map.get(dep_id)
        if not deployment and dep_id:
            for dep in self.router.deployments:
                if dep.id == dep_id or dep.model == dep_id or dep.litellm_model == dep_id:
                    deployment = dep
                    break

        if not deployment:
            model_arg = kwargs.get("model") or kwargs.get("litellm_params", {}).get("model")
            if model_arg:
                if model_arg in self.router.deployment_map:
                    deployment = self.router.deployment_map[model_arg]
                else:
                    for dep in self.router.deployments:
                        if dep.id == model_arg or dep.model == model_arg or dep.litellm_model == model_arg:
                            deployment = dep
                            break

        if deployment:
            deployment.metrics.record_attempt()

            if exc:
                info = parse_provider_error(exc, provider=deployment.provider)

                # Update normalized quota metadata (Rule 4, 8)
                deployment.update_quota(
                    rpm_limit=info.rpm_limit,
                    rpm_remaining=info.rpm_remaining,
                    rpd_limit=info.rpd_limit,
                    rpd_remaining=info.rpd_remaining,
                    reset_at=info.reset_timestamp,
                    retry_after=info.cooldown_seconds,
                    quota_scope=info.quota_scope,
                    limit_type=info.limit_type,
                )
                self.router.state_manager.record_quota(deployment.id, deployment.quota)

                # Error categorization & metrics
                if info.category in (CATEGORY_PROVIDER_LIMIT, CATEGORY_DAILY_QUOTA, CATEGORY_UNKNOWN_LIMIT):
                    deployment.metrics.record_rate_limit()
                    deployment.metrics.record_failure()
                elif info.category == CATEGORY_AUTH_ERROR:
                    deployment.metrics.record_auth_failure()
                    deployment.metrics.record_failure()
                else:
                    deployment.metrics.record_failure()

                # NTG Confirmed Authentication Failure Quarantine (Rule 1, 2, 8)
                if info.category == CATEGORY_AUTH_ERROR:
                    safe_reason = sanitize_secret(f"Authentication failure: {info.message}", deployment.api_key)
                    self.router._propagate_account_auth_quarantine(
                        provider=deployment.provider,
                        account=deployment.account,
                        reason=safe_reason,
                        failed_key=deployment.api_key,
                    )

                # NTG General Quarantine (e.g. 404 Model Not Found)
                elif info.should_quarantine:
                    deployment.circuit_state = CircuitState.QUARANTINED
                    deployment.state_reason = sanitize_secret(f"Model unavailable: {info.message}", deployment.api_key)
                    self.router.state_manager.record_quarantine(
                        deployment.id, deployment.state_reason, state=CircuitState.QUARANTINED
                    )

                # NTG Authoritative Rate Limit / Quota Exhaustion handling (429, daily, rpm, etc.)
                elif info.status_code == 429 or info.category in (
                    CATEGORY_DAILY_QUOTA,
                    CATEGORY_PROVIDER_LIMIT,
                    CATEGORY_UNKNOWN_LIMIT,
                ):
                    deployment.circuit_state = CircuitState.OPEN
                    deployment.quota.reset_at = info.reset_timestamp
                    deployment.quota.retry_after = info.cooldown_seconds
                    deployment.state_reason = (
                        f"Daily quota exhausted: {info.message}"
                        if info.category == CATEGORY_DAILY_QUOTA
                        else f"Rate limit reached ({info.limit_type}): {info.message}"
                    )
                    self.router.state_manager.record_cooldown(
                        deployment.id,
                        cooldown_seconds_or_until=deployment.quota.reset_at,
                        reason_or_limit_type=info.limit_type or "429",
                        reset_timestamp=deployment.quota.reset_at,
                        quota_scope=info.quota_scope,
                    )
                    self.router.state_manager.record_circuit_state(
                        deployment.id, CircuitState.OPEN, reason=deployment.state_reason
                    )

                    # Propagate account-wide quota block to peer deployments under the same account ONLY if scope is account
                    # Model-level limits (QuotaScope.ACCOUNT_MODEL) do NOT exhaust the entire account (Rule 7)
                    if info.quota_scope in (QuotaScope.ACCOUNT, QuotaScope.PROVIDER):
                        self.router._propagate_account_quota_exhaustion(
                            deployment.provider,
                            deployment.account,
                            deployment.quota.reset_at,
                            info.message,
                        )

                elif info.category == CATEGORY_REQUEST_ERROR:
                    # Client-side request error (bad request, parameter mismatch, context exceeded, content violation).
                    # Rule 8: Capability incompatibility and client errors must never open the circuit breaker.
                    pass

                else:
                    # Qualifying upstream failure (server error 500/502/503/504, connection error, timeout):
                    # LiteLLM owns routing cooldown and immediate failover.
                    # NTG updates failure tracking and trips circuit breaker only if threshold is reached.
                    deployment.record_failure()
                    if deployment.circuit_state == CircuitState.OPEN:
                        self.router.state_manager.record_circuit_state(
                            deployment.id,
                            CircuitState.OPEN,
                            reason=deployment.state_reason or "Qualifying upstream failure",
                            recovery_time=deployment.circuit_open_until,
                        )
            else:
                deployment.metrics.record_failure()

            self.router.state_manager.record_metrics(deployment.id, deployment.metrics)

    async def async_log_pre_api_call(self, model: Any, messages: Any, kwargs: Any = None) -> None:
        """Async LiteLLM pre-API call callback: delegates to log_pre_api_call."""
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Async LiteLLM success callback: delegates to log_success_event."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Async LiteLLM failure callback: delegates to log_failure_event."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


class NoEligibleDeploymentsError(RuntimeError):
    """Raised when no deployments satisfy strict request requirements or eligibility checks."""
    pass


class RequestRequirements:
    """Extracted capability and constraint requirements from a user request."""

    def __init__(
        self,
        capabilities: Optional[ModelCapabilities] = None,
        coding_preferred: bool = False,
        strict_requirements: Optional[Set[str]] = None,
    ):
        self.capabilities = capabilities or ModelCapabilities()
        self.coding_preferred = coding_preferred
        self.strict_requirements = strict_requirements or set()

    def describe(self) -> str:
        """Human-readable summary of active requirements."""
        reqs = []
        if self.capabilities.tool_calling is True:
            reqs.append("tool_calling")
        if self.capabilities.structured_output is True:
            reqs.append("structured_output")
        if self.capabilities.vision is True:
            reqs.append("vision")
        if self.capabilities.streaming is True:
            reqs.append("streaming")
        if self.capabilities.reasoning is True:
            reqs.append("reasoning")
        if self.capabilities.coding is True:
            reqs.append("coding")
        elif self.coding_preferred:
            reqs.append("coding (preferred)")
        if self.capabilities.context_window > 4096:
            reqs.append(f"context_window>={self.capabilities.context_window:,}")
        return ", ".join(reqs) if reqs else "none"


def extract_request_requirements(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
    min_context: Optional[int] = None,
    **kwargs: Any,
) -> RequestRequirements:
    """Extract deterministic capability requirements from OpenAI/LiteLLM request arguments."""
    req_caps = ModelCapabilities()
    strict_reqs: Set[str] = set()
    coding_preferred = False

    # 1. Tools / functions requested -> require tool_calling (Rule 1)
    tools = kwargs.get("tools")
    functions = kwargs.get("functions")
    tool_choice = kwargs.get("tool_choice")
    if (tools and len(tools) > 0) or (functions and len(functions) > 0) or tool_choice:
        req_caps.tool_calling = True
        strict_reqs.add("tool_calling")

    # 2. Structured output requested -> require structured_output (Rule 2)
    response_format = kwargs.get("response_format")
    schema = kwargs.get("schema")
    fmt = kwargs.get("format")
    if response_format is not None or schema is not None or fmt == "json":
        req_caps.structured_output = True
        strict_reqs.add("structured_output")

    # 3. Image / multimodal input -> require vision (Rule 3)
    has_image = False
    if kwargs.get("images"):
        has_image = True
    if messages and isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_type = str(part.get("type", "")).lower()
                        if part_type in ("image_url", "image") or "image_url" in part:
                            has_image = True
                            break
            elif isinstance(content, str) and ("data:image/" in content or "base64," in content):
                has_image = True
            if has_image:
                break
    if prompt and ("data:image/" in prompt or "base64," in prompt):
        has_image = True

    if has_image:
        req_caps.vision = True
        strict_reqs.add("vision")

    # 4. Streaming requested -> require streaming (Rule 4)
    if kwargs.get("stream") is True or kwargs.get("streaming") is True:
        req_caps.streaming = True
        strict_reqs.add("streaming")

    # 5. Reasoning-specific routing (Rule 6 - explicit requirements only)
    if (
        model == "reasoning"
        or kwargs.get("reasoning") is True
        or kwargs.get("reasoning_effort") is not None
    ):
        req_caps.reasoning = True
        strict_reqs.add("reasoning")

    # 6. Coding-oriented requests (Rule 5 - prefer coding, no AI intent classifier)
    if model == "coding" or kwargs.get("coding") is True or kwargs.get("task") in ("coding", "code"):
        req_caps.coding = True
        strict_reqs.add("coding")
    else:
        text_to_check = prompt or ""
        if not text_to_check and messages and isinstance(messages, list):
            for m in messages:
                c = m.get("content") if isinstance(m, dict) else ""
                if isinstance(c, str):
                    text_to_check += " " + c
        if "```" in text_to_check or "def " in text_to_check or "class " in text_to_check:
            coding_preferred = True

    # 7. Context window requirement
    if min_context and min_context > 0:
        req_caps.context_window = min_context
        strict_reqs.add("context_window")

    # 8. Explicit capabilities argument
    if capabilities is not None:
        explicit_parsed = UnifiedNTGRouter._parse_capabilities(capabilities)
        if explicit_parsed:
            if explicit_parsed.coding is True:
                req_caps.coding = True
                strict_reqs.add("coding")
            if explicit_parsed.reasoning is True:
                req_caps.reasoning = True
                strict_reqs.add("reasoning")
            if explicit_parsed.vision is True:
                req_caps.vision = True
                strict_reqs.add("vision")
            if explicit_parsed.tool_calling is True:
                req_caps.tool_calling = True
                strict_reqs.add("tool_calling")
            if explicit_parsed.structured_output is True:
                req_caps.structured_output = True
                strict_reqs.add("structured_output")
            if explicit_parsed.streaming is True:
                req_caps.streaming = True
                strict_reqs.add("streaming")
            if explicit_parsed.context_window > req_caps.context_window:
                req_caps.context_window = explicit_parsed.context_window
                strict_reqs.add("context_window")

    return RequestRequirements(
        capabilities=req_caps,
        coding_preferred=coding_preferred,
        strict_requirements=strict_reqs,
    )


class UnifiedNTGRouter:
    """Unified multi-provider router delegating load balancing, retries, and failovers to LiteLLM.

    NTG provides eligibility filtering, capability requirements, quota intelligence,
    circuit breaker lifecycle, and persisted telemetry.
    LiteLLM owns deployment selection, distributed load balancing, and failover.
    """

    def __init__(
        self,
        accounts: Optional[List[Deployment]] = None,
        default_model: Optional[str] = None,
        num_retries: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        provider_fallbacks: Optional[Dict[str, List[str]]] = None,
        openrouter_keys: Optional[Dict[str, str]] = None,
        groq_keys: Optional[Dict[str, str]] = None,
        nvidia_keys: Optional[Dict[str, str]] = None,
        cohere_keys: Optional[Dict[str, str]] = None,
        gemini_keys: Optional[Dict[str, str]] = None,
        gemini_api_key: Optional[str] = None,
        state_manager: Optional[StateManager] = None,
    ):
        # Configure logging to suppress verbose notices
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

        self.state_manager = state_manager or StateManager()

        # Explicit cross-provider fallback mappings (default: empty to prevent unexpected jumps)
        self.provider_fallbacks = (
            dict(provider_fallbacks)
            if provider_fallbacks is not None
            else dict(PROVIDER_FALLBACKS)
        )

        # 1. Build all deployments across providers and restore persisted state
        if accounts is not None:
            self.deployments: List[Deployment] = list(accounts)
            for dep in self.deployments:
                self.state_manager.apply_to_deployment(dep)
                dep.metrics = self.state_manager.get_metrics(dep.id)
        else:
            g_keys = gemini_keys
            if g_keys is None and gemini_api_key and gemini_api_key != "YOUR_NEW_GOOGLE_API_KEY":
                g_keys = {"account_1": gemini_api_key}

            self.deployments = build_all_deployments(
                openrouter_keys=openrouter_keys,
                groq_keys=groq_keys,
                nvidia_keys=nvidia_keys,
                cohere_keys=cohere_keys,
                gemini_keys=g_keys,
                state_manager=self.state_manager,
            )

        self.deployment_map: Dict[str, Deployment] = {dep.id: dep for dep in self.deployments}

        # 2. Build LiteLLM model_list deployment pool
        active_deployments = [
            dep for dep in self.deployments
            if dep.api_key and dep.api_key.strip() and dep.available
        ]

        model_list: List[Dict[str, Any]] = []
        registered_pairs: Set[tuple[str, str]] = set()

        for dep in active_deployments:
            litellm_dict = dep.to_litellm_dict()

            # Target groups:
            # 1. Provider group (e.g. openrouter, groq, nvidia, cohere, gemini)
            # 2. Global auto group (auto)
            # 3. Individual deployment ID for direct, isolated eligible pool routing
            # 4. Capability groups (coding, reasoning, vision, tool_calling, etc.)
            groups = [dep.logical_model, MODEL_GROUP_AUTO, dep.id]

            if dep.capabilities.coding:
                groups.append("coding")
            if dep.capabilities.reasoning:
                groups.append("reasoning")
            if dep.capabilities.vision:
                groups.append("vision")
            if dep.capabilities.tool_calling:
                groups.append("tool_calling")
            if dep.capabilities.structured_output:
                groups.append("structured_output")
            if dep.capabilities.streaming:
                groups.append("streaming")

            for g in groups:
                pair = (g, dep.id)
                if pair not in registered_pairs:
                    registered_pairs.add(pair)
                    entry = dict(litellm_dict)
                    entry["model_name"] = g
                    model_list.append(entry)

        # 3. Resolve default model
        self.default_model = default_model or DEFAULT_LOGICAL_MODEL

        # 4. Configure bounded retry budget (never derived from deployment count; default: 1 retry)
        if num_retries is not None:
            self.num_retries = max(0, min(num_retries, MAX_RETRY_BUDGET))
        else:
            self.num_retries = DEFAULT_NUM_RETRIES

        self.retry_policy = retry_policy or RetryPolicy(
            BadRequestErrorRetries=0,
            ContentPolicyViolationErrorRetries=0,
            AuthenticationErrorRetries=0,
            RateLimitErrorRetries=1,
            TimeoutErrorRetries=1,
            InternalServerErrorRetries=1,
        )

        # 5. Instantiate LiteLLM Router with bounded retries and failover
        # Note: Broad static fallbacks are removed. Compatibility-aware fallbacks are constructed
        # dynamically per request based on verified eligibility, quotas, and capability requirements.
        self.router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            cooldown_time=60.0,
            allowed_fails=1,
            num_retries=self.num_retries,
            retry_policy=self.retry_policy,
            fallbacks=None,
            model_group_alias={
                MODEL_GROUP_NTG_AUTO: MODEL_GROUP_AUTO,
            },
        )

        # 7. Register telemetry and circuit breaker callback via public litellm.callbacks contract
        self.telemetry_logger = NTGTelemetryLogger(self)
        litellm.callbacks = [cb for cb in litellm.callbacks if not isinstance(cb, NTGTelemetryLogger)]
        litellm.callbacks.append(self.telemetry_logger)

    def _propagate_account_auth_quarantine(
        self,
        provider: str,
        account: str,
        reason: str,
        failed_key: Optional[str] = None,
    ) -> None:
        """Propagate an authentication failure quarantine across all deployments under the same account.

        Guarantees:
        1. Quarantines only deployments sharing the exact same provider and account (Account Isolation).
        2. Sets circuit state to AUTH_FAILED and records key hash for dynamic change detection.
        3. Never exposes raw credentials in logs or reason messages.
        4. Excludes them from routing pools without relying on fixed arbitrary cooldowns.
        """
        for dep in self.deployments:
            if dep.provider == provider and dep.account == account:
                dep.circuit_state = CircuitState.AUTH_FAILED
                key_to_hash = failed_key or dep.api_key or ""
                if key_to_hash:
                    dep.auth_failed_key_hash = hashlib.sha256(key_to_hash.encode("utf-8")).hexdigest()
                dep.state_reason = sanitize_secret(reason, dep.api_key)
                self.state_manager.record_quarantine(
                    dep.id,
                    dep.state_reason,
                    state=CircuitState.AUTH_FAILED,
                )
                self.state_manager.record_circuit_state(
                    dep.id,
                    CircuitState.AUTH_FAILED,
                    reason=dep.state_reason,
                )

    def _probe_deployment_credentials(self, deployment: Deployment) -> bool:
        """Probe deployment credentials with a minimal lightweight request."""
        if not deployment.api_key or not deployment.api_key.strip():
            return False

        try:
            litellm.completion(
                model=deployment.litellm_model,
                messages=[{"role": "user", "content": "ping"}],
                api_key=deployment.api_key,
                api_base=deployment.api_base,
                max_tokens=1,
                timeout=10,
            )
            return True
        except Exception as e:
            info = parse_provider_error(e, provider=deployment.provider)
            if info.category == CATEGORY_AUTH_ERROR:
                return False
            # If provider accepted auth but returned 429, 400, or 404, credentials themselves are valid
            if info.status_code and info.status_code in (429, 400, 404):
                return True
            return False

    def revalidate_deployment(
        self,
        deployment_id: str,
        force: bool = False,
        probe_fn: Optional[Callable[[Deployment], bool]] = None,
    ) -> bool:
        """Attempt to revalidate a quarantined deployment.

        Enforces rate-limiting backoff unless force=True to prevent hammering providers.
        If credentials validate successfully, lifts quarantine and returns deployment to HEALTHY.
        """
        dep = next((d for d in self.deployments if d.id == deployment_id or d.model == deployment_id), None)
        if not dep:
            return False

        # If not quarantined, it's already eligible
        if not dep.is_quarantined:
            return True

        now = time.time()
        if not force and dep.last_auth_probe_at > 0:
            elapsed = now - dep.last_auth_probe_at
            if elapsed < dep.auth_probe_backoff:
                # Suppressed by backoff to prevent continuous hammering
                return False

        dep.last_auth_probe_at = now

        probe = probe_fn if probe_fn is not None else self._probe_deployment_credentials
        success = probe(dep)

        if success:
            dep.circuit_state = CircuitState.HEALTHY
            dep.state_reason = "Credentials revalidated; returned to service"
            dep.auth_failed_key_hash = None
            dep.auth_probe_backoff = 300.0
            dep.consecutive_failures = 0
            self.state_manager.record_success(dep.id)
            self.state_manager.record_circuit_state(dep.id, CircuitState.HEALTHY, reason=dep.state_reason)
            return True
        else:
            dep.auth_probe_backoff = min(3600.0, dep.auth_probe_backoff * 2.0)
            dep.state_reason = "Revalidation probe failed: credentials invalid"
            self.state_manager.record_circuit_state(dep.id, CircuitState.AUTH_FAILED, reason=dep.state_reason)
            return False

    def revalidate_account(
        self,
        provider: str,
        account: str,
        force: bool = False,
        probe_fn: Optional[Callable[[Deployment], bool]] = None,
    ) -> Dict[str, bool]:
        """Attempt to revalidate all deployments belonging to an account.

        Probes the account once to avoid hammering, then applies result to all sibling deployments.
        """
        account_deps = [d for d in self.deployments if d.provider == provider and d.account == account]
        if not account_deps:
            return {}

        probe_target = next((d for d in account_deps if d.is_quarantined), account_deps[0])
        success = self.revalidate_deployment(probe_target.id, force=force, probe_fn=probe_fn)

        results: Dict[str, bool] = {}
        for dep in account_deps:
            if success:
                dep.circuit_state = CircuitState.HEALTHY
                dep.state_reason = "Account credentials revalidated; returned to service"
                dep.auth_failed_key_hash = None
                dep.auth_probe_backoff = 300.0
                dep.consecutive_failures = 0
                self.state_manager.record_success(dep.id)
                self.state_manager.record_circuit_state(dep.id, CircuitState.HEALTHY, reason=dep.state_reason)
                results[dep.id] = True
            else:
                results[dep.id] = False

        if success:
            try:
                self.rediscover_models(provider=provider, account=account, force=True)
            except Exception:
                pass

        return results

    def rediscover_models(
        self,
        provider: Optional[str] = None,
        account: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Retry discovery through existing application lifecycle for stale or uninitialized accounts (Rules 3, 5).

        - Re-discovers models using authoritative provider APIs.
        - On success, replaces stale information with new authoritative result (Rule 5).
        - On temporary failure, preserves existing deployments marked stale (Rules 1, 4).
        - Preserves account isolation (Rule 9).
        """
        results: Dict[str, Any] = {"success": {}, "failed": {}}

        # 1. Groq rediscovery
        if provider is None or provider == MODEL_GROUP_GROQ:
            groq_accounts = {
                dep.account: dep.api_key
                for dep in self.deployments
                if dep.provider == MODEL_GROUP_GROQ and dep.api_key
            }
            from ntg.config import GROQ_KEYS
            for acc, key in GROQ_KEYS.items():
                if key and acc not in groq_accounts:
                    groq_accounts[acc] = key

            for acc_name, key in groq_accounts.items():
                if account and acc_name != account:
                    continue

                cache_key = f"groq_{acc_name}"
                is_stale = self.state_manager.is_discovery_stale(cache_key)
                if not force and not is_stale:
                    continue

                try:
                    from ntg.discovery import discover_groq_models
                    discovered = discover_groq_models(key, account_name=acc_name, state_manager=self.state_manager)
                    if discovered and not self.state_manager.is_discovery_stale(cache_key):
                        # Successful discovery! Clear stale flag on existing deployments
                        for dep in self.deployments:
                            if dep.provider == MODEL_GROUP_GROQ and dep.account == acc_name:
                                dep.is_stale = False
                                dep.stale_reason = None
                        results["success"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = len(discovered)
                    else:
                        results["failed"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = "Discovery failed or remained stale"
                except Exception as err:
                    results["failed"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = sanitize_secret(str(err), key)

        # 2. Gemini rediscovery
        if provider is None or provider == MODEL_GROUP_GEMINI:
            gemini_accounts = {
                dep.account: dep.api_key
                for dep in self.deployments
                if dep.provider == MODEL_GROUP_GEMINI and dep.api_key
            }
            from ntg.config import GEMINI_KEYS
            for acc, key in GEMINI_KEYS.items():
                if key and key != "YOUR_NEW_GOOGLE_API_KEY" and acc not in gemini_accounts:
                    gemini_accounts[acc] = key

            for acc_name, key in gemini_accounts.items():
                if account and acc_name != account:
                    continue

                cache_key = f"gemini_{acc_name}"
                is_stale = self.state_manager.is_discovery_stale(cache_key)
                if not force and not is_stale:
                    continue

                try:
                    from gemini import discover_models as gemini_discover
                    discovered = gemini_discover(key, account_name=acc_name, state_manager=self.state_manager)
                    if discovered and not self.state_manager.is_discovery_stale(cache_key):
                        for dep in self.deployments:
                            if dep.provider == MODEL_GROUP_GEMINI and dep.account == acc_name:
                                dep.is_stale = False
                                dep.stale_reason = None
                        results["success"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = len(discovered)
                    else:
                        results["failed"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = "Discovery failed or remained stale"
                except Exception as err:
                    results["failed"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = sanitize_secret(str(err), key)

        return results

    def reset_auth_quarantine(
        self,
        provider: Optional[str] = None,
        account: Optional[str] = None,
    ) -> int:
        """Administratively reset authentication quarantine for specified provider/account or all deployments.

        Returns the number of deployments restored.
        """
        reset_count = 0
        for dep in self.deployments:
            if provider and dep.provider != provider:
                continue
            if account and dep.account != account:
                continue
            if dep.is_quarantined:
                dep.circuit_state = CircuitState.HEALTHY
                dep.state_reason = "Authentication quarantine reset administratively"
                dep.auth_failed_key_hash = None
                dep.auth_probe_backoff = 300.0
                dep.consecutive_failures = 0
                self.state_manager.record_success(dep.id)
                self.state_manager.record_circuit_state(dep.id, CircuitState.HEALTHY, reason=dep.state_reason)
                reset_count += 1
        return reset_count

    def _propagate_account_quota_exhaustion(
        self,
        provider: str,
        account: str,
        reset_at: float,
        reason: str,
    ) -> None:
        """Propagate an account-wide quota exhaustion across all deployments under the same account.

        Updates NTG quota knowledge, circuit state, and persistent state so that
        the exhausted account's deployments are excluded from future candidate pools until reset.
        LiteLLM manages its own routing-level cooldowns without internal cache tampering.
        """
        for dep in self.deployments:
            if dep.provider == provider and dep.account == account:
                dep.circuit_state = CircuitState.OPEN
                dep.quota.reset_at = reset_at
                dep.state_reason = reason
                self.state_manager.record_cooldown(dep.id, reset_at, "account_quota")
                self.state_manager.record_circuit_state(dep.id, CircuitState.OPEN)

    def _propagate_account_cooldown(
        self,
        provider: str,
        account: str,
        cooldown_until: float,
        reason: str,
    ) -> None:
        """Backward-compatibility alias for _propagate_account_quota_exhaustion."""
        self._propagate_account_quota_exhaustion(provider, account, cooldown_until, reason)

    def get_global_pool(self, now: Optional[float] = None) -> List[Deployment]:
        """Return all active, eligible deployments across all configured providers.

        This forms the authoritative base global deployment pool for 'auto' / 'ntg-auto'.
        Every configured provider (OpenRouter, Groq, NVIDIA, Cohere, Gemini) with valid
        credentials and a healthy/recovering circuit is included.
        """
        ts = now if now is not None else time.time()
        pool: List[Deployment] = []
        for dep in self.deployments:
            dep.refresh(ts)
            if not dep.api_key or not dep.api_key.strip():
                continue
            if dep.is_quarantined:
                continue
            if not dep.is_available(ts):
                continue
            pool.append(dep)
        return pool

    def get_eligible_deployments(
        self,
        model_group: str = DEFAULT_LOGICAL_MODEL,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        now: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Deployment]:
        """Public helper to inspect eligible deployments for any model group or requirements."""
        reqs = extract_request_requirements(
            model=model_group,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )
        return self._filter_eligible_deployments(
            model_group=model_group,
            requirements=reqs,
            now=now,
        )

    def _filter_eligible_deployments(
        self,
        model_group: str,
        requirements: Optional[RequestRequirements] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[Deployment]:
        """NTG Eligibility Filter: evaluate health, capabilities, quotas, and circuit state."""
        ts = now if now is not None else time.time()
        for dep in self.deployments:
            dep.refresh(ts)

        is_global = model_group in (MODEL_GROUP_AUTO, MODEL_GROUP_NTG_AUTO, "auto", "ntg-auto")

        candidates = self.get_global_pool(ts) if is_global else [
            d
            for d in self.deployments
            if d.api_key and d.api_key.strip() and not d.is_quarantined and d.is_available(ts)
        ]

        # Non-global requests filter to matching logical model
        if not is_global and model_group not in (
            "coding",
            "reasoning",
            "vision",
            "tool_calling",
            "structured_output",
            "streaming",
        ):
            # 1. Match primary requested provider or underlying model
            primary_candidates = [
                d for d in candidates
                if d.logical_model == model_group
                or d.model == model_group
                or d.id == model_group
                or d.litellm_model == model_group
            ]

            # 2. Check if explicit compatible fallback providers are configured (Rule 5)
            # Provider-specific requests must not jump to unrelated providers unless explicitly configured
            compatible_provider_names = self.provider_fallbacks.get(model_group, [])
            if compatible_provider_names:
                fallback_candidates = [
                    d for d in candidates
                    if d.logical_model in compatible_provider_names and d not in primary_candidates
                ]
                candidates = primary_candidates + fallback_candidates
            else:
                candidates = primary_candidates

        # Normalize requirements
        reqs = requirements
        if reqs is None:
            parsed_caps = self._parse_capabilities(capabilities) or ModelCapabilities()
            if min_context and min_context > parsed_caps.context_window:
                parsed_caps.context_window = min_context
            reqs = RequestRequirements(capabilities=parsed_caps)

        eligible: List[Deployment] = []
        for dep in candidates:
            # Check capability groups if model_group was a capability group name
            if model_group == "coding" and not dep.capabilities.coding:
                continue
            if model_group == "reasoning" and not dep.capabilities.reasoning:
                continue
            if model_group == "vision" and not dep.capabilities.vision:
                continue
            if model_group == "tool_calling" and not dep.capabilities.tool_calling:
                continue
            if model_group == "structured_output" and not dep.capabilities.structured_output:
                continue
            if model_group == "streaming" and not dep.capabilities.streaming:
                continue

            # Strict capability filter: unknown (None) or unsupported (False) fails strict requirement
            if not dep.capabilities.satisfies(reqs.capabilities):
                continue

            eligible.append(dep)

        # Rule 5: Coding-oriented requests prefer deployments marked coding=True
        if reqs.coding_preferred and not reqs.capabilities.coding:
            coding_matches = [d for d in eligible if d.capabilities.coding is True]
            if coding_matches:
                eligible = coding_matches

        return eligible

    def _rank_eligible_deployments(
        self,
        eligible: List[Deployment],
        primary_group: Optional[str] = None,
    ) -> List[Deployment]:
        """Rank and shuffle eligible deployments for LiteLLM load balancing and failover."""
        if len(eligible) <= 1:
            return list(eligible)

        import random

        def _shuffle_group(group: List[Deployment]) -> List[Deployment]:
            healthy = [d for d in group if d.circuit_state == CircuitState.HEALTHY]
            probes = [d for d in group if d.circuit_state == CircuitState.HALF_OPEN]

            shuffled: List[Deployment] = []
            pool = list(healthy) if healthy else list(group)
            while pool:
                weights = [max(0.1, d.weight) for d in pool]
                chosen = random.choices(pool, weights=weights, k=1)[0]
                shuffled.append(chosen)
                pool.remove(chosen)

            if healthy and probes:
                probe_pool = list(probes)
                while probe_pool:
                    p_weights = [max(0.1, d.weight) for d in probe_pool]
                    chosen = random.choices(probe_pool, weights=p_weights, k=1)[0]
                    shuffled.append(chosen)
                    probe_pool.remove(chosen)
            return shuffled

        if primary_group and primary_group not in (MODEL_GROUP_AUTO, MODEL_GROUP_NTG_AUTO, "auto", "ntg-auto"):
            primaries = [
                d for d in eligible
                if d.logical_model == primary_group
                or d.model == primary_group
                or d.id == primary_group
                or d.litellm_model == primary_group
            ]
            secondaries = [d for d in eligible if d not in primaries]
            if primaries and secondaries:
                return _shuffle_group(primaries) + _shuffle_group(secondaries)

        return _shuffle_group(eligible)

    def _build_compatible_fallbacks(
        self,
        ordered_eligible: List[Deployment],
    ) -> Optional[List[Dict[str, List[str]]]]:
        """Construct compatibility-aware fallback chain for LiteLLM Router.

        Ensures that every deployment in the fallback chain has already been verified
        to satisfy the request's exact capability requirements and health/circuit state.
        Each deployment points to all subsequent eligible deployments in order.
        """
        if len(ordered_eligible) <= 1:
            return None

        fallbacks: List[Dict[str, List[str]]] = []
        for i in range(len(ordered_eligible) - 1):
            cur_id = ordered_eligible[i].id
            rem_ids = [d.id for d in ordered_eligible[i + 1:]]
            fallbacks.append({cur_id: rem_ids})
        return fallbacks

    def _validate_user_fallbacks(
        self,
        user_fallbacks: Any,
        requirements: RequestRequirements,
        now: Optional[float] = None,
    ) -> List[Dict[str, List[str]]]:
        """Validate caller-supplied fallbacks to ensure compatibility and eligibility.

        Ensures caller-supplied fallbacks do not bypass:
        - authentication quarantine (CircuitState.QUARANTINED)
        - quota exhaustion or cooldown timers
        - request capability requirements.
        """
        ts = now if now is not None else time.time()
        dep_by_id = {d.id: d for d in self.deployments}
        dep_by_group: Dict[str, List[Deployment]] = {}
        for d in self.deployments:
            dep_by_group.setdefault(d.logical_model, []).append(d)
            dep_by_group.setdefault(d.model, []).append(d)

        validated: List[Dict[str, List[str]]] = []
        if isinstance(user_fallbacks, list):
            for item in user_fallbacks:
                if not isinstance(item, dict):
                    continue
                for src, dst_list in item.items():
                    if not isinstance(dst_list, list):
                        continue
                    clean_dst: List[str] = []
                    for dst in dst_list:
                        candidates: List[Deployment] = []
                        if dst in dep_by_id:
                            candidates = [dep_by_id[dst]]
                        elif dst in dep_by_group:
                            candidates = dep_by_group[dst]

                        for d in candidates:
                            if (
                                d.api_key
                                and d.api_key.strip()
                                and not d.is_quarantined
                                and d.is_available(ts)
                                and d.capabilities.satisfies(requirements.capabilities)
                            ):
                                if d.id not in clean_dst:
                                    clean_dst.append(d.id)

                        if not candidates:
                            clean_dst.append(dst)

                    if clean_dst:
                        validated.append({src: clean_dst})
        return validated

    @staticmethod
    def _parse_capabilities(
        caps: Optional[ModelCapabilities | dict | list[str] | str] = None,
    ) -> Optional[ModelCapabilities]:
        """Normalize capability parameter into a ModelCapabilities object."""
        if caps is None:
            return None
        if isinstance(caps, ModelCapabilities):
            return caps
        if isinstance(caps, dict):
            return ModelCapabilities(**caps)
        if isinstance(caps, list):
            valid_fields = {
                "coding",
                "reasoning",
                "vision",
                "tool_calling",
                "structured_output",
                "streaming",
            }
            return ModelCapabilities(**{c: True for c in caps if c in valid_fields})
        if isinstance(caps, str):
            c_clean = caps.strip().lower()
            if c_clean in ("coding", "code"):
                return ModelCapabilities(coding=True)
            if c_clean in ("reasoning", "reason"):
                return ModelCapabilities(reasoning=True)
            if c_clean in ("vision", "image"):
                return ModelCapabilities(vision=True)
            if c_clean in ("tool_calling", "tools", "tool"):
                return ModelCapabilities(tool_calling=True)
            if c_clean in ("structured_output", "structured", "json"):
                return ModelCapabilities(structured_output=True)
            if c_clean in ("streaming", "stream"):
                return ModelCapabilities(streaming=True)
        return None

    def _get_response_deployment(
        self,
        response: Any,
        eligible: Optional[List[Deployment]] = None,
    ) -> Optional[Deployment]:
        """Extract fulfilling deployment directly from response metadata or candidate pool.

        Guarantees 100% concurrency safety with zero shared mutable state.
        """
        if response is None:
            return None

        # 1. Direct attachment from telemetry event callback on this specific response instance
        dep = getattr(response, "_ntg_deployment", None)
        if dep is not None and isinstance(dep, Deployment):
            return dep

        # 2. Check response model_info or hidden metadata if present
        model_info = getattr(response, "model_info", None)
        if isinstance(model_info, dict):
            dep_id = model_info.get("id")
            if dep_id and dep_id in self.deployment_map:
                dep = self.deployment_map[dep_id]
                try:
                    setattr(response, "_ntg_deployment", dep)
                except Exception:
                    pass
                return dep

        hidden = getattr(response, "_hidden_params", {}) or {}
        if isinstance(hidden, dict):
            dep_id = hidden.get("model_info", {}).get("id") or hidden.get("model_id")
            if dep_id and dep_id in self.deployment_map:
                dep = self.deployment_map[dep_id]
                try:
                    setattr(response, "_ntg_deployment", dep)
                except Exception:
                    pass
                return dep

        # 3. Model attribute matching from candidate pool
        resp_model = getattr(response, "model", "")
        candidates = eligible if eligible is not None else self.deployments
        if resp_model and candidates:
            for d in candidates:
                if d.model in resp_model or d.litellm_model in resp_model:
                    try:
                        setattr(response, "_ntg_deployment", d)
                    except Exception:
                        pass
                    return d

        fallback = candidates[0] if (candidates and len(candidates) > 0) else None
        if fallback:
            try:
                setattr(response, "_ntg_deployment", fallback)
            except Exception:
                pass
        return fallback

    def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Route user prompt through LiteLLM Router with NTG capability & eligibility validation."""
        target_model = model or self.default_model
        if target_model in ("ntg-auto", MODEL_GROUP_NTG_AUTO):
            target_model = MODEL_GROUP_AUTO

        # Extract deterministic request requirements (Rules 1-6)
        reqs = extract_request_requirements(
            prompt=prompt,
            model=target_model,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )

        # Enforce bounded retry budget (never derived from deployment count; capped at MAX_RETRY_BUDGET)
        req_retries = kwargs.pop("num_retries", None)
        effective_retries = (
            min(max(0, req_retries), MAX_RETRY_BUDGET)
            if req_retries is not None
            else self.num_retries
        )
        kwargs["num_retries"] = effective_retries

        # Run NTG eligibility filter (Rules 7, 9, 10)
        eligible = self._filter_eligible_deployments(
            model_group=target_model,
            requirements=reqs,
        )

        now = time.time()
        # Rule 8: If no deployment satisfies strict requirements, return clear eligibility error
        if not eligible:
            print_divider("NO ELIGIBLE DEPLOYMENTS")
            print(f"No available deployments satisfy request for model group '{target_model}'")
            print(f"Required capabilities: {reqs.describe()}")
            if min_context:
                print(f"Required context     : >={min_context:,}")

            resets = [dep.remaining_cooldown for dep in self.deployments if dep.remaining_cooldown > 0]
            if resets:
                next_capacity = min(resets)
                print(f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.deployments)
            return None

        # Pass eligible pool to LiteLLM Router with compatibility-aware fallbacks
        ordered_eligible = self._rank_eligible_deployments(eligible, primary_group=target_model)
        primary = ordered_eligible[0]

        caller_fallbacks = kwargs.pop("fallbacks", None)
        if caller_fallbacks:
            fallbacks = self._validate_user_fallbacks(caller_fallbacks, reqs, now)
        else:
            fallbacks = self._build_compatible_fallbacks(ordered_eligible)

        call_kwargs = dict(kwargs)
        if fallbacks:
            call_kwargs["fallbacks"] = fallbacks

        try:
            # Delegate routing, load-balancing, and failover directly to LiteLLM Router
            response = self.router.completion(
                model=primary.id,
                messages=[{"role": "user", "content": prompt}],
                **call_kwargs,
            )

            # Retrieve fulfilling deployment directly from response instance (100% concurrency-safe)
            deployment = self._get_response_deployment(response, eligible)

            if deployment:
                deployment.record_user_request()
                self.state_manager.record_metrics(deployment.id, deployment.metrics)

            if hasattr(response, "choices") and response.choices:
                print("\n" + str(response.choices[0].message.content))

            if deployment:
                print_request_execution(deployment, getattr(response, "model", None))

            return response

        except Exception as error:
            if primary:
                primary.record_user_request()
                self.state_manager.record_metrics(primary.id, primary.metrics)

            print_divider("ALL ELIGIBLE DEPLOYMENTS EXHAUSTED")
            print(f"Request failed across healthy deployments for model group '{target_model}'.")
            print(f"Final error: {error}")
            print_account_status(self.deployments)
            return None

    def completion(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion conforming to standard OpenAI/LiteLLM signature."""
        target_model = model or self.default_model
        if target_model in ("ntg-auto", MODEL_GROUP_NTG_AUTO):
            target_model = MODEL_GROUP_AUTO

        # Extract deterministic request requirements (Rules 1-6)
        reqs = extract_request_requirements(
            messages=messages,
            model=target_model,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )

        # Enforce bounded retry budget (never derived from deployment count; capped at MAX_RETRY_BUDGET)
        req_retries = kwargs.pop("num_retries", None)
        effective_retries = (
            min(max(0, req_retries), MAX_RETRY_BUDGET)
            if req_retries is not None
            else self.num_retries
        )
        kwargs["num_retries"] = effective_retries

        # Run NTG eligibility filter (Rules 7, 9, 10)
        eligible = self._filter_eligible_deployments(
            model_group=target_model,
            requirements=reqs,
        )

        # Rule 8: If no deployment satisfies strict requirements, return clear eligibility error
        if not eligible:
            raise NoEligibleDeploymentsError(
                f"No eligible deployments for model='{target_model}' satisfying requirements: "
                f"{reqs.describe()}."
            )

        now = time.time()
        # Pass eligible pool to LiteLLM Router with compatibility-aware fallbacks
        ordered_eligible = self._rank_eligible_deployments(eligible, primary_group=target_model)
        primary = ordered_eligible[0]

        caller_fallbacks = kwargs.pop("fallbacks", None)
        if caller_fallbacks:
            fallbacks = self._validate_user_fallbacks(caller_fallbacks, reqs, now)
        else:
            fallbacks = self._build_compatible_fallbacks(ordered_eligible)

        call_kwargs = dict(kwargs)
        if fallbacks:
            call_kwargs["fallbacks"] = fallbacks

        try:
            response = self.router.completion(
                model=primary.id,
                messages=messages,
                **call_kwargs,
            )

            deployment = self._get_response_deployment(response, eligible)
            if deployment:
                deployment.record_user_request()
                self.state_manager.record_metrics(deployment.id, deployment.metrics)

            return response
        except Exception:
            if primary:
                primary.record_user_request()
                self.state_manager.record_metrics(primary.id, primary.metrics)
            raise


# Maintain backwards compatibility
OpenRouterFallbackRouter = UnifiedNTGRouter