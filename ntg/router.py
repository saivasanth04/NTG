"""LiteLLM-based multi-provider smart router with NTG intelligence."""

from __future__ import annotations

import contextvars
import logging
import time
from typing import Any, Dict, List, Optional, Set

import litellm
from litellm import RetryPolicy, Router
from litellm.integrations.custom_logger import CustomLogger

from ntg.config import (
    DEFAULT_LOGICAL_MODEL,
    MODEL_GROUP_AUTO,
    MODEL_GROUP_COHERE,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NTG_AUTO,
    MODEL_GROUP_NVIDIA,
    MODEL_GROUP_OPENROUTER,
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
    parse_provider_error,
)
from ntg.models import CircuitState, Deployment, ModelCapabilities, utc_string
from ntg.state import StateManager

# ContextVar for thread-safe, concurrency-safe request deployment tracking
_current_request_deployment: contextvars.ContextVar[Optional[Deployment]] = contextvars.ContextVar(
    "_current_request_deployment", default=None
)


class NTGTelemetryLogger(CustomLogger):
    """LiteLLM custom logger callback for telemetry, circuit breaker updates, and quota intelligence.

    Extracts per-attempt deployment metadata directly from event kwargs to ensure
    100% thread safety and eliminate shared mutable request state.
    """

    def __init__(self, router: UnifiedNTGRouter):
        super().__init__()
        self.router = router

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Handle upstream success callback."""
        dep_id = kwargs.get("litellm_params", {}).get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kwargs.get("model_info", {}).get("id")

        deployment = self.router.deployment_map.get(dep_id)
        if deployment:
            deployment.metrics.upstream_attempts += 1
            deployment.metrics.successes += 1
            deployment.record_success()

            # Record healthy circuit state and update persisted state
            self.router.state_manager.record_circuit_state(dep_id, CircuitState.HEALTHY)
            self.router.state_manager.record_metrics(dep_id, deployment.metrics)

            # Store in thread-safe contextvar for the current request
            _current_request_deployment.set(deployment)

    def log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        """Handle upstream failure callback with granular error classification."""
        dep_id = kwargs.get("litellm_params", {}).get("model_info", {}).get("id")
        if not dep_id:
            dep_id = kwargs.get("model_info", {}).get("id")

        exc = kwargs.get("exception")
        if not exc and isinstance(response_obj, Exception):
            exc = response_obj

        deployment = self.router.deployment_map.get(dep_id)
        if deployment:
            deployment.metrics.upstream_attempts += 1
            deployment.metrics.failures += 1

            if exc:
                info = parse_provider_error(exc, provider=deployment.provider)

                # Update quota information
                deployment.quota.limit_type = info.limit_type
                deployment.quota.quota_scope = info.quota_scope
                deployment.quota.reset_at = info.reset_timestamp
                deployment.quota.retry_after = info.cooldown_seconds

                # Error categorization
                if info.category in (CATEGORY_PROVIDER_LIMIT, CATEGORY_DAILY_QUOTA):
                    deployment.metrics.rate_limits += 1
                elif info.category == CATEGORY_AUTH_ERROR:
                    deployment.metrics.auth_errors += 1

                # Circuit breaker & quarantine handling
                if info.should_quarantine:
                    deployment.circuit_state = CircuitState.QUARANTINED
                    deployment.blocked_reason = f"Authentication / Model error: {info.message}"
                    self.router.state_manager.record_circuit_state(dep_id, CircuitState.QUARANTINED)
                else:
                    deployment.record_failure()
                    deployment.circuit_state = CircuitState.OPEN
                    deployment.cooldown_until = time.time() + info.cooldown_seconds
                    deployment.blocked_reason = info.message

                    self.router.state_manager.record_cooldown(
                        dep_id, deployment.cooldown_until, info.limit_type
                    )
                    self.router.state_manager.record_circuit_state(dep_id, CircuitState.OPEN)

                    # Add to LiteLLM router's cooldown cache
                    try:
                        status_code = getattr(exc, "status_code", 500) or 500
                        self.router.router.cooldown_cache.add_deployment_to_cooldown(
                            model_id=dep_id,
                            original_exception=exc,
                            exception_status=status_code,
                            cooldown_time=info.cooldown_seconds,
                        )
                    except Exception:
                        pass

                # If quota scope is account-level, propagate cooldown to all deployments of that account
                if info.quota_scope == "account":
                    self.router._propagate_account_cooldown(
                        deployment.provider,
                        deployment.account,
                        deployment.cooldown_until,
                        info.message,
                    )

                self.router.state_manager.record_metrics(dep_id, deployment.metrics)


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

        # 1. Build all deployments across providers and restore persisted state
        if accounts is not None:
            self.deployments: List[Deployment] = list(accounts)
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
            # 3. Capability groups (coding, reasoning, vision, tool_calling)
            groups = [dep.logical_model, MODEL_GROUP_AUTO]

            if dep.capabilities.coding:
                groups.append("coding")
            if dep.capabilities.reasoning:
                groups.append("reasoning")
            if dep.capabilities.vision:
                groups.append("vision")
            if dep.capabilities.tool_calling:
                groups.append("tool_calling")

            for g in groups:
                pair = (g, dep.id)
                if pair not in registered_pairs:
                    registered_pairs.add(pair)
                    entry = dict(litellm_dict)
                    entry["model_name"] = g
                    model_list.append(entry)

        # 3. Configure compatible capability-aware fallbacks
        fallbacks: List[Dict[str, List[str]]] = [
            {MODEL_GROUP_OPENROUTER: [MODEL_GROUP_NVIDIA, MODEL_GROUP_COHERE, MODEL_GROUP_GROQ]},
            {MODEL_GROUP_NVIDIA: [MODEL_GROUP_OPENROUTER, MODEL_GROUP_COHERE, MODEL_GROUP_GROQ]},
            {MODEL_GROUP_COHERE: [MODEL_GROUP_NVIDIA, MODEL_GROUP_OPENROUTER, MODEL_GROUP_GROQ]},
            {MODEL_GROUP_GROQ: [MODEL_GROUP_NVIDIA, MODEL_GROUP_OPENROUTER, MODEL_GROUP_COHERE]},
            {MODEL_GROUP_GEMINI: [MODEL_GROUP_GROQ]},
            {"coding": [MODEL_GROUP_AUTO]},
            {"reasoning": [MODEL_GROUP_AUTO]},
            {"vision": [MODEL_GROUP_GEMINI]},
            {"tool_calling": [MODEL_GROUP_NVIDIA, MODEL_GROUP_COHERE, MODEL_GROUP_GROQ]},
        ]

        # 4. Resolve default model
        self.default_model = default_model or DEFAULT_LOGICAL_MODEL

        # 5. Configure RetryPolicy with bounded request retry budget (max 2 retries)
        retry_policy = RetryPolicy(
            BadRequestErrorRetries=0,
            AuthenticationErrorRetries=1,
            RateLimitErrorRetries=1,
            TimeoutErrorRetries=1,
            InternalServerErrorRetries=1,
        )

        # 6. Instantiate LiteLLM Router
        self.router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            cooldown_time=60.0,
            allowed_fails=1,
            num_retries=2,
            retry_policy=retry_policy,
            fallbacks=fallbacks,
            model_group_alias={
                MODEL_GROUP_NTG_AUTO: MODEL_GROUP_AUTO,
            },
        )

        # 7. Register telemetry and circuit breaker callback
        self.telemetry_logger = NTGTelemetryLogger(self)
        if self.telemetry_logger not in litellm.callbacks:
            litellm.callbacks.append(self.telemetry_logger)

    def _propagate_account_cooldown(
        self,
        provider: str,
        account: str,
        cooldown_until: float,
        reason: str,
    ) -> None:
        """Propagate an account-wide quota limit across all deployments under the same account."""
        now = time.time()
        cooldown_secs = max(1.0, cooldown_until - now)
        for dep in self.deployments:
            if dep.provider == provider and dep.account == account:
                dep.circuit_state = CircuitState.OPEN
                dep.cooldown_until = cooldown_until
                dep.blocked_reason = reason
                self.state_manager.record_cooldown(dep.id, cooldown_until, "account_quota")
                self.state_manager.record_circuit_state(dep.id, CircuitState.OPEN)
                try:
                    self.router.cooldown_cache.add_deployment_to_cooldown(
                        model_id=dep.id,
                        original_exception=Exception(reason),
                        exception_status=429,
                        cooldown_time=cooldown_secs,
                    )
                except Exception:
                    pass

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
            if dep.circuit_state == CircuitState.QUARANTINED:
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
    ) -> List[Deployment]:
        """Public helper to inspect eligible deployments for any model group or capabilities."""
        req_caps = self._parse_capabilities(capabilities)
        return self._filter_eligible_deployments(
            model_group=model_group,
            capabilities=req_caps,
            min_context=min_context,
            now=now,
        )

    def _filter_eligible_deployments(
        self,
        model_group: str,
        capabilities: Optional[ModelCapabilities] = None,
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
            if d.api_key and d.api_key.strip() and d.circuit_state != CircuitState.QUARANTINED and d.is_available(ts)
        ]

        eligible: List[Deployment] = []
        for dep in candidates:
            # Check capability groups
            if model_group == "coding" and not dep.capabilities.coding:
                continue
            if model_group == "reasoning" and not dep.capabilities.reasoning:
                continue
            if model_group == "vision" and not dep.capabilities.vision:
                continue
            if model_group == "tool_calling" and not dep.capabilities.tool_calling:
                continue

            # Check provider group match for non-global requests
            if not is_global and model_group not in ("coding", "reasoning", "vision", "tool_calling"):
                if dep.logical_model != model_group:
                    continue

            # Check capabilities
            if capabilities and not dep.capabilities.satisfies(capabilities):
                continue

            # Check min context window
            if min_context and dep.capabilities.context_window < min_context:
                continue

            eligible.append(dep)

        return eligible

    def _parse_capabilities(
        self,
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
            return ModelCapabilities(**{c: True for c in caps})
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
            if c_clean in ("streaming", "stream"):
                return ModelCapabilities(streaming=True)
        return None

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
        req_caps = self._parse_capabilities(capabilities)

        # Run NTG eligibility filter
        eligible = self._filter_eligible_deployments(
            model_group=target_model,
            capabilities=req_caps,
            min_context=min_context,
        )

        now = time.time()
        if not eligible:
            print_divider("NO ELIGIBLE DEPLOYMENTS")
            print(f"No available deployments satisfy model group '{target_model}'")
            if req_caps:
                print(f"Required capabilities: {', '.join(req_caps.to_tags())}")
            if min_context:
                print(f"Required context     : >={min_context:,}")

            resets = [dep.remaining_cooldown for dep in self.deployments if dep.remaining_cooldown > 0]
            if resets:
                next_capacity = min(resets)
                print(f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})")
            print_account_status(self.deployments)
            return None

        # Reset contextvar for this request
        _current_request_deployment.set(None)

        try:
            # Delegate routing, load-balancing, and failover directly to LiteLLM Router
            response = self.router.completion(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            # Retrieve the fulfilling deployment from the thread-safe context
            deployment = _current_request_deployment.get()
            if not deployment:
                # Fallback to response model matching
                resp_model = getattr(response, "model", "")
                for dep in eligible:
                    if dep.model in resp_model or dep.litellm_model in resp_model:
                        deployment = dep
                        break
                if not deployment and eligible:
                    deployment = eligible[0]

            if deployment:
                deployment.metrics.user_requests += 1

            if hasattr(response, "choices") and response.choices:
                print("\n" + str(response.choices[0].message.content))

            if deployment:
                print_request_execution(deployment, getattr(response, "model", None))

            return response

        except Exception as error:
            print_divider("ALL ELIGIBLE DEPLOYMENTS EXHAUSTED")
            print(f"Request failed across healthy deployments for model group '{target_model}'.")
            print(f"Final error: {error}")
            print_account_status(self.deployments)
            return None

    def completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion conforming to standard OpenAI/LiteLLM signature."""
        target_model = model or self.default_model
        if target_model in ("ntg-auto", MODEL_GROUP_NTG_AUTO):
            target_model = MODEL_GROUP_AUTO
        req_caps = self._parse_capabilities(capabilities)

        eligible = self._filter_eligible_deployments(
            model_group=target_model,
            capabilities=req_caps,
            min_context=min_context,
        )

        if not eligible:
            raise RuntimeError(
                f"No eligible deployments for model='{target_model}' satisfying requirements."
            )

        _current_request_deployment.set(None)
        response = self.router.completion(
            model=target_model,
            messages=messages,
            **kwargs,
        )

        deployment = _current_request_deployment.get()
        if deployment:
            deployment.metrics.user_requests += 1

        return response


# Maintain backwards compatibility
OpenRouterFallbackRouter = UnifiedNTGRouter