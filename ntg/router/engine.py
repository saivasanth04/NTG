"""LiteLLM-based multi-provider smart routing engine with NTG intelligence."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

import litellm
from litellm import RetryPolicy, Router
from litellm.router_strategy.simple_shuffle import simple_shuffle

from ntg.cli.diagnostics import (
    print_account_status,
    print_divider,
    print_request_execution,
)
from ntg.core.config import (
    DEFAULT_LOGICAL_MODEL,
    DEFAULT_NUM_RETRIES,
    GEMINI_KEYS,
    GROQ_KEYS,
    MAX_RETRY_BUDGET,
    MODEL_GROUP_AUTO,
    MODEL_GROUP_GEMINI,
    MODEL_GROUP_GROQ,
    MODEL_GROUP_NTG_AUTO,
    PROVIDER_FALLBACKS,
)
from ntg.core.exceptions import (
    CATEGORY_AUTH_ERROR,
    parse_provider_error,
)
from ntg.core.models import (
    CircuitState,
    Deployment,
    DeploymentMetrics,
    ModelCapabilities,
    QuotaScope,
)
from ntg.core.utils import sanitize_secret, utc_string
from ntg.providers.discovery import build_all_deployments
from ntg.providers.gemini import discover_gemini_models
from ntg.providers.groq import discover_groq_models
from ntg.router.requirements import (
    NoEligibleDeploymentsError,
    RequestRequirements,
    extract_request_requirements,
    parse_capabilities,
)
from ntg.router.state import StateManager
from ntg.router.telemetry import ActiveRequest, NTGTelemetryLogger


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
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

        self.state_manager = state_manager or StateManager()

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
            dep
            for dep in self.deployments
            if dep.api_key and dep.api_key.strip() and dep.available
        ]

        model_list: List[Dict[str, Any]] = []
        registered_pairs: Set[tuple[str, str]] = set()

        for dep in active_deployments:
            litellm_dict = dep.to_litellm_dict()
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

        # 4. Configure bounded retry budget
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

        # 6. Synchronization and request context
        self._model_pool_lock = threading.Lock()
        self._last_pool_fingerprint: Optional[tuple] = tuple(
            sorted((dep.id, dep.routing_config_fingerprint()) for dep in self.deployments)
        )
        self._request_context = threading.local()
        self._active_requests: Dict[str, ActiveRequest] = {}
        self._active_requests_lock = threading.Lock()
        self.unknown_metrics = DeploymentMetrics()

        self.router_id = uuid.uuid4().hex[:8]

        # 7. Register telemetry and circuit breaker callback via public litellm.callbacks contract
        self.telemetry_logger = NTGTelemetryLogger(self)

        def _keep_cb(cb: Any) -> bool:
            if not isinstance(cb, NTGTelemetryLogger):
                return True
            owner = getattr(cb, "router", None)
            if owner is None or owner is self:
                return False
            return True

        litellm.callbacks = [cb for cb in litellm.callbacks if _keep_cb(cb)]
        litellm.callbacks.append(self.telemetry_logger)
        litellm.input_callback = [cb for cb in litellm.input_callback if _keep_cb(cb)]
        litellm.input_callback.append(self.telemetry_logger)

    def close(self) -> None:
        """Deregister callbacks and release resources."""
        if hasattr(self, "telemetry_logger") and self.telemetry_logger:
            litellm.callbacks = [cb for cb in litellm.callbacks if cb is not self.telemetry_logger]
            litellm.input_callback = [
                cb for cb in litellm.input_callback if cb is not self.telemetry_logger
            ]
        with self._active_requests_lock:
            self._active_requests.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def add_deployment(self, deployment: Deployment) -> None:
        """Add a deployment to the router and synchronize the LiteLLM model pool."""
        self.state_manager.apply_to_deployment(deployment)
        deployment.metrics = self.state_manager.get_metrics(deployment.id)
        if deployment not in self.deployments:
            self.deployments.append(deployment)
        self.deployment_map[deployment.id] = deployment
        self._sync_router_model_pool()

    def remove_deployment(self, deployment_id: str) -> bool:
        """Remove a deployment from the router and synchronize the LiteLLM model pool."""
        initial_len = len(self.deployments)
        self.deployments = [d for d in self.deployments if d.id != deployment_id]
        self.deployment_map.pop(deployment_id, None)
        if len(self.deployments) < initial_len:
            self._sync_router_model_pool()
            return True
        return False

    def update_deployment(self, deployment: Deployment) -> bool:
        """Update an existing deployment and synchronize the LiteLLM model pool."""
        for i, d in enumerate(self.deployments):
            if d.id == deployment.id:
                self.deployments[i] = deployment
                self.deployment_map[deployment.id] = deployment
                self._sync_router_model_pool()
                return True
        return False

    def _register_active_request(
        self, request_id: str, selected_dep: Optional[Deployment] = None
    ) -> ActiveRequest:
        """Register active request context for cross-thread attempt coordination."""
        with self._active_requests_lock:
            req = ActiveRequest(
                request_id=request_id,
                router_id=self.router_id,
                selected_dep=selected_dep,
            )
            self._active_requests[request_id] = req
            return req

    def _get_active_request(self, request_id: Optional[str]) -> Optional[ActiveRequest]:
        """Lookup active request context by request ID."""
        if not request_id:
            return None
        with self._active_requests_lock:
            return self._active_requests.get(request_id)

    def _unregister_active_request(self, request_id: str) -> Optional[ActiveRequest]:
        """Unregister active request context when request terminates."""
        with self._active_requests_lock:
            return self._active_requests.pop(request_id, None)

    def _sync_router_model_pool(self) -> None:
        """Fast, thread-safe synchronization of the shared LiteLLM model pool."""
        with self._model_pool_lock:
            fingerprint_items = [
                (dep.id, dep.routing_config_fingerprint()) for dep in self.deployments
            ]
            pool_fingerprint = tuple(sorted(fingerprint_items))
            if self._last_pool_fingerprint == pool_fingerprint:
                return

            self.deployment_map = {dep.id: dep for dep in self.deployments}

            models: List[Dict[str, Any]] = []
            registered_pairs: Set[tuple[str, str]] = set()

            for dep in self.deployments:
                if not dep.api_key or not dep.api_key.strip():
                    continue
                if dep.is_quarantined or not dep.available:
                    continue

                litellm_dict = dep.to_litellm_dict()
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
                        models.append(entry)

            self.router.set_model_list(models)
            self._last_pool_fingerprint = pool_fingerprint

    def _record_request_attempt(self, deployment: Deployment) -> None:
        """Track deployment attempted in thread-local request context and record attempt metric."""
        attempts = getattr(self._request_context, "attempted_deployments", None)
        if attempts is not None and isinstance(attempts, list):
            attempts.append(deployment)
        deployment.metrics.record_attempt()

    def _propagate_account_auth_quarantine(
        self,
        provider: str,
        account: str,
        reason: str,
        failed_key: Optional[str] = None,
    ) -> None:
        """Propagate an authentication failure quarantine across all deployments under the same account."""
        for dep in self.deployments:
            if dep.provider == provider and dep.account == account:
                dep.transition_to_quarantine(
                    reason=reason,
                    state=CircuitState.AUTH_FAILED,
                    failed_key=failed_key,
                )
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
        self._sync_router_model_pool()

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
            if info.status_code and info.status_code in (429, 400, 404):
                return True
            return False

    def revalidate_deployment(
        self,
        deployment_id: str,
        force: bool = False,
        probe_fn: Optional[Callable[[Deployment], bool]] = None,
    ) -> bool:
        """Attempt to revalidate a quarantined deployment."""
        dep = next(
            (d for d in self.deployments if d.id == deployment_id or d.model == deployment_id),
            None,
        )
        if not dep:
            return False

        if not dep.is_quarantined:
            return True

        now = time.time()
        if not force and dep.last_auth_probe_at > 0:
            elapsed = now - dep.last_auth_probe_at
            if elapsed < dep.auth_probe_backoff:
                return False

        dep.last_auth_probe_at = now

        probe = probe_fn if probe_fn is not None else self._probe_deployment_credentials
        success = probe(dep)

        if success:
            dep.transition_to_healthy(reason="Credentials revalidated; returned to service")
            self.state_manager.record_success(dep.id)
            self.state_manager.record_circuit_state(
                dep.id, CircuitState.HEALTHY, reason=dep.state_reason
            )
            self._sync_router_model_pool()
            return True
        else:
            with dep._state_lock:
                dep.auth_probe_backoff = min(3600.0, dep.auth_probe_backoff * 2.0)
                dep.state_reason = "Revalidation probe failed: credentials invalid"
            self.state_manager.record_circuit_state(
                dep.id, CircuitState.AUTH_FAILED, reason=dep.state_reason
            )
            return False

    def revalidate_account(
        self,
        provider: str,
        account: str,
        force: bool = False,
        probe_fn: Optional[Callable[[Deployment], bool]] = None,
    ) -> Dict[str, bool]:
        """Attempt to revalidate all deployments belonging to an account."""
        account_deps = [
            d for d in self.deployments if d.provider == provider and d.account == account
        ]
        if not account_deps:
            return {}

        probe_target = next((d for d in account_deps if d.is_quarantined), account_deps[0])
        success = self.revalidate_deployment(probe_target.id, force=force, probe_fn=probe_fn)

        results: Dict[str, bool] = {}
        for dep in account_deps:
            if success:
                dep.transition_to_healthy(
                    reason="Account credentials revalidated; returned to service"
                )
                self.state_manager.record_success(dep.id)
                self.state_manager.record_circuit_state(
                    dep.id, CircuitState.HEALTHY, reason=dep.state_reason
                )
                results[dep.id] = True
            else:
                results[dep.id] = False

        if success:
            self._sync_router_model_pool()
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
        """Retry discovery through existing application lifecycle for stale or uninitialized accounts."""
        results: Dict[str, Any] = {"success": {}, "failed": {}}

        # 1. Groq rediscovery
        if provider is None or provider == MODEL_GROUP_GROQ:
            groq_accounts = {
                dep.account: dep.api_key
                for dep in self.deployments
                if dep.provider == MODEL_GROUP_GROQ and dep.api_key
            }
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
                    discovered = discover_groq_models(
                        key, account_name=acc_name, state_manager=self.state_manager
                    )
                    if discovered and not self.state_manager.is_discovery_stale(cache_key):
                        for dep in self.deployments:
                            if dep.provider == MODEL_GROUP_GROQ and dep.account == acc_name:
                                dep.is_stale = False
                                dep.stale_reason = None
                        results["success"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = len(discovered)
                    else:
                        results["failed"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = (
                            "Discovery failed or remained stale"
                        )
                except Exception as err:
                    results["failed"][f"{MODEL_GROUP_GROQ}_{acc_name}"] = sanitize_secret(
                        str(err), key
                    )

        # 2. Gemini rediscovery
        if provider is None or provider == MODEL_GROUP_GEMINI:
            gemini_accounts = {
                dep.account: dep.api_key
                for dep in self.deployments
                if dep.provider == MODEL_GROUP_GEMINI and dep.api_key
            }
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
                    discovered = discover_gemini_models(
                        key, account_name=acc_name, state_manager=self.state_manager
                    )
                    if discovered and not self.state_manager.is_discovery_stale(cache_key):
                        for dep in self.deployments:
                            if dep.provider == MODEL_GROUP_GEMINI and dep.account == acc_name:
                                dep.is_stale = False
                                dep.stale_reason = None
                        results["success"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = len(discovered)
                    else:
                        results["failed"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = (
                            "Discovery failed or remained stale"
                        )
                except Exception as err:
                    results["failed"][f"{MODEL_GROUP_GEMINI}_{acc_name}"] = sanitize_secret(
                        str(err), key
                    )

        self._sync_router_model_pool()
        return results

    def reset_auth_quarantine(
        self,
        provider: Optional[str] = None,
        account: Optional[str] = None,
    ) -> int:
        """Administratively reset authentication quarantine for specified provider/account or all deployments."""
        reset_count = 0
        for dep in self.deployments:
            if provider and dep.provider != provider:
                continue
            if account and dep.account != account:
                continue
            if dep.is_quarantined:
                dep.transition_to_healthy(reason="Authentication quarantine reset administratively")
                self.state_manager.record_success(dep.id)
                self.state_manager.record_circuit_state(
                    dep.id, CircuitState.HEALTHY, reason=dep.state_reason
                )
                reset_count += 1
        if reset_count > 0:
            self._sync_router_model_pool()
        return reset_count

    def _propagate_account_quota_exhaustion(
        self,
        provider: str,
        account: str,
        reset_at: float,
        reason: str,
        api_key: Optional[str] = None,
        quota_scope: str = QuotaScope.ACCOUNT,
    ) -> None:
        """Propagate an account-wide or provider-wide quota exhaustion across all affected deployments."""
        now = time.time()
        cooldown = max(0.0, reset_at - now)
        for dep in self.deployments:
            should_propagate = False
            if quota_scope == QuotaScope.PROVIDER and dep.provider == provider:
                should_propagate = True
            elif dep.provider == provider and dep.account == account:
                should_propagate = True
            elif api_key and dep.api_key and dep.api_key == api_key:
                should_propagate = True

            if should_propagate:
                dep.transition_to_open(
                    cooldown_seconds=cooldown,
                    reason=reason,
                    reset_at=reset_at,
                )
                self.state_manager.record_cooldown(
                    dep.id, reset_at, "account_quota", quota_scope=quota_scope
                )
                self.state_manager.record_circuit_state(
                    dep.id, CircuitState.OPEN, reason=reason
                )
        self._sync_router_model_pool()

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
        """Return all active, eligible deployments across all configured providers."""
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
        claim_probe: bool = False,
        claimed_probes: Optional[List[Deployment]] = None,
        request_id: Optional[str] = None,
    ) -> List[Deployment]:
        """NTG Eligibility Filter: evaluate health, capabilities, quotas, and circuit state."""
        ts = now if now is not None else time.time()
        for dep in self.deployments:
            dep.refresh(ts)

        is_global = model_group in (
            MODEL_GROUP_AUTO,
            MODEL_GROUP_NTG_AUTO,
            "auto",
            "ntg-auto",
        )

        if claim_probe:
            candidates = [
                d
                for d in self.deployments
                if d.api_key and d.api_key.strip() and not d.is_quarantined
            ]
        else:
            candidates = (
                self.get_global_pool(ts)
                if is_global
                else [
                    d
                    for d in self.deployments
                    if d.api_key
                    and d.api_key.strip()
                    and not d.is_quarantined
                    and d.is_available(ts)
                ]
            )

        if not is_global and model_group not in (
            "coding",
            "reasoning",
            "vision",
            "tool_calling",
            "structured_output",
            "streaming",
        ):
            candidates = [
                d
                for d in candidates
                if d.logical_model == model_group
                or d.model == model_group
                or d.id == model_group
                or d.litellm_model == model_group
                or d.provider == model_group
            ]

        reqs = requirements
        if reqs is None:
            parsed_caps = self._parse_capabilities(capabilities) or ModelCapabilities()
            if min_context and min_context > parsed_caps.context_window:
                parsed_caps.context_window = min_context
            reqs = RequestRequirements(capabilities=parsed_caps)

        has_explicit_coding = any(d.capabilities.coding is True for d in candidates)
        effective_req_caps = reqs.capabilities
        if effective_req_caps.coding is True and not has_explicit_coding:
            effective_req_caps = ModelCapabilities(
                coding=None,
                reasoning=effective_req_caps.reasoning,
                vision=effective_req_caps.vision,
                tool_calling=effective_req_caps.tool_calling,
                structured_output=effective_req_caps.structured_output,
                streaming=effective_req_caps.streaming,
                context_window=effective_req_caps.context_window,
            )

        eligible: List[Deployment] = []
        for dep in candidates:
            if model_group == "coding":
                if has_explicit_coding and not dep.capabilities.coding:
                    continue
                if not has_explicit_coding and dep.capabilities.coding is False:
                    continue
            if reqs.capabilities.coding is True and dep.capabilities.coding is False:
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

            if not dep.capabilities.satisfies(effective_req_caps):
                continue

            if claim_probe:
                if not dep.admit_to_request_pool(ts, request_id=request_id):
                    continue
                if dep.circuit_state == CircuitState.HALF_OPEN and claimed_probes is not None:
                    if dep not in claimed_probes:
                        claimed_probes.append(dep)
            else:
                if not dep.is_available(ts):
                    continue

            eligible.append(dep)

        if reqs.coding_preferred and not reqs.capabilities.coding:
            coding_matches = [d for d in eligible if d.capabilities.coding is True]
            if coding_matches:
                if claim_probe and claimed_probes is not None:
                    for d in eligible:
                        if d not in coding_matches and d in claimed_probes:
                            d.release_half_open_probe(request_id=request_id)
                            claimed_probes.remove(d)
                eligible = coding_matches

        return eligible

    def _build_request_pool_and_fallbacks(
        self,
        target_model: str,
        requirements: RequestRequirements,
        caller_fallbacks: Any = None,
        now: Optional[float] = None,
        claim_probe: bool = False,
        claimed_probes: Optional[List[Deployment]] = None,
        request_id: Optional[str] = None,
    ) -> tuple[Optional[Deployment], Optional[List[Dict[str, List[str]]]], List[Deployment]]:
        """Determine initial deployment via LiteLLM load balancing and build LiteLLM fallbacks."""
        ts = now if now is not None else time.time()
        primary_eligible = self._filter_eligible_deployments(
            model_group=target_model,
            requirements=requirements,
            now=ts,
            claim_probe=claim_probe,
            claimed_probes=claimed_probes,
            request_id=request_id,
        )

        fallback_deployments: List[Deployment] = []
        user_specified_dests: Optional[List[str]] = None
        if caller_fallbacks:
            validated_fb = self._validate_user_fallbacks(
                caller_fallbacks,
                requirements,
                ts,
                claim_probe=claim_probe,
                claimed_probes=claimed_probes,
                request_id=request_id,
            )
            user_specified_dests = []
            for fb_item in validated_fb:
                for dests in fb_item.values():
                    for dst in dests:
                        dst_dep = self.deployment_map.get(dst)
                        if dst_dep and dst_dep not in fallback_deployments:
                            fallback_deployments.append(dst_dep)
                        if dst not in user_specified_dests:
                            user_specified_dests.append(dst)
        elif target_model in self.provider_fallbacks:
            for fb_group in self.provider_fallbacks[target_model]:
                fb_eligible = self._filter_eligible_deployments(
                    model_group=fb_group,
                    requirements=requirements,
                    now=ts,
                    claim_probe=claim_probe,
                    claimed_probes=claimed_probes,
                    request_id=request_id,
                )
                for fb_dep in fb_eligible:
                    if fb_dep not in primary_eligible and fb_dep not in fallback_deployments:
                        fallback_deployments.append(fb_dep)

        if not primary_eligible and not fallback_deployments:
            return None, None, []

        candidates = primary_eligible if primary_eligible else fallback_deployments

        if len(candidates) == 1:
            selected_dep = candidates[0]
        else:
            cand_dicts = [d.to_litellm_dict(group_override=d.id) for d in candidates]
            chosen = simple_shuffle(self.router, cand_dicts, target_model)
            chosen_id = chosen.get("model_info", {}).get("id") or chosen.get("model_name")
            selected_dep = self.deployment_map.get(chosen_id, candidates[0])

        if not selected_dep.is_eligible(ts):
            still_eligible = [d for d in candidates if d.is_eligible(ts)]
            if still_eligible:
                selected_dep = still_eligible[0]
            else:
                return None, None, []

        if user_specified_dests is not None:
            remaining_fallbacks = [
                dst for dst in user_specified_dests if dst != selected_dep.id
            ]
        else:
            rem_primary = [d.id for d in primary_eligible if d.id != selected_dep.id]
            rem_secondary = [
                d.id
                for d in fallback_deployments
                if d not in primary_eligible and d.id != selected_dep.id
            ]
            if len(rem_primary) > 1:
                rem_primary = random.sample(rem_primary, len(rem_primary))
            if len(rem_secondary) > 1:
                rem_secondary = random.sample(rem_secondary, len(rem_secondary))
            remaining_fallbacks = rem_primary + rem_secondary

        seen_fb = set()
        clean_fallbacks = []
        for dst in remaining_fallbacks:
            if dst not in seen_fb and dst != selected_dep.id:
                seen_fb.add(dst)
                clean_fallbacks.append(dst)

        fallbacks = [{selected_dep.id: clean_fallbacks}] if clean_fallbacks else None

        return selected_dep, fallbacks, primary_eligible

    def _rank_eligible_deployments(
        self,
        eligible: List[Deployment],
        primary_group: Optional[str] = None,
    ) -> List[Deployment]:
        """Deprecated: NTG no longer makes ranking decisions. LiteLLM handles selection."""
        return list(eligible)

    def _build_compatible_fallbacks(
        self,
        ordered_eligible: List[Deployment],
    ) -> Optional[List[Dict[str, List[str]]]]:
        """Deprecated: Sequential fallback chains removed. LiteLLM handles pool failover."""
        return None

    def _validate_user_fallbacks(
        self,
        user_fallbacks: Any,
        requirements: RequestRequirements,
        now: Optional[float] = None,
        claim_probe: bool = False,
        claimed_probes: Optional[List[Deployment]] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, List[str]]]:
        """Validate caller-supplied fallbacks to ensure compatibility and eligibility."""
        ts = now if now is not None else time.time()
        dep_by_id = {d.id: d for d in self.deployments}
        dep_by_group: Dict[str, List[Deployment]] = {}
        for d in self.deployments:
            dep_by_group.setdefault(d.logical_model, []).append(d)
            dep_by_group.setdefault(d.model, []).append(d)

        valid_sources_and_dests = (
            set(dep_by_id.keys())
            | set(dep_by_group.keys())
            | {"auto", "ntg-auto", MODEL_GROUP_AUTO, MODEL_GROUP_NTG_AUTO}
        )

        validated: List[Dict[str, List[str]]] = []
        if isinstance(user_fallbacks, list):
            for item in user_fallbacks:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Invalid fallback mapping: expected dict, got {type(item).__name__}"
                    )
                for src, dst_list in item.items():
                    if src not in valid_sources_and_dests:
                        continue
                    if not isinstance(dst_list, list):
                        continue
                    clean_dst: List[str] = []
                    for dst in dst_list:
                        if dst not in valid_sources_and_dests:
                            continue
                        candidates: List[Deployment] = []
                        if dst in dep_by_id:
                            candidates = [dep_by_id[dst]]
                        elif dst in dep_by_group:
                            candidates = dep_by_group[dst]

                        for d in candidates:
                            if not d.api_key or not d.api_key.strip() or d.is_quarantined:
                                continue
                            if not d.capabilities.satisfies(requirements.capabilities):
                                continue
                            if claim_probe:
                                if not d.admit_to_request_pool(ts, request_id=request_id):
                                    continue
                                if (
                                    d.circuit_state == CircuitState.HALF_OPEN
                                    and claimed_probes is not None
                                ):
                                    if d not in claimed_probes:
                                        claimed_probes.append(d)
                            else:
                                if not d.is_available(ts):
                                    continue

                            if d.id not in clean_dst:
                                clean_dst.append(d.id)

                    if clean_dst:
                        validated.append({src: clean_dst})
        return validated

    @staticmethod
    def _parse_capabilities(
        caps: Optional[ModelCapabilities | dict | list[str] | str] = None,
    ) -> Optional[ModelCapabilities]:
        """Normalize capability parameter into a ModelCapabilities object."""
        return parse_capabilities(caps)

    def _get_response_deployment(
        self,
        response: Any,
        eligible: Optional[List[Deployment]] = None,
    ) -> Optional[Deployment]:
        """Extract fulfilling deployment strictly from verified response metadata."""
        if response is None:
            return None

        dep = getattr(response, "_ntg_deployment", None)
        if dep is not None and isinstance(dep, Deployment):
            return dep

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
            dep_id = hidden.get("model_id")
            if not dep_id and isinstance(hidden.get("model_info"), dict):
                dep_id = hidden.get("model_info", {}).get("id")
            if dep_id and dep_id in self.deployment_map:
                dep = self.deployment_map[dep_id]
                try:
                    setattr(response, "_ntg_deployment", dep)
                except Exception:
                    pass
                return dep

        return None

    def _record_user_request_metric(
        self,
        active_req: Optional[ActiveRequest],
        fulfilling_dep: Optional[Deployment],
        selected_dep: Optional[Deployment],
    ) -> None:
        """Atomically increment user_requests metric exactly once for this public request."""
        if active_req:
            with active_req.lock:
                if active_req.user_request_recorded:
                    return
                active_req.user_request_recorded = True

            attempted = (
                active_req.attempted_deployments
                if active_req.attempted_deployments
                else getattr(self._request_context, "attempted_deployments", [])
            )
        else:
            attempted = getattr(self._request_context, "attempted_deployments", [])

        target_dep = None
        if fulfilling_dep:
            target_dep = fulfilling_dep
        elif selected_dep and selected_dep in attempted and len(attempted) == 1:
            target_dep = selected_dep
        elif len(attempted) == 1:
            target_dep = attempted[0]

        if target_dep:
            target_dep.record_user_request()
            self.state_manager.record_metrics(target_dep.id, target_dep.metrics)
        else:
            self.unknown_metrics.increment_user_requests()
            self.state_manager.record_metrics("unknown", self.unknown_metrics)

    def _route_request(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[
        Optional[Any],
        Optional[Deployment],
        Optional[Exception],
        RequestRequirements,
        float,
    ]:
        """Unified internal routing execution pipeline shared by ask() and completion()."""
        target_model = model or self.default_model
        if target_model in ("ntg-auto", MODEL_GROUP_NTG_AUTO):
            target_model = MODEL_GROUP_AUTO

        reqs = extract_request_requirements(
            messages=messages,
            prompt=prompt,
            model=target_model,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )

        req_retries = kwargs.pop("num_retries", None)
        effective_retries = (
            min(max(0, req_retries), MAX_RETRY_BUDGET)
            if req_retries is not None
            else self.num_retries
        )
        kwargs["num_retries"] = effective_retries

        caller_fallbacks = kwargs.pop("fallbacks", None)
        now = time.time()
        claimed_probes: List[Deployment] = []
        request_id = uuid.uuid4().hex[:12]
        self._request_context.attempted_deployments = []
        self._request_context.recorded_call_ids = set()

        selected_dep, fallbacks, eligible = self._build_request_pool_and_fallbacks(
            target_model=target_model,
            requirements=reqs,
            caller_fallbacks=caller_fallbacks,
            now=now,
            claim_probe=True,
            claimed_probes=claimed_probes,
            request_id=request_id,
        )

        if not selected_dep:
            for dep in claimed_probes:
                dep.release_half_open_probe(request_id=request_id)
            self.unknown_metrics.increment_user_requests()
            self.state_manager.record_metrics("unknown", self.unknown_metrics)
            err = NoEligibleDeploymentsError(
                f"No eligible deployments for model='{target_model}' satisfying requirements: "
                f"{reqs.describe()}."
            )
            return None, None, err, reqs, now

        call_kwargs = dict(kwargs)
        call_kwargs["num_retries"] = effective_retries
        if fallbacks:
            call_kwargs["fallbacks"] = fallbacks
        meta = dict(call_kwargs.get("metadata", {}) or {})
        meta["ntg_request_id"] = request_id
        meta["ntg_router_id"] = self.router_id
        call_kwargs["metadata"] = meta

        active_req = self._register_active_request(request_id, selected_dep)
        self._request_context.active_request_id = request_id

        try:
            response = self.router.completion(
                model=selected_dep.id,
                messages=messages,
                **call_kwargs,
            )

            deployment = self._get_response_deployment(response, eligible)
            self._record_user_request_metric(active_req, deployment, selected_dep)

            return response, deployment, None, reqs, now

        except Exception as error:
            self._record_user_request_metric(active_req, None, selected_dep)
            return None, None, error, reqs, now

        finally:
            self._record_user_request_metric(active_req, None, selected_dep)
            attempted = (
                active_req.attempted_deployments
                if active_req.attempted_deployments
                else getattr(self._request_context, "attempted_deployments", [])
            )
            attempted_ids = {d.id for d in attempted}
            for dep in claimed_probes:
                if dep.id not in attempted_ids:
                    dep.release_half_open_probe(request_id=request_id)
                elif (
                    dep.circuit_state == CircuitState.HALF_OPEN
                    and request_id in dep._active_probe_request_ids
                ):
                    dep.release_half_open_probe(request_id=request_id)
            self._unregister_active_request(request_id)
            self._request_context.active_request_id = None
            self._request_context.attempted_deployments = []
            self._request_context.recorded_call_ids = set()

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

        messages = [{"role": "user", "content": prompt}]
        response, deployment, error, reqs, now = self._route_request(
            messages=messages,
            model=target_model,
            capabilities=capabilities,
            min_context=min_context,
            prompt=prompt,
            **kwargs,
        )

        if error is not None:
            if isinstance(error, NoEligibleDeploymentsError):
                print_divider("NO ELIGIBLE DEPLOYMENTS")
                print(
                    f"No available deployments satisfy request for model group '{target_model}'"
                )
                print(f"Required capabilities: {reqs.describe()}")
                if min_context:
                    print(f"Required context     : >={min_context:,}")

                resets = [
                    dep.remaining_cooldown
                    for dep in self.deployments
                    if dep.remaining_cooldown > 0
                ]
                if resets:
                    next_capacity = min(resets)
                    print(
                        f"Next available deployment capacity in: {next_capacity:.1f}s ({utc_string(now + next_capacity)})"
                    )
                print_account_status(self.deployments)
                return None
            else:
                print_divider("ALL ELIGIBLE DEPLOYMENTS EXHAUSTED")
                print(
                    f"Request failed across healthy deployments for model group '{target_model}'."
                )
                print(f"Final error: {error}")
                print_account_status(self.deployments)
                return None

        if hasattr(response, "choices") and response.choices:
            print("\n" + str(response.choices[0].message.content))

        if deployment:
            print_request_execution(deployment, getattr(response, "model", None))

        return response

    def completion(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        capabilities: Optional[ModelCapabilities | dict | list[str] | str] = None,
        min_context: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute chat completion conforming to standard OpenAI/LiteLLM signature."""
        response, _, error, _, _ = self._route_request(
            messages=messages,
            model=model,
            capabilities=capabilities,
            min_context=min_context,
            **kwargs,
        )
        if error is not None:
            raise error
        return response


OpenRouterFallbackRouter = UnifiedNTGRouter
